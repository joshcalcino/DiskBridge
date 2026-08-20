"""External non-LTE level populations for RADMC-3D ``lines_mode = 50``.

This module is the public face of the DiskBridge-native 3D HEALPix
escape-probability statistical-equilibrium solver. It:

1. Parses the molecule file (validated upstream for collider order).
2. Pulls species density, collider densities, gas temperature, gas velocity,
   and microturbulence from the in-memory ``RadModel`` and ``ChemistryResult``.
3. Reads the full species-density field in RADMC-3D cell order.
4. Iterates HEALPix escape probability + statistical equilibrium until
   converged.
5. Writes ``levelpop_<species>.dat`` in RADMC-3D cell order and an audit
   manifest.
"""

# db-keywords: shielding, healpix-columns, gow17, nonlte, line-transfer, gas-temperature, units, radmc3d, model, mesh, field
# db-role: helper
# db-scope: package
# db-purpose: External non-LTE level populations for RADMC-3D ``lines_mode = 50``.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json
import time

import numpy as np
from numba import njit, prange

from diskbridge._constants import K_B
from diskbridge._logging import logger
from diskbridge._units import units
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.model.kinematics import (
    cartesian_velocity_cm_s,
    microturbulence_cm_s,
    molecular_doppler_width_cm_s,
)
from diskbridge.radmc3d.colliders import (
    gow17_lamda_colliders,
    install_validated_molecule_file,
)
from diskbridge.radmc3d.writer import RadWriter
from diskbridge.utils import sha256_array, sha256_file
from diskbridge.chemistry.shielding.healpix_columns import (
    SphericalHealpixRayTracer,
    CartesianHealpixRayTracer,
)
from .escape_healpix import (
    _spherical_boundary_mode_codes,
    compute_escape_probabilities_healpix,
)
from .molecular_rates import (
    MoleculeData,
    lte_populations,
    parse_lamda_molecule_file,
    stack_collider_tables,
)
from .se_solver import (
    compute_collisional_rates,
    compute_line_center_opacity,
    population_change_diagnostics,
    solve_statistical_equilibrium,
)


@dataclass(frozen=True)
class HealpixSEConfig:
    """Configuration for the 3D HEALPix non-LTE solver.

    Parameters
    ----------
    spherical_inner_boundary : {"stop", "vacuum_cavity"}, optional
        Spherical-grid treatment at ``rmin``. ``"stop"`` ends the ray;
        ``"vacuum_cavity"`` crosses the central cavity with zero opacity and
        resumes integration on the far side.
    spherical_theta_boundary : {"vacuum", "boundary_cell_to_rmax"}, optional
        Spherical-grid treatment outside a truncated theta range. ``"vacuum"``
        ends the ray at the theta edge; ``"boundary_cell_to_rmax"`` extends
        the boundary-cell fields through the uncovered angle to ``rmax``.
    species_abundance_floor : float, optional
        Minimum emitting-species abundance relative to hydrogen nuclei for the
        iterative non-LTE solve. Positive-density cells below the floor retain
        local gas-temperature LTE populations in the written all-cell output.
        The default of zero disables abundance selection.
    Collision coefficients are held at the nearest tabulated temperature
    boundary outside each collider table's range. The physical gas temperature
    used for LTE populations, detailed balance, and Doppler broadening is not
    clipped.
    """

    nside: int = 4
    maxiter: int = 60
    convcrit: float = 1.0e-5
    tbg_K: float = 2.7255
    memory_budget_gib: float = 8.0
    max_ray_steps: int = 200_000
    relaxation: float = 1.0
    escape_chunk_size: int = 50_000
    allow_unconverged: bool = False
    species_density_floor_cm3: float = 0.0
    species_abundance_floor: float = 0.0
    overwrite: bool = True
    checkpoint_interval: int = 1
    checkpoint_levelpop: bool = False
    resume_from_checkpoint: bool = False
    spherical_inner_boundary: Literal["stop", "vacuum_cavity"] = "vacuum_cavity"
    spherical_theta_boundary: Literal[
        "vacuum", "boundary_cell_to_rmax"
    ] = "boundary_cell_to_rmax"

    def __post_init__(self) -> None:
        """Validate spherical ray-boundary choices at configuration time."""

        _spherical_boundary_mode_codes(
            self.spherical_inner_boundary,
            self.spherical_theta_boundary,
        )
        abundance_floor = float(self.species_abundance_floor)
        if not np.isfinite(abundance_floor) or abundance_floor < 0.0:
            raise ValueError(
                "species_abundance_floor must be a non-negative finite value, "
                f"got {abundance_floor}"
            )


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------


def _field_to_f64(field_quantity, *, unit: str) -> np.ndarray:
    """Convert a model Field/Quantity into a plain f64 array in ``unit``."""
    if hasattr(field_quantity, "data"):
        q = field_quantity.data
    else:
        q = field_quantity
    if hasattr(q, "to"):
        arr = np.asarray(q.to(unit).magnitude, dtype=np.float64)
    else:
        arr = np.asarray(q, dtype=np.float64)
    return np.ascontiguousarray(arr)


def _gas_field(rad, name: str, *, unit: str) -> np.ndarray:
    if rad.model.gas is None or name not in rad.model.gas:
        raise KeyError(f"model.gas[{name!r}] is required for the non-LTE solver")
    return _field_to_f64(rad.model.gas[name], unit=unit)


def _species_density_cm3(result, species: str) -> np.ndarray:
    if species not in result.number_densities:
        raise KeyError(
            f"chemistry_result.number_densities[{species!r}] is required"
        )
    return _field_to_f64(result.number_densities[species], unit="cm^-3")


def _species_abundance_per_h_nucleus(result, species: str) -> np.ndarray:
    """Return an emitting-species abundance relative to hydrogen nuclei."""

    abundances = getattr(result, "abundances", {})
    if species not in abundances:
        raise KeyError(
            f"chemistry_result.abundances[{species!r}] is required when "
            "species_abundance_floor is positive"
        )
    return _field_to_f64(abundances[species], unit="dimensionless")


def _collider_density_cm3(result, name: str, opr: float | None = None) -> np.ndarray:
    """Return collider number density for the strict GOW17/LAMDA labels.

    ``h2`` is computed as ``p-h2 + o-h2`` if necessary.
    """
    nds = result.number_densities
    if name in nds:
        return _field_to_f64(nds[name], unit="cm^-3")
    if name == "h2":
        if "p-h2" in nds and "o-h2" in nds:
            return (
                _field_to_f64(nds["p-h2"], unit="cm^-3")
                + _field_to_f64(nds["o-h2"], unit="cm^-3")
            )
        if "h2" in nds:
            return _field_to_f64(nds["h2"], unit="cm^-3")
    raise KeyError(
        f"Cannot resolve collider {name!r} from chemistry_result.number_densities"
    )


def _radmc_transfer_mesh(mesh):
    """Return the spherical geometry convention used by RADMC-3D input files.

    Fresh hydro loads should already be canonicalized to phi=0..2pi. The
    fallback here preserves compatibility with older saved snapshots.
    """

    if mesh.coord_system != "spherical":
        return mesh
    return Mesh.spherical(
        r=Axis(edges=mesh.edges("r")),
        theta=Axis(edges=mesh.edges("theta")),
        phi=Axis(edges=RadWriter.radmc_spherical_phi_edges_rad(mesh) * units("radian")),
    )


def _range_pair(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [0.0, 0.0]
    return [float(np.min(arr)), float(np.max(arr))]


def _collision_temperature_diagnostics(
    molecule: MoleculeData,
    Tgas: np.ndarray,
    cell_mask: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Summarize selected cells outside each collider's temperature table."""

    selected = np.asarray(Tgas, dtype=np.float64)[np.asarray(cell_mask, dtype=bool)]
    diagnostics: dict[str, dict[str, float | int]] = {}
    for name, tgrid in zip(
        molecule.collider_names,
        molecule.collider_tgrid_K,
        strict=True,
    ):
        low = float(tgrid[0])
        high = float(tgrid[-1])
        diagnostics[name] = {
            "table_min_K": low,
            "table_max_K": high,
            "selected_cells_below": int(np.count_nonzero(selected < low)),
            "selected_cells_above": int(np.count_nonzero(selected > high)),
        }
    return diagnostics


def _beta_diagnostics(beta: np.ndarray) -> dict[str, float]:
    arr = np.asarray(beta, dtype=np.float64)
    if arr.size == 0:
        return {
            "beta_min": 1.0,
            "beta_p01": 1.0,
            "beta_p10": 1.0,
            "beta_median": 1.0,
            "beta_p90": 1.0,
            "beta_max": 1.0,
        }
    qs = np.percentile(arr, [1.0, 10.0, 50.0, 90.0])
    return {
        "beta_min": float(np.min(arr)),
        "beta_p01": float(qs[0]),
        "beta_p10": float(qs[1]),
        "beta_median": float(qs[2]),
        "beta_p90": float(qs[3]),
        "beta_max": float(np.max(arr)),
    }


def _velocity_range_diagnostics(velocity_xyz: np.ndarray) -> dict[str, list[float]]:
    speed = np.linalg.norm(velocity_xyz, axis=-1)
    return {
        "vx": _range_pair(velocity_xyz[..., 0]),
        "vy": _range_pair(velocity_xyz[..., 1]),
        "vz": _range_pair(velocity_xyz[..., 2]),
        "speed": _range_pair(speed),
    }


def _gas_velocity_components_cm_s(rad) -> tuple:
    mesh = rad.model.mesh
    gas = rad.model.gas
    if mesh.coord_system == "spherical":
        names = ("vr", "vtheta", "vphi")
    elif mesh.coord_system == "cartesian":
        names = ("vx", "vy", "vz")
    else:
        raise ValueError(f"Unsupported coordinate system: {mesh.coord_system!r}")
    components = tuple(
        _gas_field(rad, name, unit="cm/s")
        for name in names
    )
    return mesh, components


def _mesh_edge_hashes(mesh) -> dict[str, str]:
    units_by_axis = {
        "r": "cm",
        "theta": "rad",
        "phi": "rad",
        "x": "cm",
        "y": "cm",
        "z": "cm",
    }
    return {
        name: sha256_array(mesh.edges_f64(name, units_by_axis[name]))
        for name in mesh.axis_names()
    }


def _checkpoint_paths(output_dir: Path, species: str, iteration: int | None = None) -> dict[str, Path]:
    root = output_dir / "checkpoints"
    paths = {
        "root": root,
        "latest": root / f"checkpoint_latest_{species}.json",
    }
    if iteration is not None:
        tag = f"iter{int(iteration):04d}"
        paths.update({
            "fracpop": root / f"fracpop_{species}.{tag}.npz",
            "manifest": root / f"checkpoint_manifest_{species}.{tag}.json",
            "levelpop": root / f"levelpop_{species}.{tag}.dat",
        })
    return paths


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _checkpoint_fingerprint(
    *,
    species: str,
    config: HealpixSEConfig,
    molecule: MoleculeData,
    molecule_sha256: str,
    gas_velocity_sha256: str,
    transfer_mesh,
    n_species: np.ndarray,
    species_abundance: np.ndarray | None,
    Tgas: np.ndarray,
    a_turb: np.ndarray,
    cell_idx: np.ndarray,
    collider_dens_cand: np.ndarray,
) -> dict:
    return {
        "version": 3,
        "species": species,
        "solver": "healpix_escape_probability_statistical_equilibrium",
        "nside": int(config.nside),
        "tbg_K": float(config.tbg_K),
        "relaxation": float(config.relaxation),
        "max_ray_steps": int(config.max_ray_steps),
        "spherical_inner_boundary": config.spherical_inner_boundary,
        "spherical_theta_boundary": config.spherical_theta_boundary,
        "collision_temperature_policy": "nearest_table_boundary",
        "species_density_floor_cm3": float(config.species_density_floor_cm3),
        "species_abundance_floor": float(config.species_abundance_floor),
        "molecule_sha256": molecule_sha256,
        "gas_velocity_sha256": gas_velocity_sha256,
        "n_levels": int(molecule.nlev),
        "n_lines": int(molecule.nlin),
        "mesh_coord_system": transfer_mesh.coord_system,
        "mesh_shape": [int(x) for x in n_species.shape],
        "mesh_edge_hashes": _mesh_edge_hashes(transfer_mesh),
        "species_density_sha256": sha256_array(n_species),
        "species_abundance_sha256": (
            None
            if species_abundance is None
            else sha256_array(species_abundance)
        ),
        "temperature_sha256": sha256_array(Tgas),
        "microturbulence_sha256": sha256_array(a_turb),
        "candidate_idx_sha256": sha256_array(cell_idx),
        "collider_density_candidate_sha256": sha256_array(collider_dens_cand),
    }


def _flat_levelpop_radmc_order(
    *,
    mesh,
    molecule: MoleculeData,
    fracpop_cand: np.ndarray,
    n_species_cand: np.ndarray,
    cell_idx: np.ndarray,
    n_species: np.ndarray,
    Tgas: np.ndarray,
) -> np.ndarray:
    """Return all-cell level populations with LTE outside the solve mask."""

    if Tgas.shape != n_species.shape:
        raise ValueError(
            f"Tgas shape {Tgas.shape} does not match species-density shape "
            f"{n_species.shape}"
        )
    n0, n1, n2 = n_species.shape
    levelpop_full = _scatter_levelpop_to_full(
        fracpop_cand, n_species_cand, cell_idx, molecule.nlev, n0, n1, n2,
    )
    _normalize_and_fill_levelpop_lte(
        levelpop_full,
        n_species,
        Tgas,
        molecule.energy_erg,
        molecule.weight,
    )
    flat_per_level = np.stack(
        [RadWriter.flatten_scalar_to_radmc_order(mesh, levelpop_full[..., k])
         for k in range(molecule.nlev)],
        axis=1,
    )
    n_species_flat = RadWriter.flatten_scalar_to_radmc_order(mesh, n_species)
    sum_levels = flat_per_level.sum(axis=1)
    diff = np.abs(sum_levels - n_species_flat)
    tol = 1.0e-8 * np.maximum(n_species_flat, 1.0)
    bad = int(np.count_nonzero(diff > tol))
    if bad:
        raise RuntimeError(
            f"Population sum mismatch in {bad} cells "
            f"(max diff {float(diff.max()):.3e})"
        )
    return flat_per_level


def _write_iteration_checkpoint(
    *,
    output_dir: Path,
    species: str,
    iteration: int,
    fracpop: np.ndarray,
    fingerprint: dict,
    history: list[dict],
    maser_diagnostics: dict,
    final_err: float,
    final_beta_range: tuple[float, float],
    converged: bool,
    write_levelpop: bool,
    mesh,
    molecule: MoleculeData,
    n_species_cand: np.ndarray,
    cell_idx: np.ndarray,
    n_species: np.ndarray,
    Tgas: np.ndarray,
) -> None:
    paths = _checkpoint_paths(output_dir, species, iteration)
    paths["root"].mkdir(parents=True, exist_ok=True)
    fracpop_tmp = paths["fracpop"].with_name(paths["fracpop"].name + ".tmp")
    with fracpop_tmp.open("wb") as fobj:
        # This compact checkpoint is the resume source. It stores fractional
        # populations, not multiplied level populations, so zero-density cells
        # do not need to be materialized at every iteration.
        np.savez(
            fobj,
            fracpop=np.ascontiguousarray(fracpop, dtype=np.float64),
            iteration=np.asarray([int(iteration)], dtype=np.int64),
            final_err=np.asarray([float(final_err)], dtype=np.float64),
            beta_range=np.asarray(final_beta_range, dtype=np.float64),
        )
    fracpop_tmp.replace(paths["fracpop"])

    levelpop_path = None
    if write_levelpop:
        flat_per_level = _flat_levelpop_radmc_order(
            mesh=mesh,
            molecule=molecule,
            fracpop_cand=fracpop,
            n_species_cand=n_species_cand,
            cell_idx=cell_idx,
            n_species=n_species,
            Tgas=Tgas,
        )
        levelpop_tmp = paths["levelpop"].with_name(paths["levelpop"].name + ".tmp")
        RadWriter.write_levelpop_dat(
            levelpop_tmp,
            levelpop_cm3_radmc_order=flat_per_level,
            level_numbers_1based=np.arange(1, molecule.nlev + 1, dtype=np.int64),
        )
        levelpop_tmp.replace(paths["levelpop"])
        levelpop_path = str(paths["levelpop"])

    payload = {
        "checkpoint_type": "healpix_se_iteration",
        "species": species,
        "iteration": int(iteration),
        "fracpop_file": str(paths["fracpop"]),
        "levelpop_file": levelpop_path,
        "fingerprint": fingerprint,
        "history": history,
        "maser_suppression": maser_diagnostics,
        "final_convergence_error": float(final_err),
        "beta_range": [float(final_beta_range[0]), float(final_beta_range[1])],
        "converged": bool(converged),
    }
    _write_json_atomic(paths["manifest"], payload)
    _write_json_atomic(paths["latest"], payload)


def _load_latest_checkpoint(
    *,
    output_dir: Path,
    species: str,
    fingerprint: dict,
    expected_shape: tuple[int, int],
) -> dict | None:
    paths = _checkpoint_paths(output_dir, species)
    manifest_path = paths["latest"]
    if not manifest_path.exists():
        candidates = sorted(paths["root"].glob(f"checkpoint_manifest_{species}.iter*.json"))
        if not candidates:
            return None
        manifest_path = candidates[-1]
    payload = json.loads(manifest_path.read_text())
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(
            f"Checkpoint {manifest_path} does not match the current inputs; "
            "delete it or run without resume_from_checkpoint."
        )
    fracpop_path = Path(payload["fracpop_file"])
    if not fracpop_path.exists():
        raise FileNotFoundError(f"Checkpoint fractional populations are missing: {fracpop_path}")
    with np.load(fracpop_path) as data:
        fracpop = np.ascontiguousarray(data["fracpop"], dtype=np.float64)
    if fracpop.shape != expected_shape:
        raise ValueError(
            f"Checkpoint {fracpop_path} has fracpop shape {fracpop.shape}; "
            f"expected {expected_shape}"
        )
    return {
        "fracpop": fracpop,
        "iteration": int(payload["iteration"]),
        "history": list(payload.get("history", [])),
        "maser_diagnostics": payload.get("maser_suppression", _empty_maser_diagnostics()),
        "final_err": float(payload.get("final_convergence_error", np.inf)),
        "final_beta_range": tuple(float(x) for x in payload.get("beta_range", [1.0, 1.0])),
    }


def _selected_cell_diagnostics(
    *,
    molecule: MoleculeData,
    cell_idx: np.ndarray,
    Tgas_cand: np.ndarray,
    n_species_cand: np.ndarray,
    collider_dens_cand: np.ndarray,
    collider_names: list[str],
    beta: np.ndarray,
    alpha0_raw: np.ndarray,
    fracpop: np.ndarray,
    tgrids: np.ndarray,
    ntemps: np.ndarray,
    tables: np.ndarray,
) -> list[dict]:
    """Return compact per-cell diagnostics for interpreting failed solves."""

    n_cell = int(cell_idx.shape[0])
    if n_cell == 0:
        return []
    candidates: list[tuple[str, int]] = [
        ("highest_species_density_cell", int(np.argmax(n_species_cand))),
        ("lowest_species_density_cell", int(np.argmin(n_species_cand))),
        ("lowest_beta_cell", int(np.argmin(np.min(beta, axis=1)))),
        ("highest_opacity_cell", int(np.argmax(np.max(alpha0_raw, axis=1)))),
    ]
    positive = n_species_cand[n_species_cand > 0.0]
    if positive.size:
        median_value = float(np.median(positive))
        candidates.append((
            "median_species_density_cell",
            int(np.argmin(np.abs(n_species_cand - median_value))),
        ))

    seen: set[int] = set()
    selected: list[tuple[str, int]] = []
    for label, idx in candidates:
        if idx in seen:
            continue
        seen.add(idx)
        selected.append((label, idx))

    selected_idx = np.asarray([idx for _label, idx in selected], dtype=np.int64)
    C = compute_collisional_rates(
        np.ascontiguousarray(Tgas_cand[selected_idx]),
        np.ascontiguousarray(collider_dens_cand[selected_idx]),
        tgrids,
        ntemps,
        tables,
        molecule.weight,
        molecule.energy_erg,
    )
    lte = lte_populations(molecule, Tgas_cand[selected_idx])

    n_report_lines = min(5, molecule.nlin)
    n_report_levels = min(8, molecule.nlev)
    out: list[dict] = []
    for local, (label, idx) in enumerate(selected):
        collider_ranges = {
            name: float(collider_dens_cand[idx, k])
            for k, name in enumerate(collider_names)
        }
        cul_over_aul = []
        for m in range(n_report_lines):
            u = int(molecule.iup[m])
            l = int(molecule.ilow[m])
            aul = float(molecule.aud[m])
            cul = float(C[local, u, l])
            cul_over_aul.append({
                "line_index": int(m),
                "radmc_line_number": int(m + 1),
                "upper_level": int(u + 1),
                "lower_level": int(l + 1),
                "C_ul_s-1": cul,
                "A_ul_s-1": aul,
                "C_ul_over_A_ul": cul / aul if aul > 0.0 else None,
            })
        out.append({
            "label": label,
            "candidate_index": int(idx),
            "grid_index": [int(x) for x in cell_idx[idx].tolist()],
            "Tgas_K": float(Tgas_cand[idx]),
            "n_species_cm3": float(n_species_cand[idx]),
            "collider_densities_cm3": collider_ranges,
            "beta_first_lines": [float(x) for x in beta[idx, :n_report_lines].tolist()],
            "alpha0_raw_first_lines_cm1": [
                float(x) for x in alpha0_raw[idx, :n_report_lines].tolist()
            ],
            "lte_fractions_first_levels": [
                float(x) for x in lte[local, :n_report_levels].tolist()
            ],
            "final_fractions_first_levels": [
                float(x) for x in fracpop[idx, :n_report_levels].tolist()
            ],
            "C_ul_over_A_ul_first_lines": cul_over_aul,
        })
    return out


def _empty_maser_diagnostics() -> dict:
    return {
        "policy": "radmc3d_compatible_suppress_negative_opacity",
        "applied": False,
        "warning": (
            "Population inversions are written unchanged. Negative opacity is "
            "set to zero only for DiskBridge escape-probability updates; "
            "RADMC-3D suppresses negative opacity again during final ray tracing. "
            "This output does not include maser amplification."
        ),
        "total_negative_opacity_entries": 0,
        "max_negative_opacity_cells_in_iteration": 0,
        "min_negative_alpha0_cm1": None,
        "iterations": [],
        "transitions": [],
    }


def _iteration_maser_diagnostics(
    *,
    molecule: MoleculeData,
    fracpop: np.ndarray,
    alpha0_raw: np.ndarray,
    iteration: int,
) -> dict:
    negative = alpha0_raw < 0.0
    negative_entries = int(np.count_nonzero(negative))
    negative_cells = int(np.count_nonzero(np.any(negative, axis=1)))
    min_alpha = (
        float(np.min(alpha0_raw[negative]))
        if negative_entries
        else float(np.min(alpha0_raw))
    )
    transitions = []
    if negative_entries:
        for line_index in range(molecule.nlin):
            line_negative = negative[:, line_index]
            n_line_cells = int(np.count_nonzero(line_negative))
            if n_line_cells == 0:
                continue
            upper = int(molecule.iup[line_index])
            lower = int(molecule.ilow[line_index])
            opacity_factor = (
                molecule.weight[upper] / molecule.weight[lower] * fracpop[:, lower]
                - fracpop[:, upper]
            )
            transitions.append({
                "line_index": int(line_index),
                "radmc_line_number": int(line_index + 1),
                "upper_level": int(upper + 1),
                "lower_level": int(lower + 1),
                "n_cells": n_line_cells,
                "min_opacity_factor": float(np.min(opacity_factor[line_negative])),
                "min_alpha0_cm1": float(np.min(alpha0_raw[line_negative, line_index])),
            })
    return {
        "iteration": int(iteration),
        "applied": bool(negative_entries),
        "negative_opacity_entries": negative_entries,
        "negative_opacity_cells": negative_cells,
        "negative_opacity_min_cm1": min_alpha,
        "transitions": transitions,
    }


def _update_maser_diagnostics(aggregate: dict, iteration_diag: dict) -> None:
    aggregate["iterations"].append(iteration_diag)
    if not iteration_diag["applied"]:
        return
    if not aggregate["applied"]:
        logger.warning(
            "Population inversion / negative line opacity detected. "
            "RADMC-3D-compatible behavior is being used: negative optical "
            "depths are set to zero in the escape-probability solve. "
            "RADMC-3D will also suppress negative opacity during final ray "
            "tracing. This output does not include maser amplification."
        )
    aggregate["applied"] = True
    aggregate["total_negative_opacity_entries"] += int(
        iteration_diag["negative_opacity_entries"]
    )
    aggregate["max_negative_opacity_cells_in_iteration"] = max(
        int(aggregate["max_negative_opacity_cells_in_iteration"]),
        int(iteration_diag["negative_opacity_cells"]),
    )
    min_alpha = float(iteration_diag["negative_opacity_min_cm1"])
    current = aggregate["min_negative_alpha0_cm1"]
    aggregate["min_negative_alpha0_cm1"] = (
        min_alpha if current is None else min(float(current), min_alpha)
    )

    by_line = {
        int(row["line_index"]): row
        for row in aggregate["transitions"]
    }
    for transition in iteration_diag["transitions"]:
        line_index = int(transition["line_index"])
        if line_index not in by_line:
            by_line[line_index] = {
                "line_index": line_index,
                "radmc_line_number": int(transition["radmc_line_number"]),
                "upper_level": int(transition["upper_level"]),
                "lower_level": int(transition["lower_level"]),
                "total_negative_entries": 0,
                "max_negative_cells_in_iteration": 0,
                "min_opacity_factor": float("inf"),
                "min_alpha0_cm1": float("inf"),
            }
        row = by_line[line_index]
        row["total_negative_entries"] += int(transition["n_cells"])
        row["max_negative_cells_in_iteration"] = max(
            int(row["max_negative_cells_in_iteration"]),
            int(transition["n_cells"]),
        )
        row["min_opacity_factor"] = min(
            float(row["min_opacity_factor"]),
            float(transition["min_opacity_factor"]),
        )
        row["min_alpha0_cm1"] = min(
            float(row["min_alpha0_cm1"]),
            float(transition["min_alpha0_cm1"]),
        )
    aggregate["transitions"] = [
        by_line[key] for key in sorted(by_line)
    ]


# ---------------------------------------------------------------------------
# Geometry: selected cells, dirs, cell centers
# ---------------------------------------------------------------------------


def _build_tracer_and_geometry(
    mesh,
    nside: int,
    cell_mask: np.ndarray,
):
    if mesh.coord_system == "spherical":
        tracer = SphericalHealpixRayTracer(mesh, nside=int(nside))
    elif mesh.coord_system == "cartesian":
        tracer = CartesianHealpixRayTracer(mesh, nside=int(nside))
    else:
        raise ValueError(
            f"HEALPix SE requires spherical or cartesian mesh, got {mesh.coord_system!r}"
        )
    dirs = np.ascontiguousarray(tracer.dirs, dtype=np.float64)
    cell_idx = np.argwhere(cell_mask).astype(np.int64)
    n_cell = cell_idx.shape[0]
    cell_centers = np.zeros((n_cell, 3), dtype=np.float64)
    for i in range(n_cell):
        idx = cell_idx[i]
        cell_centers[i, :] = tracer.cell_center_xyz(
            int(idx[0]), int(idx[1]), int(idx[2])
        )
    return tracer, dirs, cell_idx, cell_centers


# ---------------------------------------------------------------------------
# Scatter helpers (Numba parallel)
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _scatter_alpha0_to_full(
    alpha0_cand: np.ndarray,        # (Ncand, nlin)
    cell_idx: np.ndarray,      # (Ncand, 3) int64
    nlin: int,
    n0: int,
    n1: int,
    n2: int,
) -> np.ndarray:
    """Scatter selected-cell alpha0 into the full ``(nlin, n0, n1, n2)`` grid."""
    out = np.zeros((nlin, n0, n1, n2), dtype=np.float64)
    n_cell = cell_idx.shape[0]
    for c in prange(n_cell):
        i0 = cell_idx[c, 0]
        i1 = cell_idx[c, 1]
        i2 = cell_idx[c, 2]
        for m in range(nlin):
            out[m, i0, i1, i2] = alpha0_cand[c, m]
    return out


@njit(parallel=True, cache=True)
def _scatter_levelpop_to_full(
    fracpop_cand: np.ndarray,        # (Ncand, nlev)
    n_species_cand: np.ndarray,      # (Ncand,)
    cell_idx: np.ndarray,       # (Ncand, 3) int64
    nlev: int,
    n0: int,
    n1: int,
    n2: int,
) -> np.ndarray:
    """Scatter level populations into the full ``(n0, n1, n2, nlev)`` grid."""
    out = np.zeros((n0, n1, n2, nlev), dtype=np.float64)
    n_cell = cell_idx.shape[0]
    for c in prange(n_cell):
        i0 = cell_idx[c, 0]
        i1 = cell_idx[c, 1]
        i2 = cell_idx[c, 2]
        n_sp = n_species_cand[c]
        for k in range(nlev):
            out[i0, i1, i2, k] = n_sp * fracpop_cand[c, k]
    return out


@njit(parallel=True, cache=True)
def _normalize_and_fill_levelpop_lte(
    levelpop_full: np.ndarray,
    n_species: np.ndarray,
    Tgas: np.ndarray,
    energy_erg: np.ndarray,
    weight: np.ndarray,
) -> None:
    """Normalize solved cells and fill unsolved positive cells with LTE."""

    nlev = levelpop_full.shape[-1]
    populations = levelpop_full.reshape((-1, nlev))
    density = n_species.reshape(-1)
    temperature = Tgas.reshape(-1)
    for cell in prange(density.size):
        n_sp = density[cell]
        if n_sp <= 0.0:
            continue
        population_sum = 0.0
        for level in range(nlev):
            population_sum += populations[cell, level]
        if population_sum > 0.0:
            scale = n_sp / population_sum
            for level in range(nlev):
                populations[cell, level] *= scale
            continue
        if temperature[cell] <= 0.0:
            populations[cell, 0] = n_sp
            continue
        partition = 0.0
        kT = K_B * temperature[cell]
        energy_zero = energy_erg[0]
        for level in range(nlev):
            value = weight[level] * np.exp(
                -(energy_erg[level] - energy_zero) / kT
            )
            populations[cell, level] = value
            partition += value
        if partition > 0.0:
            scale = n_sp / partition
            for level in range(nlev):
                populations[cell, level] *= scale
        else:
            populations[cell, 0] = n_sp


# ---------------------------------------------------------------------------
# Public API: solve and write
# ---------------------------------------------------------------------------


def solve_and_write_healpix_levelpop(
    *,
    rad,
    chemistry_result,
    species: str,
    output_dir: str | Path,
    config: HealpixSEConfig,
) -> Path:
    """Solve non-LTE level populations and write ``levelpop_<species>.dat``.

    Parameters
    ----------
    rad : RadModel
    chemistry_result : ChemistryResult
        Provides ``number_densities`` for the species and its colliders. When
        ``config.species_abundance_floor`` is positive, it must also provide
        the dimensionless abundance in ``abundances[species]``.
    species : str
        Lowercased species name (``"co"``, ``"catom"``, or ``"hco+"``).
    output_dir : path
        Directory in which to stage the installed LAMDA molecule file and write
        ``levelpop_<species>.dat`` and its manifest.
    config : HealpixSEConfig

    Returns
    -------
    Path
        The written ``levelpop_<species>.dat`` path.
    """

    species = str(species).lower().strip()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    levelpop_path = output_dir / f"levelpop_{species}.dat"
    manifest_path = output_dir / f"external_levelpop_manifest_{species}.json"
    levelpop_tmp = output_dir / f"levelpop_{species}.dat.tmp"
    manifest_tmp = output_dir / f"external_levelpop_manifest_{species}.json.tmp"
    for tmp in (levelpop_tmp, manifest_tmp):
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
    if not bool(config.overwrite) and (levelpop_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"External level-population output already exists for {species} in {output_dir}; "
            "set overwrite=True to replace it."
        )
    for stale in (levelpop_path, manifest_path):
        if bool(config.overwrite) and (stale.exists() or stale.is_symlink()):
            # Remove stale products before the solve so a failed production run
            # cannot leave RADMC-3D pointed at an older levelpop/manifest pair.
            stale.unlink()

    molecule_file = install_validated_molecule_file(
        species=species,
        inputs_dir=output_dir,
    )
    molecule = parse_lamda_molecule_file(molecule_file)
    molecule_sha256 = sha256_file(molecule_file)

    expected_colliders = gow17_lamda_colliders(species)
    actual_colliders = list(molecule.collider_names)
    if actual_colliders != expected_colliders:
        raise ValueError(
            f"Molecule file collider order {actual_colliders} does not match "
            f"strict GOW17/LAMDA policy {expected_colliders} for {species}"
        )

    mesh = rad.model.mesh
    if mesh is None:
        raise ValueError("RadModel has no mesh defined")
    transfer_mesh = _radmc_transfer_mesh(mesh)

    # Gas fields (native mesh order).
    Tgas = _gas_field(rad, "gas_temperature", unit="K")
    a_turb = microturbulence_cm_s(rad)
    a_line = molecular_doppler_width_cm_s(Tgas, a_turb, molecule.molweight)
    n_species = _species_density_cm3(chemistry_result, species)
    velocity_xyz = cartesian_velocity_cm_s(rad, basis_mesh=transfer_mesh)
    velocity_range = _velocity_range_diagnostics(velocity_xyz)
    velocity_mesh, velocity_components = _gas_velocity_components_cm_s(rad)
    gas_velocity_sha256 = RadWriter.gas_velocity_binp_sha256(
        velocity_mesh,
        velocity_components,
    )

    if n_species.shape != Tgas.shape or n_species.shape != a_turb.shape:
        raise ValueError(
            "Inconsistent field shapes: "
            f"n_species={n_species.shape}, Tgas={Tgas.shape}, "
            f"a_turb={a_turb.shape}"
        )

    collider_full: list[np.ndarray] = []
    for name in expected_colliders:
        collider_full.append(_collider_density_cm3(chemistry_result, name))
    collider_density_ranges = {
        name: _range_pair(values)
        for name, values in zip(expected_colliders, collider_full, strict=True)
    }

    if not np.all(np.isfinite(n_species)):
        raise ValueError(f"number density for {species} contains non-finite values")
    if np.any(n_species < 0.0):
        raise ValueError(f"number density for {species} contains negative values")
    density_floor = float(config.species_density_floor_cm3)
    if not np.isfinite(density_floor) or density_floor < 0.0:
        raise ValueError(
            "species_density_floor_cm3 must be a non-negative finite value, "
            f"got {density_floor}"
        )
    abundance_floor = float(config.species_abundance_floor)
    species_abundance: np.ndarray | None = None
    if abundance_floor > 0.0:
        species_abundance = _species_abundance_per_h_nucleus(
            chemistry_result,
            species,
        )
        if species_abundance.shape != n_species.shape:
            raise ValueError(
                f"abundance shape {species_abundance.shape} for {species} does "
                f"not match species-density shape {n_species.shape}"
            )
        if not np.all(np.isfinite(species_abundance)):
            raise ValueError(f"abundance for {species} contains non-finite values")
        if np.any(species_abundance < 0.0):
            raise ValueError(f"abundance for {species} contains negative values")

    # The default skips only exact zero-species cells. Configured floors limit
    # the expensive non-LTE iteration; excluded positive-density cells are
    # filled with local LTE populations when the all-cell file is written.
    cell_mask = n_species > density_floor
    if species_abundance is not None:
        cell_mask &= species_abundance >= abundance_floor

    collision_temperature_diagnostics = _collision_temperature_diagnostics(
        molecule,
        Tgas,
        cell_mask,
    )
    for collider, diagnostics in collision_temperature_diagnostics.items():
        below = int(diagnostics["selected_cells_below"])
        above = int(diagnostics["selected_cells_above"])
        if below or above:
            logger.warning(
                "Holding %s collision coefficients at their tabulated "
                "temperature boundaries for %d cells below and %d cells "
                "above the table range",
                collider,
                below,
                above,
            )

    tracer, dirs, cell_idx, cell_centers = _build_tracer_and_geometry(
        transfer_mesh, config.nside, cell_mask,
    )
    n_cell = cell_idx.shape[0]
    n_total_cells = int(np.prod(n_species.shape))
    positive_species = n_species > 0.0
    n_zero_species_cells = int(np.count_nonzero(~positive_species))
    n_below_density_floor_cells = int(
        np.count_nonzero(positive_species & (n_species <= density_floor))
    )
    if species_abundance is None:
        n_below_abundance_floor_cells = 0
    else:
        n_below_abundance_floor_cells = int(
            np.count_nonzero(
                (n_species > density_floor)
                & (species_abundance < abundance_floor)
            )
        )
    n_lte_fallback_cells = int(np.count_nonzero(positive_species & ~cell_mask))
    n_excluded_cells = n_total_cells - n_cell
    logger.info(
        f"HEALPix SE for {species}: nside={config.nside}, n_cells={n_cell}, "
        f"zero_species_cells={n_zero_species_cells}, "
        f"lte_fallback_cells={n_lte_fallback_cells}, "
        f"nlev={molecule.nlev}, nlin={molecule.nlin}, "
        f"colliders={list(molecule.collider_names)}"
    )
    cell_counts = {
        "total_grid_cells": n_total_cells,
        "species_candidate_cells": int(n_cell),
        "species_zero_cells": int(n_zero_species_cells),
        "species_below_density_floor_cells": n_below_density_floor_cells,
        "species_below_abundance_floor_cells": n_below_abundance_floor_cells,
        "lte_fallback_species_cells": n_lte_fallback_cells,
        "solved_nonzero_species_cells": int(n_cell),
        "zero_or_below_floor_species_cells": int(n_excluded_cells),
    }
    if n_cell == 0:
        logger.warning(
            "No cells meet the %s non-LTE selection floors; writing LTE "
            "populations in positive-density cells",
            species,
        )
        flat_per_level = _flat_levelpop_radmc_order(
            mesh=mesh,
            molecule=molecule,
            fracpop_cand=np.empty((0, molecule.nlev), dtype=np.float64),
            n_species_cand=np.empty(0, dtype=np.float64),
            cell_idx=cell_idx,
            n_species=n_species,
            Tgas=Tgas,
        )
        RadWriter.write_levelpop_dat(
            levelpop_tmp,
            levelpop_cm3_radmc_order=flat_per_level,
            level_numbers_1based=np.arange(1, molecule.nlev + 1, dtype=np.int64),
        )
        levelpop_tmp.replace(levelpop_path)
        _write_manifest(
            manifest_path=manifest_tmp,
            species=species,
            config=config,
            molecule=molecule,
            molecule_file=molecule_file,
            molecule_sha256=molecule_sha256,
            levelpop_path=levelpop_path,
            iterations=0,
            final_err=0.0,
            tgas_range=(float(np.min(Tgas)), float(np.max(Tgas))),
            aturb_range=(float(np.min(a_turb)), float(np.max(a_turb))),
            nsp_range=(float(np.min(n_species)), float(np.max(n_species))),
            abundance_range=(
                None
                if species_abundance is None
                else (
                    float(np.min(species_abundance)),
                    float(np.max(species_abundance)),
                )
            ),
            velocity_range=velocity_range,
            gas_velocity_sha256=gas_velocity_sha256,
            beta_range=(1.0, 1.0),
            iteration_history=[],
            converged=True,
            cell_counts=cell_counts,
            collider_density_ranges=collider_density_ranges,
            collision_temperature_diagnostics=collision_temperature_diagnostics,
            maser_diagnostics=_empty_maser_diagnostics(),
            selected_cell_diagnostics=[],
            resumed_from_checkpoint_iteration=None,
        )
        manifest_tmp.replace(manifest_path)
        return levelpop_path

    # Selected-cell views.
    idx0 = cell_idx[:, 0]
    idx1 = cell_idx[:, 1]
    idx2 = cell_idx[:, 2]
    Tgas_cand = np.ascontiguousarray(Tgas[idx0, idx1, idx2])
    a_line_cand = np.ascontiguousarray(a_line[idx0, idx1, idx2])
    n_species_cand = np.ascontiguousarray(n_species[idx0, idx1, idx2])
    collider_dens_cand = np.empty((n_cell, len(expected_colliders)), dtype=np.float64)
    for k, full in enumerate(collider_full):
        collider_dens_cand[:, k] = full[idx0, idx1, idx2]

    tgrids, ntemps, tables = stack_collider_tables(molecule)

    # Chunk size for SE solve (the matrix factor dominates).
    bytes_per_cell = 8.0 * molecule.nlev * molecule.nlev
    budget = max(int(config.memory_budget_gib * (1 << 30) / max(bytes_per_cell, 1.0)), 1)
    se_chunk = min(n_cell, budget)
    checkpoint_interval = int(config.checkpoint_interval)
    if checkpoint_interval < 0:
        raise ValueError(
            f"checkpoint_interval must be non-negative, got {checkpoint_interval}"
        )

    # LTE initialization.
    f = lte_populations(molecule, Tgas_cand)
    checkpoint_fingerprint = _checkpoint_fingerprint(
        species=species,
        config=config,
        molecule=molecule,
        molecule_sha256=molecule_sha256,
        gas_velocity_sha256=gas_velocity_sha256,
        transfer_mesh=transfer_mesh,
        n_species=n_species,
        species_abundance=species_abundance,
        Tgas=Tgas,
        a_turb=a_turb,
        cell_idx=cell_idx,
        collider_dens_cand=collider_dens_cand,
    )

    history: list[dict] = []
    final_err = np.inf
    final_beta_range = (1.0, 1.0)
    last_iter = 0
    converged = False
    maser_diagnostics = _empty_maser_diagnostics()
    resumed_from_checkpoint_iteration: int | None = None
    if bool(config.resume_from_checkpoint):
        loaded = _load_latest_checkpoint(
            output_dir=output_dir,
            species=species,
            fingerprint=checkpoint_fingerprint,
            expected_shape=(n_cell, molecule.nlev),
        )
        if loaded is None:
            logger.info("No HEALPix SE checkpoint found for %s; starting from LTE", species)
        else:
            f = loaded["fracpop"]
            history = loaded["history"]
            maser_diagnostics = loaded["maser_diagnostics"]
            final_err = loaded["final_err"]
            final_beta_range = loaded["final_beta_range"]
            last_iter = loaded["iteration"]
            resumed_from_checkpoint_iteration = int(last_iter)
            logger.info(
                "Resuming HEALPix SE for %s from checkpoint iteration %d",
                species,
                last_iter,
            )
            if last_iter >= int(config.maxiter):
                raise ValueError(
                    f"Latest checkpoint for {species} is already at iteration "
                    f"{last_iter}; increase maxiter above {last_iter} to continue."
                )

    n0, n1, n2 = n_species.shape

    for iteration in range(last_iter + 1, int(config.maxiter) + 1):
        t0 = time.perf_counter()
        logger.info("  iter %3d/%d: computing line-center opacity", iteration, config.maxiter)
        alpha0_raw = compute_line_center_opacity(
            f,
            n_species_cand,
            a_line_cand,
            molecule.iup,
            molecule.ilow,
            molecule.aud,
            molecule.freq_hz,
            molecule.weight,
        )
        iteration_maser = _iteration_maser_diagnostics(
            molecule=molecule,
            fracpop=f,
            alpha0_raw=alpha0_raw,
            iteration=iteration,
        )
        _update_maser_diagnostics(maser_diagnostics, iteration_maser)
        if iteration_maser["applied"]:
            logger.warning(
                f"  iter {iteration:3d}: detected negative line-center opacity in "
                f"{iteration_maser['negative_opacity_cells']} cells "
                f"({iteration_maser['negative_opacity_entries']} line entries); "
                "setting those opacity entries to zero for the RADMC-3D-compatible "
                "escape-probability update"
            )
        # Keep the populations unchanged. Only suppress negative opacity in the
        # transfer update, matching RADMC-3D's no-maser-amplification behavior.
        alpha0_used = np.where(alpha0_raw < 0.0, 0.0, alpha0_raw)

        logger.info("  iter %3d/%d: scattering opacity to full grid", iteration, config.maxiter)
        alpha0_full = _scatter_alpha0_to_full(
            alpha0_used, cell_idx, molecule.nlin, n0, n1, n2,
        )

        logger.info(
            "  iter %3d/%d: computing HEALPix escape probabilities "
            "(chunk_size=%d)",
            iteration,
            config.maxiter,
            int(config.escape_chunk_size),
        )
        beta = compute_escape_probabilities_healpix(
            mesh=transfer_mesh,
            tracer=tracer,
            candidate_idx=cell_idx,
            cell_centers=cell_centers,
            dirs=dirs,
            alpha0_stack=alpha0_full,
            velocity_xyz=velocity_xyz,
            a_line=a_line,
            max_ray_steps=int(config.max_ray_steps),
            chunk_size=int(config.escape_chunk_size),
            spherical_inner_boundary=config.spherical_inner_boundary,
            spherical_theta_boundary=config.spherical_theta_boundary,
        )

        logger.info("  iter %3d/%d: solving statistical equilibrium", iteration, config.maxiter)
        f_new = solve_statistical_equilibrium(
            molecule=molecule,
            Tgas_cand=Tgas_cand,
            n_species_cand=n_species_cand,
            a_line_cand=a_line_cand,
            collider_dens_cand=collider_dens_cand,
            beta=beta,
            tbg_K=float(config.tbg_K),
            tgrids=tgrids,
            ntemps=ntemps,
            tables=tables,
            chunk_size=se_chunk,
        )

        # Under-relaxation is useful for diagnostics on coarse HEALPix angular
        # grids, where the escape-probability update can otherwise oscillate.
        relaxation = float(config.relaxation)
        if not (0.0 < relaxation <= 1.0):
            raise ValueError(f"relaxation must be in (0, 1], got {relaxation}")
        f_next = relaxation * f_new + (1.0 - relaxation) * f
        sums = f_next.sum(axis=1, keepdims=True)
        sums = np.where(sums <= 0.0, 1.0, sums)
        f_next = f_next / sums

        pop_diag = population_change_diagnostics(f_next, f)
        err = float(pop_diag["population_convergence_error"])
        beta_diag = _beta_diagnostics(beta)
        beta_min = float(beta_diag["beta_min"])
        beta_max = float(beta_diag["beta_max"])
        elapsed = time.perf_counter() - t0
        logger.info(
            f"  iter {iteration:3d}/{config.maxiter}: err={err:.3e}  "
            f"tv={pop_diag['population_tv_change_max']:.3e}  "
            f"rel_imp={pop_diag['population_rel_important_max']:.3e}  "
            f"rel_all={pop_diag['population_rel_all_levels_max']:.3e}  "
            f"beta=[{beta_min:.3e}, {beta_max:.3e}]  ({elapsed:.2f} s)"
        )
        iteration_row = {
            "iteration": int(iteration),
            "err": float(err),
            "elapsed_s": float(elapsed),
            "negative_opacity_cells": int(iteration_maser["negative_opacity_cells"]),
            "negative_opacity_entries": int(iteration_maser["negative_opacity_entries"]),
            "negative_opacity_min_cm1": float(iteration_maser["negative_opacity_min_cm1"]),
        }
        iteration_row.update(pop_diag)
        iteration_row.update(beta_diag)
        history.append(iteration_row)
        f = f_next
        final_err = err
        final_beta_range = (beta_min, beta_max)
        last_iter = iteration
        if err < float(config.convcrit):
            converged = True
        should_checkpoint = (
            checkpoint_interval > 0
            and (
                iteration % checkpoint_interval == 0
                or iteration == int(config.maxiter)
                or converged
            )
        )
        if should_checkpoint:
            _write_iteration_checkpoint(
                output_dir=output_dir,
                species=species,
                iteration=iteration,
                fracpop=f,
                fingerprint=checkpoint_fingerprint,
                history=history,
                maser_diagnostics=maser_diagnostics,
                final_err=final_err,
                final_beta_range=final_beta_range,
                converged=converged,
                write_levelpop=bool(config.checkpoint_levelpop),
                mesh=mesh,
                molecule=molecule,
                n_species_cand=n_species_cand,
                cell_idx=cell_idx,
                n_species=n_species,
                Tgas=Tgas,
            )
        if converged:
            break
    else:
        if not bool(config.allow_unconverged):
            raise RuntimeError(
                f"HEALPix SE for {species} did not converge in {config.maxiter} "
                f"iterations (final err={final_err:.3e}, convcrit={config.convcrit:.3e})"
            )
        # Diagnostic laptop runs deliberately stop after a few iterations to
        # inspect error trends. Still write the manifest/levelpop, but mark the
        # populations as unconverged so they cannot be mistaken for production.
        logger.warning(
            "HEALPix SE for %s stopped unconverged after %d iterations "
            "(final err=%.3e, convcrit=%.3e)",
            species,
            config.maxiter,
            final_err,
            config.convcrit,
        )

    alpha0_final_raw = compute_line_center_opacity(
        f,
        n_species_cand,
        a_line_cand,
        molecule.iup,
        molecule.ilow,
        molecule.aud,
        molecule.freq_hz,
        molecule.weight,
    )
    selected_cell_diagnostics = _selected_cell_diagnostics(
        molecule=molecule,
        cell_idx=cell_idx,
        Tgas_cand=Tgas_cand,
        n_species_cand=n_species_cand,
        collider_dens_cand=collider_dens_cand,
        collider_names=list(expected_colliders),
        beta=beta,
        alpha0_raw=alpha0_final_raw,
        fracpop=f,
        tgrids=tgrids,
        ntemps=ntemps,
        tables=tables,
    )

    # Build full level-population grid and write file.
    flat_per_level = _flat_levelpop_radmc_order(
        mesh=mesh,
        molecule=molecule,
        fracpop_cand=f,
        n_species_cand=n_species_cand,
        cell_idx=cell_idx,
        n_species=n_species,
        Tgas=Tgas,
    )
    n_total = flat_per_level.shape[0]

    RadWriter.write_levelpop_dat(
        levelpop_tmp,
        levelpop_cm3_radmc_order=flat_per_level,
        level_numbers_1based=np.arange(1, molecule.nlev + 1, dtype=np.int64),
    )
    levelpop_tmp.replace(levelpop_path)

    _write_manifest(
        manifest_path=manifest_tmp,
        species=species,
        config=config,
        molecule=molecule,
        molecule_file=molecule_file,
        molecule_sha256=molecule_sha256,
        levelpop_path=levelpop_path,
        iterations=last_iter,
        final_err=final_err,
        tgas_range=(float(np.min(Tgas)), float(np.max(Tgas))),
        aturb_range=(float(np.min(a_turb)), float(np.max(a_turb))),
        nsp_range=(float(np.min(n_species)), float(np.max(n_species))),
        abundance_range=(
            None
            if species_abundance is None
            else (
                float(np.min(species_abundance)),
                float(np.max(species_abundance)),
            )
        ),
        velocity_range=velocity_range,
        gas_velocity_sha256=gas_velocity_sha256,
        beta_range=final_beta_range,
        iteration_history=history,
        converged=converged,
        cell_counts=cell_counts,
        collider_density_ranges=collider_density_ranges,
        collision_temperature_diagnostics=collision_temperature_diagnostics,
        maser_diagnostics=maser_diagnostics,
        selected_cell_diagnostics=selected_cell_diagnostics,
        resumed_from_checkpoint_iteration=resumed_from_checkpoint_iteration,
    )
    manifest_tmp.replace(manifest_path)

    logger.info(f"Wrote {levelpop_path} ({n_total} cells, {molecule.nlev} levels)")
    return levelpop_path


def _write_manifest(
    *,
    manifest_path: Path,
    species: str,
    config: HealpixSEConfig,
    molecule: MoleculeData,
    molecule_file: Path,
    molecule_sha256: str,
    levelpop_path: Path,
    iterations: int,
    final_err: float,
    tgas_range: tuple[float, float],
    aturb_range: tuple[float, float],
    nsp_range: tuple[float, float],
    abundance_range: tuple[float, float] | None,
    velocity_range: dict[str, list[float]],
    gas_velocity_sha256: str,
    beta_range: tuple[float, float],
    iteration_history: list[dict],
    converged: bool,
    cell_counts: dict[str, int],
    collider_density_ranges: dict[str, list[float]],
    collision_temperature_diagnostics: dict[str, dict[str, float | int]],
    maser_diagnostics: dict,
    selected_cell_diagnostics: list[dict],
    resumed_from_checkpoint_iteration: int | None,
) -> None:
    payload = {
        "species": species,
        "line_mode": 50,
        "solver": "healpix_escape_probability_statistical_equilibrium",
        "nside": int(config.nside),
        "maxiter": int(config.maxiter),
        "convcrit": float(config.convcrit),
        "escape_chunk_size": int(config.escape_chunk_size),
        "spherical_inner_boundary": config.spherical_inner_boundary,
        "spherical_theta_boundary": config.spherical_theta_boundary,
        "collision_temperature_policy": "nearest_table_boundary",
        "allow_unconverged": bool(config.allow_unconverged),
        "species_density_floor_cm3": float(config.species_density_floor_cm3),
        "species_abundance_floor": float(config.species_abundance_floor),
        "overwrite": bool(config.overwrite),
        "checkpoint_interval": int(config.checkpoint_interval),
        "checkpoint_levelpop": bool(config.checkpoint_levelpop),
        "resume_from_checkpoint": bool(config.resume_from_checkpoint),
        "resumed_from_checkpoint_iteration": (
            None
            if resumed_from_checkpoint_iteration is None
            else int(resumed_from_checkpoint_iteration)
        ),
        "converged": bool(converged),
        "iterations": int(iterations),
        "final_convergence_error": float(final_err),
        "tbg_K": float(config.tbg_K),
        "relaxation": float(config.relaxation),
        "molecule_file": str(molecule_file),
        "molecule_sha256": molecule_sha256,
        "gas_velocity_sha256": gas_velocity_sha256,
        "n_levels": int(molecule.nlev),
        "n_lines": int(molecule.nlin),
        "cell_counts": {
            key: int(value) for key, value in cell_counts.items()
        },
        "colliders": list(molecule.collider_names),
        "collider_density_ranges_cm3": collider_density_ranges,
        "collision_temperature_diagnostics": collision_temperature_diagnostics,
        "temperature_range_K": [float(tgas_range[0]), float(tgas_range[1])],
        "microturbulence_range_cm_s": [float(aturb_range[0]), float(aturb_range[1])],
        "species_density_range_cm3": [float(nsp_range[0]), float(nsp_range[1])],
        "species_abundance_range_per_h_nucleus": (
            None
            if abundance_range is None
            else [float(abundance_range[0]), float(abundance_range[1])]
        ),
        "velocity_range_cm_s": velocity_range,
        "beta_range": [float(beta_range[0]), float(beta_range[1])],
        "levelpop_file": str(levelpop_path),
        "iteration_history": iteration_history,
        "selected_cell_diagnostics": selected_cell_diagnostics,
        "maser_suppression": maser_diagnostics,
        "maser_suppression_applied": bool(maser_diagnostics["applied"]),
        "negative_opacity_entries": int(
            maser_diagnostics["total_negative_opacity_entries"]
        ),
        "negative_opacity_cells": int(
            maser_diagnostics["max_negative_opacity_cells_in_iteration"]
        ),
        "negative_opacity_transitions": maser_diagnostics["transitions"],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "HealpixSEConfig",
    "solve_and_write_healpix_levelpop",
]
