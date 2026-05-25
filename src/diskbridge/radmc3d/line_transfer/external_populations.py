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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
from numba import njit, prange

from diskbridge._constants import K_B, M_H
from diskbridge._logging import logger
from diskbridge.radmc3d.colliders import gow17_lamda_colliders
from diskbridge.radmc3d.writer import RadWriter
from diskbridge.chemistry.shielding.healpix_columns import (
    SphericalHealpixRayTracer,
    CartesianHealpixRayTracer,
)
from .escape_healpix import compute_escape_probabilities_healpix
from .molecular_rates import (
    MoleculeData,
    assert_temperature_in_collision_range,
    lte_populations,
    parse_lamda_molecule_file,
    stack_collider_tables,
)
from .se_solver import (
    compute_line_center_opacity,
    convergence_error,
    solve_statistical_equilibrium,
)


@dataclass(frozen=True)
class HealpixSEConfig:
    """Configuration for the 3D HEALPix non-LTE solver."""

    nside: int = 4
    maxiter: int = 60
    convcrit: float = 1.0e-5
    tbg_K: float = 2.7255
    memory_budget_gib: float = 8.0
    max_ray_steps: int = 200_000
    relaxation: float = 1.0


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


def _velocity_xyz_cm_s(rad) -> np.ndarray:
    """Build a Cartesian velocity field ``v_xyz[n0, n1, n2, 3]`` in cm/s."""
    mesh = rad.model.mesh
    gas = rad.model.gas

    if mesh.coord_system == "cartesian":
        vx = _gas_field(rad, "vx", unit="cm/s") if "vx" in gas else None
        vy = _gas_field(rad, "vy", unit="cm/s") if "vy" in gas else None
        vz = _gas_field(rad, "vz", unit="cm/s") if "vz" in gas else None
        if vx is None or vy is None or vz is None:
            raise KeyError(
                "Cartesian mesh requires model.gas['vx', 'vy', 'vz'] for "
                "Cartesian-velocity HEALPix integration"
            )
        out = np.empty(vx.shape + (3,), dtype=np.float64)
        out[..., 0] = vx
        out[..., 1] = vy
        out[..., 2] = vz
        return np.ascontiguousarray(out)

    if mesh.coord_system != "spherical":
        raise ValueError(
            f"Unsupported coordinate system for HEALPix SE: {mesh.coord_system!r}"
        )

    vr = _gas_field(rad, "vr", unit="cm/s")
    vtheta = _gas_field(rad, "vtheta", unit="cm/s")
    vphi = _gas_field(rad, "vphi", unit="cm/s")
    vx, vy, vz = mesh.spherical_vector_components_to_cartesian(
        vr,
        vtheta,
        vphi,
        axis_order=mesh.axis_names(),
    )

    out = np.empty(vr.shape + (3,), dtype=np.float64)
    out[..., 0] = vx
    out[..., 1] = vy
    out[..., 2] = vz
    return np.ascontiguousarray(out)


def _microturbulence_cm_s(rad) -> np.ndarray:
    """Return microturbulence ``a_turb`` (cm/s) on the mesh.

    RADMC-3D reads ``microturbulence.*`` as the turbulent line-width parameter
    ``a_turb`` and combines it as ``sqrt(a_turb**2 + 2 k T / m_mol)``. Use the
    same convention in the external population solver.
    """
    return _gas_field(rad, "microturbulence", unit="cm/s")


def _build_a_line(Tgas: np.ndarray, a_turb: np.ndarray, molweight: float) -> np.ndarray:
    """Total Doppler line-width parameter in cm/s.

    ``a_line = sqrt(a_turb^2 + 2 k T / m_mol)``.
    """
    m_mol = molweight * M_H
    thermal_sq = 2.0 * K_B * Tgas / m_mol
    return np.sqrt(np.maximum(a_turb * a_turb + thermal_sq, 0.0))


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


# ---------------------------------------------------------------------------
# Public API: solve and write
# ---------------------------------------------------------------------------


def solve_and_write_healpix_levelpop(
    *,
    rad,
    chemistry_result,
    species: str,
    molecule_file: str | Path,
    output_dir: str | Path,
    config: HealpixSEConfig,
) -> Path:
    """Solve non-LTE level populations and write ``levelpop_<species>.dat``.

    Parameters
    ----------
    rad : RadModel
    chemistry_result : ChemistryResult
        Provides ``number_densities`` for the species and its colliders.
    species : str
        Lowercased species name (``"co"``, ``"catom"``, or ``"hco+"``).
    molecule_file : path
        Validated molecule file (collider order matches GOW17/LAMDA policy).
    output_dir : path
        Directory in which to write ``levelpop_<species>.dat`` and the manifest.
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

    molecule = parse_lamda_molecule_file(molecule_file)

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

    # Gas fields (native mesh order).
    Tgas = _gas_field(rad, "gas_temperature", unit="K")
    a_turb = _microturbulence_cm_s(rad)
    a_line = _build_a_line(Tgas, a_turb, molecule.molweight)
    n_species = _species_density_cm3(chemistry_result, species)
    velocity_xyz = _velocity_xyz_cm_s(rad)

    if n_species.shape != Tgas.shape or n_species.shape != a_turb.shape:
        raise ValueError(
            "Inconsistent field shapes: "
            f"n_species={n_species.shape}, Tgas={Tgas.shape}, "
            f"a_turb={a_turb.shape}"
        )

    collider_full: list[np.ndarray] = []
    for name in expected_colliders:
        collider_full.append(_collider_density_cm3(chemistry_result, name))

    if not np.all(np.isfinite(n_species)):
        raise ValueError(f"number density for {species} contains non-finite values")
    # RADMC-3D reads the complete numberdens_<species>.binp array. Keep the
    # external solver on that same full grid without density-based pruning.
    cell_mask = np.ones(n_species.shape, dtype=bool)

    assert_temperature_in_collision_range(molecule, Tgas, cell_mask)

    tracer, dirs, cell_idx, cell_centers = _build_tracer_and_geometry(
        mesh, config.nside, cell_mask,
    )
    n_cell = cell_idx.shape[0]
    logger.info(
        f"HEALPix SE for {species}: nside={config.nside}, n_cells={n_cell}, "
        f"nlev={molecule.nlev}, nlin={molecule.nlin}, "
        f"colliders={list(molecule.collider_names)}"
    )

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

    # LTE initialization.
    f = lte_populations(molecule, Tgas_cand)

    history: list[dict] = []
    final_err = np.inf
    final_beta_range = (1.0, 1.0)
    last_iter = 0

    n0, n1, n2 = n_species.shape

    for iteration in range(1, int(config.maxiter) + 1):
        t0 = time.perf_counter()
        alpha0_cand = compute_line_center_opacity(
            f,
            n_species_cand,
            a_line_cand,
            molecule.iup,
            molecule.ilow,
            molecule.aud,
            molecule.freq_hz,
            molecule.weight,
        )
        # RADMC-3D's LVG path does not abort on masering cells; it removes the
        # negative opacity contribution from the transfer update. Match that
        # convention here and record the event in the iteration history.
        negative_opacity_entries = int(np.count_nonzero(alpha0_cand < 0.0))
        negative_opacity_cells = int(np.count_nonzero(np.any(alpha0_cand < 0.0, axis=1)))
        min_alpha = float(np.min(alpha0_cand))
        if negative_opacity_entries:
            logger.warning(
                f"  iter {iteration:3d}: detected negative line-center opacity in "
                f"{negative_opacity_cells} cells ({negative_opacity_entries} line entries); "
                "setting those opacity entries to zero for the escape-probability update"
            )
            alpha0_cand = np.where(alpha0_cand < 0.0, 0.0, alpha0_cand)

        alpha0_full = _scatter_alpha0_to_full(
            alpha0_cand, cell_idx, molecule.nlin, n0, n1, n2,
        )

        beta = compute_escape_probabilities_healpix(
            mesh=mesh,
            tracer=tracer,
            candidate_idx=cell_idx,
            cell_centers=cell_centers,
            dirs=dirs,
            alpha0_stack=alpha0_full,
            velocity_xyz=velocity_xyz,
            a_line=a_line,
            max_ray_steps=int(config.max_ray_steps),
        )

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

        err = convergence_error(f_next, f)
        beta_min = float(np.min(beta))
        beta_max = float(np.max(beta))
        elapsed = time.perf_counter() - t0
        logger.info(
            f"  iter {iteration:3d}/{config.maxiter}: err={err:.3e}  "
            f"beta=[{beta_min:.3e}, {beta_max:.3e}]  ({elapsed:.2f} s)"
        )
        history.append({
            "iteration": int(iteration),
            "err": float(err),
            "beta_min": beta_min,
            "beta_max": beta_max,
            "elapsed_s": float(elapsed),
            "negative_opacity_cells": negative_opacity_cells,
            "negative_opacity_entries": negative_opacity_entries,
            "negative_opacity_min_cm1": min_alpha,
        })
        f = f_next
        final_err = err
        final_beta_range = (beta_min, beta_max)
        last_iter = iteration
        if err < float(config.convcrit):
            break
    else:
        raise RuntimeError(
            f"HEALPix SE for {species} did not converge in {config.maxiter} "
            f"iterations (final err={final_err:.3e}, convcrit={config.convcrit:.3e})"
        )

    # Build full level-population grid and write file.
    levelpop_full = _scatter_levelpop_to_full(
        f, n_species_cand, cell_idx, molecule.nlev, n0, n1, n2,
    )
    flat_per_level = np.stack(
        [RadWriter.flatten_scalar_to_radmc_order(mesh, levelpop_full[..., k])
         for k in range(molecule.nlev)],
        axis=1,
    )
    n_total = flat_per_level.shape[0]
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

    RadWriter.write_levelpop_dat(
        levelpop_path,
        levelpop_cm3_radmc_order=flat_per_level,
        level_numbers_1based=np.arange(1, molecule.nlev + 1, dtype=np.int64),
    )

    _write_manifest(
        manifest_path=manifest_path,
        species=species,
        config=config,
        molecule=molecule,
        molecule_file=Path(molecule_file),
        levelpop_path=levelpop_path,
        iterations=last_iter,
        final_err=final_err,
        tgas_range=(float(np.min(Tgas)), float(np.max(Tgas))),
        aturb_range=(float(np.min(a_turb)), float(np.max(a_turb))),
        nsp_range=(float(np.min(n_species_cand)), float(np.max(n_species_cand))),
        beta_range=final_beta_range,
        iteration_history=history,
    )

    logger.info(f"Wrote {levelpop_path} ({n_total} cells, {molecule.nlev} levels)")
    return levelpop_path


def _write_levelpop_all_zero(*, mesh, molecule, n_cells_shape, path) -> None:
    """Write a levelpop file with all-zero populations."""
    n_total = int(np.prod(n_cells_shape))
    pop = np.zeros((n_total, molecule.nlev), dtype=np.float64)
    RadWriter.write_levelpop_dat(
        path,
        levelpop_cm3_radmc_order=pop,
        level_numbers_1based=np.arange(1, molecule.nlev + 1, dtype=np.int64),
    )


def _write_manifest(
    *,
    manifest_path: Path,
    species: str,
    config: HealpixSEConfig,
    molecule: MoleculeData,
    molecule_file: Path,
    levelpop_path: Path,
    iterations: int,
    final_err: float,
    tgas_range: tuple[float, float],
    aturb_range: tuple[float, float],
    nsp_range: tuple[float, float],
    beta_range: tuple[float, float],
    iteration_history: list[dict],
) -> None:
    payload = {
        "species": species,
        "line_mode": 50,
        "solver": "healpix_escape_probability_statistical_equilibrium",
        "nside": int(config.nside),
        "maxiter": int(config.maxiter),
        "convcrit": float(config.convcrit),
        "iterations": int(iterations),
        "final_convergence_error": float(final_err),
        "tbg_K": float(config.tbg_K),
        "relaxation": float(config.relaxation),
        "molecule_file": str(molecule_file),
        "n_levels": int(molecule.nlev),
        "n_lines": int(molecule.nlin),
        "colliders": list(molecule.collider_names),
        "temperature_range_K": [float(tgas_range[0]), float(tgas_range[1])],
        "microturbulence_range_cm_s": [float(aturb_range[0]), float(aturb_range[1])],
        "species_density_range_cm3": [float(nsp_range[0]), float(nsp_range[1])],
        "beta_range": [float(beta_range[0]), float(beta_range[1])],
        "levelpop_file": str(levelpop_path),
        "iteration_history": iteration_history,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = [
    "HealpixSEConfig",
    "solve_and_write_healpix_levelpop",
]
