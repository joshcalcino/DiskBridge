"""Validation checks for a staged RADMC-3D ``lines_mode = 50`` run."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .validation import (
    _check_required,
    _grid_cell_count,
    check_radmc_binp_file,
    parse_radmc3d_inp,
)


def _inputs_dir(work_dir: Path) -> Path:
    candidate = work_dir / "radmc3d_inputs"
    return candidate if candidate.exists() else work_dir


def _read_levelpop_dat(path: Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return (n_cells, n_levels, levels_1based, populations[n_cells, n_levels])."""
    with path.open("r") as f:
        tokens: list[str] = []
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            tokens.extend(stripped.split())
    if len(tokens) < 3:
        raise ValueError(f"{path} is too short to be a valid levelpop file")
    _fmt = int(tokens[0])
    n_cells = int(tokens[1])
    n_levels = int(tokens[2])
    expected = 3 + n_levels + n_cells * n_levels
    if len(tokens) < expected:
        raise ValueError(
            f"{path}: expected at least {expected} tokens, got {len(tokens)}"
        )
    levels = np.asarray(tokens[3 : 3 + n_levels], dtype=np.int64)
    payload = np.asarray(tokens[3 + n_levels : 3 + n_levels + n_cells * n_levels], dtype=np.float64)
    populations = payload.reshape(n_cells, n_levels)
    return n_cells, n_levels, levels, populations


def _read_lines_inp(path: Path) -> tuple[str, int]:
    """Return ``(species, n_colliders_declared)`` from a single-species ``lines.inp``."""
    text_lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(text_lines) < 3:
        raise ValueError(f"{path} is too short to be a valid lines.inp")
    if text_lines[0] != "2":
        raise ValueError(f"{path}: expected first token '2', got {text_lines[0]!r}")
    if text_lines[1] != "1":
        raise ValueError(
            f"{path}: external-population staging requires exactly one species; "
            f"got {text_lines[1]!r}"
        )
    parts = text_lines[2].split()
    if len(parts) < 5:
        raise ValueError(f"{path}: malformed species declaration: {text_lines[2]!r}")
    species = parts[0].lower()
    n_coll = int(parts[4])
    return species, n_coll


def validate_external_population_run(
    work_dir: str | Path,
    species: str,
) -> dict:
    """Validate a staged ``lines_mode = 50`` RADMC-3D directory."""

    work = Path(work_dir)
    inp = _inputs_dir(work)
    species = str(species).lower().strip()

    radmc3d_inp = _check_required(inp / "radmc3d.inp", "Missing radmc3d.inp")
    settings = parse_radmc3d_inp(radmc3d_inp)
    if int(settings.get("lines_mode", "0")) != 50:
        raise ValueError(
            f"radmc3d.inp must set lines_mode = 50 for external populations; "
            f"got {settings.get('lines_mode')!r}"
        )
    if settings.get("incl_lines") != "1":
        raise ValueError("radmc3d.inp must contain incl_lines = 1")
    if str(settings.get("tgas_eq_tdust", "0")).strip() != "0":
        raise ValueError(
            "External-population runs require tgas_eq_tdust = 0 (RADMC-3D reads "
            "gas_temperature.* for line broadening)."
        )

    lines_inp = _check_required(inp / "lines.inp", "Missing lines.inp")
    lines_species, n_colliders = _read_lines_inp(lines_inp)
    if lines_species != species:
        raise ValueError(
            f"lines.inp species {lines_species!r} does not match expected {species!r}"
        )
    if n_colliders != 0:
        raise ValueError(
            "External-population lines.inp must declare zero RADMC collision "
            f"partners; got {n_colliders}"
        )

    molecule = _check_required(
        inp / f"molecule_{species}.inp",
        f"Missing molecule_{species}.inp",
    )
    levelpop = _check_required(
        inp / f"levelpop_{species}.dat",
        f"Missing levelpop_{species}.dat",
    )
    emitter = _check_required(
        inp / f"numberdens_{species}.binp",
        f"Missing numberdens_{species}.binp",
    )
    gas_temp = _check_required(
        inp / "gas_temperature.binp",
        "Missing gas_temperature.binp",
    )
    gas_vel = _check_required(inp / "gas_velocity.binp", "Missing gas_velocity.binp")
    micro = _check_required(inp / "microturbulence.binp", "Missing microturbulence.binp")

    amr = _check_required(inp / "amr_grid.inp", "Missing amr_grid.inp")
    ncells = _grid_cell_count(amr)

    emitter_values = check_radmc_binp_file(
        emitter, expected_ncells=ncells, components=1, nonnegative=True
    )
    check_radmc_binp_file(
        gas_temp,
        expected_ncells=ncells,
        components=1,
        positive=True,
    )
    check_radmc_binp_file(
        micro,
        expected_ncells=ncells,
        components=1,
        nonnegative=True,
    )
    check_radmc_binp_file(gas_vel, expected_ncells=ncells, components=3)

    n_cells_lp, n_levels_lp, levels, populations = _read_levelpop_dat(levelpop)
    if ncells is not None and n_cells_lp != ncells:
        raise ValueError(
            f"{levelpop.name} has {n_cells_lp} cells, amr_grid expects {ncells}"
        )
    expected_levels = np.arange(1, n_levels_lp + 1, dtype=np.int64)
    if not np.array_equal(levels, expected_levels):
        raise ValueError(
            f"{levelpop.name} level list must be 1..{n_levels_lp}; got {levels.tolist()}"
        )
    if not np.all(np.isfinite(populations)):
        raise ValueError(f"{levelpop.name} contains non-finite populations")
    if np.any(populations < 0.0):
        raise ValueError(f"{levelpop.name} contains negative populations")

    if emitter_values is not None:
        if emitter_values.size != n_cells_lp:
            raise ValueError(
                f"numberdens cell count {emitter_values.size} does not match "
                f"levelpop cell count {n_cells_lp}"
            )
        level_sums = populations.sum(axis=1)
        ref = emitter_values
        scale = np.maximum(ref, 1.0)
        diff = np.abs(level_sums - ref) / scale
        worst = float(diff.max()) if diff.size else 0.0
        if worst > 1.0e-6:
            n_bad = int(np.count_nonzero(diff > 1.0e-6))
            raise ValueError(
                f"levelpop sum disagrees with numberdens in {n_bad} cells "
                f"(max relative error {worst:.3e})"
            )

    return {
        "work_dir": str(work),
        "input_dir": str(inp),
        "species": species,
        "line_mode": 50,
        "molecule_file": str(molecule),
        "levelpop_file": str(levelpop),
        "gas_temperature_file": str(gas_temp),
        "gas_velocity_file": str(gas_vel),
        "microturbulence_file": str(micro),
        "numberdens_file": str(emitter),
        "ncells": ncells,
        "n_levels": int(n_levels_lp),
    }


__all__ = ["validate_external_population_run"]
