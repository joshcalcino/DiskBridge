from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .config import NonLTELineTransferConfig, SpeciesLineConfig
from .lines_inp import expected_lines_inp_content, normalize_collider_name
from diskbridge.radmc3d.colliders import assert_lamda_collision_order, gow17_lamda_colliders
from diskbridge.radmc3d.data import read_amr_grid_cell_count, read_radmc_binp_field


def _inputs_dir(work_dir: Path) -> Path:
    candidate = work_dir / "radmc3d_inputs"
    return candidate if candidate.exists() else work_dir


def _outputs_dir(work_dir: Path) -> Path:
    candidate = work_dir / "radmc3d_outputs"
    return candidate if candidate.exists() else work_dir


def parse_radmc3d_inp(path: str | Path) -> dict[str, str]:
    """Parse simple ``key = value`` lines from ``radmc3d.inp``."""

    settings: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        settings[key.strip()] = value.strip()
    return settings


def update_key_value_file(path: str | Path, settings: dict[str, str]) -> Path:
    """Update or append RADMC-style ``key = value`` settings."""

    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    seen: set[str] = set()
    updated: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if "=" not in stripped or stripped.startswith("#"):
            updated.append(raw)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in settings:
            updated.append(f"{key} = {settings[key]}")
            seen.add(key)
        else:
            updated.append(raw)
    for key, value in settings.items():
        if key not in seen:
            updated.append(f"{key} = {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(updated) + "\n")
    return path


def write_line_radmc3d_inp(
    path: str | Path,
    *,
    line_mode: int,
    config: NonLTELineTransferConfig,
) -> Path:
    """Write or update RADMC-3D settings for a staged line-transfer run."""

    line_mode = int(line_mode)
    settings = {
        "incl_dust": str(int(config.incl_dust)),
        "incl_lines": "1",
        "lines_mode": str(line_mode),
        "tgas_eq_tdust": "0"
        if abs(line_mode) == 50 or config.use_gow17_tgas
        else "1",
        "itempdecoup": str(int(config.itempdecoup)),
        "rto_style": str(int(config.rto_style)),
        "lines_nonlte_maxiter": str(int(config.lines_nonlte_maxiter)),
        "lines_nonlte_convcrit": f"{float(config.lines_nonlte_convcrit):.6g}",
        "lines_slowlvg_as_alternative": "1" if config.lines_slowlvg_as_alternative else "0",
    }
    return update_key_value_file(path, settings)


def ensure_gas_temperature_for_nonlte(
    work_dir: str | Path,
    radmc3d_settings: dict[str, str],
    *,
    use_gow17_tgas: bool = True,
) -> Path | None:
    """Require GOW17 gas temperature for non-LTE runs when configured."""

    if not use_gow17_tgas:
        return None
    inp = _inputs_dir(Path(work_dir))
    path = inp / "gas_temperature.binp"
    if not path.exists():
        raise FileNotFoundError(
            "Non-LTE line transfer with use_gow17_tgas=True requires "
            "gas_temperature.binp in the staged RADMC-3D input directory."
        )
    if str(radmc3d_settings.get("tgas_eq_tdust", "0")).strip() == "1":
        raise ValueError(
            "Non-LTE GOW17 run is configured with tgas_eq_tdust=1. "
            "Set tgas_eq_tdust=0 so RADMC-3D reads gas_temperature.*."
        )
    return path


def _grid_cell_count(amr_grid: Path) -> int | None:
    if not amr_grid.exists():
        return None
    return read_amr_grid_cell_count(amr_grid)


def check_radmc_binp_file(
    path: str | Path,
    *,
    expected_ncells: int | None,
    components: int,
    nonnegative: bool = False,
    positive: bool = False,
) -> np.ndarray:
    path = Path(path)
    values, ncells = read_radmc_binp_field(path, components=components)
    if ncells is not None and expected_ncells is not None and ncells != expected_ncells:
        raise ValueError(f"{path.name} has {ncells} cells, expected {expected_ncells}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name} contains non-finite values")
    if nonnegative and np.any(values < 0.0):
        raise ValueError(f"{path.name} contains negative values")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{path.name} contains non-positive values")
    return values


def _check_required(path: Path, message: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(message)
    return path


def _density_path(inp: Path, species: str) -> Path | None:
    species = str(species).lower()
    path = inp / f"numberdens_{species}.binp"
    return path if path.exists() else None


def validate_line_run(
    work_dir: str | Path,
    config: NonLTELineTransferConfig,
    species_config: SpeciesLineConfig,
) -> dict:
    """Validate a staged single-species non-LTE line-transfer directory."""

    work = Path(work_dir)
    inp = _inputs_dir(work)
    out = _outputs_dir(work)
    species = str(species_config.species).lower()
    colliders = [normalize_collider_name(c) for c in species_config.colliders]
    if abs(int(species_config.line_mode)) in {3, 4}:
        expected_colliders = gow17_lamda_colliders(species)
        if colliders != expected_colliders:
            raise ValueError(
                "Non-LTE GOW17 line transfer supports only the strict "
                f"GOW17/LAMDA collider order for {species}: {expected_colliders}; "
                f"got {colliders}."
            )

    radmc3d_inp = _check_required(inp / "radmc3d.inp", "Missing radmc3d.inp")
    settings = parse_radmc3d_inp(radmc3d_inp)
    if settings.get("incl_lines") != "1":
        raise ValueError("radmc3d.inp must contain incl_lines = 1")
    if int(settings.get("lines_mode", "0")) != int(species_config.line_mode):
        raise ValueError("radmc3d.inp lines_mode does not match species config")
    if abs(int(species_config.line_mode)) in {3, 4} and not colliders:
        raise ValueError("non-LTE line modes require at least one collider")
    gas_temp = ensure_gas_temperature_for_nonlte(
        work,
        settings,
        use_gow17_tgas=config.use_gow17_tgas,
    )

    lines_inp = _check_required(inp / "lines.inp", "Missing lines.inp")
    actual = lines_inp.read_text()
    expected = expected_lines_inp_content(species_config)
    if actual != expected:
        raise ValueError("lines.inp does not match the requested species/collider setup")

    molecule = _check_required(
        inp / f"molecule_{species}.inp",
        f"Missing molecule_{species}.inp",
    )
    if colliders:
        assert_lamda_collision_order(molecule, colliders)
    emitter = _density_path(inp, species)
    if emitter is None:
        raise FileNotFoundError(f"Missing numberdens_{species}.binp")

    collider_paths = {}
    for collider in colliders:
        path = _density_path(inp, collider)
        if path is None:
            raise FileNotFoundError(f"Missing numberdens_{collider}.binp")
        collider_paths[collider] = path

    gas_velocity = inp / "gas_velocity.binp"
    if not gas_velocity.exists():
        if config.require_gas_velocity:
            raise FileNotFoundError("Missing gas_velocity.binp")
        gas_velocity = None

    _check_required(inp / "amr_grid.inp", "Missing amr_grid.inp")
    ncells = _grid_cell_count(inp / "amr_grid.inp")
    if gas_temp is not None:
        check_radmc_binp_file(
            gas_temp,
            expected_ncells=ncells,
            components=1,
            positive=True,
        )
    check_radmc_binp_file(
        emitter,
        expected_ncells=ncells,
        components=1,
        nonnegative=True,
    )
    for path in collider_paths.values():
        check_radmc_binp_file(
            path,
            expected_ncells=ncells,
            components=1,
            nonnegative=True,
        )
    if gas_velocity is not None:
        check_radmc_binp_file(gas_velocity, expected_ncells=ncells, components=3)

    return {
        "work_dir": str(work),
        "input_dir": str(inp),
        "output_dir": str(out),
        "species": species,
        "line_mode": int(species_config.line_mode),
        "colliders": colliders,
        "molecule_file": str(molecule),
        "gas_temperature_file": str(gas_temp) if gas_temp is not None else None,
        "gas_velocity_file": str(gas_velocity) if gas_velocity is not None else None,
        "number_density_files": {
            species: str(emitter),
            **{name: str(path) for name, path in collider_paths.items()},
        },
        "ncells": ncells,
    }
