from __future__ import annotations

from pathlib import Path
import glob
import json

import numpy as np

from .config import NonLTELineTransferConfig, SpeciesLineConfig
from .lines_inp import expected_lines_inp_content, normalize_collider_name
from diskbridge.radmc3d.colliders import assert_lamda_collision_order, gow17_lamda_colliders


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

    settings = {
        "incl_dust": str(int(config.incl_dust)),
        "incl_lines": "1",
        "lines_mode": str(int(line_mode)),
        "tgas_eq_tdust": "0" if config.use_gow17_tgas else "1",
        "itempdecoup": str(int(config.itempdecoup)),
        "rto_style": str(int(config.rto_style)),
        "lines_nonlte_maxiter": str(int(config.lines_nonlte_maxiter)),
        "lines_nonlte_convcrit": f"{float(config.lines_nonlte_convcrit):.6g}",
        "lines_slowlvg_as_alternative": "1" if config.lines_slowlvg_as_alternative else "0",
    }
    return update_key_value_file(path, settings)


def _find_one(base: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        for name in glob.glob(str(base / pattern)):
            path = Path(name)
            if path.exists():
                return path
    return None


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
    path = _find_one(inp, ["gas_temperature.binp", "gas_temperature.inp"])
    if path is None:
        raise FileNotFoundError(
            "Non-LTE line transfer with use_gow17_tgas=True requires "
            "gas_temperature.binp or gas_temperature.inp in the staged RADMC-3D input directory."
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
    lines = [line.strip() for line in amr_grid.read_text().splitlines() if line.strip()]
    if len(lines) < 6:
        return None
    parts = lines[5].split()
    if len(parts) < 3:
        return None
    dims = [int(float(x)) for x in parts[:3]]
    return int(dims[0] * dims[1] * dims[2])


def _read_scalar_values(path: Path) -> tuple[np.ndarray | None, int | None]:
    if path.suffix in {".binp", ".bdat"}:
        with path.open("rb") as f:
            n_header = 4 if path.suffix == ".bdat" else 3
            header = np.fromfile(f, dtype=np.int64, count=n_header)
            if header.size < 3:
                return None, None
            ncells = int(header[2])
            data = np.fromfile(f, dtype=np.float64)
        return data, ncells
    text = path.read_text().split()
    if len(text) < 2:
        return None, None
    ncells = int(float(text[1]))
    vals = np.asarray([float(x) for x in text[2:]], dtype=float)
    return vals, ncells


def _check_scalar_file(
    path: Path,
    *,
    expected_ncells: int | None,
    nonnegative: bool = False,
    positive: bool = False,
) -> None:
    values, ncells = _read_scalar_values(path)
    if ncells is not None and expected_ncells is not None and ncells != expected_ncells:
        raise ValueError(f"{path.name} has {ncells} cells, expected {expected_ncells}")
    if values is None or values.size == 0:
        return
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name} contains non-finite values")
    if nonnegative and np.any(values < 0.0):
        raise ValueError(f"{path.name} contains negative values")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{path.name} contains non-positive values")


def _check_required(path: Path, message: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(message)
    return path


def _density_path(inp: Path, species: str) -> Path | None:
    species = str(species).lower()
    return _find_one(inp, [f"numberdens_{species}.binp", f"numberdens_{species}.inp"])


def run_line_preflight(
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
        raise FileNotFoundError(f"Missing numberdens_{species}.binp or .inp")

    collider_paths = {}
    for collider in colliders:
        path = _density_path(inp, collider)
        if path is None:
            raise FileNotFoundError(f"Missing numberdens_{collider}.binp or .inp")
        collider_paths[collider] = path

    if config.require_gas_velocity and _find_one(inp, ["gas_velocity.binp", "gas_velocity.inp"]) is None:
        raise FileNotFoundError("Missing gas_velocity.binp or gas_velocity.inp")

    _check_required(inp / "amr_grid.inp", "Missing amr_grid.inp")
    ncells = _grid_cell_count(inp / "amr_grid.inp")
    if gas_temp is not None:
        _check_scalar_file(gas_temp, expected_ncells=ncells, positive=True)
    _check_scalar_file(emitter, expected_ncells=ncells, nonnegative=True)
    for path in collider_paths.values():
        _check_scalar_file(path, expected_ncells=ncells, nonnegative=True)

    return {
        "work_dir": str(work),
        "input_dir": str(inp),
        "output_dir": str(out),
        "species": species,
        "line_mode": int(species_config.line_mode),
        "colliders": colliders,
        "molecule_file": str(molecule),
        "gas_temperature_file": str(gas_temp) if gas_temp is not None else None,
        "number_density_files": {
            species: str(emitter),
            **{name: str(path) for name, path in collider_paths.items()},
        },
        "ncells": ncells,
    }
