# db-keywords: gow17, nonlte, line-transfer, gas-temperature, config, radmc3d, field, io, serialization
# db-role: helper
# db-scope: package
# db-purpose: Package module for gow17, nonlte, line-transfer, gas-temperature.

from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np
from diskbridge.utils import sha256_file

from .config import NonLTELineTransferConfig, SpeciesLineConfig
from .lines_inp import normalize_collider_name, write_lines_inp
from .validation import validate_line_run, write_line_radmc3d_inp
from .external_validation import validate_external_population_run
from diskbridge.radmc3d.colliders import (
    gow17_lamda_colliders,
    install_validated_molecule_file,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_manifest_path(path_value: str, *, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def _link_or_copy(src: Path, dst: Path, *, copy_mode: str) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError("copy_mode must be 'symlink' or 'copy'")
    return dst


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _stage_optional(
    source_dir: Path,
    work_inputs: Path,
    names: list[str],
    *,
    copy_mode: str,
) -> dict[str, str]:
    staged = {}
    for name in names:
        src = source_dir / name
        if src.exists():
            dst = _link_or_copy(src, work_inputs / name, copy_mode=copy_mode)
            staged[name] = str(dst)
    return staged


def _stage_required_one(
    sources: list[Path],
    dst_dir: Path,
    *,
    label: str,
    copy_mode: str,
) -> Path:
    src = _first_existing(sources)
    if src is None:
        raise FileNotFoundError(f"Missing required line-transfer input: {label}")
    dst = dst_dir / src.name
    return _link_or_copy(src, dst, copy_mode=copy_mode)


def _stage_clipped_gas_temperature(
    src: Path,
    dst: Path,
    *,
    tmin: float,
    tmax: float,
) -> dict[str, float | int | str]:
    if src.suffix != ".binp":
        raise ValueError("line-transfer gas-temperature clipping currently requires gas_temperature.binp")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f:
        header = np.fromfile(f, dtype=np.int64, count=3)
        if header.size != 3:
            raise ValueError(f"gas temperature file has an incomplete binary header: {src}")
        if int(header[1]) != 8:
            raise ValueError(f"gas temperature clipping requires float64 data: {src}")
        data = np.fromfile(f, dtype=np.float64)
    before_min = float(np.nanmin(data)) if data.size else float("nan")
    before_max = float(np.nanmax(data)) if data.size else float("nan")
    clipped = np.clip(data, float(tmin), float(tmax))
    n_clipped = int(np.count_nonzero(clipped != data))
    with open(dst, "wb") as f:
        header.astype(np.int64).tofile(f)
        clipped.astype(np.float64).tofile(f)
    return {
        "source": str(src),
        "staged": str(dst),
        "before_min_K": before_min,
        "before_max_K": before_max,
        "after_min_K": float(np.nanmin(clipped)) if clipped.size else float("nan"),
        "after_max_K": float(np.nanmax(clipped)) if clipped.size else float("nan"),
        "n_clipped": n_clipped,
    }


# db-keywords: nonlte, line-transfer, radmc3d
# db-role: entrypoint
def prepare_nonlte_line_run(
    *,
    source_inputs_dir: str | Path,
    work_dir: str | Path,
    config: NonLTELineTransferConfig,
    species_config: SpeciesLineConfig,
    chemistry_inputs_dir: str | Path | None = None,
    copy_mode: str = "symlink",
    exist_ok: bool = False,
) -> Path:
    """Create and validate an isolated RADMC-3D non-LTE line-transfer work dir."""

    source_inputs = Path(source_inputs_dir)
    if not source_inputs.exists():
        raise FileNotFoundError(f"source_inputs_dir does not exist: {source_inputs}")
    chemistry_inputs = Path(chemistry_inputs_dir) if chemistry_inputs_dir is not None else source_inputs
    if not chemistry_inputs.exists():
        raise FileNotFoundError(f"chemistry_inputs_dir does not exist: {chemistry_inputs}")
    source_run = source_inputs.parent if source_inputs.name == "radmc3d_inputs" else source_inputs
    source_outputs = source_run / "radmc3d_outputs"

    work = Path(work_dir)
    if work.exists() and not exist_ok:
        raise FileExistsError(f"line-transfer work directory already exists: {work}")
    work_inputs = work / "radmc3d_inputs"
    work_outputs = work / "radmc3d_outputs"
    work_inputs.mkdir(parents=True, exist_ok=True)
    work_outputs.mkdir(parents=True, exist_ok=True)

    species = str(species_config.species).lower()
    colliders = [normalize_collider_name(c) for c in species_config.colliders]
    if abs(int(species_config.line_mode)) in {3, 4}:
        expected_colliders = gow17_lamda_colliders(species)
        if colliders != expected_colliders:
            raise ValueError(
                "Non-LTE GOW17 line-transfer staging supports only "
                f"gow17_lamda colliders for {species}: {expected_colliders}; "
                f"got {colliders}."
            )
    staged: dict[str, str] = {}
    tgas_clip_info: dict[str, float | int | str] | None = None
    clip_line_tgas = (
        bool(config.use_gow17_tgas)
        and not bool(config.tgas_eq_tdust)
        and abs(int(species_config.line_mode)) in {3, 4}
    )

    for name in (
        "amr_grid.inp",
        "wavelength_micron.inp",
        "dust_density.binp",
        "dustopac.inp",
        "gas_velocity.binp",
        "gas_temperature.binp",
        "microturbulence.binp",
        "stars.inp",
        "external_source.inp",
    ):
        if clip_line_tgas and name.startswith("gas_temperature"):
            continue
        search_dirs = [source_inputs]
        if name.startswith(("gas_temperature", "numberdens_")) and chemistry_inputs != source_inputs:
            search_dirs.insert(0, chemistry_inputs)
        path = _first_existing([d / name for d in search_dirs])
        if path is not None:
            staged[name] = str(_link_or_copy(path, work_inputs / name, copy_mode=copy_mode))

    if clip_line_tgas:
        tgas_src = _first_existing(
            [
                chemistry_inputs / "gas_temperature.binp",
                source_inputs / "gas_temperature.binp",
            ]
        )
        if tgas_src is not None:
            tgas_clip_info = _stage_clipped_gas_temperature(
                tgas_src,
                work_inputs / "gas_temperature.binp",
                tmin=float(config.line_tgas_min_K),
                tmax=float(config.line_tgas_max_K),
            )
            staged["gas_temperature.binp"] = str(work_inputs / "gas_temperature.binp")

    for opac in sorted(source_inputs.glob("dustkappa_*.inp")):
        staged[opac.name] = str(_link_or_copy(opac, work_inputs / opac.name, copy_mode=copy_mode))

    dust_temp = _first_existing(
        [
            source_outputs / "temperature" / "dust_temperature.bdat",
            source_outputs / "dust_temperature.bdat",
            source_inputs / "dust_temperature.bdat",
            source_run / "dust_temperature.bdat",
        ]
    )
    if dust_temp is not None:
        staged[dust_temp.name] = str(
            _link_or_copy(dust_temp, work_outputs / "dust_temperature.bdat", copy_mode=copy_mode)
        )

    _stage_required_one(
        [
            chemistry_inputs / f"numberdens_{species}.binp",
            source_inputs / f"numberdens_{species}.binp",
        ],
        work_inputs,
        label=f"numberdens_{species}",
        copy_mode=copy_mode,
    )
    for collider in colliders:
        _stage_required_one(
            [
                chemistry_inputs / f"numberdens_{collider}.binp",
                source_inputs / f"numberdens_{collider}.binp",
            ],
            work_inputs,
            label=f"numberdens_{collider}",
            copy_mode=copy_mode,
        )

    molecule_dst = install_validated_molecule_file(
        species=species,
        moldata_dir=_REPO_ROOT / "data" / "moldata",
        inputs_dir=work_inputs,
    )

    radmc3d_src = source_inputs / "radmc3d.inp"
    if radmc3d_src.exists():
        _link_or_copy(radmc3d_src, work_inputs / "radmc3d.inp", copy_mode="copy")
    write_line_radmc3d_inp(
        work_inputs / "radmc3d.inp",
        line_mode=species_config.line_mode,
        config=config,
    )
    write_lines_inp(work_inputs / "lines.inp", species_config)

    manifest = {
        "species": species,
        "transition": species_config.transition,
        "line_mode": int(species_config.line_mode),
        "tgas_eq_tdust": bool(config.tgas_eq_tdust),
        "use_gow17_tgas": bool(config.use_gow17_tgas),
        "colliders": colliders,
        "collider_policy": "gow17_lamda",
        "h2_opr_mode": config.h2_opr_mode,
        "molecule_file": str(molecule_dst),
        "source_inputs_dir": str(source_inputs),
        "chemistry_inputs_dir": str(chemistry_inputs),
        "staged_inputs_dir": str(work_inputs),
        "copy_mode": copy_mode,
        "input_files": staged,
        "line_tgas_clip": tgas_clip_info,
    }
    (work / "line_rt_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    validate_line_run(work, config, species_config)
    return work


# db-keywords: nonlte, line-transfer, radmc3d
# db-role: entrypoint
def prepare_external_population_line_run(
    *,
    source_inputs_dir: str | Path,
    chemistry_inputs_dir: str | Path | None = None,
    work_dir: str | Path,
    species: str,
    levelpop_file: str | Path,
    molecule_file: str | Path | None = None,
    copy_mode: str = "symlink",
    exist_ok: bool = False,
    allow_unconverged: bool = False,
    incl_dust: int = 1,
    itempdecoup: int = 1,
    rto_style: int = 3,
) -> Path:
    """Stage a single-species RADMC-3D ``lines_mode = 50`` run."""

    species = str(species).lower().strip()
    source_inputs = Path(source_inputs_dir)
    if not source_inputs.exists():
        raise FileNotFoundError(f"source_inputs_dir does not exist: {source_inputs}")
    chemistry_inputs = (
        Path(chemistry_inputs_dir) if chemistry_inputs_dir is not None else source_inputs
    )
    if not chemistry_inputs.exists():
        raise FileNotFoundError(
            f"chemistry_inputs_dir does not exist: {chemistry_inputs}"
        )
    levelpop_file = Path(levelpop_file)
    if not levelpop_file.exists():
        raise FileNotFoundError(f"levelpop file does not exist: {levelpop_file}")
    solver_manifest_file = levelpop_file.with_name(
        f"external_levelpop_manifest_{species}.json"
    )
    if not solver_manifest_file.exists():
        raise FileNotFoundError(
            f"Missing solver manifest next to levelpop file: {solver_manifest_file}"
        )
    solver_manifest = json.loads(solver_manifest_file.read_text())
    if solver_manifest.get("species") != species:
        raise ValueError(
            f"Solver manifest species {solver_manifest.get('species')!r} does not "
            f"match requested species {species!r}"
        )
    if not bool(solver_manifest.get("converged", False)) and not bool(allow_unconverged):
        raise ValueError(
            f"Refusing to stage unconverged external populations from {solver_manifest_file}"
        )
    if int(solver_manifest.get("line_mode", 0)) != 50:
        raise ValueError("External-population solver manifest must declare line_mode = 50")
    manifest_levelpop = solver_manifest.get("levelpop_file")
    if manifest_levelpop:
        expected_levelpop = _resolve_manifest_path(
            str(manifest_levelpop),
            base_dir=solver_manifest_file.parent,
        )
        if expected_levelpop.resolve() != levelpop_file.resolve():
            raise ValueError(
                "levelpop_file does not match the solver manifest: "
                f"{levelpop_file} != {expected_levelpop}"
            )

    work = Path(work_dir)
    if work.exists() and not exist_ok:
        raise FileExistsError(f"work_dir already exists: {work}")
    work_inputs = work / "radmc3d_inputs"
    work_outputs = work / "radmc3d_outputs"
    work_inputs.mkdir(parents=True, exist_ok=True)
    work_outputs.mkdir(parents=True, exist_ok=True)

    staged: dict[str, str] = {}
    for name in ("amr_grid.inp", "wavelength_micron.inp"):
        dst = _stage_required_one(
            [source_inputs / name],
            work_inputs,
            label=name,
            copy_mode=copy_mode,
        )
        staged[name] = str(dst)

    staged.update(
        _stage_optional(
            source_inputs,
            work_inputs,
            ["dust_density.binp", "dustopac.inp", "stars.inp", "external_source.inp"],
            copy_mode=copy_mode,
        )
    )
    for opac in sorted(source_inputs.glob("dustkappa_*.inp")):
        staged[opac.name] = str(
            _link_or_copy(opac, work_inputs / opac.name, copy_mode=copy_mode)
        )

    source_run = (
        source_inputs.parent if source_inputs.name == "radmc3d_inputs" else source_inputs
    )
    source_outputs = source_run / "radmc3d_outputs"
    dust_temp = _first_existing(
        [
            source_outputs / "temperature" / "dust_temperature.bdat",
            source_outputs / "dust_temperature.bdat",
            source_inputs / "dust_temperature.bdat",
            source_run / "dust_temperature.bdat",
        ]
    )
    if dust_temp is not None:
        staged[dust_temp.name] = str(
            _link_or_copy(
                dust_temp,
                work_outputs / "dust_temperature.bdat",
                copy_mode=copy_mode,
            )
        )

    for name in (
        f"numberdens_{species}.binp",
        "gas_temperature.binp",
        "gas_velocity.binp",
        "microturbulence.binp",
    ):
        dst = _stage_required_one(
            [chemistry_inputs / name, source_inputs / name],
            work_inputs,
            label=name,
            copy_mode=copy_mode,
        )
        staged[name] = str(dst)
    expected_gas_velocity_sha256 = solver_manifest.get("gas_velocity_sha256")
    if expected_gas_velocity_sha256:
        staged_gas_velocity_sha256 = sha256_file(work_inputs / "gas_velocity.binp")
        if staged_gas_velocity_sha256 != expected_gas_velocity_sha256:
            raise ValueError(
                "Staged gas_velocity.binp does not match the velocity field used "
                "by the external population solver."
            )
    else:
        staged_gas_velocity_sha256 = sha256_file(work_inputs / "gas_velocity.binp")

    if molecule_file is None:
        manifest_molecule = solver_manifest.get("molecule_file")
        if not manifest_molecule:
            raise ValueError(
                "Solver manifest does not record molecule_file; cannot safely stage "
                "external populations."
            )
        molecule_src = _resolve_manifest_path(
            str(manifest_molecule),
            base_dir=solver_manifest_file.parent,
        )
    else:
        molecule_src = Path(molecule_file)
    if not molecule_src.exists():
        raise FileNotFoundError(f"molecule_file does not exist: {molecule_src}")
    molecule_sha256 = sha256_file(molecule_src)
    expected_molecule_sha256 = solver_manifest.get("molecule_sha256")
    if expected_molecule_sha256 and molecule_sha256 != expected_molecule_sha256:
        raise ValueError(
            "Staged molecule file does not match the molecule used by the external "
            "population solver."
        )
    molecule_dst = _link_or_copy(
        molecule_src,
        work_inputs / f"molecule_{species}.inp",
        copy_mode=copy_mode,
    )
    staged[molecule_dst.name] = str(molecule_dst)

    levelpop_dst = _link_or_copy(
        levelpop_file,
        work_inputs / f"levelpop_{species}.dat",
        copy_mode=copy_mode,
    )
    staged[levelpop_dst.name] = str(levelpop_dst)
    solver_manifest_dst = _link_or_copy(
        solver_manifest_file,
        work_inputs / f"external_levelpop_manifest_{species}.json",
        copy_mode=copy_mode,
    )
    staged[solver_manifest_dst.name] = str(solver_manifest_dst)

    radmc3d_src = source_inputs / "radmc3d.inp"
    if radmc3d_src.exists():
        _link_or_copy(radmc3d_src, work_inputs / "radmc3d.inp", copy_mode="copy")
    line_config = SpeciesLineConfig(species=species, line_mode=50, colliders=[])
    write_line_radmc3d_inp(
        work_inputs / "radmc3d.inp",
        line_mode=50,
        config=NonLTELineTransferConfig(
            use_gow17_tgas=True,
            incl_dust=incl_dust,
            itempdecoup=itempdecoup,
            rto_style=rto_style,
        ),
    )
    write_lines_inp(work_inputs / "lines.inp", line_config)

    manifest = {
        "species": species,
        "line_mode": 50,
        "solver": "external_healpix_escape_probability",
        "source_inputs_dir": str(source_inputs),
        "chemistry_inputs_dir": str(chemistry_inputs),
        "staged_inputs_dir": str(work_inputs),
        "copy_mode": copy_mode,
        "input_files": staged,
        "levelpop_file": str(levelpop_dst),
        "molecule_file": str(molecule_dst),
        "molecule_sha256": molecule_sha256,
        "gas_velocity_sha256": staged_gas_velocity_sha256,
        "solver_manifest_file": str(solver_manifest_file),
        "solver_manifest": solver_manifest,
    }
    validation = validate_external_population_run(
        work,
        species=species,
        allow_unconverged=allow_unconverged,
    )
    manifest["validation"] = validation
    (work / "external_levelpop_staging.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    return work
