# db-keywords: uv-products, gow17, disk-mask, gas-temperature, config, units, radmc3d, chemistry, model, field, api, hashing
# db-role: entrypoint
# db-scope: package
# db-purpose: Package module for uv-products, gow17, disk-mask, gas-temperature.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional
from pathlib import Path
import hashlib
import json

import numpy as np

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

from diskbridge._logging import logger
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.registry import available_models, get_model_callable
from diskbridge.chemistry.io import (
    add_default_line_colliders_from_gow17,
    attach_chemistry_result_to_model,
    write_gas_temperature,
    write_many,
)
from diskbridge.model.microturbulence import ensure_microturbulence_field
from diskbridge.radmc3d.cache import should_use_cache, stable_json_hash, write_cache_context
from diskbridge.radmc3d.uv_products import UV_PRODUCT_MERGED_FIELD_NAMES
from diskbridge.serialization import jsonable
from diskbridge.utils import sha256_file


def _quantity_context(q, unit: str | None = None) -> dict[str, Any]:
    data = q.to(unit) if unit is not None else q
    arr = np.ascontiguousarray(np.asarray(data.magnitude, dtype=np.float64))
    return {
        "unit": str(data.units),
        "shape": list(arr.shape),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def _field_context(field: Field, unit: str | None = None) -> dict[str, Any]:
    out = _quantity_context(field.data, unit=unit)
    out["axis_order"] = list(field.axis_order)
    return out


def _file_hashes(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                out[str(path.relative_to(root))] = sha256_file(path)
    return out


def _gow17_cache_context(rad: "RadModel", config: dict) -> dict[str, Any]:
    """Build the explicit dependency context for GOW17 chemistry."""
    config = {
        key: value
        for key, value in dict(config).items()
        if key not in {"checkpoint", "_output_dir"}
    }
    model = rad.model
    if model.gas is None:
        raise ValueError("GOW17 cache context requires model.gas")

    ensure_microturbulence_field(model, rad.params)

    gas_units = {
        "density": "g/cm^3",
        "dust_temperature": "K",
        "microturbulence": "km/s",
        "chi": "dimensionless",
        "chi_broad": "dimensionless",
        "chi_h2": "dimensionless",
        "chi_co": "dimensionless",
        "chi_c": "dimensionless",
        "G_CO_pdes": "dimensionless",
        "F_CO_pdes_photon": "1/(cm^2 s)",
        "F_CO_pdes_photon_bands": "1/(cm^2 s)",
        "uv_product_measured_mask": "dimensionless",
        "segment_id": "dimensionless",
    }
    gas_fields: dict[str, Any] = {}
    required = ("density", "dust_temperature", "microturbulence")
    optional = (
        "chi",
        "uv_product_measured_mask",
        "segment_id",
        *UV_PRODUCT_MERGED_FIELD_NAMES,
    )
    for name in (*required, *optional):
        if name in model.gas:
            gas_fields[name] = _field_context(model.gas[name], unit=gas_units.get(name))
        elif name in required:
            raise KeyError(f"GOW17 cache context requires gas field {name!r}")

    dust_context: dict[str, Any] = {"present": model.dust is not None}
    if model.dust is not None:
        bins: dict[str, Any] = {}
        for bin_name, dust_bin in sorted(model.dust.bins.items()):
            bins[bin_name] = {
                "size": jsonable(dust_bin.size),
                "size_min": jsonable(dust_bin.size_min),
                "size_max": jsonable(dust_bin.size_max),
                "mass_fraction": float(dust_bin.mass_fraction),
                "density_material": jsonable(dust_bin.density_material),
                "density": _field_context(dust_bin["density"], unit="g/cm^3"),
            }
        dust_context.update(
            {
                "nbin": int(model.dust.nbin),
                "bins": bins,
            }
        )

    model_dir = Path(rad.model_dir)
    upstream_contexts = _file_hashes(
        model_dir,
        (
            "radmc3d_outputs/**/cache_context.json",
            "segments/**/cache_context.json",
        ),
    )
    opacity_inputs = _file_hashes(
        Path(rad.inputs_dir),
        (
            "dustopac.inp",
            "dustkappa_*.inp",
        ),
    )

    return {
        "kind": "gow17",
        "config_sha256": stable_json_hash(config),
        "params": {
            "nside": int(rad.params.nside),
        },
        "model": {
            "mesh_shape": list(model.mesh.shape) if model.mesh is not None else None,
            "mesh_axes": list(model.mesh.axis_names()) if model.mesh is not None else None,
            "gas_fields": gas_fields,
            "dust": dust_context,
        },
        "radmc": {
            "upstream_cache_contexts": upstream_contexts,
            "opacity_inputs": opacity_inputs,
        },
    }


def load_chemistry_outputs(
    rad: 'RadModel',
    output_dir: Path,
) -> ChemistryResult:
    """Load saved chemistry products from RADMC-3D input files.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper used to read fields onto the active mesh.
    output_dir : pathlib.Path
        Directory containing saved chemistry outputs. Files may live directly
        in this directory or under its ``radmc3d_inputs`` subdirectory.

    Returns
    -------
    ChemistryResult
        Chemistry result reconstructed from saved number-density files and
        optional gas-temperature output.

    Raises
    ------
    FileNotFoundError
        If no ``numberdens_*.binp`` files exist in the output directory.
    """
    output_dir = Path(output_dir)
    inputs_dir = output_dir / "radmc3d_inputs"
    data_dir = inputs_dir if inputs_dir.exists() else output_dir

    number_densities = {}
    for path in sorted(data_dir.glob("numberdens_*.binp")):
        species = path.stem.removeprefix("numberdens_")
        number_densities[species] = rad.data.readGasDens(fname=path, ispec=species)

    if not number_densities:
        raise FileNotFoundError(f"No numberdens_*.binp files found in {data_dir}")

    gas_temp_path = data_dir / "gas_temperature.binp"
    if gas_temp_path.exists():
        rad.gas_temperature = rad.data.readGasTemp(fname=gas_temp_path)

    meta = {
        "source": "saved_outputs",
        "output_dir": str(output_dir),
        "data_dir": str(data_dir),
        "loaded_species": sorted(number_densities),
    }
    for meta_path in (
        output_dir / "chemistry_meta.json",
        output_dir / "gow17_meta.json",
    ):
        if meta_path.exists():
            with meta_path.open("r") as f:
                saved_meta = json.load(f)
            if isinstance(saved_meta, dict):
                meta.update(saved_meta)
            break

    return ChemistryResult(
        number_densities=number_densities,
        meta=meta,
    )


def _attach_chemistry_result_to_model(rad: 'RadModel', result: ChemistryResult) -> None:
    attach_chemistry_result_to_model(rad, result)


def _gow17_validation_summary(meta: dict) -> dict:
    """Return a compact summary for GOW17 validation runs."""
    diag = meta.get("gow17_diagnostics", {})
    time_cvode_failure_cells_total = int(sum(diag.get("time_cvode_failure_cells_hist", [])))
    time_exception_failure_cells_total = int(sum(diag.get("time_exception_failure_cells_hist", [])))
    time_negative_abundance_corrections_total = int(
        sum(diag.get("time_negative_abundance_corrections_hist", []))
    )
    return {
        "n_fail": int(meta.get("n_fail", -1)),
        "max_status": int(meta.get("max_status", 0)),
        "temperature_mode": meta.get("temperature_mode"),
        "enable_co_phase": bool(meta.get("enable_co_phase", False)),
        "b_CO_requested_scalar_kms": diag.get("b_CO_requested_scalar_kms"),
        "b_CO_table_kms": diag.get("b_CO_table_kms"),
        "b_CO_scalar_approximation": bool(diag.get("b_CO_scalar_approximation", False)),
        "rhs_status": int(diag.get("gow17_rhs_status", 0)),
        "rhs_residual_max_global": float(diag.get("gow17_rhs_residual_max_global", float("nan"))),
        "thermal_balance_residual_max_abs": float(
            diag.get("thermal_balance_residual_max_abs", float("nan"))
        ),
        "tevol_max_cells_total": int(diag.get("tevol_max_cells_total", 0)),
        "negative_abundance_corrections_total": int(
            diag.get("negative_abundance_corrections_total", 0)
        ),
        "cvode_failure_cells_total": int(diag.get("cvode_failure_cells_total", 0)),
        "exception_failure_cells_total": int(diag.get("exception_failure_cells_total", 0)),
        "time_cvode_failure_cells_total": time_cvode_failure_cells_total,
        "time_exception_failure_cells_total": time_exception_failure_cells_total,
        "time_negative_abundance_corrections_total": time_negative_abundance_corrections_total,
    }


def run_chemistry(
    rad: 'RadModel',
    model: str,
    config: Optional[dict] = None,
    write: bool = False,
    output_dir: Optional[Path] = None,
    diagnostic_plots: bool = False,
    plots_dir: Optional[Path] = None,
    load_existing: bool = False,
    force: bool = False,
    use_cache: bool = True,
) -> ChemistryResult:
    """Run chemistry computation with the specified model.
    
    This is the main entry point for all chemistry workflows. It:
    1. Ensures required RAD fields exist (temperature, nH, chi via RadModel methods)
    2. Calls the model callable from the registry
    3. Optionally writes outputs
    4. Returns a ChemistryResult
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper (provides ensure_temperature, ensure_nH, ensure_chi)
    model : str
        Chemistry model name (must be in REGISTRY)
    config : dict, optional
        Model-specific configuration parameters (default: empty dict)
    write : bool, optional
        Whether to write number density outputs (default: False)
    output_dir : Path, optional
        Output directory for writing (default: rad.model_dir)
    diagnostic_plots : bool, optional
        Whether to make diagnostic plots after chemistry completes.
    plots_dir : Path, optional
        Directory for diagnostic plots. Defaults to ``output_dir / "plots"``.
    load_existing : bool, optional
        Load saved chemistry outputs from ``output_dir`` instead of running the
        chemistry model.
    force : bool, optional
        Recompute chemistry even if a valid chemistry cache exists.
    use_cache : bool, optional
        If True, GOW17 loads saved outputs when the chemistry cache context matches.
        
    Returns
    -------
    ChemistryResult
        Result object containing abundances, number_densities, fields, and metadata
        
    Raises
    ------
    ValueError
        If model name is not found in registry
        
    Examples
    --------
    >>> res = run_chemistry(
    ...     rad,
    ...     model="gow17",
    ...     config=dict(enable_co_phase=True),
    ...     write=True,
    ... )
    >>> nco = res.number_densities["co"]
    """
    if config is None:
        config = {}
    
    model_lower = model.lower()
    output_path = Path(output_dir) if output_dir is not None else Path(rad.model_dir)
    cache_context = None
    loaded_from_cache = False

    if load_existing:
        result = load_chemistry_outputs(rad, output_path)
    else:
        if use_cache and write and model_lower == "gow17":
            cache_context = _gow17_cache_context(rad, config)
            cache_ok, _cached = should_use_cache(
                output_dir=output_path,
                candidate_files=["radmc3d_inputs/numberdens_co.binp"],
                force=force,
                cache_context=cache_context,
            )
            if cache_ok:
                logger.info("Using cached GOW17 chemistry outputs: %s", output_path)
                result = load_chemistry_outputs(rad, output_path)
                loaded_from_cache = True
            else:
                result = None
        else:
            result = None

        if result is None:
            try:
                model_fn = get_model_callable(model_lower)
            except ValueError:
                available = ", ".join(available_models())
                raise ValueError(
                    f"Unknown chemistry model: {model!r}. "
                    f"Available models: {available}"
                )
            model_config = dict(config)
            model_config.setdefault("_output_dir", str(output_path))
            result = model_fn(rad, model_config)

    if model_lower in {"gow17", "gow17_slab_equilibrium"}:
        result = add_default_line_colliders_from_gow17(
            result,
            rad=rad,
            opr=config.get("line_h2_opr", 3.0),
        )

    _attach_chemistry_result_to_model(rad, result)
    
    if write and result.number_densities and (not load_existing or model_lower == "gow17"):
        write_many(rad, result.number_densities, output_path)

    if write and getattr(rad, 'Tgas_gow17', None) is not None and not load_existing:
        write_gas_temperature(rad, output_dir=output_path, binary=True)

    result.meta["loaded_from_cache"] = bool(loaded_from_cache or load_existing)

    if diagnostic_plots:
        from diskbridge.visualization.diagnostics import make_chemistry_diagnostic_plots

        plot_output_dir = Path(plots_dir) if plots_dir is not None else output_path / "plots"
        made = make_chemistry_diagnostic_plots(rad, result, plot_output_dir)
        result.meta["diagnostic_plots"] = [str(path) for path in made]

    if write:
        output_path.mkdir(parents=True, exist_ok=True)
        if cache_context is None and model_lower == "gow17":
            cache_context = _gow17_cache_context(rad, config)
        if cache_context is not None and not load_existing:
            write_cache_context(output_path, cache_context)
        (output_path / "chemistry_meta.json").write_text(
            json.dumps(jsonable(result.meta), indent=2, sort_keys=True) + "\n"
        )
        if model_lower == "gow17":
            (output_path / "gow17_validation_summary.json").write_text(
                json.dumps(jsonable(_gow17_validation_summary(result.meta)), indent=2, sort_keys=True) + "\n"
            )
    
    return result
