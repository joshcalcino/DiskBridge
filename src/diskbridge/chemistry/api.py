# db-keywords: uv-products, gow17, disk-mask, gas-temperature, config, units, radmc3d, chemistry, model, field, api, hashing
# db-role: entrypoint
# db-scope: package
# db-purpose: Package module for uv-products, gow17, disk-mask, gas-temperature.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence
from pathlib import Path
import json

import numpy as np

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

from diskbridge._logging import logger
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.registry import available_models, get_model_callable
from diskbridge.chemistry.models.gow17.cache import _gow17_cache_context
from diskbridge.chemistry.models.gow17.io import add_default_line_colliders_from_gow17
from diskbridge.chemistry.models.gow17.validation import _gow17_validation_summary
from diskbridge.chemistry.io import (
    attach_chemistry_result_to_model,
    write_gas_temperature,
    write_many,
)
from diskbridge.radmc3d.cache import should_use_cache, write_cache_context
from diskbridge.utils.serialization import jsonable


def load_chemistry_outputs(
    rad: 'RadModel',
    output_dir: Path,
    *,
    species: Sequence[str] | None = None,
) -> ChemistryResult:
    """Load saved chemistry products and reconstruct their abundances.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper used to read fields onto the active mesh.
    output_dir : pathlib.Path
        Directory containing saved chemistry outputs. Files may live directly
        in this directory or under its ``radmc3d_inputs`` subdirectory.
    species : sequence of str, optional
        Exact species names to load. By default, load every saved
        ``numberdens_*.binp`` field.

    Returns
    -------
    ChemistryResult
        Chemistry result reconstructed from saved number-density files and the
        active model's hydrogen-nuclei density. Each loaded abundance is
        dimensionless and relative to hydrogen nuclei.

    Raises
    ------
    FileNotFoundError
        If no saved number-density files exist, or if an explicitly requested
        species file is missing.
    ValueError
        If ``species`` is empty or the model hydrogen-nuclei density is invalid.
    """
    output_dir = Path(output_dir)
    inputs_dir = output_dir / "radmc3d_inputs"
    data_dir = inputs_dir if inputs_dir.exists() else output_dir

    if species is None:
        species_paths = [
            (path.stem.removeprefix("numberdens_"), path)
            for path in sorted(data_dir.glob("numberdens_*.binp"))
        ]
    else:
        names = list(dict.fromkeys(str(name).lower().strip() for name in species))
        if not names or any(not name for name in names):
            raise ValueError("species must contain at least one non-empty name")
        species_paths = [(name, data_dir / f"numberdens_{name}.binp") for name in names]
        missing = [str(path) for _, path in species_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing requested chemistry number-density files: " + ", ".join(missing)
            )

    number_densities = {
        name: rad.data.readGasDens(fname=path, ispec=name)
        for name, path in species_paths
    }

    if not number_densities:
        raise FileNotFoundError(f"No numberdens_*.binp files found in {data_dir}")

    nH = rad.ensure_nH().to("cm^-3")
    nH_values = np.asarray(nH.magnitude, dtype=np.float64)
    if not np.all(np.isfinite(nH_values)) or np.any(nH_values <= 0.0):
        raise ValueError(
            "Saved chemistry abundances require a finite, positive hydrogen-"
            "nuclei density in every model cell"
        )
    abundances = {
        name: (number_density.to("cm^-3") / nH).to("dimensionless")
        for name, number_density in number_densities.items()
    }

    gas_temp_path = data_dir / "gas_temperature.binp"
    if gas_temp_path.exists():
        rad.read_gas_temperature(gas_temp_path)

    meta = {
        "source": "saved_outputs",
        "output_dir": str(output_dir),
        "data_dir": str(data_dir),
        "loaded_species": sorted(number_densities),
        "abundance_reference": "hydrogen_nuclei",
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
        abundances=abundances,
        number_densities=number_densities,
        meta=meta,
    )


def _attach_chemistry_result_to_model(rad: 'RadModel', result: ChemistryResult) -> None:
    attach_chemistry_result_to_model(rad, result)


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

    if model_lower == "gow17" and write and not load_existing:
        rad.ensure_dust_temperature()

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
        if cache_context is None and model_lower == "gow17" and not load_existing:
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
