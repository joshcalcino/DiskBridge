from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from pathlib import Path
import json

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

from diskbridge._logging import logger
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.registry import available_models, get_model_callable
from diskbridge.chemistry.io import write_gas_temperature, write_many
from diskbridge.model.field import Field
from diskbridge.serialization import jsonable


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
    """Attach chemistry outputs to the in-memory Model.

    Contract (no new arguments): whenever chemistry runs, the per-cell products are
    stored on `rad.model.gas` as realized `Field`s so they can be persisted via
    `model.save_hdf5()`.

    Naming convention:
      * number density [cm^-3]:  number_density_<species>
      * abundance fraction [-]:  abundance_<species>
      * extra diagnostics:       chem_<name>

    Temperature:
      * If `rad.gas_temperature` exists (either
        read from file or set by a thermal solver), we ensure it is registered as
        `gas_temperature`.
    """

    model = getattr(rad, 'model', None)
    if model is None or getattr(model, 'gas', None) is None or getattr(model, 'mesh', None) is None:
        return

    axis_order = model.mesh.axis_names()
    if hasattr(rad, '_chem_axis_order'):
        axis_order = rad._chem_axis_order()

    tgas = None
    if hasattr(rad, 'ensure_gas_temperature'):
        tgas = rad.ensure_gas_temperature()
    if tgas is None:
        tgas = getattr(rad, 'gas_temperature', None)
    tgas_gow17 = getattr(rad, 'Tgas_gow17', None)
    if tgas_gow17 is not None:
        tgas = tgas_gow17

    if tgas is not None:
        model.gas_register(
            'gas_temperature',
            Field(
                quantity='gas_temperature',
                data=tgas,
                axis_order=axis_order,
                attrs={'source': 'rad'},
            ),
        )

    for sp, x in result.abundances.items():
        model.gas_register(
            f'abundance_{sp}',
            Field(
                quantity=f'abundance_{sp}',
                data=x,
                axis_order=axis_order,
                attrs={'source': 'chemistry', 'kind': 'abundance', 'species': sp},
            ),
        )

    for sp, n in result.number_densities.items():
        model.gas_register(
            f'number_density_{sp}',
            Field(
                quantity=f'number_density_{sp}',
                data=n,
                axis_order=axis_order,
                attrs={'source': 'chemistry', 'kind': 'number_density', 'species': sp},
            ),
        )

    for name, q in result.fields.items():
        model.gas_register(
            f'chem_{name}',
            Field(
                quantity=f'chem_{name}',
                data=q,
                axis_order=axis_order,
                attrs={'source': 'chemistry', 'kind': 'diagnostic', 'name': name},
            ),
        )


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
        "b_CO_mode": meta.get("b_CO_mode"),
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
    ...     config=dict(enable_co_phase=True, b_kms=0.3),
    ...     write=True,
    ... )
    >>> nco = res.number_densities["co"]
    """
    if config is None:
        config = {}
    
    model_lower = model.lower()

    if load_existing:
        if output_dir is None:
            raise ValueError("load_existing=True requires output_dir")
        result = load_chemistry_outputs(rad, Path(output_dir))
    else:
        try:
            model_fn = get_model_callable(model_lower)
        except ValueError:
            available = ", ".join(available_models())
            raise ValueError(
                f"Unknown chemistry model: {model!r}. "
                f"Available models: {available}"
            )
        result = model_fn(rad, config)

    _attach_chemistry_result_to_model(rad, result)
    
    if write and result.number_densities and not load_existing:
        write_many(rad, result.number_densities, output_dir)

    if write and getattr(rad, 'Tgas_gow17', None) is not None and not load_existing:
        write_gas_temperature(rad, output_dir=output_dir, binary=True)

    if diagnostic_plots:
        from diskbridge.visualization.diagnostics import make_chemistry_diagnostic_plots

        base_dir = Path(output_dir) if output_dir is not None else Path(rad.model_dir)
        plot_output_dir = Path(plots_dir) if plots_dir is not None else base_dir / "plots"
        made = make_chemistry_diagnostic_plots(rad, result, plot_output_dir)
        result.meta["diagnostic_plots"] = [str(path) for path in made]

    if write and output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "chemistry_meta.json").write_text(
            json.dumps(jsonable(result.meta), indent=2, sort_keys=True) + "\n"
        )
        if model_lower == "gow17":
            (Path(output_dir) / "gow17_validation_summary.json").write_text(
                json.dumps(jsonable(_gow17_validation_summary(result.meta)), indent=2, sort_keys=True) + "\n"
            )
    
    return result
