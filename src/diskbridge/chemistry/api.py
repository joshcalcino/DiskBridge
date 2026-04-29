from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

from diskbridge._logging import logger
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.registry import available_models, get_model_callable
from diskbridge.chemistry.io import write_gas_temperature, write_many
from diskbridge.model.field import Field


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


def run_chemistry(
    rad: 'RadModel',
    model: str,
    config: Optional[dict] = None,
    write: bool = False,
    output_dir: Optional[Path] = None,
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
    
    if write and result.number_densities:
        write_many(rad, result.number_densities, output_dir)

    if write and getattr(rad, 'Tgas_gow17', None) is not None:
        write_gas_temperature(rad, output_dir=output_dir, binary=True)
    
    return result

