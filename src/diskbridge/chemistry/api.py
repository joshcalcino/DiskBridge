from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
from pathlib import Path

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel
    from diskbridge.chemistry.thermal.types import ThermalResult

from diskbridge._logging import logger
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.registry import REGISTRY
from diskbridge.chemistry.io import write_many


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
    ...     model="co_two_phase_steady",
    ...     config=dict(Xco_tot=1e-4, nside=8, b_kms=0.3),
    ...     write=True,
    ... )
    >>> nco = res.number_densities["co"]
    """
    if config is None:
        config = {}
    
    model_lower = model.lower()
    
    if model_lower not in REGISTRY:
        available = ', '.join(REGISTRY.keys())
        raise ValueError(
            f"Unknown chemistry model: {model!r}. "
            f"Available models: {available}"
        )
    
    model_fn = REGISTRY[model_lower]
    result = model_fn(rad, config)
    
    if write and result.number_densities:
        write_many(rad, result.number_densities, output_dir)
    
    return result


def run_thermochemistry(
    rad: "RadModel",
    *,
    chemistry_model: str,
    chemistry_config: Optional[dict] = None,
    thermal_model: str = "thermal_balance_v1",
    thermal_config: Optional[dict] = None,
    n_iter: int = 3,
    write: bool = True,
) -> Tuple[ChemistryResult, "ThermalResult"]:
    """One-call thermochemistry: iterate chemistry <-> thermal balance.
    
    This orchestrates the coupled thermochemistry workflow:
    1. Run chemistry to get CO abundances
    2. Build thermal state using chemistry outputs (CO, chi_eff)
    3. Solve for gas temperature
    4. Optionally re-run chemistry with updated Tgas
    5. Iterate n_iter times
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    chemistry_model : str
        Chemistry model name (from chemistry registry)
    chemistry_config : dict, optional
        Chemistry model configuration
    thermal_model : str, optional
        Thermal model name (default "thermal_balance_v1")
    thermal_config : dict, optional
        Thermal model configuration
    n_iter : int, optional
        Number of thermochemistry iterations (default 3)
    write : bool, optional
        Write outputs (default True)
        
    Returns
    -------
    chem_result : ChemistryResult
        Final chemistry result
    therm_result : ThermalResult
        Final thermal result
        
    Examples
    --------
    >>> from diskbridge.chemistry import run_thermochemistry
    >>> 
    >>> chem, therm = run_thermochemistry(
    ...     rad,
    ...     chemistry_model="co_two_phase_time",
    ...     chemistry_config={"Xco_tot": 1e-4, "nside": 4},
    ...     thermal_model="thermal_balance_v1",
    ...     thermal_config={"pah_scale": 0.5},
    ...     n_iter=3,
    ... )
    """
    from diskbridge.chemistry.thermal import run_thermal
    
    if chemistry_config is None:
        chemistry_config = {}
    if thermal_config is None:
        thermal_config = {}
    
    logger.info("=" * 60)
    logger.info("Running coupled thermochemistry")
    logger.info("=" * 60)
    
    chem_result = None
    therm_result = None
    
    for iteration in range(n_iter):
        logger.info(f"\nThermochemistry iteration {iteration + 1}/{n_iter}")
        logger.info("-" * 60)
        
        logger.info("Step 1: Chemistry")
        chem_result = run_chemistry(
            rad,
            model=chemistry_model,
            config=chemistry_config,
            write=(write and iteration == n_iter - 1),
        )
        
        if chem_result.number_densities:
            if "co_gas" in chem_result.number_densities:
                rad.nco_gas = chem_result.number_densities["co_gas"]
            if "co_ice" in chem_result.number_densities:
                rad.nco_ice = chem_result.number_densities["co_ice"]
        
        logger.info("Step 2: Thermal balance")
        therm_result = run_thermal(
            rad,
            model=thermal_model,
            config=thermal_config,
            write=(write and iteration == n_iter - 1),
        )
        
        logger.info(f"Iteration {iteration + 1} complete")
        logger.info(f"  Tgas range: {therm_result.tgas.to('K').magnitude.min():.1f} - "
                   f"{therm_result.tgas.to('K').magnitude.max():.1f} K")
    
    logger.info("=" * 60)
    logger.info("Thermochemistry complete!")
    logger.info("=" * 60)
    
    return chem_result, therm_result
