from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
from pathlib import Path

import numpy as np

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
    convergence: Optional[dict] = None,
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
    from diskbridge.chemistry.hydrogen.partition import compute_h2_partition
    
    if chemistry_config is None:
        chemistry_config = {}
    if thermal_config is None:
        thermal_config = {}
    
    logger.info("=" * 60)
    logger.info("Running coupled thermochemistry")
    logger.info("=" * 60)
    
    chem_result = None
    therm_result = None

    chi_dust = rad.chi if rad.chi is not None else rad.ensure_chi()
    nH = rad.ensure_nH()

    rad.nH2, rad.nH_atom = compute_h2_partition(
        rad=rad,
        mesh=rad.model.mesh,
        nH=nH,
        chi_dust=chi_dust,
    )

    enable_convergence = convergence is not None
    if enable_convergence:
        if not convergence:
            raise ValueError(
                "convergence must specify at least one threshold: "
                "tgas_atol, tgas_rtol, nco_atol, nco_rtol"
            )
        tgas_atol = convergence.get("tgas_atol", None)
        tgas_rtol = convergence.get("tgas_rtol", None)
        nco_atol = convergence.get("nco_atol", None)
        nco_rtol = convergence.get("nco_rtol", None)
        if tgas_atol is None and tgas_rtol is None and nco_atol is None and nco_rtol is None:
            raise ValueError(
                "convergence must specify at least one threshold: "
                "tgas_atol, tgas_rtol, nco_atol, nco_rtol"
            )
    else:
        tgas_atol = None
        tgas_rtol = None
        nco_atol = None
        nco_rtol = None

    prev_tgas_K = None
    prev_nco_total = None
    
    for iteration in range(n_iter):
        logger.info(f"\nThermochemistry iteration {iteration + 1}/{n_iter}")
        logger.info("-" * 60)
        
        logger.info("Step 1: Chemistry")
        chem_result = run_chemistry(
            rad,
            model=chemistry_model,
            config=chemistry_config,
            write=(write and (not enable_convergence) and iteration == n_iter - 1),
        )
        
        if chem_result.number_densities:
            if "co_gas" in chem_result.number_densities:
                rad.nco_gas = chem_result.number_densities["co_gas"]
            if "co_ice" in chem_result.number_densities:
                rad.nco_ice = chem_result.number_densities["co_ice"]
            # Store carbon closure products from chemistry
            if "cplus" in chem_result.number_densities:
                rad.nCplus = chem_result.number_densities["cplus"]
            if "c" in chem_result.number_densities:
                rad.nC = chem_result.number_densities["c"]
            if "e" in chem_result.number_densities:
                rad.ne = chem_result.number_densities["e"]
        
        logger.info("Step 2: Thermal balance")
        therm_result = run_thermal(
            rad,
            model=thermal_model,
            config=thermal_config,
            write=(write and (not enable_convergence) and iteration == n_iter - 1),
        )
        
        logger.info(f"Iteration {iteration + 1} complete")
        logger.info(f"  Tgas range: {therm_result.tgas.to('K').magnitude.min():.1f} - "
                   f"{therm_result.tgas.to('K').magnitude.max():.1f} K")

        if enable_convergence:
            tgas_K = therm_result.tgas.to('K').magnitude

            nco_total = None
            if rad.nco_gas is not None and rad.nco_ice is not None:
                nco_total = (
                    rad.nco_gas.to('cm^-3').magnitude
                    + rad.nco_ice.to('cm^-3').magnitude
                )
            elif rad.nco_gas is not None:
                nco_total = rad.nco_gas.to('cm^-3').magnitude
            elif rad.nco_ice is not None:
                nco_total = rad.nco_ice.to('cm^-3').magnitude

            if (nco_atol is not None or nco_rtol is not None) and nco_total is None:
                raise ValueError(
                    "CO convergence requested (nco_atol/nco_rtol) but no CO fields "
                    "are available on rad (nco_gas/nco_ice)."
                )

            if prev_tgas_K is not None:
                need_t = (tgas_atol is not None) or (tgas_rtol is not None)
                need_co = (nco_atol is not None) or (nco_rtol is not None)

                t_ok = True
                if need_t:
                    dt = np.abs(tgas_K - prev_tgas_K)
                    if tgas_atol is not None:
                        t_ok = t_ok and (np.max(dt) <= float(tgas_atol))
                    if tgas_rtol is not None:
                        denom = np.where(prev_tgas_K != 0.0, np.abs(prev_tgas_K), np.inf)
                        t_ok = t_ok and (np.max(dt / denom) <= float(tgas_rtol))

                co_ok = True
                if need_co:
                    dn = np.abs(nco_total - prev_nco_total)
                    if nco_atol is not None:
                        co_ok = co_ok and (np.max(dn) <= float(nco_atol))
                    if nco_rtol is not None:
                        denom = np.where(prev_nco_total != 0.0, np.abs(prev_nco_total), np.inf)
                        co_ok = co_ok and (np.max(dn / denom) <= float(nco_rtol))

                converged = (not need_t or t_ok) and (not need_co or co_ok)
                if converged:
                    logger.info(
                        f"Converged after {iteration + 1} iterations; stopping early"
                    )
                    break

            prev_tgas_K = tgas_K
            prev_nco_total = nco_total
    
    logger.info("=" * 60)
    logger.info("Thermochemistry complete!")
    logger.info("=" * 60)

    if write and enable_convergence:
        if chem_result is not None and chem_result.number_densities:
            write_many(rad, chem_result.number_densities)
        if therm_result is not None:
            rad.writer.write_gas_temperature(therm_result.tgas, output_dir=rad.model_dir)
    
    return chem_result, therm_result
