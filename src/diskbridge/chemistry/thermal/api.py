"""High-level thermal balance API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np

from diskbridge._logging import logger
from diskbridge._units import Quantity
from diskbridge._config import resolve_model_config
from diskbridge.model.field import Field
from diskbridge.chemistry.thermal.types import ThermalState, ThermalResult
from diskbridge.chemistry.thermal.registry import get_thermal_model


def run_thermal(
    rad: "RadModel",
    model: str = "thermal_balance_v1",
    config: Optional[dict] = None,
    write: bool = True,
) -> ThermalResult:
    """Run thermal balance calculation to solve for gas temperature.
    
    This is the main user-facing API for thermal calculations. It:
    1. Ensures required fields (nH, Tdust, chi)
    2. Pulls chemistry products if available (CO, theta_co, chi_eff)
    3. Runs the thermal solver
    4. Stores and optionally writes results
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper with radiative transfer outputs
    model : str, optional
        Thermal model name (default "thermal_balance_v1")
    config : dict, optional
        Model-specific configuration (merged with defaults)
    write : bool, optional
        Write gas_temperature.inp to model directory (default True)
        
    Returns
    -------
    ThermalResult
        Result object with solved Tgas and auxiliary fields
        
    Examples
    --------
    >>> from diskbridge.chemistry.thermal import run_thermal
    >>> 
    >>> result = run_thermal(rad, model="thermal_balance_v1")
    >>> print(f"Gas temperature: {result.tgas.to('K')}")
    >>> 
    >>> result = run_thermal(
    ...     rad,
    ...     config={
    ...         'zeta_cr': 1e-16,
    ...         'pah_scale': 0.5,
    ...         'n_iter': 5,
    ...     }
    ... )
    """
    if config is None:
        config = {}
    
    logger.info("=" * 60)
    logger.info(f"Running thermal balance: {model}")
    logger.info("=" * 60)
    
    logger.info("Ensuring required fields...")
    nH = rad.ensure_nH()
    Tdust = rad.ensure_dust_temperature()
    
    if rad.chi_eff is not None:
        logger.info("Using chi_eff from shielding calculation")
        chi_eff = rad.chi_eff
    elif rad.chi is not None:
        logger.info("Using chi (no shielding correction)")
        chi_eff = rad.chi
    else:
        logger.info("Computing chi from mean intensity")
        chi_eff = rad.ensure_chi()
    
    logger.info("Building thermal state...")
    state = ThermalState(
        nH=nH,
        Tdust=Tdust,
        chi_eff=chi_eff,
        mesh=rad.model.mesh,
    )
    
    if rad.theta_co is not None:
        logger.info("Including CO shielding factor")
        state.theta_co = rad.theta_co
    
    if rad.chi is not None:
        state.chi = rad.chi
    
    if rad.nco_gas is not None and rad.nco_ice is not None:
        logger.info("Including CO number densities from chemistry")
        state.nco_gas = rad.nco_gas
        state.nco_ice = rad.nco_ice
    
    cfg = resolve_model_config(("thermal", model), overrides=config)
    
    def get_val(key, default_key=None):
        """Get parameter from user config or defaults, converting strings to Quantity."""
        if key in config:
            val = config[key]
            if isinstance(val, str):
                return Quantity(val)
            return val
        default_key = default_key or key
        if default_key in cfg:
            val = cfg[default_key]
            if isinstance(val, str):
                return Quantity(val)
            return val
        return None
    
    params = {
        'zeta_cr': get_val('zeta_cr', 'zeta_cr_default'),
        'pah_scale': get_val('pah_scale', 'pah_scale_default'),
        'X_C_tot': get_val('X_C_tot', 'X_C_tot_default'),
        'Gamma_C0': get_val('Gamma_C0', 'Gamma_C0_default'),
        'T_min': get_val('T_min', 'T_min_solve'),
        'T_max': get_val('T_max', 'T_max_solve'),
        'n_iter': config.get('n_iter', cfg.get('n_iter_default', 3)),
        'tol': config.get('tol', cfg.get('tol_default', 0.01)),
        'beta_cii': get_val('beta_cii', 'beta_cii_default'),
        'alpha_acc': get_val('alpha_acc', 'alpha_acc_default'),
        'backend': config.get('backend', 'auto'),
        'max_bisect_iter': config.get('max_bisect_iter', cfg.get('max_bisect_iter_default', 60)),
        'bisect_tol': config.get('bisect_tol', cfg.get('bisect_tol_default', 1e-6)),
        'store_terms': config.get('store_terms', False),
        'alpha_rec_c0': cfg['alpha_rec_c0'],
        'T_rec_exp': cfg['T_rec_exp'],
        'heating_per_cr': get_val('heating_per_cr', 'heating_per_cr_ionization'),
        'pe_heating_rate_0': get_val('pe_heating_rate_0', 'pe_heating_rate_0'),
        'gamma_cii': get_val('gamma_cii', 'gamma_cii'),
        'E_cii': get_val('E_cii', 'E_cii'),
        'n_crit_cii': cfg['n_crit_cii'],
        'sigma_dust': cfg['sigma_dust'],
        'f_dust': cfg['f_dust'],
    }
    
    logger.info("Solver parameters:")
    logger.info(f"  backend: {params['backend']}")
    logger.info(f"  zeta_cr: {params['zeta_cr']}")
    logger.info(f"  pah_scale: {params['pah_scale']}")
    logger.info(f"  X_C_tot: {params['X_C_tot']}")
    logger.info(f"  n_iter: {params['n_iter']}")
    logger.info(f"  tol: {params['tol']}")
    
    thermal_func = get_thermal_model(model)
    
    logger.info("Running thermal solver...")
    result = thermal_func(state, params)
    
    logger.info("=" * 60)
    logger.info("Thermal solve complete!")
    logger.info(f"  Tgas: min={result.tgas.to('K').magnitude.min():.1f} K, "
               f"max={result.tgas.to('K').magnitude.max():.1f} K, "
               f"median={np.median(result.tgas.to('K').magnitude):.1f} K")
    logger.info(f"  Converged: {result.meta.get('converged', False)}")
    logger.info(f"  Iterations: {result.meta.get('n_iter', 0)}")
    logger.info("=" * 60)
    
    rad.gas_temperature = result.tgas
    
    axis_order = rad.model.mesh.axis_names()
    
    rad.model.gas_register(
        'gas_temperature',
        Field(
            quantity='temperature',
            data=result.tgas,
            axis_order=axis_order,
        ),
    )
    
    if 'nCplus' in result.fields:
        logger.info("Storing auxiliary fields (nCplus, nC, ne)...")
        rad.model.gas_register(
            'nCplus',
            Field(
                quantity='number_density',
                data=result.fields['nCplus'],
                axis_order=axis_order,
            ),
        )
    
    if 'nC' in result.fields:
        rad.model.gas_register(
            'nC',
            Field(
                quantity='number_density',
                data=result.fields['nC'],
                axis_order=axis_order,
            ),
        )
    
    if 'ne' in result.fields:
        rad.model.gas_register(
            'ne',
            Field(
                quantity='number_density',
                data=result.fields['ne'],
                axis_order=axis_order,
            ),
        )
    
    if write:
        logger.info(f"Writing gas_temperature.inp to {rad.model_dir}")
        rad.writer.write_gas_temperature(result.tgas, output_dir=rad.model_dir)
    
    return result
