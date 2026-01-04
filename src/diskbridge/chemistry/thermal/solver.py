"""Thermal balance solver with modular term evaluation.

Solves Gamma_tot(Tg) - Lambda_tot(Tg) = 0 for gas temperature.
"""

from __future__ import annotations

import numpy as np

from diskbridge._units import Quantity
from diskbridge._logging import logger
from diskbridge._constants import K_B, M_H
from diskbridge.chemistry.thermal.types import ThermalState, ThermalResult

from diskbridge.chemistry.thermal._kernels import (
    solve_tgas_kernel_se,
    compute_terms_kernel_se,
    max_fractional_change,
    TERM_CR,
    TERM_PE,
    TERM_CII,
    TERM_GD,
    TERM_CI,
    TERM_OI,
    TERM_CO,
)


def _build_term_mask(
    heating_terms: list[str],
    cooling_terms: list[str],
    exchange_terms: list[str],
) -> int:
    """Build bitmask for term dispatch in kernels."""
    mask = 0
    
    if "cosmic_ray" in heating_terms:
        mask |= TERM_CR
    if "photoelectric" in heating_terms:
        mask |= TERM_PE
    
    if "cii" in cooling_terms:
        mask |= TERM_CII
    if "ci" in cooling_terms:
        mask |= TERM_CI
    if "oi" in cooling_terms:
        mask |= TERM_OI
    if "co" in cooling_terms:
        mask |= TERM_CO
    
    if "gas_dust" in exchange_terms:
        mask |= TERM_GD
    
    return mask


def solve_thermal_balance_numba(
    state: ThermalState,
    heating_terms: list[str],
    cooling_terms: list[str],
    exchange_terms: list[str],
    params: dict,
    n_iter: int = 3,
    tol: float = 0.01,
    max_bisect_iter: int = 60,
    bisect_tol: float = 1e-6,
    store_terms: bool = False,
) -> ThermalResult:
    """Numba-accelerated thermal balance solver.
    
    Uses global constants from _constants.py. The params dict only contains
    solver control parameters (n_iter, tol, etc.) and optional user overrides.
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with required fields
    heating_terms : list[str]
        List of heating term names
    cooling_terms : list[str]
        List of cooling term names
    exchange_terms : list[str]
        List of exchange term names
    params : dict
        Solver control parameters 
    n_iter : int
        Maximum outer iterations (default 3)
    tol : float
        Convergence tolerance (fractional change in Tg, default 0.01)
    max_bisect_iter : int
        Maximum bisection iterations per cell (default 60)
    bisect_tol : float
        Bisection convergence tolerance [erg cm^-3 s^-1] (default 1e-6)
    store_terms : bool
        Store per-term heating/cooling rates (default False)
        
    Returns
    -------
    ThermalResult
        Result with solved Tgas and auxiliary fields
    """
    from diskbridge._constants import (
        T_MIN_SOLVE, T_MAX_SOLVE, ZETA_CR, PAH_SCALE,
        X_O_TOT, ALPHA_ACC, BETA_CII, BETA_CI10, BETA_CI20, BETA_CI21,
        BETA_OI10, BETA_OI20, BETA_OI21, BETA_CO,
        HEATING_PER_CR, PE_HEATING_RATE_0, SIGMA_DUST, F_DUST,
    )
    
    CO_JMAX = 15
    
    mask = _build_term_mask(heating_terms, cooling_terms, exchange_terms)
    
    shape = state.nH.shape
    ncells = state.nH.size
    
    nH_flat = np.ascontiguousarray(state.nH.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    Td_flat = np.ascontiguousarray(state.Tdust.to('K').magnitude.flatten(), dtype=np.float64)
    chi_flat = np.ascontiguousarray(state.chi_eff.magnitude.flatten(), dtype=np.float64)
    
    if state.nH2 is None:
        raise ValueError(
            "ThermalState.nH2 is required for SE-based cooling. "
            "Provide H2 number density explicitly."
        )
    nH2_flat = np.ascontiguousarray(state.nH2.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    
    if state.nH_atom is None:
        raise ValueError(
            "ThermalState.nH_atom is required for SE-based cooling. "
            "Provide atomic H number density explicitly."
        )
    nHI_flat = np.ascontiguousarray(state.nH_atom.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    
    if state.nco_gas is not None:
        nco_gas_flat = state.nco_gas.to('cm^-3').magnitude.flatten()
    else:
        nco_gas_flat = np.zeros(ncells, dtype=np.float64)
    
    if state.nco_ice is not None:
        nco_ice_flat = state.nco_ice.to('cm^-3').magnitude.flatten()
    else:
        nco_ice_flat = np.zeros(ncells, dtype=np.float64)
    
    nco_total = np.ascontiguousarray(nco_gas_flat + nco_ice_flat, dtype=np.float64)
    nco_gas_flat = np.ascontiguousarray(nco_gas_flat, dtype=np.float64)
    
    nO_flat = np.maximum(X_O_TOT * nH_flat - nco_total, 0.0)
    nO_flat = np.ascontiguousarray(nO_flat, dtype=np.float64)
    
    # Require carbon closure products from chemistry (not computed here)
    if state.nCplus is None:
        raise ValueError(
            "ThermalState.nCplus is required. Run chemistry first to compute "
            "carbon closure, or provide C+ number density explicitly."
        )
    if state.nC is None:
        raise ValueError(
            "ThermalState.nC is required. Run chemistry first to compute "
            "carbon closure, or provide neutral C number density explicitly."
        )
    if state.ne is None:
        raise ValueError(
            "ThermalState.ne is required. Run chemistry first to compute "
            "carbon closure, or provide electron number density explicitly."
        )
    
    nCplus_flat = np.ascontiguousarray(state.nCplus.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    nC_flat = np.ascontiguousarray(state.nC.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    ne_flat = np.ascontiguousarray(state.ne.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    
    Tg_flat = np.full(ncells, 0.5 * (T_MIN_SOLVE + T_MAX_SOLVE), dtype=np.float64)
    
    converged = False
    max_change = 1.0
    
    for iteration in range(n_iter):
        logger.info(f"Thermal iteration {iteration + 1}/{n_iter}")
        
        Tg_old = Tg_flat.copy()
        
        solve_tgas_kernel_se(
            nH_flat, nH2_flat, nHI_flat, Td_flat, chi_flat, ne_flat,
            nCplus_flat, nC_flat, nO_flat, nco_gas_flat,
            mask, T_MIN_SOLVE, T_MAX_SOLVE, max_bisect_iter, bisect_tol,
            ZETA_CR, PAH_SCALE, ALPHA_ACC, BETA_CII,
            BETA_CI10, BETA_CI20, BETA_CI21,
            BETA_OI10, BETA_OI20, BETA_OI21,
            BETA_CO, CO_JMAX,
            HEATING_PER_CR, PE_HEATING_RATE_0, SIGMA_DUST, F_DUST,
            Tg_flat,
        )
        
        max_change = max_fractional_change(Tg_flat, Tg_old)
        
        logger.info(f"  Tgas: min={np.min(Tg_flat):.1f} K, "
                   f"max={np.max(Tg_flat):.1f} K, "
                   f"median={np.median(Tg_flat):.1f} K, "
                   f"max_change={max_change:.3f}")
        
        if max_change < tol:
            converged = True
            logger.info(f"  Converged after {iteration + 1} iterations")
            break
    
    Tgas_K = Tg_flat.reshape(shape)
    state.Tgas = Quantity(Tgas_K, 'K')
    
    # Note: nCplus/nC/ne are inputs from chemistry, not outputs from thermal
    # They are not modified or overwritten here
    
    fields = {}
    
    if store_terms:
        rate_cr = np.zeros(ncells, dtype=np.float64)
        rate_pe = np.zeros(ncells, dtype=np.float64)
        rate_cii = np.zeros(ncells, dtype=np.float64)
        rate_ci = np.zeros(ncells, dtype=np.float64)
        rate_oi = np.zeros(ncells, dtype=np.float64)
        rate_co = np.zeros(ncells, dtype=np.float64)
        rate_gd = np.zeros(ncells, dtype=np.float64)
        
        compute_terms_kernel_se(
            Tg_flat, nH_flat, nH2_flat, nHI_flat, Td_flat, chi_flat, ne_flat,
            nCplus_flat, nC_flat, nO_flat, nco_gas_flat,
            ZETA_CR, PAH_SCALE, ALPHA_ACC, BETA_CII,
            BETA_CI10, BETA_CI20, BETA_CI21,
            BETA_OI10, BETA_OI20, BETA_OI21,
            BETA_CO, CO_JMAX, HEATING_PER_CR, PE_HEATING_RATE_0,
            SIGMA_DUST, F_DUST,
            rate_cr, rate_pe, rate_cii, rate_ci, rate_oi, rate_co, rate_gd,
        )
        
        fields['rate_cosmic_ray'] = Quantity(rate_cr.reshape(shape), 'erg/(cm^3 * s)')
        fields['rate_photoelectric'] = Quantity(rate_pe.reshape(shape), 'erg/(cm^3 * s)')
        fields['rate_cii'] = Quantity(rate_cii.reshape(shape), 'erg/(cm^3 * s)')
        fields['rate_ci'] = Quantity(rate_ci.reshape(shape), 'erg/(cm^3 * s)')
        fields['rate_oi'] = Quantity(rate_oi.reshape(shape), 'erg/(cm^3 * s)')
        fields['rate_co'] = Quantity(rate_co.reshape(shape), 'erg/(cm^3 * s)')
        fields['rate_gas_dust'] = Quantity(rate_gd.reshape(shape), 'erg/(cm^3 * s)')
    
    meta = {
        'n_iter': iteration + 1,
        'converged': converged,
        'max_change': float(max_change),
        'heating_terms': heating_terms,
        'cooling_terms': cooling_terms,
        'exchange_terms': exchange_terms,
        'params': params,
    }
    
    return ThermalResult(
        tgas=state.Tgas,
        fields=fields,
        meta=meta,
    )



def solve_thermal_balance(
    state: ThermalState,
    heating_terms: list[str],
    cooling_terms: list[str],
    exchange_terms: list[str],
    params: dict,
    n_iter: int = 3,
    tol: float = 0.01,
    max_bisect_iter: int = 60,
    bisect_tol: float = 1e-6,
    store_terms: bool = False,
) -> ThermalResult:
    """Solve thermal balance using Numba-accelerated kernels.
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with required fields (including nCplus, nC, ne from chemistry)
    heating_terms : list[str]
        List of heating term names
    cooling_terms : list[str]
        List of cooling term names
    exchange_terms : list[str]
        List of exchange term names
    params : dict
        Parameters for term evaluation
    n_iter : int
        Maximum outer iterations (default 3)
    tol : float
        Convergence tolerance (fractional change in Tg, default 0.01)
    max_bisect_iter : int
        Maximum bisection iterations (default 60)
    bisect_tol : float
        Bisection tolerance (default 1e-6)
    store_terms : bool
        Store per-term rates (default False)
        
    Returns
    -------
    ThermalResult
        Result with solved Tgas and auxiliary fields
    """

    return solve_thermal_balance_numba(
        state=state,
        heating_terms=heating_terms,
        cooling_terms=cooling_terms,
        exchange_terms=exchange_terms,
        params=params,
        n_iter=n_iter,
        tol=tol,
        max_bisect_iter=max_bisect_iter,
        bisect_tol=bisect_tol,
        store_terms=store_terms,
    )
