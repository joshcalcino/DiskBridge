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
    carbon_closure_kernel,
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
    update_closure: bool = True,
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
    update_closure : bool
        Whether to update carbon closure (default True)
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
        T_MIN_SOLVE, T_MAX_SOLVE, ZETA_CR, GAMMA_C0, PAH_SCALE,
        X_C_TOT, X_O_TOT, ALPHA_ACC, BETA_CII, BETA_CI10, BETA_CI20, BETA_CI21,
        BETA_OI10, BETA_OI20, BETA_OI21, BETA_CO, ALPHA_REC_C0, T_REC_EXP,
        HEATING_PER_CR, PE_HEATING_RATE_0, SIGMA_DUST, F_DUST,
    )
    
    CO_JMAX = 15
    
    mask = _build_term_mask(heating_terms, cooling_terms, exchange_terms)
    
    shape = state.nH.shape
    ncells = state.nH.size
    
    nH_flat = np.ascontiguousarray(state.nH.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    Td_flat = np.ascontiguousarray(state.Tdust.to('K').magnitude.flatten(), dtype=np.float64)
    chi_flat = np.ascontiguousarray(state.chi_eff.magnitude.flatten(), dtype=np.float64)
    
    if state.nH2 is not None:
        nH2_flat = np.ascontiguousarray(state.nH2.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    else:
        nH2_flat = np.ascontiguousarray(0.5 * nH_flat, dtype=np.float64)
        logger.warning("nH2 not provided, assuming nH2 = 0.5 * nH")
    
    if state.nH_atom is not None:
        nHI_flat = np.ascontiguousarray(state.nH_atom.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    else:
        nHI_flat = np.ascontiguousarray(nH_flat - 2.0 * nH2_flat, dtype=np.float64)
        nHI_flat = np.maximum(nHI_flat, 0.0)
    
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
    
    Tg_flat = np.full(ncells, 0.5 * (T_MIN_SOLVE + T_MAX_SOLVE), dtype=np.float64)
    nCplus_flat = np.zeros(ncells, dtype=np.float64)
    nC_flat = np.zeros(ncells, dtype=np.float64)
    ne_flat = np.zeros(ncells, dtype=np.float64)
    
    converged = False
    max_change = 1.0
    
    for iteration in range(n_iter):
        logger.info(f"Thermal iteration {iteration + 1}/{n_iter}")
        
        if update_closure:
            carbon_closure_kernel(
                nH_flat, chi_flat, Tg_flat, nco_total,
                X_C_TOT, GAMMA_C0, ALPHA_REC_C0, T_REC_EXP,
                nCplus_flat, nC_flat, ne_flat,
            )
            logger.info(f"  Updated carbon closure: max(nCplus)={np.max(nCplus_flat):.2e} cm^-3")
        
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
    state.nCplus = Quantity(nCplus_flat.reshape(shape), 'cm^-3')
    state.nC = Quantity(nC_flat.reshape(shape), 'cm^-3')
    state.ne = Quantity(ne_flat.reshape(shape), 'cm^-3')
    
    fields = {
        'nCplus': state.nCplus,
        'nC': state.nC,
        'ne': state.ne,
    }
    
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
    update_closure: bool = True,
    max_bisect_iter: int = 60,
    bisect_tol: float = 1e-6,
    store_terms: bool = False,
) -> ThermalResult:
    """Solve thermal balance using Numba-accelerated kernels.
    
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
        Parameters for term evaluation
    n_iter : int
        Maximum outer iterations (default 3)
    tol : float
        Convergence tolerance (fractional change in Tg, default 0.01)
    update_closure : bool
        Whether to update carbon closure (default True)
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
        update_closure=update_closure,
        max_bisect_iter=max_bisect_iter,
        bisect_tol=bisect_tol,
        store_terms=store_terms,
    )
