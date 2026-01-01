"""Thermal balance solver with modular term evaluation.

Solves Gamma_tot(Tg) - Lambda_tot(Tg) = 0 for gas temperature.

Supports two backends:
- "numba": Parallel Numba-accelerated kernels (default if numba available)
- "python": Pure Python with SciPy brentq (fallback)
"""

from __future__ import annotations

import numpy as np

from diskbridge._units import Quantity
from diskbridge._logging import logger
from diskbridge._constants import K_B, M_H
from diskbridge.chemistry.thermal.types import ThermalState, ThermalResult
from diskbridge.chemistry.thermal.terms import HEATING_TERMS, COOLING_TERMS, EXCHANGE_TERMS
from diskbridge.chemistry.thermal.carbon_closure import update_carbon_ions

try:
    from diskbridge.chemistry.thermal._kernels import (
        carbon_closure_kernel,
        solve_tgas_kernel,
        compute_terms_kernel,
        max_fractional_change,
        TERM_CR,
        TERM_PE,
        TERM_CII,
        TERM_GD,
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    TERM_CR = 1 << 0
    TERM_PE = 1 << 1
    TERM_CII = 1 << 2
    TERM_GD = 1 << 3


def evaluate_net_heating(
    Tgas_K: float,
    state: ThermalState,
    heating_terms: list[str],
    cooling_terms: list[str],
    exchange_terms: list[str],
    params: dict,
) -> float:
    """Evaluate net heating rate at a given gas temperature.
    
    Parameters
    ----------
    Tgas_K : float
        Gas temperature [K]
    state : ThermalState
        Thermal state (Tgas will be updated)
    heating_terms : list[str]
        List of heating term names to evaluate
    cooling_terms : list[str]
        List of cooling term names to evaluate
    exchange_terms : list[str]
        List of exchange term names to evaluate
    params : dict
        Parameters for term evaluation
        
    Returns
    -------
    float
        Net heating rate [erg cm^-3 s^-1] (positive = net heating)
    """
    state.Tgas = Quantity(Tgas_K, 'K')
    
    net_heating = 0.0
    
    for term_name in heating_terms:
        if term_name not in HEATING_TERMS:
            raise ValueError(f"Unknown heating term: {term_name}")
        term_func = HEATING_TERMS[term_name]
        rate = term_func(state, params)
        net_heating += rate.to('erg/(cm^3 * s)').magnitude
    
    for term_name in cooling_terms:
        if term_name not in COOLING_TERMS:
            raise ValueError(f"Unknown cooling term: {term_name}")
        term_func = COOLING_TERMS[term_name]
        rate = term_func(state, params)
        net_heating -= rate.to('erg/(cm^3 * s)').magnitude
    
    for term_name in exchange_terms:
        if term_name not in EXCHANGE_TERMS:
            raise ValueError(f"Unknown exchange term: {term_name}")
        term_func = EXCHANGE_TERMS[term_name]
        rate = term_func(state, params)
        net_heating += rate.to('erg/(cm^3 * s)').magnitude
    
    return net_heating


def solve_thermal_balance_cell(
    state: ThermalState,
    heating_terms: list[str],
    cooling_terms: list[str],
    exchange_terms: list[str],
    params: dict,
    T_min: float,
    T_max: float,
) -> float:
    """Solve thermal balance for a single cell.
    
    Parameters
    ----------
    state : ThermalState
        Thermal state for this cell
    heating_terms : list[str]
        Heating terms to include
    cooling_terms : list[str]
        Cooling terms to include
    exchange_terms : list[str]
        Exchange terms to include
    params : dict
        Parameters for term evaluation
    T_min : float
        Minimum temperature [K]
    T_max : float
        Maximum temperature [K]
        
    Returns
    -------
    float
        Solved gas temperature [K]
    """
    def residual(T_K):
        return evaluate_net_heating(
            T_K, state, heating_terms, cooling_terms, exchange_terms, params
        )
    
    f_min = residual(T_min)
    f_max = residual(T_max)
    
    Tdust_K = state.Tdust.to('K').magnitude
    
    if f_min * f_max > 0:
        if T_min < Tdust_K < T_max:
            return Tdust_K
        elif abs(f_min) < abs(f_max):
            return T_min
        else:
            return T_max
    
    try:
        from scipy.optimize import brentq
        T_solved = brentq(residual, T_min, T_max, xtol=1e-3)
        return T_solved
    except ValueError:
        return Tdust_K if T_min < Tdust_K < T_max else 0.5 * (T_min + T_max)


def _build_term_mask(
    heating_terms: list[str],
    cooling_terms: list[str],
    exchange_terms: list[str],
) -> int:
    """Convert term lists to bitmask for kernel dispatch."""
    mask = 0
    if "cosmic_ray" in heating_terms:
        mask |= TERM_CR
    if "photoelectric" in heating_terms:
        mask |= TERM_PE
    if "cii" in cooling_terms:
        mask |= TERM_CII
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
    def _to_float(val):
        """Convert Quantity or value to float magnitude in base units."""
        if hasattr(val, 'to_base_units'):
            return val.to_base_units().magnitude
        return float(val)
    
    T_min = _to_float(params['T_min'])
    T_max = _to_float(params['T_max'])
    zeta_cr = _to_float(params['zeta_cr'])
    Gamma_C0 = _to_float(params['Gamma_C0'])
    pah_scale = float(params['pah_scale'])
    X_C_tot = float(params['X_C_tot'])
    alpha_acc = float(params['alpha_acc'])
    beta_cii = float(params['beta_cii'])
    alpha_rec_c0 = float(params['alpha_rec_c0'])
    T_rec_exp = float(params['T_rec_exp'])
    heating_per_cr = _to_float(params['heating_per_cr'])
    pe_heating_rate_0 = _to_float(params['pe_heating_rate_0'])
    gamma_cii = _to_float(params['gamma_cii'])
    E_cii = _to_float(params['E_cii'])
    n_crit_cii = float(params['n_crit_cii'])
    sigma_dust = float(params['sigma_dust'])
    f_dust = float(params['f_dust'])
    
    mask = _build_term_mask(heating_terms, cooling_terms, exchange_terms)
    
    shape = state.nH.shape
    ncells = state.nH.size
    
    nH_flat = np.ascontiguousarray(state.nH.to('cm^-3').magnitude.flatten(), dtype=np.float64)
    Td_flat = np.ascontiguousarray(state.Tdust.to('K').magnitude.flatten(), dtype=np.float64)
    chi_flat = np.ascontiguousarray(state.chi_eff.magnitude.flatten(), dtype=np.float64)
    
    if state.nco_gas is not None:
        nco_gas_flat = state.nco_gas.to('cm^-3').magnitude.flatten()
    else:
        nco_gas_flat = np.zeros(ncells, dtype=np.float64)
    
    if state.nco_ice is not None:
        nco_ice_flat = state.nco_ice.to('cm^-3').magnitude.flatten()
    else:
        nco_ice_flat = np.zeros(ncells, dtype=np.float64)
    
    nco_total = np.ascontiguousarray(nco_gas_flat + nco_ice_flat, dtype=np.float64)
    
    Tg_flat = np.full(ncells, 0.5 * (T_min + T_max), dtype=np.float64)
    nCplus_flat = np.zeros(ncells, dtype=np.float64)
    nC_flat = np.zeros(ncells, dtype=np.float64)
    ne_flat = np.zeros(ncells, dtype=np.float64)
    
    converged = False
    max_change = 1.0
    
    for iteration in range(n_iter):
        logger.info(f"Thermal iteration {iteration + 1}/{n_iter} (numba backend)")
        
        if update_closure:
            carbon_closure_kernel(
                nH_flat, chi_flat, Tg_flat, nco_total,
                X_C_tot, Gamma_C0, alpha_rec_c0, T_rec_exp,
                nCplus_flat, nC_flat, ne_flat,
            )
            logger.info(f"  Updated carbon closure: max(nCplus)={np.max(nCplus_flat):.2e} cm^-3")
        
        Tg_old = Tg_flat.copy()
        
        solve_tgas_kernel(
            nH_flat, Td_flat, chi_flat, ne_flat, nCplus_flat,
            mask, T_min, T_max, max_bisect_iter, bisect_tol,
            zeta_cr, pah_scale, alpha_acc, beta_cii,
            heating_per_cr, pe_heating_rate_0,
            gamma_cii, E_cii, n_crit_cii,
            sigma_dust, f_dust, K_B, M_H,
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
        rate_gd = np.zeros(ncells, dtype=np.float64)
        
        compute_terms_kernel(
            Tg_flat, nH_flat, Td_flat, chi_flat, ne_flat, nCplus_flat,
            zeta_cr, pah_scale, alpha_acc, beta_cii,
            heating_per_cr, pe_heating_rate_0,
            gamma_cii, E_cii, n_crit_cii,
            sigma_dust, f_dust, K_B, M_H,
            rate_cr, rate_pe, rate_cii, rate_gd,
        )
        
        fields['rate_cosmic_ray'] = Quantity(rate_cr.reshape(shape), 'erg/(cm^3 * s)')
        fields['rate_photoelectric'] = Quantity(rate_pe.reshape(shape), 'erg/(cm^3 * s)')
        fields['rate_cii'] = Quantity(rate_cii.reshape(shape), 'erg/(cm^3 * s)')
        fields['rate_gas_dust'] = Quantity(rate_gd.reshape(shape), 'erg/(cm^3 * s)')
    
    meta = {
        'n_iter': iteration + 1,
        'converged': converged,
        'max_change': float(max_change),
        'heating_terms': heating_terms,
        'cooling_terms': cooling_terms,
        'exchange_terms': exchange_terms,
        'backend': 'numba',
        'params': params,
    }
    
    return ThermalResult(
        tgas=state.Tgas,
        fields=fields,
        meta=meta,
    )


def solve_thermal_balance_python(
    state: ThermalState,
    heating_terms: list[str],
    cooling_terms: list[str],
    exchange_terms: list[str],
    params: dict,
    n_iter: int = 3,
    tol: float = 0.01,
    update_closure: bool = True,
) -> ThermalResult:
    """Pure Python thermal balance solver (fallback).
    
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
    n_iter : int, optional
        Maximum outer iterations (default 3)
    tol : float, optional
        Convergence tolerance (fractional change in Tg, default 0.01)
    update_closure : bool, optional
        Whether to update carbon closure (default True)
        
    Returns
    -------
    ThermalResult
        Result with solved Tgas and auxiliary fields
    """
    def _to_float(val):
        if hasattr(val, 'to_base_units'):
            return val.to_base_units().magnitude
        return float(val)
    
    T_min = _to_float(params['T_min'])
    T_max = _to_float(params['T_max'])
    
    shape = state.nH.shape
    ncells = state.nH.size
    
    Tgas_K = np.full(shape, 0.5 * (T_min + T_max), dtype=float)
    state.Tgas = Quantity(Tgas_K, 'K')
    
    converged = False
    
    for iteration in range(n_iter):
        logger.info(f"Thermal iteration {iteration + 1}/{n_iter}")
        
        if update_closure:
            update_carbon_ions(state, params)
            logger.info(f"  Updated carbon closure: "
                       f"max(nCplus)={np.max(state.nCplus.magnitude):.2e} cm^-3")
        
        Tgas_old = Tgas_K.copy()
        
        Tgas_K_flat = Tgas_K.flatten()
        
        nH_flat = state.nH.to('cm^-3').magnitude.flatten()
        Tdust_flat = state.Tdust.to('K').magnitude.flatten()
        chi_eff_flat = state.chi_eff.magnitude.flatten()
        
        if state.nCplus is not None:
            nCplus_flat = state.nCplus.to('cm^-3').magnitude.flatten()
        else:
            nCplus_flat = None
        
        if state.ne is not None:
            ne_flat = state.ne.to('cm^-3').magnitude.flatten()
        else:
            ne_flat = None
        
        if state.nco_gas is not None:
            nco_gas_flat = state.nco_gas.to('cm^-3').magnitude.flatten()
        else:
            nco_gas_flat = None
        
        if state.nco_ice is not None:
            nco_ice_flat = state.nco_ice.to('cm^-3').magnitude.flatten()
        else:
            nco_ice_flat = None
        
        chunk_size = 10000
        n_chunks = (ncells + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            i_start = chunk_idx * chunk_size
            i_end = min((chunk_idx + 1) * chunk_size, ncells)
            
            for i in range(i_start, i_end):
                cell_state = ThermalState(
                    nH=Quantity(nH_flat[i], 'cm^-3'),
                    Tdust=Quantity(Tdust_flat[i], 'K'),
                    chi_eff=Quantity(chi_eff_flat[i], 'dimensionless'),
                    mesh=state.mesh,
                    Tgas=Quantity(Tgas_K_flat[i], 'K'),
                    nCplus=Quantity(nCplus_flat[i], 'cm^-3') if nCplus_flat is not None else None,
                    ne=Quantity(ne_flat[i], 'cm^-3') if ne_flat is not None else None,
                    nco_gas=Quantity(nco_gas_flat[i], 'cm^-3') if nco_gas_flat is not None else None,
                    nco_ice=Quantity(nco_ice_flat[i], 'cm^-3') if nco_ice_flat is not None else None,
                )
                
                Tgas_K_flat[i] = solve_thermal_balance_cell(
                    cell_state,
                    heating_terms,
                    cooling_terms,
                    exchange_terms,
                    params,
                    T_min,
                    T_max,
                )
        
        Tgas_K = Tgas_K_flat.reshape(shape)
        state.Tgas = Quantity(Tgas_K, 'K')
        
        max_change = np.max(np.abs(Tgas_K - Tgas_old) / np.maximum(Tgas_old, 1.0))
        logger.info(f"  Tgas: min={np.min(Tgas_K):.1f} K, "
                   f"max={np.max(Tgas_K):.1f} K, "
                   f"median={np.median(Tgas_K):.1f} K, "
                   f"max_change={max_change:.3f}")
        
        if max_change < tol:
            converged = True
            logger.info(f"  Converged after {iteration + 1} iterations")
            break
    
    fields = {}
    if state.nCplus is not None:
        fields['nCplus'] = state.nCplus
    if state.nC is not None:
        fields['nC'] = state.nC
    if state.ne is not None:
        fields['ne'] = state.ne
    
    meta = {
        'n_iter': iteration + 1,
        'converged': converged,
        'max_change': float(max_change),
        'heating_terms': heating_terms,
        'cooling_terms': cooling_terms,
        'exchange_terms': exchange_terms,
        'backend': 'python',
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
    backend: str = "auto",
    max_bisect_iter: int = 60,
    bisect_tol: float = 1e-6,
    store_terms: bool = False,
) -> ThermalResult:
    """Solve thermal balance with backend dispatch.
    
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
    backend : str
        Backend to use: "auto", "numba", or "python" (default "auto")
    max_bisect_iter : int
        Maximum bisection iterations (numba only, default 60)
    bisect_tol : float
        Bisection tolerance (numba only, default 1e-6)
    store_terms : bool
        Store per-term rates (numba only, default False)
        
    Returns
    -------
    ThermalResult
        Result with solved Tgas and auxiliary fields
    """
    if backend == "auto":
        backend = "numba" if NUMBA_AVAILABLE else "python"
    
    if backend == "numba":
        if not NUMBA_AVAILABLE:
            raise ImportError(
                "Numba backend requested but numba is not available. "
                "Install numba or use backend='python'."
            )
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
    elif backend == "python":
        return solve_thermal_balance_python(
            state=state,
            heating_terms=heating_terms,
            cooling_terms=cooling_terms,
            exchange_terms=exchange_terms,
            params=params,
            n_iter=n_iter,
            tol=tol,
            update_closure=update_closure,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'auto', 'numba', or 'python'.")
