"""Thermal balance solver with modular term evaluation.

Solves Gamma_tot(Tg) - Lambda_tot(Tg) = 0 for gas temperature.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import diskbridge
from diskbridge._units import Quantity
from diskbridge._logging import logger
from diskbridge.chemistry.thermal.types import ThermalState, ThermalResult
from diskbridge.chemistry.thermal.terms import HEATING_TERMS, COOLING_TERMS, EXCHANGE_TERMS
from diskbridge.chemistry.thermal.carbon_closure import update_carbon_ions


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
    
    if f_min * f_max > 0:
        if abs(f_min) < abs(f_max):
            return T_min
        else:
            return T_max
    
    try:
        T_solved = brentq(residual, T_min, T_max, xtol=1e-3)
        return T_solved
    except ValueError:
        return 0.5 * (T_min + T_max)


def solve_thermal_balance(
    state: ThermalState,
    heating_terms: list[str],
    cooling_terms: list[str],
    exchange_terms: list[str],
    params: dict,
    n_iter: int = 3,
    tol: float = 0.01,
    update_closure: bool = True,
) -> ThermalResult:
    """Solve thermal balance with optional iteration for closure.
    
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
    T_min_qty = params.get('T_min', diskbridge.params.T_min_solve)
    T_max_qty = params.get('T_max', diskbridge.params.T_max_solve)
    T_min = T_min_qty.to('K').magnitude
    T_max = T_max_qty.to('K').magnitude
    
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
        'params': params,
    }
    
    return ThermalResult(
        tgas=state.Tgas,
        fields=fields,
        meta=meta,
    )
