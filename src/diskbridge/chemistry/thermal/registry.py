"""Thermal model registry.

Maps model names to solver configurations, following the same pattern as
chemistry registry.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from diskbridge.chemistry.thermal.types import ThermalState, ThermalResult


def run_thermal_balance(
    state: ThermalState,
    params: dict,
) -> "ThermalResult":
    """Unified thermal balance solver with configurable heating/cooling terms.
    
    Solves for gas temperature by balancing heating and cooling processes.
    By default includes all available terms:
    - Heating: photoelectric (on dust/PAHs), cosmic ray
    - Cooling: [C II] 158um, [C I] 609/370um, [O I] 63/145um, CO rotational
    - Exchange: gas-dust collisional coupling
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with required fields
    params : dict
        Solver parameters. Can override default term lists:
        - 'heating_terms': list of heating term names (default: all)
        - 'cooling_terms': list of cooling term names (default: all)
        - 'exchange_terms': list of exchange term names (default: all)
        
    Returns
    -------
    ThermalResult
        Solved gas temperature and auxiliary fields
        
    Examples
    --------
    Use all terms (default):
    >>> result = run_thermal_balance(state, params)
    
    Use only [C II] cooling:
    >>> params['cooling_terms'] = ['cii']
    >>> result = run_thermal_balance(state, params)
    """
    from diskbridge.chemistry.thermal.solver import solve_thermal_balance
    
    heating_terms = params.get('heating_terms', ["photoelectric", "cosmic_ray"])
    cooling_terms = params.get('cooling_terms', ["cii", "ci", "oi", "co"])
    exchange_terms = params.get('exchange_terms', ["gas_dust"])
    
    n_iter = params.get('n_iter', 3)
    tol = params.get('tol', 0.01)
    max_bisect_iter = params.get('max_bisect_iter', 60)
    bisect_tol = params.get('bisect_tol', 1e-6)
    store_terms = params.get('store_terms', False)
    
    return solve_thermal_balance(
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


THERMAL_REGISTRY: dict[str, Callable] = {
    "thermal_balance": run_thermal_balance,
}


def get_thermal_model(model_name: str) -> Callable:
    """Get thermal model function from registry.
    
    Parameters
    ----------
    model_name : str
        Name of thermal model
        
    Returns
    -------
    Callable
        Thermal model function
        
    Raises
    ------
    ValueError
        If model_name not in registry
    """
    if model_name not in THERMAL_REGISTRY:
        available = ', '.join(THERMAL_REGISTRY.keys())
        raise ValueError(
            f"Unknown thermal model: {model_name!r}. "
            f"Available models: {available}"
        )
    return THERMAL_REGISTRY[model_name]
