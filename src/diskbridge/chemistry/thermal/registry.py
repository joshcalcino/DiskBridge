"""Thermal model registry.

Maps model names to solver configurations, following the same pattern as
chemistry registry.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from diskbridge.chemistry.thermal.types import ThermalState, ThermalResult


def run_thermal_balance_v1(
    state: ThermalState,
    params: dict,
) -> "ThermalResult":
    """Milestone 1 thermal balance: PE + CR heating, C II + gas-dust cooling.
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with required fields
    params : dict
        Solver parameters (passed through to solver)
        
    Returns
    -------
    ThermalResult
        Solved gas temperature and auxiliary fields
    """
    from diskbridge.chemistry.thermal.solver import solve_thermal_balance
    
    heating_terms = ["photoelectric", "cosmic_ray"]
    cooling_terms = ["cii"]
    exchange_terms = ["gas_dust"]
    
    n_iter = params.get('n_iter', 3)
    tol = params.get('tol', 0.01)
    backend = params.get('backend', 'auto')
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
        update_closure=True,
        backend=backend,
        max_bisect_iter=max_bisect_iter,
        bisect_tol=bisect_tol,
        store_terms=store_terms,
    )


THERMAL_REGISTRY: dict[str, Callable] = {
    "thermal_balance_v1": run_thermal_balance_v1,
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
