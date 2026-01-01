"""Carbon ionization closure for thermal balance.

Provides a lightweight carbon chemistry closure to compute nCplus, nC, and ne
without requiring a full UMIST network.
"""

from __future__ import annotations

import numpy as np

from diskbridge._units import Quantity
from diskbridge.chemistry.thermal.types import ThermalState


def alpha_rec_c(T_K: np.ndarray, alpha_rec_c0: float, T_rec_exp: float) -> np.ndarray:
    """Radiative recombination rate for C+ + e- -> C.
    
    Parameters
    ----------
    T_K : ndarray
        Gas temperature [K]
    alpha_rec_c0 : float
        Recombination rate coefficient at 300K [cm^3/s]
    T_rec_exp : float
        Temperature exponent
        
    Returns
    -------
    ndarray
        Recombination rate [cm^3 s^-1]
        
    Notes
    -----
    UMIST-style fit: alpha_rec = A * (T/300)^beta
    """
    return alpha_rec_c0 * (T_K / 300.0)**T_rec_exp


def update_carbon_ions(state: ThermalState, params: dict) -> None:
    """Update carbon ionization state and electron density.
    
    Solves photoionization-recombination equilibrium:
    Gamma_photo * n(C) = alpha_rec(T) * n(C+) * ne
    
    With constraints:
    - n(C) + n(C+) + n(CO) = X_C_tot * nH
    - ne = n(C+) (Milestone 1 closure)
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with nH, chi_eff, Tgas
        Updates state.nCplus, state.nC, state.ne in place
    params : dict
        Optional keys:
        - 'X_C_tot': total carbon abundance (default 1.4e-4)
        - 'Gamma_C0': C photoionization rate at chi=1 [s^-1] (default 3e-10)
        
    Notes
    -----
    This is a simplified closure for Milestone 1. Later can be replaced
    with full UMIST network without touching thermal solver.
    """
    X_C_tot = float(params['X_C_tot'])
    Gamma_C0 = params['Gamma_C0']
    if hasattr(Gamma_C0, 'to_base_units'):
        Gamma_C0 = Gamma_C0.to_base_units().magnitude
    else:
        Gamma_C0 = float(Gamma_C0)
    alpha_rec_c0 = float(params['alpha_rec_c0'])
    T_rec_exp = float(params['T_rec_exp'])
    
    if state.Tgas is None:
        raise ValueError("update_carbon_ions requires Tgas in state")
    
    nH_cm3 = state.nH.to_base_units().magnitude
    chi = state.chi_eff.magnitude
    Tgas_K = state.Tgas.to_base_units().magnitude
    
    n_C_tot = X_C_tot * nH_cm3
    
    if state.nco_gas is not None:
        nco_gas_cm3 = state.nco_gas.to_base_units().magnitude
    else:
        nco_gas_cm3 = np.zeros_like(nH_cm3)
    
    if state.nco_ice is not None:
        nco_ice_cm3 = state.nco_ice.to_base_units().magnitude
    else:
        nco_ice_cm3 = np.zeros_like(nH_cm3)
    
    nco_total = nco_gas_cm3 + nco_ice_cm3
    
    n_C_available = np.maximum(n_C_tot - nco_total, 0.0)
    
    Gamma_photo = Gamma_C0 * chi
    alpha = alpha_rec_c(Tgas_K, alpha_rec_c0, T_rec_exp)
    
    ratio = Gamma_photo / np.maximum(alpha, 1e-30)
    
    nCplus_cm3 = n_C_available * np.sqrt(ratio) / (1.0 + np.sqrt(ratio))
    nC_cm3 = n_C_available - nCplus_cm3
    
    ne_cm3 = nCplus_cm3
    
    state.nCplus = Quantity(nCplus_cm3, 'cm^-3')
    state.nC = Quantity(nC_cm3, 'cm^-3')
    state.ne = Quantity(ne_cm3, 'cm^-3')
