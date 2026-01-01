"""Modular heating and cooling terms for thermal balance.

Each term is a pure function: state + params -> rate [erg cm^-3 s^-1]

Design principle: 
- Heating terms return positive rates
- Cooling terms return positive rates (will be subtracted in solver)
- Exchange terms (gas-dust) return signed rates (+ heats gas, - cools gas)
"""

from __future__ import annotations

import numpy as np

from diskbridge._units import Quantity
from diskbridge._constants import K_B, M_H
from diskbridge.chemistry.thermal.types import ThermalState


def term_cosmic_ray(state: ThermalState, params: dict) -> Quantity:
    """Cosmic ray heating.
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with nH
    params : dict
        Must contain 'zeta_cr' (ionization rate in s^-1)
        
    Returns
    -------
    Quantity
        Heating rate [erg cm^-3 s^-1]
        
    Notes
    -----
    Simple prescription: each ionization deposits ~20 eV into gas heating.
    """
    zeta_cr = params['zeta_cr']
    if hasattr(zeta_cr, 'to_base_units'):
        zeta_cr = zeta_cr.to_base_units().magnitude
    else:
        zeta_cr = float(zeta_cr)
    
    heating_per_cr = params['heating_per_cr']
    if hasattr(heating_per_cr, 'to_base_units'):
        heating_per_cr = heating_per_cr.to_base_units().magnitude
    else:
        heating_per_cr = float(heating_per_cr)
    
    nH_cm3 = state.nH.to_base_units().magnitude
    
    rate_erg_cm3_s = nH_cm3 * zeta_cr * heating_per_cr
    
    return Quantity(rate_erg_cm3_s, 'erg/(cm^3 * s)')


def term_photoelectric(state: ThermalState, params: dict) -> Quantity:
    """Photoelectric heating on dust grains and PAHs.
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with nH, chi_eff, Tgas, ne
    params : dict
        Must contain 'pah_scale' (dimensionless scaling factor)
        
    Returns
    -------
    Quantity
        Heating rate [erg cm^-3 s^-1]
        
    Notes
    -----
    Uses Bakes & Tielens (1994) fit with grain charging parameter.
    Heating efficiency epsilon ~ 0.05 * (Tg/1e4)^0.5 / (1 + psi)
    where psi ~ sqrt(Tg) * chi / ne characterizes grain charge.
    """
    pah_scale = float(params['pah_scale'])
    pe_heating_rate_0 = params['pe_heating_rate_0']
    if hasattr(pe_heating_rate_0, 'to_base_units'):
        pe_heating_rate_0 = pe_heating_rate_0.to_base_units().magnitude
    else:
        pe_heating_rate_0 = float(pe_heating_rate_0)
    
    nH_cm3 = state.nH.to_base_units().magnitude
    chi = state.chi_eff.magnitude
    
    if state.Tgas is None:
        raise ValueError("term_photoelectric requires Tgas in state")
    Tgas_K = state.Tgas.to_base_units().magnitude
    
    if state.ne is not None:
        ne_cm3 = state.ne.to_base_units().magnitude
    else:
        ne_cm3 = 1e-4 * nH_cm3
    
    ne_cm3 = np.maximum(ne_cm3, 1e-10)
    
    psi = np.sqrt(Tgas_K) * chi / ne_cm3
    epsilon = 0.05 * np.sqrt(Tgas_K / 1e4) / (1.0 + 4e-3 * psi)
    epsilon = np.clip(epsilon, 1e-4, 0.1)
    
    rate_erg_cm3_s = pah_scale * pe_heating_rate_0 * nH_cm3 * chi * epsilon
    
    return Quantity(rate_erg_cm3_s, 'erg/(cm^3 * s)')


def term_gas_dust_exchange(state: ThermalState, params: dict) -> Quantity:
    """Gas-dust collisional energy exchange.
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with nH, Tgas, Tdust
    params : dict
        Optional: 'alpha_acc' (accommodation coefficient, default 0.3)
        
    Returns
    -------
    Quantity
        Signed rate [erg cm^-3 s^-1]: positive heats gas, negative cools gas
        
    Notes
    -----
    Lambda_gd = alpha * n_gas * n_dust * sigma_d * v_th * 2 k_B (Tg - Td)
    
    Simplified: assume dust-to-gas ratio and grain properties,
    rate ~ nH * (Tg - Td) with a constant.
    """
    alpha_acc = float(params['alpha_acc'])
    sigma_dust = float(params['sigma_dust'])
    f_dust = float(params['f_dust'])
    
    if state.Tgas is None:
        raise ValueError("term_gas_dust_exchange requires Tgas in state")
    
    nH_cm3 = state.nH.to_base_units().magnitude
    Tgas_K = state.Tgas.to_base_units().magnitude
    Tdust_K = state.Tdust.to_base_units().magnitude
    
    T_mean = 0.5 * (Tgas_K + Tdust_K)
    v_th = np.sqrt(8.0 * K_B * T_mean / (np.pi * M_H))
    
    n_dust = f_dust * nH_cm3
    
    rate_erg_cm3_s = (
        alpha_acc * nH_cm3 * n_dust * sigma_dust * v_th 
        * 2.0 * K_B * (Tdust_K - Tgas_K)
    )
    
    return Quantity(rate_erg_cm3_s, 'erg/(cm^3 * s)')


def term_cii_cooling(state: ThermalState, params: dict) -> Quantity:
    """[C II] 158 micron fine-structure cooling.
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with nH, Tgas, nCplus, ne
    params : dict
        Optional: 'beta_cii' (escape probability, default 1.0 = optically thin)
        
    Returns
    -------
    Quantity
        Cooling rate [erg cm^-3 s^-1] (positive)
        
    Notes
    -----
    Two-level atom approximation:
    Lambda_CII = n(C+) * n_coll * gamma * E_CII * exp(-E_CII/Tg) / (1 + n_crit/n_coll)
    
    Colliders: e-, H, H2 (for now use simplified n_coll ~ ne + 0.1*nH)
    """
    beta_cii = float(params['beta_cii'])
    gamma_cii = params['gamma_cii']
    if hasattr(gamma_cii, 'to_base_units'):
        gamma_cii = gamma_cii.to_base_units().magnitude
    else:
        gamma_cii = float(gamma_cii)
    E_cii = params['E_cii']
    if hasattr(E_cii, 'to_base_units'):
        E_cii_K = E_cii.to_base_units().magnitude
    else:
        E_cii_K = float(E_cii)
    n_crit_cii = float(params['n_crit_cii'])
    
    if state.Tgas is None:
        raise ValueError("term_cii_cooling requires Tgas in state")
    if state.nCplus is None:
        return Quantity(np.zeros_like(state.nH.magnitude), 'erg/(cm^3 * s)')
    
    nCplus_cm3 = state.nCplus.to_base_units().magnitude
    nH_cm3 = state.nH.to_base_units().magnitude
    Tgas_K = state.Tgas.to_base_units().magnitude
    
    if state.ne is not None:
        ne_cm3 = state.ne.to_base_units().magnitude
    else:
        ne_cm3 = 1e-4 * nH_cm3
    
    n_coll = ne_cm3 + 0.1 * nH_cm3
    n_coll = np.maximum(n_coll, 1e-10)
    
    excitation_factor = np.exp(-E_cii_K / np.maximum(Tgas_K, 10.0))
    
    rate_erg_cm3_s = (
        nCplus_cm3 * n_coll * gamma_cii * E_cii_K * K_B
        * excitation_factor / (1.0 + n_crit_cii / n_coll)
        * beta_cii
    )
    
    return Quantity(rate_erg_cm3_s, 'erg/(cm^3 * s)')


HEATING_TERMS = {
    "cosmic_ray": term_cosmic_ray,
    "photoelectric": term_photoelectric,
}

COOLING_TERMS = {
    "cii": term_cii_cooling,
}

EXCHANGE_TERMS = {
    "gas_dust": term_gas_dust_exchange,
}
