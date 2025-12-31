"""Modular heating and cooling terms for thermal balance.

Each term is a pure function: state + params -> rate [erg cm^-3 s^-1]

Design principle: 
- Heating terms return positive rates
- Cooling terms return positive rates (will be subtracted in solver)
- Exchange terms (gas-dust) return signed rates (+ heats gas, - cools gas)
"""

from __future__ import annotations

import numpy as np

from diskbridge._units import Quantity, units
from diskbridge.chemistry.thermal.types import ThermalState
from diskbridge.chemistry.thermal import constants as thermal_const

K_B = units('k_B')
M_H = units('m_H')


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
    zeta_cr = params.get('zeta_cr', thermal_const.zeta_cr_default)
    if not hasattr(zeta_cr, 'to'):
        zeta_cr = Quantity(zeta_cr, 's^-1')
    
    heating_per_ionization = Quantity(20.0, 'eV').to('erg')
    
    nH_cm3 = state.nH.to('cm^-3').magnitude
    zeta_s = zeta_cr.to('s^-1').magnitude
    
    rate_erg_cm3_s = nH_cm3 * zeta_s * heating_per_ionization.magnitude
    
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
    pah_scale = params.get('pah_scale', thermal_const.pah_scale_default)
    
    nH_cm3 = state.nH.to('cm^-3').magnitude
    chi = state.chi_eff.magnitude
    
    if state.Tgas is None:
        raise ValueError("term_photoelectric requires Tgas in state")
    Tgas_K = state.Tgas.to('K').magnitude
    
    if state.ne is not None:
        ne_cm3 = state.ne.to('cm^-3').magnitude
    else:
        ne_cm3 = 1e-4 * nH_cm3
    
    ne_cm3 = np.maximum(ne_cm3, 1e-10)
    
    psi = np.sqrt(Tgas_K) * chi / ne_cm3
    epsilon = 0.05 * np.sqrt(Tgas_K / 1e4) / (1.0 + 4e-3 * psi)
    epsilon = np.clip(epsilon, 1e-4, 0.1)
    
    heating_rate_0 = 1.3e-24
    
    rate_erg_cm3_s = pah_scale * heating_rate_0 * nH_cm3 * chi * epsilon
    
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
    alpha_acc = params.get('alpha_acc', 0.3)
    
    if state.Tgas is None:
        raise ValueError("term_gas_dust_exchange requires Tgas in state")
    
    nH_cm3 = state.nH.to('cm^-3').magnitude
    Tgas_K = state.Tgas.to('K').magnitude
    Tdust_K = state.Tdust.to('K').magnitude
    
    T_mean = 0.5 * (Tgas_K + Tdust_K)
    v_th = np.sqrt(8.0 * K_B.to('erg/K').magnitude * T_mean / (np.pi * M_H.to('g').magnitude))
    
    sigma_d = 1e-21
    f_dust = 0.01
    n_dust = f_dust * nH_cm3
    
    rate_erg_cm3_s = (
        alpha_acc * nH_cm3 * n_dust * sigma_d * v_th 
        * 2.0 * K_B.to('erg/K').magnitude * (Tdust_K - Tgas_K)
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
    beta_cii = params.get('beta_cii', 1.0)
    
    if state.Tgas is None:
        raise ValueError("term_cii_cooling requires Tgas in state")
    if state.nCplus is None:
        return Quantity(np.zeros_like(state.nH.magnitude), 'erg/(cm^3 * s)')
    
    nCplus_cm3 = state.nCplus.to('cm^-3').magnitude
    nH_cm3 = state.nH.to('cm^-3').magnitude
    Tgas_K = state.Tgas.to('K').magnitude
    
    if state.ne is not None:
        ne_cm3 = state.ne.to('cm^-3').magnitude
    else:
        ne_cm3 = 1e-4 * nH_cm3
    
    n_coll = ne_cm3 + 0.1 * nH_cm3
    n_coll = np.maximum(n_coll, 1e-10)
    
    gamma_cii = thermal_const.gamma_cii.to('cm^3/s').magnitude
    E_cii_K = thermal_const.E_cii.to('K').magnitude
    
    n_crit = 3e3
    
    excitation_factor = np.exp(-E_cii_K / np.maximum(Tgas_K, 10.0))
    
    rate_erg_cm3_s = (
        nCplus_cm3 * n_coll * gamma_cii * E_cii_K * K_B.to('erg/K').magnitude
        * excitation_factor / (1.0 + n_crit / n_coll)
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
