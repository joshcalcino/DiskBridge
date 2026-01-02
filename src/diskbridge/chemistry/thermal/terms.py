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

def term_ci_cooling(state: ThermalState, params: dict) -> Quantity:
    """[C I] fine-structure cooling (609 um and 370 um lines).
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with nH, Tgas, nC
    params : dict
        Must contain parameters for both CI(1-0) 609um and CI(2-1) 370um transitions:
        'E_ci10', 'gamma_ci10', 'n_crit_ci10', 'beta_ci10' for 609um line
        'E_ci21', 'gamma_ci21', 'n_crit_ci21', 'beta_ci21' for 370um line
        
    Returns
    -------
    Quantity
        Cooling rate [erg cm^-3 s^-1] (positive)
        
    Notes (physics)
    -----
    Cooling is computed as radiative loss from collisionally excited line emission.
    We use a two-level approximation with an escape-probability factor (beta) to mimic
    optical-depth effects. This approach is widely used in thermo-chemical disk and PDR
    modeling to compute T_gas from a local heating-cooling balance without performing
    full line radiative transfer inside the thermal solver. [C I] is a key coolant in
    intermediate layers where carbon transitions from C+ to CO (Tielens & Hollenbach 1985).
    """
    if state.Tgas is None:
        raise ValueError("term_ci_cooling requires Tgas in state")
    if state.nC is None:
        return Quantity(np.zeros_like(state.nH.magnitude), 'erg/(cm^3 * s)')
    
    nC_cm3 = state.nC.to_base_units().magnitude
    nH_cm3 = state.nH.to_base_units().magnitude
    Tgas_K = state.Tgas.to_base_units().magnitude
    
    if state.ne is not None:
        ne_cm3 = state.ne.to_base_units().magnitude
    else:
        ne_cm3 = 1e-4 * nH_cm3
    
    n_coll = ne_cm3 + 0.1 * nH_cm3
    n_coll = np.maximum(n_coll, 1e-10)
    
    Tgas_safe = np.maximum(Tgas_K, 10.0)
    
    beta_ci10 = float(params['beta_ci10'])
    gamma_ci10 = params['gamma_ci10']
    if hasattr(gamma_ci10, 'to_base_units'):
        gamma_ci10 = gamma_ci10.to_base_units().magnitude
    else:
        gamma_ci10 = float(gamma_ci10)
    E_ci10 = params['E_ci10']
    if hasattr(E_ci10, 'to_base_units'):
        E_ci10_K = E_ci10.to_base_units().magnitude
    else:
        E_ci10_K = float(E_ci10)
    n_crit_ci10 = float(params['n_crit_ci10'])
    
    excitation_10 = np.exp(-E_ci10_K / Tgas_safe)
    rate_10 = (
        nC_cm3 * n_coll * gamma_ci10 * E_ci10_K * K_B
        * excitation_10 / (1.0 + n_crit_ci10 / n_coll)
        * beta_ci10
    )
    
    beta_ci21 = float(params['beta_ci21'])
    gamma_ci21 = params['gamma_ci21']
    if hasattr(gamma_ci21, 'to_base_units'):
        gamma_ci21 = gamma_ci21.to_base_units().magnitude
    else:
        gamma_ci21 = float(gamma_ci21)
    E_ci21 = params['E_ci21']
    if hasattr(E_ci21, 'to_base_units'):
        E_ci21_K = E_ci21.to_base_units().magnitude
    else:
        E_ci21_K = float(E_ci21)
    n_crit_ci21 = float(params['n_crit_ci21'])
    
    excitation_21 = np.exp(-E_ci21_K / Tgas_safe)
    rate_21 = (
        nC_cm3 * n_coll * gamma_ci21 * E_ci21_K * K_B
        * excitation_21 / (1.0 + n_crit_ci21 / n_coll)
        * beta_ci21
    )
    
    return Quantity(rate_10 + rate_21, 'erg/(cm^3 * s)')


def term_oi_cooling(state: ThermalState, params: dict) -> Quantity:
    """[O I] fine-structure cooling (63 um and 145 um lines).
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with nH, Tgas, nco_gas, nco_ice (or nO if available)
    params : dict
        Must contain parameters for OI transitions:
        'E_oi63', 'gamma_oi63', 'n_crit_oi63', 'beta_oi63' for 63um line
        'E_oi145', 'gamma_oi145', 'n_crit_oi145', 'beta_oi145' for 145um line
        'X_O_tot' for total oxygen abundance if nO not in state
        
    Returns
    -------
    Quantity
        Cooling rate [erg cm^-3 s^-1] (positive)
        
    Notes (physics)
    -----
    Cooling is computed as radiative loss from collisionally excited line emission.
    We use a two-level approximation with an escape-probability factor (beta) to mimic
    optical-depth effects. This approach is widely used in thermo-chemical disk and PDR
    modeling to compute T_gas from a local heating-cooling balance without performing
    full line radiative transfer inside the thermal solver. [O I] 63um is a principal
    coolant in warm neutral gas / PDR surfaces (Hollenbach & Tielens 1997).
    """
    if state.Tgas is None:
        raise ValueError("term_oi_cooling requires Tgas in state")
    
    nH_cm3 = state.nH.to_base_units().magnitude
    Tgas_K = state.Tgas.to_base_units().magnitude
    
    if state.nO is not None:
        nO_cm3 = state.nO.to_base_units().magnitude
    else:
        X_O_tot = float(params['X_O_tot'])
        if state.nco_gas is not None and state.nco_ice is not None:
            nco_gas_cm3 = state.nco_gas.to_base_units().magnitude
            nco_ice_cm3 = state.nco_ice.to_base_units().magnitude
            nco_total_cm3 = nco_gas_cm3 + nco_ice_cm3
        else:
            nco_total_cm3 = 0.0
        
        nO_cm3 = np.maximum(X_O_tot * nH_cm3 - nco_total_cm3, 0.0)
    
    if state.ne is not None:
        ne_cm3 = state.ne.to_base_units().magnitude
    else:
        ne_cm3 = 1e-4 * nH_cm3
    
    n_coll = ne_cm3 + 0.1 * nH_cm3
    n_coll = np.maximum(n_coll, 1e-10)
    
    Tgas_safe = np.maximum(Tgas_K, 10.0)
    
    beta_oi63 = float(params['beta_oi63'])
    gamma_oi63 = params['gamma_oi63']
    if hasattr(gamma_oi63, 'to_base_units'):
        gamma_oi63 = gamma_oi63.to_base_units().magnitude
    else:
        gamma_oi63 = float(gamma_oi63)
    E_oi63 = params['E_oi63']
    if hasattr(E_oi63, 'to_base_units'):
        E_oi63_K = E_oi63.to_base_units().magnitude
    else:
        E_oi63_K = float(E_oi63)
    n_crit_oi63 = float(params['n_crit_oi63'])
    
    excitation_63 = np.exp(-E_oi63_K / Tgas_safe)
    rate_63 = (
        nO_cm3 * n_coll * gamma_oi63 * E_oi63_K * K_B
        * excitation_63 / (1.0 + n_crit_oi63 / n_coll)
        * beta_oi63
    )
    
    beta_oi145 = float(params['beta_oi145'])
    gamma_oi145 = params['gamma_oi145']
    if hasattr(gamma_oi145, 'to_base_units'):
        gamma_oi145 = gamma_oi145.to_base_units().magnitude
    else:
        gamma_oi145 = float(gamma_oi145)
    E_oi145 = params['E_oi145']
    if hasattr(E_oi145, 'to_base_units'):
        E_oi145_K = E_oi145.to_base_units().magnitude
    else:
        E_oi145_K = float(E_oi145)
    n_crit_oi145 = float(params['n_crit_oi145'])
    
    excitation_145 = np.exp(-E_oi145_K / Tgas_safe)
    rate_145 = (
        nO_cm3 * n_coll * gamma_oi145 * E_oi145_K * K_B
        * excitation_145 / (1.0 + n_crit_oi145 / n_coll)
        * beta_oi145
    )
    
    return Quantity(rate_63 + rate_145, 'erg/(cm^3 * s)')


def term_co_rot_cooling(state: ThermalState, params: dict) -> Quantity:
    """CO rotational cooling (multi-level).
    
    Parameters
    ----------
    state : ThermalState
        Thermal state with nH, Tgas, nco_gas
    params : dict
        Must contain 'L_co_coeff' and 'beta_co'
        
    Returns
    -------
    Quantity
        Cooling rate [erg cm^-3 s^-1] (positive)
        
    Notes (physics)
    -----
    CO has many rotational levels, so "two-level" is too crude. Standard practice is to
    use tabulated/fitted cooling functions derived from escape-probability/LVG calculations
    (Neufeld & Kaufman 1993). This implementation uses a simple optically-thin fit:
    Lambda_CO = n(CO) * n_coll * L_CO(T) * beta_CO
    where L_CO is a cooling function. For v1, we use a power-law approximation:
    L_CO ~ T^0.5 scaled to match typical LVG results at T~100K.
    """
    if state.Tgas is None:
        raise ValueError("term_co_rot_cooling requires Tgas in state")
    if state.nco_gas is None:
        return Quantity(np.zeros_like(state.nH.magnitude), 'erg/(cm^3 * s)')
    
    nco_gas_cm3 = state.nco_gas.to_base_units().magnitude
    nH_cm3 = state.nH.to_base_units().magnitude
    Tgas_K = state.Tgas.to_base_units().magnitude
    
    if state.ne is not None:
        ne_cm3 = state.ne.to_base_units().magnitude
    else:
        ne_cm3 = 1e-4 * nH_cm3
    
    n_coll = ne_cm3 + 0.5 * nH_cm3
    n_coll = np.maximum(n_coll, 1e-10)
    
    Tgas_safe = np.maximum(Tgas_K, 10.0)
    
    beta_co = float(params['beta_co'])
    L_co_coeff = params['L_co_coeff']
    if hasattr(L_co_coeff, 'to_base_units'):
        L_co_coeff = L_co_coeff.to_base_units().magnitude
    else:
        L_co_coeff = float(L_co_coeff)
    
    L_co = L_co_coeff * np.sqrt(Tgas_safe / 100.0)
    
    rate_erg_cm3_s = nco_gas_cm3 * n_coll * L_co * beta_co
    
    return Quantity(rate_erg_cm3_s, 'erg/(cm^3 * s)')


COOLING_TERMS = {
    "cii": term_cii_cooling,
    "ci": term_ci_cooling,
    "oi": term_oi_cooling,
    "co": term_co_rot_cooling,
}

EXCHANGE_TERMS = {
    "gas_dust": term_gas_dust_exchange,
}
