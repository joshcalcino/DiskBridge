"""Numba-accelerated kernels for thermal balance calculations.

This module contains pure numeric kernels that operate on flat float64 arrays.
No Pint, no dicts, no dataclasses - only numpy and numba.

All quantities are in CGS units:
- Temperature: K
- Number density: cm^-3
- Rates: erg cm^-3 s^-1
"""

import numpy as np
from numba import njit, prange

from diskbridge._constants import K_B, M_H

# Term bitmask flags for kernel dispatch
TERM_CR = 1 << 0     # cosmic ray heating
TERM_PE = 1 << 1     # photoelectric heating
TERM_CII = 1 << 2    # C II 158 um cooling
TERM_GD = 1 << 3     # gas-dust exchange


# -----------------------------------------------------------------------------
# Carbon ionization closure kernel
# -----------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def carbon_closure_kernel(
    nH: np.ndarray,
    chi: np.ndarray,
    Tg: np.ndarray,
    nco_total: np.ndarray,
    X_C_tot: float,
    Gamma_C0: float,
    alpha_rec_c0: float,
    T_rec_exp: float,
    nCplus_out: np.ndarray,
    nC_out: np.ndarray,
    ne_out: np.ndarray,
) -> None:
    """Update carbon ionization state for all cells.
    
    Solves photoionization-recombination equilibrium:
        Gamma_photo * n(C) = alpha_rec(T) * n(C+) * ne
    
    With constraint: n(C) + n(C+) + n(CO) = X_C_tot * nH
    And closure: ne = n(C+)
    
    Parameters
    ----------
    nH : ndarray
        H nuclei number density [cm^-3]
    chi : ndarray
        Effective UV field (dimensionless)
    Tg : ndarray
        Gas temperature [K]
    nco_total : ndarray
        Total CO number density (gas + ice) [cm^-3]
    X_C_tot : float
        Total carbon abundance relative to H
    Gamma_C0 : float
        C photoionization rate at chi=1 [s^-1]
    alpha_rec_c0 : float
        Recombination rate coefficient at 300K [cm^3/s]
    T_rec_exp : float
        Temperature exponent for recombination
    nCplus_out : ndarray
        Output: C+ number density [cm^-3]
    nC_out : ndarray
        Output: neutral C number density [cm^-3]
    ne_out : ndarray
        Output: electron number density [cm^-3]
    """
    N = nH.size
    for i in prange(N):
        n_C_tot = X_C_tot * nH[i]
        n_C_available = max(n_C_tot - nco_total[i], 0.0)
        
        Gamma_photo = Gamma_C0 * chi[i]
        alpha = alpha_rec_c0 * (Tg[i] / 300.0) ** T_rec_exp
        
        ratio = Gamma_photo / max(alpha, 1e-30)
        sqrt_ratio = np.sqrt(ratio)
        
        nCplus = n_C_available * sqrt_ratio / (1.0 + sqrt_ratio)
        nC = n_C_available - nCplus
        
        nCplus_out[i] = nCplus
        nC_out[i] = nC
        ne_out[i] = nCplus  # Milestone 1 closure: ne = nCplus


# -----------------------------------------------------------------------------
# Individual heating/cooling term functions (scalar, inlined)
# -----------------------------------------------------------------------------

@njit(inline='always', fastmath=True, cache=True)
def cosmic_ray_heating(nH: float, zeta_cr: float, heating_per_cr: float) -> float:
    """Cosmic ray heating rate.
    
    Returns
    -------
    float
        Heating rate [erg cm^-3 s^-1]
    """
    return nH * zeta_cr * heating_per_cr


@njit(inline='always', fastmath=True, cache=True)
def photoelectric_heating(
    nH: float,
    chi: float,
    ne: float,
    Tg: float,
    pah_scale: float,
    pe_heating_rate_0: float,
) -> float:
    """Photoelectric heating on dust grains (Bakes & Tielens 1994).
    
    Returns
    -------
    float
        Heating rate [erg cm^-3 s^-1]
    """
    ne_safe = max(ne, 1e-10)
    sqrt_Tg = np.sqrt(Tg)
    psi = sqrt_Tg * chi / ne_safe
    epsilon = 0.05 * np.sqrt(Tg / 1e4) / (1.0 + 4e-3 * psi)
    epsilon = min(max(epsilon, 1e-4), 0.1)
    
    return pah_scale * pe_heating_rate_0 * nH * chi * epsilon


@njit(inline='always', fastmath=True, cache=True)
def cii_cooling(
    nH: float,
    ne: float,
    nCplus: float,
    Tg: float,
    beta_cii: float,
    gamma_cii: float,
    E_cii: float,
    n_crit_cii: float,
    k_B: float,
) -> float:
    """C II 158 um fine-structure cooling.
    
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1] (positive)
    """
    n_coll = ne + 0.1 * nH
    n_coll = max(n_coll, 1e-10)
    
    Tg_safe = max(Tg, 10.0)
    excitation = np.exp(-E_cii / Tg_safe)
    
    rate = (
        nCplus * n_coll * gamma_cii * E_cii * k_B
        * excitation / (1.0 + n_crit_cii / n_coll)
        * beta_cii
    )
    return rate


@njit(inline='always', fastmath=True, cache=True)
def gas_dust_exchange(
    nH: float,
    Td: float,
    Tg: float,
    alpha_acc: float,
    sigma_dust: float,
    f_dust: float,
    k_B: float,
    m_H: float,
) -> float:
    """Gas-dust collisional energy exchange.
    
    Returns positive when Td > Tg (gas is heated by dust).
    
    Returns
    -------
    float
        Signed rate [erg cm^-3 s^-1]
    """
    T_mean = 0.5 * (Tg + Td)
    v_th = np.sqrt(8.0 * k_B * T_mean / (np.pi * m_H))
    
    n_dust = f_dust * nH
    
    rate = (
        alpha_acc * nH * n_dust * sigma_dust * v_th
        * 2.0 * k_B * (Td - Tg)
    )
    return rate


# -----------------------------------------------------------------------------
# Net heating function for a single cell
# -----------------------------------------------------------------------------

@njit(inline='always', fastmath=True, cache=True)
def net_heating_cell(
    Tg: float,
    nH: float,
    Td: float,
    chi: float,
    ne: float,
    nCplus: float,
    mask: int,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    gamma_cii: float,
    E_cii: float,
    n_crit_cii: float,
    sigma_dust: float,
    f_dust: float,
    k_B: float,
    m_H: float,
) -> float:
    """Compute net heating rate for a single cell.
    
    Returns
    -------
    float
        Net heating rate [erg cm^-3 s^-1] (positive = net heating)
    """
    net = 0.0
    
    if mask & TERM_CR:
        net += cosmic_ray_heating(nH, zeta_cr, heating_per_cr)
    
    if mask & TERM_PE:
        net += photoelectric_heating(nH, chi, ne, Tg, pah_scale, pe_heating_rate_0)
    
    if mask & TERM_CII:
        net -= cii_cooling(nH, ne, nCplus, Tg, beta_cii, gamma_cii, E_cii, n_crit_cii, k_B)
    
    if mask & TERM_GD:
        net += gas_dust_exchange(nH, Td, Tg, alpha_acc, sigma_dust, f_dust, k_B, m_H)
    
    return net


# -----------------------------------------------------------------------------
# Bisection solver for a single cell
# -----------------------------------------------------------------------------

@njit(inline='always', fastmath=True, cache=True)
def bisect_solve_cell(
    nH: float,
    Td: float,
    chi: float,
    ne: float,
    nCplus: float,
    mask: int,
    T_min: float,
    T_max: float,
    max_iter: int,
    tol: float,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    gamma_cii: float,
    E_cii: float,
    n_crit_cii: float,
    sigma_dust: float,
    f_dust: float,
    k_B: float,
    m_H: float,
) -> float:
    """Solve thermal balance for a single cell using bisection.
    
    Returns
    -------
    float
        Solved gas temperature [K]
    """
    a = T_min
    b = T_max
    
    fa = net_heating_cell(a, nH, Td, chi, ne, nCplus, mask,
                          zeta_cr, pah_scale, alpha_acc, beta_cii,
                          heating_per_cr, pe_heating_rate_0,
                          gamma_cii, E_cii, n_crit_cii,
                          sigma_dust, f_dust, k_B, m_H)
    fb = net_heating_cell(b, nH, Td, chi, ne, nCplus, mask,
                          zeta_cr, pah_scale, alpha_acc, beta_cii,
                          heating_per_cr, pe_heating_rate_0,
                          gamma_cii, E_cii, n_crit_cii,
                          sigma_dust, f_dust, k_B, m_H)
    
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    
    if fa * fb > 0.0:
        if Td > T_min and Td < T_max:
            return Td
        elif abs(fa) < abs(fb):
            return a
        else:
            return b
    
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = net_heating_cell(m, nH, Td, chi, ne, nCplus, mask,
                              zeta_cr, pah_scale, alpha_acc, beta_cii,
                              heating_per_cr, pe_heating_rate_0,
                              gamma_cii, E_cii, n_crit_cii,
                              sigma_dust, f_dust, k_B, m_H)
        
        if abs(fm) < tol:
            return m
        
        if fa * fm < 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    
    return 0.5 * (a + b)


# -----------------------------------------------------------------------------
# Parallel solver for all cells
# -----------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def solve_tgas_kernel(
    nH: np.ndarray,
    Td: np.ndarray,
    chi: np.ndarray,
    ne: np.ndarray,
    nCplus: np.ndarray,
    mask: int,
    T_min: float,
    T_max: float,
    max_iter: int,
    tol: float,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    gamma_cii: float,
    E_cii: float,
    n_crit_cii: float,
    sigma_dust: float,
    f_dust: float,
    k_B: float,
    m_H: float,
    Tg_out: np.ndarray,
) -> None:
    """Solve thermal balance for all cells in parallel.
    
    Parameters
    ----------
    nH : ndarray
        H nuclei number density [cm^-3]
    Td : ndarray
        Dust temperature [K]
    chi : ndarray
        Effective UV field (dimensionless)
    ne : ndarray
        Electron number density [cm^-3]
    nCplus : ndarray
        C+ number density [cm^-3]
    mask : int
        Bitmask of terms to include
    T_min : float
        Minimum temperature for bracket [K]
    T_max : float
        Maximum temperature for bracket [K]
    max_iter : int
        Maximum bisection iterations
    tol : float
        Convergence tolerance [erg cm^-3 s^-1]
    zeta_cr : float
        Cosmic ray ionization rate [s^-1]
    pah_scale : float
        PAH abundance scaling factor
    alpha_acc : float
        Accommodation coefficient for gas-dust
    beta_cii : float
        C II escape probability
    heating_per_cr : float
        Energy deposited per CR ionization [erg]
    pe_heating_rate_0 : float
        Base photoelectric heating rate [erg/s]
    gamma_cii : float
        C II collisional de-excitation rate [cm^3/s]
    E_cii : float
        C II excitation energy [K]
    n_crit_cii : float
        C II critical density [cm^-3]
    sigma_dust : float
        Dust cross section [cm^2]
    f_dust : float
        Dust-to-gas mass ratio
    k_B : float
        Boltzmann constant [CGS]
    m_H : float
        Hydrogen mass [g]
    Tg_out : ndarray
        Output: solved gas temperature [K]
    """
    N = nH.size
    for i in prange(N):
        Tg_out[i] = bisect_solve_cell(
            nH[i], Td[i], chi[i], ne[i], nCplus[i],
            mask, T_min, T_max, max_iter, tol,
            zeta_cr, pah_scale, alpha_acc, beta_cii,
            heating_per_cr, pe_heating_rate_0,
            gamma_cii, E_cii, n_crit_cii,
            sigma_dust, f_dust, k_B, m_H
        )


# -----------------------------------------------------------------------------
# Diagnostic kernel: compute per-term rates at final temperature
# -----------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def compute_terms_kernel(
    Tg: np.ndarray,
    nH: np.ndarray,
    Td: np.ndarray,
    chi: np.ndarray,
    ne: np.ndarray,
    nCplus: np.ndarray,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    gamma_cii: float,
    E_cii: float,
    n_crit_cii: float,
    sigma_dust: float,
    f_dust: float,
    k_B: float,
    m_H: float,
    rate_cr_out: np.ndarray,
    rate_pe_out: np.ndarray,
    rate_cii_out: np.ndarray,
    rate_gd_out: np.ndarray,
) -> None:
    """Compute individual heating/cooling rates for diagnostics.
    
    All output arrays have units [erg cm^-3 s^-1].
    Heating terms are positive, cooling terms are negative.
    """
    N = nH.size
    for i in prange(N):
        rate_cr_out[i] = cosmic_ray_heating(nH[i], zeta_cr, heating_per_cr)
        rate_pe_out[i] = photoelectric_heating(nH[i], chi[i], ne[i], Tg[i], pah_scale, pe_heating_rate_0)
        rate_cii_out[i] = -cii_cooling(nH[i], ne[i], nCplus[i], Tg[i], beta_cii, gamma_cii, E_cii, n_crit_cii, k_B)
        rate_gd_out[i] = gas_dust_exchange(nH[i], Td[i], Tg[i], alpha_acc, sigma_dust, f_dust, k_B, m_H)


# -----------------------------------------------------------------------------
# Utility: check convergence
# -----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def max_fractional_change(Tg_new: np.ndarray, Tg_old: np.ndarray) -> float:
    """Compute maximum fractional change in temperature.
    
    Note: Not parallelized to avoid reduction race conditions.
    This is a lightweight O(N) pass after the expensive solve.
    """
    N = Tg_new.size
    max_change = 0.0
    for i in range(N):
        denom = max(Tg_old[i], 1.0)
        change = abs(Tg_new[i] - Tg_old[i]) / denom
        if change > max_change:
            max_change = change
    return max_change
