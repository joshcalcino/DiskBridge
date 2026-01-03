"""Numba-accelerated kernels for thermal balance calculations.

This module contains pure numeric kernels that operate on flat float64 arrays.
No Pint, no dicts, no dataclasses - only numpy and numba.

All quantities are in CGS units:
- Temperature: K
- Number density: cm^-3
- Rates: erg cm^-3 s^-1

References
----------
Atomic/molecular data (level energies, Einstein A values, and collisional
rate coefficients) are taken from the Leiden Atomic and Molecular Database
(LAMDA; Schoier et al. 2005, A&A 432, 369; van der Tak et al. 2020, Atoms 8, 15).

Species-specific collision rate sources:
- C+: Wiesenfeld & Goldsmith (2014, ApJ 780, 183)
- C, O: Abrahamsson et al. (2007, ApJ 654, 1171)
- CO: Yang et al. (2010, ApJ 718, 1062)
"""

import numpy as np
from numba import njit, prange

from diskbridge._constants import K_B, M_H

from diskbridge.chemistry.thermal.lamda import (
    CPLUS_E_UL_K, CPLUS_G_LEVELS, CPLUS_A_UL, CPLUS_HNU,
    CPLUS_COLL_PH2_T, CPLUS_COLL_PH2_Q,
    CPLUS_COLL_OH2_T, CPLUS_COLL_OH2_Q,
    CPLUS_COLL_H_T, CPLUS_COLL_H_Q,
    CPLUS_COLL_E_T, CPLUS_COLL_E_Q,
    C_E_LEVELS_K, C_G_LEVELS, C_TRANS_U, C_TRANS_L, C_A_UL, C_HNU,
    C_COLL_PH2_T, C_COLL_PH2_Q,
    C_COLL_OH2_T, C_COLL_OH2_Q,
    C_COLL_H_T, C_COLL_H_Q,
    C_COLL_E_T, C_COLL_E_Q,
    O_E_LEVELS_K, O_G_LEVELS, O_TRANS_U, O_TRANS_L, O_A_UL, O_HNU,
    O_COLL_PH2_T, O_COLL_PH2_Q,
    O_COLL_OH2_T, O_COLL_OH2_Q,
    O_COLL_H_T, O_COLL_H_Q,
    O_COLL_E_T, O_COLL_E_Q,
    CO_N_LEVELS, CO_E_LEVELS_K, CO_G_LEVELS, CO_N_TRANS,
    CO_TRANS_U, CO_TRANS_L, CO_A_UL, CO_HNU,
    CO_COLL_PH2_T, CO_COLL_PH2_Q,
    CO_COLL_OH2_T, CO_COLL_OH2_Q,
)

from diskbridge._constants import (
    OPR_DIFFUSE, OPR_T0, OPR_T1, OPR_F0, OPR_F1, OPR_CAP,
)

# Term bitmask flags for kernel dispatch
TERM_CR = 1 << 0     # cosmic ray heating
TERM_PE = 1 << 1     # photoelectric heating
TERM_CII = 1 << 2    # C II 158 um cooling
TERM_GD = 1 << 3     # gas-dust exchange
TERM_CI = 1 << 4     # C I 609/370 um cooling
TERM_OI = 1 << 5     # O I 63/145 um cooling
TERM_CO = 1 << 6     # CO rotational cooling


# -----------------------------------------------------------------------------
# Blended ortho/para H2 ratio model
# -----------------------------------------------------------------------------
# Uses environment-dependent OPR: diffuse/warm -> OPR=3, cold+molecular -> LTE
# No H2 chemistry required - just mixes collision rates based on T and fH2

@njit(fastmath=True, cache=True)
def h2_molecular_fraction(nH2: float, nH_atom: float) -> float:
    """Compute molecular hydrogen fraction fH2 = 2*nH2 / (2*nH2 + nH_atom)."""
    denom = 2.0 * nH2 + nH_atom
    if denom <= 0.0:
        return 0.0
    return (2.0 * nH2) / denom


@njit(fastmath=True, cache=True)
def opr_lte(T: float) -> float:
    """LTE ortho/para ratio approximation: OPR_LTE(T) ~ 9*exp(-170.5/T).
    
    This is a cheap approximation valid for collider mixing purposes.
    Capped at OPR_CAP (statistical limit = 3).
    """
    if T <= 0.0:
        return 0.0
    opr = 9.0 * np.exp(-170.5 / T)
    if opr > OPR_CAP:
        opr = OPR_CAP
    return opr


@njit(fastmath=True, cache=True)
def clamp01(x: float) -> float:
    """Clamp value to [0, 1] range."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@njit(fastmath=True, cache=True)
def f_ortho_h2(T: float, nH2: float, nH_atom: float) -> float:
    """Compute effective ortho-H2 fraction using blended OPR model.
    
    Blends between:
    - Diffuse/warm limit: OPR = OPR_DIFFUSE (3.0, statistical)
    - Cold molecular limit: OPR -> LTE(T)
    
    The blend weight depends on:
    - Molecular fraction fH2 (high fH2 -> push toward LTE)
    - Temperature (low T -> push toward LTE)
    
    Returns f_ortho = OPR / (1 + OPR), i.e. ortho fraction.
    """
    fH2 = h2_molecular_fraction(nH2, nH_atom)
    
    # Weight for molecular fraction: 0 if fH2 < F0, 1 if fH2 > F1
    wf = clamp01((fH2 - OPR_F0) / (OPR_F1 - OPR_F0))
    
    # Weight for temperature: 1 if T < T0 (cold), 0 if T > T1 (warm)
    wT = clamp01((OPR_T1 - T) / (OPR_T1 - OPR_T0))
    
    # Combined weight: push toward LTE only if both molecular AND cold
    w = wf * wT
    
    # Blend OPR
    opr = (1.0 - w) * OPR_DIFFUSE + w * opr_lte(T)
    if opr < 0.0:
        opr = 0.0
    if opr > OPR_CAP:
        opr = OPR_CAP
    
    # Convert OPR to ortho fraction: f = OPR / (1 + OPR)
    return opr / (1.0 + opr)


@njit(fastmath=True, cache=True)
def interp_rate_mix(
    T: float,
    T_grid: np.ndarray,
    q_p: np.ndarray,
    q_o: np.ndarray,
    f_ortho: float,
) -> float:
    """Interpolate and mix para-H2 and ortho-H2 collision rates.
    
    q_mixed = (1 - f_ortho) * q_para + f_ortho * q_ortho
    
    Parameters
    ----------
    T : float
        Temperature [K]
    T_grid : ndarray
        Temperature grid (same for both p and o)
    q_p : ndarray
        Para-H2 collision rate at each grid point [cm^3/s]
    q_o : ndarray
        Ortho-H2 collision rate at each grid point [cm^3/s]
    f_ortho : float
        Ortho-H2 fraction (from f_ortho_h2)
        
    Returns
    -------
    float
        Mixed collision rate [cm^3/s]
    """
    # Interpolate both rates
    n = len(T_grid)
    
    # Clamp to grid bounds
    if T <= T_grid[0]:
        qp = q_p[0]
        qo = q_o[0]
    elif T >= T_grid[n - 1]:
        qp = q_p[n - 1]
        qo = q_o[n - 1]
    else:
        # Find bracketing indices
        i = 0
        while i < n - 1 and T_grid[i + 1] < T:
            i += 1
        # Linear interpolation
        t = (T - T_grid[i]) / (T_grid[i + 1] - T_grid[i])
        qp = q_p[i] + t * (q_p[i + 1] - q_p[i])
        qo = q_o[i] + t * (q_o[i + 1] - q_o[i])
    
    # Mix
    return (1.0 - f_ortho) * qp + f_ortho * qo


@njit(fastmath=True, cache=True)
def interp_rate_mix_2grid(
    T: float,
    T_p: np.ndarray,
    q_p: np.ndarray,
    T_o: np.ndarray,
    q_o: np.ndarray,
    f_ortho: float,
) -> float:
    """Interpolate and mix para-H2 and ortho-H2 rates with different T-grids.
    
    For species like CO where pH2 and oH2 may have different temperature grids.
    
    Parameters
    ----------
    T : float
        Temperature [K]
    T_p : ndarray
        Temperature grid for para-H2
    q_p : ndarray
        Para-H2 collision rate at each grid point [cm^3/s]
    T_o : ndarray
        Temperature grid for ortho-H2
    q_o : ndarray
        Ortho-H2 collision rate at each grid point [cm^3/s]
    f_ortho : float
        Ortho-H2 fraction (from f_ortho_h2)
        
    Returns
    -------
    float
        Mixed collision rate [cm^3/s]
    """
    qp = interp_rate(T, T_p, q_p)
    qo = interp_rate(T, T_o, q_o)
    return (1.0 - f_ortho) * qp + f_ortho * qo


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
# Linear interpolation helper for collision rates
# -----------------------------------------------------------------------------

@njit(inline='always', fastmath=True, cache=True)
def interp_rate(T: float, T_grid: np.ndarray, q_grid: np.ndarray) -> float:
    """Linear interpolation of collision rate coefficient.
    
    Parameters
    ----------
    T : float
        Temperature [K]
    T_grid : ndarray
        Temperature grid points [K]
    q_grid : ndarray
        Rate coefficients at grid points [cm^3/s]
        
    Returns
    -------
    float
        Interpolated rate coefficient [cm^3/s]
    """
    n = len(T_grid)
    if T <= T_grid[0]:
        return q_grid[0]
    if T >= T_grid[n-1]:
        return q_grid[n-1]
    
    for i in range(n - 1):
        if T_grid[i] <= T < T_grid[i+1]:
            frac = (T - T_grid[i]) / (T_grid[i+1] - T_grid[i])
            return q_grid[i] + frac * (q_grid[i+1] - q_grid[i])
    
    return q_grid[n-1]


# -----------------------------------------------------------------------------
# 2-level statistical equilibrium solver (for C+)
# -----------------------------------------------------------------------------

@njit(inline='always', fastmath=True, cache=True)
def solve_2level_se(
    n_X: float,
    Tg: float,
    n_H2: float,
    n_HI: float,
    ne: float,
    beta: float,
    E_ul: float,
    g_u: float,
    g_l: float,
    A_ul: float,
    h_nu: float,
    T_H2: np.ndarray,
    q_H2: np.ndarray,
    T_HI: np.ndarray,
    q_HI: np.ndarray,
    T_e: np.ndarray,
    q_e: np.ndarray,
) -> float:
    """Solve 2-level statistical equilibrium and return cooling rate.
    
    Uses de-excitation rates from LAMDA and computes excitation by detailed balance.
    
    Parameters
    ----------
    n_X : float
        Total species number density [cm^-3]
    Tg : float
        Gas temperature [K]
    n_H2 : float
        H2 number density [cm^-3]
    n_HI : float
        Atomic H number density [cm^-3]
    ne : float
        Electron number density [cm^-3]
    beta : float
        Escape probability (1 = optically thin)
    E_ul : float
        Transition energy [K]
    g_u, g_l : float
        Statistical weights of upper and lower levels
    A_ul : float
        Einstein A coefficient [s^-1]
    h_nu : float
        Photon energy h*nu [erg]
    T_H2, q_H2 : ndarray
        Temperature grid and de-excitation rates for H2 collisions
    T_HI, q_HI : ndarray
        Temperature grid and de-excitation rates for H collisions
    T_e, q_e : ndarray
        Temperature grid and de-excitation rates for electron collisions
        
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1]
    """
    if n_X <= 0.0 or Tg <= 0.0:
        return 0.0
    
    q_ul_H2 = interp_rate(Tg, T_H2, q_H2) if len(T_H2) > 0 else 0.0
    q_ul_HI = interp_rate(Tg, T_HI, q_HI) if len(T_HI) > 0 else 0.0
    q_ul_e = interp_rate(Tg, T_e, q_e) if len(T_e) > 0 else 0.0
    
    boltz = np.exp(-E_ul / max(Tg, 1.0))
    g_ratio = g_u / g_l
    
    q_lu_H2 = q_ul_H2 * g_ratio * boltz
    q_lu_HI = q_ul_HI * g_ratio * boltz
    q_lu_e = q_ul_e * g_ratio * boltz
    
    C_ul = n_H2 * q_ul_H2 + n_HI * q_ul_HI + ne * q_ul_e
    C_lu = n_H2 * q_lu_H2 + n_HI * q_lu_HI + ne * q_lu_e
    
    denom = C_lu + C_ul + A_ul * beta
    if denom <= 0.0:
        return 0.0
    
    n_u = n_X * C_lu / denom
    
    cooling = n_u * A_ul * h_nu * beta
    
    return cooling


# -----------------------------------------------------------------------------
# 3-level statistical equilibrium solver (for C and O)
# -----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def solve_3level_se(
    n_X: float,
    Tg: float,
    n_H2: float,
    n_HI: float,
    ne: float,
    beta_01: float,
    beta_02: float,
    beta_12: float,
    E_levels: np.ndarray,
    g_levels: np.ndarray,
    trans_u: np.ndarray,
    trans_l: np.ndarray,
    A_ul: np.ndarray,
    h_nu: np.ndarray,
    T_H2: np.ndarray,
    q_H2: np.ndarray,
    T_HI: np.ndarray,
    q_HI: np.ndarray,
    T_e: np.ndarray,
    q_e: np.ndarray,
) -> float:
    """Solve 3-level statistical equilibrium and return total cooling rate.
    
    Solves the coupled rate equations for a 3-level system using Gaussian elimination.
    
    Parameters
    ----------
    n_X : float
        Total species number density [cm^-3]
    Tg : float
        Gas temperature [K]
    n_H2, n_HI, ne : float
        Collider densities [cm^-3]
    beta_01, beta_02, beta_12 : float
        Escape probabilities for each transition
    E_levels : ndarray (3,)
        Level energies [K]
    g_levels : ndarray (3,)
        Statistical weights
    trans_u, trans_l : ndarray
        Upper and lower level indices for each transition
    A_ul : ndarray
        Einstein A coefficients [s^-1]
    h_nu : ndarray
        Photon energies h*nu [erg]
    T_H2, q_H2 : ndarray
        Temperature grid and de-excitation rates for H2 (shape: n_trans x n_T)
    T_HI, q_HI : ndarray
        Temperature grid and de-excitation rates for H
    T_e, q_e : ndarray
        Temperature grid and de-excitation rates for electrons
        
    Returns
    -------
    float
        Total cooling rate [erg cm^-3 s^-1]
    """
    if n_X <= 0.0 or Tg <= 0.0:
        return 0.0
    
    # Use np.empty + manual zeroing to minimize allocation overhead
    C = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            C[i, j] = 0.0
    
    n_trans = len(trans_u)
    for t in range(n_trans):
        u = trans_u[t]
        l = trans_l[t]
        
        E_ul = E_levels[u] - E_levels[l]
        g_ratio = g_levels[u] / g_levels[l]
        boltz = np.exp(-E_ul / max(Tg, 1.0))
        
        q_ul_H2 = interp_rate(Tg, T_H2, q_H2[t]) if len(T_H2) > 0 else 0.0
        q_ul_HI = interp_rate(Tg, T_HI, q_HI[t]) if len(T_HI) > 0 else 0.0
        q_ul_e = interp_rate(Tg, T_e, q_e[t]) if len(T_e) > 0 else 0.0
        
        q_lu_H2 = q_ul_H2 * g_ratio * boltz
        q_lu_HI = q_ul_HI * g_ratio * boltz
        q_lu_e = q_ul_e * g_ratio * boltz
        
        C_ul = n_H2 * q_ul_H2 + n_HI * q_ul_HI + ne * q_ul_e
        C_lu = n_H2 * q_lu_H2 + n_HI * q_lu_HI + ne * q_lu_e
        
        if u == 1 and l == 0:
            beta_t = beta_01
        elif u == 2 and l == 0:
            beta_t = beta_02
        else:
            beta_t = beta_12
        
        R_ul = A_ul[t] * beta_t + C_ul
        R_lu = C_lu
        
        C[l, u] += R_ul
        C[u, l] += R_lu
        C[u, u] -= R_ul
        C[l, l] -= R_lu
    
    # Use np.empty + explicit initialization
    M = np.empty((3, 3), dtype=np.float64)
    b = np.empty(3, dtype=np.float64)
    b[0] = 0.0
    b[1] = 0.0
    
    M[0, :] = C[0, :]
    M[1, :] = C[1, :]
    M[2, 0] = 1.0
    M[2, 1] = 1.0
    M[2, 2] = 1.0
    b[2] = n_X
    
    for col in range(2):
        pivot = col
        for row in range(col + 1, 3):
            if abs(M[row, col]) > abs(M[pivot, col]):
                pivot = row
        if pivot != col:
            for k in range(3):
                M[col, k], M[pivot, k] = M[pivot, k], M[col, k]
            b[col], b[pivot] = b[pivot], b[col]
        
        if abs(M[col, col]) < 1e-30:
            continue
            
        for row in range(col + 1, 3):
            factor = M[row, col] / M[col, col]
            for k in range(col, 3):
                M[row, k] -= factor * M[col, k]
            b[row] -= factor * b[col]
    
    n = np.empty(3, dtype=np.float64)
    n[0] = 0.0
    n[1] = 0.0
    n[2] = 0.0
    for i in range(2, -1, -1):
        if abs(M[i, i]) < 1e-30:
            n[i] = 0.0
        else:
            s = b[i]
            for j in range(i + 1, 3):
                s -= M[i, j] * n[j]
            n[i] = s / M[i, i]
    
    n[0] = max(n[0], 0.0)
    n[1] = max(n[1], 0.0)
    n[2] = max(n[2], 0.0)
    
    cooling = 0.0
    for t in range(n_trans):
        u = trans_u[t]
        l = trans_l[t]
        if u == 1 and l == 0:
            beta_t = beta_01
        elif u == 2 and l == 0:
            beta_t = beta_02
        else:
            beta_t = beta_12
        cooling += n[u] * A_ul[t] * h_nu[t] * beta_t
    
    return cooling


@njit(fastmath=True, cache=True)
def solve_3level_se_blended(
    n_X: float,
    Tg: float,
    n_H2: float,
    n_HI: float,
    ne: float,
    f_ortho: float,
    beta_01: float,
    beta_02: float,
    beta_12: float,
    E_levels: np.ndarray,
    g_levels: np.ndarray,
    trans_u: np.ndarray,
    trans_l: np.ndarray,
    A_ul: np.ndarray,
    h_nu: np.ndarray,
    T_pH2: np.ndarray,
    q_pH2: np.ndarray,
    T_oH2: np.ndarray,
    q_oH2: np.ndarray,
    T_HI: np.ndarray,
    q_HI: np.ndarray,
    T_e: np.ndarray,
    q_e: np.ndarray,
) -> float:
    """Solve 3-level SE with blended ortho/para H2 collision rates.
    
    Mixes pH2 and oH2 rates at runtime using f_ortho. No pre-allocation needed.
    
    Parameters
    ----------
    n_X : float
        Total species number density [cm^-3]
    Tg : float
        Gas temperature [K]
    n_H2, n_HI, ne : float
        Collider densities [cm^-3]
    f_ortho : float
        Ortho-H2 fraction from blended OPR model
    beta_01, beta_02, beta_12 : float
        Escape probabilities for each transition
    E_levels, g_levels : ndarray (3,)
        Level energies [K] and statistical weights
    trans_u, trans_l : ndarray
        Upper and lower level indices
    A_ul, h_nu : ndarray
        Einstein A coefficients and photon energies
    T_pH2, q_pH2 : ndarray
        Para-H2 collision rate table
    T_oH2, q_oH2 : ndarray
        Ortho-H2 collision rate table
    T_HI, q_HI : ndarray
        Atomic H collision rate table
    T_e, q_e : ndarray
        Electron collision rate table
        
    Returns
    -------
    float
        Total cooling rate [erg cm^-3 s^-1]
    """
    if n_X <= 0.0 or Tg <= 0.0:
        return 0.0
    
    # Use np.empty + manual zeroing to minimize allocation overhead
    C = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            C[i, j] = 0.0
    
    n_trans = len(trans_u)
    for t in range(n_trans):
        u = trans_u[t]
        l = trans_l[t]
        
        E_ul = E_levels[u] - E_levels[l]
        g_ratio = g_levels[u] / g_levels[l]
        boltz = np.exp(-E_ul / max(Tg, 1.0))
        
        # Mix pH2 and oH2 rates at runtime
        q_ul_H2 = interp_rate_mix(Tg, T_pH2, q_pH2[t], q_oH2[t], f_ortho) if len(T_pH2) > 0 else 0.0
        q_ul_HI = interp_rate(Tg, T_HI, q_HI[t]) if len(T_HI) > 0 else 0.0
        q_ul_e = interp_rate(Tg, T_e, q_e[t]) if len(T_e) > 0 else 0.0
        
        q_lu_H2 = q_ul_H2 * g_ratio * boltz
        q_lu_HI = q_ul_HI * g_ratio * boltz
        q_lu_e = q_ul_e * g_ratio * boltz
        
        C_ul = n_H2 * q_ul_H2 + n_HI * q_ul_HI + ne * q_ul_e
        C_lu = n_H2 * q_lu_H2 + n_HI * q_lu_HI + ne * q_lu_e
        
        if u == 1 and l == 0:
            beta_t = beta_01
        elif u == 2 and l == 0:
            beta_t = beta_02
        else:
            beta_t = beta_12
        
        R_ul = A_ul[t] * beta_t + C_ul
        R_lu = C_lu
        
        C[l, u] += R_ul
        C[u, l] += R_lu
        C[u, u] -= R_ul
        C[l, l] -= R_lu
    
    # Use np.empty + explicit initialization
    M = np.empty((3, 3), dtype=np.float64)
    b = np.empty(3, dtype=np.float64)
    b[0] = 0.0
    b[1] = 0.0
    
    M[0, :] = C[0, :]
    M[1, :] = C[1, :]
    M[2, 0] = 1.0
    M[2, 1] = 1.0
    M[2, 2] = 1.0
    b[2] = n_X
    
    # Gaussian elimination with partial pivoting
    for col in range(2):
        pivot = col
        for row in range(col + 1, 3):
            if abs(M[row, col]) > abs(M[pivot, col]):
                pivot = row
        if pivot != col:
            for k in range(3):
                M[col, k], M[pivot, k] = M[pivot, k], M[col, k]
            b[col], b[pivot] = b[pivot], b[col]
        
        if abs(M[col, col]) < 1e-30:
            continue
            
        for row in range(col + 1, 3):
            factor = M[row, col] / M[col, col]
            for k in range(col, 3):
                M[row, k] -= factor * M[col, k]
            b[row] -= factor * b[col]
    
    n = np.empty(3, dtype=np.float64)
    n[0] = 0.0
    n[1] = 0.0
    n[2] = 0.0
    for i in range(2, -1, -1):
        if abs(M[i, i]) < 1e-30:
            n[i] = 0.0
        else:
            s = b[i]
            for j in range(i + 1, 3):
                s -= M[i, j] * n[j]
            n[i] = s / M[i, i]
    
    n[0] = max(n[0], 0.0)
    n[1] = max(n[1], 0.0)
    n[2] = max(n[2], 0.0)
    
    cooling = 0.0
    for t in range(n_trans):
        u = trans_u[t]
        l = trans_l[t]
        if u == 1 and l == 0:
            beta_t = beta_01
        elif u == 2 and l == 0:
            beta_t = beta_02
        else:
            beta_t = beta_12
        cooling += n[u] * A_ul[t] * h_nu[t] * beta_t
    
    return cooling


# -----------------------------------------------------------------------------
# N-level statistical equilibrium solver (for CO)
# -----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def solve_nlevel_se(
    n_X: float,
    Tg: float,
    n_H2: float,
    beta_arr: np.ndarray,
    n_levels: int,
    E_levels: np.ndarray,
    g_levels: np.ndarray,
    n_trans: int,
    trans_u: np.ndarray,
    trans_l: np.ndarray,
    A_ul: np.ndarray,
    h_nu: np.ndarray,
    T_coll: np.ndarray,
    q_coll: np.ndarray,
) -> float:
    """Solve N-level statistical equilibrium and return total cooling rate.
    
    For CO, uses H2 as the primary collider.
    
    Parameters
    ----------
    n_X : float
        Total species number density [cm^-3]
    Tg : float
        Gas temperature [K]
    n_H2 : float
        H2 number density [cm^-3]
    beta_arr : ndarray (n_trans,)
        Escape probabilities for each transition
    n_levels : int
        Number of levels to include
    E_levels : ndarray (n_levels,)
        Level energies [K]
    g_levels : ndarray (n_levels,)
        Statistical weights
    n_trans : int
        Number of radiative transitions
    trans_u, trans_l : ndarray (n_trans,)
        Upper and lower level indices
    A_ul : ndarray (n_trans,)
        Einstein A coefficients [s^-1]
    h_nu : ndarray (n_trans,)
        Photon energies [erg]
    T_coll : ndarray
        Temperature grid for collision rates
    q_coll : ndarray (n_coll_trans, n_T)
        De-excitation collision rates [cm^3/s]
        
    Returns
    -------
    float
        Total cooling rate [erg cm^-3 s^-1]
    """
    if n_X <= 0.0 or Tg <= 0.0 or n_H2 <= 0.0:
        return 0.0
    
    MAX_LEVELS = 21
    if n_levels > MAX_LEVELS:
        n_levels = MAX_LEVELS
    
    # Use np.empty + manual zeroing to minimize allocation overhead
    M = np.empty((MAX_LEVELS, MAX_LEVELS), dtype=np.float64)
    b = np.empty(MAX_LEVELS, dtype=np.float64)
    n = np.empty(MAX_LEVELS, dtype=np.float64)
    for i in range(n_levels):
        b[i] = 0.0
        n[i] = 0.0
        for j in range(n_levels):
            M[i, j] = 0.0
    
    n_coll_trans = q_coll.shape[0]
    
    for t in range(min(n_trans, n_coll_trans)):
        u = trans_u[t]
        l = trans_l[t]
        if u >= n_levels or l >= n_levels:
            continue
        
        E_ul = E_levels[u] - E_levels[l]
        g_ratio = g_levels[u] / g_levels[l]
        boltz = np.exp(-E_ul / max(Tg, 1.0))
        
        q_ul = interp_rate(Tg, T_coll, q_coll[t])
        q_lu = q_ul * g_ratio * boltz
        
        C_ul = n_H2 * q_ul
        C_lu = n_H2 * q_lu
        
        beta_t = beta_arr[t] if t < len(beta_arr) else 1.0
        R_ul = A_ul[t] * beta_t + C_ul
        R_lu = C_lu
        
        M[l, u] += R_ul
        M[u, l] += R_lu
        M[u, u] -= R_ul
        M[l, l] -= R_lu
    
    M[n_levels - 1, :n_levels] = 1.0
    b[n_levels - 1] = n_X
    
    for col in range(n_levels - 1):
        pivot = col
        for row in range(col + 1, n_levels):
            if abs(M[row, col]) > abs(M[pivot, col]):
                pivot = row
        if pivot != col:
            for k in range(n_levels):
                M[col, k], M[pivot, k] = M[pivot, k], M[col, k]
            b[col], b[pivot] = b[pivot], b[col]
        
        if abs(M[col, col]) < 1e-30:
            continue
            
        for row in range(col + 1, n_levels):
            factor = M[row, col] / M[col, col]
            for k in range(col, n_levels):
                M[row, k] -= factor * M[col, k]
            b[row] -= factor * b[col]
    
    for i in range(n_levels - 1, -1, -1):
        if abs(M[i, i]) < 1e-30:
            n[i] = 0.0
        else:
            s = b[i]
            for j in range(i + 1, n_levels):
                s -= M[i, j] * n[j]
            n[i] = s / M[i, i]
        n[i] = max(n[i], 0.0)
    
    cooling = 0.0
    for t in range(min(n_trans, n_levels - 1)):
        u = trans_u[t]
        l = trans_l[t]
        if u >= n_levels or l >= n_levels:
            continue
        beta_t = beta_arr[t] if t < len(beta_arr) else 1.0
        cooling += n[u] * A_ul[t] * h_nu[t] * beta_t
    
    return cooling


@njit(fastmath=True, cache=True)
def solve_nlevel_se_blended(
    n_X: float,
    Tg: float,
    n_H2: float,
    n_HI: float,
    f_ortho: float,
    beta_arr: np.ndarray,
    n_levels: int,
    E_levels: np.ndarray,
    g_levels: np.ndarray,
    n_trans: int,
    trans_u: np.ndarray,
    trans_l: np.ndarray,
    A_ul: np.ndarray,
    h_nu: np.ndarray,
    T_pH2: np.ndarray,
    q_pH2: np.ndarray,
    T_oH2: np.ndarray,
    q_oH2: np.ndarray,
) -> float:
    """Solve N-level SE with blended ortho/para H2 collision rates.
    
    For CO, mixes pH2 and oH2 rates at runtime using f_ortho.
    
    Parameters
    ----------
    n_X : float
        Total species number density [cm^-3]
    Tg : float
        Gas temperature [K]
    n_H2 : float
        H2 number density [cm^-3]
    n_HI : float
        Atomic H number density [cm^-3] (used to compute f_ortho)
    f_ortho : float
        Ortho-H2 fraction from blended OPR model
    beta_arr : ndarray (n_trans,)
        Escape probabilities for each transition
    n_levels : int
        Number of levels to include
    E_levels, g_levels : ndarray
        Level energies [K] and statistical weights
    n_trans : int
        Number of radiative transitions
    trans_u, trans_l : ndarray
        Upper and lower level indices
    A_ul, h_nu : ndarray
        Einstein A coefficients and photon energies
    T_pH2, q_pH2 : ndarray
        Para-H2 collision rate table
    T_oH2, q_oH2 : ndarray
        Ortho-H2 collision rate table
        
    Returns
    -------
    float
        Total cooling rate [erg cm^-3 s^-1]
    """
    if n_X <= 0.0 or Tg <= 0.0 or n_H2 <= 0.0:
        return 0.0
    
    MAX_LEVELS = 21
    if n_levels > MAX_LEVELS:
        n_levels = MAX_LEVELS
    
    # Use np.empty + manual zeroing to minimize allocation overhead
    M = np.empty((MAX_LEVELS, MAX_LEVELS), dtype=np.float64)
    b = np.empty(MAX_LEVELS, dtype=np.float64)
    n = np.empty(MAX_LEVELS, dtype=np.float64)
    for i in range(n_levels):
        b[i] = 0.0
        n[i] = 0.0
        for j in range(n_levels):
            M[i, j] = 0.0
    
    n_coll_trans = q_pH2.shape[0]
    
    for t in range(min(n_trans, n_coll_trans)):
        u = trans_u[t]
        l = trans_l[t]
        if u >= n_levels or l >= n_levels:
            continue
        
        E_ul = E_levels[u] - E_levels[l]
        g_ratio = g_levels[u] / g_levels[l]
        boltz = np.exp(-E_ul / max(Tg, 1.0))
        
        # Mix pH2 and oH2 rates at runtime using different T-grids
        q_ul = interp_rate_mix_2grid(Tg, T_pH2, q_pH2[t], T_oH2, q_oH2[t], f_ortho)
        q_lu = q_ul * g_ratio * boltz
        
        C_ul = n_H2 * q_ul
        C_lu = n_H2 * q_lu
        
        beta_t = beta_arr[t] if t < len(beta_arr) else 1.0
        R_ul = A_ul[t] * beta_t + C_ul
        R_lu = C_lu
        
        M[l, u] += R_ul
        M[u, l] += R_lu
        M[u, u] -= R_ul
        M[l, l] -= R_lu
    
    M[n_levels - 1, :n_levels] = 1.0
    b[n_levels - 1] = n_X
    
    # Gaussian elimination with partial pivoting
    for col in range(n_levels - 1):
        pivot = col
        for row in range(col + 1, n_levels):
            if abs(M[row, col]) > abs(M[pivot, col]):
                pivot = row
        if pivot != col:
            for k in range(n_levels):
                M[col, k], M[pivot, k] = M[pivot, k], M[col, k]
            b[col], b[pivot] = b[pivot], b[col]
        
        if abs(M[col, col]) < 1e-30:
            continue
            
        for row in range(col + 1, n_levels):
            factor = M[row, col] / M[col, col]
            for k in range(col, n_levels):
                M[row, k] -= factor * M[col, k]
            b[row] -= factor * b[col]
    
    for i in range(n_levels - 1, -1, -1):
        if abs(M[i, i]) < 1e-30:
            n[i] = 0.0
        else:
            s = b[i]
            for j in range(i + 1, n_levels):
                s -= M[i, j] * n[j]
            n[i] = s / M[i, i]
        n[i] = max(n[i], 0.0)
    
    cooling = 0.0
    for t in range(min(n_trans, n_levels - 1)):
        u = trans_u[t]
        l = trans_l[t]
        if u >= n_levels or l >= n_levels:
            continue
        beta_t = beta_arr[t] if t < len(beta_arr) else 1.0
        cooling += n[u] * A_ul[t] * h_nu[t] * beta_t
    
    return cooling


# -----------------------------------------------------------------------------
# Individual heating/cooling term functions (scalar, inlined)
# -----------------------------------------------------------------------------
#
# NOTE: The placeholder cooling functions (cii_cooling, ci_cooling, oi_cooling,
# co_rot_cooling) below are DEPRECATED. The solver now uses SE-based cooling
# functions (cii_cooling_lamda, ci_cooling_lamda, oi_cooling_lamda, co_cooling_lamda)
# which use LAMDA collision rates with explicit collider densities.
#
# The old functions are retained temporarily for backwards compatibility with
# diagnostic tools, but should not be used for new development.
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


@njit(inline='always', fastmath=True, cache=True)
def ci_cooling(
    nH: float,
    ne: float,
    nC: float,
    Tg: float,
    beta_ci10: float,
    gamma_ci10: float,
    E_ci10: float,
    n_crit_ci10: float,
    beta_ci21: float,
    gamma_ci21: float,
    E_ci21: float,
    n_crit_ci21: float,
    k_B: float,
) -> float:
    """C I 609 um and 370 um fine-structure cooling.
    
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1] (positive)
    """
    n_coll = ne + 0.1 * nH
    n_coll = max(n_coll, 1e-10)
    
    Tg_safe = max(Tg, 10.0)
    
    excitation_10 = np.exp(-E_ci10 / Tg_safe)
    rate_10 = (
        nC * n_coll * gamma_ci10 * E_ci10 * k_B
        * excitation_10 / (1.0 + n_crit_ci10 / n_coll)
        * beta_ci10
    )
    
    excitation_21 = np.exp(-E_ci21 / Tg_safe)
    rate_21 = (
        nC * n_coll * gamma_ci21 * E_ci21 * k_B
        * excitation_21 / (1.0 + n_crit_ci21 / n_coll)
        * beta_ci21
    )
    
    return rate_10 + rate_21


@njit(inline='always', fastmath=True, cache=True)
def oi_cooling(
    nH: float,
    ne: float,
    nO: float,
    Tg: float,
    beta_oi63: float,
    gamma_oi63: float,
    E_oi63: float,
    n_crit_oi63: float,
    beta_oi145: float,
    gamma_oi145: float,
    E_oi145: float,
    n_crit_oi145: float,
    k_B: float,
) -> float:
    """O I 63 um and 145 um fine-structure cooling.
    
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1] (positive)
    """
    n_coll = ne + 0.1 * nH
    n_coll = max(n_coll, 1e-10)
    
    Tg_safe = max(Tg, 10.0)
    
    excitation_63 = np.exp(-E_oi63 / Tg_safe)
    rate_63 = (
        nO * n_coll * gamma_oi63 * E_oi63 * k_B
        * excitation_63 / (1.0 + n_crit_oi63 / n_coll)
        * beta_oi63
    )
    
    excitation_145 = np.exp(-E_oi145 / Tg_safe)
    rate_145 = (
        nO * n_coll * gamma_oi145 * E_oi145 * k_B
        * excitation_145 / (1.0 + n_crit_oi145 / n_coll)
        * beta_oi145
    )
    
    return rate_63 + rate_145


@njit(inline='always', fastmath=True, cache=True)
def co_rot_cooling(
    nH: float,
    ne: float,
    nco_gas: float,
    Tg: float,
    beta_co: float,
    L_co_coeff: float,
) -> float:
    """CO rotational cooling.
    
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1] (positive)
    """
    n_coll = ne + 0.5 * nH
    n_coll = max(n_coll, 1e-10)
    
    Tg_safe = max(Tg, 10.0)
    
    L_co = L_co_coeff * np.sqrt(Tg_safe / 100.0)
    
    rate = nco_gas * n_coll * L_co * beta_co
    
    return rate


# -----------------------------------------------------------------------------
# SE-based cooling functions using LAMDA data
# -----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def cii_cooling_se(
    nCplus: float,
    Tg: float,
    n_H2: float,
    n_HI: float,
    ne: float,
    beta: float,
    E_ul: float,
    g_u: float,
    g_l: float,
    A_ul: float,
    h_nu: float,
    T_H2: np.ndarray,
    q_H2: np.ndarray,
    T_HI: np.ndarray,
    q_HI: np.ndarray,
    T_e: np.ndarray,
    q_e: np.ndarray,
) -> float:
    """C+ 158um cooling using 2-level SE with LAMDA data.
    
    Parameters
    ----------
    nCplus : float
        C+ number density [cm^-3]
    Tg : float
        Gas temperature [K]
    n_H2, n_HI, ne : float
        Collider densities [cm^-3]
    beta : float
        Escape probability
    E_ul, g_u, g_l, A_ul, h_nu : float
        Atomic data from LAMDA
    T_H2, q_H2, T_HI, q_HI, T_e, q_e : ndarray
        Collision rate data from LAMDA
        
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1]
    """
    return solve_2level_se(
        nCplus, Tg, n_H2, n_HI, ne, beta,
        E_ul, g_u, g_l, A_ul, h_nu,
        T_H2, q_H2, T_HI, q_HI, T_e, q_e
    )


@njit(fastmath=True, cache=True)
def ci_cooling_se(
    nC: float,
    Tg: float,
    n_H2: float,
    n_HI: float,
    ne: float,
    beta_01: float,
    beta_02: float,
    beta_12: float,
    E_levels: np.ndarray,
    g_levels: np.ndarray,
    trans_u: np.ndarray,
    trans_l: np.ndarray,
    A_ul: np.ndarray,
    h_nu: np.ndarray,
    T_H2: np.ndarray,
    q_H2: np.ndarray,
    T_HI: np.ndarray,
    q_HI: np.ndarray,
    T_e: np.ndarray,
    q_e: np.ndarray,
) -> float:
    """C fine-structure cooling using 3-level SE with LAMDA data.
    
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1]
    """
    return solve_3level_se(
        nC, Tg, n_H2, n_HI, ne,
        beta_01, beta_02, beta_12,
        E_levels, g_levels, trans_u, trans_l, A_ul, h_nu,
        T_H2, q_H2, T_HI, q_HI, T_e, q_e
    )


@njit(fastmath=True, cache=True)
def oi_cooling_se(
    nO: float,
    Tg: float,
    n_H2: float,
    n_HI: float,
    ne: float,
    beta_01: float,
    beta_02: float,
    beta_12: float,
    E_levels: np.ndarray,
    g_levels: np.ndarray,
    trans_u: np.ndarray,
    trans_l: np.ndarray,
    A_ul: np.ndarray,
    h_nu: np.ndarray,
    T_H2: np.ndarray,
    q_H2: np.ndarray,
    T_HI: np.ndarray,
    q_HI: np.ndarray,
    T_e: np.ndarray,
    q_e: np.ndarray,
) -> float:
    """O fine-structure cooling using 3-level SE with LAMDA data.
    
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1]
    """
    return solve_3level_se(
        nO, Tg, n_H2, n_HI, ne,
        beta_01, beta_02, beta_12,
        E_levels, g_levels, trans_u, trans_l, A_ul, h_nu,
        T_H2, q_H2, T_HI, q_HI, T_e, q_e
    )


@njit(fastmath=True, cache=True)
def co_cooling_se(
    nco: float,
    Tg: float,
    n_H2: float,
    beta_arr: np.ndarray,
    n_levels: int,
    E_levels: np.ndarray,
    g_levels: np.ndarray,
    n_trans: int,
    trans_u: np.ndarray,
    trans_l: np.ndarray,
    A_ul: np.ndarray,
    h_nu: np.ndarray,
    T_coll: np.ndarray,
    q_coll: np.ndarray,
) -> float:
    """CO rotational cooling using N-level SE with LAMDA data.
    
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1]
    """
    return solve_nlevel_se(
        nco, Tg, n_H2, beta_arr,
        n_levels, E_levels, g_levels,
        n_trans, trans_u, trans_l, A_ul, h_nu,
        T_coll, q_coll
    )


# -----------------------------------------------------------------------------
# Net heating function for a single cell
# -----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def net_heating_cell(
    Tg: float,
    nH: float,
    Td: float,
    chi: float,
    ne: float,
    nCplus: float,
    nC: float,
    nO: float,
    nco_gas: float,
    mask: int,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    beta_ci10: float,
    beta_ci21: float,
    beta_oi63: float,
    beta_oi145: float,
    beta_co: float,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    gamma_cii: float,
    E_cii: float,
    n_crit_cii: float,
    gamma_ci10: float,
    E_ci10: float,
    n_crit_ci10: float,
    gamma_ci21: float,
    E_ci21: float,
    n_crit_ci21: float,
    gamma_oi63: float,
    E_oi63: float,
    n_crit_oi63: float,
    gamma_oi145: float,
    E_oi145: float,
    n_crit_oi145: float,
    L_co_coeff: float,
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
    
    if mask & TERM_CI:
        net -= ci_cooling(nH, ne, nC, Tg, beta_ci10, gamma_ci10, E_ci10, n_crit_ci10,
                         beta_ci21, gamma_ci21, E_ci21, n_crit_ci21, k_B)
    
    if mask & TERM_OI:
        net -= oi_cooling(nH, ne, nO, Tg, beta_oi63, gamma_oi63, E_oi63, n_crit_oi63,
                         beta_oi145, gamma_oi145, E_oi145, n_crit_oi145, k_B)
    
    if mask & TERM_CO:
        net -= co_rot_cooling(nH, ne, nco_gas, Tg, beta_co, L_co_coeff)
    
    if mask & TERM_GD:
        net += gas_dust_exchange(nH, Td, Tg, alpha_acc, sigma_dust, f_dust, k_B, m_H)
    
    return net


# -----------------------------------------------------------------------------
# Bisection solver for a single cell
# -----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def bisect_solve_cell(
    nH: float,
    Td: float,
    chi: float,
    ne: float,
    nCplus: float,
    nC: float,
    nO: float,
    nco_gas: float,
    mask: int,
    T_min: float,
    T_max: float,
    max_iter: int,
    tol: float,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    beta_ci10: float,
    beta_ci21: float,
    beta_oi63: float,
    beta_oi145: float,
    beta_co: float,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    gamma_cii: float,
    E_cii: float,
    n_crit_cii: float,
    gamma_ci10: float,
    E_ci10: float,
    n_crit_ci10: float,
    gamma_ci21: float,
    E_ci21: float,
    n_crit_ci21: float,
    gamma_oi63: float,
    E_oi63: float,
    n_crit_oi63: float,
    gamma_oi145: float,
    E_oi145: float,
    n_crit_oi145: float,
    L_co_coeff: float,
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
    
    fa = net_heating_cell(a, nH, Td, chi, ne, nCplus, nC, nO, nco_gas, mask,
                          zeta_cr, pah_scale, alpha_acc, beta_cii,
                          beta_ci10, beta_ci21, beta_oi63, beta_oi145, beta_co,
                          heating_per_cr, pe_heating_rate_0,
                          gamma_cii, E_cii, n_crit_cii,
                          gamma_ci10, E_ci10, n_crit_ci10,
                          gamma_ci21, E_ci21, n_crit_ci21,
                          gamma_oi63, E_oi63, n_crit_oi63,
                          gamma_oi145, E_oi145, n_crit_oi145,
                          L_co_coeff, sigma_dust, f_dust, k_B, m_H)
    fb = net_heating_cell(b, nH, Td, chi, ne, nCplus, nC, nO, nco_gas, mask,
                          zeta_cr, pah_scale, alpha_acc, beta_cii,
                          beta_ci10, beta_ci21, beta_oi63, beta_oi145, beta_co,
                          heating_per_cr, pe_heating_rate_0,
                          gamma_cii, E_cii, n_crit_cii,
                          gamma_ci10, E_ci10, n_crit_ci10,
                          gamma_ci21, E_ci21, n_crit_ci21,
                          gamma_oi63, E_oi63, n_crit_oi63,
                          gamma_oi145, E_oi145, n_crit_oi145,
                          L_co_coeff, sigma_dust, f_dust, k_B, m_H)
    
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
        fm = net_heating_cell(m, nH, Td, chi, ne, nCplus, nC, nO, nco_gas, mask,
                              zeta_cr, pah_scale, alpha_acc, beta_cii,
                              beta_ci10, beta_ci21, beta_oi63, beta_oi145, beta_co,
                              heating_per_cr, pe_heating_rate_0,
                              gamma_cii, E_cii, n_crit_cii,
                              gamma_ci10, E_ci10, n_crit_ci10,
                              gamma_ci21, E_ci21, n_crit_ci21,
                              gamma_oi63, E_oi63, n_crit_oi63,
                              gamma_oi145, E_oi145, n_crit_oi145,
                              L_co_coeff, sigma_dust, f_dust, k_B, m_H)
        
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
    nC: np.ndarray,
    nO: np.ndarray,
    nco_gas: np.ndarray,
    mask: int,
    T_min: float,
    T_max: float,
    max_iter: int,
    tol: float,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    beta_ci10: float,
    beta_ci21: float,
    beta_oi63: float,
    beta_oi145: float,
    beta_co: float,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    gamma_cii: float,
    E_cii: float,
    n_crit_cii: float,
    gamma_ci10: float,
    E_ci10: float,
    n_crit_ci10: float,
    gamma_ci21: float,
    E_ci21: float,
    n_crit_ci21: float,
    gamma_oi63: float,
    E_oi63: float,
    n_crit_oi63: float,
    gamma_oi145: float,
    E_oi145: float,
    n_crit_oi145: float,
    L_co_coeff: float,
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
            nH[i], Td[i], chi[i], ne[i], nCplus[i], nC[i], nO[i], nco_gas[i],
            mask, T_min, T_max, max_iter, tol,
            zeta_cr, pah_scale, alpha_acc, beta_cii,
            beta_ci10, beta_ci21, beta_oi63, beta_oi145, beta_co,
            heating_per_cr, pe_heating_rate_0,
            gamma_cii, E_cii, n_crit_cii,
            gamma_ci10, E_ci10, n_crit_ci10,
            gamma_ci21, E_ci21, n_crit_ci21,
            gamma_oi63, E_oi63, n_crit_oi63,
            gamma_oi145, E_oi145, n_crit_oi145,
            L_co_coeff, sigma_dust, f_dust, k_B, m_H
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
    nC: np.ndarray,
    nO: np.ndarray,
    nco_gas: np.ndarray,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    beta_ci10: float,
    beta_ci21: float,
    beta_oi63: float,
    beta_oi145: float,
    beta_co: float,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    gamma_cii: float,
    E_cii: float,
    n_crit_cii: float,
    gamma_ci10: float,
    E_ci10: float,
    n_crit_ci10: float,
    gamma_ci21: float,
    E_ci21: float,
    n_crit_ci21: float,
    gamma_oi63: float,
    E_oi63: float,
    n_crit_oi63: float,
    gamma_oi145: float,
    E_oi145: float,
    n_crit_oi145: float,
    L_co_coeff: float,
    sigma_dust: float,
    f_dust: float,
    k_B: float,
    m_H: float,
    rate_cr_out: np.ndarray,
    rate_pe_out: np.ndarray,
    rate_cii_out: np.ndarray,
    rate_ci_out: np.ndarray,
    rate_oi_out: np.ndarray,
    rate_co_out: np.ndarray,
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
        rate_ci_out[i] = -ci_cooling(nH[i], ne[i], nC[i], Tg[i], beta_ci10, gamma_ci10, E_ci10, n_crit_ci10,
                                      beta_ci21, gamma_ci21, E_ci21, n_crit_ci21, k_B)
        rate_oi_out[i] = -oi_cooling(nH[i], ne[i], nO[i], Tg[i], beta_oi63, gamma_oi63, E_oi63, n_crit_oi63,
                                      beta_oi145, gamma_oi145, E_oi145, n_crit_oi145, k_B)
        rate_co_out[i] = -co_rot_cooling(nH[i], ne[i], nco_gas[i], Tg[i], beta_co, L_co_coeff)
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


# =============================================================================
# SE-BASED COOLING KERNELS (LAMDA DATA)
# =============================================================================
#
# These kernels use statistical equilibrium with LAMDA collision rates.
# They require explicit collider densities (nH2, nHI, ne) - no hidden hacks.
#
# References:
#   - Schoier et al. 2005, A&A 432, 369 (LAMDA database)
#   - van der Tak et al. 2020, Atoms 8, 15 (LAMDA update)
#   - Goldsmith et al. 2012, ApJS 203, 13 (C+ excitation)
#   - Kamp et al. 2010, A&A 510, A18 (disk thermal balance)
# =============================================================================


@njit(fastmath=True, cache=True)
def cii_cooling_lamda(
    nCplus: float,
    Tg: float,
    nH2: float,
    nHI: float,
    ne: float,
    beta: float,
) -> float:
    """[C II] 158 um cooling using 2-level SE with LAMDA rates.
    
    Uses collision rates from Wiesenfeld & Goldsmith (2014) for H2,
    Barinovs et al. (2005) for H, and Wilson & Bell (2002) for electrons.
    H2 rates are mixed using blended OPR model based on T and fH2.
    
    Parameters
    ----------
    nCplus : float
        C+ number density [cm^-3]
    Tg : float
        Gas temperature [K]
    nH2 : float
        H2 number density [cm^-3]
    nHI : float
        Atomic H number density [cm^-3]
    ne : float
        Electron number density [cm^-3]
    beta : float
        Escape probability (1.0 = optically thin)
        
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1]
    """
    if nCplus <= 0.0 or Tg <= 0.0:
        return 0.0
    
    # Compute blended ortho fraction based on T and molecular fraction
    f_o = f_ortho_h2(Tg, nH2, nHI)
    
    # Mix pH2 and oH2 collision rates
    q_ul_H2 = interp_rate_mix(Tg, CPLUS_COLL_PH2_T, CPLUS_COLL_PH2_Q[0], CPLUS_COLL_OH2_Q[0], f_o)
    q_ul_HI = interp_rate(Tg, CPLUS_COLL_H_T, CPLUS_COLL_H_Q[0])
    q_ul_e = interp_rate(Tg, CPLUS_COLL_E_T, CPLUS_COLL_E_Q[0])
    
    E_ul = CPLUS_E_UL_K[0]
    g_u = CPLUS_G_LEVELS[1]
    g_l = CPLUS_G_LEVELS[0]
    A_ul = CPLUS_A_UL[0]
    h_nu = CPLUS_HNU[0]
    
    boltz = np.exp(-E_ul / max(Tg, 1.0))
    g_ratio = g_u / g_l
    
    q_lu_H2 = q_ul_H2 * g_ratio * boltz
    q_lu_HI = q_ul_HI * g_ratio * boltz
    q_lu_e = q_ul_e * g_ratio * boltz
    
    C_ul = nH2 * q_ul_H2 + nHI * q_ul_HI + ne * q_ul_e
    C_lu = nH2 * q_lu_H2 + nHI * q_lu_HI + ne * q_lu_e
    
    denom = C_lu + C_ul + A_ul * beta
    if denom <= 0.0:
        return 0.0
    
    n_u = nCplus * C_lu / denom
    return n_u * A_ul * h_nu * beta


@njit(fastmath=True, cache=True)
def ci_cooling_lamda(
    nC: float,
    Tg: float,
    nH2: float,
    nHI: float,
    ne: float,
    beta_10: float,
    beta_20: float,
    beta_21: float,
) -> float:
    """[C I] 609 um and 370 um cooling using 3-level SE with LAMDA rates.
    
    Uses collision rates from Launay & Roueff (1977) and Schroeder et al. (1991).
    H2 rates are mixed at runtime using blended OPR model.
    
    Parameters
    ----------
    nC : float
        Neutral carbon number density [cm^-3]
    Tg : float
        Gas temperature [K]
    nH2 : float
        H2 number density [cm^-3]
    nHI : float
        Atomic H number density [cm^-3]
    ne : float
        Electron number density [cm^-3]
    beta_10, beta_20, beta_21 : float
        Escape probabilities for each transition (1.0 = optically thin)
        
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1]
    """
    if nC <= 0.0 or Tg <= 0.0:
        return 0.0
    
    # Compute blended ortho fraction
    f_o = f_ortho_h2(Tg, nH2, nHI)
    
    return solve_3level_se_blended(
        nC, Tg, nH2, nHI, ne, f_o,
        beta_10, beta_20, beta_21,
        C_E_LEVELS_K, C_G_LEVELS, C_TRANS_U, C_TRANS_L, C_A_UL, C_HNU,
        C_COLL_PH2_T, C_COLL_PH2_Q, C_COLL_OH2_T, C_COLL_OH2_Q,
        C_COLL_H_T, C_COLL_H_Q, C_COLL_E_T, C_COLL_E_Q
    )


@njit(fastmath=True, cache=True)
def oi_cooling_lamda(
    nO: float,
    Tg: float,
    nH2: float,
    nHI: float,
    ne: float,
    beta_10: float,
    beta_20: float,
    beta_21: float,
) -> float:
    """[O I] 63 um and 145 um cooling using 3-level SE with LAMDA rates.
    
    Uses collision rates from Abrahamsson et al. (2007) and Jaquet et al. (1992).
    H2 rates are mixed at runtime using blended OPR model.
    
    Parameters
    ----------
    nO : float
        Atomic oxygen number density [cm^-3]
    Tg : float
        Gas temperature [K]
    nH2 : float
        H2 number density [cm^-3]
    nHI : float
        Atomic H number density [cm^-3]
    ne : float
        Electron number density [cm^-3]
    beta_10, beta_20, beta_21 : float
        Escape probabilities for each transition (1.0 = optically thin)
        
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1]
    """
    if nO <= 0.0 or Tg <= 0.0:
        return 0.0
    
    # Compute blended ortho fraction
    f_o = f_ortho_h2(Tg, nH2, nHI)
    
    return solve_3level_se_blended(
        nO, Tg, nH2, nHI, ne, f_o,
        beta_10, beta_20, beta_21,
        O_E_LEVELS_K, O_G_LEVELS, O_TRANS_U, O_TRANS_L, O_A_UL, O_HNU,
        O_COLL_PH2_T, O_COLL_PH2_Q, O_COLL_OH2_T, O_COLL_OH2_Q,
        O_COLL_H_T, O_COLL_H_Q, O_COLL_E_T, O_COLL_E_Q
    )


@njit(fastmath=True, cache=True)
def co_cooling_lamda(
    nco: float,
    Tg: float,
    nH2: float,
    nHI: float,
    beta_co: float,
    co_jmax: int,
) -> float:
    """CO rotational cooling using N-level SE with LAMDA rates.
    
    Uses collision rates from Yang et al. (2010).
    H2 rates are mixed at runtime using blended OPR model.
    
    Parameters
    ----------
    nco : float
        CO number density [cm^-3]
    Tg : float
        Gas temperature [K]
    nH2 : float
        H2 number density [cm^-3] (primary collider for CO)
    nHI : float
        Atomic H number density [cm^-3] (for blended OPR model)
    beta_co : float
        Escape probability (uniform for all lines, 1.0 = optically thin)
    co_jmax : int
        Maximum J level to include (must be <= 20, affects accuracy at high T)
        
    Returns
    -------
    float
        Cooling rate [erg cm^-3 s^-1]
        
    Notes
    -----
    CO has many rotational levels. For T < 100 K, J_max ~ 10 is sufficient.
    For T ~ 300 K, J_max ~ 15-20 is better. The solver caps at 21 levels
    to keep the matrix inversion fast.
    """
    if nco <= 0.0 or Tg <= 0.0 or nH2 <= 0.0:
        return 0.0
    
    # Compute blended ortho fraction
    f_o = f_ortho_h2(Tg, nH2, nHI)
    
    n_levels = min(co_jmax + 1, CO_N_LEVELS, 21)
    n_trans = min(n_levels - 1, CO_N_TRANS)
    
    beta_arr = np.full(n_trans, beta_co, dtype=np.float64)
    
    return solve_nlevel_se_blended(
        nco, Tg, nH2, nHI, f_o, beta_arr,
        n_levels, CO_E_LEVELS_K, CO_G_LEVELS,
        n_trans, CO_TRANS_U, CO_TRANS_L, CO_A_UL, CO_HNU,
        CO_COLL_PH2_T, CO_COLL_PH2_Q,
        CO_COLL_OH2_T, CO_COLL_OH2_Q
    )


# -----------------------------------------------------------------------------
# Net heating function using SE-based cooling (LAMDA data)
# -----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def net_heating_cell_se(
    Tg: float,
    nH: float,
    nH2: float,
    nHI: float,
    Td: float,
    chi: float,
    ne: float,
    nCplus: float,
    nC: float,
    nO: float,
    nco_gas: float,
    mask: int,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    beta_ci10: float,
    beta_ci20: float,
    beta_ci21: float,
    beta_oi10: float,
    beta_oi20: float,
    beta_oi21: float,
    beta_co: float,
    co_jmax: int,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    sigma_dust: float,
    f_dust: float,
) -> float:
    """Compute net heating rate using SE-based cooling with LAMDA data.
    
    This kernel uses statistical equilibrium solvers with LAMDA collision
    rates for all atomic/molecular line cooling. Collider densities (nH2,
    nHI, ne) must be provided explicitly - no hidden hacks.
    
    Parameters
    ----------
    Tg : float
        Gas temperature [K]
    nH : float
        Total H nuclei density [cm^-3]
    nH2 : float
        H2 number density [cm^-3]
    nHI : float
        Atomic H number density [cm^-3]
    Td : float
        Dust temperature [K]
    chi : float
        Effective UV field (dimensionless)
    ne : float
        Electron number density [cm^-3]
    nCplus : float
        C+ number density [cm^-3]
    nC : float
        Neutral C number density [cm^-3]
    nO : float
        Atomic O number density [cm^-3]
    nco_gas : float
        CO gas number density [cm^-3]
    mask : int
        Bitmask of terms to include
    zeta_cr : float
        Cosmic ray ionization rate [s^-1]
    pah_scale : float
        PAH abundance scaling
    alpha_acc : float
        Gas-dust accommodation coefficient
    beta_cii : float
        C II 158 um escape probability
    beta_ci10, beta_ci20, beta_ci21 : float
        C I escape probabilities (609um, 370um forbidden, 370um)
    beta_oi10, beta_oi20, beta_oi21 : float
        O I escape probabilities (63um, 145um forbidden, 145um)
    beta_co : float
        CO escape probability (uniform for all lines)
    co_jmax : int
        Maximum CO J level to include
    heating_per_cr : float
        Energy per CR ionization [erg]
    pe_heating_rate_0 : float
        Base photoelectric heating rate
    sigma_dust : float
        Dust cross section [cm^2]
    f_dust : float
        Dust abundance factor
        
    Returns
    -------
    float
        Net heating rate [erg cm^-3 s^-1] (positive = heating)
    """
    net = 0.0
    
    if mask & TERM_CR:
        net += cosmic_ray_heating(nH, zeta_cr, heating_per_cr)
    
    if mask & TERM_PE:
        net += photoelectric_heating(nH, chi, ne, Tg, pah_scale, pe_heating_rate_0)
    
    if mask & TERM_CII:
        net -= cii_cooling_lamda(nCplus, Tg, nH2, nHI, ne, beta_cii)
    
    if mask & TERM_CI:
        net -= ci_cooling_lamda(nC, Tg, nH2, nHI, ne, beta_ci10, beta_ci20, beta_ci21)
    
    if mask & TERM_OI:
        net -= oi_cooling_lamda(nO, Tg, nH2, nHI, ne, beta_oi10, beta_oi20, beta_oi21)
    
    if mask & TERM_CO:
        net -= co_cooling_lamda(nco_gas, Tg, nH2, nHI, beta_co, co_jmax)
    
    if mask & TERM_GD:
        net += gas_dust_exchange(nH, Td, Tg, alpha_acc, sigma_dust, f_dust, K_B, M_H)
    
    return net


# -----------------------------------------------------------------------------
# Bisection solver using SE-based cooling
# -----------------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def bisect_solve_cell_se(
    nH: float,
    nH2: float,
    nHI: float,
    Td: float,
    chi: float,
    ne: float,
    nCplus: float,
    nC: float,
    nO: float,
    nco_gas: float,
    mask: int,
    T_min: float,
    T_max: float,
    max_iter: int,
    tol: float,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    beta_ci10: float,
    beta_ci20: float,
    beta_ci21: float,
    beta_oi10: float,
    beta_oi20: float,
    beta_oi21: float,
    beta_co: float,
    co_jmax: int,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    sigma_dust: float,
    f_dust: float,
) -> float:
    """Solve thermal balance using SE-based cooling with bisection."""
    a = T_min
    b = T_max
    
    fa = net_heating_cell_se(a, nH, nH2, nHI, Td, chi, ne, nCplus, nC, nO, nco_gas, mask,
                              zeta_cr, pah_scale, alpha_acc, beta_cii,
                              beta_ci10, beta_ci20, beta_ci21,
                              beta_oi10, beta_oi20, beta_oi21,
                              beta_co, co_jmax, heating_per_cr, pe_heating_rate_0,
                              sigma_dust, f_dust)
    fb = net_heating_cell_se(b, nH, nH2, nHI, Td, chi, ne, nCplus, nC, nO, nco_gas, mask,
                              zeta_cr, pah_scale, alpha_acc, beta_cii,
                              beta_ci10, beta_ci20, beta_ci21,
                              beta_oi10, beta_oi20, beta_oi21,
                              beta_co, co_jmax, heating_per_cr, pe_heating_rate_0,
                              sigma_dust, f_dust)
    
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
        fm = net_heating_cell_se(m, nH, nH2, nHI, Td, chi, ne, nCplus, nC, nO, nco_gas, mask,
                                  zeta_cr, pah_scale, alpha_acc, beta_cii,
                                  beta_ci10, beta_ci20, beta_ci21,
                                  beta_oi10, beta_oi20, beta_oi21,
                                  beta_co, co_jmax, heating_per_cr, pe_heating_rate_0,
                                  sigma_dust, f_dust)
        
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
# Parallel solver using SE-based cooling
# -----------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def solve_tgas_kernel_se(
    nH: np.ndarray,
    nH2: np.ndarray,
    nHI: np.ndarray,
    Td: np.ndarray,
    chi: np.ndarray,
    ne: np.ndarray,
    nCplus: np.ndarray,
    nC: np.ndarray,
    nO: np.ndarray,
    nco_gas: np.ndarray,
    mask: int,
    T_min: float,
    T_max: float,
    max_iter: int,
    tol: float,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    beta_ci10: float,
    beta_ci20: float,
    beta_ci21: float,
    beta_oi10: float,
    beta_oi20: float,
    beta_oi21: float,
    beta_co: float,
    co_jmax: int,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    sigma_dust: float,
    f_dust: float,
    Tg_out: np.ndarray,
) -> None:
    """Solve thermal balance using SE-based cooling for all cells.
    
    Uses LAMDA collision rates for C+, C, O, CO cooling. Requires explicit
    collider densities (nH2, nHI, ne) - no placeholder physics.
    
    Parameters
    ----------
    nH : ndarray
        Total H nuclei density [cm^-3]
    nH2 : ndarray
        H2 number density [cm^-3]
    nHI : ndarray
        Atomic H number density [cm^-3]
    Td : ndarray
        Dust temperature [K]
    chi : ndarray
        Effective UV field (dimensionless)
    ne : ndarray
        Electron number density [cm^-3]
    nCplus : ndarray
        C+ number density [cm^-3]
    nC : ndarray
        Neutral C number density [cm^-3]
    nO : ndarray
        Atomic O number density [cm^-3]
    nco_gas : ndarray
        CO gas number density [cm^-3]
    mask : int
        Bitmask of terms to include
    T_min, T_max : float
        Temperature bracket [K]
    max_iter : int
        Maximum bisection iterations
    tol : float
        Convergence tolerance [erg cm^-3 s^-1]
    zeta_cr : float
        Cosmic ray ionization rate [s^-1]
    pah_scale : float
        PAH abundance scaling
    alpha_acc : float
        Gas-dust accommodation coefficient
    beta_cii : float
        C II escape probability
    beta_ci10, beta_ci20, beta_ci21 : float
        C I escape probabilities
    beta_oi10, beta_oi20, beta_oi21 : float
        O I escape probabilities
    beta_co : float
        CO escape probability
    co_jmax : int
        Maximum CO J level
    heating_per_cr : float
        Energy per CR ionization [erg]
    pe_heating_rate_0 : float
        Base photoelectric heating rate
    sigma_dust : float
        Dust cross section [cm^2]
    f_dust : float
        Dust abundance factor
    Tg_out : ndarray
        Output: solved gas temperature [K]
    """
    N = nH.size
    for i in prange(N):
        Tg_out[i] = bisect_solve_cell_se(
            nH[i], nH2[i], nHI[i], Td[i], chi[i], ne[i],
            nCplus[i], nC[i], nO[i], nco_gas[i],
            mask, T_min, T_max, max_iter, tol,
            zeta_cr, pah_scale, alpha_acc, beta_cii,
            beta_ci10, beta_ci20, beta_ci21,
            beta_oi10, beta_oi20, beta_oi21,
            beta_co, co_jmax, heating_per_cr, pe_heating_rate_0,
            sigma_dust, f_dust
        )


# -----------------------------------------------------------------------------
# Diagnostic kernel: compute per-term rates using SE cooling
# -----------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def compute_terms_kernel_se(
    Tg: np.ndarray,
    nH: np.ndarray,
    nH2: np.ndarray,
    nHI: np.ndarray,
    Td: np.ndarray,
    chi: np.ndarray,
    ne: np.ndarray,
    nCplus: np.ndarray,
    nC: np.ndarray,
    nO: np.ndarray,
    nco_gas: np.ndarray,
    zeta_cr: float,
    pah_scale: float,
    alpha_acc: float,
    beta_cii: float,
    beta_ci10: float,
    beta_ci20: float,
    beta_ci21: float,
    beta_oi10: float,
    beta_oi20: float,
    beta_oi21: float,
    beta_co: float,
    co_jmax: int,
    heating_per_cr: float,
    pe_heating_rate_0: float,
    sigma_dust: float,
    f_dust: float,
    rate_cr_out: np.ndarray,
    rate_pe_out: np.ndarray,
    rate_cii_out: np.ndarray,
    rate_ci_out: np.ndarray,
    rate_oi_out: np.ndarray,
    rate_co_out: np.ndarray,
    rate_gd_out: np.ndarray,
) -> None:
    """Compute individual heating/cooling rates using SE-based cooling.
    
    All output arrays have units [erg cm^-3 s^-1].
    Heating terms are positive, cooling terms are negative.
    """
    N = nH.size
    for i in prange(N):
        rate_cr_out[i] = cosmic_ray_heating(nH[i], zeta_cr, heating_per_cr)
        rate_pe_out[i] = photoelectric_heating(nH[i], chi[i], ne[i], Tg[i], pah_scale, pe_heating_rate_0)
        rate_cii_out[i] = -cii_cooling_lamda(nCplus[i], Tg[i], nH2[i], nHI[i], ne[i], beta_cii)
        rate_ci_out[i] = -ci_cooling_lamda(nC[i], Tg[i], nH2[i], nHI[i], ne[i], beta_ci10, beta_ci20, beta_ci21)
        rate_oi_out[i] = -oi_cooling_lamda(nO[i], Tg[i], nH2[i], nHI[i], ne[i], beta_oi10, beta_oi20, beta_oi21)
        rate_co_out[i] = -co_cooling_lamda(nco_gas[i], Tg[i], nH2[i], nHI[i], beta_co, co_jmax)
        rate_gd_out[i] = gas_dust_exchange(nH[i], Td[i], Tg[i], alpha_acc, sigma_dust, f_dust, K_B, M_H)
