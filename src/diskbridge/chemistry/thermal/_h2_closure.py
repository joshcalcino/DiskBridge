"""Hydrogen partition closure for H/H2 balance.

Computes nH2 and nHI from a local steady-state balance between:
- H2 formation on grains (rate R_form)
- H2 photodissociation by FUV (attenuated by dust + H2 self-shielding)

This is NOT a time-dependent chemistry network - just a local equilibrium closure
that makes diffuse regions naturally atomic and dense/shielded regions molecular.

References
----------
- Hollenbach & McKee 1979, ApJS 41, 555 (grain formation rates)
- Draine & Bertoldi 1996, ApJ 468, 269 (H2 self-shielding)
- Krumholz et al. 2009, ApJ 693, 216 (analytic H/H2 partitioning)
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Physical constants
AU_CM = 1.49597871e13  # cm per AU


@njit(fastmath=True, cache=True)
def h2_self_shielding_db96(N_H2: float, b5: float = 2.0, alpha: float = -0.75) -> float:
    """H2 self-shielding factor from Draine & Bertoldi (1996).
    
    Parameters
    ----------
    N_H2 : float
        H2 column density [cm^-2]
    b5 : float
        Doppler parameter in units of 10^5 cm/s (default 2.0 for ~2 km/s)
    alpha : float
        Power-law exponent (default -0.75)
        
    Returns
    -------
    float
        Self-shielding factor f_shield in [0, 1]
        (1 = unshielded, 0 = fully shielded)
    """
    x = N_H2 / 5.0e14
    
    if x < 1e-10:
        return 1.0
    
    term1 = (1.0 + x / b5) ** alpha
    term2 = np.exp(-5.0e-4 * np.sqrt(1.0 + x))
    
    f_shield = term1 * term2
    
    return max(f_shield, 0.0)


@njit(fastmath=True, cache=True)
def h2_dissociation_rate(
    chi_eff: float,
    N_H2: float,
    N_H: float,
    k0_diss: float,
    use_dust_attn: bool,
    sigma_d: float,
    b5: float,
    alpha: float,
) -> float:
    """Compute effective H2 photodissociation rate including shielding.
    
    Parameters
    ----------
    chi_eff : float
        Effective FUV field strength (relative to Draine field)
    N_H2 : float
        H2 column density [cm^-2]
    N_H : float
        Total H nuclei column density [cm^-2]
    k0_diss : float
        Unshielded H2 dissociation rate at chi=1 [s^-1]
    use_dust_attn : bool
        If True, apply dust attenuation (set False if chi_eff already includes dust)
    sigma_d : float
        Dust FUV absorption cross-section per H [cm^2]
    b5 : float
        Doppler parameter for self-shielding
    alpha : float
        Power-law exponent for self-shielding
        
    Returns
    -------
    float
        Effective H2 dissociation rate [s^-1]
    """
    # Self-shielding by H2
    f_shield = h2_self_shielding_db96(N_H2, b5, alpha)
    
    # Dust attenuation (only if not already in chi_eff)
    if use_dust_attn and N_H > 0.0:
        tau_dust = sigma_d * N_H
        f_dust = np.exp(-tau_dust)
    else:
        f_dust = 1.0
    
    k_diss = k0_diss * chi_eff * f_shield * f_dust
    
    return k_diss


@njit(fastmath=True, cache=True)
def solve_h2_fraction(
    n_H: float,
    chi_eff: float,
    L_shield: float,
    R_form: float,
    k0_diss: float,
    use_dust_attn: bool,
    sigma_d: float,
    b5: float,
    alpha: float,
    max_iter: int = 30,
    tol: float = 1.0e-6,
) -> float:
    """Solve for H2 fraction using bisection on fH2 = 2*nH2/nH.
    
    Steady-state balance:
        R_form * n_H * n_HI = k_diss(N_H2) * n_H2
        
    where k_diss depends on N_H2 = n_H2 * L_shield (self-shielding).
    
    Parameters
    ----------
    n_H : float
        Total H nuclei density [cm^-3]
    chi_eff : float
        Effective FUV field strength (relative to Draine field)
    L_shield : float
        Shielding length [cm]
    R_form : float
        H2 grain-surface formation rate coefficient [cm^3 s^-1]
    k0_diss : float
        Unshielded H2 dissociation rate at chi=1 [s^-1]
    use_dust_attn : bool
        Apply dust attenuation to FUV
    sigma_d : float
        Dust FUV cross-section per H [cm^2]
    b5 : float
        Doppler parameter for self-shielding
    alpha : float
        Power-law exponent for self-shielding
    max_iter : int
        Maximum bisection iterations
    tol : float
        Convergence tolerance on fH2
        
    Returns
    -------
    float
        fH2 = 2*nH2/nH, the molecular hydrogen fraction in [0, 1]
    """
    if n_H <= 0.0:
        return 0.0
    
    if chi_eff <= 0.0:
        return 1.0
    
    # Bisection on fH2 in [0, 1]
    # At equilibrium: R_form * nH * nHI = k_diss * nH2
    # With nHI = (1 - fH2) * nH and nH2 = 0.5 * fH2 * nH
    # => R_form * nH * (1 - fH2) * nH = k_diss(fH2) * 0.5 * fH2 * nH
    # => 2 * R_form * nH * (1 - fH2) = k_diss(fH2) * fH2
    
    fH2_lo = 0.0
    fH2_hi = 1.0
    
    for _ in range(max_iter):
        fH2_mid = 0.5 * (fH2_lo + fH2_hi)
        
        # Compute column densities
        n_H2 = 0.5 * fH2_mid * n_H
        N_H2 = n_H2 * L_shield
        N_H = n_H * L_shield
        
        # Compute dissociation rate at this fH2
        k_diss = h2_dissociation_rate(
            chi_eff, N_H2, N_H, k0_diss, use_dust_attn, sigma_d, b5, alpha
        )
        
        # Residual: formation - destruction
        # formation = R_form * nH * nHI = R_form * nH * (1 - fH2) * nH
        # destruction = k_diss * nH2 = k_diss * 0.5 * fH2 * nH
        # At equilibrium: 2 * R_form * nH * (1 - fH2) = k_diss * fH2
        
        lhs = 2.0 * R_form * n_H * (1.0 - fH2_mid)  # formation rate per H2
        rhs = k_diss * fH2_mid  # destruction rate per H2 (normalized)
        
        residual = lhs - rhs
        
        if residual > 0:
            # Formation exceeds destruction => need more H2
            fH2_lo = fH2_mid
        else:
            # Destruction exceeds formation => need less H2
            fH2_hi = fH2_mid
        
        if fH2_hi - fH2_lo < tol:
            break
    
    return 0.5 * (fH2_lo + fH2_hi)


@njit(fastmath=True, cache=True)
def compute_h_partition(
    n_H: float,
    chi_eff: float,
    L_shield: float,
    R_form: float,
    k0_diss: float,
    use_dust_attn: bool,
    sigma_d: float,
    b5: float,
    alpha: float,
) -> tuple:
    """Compute H/H2 partition from closure model.
    
    Parameters
    ----------
    n_H : float
        Total H nuclei density [cm^-3]
    chi_eff : float
        Effective FUV field strength
    L_shield : float
        Shielding length [cm]
    R_form : float
        H2 formation rate coefficient [cm^3 s^-1]
    k0_diss : float
        Unshielded H2 dissociation rate [s^-1]
    use_dust_attn : bool
        Apply dust attenuation
    sigma_d : float
        Dust cross-section [cm^2]
    b5 : float
        Self-shielding Doppler parameter
    alpha : float
        Self-shielding exponent
        
    Returns
    -------
    tuple (nH2, nHI, fH2)
        H2 density [cm^-3], atomic H density [cm^-3], molecular fraction
    """
    fH2 = solve_h2_fraction(
        n_H, chi_eff, L_shield, R_form, k0_diss,
        use_dust_attn, sigma_d, b5, alpha
    )
    
    nH2 = 0.5 * fH2 * n_H
    nHI = (1.0 - fH2) * n_H
    
    return nH2, nHI, fH2


def compute_h_partition_array(
    n_H: np.ndarray,
    chi_eff: np.ndarray,
    L_shield: np.ndarray,
    R_form: float,
    k0_diss: float,
    use_dust_attn: bool,
    sigma_d: float,
    b5: float,
    alpha: float,
) -> tuple:
    """Vectorized H/H2 partition for arrays.
    
    Parameters
    ----------
    n_H : ndarray
        Total H nuclei density [cm^-3]
    chi_eff : ndarray
        Effective FUV field strength
    L_shield : ndarray
        Shielding length [cm]
    R_form, k0_diss, use_dust_attn, sigma_d, b5, alpha : float
        Closure model parameters
        
    Returns
    -------
    tuple (nH2, nHI, fH2)
        Arrays of H2 density, atomic H density, molecular fraction
    """
    n = len(n_H)
    nH2 = np.empty(n, dtype=np.float64)
    nHI = np.empty(n, dtype=np.float64)
    fH2 = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        nH2[i], nHI[i], fH2[i] = compute_h_partition(
            n_H[i], chi_eff[i], L_shield[i],
            R_form, k0_diss, use_dust_attn, sigma_d, b5, alpha
        )
    
    return nH2, nHI, fH2


def check_h_conservation(
    nH2: np.ndarray,
    nHI: np.ndarray,
    n_H: np.ndarray,
    rtol: float = 1.0e-6,
) -> None:
    """Check H nuclei conservation: nHI + 2*nH2 == n_H.
    
    Raises ValueError if conservation is violated beyond tolerance.
    
    Parameters
    ----------
    nH2 : ndarray
        H2 density [cm^-3]
    nHI : ndarray
        Atomic H density [cm^-3]
    n_H : ndarray
        Total H nuclei density [cm^-3]
    rtol : float
        Relative tolerance
    """
    computed = nHI + 2.0 * nH2
    rel_err = np.abs(computed - n_H) / np.maximum(n_H, 1e-30)
    max_err = np.max(rel_err)
    
    if max_err > rtol:
        idx = np.argmax(rel_err)
        raise ValueError(
            f"H conservation violated: max relative error = {max_err:.2e} "
            f"at index {idx} (nHI={nHI[idx]:.2e}, nH2={nH2[idx]:.2e}, "
            f"nH={n_H[idx]:.2e}, computed={computed[idx]:.2e})"
        )
