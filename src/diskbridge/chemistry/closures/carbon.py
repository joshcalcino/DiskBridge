"""Carbon ionization closure model.

Computes C/C+/e- abundances from a local steady-state balance between:
- C photoionization by FUV (rate Gamma_C0 * chi)
- C+ recombination with electrons (rate alpha_rec(T) * ne)

This is NOT a full chemistry network - just a local equilibrium closure
that partitions available carbon (after CO formation) into C and C+.

References
----------
- Standard ISM carbon photoionization rates
- Recombination rates follow T^-0.6 scaling
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from diskbridge._units import Quantity


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
        UV field strength (dust-attenuated, NOT CO-shielded) [dimensionless]
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
        Temperature exponent for recombination (typically -0.6)
    nCplus_out : ndarray
        Output: C+ number density [cm^-3]
    nC_out : ndarray
        Output: neutral C number density [cm^-3]
    ne_out : ndarray
        Output: electron number density [cm^-3]
        
    Notes
    -----
    The UV field `chi` should be the dust-attenuated field, NOT the CO-shielded
    field (chi_eff = chi * theta_co). Carbon photoionization occurs in the FUV
    continuum which is attenuated by dust but not by CO line absorption.
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
        ne_out[i] = nCplus


def compute_carbon_closure(
    nH: Quantity,
    chi: Quantity,
    Tg: Quantity,
    nco_total: Quantity,
    X_C_tot: float = None,
    Gamma_C0: float = None,
    alpha_rec_c0: float = None,
    T_rec_exp: float = None,
) -> tuple[Quantity, Quantity, Quantity]:
    """Compute carbon ionization closure (Python wrapper).
    
    This is the high-level interface that accepts Quantities and returns Quantities.
    
    Parameters
    ----------
    nH : Quantity
        H nuclei number density [cm^-3]
    chi : Quantity or array
        UV field strength (dust-attenuated, NOT CO-shielded) [dimensionless]
    Tg : Quantity
        Gas temperature [K]
    nco_total : Quantity
        Total CO number density (gas + ice) [cm^-3]
    X_C_tot : float, optional
        Total carbon abundance. If None, uses value from _constants.
    Gamma_C0 : float, optional
        C photoionization rate at chi=1. If None, uses value from _constants.
    alpha_rec_c0 : float, optional
        Recombination rate at 300K. If None, uses value from _constants.
    T_rec_exp : float, optional
        Recombination temperature exponent. If None, uses value from _constants.
        
    Returns
    -------
    nCplus : Quantity
        C+ number density [cm^-3]
    nC : Quantity
        Neutral C number density [cm^-3]
    ne : Quantity
        Electron number density [cm^-3]
        
    Notes
    -----
    The UV field `chi` should be dust-attenuated only (e.g., rad.chi or
    rad.ensure_chi()), NOT the CO-shielded field (rad.chi_eff).
    """
    from diskbridge._constants import (
        X_C_TOT as _X_C_TOT,
        GAMMA_C0 as _GAMMA_C0,
        ALPHA_REC_C0 as _ALPHA_REC_C0,
        T_REC_EXP as _T_REC_EXP,
    )
    
    if X_C_tot is None:
        X_C_tot = _X_C_TOT
    if Gamma_C0 is None:
        Gamma_C0 = _GAMMA_C0
    if alpha_rec_c0 is None:
        alpha_rec_c0 = _ALPHA_REC_C0
    if T_rec_exp is None:
        T_rec_exp = _T_REC_EXP
    
    orig_shape = nH.magnitude.shape
    ncells = nH.magnitude.size
    
    nH_flat = np.ascontiguousarray(
        nH.to('cm^-3').magnitude.flatten(), dtype=np.float64
    )
    
    if hasattr(chi, 'magnitude'):
        chi_flat = np.ascontiguousarray(
            chi.magnitude.flatten(), dtype=np.float64
        )
    else:
        chi_flat = np.ascontiguousarray(
            np.asarray(chi).flatten(), dtype=np.float64
        )
    
    Tg_flat = np.ascontiguousarray(
        Tg.to('K').magnitude.flatten(), dtype=np.float64
    )
    
    nco_flat = np.ascontiguousarray(
        nco_total.to('cm^-3').magnitude.flatten(), dtype=np.float64
    )
    
    nCplus_flat = np.zeros(ncells, dtype=np.float64)
    nC_flat = np.zeros(ncells, dtype=np.float64)
    ne_flat = np.zeros(ncells, dtype=np.float64)
    
    carbon_closure_kernel(
        nH_flat, chi_flat, Tg_flat, nco_flat,
        X_C_tot, Gamma_C0, alpha_rec_c0, T_rec_exp,
        nCplus_flat, nC_flat, ne_flat,
    )
    
    nCplus = Quantity(nCplus_flat.reshape(orig_shape), 'cm^-3')
    nC = Quantity(nC_flat.reshape(orig_shape), 'cm^-3')
    ne = Quantity(ne_flat.reshape(orig_shape), 'cm^-3')
    
    return nCplus, nC, ne


def check_carbon_budget(
    nC: Quantity,
    nCplus: Quantity,
    nco_total: Quantity,
    nH: Quantity,
    X_C_tot: float = None,
    rtol: float = 1e-4,
) -> None:
    """Check carbon budget conservation.
    
    Validates: nC + nCplus + nCO <= X_C_tot * nH
    
    Parameters
    ----------
    nC : Quantity
        Neutral C density [cm^-3]
    nCplus : Quantity
        C+ density [cm^-3]
    nco_total : Quantity
        Total CO density [cm^-3]
    nH : Quantity
        H nuclei density [cm^-3]
    X_C_tot : float, optional
        Total carbon abundance. If None, uses value from _constants.
    rtol : float
        Relative tolerance
        
    Raises
    ------
    ValueError
        If carbon budget is violated
    """
    from diskbridge._constants import X_C_TOT as _X_C_TOT
    
    if X_C_tot is None:
        X_C_tot = _X_C_TOT
    
    nC_flat = nC.to('cm^-3').magnitude.flatten()
    nCplus_flat = nCplus.to('cm^-3').magnitude.flatten()
    nco_flat = nco_total.to('cm^-3').magnitude.flatten()
    nH_flat = nH.to('cm^-3').magnitude.flatten()
    
    n_C_total = nC_flat + nCplus_flat + nco_flat
    n_C_max = X_C_tot * nH_flat
    
    excess = n_C_total - n_C_max
    rel_excess = excess / np.maximum(n_C_max, 1e-30)
    max_excess = np.max(rel_excess)
    
    if max_excess > rtol:
        idx = np.argmax(rel_excess)
        raise ValueError(
            f"Carbon budget violated: nC + nCplus + nCO exceeds X_C*nH by "
            f"{max_excess:.2e} (relative) at index {idx}. "
            f"nC={nC_flat[idx]:.2e}, nCplus={nCplus_flat[idx]:.2e}, "
            f"nCO={nco_flat[idx]:.2e}, X_C*nH={n_C_max[idx]:.2e}"
        )
