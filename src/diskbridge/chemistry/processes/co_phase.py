from __future__ import annotations

import numpy as np
from numba import njit, prange

from diskbridge._constants import (
    K_B,
    M_H,
    E_BIND_CO,
    NU0_CO,
    F_DRAINE,
    N_LAY,
    N_SURF,
    Y_CO,
)

M_CO_CGS = 28.0 * M_H


@njit(inline='always')
def _v_th_co_cgs(Tgas_K: float) -> float:
    if Tgas_K <= 0.0:
        return 0.0
    pi = 3.141592653589793
    return np.sqrt(8.0 * K_B * Tgas_K / (pi * M_CO_CGS))


@njit(inline='always')
def co_freezeout_rate_cgs(nH_cm3: float, Tgas_K: float, sigma_d_per_H_cm2: float) -> float:
    return (sigma_d_per_H_cm2 * nH_cm3) * _v_th_co_cgs(Tgas_K)


@njit(inline='always')
def co_thermal_desorption_rate_cgs(Tdust_K: float) -> float:
    if Tdust_K <= 0.0:
        return 0.0
    return NU0_CO * np.exp(-E_BIND_CO / Tdust_K)


@njit(inline='always')
def co_photodesorption_surface_rate_cgs(chi: float) -> float:
    return (chi * F_DRAINE) * (Y_CO / (4.0 * N_SURF * float(N_LAY)))


@njit(inline='always')
def co_active_ice_max_cgs(nH_cm3: float, sigma_d_per_H_cm2: float) -> float:
    return (sigma_d_per_H_cm2 * nH_cm3) * N_SURF * float(N_LAY)


@njit(inline='always')
def co_active_ice_cgs(
    nH_cm3: float,
    sigma_d_per_H_cm2: float,
    nco_ice_cm3: float,
) -> tuple[float, float]:
    n_ice_act_max = co_active_ice_max_cgs(nH_cm3, sigma_d_per_H_cm2)
    n_ice_act = nco_ice_cm3 if nco_ice_cm3 <= n_ice_act_max else n_ice_act_max
    return n_ice_act_max, n_ice_act


@njit(inline='always')
def co_photodesorption_R_cgs(k_pd_surf_s: float, n_ice_act_cm3: float) -> float:
    return k_pd_surf_s * n_ice_act_cm3


@njit(parallel=True, cache=True)
def co_freezeout_rate_field_cgs(
    nH_cm3: np.ndarray,
    Tgas_K: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
) -> np.ndarray:
    out = np.empty(nH_cm3.shape, dtype=np.float64)
    out_flat = out.ravel()
    nH_flat = nH_cm3.ravel()
    T_flat = Tgas_K.ravel()
    sig_flat = sigma_d_per_H_cm2.ravel()
    for i in prange(out_flat.size):
        out_flat[i] = co_freezeout_rate_cgs(nH_flat[i], T_flat[i], sig_flat[i])
    return out


@njit(parallel=True, cache=True)
def co_thermal_desorption_rate_field_cgs(Tdust_K: np.ndarray) -> np.ndarray:
    out = np.empty(Tdust_K.shape, dtype=np.float64)
    out_flat = out.ravel()
    T_flat = Tdust_K.ravel()
    for i in prange(out_flat.size):
        out_flat[i] = co_thermal_desorption_rate_cgs(T_flat[i])
    return out


@njit(parallel=True, cache=True)
def co_photodesorption_surface_rate_field_cgs(chi: np.ndarray) -> np.ndarray:
    out = np.empty(chi.shape, dtype=np.float64)
    out_flat = out.ravel()
    chi_flat = chi.ravel()
    for i in prange(out_flat.size):
        out_flat[i] = co_photodesorption_surface_rate_cgs(chi_flat[i])
    return out


@njit(parallel=True, cache=True)
def co_active_ice_field_cgs(
    nH_cm3: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
    nco_ice_cm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_max = np.empty(nH_cm3.shape, dtype=np.float64)
    out_act = np.empty(nH_cm3.shape, dtype=np.float64)
    out_max_flat = out_max.ravel()
    out_act_flat = out_act.ravel()
    nH_flat = nH_cm3.ravel()
    sig_flat = sigma_d_per_H_cm2.ravel()
    nco_flat = nco_ice_cm3.ravel()
    for i in prange(out_max_flat.size):
        n_max, n_act = co_active_ice_cgs(nH_flat[i], sig_flat[i], nco_flat[i])
        out_max_flat[i] = n_max
        out_act_flat[i] = n_act
    return out_max, out_act


@njit(parallel=True, cache=True)
def co_photodesorption_R_field_cgs(k_pd_surf_s: np.ndarray, n_ice_act_cm3: np.ndarray) -> np.ndarray:
    out = np.empty(n_ice_act_cm3.shape, dtype=np.float64)
    out_flat = out.ravel()
    k_flat = k_pd_surf_s.ravel()
    n_flat = n_ice_act_cm3.ravel()
    for i in prange(out_flat.size):
        out_flat[i] = co_photodesorption_R_cgs(k_flat[i], n_flat[i])
    return out
