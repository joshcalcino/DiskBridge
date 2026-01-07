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


@njit(cache=True)
def _v_th_co_cgs(Tgas_K: float) -> float:
    if Tgas_K <= 0.0:
        return 0.0
    return np.sqrt(8.0 * K_B * Tgas_K / (np.pi * M_CO_CGS))


@njit(cache=True)
def co_freezeout_rate_cgs(nH_cm3: float, Tgas_K: float, sigma_d_per_H_cm2: float) -> float:
    return (sigma_d_per_H_cm2 * nH_cm3) * _v_th_co_cgs(Tgas_K)


@njit(cache=True)
def co_thermal_desorption_rate_cgs(Tdust_K: float) -> float:
    if Tdust_K <= 0.0:
        return 0.0
    return NU0_CO * np.exp(-E_BIND_CO / Tdust_K)


@njit(cache=True)
def co_photodesorption_surface_rate_cgs(chi: float) -> float:
    return (chi * F_DRAINE) * (Y_CO / (4.0 * N_SURF * float(N_LAY)))


@njit(cache=True)
def co_active_ice_max_cgs(nH_cm3: float, sigma_d_per_H_cm2: float) -> float:
    return (sigma_d_per_H_cm2 * nH_cm3) * N_SURF * float(N_LAY)


@njit(cache=True)
def co_active_ice_cgs(
    nH_cm3: float,
    sigma_d_per_H_cm2: float,
    nco_ice_cm3: float,
) -> tuple[float, float]:
    n_ice_act_max = co_active_ice_max_cgs(nH_cm3, sigma_d_per_H_cm2)
    n_ice_act = nco_ice_cm3
    if n_ice_act > n_ice_act_max:
        n_ice_act = n_ice_act_max
    return n_ice_act_max, n_ice_act


@njit(cache=True)
def co_photodesorption_R_cgs(k_pd_surf_s: float, n_ice_act_cm3: float) -> float:
    return k_pd_surf_s * n_ice_act_cm3


@njit(parallel=True, cache=True)
def co_freezeout_rate_field_cgs(
    nH_cm3: np.ndarray,
    Tgas_K: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
) -> np.ndarray:
    out = np.empty(nH_cm3.size, dtype=np.float64)
    nH_flat = nH_cm3.ravel()
    T_flat = Tgas_K.ravel()
    sig_flat = sigma_d_per_H_cm2.ravel()
    for i in prange(out.size):
        out[i] = co_freezeout_rate_cgs(float(nH_flat[i]), float(T_flat[i]), float(sig_flat[i]))
    return out.reshape(nH_cm3.shape)


@njit(parallel=True, cache=True)
def co_thermal_desorption_rate_field_cgs(Tdust_K: np.ndarray) -> np.ndarray:
    out = np.empty(Tdust_K.size, dtype=np.float64)
    T_flat = Tdust_K.ravel()
    for i in prange(out.size):
        out[i] = co_thermal_desorption_rate_cgs(float(T_flat[i]))
    return out.reshape(Tdust_K.shape)


@njit(parallel=True, cache=True)
def co_photodesorption_surface_rate_field_cgs(chi: np.ndarray) -> np.ndarray:
    out = np.empty(chi.size, dtype=np.float64)
    chi_flat = chi.ravel()
    for i in prange(out.size):
        out[i] = co_photodesorption_surface_rate_cgs(float(chi_flat[i]))
    return out.reshape(chi.shape)


@njit(parallel=True, cache=True)
def co_active_ice_field_cgs(
    nH_cm3: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
    nco_ice_cm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_max = np.empty(nH_cm3.size, dtype=np.float64)
    out_act = np.empty(nH_cm3.size, dtype=np.float64)

    nH_flat = nH_cm3.ravel()
    sig_flat = sigma_d_per_H_cm2.ravel()
    nco_ice_flat = nco_ice_cm3.ravel()

    for i in prange(out_max.size):
        n_ice_act_max, n_ice_act = co_active_ice_cgs(
            float(nH_flat[i]),
            float(sig_flat[i]),
            float(nco_ice_flat[i]),
        )
        out_max[i] = n_ice_act_max
        out_act[i] = n_ice_act

    return out_max.reshape(nH_cm3.shape), out_act.reshape(nH_cm3.shape)


@njit(parallel=True, cache=True)
def co_photodesorption_R_field_cgs(k_pd_surf_s: np.ndarray, n_ice_act_cm3: np.ndarray) -> np.ndarray:
    out = np.empty(n_ice_act_cm3.size, dtype=np.float64)
    k_flat = k_pd_surf_s.ravel()
    n_flat = n_ice_act_cm3.ravel()
    for i in prange(out.size):
        out[i] = co_photodesorption_R_cgs(float(k_flat[i]), float(n_flat[i]))
    return out.reshape(n_ice_act_cm3.shape)
