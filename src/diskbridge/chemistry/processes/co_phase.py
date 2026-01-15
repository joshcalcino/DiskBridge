from __future__ import annotations

import numpy as np
import diskbridge._gow17 as _gow17

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


def co_freezeout_rate_cgs(nH_cm3: float, Tgas_K: float, sigma_d_per_H_cm2: float) -> float:
    return float(
        _gow17.co_freezeout_rate_cgs(
            float(nH_cm3),
            float(Tgas_K),
            float(sigma_d_per_H_cm2),
            float(K_B),
            float(M_CO_CGS),
        )
    )


def co_thermal_desorption_rate_cgs(Tdust_K: float) -> float:
    return float(
        _gow17.co_thermal_desorption_rate_cgs(
            float(Tdust_K),
            float(NU0_CO),
            float(E_BIND_CO),
        )
    )


def co_photodesorption_surface_rate_cgs(chi: float) -> float:
    return float(
        _gow17.co_photodesorption_surface_rate_cgs(
            float(chi),
            float(F_DRAINE),
            float(Y_CO),
            float(N_SURF),
            int(N_LAY),
        )
    )


def co_active_ice_max_cgs(nH_cm3: float, sigma_d_per_H_cm2: float) -> float:
    n_ice_act_max, _ = co_active_ice_cgs(
        nH_cm3=float(nH_cm3),
        sigma_d_per_H_cm2=float(sigma_d_per_H_cm2),
        nco_ice_cm3=float(0.0),
    )
    return float(n_ice_act_max)


def co_active_ice_cgs(
    nH_cm3: float,
    sigma_d_per_H_cm2: float,
    nco_ice_cm3: float,
) -> tuple[float, float]:
    n_ice_act_max, n_ice_act = _gow17.co_active_ice_cgs(
        float(nH_cm3),
        float(sigma_d_per_H_cm2),
        float(nco_ice_cm3),
        float(N_SURF),
        int(N_LAY),
    )
    return float(n_ice_act_max), float(n_ice_act)


def co_photodesorption_R_cgs(k_pd_surf_s: float, n_ice_act_cm3: float) -> float:
    return float(_gow17.co_photodesorption_R_cgs(float(k_pd_surf_s), float(n_ice_act_cm3)))


def co_freezeout_rate_field_cgs(
    nH_cm3: np.ndarray,
    Tgas_K: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
) -> np.ndarray:
    out = np.empty(np.shape(nH_cm3), dtype=np.float64)
    nH_flat = np.asarray(nH_cm3, dtype=np.float64).ravel()
    T_flat = np.asarray(Tgas_K, dtype=np.float64).ravel()
    sig_flat = np.asarray(sigma_d_per_H_cm2, dtype=np.float64).ravel()
    out_flat = out.ravel()
    for i in range(out_flat.size):
        out_flat[i] = co_freezeout_rate_cgs(float(nH_flat[i]), float(T_flat[i]), float(sig_flat[i]))
    return out


def co_thermal_desorption_rate_field_cgs(Tdust_K: np.ndarray) -> np.ndarray:
    out = np.empty(np.shape(Tdust_K), dtype=np.float64)
    T_flat = np.asarray(Tdust_K, dtype=np.float64).ravel()
    out_flat = out.ravel()
    for i in range(out_flat.size):
        out_flat[i] = co_thermal_desorption_rate_cgs(float(T_flat[i]))
    return out


def co_photodesorption_surface_rate_field_cgs(chi: np.ndarray) -> np.ndarray:
    out = np.empty(np.shape(chi), dtype=np.float64)
    chi_flat = np.asarray(chi, dtype=np.float64).ravel()
    out_flat = out.ravel()
    for i in range(out_flat.size):
        out_flat[i] = co_photodesorption_surface_rate_cgs(float(chi_flat[i]))
    return out


def co_active_ice_field_cgs(
    nH_cm3: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
    nco_ice_cm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_max = np.empty(np.shape(nH_cm3), dtype=np.float64)
    out_act = np.empty(np.shape(nH_cm3), dtype=np.float64)

    nH_flat = np.asarray(nH_cm3, dtype=np.float64).ravel()
    sig_flat = np.asarray(sigma_d_per_H_cm2, dtype=np.float64).ravel()
    nco_ice_flat = np.asarray(nco_ice_cm3, dtype=np.float64).ravel()

    out_max_flat = out_max.ravel()
    out_act_flat = out_act.ravel()
    for i in range(out_max_flat.size):
        n_ice_act_max, n_ice_act = co_active_ice_cgs(
            float(nH_flat[i]),
            float(sig_flat[i]),
            float(nco_ice_flat[i]),
        )
        out_max_flat[i] = n_ice_act_max
        out_act_flat[i] = n_ice_act

    return out_max, out_act


def co_photodesorption_R_field_cgs(k_pd_surf_s: np.ndarray, n_ice_act_cm3: np.ndarray) -> np.ndarray:
    out = np.empty(np.shape(n_ice_act_cm3), dtype=np.float64)
    k_flat = np.asarray(k_pd_surf_s, dtype=np.float64).ravel()
    n_flat = np.asarray(n_ice_act_cm3, dtype=np.float64).ravel()
    out_flat = out.ravel()
    for i in range(out_flat.size):
        out_flat[i] = co_photodesorption_R_cgs(float(k_flat[i]), float(n_flat[i]))
    return out
