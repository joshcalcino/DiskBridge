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
    return np.asarray(
        _gow17.co_freezeout_rate_field_cgs(
            np.asarray(nH_cm3, dtype=np.float64),
            np.asarray(Tgas_K, dtype=np.float64),
            np.asarray(sigma_d_per_H_cm2, dtype=np.float64),
            float(K_B),
            float(M_CO_CGS),
        ),
        dtype=np.float64,
    )


def co_thermal_desorption_rate_field_cgs(Tdust_K: np.ndarray) -> np.ndarray:
    return np.asarray(
        _gow17.co_thermal_desorption_rate_field_cgs(
            np.asarray(Tdust_K, dtype=np.float64),
            float(NU0_CO),
            float(E_BIND_CO),
        ),
        dtype=np.float64,
    )


def co_photodesorption_surface_rate_field_cgs(chi: np.ndarray) -> np.ndarray:
    return np.asarray(
        _gow17.co_photodesorption_surface_rate_field_cgs(
            np.asarray(chi, dtype=np.float64),
            float(F_DRAINE),
            float(Y_CO),
            float(N_SURF),
            int(N_LAY),
        ),
        dtype=np.float64,
    )


def co_active_ice_field_cgs(
    nH_cm3: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
    nco_ice_cm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_max, out_act = _gow17.co_active_ice_field_cgs(
        np.asarray(nH_cm3, dtype=np.float64),
        np.asarray(sigma_d_per_H_cm2, dtype=np.float64),
        np.asarray(nco_ice_cm3, dtype=np.float64),
        float(N_SURF),
        int(N_LAY),
    )
    return np.asarray(out_max, dtype=np.float64), np.asarray(out_act, dtype=np.float64)


def co_photodesorption_R_field_cgs(k_pd_surf_s: np.ndarray, n_ice_act_cm3: np.ndarray) -> np.ndarray:
    return np.asarray(
        _gow17.co_photodesorption_R_field_cgs(
            np.asarray(k_pd_surf_s, dtype=np.float64),
            np.asarray(n_ice_act_cm3, dtype=np.float64),
        ),
        dtype=np.float64,
    )
