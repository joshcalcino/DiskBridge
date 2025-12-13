from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, 0.0, 1.0)
    return x_clipped * x_clipped * (3.0 - 2.0 * x_clipped)


def compute_freezeout_factor(
    T_vals_K: np.ndarray,
    Tfrz_K: float,
    eps: float,
    smooth_Tfrz_K: float,
) -> Tuple[np.ndarray, np.ndarray]:
    freeze_factor = np.ones_like(T_vals_K, dtype=float)

    if smooth_Tfrz_K > 0.0:
        half_T = 0.5 * float(smooth_Tfrz_K)
        T_low = float(Tfrz_K) - half_T
        T_high = float(Tfrz_K) + half_T

        mask_cold = T_vals_K <= T_low
        mask_warm = T_vals_K >= T_high
        mask_mid = (~mask_cold) & (~mask_warm)

        freeze_factor[mask_cold] = float(eps)
        freeze_factor[mask_warm] = 1.0
        if np.any(mask_mid):
            t_mid = (T_vals_K[mask_mid] - T_low) / (2.0 * half_T)
            s_mid = _smoothstep01(t_mid)
            freeze_factor[mask_mid] = float(eps) + (1.0 - float(eps)) * s_mid

        mask_frz = freeze_factor < 1.0 - 1e-12
    else:
        mask_frz = T_vals_K < float(Tfrz_K)
        freeze_factor[mask_frz] = float(eps)

    return freeze_factor, mask_frz


def apply_photodesorption_escape(
    X: Any,
    *,
    X0: float,
    freezeout: bool,
    freeze_factor: Optional[np.ndarray],
    mask_frz: Optional[np.ndarray],
    chi_over_nH: np.ndarray,
    log_thr_pdes: float,
    smooth_log_chi_nH_dex: float,
) -> Tuple[Any, np.ndarray]:
    mask_pdes = np.zeros_like(chi_over_nH, dtype=bool)

    if smooth_log_chi_nH_dex > 0.0:
        half_dex = 0.5 * float(smooth_log_chi_nH_dex)
        lower = float(log_thr_pdes) - half_dex
        width = max(float(smooth_log_chi_nH_dex), 1e-30)
        t = (chi_over_nH - lower) / width
        w = _smoothstep01(t)

        if freezeout:
            if freeze_factor is None:
                raise ValueError("freeze_factor is required when freezeout=True")

            has_freeze = freeze_factor < 1.0 - 1e-12
            if np.any(has_freeze):
                w_eff = w * has_freeze
                F = freeze_factor
                F_new = F + w_eff * (1.0 - F)
                ratio_F = np.ones_like(F, dtype=float)
                positive = F > 0.0
                ratio_F[positive] = F_new[positive] / F[positive]
                X *= ratio_F

        mask_pdes = chi_over_nH > float(log_thr_pdes)
    else:
        mask_pdes = chi_over_nH > float(log_thr_pdes)
        if freezeout:
            if mask_frz is None:
                raise ValueError("mask_frz is required when freezeout=True")
            unfreeze_mask = mask_pdes & mask_frz
            X[unfreeze_mask] = float(X0)

    return X, mask_pdes


def apply_photodissociation(
    X: Any,
    *,
    chi_over_nH: np.ndarray,
    log_thr_pdiss: float,
    smooth_log_chi_nH_dex: float,
) -> Tuple[Any, np.ndarray]:
    if smooth_log_chi_nH_dex > 0.0:
        half_dex = 0.5 * float(smooth_log_chi_nH_dex)
        lower = float(log_thr_pdiss) - half_dex
        width = max(float(smooth_log_chi_nH_dex), 1e-30)
        t = (chi_over_nH - lower) / width
        w_pdiss = _smoothstep01(t)
        kill_factor = 1.0 - w_pdiss
        X *= kill_factor
        mask_pdiss = chi_over_nH > float(log_thr_pdiss)
    else:
        mask_pdiss = chi_over_nH > float(log_thr_pdiss)
        X[mask_pdiss] = 0.0

    return X, mask_pdiss
