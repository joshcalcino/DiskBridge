from __future__ import annotations

import numpy as np
from numba import njit

from diskbridge._constants import X_C_TOT, GAMMA_C0, ALPHA_REC_C0, T_REC_EXP


@njit(inline='always')
def carbon_closure_cell_param_cgs(
    nH_cm3: float,
    chi: float,
    Tg_K: float,
    nco_total_cm3: float,
    X_C_tot: float,
    Gamma_C0: float,
    alpha_rec_c0: float,
    T_rec_exp: float,
) -> tuple[float, float, float]:
    n_C_tot = X_C_tot * nH_cm3
    n_C_available = max(n_C_tot - nco_total_cm3, 0.0)

    Gamma_photo = Gamma_C0 * chi
    alpha = 0.0
    if Tg_K > 0.0:
        alpha = alpha_rec_c0 * (Tg_K / 300.0) ** T_rec_exp

    if Gamma_photo <= 0.0 or n_C_available <= 0.0:
        nCplus = 0.0
    elif alpha <= 0.0:
        nCplus = n_C_available
    else:
        disc = Gamma_photo * Gamma_photo + 4.0 * alpha * Gamma_photo * n_C_available
        sqrt_disc = np.sqrt(disc)
        nCplus = (2.0 * Gamma_photo * n_C_available) / (Gamma_photo + sqrt_disc)
        if nCplus < 0.0:
            nCplus = 0.0
        elif nCplus > n_C_available:
            nCplus = n_C_available

    nC = n_C_available - nCplus
    ne = nCplus
    return nCplus, nC, ne


@njit(inline='always')
def carbon_closure_cell_cgs(
    nH_cm3: float,
    chi: float,
    Tg_K: float,
    nco_total_cm3: float,
) -> tuple[float, float, float]:
    return carbon_closure_cell_param_cgs(
        nH_cm3=nH_cm3,
        chi=chi,
        Tg_K=Tg_K,
        nco_total_cm3=nco_total_cm3,
        X_C_tot=float(X_C_TOT),
        Gamma_C0=float(GAMMA_C0),
        alpha_rec_c0=float(ALPHA_REC_C0),
        T_rec_exp=float(T_REC_EXP),
    )
