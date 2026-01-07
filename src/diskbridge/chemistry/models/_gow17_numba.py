from __future__ import annotations

import numpy as np
from numba import njit, prange

from diskbridge._constants import X_C_TOT, X_O_TOT
from diskbridge.chemistry.processes.co_phase import (
    co_freezeout_rate_cgs,
    co_thermal_desorption_rate_cgs,
    co_photodesorption_surface_rate_cgs,
    co_active_ice_cgs,
    co_photodesorption_R_cgs,
)

from diskbridge.chemistry.models._gow17_network import (
    N_Y,
    N_GHOST,
    GHOST_OFFSET,
    I_HEP,
    I_OHX,
    I_CHX,
    I_CO,
    I_CP,
    I_HCOP,
    I_H2,
    I_HP,
    I_H3P,
    I_H2P,
    I_SP,
    I_SIP,
    I_OP,
    I_CO_ICE,
    G_SI,
    G_S,
    G_C,
    G_O,
    G_HE,
    G_E,
    G_H,
    X_HE_STD,
    X_S_STD,
    X_SI_STD,
    TEMP_COLL_K,
    N_CR,
    IN_CR,
    OUT_CR,
    K_CR_BASE,
    N_2BODY,
    IN_2BODY1,
    IN_2BODY2,
    OUT_2BODY1,
    OUT_2BODY2,
    K2_TEXP,
    K2_BASE,
    A_KCHX,
    N_KCHX,
    C_KCHX,
    TI_KCHX,
    N_PH,
    IPH_C,
    IPH_CO,
    IPH_H2,
    IN_PH,
    OUT_PH1,
    KPH_BASE,
    N_GR,
    IN_GR,
    OUT_GR,
    C_HP,
    C_CP,
    C_HEP,
    C_SP,
    C_SIP,
)


@njit(cache=True)
def _poly_lnTe_HI_ion(lnTe: float) -> float:
    p = -2.03914985e-6
    p = 1.11954395e-4 + p * lnTe
    p = -2.63197617e-3 + p * lnTe
    p = 3.48255977e-2 + p * lnTe
    p = -2.877056e-1 + p * lnTe
    p = 1.56315498 + p * lnTe
    p = -5.73932875 + p * lnTe
    p = 1.35365560e1 + p * lnTe
    p = -3.271396786e1 + p * lnTe
    return p


@njit(cache=True)
def _kgr_ion_rate(
    c: np.ndarray,
    psi: float,
    T_K: float,
    logT: float,
    nH_cm3: float,
    Zd: float,
    fscale: float,
) -> float:
    denom = 1.0 + c[1] * (psi ** c[2]) * (1.0 + c[3] * (T_K ** c[4]) * (psi ** (-c[5] - c[6] * logT)))
    return 1.0e-14 * c[0] / denom * nH_cm3 * Zd * fscale


@njit(cache=True)
def _solve_linear_system_gauss(A: np.ndarray, b: np.ndarray, x_out: np.ndarray) -> int:
    n = A.shape[0]
    A_work = A.copy()
    b_work = b.copy()

    max_abs = 0.0
    for i in range(n):
        for j in range(n):
            aij = abs(A_work[i, j])
            if aij > max_abs:
                max_abs = aij

    eps = np.finfo(np.float64).eps
    piv_tol = eps * max(1.0, max_abs)

    for k in range(n):
        piv_row = k
        piv = abs(A_work[k, k])
        for r in range(k + 1, n):
            v = abs(A_work[r, k])
            if v > piv:
                piv = v
                piv_row = r

        if piv <= piv_tol:
            return 1

        if piv_row != k:
            tmp = b_work[k]
            b_work[k] = b_work[piv_row]
            b_work[piv_row] = tmp
            for j in range(k, n):
                tmpa = A_work[k, j]
                A_work[k, j] = A_work[piv_row, j]
                A_work[piv_row, j] = tmpa

        piv_val = A_work[k, k]
        for i in range(k + 1, n):
            m = A_work[i, k] / piv_val
            A_work[i, k] = 0.0
            for j in range(k + 1, n):
                A_work[i, j] = A_work[i, j] - m * A_work[k, j]
            b_work[i] = b_work[i] - m * b_work[k]

    for i in range(n - 1, -1, -1):
        s = b_work[i]
        for j in range(i + 1, n):
            s = s - A_work[i, j] * x_out[j]
        diag = A_work[i, i]
        if abs(diag) <= piv_tol:
            return 1
        x_out[i] = s / diag

    return 0


@njit(cache=True)
def _cii_rec_rate(T: float) -> float:
    A = 2.995e-9
    B = 0.7849
    T0 = 6.670e-3
    T1 = 1.943e6
    C = 0.1597
    T2 = 4.955e4

    if T <= 0.0:
        return 0.0

    BN = B + C * np.exp(-T2 / T)
    term1 = np.sqrt(T / T0)
    term2 = np.sqrt(T / T1)
    alpharr = A / (term1 * (1.0 + term1) ** (1.0 - BN) * (1.0 + term2) ** (1.0 + BN))
    alphadr = (T ** (-1.5)) * (
        6.346e-9 * np.exp(-1.217e1 / T)
        + 9.793e-09 * np.exp(-7.38e1 / T)
        + 1.634e-06 * np.exp(-1.523e4 / T)
    )
    return alpharr + alphadr


@njit(cache=True)
def gow17_set_ghost_species(
    y: np.ndarray,
    Zg: float,
    out_yghost: np.ndarray,
) -> None:
    for i in range(N_Y):
        out_yghost[i] = y[i]

    xC = float(Zg) * float(X_C_TOT)
    xO = float(Zg) * float(X_O_TOT)
    xS = float(Zg) * float(X_S_STD)
    xSi = float(Zg) * float(X_SI_STD)

    xCO_ice = out_yghost[I_CO_ICE]

    out_yghost[GHOST_OFFSET + G_C] = xC - out_yghost[I_HCOP] - out_yghost[I_CHX] - out_yghost[I_CO] - out_yghost[I_CP] - xCO_ice
    out_yghost[GHOST_OFFSET + G_O] = xO - out_yghost[I_HCOP] - out_yghost[I_OHX] - out_yghost[I_CO] - out_yghost[I_OP] - xCO_ice
    out_yghost[GHOST_OFFSET + G_HE] = float(X_HE_STD) - out_yghost[I_HEP]
    out_yghost[GHOST_OFFSET + G_S] = xS - out_yghost[I_SP]
    out_yghost[GHOST_OFFSET + G_SI] = xSi - out_yghost[I_SIP]

    out_yghost[GHOST_OFFSET + G_E] = (
        out_yghost[I_HEP]
        + out_yghost[I_CP]
        + out_yghost[I_HCOP]
        + out_yghost[I_H3P]
        + out_yghost[I_H2P]
        + out_yghost[I_HP]
        + out_yghost[I_SP]
        + out_yghost[I_SIP]
        + out_yghost[I_OP]
    )

    out_yghost[GHOST_OFFSET + G_H] = 1.0 - (
        out_yghost[I_OHX]
        + out_yghost[I_CHX]
        + out_yghost[I_HCOP]
        + 3.0 * out_yghost[I_H3P]
        + 2.0 * out_yghost[I_H2P]
        + out_yghost[I_HP]
        + 2.0 * out_yghost[I_H2]
    )

    for j in range(N_GHOST):
        idx = GHOST_OFFSET + j
        if out_yghost[idx] < 0.0:
            out_yghost[idx] = 0.0


@njit(cache=True)
def gow17_compute_rates(
    yghost: np.ndarray,
    nH_cm3: float,
    T_K: float,
    chi: float,
    theta_h2: float,
    theta_co: float,
    theta_c: float,
    ion_rate_s: float,
    Zd: float,
    fH2gr: float,
    fHplusgr: float,
    fCplusgr: float,
    fHeplusgr: float,
    fSplusgr: float,
    fSiplusgr: float,
    fCplusCR: float,
    out_kcr: np.ndarray,
    out_k2: np.ndarray,
    out_kph: np.ndarray,
    out_kgr: np.ndarray,
) -> None:
    xH2 = yghost[I_H2]
    xH = yghost[GHOST_OFFSET + G_H]
    xe = yghost[GHOST_OFFSET + G_E]

    for i in range(N_CR):
        out_kcr[i] = K_CR_BASE[i] * ion_rate_s

    kcr_H_fac = 1.15 * 2.0 * xH2 + 1.5 * xH
    out_kcr[0] *= kcr_H_fac
    out_kcr[2] *= kcr_H_fac
    out_kcr[3] *= (2.0 * xH2 + 3.85 / K_CR_BASE[3]) * fCplusCR
    out_kcr[4] *= 2.0 * xH2
    out_kcr[6] *= 2.0 * xH2
    out_kcr[7] *= 2.0 * xH2

    if T_K <= 0.0:
        for i in range(N_2BODY):
            out_k2[i] = 0.0
    else:
        for i in range(N_2BODY):
            out_k2[i] = K2_BASE[i] * (T_K ** K2_TEXP[i]) * nH_cm3

        t1_chx = A_KCHX * (300.0 / T_K) ** N_KCHX
        t2_chx = (
            C_KCHX[0] * np.exp(-TI_KCHX[0] / T_K)
            + C_KCHX[1] * np.exp(-TI_KCHX[1] / T_K)
            + C_KCHX[2] * np.exp(-TI_KCHX[2] / T_K)
            + C_KCHX[3] * np.exp(-TI_KCHX[3] / T_K)
        )
        out_k2[0] *= t1_chx + (T_K ** (-1.5)) * t2_chx

        out_k2[3] *= np.exp(-22.5 / T_K)
        out_k2[5] *= np.exp(-23.0 / T_K)
        out_k2[22] *= np.exp(-23.0 / T_K)

        out_k2[6] = 9.15e-10 * (0.62 + 45.41 / np.sqrt(T_K)) * nH_cm3
        out_k2[8] *= np.exp(0.108 / T_K)

        logT = np.log10(T_K)
        out_k2[9] *= 11.19 + (-1.676 + (-0.2852 + 0.04433 * logT) * logT) * logT

        out_k2[11] = _cii_rec_rate(T_K) * nH_cm3

        out_k2[13] *= np.exp(-T_K / 46600.0)

        out_k2[14] *= (315614.0 / T_K) ** 1.5 * (1.0 + (115188.0 / T_K) ** 0.407) ** (-2.242)

        out_k2[28] = 1.35e-9 * (0.62 + 0.4767 * 5.5 * np.sqrt(300.0 / T_K)) * nH_cm3

        if xe > 0.0:
            h2oplus_ratio = 6e-10 * xH2 / ((5.3e-6 / np.sqrt(T_K)) * xe)
            fac_h2 = h2oplus_ratio / (h2oplus_ratio + 1.0)
            fac_e = 1.0 / (h2oplus_ratio + 1.0)
            out_k2[1] *= fac_h2
            out_k2[27] *= fac_e
            out_k2[32] *= fac_h2
            out_k2[33] *= fac_e
        else:
            out_k2[1] = 0.0
            out_k2[27] = 0.0
            out_k2[32] = 0.0
            out_k2[33] = 0.0

        out_k2[30] *= (1.1e-11 * (T_K ** 0.517) + 4.0e-10 * (T_K ** 6.69e-3)) * np.exp(-227.0 / T_K)
        out_k2[31] *= 4.99e-11 * (T_K ** 0.405) + 7.5e-10 * (T_K ** (-0.458))

        out_k2[26] = 0.0

        if T_K > TEMP_COLL_K:
            logT4 = np.log10(T_K / 1.0e4)
            lnTe = np.log(T_K * 8.6173e-5)

            k9l = 6.67e-12 * np.sqrt(T_K) * np.exp(-(1.0 + 63590.0 / T_K))
            k9h = 3.52e-9 * np.exp(-43900.0 / T_K)
            k10l = (
                5.996e-30
                * (T_K ** 4.1881)
                / ((1.0 + 6.761e-6 * T_K) ** 5.6881)
                * np.exp(-54657.4 / T_K)
            )
            k10h = 1.3e-9 * np.exp(-53300.0 / T_K)

            ncrH = 10.0 ** (3.0 - 0.416 * logT4 - 0.327 * logT4 * logT4)
            ncrH2 = 10.0 ** (4.845 - 1.3 * logT4 + 1.62 * logT4 * logT4)

            denom = xH / ncrH + xH2 / ncrH2
            if denom > 0.0:
                ncr = 1.0 / denom
                n2ncr = nH_cm3 / ncr

                out_k2[15] = 10.0 ** (
                    np.log10(k9h) * n2ncr / (1.0 + n2ncr) + np.log10(k9l) / (1.0 + n2ncr)
                ) * nH_cm3
                out_k2[16] = 10.0 ** (
                    np.log10(k10h) * n2ncr / (1.0 + n2ncr) + np.log10(k10l) / (1.0 + n2ncr)
                ) * nH_cm3

                out_k2[17] *= np.exp(_poly_lnTe_HI_ion(lnTe))
            else:
                out_k2[15] = 0.0
                out_k2[16] = 0.0
                out_k2[17] = 0.0
        else:
            out_k2[15] = 0.0
            out_k2[16] = 0.0
            out_k2[17] = 0.0

    Gph = chi
    for i in range(N_PH):
        out_kph[i] = KPH_BASE[i] * Gph

    out_kph[IPH_CO] *= theta_co
    out_kph[IPH_H2] *= theta_h2
    out_kph[IPH_C] *= theta_c

    out_kgr[0] = 3.0e-17 * nH_cm3 * Zd * fH2gr

    if nH_cm3 <= 0.0 or T_K <= 0.0 or xe <= 0.0:
        for i in range(1, N_GR):
            out_kgr[i] = 0.0
        return

    psi_gr_fac = 1.7 * chi * np.sqrt(T_K) / nH_cm3
    psi = psi_gr_fac / xe

    logT = np.log(T_K)

    out_kgr[1] = _kgr_ion_rate(C_HP, psi, T_K, logT, nH_cm3, Zd, fHplusgr)
    out_kgr[2] = _kgr_ion_rate(C_CP, psi, T_K, logT, nH_cm3, Zd, fCplusgr)
    out_kgr[3] = _kgr_ion_rate(C_HEP, psi, T_K, logT, nH_cm3, Zd, fHeplusgr)
    out_kgr[4] = _kgr_ion_rate(C_SP, psi, T_K, logT, nH_cm3, Zd, fSplusgr)
    out_kgr[5] = _kgr_ion_rate(C_SIP, psi, T_K, logT, nH_cm3, Zd, fSiplusgr)


@njit(cache=True)
def gow17_rhs_cgs(
    y: np.ndarray,
    nH_cm3: float,
    T_K: float,
    chi: float,
    theta_h2: float,
    theta_co: float,
    theta_c: float,
    ion_rate_s: float,
    sigma_d_per_H_cm2: float,
    Tdust_K: float,
    chi_eff_pdr: float,
    Zg: float,
    Zd: float,
    fH2gr: float,
    fHplusgr: float,
    fCplusgr: float,
    fHeplusgr: float,
    fSplusgr: float,
    fSiplusgr: float,
    fCplusCR: float,
    out_rhs: np.ndarray,
) -> None:
    yghost = np.empty(N_Y + N_GHOST, dtype=np.float64)
    gow17_set_ghost_species(y, Zg, yghost)

    kcr = np.empty(N_CR, dtype=np.float64)
    k2 = np.empty(N_2BODY, dtype=np.float64)
    kph = np.empty(N_PH, dtype=np.float64)
    kgr = np.empty(N_GR, dtype=np.float64)

    gow17_compute_rates(
        yghost,
        nH_cm3=nH_cm3,
        T_K=T_K,
        chi=chi,
        theta_h2=theta_h2,
        theta_co=theta_co,
        theta_c=theta_c,
        ion_rate_s=ion_rate_s,
        Zd=Zd,
        fH2gr=fH2gr,
        fHplusgr=fHplusgr,
        fCplusgr=fCplusgr,
        fHeplusgr=fHeplusgr,
        fSplusgr=fSplusgr,
        fSiplusgr=fSiplusgr,
        fCplusCR=fCplusCR,
        out_kcr=kcr,
        out_k2=k2,
        out_kph=kph,
        out_kgr=kgr,
    )

    for i in range(N_Y):
        out_rhs[i] = 0.0

    for i in range(N_CR):
        rate = kcr[i] * yghost[IN_CR[i]]
        out_rhs[IN_CR[i]] -= rate
        out_rhs[OUT_CR[i]] += rate

    for i in range(N_2BODY):
        a = IN_2BODY1[i]
        b = IN_2BODY2[i]
        rate = k2[i] * yghost[a] * yghost[b]
        out_rhs[a] -= rate
        out_rhs[b] -= rate
        out_rhs[OUT_2BODY1[i]] += rate
        out_rhs[OUT_2BODY2[i]] += rate

    for i in range(N_PH):
        a = IN_PH[i]
        rate = kph[i] * yghost[a]
        out_rhs[a] -= rate
        out_rhs[OUT_PH1[i]] += rate

    for i in range(N_GR):
        a = IN_GR[i]
        rate = kgr[i] * yghost[a]
        out_rhs[a] -= rate
        out_rhs[OUT_GR[i]] += rate

    xco = y[I_CO]
    xco_ice = y[I_CO_ICE]

    k_fo = co_freezeout_rate_cgs(nH_cm3, T_K, sigma_d_per_H_cm2)
    k_td = co_thermal_desorption_rate_cgs(Tdust_K)

    k_pd_surf = co_photodesorption_surface_rate_cgs(chi_eff_pdr)

    nco_ice_cm3 = xco_ice * nH_cm3
    _, n_ice_act = co_active_ice_cgs(nH_cm3, sigma_d_per_H_cm2, nco_ice_cm3)
    R_pd_cm3s = co_photodesorption_R_cgs(k_pd_surf, n_ice_act)

    if nH_cm3 > 0.0:
        r_pd_per_H = R_pd_cm3s / nH_cm3
    else:
        r_pd_per_H = 0.0

    out_rhs[I_CO] += (-k_fo * xco + k_td * xco_ice + r_pd_per_H)
    out_rhs[I_CO_ICE] += (k_fo * xco - k_td * xco_ice - r_pd_per_H)


@njit(cache=True)
def _fd_jacobian(
    y: np.ndarray,
    f0: np.ndarray,
    mode: int,
    y_prev: np.ndarray,
    dt_s: float,
    nH_cm3: float,
    T_K: float,
    chi: float,
    theta_h2: float,
    theta_co: float,
    theta_c: float,
    ion_rate_s: float,
    sigma_d_per_H_cm2: float,
    Tdust_K: float,
    chi_eff_pdr: float,
    Zg: float,
    Zd: float,
    fH2gr: float,
    fHplusgr: float,
    fCplusgr: float,
    fHeplusgr: float,
    fSplusgr: float,
    fSiplusgr: float,
    fCplusCR: float,
    reltol: float,
    abstol: np.ndarray,
    J_out: np.ndarray,
) -> None:
    y_work = y.copy()
    f_work = np.empty(N_Y, dtype=np.float64)

    eps = np.finfo(np.float64).eps
    sqrt_eps = np.sqrt(eps)

    for j in range(N_Y):
        yj = y[j]

        scale = float(abstol[j]) + float(reltol) * abs(yj)
        h = sqrt_eps * max(scale, 0.0)
        if h == 0.0:
            h = sqrt_eps

        y_work[j] = yj + h

        gow17_rhs_cgs(
            y_work,
            nH_cm3=nH_cm3,
            T_K=T_K,
            chi=chi,
            theta_h2=theta_h2,
            theta_co=theta_co,
            theta_c=theta_c,
            ion_rate_s=ion_rate_s,
            sigma_d_per_H_cm2=sigma_d_per_H_cm2,
            Tdust_K=Tdust_K,
            chi_eff_pdr=chi_eff_pdr,
            Zg=Zg,
            Zd=Zd,
            fH2gr=fH2gr,
            fHplusgr=fHplusgr,
            fCplusgr=fCplusgr,
            fHeplusgr=fHeplusgr,
            fSplusgr=fSplusgr,
            fSiplusgr=fSiplusgr,
            fCplusCR=fCplusCR,
            out_rhs=f_work,
        )

        if mode == 0:
            for i in range(N_Y):
                f_work[i] = f_work[i]
        else:
            for i in range(N_Y):
                f_work[i] = y_work[i] - y_prev[i] - dt_s * f_work[i]

        for i in range(N_Y):
            J_out[i, j] = (f_work[i] - f0[i]) / h

        y_work[j] = yj


@njit(cache=True)
def newton_solve_fd(
    y0: np.ndarray,
    mode: int,
    y_prev: np.ndarray,
    dt_s: float,
    nH_cm3: float,
    T_K: float,
    chi: float,
    theta_h2: float,
    theta_co: float,
    theta_c: float,
    ion_rate_s: float,
    sigma_d_per_H_cm2: float,
    Tdust_K: float,
    chi_eff_pdr: float,
    Zg: float,
    Zd: float,
    fH2gr: float,
    fHplusgr: float,
    fCplusgr: float,
    fHeplusgr: float,
    fSplusgr: float,
    fSiplusgr: float,
    fCplusCR: float,
    max_iter: int,
    reltol: float,
    abstol: np.ndarray,
    out_y: np.ndarray,
) -> int:
    y = y0.copy()
    f = np.empty(N_Y, dtype=np.float64)
    J = np.empty((N_Y, N_Y), dtype=np.float64)
    dx = np.empty(N_Y, dtype=np.float64)

    for _ in range(max_iter):
        gow17_rhs_cgs(
            y,
            nH_cm3=nH_cm3,
            T_K=T_K,
            chi=chi,
            theta_h2=theta_h2,
            theta_co=theta_co,
            theta_c=theta_c,
            ion_rate_s=ion_rate_s,
            sigma_d_per_H_cm2=sigma_d_per_H_cm2,
            Tdust_K=Tdust_K,
            chi_eff_pdr=chi_eff_pdr,
            Zg=Zg,
            Zd=Zd,
            fH2gr=fH2gr,
            fHplusgr=fHplusgr,
            fCplusgr=fCplusgr,
            fHeplusgr=fHeplusgr,
            fSplusgr=fSplusgr,
            fSiplusgr=fSiplusgr,
            fCplusCR=fCplusCR,
            out_rhs=f,
        )

        if mode == 0:
            pass
        else:
            for i in range(N_Y):
                f[i] = y[i] - y_prev[i] - dt_s * f[i]

        max_scaled = 0.0
        for i in range(N_Y):
            denom = float(abstol[i]) + float(reltol) * abs(y[i])
            if denom <= 0.0:
                denom = 1.0
            si = abs(f[i]) / denom
            if si > max_scaled:
                max_scaled = si

        if max_scaled <= 1.0:
            for i in range(N_Y):
                out_y[i] = y[i]
            return 0

        _fd_jacobian(
            y,
            f,
            mode=mode,
            y_prev=y_prev,
            dt_s=dt_s,
            nH_cm3=nH_cm3,
            T_K=T_K,
            chi=chi,
            theta_h2=theta_h2,
            theta_co=theta_co,
            theta_c=theta_c,
            ion_rate_s=ion_rate_s,
            sigma_d_per_H_cm2=sigma_d_per_H_cm2,
            Tdust_K=Tdust_K,
            chi_eff_pdr=chi_eff_pdr,
            Zg=Zg,
            Zd=Zd,
            fH2gr=fH2gr,
            fHplusgr=fHplusgr,
            fCplusgr=fCplusgr,
            fHeplusgr=fHeplusgr,
            fSplusgr=fSplusgr,
            fSiplusgr=fSiplusgr,
            fCplusCR=fCplusCR,
            reltol=reltol,
            abstol=abstol,
            J_out=J,
        )

        rhs = -f
        st = _solve_linear_system_gauss(J, rhs, dx)
        if st != 0:
            return 2

        for i in range(N_Y):
            y[i] = y[i] + dx[i]
            if y[i] < 0.0:
                y[i] = 0.0

    for i in range(N_Y):
        out_y[i] = y[i]
    return 1


@njit(parallel=True, cache=True)
def solve_gow17_equilibrium_cells_cgs(
    nH_cm3: np.ndarray,
    T_K: np.ndarray,
    Tdust_K: np.ndarray,
    chi: np.ndarray,
    theta_h2: np.ndarray,
    theta_co: np.ndarray,
    theta_c: np.ndarray,
    chi_eff_pdr: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
    y0: np.ndarray,
    Zg: float,
    Zd: float,
    ion_rate_s: float,
    fH2gr: float,
    fHplusgr: float,
    fCplusgr: float,
    fHeplusgr: float,
    fSplusgr: float,
    fSiplusgr: float,
    fCplusCR: float,
    max_iter: int,
    reltol: float,
    abstol: np.ndarray,
    y_out: np.ndarray,
    status_out: np.ndarray,
) -> None:
    ncells = nH_cm3.size

    for i in prange(ncells):
        y_init = np.empty(N_Y, dtype=np.float64)
        for j in range(N_Y):
            y_init[j] = y0[j]

        y_prev = y_init
        y_sol = np.empty(N_Y, dtype=np.float64)

        st = newton_solve_fd(
            y_init,
            mode=0,
            y_prev=y_prev,
            dt_s=0.0,
            nH_cm3=float(nH_cm3[i]),
            T_K=float(T_K[i]),
            chi=float(chi[i]),
            theta_h2=float(theta_h2[i]),
            theta_co=float(theta_co[i]),
            theta_c=float(theta_c[i]),
            ion_rate_s=float(ion_rate_s),
            sigma_d_per_H_cm2=float(sigma_d_per_H_cm2[i]),
            Tdust_K=float(Tdust_K[i]),
            chi_eff_pdr=float(chi_eff_pdr[i]),
            Zg=float(Zg),
            Zd=float(Zd),
            fH2gr=float(fH2gr),
            fHplusgr=float(fHplusgr),
            fCplusgr=float(fCplusgr),
            fHeplusgr=float(fHeplusgr),
            fSplusgr=float(fSplusgr),
            fSiplusgr=float(fSiplusgr),
            fCplusCR=float(fCplusCR),
            max_iter=int(max_iter),
            reltol=float(reltol),
            abstol=abstol,
            out_y=y_sol,
        )

        status_out[i] = st
        for j in range(N_Y):
            y_out[i, j] = y_sol[j]


@njit(parallel=True, cache=True)
def evolve_gow17_be_cells_cgs(
    nH_cm3: np.ndarray,
    T_K: np.ndarray,
    Tdust_K: np.ndarray,
    chi: np.ndarray,
    theta_h2: np.ndarray,
    theta_co: np.ndarray,
    theta_c: np.ndarray,
    chi_eff_pdr: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
    dt_s: float,
    y_inout: np.ndarray,
    Zg: float,
    Zd: float,
    ion_rate_s: float,
    fH2gr: float,
    fHplusgr: float,
    fCplusgr: float,
    fHeplusgr: float,
    fSplusgr: float,
    fSiplusgr: float,
    fCplusCR: float,
    max_iter: int,
    reltol: float,
    abstol: np.ndarray,
    status_out: np.ndarray,
) -> None:
    ncells = nH_cm3.size

    for i in prange(ncells):
        y_prev = np.empty(N_Y, dtype=np.float64)
        y_init = np.empty(N_Y, dtype=np.float64)
        for j in range(N_Y):
            y_prev[j] = y_inout[i, j]
            y_init[j] = y_inout[i, j]

        y_sol = np.empty(N_Y, dtype=np.float64)

        st = newton_solve_fd(
            y_init,
            mode=1,
            y_prev=y_prev,
            dt_s=float(dt_s),
            nH_cm3=float(nH_cm3[i]),
            T_K=float(T_K[i]),
            chi=float(chi[i]),
            theta_h2=float(theta_h2[i]),
            theta_co=float(theta_co[i]),
            theta_c=float(theta_c[i]),
            ion_rate_s=float(ion_rate_s),
            sigma_d_per_H_cm2=float(sigma_d_per_H_cm2[i]),
            Tdust_K=float(Tdust_K[i]),
            chi_eff_pdr=float(chi_eff_pdr[i]),
            Zg=float(Zg),
            Zd=float(Zd),
            fH2gr=float(fH2gr),
            fHplusgr=float(fHplusgr),
            fCplusgr=float(fCplusgr),
            fHeplusgr=float(fHeplusgr),
            fSplusgr=float(fSplusgr),
            fSiplusgr=float(fSiplusgr),
            fCplusCR=float(fCplusCR),
            max_iter=int(max_iter),
            reltol=float(reltol),
            abstol=abstol,
            out_y=y_sol,
        )

        status_out[i] = st
        for j in range(N_Y):
            y_inout[i, j] = y_sol[j]
