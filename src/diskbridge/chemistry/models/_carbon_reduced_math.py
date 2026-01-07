from __future__ import annotations

from typing import Optional, Tuple

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from diskbridge._units import Quantity
from diskbridge._constants import (
    K_B, M_H, K0_CO, E_BIND_CO, NU0_CO, F_DRAINE, N_LAY, N_SURF, Y_CO
)
from diskbridge._constants import (
    X_C_TOT, GAMMA_C0, ALPHA_REC_C0, T_REC_EXP,
)
from diskbridge.chemistry.processes.co_phase import (
    M_CO_CGS as _M_CO_CGS,
    co_freezeout_rate_cgs,
    co_thermal_desorption_rate_cgs,
    co_photodesorption_surface_rate_cgs,
    co_active_ice_cgs,
    co_photodesorption_R_cgs,
    co_freezeout_rate_field_cgs,
    co_thermal_desorption_rate_field_cgs,
    co_photodesorption_surface_rate_field_cgs,
    co_active_ice_field_cgs,
    co_photodesorption_R_field_cgs,
)

# constant below should probably be in _constants.py 
M_CO_CGS = _M_CO_CGS


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


@njit(inline='always')
def _apply_co_budget_clamp(g: float, i_ice: float, nco_max: float) -> tuple[float, float]:
    total = g + i_ice
    excess = total - nco_max
    if excess > 0.0:
        reduce_ice = i_ice
        if reduce_ice > excess:
            reduce_ice = excess
        i_ice = i_ice - reduce_ice

        excess2 = excess - reduce_ice
        if excess2 > 0.0:
            g = g - excess2
            if g < 0.0:
                g = 0.0
    return g, i_ice


@njit(inline='always')
def _nl97_eval_F_thin(
    g_in: float,
    nco_max: float,
    a: float,
    k_pd_eff: float,
    k0_nl97: float,
    beta: float,
    nH_i: float,
    chi_i: float,
    Tg_i: float,
    nH2_i: float,
) -> tuple[float, float]:
    g = g_in
    i_ice = a * g
    if i_ice < 0.0:
        i_ice = 0.0

    g, i_ice = _apply_co_budget_clamp(g, i_ice, nco_max)
    nco_total = g + i_ice
    nCplus, _, _ = carbon_closure_cell_cgs(nH_i, chi_i, Tg_i, nco_total)
    R_form = k0_nl97 * beta * nCplus * nH2_i
    return (k_pd_eff * g - R_form), i_ice


@njit(inline='always')
def _nl97_eval_F_thick(
    g_in: float,
    nco_max: float,
    k_fo: float,
    k_td: float,
    R0: float,
    k_pd_eff: float,
    k0_nl97: float,
    beta: float,
    nH_i: float,
    chi_i: float,
    Tg_i: float,
    nH2_i: float,
) -> tuple[float, float]:
    g = g_in
    i_ice = 0.0
    if k_td > 0.0:
        i_ice = (k_fo * g - R0) / k_td
    if i_ice < 0.0:
        i_ice = 0.0

    g, i_ice = _apply_co_budget_clamp(g, i_ice, nco_max)
    nco_total = g + i_ice
    nCplus, _, _ = carbon_closure_cell_cgs(nH_i, chi_i, Tg_i, nco_total)
    R_form = k0_nl97 * beta * nCplus * nH2_i
    return (k_pd_eff * g - R_form), i_ice


@njit(parallel=True, cache=True)
def solve_carbon_reduced_steady_state_cgs(
    nH_cm3: np.ndarray,
    T_K: np.ndarray,
    chi: np.ndarray,
    theta_co: np.ndarray,
    tau_form_s: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
    nH2_cm3: np.ndarray,
    Tg_K: np.ndarray,
    formation_model: int,
    k0_nl97: float,
    k1_nl97: float,
    xO: float,
    gamma_chx0: float,
    Xco_tot: float,
    min_rate: float,
    out_Xco_gas: np.ndarray,
    out_nco_gas: np.ndarray,
    out_nco_ice: np.ndarray,
    out_k_pd: np.ndarray,
    out_tau_pd: np.ndarray,
    out_n_ice_act_max: np.ndarray,
    out_n_ice_act: np.ndarray,
    out_k_pd_surf: np.ndarray,
    out_R_pd: np.ndarray,
) -> None:
    ncells = nH_cm3.size
    for i in prange(ncells):
        nH_i = nH_cm3[i]
        T_i = T_K[i]
        chi_i = chi[i]

        theta_i = theta_co[i]
        if theta_i < 0.0:
            theta_i = 0.0
        elif theta_i > 1.0:
            theta_i = 1.0

        tau_form_i = tau_form_s[i]
        sigma_i = sigma_d_per_H_cm2[i]

        nH2_i = nH2_cm3[i]
        Tg_i = Tg_K[i]

        nco_max = Xco_tot * nH_i

        k_pd = K0_CO * chi_i * theta_i
        out_k_pd[i] = k_pd
        if k_pd <= 0.0:
            out_tau_pd[i] = np.inf
        else:
            k_pd_safe = k_pd
            if min_rate > 0.0 and k_pd_safe < min_rate:
                k_pd_safe = min_rate
            out_tau_pd[i] = 1.0 / k_pd_safe

        k_fo = co_freezeout_rate_cgs(nH_i, T_i, sigma_i)
        k_td = co_thermal_desorption_rate_cgs(T_i)
        k_pd_surf = co_photodesorption_surface_rate_cgs(chi_i)
        out_k_pd_surf[i] = k_pd_surf

        n_ice_act_max, _ = co_active_ice_cgs(nH_i, sigma_i, 0.0)
        out_n_ice_act_max[i] = n_ice_act_max

        # Thin-ice regime
        denom_thin = k_td + k_pd_surf
        a = 0.0
        if denom_thin > 0.0:
            a = k_fo / denom_thin

        if int(formation_model) == 0:
            # Timescale (tau) formation law
            denom_g_thin = 1.0 + tau_form_i * k_pd + a
            g_thin = 0.0
            if denom_g_thin > 0.0:
                g_thin = nco_max / denom_g_thin
            if g_thin < 0.0:
                g_thin = 0.0
            elif g_thin > nco_max:
                g_thin = nco_max

            i_thin = a * g_thin
            if i_thin < 0.0:
                i_thin = 0.0

            # Thick-ice regime
            R0 = k_pd_surf * n_ice_act_max
            denom_g_thick = 1.0 + tau_form_i * k_pd
            if k_td > 0.0:
                denom_g_thick = denom_g_thick + (k_fo / k_td)
            numer_g_thick = nco_max
            if k_td > 0.0:
                numer_g_thick = numer_g_thick + (R0 / k_td)

            g_thick = 0.0
            if denom_g_thick > 0.0:
                g_thick = numer_g_thick / denom_g_thick
            if g_thick < 0.0:
                g_thick = 0.0
            elif g_thick > nco_max:
                g_thick = nco_max

            i_thick = 0.0
            if k_td > 0.0:
                i_thick = (k_fo * g_thick - R0) / k_td
            if i_thick < 0.0:
                i_thick = 0.0
        else:
            # NL97 formation law
            k_pd_eff = k_pd
            if min_rate > 0.0 and k_pd_eff < min_rate:
                k_pd_eff = min_rate

            # Precompute constants for formation
            Gamma_CHx = gamma_chx0 * chi_i
            beta = 0.0
            denom_beta = (k1_nl97 * xO * nH_i) + Gamma_CHx
            if denom_beta > 0.0:
                beta = (k1_nl97 * xO * nH_i) / denom_beta

            # Fixed-iteration bisection per regime
            n_bisect = 35

            # Solve thin
            lo = 0.0
            hi = nco_max
            f_lo, _ = _nl97_eval_F_thin(
                g_in=lo,
                nco_max=nco_max,
                a=a,
                k_pd_eff=k_pd_eff,
                k0_nl97=k0_nl97,
                beta=beta,
                nH_i=nH_i,
                chi_i=chi_i,
                Tg_i=Tg_i,
                nH2_i=nH2_i,
            )
            f_hi, _ = _nl97_eval_F_thin(
                g_in=hi,
                nco_max=nco_max,
                a=a,
                k_pd_eff=k_pd_eff,
                k0_nl97=k0_nl97,
                beta=beta,
                nH_i=nH_i,
                chi_i=chi_i,
                Tg_i=Tg_i,
                nH2_i=nH2_i,
            )
            if f_hi < 0.0:
                g_thin = nco_max
                _, i_thin = _nl97_eval_F_thin(
                    g_in=g_thin,
                    nco_max=nco_max,
                    a=a,
                    k_pd_eff=k_pd_eff,
                    k0_nl97=k0_nl97,
                    beta=beta,
                    nH_i=nH_i,
                    chi_i=chi_i,
                    Tg_i=Tg_i,
                    nH2_i=nH2_i,
                )
            elif f_lo > 0.0:
                g_thin = 0.0
                _, i_thin = _nl97_eval_F_thin(
                    g_in=g_thin,
                    nco_max=nco_max,
                    a=a,
                    k_pd_eff=k_pd_eff,
                    k0_nl97=k0_nl97,
                    beta=beta,
                    nH_i=nH_i,
                    chi_i=chi_i,
                    Tg_i=Tg_i,
                    nH2_i=nH2_i,
                )
            else:
                for _ in range(n_bisect):
                    mid = 0.5 * (lo + hi)
                    f_mid, _ = _nl97_eval_F_thin(
                        g_in=mid,
                        nco_max=nco_max,
                        a=a,
                        k_pd_eff=k_pd_eff,
                        k0_nl97=k0_nl97,
                        beta=beta,
                        nH_i=nH_i,
                        chi_i=chi_i,
                        Tg_i=Tg_i,
                        nH2_i=nH2_i,
                    )
                    if f_mid <= 0.0:
                        lo = mid
                        f_lo = f_mid
                    else:
                        hi = mid
                        f_hi = f_mid
                g_thin = 0.5 * (lo + hi)
                _, i_thin = _nl97_eval_F_thin(
                    g_in=g_thin,
                    nco_max=nco_max,
                    a=a,
                    k_pd_eff=k_pd_eff,
                    k0_nl97=k0_nl97,
                    beta=beta,
                    nH_i=nH_i,
                    chi_i=chi_i,
                    Tg_i=Tg_i,
                    nH2_i=nH2_i,
                )

            # Solve thick
            if k_td <= 0.0:
                g_thick = g_thin
                i_thick = i_thin
            else:
                R0 = k_pd_surf * n_ice_act_max
                lo = 0.0
                hi = nco_max
                f_lo, _ = _nl97_eval_F_thick(
                    g_in=lo,
                    nco_max=nco_max,
                    k_fo=k_fo,
                    k_td=k_td,
                    R0=R0,
                    k_pd_eff=k_pd_eff,
                    k0_nl97=k0_nl97,
                    beta=beta,
                    nH_i=nH_i,
                    chi_i=chi_i,
                    Tg_i=Tg_i,
                    nH2_i=nH2_i,
                )
                f_hi, _ = _nl97_eval_F_thick(
                    g_in=hi,
                    nco_max=nco_max,
                    k_fo=k_fo,
                    k_td=k_td,
                    R0=R0,
                    k_pd_eff=k_pd_eff,
                    k0_nl97=k0_nl97,
                    beta=beta,
                    nH_i=nH_i,
                    chi_i=chi_i,
                    Tg_i=Tg_i,
                    nH2_i=nH2_i,
                )
                if f_hi < 0.0:
                    g_thick = nco_max
                    _, i_thick = _nl97_eval_F_thick(
                        g_in=g_thick,
                        nco_max=nco_max,
                        k_fo=k_fo,
                        k_td=k_td,
                        R0=R0,
                        k_pd_eff=k_pd_eff,
                        k0_nl97=k0_nl97,
                        beta=beta,
                        nH_i=nH_i,
                        chi_i=chi_i,
                        Tg_i=Tg_i,
                        nH2_i=nH2_i,
                    )
                elif f_lo > 0.0:
                    g_thick = 0.0
                    _, i_thick = _nl97_eval_F_thick(
                        g_in=g_thick,
                        nco_max=nco_max,
                        k_fo=k_fo,
                        k_td=k_td,
                        R0=R0,
                        k_pd_eff=k_pd_eff,
                        k0_nl97=k0_nl97,
                        beta=beta,
                        nH_i=nH_i,
                        chi_i=chi_i,
                        Tg_i=Tg_i,
                        nH2_i=nH2_i,
                    )
                else:
                    for _ in range(n_bisect):
                        mid = 0.5 * (lo + hi)
                        f_mid, _ = _nl97_eval_F_thick(
                            g_in=mid,
                            nco_max=nco_max,
                            k_fo=k_fo,
                            k_td=k_td,
                            R0=R0,
                            k_pd_eff=k_pd_eff,
                            k0_nl97=k0_nl97,
                            beta=beta,
                            nH_i=nH_i,
                            chi_i=chi_i,
                            Tg_i=Tg_i,
                            nH2_i=nH2_i,
                        )
                        if f_mid <= 0.0:
                            lo = mid
                            f_lo = f_mid
                        else:
                            hi = mid
                            f_hi = f_mid
                    g_thick = 0.5 * (lo + hi)
                    _, i_thick = _nl97_eval_F_thick(
                        g_in=g_thick,
                        nco_max=nco_max,
                        k_fo=k_fo,
                        k_td=k_td,
                        R0=R0,
                        k_pd_eff=k_pd_eff,
                        k0_nl97=k0_nl97,
                        beta=beta,
                        nH_i=nH_i,
                        chi_i=chi_i,
                        Tg_i=Tg_i,
                        nH2_i=nH2_i,
                    )

        use_thin = i_thin < n_ice_act_max
        nco_gas = g_thin if use_thin else g_thick
        nco_ice = i_thin if use_thin else i_thick

        # Clamp total to nco_max (reduce ice first, then gas)
        total = nco_gas + nco_ice
        excess = total - nco_max
        if excess > 0.0:
            reduce_ice = nco_ice
            if reduce_ice > excess:
                reduce_ice = excess
            nco_ice = nco_ice - reduce_ice
            excess2 = excess - reduce_ice
            if excess2 > 0.0:
                nco_gas = nco_gas - excess2
                if nco_gas < 0.0:
                    nco_gas = 0.0

        out_nco_gas[i] = nco_gas
        out_nco_ice[i] = nco_ice

        _, n_ice_act = co_active_ice_cgs(nH_i, sigma_i, nco_ice)
        out_n_ice_act[i] = n_ice_act
        out_R_pd[i] = co_photodesorption_R_cgs(k_pd_surf, n_ice_act)

        if nH_i > 0.0:
            out_Xco_gas[i] = nco_gas / nH_i
        else:
            out_Xco_gas[i] = 0.0


@dataclass(frozen=True)
class CarbonReducedRates:
    k_pd: Quantity
    k_fo: Quantity
    k_td: Quantity
    k_pd_surf: Quantity
    tau_pd: Optional[Quantity]


def compute_co_photodissociation_rate(
    chi: Quantity,
    theta_co: Optional[Quantity],
    k0_co: Quantity,
    *,
    min_rate: float = 0.0,
) -> Tuple[Quantity, Optional[Quantity]]:
    chi_arr = chi.to('dimensionless').magnitude

    if theta_co is None:
        theta_arr = np.ones_like(chi_arr, dtype=float)
    else:
        theta_arr = theta_co.to('dimensionless').magnitude

    theta_arr = np.clip(theta_arr, 0.0, 1.0)

    k0_val = k0_co.to('1/s').magnitude

    k_diss_arr = k0_val * chi_arr * theta_arr
    k_diss_co = Quantity(k_diss_arr, '1/s')

    tau_diss_co = None
    if min_rate is not None:
        rate_floor = float(min_rate)
        if rate_floor > 0.0:
            k_safe = np.maximum(k_diss_arr, rate_floor)
            tau_arr = 1.0 / k_safe
            tau_diss_co = Quantity(tau_arr, 's')

    return k_diss_co, tau_diss_co


def co_freezeout_rate(T: Quantity, nH: Quantity, sigma_d_per_H: Quantity) -> Quantity:
    """CO freeze-out rate using mesh sigma_d_per_H."""
    T_K = np.asarray(T.to('K').magnitude, dtype=float)
    nH_cm3 = np.asarray(nH.to('cm^-3').magnitude, dtype=float)
    sigma = np.asarray(sigma_d_per_H.to('cm^2').magnitude, dtype=float)
    k_fo = co_freezeout_rate_field_cgs(nH_cm3, T_K, sigma)
    return Quantity(k_fo, '1/s')


def co_thermal_desorption_rate(T_d: Quantity) -> Quantity:
    """CO thermal desorption rate using E_BIND_CO and NU0_CO from config."""
    T_K = np.asarray(T_d.to('K').magnitude, dtype=float)
    k_td = co_thermal_desorption_rate_field_cgs(T_K)
    return Quantity(k_td, '1/s')


def co_photodesorption_rate_surface(chi: Quantity) -> Quantity:
    chi_val = np.asarray(chi.to('dimensionless').magnitude, dtype=float)
    k_pd_surf = co_photodesorption_surface_rate_field_cgs(chi_val)
    return Quantity(k_pd_surf, '1/s')


def co_photodesorption_sink(
    *,
    nH: Quantity,
    nco_ice: Quantity,
    sigma_d_per_H: Quantity,
    k_pd_surf: Quantity,
) -> tuple[Quantity, Quantity, Quantity, Quantity]:
    nH_cm3 = np.asarray(nH.to('cm^-3').magnitude, dtype=float)
    nco_ice_cm3 = np.asarray(nco_ice.to('cm^-3').magnitude, dtype=float)
    sigma = np.asarray(sigma_d_per_H.to('cm^2').magnitude, dtype=float)

    n_ice_act_max, n_ice_act = co_active_ice_field_cgs(nH_cm3, sigma, nco_ice_cm3)

    k_pd_s = np.asarray(k_pd_surf.to('1/s').magnitude, dtype=float)
    if np.ndim(k_pd_s) == 0:
        k_pd_s = np.broadcast_to(k_pd_s, n_ice_act.shape)

    R_pd = co_photodesorption_R_field_cgs(k_pd_s, n_ice_act)

    return (
        Quantity(n_ice_act_max, 'cm^-3'),
        Quantity(n_ice_act, 'cm^-3'),
        k_pd_surf,
        Quantity(R_pd, 'cm^-3/s'),
    )


def compute_carbon_reduced_rates(
    *,
    nH: Quantity,
    T: Quantity,
    chi: Quantity,
    theta_co: Optional[Quantity],
    sigma_d_per_H: Quantity,
    min_rate: float = 0.0,
) -> CarbonReducedRates:
    """Compute all CO two-phase chemistry rates using constants from config."""
    k_pd, tau_pd = compute_co_photodissociation_rate(
        chi=chi,
        theta_co=theta_co,
        k0_co=Quantity(K0_CO, '1/s'),
        min_rate=min_rate,
    )
    k_fo = co_freezeout_rate(T, nH, sigma_d_per_H)
    k_td = co_thermal_desorption_rate(T)
    k_pd_surf = co_photodesorption_rate_surface(chi)
    return CarbonReducedRates(k_pd=k_pd, k_fo=k_fo, k_td=k_td, k_pd_surf=k_pd_surf, tau_pd=tau_pd)


def solve_carbon_reduced_steady_state(
    *,
    nH: Quantity,
    T: Quantity,
    chi: Quantity,
    theta_co,
    Xco_tot: float,
    tau_form: Quantity,
    sigma_d_per_H: Quantity,
    min_rate: float = 0.0,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Optional[Quantity], dict]:
    """Solve steady-state CO chemistry using an analytic reduced model."""
    orig_shape = nH.to('cm^-3').magnitude.shape
    ncells = nH.to('cm^-3').magnitude.size

    nH_flat = np.ascontiguousarray(nH.to('cm^-3').magnitude.reshape(ncells), dtype=np.float64)
    T_flat = np.ascontiguousarray(T.to('K').magnitude.reshape(ncells), dtype=np.float64)
    chi_flat = np.ascontiguousarray(chi.to('dimensionless').magnitude.reshape(ncells), dtype=np.float64)
    if theta_co is None:
        theta_flat = np.ones(ncells, dtype=np.float64)
    else:
        if isinstance(theta_co, Quantity):
            theta_arr = theta_co.to('dimensionless').magnitude
        else:
            theta_arr = np.asarray(theta_co)
        theta_flat = np.ascontiguousarray(theta_arr.reshape(ncells), dtype=np.float64)
    tau_form_flat = np.ascontiguousarray(tau_form.to('s').magnitude.reshape(ncells), dtype=np.float64)
    sigma_flat = np.ascontiguousarray(sigma_d_per_H.to('cm^2').magnitude.reshape(ncells), dtype=np.float64)

    out_Xco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_ice = np.empty(ncells, dtype=np.float64)
    out_k_pd = np.empty(ncells, dtype=np.float64)
    out_tau_pd = np.empty(ncells, dtype=np.float64)
    out_n_ice_act_max = np.empty(ncells, dtype=np.float64)
    out_n_ice_act = np.empty(ncells, dtype=np.float64)
    out_k_pd_surf = np.empty(ncells, dtype=np.float64)
    out_R_pd = np.empty(ncells, dtype=np.float64)

    nH2_flat = np.zeros_like(nH_flat, dtype=np.float64)
    Tg_flat = np.ascontiguousarray(T_flat, dtype=np.float64)

    solve_carbon_reduced_steady_state_cgs(
        nH_flat,
        T_flat,
        chi_flat,
        theta_flat,
        tau_form_flat,
        sigma_flat,
        nH2_flat,
        Tg_flat,
        int(0),
        float(0.0),
        float(0.0),
        float(0.0),
        float(0.0),
        float(Xco_tot),
        float(min_rate),
        out_Xco_gas,
        out_nco_gas,
        out_nco_ice,
        out_k_pd,
        out_tau_pd,
        out_n_ice_act_max,
        out_n_ice_act,
        out_k_pd_surf,
        out_R_pd,
    )

    X_co = Quantity(out_Xco_gas.reshape(orig_shape), 'dimensionless')
    nco_gas_q = Quantity(out_nco_gas.reshape(orig_shape), 'cm^-3')
    nco_ice_q = Quantity(out_nco_ice.reshape(orig_shape), 'cm^-3')
    k_pd = Quantity(out_k_pd.reshape(orig_shape), '1/s')
    tau_pd = Quantity(out_tau_pd.reshape(orig_shape), 's')

    diag = {
        'n_ice_act_max': Quantity(out_n_ice_act_max.reshape(orig_shape), 'cm^-3'),
        'n_ice_act': Quantity(out_n_ice_act.reshape(orig_shape), 'cm^-3'),
        'k_pd_surf': Quantity(out_k_pd_surf.reshape(orig_shape), '1/s'),
        'R_pd': Quantity(out_R_pd.reshape(orig_shape), 'cm^-3/s'),
    }

    return X_co, nco_gas_q, nco_ice_q, k_pd, tau_pd, diag


def evolve_carbon_reduced_time_dependent(
    *,
    nH: Quantity,
    T: Quantity,
    chi: Quantity,
    theta_co,
    t_end: Quantity,
    Xco_tot: float,
    tau_form: Quantity,
    sigma_d_per_H: Quantity,
    Xco_gas_init: Optional[float] = None,
    Xco_ice_init: Optional[float] = None,
    dt: Optional[Quantity] = None,
    min_rate: float = 0.0,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Optional[Quantity], dict]:
    """Evolve CO two-phase chemistry using constants from config."""
    rates = compute_carbon_reduced_rates(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        sigma_d_per_H=sigma_d_per_H,
        min_rate=min_rate,
    )

    k_pd = rates.k_pd
    tau_pd = rates.tau_pd
    k_fo = rates.k_fo
    k_td = rates.k_td
    k_pd_surf = rates.k_pd_surf

    k_fo_s = k_fo.to('1/s').magnitude
    k_td_s = k_td.to('1/s').magnitude
    k_pd_s = k_pd.to('1/s').magnitude
    tau_s = tau_form.to('s').magnitude
    t_end_s = t_end.to('s').magnitude

    nH_cm3 = nH.to('cm^-3').magnitude
    nco_max = float(Xco_tot) * nH_cm3

    if np.ndim(t_end_s) != 0:
        if np.shape(t_end_s) != np.shape(nH_cm3):
            raise ValueError(
                f"t_end shape {np.shape(t_end_s)} must match nH shape {np.shape(nH_cm3)} "
                "when providing a per-cell t_end"
            )
    if dt is not None and np.ndim(t_end_s) != 0:
        raise ValueError("dt-based evolution requires scalar t_end")

    if np.ndim(t_end_s) == 0:
        t_end_s_arr = np.full_like(nH_cm3, float(t_end_s), dtype=float)
    else:
        t_end_s_arr = np.asarray(t_end_s, dtype=float)

    if Xco_gas_init is None:
        nco_gas = nco_max.copy()
    else:
        if isinstance(Xco_gas_init, Quantity):
            X_init = Xco_gas_init.to('dimensionless').magnitude
        else:
            X_init = Xco_gas_init
        X_init = np.asarray(X_init, dtype=float)
        if np.ndim(X_init) != 0 and np.shape(X_init) != np.shape(nH_cm3):
            raise ValueError(
                f"Xco_gas_init shape {np.shape(X_init)} must match nH shape {np.shape(nH_cm3)} "
                "when providing a per-cell initial condition"
            )
        nco_gas = X_init * nH_cm3

    if Xco_ice_init is None:
        nco_ice = np.zeros_like(nco_gas)
    else:
        if isinstance(Xco_ice_init, Quantity):
            X_init_ice = Xco_ice_init.to('dimensionless').magnitude
        else:
            X_init_ice = Xco_ice_init
        X_init_ice = np.asarray(X_init_ice, dtype=float)
        if np.ndim(X_init_ice) != 0 and np.shape(X_init_ice) != np.shape(nH_cm3):
            raise ValueError(
                f"Xco_ice_init shape {np.shape(X_init_ice)} must match nH shape {np.shape(nH_cm3)} "
                "when providing a per-cell initial condition"
            )
        nco_ice = X_init_ice * nH_cm3

    if dt is None:
        raise ValueError("dt is required for time-dependent carbon_reduced evolution")

    dt_s = float(dt.to('s').magnitude)
    if dt_s <= 0.0:
        raise ValueError("dt must be > 0")

    t = 0.0
    n_steps = 0
    n_steps_total = int(np.ceil(float(t_end_s) / float(dt_s)))

    n_ice_act_max_last = None
    n_ice_act_last = None
    R_pd_last = None

    while t < float(t_end_s) and n_steps < n_steps_total:
        dt_step = min(float(dt_s), float(t_end_s) - t)

        def _rhs(nco_g, nco_i):
            R_form = (nco_max - nco_g - nco_i) / tau_s

            nco_ice_q = Quantity(nco_i, 'cm^-3')
            n_ice_act_max_q, n_ice_act_q, _, R_pd_q = co_photodesorption_sink(
                nH=nH,
                nco_ice=nco_ice_q,
                sigma_d_per_H=sigma_d_per_H,
                k_pd_surf=k_pd_surf,
            )
            R_pd = R_pd_q.to('cm^-3/s').magnitude

            dnco_gas_dt = (
                R_form
                + k_td_s * nco_i
                + R_pd
                - k_fo_s * nco_g
                - k_pd_s * nco_g
            )
            dnco_ice_dt = k_fo_s * nco_g - k_td_s * nco_i - R_pd
            return dnco_gas_dt, dnco_ice_dt, n_ice_act_max_q, n_ice_act_q, R_pd_q

        k1g, k1i, n_ice_act_max_q, n_ice_act_q, R_pd_q = _rhs(nco_gas, nco_ice)
        g_mid = np.maximum(nco_gas + 0.5 * dt_step * k1g, 0.0)
        i_mid = np.maximum(nco_ice + 0.5 * dt_step * k1i, 0.0)
        k2g, k2i, n_ice_act_max_q, n_ice_act_q, R_pd_q = _rhs(g_mid, i_mid)
        nco_gas = np.maximum(nco_gas + dt_step * k2g, 0.0)
        nco_ice = np.maximum(nco_ice + dt_step * k2i, 0.0)

        n_ice_act_max_last = n_ice_act_max_q
        n_ice_act_last = n_ice_act_q
        R_pd_last = R_pd_q

        t += dt_step
        n_steps += 1

    nco_gas_q = Quantity(nco_gas, 'cm^-3')
    nco_ice_q = Quantity(nco_ice, 'cm^-3')
    X_co = Quantity(np.where(nH_cm3 > 0.0, nco_gas / nH_cm3, 0.0), 'dimensionless')

    diag = {
        'n_ice_act_max': n_ice_act_max_last,
        'n_ice_act': n_ice_act_last,
        'k_pd_surf': k_pd_surf,
        'R_pd': R_pd_last,
    }

    return X_co, nco_gas_q, nco_ice_q, k_pd, tau_pd, diag
