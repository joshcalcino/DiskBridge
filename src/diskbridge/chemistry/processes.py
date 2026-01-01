from __future__ import annotations

from typing import Optional, Tuple

from dataclasses import dataclass

import numpy as np

from diskbridge._units import Quantity, units
from diskbridge._constants import (
    K_B, M_H, K0_CO, SIGMA_D_PER_H, E_BIND_CO, NU0_CO, ALPHA_PD_ICE
)

M_CO_CGS = 28.0 * M_H


@dataclass(frozen=True)
class CoTwoPhaseRates:
    k_pd: Quantity
    k_fo: Quantity
    k_td: Quantity
    k_pd_ice: Quantity
    tau_pd: Optional[Quantity]


def compute_co_photodissociation_rate(
    chi,
    theta_co,
    k0_co,
    *,
    candidate_mask=None,
    min_rate: float = 1.0e-30,
) -> Tuple[Quantity, Optional[Quantity]]:
    if isinstance(chi, Quantity):
        chi_arr = chi.to('dimensionless').magnitude
    else:
        chi_arr = np.asarray(chi, dtype=float)

    if theta_co is None:
        theta_arr = np.ones_like(chi_arr, dtype=float)
    elif isinstance(theta_co, Quantity):
        theta_arr = theta_co.to('dimensionless').magnitude
    else:
        theta_arr = np.asarray(theta_co, dtype=float)

    if candidate_mask is not None:
        mask = np.asarray(candidate_mask, dtype=bool)
        if mask.shape != theta_arr.shape:
            raise ValueError(
                f'candidate_mask shape {mask.shape} does not match theta_co shape {theta_arr.shape}'
            )
        theta_arr = np.where(mask, theta_arr, 1.0)

    theta_arr = np.clip(theta_arr, 0.0, 1.0)

    if isinstance(k0_co, Quantity):
        k0_val = k0_co.to('1/s').magnitude
    else:
        k0_val = float(k0_co)

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


def co_freezeout_rate(T: Quantity, nH: Quantity) -> Quantity:
    """CO freeze-out rate using SIGMA_D_PER_H from config."""
    T_K = T.to('K').magnitude
    nH_cm3 = nH.to('cm^-3').magnitude
    v_th = np.sqrt(8.0 * K_B * T_K / (np.pi * M_CO_CGS))
    sigma_nd = SIGMA_D_PER_H * nH_cm3
    k_fo = sigma_nd * v_th
    return Quantity(k_fo, '1/s')


def co_thermal_desorption_rate(T_d: Quantity) -> Quantity:
    """CO thermal desorption rate using E_BIND_CO and NU0_CO from config."""
    T_K = T_d.to('K').magnitude
    k_td = NU0_CO * np.exp(-E_BIND_CO / np.maximum(T_K, 1e-6))
    return Quantity(k_td, '1/s')


def co_photodesorption_rate(chi: Quantity) -> Quantity:
    """CO photodesorption rate using ALPHA_PD_ICE from config."""
    chi_val = chi.to('dimensionless').magnitude
    k_pd_ice = ALPHA_PD_ICE * chi_val
    return Quantity(k_pd_ice, '1/s')


def compute_co_two_phase_rates(
    *,
    nH: Quantity,
    T: Quantity,
    chi: Quantity,
    theta_co,
    min_rate: float = 1.0e-30,
) -> CoTwoPhaseRates:
    """Compute all CO two-phase chemistry rates using constants from config."""
    k_pd, tau_pd = compute_co_photodissociation_rate(
        chi=chi,
        theta_co=theta_co,
        k0_co=Quantity(K0_CO, '1/s'),
        candidate_mask=None,
        min_rate=min_rate,
    )
    k_fo = co_freezeout_rate(T, nH)
    k_td = co_thermal_desorption_rate(T)
    k_pd_ice = co_photodesorption_rate(chi)
    return CoTwoPhaseRates(k_pd=k_pd, k_fo=k_fo, k_td=k_td, k_pd_ice=k_pd_ice, tau_pd=tau_pd)


def solve_co_two_phase_steady_state(
    *,
    nH: Quantity,
    T: Quantity,
    chi: Quantity,
    theta_co,
    Xco_tot: float,
    tau_form: Quantity,
    min_rate: float = 1.0e-30,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Optional[Quantity]]:
    """Solve steady-state CO two-phase chemistry using constants from config."""
    rates = compute_co_two_phase_rates(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        min_rate=min_rate,
    )

    k_pd = rates.k_pd
    tau_pd = rates.tau_pd
    k_fo = rates.k_fo
    k_td = rates.k_td
    k_pd_ice = rates.k_pd_ice

    k_td_plus = k_td.to('1/s').magnitude + k_pd_ice.to('1/s').magnitude
    k_td_plus_safe = np.maximum(k_td_plus, 1e-60)
    eta = k_fo.to('1/s').magnitude / k_td_plus_safe

    nH_cm3 = nH.to('cm^-3').magnitude
    nco_max = float(Xco_tot) * nH_cm3
    tau_s = tau_form.to('s').magnitude

    k_pd_s = k_pd.to('1/s').magnitude
    denom = (1.0 + eta) + k_pd_s * tau_s
    nco_gas_arr = nco_max / denom
    nco_ice_arr = eta * nco_gas_arr

    nco_gas = Quantity(nco_gas_arr, 'cm^-3')
    nco_ice = Quantity(nco_ice_arr, 'cm^-3')
    X_co = Quantity(nco_gas_arr / (nH_cm3 + 1e-99), 'dimensionless')

    return X_co, nco_gas, nco_ice, k_pd, tau_pd


def evolve_co_two_phase_time_dependent(
    *,
    nH: Quantity,
    T: Quantity,
    chi: Quantity,
    theta_co,
    t_end: Quantity,
    Xco_tot: float,
    tau_form: Quantity,
    Xco_gas_init: Optional[float] = None,
    Xco_ice_init: Optional[float] = None,
    min_rate: float = 1.0e-30,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Optional[Quantity]]:
    """Evolve CO two-phase chemistry using constants from config."""
    rates = compute_co_two_phase_rates(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        min_rate=min_rate,
    )

    k_pd = rates.k_pd
    tau_pd = rates.tau_pd
    k_fo = rates.k_fo
    k_td = rates.k_td
    k_pd_ice = rates.k_pd_ice

    k_fo_s = k_fo.to('1/s').magnitude
    k_td_s = k_td.to('1/s').magnitude
    k_pd_s = k_pd.to('1/s').magnitude
    k_pd_ice_s = k_pd_ice.to('1/s').magnitude
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
        inv_tau = 1.0 / tau_s
        b_des = k_td_s + k_pd_ice_s

        a = -(inv_tau + k_fo_s + k_pd_s)
        b = b_des - inv_tau
        c = k_fo_s
        d = -b_des

        f0 = nco_max * inv_tau

        det = a * d - b * c
        if np.any(det == 0.0):
            raise ZeroDivisionError('Singular CO two-phase linear system (determinant is zero)')

        nco_gas_ss = (-d * f0) / det
        nco_ice_ss = (c * f0) / det

        m = 0.5 * (a + d)
        b11 = a - m
        b12 = b
        b21 = c
        b22 = d - m

        disc = b11 * b11 + b12 * b21

        mask_pos = disc > 0.0
        mask_neg = disc < 0.0
        mask_zero = disc == 0.0

        e11 = np.empty_like(disc, dtype=float)
        e12 = np.empty_like(disc, dtype=float)
        e21 = np.empty_like(disc, dtype=float)
        e22 = np.empty_like(disc, dtype=float)

        if np.any(mask_pos):
            delta = np.sqrt(disc[mask_pos])
            l1 = (m[mask_pos] + delta)
            l2 = (m[mask_pos] - delta)
            denom = (l1 - l2)
            e1 = np.exp(l1 * t_end_s_arr[mask_pos])
            e2 = np.exp(l2 * t_end_s_arr[mask_pos])

            e11[mask_pos] = (e1 * (a[mask_pos] - l2) - e2 * (a[mask_pos] - l1)) / denom
            e22[mask_pos] = (e1 * (d[mask_pos] - l2) - e2 * (d[mask_pos] - l1)) / denom
            e12[mask_pos] = b[mask_pos] * (e1 - e2) / denom
            e21[mask_pos] = c[mask_pos] * (e1 - e2) / denom

        if np.any(mask_neg):
            omega = np.sqrt(-disc[mask_neg])
            x = omega * t_end_s_arr[mask_neg]
            cosx = np.cos(x)
            sin_over = np.sin(x) / omega
            exp_mt = np.exp(m[mask_neg] * t_end_s_arr[mask_neg])

            e11[mask_neg] = exp_mt * (cosx + sin_over * b11[mask_neg])
            e22[mask_neg] = exp_mt * (cosx + sin_over * b22[mask_neg])
            e12[mask_neg] = exp_mt * (sin_over * b12[mask_neg])
            e21[mask_neg] = exp_mt * (sin_over * b21[mask_neg])

        if np.any(mask_zero):
            exp_mt = np.exp(m[mask_zero] * t_end_s_arr[mask_zero])
            t = t_end_s_arr[mask_zero]
            e11[mask_zero] = exp_mt * (1.0 + b11[mask_zero] * t)
            e22[mask_zero] = exp_mt * (1.0 + b22[mask_zero] * t)
            e12[mask_zero] = exp_mt * (b12[mask_zero] * t)
            e21[mask_zero] = exp_mt * (b21[mask_zero] * t)

        dg = nco_gas - nco_gas_ss
        di = nco_ice - nco_ice_ss

        nco_gas = nco_gas_ss + e11 * dg + e12 * di
        nco_ice = nco_ice_ss + e21 * dg + e22 * di

        nco_gas = np.maximum(nco_gas, 0.0)
        nco_ice = np.maximum(nco_ice, 0.0)
    else:
        dt_s = dt.to('s').magnitude
        t = 0.0
        n_steps = 0
        max_steps = int(1e8)
        while t < t_end_s and n_steps < max_steps:
            dt_step = min(dt_s, t_end_s - t)
            R_form = (nco_max - nco_gas - nco_ice) / tau_s
            dnco_gas_dt = (
                R_form
                + k_td_s * nco_ice
                + k_pd_ice_s * nco_ice
                - k_fo_s * nco_gas
                - k_pd_s * nco_gas
            )
            dnco_ice_dt = k_fo_s * nco_gas - k_td_s * nco_ice - k_pd_ice_s * nco_ice
            nco_gas = nco_gas + dt_step * dnco_gas_dt
            nco_ice = nco_ice + dt_step * dnco_ice_dt
            nco_gas = np.maximum(nco_gas, 0.0)
            nco_ice = np.maximum(nco_ice, 0.0)
            t += dt_step
            n_steps += 1

    nco_gas_q = Quantity(nco_gas, 'cm^-3')
    nco_ice_q = Quantity(nco_ice, 'cm^-3')
    X_co = Quantity(nco_gas / (nH_cm3 + 1e-99), 'dimensionless')

    return X_co, nco_gas_q, nco_ice_q, k_pd, tau_pd
