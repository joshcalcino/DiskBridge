from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from diskbridge._units import Quantity, units

M_CO_CGS = (28.0 * units('m_H')).to('g').magnitude
K_BOLTZ_CGS = units('k_B').to('erg/K').magnitude


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


def co_freezeout_rate(
    T: Quantity,
    nH: Quantity,
    sigma_d_per_H: float,
) -> Quantity:
    T_K = T.to('K').magnitude
    nH_cm3 = nH.to('cm^-3').magnitude
    v_th = np.sqrt(8.0 * K_BOLTZ_CGS * T_K / (np.pi * M_CO_CGS))
    sigma_nd = float(sigma_d_per_H) * nH_cm3
    k_fo = sigma_nd * v_th
    return Quantity(k_fo, '1/s')


def co_thermal_desorption_rate(
    T_d: Quantity,
    E_bind: float,
    nu0: float,
) -> Quantity:
    T_K = T_d.to('K').magnitude
    k_td = float(nu0) * np.exp(-float(E_bind) / np.maximum(T_K, 1e-6))
    return Quantity(k_td, '1/s')


def co_photodesorption_rate(
    chi: Quantity,
    alpha_pd: float,
) -> Quantity:
    chi_val = chi.to('dimensionless').magnitude
    k_pd_ice = float(alpha_pd) * chi_val
    return Quantity(k_pd_ice, '1/s')


def solve_co_two_phase_steady_state(
    *,
    nH: Quantity,
    T: Quantity,
    chi: Quantity,
    theta_co,
    Xco_tot: float,
    tau_form: Quantity,
    k0_co: Quantity,
    sigma_d_per_H: float,
    E_bind: float,
    nu0: float,
    alpha_pd_ice: float,
    min_rate: float = 1.0e-30,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Optional[Quantity]]:
    k_pd, tau_pd = compute_co_photodissociation_rate(
        chi=chi,
        theta_co=theta_co,
        k0_co=k0_co,
        candidate_mask=None,
        min_rate=min_rate,
    )

    k_fo = co_freezeout_rate(T, nH, sigma_d_per_H)
    k_td = co_thermal_desorption_rate(T, E_bind, nu0)
    k_pd_ice = co_photodesorption_rate(chi, alpha_pd_ice)

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
    dt: Optional[Quantity],
    Xco_tot: float,
    tau_form: Quantity,
    k0_co: Quantity,
    sigma_d_per_H: float,
    E_bind: float,
    nu0: float,
    alpha_pd_ice: float,
    Xco_gas_init: Optional[float],
    Xco_ice_init: Optional[float],
    min_rate: float = 1.0e-30,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Optional[Quantity]]:
    k_pd, tau_pd = compute_co_photodissociation_rate(
        chi=chi,
        theta_co=theta_co,
        k0_co=k0_co,
        candidate_mask=None,
        min_rate=min_rate,
    )

    k_fo = co_freezeout_rate(T, nH, sigma_d_per_H)
    k_td = co_thermal_desorption_rate(T, E_bind, nu0)
    k_pd_ice = co_photodesorption_rate(chi, alpha_pd_ice)

    k_fo_s = k_fo.to('1/s').magnitude
    k_td_s = k_td.to('1/s').magnitude
    k_pd_s = k_pd.to('1/s').magnitude
    k_pd_ice_s = k_pd_ice.to('1/s').magnitude
    tau_s = tau_form.to('s').magnitude
    t_end_s = t_end.to('s').magnitude

    nH_cm3 = nH.to('cm^-3').magnitude
    nco_max = float(Xco_tot) * nH_cm3

    if Xco_gas_init is None:
        nco_gas = nco_max.copy()
    else:
        nco_gas = float(Xco_gas_init) * nH_cm3

    if Xco_ice_init is None:
        nco_ice = np.zeros_like(nco_gas)
    else:
        nco_ice = float(Xco_ice_init) * nH_cm3

    if dt is None:
        total_rate = k_fo_s + k_td_s + k_pd_s + k_pd_ice_s + 1.0 / tau_s
        min_timescale = 1.0 / np.maximum(total_rate, 1e-60)
        dt_s = 0.1 * np.nanmin(min_timescale)
        dt_s = np.clip(dt_s, 1e-10 * t_end_s, 0.01 * t_end_s)
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
