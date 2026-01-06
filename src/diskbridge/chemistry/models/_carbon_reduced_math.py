from __future__ import annotations

from typing import Optional, Tuple

from dataclasses import dataclass

import numpy as np

from diskbridge._units import Quantity
from diskbridge._constants import (
    K_B, M_H, K0_CO, E_BIND_CO, NU0_CO, F_DRAINE, N_LAY, N_SURF, Y_CO
)

# constant below should probably be in _constants.py 
M_CO_CGS = 28.0 * M_H


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
    min_rate: float = 1.0e-30,
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
    T_K = T.to('K').magnitude
    nH_cm3 = nH.to('cm^-3').magnitude
    v_th = np.sqrt(8.0 * K_B * T_K / (np.pi * M_CO_CGS))
    sigma_nd = sigma_d_per_H.to('cm^2').magnitude * nH_cm3
    k_fo = sigma_nd * v_th
    return Quantity(k_fo, '1/s')


def co_thermal_desorption_rate(T_d: Quantity) -> Quantity:
    """CO thermal desorption rate using E_BIND_CO and NU0_CO from config."""
    T_K = T_d.to('K').magnitude
    k_td = NU0_CO * np.exp(-E_BIND_CO / np.maximum(T_K, 1e-6))
    return Quantity(k_td, '1/s')


def co_photodesorption_rate_surface(chi: Quantity) -> Quantity:
    chi_val = chi.to('dimensionless').magnitude
    k_pd_surf = (chi_val * F_DRAINE) * (Y_CO / (4.0 * N_SURF * float(N_LAY)))
    return Quantity(k_pd_surf, '1/s')


def co_photodesorption_sink(
    *,
    nH: Quantity,
    nco_ice: Quantity,
    sigma_d_per_H: Quantity,
    k_pd_surf: Quantity,
) -> tuple[Quantity, Quantity, Quantity, Quantity]:
    nH_cm3 = nH.to('cm^-3').magnitude
    nco_ice_cm3 = nco_ice.to('cm^-3').magnitude
    sigma = sigma_d_per_H.to('cm^2').magnitude
    A_d = sigma * nH_cm3

    n_ice_act_max = A_d * N_SURF * float(N_LAY)
    n_ice_act = np.minimum(nco_ice_cm3, n_ice_act_max)

    k_pd_s = k_pd_surf.to('1/s').magnitude
    R_pd = k_pd_s * n_ice_act

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
    min_rate: float = 1.0e-30,
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
    min_rate: float = 1.0e-30,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Optional[Quantity], dict]:
    """Solve steady-state CO chemistry using an analytic reduced model."""
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

    nH_cm3 = nH.to('cm^-3').magnitude
    nco_max = float(Xco_tot) * nH_cm3
    tau_s = tau_form.to('s').magnitude

    k_fo_s = k_fo.to('1/s').magnitude
    k_td_s = k_td.to('1/s').magnitude
    k_pd_s = k_pd.to('1/s').magnitude

    k_pd_surf_s = k_pd_surf.to('1/s').magnitude

    # Surface-active ice cap, i_max (cm^-3)
    n_ice_act_max = (
        sigma_d_per_H.to('cm^2').magnitude
        * nH.to('cm^-3').magnitude
        * N_SURF
        * float(N_LAY)
    )

    # Thin-ice regime: R_pd = k_surf * i
    denom_thin = (k_td_s + k_pd_surf_s)
    a = np.where(denom_thin > 0.0, k_fo_s / denom_thin, 0.0)
    denom_g_thin = 1.0 + tau_s * k_pd_s + a
    g_thin = np.where(denom_g_thin > 0.0, nco_max / denom_g_thin, 0.0)
    g_thin = np.clip(g_thin, 0.0, nco_max)
    i_thin = a * g_thin
    i_thin = np.maximum(i_thin, 0.0)

    # Thick-ice regime: R_pd saturates at k_surf * i_max
    R0 = k_pd_surf_s * n_ice_act_max
    denom_g_thick = 1.0 + tau_s * k_pd_s + np.where(k_td_s > 0.0, (k_fo_s / k_td_s), 0.0)
    numer_g_thick = nco_max + np.where(k_td_s > 0.0, (R0 / k_td_s), 0.0)
    g_thick = np.where(denom_g_thick > 0.0, numer_g_thick / denom_g_thick, 0.0)
    g_thick = np.clip(g_thick, 0.0, nco_max)
    i_thick = np.where(k_td_s > 0.0, (k_fo_s * g_thick - R0) / k_td_s, 0.0)
    i_thick = np.maximum(i_thick, 0.0)

    use_thin = i_thin < n_ice_act_max
    nco_gas = np.where(use_thin, g_thin, g_thick)
    nco_ice = np.where(use_thin, i_thin, i_thick)

    nco_total = nco_gas + nco_ice
    excess = nco_total - nco_max
    if np.any(excess > 0.0):
        # Clamp by reducing ice (preferred) then gas if needed.
        reduce_ice = np.minimum(nco_ice, excess)
        nco_ice = nco_ice - reduce_ice
        excess2 = excess - reduce_ice
        nco_gas = np.maximum(nco_gas - excess2, 0.0)

    n_ice_act = np.minimum(nco_ice, n_ice_act_max)
    R_pd = k_pd_surf_s * n_ice_act
    R_pd_q = Quantity(R_pd, 'cm^-3/s')

    nco_gas_q = Quantity(nco_gas, 'cm^-3')
    nco_ice_q = Quantity(nco_ice, 'cm^-3')
    X_co = Quantity(nco_gas / (nH_cm3 + 1e-99), 'dimensionless')

    diag = {
        'n_ice_act_max': Quantity(n_ice_act_max, 'cm^-3'),
        'n_ice_act': Quantity(n_ice_act, 'cm^-3'),
        'k_pd_surf': k_pd_surf,
        'R_pd': R_pd_q,
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
    min_rate: float = 1.0e-30,
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
        if np.ndim(t_end_s) != 0:
            raise ValueError("dt is required when evolving with per-cell t_end")
        dt_s = 0.05 * float(np.median(tau_s)) if np.ndim(tau_s) != 0 else 0.05 * float(tau_s)
        dt_s = max(dt_s, 0.0)
    else:
        dt_s = float(dt.to('s').magnitude)

    if float(dt_s) <= 0.0:
        dt_s = 1.0

    t = 0.0
    n_steps = 0
    max_steps = int(1e8)

    n_ice_act_max_last = None
    n_ice_act_last = None
    R_pd_last = None

    while t < float(t_end_s) and n_steps < max_steps:
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
    X_co = Quantity(nco_gas / (nH_cm3 + 1e-99), 'dimensionless')

    diag = {
        'n_ice_act_max': n_ice_act_max_last,
        'n_ice_act': n_ice_act_last,
        'k_pd_surf': k_pd_surf,
        'R_pd': R_pd_last,
    }

    return X_co, nco_gas_q, nco_ice_q, k_pd, tau_pd, diag
