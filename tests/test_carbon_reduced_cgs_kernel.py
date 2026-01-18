import pytest

pytest.skip("carbon_reduced kernel is not part of the unified gow17 backend", allow_module_level=True)

import numpy as np

from diskbridge._constants import (
    K_B,
    M_H,
    K0_CO,
    E_BIND_CO,
    NU0_CO,
    F_DRAINE,
    N_LAY,
    N_SURF,
    Y_CO,
)
from diskbridge.chemistry.models._carbon_reduced_math import solve_carbon_reduced_steady_state_cgs


M_CO_CGS = 28.0 * M_H


def _reference_solve(
    nH_cm3: np.ndarray,
    T_K: np.ndarray,
    chi: np.ndarray,
    theta_co: np.ndarray,
    tau_form_s: np.ndarray,
    sigma_d_per_H_cm2: np.ndarray,
    Xco_tot: float,
    min_rate: float,
) -> dict[str, np.ndarray]:
    ncells = nH_cm3.size

    out_Xco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_ice = np.empty(ncells, dtype=np.float64)
    out_k_pd = np.empty(ncells, dtype=np.float64)
    out_tau_pd = np.empty(ncells, dtype=np.float64)
    out_n_ice_act_max = np.empty(ncells, dtype=np.float64)
    out_n_ice_act = np.empty(ncells, dtype=np.float64)
    out_k_pd_surf = np.empty(ncells, dtype=np.float64)
    out_R_pd = np.empty(ncells, dtype=np.float64)

    for i in range(ncells):
        nH_i = float(nH_cm3[i])
        T_i = float(T_K[i])
        chi_i = float(chi[i])

        theta_i = float(theta_co[i])
        if theta_i < 0.0:
            theta_i = 0.0
        elif theta_i > 1.0:
            theta_i = 1.0

        tau_form_i = float(tau_form_s[i])
        sigma_i = float(sigma_d_per_H_cm2[i])

        nco_max = float(Xco_tot) * nH_i

        k_pd = float(K0_CO) * chi_i * theta_i
        out_k_pd[i] = k_pd
        if k_pd <= 0.0:
            out_tau_pd[i] = np.inf
        else:
            k_pd_safe = k_pd
            if float(min_rate) > 0.0 and k_pd_safe < float(min_rate):
                k_pd_safe = float(min_rate)
            out_tau_pd[i] = 1.0 / k_pd_safe

        v_th = 0.0
        if T_i > 0.0:
            v_th = np.sqrt(8.0 * float(K_B) * T_i / (np.pi * float(M_CO_CGS)))

        k_fo = (sigma_i * nH_i) * v_th

        k_td = 0.0
        if T_i > 0.0:
            k_td = float(NU0_CO) * np.exp(-float(E_BIND_CO) / T_i)

        k_pd_surf = (chi_i * float(F_DRAINE)) * (float(Y_CO) / (4.0 * float(N_SURF) * float(N_LAY)))
        out_k_pd_surf[i] = k_pd_surf

        n_ice_act_max = (sigma_i * nH_i) * float(N_SURF) * float(N_LAY)
        out_n_ice_act_max[i] = n_ice_act_max

        denom_thin = k_td + k_pd_surf
        a = 0.0
        if denom_thin > 0.0:
            a = k_fo / denom_thin

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

        use_thin = i_thin < n_ice_act_max
        nco_gas = g_thin if use_thin else g_thick
        nco_ice = i_thin if use_thin else i_thick

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

        n_ice_act = nco_ice if nco_ice < n_ice_act_max else n_ice_act_max
        out_n_ice_act[i] = n_ice_act
        out_R_pd[i] = k_pd_surf * n_ice_act

        if nH_i > 0.0:
            out_Xco_gas[i] = nco_gas / nH_i
        else:
            out_Xco_gas[i] = 0.0

    return {
        "Xco_gas": out_Xco_gas,
        "nco_gas": out_nco_gas,
        "nco_ice": out_nco_ice,
        "k_pd": out_k_pd,
        "tau_pd": out_tau_pd,
        "n_ice_act_max": out_n_ice_act_max,
        "n_ice_act": out_n_ice_act,
        "k_pd_surf": out_k_pd_surf,
        "R_pd": out_R_pd,
    }


def test_cgs_kernel_matches_reference_small_grid() -> None:
    rng = np.random.default_rng(0)
    ncells = 32

    nH_cm3 = rng.lognormal(mean=6.0, sigma=1.0, size=ncells).astype(np.float64)
    nH_cm3[0] = 0.0

    T_K = rng.uniform(low=0.0, high=200.0, size=ncells).astype(np.float64)
    T_K[1] = 0.0

    chi = rng.uniform(low=0.0, high=10.0, size=ncells).astype(np.float64)
    theta = rng.uniform(low=-0.5, high=1.5, size=ncells).astype(np.float64)

    tau_form_s = rng.lognormal(mean=10.0, sigma=0.5, size=ncells).astype(np.float64)
    sigma_cm2 = rng.lognormal(mean=-21.0, sigma=0.2, size=ncells).astype(np.float64)

    Xco_tot = 1.0e-4
    min_rate = 0.0

    ref = _reference_solve(
        nH_cm3=nH_cm3,
        T_K=T_K,
        chi=chi,
        theta_co=theta,
        tau_form_s=tau_form_s,
        sigma_d_per_H_cm2=sigma_cm2,
        Xco_tot=Xco_tot,
        min_rate=min_rate,
    )

    out_Xco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_ice = np.empty(ncells, dtype=np.float64)
    out_k_pd = np.empty(ncells, dtype=np.float64)
    out_tau_pd = np.empty(ncells, dtype=np.float64)
    out_n_ice_act_max = np.empty(ncells, dtype=np.float64)
    out_n_ice_act = np.empty(ncells, dtype=np.float64)
    out_k_pd_surf = np.empty(ncells, dtype=np.float64)
    out_R_pd = np.empty(ncells, dtype=np.float64)

    nH2_cm3 = np.zeros_like(nH_cm3)
    Tg_K = np.ascontiguousarray(T_K, dtype=np.float64)

    solve_carbon_reduced_steady_state_cgs(
        nH_cm3,
        T_K,
        chi,
        theta,
        tau_form_s,
        sigma_cm2,
        nH2_cm3,
        Tg_K,
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

    eps = np.finfo(np.float64).eps
    rtol = 32.0 * eps

    np.testing.assert_allclose(out_Xco_gas, ref["Xco_gas"], rtol=rtol, atol=0.0)
    np.testing.assert_allclose(out_nco_gas, ref["nco_gas"], rtol=rtol, atol=0.0)
    np.testing.assert_allclose(out_nco_ice, ref["nco_ice"], rtol=rtol, atol=0.0)
    np.testing.assert_allclose(out_k_pd, ref["k_pd"], rtol=rtol, atol=0.0)

    tau_is_inf = ~np.isfinite(ref["tau_pd"])
    assert np.array_equal(~np.isfinite(out_tau_pd), tau_is_inf)
    np.testing.assert_allclose(
        out_tau_pd[~tau_is_inf],
        ref["tau_pd"][~tau_is_inf],
        rtol=rtol,
        atol=0.0,
    )

    np.testing.assert_allclose(out_n_ice_act_max, ref["n_ice_act_max"], rtol=rtol, atol=0.0)
    np.testing.assert_allclose(out_n_ice_act, ref["n_ice_act"], rtol=rtol, atol=0.0)
    np.testing.assert_allclose(out_k_pd_surf, ref["k_pd_surf"], rtol=rtol, atol=0.0)
    np.testing.assert_allclose(out_R_pd, ref["R_pd"], rtol=rtol, atol=0.0)
