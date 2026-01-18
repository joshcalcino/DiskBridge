import numpy as np

from diskbridge._constants import GAMMA_CHX0, K0_NL97, K1_NL97, X_C_TOT, X_O_NL97
from diskbridge.chemistry.models._carbon_reduced_math import solve_carbon_reduced_steady_state_cgs


def test_solve_carbon_reduced_steady_state_cgs_numba_compiles():
    ncells = 1

    nH_cm3 = np.full(ncells, 1.0e5, dtype=np.float64)
    T_K = np.full(ncells, 30.0, dtype=np.float64)
    chi = np.full(ncells, 1.0, dtype=np.float64)
    theta_co = np.full(ncells, 1.0, dtype=np.float64)
    tau_form_s = np.full(ncells, 1.0e12, dtype=np.float64)
    sigma_d_per_H_cm2 = np.full(ncells, 1.0e-21, dtype=np.float64)
    nH2_cm3 = np.full(ncells, 0.5 * nH_cm3[0], dtype=np.float64)
    Tg_K = np.full(ncells, 30.0, dtype=np.float64)

    formation_model = 0

    out_Xco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_ice = np.empty(ncells, dtype=np.float64)
    out_k_pd = np.empty(ncells, dtype=np.float64)
    out_tau_pd = np.empty(ncells, dtype=np.float64)
    out_n_ice_act_max = np.empty(ncells, dtype=np.float64)
    out_n_ice_act = np.empty(ncells, dtype=np.float64)
    out_k_pd_surf = np.empty(ncells, dtype=np.float64)
    out_R_pd = np.empty(ncells, dtype=np.float64)

    solve_carbon_reduced_steady_state_cgs(
        nH_cm3=nH_cm3,
        T_K=T_K,
        chi=chi,
        theta_co=theta_co,
        tau_form_s=tau_form_s,
        sigma_d_per_H_cm2=sigma_d_per_H_cm2,
        nH2_cm3=nH2_cm3,
        Tg_K=Tg_K,
        formation_model=formation_model,
        k0_nl97=float(K0_NL97),
        k1_nl97=float(K1_NL97),
        xO=float(X_O_NL97),
        gamma_chx0=float(GAMMA_CHX0),
        Xco_tot=float(X_C_TOT),
        min_rate=0.0,
        out_Xco_gas=out_Xco_gas,
        out_nco_gas=out_nco_gas,
        out_nco_ice=out_nco_ice,
        out_k_pd=out_k_pd,
        out_tau_pd=out_tau_pd,
        out_n_ice_act_max=out_n_ice_act_max,
        out_n_ice_act=out_n_ice_act,
        out_k_pd_surf=out_k_pd_surf,
        out_R_pd=out_R_pd,
    )

    assert out_Xco_gas.shape == (ncells,)
    assert np.all(np.isfinite(out_Xco_gas))
    assert np.all(out_Xco_gas >= 0.0)
