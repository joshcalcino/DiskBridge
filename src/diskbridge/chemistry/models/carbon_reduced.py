from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np
from numba import njit, prange

import diskbridge
from diskbridge._units import Quantity
from diskbridge._constants import (
    TAU_CO_FORM,
    TAU_FORM_N0, TAU_FORM_TAU0, TAU_FORM_TAU_MIN, TAU_FORM_ALPHA
)
from diskbridge._constants import K0_NL97, K1_NL97, GAMMA_CHX0, X_O_NL97
from diskbridge._config import resolve_model_config
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.hydrogen.api import ensure_h2_partition
from diskbridge.chemistry.models._carbon_reduced_math import (
    solve_carbon_reduced_steady_state_cgs,
    evolve_carbon_reduced_time_dependent,
)
from diskbridge.chemistry.processes.carbon_closure import carbon_closure_cell_param_cgs
from diskbridge.chemistry.tau_form import compute_tau_form_co_cgs
from diskbridge.chemistry.validation import validate_chemistry_state
from diskbridge._logging import logger


def _as_cgs_f64(q: Quantity, unit: str) -> np.ndarray:
    return np.ascontiguousarray(q.to(unit).magnitude, dtype=np.float64)


def _flat_view(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a.reshape(-1), dtype=np.float64)


def _parse_value_to_cgs_float(val, unit: str, default_cgs: float) -> float:
    if val is None:
        return float(default_cgs)
    if isinstance(val, str):
        return float(Quantity(val).to(unit).magnitude)
    if hasattr(val, 'to'):
        return float(val.to(unit).magnitude)
    return float(val)


@njit(parallel=True, fastmath=True, cache=True)
def _carbon_closure_kernel(
    nH: np.ndarray,
    chi: np.ndarray,
    Tg: np.ndarray,
    nco_total: np.ndarray,
    X_C_tot: float,
    Gamma_C0: float,
    alpha_rec_c0: float,
    T_rec_exp: float,
    nCplus_out: np.ndarray,
    nC_out: np.ndarray,
    ne_out: np.ndarray,
) -> None:
    ncells = nH.size
    for i in prange(ncells):
        nCplus, nC, ne = carbon_closure_cell_param_cgs(
            nH_cm3=nH[i],
            chi=chi[i],
            Tg_K=Tg[i],
            nco_total_cm3=nco_total[i],
            X_C_tot=X_C_tot,
            Gamma_C0=Gamma_C0,
            alpha_rec_c0=alpha_rec_c0,
            T_rec_exp=T_rec_exp,
        )

        nCplus_out[i] = nCplus
        nC_out[i] = nC
        ne_out[i] = ne


def _carbon_closure(
    *,
    nH: Quantity,
    chi: Quantity,
    Tg: Quantity,
    nco_total: Quantity,
    X_C_tot: float = None,
    Gamma_C0: float = None,
    alpha_rec_c0: float = None,
    T_rec_exp: float = None,
) -> tuple[Quantity, Quantity, Quantity]:
    from diskbridge._constants import (
        X_C_TOT as _X_C_TOT,
        GAMMA_C0 as _GAMMA_C0,
        ALPHA_REC_C0 as _ALPHA_REC_C0,
        T_REC_EXP as _T_REC_EXP,
    )

    if X_C_tot is None:
        X_C_tot = _X_C_TOT
    if Gamma_C0 is None:
        Gamma_C0 = _GAMMA_C0
    if alpha_rec_c0 is None:
        alpha_rec_c0 = _ALPHA_REC_C0
    if T_rec_exp is None:
        T_rec_exp = _T_REC_EXP

    orig_shape = nH.magnitude.shape
    ncells = nH.magnitude.size

    nH_flat = np.ascontiguousarray(nH.to('cm^-3').magnitude.reshape(ncells), dtype=np.float64)
    chi_flat = np.ascontiguousarray(chi.to('dimensionless').magnitude.reshape(ncells), dtype=np.float64)
    Tg_flat = np.ascontiguousarray(Tg.to('K').magnitude.reshape(ncells), dtype=np.float64)
    nco_flat = np.ascontiguousarray(nco_total.to('cm^-3').magnitude.reshape(ncells), dtype=np.float64)

    nCplus_flat = np.zeros(ncells, dtype=np.float64)
    nC_flat = np.zeros(ncells, dtype=np.float64)
    ne_flat = np.zeros(ncells, dtype=np.float64)

    _carbon_closure_kernel(
        nH_flat,
        chi_flat,
        Tg_flat,
        nco_flat,
        float(X_C_tot),
        float(Gamma_C0),
        float(alpha_rec_c0),
        float(T_rec_exp),
        nCplus_flat,
        nC_flat,
        ne_flat,
    )

    nCplus = Quantity(nCplus_flat.reshape(orig_shape), 'cm^-3')
    nC = Quantity(nC_flat.reshape(orig_shape), 'cm^-3')
    ne = Quantity(ne_flat.reshape(orig_shape), 'cm^-3')

    return nCplus, nC, ne


def _resolve_tau_form(
    rad: 'RadModel',
    *,
    config: dict,
    nH2_cm3: np.ndarray,
) -> np.ndarray:
    if 'tau_form' in config:
        tau_form = config['tau_form']
        if isinstance(tau_form, Quantity):
            tau_arr = _as_cgs_f64(tau_form, 's')
        else:
            tau_arr = np.asarray(tau_form, dtype=np.float64)

        if np.ndim(tau_arr) == 0:
            return np.ascontiguousarray(np.full_like(nH2_cm3, float(tau_arr), dtype=np.float64))

        if np.shape(tau_arr) != np.shape(nH2_cm3):
            raise ValueError(
                f"tau_form shape {np.shape(tau_arr)} must match nH2 shape {np.shape(nH2_cm3)}"
            )
        return np.ascontiguousarray(tau_arr, dtype=np.float64)

    model = str(diskbridge.params.co_tau_form_model).lower()
    if model == "density_capped":
        return compute_tau_form_co_cgs(
            nH2_cm3,
            n0_cm3=float(TAU_FORM_N0),
            tau0_s=float(TAU_FORM_TAU0),
            tau_min_s=float(TAU_FORM_TAU_MIN),
            alpha=float(TAU_FORM_ALPHA),
        )
    if model == "off":
        return np.ascontiguousarray(np.full_like(nH2_cm3, float(TAU_CO_FORM), dtype=np.float64))
    raise ValueError(f"Unknown co_tau_form_model={diskbridge.params.co_tau_form_model!r}")


def _compute_co_shielding(
    *,
    rad: 'RadModel',
    nH_cm3: np.ndarray,
    chi: np.ndarray,
    nCO_cm3: np.ndarray,
    nH2_cm3: np.ndarray,
    nside: int,
    b_kms: float,
    avg_mode: str = "isotropic_mean",
) -> tuple[np.ndarray, np.ndarray]:
    from diskbridge.chemistry.shielding.columns_1d import is_effectively_1d
    from diskbridge.chemistry.shielding.visser_shielding import VisserShielding

    if is_effectively_1d(rad.model.mesh, tuple(nH_cm3.shape)):
        from diskbridge.chemistry.shielding.columns_1d import compute_co_shielding_1d

    visser = VisserShielding(b_kms=float(b_kms))

    if is_effectively_1d(rad.model.mesh, tuple(nH_cm3.shape)):
        theta_co, chi_eff = compute_co_shielding_1d(
            rad.model.mesh,
            nH_cm3,
            chi,
            visser=visser,
            nCO=nCO_cm3,
            nH2=nH2_cm3,
            b_kms=float(b_kms),
            outer="min",
        )
    else:
        from diskbridge.chemistry.shielding.healpix_columns import compute_co_shielding_healpix

        theta_co, chi_eff = compute_co_shielding_healpix(
            mesh=rad.model.mesh,
            nH=nH_cm3,
            chi=chi,
            visser=visser,
            nCO=nCO_cm3,
            nH2=nH2_cm3,
            nside=int(nside),
            b_kms=float(b_kms),
            progress_chunks=None,
            cache_dir=None,
            avg_mode=avg_mode,
        )
    return theta_co, chi_eff


def _run_carbon_closure(
    *,
    rad: 'RadModel',
    nH: Quantity,
    chi: Quantity,
    Tdust: Quantity,
    nco_gas: Quantity,
    nco_ice: Quantity,
) -> tuple[Quantity, Quantity, Quantity, Quantity]:
    nco_total = Quantity(
        nco_gas.to('cm^-3').magnitude + nco_ice.to('cm^-3').magnitude,
        'cm^-3'
    )

    if rad.gas_temperature is not None:
        Tg = rad.gas_temperature
        logger.info("Carbon closure using gas temperature")
    else:
        Tg = Tdust
        logger.info("Carbon closure using dust temperature (gas T not available)")

    nCplus, nC, ne = _carbon_closure(nH=nH, chi=chi, Tg=Tg, nco_total=nco_total)
    logger.info(f"Carbon closure: max(nCplus)={np.max(nCplus.magnitude):.2e} cm^-3")
    return nCplus, nC, ne, nco_total


def _carbon_closure_cgs(
    *,
    nH_cm3: np.ndarray,
    chi: np.ndarray,
    Tg_K: np.ndarray,
    nco_total_cm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from diskbridge._constants import (
        X_C_TOT as _X_C_TOT,
        GAMMA_C0 as _GAMMA_C0,
        ALPHA_REC_C0 as _ALPHA_REC_C0,
        T_REC_EXP as _T_REC_EXP,
    )

    nH_flat = _flat_view(nH_cm3)
    chi_flat = _flat_view(chi)
    Tg_flat = _flat_view(Tg_K)
    nco_flat = _flat_view(nco_total_cm3)

    ncells = nH_flat.size
    nCplus_flat = np.empty(ncells, dtype=np.float64)
    nC_flat = np.empty(ncells, dtype=np.float64)
    ne_flat = np.empty(ncells, dtype=np.float64)

    _carbon_closure_kernel(
        nH_flat,
        chi_flat,
        Tg_flat,
        nco_flat,
        float(_X_C_TOT),
        float(_GAMMA_C0),
        float(_ALPHA_REC_C0),
        float(_T_REC_EXP),
        nCplus_flat,
        nC_flat,
        ne_flat,
    )

    return nCplus_flat, nC_flat, ne_flat


def _prepare_common(
    rad: 'RadModel',
    config: dict,
) -> tuple[Quantity, Quantity, Quantity, np.ndarray, float, bool, int, float, int]:
    Xco_tot = float(config.get('Xco_tot', float(diskbridge.params.abundance)))
    skip_shielding = bool(config.get('skip_shielding', False))
    nside = int(diskbridge.params.nside)
    b_kms = float(config.get('b_kms', 0.3))
    shielding_iter = int(config.get('shielding_iter', 1))

    nH = rad.ensure_nH()
    Tdust = rad.ensure_dust_temperature()
    chi = rad.ensure_chi()

    ensure_h2_partition(rad, nH=nH, chi_dust=chi)
    nH2_cm3 = _as_cgs_f64(rad.nH2, 'cm^-3')
    tau_form_s = _resolve_tau_form(rad, config=config, nH2_cm3=nH2_cm3)

    return nH, Tdust, chi, tau_form_s, Xco_tot, skip_shielding, nside, b_kms, shielding_iter


def run_steady(rad: 'RadModel', config: dict) -> ChemistryResult:
    """Run steady-state CO two-phase chemistry.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    config : dict
        Configuration with keys:
        - Xco_tot: float (default: from diskbridge.params.abundance)
        - tau_form: Quantity (optional; if absent uses diskbridge.params.co_tau_form_model)
        - skip_shielding: bool (default: False)
        - nside: int (default: 4)
        - b_kms: float (default: 0.3)
        
    Returns
    -------
    ChemistryResult
        Result with abundances, number_densities, and diagnostic fields
    """
    cfg = resolve_model_config(("chemistry", "carbon_reduced"), overrides=config)
    nH, Tdust, chi, tau_form_s, Xco_tot, skip_shielding, nside, b_kms, shielding_iter = _prepare_common(rad, cfg)

    sigma_d_per_H = rad.ensure_sigma_d_per_H()

    nH_cm3 = _as_cgs_f64(nH, 'cm^-3')
    T_K = _as_cgs_f64(Tdust, 'K')
    chi_dim = _as_cgs_f64(chi, 'dimensionless')
    sigma_cm2 = _as_cgs_f64(sigma_d_per_H, 'cm^2')
    nH2_cm3 = _as_cgs_f64(rad.nH2, 'cm^-3')
    nH_atom_cm3 = _as_cgs_f64(rad.nH_atom, 'cm^-3')

    if rad.gas_temperature is not None:
        Tg_K = _as_cgs_f64(rad.gas_temperature, 'K')
        logger.info("CO formation using gas temperature")
    else:
        Tg_K = T_K
        logger.info("CO formation using dust temperature (gas T not available)")

    orig_shape = nH_cm3.shape
    ncells = nH_cm3.size

    nH_flat = _flat_view(nH_cm3)
    T_flat = _flat_view(T_K)
    chi_flat = _flat_view(chi_dim)
    sigma_flat = _flat_view(sigma_cm2)
    tau_form_flat = _flat_view(tau_form_s)
    nH2_flat = _flat_view(nH2_cm3)
    Tg_flat = _flat_view(Tg_K)

    formation_model = str(cfg.get('formation_model', 'tau')).lower()
    if formation_model == 'tau':
        formation_model_flag = 0
    elif formation_model == 'nl97':
        formation_model_flag = 1
    else:
        raise ValueError(f"Unknown formation_model={formation_model!r} (expected 'tau' or 'nl97')")

    k0_nl97 = _parse_value_to_cgs_float(cfg.get('k0_nl97', None), 'cm^3/s', float(K0_NL97))
    k1_nl97 = _parse_value_to_cgs_float(cfg.get('k1_nl97', None), 'cm^3/s', float(K1_NL97))
    gamma_chx0 = _parse_value_to_cgs_float(cfg.get('gamma_chx0', None), '1/s', float(GAMMA_CHX0))
    xO = float(cfg.get('xO', X_O_NL97))

    if getattr(rad, 'nco_gas', None) is not None:
        nco_guess_cm3 = _as_cgs_f64(rad.nco_gas, 'cm^-3')
    else:
        nco_guess_cm3 = np.ascontiguousarray(float(Xco_tot) * nH_cm3, dtype=np.float64)

    theta_arr = np.ones_like(nH_cm3, dtype=np.float64)
    chi_eff_arr = np.ascontiguousarray(chi_dim, dtype=np.float64)

    out_Xco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_gas = np.empty(ncells, dtype=np.float64)
    out_nco_ice = np.empty(ncells, dtype=np.float64)
    out_k_pd = np.empty(ncells, dtype=np.float64)
    out_tau_pd = np.empty(ncells, dtype=np.float64)
    out_n_ice_act_max = np.empty(ncells, dtype=np.float64)
    out_n_ice_act = np.empty(ncells, dtype=np.float64)
    out_k_pd_surf = np.empty(ncells, dtype=np.float64)
    out_R_pd = np.empty(ncells, dtype=np.float64)

    if bool(skip_shielding):
        shielding_iter = 0

    min_rate = float(cfg.get('min_rate', 0.0))

    for _ in range(int(shielding_iter) + 1):
        if not bool(skip_shielding):
            theta_arr, chi_eff_arr = _compute_co_shielding(
                rad=rad,
                nH_cm3=nH_cm3,
                chi=chi_dim,
                nCO_cm3=nco_guess_cm3,
                nH2_cm3=nH2_cm3,
                nside=int(nside),
                b_kms=float(b_kms),
            )

        theta_flat = _flat_view(theta_arr)

        solve_carbon_reduced_steady_state_cgs(
            nH_flat,
            T_flat,
            chi_flat,
            theta_flat,
            tau_form_flat,
            sigma_flat,
            nH2_flat,
            Tg_flat,
            int(formation_model_flag),
            float(k0_nl97),
            float(k1_nl97),
            float(xO),
            float(gamma_chx0),
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

        nco_guess_cm3 = np.ascontiguousarray(out_nco_gas.reshape(orig_shape), dtype=np.float64)

    Xco_gas = Quantity(out_Xco_gas.reshape(orig_shape), 'dimensionless')
    nco_gas = Quantity(out_nco_gas.reshape(orig_shape), 'cm^-3')
    nco_ice = Quantity(out_nco_ice.reshape(orig_shape), 'cm^-3')
    k_pd = Quantity(out_k_pd.reshape(orig_shape), '1/s')
    tau_pd = Quantity(out_tau_pd.reshape(orig_shape), 's')

    theta_co = Quantity(theta_arr, 'dimensionless')
    chi_eff = Quantity(chi_eff_arr, 'dimensionless')

    diag = {
        'n_ice_act_max': Quantity(out_n_ice_act_max.reshape(orig_shape), 'cm^-3'),
        'n_ice_act': Quantity(out_n_ice_act.reshape(orig_shape), 'cm^-3'),
        'k_pd_surf': Quantity(out_k_pd_surf.reshape(orig_shape), '1/s'),
        'R_pd': Quantity(out_R_pd.reshape(orig_shape), 'cm^-3/s'),
    }

    nco_total_cm3 = out_nco_gas + out_nco_ice
    if rad.gas_temperature is not None:
        Tg_K = _as_cgs_f64(rad.gas_temperature, 'K')
        logger.info("Carbon closure using gas temperature")
    else:
        Tg_K = T_K
        logger.info("Carbon closure using dust temperature (gas T not available)")

    nCplus_flat, nC_flat, ne_flat = _carbon_closure_cgs(
        nH_cm3=nH_cm3,
        chi=chi_dim,
        Tg_K=Tg_K,
        nco_total_cm3=nco_total_cm3.reshape(orig_shape),
    )
    nCplus = Quantity(nCplus_flat.reshape(orig_shape), 'cm^-3')
    nC = Quantity(nC_flat.reshape(orig_shape), 'cm^-3')
    ne = Quantity(ne_flat.reshape(orig_shape), 'cm^-3')
    nco_total = Quantity(nco_total_cm3.reshape(orig_shape), 'cm^-3')

    validate_chemistry_state(
        nH=nH,
        nH2=rad.nH2,
        nHI=rad.nH_atom,
        nC=nC,
        nCplus=nCplus,
        nco_total=nco_total,
        check_h=True,
        check_c=True,
        check_pd=True,
        n_ice_act_max=diag.get('n_ice_act_max'),
        n_ice_act=diag.get('n_ice_act'),
        k_pd_surf=diag.get('k_pd_surf'),
        R_pd=diag.get('R_pd'),
    )
    
    rad.nco_gas = nco_gas
    rad.nco_ice = nco_ice
    rad.theta_co = theta_co
    rad.chi_eff = chi_eff
    rad.k_diss_co = k_pd
    rad.tau_diss_co = tau_pd
    rad.nCplus = nCplus
    rad.nC = nC
    rad.ne = ne

    return ChemistryResult(
        abundances={'co': Xco_gas},
        number_densities={
            'co': nco_gas,
            'h2': rad.nH2,
            'h': rad.nH_atom,
            'e': ne,
            'c+': nCplus,
            'catom': nC,
        },
        fields={
            'k_diss_co': k_pd,
            'tau_diss_co': tau_pd,
            'theta_co': theta_co,
            'chi_eff': chi_eff,
            'co_ice': nco_ice,
            'n_ice_act_max': diag.get('n_ice_act_max'),
            'n_ice_act': diag.get('n_ice_act'),
            'k_pd_surf': diag.get('k_pd_surf'),
            'R_pd': diag.get('R_pd'),
        },
        meta={'model': 'carbon_reduced'},
    )


def run_time_dependent(rad: 'RadModel', config: dict) -> ChemistryResult:
    """Run time-dependent CO two-phase chemistry.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    config : dict
        Configuration with keys:
        - t_end: Quantity (required, scalar or array for per-cell time)
        - dt: Quantity (optional)
        - Xco_tot: float (default: from diskbridge.params.abundance)
        - tau_form: Quantity (optional; if absent uses diskbridge.params.co_tau_form_model)
        - skip_shielding: bool (default: False)
        - nside: int (default: 4)
        - b_kms: float (default: 0.3)
        - Xco_gas_init: float or array (optional, scalar or per-cell IC)
        - Xco_ice_init: float or array (optional, scalar or per-cell IC)
        
    Returns
    -------
    ChemistryResult
        Result with abundances, number_densities, and diagnostic fields
    """
    if 't_end' not in config:
        raise ValueError("t_end is required for time-dependent CO chemistry")
    
    t_end = config['t_end']
    dt = config.get('dt', None)
    
    Xco_gas_init = config.get('Xco_gas_init', None)
    if Xco_gas_init is not None and hasattr(Xco_gas_init, 'magnitude'):
        Xco_gas_init = Xco_gas_init.magnitude
    
    Xco_ice_init = config.get('Xco_ice_init', None)
    if Xco_ice_init is not None and hasattr(Xco_ice_init, 'magnitude'):
        Xco_ice_init = Xco_ice_init.magnitude

    cfg = resolve_model_config(("chemistry", "carbon_reduced"), overrides=config)
    formation_model = str(cfg.get('formation_model', 'tau')).lower()
    if formation_model == 'nl97':
        raise ValueError("formation_model='nl97' is not supported in time-dependent carbon_reduced")

    nH, Tdust, chi, tau_form_s, Xco_tot, skip_shielding, nside, b_kms, shielding_iter = _prepare_common(rad, cfg)
    tau_form = Quantity(tau_form_s, 's')

    if int(shielding_iter) != 0:
        raise ValueError("shielding_iter must be 0 for time-dependent carbon_reduced")

    if bool(skip_shielding):
        theta_co = Quantity(np.ones_like(chi.magnitude), 'dimensionless')
        chi_eff = chi
    else:
        nco_guess = Quantity(float(Xco_tot) * nH.to('cm^-3').magnitude, 'cm^-3')
        if getattr(rad, 'nco_gas', None) is not None:
            nco_guess = rad.nco_gas

        theta_co, chi_eff = _compute_co_shielding(
            rad=rad,
            nH_cm3=_as_cgs_f64(nH, 'cm^-3'),
            chi=_as_cgs_f64(chi, 'dimensionless'),
            nCO_cm3=_as_cgs_f64(nco_guess, 'cm^-3'),
            nH2_cm3=_as_cgs_f64(rad.nH2, 'cm^-3'),
            nside=nside,
            b_kms=b_kms,
        )

        theta_co = Quantity(theta_co, 'dimensionless')
        chi_eff = Quantity(chi_eff, 'dimensionless')

    sigma_d_per_H = rad.ensure_sigma_d_per_H()

    Xco_gas, nco_gas, nco_ice, k_pd, tau_pd, diag = evolve_carbon_reduced_time_dependent(
        nH=nH,
        T=Tdust,
        chi=chi,
        theta_co=theta_co,
        t_end=t_end,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form,
        sigma_d_per_H=sigma_d_per_H,
        Xco_gas_init=Xco_gas_init,
        Xco_ice_init=Xco_ice_init,
        dt=dt,
    )

    nCplus, nC, ne, nco_total = _run_carbon_closure(
        rad=rad,
        nH=nH,
        chi=chi,
        Tdust=Tdust,
        nco_gas=nco_gas,
        nco_ice=nco_ice,
    )

    validate_chemistry_state(
        nH=nH,
        nH2=rad.nH2,
        nHI=rad.nH_atom,
        nC=nC,
        nCplus=nCplus,
        nco_total=nco_total,
        check_h=True,
        check_c=True,
        check_pd=True,
        n_ice_act_max=diag.get('n_ice_act_max'),
        n_ice_act=diag.get('n_ice_act'),
        k_pd_surf=diag.get('k_pd_surf'),
        R_pd=diag.get('R_pd'),
    )

    rad.nco_gas = nco_gas
    rad.nco_ice = nco_ice
    rad.theta_co = theta_co
    rad.chi_eff = chi_eff
    rad.k_diss_co = k_pd
    rad.tau_diss_co = tau_pd
    rad.nCplus = nCplus
    rad.nC = nC
    rad.ne = ne

    return ChemistryResult(
        abundances={'co': Xco_gas},
        number_densities={
            'co': nco_gas,
            'h2': rad.nH2,
            'h': rad.nH_atom,
            'e': ne,
            'c+': nCplus,
            'catom': nC,
        },
        fields={
            'k_diss_co': k_pd,
            'tau_diss_co': tau_pd,
            'theta_co': theta_co,
            'chi_eff': chi_eff,
            'co_ice': nco_ice,
            'n_ice_act_max': diag.get('n_ice_act_max'),
            'n_ice_act': diag.get('n_ice_act'),
            'k_pd_surf': diag.get('k_pd_surf'),
            'R_pd': diag.get('R_pd'),
        },
        meta={'model': 'carbon_reduced'},
    )


def compute_boundary_co_ic(
    rad: 'RadModel',
    *,
    r_boundary: Quantity,
    shell_dr: Optional[Quantity] = None,
    age: Quantity,
    base_config: Optional[dict] = None,
    reducer: str = "median",
) -> tuple[float, float]:
    """Compute boundary CO initial conditions from outer shell chemistry.
    
    Solves time-dependent chemistry in a thin shell near r_boundary for the
    specified age, then reduces to scalar abundances for use as ICs.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    r_boundary : Quantity
        Boundary radius
    shell_dr : Quantity, optional
        Shell thickness (default: 2 * dr at boundary)
    age : Quantity
        Evolution time for boundary chemistry
    base_config : dict, optional
        Base chemistry config (default: {})
    reducer : str, optional
        "median" or "mean" (default: "median")
        
    Returns
    -------
    xco_gas0 : float
        Boundary gas abundance (dimensionless)
    xco_ice0 : float
        Boundary ice abundance (dimensionless)
        
    Examples
    --------
    >>> xco_gas0, xco_ice0 = compute_boundary_co_ic(
    ...     rad,
    ...     r_boundary=Quantity(100, "au"),
    ...     age=Quantity(1e5, "yr"),
    ... )
    """
    from diskbridge._logging import logger
    
    if reducer not in ("median", "mean"):
        raise ValueError(f"Invalid reducer='{reducer}'")
    
    mesh = rad.model.mesh
    r_centers = mesh.centers('r')
    r_edges = mesh.edges('r')
    
    r_au = r_centers.to('au').magnitude
    r_edges_au = r_edges.to('au').magnitude
    r_boundary_au = r_boundary.to('au').magnitude
    
    r_idx = np.argmin(np.abs(r_au - r_boundary_au))
    
    if shell_dr is None:
        if r_idx < len(r_edges_au) - 1:
            dr = r_edges_au[r_idx + 1] - r_edges_au[r_idx]
        else:
            dr = r_edges_au[r_idx] - r_edges_au[r_idx - 1]
        shell_dr = Quantity(2.0 * dr, 'au')
    
    shell_dr_au = shell_dr.to('au').magnitude
    shell_mask = np.abs(r_au - r_boundary_au) <= shell_dr_au / 2.0
    shell_mask_3d = shell_mask[:, None, None]
    
    if not np.any(shell_mask):
        raise ValueError(f"No cells in shell at r={r_boundary_au:.3g} AU")
    
    if base_config is None:
        base_config = {}
    
    config = base_config.copy()
    config['t_end'] = age
    config['dt'] = None
    
    result = run_time_dependent(rad, config)
    
    nH = rad.ensure_nH()
    if hasattr(nH, 'magnitude'):
        nH_cm3 = nH.to('cm^-3').magnitude
    else:
        nH_cm3 = nH
    
    nco_gas = result.number_densities['co']
    nco_gas_cm3 = nco_gas.to('cm^-3').magnitude

    nco_ice = result.fields['co_ice']
    nco_ice_cm3 = nco_ice.to('cm^-3').magnitude
    
    xco_gas = nco_gas_cm3 / (nH_cm3 + 1e-99)
    xco_ice = nco_ice_cm3 / (nH_cm3 + 1e-99)
    
    xco_gas_shell = xco_gas[shell_mask_3d]
    xco_ice_shell = xco_ice[shell_mask_3d]
    
    if reducer == "median":
        xco_gas0 = float(np.median(xco_gas_shell))
        xco_ice0 = float(np.median(xco_ice_shell))
    else:
        xco_gas0 = float(np.mean(xco_gas_shell))
        xco_ice0 = float(np.mean(xco_ice_shell))
    
    logger.info(
        f"Boundary IC: r={r_boundary_au:.3g} AU, "
        f"Xco_gas={xco_gas0:.3e}, Xco_ice={xco_ice0:.3e}"
    )
    
    return xco_gas0, xco_ice0


def run_carbon_reduced(rad: 'RadModel', config: dict) -> ChemistryResult:
    if 't_end' in config:
        return run_time_dependent(rad, config)
    return run_steady(rad, config)


__all__ = [
    'run_carbon_reduced',
]
