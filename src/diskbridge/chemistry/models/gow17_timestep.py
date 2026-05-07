from __future__ import annotations

import numpy as np

import diskbridge
import diskbridge._gow17 as _gow17

from diskbridge._config import get_config, resolve_model_config
from diskbridge._units import Quantity
from diskbridge._constants import (
    E_BIND_CO,
    NU0_CO,
    N_LAY,
    N_SURF,
    Y_CO,
)

from diskbridge.chemistry.models.gow17 import (
    N_Y,
    N_PH,
    IPH_C,
    IPH_CO,
    IPH_H2,
    XC_STD,
    XO_STD,
    XHE,
    I_HEP,
    I_OHX,
    I_CHX,
    I_CO,
    I_CO_ICE,
    I_CP,
    I_HCOP,
    I_H2,
    I_HP,
    I_H3P,
    I_H2P,
    I_SP,
    I_SIP,
    I_OP,
    I_E,
    _cv_cold,
    _electron_abundance,
    _as_cgs_f64,
    _broadcast_scalar_or_array,
    _actual_solver_uv_fields,
    _build_gow17_abstol,
    _compute_shielding_and_gph,
    _co_pdes_draine_flux,
    _maybe_quantity_to_float,
    _resolve_co_dust_scalings,
    _resolve_co_phase_controls,
    _resolve_co_phase_runtime_params,
    _resolve_local_chi_factor,
    _resolve_shielding_linewidth,
    _resolve_temperature_config,
    _resolve_visser_table_linewidth,
    _warn_if_co_phase_settings_ignored,
)
from diskbridge.chemistry.validation import project_gow17_state_to_budgets
from diskbridge.radmc3d.uv_products import validate_uv_chemistry_config

KB_CGS = 1.380649e-16
YR_TO_S = 365.25 * 24.0 * 3600.0


def _default_y0_single() -> np.ndarray:
    y0 = np.zeros(N_Y, dtype=np.float64)
    y0[I_HEP] = 1.45e-08
    y0[I_H3P] = 2.68e-07
    y0[I_CP] = 1.0e-4
    y0[I_CO] = 1.0e-7
    y0[I_H2] = 0.1
    y0[I_CO_ICE] = 0.0
    return y0


class Gow17TimeStepper:
    """
    Advance GOW17 chemistry (and optionally energy) by a single dt with
    shielding re-evaluated from the current state.

    This is designed for external time-dependent problems where chi/Tdust/etc
    change each hydro step.
    """

    def __init__(self, rad, config: dict):
        cfg = resolve_model_config(("chemistry", "gow17"), overrides=config)
        self.cfg = cfg
        self.runtime_mode = None

        self.temperature_cfg = _resolve_temperature_config(cfg)
        self.temperature_mode = str(self.temperature_cfg["mode"])
        self.const_temp = bool(self.temperature_cfg["const_temp"])
        if self.temperature_mode == "constant_debug":
            import logging

            logging.getLogger(__name__).warning(
                "Gow17TimeStepper: using constant_debug Tgas; this is for "
                "debugging only and is not a physically valid "
                "disc/PDR temperature model."
            )
        self.enable_co_phase = bool(cfg.get("enable_co_phase", True))
        _warn_if_co_phase_settings_ignored(
            cfg,
            enable_co_phase=self.enable_co_phase,
            context="Gow17TimeStepper",
        )
        self.chi_is_incident = bool(cfg.get("chi_is_incident", False))
        self.incident_uv_products = bool(cfg.get("incident_uv_products", False))
        if self.chi_is_incident:
            if getattr(rad, "Av", None) is None:
                raise ValueError("Gow17TimeStepper: chi_is_incident=True requires rad.Av to be set")
            if rad.has_uv_product("G_CO_diss") and not self.incident_uv_products:
                raise ValueError(
                    "Gow17TimeStepper: chi_is_incident=True is incompatible "
                    "with registered UV product fields unless "
                    "incident_uv_products=True"
                )
        else:
            full_cfg = dict(get_config())
            chemistry_cfg = dict(full_cfg.get("chemistry", {}))
            chemistry_cfg["gow17"] = dict(cfg)
            full_cfg["chemistry"] = chemistry_cfg
            self.runtime_mode = validate_uv_chemistry_config(full_cfg)

        self.reltol = float(cfg.get("reltol", 1e-4))
        self.abstol0 = float(cfg.get("abstol0", 1e-15))
        self.tolfac = float(cfg.get("tolfac", 10.0))
        self.mxsteps = int(cfg.get("mxsteps", 10000))
        self.maxord = int(cfg.get("maxord", 5))
        self.userJac = bool(cfg.get("userJac", False))
        self.verbose = bool(cfg.get("verbose", False))

        self.b_kms = float(cfg.get("b_kms", 0.3))
        self.ion_rate_s = Quantity(cfg.get("ion_rate", "2e-16 s^-1")).to("1/s").magnitude

        self.Zg = float(cfg.get("Zg", 1.0))
        self.gradv_scalar = float(cfg.get("gradv", 1.0e-14))
        self.Leff_CO_max_scalar = float(cfg.get("Leff_CO_max", 3.0e20))
        self.isDust_cooling = bool(cfg.get("isDust_cooling", True))
        self.isCoolingCOThin = bool(cfg.get("isCoolingCOThin", False))

        self.local_chi_factor = _resolve_local_chi_factor(
            cfg,
            chi_is_incident=self.chi_is_incident,
        )
        self.shielding_outer_1d = str(cfg.get("shielding_outer_1d", "min"))

        self.rad = rad
        self.nside = int(diskbridge.params.nside)

        nH = rad.ensure_nH()
        chi = rad.ensure_chi()
        Tdust = rad.ensure_dust_temperature()
        if self.temperature_mode == "constant_debug":
            Tgas = Quantity(
                np.full(
                    _as_cgs_f64(Tdust, "K").shape,
                    float(self.temperature_cfg["constant_debug_Tgas"]),
                    dtype=np.float64,
                ),
                "K",
            )
        elif self.temperature_mode == "dust":
            Tgas = Tdust
        else:
            Tgas = rad.ensure_gas_temperature()
            if Tgas is None:
                Tgas = Tdust

        self.shape = _as_cgs_f64(nH, "cm^-3").shape
        self.ncells = int(np.prod(self.shape))

        self.nH_cm3 = _as_cgs_f64(nH, "cm^-3")
        self.nH_flat = self.nH_cm3.reshape(self.ncells)
        T_init = np.clip(
            _as_cgs_f64(Tgas, "K"),
            float(self.temperature_cfg["Tgas_floor"]),
            float(self.temperature_cfg["Tgas_ceiling"]),
        )
        self.b_kms, self.b_CO_kms_arr, self.shielding_linewidth_meta = _resolve_shielding_linewidth(
            cfg,
            T_init,
        )

        self.abstol = _build_gow17_abstol(cfg, self.abstol0)

        sigma_d = rad.ensure_sigma_d_per_H()
        sigma_d_cm2 = _as_cgs_f64(sigma_d, "cm^2").reshape(self.ncells)

        self.Zg_arr = _broadcast_scalar_or_array(self.Zg, self.ncells)
        self.ion_rate_arr = _broadcast_scalar_or_array(self.ion_rate_s, self.ncells)
        self.sigma_d_CO_per_H, self.Zd_arr, self.sigma_d_ISM_ref = _resolve_co_dust_scalings(
            cfg=cfg,
            sigma_d_cm2=sigma_d_cm2,
            ncells=self.ncells,
        )
        self.S_CO, self.F_CRUV_CO_pdes_arr, self.k_crdes_CO_arr = _resolve_co_phase_controls(
            cfg=cfg,
            ion_rate_arr=self.ion_rate_arr,
            ncells=self.ncells,
        )
        self.gradv_arr = _broadcast_scalar_or_array(self.gradv_scalar, self.ncells)
        self.Leff_CO_max_arr = _broadcast_scalar_or_array(self.Leff_CO_max_scalar, self.ncells)

        self.xCtot_flat = self.Zg_arr * float(XC_STD)
        self.xOtot_flat = self.Zg_arr * float(XO_STD)

        self.Av_flat = None
        if self.chi_is_incident:
            Av_q = getattr(rad, "Av", None)
            Av_arr = _as_cgs_f64(Av_q, "dimensionless")
            self.Av_flat = Av_arr.reshape(self.ncells)

        self.b_kms, self.visser, self.shielding_linewidth_meta = _resolve_visser_table_linewidth(
            self.b_kms,
            self.shielding_linewidth_meta,
        )
        self.co_phase_params = _resolve_co_phase_runtime_params(cfg)

        zero_cell = np.zeros(self.ncells, dtype=np.float64)
        self._co_phase_kw = dict(
            co_E_bind_co=(float(self.co_phase_params["E_bind_CO"]) if self.enable_co_phase else 0.0),
            co_nu0_co=(float(self.co_phase_params["nu0_CO"]) if self.enable_co_phase else 0.0),
            co_Y_CO=(float(self.co_phase_params["Y_CO"]) if self.enable_co_phase else 0.0),
            co_N_SURF=(float(self.co_phase_params["N_SURF"]) if self.enable_co_phase else 0.0),
            co_N_LAY=(int(self.co_phase_params["N_LAY"]) if self.enable_co_phase else 0),
            co_S_CO=(float(self.S_CO) if self.enable_co_phase else 0.0),
            co_F_CRUV_CO_pdes=(
                np.ascontiguousarray(self.F_CRUV_CO_pdes_arr, dtype=np.float64)
                if self.enable_co_phase
                else zero_cell
            ),
            co_k_crdes_CO=(
                np.ascontiguousarray(self.k_crdes_CO_arr, dtype=np.float64)
                if self.enable_co_phase
                else zero_cell
            ),
        )

        y_prev = getattr(rad, "gow17_y", None)
        if y_prev is not None and np.shape(y_prev) == tuple(self.shape) + (N_Y,):
            self.y_state = np.ascontiguousarray(np.asarray(y_prev, dtype=np.float64).reshape(self.ncells, N_Y))
        else:
            y0 = _default_y0_single()
            self.y_state = np.tile(y0[None, :], (self.ncells, 1)).astype(np.float64)

        if not self.enable_co_phase:
            self.y_state[:, I_CO_ICE] = 0.0
        self.y_state[:, :] = self._project_state(self.y_state)

        if not self.const_temp and (y_prev is None):
            T_flat = np.clip(
                _as_cgs_f64(Tgas, "K"),
                float(self.temperature_cfg["Tgas_floor"]),
                float(self.temperature_cfg["Tgas_ceiling"]),
            ).reshape(self.ncells)
            xe0 = _electron_abundance(self.y_state)
            Cv0 = _cv_cold(self.y_state[:, I_H2], xe0)
            self.y_state[:, I_E] = Cv0 * T_flat

        self.eq_tmin_s = _maybe_quantity_to_float(cfg.get("tmin", 3.16e10), "s")
        tmax_val = cfg.get("tmax", None)
        if tmax_val is None:
            tmax_val = cfg.get("t_end", 3.16e14)
        self.eq_tmax_s = _maybe_quantity_to_float(tmax_val, "s")
        self.astrochem_n_updates = int(cfg.get("astrochem_n_updates", 5))
        self.astrochem_t_end_s = float(cfg.get("astrochem_t_end_yr", 1.0e6)) * YR_TO_S
        self.shielding_max_iter = int(cfg.get("shielding_max_iter", max(self.astrochem_n_updates, 50)))
        self.shielding_reltol = float(cfg.get("shielding_reltol", 1.0e-3))
        self.shielding_abstol = float(cfg.get("shielding_abstol", 1.0e-15))

    def _project_state(self, y: np.ndarray) -> np.ndarray:
        y_proj = project_gow17_state_to_budgets(
            np.asarray(y, dtype=np.float64),
            xCtot=self.xCtot_flat,
            xOtot=self.xOtot_flat,
        )
        if not self.enable_co_phase:
            y_proj[:, I_CO_ICE] = 0.0
        return np.ascontiguousarray(y_proj, dtype=np.float64)

    def _prepare_environment(
        self,
        *,
        y_state: np.ndarray | None = None,
    ):
        rad = self.rad
        if y_state is None:
            y_state = self.y_state
        y_state = self._project_state(y_state)

        if self.chi_is_incident:
            if self.incident_uv_products:
                chi_broad = rad.ensure_uv_product("chi_broad", fallback_to_chi=False)
                G_CO_diss = rad.ensure_uv_product("G_CO_diss", fallback_to_chi=False)
                G_H2_diss = rad.ensure_uv_product("G_H2_diss", fallback_to_chi=False)
                G_C_ion = rad.ensure_uv_product("G_C_ion", fallback_to_chi=False)
                G_CO_pdes = rad.ensure_uv_product("G_CO_pdes", fallback_to_chi=False)
                F_CO_pdes_photon = rad.ensure_uv_product(
                    "F_CO_pdes_photon",
                    fallback_to_chi=False,
                )
            else:
                chi_broad = rad.ensure_chi()
                G_CO_diss = chi_broad
                G_H2_diss = chi_broad
                G_C_ion = chi_broad
                G_CO_pdes = chi_broad
                F_CO_pdes_photon = None
        elif self.runtime_mode is not None and self.runtime_mode.products_enabled:
            chi_broad = rad.ensure_uv_product("chi_broad", fallback_to_chi=False)
            G_CO_diss = rad.ensure_uv_product("G_CO_diss", fallback_to_chi=False)
            G_H2_diss = rad.ensure_uv_product("G_H2_diss", fallback_to_chi=False)
            G_C_ion = rad.ensure_uv_product("G_C_ion", fallback_to_chi=False)
            G_CO_pdes = rad.ensure_uv_product("G_CO_pdes", fallback_to_chi=False)
            F_CO_pdes_photon = rad.ensure_uv_product(
                "F_CO_pdes_photon",
                fallback_to_chi=False,
            )
        else:
            chi_broad = rad.ensure_chi()
            G_CO_diss = chi_broad
            G_H2_diss = chi_broad
            G_C_ion = chi_broad
            G_CO_pdes = chi_broad
            F_CO_pdes_photon = None

        chi_dust_arr = _as_cgs_f64(chi_broad, "dimensionless")
        chi_dust_flat = chi_dust_arr.reshape(self.ncells)
        G_CO_diss_flat = _as_cgs_f64(G_CO_diss, "dimensionless").reshape(self.ncells)
        G_H2_diss_flat = _as_cgs_f64(G_H2_diss, "dimensionless").reshape(self.ncells)
        G_C_ion_flat = _as_cgs_f64(G_C_ion, "dimensionless").reshape(self.ncells)
        G_CO_pdes_flat = _as_cgs_f64(G_CO_pdes, "dimensionless").reshape(self.ncells)
        if F_CO_pdes_photon is None:
            F_CO_pdes_photon_flat = (
                G_CO_pdes_flat * _co_pdes_draine_flux()
            )
        else:
            F_CO_pdes_photon_flat = _as_cgs_f64(
                F_CO_pdes_photon,
                "1/(cm^2 s)",
            ).reshape(self.ncells)

        Tdust_flat = _as_cgs_f64(rad.ensure_dust_temperature(), "K").reshape(self.ncells)

        if self.temperature_mode == "constant_debug":
            Tgas = Quantity(
                np.full(
                    self.shape,
                    float(self.temperature_cfg["constant_debug_Tgas"]),
                    dtype=np.float64,
                ),
                "K",
            )
        elif self.temperature_mode == "dust":
            Tgas = rad.ensure_dust_temperature()
        else:
            Tgas = rad.ensure_gas_temperature()
            if Tgas is None:
                Tgas = rad.ensure_dust_temperature()
        T_flat = np.clip(
            _as_cgs_f64(Tgas, "K"),
            float(self.temperature_cfg["Tgas_floor"]),
            float(self.temperature_cfg["Tgas_ceiling"]),
        ).reshape(self.ncells)
        self.b_kms, self.b_CO_kms_arr, self.shielding_linewidth_meta = _resolve_shielding_linewidth(
            self.cfg,
            T_flat.reshape(self.shape),
        )
        self.b_kms, self.visser, self.shielding_linewidth_meta = _resolve_visser_table_linewidth(
            self.b_kms,
            self.shielding_linewidth_meta,
        )

        theta_h2, theta_co, theta_c, Gph, GPE, GISRF = _compute_shielding_and_gph(
            y_flat=y_state,
            nH_flat=self.nH_flat,
            chi_dust_flat=chi_dust_flat,
            G_CO_diss_flat=G_CO_diss_flat,
            G_H2_diss_flat=G_H2_diss_flat,
            G_C_ion_flat=G_C_ion_flat,
            G_CO_pdes_flat=G_CO_pdes_flat,
            F_CO_pdes_photon_flat=F_CO_pdes_photon_flat,
            xCtot_flat=self.xCtot_flat,
            Zd_arr=self.Zd_arr,
            Av_flat=self.Av_flat,
            shape=self.shape,
            ncells=self.ncells,
            rad=rad,
            nH_cm3=self.nH_cm3,
            chi_dust_arr=chi_dust_arr,
            visser=self.visser,
            b_kms=self.b_kms,
            nside=self.nside,
            chi_is_incident=self.chi_is_incident,
            local_chi_factor=self.local_chi_factor,
            shielding_outer_1d=self.shielding_outer_1d,
        )
        return chi_dust_arr, Tdust_flat, T_flat, theta_h2, theta_co, theta_c, Gph, GPE, GISRF

    def _commit_solution(
        self,
        *,
        y_new: np.ndarray,
        status: np.ndarray,
        theta_h2: np.ndarray,
        theta_co: np.ndarray,
        theta_c: np.ndarray,
        chi_dust_arr: np.ndarray,
        Gph: np.ndarray,
        GISRF: np.ndarray,
    ) -> dict:
        rad = self.rad

        self.y_state[:, :] = self._project_state(y_new)
        status = np.asarray(status, dtype=np.int32)

        y_out = self.y_state.reshape(self.shape + (N_Y,))
        xCO = y_out[..., I_CO]
        xCO_ice = y_out[..., I_CO_ICE]
        xH2 = y_out[..., I_H2]

        nco_gas = Quantity(xCO * self.nH_cm3, "cm^-3")
        nco_ice = Quantity(xCO_ice * self.nH_cm3, "cm^-3")
        nH2_out = Quantity(xH2 * self.nH_cm3, "cm^-3")

        xe = _electron_abundance(y_out)

        xH_atom = 1.0 - (
            y_out[..., I_OHX]
            + y_out[..., I_CHX]
            + y_out[..., I_HCOP]
            + 3.0 * y_out[..., I_H3P]
            + 2.0 * y_out[..., I_H2P]
            + y_out[..., I_HP]
            + 2.0 * y_out[..., I_H2]
        )
        xH_atom = np.maximum(xH_atom, 0.0)

        xCtot = self.Zg_arr.reshape(self.shape) * float(XC_STD)
        xC_neutral = xCtot - (
            y_out[..., I_HCOP]
            + y_out[..., I_CHX]
            + y_out[..., I_CO]
            + y_out[..., I_CO_ICE]
            + y_out[..., I_CP]
        )
        xC_neutral = np.maximum(xC_neutral, 0.0)

        rad.gow17_y = y_out
        rad.nco_gas = nco_gas
        rad.nco_ice = nco_ice
        rad.nH2 = nH2_out
        rad.nH_atom = Quantity(xH_atom * self.nH_cm3, "cm^-3")
        rad.nCplus = Quantity(y_out[..., I_CP] * self.nH_cm3, "cm^-3")
        rad.nC = Quantity(xC_neutral * self.nH_cm3, "cm^-3")
        rad.ne = Quantity(xe * self.nH_cm3, "cm^-3")

        theta_co_arr = np.asarray(theta_co, dtype=np.float64).reshape(self.shape)
        theta_h2_arr = np.asarray(theta_h2, dtype=np.float64).reshape(self.shape)
        theta_c_arr = np.asarray(theta_c, dtype=np.float64).reshape(self.shape)

        rad.theta_co = Quantity(theta_co_arr, "dimensionless")
        rad.theta_h2 = Quantity(theta_h2_arr, "dimensionless")
        rad.theta_c = Quantity(theta_c_arr, "dimensionless")
        actual_uv = _actual_solver_uv_fields(
            Gph=Gph,
            F_CO_pdes_external=GISRF,
            shape=self.shape,
        )
        rad.G_CO_diss_actual = Quantity(actual_uv["G_CO_diss_actual"], "dimensionless")
        rad.G_C_ion_actual = Quantity(actual_uv["G_C_ion_actual"], "dimensionless")
        rad.G_H2_diss_actual = Quantity(actual_uv["G_H2_diss_actual"], "dimensionless")
        rad.F_CO_pdes_external_actual = Quantity(
            actual_uv["F_CO_pdes_external_actual"],
            "1/(cm^2 s)",
        )
        rad.chi_eff = rad.G_CO_diss_actual

        if not self.const_temp:
            xe_out = _electron_abundance(y_out)
            Cv_out = _cv_cold(y_out[..., I_H2], xe_out)
            T_out = y_out[..., I_E] / Cv_out
            T_out = np.clip(
                T_out,
                float(self.temperature_cfg["Tgas_floor"]),
                float(self.temperature_cfg["Tgas_ceiling"]),
            )
            rad.Tgas_gow17 = Quantity(T_out, "K")
            rad.gas_temperature = rad.Tgas_gow17
        rad.b_CO_kms = Quantity(self.b_CO_kms_arr.reshape(self.shape), "km/s")

        return {
            "status": status,
            "theta_co": theta_co_arr,
            "theta_h2": theta_h2_arr,
            "theta_c": theta_c_arr,
            "b_CO_kms": self.b_CO_kms_arr.reshape(self.shape),
            "shielding_linewidth": dict(self.shielding_linewidth_meta),
            "co_phase_params": dict(self.co_phase_params),
        }

    def _repair_failed_cells(self, y_new: np.ndarray, status: np.ndarray) -> np.ndarray:
        y = np.asarray(y_new, dtype=np.float64).copy()
        status_arr = np.asarray(status, dtype=np.int32).reshape(-1)
        if y.ndim != 2 or y.shape[0] != self.ncells:
            raise ValueError("Gow17TimeStepper._repair_failed_cells: y_new must have shape (ncells, N_Y)")

        failed = status_arr != 0
        if not np.any(failed):
            return y
        if np.all(failed):
            return y

        i = 0
        while i < self.ncells:
            if not failed[i]:
                i += 1
                continue

            j = i
            while j + 1 < self.ncells and failed[j + 1]:
                j += 1

            left = i - 1 if i > 0 and not failed[i - 1] else None
            right = j + 1 if (j + 1) < self.ncells and not failed[j + 1] else None

            if left is not None and right is not None:
                span = float(right - left)
                for k in range(i, j + 1):
                    w = float(k - left) / span
                    y[k, :] = (1.0 - w) * y[left, :] + w * y[right, :]
            elif left is not None:
                y[i : j + 1, :] = y[left, :]
            elif right is not None:
                y[i : j + 1, :] = y[right, :]

            i = j + 1

        return y

    def step(self, dt_s: float) -> dict:
        """
        Advance by dt_s seconds. Assumes caller has already updated:
          - rad.chi (dimensionless, incident scaling if chi_is_incident=True)
          - rad.dust_temperature (K)
          - rad.gas_temperature (K) if const_temp=True (or initial guess)
        """
        chi_dust_arr, Tdust_flat, T_flat, theta_h2, theta_co, theta_c, Gph, GPE, GISRF = self._prepare_environment()

        result = _gow17.solve_batch_time(
            y0=self._project_state(self.y_state),
            nH=self.nH_flat,
            Tgas=T_flat,
            Tdust=Tdust_flat,
            Zd=self.Zd_arr,
            Zg=self.Zg_arr,
            ion_rate=self.ion_rate_arr,
            GPE=GPE,
            F_CO_pdes_photon=GISRF,
            Gph=Gph,
            sigma_d_CO_per_H=(
                self.sigma_d_CO_per_H
                if self.enable_co_phase
                else np.zeros(self.ncells, dtype=np.float64)
            ),
            reltol=self.reltol,
            abstol=self.abstol,
            mxsteps=self.mxsteps,
            maxord=self.maxord,
            t_end=float(dt_s),
            const_temp=self.const_temp,
            gradv=self.gradv_arr,
            Leff_CO_max=self.Leff_CO_max_arr,
            isDust_cooling=self.isDust_cooling,
            isCoolingCOThin=self.isCoolingCOThin,
            fH2gr=float(1.0),
            fHplusgr=float(1.0),
            fCplusgr=float(1.0),
            fHeplusgr=float(1.0),
            fSplusgr=float(1.0),
            fSiplusgr=float(1.0),
            fCplusCR=float(1.0),
            **self._co_phase_kw,
            userJac=self.userJac,
            verbose=self.verbose,
        )
        y_new = self._project_state(self._repair_failed_cells(result["y"], result["status"]))
        return self._commit_solution(
            y_new=y_new,
            status=np.asarray(result["status"], dtype=np.int32),
            theta_h2=theta_h2,
            theta_co=theta_co,
            theta_c=theta_c,
            chi_dust_arr=chi_dust_arr,
            Gph=Gph,
            GISRF=GISRF,
        )

    def solve_equilibrium(self) -> dict:
        N = self.astrochem_n_updates
        if N < 1:
            raise ValueError("Gow17TimeStepper.solve_equilibrium: astrochem_n_updates must be >= 1")
        max_iter = self.shielding_max_iter
        if max_iter < N:
            raise ValueError("Gow17TimeStepper.solve_equilibrium: shielding_max_iter must be >= astrochem_n_updates")

        y_state = self._project_state(self.y_state.copy())

        if N == 1:
            t_targets = np.array([self.astrochem_t_end_s], dtype=np.float64)
        else:
            r = 10.0 ** (1.0 / (N - 1))
            k = np.arange(1, N + 1, dtype=np.float64)
            t_targets = self.astrochem_t_end_s * (r**k - 1.0) / (r**N - 1.0)
            t_targets[-1] = self.astrochem_t_end_s
        dt = np.empty(N, dtype=np.float64)
        dt[0] = t_targets[0]
        dt[1:] = np.diff(t_targets)

        t_schedule = np.empty(max_iter, dtype=np.float64)
        if max_iter == N:
            t_schedule[:] = t_targets
        else:
            r_full = 10.0 ** (1.0 / (max_iter - 1))
            k_full = np.arange(1, max_iter + 1, dtype=np.float64)
            t_schedule[:] = self.astrochem_t_end_s * (r_full**k_full - 1.0) / (r_full**max_iter - 1.0)
            t_schedule[-1] = self.astrochem_t_end_s
        dt = np.empty(max_iter, dtype=np.float64)
        dt[0] = t_schedule[0]
        dt[1:] = np.diff(t_schedule)

        for k_step in range(max_iter):
            xCO_old = np.ascontiguousarray(y_state[:, I_CO].copy(), dtype=np.float64)
            xH2_old = np.ascontiguousarray(y_state[:, I_H2].copy(), dtype=np.float64)
            _, Tdust_flat, T_flat, theta_h2, theta_co, theta_c, Gph, GPE, GISRF = self._prepare_environment(
                y_state=y_state,
            )
            result_time = _gow17.solve_batch_time(
                    y0=self._project_state(y_state),
                nH=self.nH_flat,
                Tgas=T_flat,
                Tdust=Tdust_flat,
                Zd=self.Zd_arr,
                Zg=self.Zg_arr,
                ion_rate=self.ion_rate_arr,
                GPE=GPE,
                F_CO_pdes_photon=GISRF,
                Gph=Gph,
                sigma_d_CO_per_H=(
                    self.sigma_d_CO_per_H
                    if self.enable_co_phase
                    else np.zeros(self.ncells, dtype=np.float64)
                ),
                reltol=self.reltol,
                abstol=self.abstol,
                mxsteps=self.mxsteps,
                maxord=self.maxord,
                t_end=float(dt[k_step]),
                const_temp=self.const_temp,
                gradv=self.gradv_arr,
                Leff_CO_max=self.Leff_CO_max_arr,
                isDust_cooling=self.isDust_cooling,
                isCoolingCOThin=self.isCoolingCOThin,
                fH2gr=float(1.0),
                fHplusgr=float(1.0),
                fCplusgr=float(1.0),
                fHeplusgr=float(1.0),
                fSplusgr=float(1.0),
                fSiplusgr=float(1.0),
                fCplusCR=float(1.0),
                **self._co_phase_kw,
                userJac=self.userJac,
                verbose=self.verbose,
            )
            y_state[:, :] = self._project_state(
                self._repair_failed_cells(result_time["y"], result_time["status"])
            )

            xCO_new = y_state[:, I_CO]
            xH2_new = y_state[:, I_H2]
            denom_h2 = np.maximum(np.abs(xH2_new), self.shielding_abstol)
            denom_co = np.maximum(np.abs(xCO_new), self.shielding_abstol)
            d_h2 = float(np.max(np.abs(xH2_new - xH2_old) / denom_h2))
            d_co = float(np.max(np.abs(xCO_new - xCO_old) / denom_co))
            if (k_step + 1) >= N and max(d_h2, d_co) <= self.shielding_reltol:
                break

        chi_dust_arr, Tdust_flat, T_flat, theta_h2, theta_co, theta_c, Gph, GPE, GISRF = self._prepare_environment(
            y_state=y_state,
        )
        result = _gow17.solve_batch_equilibrium(
            y0=self._project_state(y_state),
            nH=self.nH_flat,
            Tgas=T_flat,
            Tdust=Tdust_flat,
            Zd=self.Zd_arr,
            Zg=self.Zg_arr,
            ion_rate=self.ion_rate_arr,
            GPE=GPE,
            F_CO_pdes_photon=GISRF,
            Gph=Gph,
            sigma_d_CO_per_H=(
                self.sigma_d_CO_per_H
                if self.enable_co_phase
                else np.zeros(self.ncells, dtype=np.float64)
            ),
            reltol=self.reltol,
            abstol=self.abstol,
            mxsteps=self.mxsteps,
            maxord=self.maxord,
            tolfac=self.tolfac,
            tmin=self.eq_tmin_s,
            tmax=self.eq_tmax_s,
            const_temp=self.const_temp,
            gradv=self.gradv_arr,
            Leff_CO_max=self.Leff_CO_max_arr,
            isDust_cooling=self.isDust_cooling,
            isCoolingCOThin=self.isCoolingCOThin,
            fH2gr=float(1.0),
            fHplusgr=float(1.0),
            fCplusgr=float(1.0),
            fHeplusgr=float(1.0),
            fSplusgr=float(1.0),
            fSiplusgr=float(1.0),
            fCplusCR=float(1.0),
            **self._co_phase_kw,
            userJac=self.userJac,
            verbose=self.verbose,
        )
        y_final = self._project_state(self._repair_failed_cells(result["y"], result["status"]))
        return self._commit_solution(
            y_new=y_final,
            status=np.asarray(result["status"], dtype=np.int32),
            theta_h2=theta_h2,
            theta_co=theta_co,
            theta_c=theta_c,
            chi_dust_arr=chi_dust_arr,
            Gph=Gph,
            GISRF=GISRF,
        )
