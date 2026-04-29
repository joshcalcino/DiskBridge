from __future__ import annotations

import numpy as np

import diskbridge
import diskbridge._gow17 as _gow17

from diskbridge._config import resolve_model_config
from diskbridge._units import Quantity
from diskbridge._constants import (
    E_BIND_CO,
    NU0_CO,
    F_DRAINE,
    N_LAY,
    N_SURF,
    Y_CO,
)

from diskbridge.chemistry.shielding.visser_shielding import VisserShielding

from diskbridge.chemistry.models.gow17 import (
    N_Y,
    N_PH,
    IPH_C,
    IPH_CO,
    IPH_H2,
    XC_STD,
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
    _maybe_quantity_to_float,
    _compute_shielding_and_gph,
)

KB_CGS = 1.380649e-16
YR_TO_S = 365.25 * 24.0 * 3600.0


def _build_abstol(abstol0: float) -> np.ndarray:
    abstol = np.full(N_Y, float(abstol0), dtype=np.float64)
    abstol[I_HEP] = 1.0e-15
    abstol[I_OHX] = 1.0e-15
    abstol[I_CHX] = 1.0e-15
    abstol[I_CO] = 1.0e-15
    abstol[I_CO_ICE] = 1.0e-15
    abstol[I_CP] = 1.0e-15
    abstol[I_HCOP] = max(float(abstol0), 1.0e-20)
    abstol[I_H2] = 1.0e-8
    abstol[I_HP] = 1.0e-15
    abstol[I_H3P] = 1.0e-15
    abstol[I_H2P] = 1.0e-15
    return abstol


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

        self.const_temp = bool(cfg.get("const_temp", True))
        self.enable_co_phase = bool(cfg.get("enable_co_phase", False))
        self.chi_is_incident = bool(cfg.get("chi_is_incident", False))
        if self.chi_is_incident:
            if getattr(rad, "Av", None) is None:
                raise ValueError("Gow17TimeStepper: chi_is_incident=True requires rad.Av to be set")

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
        self.isDust_cooling = bool(cfg.get("isDust_cooling", False))
        self.isCoolingCOThin = bool(cfg.get("isCoolingCOThin", False))

        self.local_chi_factor = float(cfg.get("local_chi_factor", 0.5))
        self.shielding_outer_1d = str(cfg.get("shielding_outer_1d", "min"))

        self.rad = rad
        self.nside = int(diskbridge.params.nside)

        nH = rad.ensure_nH()
        chi = rad.ensure_chi()
        Tdust = rad.ensure_dust_temperature()
        Tgas = rad.ensure_gas_temperature()
        if Tgas is None:
            Tgas = Tdust

        self.shape = _as_cgs_f64(nH, "cm^-3").shape
        self.ncells = int(np.prod(self.shape))

        self.nH_cm3 = _as_cgs_f64(nH, "cm^-3")
        self.nH_flat = self.nH_cm3.reshape(self.ncells)

        self.abstol = _build_abstol(self.abstol0)

        sigma_d = rad.ensure_sigma_d_per_H()
        sigma_d_cm2 = _as_cgs_f64(sigma_d, "cm^2").reshape(self.ncells)

        sigma_ref_cfg = cfg.get("sigma_d_per_H_ref", None)
        if sigma_ref_cfg is None:
            valid = np.isfinite(sigma_d_cm2) & (sigma_d_cm2 > 0.0)
            if not np.any(valid):
                raise ValueError("Gow17TimeStepper: sigma_d_per_H has no positive finite values")
            self.sigma_d_per_H_ref = float(np.nanmedian(sigma_d_cm2[valid]))
        else:
            if isinstance(sigma_ref_cfg, str):
                self.sigma_d_per_H_ref = float(Quantity(sigma_ref_cfg).to("cm^2").magnitude)
            else:
                self.sigma_d_per_H_ref = float(sigma_ref_cfg)

        self.Zd_arr = np.divide(
            sigma_d_cm2,
            self.sigma_d_per_H_ref,
            out=np.ones(self.ncells, dtype=np.float64),
            where=(self.sigma_d_per_H_ref > 0.0),
        )

        self.Zg_arr = _broadcast_scalar_or_array(self.Zg, self.ncells)
        self.ion_rate_arr = _broadcast_scalar_or_array(self.ion_rate_s, self.ncells)
        self.gradv_arr = _broadcast_scalar_or_array(self.gradv_scalar, self.ncells)
        self.Leff_CO_max_arr = _broadcast_scalar_or_array(self.Leff_CO_max_scalar, self.ncells)

        self.xCtot_flat = self.Zg_arr * float(XC_STD)

        self.Av_flat = None
        if self.chi_is_incident:
            Av_q = getattr(rad, "Av", None)
            Av_arr = _as_cgs_f64(Av_q, "dimensionless")
            self.Av_flat = Av_arr.reshape(self.ncells)

        self.visser = VisserShielding(b_kms=float(self.b_kms))

        self._co_phase_kw = dict(
            co_sigma_d_per_H_ref=(float(self.sigma_d_per_H_ref) if self.enable_co_phase else 0.0),
            co_E_bind_co=(float(E_BIND_CO) if self.enable_co_phase else 0.0),
            co_nu0_co=(float(NU0_CO) if self.enable_co_phase else 0.0),
            co_F_DRAINE=(float(F_DRAINE) if self.enable_co_phase else 0.0),
            co_Y_CO=(float(Y_CO) if self.enable_co_phase else 0.0),
            co_N_SURF=(float(N_SURF) if self.enable_co_phase else 0.0),
            co_N_LAY=(int(N_LAY) if self.enable_co_phase else 0),
        )

        y_prev = getattr(rad, "gow17_y", None)
        if y_prev is not None and np.shape(y_prev) == tuple(self.shape) + (N_Y,):
            self.y_state = np.ascontiguousarray(np.asarray(y_prev, dtype=np.float64).reshape(self.ncells, N_Y))
        else:
            y0 = _default_y0_single()
            self.y_state = np.tile(y0[None, :], (self.ncells, 1)).astype(np.float64)

        if not self.enable_co_phase:
            self.y_state[:, I_CO_ICE] = 0.0

        if not self.const_temp and (y_prev is None):
            Tgas = rad.ensure_gas_temperature()
            if Tgas is None:
                Tgas = rad.ensure_dust_temperature()
            T_flat = _as_cgs_f64(Tgas, "K").reshape(self.ncells)
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

    def _prepare_environment(
        self,
        *,
        y_state: np.ndarray | None = None,
    ):
        rad = self.rad
        if y_state is None:
            y_state = self.y_state

        chi_dust_arr = _as_cgs_f64(rad.ensure_chi(), "dimensionless")
        chi_dust_flat = chi_dust_arr.reshape(self.ncells)

        Tdust_flat = _as_cgs_f64(rad.ensure_dust_temperature(), "K").reshape(self.ncells)

        Tgas = rad.ensure_gas_temperature()
        if Tgas is None:
            Tgas = rad.ensure_dust_temperature()
        T_flat = _as_cgs_f64(Tgas, "K").reshape(self.ncells)

        theta_h2, theta_co, theta_c, Gph, GPE, GISRF = _compute_shielding_and_gph(
            y_flat=y_state,
            nH_flat=self.nH_flat,
            chi_dust_flat=chi_dust_flat,
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
    ) -> dict:
        rad = self.rad

        self.y_state[:, :] = np.asarray(y_new, dtype=np.float64)
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
        rad.chi_eff = Quantity(chi_dust_arr * theta_co_arr, "dimensionless")

        if not self.const_temp:
            xe_out = _electron_abundance(y_out)
            Cv_out = _cv_cold(y_out[..., I_H2], xe_out)
            T_out = y_out[..., I_E] / Cv_out
            T_out = np.clip(T_out, 2.7, 1e6)
            rad.Tgas_gow17 = Quantity(T_out, "K")
            rad.gas_temperature = rad.Tgas_gow17

        return {
            "status": status,
            "theta_co": theta_co_arr,
            "theta_h2": theta_h2_arr,
            "theta_c": theta_c_arr,
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
            y0=np.ascontiguousarray(self.y_state, dtype=np.float64),
            nH=self.nH_flat,
            Tgas=T_flat,
            Tdust=Tdust_flat,
            Zd=self.Zd_arr,
            Zg=self.Zg_arr,
            ion_rate=self.ion_rate_arr,
            GPE=GPE,
            GISRF=GISRF,
            Gph=Gph,
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
        y_new = self._repair_failed_cells(result["y"], result["status"])
        return self._commit_solution(
            y_new=y_new,
            status=np.asarray(result["status"], dtype=np.int32),
            theta_h2=theta_h2,
            theta_co=theta_co,
            theta_c=theta_c,
            chi_dust_arr=chi_dust_arr,
        )

    def solve_equilibrium(self) -> dict:
        N = self.astrochem_n_updates
        if N < 1:
            raise ValueError("Gow17TimeStepper.solve_equilibrium: astrochem_n_updates must be >= 1")
        max_iter = self.shielding_max_iter
        if max_iter < N:
            raise ValueError("Gow17TimeStepper.solve_equilibrium: shielding_max_iter must be >= astrochem_n_updates")

        y_state = np.ascontiguousarray(self.y_state.copy(), dtype=np.float64)

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
                y0=np.ascontiguousarray(y_state, dtype=np.float64),
                nH=self.nH_flat,
                Tgas=T_flat,
                Tdust=Tdust_flat,
                Zd=self.Zd_arr,
                Zg=self.Zg_arr,
                ion_rate=self.ion_rate_arr,
                GPE=GPE,
                GISRF=GISRF,
                Gph=Gph,
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
            y_state[:, :] = self._repair_failed_cells(result_time["y"], result_time["status"])

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
            y0=np.ascontiguousarray(y_state, dtype=np.float64),
            nH=self.nH_flat,
            Tgas=T_flat,
            Tdust=Tdust_flat,
            Zd=self.Zd_arr,
            Zg=self.Zg_arr,
            ion_rate=self.ion_rate_arr,
            GPE=GPE,
            GISRF=GISRF,
            Gph=Gph,
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
        y_final = self._repair_failed_cells(result["y"], result["status"])
        return self._commit_solution(
            y_new=y_final,
            status=np.asarray(result["status"], dtype=np.int32),
            theta_h2=theta_h2,
            theta_co=theta_co,
            theta_c=theta_c,
            chi_dust_arr=chi_dust_arr,
        )
