# db-keywords: shielding, photodesorption, gow17, gas-temperature, config, units, radmc3d, chemistry, model, mesh
# db-role: entrypoint
# db-scope: package
# db-purpose: Package module for shielding, photodesorption, gow17, gas-temperature.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import diskbridge
import diskbridge._gow17 as _gow17
from diskbridge._config import resolve_model_config
from diskbridge._units import Quantity
from diskbridge.chemistry.models.gow17 import (
    I_CHX,
    I_CO,
    I_CO_ICE,
    I_CP,
    I_E,
    I_H2,
    I_H2P,
    I_H3P,
    I_HCOP,
    I_HEP,
    I_HP,
    I_OHX,
    N_Y,
    XC_STD,
    _as_cgs_f64,
    _build_gow17_abstol,
    _co_pdes_draine_flux,
    _cv_cold,
    _electron_abundance,
    _gow17_species_outputs,
    _initial_gas_temperature,
    _maybe_quantity_to_float,
    _model_microturbulence_grid_kms,
    _resolve_co_phase_controls,
    _resolve_shielding_linewidth,
    _resolve_temperature_config,
    _resolve_co_phase_runtime_params,
    _warn_if_co_phase_settings_ignored,
)
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.model.microturbulence import ensure_microturbulence_field

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel


_MODEL_NAME = "gow17_slab_equilibrium"


def _require_slab_shape(shape: tuple[int, ...]) -> None:
    if sum(int(n) > 1 for n in shape) != 1:
        raise ValueError(
            f"{_MODEL_NAME} requires exactly one varying mesh axis; got shape={shape}"
        )


def _uniform_scalar(name: str, values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    value = float(flat[0])
    if not np.allclose(flat, value, rtol=0.0, atol=0.0):
        raise ValueError(f"{_MODEL_NAME} requires uniform {name}")
    return value


def _default_y0(*, const_temp: bool, Tgas: float) -> np.ndarray:
    y0 = np.zeros(N_Y, dtype=np.float64)
    y0[I_HEP] = 1.450654e-08
    y0[I_H3P] = 2.681411e-07
    y0[I_CP] = 1.0e-4
    y0[I_CO] = 1.0e-7
    y0[I_H2] = 0.1
    y0[I_CO_ICE] = 0.0
    if not const_temp:
        Cv0 = _cv_cold(
            np.asarray([y0[I_H2]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
        )[0]
        y0[I_E] = Cv0 * float(Tgas)
    return y0


def run_gow17_slab_equilibrium(rad: "RadModel", config: dict) -> ChemistryResult:
    cfg = resolve_model_config(("chemistry", "gow17"), overrides=config)

    Tdust = rad.ensure_dust_temperature()
    nH = rad.ensure_nH()
    chi = rad.ensure_chi()
    Av_q = getattr(rad, "Av", None)
    if Av_q is None:
        raise ValueError(f"{_MODEL_NAME} requires rad.Av")

    nH_cm3 = _as_cgs_f64(nH, "cm^-3")
    shape = nH_cm3.shape
    _require_slab_shape(shape)
    ncells = int(nH_cm3.size)

    temperature_cfg = _resolve_temperature_config(cfg)
    temperature_mode = str(temperature_cfg["mode"])
    const_temp = bool(temperature_cfg["const_temp"])
    if temperature_mode == "dust":
        Tgas = Tdust
    else:
        Tgas = _initial_gas_temperature(rad, Tdust, temperature_cfg)

    T_K = np.clip(
        _as_cgs_f64(Tgas, "K"),
        float(temperature_cfg["Tgas_floor"]),
        float(temperature_cfg["Tgas_ceiling"]),
    )
    Tdust_K = _as_cgs_f64(Tdust, "K")
    chi_arr = _as_cgs_f64(chi, "dimensionless")
    Av_arr = _as_cgs_f64(Av_q, "dimensionless")
    if Av_arr.shape != shape:
        raise ValueError(
            f"{_MODEL_NAME} rad.Av shape {Av_arr.shape} does not match nH shape {shape}"
        )

    nH0 = _uniform_scalar("nH", nH_cm3)
    chi0 = _uniform_scalar("incident chi", chi_arr)
    T0 = _uniform_scalar("Tgas", T_K) if const_temp else float(T_K.reshape(-1)[0])

    Av_flat = np.asarray(Av_arr, dtype=np.float64).reshape(-1)
    NH_grid = Av_flat * 1.87e21
    positive_NH = NH_grid[np.isfinite(NH_grid) & (NH_grid > 0.0)]
    if positive_NH.size == 0:
        raise ValueError(f"{_MODEL_NAME} requires positive finite Av values")
    NH_min = _maybe_quantity_to_float(cfg.get("NH_min", float(np.min(positive_NH))), "cm^-2")
    NH_total = _maybe_quantity_to_float(cfg.get("NH_total", float(np.max(positive_NH))), "cm^-2")

    enable_co_phase = bool(cfg["enable_co_phase"])
    _warn_if_co_phase_settings_ignored(
        cfg,
        enable_co_phase=enable_co_phase,
        context=_MODEL_NAME,
    )

    ensure_microturbulence_field(rad.model, diskbridge.params)
    v_turb_grid_kms = _model_microturbulence_grid_kms(rad, shape)
    (
        b_H2_kms,
        b_H2_kms_arr,
        b_CO_kms,
        b_CO_kms_arr,
        shielding_linewidth_meta,
    ) = _resolve_shielding_linewidth(
        T_K,
        v_turb_grid_kms,
    )

    ion_rate_s = Quantity(cfg["ion_rate"]).to("1/s").magnitude
    ion_rate_arr = np.full(ncells, float(ion_rate_s), dtype=np.float64)
    Zg = float(cfg["Zg"])
    Zd = float(cfg["Zd"])
    reltol = float(cfg["reltol"])
    abstol0 = float(cfg["abstol0"])
    tolfac = float(cfg["tolfac"])
    tmin = _maybe_quantity_to_float(cfg["tmin"], "s")
    tmax = _maybe_quantity_to_float(cfg["tmax"], "s")
    mxsteps = int(cfg["mxsteps"])
    maxord = int(cfg["maxord"])
    verbose = bool(cfg["verbose"])
    userJac = bool(cfg["userJac"])
    gradv = float(cfg["gradv"])
    co_phase_params = _resolve_co_phase_runtime_params(cfg)
    S_CO, F_CRUV_CO_pdes_arr, k_crdes_CO_arr = _resolve_co_phase_controls(
        cfg=cfg,
        ion_rate_arr=ion_rate_arr,
        ncells=ncells,
    )
    co_sigma_d_per_H_ref = (
        _maybe_quantity_to_float(cfg["dust"]["sigma_d_ISM_ref"], "cm^2")
        if enable_co_phase
        else 0.0
    )

    abstol = _build_gow17_abstol(cfg, abstol0)
    abstol[I_E] = float(
        _cv_cold(
            np.asarray([0.1], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
        )[0]
    )

    slab = _gow17.solve_slab_1d_equilibrium(
        nH=float(nH0),
        G0=2.0 * float(chi0),
        ngrid=int(ncells),
        NH_total=float(NH_total),
        logNH=bool(cfg["logNH"]),
        NH_min=float(NH_min),
        field_geo=int(cfg["field_geo"]),
        isdust=bool(cfg["isdust"]),
        isfsH2=bool(cfg["isfsH2"]),
        isfsCO=bool(cfg["isfsCO"]),
        isfsC=bool(cfg["isfsC"]),
        Zg=float(Zg),
        Zd=float(Zd),
        ion_rate=float(ion_rate_s),
        reltol=float(reltol),
        abstol=abstol,
        mxsteps=int(mxsteps),
        maxord=int(maxord),
        tolfac=float(tolfac),
        tmin=float(tmin),
        tmax=float(tmax),
        verbose=bool(verbose),
        y0=_default_y0(const_temp=const_temp, Tgas=T0),
        const_temp=bool(const_temp),
        Tgas=float(T0),
        gradv=float(gradv),
        NCOeff_global=bool(cfg["NCOeff_global"]),
        bCO_L=bool(cfg["bCO_L"]),
        fH2gr=float(cfg["fH2gr"]),
        fHplusgr=float(cfg["fHplusgr"]),
        fCplusgr=float(cfg["fCplusgr"]),
        fHeplusgr=float(cfg["fHeplusgr"]),
        fSplusgr=float(cfg["fSplusgr"]),
        fSiplusgr=float(cfg["fSiplusgr"]),
        fCplusCR=float(cfg["fCplusCR"]),
        co_sigma_d_per_H_ref=float(co_sigma_d_per_H_ref),
        co_E_bind_co=(float(co_phase_params["E_bind_CO"]) if enable_co_phase else 0.0),
        co_nu0_co=(float(co_phase_params["nu0_CO"]) if enable_co_phase else 0.0),
        co_F_DRAINE=float(_co_pdes_draine_flux()) if enable_co_phase else 0.0,
        co_Y_CO=(float(co_phase_params["Y_CO"]) if enable_co_phase else 0.0),
        co_N_SURF=(float(co_phase_params["N_SURF"]) if enable_co_phase else 0.0),
        co_N_LAY=(int(co_phase_params["N_LAY"]) if enable_co_phase else 0),
        userJac=bool(userJac),
        co_S_CO=(float(S_CO) if enable_co_phase else 0.0),
        co_F_CRUV_CO_pdes=(
            float(F_CRUV_CO_pdes_arr.reshape(-1)[0]) if enable_co_phase else 0.0
        ),
        co_k_crdes_CO=(
            float(k_crdes_CO_arr.reshape(-1)[0]) if enable_co_phase else 0.0
        ),
    )

    y_out = np.asarray(slab["y"], dtype=np.float64).reshape(shape + (N_Y,))
    theta_h2_arr = np.asarray(slab["fShieldH2"], dtype=np.float64).reshape(shape)
    theta_co_arr = np.asarray(slab["fShieldCO"], dtype=np.float64).reshape(shape)
    theta_c_arr = np.ones_like(theta_co_arr, dtype=np.float64)

    xe = _electron_abundance(y_out)
    if const_temp:
        T_out = T_K
    else:
        Cv = _cv_cold(y_out[..., I_H2], xe)
        with np.errstate(divide="ignore", invalid="ignore"):
            T_out = y_out[..., I_E] / Cv
        T_out = np.clip(
            T_out,
            float(temperature_cfg["Tgas_floor"]),
            float(temperature_cfg["Tgas_ceiling"]),
        )

    xH_atom = 1.0 - (
        y_out[..., I_OHX]
        + y_out[..., I_CHX]
        + y_out[..., I_HCOP]
        + 3.0 * y_out[..., I_H3P]
        + 2.0 * y_out[..., I_H2P]
        + y_out[..., I_HP]
        + 2.0 * y_out[..., I_H2]
    )
    xCtot = np.full(shape, float(Zg) * float(XC_STD), dtype=np.float64)
    xC_neutral = xCtot - (
        y_out[..., I_HCOP]
        + y_out[..., I_CHX]
        + y_out[..., I_CO]
        + y_out[..., I_CO_ICE]
        + y_out[..., I_CP]
    )

    abundances, number_densities = _gow17_species_outputs(
        y_out,
        nH_cm3,
        x_h=xH_atom,
        x_catom=xC_neutral,
        x_e=xe,
        line_h2_opr=cfg["line_h2_opr"],
    )

    F_CO_pdes_photon = chi_arr * _co_pdes_draine_flux()
    F_CO_pdes_total = F_CO_pdes_photon + (
        F_CRUV_CO_pdes_arr.reshape(shape) if enable_co_phase else 0.0
    )
    G_CO = chi_arr * theta_co_arr
    fields = {
        "co_ice": Quantity(y_out[..., I_CO_ICE] * nH_cm3, "cm^-3"),
        "Tgas": Quantity(T_out, "K"),
        "Tgas_minus_Tdust": Quantity(T_out - Tdust_K, "K"),
        "Tgas_status": Quantity(np.zeros(shape, dtype=np.int32), "dimensionless"),
        "b_H2_kms": Quantity(b_H2_kms_arr, "km/s"),
        "b_CO_kms": Quantity(b_CO_kms_arr, "km/s"),
        "chi_broad": Quantity(chi_arr, "dimensionless"),
        "G_CO_diss": Quantity(chi_arr, "dimensionless"),
        "G_H2_diss": Quantity(chi_arr, "dimensionless"),
        "G_C_ion": Quantity(chi_arr, "dimensionless"),
        "G_CO_pdes": Quantity(chi_arr, "dimensionless"),
        "F_CO_pdes_photon": Quantity(F_CO_pdes_photon, "1/(cm^2 s)"),
        "theta_co": Quantity(theta_co_arr, "dimensionless"),
        "theta_h2": Quantity(theta_h2_arr, "dimensionless"),
        "theta_c": Quantity(theta_c_arr, "dimensionless"),
        "chi_eff": Quantity(G_CO, "dimensionless"),
        "G_CO_diss_actual": Quantity(G_CO, "dimensionless"),
        "G_C_ion_actual": Quantity(chi_arr * theta_c_arr, "dimensionless"),
        "G_H2_diss_actual": Quantity(chi_arr * theta_h2_arr, "dimensionless"),
        "F_CRUV_CO_pdes": Quantity(
            F_CRUV_CO_pdes_arr.reshape(shape) if enable_co_phase else np.zeros(shape, dtype=np.float64),
            "1/(cm^2 s)",
        ),
        "F_CO_pdes_photon_total": Quantity(F_CO_pdes_total, "1/(cm^2 s)"),
    }

    rad.gow17_y = y_out
    rad.nco_gas = Quantity(y_out[..., I_CO] * nH_cm3, "cm^-3")
    rad.nco_ice = fields["co_ice"]
    rad.theta_co = fields["theta_co"]
    rad.chi_eff = fields["chi_eff"]
    rad.nH2 = Quantity(y_out[..., I_H2] * nH_cm3, "cm^-3")
    rad.nH_atom = Quantity(np.maximum(xH_atom, 0.0) * nH_cm3, "cm^-3")
    rad.nCplus = Quantity(y_out[..., I_CP] * nH_cm3, "cm^-3")
    rad.nC = Quantity(np.maximum(xC_neutral, 0.0) * nH_cm3, "cm^-3")
    rad.ne = Quantity(xe * nH_cm3, "cm^-3")
    rad.Tgas_gow17 = fields["Tgas"]
    rad.gas_temperature = rad.Tgas_gow17

    return ChemistryResult(
        abundances=abundances,
        number_densities=number_densities,
        fields=fields,
        meta={
            "model": _MODEL_NAME,
            "enable_co_phase": bool(enable_co_phase),
            "radiation_mode": "incident_slab",
            "b_H2_kms_scalar": float(b_H2_kms),
            "b_CO_kms_scalar": float(b_CO_kms),
            "shielding_linewidth": shielding_linewidth_meta,
            "ion_rate_s": float(ion_rate_s),
            "Zg": float(Zg),
            "Zd": float(Zd),
            "NH_min": float(NH_min),
            "NH_total": float(NH_total),
            "reltol": float(reltol),
            "abstol0": float(abstol0),
            "n_fail": 0,
            "max_status": 0,
            "fail_idx_head": [],
            "status_hist": {0: int(ncells), -1: 0},
            "co_sigma_d_per_H_ref": float(co_sigma_d_per_H_ref),
        },
    )


__all__ = ["run_gow17_slab_equilibrium"]
