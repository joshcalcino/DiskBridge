from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np

import diskbridge
from diskbridge._units import Quantity
from diskbridge._config import resolve_model_config
from diskbridge._logging import logger
from diskbridge._constants import X_C_TOT
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.shielding.columns_1d import (
    compute_pdr_shielding_1d,
    is_effectively_1d,
    column_to_outer_boundary_1d,
    effective_1d_axis,
)
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
from diskbridge.chemistry.validation import validate_chemistry_state

from diskbridge.chemistry.models._gow17_network import (
    N_Y,
    I_HEP,
    I_OHX,
    I_CHX,
    I_CO,
    I_CP,
    I_HCOP,
    I_H2,
    I_HP,
    I_H3P,
    I_H2P,
    I_SP,
    I_SIP,
    I_OP,
    I_CO_ICE,
)

from diskbridge.chemistry.models._gow17_numba import (
    solve_gow17_equilibrium_cells_cgs,
    evolve_gow17_be_cells_cgs,
)


def _as_cgs_f64(q: Quantity, unit: str) -> np.ndarray:
    return np.ascontiguousarray(q.to(unit).magnitude, dtype=np.float64)


def run_gow17_pdr(rad: "RadModel", config: dict) -> ChemistryResult:
    cfg = resolve_model_config(("chemistry", "gow17_pdr"), overrides=config)

    if "mode" not in cfg:
        raise ValueError("gow17_pdr requires config['mode']")

    if "max_iter" not in cfg:
        raise ValueError("gow17_pdr requires config['max_iter']")
    if "reltol" not in cfg:
        raise ValueError("gow17_pdr requires config['reltol']")
    if "abstol0" not in cfg:
        raise ValueError("gow17_pdr requires config['abstol0']")

    mode = str(cfg.get("mode")).lower()
    if mode not in {"equilibrium", "time_dependent"}:
        raise ValueError(f"gow17_pdr: unknown mode={mode!r} (expected 'equilibrium' or 'time_dependent')")

    nside = int(cfg.get("nside"))
    b_kms = float(cfg.get("b_kms"))

    chi0_cfg = cfg.get("chi0", None)

    ion_rate_s = Quantity(cfg.get("ion_rate")).to("1/s").magnitude

    Zg = float(cfg.get("Zg"))
    Zd = float(cfg.get("Zd"))

    fH2gr = float(cfg.get("fH2gr"))
    fHplusgr = float(cfg.get("fHplusgr"))
    fCplusgr = float(cfg.get("fCplusgr"))
    fHeplusgr = float(cfg.get("fHeplusgr"))
    fSplusgr = float(cfg.get("fSplusgr"))
    fSiplusgr = float(cfg.get("fSiplusgr"))
    fCplusCR = float(cfg.get("fCplusCR"))

    const_temp = bool(cfg.get("const_temp"))

    max_iter = int(cfg.get("max_iter"))

    reltol = float(cfg.get("reltol"))

    abstol0 = float(cfg.get("abstol0"))

    nH = rad.ensure_nH()
    Tdust = rad.ensure_dust_temperature()

    if const_temp:
        Tgas = Tdust
    else:
        Tgas = rad.ensure_gas_temperature()
        if Tgas is None:
            Tgas = Tdust

    chi = rad.ensure_chi()

    sigma_d_per_H = rad.ensure_sigma_d_per_H()

    nH_cm3 = _as_cgs_f64(nH, "cm^-3")
    T_K = _as_cgs_f64(Tgas, "K")
    Tdust_K = _as_cgs_f64(Tdust, "K")
    chi_pe_arr = _as_cgs_f64(chi, "dimensionless")
    sigma_cm2 = _as_cgs_f64(sigma_d_per_H, "cm^2")

    if chi0_cfg is None:
        chi0_arr = 2.0 * chi_pe_arr
        Av_arr = np.zeros_like(chi_pe_arr, dtype=np.float64)
    else:
        chi0_val = float(chi0_cfg)
        chi0_arr = np.full_like(chi_pe_arr, chi0_val, dtype=np.float64)

        Av_in = getattr(rad, "Av", None)
        if Av_in is not None:
            Av_arr = _as_cgs_f64(Av_in, "dimensionless")
            if Av_arr.shape != chi_pe_arr.shape:
                raise ValueError(
                    f"gow17_pdr: rad.Av must match chi shape, got {Av_arr.shape} vs {chi_pe_arr.shape}"
                )
        elif is_effectively_1d(rad.model.mesh, nH_cm3.shape):
            axis_name, axis_index = effective_1d_axis(rad.model.mesh, nH_cm3.shape)
            NH = column_to_outer_boundary_1d(
                rad.model.mesh,
                nH_cm3,
                axis_name=axis_name,
                axis_index=axis_index,
                outer="max",
            )
            Av_arr = (NH * float(Zd)) / 1.87e21
        else:
            Av_arr = np.zeros_like(chi_pe_arr, dtype=np.float64)

    visser = VisserShielding(b_kms=float(b_kms))

    y0 = np.zeros(N_Y, dtype=np.float64)
    y0[I_HEP] = 1.450654e-08
    y0[I_H3P] = 2.681411e-07
    y0[I_CP] = 1.0e-4
    y0[I_CO] = 1.0e-7
    y0[I_H2] = 0.1
    y0[I_CO_ICE] = 0.0

    abstol = np.full(N_Y, abstol0, dtype=np.float64)
    abstol[I_HEP] = 1.0e-15
    abstol[I_OHX] = 1.0e-15
    abstol[I_CHX] = 1.0e-15
    abstol[I_CO] = 1.0e-15
    abstol[I_CP] = 1.0e-15
    abstol[I_HCOP] = 1.0e-30
    abstol[I_H2] = 1.0e-8
    abstol[I_HP] = 1.0e-15
    abstol[I_H3P] = 1.0e-15
    abstol[I_H2P] = 1.0e-15

    shape = nH_cm3.shape
    ncells = nH_cm3.size

    theta_h2_arr = np.ones_like(nH_cm3, dtype=np.float64)
    theta_co_arr = np.ones_like(nH_cm3, dtype=np.float64)
    theta_c_arr = np.ones_like(nH_cm3, dtype=np.float64)
    theta_pdr_arr = np.ones_like(nH_cm3, dtype=np.float64)
    chi_eff_pdr_arr = np.ascontiguousarray(chi_pe_arr, dtype=np.float64)

    if mode == "equilibrium":
        if "t_end" not in cfg:
            raise ValueError("gow17_pdr equilibrium mode requires config['t_end']")
        dt_eq_s = float(Quantity(cfg.get("t_end")).to("s").magnitude)
        if dt_eq_s <= 0.0:
            raise ValueError("gow17_pdr: t_end must be > 0")

        if "shielding_max_iter" not in cfg:
            raise ValueError("gow17_pdr equilibrium mode requires config['shielding_max_iter']")
        if "shielding_reltol" not in cfg:
            raise ValueError("gow17_pdr equilibrium mode requires config['shielding_reltol']")
        if "shielding_abstol" not in cfg:
            raise ValueError("gow17_pdr equilibrium mode requires config['shielding_abstol']")

        shielding_max_iter = int(cfg.get("shielding_max_iter"))
        shielding_reltol = float(cfg.get("shielding_reltol"))
        shielding_abstol = float(cfg.get("shielding_abstol"))
        if shielding_max_iter <= 0:
            raise ValueError("gow17_pdr: shielding_max_iter must be >= 1")
        if shielding_reltol <= 0.0:
            raise ValueError("gow17_pdr: shielding_reltol must be > 0")
        if shielding_abstol <= 0.0:
            raise ValueError("gow17_pdr: shielding_abstol must be > 0")

        shielding_mix_cfg = cfg.get("shielding_mix", None)
        if shielding_mix_cfg is None:
            shielding_mix = 1.0
        else:
            shielding_mix = float(shielding_mix_cfg)
        if not (0.0 < shielding_mix <= 1.0):
            raise ValueError(
                f"gow17_pdr: shielding_mix must be in (0, 1], got {shielding_mix}"
            )

        y_inout = np.zeros((ncells, N_Y), dtype=np.float64)
        y_prev = getattr(rad, "gow17_y", None)
        if y_prev is not None and np.shape(y_prev) == tuple(shape) + (N_Y,):
            y_inout[:, :] = np.ascontiguousarray(np.asarray(y_prev, dtype=np.float64).reshape(ncells, N_Y))
        else:
            for j in range(N_Y):
                y_inout[:, j] = y0[j]

        y_prev_snapshot = np.zeros((ncells, N_Y), dtype=np.float64)
        status_step = np.zeros(ncells, dtype=np.int64)
        status_acc = np.zeros(ncells, dtype=np.int64)

        xCtot = float(Zg) * float(X_C_TOT)
        theta_h2_prev_flat = np.ones(ncells, dtype=np.float64)
        theta_co_prev_flat = np.ones(ncells, dtype=np.float64)
        theta_c_prev_flat = np.ones(ncells, dtype=np.float64)
        theta_pdr_prev_flat = np.ones(ncells, dtype=np.float64)
        tiny = float(np.finfo(np.float64).tiny)

        for it in range(shielding_max_iter):
            y_prev_snapshot[:, :] = y_inout
            xCO_old = np.ascontiguousarray(y_inout[:, I_CO].copy(), dtype=np.float64)
            xH2_old = np.ascontiguousarray(y_inout[:, I_H2].copy(), dtype=np.float64)

            xCO = y_inout[:, I_CO]
            xCO_ice = y_inout[:, I_CO_ICE]
            xH2 = y_inout[:, I_H2]

            xC_neutral = xCtot - (
                y_inout[:, I_HCOP]
                + y_inout[:, I_CHX]
                + xCO
                + y_inout[:, I_CP]
                + xCO_ice
            )
            xC_neutral = np.maximum(xC_neutral, 0.0)

            nCO_cm3 = np.ascontiguousarray((xCO * nH_cm3.reshape(ncells)).reshape(shape), dtype=np.float64)
            nH2_cm3 = np.ascontiguousarray((xH2 * nH_cm3.reshape(ncells)).reshape(shape), dtype=np.float64)
            nC_cm3 = np.ascontiguousarray((xC_neutral * nH_cm3.reshape(ncells)).reshape(shape), dtype=np.float64)

            if is_effectively_1d(rad.model.mesh, nH_cm3.shape):
                theta_h2_arr, theta_co_arr, theta_c_arr, theta_pdr_arr, chi_eff_pdr_arr = compute_pdr_shielding_1d(
                    mesh=rad.model.mesh,
                    nH=nH_cm3,
                    chi=chi_pe_arr,
                    visser=visser,
                    nCO=nCO_cm3,
                    nC=nC_cm3,
                    nH2=nH2_cm3,
                    b_kms=b_kms,
                    outer="max",
                    return_quantity=False,
                )
            else:
                from diskbridge.chemistry.shielding.healpix_columns import compute_pdr_shielding_healpix

                theta_h2_arr, theta_co_arr, theta_c_arr, theta_pdr_arr, chi_eff_pdr_arr = compute_pdr_shielding_healpix(
                    mesh=rad.model.mesh,
                    nH=nH_cm3,
                    chi=chi_pe_arr,
                    visser=visser,
                    nCO=nCO_cm3,
                    nC=nC_cm3,
                    nH2=nH2_cm3,
                    nside=nside,
                    b_kms=b_kms,
                    return_quantity=False,
                )

            alpha = float(shielding_mix)

            theta_h2_new_flat = theta_h2_arr.reshape(ncells)
            theta_co_new_flat = theta_co_arr.reshape(ncells)
            theta_c_new_flat = theta_c_arr.reshape(ncells)
            theta_pdr_new_flat = theta_pdr_arr.reshape(ncells)

            theta_h2_prev_pos = np.maximum(theta_h2_prev_flat, tiny)
            theta_co_prev_pos = np.maximum(theta_co_prev_flat, tiny)
            theta_c_prev_pos = np.maximum(theta_c_prev_flat, tiny)
            theta_pdr_prev_pos = np.maximum(theta_pdr_prev_flat, tiny)

            theta_h2_new_pos = np.maximum(theta_h2_new_flat, tiny)
            theta_co_new_pos = np.maximum(theta_co_new_flat, tiny)
            theta_c_new_pos = np.maximum(theta_c_new_flat, tiny)
            theta_pdr_new_pos = np.maximum(theta_pdr_new_flat, tiny)

            theta_h2_mix_flat = np.exp(
                (1.0 - alpha) * np.log(theta_h2_prev_pos) + alpha * np.log(theta_h2_new_pos)
            )
            theta_co_mix_flat = np.exp(
                (1.0 - alpha) * np.log(theta_co_prev_pos) + alpha * np.log(theta_co_new_pos)
            )
            theta_c_mix_flat = np.exp(
                (1.0 - alpha) * np.log(theta_c_prev_pos) + alpha * np.log(theta_c_new_pos)
            )
            theta_pdr_mix_flat = np.exp(
                (1.0 - alpha) * np.log(theta_pdr_prev_pos) + alpha * np.log(theta_pdr_new_pos)
            )

            theta_h2_prev_flat = np.ascontiguousarray(theta_h2_mix_flat, dtype=np.float64)
            theta_co_prev_flat = np.ascontiguousarray(theta_co_mix_flat, dtype=np.float64)
            theta_c_prev_flat = np.ascontiguousarray(theta_c_mix_flat, dtype=np.float64)
            theta_pdr_prev_flat = np.ascontiguousarray(theta_pdr_mix_flat, dtype=np.float64)

            theta_h2_arr = theta_h2_mix_flat.reshape(shape)
            theta_co_arr = theta_co_mix_flat.reshape(shape)
            theta_c_arr = theta_c_mix_flat.reshape(shape)
            theta_pdr_arr = theta_pdr_mix_flat.reshape(shape)
            chi_eff_pdr_arr = (chi_pe_arr.reshape(ncells) * theta_pdr_mix_flat).reshape(shape)

            y_tmp = np.zeros_like(y_inout)
            status_step[:] = 0
            solve_gow17_equilibrium_cells_cgs(
                nH_cm3.reshape(ncells),
                T_K.reshape(ncells),
                Tdust_K.reshape(ncells),
                chi_pe_arr.reshape(ncells),
                chi0_arr.reshape(ncells),
                Av_arr.reshape(ncells),
                theta_h2_arr.reshape(ncells),
                theta_co_arr.reshape(ncells),
                theta_c_arr.reshape(ncells),
                chi_eff_pdr_arr.reshape(ncells),
                sigma_cm2.reshape(ncells),
                float(dt_eq_s),
                y0=y_inout,
                Zg=Zg,
                Zd=Zd,
                ion_rate_s=float(ion_rate_s),
                fH2gr=fH2gr,
                fHplusgr=fHplusgr,
                fCplusgr=fCplusgr,
                fHeplusgr=fHeplusgr,
                fSplusgr=fSplusgr,
                fSiplusgr=fSiplusgr,
                fCplusCR=fCplusCR,
                max_iter=max_iter,
                reltol=reltol,
                abstol=abstol,
                y_out=y_tmp,
                status_out=status_step,
            )

            status_acc = np.maximum(status_acc, status_step)
            bad_idx = np.flatnonzero(status_step == 2)
            if bad_idx.size:
                y_tmp[bad_idx, :] = y_prev_snapshot[bad_idx, :]

            y_inout[:, :] = y_tmp

            xCO_new = y_inout[:, I_CO]
            xH2_new = y_inout[:, I_H2]

            denom_h2 = np.maximum(np.abs(xH2_new), shielding_abstol)
            denom_co = np.maximum(np.abs(xCO_new), shielding_abstol)
            d_h2 = np.max(np.abs(xH2_new - xH2_old) / denom_h2)
            d_co = np.max(np.abs(xCO_new - xCO_old) / denom_co)
            if max(d_h2, d_co) <= shielding_reltol:
                break

        xCO = y_inout[:, I_CO]
        xCO_ice = y_inout[:, I_CO_ICE]
        xH2 = y_inout[:, I_H2]

        xC_neutral = xCtot - (
            y_inout[:, I_HCOP]
            + y_inout[:, I_CHX]
            + xCO
            + y_inout[:, I_CP]
            + xCO_ice
        )
        xC_neutral = np.maximum(xC_neutral, 0.0)

        nCO_cm3 = np.ascontiguousarray((xCO * nH_cm3.reshape(ncells)).reshape(shape), dtype=np.float64)
        nH2_cm3 = np.ascontiguousarray((xH2 * nH_cm3.reshape(ncells)).reshape(shape), dtype=np.float64)
        nC_cm3 = np.ascontiguousarray((xC_neutral * nH_cm3.reshape(ncells)).reshape(shape), dtype=np.float64)

        if is_effectively_1d(rad.model.mesh, nH_cm3.shape):
            theta_h2_arr, theta_co_arr, theta_c_arr, theta_pdr_arr, chi_eff_pdr_arr = compute_pdr_shielding_1d(
                mesh=rad.model.mesh,
                nH=nH_cm3,
                chi=chi_pe_arr,
                visser=visser,
                nCO=nCO_cm3,
                nC=nC_cm3,
                nH2=nH2_cm3,
                b_kms=b_kms,
                outer="max",
                return_quantity=False,
            )
        else:
            from diskbridge.chemistry.shielding.healpix_columns import compute_pdr_shielding_healpix

            theta_h2_arr, theta_co_arr, theta_c_arr, theta_pdr_arr, chi_eff_pdr_arr = compute_pdr_shielding_healpix(
                mesh=rad.model.mesh,
                nH=nH_cm3,
                chi=chi_pe_arr,
                visser=visser,
                nCO=nCO_cm3,
                nC=nC_cm3,
                nH2=nH2_cm3,
                nside=nside,
                b_kms=b_kms,
                return_quantity=False,
            )

        y_out = y_inout.reshape(shape + (N_Y,))
        status = status_acc
    else:
        if "t_end" not in cfg:
            raise ValueError("gow17_pdr time_dependent mode requires config['t_end']")
        if "dt" not in cfg:
            raise ValueError("gow17_pdr time_dependent mode requires config['dt']")
        if "shielding_update_every" not in cfg:
            raise ValueError("gow17_pdr time_dependent mode requires config['shielding_update_every']")

        t_end_s = float(Quantity(cfg.get("t_end")).to("s").magnitude)
        dt_s = float(Quantity(cfg.get("dt")).to("s").magnitude)
        shielding_update_every = int(cfg.get("shielding_update_every"))
        if t_end_s <= 0.0:
            raise ValueError("gow17_pdr: t_end must be > 0")
        if dt_s <= 0.0:
            raise ValueError("gow17_pdr: dt must be > 0")
        if shielding_update_every <= 0:
            raise ValueError("gow17_pdr: shielding_update_every must be >= 1")

        y_in = getattr(rad, "gow17_y", None)
        if y_in is not None and np.shape(y_in) == tuple(shape) + (N_Y,):
            y_inout = np.ascontiguousarray(np.asarray(y_in, dtype=np.float64).reshape(ncells, N_Y))
        else:
            y_inout = np.zeros((ncells, N_Y), dtype=np.float64)
            for j in range(N_Y):
                y_inout[:, j] = y0[j]

        status_acc = np.zeros(ncells, dtype=np.int64)
        status_step = np.zeros(ncells, dtype=np.int64)

        nH_flat = nH_cm3.reshape(ncells)
        chi_pe_flat = chi_pe_arr.reshape(ncells)
        chi0_flat = chi0_arr.reshape(ncells)
        Av_flat = Av_arr.reshape(ncells)
        T_flat = T_K.reshape(ncells)
        Td_flat = Tdust_K.reshape(ncells)
        sigma_flat = sigma_cm2.reshape(ncells)

        xCtot = float(Zg) * float(X_C_TOT)

        t = 0.0
        istep = 0
        while t < t_end_s:
            if (istep % shielding_update_every) == 0:
                xCO = y_inout[:, I_CO]
                xCO_ice = y_inout[:, I_CO_ICE]
                xH2 = y_inout[:, I_H2]

                xC_neutral = xCtot - (
                    y_inout[:, I_HCOP]
                    + y_inout[:, I_CHX]
                    + xCO
                    + y_inout[:, I_CP]
                    + xCO_ice
                )
                xC_neutral = np.maximum(xC_neutral, 0.0)

                nCO_cm3 = np.ascontiguousarray((xCO * nH_flat).reshape(shape), dtype=np.float64)
                nH2_cm3 = np.ascontiguousarray((xH2 * nH_flat).reshape(shape), dtype=np.float64)
                nC_cm3 = np.ascontiguousarray((xC_neutral * nH_flat).reshape(shape), dtype=np.float64)

                if is_effectively_1d(rad.model.mesh, nH_cm3.shape):
                    theta_h2_arr, theta_co_arr, theta_c_arr, theta_pdr_arr, chi_eff_pdr_arr = compute_pdr_shielding_1d(
                        mesh=rad.model.mesh,
                        nH=nH_cm3,
                        chi=chi_pe_arr,
                        visser=visser,
                        nCO=nCO_cm3,
                        nC=nC_cm3,
                        nH2=nH2_cm3,
                        b_kms=b_kms,
                        outer="max",
                        return_quantity=False,
                    )
                else:
                    from diskbridge.chemistry.shielding.healpix_columns import compute_pdr_shielding_healpix

                    theta_h2_arr, theta_co_arr, theta_c_arr, theta_pdr_arr, chi_eff_pdr_arr = compute_pdr_shielding_healpix(
                        mesh=rad.model.mesh,
                        nH=nH_cm3,
                        chi=chi_pe_arr,
                        visser=visser,
                        nCO=nCO_cm3,
                        nC=nC_cm3,
                        nH2=nH2_cm3,
                        nside=nside,
                        b_kms=b_kms,
                        return_quantity=False,
                    )

            dt_step = min(dt_s, t_end_s - t)

            evolve_gow17_be_cells_cgs(
                nH_flat,
                T_flat,
                Td_flat,
                chi_pe_flat,
                chi0_flat,
                Av_flat,
                theta_h2_arr.reshape(ncells),
                theta_co_arr.reshape(ncells),
                theta_c_arr.reshape(ncells),
                chi_eff_pdr_arr.reshape(ncells),
                sigma_flat,
                float(dt_step),
                y_inout,
                Zg=Zg,
                Zd=Zd,
                ion_rate_s=float(ion_rate_s),
                fH2gr=fH2gr,
                fHplusgr=fHplusgr,
                fCplusgr=fCplusgr,
                fHeplusgr=fHeplusgr,
                fSplusgr=fSplusgr,
                fSiplusgr=fSiplusgr,
                fCplusCR=fCplusCR,
                max_iter=max_iter,
                reltol=reltol,
                abstol=abstol,
                status_out=status_step,
            )

            status_acc = np.maximum(status_acc, status_step)
            t += dt_step
            istep += 1

        y_out = y_inout.reshape(shape + (N_Y,))
        status = status_acc

    xCO = y_out[..., I_CO]
    xCO_ice = y_out[..., I_CO_ICE]
    xH2 = y_out[..., I_H2]

    nco_gas = Quantity(xCO * nH_cm3, "cm^-3")
    nco_ice = Quantity(xCO_ice * nH_cm3, "cm^-3")

    nH2_cm3 = Quantity(xH2 * nH_cm3, "cm^-3")

    xe = (
        y_out[..., I_HEP]
        + y_out[..., I_CP]
        + y_out[..., I_HCOP]
        + y_out[..., I_H3P]
        + y_out[..., I_H2P]
        + y_out[..., I_HP]
        + y_out[..., I_SP]
        + y_out[..., I_SIP]
        + y_out[..., I_OP]
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

    xCtot = float(Zg) * float(X_C_TOT)
    xC_neutral = xCtot - (
        y_out[..., I_HCOP]
        + y_out[..., I_CHX]
        + y_out[..., I_CO]
        + y_out[..., I_CP]
        + y_out[..., I_CO_ICE]
    )

    nCplus = Quantity(y_out[..., I_CP] * nH_cm3, "cm^-3")
    nC = Quantity(np.maximum(xC_neutral, 0.0) * nH_cm3, "cm^-3")
    ne = Quantity(xe * nH_cm3, "cm^-3")

    nH_atom = Quantity(np.maximum(xH_atom, 0.0) * nH_cm3, "cm^-3")

    if np.any(status != 0):
        bad = int(np.sum(status != 0))
        logger.warning(f"gow17_pdr: {bad} cells did not converge")
    else:
        validate_chemistry_state(
            nH=nH,
            nH2=nH2_cm3,
            nHI=nH_atom,
            nC=nC,
            nCplus=nCplus,
            nco_total=Quantity(nco_gas.magnitude + nco_ice.magnitude, "cm^-3"),
            check_pd=False,
            rtol=float(reltol),
        )

    abundances = {
        "co": Quantity(xCO, "dimensionless"),
        "co_ice": Quantity(xCO_ice, "dimensionless"),
    }

    number_densities = {
        "co": nco_gas,
        "c+": nCplus,
        "catom": nC,
        "e": ne,
        "h2": nH2_cm3,
        "h": nH_atom,
    }

    fields = {
        "co_ice": nco_ice,
        "theta_co": Quantity(theta_co_arr, "dimensionless"),
        "theta_pdr": Quantity(theta_pdr_arr, "dimensionless"),
        "chi_eff_pdr": Quantity(chi_eff_pdr_arr, "dimensionless"),
        "theta_h2": Quantity(theta_h2_arr, "dimensionless"),
        "theta_c": Quantity(theta_c_arr, "dimensionless"),
    }

    rad.gow17_y = y_out
    rad.nco_gas = nco_gas
    rad.nco_ice = nco_ice
    rad.theta_co = Quantity(theta_co_arr, "dimensionless")
    rad.theta_pdr = Quantity(theta_pdr_arr, "dimensionless")
    rad.chi_eff = Quantity(chi_eff_pdr_arr, "dimensionless")
    rad.nH2 = nH2_cm3
    rad.nH_atom = nH_atom
    rad.nCplus = nCplus
    rad.nC = nC
    rad.ne = ne

    n_fail = int(np.sum(status != 0))
    max_status = int(np.max(status)) if status.size else 0

    fail_idx = np.flatnonzero(status != 0)
    fail_idx_head = fail_idx[:16].astype(np.int64)
    status_hist = {}
    if status.size:
        for code in (0, 1, 2):
            status_hist[int(code)] = int(np.sum(status == code))

    meta = {
        "model": "gow17_pdr",
        "mode": str(mode),
        "nside": int(nside),
        "b_kms": float(b_kms),
        "ion_rate_s": float(ion_rate_s),
        "Zg": float(Zg),
        "Zd": float(Zd),
        "const_temp": bool(const_temp),
        "reltol": float(reltol),
        "abstol0": float(abstol0),
        "max_iter": int(max_iter),
        "n_fail": int(n_fail),
        "max_status": int(max_status),
        "fail_idx_head": fail_idx_head.tolist(),
        "status_hist": status_hist,
    }

    return ChemistryResult(
        abundances=abundances,
        number_densities=number_densities,
        fields=fields,
        meta=meta,
    )
