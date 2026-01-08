from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np

from diskbridge._units import Quantity
from diskbridge._config import resolve_model_config
from diskbridge._logging import logger
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.shielding.columns_1d import (
    compute_pdr_shielding_1d,
    is_effectively_1d,
)
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
from diskbridge.chemistry.validation import validate_chemistry_state

import diskbridge._gow17 as _gow17

N_Y = _gow17.N_Y
N_PH = _gow17.N_PH
IPH_C = _gow17.IPH_C
IPH_CH = _gow17.IPH_CH
IPH_CO = _gow17.IPH_CO
IPH_OH = _gow17.IPH_OH
IPH_H2 = _gow17.IPH_H2
IPH_S = _gow17.IPH_S
IPH_SI = _gow17.IPH_SI

XC_STD = _gow17.XC_STD

I_HEP = 0
I_OHX = 1
I_CHX = 2
I_CO = 3
I_CP = 4
I_HCOP = 5
I_H2 = 6
I_HP = 7
I_H3P = 8
I_H2P = 9
I_SP = 10
I_SIP = 11
I_OP = 12
I_E = 13


def _as_cgs_f64(q: Quantity, unit: str) -> np.ndarray:
    return np.ascontiguousarray(q.to(unit).magnitude, dtype=np.float64)


def _broadcast_scalar_or_array(val, ncells: int) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
    if arr.size == 1:
        return np.full(ncells, arr[0], dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(ncells), dtype=np.float64)


def run_gow17(rad: "RadModel", config: dict) -> ChemistryResult:
    cfg = resolve_model_config(("chemistry", "gow17"), overrides=config)

    nside = int(cfg.get("nside", 8))
    b_kms = float(cfg.get("b_kms", 0.3))

    ion_rate_s = Quantity(cfg.get("ion_rate", "2e-16 s^-1")).to("1/s").magnitude

    Zg = float(cfg.get("Zg", 1.0))
    Zd = float(cfg.get("Zd", 1.0))

    fH2gr = float(cfg.get("fH2gr", 1.0))
    fHplusgr = float(cfg.get("fHplusgr", 1.0))
    fCplusgr = float(cfg.get("fCplusgr", 1.0))
    fHeplusgr = float(cfg.get("fHeplusgr", 1.0))
    fSplusgr = float(cfg.get("fSplusgr", 1.0))
    fSiplusgr = float(cfg.get("fSiplusgr", 1.0))
    fCplusCR = float(cfg.get("fCplusCR", 1.0))

    gradv = float(cfg.get("gradv", 1.0e-14))

    reltol = float(cfg.get("reltol", 1e-4))
    abstol0 = float(cfg.get("abstol0", 1e-15))
    tolfac = float(cfg.get("tolfac", 10.0))
    tmin = float(cfg.get("tmin", 3.16e10))
    tmax = float(cfg.get("tmax", 3.16e14))
    mxsteps = int(cfg.get("mxsteps", 10000))
    maxord = int(cfg.get("maxord", 5))
    userJac = bool(cfg.get("userJac", False))
    verbose = bool(cfg.get("verbose", False))

    shielding_max_iter = int(cfg.get("shielding_max_iter", 10))
    shielding_reltol = float(cfg.get("shielding_reltol", 1e-3))
    shielding_abstol = float(cfg.get("shielding_abstol", 1e-15))

    nH = rad.ensure_nH()
    Tgas = rad.ensure_gas_temperature()
    if Tgas is None:
        Tgas = rad.ensure_dust_temperature()

    chi = rad.ensure_chi()

    nH_cm3 = _as_cgs_f64(nH, "cm^-3")
    T_K = _as_cgs_f64(Tgas, "K")
    chi_dust_arr = _as_cgs_f64(chi, "dimensionless")

    shape = nH_cm3.shape
    ncells = nH_cm3.size

    nH_flat = nH_cm3.reshape(ncells)
    T_flat = T_K.reshape(ncells)
    chi_dust_flat = chi_dust_arr.reshape(ncells)

    Zd_arr = _broadcast_scalar_or_array(Zd, ncells)
    Zg_arr = _broadcast_scalar_or_array(Zg, ncells)
    ion_rate_arr = _broadcast_scalar_or_array(ion_rate_s, ncells)

    visser = VisserShielding(b_kms=float(b_kms))

    y0_single = np.zeros(N_Y, dtype=np.float64)
    y0_single[I_HEP] = 1.45e-08
    y0_single[I_H3P] = 2.68e-07
    y0_single[I_CP] = 1.0e-4
    y0_single[I_CO] = 1.0e-7
    y0_single[I_H2] = 0.1

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

    y_guess = np.zeros((ncells, N_Y), dtype=np.float64)
    y_prev = getattr(rad, "gow17_y", None)
    if y_prev is not None and np.shape(y_prev) == tuple(shape) + (N_Y,):
        y_guess[:, :] = np.ascontiguousarray(
            np.asarray(y_prev, dtype=np.float64).reshape(ncells, N_Y)
        )
    else:
        for j in range(N_Y):
            y_guess[:, j] = y0_single[j]

    theta_h2_arr = np.ones(ncells, dtype=np.float64)
    theta_co_arr = np.ones(ncells, dtype=np.float64)
    theta_c_arr = np.ones(ncells, dtype=np.float64)

    xCtot_flat = Zg_arr * float(XC_STD)

    status_acc = np.zeros(ncells, dtype=np.int32)

    for it in range(shielding_max_iter):
        xCO_old = np.ascontiguousarray(y_guess[:, I_CO].copy(), dtype=np.float64)
        xH2_old = np.ascontiguousarray(y_guess[:, I_H2].copy(), dtype=np.float64)

        xCO = y_guess[:, I_CO]
        xH2 = y_guess[:, I_H2]

        xC_neutral = xCtot_flat - (
            y_guess[:, I_HCOP]
            + y_guess[:, I_CHX]
            + xCO
            + y_guess[:, I_CP]
        )
        xC_neutral = np.maximum(xC_neutral, 0.0)

        nCO_cm3 = np.ascontiguousarray(
            (xCO * nH_flat).reshape(shape), dtype=np.float64
        )
        nH2_cm3 = np.ascontiguousarray(
            (xH2 * nH_flat).reshape(shape), dtype=np.float64
        )
        nC_cm3 = np.ascontiguousarray(
            (xC_neutral * nH_flat).reshape(shape), dtype=np.float64
        )

        if is_effectively_1d(rad.model.mesh, nH_cm3.shape):
            theta_h2_arr, theta_co_arr, theta_c_arr, _ = compute_pdr_shielding_1d(
                mesh=rad.model.mesh,
                nH=nH_cm3,
                chi=chi_dust_arr,
                visser=visser,
                nCO=nCO_cm3,
                nC=nC_cm3,
                nH2=nH2_cm3,
                b_kms=b_kms,
                outer="max",
                return_quantity=False,
            )
        else:
            from diskbridge.chemistry.shielding.healpix_columns import (
                compute_pdr_shielding_healpix,
            )

            theta_h2_arr, theta_co_arr, theta_c_arr, _ = compute_pdr_shielding_healpix(
                mesh=rad.model.mesh,
                nH=nH_cm3,
                chi=chi_dust_arr,
                visser=visser,
                nCO=nCO_cm3,
                nC=nC_cm3,
                nH2=nH2_cm3,
                nside=nside,
                b_kms=b_kms,
                return_quantity=False,
            )

        theta_h2_flat = theta_h2_arr.reshape(ncells)
        theta_co_flat = theta_co_arr.reshape(ncells)
        theta_c_flat = theta_c_arr.reshape(ncells)

        Gph = np.empty((ncells, N_PH), dtype=np.float64)
        Gph[:, :] = chi_dust_flat[:, None]
        Gph[:, IPH_C] *= theta_c_flat
        Gph[:, IPH_CO] *= theta_co_flat
        Gph[:, IPH_H2] *= theta_h2_flat

        GPE = np.ascontiguousarray(chi_dust_flat.copy(), dtype=np.float64)
        GISRF = np.ascontiguousarray(chi_dust_flat.copy(), dtype=np.float64)

        result = _gow17.solve_batch_equilibrium(
            y0=np.ascontiguousarray(y_guess, dtype=np.float64),
            nH=nH_flat,
            Tgas=T_flat,
            Zd=Zd_arr,
            Zg=Zg_arr,
            ion_rate=ion_rate_arr,
            GPE=GPE,
            GISRF=GISRF,
            Gph=Gph,
            reltol=reltol,
            abstol=abstol,
            mxsteps=mxsteps,
            maxord=maxord,
            tolfac=tolfac,
            tmin=tmin,
            tmax=tmax,
            const_temp=True,
            gradv=gradv,
            fH2gr=fH2gr,
            fHplusgr=fHplusgr,
            fCplusgr=fCplusgr,
            fHeplusgr=fHeplusgr,
            fSplusgr=fSplusgr,
            fSiplusgr=fSiplusgr,
            fCplusCR=fCplusCR,
            userJac=userJac,
            verbose=verbose,
        )

        y_new = result["y"]
        status_step = result["status"]

        status_acc = np.maximum(status_acc, np.asarray(status_step, dtype=np.int32))
        y_guess[:, :] = y_new

        xCO_new = y_guess[:, I_CO]
        xH2_new = y_guess[:, I_H2]

        denom_h2 = np.maximum(np.abs(xH2_new), shielding_abstol)
        denom_co = np.maximum(np.abs(xCO_new), shielding_abstol)
        d_h2 = np.max(np.abs(xH2_new - xH2_old) / denom_h2)
        d_co = np.max(np.abs(xCO_new - xCO_old) / denom_co)
        if max(d_h2, d_co) <= shielding_reltol:
            break

    y_out = y_guess.reshape(shape + (N_Y,))
    status = status_acc.reshape(shape)

    xCO = y_out[..., I_CO]
    xH2 = y_out[..., I_H2]

    nco_gas = Quantity(xCO * nH_cm3, "cm^-3")
    nH2_out = Quantity(xH2 * nH_cm3, "cm^-3")

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

    xCtot = xCtot_flat.reshape(shape)
    xC_neutral = xCtot - (
        y_out[..., I_HCOP]
        + y_out[..., I_CHX]
        + y_out[..., I_CO]
        + y_out[..., I_CP]
    )

    nCplus = Quantity(y_out[..., I_CP] * nH_cm3, "cm^-3")
    nC = Quantity(np.maximum(xC_neutral, 0.0) * nH_cm3, "cm^-3")
    ne = Quantity(xe * nH_cm3, "cm^-3")
    nH_atom = Quantity(np.maximum(xH_atom, 0.0) * nH_cm3, "cm^-3")

    if np.any(status != 0):
        bad = int(np.sum(status != 0))
        logger.warning(f"gow17: {bad} cells did not converge")
    else:
        validate_chemistry_state(
            nH=nH,
            nH2=nH2_out,
            nHI=nH_atom,
            nC=nC,
            nCplus=nCplus,
            nco_total=nco_gas,
            check_pd=False,
            rtol=float(reltol),
        )

    abundances = {
        "co": Quantity(xCO, "dimensionless"),
    }

    number_densities = {
        "co": nco_gas,
        "c+": nCplus,
        "catom": nC,
        "e": ne,
        "h2": nH2_out,
        "h": nH_atom,
    }

    fields = {
        "theta_co": Quantity(theta_co_arr.reshape(shape), "dimensionless"),
        "theta_h2": Quantity(theta_h2_arr.reshape(shape), "dimensionless"),
        "theta_c": Quantity(theta_c_arr.reshape(shape), "dimensionless"),
    }

    rad.gow17_y = y_out
    rad.nco_gas = nco_gas
    rad.theta_co = Quantity(theta_co_arr.reshape(shape), "dimensionless")
    rad.nH2 = nH2_out
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
        for code in (0, -1):
            status_hist[int(code)] = int(np.sum(status == code))

    meta = {
        "model": "gow17",
        "backend": "native",
        "nside": int(nside),
        "b_kms": float(b_kms),
        "ion_rate_s": float(ion_rate_s),
        "Zg": float(Zg),
        "Zd": float(Zd),
        "reltol": float(reltol),
        "abstol0": float(abstol0),
        "shielding_max_iter": int(shielding_max_iter),
        "shielding_reltol": float(shielding_reltol),
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
