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
from diskbridge.chemistry.hydrogen.api import ensure_h2_partition
from diskbridge.chemistry.shielding.healpix_columns import compute_pdr_shielding_healpix
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

from diskbridge.chemistry.models._gow17_numba import solve_gow17_equilibrium_cells_cgs


def _as_cgs_f64(q: Quantity, unit: str) -> np.ndarray:
    return np.ascontiguousarray(q.to(unit).magnitude, dtype=np.float64)


def run_gow17_pdr(rad: "RadModel", config: dict) -> ChemistryResult:
    cfg = resolve_model_config(("chemistry", "gow17_pdr"), overrides=config)

    if "max_iter" not in cfg:
        raise ValueError("gow17_pdr requires config['max_iter']")
    if "reltol" not in cfg:
        raise ValueError("gow17_pdr requires config['reltol']")
    if "abstol0" not in cfg:
        raise ValueError("gow17_pdr requires config['abstol0']")

    nside = int(cfg.get("nside"))
    b_kms = float(cfg.get("b_kms"))

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

    max_iter = int(cfg.get("max_iter"))

    reltol = float(cfg.get("reltol"))

    abstol0 = float(cfg.get("abstol0"))

    nH = rad.ensure_nH()
    Tdust = rad.ensure_dust_temperature()

    Tgas = rad.ensure_gas_temperature()
    if Tgas is None:
        Tgas = Tdust

    chi = rad.ensure_chi()

    ensure_h2_partition(rad, nH=nH, chi_dust=chi)

    sigma_d_per_H = rad.ensure_sigma_d_per_H()

    nH_cm3 = _as_cgs_f64(nH, "cm^-3")
    T_K = _as_cgs_f64(Tgas, "K")
    Tdust_K = _as_cgs_f64(Tdust, "K")
    chi_arr = _as_cgs_f64(chi, "dimensionless")
    sigma_cm2 = _as_cgs_f64(sigma_d_per_H, "cm^2")

    visser = VisserShielding(b_kms=float(b_kms))

    theta_h2, theta_co, theta_c, chi_eff_pdr = compute_pdr_shielding_healpix(
        mesh=rad.model.mesh,
        nH=nH,
        chi=chi,
        visser=visser,
        nCO=getattr(rad, "nco_gas", None),
        nH2=getattr(rad, "nH2", None),
        nside=nside,
        b_kms=b_kms,
        return_quantity=True,
    )

    theta_h2_arr = _as_cgs_f64(theta_h2, "dimensionless")
    theta_co_arr = _as_cgs_f64(theta_co, "dimensionless")
    theta_c_arr = _as_cgs_f64(theta_c, "dimensionless")
    chi_eff_pdr_arr = _as_cgs_f64(chi_eff_pdr, "dimensionless")

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

    y_out = np.zeros((ncells, N_Y), dtype=np.float64)
    status = np.zeros(ncells, dtype=np.int64)

    solve_gow17_equilibrium_cells_cgs(
        nH_cm3.reshape(ncells),
        T_K.reshape(ncells),
        Tdust_K.reshape(ncells),
        chi_arr.reshape(ncells),
        theta_h2_arr.reshape(ncells),
        theta_co_arr.reshape(ncells),
        theta_c_arr.reshape(ncells),
        chi_eff_pdr_arr.reshape(ncells),
        sigma_cm2.reshape(ncells),
        y0=y0,
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
        y_out=y_out,
        status_out=status,
    )

    y_out = y_out.reshape(shape + (N_Y,))

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

    validate_chemistry_state(
        nH=nH,
        nH2=nH2_cm3,
        nHI=nH_atom,
        nC=nC,
        nCplus=nCplus,
        nco_total=Quantity(nco_gas.magnitude + nco_ice.magnitude, "cm^-3"),
        check_pd=False,
    )

    if np.any(status != 0):
        bad = int(np.sum(status != 0))
        logger.warning(f"gow17_pdr: {bad} cells did not converge")

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
        "theta_co": theta_co,
        "chi_eff_pdr": chi_eff_pdr,
        "theta_h2": theta_h2,
    }

    rad.nco_gas = nco_gas
    rad.nco_ice = nco_ice
    rad.theta_co = theta_co
    rad.chi_eff = chi_eff_pdr
    rad.nH2 = nH2_cm3
    rad.nH_atom = nH_atom
    rad.nCplus = nCplus
    rad.nC = nC
    rad.ne = ne

    meta = {
        "model": "gow17_pdr",
        "nside": int(nside),
        "b_kms": float(b_kms),
        "ion_rate_s": float(ion_rate_s),
        "Zg": float(Zg),
        "Zd": float(Zd),
        "reltol": float(reltol),
        "abstol0": float(abstol0),
        "max_iter": int(max_iter),
    }

    return ChemistryResult(
        abundances=abundances,
        number_densities=number_densities,
        fields=fields,
        meta=meta,
    )
