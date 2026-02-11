from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np

import diskbridge
from diskbridge._units import Quantity
from diskbridge._config import resolve_model_config
from diskbridge._logging import logger
from diskbridge._constants import (
    E_BIND_CO,
    NU0_CO,
    F_DRAINE,
    N_LAY,
    N_SURF,
    Y_CO,
)
from diskbridge.model.profiles import compute_cell_volumes
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.shielding.columns_1d import (
    compute_pdr_shielding_1d,
    is_effectively_1d,
)
from diskbridge.chemistry.shielding.healpix_columns import (
    _prepare_healpix_geometry,
    compute_column_rays_healpix,
    compute_gradv_nh_weighted,
    compute_L_geo_from_pathlengths,
)
from diskbridge.chemistry.shielding.healpix_utils import (
    integrate_rays_with_pathlength,
)
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
from diskbridge.chemistry.shielding.w_rays_cache import maybe_ensure_W_rays
from diskbridge.chemistry.validation import validate_chemistry_state, gow17_budget_diagnostics

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
XHE = _gow17.XHE

I_HEP = _gow17.I_HEP
I_OHX = _gow17.I_OHX
I_CHX = _gow17.I_CHX
I_CO = _gow17.I_CO
I_CO_ICE = _gow17.I_CO_ICE
I_CP = _gow17.I_CP
I_HCOP = _gow17.I_HCOP
I_H2 = _gow17.I_H2
I_HP = _gow17.I_HP
I_H3P = _gow17.I_H3P
I_H2P = _gow17.I_H2P
I_SP = _gow17.I_SP
I_SIP = _gow17.I_SIP
I_OP = _gow17.I_OP
I_E = _gow17.I_E

KB_CGS = 1.380649e-16

_KPH_AVFAC = np.asarray([3.76, 2.12, 3.88, 2.66, 4.18, 3.10, 2.61], dtype=np.float64)
_SIGMA_PE_CGS = 1.0e-21
_SIGMA_ISRF_CGS = 3.0e-22


def _cv_cold(xH2: np.ndarray, xe: np.ndarray) -> np.ndarray:
    """Heat capacity per H nucleus for cold gas (cgs: erg/K per H)."""
    return 1.5 * KB_CGS * ((1.0 - 2.0 * xH2) + xH2 + XHE + xe)


def _electron_abundance(y: np.ndarray) -> np.ndarray:
    """Sum of all ion abundances to get electron abundance."""
    return (
        y[..., I_HEP]
        + y[..., I_CP]
        + y[..., I_HCOP]
        + y[..., I_H3P]
        + y[..., I_H2P]
        + y[..., I_HP]
        + y[..., I_SP]
        + y[..., I_SIP]
        + y[..., I_OP]
    )


def _as_cgs_f64(q: Quantity, unit: str) -> np.ndarray:
    return np.ascontiguousarray(q.to(unit).magnitude, dtype=np.float64)


def _broadcast_scalar_or_array(val, ncells: int) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
    if arr.size == 1:
        return np.full(ncells, arr[0], dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(ncells), dtype=np.float64)


def _maybe_quantity_to_float(val, unit: str) -> float:
    if isinstance(val, str):
        return float(Quantity(val).to(unit).magnitude)
    return float(val)


def run_gow17(rad: "RadModel", config: dict) -> ChemistryResult:
    cfg = resolve_model_config(("chemistry", "gow17"), overrides=config)

    nside = int(diskbridge.params.nside)
    b_kms = float(cfg.get("b_kms", 0.3))

    ion_rate_s = Quantity(cfg.get("ion_rate", "2e-16 s^-1")).to("1/s").magnitude

    Zg = float(cfg.get("Zg", 1.0))
    Zd_mode = str(cfg.get("Zd_mode", "scalar"))

    enable_co_phase = bool(cfg.get("enable_co_phase", False))

    fH2gr = float(cfg.get("fH2gr", 1.0))
    fHplusgr = float(cfg.get("fHplusgr", 1.0))
    fCplusgr = float(cfg.get("fCplusgr", 1.0))
    fHeplusgr = float(cfg.get("fHeplusgr", 1.0))
    fSplusgr = float(cfg.get("fSplusgr", 1.0))
    fSiplusgr = float(cfg.get("fSiplusgr", 1.0))
    fCplusCR = float(cfg.get("fCplusCR", 1.0))

    gradv_scalar = float(cfg.get("gradv", 1.0e-14))
    gradv_mode = str(cfg.get("gradv_mode", "scalar"))
    gradv_q = float(cfg.get("gradv_q", 1.5))
    gradv_N0 = float(cfg.get("gradv_N0", 1e21))
    gradv_p = float(cfg.get("gradv_p", 1.0))
    gradv_f_corr = float(cfg.get("gradv_f_corr", 4.0))
    gradv_gmin = float(cfg.get("gradv_gmin", 1e-20))
    gradv_gmax = float(cfg.get("gradv_gmax", 1e-8))

    Leff_CO_max_mode = str(cfg.get("Leff_CO_max_mode", "scalar"))
    L_geo_reduction = str(cfg.get("L_geo_reduction", "percentile_20"))
    L_geo_min = float(cfg.get("L_geo_min", 1e10))
    L_geo_max = float(cfg.get("L_geo_max", 1e20))
    Leff_CO_max_scalar = float(cfg.get("Leff_CO_max", 3.0e20))
    isDust_cooling = bool(cfg.get("isDust_cooling", False))
    isCoolingCOThin = bool(cfg.get("isCoolingCOThin", False))
    const_temp = bool(cfg.get("const_temp", True))

    reltol = float(cfg.get("reltol", 1e-4))
    abstol0 = float(cfg.get("abstol0", 1e-15))
    tolfac = float(cfg.get("tolfac", 10.0))
    tmin = _maybe_quantity_to_float(cfg.get("tmin", 3.16e10), "s")

    tmax_val = cfg.get("tmax", None)
    if tmax_val is None:
        tmax_val = cfg.get("t_end", 3.16e14)
    tmax = _maybe_quantity_to_float(tmax_val, "s")
    mxsteps = int(cfg.get("mxsteps", 10000))
    maxord = int(cfg.get("maxord", 5))
    userJac = bool(cfg.get("userJac", False))
    verbose = bool(cfg.get("verbose", False))

    shielding_max_iter_cfg = cfg.get("shielding_max_iter", None)
    if shielding_max_iter_cfg is None:
        shielding_max_iter_cfg = cfg.get("max_iter", 10)
    shielding_max_iter = int(shielding_max_iter_cfg)
    shielding_reltol = float(cfg.get("shielding_reltol", 1e-3))
    shielding_abstol = float(cfg.get("shielding_abstol", 1e-15))
    shielding_mix = float(cfg.get("shielding_mix", 1.0))
    shielding_theta_mix = float(cfg.get("shielding_theta_mix", 1.0))
    local_chi_factor = float(cfg.get("local_chi_factor", 0.5))
    shielding_outer_1d = str(cfg.get("shielding_outer_1d", "min"))

    chi_is_incident = bool(cfg.get("chi_is_incident", False))
    slab_1d_equilibrium = bool(cfg.get("slab_1d_equilibrium", False))

    nH = rad.ensure_nH()
    Tgas = rad.ensure_gas_temperature()
    if Tgas is None:
        Tgas = rad.ensure_dust_temperature()
    Tdust = rad.ensure_dust_temperature()

    chi = rad.ensure_chi()

    # Pre-compute directional UV weights (W_rays) once for reuse across
    # shielding iterations.  Returns None for 1-D meshes or when dustkappa
    # files are unavailable (shielding then falls back to isotropic averaging).
    maybe_ensure_W_rays(rad, nside=nside)

    nH_cm3 = _as_cgs_f64(nH, "cm^-3")
    T_K = _as_cgs_f64(Tgas, "K")
    chi_dust_arr = _as_cgs_f64(chi, "dimensionless")

    shape = nH_cm3.shape
    ncells = nH_cm3.size

    Tdust_K = _as_cgs_f64(Tdust, "K")

    nH_flat = nH_cm3.reshape(ncells)
    T_flat = T_K.reshape(ncells)
    Tdust_flat = Tdust_K.reshape(ncells)
    chi_dust_flat = chi_dust_arr.reshape(ncells)

    Av_flat = None
    if chi_is_incident:
        Av_q = getattr(rad, "Av", None)
        if Av_q is None:
            raise ValueError("gow17: chi_is_incident=True requires rad.Av to be set")
        Av_arr = _as_cgs_f64(Av_q, "dimensionless")
        if Av_arr.shape != shape:
            raise ValueError(f"gow17: rad.Av shape {Av_arr.shape} does not match nH shape {shape}")
        Av_flat = Av_arr.reshape(ncells)

    if slab_1d_equilibrium:
        if not chi_is_incident:
            raise ValueError("gow17: slab_1d_equilibrium requires chi_is_incident=True")
        if not is_effectively_1d(rad.model.mesh, nH_cm3.shape):
            raise ValueError("gow17: slab_1d_equilibrium requires an effectively-1D mesh")
        if Av_flat is None:
            raise RuntimeError("gow17: internal error (Av_flat missing)")

        NH_total_cfg = cfg.get("NH_total", None)
        NH_min_cfg = cfg.get("NH_min", None)
        if NH_total_cfg is None or NH_min_cfg is None:
            raise ValueError("gow17: slab_1d_equilibrium requires NH_total and NH_min in config")

        NH_total = _maybe_quantity_to_float(NH_total_cfg, "cm^-2")
        NH_min = _maybe_quantity_to_float(NH_min_cfg, "cm^-2")
        logNH = bool(cfg.get("logNH", True))
        field_geo = int(cfg.get("field_geo", 0))
        isdust = bool(cfg.get("isdust", True))
        isfsH2 = bool(cfg.get("isfsH2", True))
        isfsCO = bool(cfg.get("isfsCO", True))
        isfsC = bool(cfg.get("isfsC", True))

        nH0 = float(nH_flat[0])
        if not np.allclose(nH_flat, nH0, rtol=0.0, atol=0.0):
            raise ValueError("gow17: slab_1d_equilibrium requires uniform nH")

        chi0 = float(chi_dust_flat[0])
        if not np.allclose(chi_dust_flat, chi0, rtol=0.0, atol=0.0):
            raise ValueError("gow17: slab_1d_equilibrium requires uniform chi")

        sigma_d = rad.ensure_sigma_d_per_H()
        sigma_d_cm2 = _as_cgs_f64(sigma_d, "cm^2").reshape(ncells)
        valid = np.isfinite(sigma_d_cm2) & (sigma_d_cm2 > 0.0)
        if not np.any(valid):
            raise ValueError("gow17: slab_1d_equilibrium requires positive finite sigma_d_per_H")
        sigma_d_per_H_ref = float(np.nanmedian(sigma_d_cm2[valid]))
        Zd0 = float(np.nanmedian(sigma_d_cm2[valid]) / sigma_d_per_H_ref)

        Zg0 = float(Zg)
        ion0 = float(ion_rate_s)
        gradv0 = float(gradv_scalar)

        NCOeff_global = bool(cfg.get("NCOeff_global", True))
        bCO_L = bool(cfg.get("bCO_L", True))

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
            y0[I_E] = Cv0 * float(T_flat[0])

        abstol = np.full(N_Y, abstol0, dtype=np.float64)
        abstol[I_HEP] = 1.0e-15
        abstol[I_OHX] = 1.0e-15
        abstol[I_CHX] = 1.0e-15
        abstol[I_CO] = 1.0e-15
        abstol[I_CO_ICE] = 1.0e-15
        abstol[I_CP] = 1.0e-15
        abstol[I_HCOP] = max(abstol0, 1.0e-20)
        abstol[I_H2] = 1.0e-8
        abstol[I_HP] = 1.0e-15
        abstol[I_H3P] = 1.0e-15
        abstol[I_H2P] = 1.0e-15
        abstol[I_E] = float(
            _cv_cold(
                np.asarray([0.1], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
            )[0]
        )

        slab = _gow17.solve_slab_1d_equilibrium(
            nH=nH0,
            G0=(2.0 * chi0),
            ngrid=int(ncells),
            NH_total=float(NH_total),
            logNH=bool(logNH),
            NH_min=float(NH_min),
            field_geo=int(field_geo),
            isdust=bool(isdust),
            isfsH2=bool(isfsH2),
            isfsCO=bool(isfsCO),
            isfsC=bool(isfsC),
            Zg=float(Zg0),
            Zd=float(Zd0),
            ion_rate=float(ion0),
            reltol=float(reltol),
            abstol=abstol,
            mxsteps=int(mxsteps),
            maxord=int(maxord),
            tolfac=float(tolfac),
            tmin=float(tmin),
            tmax=float(tmax),
            verbose=bool(verbose),
            y0=y0,
            const_temp=bool(const_temp),
            Tgas=float(T_flat[0]),
            gradv=float(gradv0),
            NCOeff_global=bool(NCOeff_global),
            bCO_L=bool(bCO_L),
            fH2gr=float(fH2gr),
            fHplusgr=float(fHplusgr),
            fCplusgr=float(fCplusgr),
            fHeplusgr=float(fHeplusgr),
            fSplusgr=float(fSplusgr),
            fSiplusgr=float(fSiplusgr),
            fCplusCR=float(fCplusCR),
            co_sigma_d_per_H_ref=0.0,
            co_E_bind_co=0.0,
            co_nu0_co=0.0,
            co_F_DRAINE=0.0,
            co_Y_CO=0.0,
            co_N_SURF=0.0,
            co_N_LAY=0,
            userJac=bool(userJac),
        )

        y_guess = np.asarray(slab["y"], dtype=np.float64).reshape(ncells, N_Y)
        theta_h2_arr = np.asarray(slab["fShieldH2"], dtype=np.float64).reshape(shape)
        theta_co_arr = np.asarray(slab["fShieldCO"], dtype=np.float64).reshape(shape)
        theta_c_arr = np.ones_like(theta_co_arr, dtype=np.float64)

        y_out = y_guess.reshape(shape + (N_Y,))
        status = np.zeros(shape, dtype=np.int32)

        xCO = y_out[..., I_CO]
        xCO_ice = y_out[..., I_CO_ICE]
        xH2 = y_out[..., I_H2]

        nco_gas = Quantity(xCO * nH_cm3, "cm^-3")
        nco_ice = Quantity(xCO_ice * nH_cm3, "cm^-3")
        nH2_out = Quantity(xH2 * nH_cm3, "cm^-3")

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

        xCtot = np.full(shape, float(Zg0) * float(XC_STD), dtype=np.float64)
        xC_neutral = xCtot - (
            y_out[..., I_HCOP]
            + y_out[..., I_CHX]
            + y_out[..., I_CO]
            + y_out[..., I_CO_ICE]
            + y_out[..., I_CP]
        )

        nCplus = Quantity(y_out[..., I_CP] * nH_cm3, "cm^-3")
        nC = Quantity(np.maximum(xC_neutral, 0.0) * nH_cm3, "cm^-3")
        ne = Quantity(xe * nH_cm3, "cm^-3")
        nH_atom = Quantity(np.maximum(xH_atom, 0.0) * nH_cm3, "cm^-3")

        abundances = {
            "co": Quantity(xCO, "dimensionless"),
            "co_ice": Quantity(xCO_ice, "dimensionless"),
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
            "co_ice": nco_ice,
            "theta_co": Quantity(theta_co_arr, "dimensionless"),
            "theta_h2": Quantity(theta_h2_arr, "dimensionless"),
            "theta_c": Quantity(theta_c_arr, "dimensionless"),
            "chi_eff": Quantity(chi_dust_arr * theta_co_arr, "dimensionless"),
        }

        rad.gow17_y = y_out
        rad.nco_gas = nco_gas
        rad.nco_ice = nco_ice
        rad.theta_co = fields["theta_co"]
        rad.chi_eff = fields["chi_eff"]
        rad.nH2 = nH2_out
        rad.nH_atom = nH_atom
        rad.nCplus = nCplus
        rad.nC = nC
        rad.ne = ne

        n_fail = 0
        max_status = 0

        meta = {
            "model": "gow17",
            "enable_co_phase": bool(enable_co_phase),
            "nside": int(nside),
            "b_kms": float(b_kms),
            "ion_rate_s": float(ion_rate_s),
            "Zg": float(Zg0),
            "Zd": float(Zd0),
            "reltol": float(reltol),
            "abstol0": float(abstol0),
            "shielding_max_iter": int(shielding_max_iter),
            "shielding_reltol": float(shielding_reltol),
            "n_fail": int(n_fail),
            "max_status": int(max_status),
            "fail_idx_head": [],
            "status_hist": {0: int(ncells), -1: 0},
        }

        return ChemistryResult(
            abundances=abundances,
            number_densities=number_densities,
            fields=fields,
            meta=meta,
        )

    Zg_arr = _broadcast_scalar_or_array(Zg, ncells)
    ion_rate_arr = _broadcast_scalar_or_array(ion_rate_s, ncells)


    sigma_d = rad.ensure_sigma_d_per_H()
    sigma_d_cm2 = _as_cgs_f64(sigma_d, "cm^2").reshape(ncells)

    sigma_d_per_H_ref_cfg = cfg.get("sigma_d_per_H_ref", None)
    if sigma_d_per_H_ref_cfg is None:
        valid = np.isfinite(sigma_d_cm2) & (sigma_d_cm2 > 0.0)
        if not np.any(valid):
            raise ValueError(
                "gow17: cannot infer sigma_d_per_H_ref because sigma_d_per_H has no positive finite values"
            )
        sigma_d_per_H_ref = float(np.nanmedian(sigma_d_cm2[valid]))
    else:
        if isinstance(sigma_d_per_H_ref_cfg, str):
            sigma_d_per_H_ref = float(Quantity(sigma_d_per_H_ref_cfg).to("cm^2").magnitude)
        else:
            sigma_d_per_H_ref = float(sigma_d_per_H_ref_cfg)

    Zd_arr = np.divide(
        sigma_d_cm2,
        sigma_d_per_H_ref,
        out=np.ones(ncells, dtype=np.float64),
        where=(sigma_d_per_H_ref > 0.0),
    )


    Leff_CO_max_arr = _broadcast_scalar_or_array(Leff_CO_max_scalar, ncells)
    gradv_arr = _broadcast_scalar_or_array(gradv_scalar, ncells)

    candidate_mask_arr = None
    tracer = None
    dirs = None
    candidate_idx = None
    candidate_flat_idx = None
    cell_centers = None

    use_healpix_geometry = (
        (gradv_mode == "nh_weighted_shear+turb")
        or (Leff_CO_max_mode == "geo")
    )
    if use_healpix_geometry and (not is_effectively_1d(rad.model.mesh, nH_cm3.shape)):
        candidate_mask_arr = np.ones(shape, dtype=bool)
        tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
            rad.model.mesh,
            nside=int(nside),
            candidate_mask=candidate_mask_arr,
            cache_dir=None,
        )
        if candidate_idx.size > 0:
            candidate_flat_idx = np.ravel_multi_index(candidate_idx.T, dims=shape)

    if gradv_mode == "nh_weighted_shear+turb":
        if candidate_idx is None or candidate_idx.size == 0:
            logger.info("gradv_mode=nh_weighted_shear+turb: no candidate cells; using scalar gradv")
        else:
            if "vphi" not in rad.model.gas:
                raise ValueError("gradv_mode=nh_weighted_shear+turb requires gas field 'vphi'")
            vphi_cgs = _as_cgs_f64(rad.model.gas["vphi"].data, "cm/s")

            if rad.model.mesh.coord_system == "spherical":
                r_cent = rad.model.mesh.centers("r").to("cm").magnitude
                theta_cent = rad.model.mesh.centers("theta").to("radian").magnitude
                phi_cent = rad.model.mesh.centers("phi").to("radian").magnitude

                ir = candidate_idx[:, 0]
                itheta = candidate_idx[:, 1]
                iphi = candidate_idx[:, 2]

                r_cand = r_cent[ir]
                theta_cand = theta_cent[itheta]
                phi_cand = phi_cent[iphi]

                R_cyl = r_cand * np.sin(theta_cand)
                vphi_cand = np.abs(vphi_cgs[ir, itheta, iphi])
                Omega = np.where(R_cyl > 0.0, vphi_cand / R_cyl, 0.0)

                eR = np.zeros((candidate_idx.shape[0], 3), dtype=np.float64)
                eR[:, 0] = np.cos(phi_cand)
                eR[:, 1] = np.sin(phi_cand)

                volumes = compute_cell_volumes(rad.model)
                L_cell = np.cbrt(volumes)
                L_cell_cand = L_cell[ir, itheta, iphi]
            else:
                x_edges = rad.model.mesh.edges("x").to("cm").magnitude
                y_edges = rad.model.mesh.edges("y").to("cm").magnitude
                z_edges = rad.model.mesh.edges("z").to("cm").magnitude
                x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
                z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

                dx = np.diff(x_edges)
                dy = np.diff(y_edges)
                dz = np.diff(z_edges)
                volumes = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
                L_cell = np.cbrt(volumes)

                ix = candidate_idx[:, 0]
                iy = candidate_idx[:, 1]
                iz = candidate_idx[:, 2]

                x_cand = x_cent[ix]
                y_cand = y_cent[iy]
                R_cyl = np.sqrt(x_cand**2 + y_cand**2)
                vphi_cand = np.abs(vphi_cgs[ix, iy, iz])
                Omega = np.where(R_cyl > 0.0, vphi_cand / R_cyl, 0.0)

                eR = np.zeros((candidate_idx.shape[0], 3), dtype=np.float64)
                mask_R = R_cyl > 0.0
                eR[mask_R, 0] = x_cand[mask_R] / R_cyl[mask_R]
                eR[mask_R, 1] = y_cand[mask_R] / R_cyl[mask_R]
                eR[~mask_R, 0] = 1.0

                L_cell_cand = L_cell[ix, iy, iz]

            _, _, cols = compute_column_rays_healpix(
                rad.model.mesh,
                fields={"nh": nH_cm3},
                nside=int(nside),
                candidate_mask=None,
                tracer=tracer,
                dirs=dirs,
                candidate_idx=candidate_idx,
                cell_centers=cell_centers,
                self_weight=1.0,
            )
            NH_rays = cols["nh"]
            gradv_cand = compute_gradv_nh_weighted(
                NH_rays,
                dirs,
                Omega,
                eR,
                L_cell_cand,
                b_kms=float(b_kms),
                q=float(gradv_q),
                N0=float(gradv_N0),
                p=float(gradv_p),
                f_corr=float(gradv_f_corr),
                gmin=float(gradv_gmin),
                gmax=float(gradv_gmax),
            )
            gradv_arr[candidate_flat_idx] = gradv_cand
            rad.gradv_gow17 = gradv_arr.reshape(shape)

    if Leff_CO_max_mode == "geo":
        if candidate_idx is None or candidate_idx.size == 0:
            logger.info("Leff_CO_max_mode=geo: no candidate cells; using scalar Leff_CO_max")
        else:
            _, S_all = integrate_rays_with_pathlength(tracer, cell_centers, dirs, nH_cm3)
            L_geo_cand = compute_L_geo_from_pathlengths(
                S_all,
                reduction=L_geo_reduction,
                L_min=L_geo_min,
                L_max=L_geo_max,
            )
            Leff_CO_max_arr[candidate_flat_idx] = L_geo_cand
            rad.Lgeo_gow17 = Leff_CO_max_arr.reshape(shape)

    visser = VisserShielding(b_kms=float(b_kms))

    y0_single = np.zeros(N_Y, dtype=np.float64)
    y0_single[I_HEP] = 1.45e-08
    y0_single[I_H3P] = 2.68e-07
    y0_single[I_CP] = 1.0e-4
    y0_single[I_CO] = 1.0e-7
    y0_single[I_H2] = 0.1
    y0_single[I_CO_ICE] = 0.0

    abstol = np.full(N_Y, abstol0, dtype=np.float64)
    abstol[I_HEP] = 1.0e-15
    abstol[I_OHX] = 1.0e-15
    abstol[I_CHX] = 1.0e-15
    abstol[I_CO] = 1.0e-15
    abstol[I_CO_ICE] = 1.0e-15
    abstol[I_CP] = 1.0e-15
    abstol[I_HCOP] = max(abstol0, 1.0e-20)
    abstol[I_H2] = 1.0e-8
    abstol[I_HP] = 1.0e-15
    abstol[I_H3P] = 1.0e-15
    abstol[I_H2P] = 1.0e-15

    y_guess = np.zeros((ncells, N_Y), dtype=np.float64)
    y_prev = getattr(rad, "gow17_y", None)
    warm_start = y_prev is not None and np.shape(y_prev) == tuple(shape) + (N_Y,)
    if warm_start:
        y_guess[:, :] = np.ascontiguousarray(
            np.asarray(y_prev, dtype=np.float64).reshape(ncells, N_Y)
        )
    else:
        for j in range(N_Y):
            y_guess[:, j] = y0_single[j]

    if not enable_co_phase:
        y_guess[:, I_CO_ICE] = 0.0

    if not const_temp and not warm_start:
        xe0 = _electron_abundance(y_guess)
        Cv0 = _cv_cold(y_guess[:, I_H2], xe0)
        y_guess[:, I_E] = Cv0 * T_flat

    theta_h2_arr = np.ones(ncells, dtype=np.float64)
    theta_co_arr = np.ones(ncells, dtype=np.float64)
    theta_c_arr = np.ones(ncells, dtype=np.float64)

    xCtot_flat = Zg_arr * float(XC_STD)

    status_acc = np.zeros(ncells, dtype=np.int32)

    d_h2_hist = []
    d_co_hist = []

    for it in range(shielding_max_iter):
        xCO_old = np.ascontiguousarray(y_guess[:, I_CO].copy(), dtype=np.float64)
        xH2_old = np.ascontiguousarray(y_guess[:, I_H2].copy(), dtype=np.float64)

        xCO = y_guess[:, I_CO]
        xCO_ice = y_guess[:, I_CO_ICE]
        xH2 = y_guess[:, I_H2]

        xC_neutral = xCtot_flat - (
            y_guess[:, I_HCOP]
            + y_guess[:, I_CHX]
            + xCO
            + xCO_ice
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
            if chi_is_incident:
                if Av_flat is None:
                    raise RuntimeError("gow17: internal error (Av_flat missing)")

                NH_over_Zd = Av_flat * (1.87e21)
                NH_flat = np.divide(
                    NH_over_Zd,
                    Zd_arr,
                    out=np.zeros_like(NH_over_Zd, dtype=np.float64),
                    where=(Zd_arr > 0.0),
                )

                dNH = np.empty_like(NH_flat, dtype=np.float64)
                dNH[:-1] = NH_flat[1:] - NH_flat[:-1]
                dNH[-1] = 0.0

                xH2_use = np.ascontiguousarray(xH2, dtype=np.float64)
                xCO_use = np.ascontiguousarray(xCO, dtype=np.float64)
                xC_use = np.ascontiguousarray(xC_neutral, dtype=np.float64)

                N_H2 = np.zeros_like(NH_flat, dtype=np.float64)
                N_CO = np.zeros_like(NH_flat, dtype=np.float64)
                N_C = np.zeros_like(NH_flat, dtype=np.float64)
                if NH_flat.size >= 2:
                    N_H2[1:] = np.cumsum(xH2_use[:-1] * dNH[:-1])
                    N_CO[1:] = np.cumsum(xCO_use[:-1] * dNH[:-1])
                    N_C[1:] = np.cumsum(xC_use[:-1] * dNH[:-1])

                from diskbridge.chemistry.shielding.h2_db96 import h2_self_shielding_db96

                theta_h2_flat = h2_self_shielding_db96(N_H2, b5=float(b_kms))
                theta_co_flat = visser.theta("co", N_CO, N_H2, b_kms=float(b_kms))

                AH2 = 1.17e-8
                tau_H2 = 1.2e-14 * 2.0 * N_H2
                y = AH2 * tau_H2
                ry = np.exp(-y) / (1.0 + y)
                rc = np.exp(-1.6e-17 * N_C)
                theta_c_flat = rc * ry

                theta_h2_arr = theta_h2_flat.reshape(shape)
                theta_co_arr = theta_co_flat.reshape(shape)
                theta_c_arr = theta_c_flat.reshape(shape)
            else:
                theta_h2_arr, theta_co_arr, theta_c_arr, _, _ = compute_pdr_shielding_1d(
                    mesh=rad.model.mesh,
                    nH=nH_cm3,
                    chi=chi_dust_arr,
                    visser=visser,
                    nCO=nCO_cm3,
                    nC=nC_cm3,
                    nH2=nH2_cm3,
                    b_kms=b_kms,
                    outer=shielding_outer_1d,
                )
        else:
            from diskbridge.chemistry.shielding.healpix_columns import (
                compute_pdr_shielding_healpix,
            )

            W_rays = getattr(rad, "W_rays", None)
            theta_h2_arr, theta_co_arr, theta_c_arr, _, _ = compute_pdr_shielding_healpix(
                mesh=rad.model.mesh,
                nH=nH_cm3,
                chi=chi_dust_arr,
                visser=visser,
                nCO=nCO_cm3,
                nC=nC_cm3,
                nH2=nH2_cm3,
                nside=nside,
                b_kms=b_kms,
                W_rays=W_rays,
            )

        theta_h2_new = theta_h2_arr.reshape(ncells)
        theta_co_new = theta_co_arr.reshape(ncells)
        theta_c_new = theta_c_arr.reshape(ncells)

        # Optional theta mixing for shielding factor stabilisation.
        if it > 0 and shielding_theta_mix < 1.0:
            a = shielding_theta_mix
            theta_h2_flat = a * theta_h2_new + (1.0 - a) * theta_h2_flat
            theta_co_flat = a * theta_co_new + (1.0 - a) * theta_co_flat
            theta_c_flat = a * theta_c_new + (1.0 - a) * theta_c_flat
        else:
            theta_h2_flat = theta_h2_new
            theta_co_flat = theta_co_new
            theta_c_flat = theta_c_new

        Gph = np.empty((ncells, N_PH), dtype=np.float64)
        if chi_is_incident:
            if Av_flat is None:
                raise RuntimeError("gow17: internal error (Av_flat missing)")
            G0_half = 0.5 * chi_dust_flat
            Gph[:, :] = G0_half[:, None] * np.exp(-_KPH_AVFAC[None, :] * Av_flat[:, None])
            NH_over_Zd = Av_flat * (1.87e21)
            NH_flat = np.divide(
                NH_over_Zd,
                Zd_arr,
                out=np.zeros_like(NH_over_Zd, dtype=np.float64),
                where=(Zd_arr > 0.0),
            )
            GPE = np.ascontiguousarray(
                G0_half * np.exp(-NH_flat * _SIGMA_PE_CGS * Zd_arr),
                dtype=np.float64,
            )
            GISRF = np.ascontiguousarray(
                G0_half * np.exp(-NH_flat * _SIGMA_ISRF_CGS * Zd_arr),
                dtype=np.float64,
            )
        else:
            # chi_dust_flat is the dust-attenuated UV from RADMC-3D (Draine units).
            # The C solver convention (radfield.cpp) is Gph = (G0/2) * attenuation.
            # local_chi_factor controls whether 0.5 (default, one-sided flux) or
            # 1.0 (local energy-density interpretation) is applied.
            G0_scaled = local_chi_factor * chi_dust_flat
            Gph[:, :] = G0_scaled[:, None]
            GPE = np.ascontiguousarray(G0_scaled.copy(), dtype=np.float64)
            GISRF = np.ascontiguousarray(G0_scaled.copy(), dtype=np.float64)

        Gph[:, IPH_C] *= theta_c_flat
        Gph[:, IPH_CO] *= theta_co_flat
        Gph[:, IPH_H2] *= theta_h2_flat

        result = _gow17.solve_batch_equilibrium(
            y0=np.ascontiguousarray(y_guess, dtype=np.float64),
            nH=nH_flat,
            Tgas=T_flat,
            Tdust=Tdust_flat,
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
            const_temp=const_temp,
            gradv=gradv_arr,
            Leff_CO_max=Leff_CO_max_arr,
            isDust_cooling=isDust_cooling,
            isCoolingCOThin=isCoolingCOThin,
            fH2gr=fH2gr,
            fHplusgr=fHplusgr,
            fCplusgr=fCplusgr,
            fHeplusgr=fHeplusgr,
            fSplusgr=fSplusgr,
            fSiplusgr=fSiplusgr,
            fCplusCR=fCplusCR,
            co_sigma_d_per_H_ref=(float(sigma_d_per_H_ref) if enable_co_phase else 0.0),
            co_E_bind_co=(float(E_BIND_CO) if enable_co_phase else 0.0),
            co_nu0_co=(float(NU0_CO) if enable_co_phase else 0.0),
            co_F_DRAINE=(float(F_DRAINE) if enable_co_phase else 0.0),
            co_Y_CO=(float(Y_CO) if enable_co_phase else 0.0),
            co_N_SURF=(float(N_SURF) if enable_co_phase else 0.0),
            co_N_LAY=(int(N_LAY) if enable_co_phase else 0),
            userJac=userJac,
            verbose=verbose,
        )

        y_new = result["y"]
        status_step = result["status"]

        status_acc = np.maximum(status_acc, np.asarray(status_step, dtype=np.int32))
        if shielding_mix < 1.0:
            y_guess[:, :] = shielding_mix * y_new + (1.0 - shielding_mix) * y_guess
        else:
            y_guess[:, :] = y_new

        xCO_new = y_guess[:, I_CO]
        xH2_new = y_guess[:, I_H2]

        denom_h2 = np.maximum(np.abs(xH2_new), shielding_abstol)
        denom_co = np.maximum(np.abs(xCO_new), shielding_abstol)
        d_h2 = np.max(np.abs(xH2_new - xH2_old) / denom_h2)
        d_co = np.max(np.abs(xCO_new - xCO_old) / denom_co)
        d_h2_hist.append(float(d_h2))
        d_co_hist.append(float(d_co))
        if max(d_h2, d_co) <= shielding_reltol:
            break

    n_shielding_iter = len(d_h2_hist)

    y_out = y_guess.reshape(shape + (N_Y,))
    status = status_acc.reshape(shape)

    if not enable_co_phase:
        y_out[..., I_CO_ICE] = 0.0

    xCO = y_out[..., I_CO]
    xCO_ice = y_out[..., I_CO_ICE]
    xH2 = y_out[..., I_H2]

    nco_gas = Quantity(xCO * nH_cm3, "cm^-3")
    nco_ice = Quantity(xCO_ice * nH_cm3, "cm^-3")
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
        + y_out[..., I_CO_ICE]
        + y_out[..., I_CP]
    )

    nCplus = Quantity(y_out[..., I_CP] * nH_cm3, "cm^-3")
    nC = Quantity(np.maximum(xC_neutral, 0.0) * nH_cm3, "cm^-3")
    ne = Quantity(xe * nH_cm3, "cm^-3")
    nH_atom = Quantity(np.maximum(xH_atom, 0.0) * nH_cm3, "cm^-3")

    budget_diag = gow17_budget_diagnostics(
        y_out, xCtot=xCtot, rtol=float(reltol), logger=logger,
    )

    if np.any(status != 0):
        bad = int(np.sum(status != 0))
        logger.warning(f"gow17: {bad} cells did not converge")
    else:
        validate_chemistry_state(
            nH=nH,
            nH2=nH2_out,
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
        "h2": nH2_out,
        "h": nH_atom,
    }

    gow17_diag = {
        "shielding_iter": n_shielding_iter,
        "d_h2_hist": np.asarray(d_h2_hist, dtype=np.float64),
        "d_co_hist": np.asarray(d_co_hist, dtype=np.float64),
    }
    gow17_diag.update(budget_diag)

    fields = {
        "co_ice": nco_ice,
        "theta_co": Quantity(theta_co_flat.reshape(shape), "dimensionless"),
        "theta_h2": Quantity(theta_h2_flat.reshape(shape), "dimensionless"),
        "theta_c": Quantity(theta_c_flat.reshape(shape), "dimensionless"),
        "chi_eff": Quantity(chi_dust_arr * theta_co_flat.reshape(shape), "dimensionless"),
    }

    rad.gow17_y = y_out
    rad.nco_gas = nco_gas
    rad.nco_ice = nco_ice
    rad.theta_co = fields["theta_co"]
    rad.chi_eff = fields["chi_eff"]
    rad.nH2 = nH2_out
    rad.nH_atom = nH_atom
    rad.nCplus = nCplus
    rad.nC = nC
    rad.ne = ne

    if not const_temp:
        xe_out = _electron_abundance(y_out)
        Cv_out = _cv_cold(y_out[..., I_H2], xe_out)
        T_out = y_out[..., I_E] / Cv_out
        T_out = np.clip(T_out, 2.7, 1e6)
        rad.Tgas_gow17 = Quantity(T_out, "K")

    n_fail = int(np.sum(status != 0))
    max_status = int(np.max(status)) if status.size else 0

    fail_idx = np.flatnonzero(status != 0)
    fail_idx_head = fail_idx[:16].astype(np.int64)
    status_hist = {}
    if status.size:
        for code in (0, -1):
            status_hist[int(code)] = int(np.sum(status == code))

    Zd = float(np.mean(Zd_arr))

    meta = {
        "model": "gow17",
        "enable_co_phase": bool(enable_co_phase),
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
        "gow17_diagnostics": gow17_diag,
    }

    return ChemistryResult(
        abundances=abundances,
        number_densities=number_densities,
        fields=fields,
        meta=meta,
    )
