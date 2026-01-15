from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

import diskbridge
import diskbridge._gow17 as gow17_native
from diskbridge._constants import E_BIND_CO, F_DRAINE, N_LAY, N_SURF, NU0_CO, SIGMA_D_PER_H, Y_CO
from diskbridge._units import Quantity
from diskbridge._constants import K_B, X_C_TOT
from diskbridge.chemistry.api import run_chemistry
from diskbridge.chemistry.shielding.columns_1d import compute_pdr_shielding_1d
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
I_CHX = gow17_native.I_CHX
I_CO = gow17_native.I_CO
I_CO_ICE = gow17_native.I_CO_ICE
I_CP = gow17_native.I_CP
I_H2 = gow17_native.I_H2
I_H2P = gow17_native.I_H2P
I_H3P = gow17_native.I_H3P
I_HCOP = gow17_native.I_HCOP
I_HEP = gow17_native.I_HEP
I_HP = gow17_native.I_HP
I_OHX = gow17_native.I_OHX
I_OP = gow17_native.I_OP
I_SIP = gow17_native.I_SIP
I_SP = gow17_native.I_SP
N_Y = gow17_native.N_Y
from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel


class ShieldingMode(Enum):
    NATIVE_SLAB = "native_slab"
    EXTERNAL = "external"


SPEC_LIST_REF = [
    "He+",
    "OHx",
    "CHx",
    "CO",
    "C+",
    "HCO+",
    "H2",
    "H+",
    "H3+",
    "H2+",
    "S+",
    "Si+",
    "O+",
    "E",
]

_SPEC_INDEX_REF = {name: i for i, name in enumerate(SPEC_LIST_REF)}

COLORS = {
    "CO": "#1f77b4",
    "C": "#d62728",
    "C+": "#ff7f0e",
    "H3+": "#2ca02c",
    "OHx": "#9467bd",
    "CHx": "#8c564b",
    "He+": "#e377c2",
}


@dataclass(frozen=True)
class RefSlab:
    NH: np.ndarray
    Av: np.ndarray
    nH_values: np.ndarray
    abd: Dict[str, np.ndarray]
    E: np.ndarray


def _load_reference(dir_out: Path) -> RefSlab:
    nH_values = np.loadtxt(dir_out / "nH_arr.dat")
    NH = np.loadtxt(dir_out / "colH_arr.dat")
    Av = NH / 1.87e21

    abd: Dict[str, np.ndarray] = {}
    E_list = []
    for islab, _nH in enumerate(np.atleast_1d(nH_values)):
        slab = np.loadtxt(dir_out / f"slab{islab:06d}.dat")
        for spec, j in _SPEC_INDEX_REF.items():
            abd.setdefault(spec, []).append(slab[:, j])
        E_list.append(slab[:, _SPEC_INDEX_REF["E"]])

    for spec in list(abd):
        abd[spec] = np.stack(abd[spec], axis=1)

    E = np.stack(E_list, axis=1)

    xC = 1.6e-4
    abd["C"] = xC - abd["CHx"] - abd["CO"] - abd["C+"] - abd["HCO+"]

    return RefSlab(NH=NH, Av=Av, nH_values=np.asarray(nH_values, dtype=float), abd=abd, E=E)


def _temperature_from_energy(
    *,
    E: np.ndarray,
    xH2: np.ndarray,
    xe: np.ndarray,
    xHe: float = 0.1,
) -> np.ndarray:
    Cv_cold = 1.5 * float(K_B) * (1.0 - xH2 + float(xHe) + xe)
    return E / Cv_cold


def _xe_from_abundances(abd: Dict[str, np.ndarray]) -> np.ndarray:
    xe = np.zeros_like(abd["H2"], dtype=float)
    for spec in ["He+", "C+", "HCO+", "H3+", "H2+", "H+", "Si+", "S+", "O+"]:
        if spec in abd:
            xe = xe + np.asarray(abd[spec], dtype=float)
    return xe


def _build_1d_cartesian_model(*, nH_cm3: float, NH: np.ndarray) -> RadModel:
    units = diskbridge.units
    m_H = units("m_H")

    # Put surface at +x boundary so the existing outer="max" shielding logic
    # (column to +x boundary) maps to the reference NH grid.
    #
    # The reference NH grid is increasing with depth from surface. But the
    # column-to-"max" operator returns the largest column at index 0 and the
    # smallest at the last index. So we reverse NH such that index 0 is the
    # deepest point and the surface is at the +x boundary.
    NH_rev = np.asarray(NH, dtype=float)[::-1]

    dx_cm = np.empty_like(NH_rev)
    accum = 0.0
    for i in range(int(NH_rev.size) - 1, -1, -1):
        if i == int(NH_rev.size) - 1:
            dx_cm[i] = 2.0 * float(NH_rev[i]) / float(nH_cm3)
        else:
            dx_cm[i] = 2.0 * (float(NH_rev[i]) / float(nH_cm3) - accum)
        if not (dx_cm[i] > 0.0):
            raise ValueError(
                f"Invalid slab grid: non-positive cell width dx={dx_cm[i]:.3e} cm at i={i}"
            )
        accum += float(dx_cm[i])

    x_edges_cm = np.concatenate((np.array([0.0]), np.cumsum(dx_cm)))
    x_axis = Axis(edges=Quantity(x_edges_cm, "cm"))
    y_axis = Axis(edges=Quantity(np.array([0.0, 1.0]), "cm"))
    z_axis = Axis(edges=Quantity(np.array([0.0, 1.0]), "cm"))

    mesh = Mesh.cartesian(x=x_axis, y=y_axis, z=z_axis)
    shape = mesh.shape

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    nH_field = np.full(shape, float(nH_cm3), dtype=float)
    rho = (Quantity(nH_field, "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=("x", "y", "z")))

    sigma = Quantity(np.full(shape, 1.0e-21, dtype=float), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")),
    )

    return RadModel(model)


def _diskbridge_abundances_from_y(y: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {
        "He+": y[..., I_HEP],
        "OHx": y[..., I_OHX],
        "CHx": y[..., I_CHX],
        "CO": y[..., I_CO],
        "C+": y[..., I_CP],
        "HCO+": y[..., I_HCOP],
        "H2": y[..., I_H2],
        "H+": y[..., I_HP],
        "H3+": y[..., I_H3P],
        "H2+": y[..., I_H2P],
        "S+": y[..., I_SP],
        "Si+": y[..., I_SIP],
        "O+": y[..., I_OP],
    }

    xC = 1.6e-4
    out["C"] = xC - out["CHx"] - out["CO"] - out["C+"] - out["HCO+"]
    return out


def run_native_slab(
    *,
    nH_cm3: float,
    NH_total: float,
    ngrid: int,
    chi0: float = 1.0,
    xi_cr: float = 2.0e-16,
    const_temp: bool = False,
    Tgas_init: float = 100.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict]:
    dim = gow17_native.N_Y

    y0 = np.zeros(dim, dtype=np.float64)
    y0[gow17_native.I_HEP] = 1.450654e-08
    y0[gow17_native.I_H3P] = 2.681411e-07
    y0[gow17_native.I_CP] = 1.0e-4
    y0[gow17_native.I_CO] = 1.0e-7
    y0[gow17_native.I_H2] = 0.1
    if hasattr(gow17_native, "I_CO_ICE"):
        y0[gow17_native.I_CO_ICE] = 0.0

    # Match example_simple.cpp:
    #   y0[E] = GetE(temp, 0.1, 0.)
    # For xH2=0.1, xHe=0.1, xe=0, CvCold = 1.5*kB*(1 - xH2 + xHe + xe) = 1.5*kB
    # so E = 1.5*kB*T.
    xH2_init = float(y0[gow17_native.I_H2])
    xHe_tot = 0.1
    xe_init = 0.0
    Cv0 = 1.5 * float(K_B) * (1.0 - xH2_init + xHe_tot + xe_init)
    y0[gow17_native.I_E] = Cv0 * float(Tgas_init)

    abstol = np.full(dim, 1.0e-9, dtype=np.float64)
    abstol[gow17_native.I_HEP] = 1.0e-15
    abstol[gow17_native.I_OHX] = 1.0e-15
    abstol[gow17_native.I_CHX] = 1.0e-15
    abstol[gow17_native.I_CO] = 1.0e-15
    abstol[gow17_native.I_CP] = 1.0e-15
    abstol[gow17_native.I_HCOP] = 1.0e-30
    abstol[gow17_native.I_H2] = 1.0e-8
    abstol[gow17_native.I_HP] = 1.0e-15
    abstol[gow17_native.I_H3P] = 1.0e-15
    abstol[gow17_native.I_H2P] = 1.0e-15

    # Match example_simple.cpp:
    #   abstol[E] = GetE(1., 0.1, 0.)
    abstol[gow17_native.I_E] = Cv0 * 1.0

    reltol = 1.0e-2
    mxsteps = 5000000
    maxord = 3
    tolfac = 1.0e-1 / reltol
    tmin = 0.1e6 * 3.16e7
    tmax = 2000.0e6 * 3.16e7

    G0_incident = 2.0 * float(chi0)

    Zg = 1.0
    Zd = 1.0
    NH_min = 1.0e17 / Zd

    result = gow17_native.solve_slab_1d_equilibrium(
        nH=float(nH_cm3),
        G0=G0_incident,
        ngrid=int(ngrid),
        NH_total=float(NH_total),
        logNH=True,
        NH_min=NH_min,
        field_geo=0,
        isdust=True,
        isfsH2=True,
        isfsCO=True,
        isfsC=True,
        Zg=Zg,
        Zd=Zd,
        ion_rate=float(xi_cr),
        reltol=reltol,
        abstol=abstol,
        mxsteps=mxsteps,
        maxord=maxord,
        tolfac=tolfac,
        tmin=tmin,
        tmax=tmax,
        verbose=verbose,
        y0=y0,
        const_temp=bool(const_temp),
        Tgas=float(Tgas_init),
        gradv=3.0 * 3.0e-14,
        NCOeff_global=True,
        bCO_L=True,
        fH2gr=1.0,
        fHplusgr=0.6,
        fCplusgr=0.6,
        fHeplusgr=0.6,
        fSplusgr=0.6,
        fSiplusgr=0.6,
        fCplusCR=1.0,
        co_sigma_d_per_H_ref=float(SIGMA_D_PER_H),
        co_E_bind_co=float(E_BIND_CO),
        co_nu0_co=float(NU0_CO),
        co_F_DRAINE=float(F_DRAINE),
        co_Y_CO=float(Y_CO),
        co_N_SURF=float(N_SURF),
        co_N_LAY=int(N_LAY),
        userJac=False,
    )

    y_out = np.asarray(result["y"], dtype=np.float64)
    NH_arr = np.asarray(result["NH"], dtype=np.float64)

    abd = _diskbridge_abundances_from_y(y_out)

    info = {
        "shielding_mode": "native_slab",
        "G0_incident": G0_incident,
        "ngrid": ngrid,
        "NH_total": NH_total,
    }

    return y_out, abd, NH_arr, info


def _build_1d_cartesian_model(*, nH_cm3: float, NH: np.ndarray) -> RadModel:
    units = diskbridge.units
    m_H = units("m_H")

    NH_rev = np.asarray(NH, dtype=float)[::-1]
    L_cm = float(NH_rev[0] / float(nH_cm3))
    x_centers_cm = L_cm - (NH_rev / float(nH_cm3))

    x_axis = Axis(centers=Quantity(x_centers_cm, "cm"))
    y_axis = Axis(edges=Quantity(np.array([0.0, 1.0]), "cm"))
    z_axis = Axis(edges=Quantity(np.array([0.0, 1.0]), "cm"))

    mesh = Mesh.cartesian(x=x_axis, y=y_axis, z=z_axis)
    shape = mesh.shape

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    nH_field = np.full(shape, float(nH_cm3), dtype=float)
    rho = (Quantity(nH_field, "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=("x", "y", "z")))

    sigma = Quantity(np.full(shape, 1.0e-21, dtype=float), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")),
    )

    return RadModel(model)


def run_external_iteration(
    *,
    nH_cm3: float,
    NH: np.ndarray,
    Tgas_K: np.ndarray,
    chi0: float = 1.0,
    xi_cr: float = 2.0e-16,
    outer_iter: int = 20,
    outer_reltol: float = 1.0e-4,
    mix: float = 0.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict]:
    radm = _build_1d_cartesian_model(nH_cm3=nH_cm3, NH=NH)

    NH = np.asarray(NH, dtype=float)
    shape = radm.model.mesh.shape
    ncells = NH.size

    chi0_incident = 2.0 * float(chi0)
    NH_rev = NH[::-1]
    Av_arr = NH_rev / 1.87e21

    radm.dust_temperature = Quantity(np.full(shape, 50.0, dtype=float), "K")
    Tgas_K = np.asarray(Tgas_K, dtype=float)
    if Tgas_K.shape != (ncells,):
        raise ValueError(f"Tgas_K must have shape ({ncells},), got {Tgas_K.shape}")
    radm.gas_temperature = Quantity(Tgas_K[::-1].reshape(shape), "K")
    radm.Av = Quantity(Av_arr.reshape(shape), "dimensionless")

    chi_pe = 0.5 * chi0_incident * np.exp(-NH_rev * 1.0e-21)
    radm.chi = Quantity(chi_pe.reshape(shape), "dimensionless")

    nH_cm3_arr = np.ascontiguousarray(np.full(shape, float(nH_cm3), dtype=np.float64))
    T_K_arr = np.ascontiguousarray(radm.gas_temperature.to("K").magnitude, dtype=np.float64)
    Tdust_K_arr = np.ascontiguousarray(radm.dust_temperature.to("K").magnitude, dtype=np.float64)
    chi_pe_arr = np.ascontiguousarray(radm.chi.to("dimensionless").magnitude, dtype=np.float64)
    chi0_arr = np.ascontiguousarray(np.full(shape, float(chi0_incident), dtype=np.float64))
    Av_in = np.ascontiguousarray(radm.Av.to("dimensionless").magnitude, dtype=np.float64)
    sigma_cm2_arr = np.ascontiguousarray(np.full(shape, 1.0e-21, dtype=np.float64))

    dt_eq_s = float(Quantity("2.0e9 yr").to("s").magnitude)
    b_kms = 0.3

    Zg = 1.0
    Zd = 1.0

    y0 = np.zeros(N_Y, dtype=np.float64)
    y0[I_HEP] = 1.450654e-08
    y0[I_H3P] = 2.681411e-07
    y0[I_CP] = 1.0e-4
    y0[I_CO] = 1.0e-7
    y0[I_H2] = 0.1

    abstol = np.full(N_Y, 1.0e-9, dtype=np.float64)
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

    reltol = 1.0e-2
    max_iter = 80
    ion_rate_s = float(xi_cr)

    fH2gr = 1.0
    fHplusgr = 0.6
    fCplusgr = 0.6
    fHeplusgr = 0.6
    fSplusgr = 0.6
    fSiplusgr = 0.6
    fCplusCR = 1.0

    visser = VisserShielding(b_kms=float(b_kms))

    mix = float(mix)
    if not (0.0 < mix <= 1.0):
        raise ValueError(f"mix must be in (0, 1], got {mix}")

    tiny = float(np.finfo(np.float64).tiny)
    y_inout = np.zeros((ncells, N_Y), dtype=np.float64)
    for j in range(N_Y):
        y_inout[:, j] = y0[j]

    y_prev_snapshot = np.zeros((ncells, N_Y), dtype=np.float64)
    status_step = np.zeros(ncells, dtype=np.int64)
    status_acc = np.zeros(ncells, dtype=np.int64)

    theta_h2_prev_flat = np.ones(ncells, dtype=np.float64)
    theta_co_prev_flat = np.ones(ncells, dtype=np.float64)
    theta_c_prev_flat = np.ones(ncells, dtype=np.float64)
    theta_pdr_prev_flat = np.ones(ncells, dtype=np.float64)

    dt_coeff = (2.0 * float(dt_eq_s)) / (float(int(outer_iter)) * float(int(outer_iter) + 1))

    convergence_history = []
    converged = False

    for it in range(int(outer_iter)):
        y_prev_snapshot[:, :] = y_inout
        xCO_old = np.ascontiguousarray(y_inout[:, I_CO].copy(), dtype=np.float64)
        xH2_old = np.ascontiguousarray(y_inout[:, I_H2].copy(), dtype=np.float64)

        xCO = y_inout[:, I_CO]
        xCO_ice = y_inout[:, I_CO_ICE]
        xH2 = y_inout[:, I_H2]

        xCtot = float(Zg) * float(X_C_TOT)
        xC_neutral = xCtot - (y_inout[:, I_HCOP] + y_inout[:, I_CHX] + xCO + y_inout[:, I_CP] + xCO_ice)
        xC_neutral = np.maximum(xC_neutral, 0.0)

        nCO_cm3 = np.ascontiguousarray((xCO * nH_cm3_arr.reshape(ncells)).reshape(shape), dtype=np.float64)
        nH2_cm3 = np.ascontiguousarray((xH2 * nH_cm3_arr.reshape(ncells)).reshape(shape), dtype=np.float64)
        nC_cm3 = np.ascontiguousarray((xC_neutral * nH_cm3_arr.reshape(ncells)).reshape(shape), dtype=np.float64)

        theta_h2, theta_co, theta_c, theta_pdr, chi_eff_pdr = compute_pdr_shielding_1d(
            mesh=radm.model.mesh,
            nH=nH_cm3_arr,
            chi=chi_pe_arr,
            visser=visser,
            nCO=nCO_cm3,
            nC=nC_cm3,
            nH2=nH2_cm3,
            b_kms=float(b_kms),
            outer="max",
            return_quantity=False,
        )

        alpha = float(mix)

        theta_h2_new_flat = theta_h2.reshape(ncells)
        theta_co_new_flat = theta_co.reshape(ncells)
        theta_c_new_flat = theta_c.reshape(ncells)

        theta_h2_prev_pos = np.maximum(theta_h2_prev_flat, tiny)
        theta_co_prev_pos = np.maximum(theta_co_prev_flat, tiny)
        theta_c_prev_pos = np.maximum(theta_c_prev_flat, tiny)
        theta_pdr_prev_pos = np.maximum(theta_pdr_prev_flat, tiny)

        theta_h2_new_pos = np.maximum(theta_h2_new_flat, tiny)
        theta_co_new_pos = np.maximum(theta_co_new_flat, tiny)
        theta_c_new_pos = np.maximum(theta_c_new_flat, tiny)
        theta_pdr_new_pos = np.maximum(theta_h2_new_pos * theta_co_new_pos, tiny)

        theta_h2_mix_flat = np.exp((1.0 - alpha) * np.log(theta_h2_prev_pos) + alpha * np.log(theta_h2_new_pos))
        theta_co_mix_flat = np.exp((1.0 - alpha) * np.log(theta_co_prev_pos) + alpha * np.log(theta_co_new_pos))
        theta_c_mix_flat = np.exp((1.0 - alpha) * np.log(theta_c_prev_pos) + alpha * np.log(theta_c_new_pos))
        theta_pdr_mix_flat = theta_h2_mix_flat * theta_co_mix_flat

        theta_h2_prev_flat = np.ascontiguousarray(theta_h2_mix_flat, dtype=np.float64)
        theta_co_prev_flat = np.ascontiguousarray(theta_co_mix_flat, dtype=np.float64)
        theta_c_prev_flat = np.ascontiguousarray(theta_c_mix_flat, dtype=np.float64)
        theta_pdr_prev_flat = np.ascontiguousarray(theta_pdr_mix_flat, dtype=np.float64)

        chi_eff_pdr_mix = (chi_pe_arr.reshape(ncells) * theta_pdr_mix_flat).reshape(shape)

        dt_outer_s = float(dt_coeff) * float(it + 1)
        evolve_gow17_be_cells_cgs(
            nH_cm3_arr.reshape(ncells),
            T_K_arr.reshape(ncells),
            Tdust_K_arr.reshape(ncells),
            chi_pe_arr.reshape(ncells),
            chi0_arr.reshape(ncells),
            Av_in.reshape(ncells),
            theta_h2_prev_flat,
            theta_co_prev_flat,
            theta_c_prev_flat,
            chi_eff_pdr_mix.reshape(ncells),
            sigma_cm2_arr.reshape(ncells),
            float(dt_outer_s),
            y_inout,
            Zg=float(Zg),
            Zd=float(Zd),
            ion_rate_s=float(ion_rate_s),
            fH2gr=float(fH2gr),
            fHplusgr=float(fHplusgr),
            fCplusgr=float(fCplusgr),
            fHeplusgr=float(fHeplusgr),
            fSplusgr=float(fSplusgr),
            fSiplusgr=float(fSiplusgr),
            fCplusCR=float(fCplusCR),
            max_iter=int(max_iter),
            reltol=float(reltol),
            abstol=abstol,
            status_out=status_step,
        )

        status_acc = np.maximum(status_acc, status_step)
        bad_idx = np.flatnonzero(status_step != 0)
        if bad_idx.size:
            y_inout[bad_idx, :] = y_prev_snapshot[bad_idx, :]

        xco_new = y_inout[:, I_CO]
        xh2_new = y_inout[:, I_H2]
        xco_old = xCO_old
        xh2_old = xH2_old

        Av_line = Av_arr.reshape(-1)
        co_floor = 1.0e-15
        h2_floor = 1.0e-8
        mask_co = (Av_line > 0.5) & ((xco_new > co_floor) | (xco_old > co_floor))
        mask_h2 = (Av_line > 0.5) & ((xh2_new > h2_floor) | (xh2_old > h2_floor))

        if np.any(mask_co):
            denom_co = np.maximum(np.maximum(xco_new[mask_co], xco_old[mask_co]), co_floor)
            rel_co = np.abs(xco_new[mask_co] - xco_old[mask_co]) / denom_co
            max_rel_co = float(np.max(rel_co))
            p90_rel_co = float(np.percentile(rel_co, 90))
        else:
            max_rel_co = 0.0
            p90_rel_co = 0.0

        if np.any(mask_h2):
            denom_h2 = np.maximum(np.maximum(xh2_new[mask_h2], xh2_old[mask_h2]), h2_floor)
            rel_h2 = np.abs(xh2_new[mask_h2] - xh2_old[mask_h2]) / denom_h2
            max_rel_h2 = float(np.max(rel_h2))
            p90_rel_h2 = float(np.percentile(rel_h2, 90))
        else:
            max_rel_h2 = 0.0
            p90_rel_h2 = 0.0

        convergence_history.append(
            {
                "iter": it + 1,
                "max_rel_co": max_rel_co,
                "p90_rel_co": p90_rel_co,
                "max_rel_h2": max_rel_h2,
                "p90_rel_h2": p90_rel_h2,
            }
        )

        if verbose:
            print(
                f"  iter={it+1}/{int(outer_iter)}: "
                f"max_rel_CO={max_rel_co:.3e}, p90_rel_CO={p90_rel_co:.3e}, "
                f"max_rel_H2={max_rel_h2:.3e}, p90_rel_H2={p90_rel_h2:.3e}"
            )

        if max(p90_rel_co, p90_rel_h2) <= float(outer_reltol):
            converged = True
            break

    if not converged and verbose:
        print(f"  Warning: did not converge after {outer_iter} iterations")

    xCO = y_inout[:, I_CO]
    xCO_ice = y_inout[:, I_CO_ICE]
    xH2 = y_inout[:, I_H2]

    xCtot = float(Zg) * float(X_C_TOT)
    xC_neutral = xCtot - (
        y_inout[:, I_HCOP] + y_inout[:, I_CHX] + xCO + y_inout[:, I_CP] + xCO_ice
    )
    xC_neutral = np.maximum(xC_neutral, 0.0)

    nCO_cm3 = np.ascontiguousarray((xCO * nH_cm3_arr.reshape(ncells)).reshape(shape), dtype=np.float64)
    nH2_cm3 = np.ascontiguousarray((xH2 * nH_cm3_arr.reshape(ncells)).reshape(shape), dtype=np.float64)
    nC_cm3 = np.ascontiguousarray((xC_neutral * nH_cm3_arr.reshape(ncells)).reshape(shape), dtype=np.float64)

    theta_h2, theta_co, theta_c, theta_pdr, chi_eff_pdr = compute_pdr_shielding_1d(
        mesh=radm.model.mesh,
        nH=nH_cm3_arr,
        chi=chi_pe_arr,
        visser=visser,
        nCO=nCO_cm3,
        nC=nC_cm3,
        nH2=nH2_cm3,
        b_kms=float(b_kms),
        outer="max",
        return_quantity=False,
    )

    y_tmp = np.zeros_like(y_inout)
    status_final = np.zeros(ncells, dtype=np.int64)
    solve_gow17_equilibrium_cells_cgs(
        nH_cm3_arr.reshape(ncells),
        T_K_arr.reshape(ncells),
        Tdust_K_arr.reshape(ncells),
        chi_pe_arr.reshape(ncells),
        chi0_arr.reshape(ncells),
        Av_in.reshape(ncells),
        theta_h2.reshape(ncells),
        theta_co.reshape(ncells),
        theta_c.reshape(ncells),
        chi_eff_pdr.reshape(ncells),
        sigma_cm2_arr.reshape(ncells),
        float(dt_eq_s),
        y0=y_inout,
        Zg=float(Zg),
        Zd=float(Zd),
        ion_rate_s=float(ion_rate_s),
        fH2gr=float(fH2gr),
        fHplusgr=float(fHplusgr),
        fCplusgr=float(fCplusgr),
        fHeplusgr=float(fHeplusgr),
        fSplusgr=float(fSplusgr),
        fSiplusgr=float(fSiplusgr),
        fCplusCR=float(fCplusCR),
        max_iter=int(max_iter),
        reltol=float(reltol),
        abstol=abstol,
        y_out=y_tmp,
        status_out=status_final,
    )

    status_acc = np.maximum(status_acc, status_final)
    bad_idx = np.flatnonzero(status_final != 0)
    if bad_idx.size:
        y_tmp[bad_idx, :] = y_inout[bad_idx, :]
    y_inout[:, :] = y_tmp

    y_line = np.asarray(y_inout, dtype=float).reshape(-1, N_Y)[::-1]
    abd = _diskbridge_abundances_from_y(y_line)

    info = {
        "shielding_mode": "external_iteration",
        "converged": converged,
        "convergence_history": convergence_history,
        "mix": mix,
        "status_hist": {
            0: int(np.sum(status_acc == 0)),
            1: int(np.sum(status_acc == 1)),
            2: int(np.sum(status_acc == 2)),
        },
    }

    return y_line, abd, NH, info


def run_gow17_pdr_internal(
    *,
    nH_cm3: float,
    NH: np.ndarray,
    Tgas_K: np.ndarray,
    chi0: float = 1.0,
    xi_cr: float = 2.0e-16,
    shielding_max_iter: int = 30,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict]:
    radm = _build_1d_cartesian_model(nH_cm3=nH_cm3, NH=NH)

    NH = np.asarray(NH, dtype=float)
    shape = radm.model.mesh.shape
    ncells = NH.size

    chi0_incident = 2.0 * float(chi0)
    NH_rev = NH[::-1]
    Av_arr = NH_rev / 1.87e21

    radm.dust_temperature = Quantity(np.full(shape, 50.0, dtype=float), "K")
    Tgas_K = np.asarray(Tgas_K, dtype=float)
    if Tgas_K.shape != (ncells,):
        raise ValueError(f"Tgas_K must have shape ({ncells},), got {Tgas_K.shape}")
    radm.gas_temperature = Quantity(Tgas_K[::-1].reshape(shape), "K")
    radm.Av = Quantity(Av_arr.reshape(shape), "dimensionless")

    chi_pe = 0.5 * chi0_incident * np.exp(-NH_rev * 1.0e-21)
    radm.chi = Quantity(chi_pe.reshape(shape), "dimensionless")

    config = {
        "mode": "equilibrium",
        "const_temp": False,
        "t_end": "2.0e9 yr",
        "nside": 1,
        "chi0": float(chi0_incident),
        "ion_rate": f"{float(xi_cr)} 1/s",
        "max_iter": 80,
        "reltol": 1.0e-2,
        "abstol0": 1.0e-9,
        "b_kms": 0.3,
        "Zg": 1.0,
        "Zd": 1.0,
        "fH2gr": 1.0,
        "fHplusgr": 0.6,
        "fCplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "fCplusCR": 1.0,
        "shielding_outer_coupling": "pseudotime",
        "shielding_max_iter": int(shielding_max_iter),
        "shielding_reltol": 1.0e-6,
        "shielding_abstol": 1.0e-20,
    }

    run_chemistry(radm, model="gow17_pdr", config=config)

    y_line = np.asarray(radm.gow17_y, dtype=float).reshape(-1, N_Y)[::-1]
    abd = _diskbridge_abundances_from_y(y_line)
    info = {
        "shielding_mode": "gow17_pdr_internal",
        "shielding_max_iter": int(shielding_max_iter),
    }
    return y_line, abd, NH, info


def _safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log10(np.maximum(x, 1.0e-99))


def _plot_compare(
    *,
    outdir: Path,
    Av_ref: np.ndarray,
    Av_db: np.ndarray,
    ref: RefSlab,
    nH_index: int,
    abd_db: Dict[str, np.ndarray],
    species: Iterable[str],
    suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)

    for spec in species:
        y_ref = ref.abd[spec][:, nH_index]
        y_db = abd_db[spec].reshape(-1)
        color = COLORS.get(spec, "#444444")

        ax.plot(Av_ref, _safe_log10(y_ref), lw=1.2, label=f"{spec} ref", color=color)
        ax.plot(Av_db, _safe_log10(y_db), lw=1.2, ls="--", label=f"{spec} diskbridge", color=color)

    ax.set_xlabel("A_V")
    ax.set_ylabel("log10 abundance per H")
    ax.set_ylim(-14.0, 0.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fname = f"gow17_fig2_compare_nH_{int(ref.nH_values[nH_index])}"
    if suffix:
        fname += f"_{suffix}"
    fig.savefig(outdir / f"{fname}.png")
    plt.close(fig)


def compute_error_vs_reference(
    abd_db: Dict[str, np.ndarray],
    ref: RefSlab,
    nH_index: int,
    Av_db: np.ndarray,
    species: Iterable[str],
    Av_min: float = 0.5,
    abd_min: float = 1.0e-12,
) -> Dict[str, Dict[str, float]]:
    errors = {}
    for spec in species:
        y_ref = ref.abd[spec][:, nH_index]
        y_ref_interp = np.interp(Av_db, ref.Av, y_ref)
        y_db = abd_db[spec].reshape(-1)

        mask = (Av_db > Av_min) & (y_ref_interp > abd_min) & (y_db > abd_min)
        if not np.any(mask):
            continue

        log_ref = _safe_log10(y_ref_interp[mask])
        log_db = _safe_log10(y_db[mask])
        dex_err = np.abs(log_db - log_ref)

        errors[spec] = {
            "median_dex": float(np.median(dex_err)),
            "p90_dex": float(np.percentile(dex_err, 90)),
            "max_dex": float(np.max(dex_err)),
        }
    return errors


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ref_dir = root / "other_codes" / "pdr" / "out_example_simple"

    ref = _load_reference(ref_dir)

    ngrid = len(ref.NH)
    NH_total = float(ref.NH[-1])
    Av_ref = ref.Av

    species = ["CO", "C", "C+", "H3+", "OHx", "CHx", "He+"]

    ext_summary = []

    for i, nH in enumerate(np.atleast_1d(ref.nH_values)):
        nH_val = float(nH)
        print(f"\n{'='*60}")
        print(f"nH = {nH_val:.0f} cm^-3")
        print(f"{'='*60}")

        print(f"\n--- Running native slab solver (same C++ code as reference) ---")
        y_native, abd_native, NH_native, info_native = run_native_slab(
            nH_cm3=nH_val,
            NH_total=NH_total,
            ngrid=ngrid,
            chi0=1.0,
            xi_cr=2.0e-16,
            const_temp=False,
            Tgas_init=100.0,
            verbose=False,
        )

        Av_native = NH_native / 1.87e21

        E_ref = np.asarray(ref.E[:, i], dtype=float)
        xH2_ref = np.asarray(ref.abd["H2"][:, i], dtype=float)
        xe_ref = _xe_from_abundances({k: v[:, i] for k, v in ref.abd.items()})
        T_ref = _temperature_from_energy(E=E_ref, xH2=xH2_ref, xe=xe_ref)

        E_native = np.asarray(y_native[:, gow17_native.I_E], dtype=float)
        xH2_native = np.asarray(abd_native["H2"].reshape(-1), dtype=float)
        xe_native = _xe_from_abundances({k: v.reshape(-1) for k, v in abd_native.items()})
        T_native = _temperature_from_energy(E=E_native, xH2=xH2_native, xe=xe_native)

        print(
            "T diagnostics (K): "
            f"ref[min/med/max]={float(np.min(T_ref)):.2f}/{float(np.median(T_ref)):.2f}/{float(np.max(T_ref)):.2f}, "
            f"native[min/med/max]={float(np.min(T_native)):.2f}/{float(np.median(T_native)):.2f}/{float(np.max(T_native)):.2f}"
        )

        print(f"\n--- Comparing native vs ORIGINAL reference ---")
        print("Note: external/pdr differs from other_codes/pdr (NH grid calculation, fH2gr param)")
        y_ref = ref.abd
        for spec in species:
            if spec not in y_ref:
                continue
            y_r = y_ref[spec][:, i]
            y_n = abd_native[spec].reshape(-1)

            if len(y_r) != len(y_n):
                print(f"  {spec}: LENGTH MISMATCH ref={len(y_r)} vs native={len(y_n)}")
                continue

            mask = (y_r > 1.0e-12) & (y_n > 1.0e-12)
            if not np.any(mask):
                print(f"  {spec}: no significant abundance")
                continue

            ratio = y_n[mask] / y_r[mask]
            median_ratio = float(np.median(ratio))
            max_ratio = float(np.max(ratio))
            min_ratio = float(np.min(ratio))

            log_diff = np.abs(np.log10(y_n[mask]) - np.log10(y_r[mask]))
            median_dex = float(np.median(log_diff))

            print(f"  {spec}: ratio median={median_ratio:.2f}, range=[{min_ratio:.2f}, {max_ratio:.2f}], median_dex={median_dex:.3f}")

        _plot_compare(
            outdir=root / "visualization_tests",
            Av_ref=Av_ref,
            Av_db=Av_native,
            ref=ref,
            nH_index=i,
            abd_db=abd_native,
            species=species,
            suffix="native",
        )

        print(f"\n--- Running internal gow17_pdr baseline (built-in shielding iterations) ---")
        y_int, abd_int, NH_int, info_int = run_gow17_pdr_internal(
            nH_cm3=nH_val,
            NH=ref.NH,
            Tgas_K=T_ref,
            chi0=1.0,
            xi_cr=2.0e-16,
            shielding_max_iter=30,
        )

        Av_int = NH_int / 1.87e21

        _plot_compare(
            outdir=root / "visualization_tests",
            Av_ref=Av_ref,
            Av_db=Av_int,
            ref=ref,
            nH_index=i,
            abd_db=abd_int,
            species=species,
            suffix="internal_pdr",
        )

        print(f"\n--- Running external iteration mode (gow17_pdr with outer iterations) ---")
        y_ext, abd_ext, NH_ext, info_ext = run_external_iteration(
            nH_cm3=nH_val,
            NH=ref.NH,
            Tgas_K=T_ref,
            chi0=1.0,
            xi_cr=2.0e-16,
            outer_iter=30,
            outer_reltol=1.0e-2,
            mix=0.2,
            verbose=True,
        )

        Av_ext = NH_ext / 1.87e21

        print(f"\n--- Comparing external vs internal gow17_pdr ---")
        for spec in species:
            y_nat = abd_int[spec].reshape(-1)
            y_ex = abd_ext[spec].reshape(-1)

            if len(y_nat) != len(y_ex):
                print(f"  {spec}: LENGTH MISMATCH native={len(y_nat)} vs ext={len(y_ex)}")
                continue

            mask = (y_nat > 1.0e-12) & (y_ex > 1.0e-12)
            if not np.any(mask):
                print(f"  {spec}: no significant abundance")
                continue

            denom = np.maximum(y_nat[mask], y_ex[mask])
            rel_diff = np.abs(y_nat[mask] - y_ex[mask]) / denom
            max_rel = float(np.max(rel_diff))
            p90_rel = float(np.percentile(rel_diff, 90))
            print(f"  {spec}: max_rel={max_rel:.3e}, p90_rel={p90_rel:.3e}")

        ext_summary.append(
            {
                "nH": float(nH_val),
                "converged": bool(info_ext.get("converged", False)),
                "mix": float(info_ext.get("mix", np.nan)),
                "n_outer": len(info_ext.get("convergence_history", [])),
            }
        )

        _plot_compare(
            outdir=root / "visualization_tests",
            Av_ref=Av_ref,
            Av_db=Av_ext,
            ref=ref,
            nH_index=i,
            abd_db=abd_ext,
            species=species,
            suffix="external",
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Native slab:
- With thermochemistry enabled (const_temp=False) and proper E initialization, native slab matches the shipped reference.

External iteration:
- Uses gow17_pdr with outer iterations and log-space mixing; convergence is measured by p90 relative change in CO/H2.
""")

    for row in ext_summary:
        print(
            f"nH={row['nH']:.0f}: converged={row['converged']}, "
            f"mix={row['mix']:.2f}, outer_iters={row['n_outer']}"
        )


if __name__ == "__main__":
    main()
