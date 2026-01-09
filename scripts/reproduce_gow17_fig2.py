from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

import diskbridge
import diskbridge._gow17 as gow17_native
from diskbridge._units import Quantity
from diskbridge.chemistry.api import run_chemistry
from diskbridge.chemistry.models._gow17_network import (
    I_CHX,
    I_CO,
    I_CP,
    I_H2,
    I_H2P,
    I_H3P,
    I_HCOP,
    I_HEP,
    I_HP,
    I_OHX,
    I_OP,
    I_SIP,
    I_SP,
    N_Y,
)
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


def _load_reference(dir_out: Path) -> RefSlab:
    nH_values = np.loadtxt(dir_out / "nH_arr.dat")
    NH = np.loadtxt(dir_out / "colH_arr.dat")
    Av = NH / 1.87e21

    abd: Dict[str, np.ndarray] = {}
    for islab, _nH in enumerate(np.atleast_1d(nH_values)):
        slab = np.loadtxt(dir_out / f"slab{islab:06d}.dat")
        for spec, j in _SPEC_INDEX_REF.items():
            if spec == "E":
                continue
            abd.setdefault(spec, []).append(slab[:, j])

    for spec in list(abd):
        abd[spec] = np.stack(abd[spec], axis=1)

    xC = 1.6e-4
    abd["C"] = xC - abd["CHx"] - abd["CO"] - abd["C+"] - abd["HCO+"]

    return RefSlab(NH=NH, Av=Av, nH_values=np.asarray(nH_values, dtype=float), abd=abd)


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
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict]:
    dim = gow17_native.N_Y

    y0 = np.zeros(dim, dtype=np.float64)
    y0[gow17_native.I_HEP] = 1.450654e-08
    y0[gow17_native.I_H3P] = 2.681411e-07
    y0[gow17_native.I_CP] = 1.0e-4
    y0[gow17_native.I_CO] = 1.0e-7
    y0[gow17_native.I_H2] = 0.1

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
        const_temp=True,
        Tgas=100.0,
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
    chi0: float = 1.0,
    xi_cr: float = 2.0e-16,
    outer_iter: int = 20,
    outer_reltol: float = 1.0e-4,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict]:
    radm = _build_1d_cartesian_model(nH_cm3=nH_cm3, NH=NH)

    NH = np.asarray(NH, dtype=float)
    shape = radm.model.mesh.shape
    ncells = NH.size

    chi0_incident = 2.0 * float(chi0)
    NH_rev = NH[::-1]
    Av_arr = NH_rev / 1.87e21

    radm.dust_temperature = Quantity(np.full(shape, 10.0, dtype=float), "K")
    radm.gas_temperature = Quantity(np.full(shape, 100.0, dtype=float), "K")
    radm.Av = Quantity(Av_arr.reshape(shape), "dimensionless")

    chi_pe = 0.5 * chi0_incident * np.exp(-NH_rev * 1.0e-21)
    radm.chi = Quantity(chi_pe.reshape(shape), "dimensionless")

    config = {
        "mode": "equilibrium",
        "const_temp": True,
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
        "shielding_max_iter": 30,
        "shielding_reltol": 1.0e-6,
        "shielding_abstol": 1.0e-20,
    }

    convergence_history = []
    y_prev = None
    converged = False

    for it in range(int(outer_iter)):
        res = run_chemistry(radm, model="gow17_pdr", config=config)

        y_out = np.asarray(radm.gow17_y, dtype=float).reshape(-1, N_Y)

        if y_prev is not None:
            xco_new = y_out[:, I_CO]
            xco_old = y_prev[:, I_CO]
            xh2_new = y_out[:, I_H2]
            xh2_old = y_prev[:, I_H2]

            mask_co = (xco_new > 1.0e-10) & (xco_old > 1.0e-10)
            mask_h2 = (xh2_new > 1.0e-10) & (xh2_old > 1.0e-10)

            if np.any(mask_co):
                denom_co = np.maximum(xco_new[mask_co], xco_old[mask_co])
                rel_co = np.abs(xco_new[mask_co] - xco_old[mask_co]) / denom_co
                max_rel_co = float(np.max(rel_co))
            else:
                max_rel_co = 0.0

            if np.any(mask_h2):
                denom_h2 = np.maximum(xh2_new[mask_h2], xh2_old[mask_h2])
                rel_h2 = np.abs(xh2_new[mask_h2] - xh2_old[mask_h2]) / denom_h2
                max_rel_h2 = float(np.max(rel_h2))
            else:
                max_rel_h2 = 0.0

            convergence_history.append({
                "iter": it + 1,
                "max_rel_co": max_rel_co,
                "max_rel_h2": max_rel_h2,
            })

            if verbose:
                print(
                    f"  iter={it+1}/{int(outer_iter)}: "
                    f"max_rel_CO={max_rel_co:.3e}, "
                    f"max_rel_H2={max_rel_h2:.3e}"
                )

            if max(max_rel_co, max_rel_h2) <= float(outer_reltol):
                converged = True
                break

        y_prev = y_out.copy()

    if not converged and verbose:
        print(f"  Warning: did not converge after {outer_iter} iterations")

    y_line = np.asarray(radm.gow17_y, dtype=float).reshape(-1, N_Y)[::-1]
    abd = _diskbridge_abundances_from_y(y_line)

    info = {
        "shielding_mode": "external_iteration",
        "converged": converged,
        "convergence_history": convergence_history,
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
            verbose=False,
        )

        Av_native = NH_native / 1.87e21

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

    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)
    print("""
Known code differences between external/pdr (DiskBridge) and other_codes/pdr (reference):
1. NH grid: external/pdr uses (ngrid-1) divisor, other_codes/pdr uses ngrid
2. fH2gr parameter: external/pdr has configurable fH2gr, other_codes/pdr does not
3. Tdust: external/pdr configurable, other_codes/pdr hardcoded to 10K

Observations:
- H2 abundance matches reference almost exactly (ratio ~1.0)
- CO abundance is ~3x higher than reference (cause unknown - codes nearly identical)
- OHx, CHx also show significant differences

External iteration (gow17_pdr) limitations:
- Global iteration over all cells simultaneously oscillates at C+/C/CO transition (Av 0.1-0.6)
- This is inherent to global iteration vs sequential marching approach
- Native slab solver (sequential marching) is more stable for 1D problems
""")


if __name__ == "__main__":
    main()
