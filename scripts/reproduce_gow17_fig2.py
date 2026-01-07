from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

import diskbridge
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


def run_diskbridge_slab(
    *,
    nH_cm3: float,
    NH: np.ndarray,
    chi0: float = 2.0,
    xi_cr: float = 2.0e-16,
    outer_iter: int = 6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], RadModel]:
    radm = _build_1d_cartesian_model(nH_cm3=nH_cm3, NH=NH)
    shape = radm.model.mesh.shape

    # Reference code uses: GPE = (G0/2) * exp(-NH*sigmaPE) with sigmaPE=1e-21,
    # and passes G0*2 to represent one-sided illumination.
    # chi is the dust-attenuated PE field (GPE in reference). For beamed geometry:
    # GPE = (G0/2) * exp(-NH * sigmaPE). The reference sets G0*2 for one-sided slab,
    # so chi0 here should be 2.0 to make (chi0/2)=1 at the boundary.
    NH_rev = np.asarray(NH, dtype=float)[::-1]
    chi_pe = 0.5 * chi0 * np.exp(-NH_rev * 1.0e-21)
    radm.chi = Quantity(chi_pe.reshape(shape), "dimensionless")

    Av = NH_rev / 1.87e21
    radm.Av = Quantity(Av.reshape(shape), "dimensionless")

    radm.dust_temperature = Quantity(np.full(shape, 100.0, dtype=float), "K")

    # seed for shielding iteration
    radm.nco_gas = Quantity(np.full(shape, 1.0e-12 * float(nH_cm3), dtype=float), "cm^-3")

    res = None
    for _ in range(int(outer_iter)):
        res = run_chemistry(
            radm,
            model="gow17_pdr",
            config={
                "mode": "equilibrium",
                "nside": 1,
                "b_kms": 0.3,
                "chi0": float(chi0),
                "ion_rate": f"{xi_cr} 1/s",
                "Zg": 1.0,
                "Zd": 1.0,
                "fHplusgr": 0.6,
                "fHeplusgr": 0.6,
                "fCplusgr": 0.6,
                "fSplusgr": 0.6,
                "fSiplusgr": 0.6,
                "max_iter": 80,
            },
        )
        radm.nco_gas = res.number_densities["co"]

    assert res is not None
    y = np.asarray(radm.gow17_y, dtype=float)
    abd = _diskbridge_abundances_from_y(y)

    return y, abd, radm


def _safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log10(np.maximum(x, 1.0e-99))


def _plot_compare(
    *,
    outdir: Path,
    Av: np.ndarray,
    ref: RefSlab,
    nH_index: int,
    abd_db: Dict[str, np.ndarray],
    species: Iterable[str],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=160)

    for spec in species:
        y_ref = ref.abd[spec][:, nH_index][::-1]
        y_db = abd_db[spec].reshape(-1)

        ax.plot(Av, _safe_log10(y_ref), lw=1.2, label=f"{spec} ref")
        ax.plot(Av, _safe_log10(y_db), lw=1.2, ls="--", label=f"{spec} diskbridge")

    ax.set_xlabel("A_V")
    ax.set_ylabel("log10 abundance per H")
    ax.set_ylim(-14.0, 0.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"gow17_fig2_compare_nH_{int(ref.nH_values[nH_index])}.png")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ref_dir = root / "other_codes" / "pdr" / "out_example_simple"

    ref = _load_reference(ref_dir)

    # Use the same NH grid as the reference output
    NH = ref.NH
    Av = ref.Av[::-1]

    species = ["CO", "C", "C+", "H3+", "OHx", "CHx", "He+"]

    for i, nH in enumerate(np.atleast_1d(ref.nH_values)):
        _y, abd, _radm = run_diskbridge_slab(nH_cm3=float(nH), NH=NH)
        _plot_compare(outdir=root / "visualization_tests", Av=Av, ref=ref, nH_index=i, abd_db=abd, species=species)


if __name__ == "__main__":
    main()
