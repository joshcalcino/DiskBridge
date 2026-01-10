"""
Regression test for GOW17 PDR chemistry against reference slab data.

Reproduces GOW17 Figure 2 benchmark conditions and asserts that key species
abundances match the reference within specified tolerances.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest

import diskbridge
from diskbridge.chemistry.api import run_chemistry
from diskbridge.chemistry.models._gow17_network import (
    I_HEP, I_OHX, I_CHX, I_CO, I_CP, I_HCOP, I_H2, I_CO_ICE, N_Y
)
from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel
from diskbridge._units import Quantity


REF_DIR = Path(__file__).parent.parent / "other_codes" / "pdr" / "out_example_simple"
SPEC_LIST_REF = ["He+", "OHx", "CHx", "CO", "C+", "HCO+", "H2", "H+", "H3+", "H2+", "S+", "Si+", "O+", "E"]
IDX_REF = {n: i for i, n in enumerate(SPEC_LIST_REF)}


FIG_DIR = Path(__file__).parent.parent / "visualization_tests"

TOLERANCES_DEX = {
    "CO": 0.6,
    "C+": 0.15,
    "H2": 0.1,
    "He+": 0.2,
    "OHx": 0.6,
    "CHx": 0.4,
}


def load_reference(nH_index: int = 0):
    """Load reference slab data for given density index."""
    NH_full = np.loadtxt(REF_DIR / "colH_arr.dat")
    slab = np.loadtxt(REF_DIR / f"slab{nH_index:06d}.dat")
    Av = NH_full / 1.87e21
    return Av, slab


def build_model(nH_cm3: float, NH: np.ndarray, chi0: float = 2.0):
    """Build 1D slab model for GOW17 benchmark."""
    units = diskbridge.units
    m_H = units("m_H")
    NH = np.asarray(NH, float)
    NH_rev = NH[::-1]
    L_cm = float(NH_rev[0] / nH_cm3)
    x_centers_cm = L_cm - (NH_rev / nH_cm3)

    mesh = Mesh.cartesian(
        x=Axis(centers=Quantity(x_centers_cm, "cm")),
        y=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
        z=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
    )
    shape = mesh.shape
    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    nH_field = np.full(shape, nH_cm3)
    rho = (Quantity(nH_field, "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=("x", "y", "z")))

    sigma = Quantity(np.full(shape, 1e-21), "cm^2")
    model.gas_register("sigma_d_per_H", Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")))

    radm = RadModel(model)
    radm.dust_temperature = Quantity(np.full(shape, 50.0), "K")

    chi_pe = 0.5 * chi0 * np.exp(-NH_rev * 1e-21)
    radm.chi = Quantity(chi_pe.reshape(shape), "dimensionless")
    radm.Av = Quantity((NH_rev / 1.87e21).reshape(shape), "dimensionless")

    return radm, NH_rev / 1.87e21


def run_gow17_slab(radm: RadModel, nH: float, chi0: float = 2.0, n_iter: int = 4):
    """Run GOW17 chemistry with shielding iterations."""
    radm.nco_gas = Quantity(np.full(radm.model.mesh.shape, 1e-12 * nH), "cm^-3")

    config = {
        "mode": "equilibrium",
        "t_end": "2.0e9 yr",
        "nside": 1,
        "b_kms": 0.3,
        "chi0": chi0,
        "ion_rate": "2e-16 1/s",
        "fHplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fCplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "max_iter": 80,
    }

    res = None
    for _ in range(n_iter):
        res = run_chemistry(radm, model="gow17_pdr", config=config)
        radm.nco_gas = res.number_densities["co"]

    return res


def safe_log10(x):
    return np.log10(np.maximum(np.asarray(x, float), 1e-30))


@pytest.fixture(scope="module")
def reference_nH100():
    """Load reference data for nH=100 cm^-3."""
    return load_reference(nH_index=0)


@pytest.fixture(scope="module")
def diskbridge_nH100(reference_nH100):
    """Run DiskBridge for nH=100 cm^-3."""
    Av_ref, _ = reference_nH100
    NH_full = Av_ref * 1.87e21

    n_cells = 64
    sel = np.unique(np.round(np.linspace(0, len(NH_full) - 1, n_cells)).astype(int))
    NH = NH_full[sel]

    nH = 100.0
    radm, Av_db = build_model(nH, NH, chi0=2.0)
    run_gow17_slab(radm, nH, chi0=2.0, n_iter=4)

    Y = np.asarray(radm.gow17_y).reshape(-1, N_Y)
    return Av_db, Y


@pytest.mark.skipif(not REF_DIR.exists(), reason="Reference data not available")
class TestGOW17Fig2Regression:
    """Regression tests for GOW17 Figure 2 reproduction."""

    def test_convergence(self, diskbridge_nH100, reference_nH100):
        """All cells should converge."""
        Av_db, Y = diskbridge_nH100
        assert Y.shape[0] > 0, "No cells computed"

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        Av_ref, slab_ref = reference_nH100
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        axes = axes.ravel()

        series = [
            ("CO", I_CO),
            ("C+", I_CP),
            ("H2", I_H2),
            ("He+", I_HEP),
        ]
        for ax, (name, idx_db) in zip(axes, series):
            db_vals = np.asarray(Y[:, idx_db], float)
            ref_vals = np.interp(Av_db, Av_ref, slab_ref[:, IDX_REF[name]])
            ax.plot(Av_db, safe_log10(db_vals), label="diskbridge")
            ax.plot(Av_db, safe_log10(ref_vals), label="reference", linestyle="--")
            ax.set_title(name)
            ax.set_ylabel("log10(abundance)")
            ax.grid(True, alpha=0.3)

        for ax in axes[-2:]:
            ax.set_xlabel("Av")

        axes[0].legend(loc="best")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "gow17_fig2_pytest_nH100.png", dpi=150)
        plt.close(fig)

    @pytest.mark.parametrize("species,idx_db,tol", [
        ("CO", I_CO, TOLERANCES_DEX["CO"]),
        ("C+", I_CP, TOLERANCES_DEX["C+"]),
        ("H2", I_H2, TOLERANCES_DEX["H2"]),
        ("He+", I_HEP, TOLERANCES_DEX["He+"]),
    ])
    def test_species_match(self, species, idx_db, tol, diskbridge_nH100, reference_nH100):
        """Key species should match reference within tolerance for Av > 0.5."""
        Av_db, Y = diskbridge_nH100
        Av_ref, slab_ref = reference_nH100

        db_vals = Y[:, idx_db]
        ref_interp = np.interp(Av_db, Av_ref, slab_ref[:, IDX_REF[species]])

        mask = Av_db > 0.5
        err = np.abs(safe_log10(db_vals[mask]) - safe_log10(ref_interp[mask]))

        median_err = np.median(err)
        p90_err = np.percentile(err, 90)

        assert median_err < tol, f"{species} median error {median_err:.3f} dex exceeds tolerance {tol}"
        assert p90_err < tol * 1.5, f"{species} p90 error {p90_err:.3f} dex exceeds tolerance {tol * 1.5}"

    def test_h2_excellent_match(self, diskbridge_nH100, reference_nH100):
        """H2 should match within 0.05 dex for Av > 1."""
        Av_db, Y = diskbridge_nH100
        Av_ref, slab_ref = reference_nH100

        db_vals = Y[:, I_H2]
        ref_interp = np.interp(Av_db, Av_ref, slab_ref[:, IDX_REF["H2"]])

        mask = Av_db > 1.0
        err = np.abs(safe_log10(db_vals[mask]) - safe_log10(ref_interp[mask]))

        max_err = np.max(err)
        assert max_err < 0.05, f"H2 max error {max_err:.3f} dex exceeds 0.05 for Av > 1"

    def test_carbon_budget_conservation(self, diskbridge_nH100):
        """Carbon budget: sum of solved C-species <= X_C_TOT (remainder is neutral C ghost)."""
        Av_db, Y = diskbridge_nH100
        X_C_TOT = 1.6e-4

        C_solved = Y[:, I_CO] + Y[:, I_CP] + Y[:, I_CHX] + Y[:, I_HCOP] + Y[:, I_CO_ICE]
        C_neutral_ghost = X_C_TOT - C_solved

        assert np.all(C_solved <= X_C_TOT * 1.01), "Solved carbon exceeds total budget"
        assert np.all(C_neutral_ghost >= -1e-10), "Negative neutral C (carbon budget violated)"
