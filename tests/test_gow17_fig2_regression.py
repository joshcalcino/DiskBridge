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
import diskbridge._gow17 as _gow17

I_HEP = _gow17.I_HEP
I_OHX = _gow17.I_OHX
I_CHX = _gow17.I_CHX
I_CO = _gow17.I_CO
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
I_CO_ICE = _gow17.I_CO_ICE
N_Y = _gow17.N_Y
from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel
from diskbridge._units import Quantity
from diskbridge.chemistry.shielding.columns_1d import is_effectively_1d


REF_DIR = Path(__file__).parent.parent / "other_codes" / "pdr" / "out_example_simple"
SPEC_LIST_REF = ["He+", "OHx", "CHx", "CO", "C+", "HCO+", "H2", "H+", "H3+", "H2+", "S+", "Si+", "O+", "E"]
IDX_REF = {n: i for i, n in enumerate(SPEC_LIST_REF)}


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

    # Build x edges such that x_center[i] == NH[i] / nH exactly.
    # This makes the 1D column integrator reproduce the reference NH profile.
    x_centers_cm = NH / float(nH_cm3)
    x_edges_cm = np.empty(int(x_centers_cm.size) + 1, dtype=float)
    x_edges_cm[0] = 0.0
    for i in range(int(x_centers_cm.size)):
        x_edges_cm[i + 1] = 2.0 * float(x_centers_cm[i]) - float(x_edges_cm[i])
    if not np.all(np.diff(x_edges_cm) > 0.0):
        raise ValueError("NH profile does not produce a strictly increasing x_edges grid")

    mesh = Mesh.cartesian(
        x=Axis(edges=Quantity(x_edges_cm, "cm")),
        y=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
        z=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
    )
    shape = mesh.shape
    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    rho = (Quantity(np.full(shape, float(nH_cm3)), "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=("x", "y", "z")))

    sigma = Quantity(np.full(shape, 1e-21), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")),
    )

    radm = RadModel(model)

    radm.nH = Quantity(np.full(shape, float(nH_cm3)), "cm^-3")
    radm.gas_temperature = Quantity(np.full(shape, 100.0), "K")
    radm.dust_temperature = Quantity(np.full(shape, 20.0), "K")

    # Incident UV field (G0) for a one-sided slab. The reference PDR code uses
    # a (G0/2) factor internally plus explicit dust attenuation in Av/NH.
    radm.chi = Quantity(np.full(shape, float(chi0)), "dimensionless")

    # Av profile used by the reference PDR code: Av = NH * Zd / 1.87e21.
    # Here we adopt Zd=1, matching the regression reference.
    NH_arr = np.asarray(NH, float)
    Av_arr = NH_arr / 1.87e21
    radm.Av = Quantity(Av_arr.reshape(shape), "dimensionless")

    Av_db = Quantity(Av_arr, "dimensionless")
    return radm, Av_db


def run_gow17_slab(radm: RadModel, nH: float, *, chi0: float, n_iter: int):
    """Run GOW17 chemistry with shielding iterations."""
    radm.nco_gas = Quantity(np.full(radm.model.mesh.shape, 1e-12 * nH), "cm^-3")

    config = {
        "mode": "equilibrium",
        "t_end": "2.0e9 yr",
        "b_kms": 0.3,
        "chi0": chi0,
        "chi_is_incident": True,
        "ion_rate": "2e-16 1/s",
        "shielding_outer_1d": "min",
        "reltol": 1.0e-2,
        "abstol0": 1.0e-9,
        "mxsteps": 5000000,
        "maxord": 3,
        "tolfac": 10.0,
        "tmin": "1.0e5 yr",
        "const_temp": False,
        "isfsH2": True,
        "isfsCO": True,
        "isfsC": True,
        "fHplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fCplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "max_iter": 80,
    }

    res = None
    for _ in range(n_iter):
        res = run_chemistry(radm, model="gow17", config=config, write=False)
        radm.nco_gas = res.number_densities["co"]

    return res


@pytest.fixture(scope="module")
def reference_nH100():
    """Load reference data for nH=100 cm^-3."""
    return load_reference(nH_index=0)


@pytest.fixture(scope="module")
def diskbridge_nH100(reference_nH100):
    """Run DiskBridge for nH=100 cm^-3."""
    Av_ref, _ = reference_nH100
    NH_full = Av_ref * 1.87e21

    NH = np.asarray(NH_full, dtype=float)

    nH = 100.0
    radm, Av_db = build_model(nH, NH, chi0=2.0)
    assert is_effectively_1d(radm.model.mesh, radm.ensure_nH().shape)
    res = run_gow17_slab(radm, nH, chi0=2.0, n_iter=4)

    Y = np.asarray(radm.gow17_y).reshape(-1, N_Y)
    return Av_db, Y, res


@pytest.mark.skipif(not REF_DIR.exists(), reason="Reference data not available")
class TestGOW17Fig2Regression:
    """Regression tests for GOW17 Figure 2 reproduction."""

    def test_converges(self, diskbridge_nH100) -> None:
        """All cells should converge (no failed CVODE solves)."""
        _Av_db, _Y, res = diskbridge_nH100
        assert int(res.meta.get("n_fail", -1)) == 0
        assert int(res.meta.get("max_status", 1)) == 0


    @pytest.mark.parametrize(
        "species,idx_db",
        [
            ("CO", I_CO),
            ("C+", I_CP),
            ("H2", I_H2),
            ("He+", I_HEP),
            ("OHx", I_OHX),
            ("CHx", I_CHX),
            ("HCO+", I_HCOP),
        ],
    )
    def test_species_match_reference_rtol_1e5(self, species, idx_db, diskbridge_nH100, reference_nH100) -> None:
        """Match reference slab data within 1e-5 relative error where reference is nonzero."""
        _Av_ref, slab_ref = reference_nH100
        Av_db, Y, _res = diskbridge_nH100

        ref_vals = np.asarray(slab_ref[:, IDX_REF[species]], dtype=float)
        db_vals = np.asarray(Y[:, idx_db], dtype=float)

        assert ref_vals.shape == db_vals.shape
        assert np.all(np.isfinite(ref_vals))
        assert np.all(np.isfinite(db_vals))

        nonzero = ref_vals != 0.0
        if np.any(nonzero):
            rel = np.abs(db_vals[nonzero] - ref_vals[nonzero]) / np.abs(ref_vals[nonzero])
            assert float(np.max(rel)) <= 1e-5

        if np.any(~nonzero):
            assert np.all(db_vals[~nonzero] == 0.0)

    def test_carbon_budget_conservation(self, diskbridge_nH100):
        """Carbon budget: sum of solved C-species <= X_C_TOT (remainder is neutral C ghost)."""
        _Av_db, Y, _res = diskbridge_nH100
        X_C_TOT = 1.6e-4

        C_solved = Y[:, I_CO] + Y[:, I_CP] + Y[:, I_CHX] + Y[:, I_HCOP] + Y[:, I_CO_ICE]
        C_neutral_ghost = X_C_TOT - C_solved

        assert np.all(C_solved <= X_C_TOT * 1.01), "Solved carbon exceeds total budget"
        assert np.all(C_neutral_ghost >= -1e-10), "Negative neutral C (carbon budget violated)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
