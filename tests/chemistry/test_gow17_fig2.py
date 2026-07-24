"""Check DiskBridge GOW17 chemistry against the original PDR slab output.

The machine-readable reference is copied from
``other_codes/pdr/out_example_simple``. The full two-density comparison and
diagnostic plots live in ``validation/gow17_fig2``.
"""
# db-keywords: shielding, uv-products, photodesorption, gow17, gas-temperature, validation, config, units, radmc3d, chemistry, model, mesh
# db-role: validation
# db-scope: test
# db-purpose: Regression test for GOW17 PDR chemistry against reference slab data.

from __future__ import annotations

import pytest

import numpy as np
from pathlib import Path

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


REF_DIR = Path(__file__).resolve().parents[1] / "reference" / "gow17_fig2_external_pdr"
SPEC_LIST_REF = ["He+", "OHx", "CHx", "CO", "C+", "HCO+", "H2", "H+", "H3+", "H2+", "S+", "Si+", "O+", "E"]
IDX_REF = {n: i for i, n in enumerate(SPEC_LIST_REF)}
EXTERNAL_PDR_RTOL = 1.0e-2


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

    x_edges_cm = np.empty(int(NH.size) + 1, dtype=float)
    x_edges_cm[0] = 0.0
    dNH = np.diff(NH)
    if not np.all(dNH > 0.0):
        raise ValueError("NH must be strictly increasing")
    dx = dNH / float(nH_cm3)
    x_edges_cm[1:-1] = np.cumsum(dx)
    x_edges_cm[-1] = x_edges_cm[-2] + float(dx[-1])

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
    model.gas_register(
        "microturbulence",
        Field(
            quantity="microturbulence",
            data=Quantity(np.full(shape, 0.3), "km/s"),
            axis_order=("x", "y", "z"),
            attrs={"spatially_constant": True},
        ),
    )

    radm = RadModel(model)

    radm.nH = Quantity(np.full(shape, float(nH_cm3)), "cm^-3")
    radm.gas_temperature = Quantity(np.full(shape, 100.0), "K")
    radm.dust_temperature = Quantity(np.full(shape, 10.0), "K")

    # Av profile used by the reference PDR code: Av = NH * Zd / 1.87e21.
    # Here we adopt Zd=1, matching the regression reference.
    NH_arr = np.asarray(NH, float)
    Av_arr = NH_arr / 1.87e21
    radm.set_incident_uv(
        chi=Quantity(np.full(shape, float(chi0)), "dimensionless"),
        Av=Quantity(Av_arr.reshape(shape), "dimensionless"),
    )

    Av_db = Quantity(Av_arr, "dimensionless")
    return radm, Av_db


def run_gow17_slab_equilibrium(
    radm: RadModel,
    nH: float,
    *,
    chi0: float,
    n_iter: int,
    enable_co_phase: bool = False,
    reltol: float = 1.0e-2,
    tmin_s: float = 3.16e12,
    tmax_s: float = 6.32e16,
):
    """Run GOW17 chemistry with shielding iterations."""
    radm.nco_gas = Quantity(np.full(radm.model.mesh.shape, 1e-12 * nH), "cm^-3")

    config = {
        "mode": "equilibrium",
        "temperature": {"mode": "computed", "initial": "gas_temperature"},
        "tmax": f"{float(tmax_s)} s",
        "b_kms": 3.0,
        "chi0": chi0,
        "NH_total": "1.0e22 cm^-2",
        "NH_min": "1.0e17 cm^-2",
        "logNH": True,
        "field_geo": 0,
        "isdust": True,
        "NCOeff_global": True,
        "bCO_L": True,
        "ion_rate": "2e-16 1/s",
        "gradv": 9.0e-14,
        "Leff_CO_max": 3.0e20,
        "reltol": float(reltol),
        "abstol0": 1.0e-9,
        "mxsteps": 5000000,
        "maxord": 3,
        "tolfac": 10.0,
        "tmin": f"{float(tmin_s)} s",
        "isfsH2": True,
        "isfsCO": True,
        "isfsC": True,
        "enable_co_phase": bool(enable_co_phase),
        "fHplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fCplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "tolerances": {
            "abstol_default": 1.0e-9,
            "abstol_Heplus": 1.0e-15,
            "abstol_OHx": 1.0e-15,
            "abstol_CHx": 1.0e-15,
            "abstol_CO": 1.0e-15,
            "abstol_CO_ice": 1.0e-9,
            "abstol_Cplus": 1.0e-15,
            "abstol_HCOplus": 1.0e-30,
            "abstol_H2": 1.0e-8,
            "abstol_Hplus": 1.0e-15,
            "abstol_H3plus": 1.0e-15,
            "abstol_H2plus": 1.0e-15,
            "abstol_Splus": 1.0e-9,
            "abstol_Siplus": 1.0e-9,
            "abstol_Oplus": 1.0e-9,
        },
        "shielding_max_iter": 80,
    }
    if enable_co_phase:
        config["co_phase"] = {
            "S_CO": 1.0,
            "Y_CO": 0.0,
            "enable_cruv_pdes": False,
            "enable_crdes_CO": False,
        }

    res = None
    for _ in range(n_iter):
        res = run_chemistry(
            radm,
            model="gow17_slab_equilibrium",
            config=config,
            write=False,
        )
        radm.nco_gas = res.number_densities["co"]

    return res


@pytest.fixture(scope="module")
def cold_co_phase_slab():
    """Return a cold native slab with enough CO ice to test phase bookkeeping."""

    NH = np.geomspace(1.0e17, 1.0e22, 96)
    radm, _Av = build_model(1000.0, NH, chi0=1.0)
    result = run_gow17_slab_equilibrium(
        radm,
        1000.0,
        chi0=1.0,
        n_iter=1,
        enable_co_phase=True,
    )
    return radm, result


def test_slab_co_freezeout_uses_uniform_dust_temperature(cold_co_phase_slab) -> None:
    """Cold dust permits CO ice while warm dust thermally desorbs it."""

    _cold_rad, cold_result = cold_co_phase_slab
    ice_max = {
        10.0: float(np.max(cold_result.abundances["co_ice"].magnitude))
    }
    NH = np.geomspace(1.0e17, 1.0e22, 96)
    warm_rad, _Av = build_model(1000.0, NH, chi0=1.0)
    warm_rad.dust_temperature = Quantity(
        np.full(warm_rad.model.mesh.shape, 100.0),
        "K",
    )
    warm_result = run_gow17_slab_equilibrium(
        warm_rad,
        1000.0,
        chi0=1.0,
        n_iter=1,
        enable_co_phase=True,
    )
    ice_max[100.0] = float(np.max(warm_result.abundances["co_ice"].magnitude))

    assert ice_max[10.0] > 1.0e-5
    assert ice_max[100.0] < 1.0e-12


def test_native_slab_co_photodesorption_uses_fuv_attenuation(
    cold_co_phase_slab,
) -> None:
    """The native slab uses the process-specific exp(-1.8 Av) FUV field."""

    _radm, result = cold_co_phase_slab
    NH = result.fields["NH_slab"].to("cm^-2").magnitude.reshape(-1)
    G_CO_pdes = result.fields["G_CO_pdes"].magnitude.reshape(-1)
    expected = np.exp(-1.8 * NH / 1.87e21)

    np.testing.assert_allclose(G_CO_pdes, expected, rtol=2.0e-15, atol=0.0)


def test_native_slab_carbon_shielding_excludes_co_ice(
    cold_co_phase_slab,
) -> None:
    """Reconstruct native C shielding from gas C after removing CO ice."""

    radm, result = cold_co_phase_slab
    y = np.asarray(radm.gow17_y, dtype=np.float64).reshape(-1, N_Y)
    NH = result.fields["NH_slab"].to("cm^-2").magnitude.reshape(-1)
    theta_c = result.fields["theta_c"].magnitude.reshape(-1)
    x_c_neutral = 1.6e-4 - (
        y[:, I_HCOP]
        + y[:, I_CHX]
        + y[:, I_CO]
        + y[:, I_CO_ICE]
        + y[:, I_CP]
    )
    x_c_neutral = np.maximum(x_c_neutral, 0.0)

    expected = np.empty_like(theta_c)
    N_H2 = 0.0
    N_C = 0.0
    for i in range(NH.size):
        tau_h2 = 1.2e-14 * 2.0 * N_H2
        y_h2 = 1.17e-8 * tau_h2
        expected[i] = np.exp(-1.6e-17 * N_C) * np.exp(-y_h2) / (1.0 + y_h2)
        if i + 1 < NH.size:
            dNH = NH[i + 1] - NH[i]
            N_H2 += y[i, I_H2] * dNH
            N_C += x_c_neutral[i] * dNH

    assert float(np.max(y[:, I_CO_ICE])) > 1.0e-5
    np.testing.assert_allclose(theta_c, expected, rtol=2.0e-14, atol=1.0e-15)


def test_native_slab_reports_equilibrium_time_cap() -> None:
    """A deliberately short solve reports rather than hides its time cap."""

    NH = np.geomspace(1.0e17, 1.0e20, 8)
    radm, _Av = build_model(100.0, NH, chi0=1.0)
    result = run_gow17_slab_equilibrium(
        radm,
        100.0,
        chi0=1.0,
        n_iter=1,
        reltol=1.0e-12,
        tmin_s=3.16e12,
        tmax_s=5.056e13,
    )

    reached = result.fields["gow17_reached_tevol_max"].magnitude.reshape(-1)
    residual = result.fields["gow17_tevol_max_residual"].magnitude.reshape(-1)
    assert int(result.meta["n_fail"]) == int(np.count_nonzero(reached))
    assert int(result.meta["n_fail"]) > 0
    assert float(result.meta["tevol_max_residual_max"]) == pytest.approx(
        float(np.max(residual))
    )


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
    radm, Av_db = build_model(nH, NH, chi0=1.0)
    assert sum(int(n) > 1 for n in radm.ensure_nH().shape) == 1
    res = run_gow17_slab_equilibrium(radm, nH, chi0=1.0, n_iter=1)

    Y = np.asarray(radm.gow17_y).reshape(-1, N_Y)
    return Av_db, Y, res


@pytest.mark.skipif(not REF_DIR.exists(), reason="Reference data not available")
class TestGOW17Fig2Regression:
    """Regression tests against the original GOW17-style PDR calculation."""

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
    def test_species_match_external_pdr_reference(
        self,
        species,
        idx_db,
        diskbridge_nH100,
        reference_nH100,
    ) -> None:
        """Match the original PDR output within its solver and file precision."""
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
            assert float(np.max(rel)) <= EXTERNAL_PDR_RTOL

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
