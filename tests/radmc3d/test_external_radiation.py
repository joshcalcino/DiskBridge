"""Tests for RADMC-3D external radiation field assembly.

These tests use analytic blackbody/CMB checks and tiny synthetic wavelength
files to protect external UV/IR intensity scaling and writer policy.
"""

# db-keywords: uv-products, validation, units, radmc3d, field, io, paths
# db-role: validation
# db-scope: test
# db-purpose: Tests for RADMC-3D external radiation field assembly.

from pathlib import Path
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from diskbridge.radmc3d import writer as writer_mod
from diskbridge.radmc3d.writer import C_CGS, T_CMB, RadWriter
from diskbridge._constants import SIGMA_SB
from diskbridge._units import Quantity


def _writer() -> RadWriter:
    return RadWriter.__new__(RadWriter)


def _isrf_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ISRF.dat"


def _ir_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "MMP83_IR.dat"


def test_planck_B_nu_matches_cmb_component():
    w = _writer()
    lam_cm = np.array([100.0, 300.0, 1000.0]) * 1.0e-4
    nu = C_CGS / lam_cm

    np.testing.assert_allclose(w._planck_B_nu(nu, T_CMB), w._planck_B_nu(nu, 2.725), rtol=0.0, atol=0.0)


def test_external_i_nu_draine_component_scales_linearly_with_chi():
    w = _writer()
    lam_cm = np.array([100.0, 150.0, 500.0, 1500.0]) * 1.0e-7
    lam_tab_cm, i_nu_tab = w._load_leiden_draine_i_nu(_isrf_path())
    i_draine = np.exp(
        np.interp(np.log(lam_cm), np.log(lam_tab_cm), np.log(i_nu_tab))
    )

    assert np.all(i_draine > 0.0)
    np.testing.assert_allclose(1.0 * i_draine, 10.0 * (0.1 * i_draine), rtol=1.0e-14, atol=0.0)


def test_external_ir_background_disabled_gives_uv_plus_cmb():
    w = _writer()
    lam_cm = np.logspace(np.log10(0.0912), np.log10(3000.0), 1600) * 1.0e-4
    nu = C_CGS / lam_cm
    lam_tab_cm, i_nu_tab = w._load_leiden_draine_i_nu(_isrf_path())
    i_draine = np.zeros_like(lam_cm)
    m = (lam_cm >= lam_tab_cm.min()) & (lam_cm <= lam_tab_cm.max())
    i_draine[m] = np.exp(
        np.interp(np.log(lam_cm[m]), np.log(lam_tab_cm), np.log(i_nu_tab))
    )

    result = w._make_external_i_nu(
        lam_cm,
        chi=0.1,
        isrf_path=_isrf_path(),
        ir_path=_ir_path(),
        include_ir_background=False,
        T_back_ir=10.0,
        include_cmb=True,
    )

    np.testing.assert_allclose(
        result,
        0.1 * i_draine + w._planck_B_nu(nu, T_CMB),
        rtol=1.0e-14,
        atol=0.0,
    )


def test_external_ir_Tback_sets_total_integrated_intensity():
    w = _writer()
    lam_cm = np.logspace(np.log10(0.0912), np.log10(3000.0), 1600) * 1.0e-4
    nu = C_CGS / lam_cm

    result = w._make_external_i_nu(
        lam_cm,
        chi=0.1,
        isrf_path=_isrf_path(),
        ir_path=_ir_path(),
        T_back_ir=10.0,
    )
    target = (SIGMA_SB / np.pi) * 10.0**4

    assert w._integrate_i_nu(result, nu) == pytest.approx(target, rel=2.0e-3)


def test_external_uv_chi_adjusts_draine_and_forces_chi_ir_to_adjust():
    w = _writer()
    lam_cm = np.logspace(np.log10(0.0912), np.log10(3000.0), 1600) * 1.0e-4
    nu = C_CGS / lam_cm
    lam_tab_cm, i_nu_tab = w._load_leiden_draine_i_nu(_isrf_path())
    i_draine = np.zeros_like(lam_cm)
    m = (lam_cm >= lam_tab_cm.min()) & (lam_cm <= lam_tab_cm.max())
    i_draine[m] = np.exp(
        np.interp(np.log(lam_cm[m]), np.log(lam_tab_cm), np.log(i_nu_tab))
    )
    i_cmb = w._planck_B_nu(nu, T_CMB)
    i_ir_shape = w._load_mmp83_ir_i_nu(lam_cm, _ir_path())

    chi_ir_low_uv = w._compute_chi_ir(0.1 * i_draine, i_ir_shape, i_cmb, nu, 10.0)
    chi_ir_high_uv = w._compute_chi_ir(1.0 * i_draine, i_ir_shape, i_cmb, nu, 10.0)
    low = w._make_external_i_nu(
        lam_cm,
        chi=0.1,
        isrf_path=_isrf_path(),
        ir_path=_ir_path(),
        T_back_ir=10.0,
    )
    high = w._make_external_i_nu(
        lam_cm,
        chi=1.0,
        isrf_path=_isrf_path(),
        ir_path=_ir_path(),
        T_back_ir=10.0,
    )
    target = (SIGMA_SB / np.pi) * 10.0**4

    assert chi_ir_low_uv >= 0.0
    assert chi_ir_low_uv == pytest.approx(92.0, rel=0.02)
    assert chi_ir_high_uv >= 0.0
    assert chi_ir_high_uv < chi_ir_low_uv
    assert w._integrate_i_nu(high - low, nu) == pytest.approx(0.0, abs=2.0e-6)
    assert w._integrate_i_nu(low, nu) == pytest.approx(target, rel=2.0e-3)
    assert w._integrate_i_nu(high, nu) == pytest.approx(target, rel=2.0e-3)


def test_external_ir_background_requires_wavelength_grid_to_3000_micron(tmp_path):
    w = _writer()
    w.organize_files = False
    w.inputs_dir = "radmc3d_inputs"
    w.written_files = {}
    w.params = SimpleNamespace(
        external_uv_chi=0.1,
        external_ir_background=True,
        external_ir_Tback=Quantity(10.0, "K"),
        external_cmb=True,
    )
    wav = np.array([0.1, 1.0, 1000.0])
    with (tmp_path / "wavelength_micron.inp").open("w") as f:
        f.write(f"{wav.size}\n")
        for val in wav:
            f.write(f"{val:.16e}\n")

    with pytest.raises(
        ValueError,
        match="MMP83 IR background normalization requires lambda_max >= 3000 micron",
    ):
        w.write_external_source(tmp_path)


def test_configured_external_source_rejects_existing_mismatch(tmp_path):
    w = _writer()
    w.organize_files = False
    w.inputs_dir = "radmc3d_inputs"
    w.written_files = {}
    w.params = SimpleNamespace(
        external_uv_chi=1.0,
        external_ir_background=True,
        external_ir_Tback=Quantity(10.0, "K"),
        external_cmb=True,
    )
    wavelengths = np.logspace(np.log10(0.0912), np.log10(10000.0), 32)
    with (tmp_path / "wavelength_micron.inp").open("w") as handle:
        handle.write(f"{wavelengths.size}\n")
        for wavelength in wavelengths:
            handle.write(f"{wavelength:.16e}\n")

    source_path = w.ensure_external_source(tmp_path)
    expected = source_path.read_text()
    assert w.ensure_external_source(tmp_path).read_text() == expected

    source_path.write_text("different spectrum\n")
    with pytest.raises(RuntimeError, match="Remove the run directory"):
        w.ensure_external_source(tmp_path)


def test_external_source_writer_no_longer_uses_legacy_dust_normalization_path():
    source = inspect.getsource(writer_mod.RadWriter)

    assert not hasattr(RadWriter, "_make_" + "ism_background_lambda")
    assert not hasattr(RadWriter, "_parse_isrf_table")
    assert not hasattr(RadWriter, "_to_i_nu_from_table")
    assert not hasattr(RadWriter, "_planck_B_lambda")
    assert not hasattr(RadWriter, "_normalized_ir_greybody_i_nu")
    assert "T_color" not in source
    assert "beta_ir" not in source
    assert "external_ir_beta" not in source
    assert "external_ir_tau_ref" not in source
    assert "greybody" not in source.lower()
    assert ("dust" + "_norm") not in source
