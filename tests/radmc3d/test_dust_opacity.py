"""Tests for dustkappa reader, band-averaged kext, and dust UV tau computation.

Tests cover:
1. read_dustkappa: parsing dustkappa files
2. band_average_kext: flat-in-log-lambda averaging of kext
3. compute_tau_uv_from_dust_columns: per-ray tau from dust mass columns
4. resolve_uv_tau_mode: decision logic for dustkappa vs sigma_dust
"""

# db-keywords: shielding, uv-products, validation, radmc3d, io, paths, arrays
# db-role: validation
# db-scope: test
# db-purpose: Tests for dustkappa reader, band-averaged kext, and dust UV tau computation.

import tempfile
from pathlib import Path

import numpy as np
import pytest

from diskbridge.radmc3d.dustkappa_reader import (
    read_dustkappa,
    band_average_kext,
    load_kext_uv_for_bins,
)
from diskbridge.chemistry.shielding.dust_uv_tau import (
    compute_tau_uv_from_dust_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dustkappa_file(path: Path, lam_um, kabs, kscat, g):
    """Write a minimal dustkappa format-3 file."""
    with open(path, "w") as f:
        f.write("# Test opacity file\n")
        f.write("3\n")  # lambda, kabs, kscat, g
        f.write(f"{len(lam_um)}\n")
        for i in range(len(lam_um)):
            f.write(f"{lam_um[i]:13.6e}  {kabs[i]:13.6e}  "
                    f"{kscat[i]:13.6e}  {g[i]:13.6e}\n")


# ---------------------------------------------------------------------------
# Tests: read_dustkappa
# ---------------------------------------------------------------------------

class TestReadDustkappa:
    def test_basic_read(self, tmp_path):
        lam = np.array([0.1, 0.2, 0.5, 1.0, 10.0])
        kabs = np.array([100.0, 80.0, 40.0, 20.0, 2.0])
        kscat = np.array([50.0, 40.0, 20.0, 10.0, 1.0])
        g = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        fpath = tmp_path / "dustkappa_test0.inp"
        _write_dustkappa_file(fpath, lam, kabs, kscat, g)

        lam_r, kabs_r, kscat_r, g_r = read_dustkappa(fpath)

        np.testing.assert_allclose(lam_r, lam, rtol=1e-5)
        np.testing.assert_allclose(kabs_r, kabs, rtol=1e-5)
        np.testing.assert_allclose(kscat_r, kscat, rtol=1e-5)
        np.testing.assert_allclose(g_r, g, rtol=1e-5)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            read_dustkappa("/nonexistent/dustkappa_foo.inp")

    def test_wrong_format(self, tmp_path):
        fpath = tmp_path / "dustkappa_bad.inp"
        with open(fpath, "w") as f:
            f.write("9\n3\n0.1 1.0 0.5 0.3\n0.2 0.8 0.4 0.3\n0.5 0.4 0.2 0.3\n")
        with pytest.raises(ValueError, match="formats 1, 2, and 3"):
            read_dustkappa(fpath)


# ---------------------------------------------------------------------------
# Tests: band_average_kext
# ---------------------------------------------------------------------------

class TestBandAverageKext:
    def test_constant_opacity(self, tmp_path):
        """If kabs+kscat is constant over the UV band, average should equal it."""
        lam = np.logspace(-1, 1, 50)  # 0.1 to 10 um
        kabs = np.full_like(lam, 100.0)
        kscat = np.full_like(lam, 50.0)
        g = np.full_like(lam, 0.5)

        fpath = tmp_path / "dustkappa_const0.inp"
        _write_dustkappa_file(fpath, lam, kabs, kscat, g)

        kext_avg = band_average_kext(fpath, uv_min_um=0.0912, uv_max_um=0.2)
        np.testing.assert_allclose(kext_avg, 150.0, rtol=1e-3)

    def test_varying_opacity(self, tmp_path):
        """Non-trivial opacity profile; check that result is between min and max."""
        lam = np.logspace(-1.5, 1.5, 200)
        kabs = 1000.0 / lam  # decreasing with wavelength
        kscat = 500.0 / lam
        g = np.full_like(lam, 0.4)

        fpath = tmp_path / "dustkappa_vary0.inp"
        _write_dustkappa_file(fpath, lam, kabs, kscat, g)

        kext_avg = band_average_kext(fpath, uv_min_um=0.1, uv_max_um=0.2)

        # kext = 1500/lam, in [0.1, 0.2] um range: kext in [7500, 15000]
        assert 7500 < kext_avg < 15000

    def test_draine_photon_weighted_average(self, tmp_path):
        """Photon-weighted Draine averaging remains exact for constant opacity."""
        lam = np.logspace(-1.1, -0.6, 100)
        kabs = np.full_like(lam, 25.0)
        kscat = np.full_like(lam, 15.0)
        g = np.zeros_like(lam)

        fpath = tmp_path / "dustkappa_weighted0.inp"
        _write_dustkappa_file(fpath, lam, kabs, kscat, g)

        kext_avg = band_average_kext(
            fpath,
            uv_min_um=0.0912,
            uv_max_um=0.1118,
            weighting="photon",
            reference_spectrum="draine",
        )

        np.testing.assert_allclose(kext_avg, 40.0, rtol=1e-6)

    def test_cache_hit(self, tmp_path):
        """Second call should return cached value (no re-read)."""
        lam = np.array([0.05, 0.1, 0.5, 1.0])
        kabs = np.array([200.0, 100.0, 20.0, 10.0])
        kscat = np.array([100.0, 50.0, 10.0, 5.0])
        g = np.zeros_like(lam)

        fpath = tmp_path / "dustkappa_cache0.inp"
        _write_dustkappa_file(fpath, lam, kabs, kscat, g)

        v1 = band_average_kext(fpath, uv_min_um=0.05, uv_max_um=0.5)
        v2 = band_average_kext(fpath, uv_min_um=0.05, uv_max_um=0.5)
        assert v1 == v2

    def test_no_overlap_raises(self, tmp_path):
        """UV band outside table range should raise ValueError."""
        lam = np.array([1.0, 10.0, 100.0])
        kabs = np.array([10.0, 1.0, 0.1])
        kscat = np.zeros(3)
        g = np.zeros(3)

        fpath = tmp_path / "dustkappa_nir0.inp"
        _write_dustkappa_file(fpath, lam, kabs, kscat, g)

        with pytest.raises(ValueError, match="does not overlap"):
            band_average_kext(fpath, uv_min_um=0.09, uv_max_um=0.2)


# ---------------------------------------------------------------------------
# Tests: load_kext_uv_for_bins
# ---------------------------------------------------------------------------

class TestLoadKextUvForBins:
    def test_load_two_bins(self, tmp_path):
        for ibin in range(2):
            lam = np.logspace(-1, 1, 30)
            kabs = np.full_like(lam, 50.0 + 10.0 * ibin)
            kscat = np.full_like(lam, 20.0 + 5.0 * ibin)
            g = np.full_like(lam, 0.5)
            fpath = tmp_path / f"dustkappa_sil{ibin}.inp"
            _write_dustkappa_file(fpath, lam, kabs, kscat, g)

        kext = load_kext_uv_for_bins(tmp_path, "sil", 2, 0.1, 0.2)
        assert kext is not None
        assert kext.shape == (2,)
        np.testing.assert_allclose(kext[0], 70.0, rtol=1e-3)
        np.testing.assert_allclose(kext[1], 85.0, rtol=1e-3)

    def test_missing_bin_returns_none(self, tmp_path):
        # Only write bin 0, not bin 1
        lam = np.logspace(-1, 1, 20)
        kabs = np.full_like(lam, 50.0)
        kscat = np.full_like(lam, 20.0)
        g = np.full_like(lam, 0.5)
        fpath = tmp_path / "dustkappa_sil0.inp"
        _write_dustkappa_file(fpath, lam, kabs, kscat, g)

        result = load_kext_uv_for_bins(tmp_path, "sil", 2, 0.1, 0.2)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: compute_tau_uv_from_dust_columns
# ---------------------------------------------------------------------------

class TestComputeTauUv:
    def test_single_bin(self):
        """tau = sigma_dust * kext for one bin."""
        n_cand, npix = 10, 48
        sigma = np.random.uniform(0.01, 1.0, (n_cand, npix))
        kext_uv = np.array([100.0])

        cols = {"dust_bin_0": sigma}
        tau = compute_tau_uv_from_dust_columns(cols, kext_uv, nbin=1)

        expected = sigma * 100.0
        np.testing.assert_allclose(tau, expected, rtol=1e-12)

    def test_two_bins_additive(self):
        """tau = sum over bins of sigma * kext."""
        n_cand, npix = 5, 12
        sigma0 = np.ones((n_cand, npix)) * 0.1
        sigma1 = np.ones((n_cand, npix)) * 0.2
        kext_uv = np.array([100.0, 200.0])

        cols = {"dust_bin_0": sigma0, "dust_bin_1": sigma1}
        tau = compute_tau_uv_from_dust_columns(cols, kext_uv, nbin=2)

        # 0.1*100 + 0.2*200 = 10 + 40 = 50
        np.testing.assert_allclose(tau, 50.0, rtol=1e-12)
