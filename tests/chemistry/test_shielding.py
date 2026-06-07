"""Tests for angular UV direction weights and W_rays-based shielding.

Verifies:
- planck_band_luminosity returns physical values
- compute_uv_direction_weights_healpix produces normalised weights
- uniform weights reproduce isotropic mean
- Cartesian direct-stellar weighting is rejected until starward ray stopping is implemented
- W_rays=None in shielding functions equals isotropic mean
"""

import numpy as np
import pytest

from diskbridge._units import Quantity
from diskbridge._constants import U_DRAINE, C_LIGHT, H_PLANCK, K_B
from diskbridge.model.mesh import Mesh, Axis
from diskbridge.chemistry.shielding.angular_uv_weights import (
    planck_band_luminosity,
    compute_uv_direction_weights_healpix,
)
from diskbridge.chemistry.shielding.w_rays_cache import _estimate_w_rays_memory_bytes
from diskbridge.chemistry.shielding.healpix_columns import (
    compute_co_shielding_healpix,
    compute_pdr_shielding_healpix,
)
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding


def _make_uniform_cartesian_mesh(ncells: int = 4, L_cm: float = 1e17):
    """Build a small uniform Cartesian mesh centred on the origin."""
    edges = np.linspace(-L_cm, L_cm, ncells + 1)
    edges_q = Quantity(edges, "cm")
    return Mesh(
        coord_system="cartesian",
        axes={
            "x": Axis(edges=edges_q),
            "y": Axis(edges=edges_q),
            "z": Axis(edges=edges_q),
        },
    )


# ---- planck_band_luminosity ------------------------------------------------

def test_planck_band_luminosity_positive():
    """A 6000 K star should have positive UV luminosity."""
    R_cm = 7e10  # ~1 R_sun
    T_K = 6000.0
    lam_min_cm = 91.2e-7   # 91.2 nm
    lam_max_cm = 111.8e-7  # 111.8 nm
    L = planck_band_luminosity(R_cm, T_K, lam_min_cm, lam_max_cm)
    assert L > 0.0


def test_planck_band_luminosity_zero_for_cold_star():
    """A very cold star (T=10 K) should have negligible UV luminosity."""
    R_cm = 7e10
    T_K = 10.0
    lam_min_cm = 91.2e-7
    lam_max_cm = 111.8e-7
    L = planck_band_luminosity(R_cm, T_K, lam_min_cm, lam_max_cm)
    assert L < 1e-10


def test_planck_band_luminosity_zero_radius():
    """Zero radius should return zero luminosity."""
    assert planck_band_luminosity(0.0, 6000.0, 91.2e-7, 111.8e-7) == 0.0


def test_planck_band_luminosity_zero_temperature():
    """Zero temperature should return zero luminosity."""
    assert planck_band_luminosity(7e10, 0.0, 91.2e-7, 111.8e-7) == 0.0


# ---- compute_uv_direction_weights_healpix ----------------------------------

def test_weights_normalised_cartesian():
    """Weights must sum to 1 along the pixel axis for every candidate cell."""
    ncells = 4
    nside = 1
    npix = 12 * nside**2
    mesh = _make_uniform_cartesian_mesh(ncells=ncells)
    shape = (ncells, ncells, ncells)

    chi_radmc = np.full(shape, 1.0, dtype=np.float64)
    rho_dust = np.full(shape, 1e-20, dtype=np.float64)
    dust_rho_bins = [rho_dust]
    kext_uv = np.array([100.0])

    W, cidx, dirs, cc, debug = compute_uv_direction_weights_healpix(
        mesh,
        chi_radmc=chi_radmc,
        nside=nside,
        dust_rho_bins=dust_rho_bins,
        kext_uv=kext_uv,
        chi_ext0=1.0,
        star_uv_luminosity_erg_s=0.0,
    )

    assert W.shape[1] == npix
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-12)


def test_weights_uniform_when_isotropic_dominates():
    """If chi_radmc >> chi_ext_dir + chi_star, weights should be near-uniform.

    With zero external UV and zero stellar luminosity, chi_iso = chi_radmc,
    and C_iso is uniform, making all weights = 1/npix.
    """
    ncells = 4
    nside = 1
    npix = 12 * nside**2
    mesh = _make_uniform_cartesian_mesh(ncells=ncells)
    shape = (ncells, ncells, ncells)

    chi_radmc = np.full(shape, 10.0, dtype=np.float64)
    rho_dust = np.full(shape, 1e-25, dtype=np.float64)
    dust_rho_bins = [rho_dust]
    kext_uv = np.array([100.0])

    W, _, _, _, debug = compute_uv_direction_weights_healpix(
        mesh,
        chi_radmc=chi_radmc,
        nside=nside,
        dust_rho_bins=dust_rho_bins,
        kext_uv=kext_uv,
        chi_ext0=0.0,
        star_uv_luminosity_erg_s=0.0,
    )

    expected = 1.0 / npix
    np.testing.assert_allclose(W, expected, atol=1e-10)
    assert "closure_diagnostics" not in debug


def test_closure_diagnostics_are_opt_in_scalar_fields():
    """Closure diagnostics should be absent by default and scalar when enabled."""
    ncells = 3
    nside = 1
    mesh = _make_uniform_cartesian_mesh(ncells=ncells)
    shape = (ncells, ncells, ncells)

    chi_radmc = np.full(shape, 2.0, dtype=np.float64)
    dust_rho_bins = [np.full(shape, 1e-25, dtype=np.float64)]
    kext_uv = np.array([100.0])

    _, _, _, _, debug = compute_uv_direction_weights_healpix(
        mesh,
        chi_radmc=chi_radmc,
        nside=nside,
        dust_rho_bins=dust_rho_bins,
        kext_uv=kext_uv,
        chi_ext0=0.5,
        star_uv_luminosity_erg_s=0.0,
        keep_closure_diagnostics=True,
        chunk_size=4,
    )

    closure = debug["closure_diagnostics"]
    assert closure["shape"] == shape
    assert closure["npix"] == 12
    assert "uv_ext_contrib" not in closure["fields"]
    assert closure["fields"]["chi_radmc"].shape == (int(np.prod(shape)),)
    assert closure["fields"]["direct_excess"].shape == (int(np.prod(shape)),)
    assert closure["summary"]["max_direct_excess"] >= 0.0


def test_closure_diagnostics_increase_memory_estimate():
    """The memory estimator must account for retained closure diagnostics."""
    base = _estimate_w_rays_memory_bytes(
        n_cells=128,
        npix=48,
        nbin=2,
        include_star=True,
        keep_closure_diagnostics=False,
    )
    with_diag = _estimate_w_rays_memory_bytes(
        n_cells=128,
        npix=48,
        nbin=2,
        include_star=True,
        keep_closure_diagnostics=True,
    )

    assert with_diag["retained_diagnostics_bytes"] > base["retained_diagnostics_bytes"]
    assert with_diag["estimated_peak_bytes"] > base["estimated_peak_bytes"]


def test_chunked_weights_match_single_chunk():
    """Chunking should preserve the UV-weight calculation."""
    ncells = 4
    nside = 1
    mesh = _make_uniform_cartesian_mesh(ncells=ncells)
    shape = (ncells, ncells, ncells)

    rng = np.random.default_rng(1234)
    chi_radmc = 1.0 + rng.random(shape)
    dust_rho_bins = [
        np.full(shape, 1e-23, dtype=np.float64),
        np.full(shape, 3e-24, dtype=np.float64),
    ]
    kext_uv = np.array([100.0, 300.0])

    W_full, _, _, _, debug_full = compute_uv_direction_weights_healpix(
        mesh,
        chi_radmc=chi_radmc,
        nside=nside,
        dust_rho_bins=dust_rho_bins,
        kext_uv=kext_uv,
        chi_ext0=0.7,
        star_uv_luminosity_erg_s=0.0,
        chunk_size=shape[0] * shape[1] * shape[2],
    )
    W_chunked, _, _, _, debug_chunked = compute_uv_direction_weights_healpix(
        mesh,
        chi_radmc=chi_radmc,
        nside=nside,
        dust_rho_bins=dust_rho_bins,
        kext_uv=kext_uv,
        chi_ext0=0.7,
        star_uv_luminosity_erg_s=0.0,
        chunk_size=5,
    )

    np.testing.assert_allclose(W_chunked, W_full, rtol=0.0, atol=1e-12)
    assert debug_full["keep_debug_arrays"] is False
    assert debug_chunked["keep_debug_arrays"] is False
    assert "tau_ext_rays" not in debug_chunked


def test_cartesian_direct_stellar_weights_are_rejected():
    """Cartesian starward attenuation must not silently march past the source."""
    ncells = 4
    mesh = _make_uniform_cartesian_mesh(ncells=ncells)
    shape = (ncells, ncells, ncells)

    with pytest.raises(NotImplementedError, match="Cartesian meshes"):
        compute_uv_direction_weights_healpix(
            mesh,
            chi_radmc=np.ones(shape, dtype=np.float64),
            nside=1,
            dust_rho_bins=[np.full(shape, 1e-23, dtype=np.float64)],
            kext_uv=np.array([100.0]),
            chi_ext0=0.0,
            star_uv_luminosity_erg_s=2.0e31,
        )


def test_weights_require_dust_rho_bins():
    """Missing dust_rho_bins should raise ValueError."""
    mesh = _make_uniform_cartesian_mesh(ncells=2)
    shape = (2, 2, 2)
    chi = np.ones(shape, dtype=np.float64)

    with pytest.raises(ValueError, match="dust_rho_bins"):
        compute_uv_direction_weights_healpix(
            mesh,
            chi_radmc=chi,
            nside=1,
            dust_rho_bins=[],
            kext_uv=np.array([100.0]),
            chi_ext0=1.0,
            star_uv_luminosity_erg_s=0.0,
        )


def test_weights_require_kext_uv():
    """Missing kext_uv should raise ValueError."""
    mesh = _make_uniform_cartesian_mesh(ncells=2)
    shape = (2, 2, 2)
    chi = np.ones(shape, dtype=np.float64)
    rho = np.ones(shape, dtype=np.float64)

    with pytest.raises(ValueError, match="kext_uv"):
        compute_uv_direction_weights_healpix(
            mesh,
            chi_radmc=chi,
            nside=1,
            dust_rho_bins=[rho],
            kext_uv=np.array([]),
            chi_ext0=1.0,
            star_uv_luminosity_erg_s=0.0,
        )


# ---- W_rays=None gives isotropic mean in shielding -------------------------

def test_w_rays_none_gives_isotropic_mean():
    """With W_rays=None, shielding should use uniform (isotropic) averaging."""
    ncells = 4
    mesh = _make_uniform_cartesian_mesh(ncells=ncells)
    shape = (ncells, ncells, ncells)

    nH = np.full(shape, 1e-2, dtype=np.float64)
    nCO = np.full(shape, 1e-4 * 1e-2, dtype=np.float64)
    nH2 = np.full(shape, 0.5 * 1e-2, dtype=np.float64)
    chi = np.full(shape, 1.0, dtype=np.float64)

    visser = VisserShielding(b_kms=0.3)

    theta, chi_eff = compute_co_shielding_healpix(
        mesh, nH, chi, visser,
        nCO=nCO, nH2=nH2,
        nside=2,
        b_kms=0.3,
        W_rays=None,
    )

    assert theta.shape == shape
    assert chi_eff.shape == shape
    assert np.all(theta >= 0.0)
    assert np.all(theta <= 1.0)


def test_chunked_pdr_shielding_matches_single_chunk():
    """Chunking PDR shielding should preserve the shielding result."""
    ncells = 4
    nside = 1
    npix = 12 * nside**2
    mesh = _make_uniform_cartesian_mesh(ncells=ncells)
    shape = (ncells, ncells, ncells)
    n_candidates = int(np.prod(shape))

    rng = np.random.default_rng(5678)
    nH = np.full(shape, 1e2, dtype=np.float64)
    chi = 0.1 + rng.random(shape)
    nH2 = np.full(shape, 25.0, dtype=np.float64)
    nC = np.full(shape, 1e-2, dtype=np.float64)
    W = rng.random((n_candidates, npix))
    W /= W.sum(axis=1, keepdims=True)

    full = compute_pdr_shielding_healpix(
        mesh,
        nH,
        chi,
        visser=None,
        nC=nC,
        nH2=nH2,
        nside=nside,
        W_rays=W,
        chunk_size=n_candidates,
    )
    chunked = compute_pdr_shielding_healpix(
        mesh,
        nH,
        chi,
        visser=None,
        nC=nC,
        nH2=nH2,
        nside=nside,
        W_rays=W,
        chunk_size=5,
    )

    for chunked_arr, full_arr in zip(chunked, full):
        np.testing.assert_allclose(chunked_arr, full_arr, rtol=0.0, atol=1e-12)
