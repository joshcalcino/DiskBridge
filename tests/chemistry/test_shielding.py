"""Tests for angular UV direction weights and W_rays-based shielding.

Verifies:
- planck_band_luminosity returns physical values
- compute_uv_direction_weights_healpix produces normalised weights
- uniform weights reproduce isotropic mean
- direct stellar weighting uses exact starward shielding instead of HEALPix
  pixel-center boundary columns
- W_rays=None in shielding functions equals isotropic mean
"""

# db-keywords: shielding, co-shielding, healpix-columns, uv-products, gow17, validation, units, radmc3d, chemistry, model, mesh
# db-role: validation
# db-scope: test
# db-purpose: Tests for angular UV direction weights and W_rays-based shielding.

import numpy as np
import pytest
import healpy as hp

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


def _write_minimal_visser_table(path, *, tex: float, carbon_ratio: int) -> None:
    path.write_text(
        "\n".join(
            [
                "b(CO,H2,H) (km/s)    =   0.30  3.00  5.00",
                f"Tex(CO,H2) (K)       =   {tex:.2f} 11.18",
                f"[12C]/[13C]          =  {carbon_ratio}",
                "[16O]/[18O]          = 557",
                "[18O]/[17O]          =   3.6",
                "n[N(12CO)]           =  2",
                "n[N(H2)]             =  2",
                "N(12CO)",
                " 1.000E+10",
                " 1.000E+11",
                "N(H2)",
                " 1.000E+15",
                " 1.000E+16",
                "12C16O",
                " 9.000E-01 8.000E-01",
                " 7.000E-01 6.000E-01",
            ]
        )
        + "\n"
    )


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


def _star_weight_metadata_for_target(shape, target_idx, *, nside, w_star):
    """Build compact origin-star metadata in candidate C-order for one cell."""
    npix = 12 * int(nside) ** 2
    target_flat = np.ravel_multi_index(target_idx, shape)
    k_star = np.zeros(int(np.prod(shape)), dtype=np.intp)
    w = np.zeros(int(np.prod(shape)), dtype=np.float64)
    valid = np.zeros(int(np.prod(shape)), dtype=bool)
    k = hp.vec2pix(int(nside), 0.0, 1.0, 0.0)
    k_star[target_flat] = k
    w[target_flat] = float(w_star)
    valid[target_flat] = float(w_star) > 0.0
    return {
        "k_star": k_star,
        "w_star": w,
        "valid_star": valid,
        "shape": shape,
        "nside": int(nside),
        "npix": npix,
        "star_inner_radius_cm": 0.0,
    }


def _one_pixel_weight_for_target(shape, target_idx, *, nside):
    """Return W_rays with the target cell fully weighted toward the origin."""
    npix = 12 * int(nside) ** 2
    target_flat = np.ravel_multi_index(target_idx, shape)
    W = np.full((int(np.prod(shape)), npix), 1.0 / npix, dtype=np.float64)
    W[target_flat, :] = 0.0
    W[target_flat, hp.vec2pix(int(nside), 0.0, 1.0, 0.0)] = 1.0
    return W


def _star_shielding_test_fields(*, nside):
    """Create a case where the boundary pixel is shielded but starward LoS is clear."""
    shape = (3, 3, 3)
    target = (1, 0, 1)
    mesh = _make_uniform_cartesian_mesh(ncells=3, L_cm=3.0)
    nH = np.ones(shape, dtype=np.float64)
    chi = np.ones(shape, dtype=np.float64)
    nH2 = np.zeros(shape, dtype=np.float64)
    nH2[:, 2, :] = 1.0e30
    nC = np.zeros(shape, dtype=np.float64)
    W = _one_pixel_weight_for_target(shape, target, nside=nside)
    metadata = _star_weight_metadata_for_target(
        shape,
        target,
        nside=nside,
        w_star=1.0,
    )
    return mesh, shape, target, nH, chi, nH2, nC, W, metadata


def test_visser_parser_transposes_h2_major_theta_blocks(tmp_path):
    """Visser files store shielding blocks with N(CO) varying fastest."""
    table = tmp_path / "shield.03.5.35-557-36.dat"
    table.write_text(
        "\n".join(
            [
                "b(CO,H2,H) (km/s)    =   0.30  3.00  5.00",
                "Tex(CO,H2) (K)       =   5.00 11.18",
                "[12C]/[13C]          =  35",
                "[16O]/[18O]          = 557",
                "[18O]/[17O]          =   3.6",
                "n[N(12CO)]           =  3",
                "n[N(H2)]             =  2",
                "N(12CO)",
                " 1.000E+10",
                " 1.000E+11",
                " 1.000E+12",
                "N(H2)",
                " 1.000E+15",
                " 1.000E+16",
                "12C16O",
                " 9.000E-01 8.000E-01 7.000E-01",
                " 6.000E-01 5.000E-01 4.000E-01",
            ]
        )
        + "\n"
    )

    visser = VisserShielding(
        data_dir=tmp_path,
        filename=table.name,
        b_kms=0.3,
        auto_download=False,
    )

    np.testing.assert_allclose(
        visser.theta(
            "co",
            np.array([[1.0e10, 1.0e11, 1.0e12], [1.0e10, 1.0e11, 1.0e12]]),
            np.array([[1.0e15, 1.0e15, 1.0e15], [1.0e16, 1.0e16, 1.0e16]]),
            b_kms=0.3,
        ),
        np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_visser_default_prefers_original_gow17_table_family(tmp_path):
    """Same-b ties should choose Visser Table 5, matching original GOW17."""
    _write_minimal_visser_table(
        tmp_path / "shield.03.100.35-557-36.dat",
        tex=100.0,
        carbon_ratio=35,
    )
    _write_minimal_visser_table(
        tmp_path / "shield.03.5.35-557-36.dat",
        tex=5.0,
        carbon_ratio=35,
    )
    _write_minimal_visser_table(
        tmp_path / "shield.03.5.69-557-36.dat",
        tex=5.0,
        carbon_ratio=69,
    )

    visser = VisserShielding(data_dir=tmp_path, b_kms=0.3, auto_download=False)

    assert visser.filepath.name == "shield.03.5.69-557-36.dat"


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


def test_weighted_stellar_shielding_uses_exact_starward_column():
    """A star-only weight should not use gas behind the star on a boundary ray."""
    (
        mesh,
        _shape,
        target,
        nH,
        chi,
        nH2,
        nC,
        W,
        metadata,
    ) = _star_shielding_test_fields(nside=1)

    uncorrected = compute_pdr_shielding_healpix(
        mesh,
        nH,
        chi,
        visser=None,
        nC=nC,
        nH2=nH2,
        nside=1,
        W_rays=W,
    )[0]
    corrected = compute_pdr_shielding_healpix(
        mesh,
        nH,
        chi,
        visser=None,
        nC=nC,
        nH2=nH2,
        nside=1,
        W_rays=W,
        stellar_metadata=metadata,
    )[0]

    assert uncorrected[target] < 1.0e-6
    assert corrected[target] > 0.99


def test_stellar_shielding_correction_only_replaces_stellar_fraction():
    """Diffuse light in the star HEALPix pixel must keep its boundary shielding."""
    (
        mesh,
        shape,
        target,
        nH,
        chi,
        nH2,
        nC,
        W,
        _metadata,
    ) = _star_shielding_test_fields(nside=1)
    metadata = _star_weight_metadata_for_target(
        shape,
        target,
        nside=1,
        w_star=0.25,
    )

    uncorrected = compute_pdr_shielding_healpix(
        mesh,
        nH,
        chi,
        visser=None,
        nC=nC,
        nH2=nH2,
        nside=1,
        W_rays=W,
    )[0]
    corrected = compute_pdr_shielding_healpix(
        mesh,
        nH,
        chi,
        visser=None,
        nC=nC,
        nH2=nH2,
        nside=1,
        W_rays=W,
        stellar_metadata=metadata,
    )[0]

    assert corrected[target] > uncorrected[target]
    assert 0.20 < corrected[target] < 0.50


def test_starward_stellar_shielding_is_not_nside_pixel_center_limited():
    """Exact stellar shielding should stay clear as the HEALPix pixel changes."""
    for nside in (1, 2, 4):
        (
            mesh,
            _shape,
            target,
            nH,
            chi,
            nH2,
            nC,
            W,
            metadata,
        ) = _star_shielding_test_fields(nside=nside)

        corrected = compute_pdr_shielding_healpix(
            mesh,
            nH,
            chi,
            visser=None,
            nC=nC,
            nH2=nH2,
            nside=nside,
            W_rays=W,
            stellar_metadata=metadata,
        )[0]

        assert corrected[target] > 0.99


def test_source_cell_has_no_undefined_stellar_weight():
    """The origin cell should not receive a point-source direction or singular weight."""
    mesh = _make_uniform_cartesian_mesh(ncells=3, L_cm=3.0)
    shape = (3, 3, 3)
    chi = np.ones(shape, dtype=np.float64)
    dust = np.full(shape, 1.0e-30, dtype=np.float64)

    W, _candidate_idx, _dirs, _centers, debug = compute_uv_direction_weights_healpix(
        mesh,
        chi_radmc=chi,
        nside=1,
        dust_rho_bins=[dust],
        kext_uv=np.array([1.0]),
        chi_ext0=0.0,
        star_uv_luminosity_erg_s=1.0,
    )

    origin_flat = np.ravel_multi_index((1, 1, 1), shape)
    stellar = debug["stellar"]
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1.0e-12)
    assert not bool(stellar["valid_star"][origin_flat])
    assert stellar["w_star"][origin_flat] == 0.0
