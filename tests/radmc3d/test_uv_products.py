"""Tests for RADMC-3D UV product normalization and partition contracts.

These tests use analytic spectra, Draine reference integrals, and small
synthetic RADMC workflows to protect deterministic UV-band product behavior.
"""

# db-keywords: uv-products, disk-mask, validation, units, radmc3d, model, mesh, field
# db-role: validation
# db-scope: test
# db-purpose: Tests for RADMC-3D UV product normalization and partition contracts.

import numpy as np
import pytest

from diskbridge._units import Quantity, units
from diskbridge.radmc3d.model import RadModel
from diskbridge.radmc3d.uv_products import (
    C_CGS,
    H_CGS,
    UV_PRODUCT_EDGES_NM,
    UV_PRODUCT_MERGED_FIELD_NAMES,
    UVPartition,
    build_partition_weights,
    compute_chi_broad,
    compute_uv_products,
    default_isrf_path,
    draine_references_for_product_partitions,
    draine_reference_for_product,
    integrate_partitioned,
    load_draine_reference,
    _loglinear_interp_strict,
)
from diskbridge.radmc3d.segmented import SegmentedRadRunner
from diskbridge.radmc3d.wavelengths import build_mcmono_wavelengths


def _draine_jnu_on_grid(lam_nm: np.ndarray, chi: float = 1.0) -> tuple[Quantity, Quantity]:
    lam_ref_nm, photon_ref = load_draine_reference(default_isrf_path())
    photon_nm = np.interp(lam_nm, lam_ref_nm, photon_ref)
    lam_cm = lam_nm * 1.0e-7
    j_lambda = photon_nm * (H_CGS * C_CGS / lam_cm) / 1.0e-7 / (4.0 * np.pi)
    j_nu = chi * j_lambda * lam_cm**2 / C_CGS
    freq_hz = C_CGS / lam_cm
    return Quantity(freq_hz, "Hz"), Quantity(j_nu[None, :], "erg/(s*cm^2*Hz*sr)")


def test_loglinear_interp_strict_reconstructs_power_law_and_rejects_bad_values():
    lam_sample = np.array([90.0, 120.0, 180.0, 240.0])
    lam_quad = np.array([100.0, 150.0, 210.0])
    J_sample = np.vstack([lam_sample**2, 3.0 * lam_sample**2])

    J_quad = _loglinear_interp_strict(lam_sample, J_sample, lam_quad)

    np.testing.assert_allclose(J_quad[0], lam_quad**2, rtol=1.0e-14)
    np.testing.assert_allclose(J_quad[1], 3.0 * lam_quad**2, rtol=1.0e-14)
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        _loglinear_interp_strict(lam_sample, np.array([[1.0, 0.0, 2.0, 3.0]]), lam_quad)


def test_uv_products_normalize_to_scaled_draine_field():
    lam_ref_nm, _ = load_draine_reference(default_isrf_path())
    lam_nm = np.unique(
        np.concatenate(
            [
                lam_ref_nm[(lam_ref_nm >= 91.2) & (lam_ref_nm <= 206.7)],
                UV_PRODUCT_EDGES_NM,
            ]
        )
    )
    chi0 = 3.7
    freq_hz, jnu = _draine_jnu_on_grid(lam_nm, chi=chi0)

    products = compute_uv_products(freq_hz, jnu, (1, 1, 1), isrf_path=default_isrf_path())

    for name in ["chi_broad", "G_CO_diss", "G_H2_diss", "G_C_ion", "G_CO_pdes"]:
        assert products[name].to("dimensionless").magnitude.item() == pytest.approx(
            chi0,
            rel=2e-3,
        )


def test_pure_draine_uv_products_do_not_alias_band_normalizations():
    lam_ref_nm, photon_ref = load_draine_reference(default_isrf_path())
    lam_nm = np.unique(
        np.concatenate(
            [
                lam_ref_nm[(lam_ref_nm >= 91.2) & (lam_ref_nm <= 206.7)],
                UV_PRODUCT_EDGES_NM,
            ]
        )
    )
    freq_hz, jnu = _draine_jnu_on_grid(lam_nm, chi=1.0)

    products = compute_uv_products(freq_hz, jnu, (1, 1, 1), isrf_path=default_isrf_path())

    normalized_names = ["chi_broad", "G_CO_diss", "G_H2_diss", "G_C_ion", "G_CO_pdes"]
    values = np.array(
        [products[name].to("dimensionless").magnitude.item() for name in normalized_names],
        dtype=np.float64,
    )
    assert np.allclose(values, np.ones_like(values), rtol=2e-3, atol=0.0)

    mask = (lam_ref_nm > 91.2) & (lam_ref_nm < 205.0)
    lam = np.concatenate(([91.2], lam_ref_nm[mask], [205.0])).astype(np.float64)
    photon = np.interp(lam, lam_ref_nm, photon_ref).astype(np.float64)
    pdes_flux_ref = float(np.trapezoid(photon, lam))
    assert products["F_CO_pdes_photon"].to("1/(cm^2 s)").magnitude.item() == pytest.approx(
        pdes_flux_ref,
        rel=2e-3,
    )
    band_flux = products["F_CO_pdes_photon_bands"].to("1/(cm^2 s)").magnitude
    assert band_flux.shape[1:] == (1, 1, 1)
    assert float(np.sum(band_flux[:, 0, 0, 0])) == pytest.approx(pdes_flux_ref, rel=2e-3)


def test_non_product_chi_uses_canonical_broad_band():
    lam_ref_nm, _ = load_draine_reference(default_isrf_path())
    lam_nm = np.unique(
        np.concatenate(
            [
                lam_ref_nm[(lam_ref_nm >= 91.2) & (lam_ref_nm <= 206.7)],
                UV_PRODUCT_EDGES_NM,
            ]
        )
    )
    freq_hz, jnu = _draine_jnu_on_grid(lam_nm, chi=2.25)

    chi_broad = compute_chi_broad(freq_hz, jnu, (1, 1, 1), isrf_path=default_isrf_path())

    assert chi_broad.to("dimensionless").magnitude.item() == pytest.approx(2.25, rel=2e-3)


def test_co_pdes_reference_uses_pdes_band_not_legacy_broad_flux():
    ref = draine_reference_for_product("F_CO_pdes_photon").to("1/(cm^2 s)").magnitude
    lam_ref_nm, photon_ref = load_draine_reference(default_isrf_path())
    mask = (lam_ref_nm > 91.2) & (lam_ref_nm < 205.0)
    lam = np.concatenate(([91.2], lam_ref_nm[mask], [205.0])).astype(np.float64)
    photon = np.interp(lam, lam_ref_nm, photon_ref).astype(np.float64)

    assert ref == pytest.approx(float(np.trapezoid(photon, lam)), rel=2e-3)
    assert ref != pytest.approx(2.0e8, rel=1e-3)


def test_co_pdes_partition_references_sum_to_product_reference():
    ref = draine_reference_for_product("F_CO_pdes_photon").to("1/(cm^2 s)").magnitude
    band_refs = draine_references_for_product_partitions("F_CO_pdes_photon").to(
        "1/(cm^2 s)"
    ).magnitude

    assert band_refs.ndim == 1
    assert np.all(band_refs > 0.0)
    assert float(np.sum(band_refs)) == pytest.approx(float(ref), rel=2e-3)


def test_partition_weights_sum_over_expected_intervals():
    lam_nm = np.array([91.2, 100.0, 110.1, 111.8, 150.0, 205.0, 206.7])
    freq_hz = C_CGS / (lam_nm * 1.0e-7)
    partitions = [
        UVPartition("A", 91.2, 110.1),
        UVPartition("B", 110.1, 111.8),
        UVPartition("C", 111.8, 205.0),
        UVPartition("D", 205.0, 206.7),
    ]
    weights = build_partition_weights(freq_hz, lam_nm, partitions, weighting="energy")
    parts = integrate_partitioned(np.ones((1, lam_nm.size)), weights)

    broad = parts[0, 0] + parts[1, 0] + parts[2, 0] + parts[3, 0]
    hard = parts[0, 0] + parts[1, 0]
    c_ion = parts[0, 0]
    pdes = parts[0, 0] + parts[1, 0] + parts[2, 0]

    assert broad == pytest.approx(np.sum(parts[:, 0]))
    assert hard == pytest.approx(np.sum(parts[:2, 0]))
    assert c_ion == pytest.approx(parts[0, 0])
    assert pdes == pytest.approx(np.sum(parts[:3, 0]))


def test_blackbody_spectrum_has_hard_broad_contrast():
    lam_nm = np.unique(np.concatenate([np.linspace(91.2, 206.7, 240), UV_PRODUCT_EDGES_NM]))
    lam_cm = lam_nm * 1.0e-7
    freq_hz = C_CGS / lam_cm
    temp = 10000.0
    x = H_CGS * freq_hz / (units("k_B").to("erg/K").magnitude * temp)
    jnu = (2.0 * H_CGS * freq_hz**3 / C_CGS**2) / np.expm1(x)

    products = compute_uv_products(
        Quantity(freq_hz, "Hz"),
        Quantity(jnu[None, :], "erg/(s*cm^2*Hz*sr)"),
        (1, 1, 1),
        isrf_path=default_isrf_path(),
    )

    broad = products["chi_broad"].magnitude.item()
    assert products["G_CO_diss"].magnitude.item() / broad != pytest.approx(1.0, rel=1e-2)
    assert products["G_C_ion"].magnitude.item() / broad != pytest.approx(1.0, rel=1e-2)


def test_c_ion_product_uses_shorter_edge_than_hard_band():
    lam_nm = np.array([91.2, 110.1, 111.8, 205.0, 206.7])
    freq_hz = C_CGS / (lam_nm * 1.0e-7)
    jnu = np.ones((1, lam_nm.size))

    products = compute_uv_products(
        Quantity(freq_hz, "Hz"),
        Quantity(jnu, "erg/(s*cm^2*Hz*sr)"),
        (1, 1, 1),
        isrf_path=default_isrf_path(),
    )

    assert products["G_C_ion"].magnitude.item() != pytest.approx(
        products["G_CO_diss"].magnitude.item()
    )


def test_mcmono_wavelengths_include_extra_uv_product_edges():
    wavelengths = build_mcmono_wavelengths(
        wavelength_source="uv",
        wavelength_file=None,
        uv_min_um=0.0912,
        uv_max_um=0.2067,
        n_wavelengths=5,
        n_uv_enforce=0,
        extra_enforced_wavelengths_um=UV_PRODUCT_EDGES_NM * 1.0e-3,
    )

    for edge_um in UV_PRODUCT_EDGES_NM * 1.0e-3:
        assert np.any(np.isclose(wavelengths, edge_um))


def test_radmodel_uv_product_fallback_to_chi():
    rad = RadModel.__new__(RadModel)
    rad.uv_products = {}
    rad.chi = Quantity(np.ones((2, 1, 1)), "dimensionless")
    rad.model = type("DummyModel", (), {"gas": {}})()

    product = RadModel.ensure_uv_product(rad, "G_CO_diss", fallback_to_chi=True)

    assert product is rad.chi
    with pytest.raises(KeyError):
        RadModel.ensure_uv_product(rad, "F_CO_pdes_photon", fallback_to_chi=False)


def test_segment_uv_products_assign_draine_equivalent_outer_policy():
    rad = RadModel.__new__(RadModel)
    chi = Quantity(np.full((2, 1, 1), 4.0), "dimensionless")
    rad.uv_products = {"chi_broad": chi}
    rad.chi = chi
    rad.model = type("DummyModel", (), {"gas": {}})()

    runner = SegmentedRadRunner.__new__(SegmentedRadRunner)
    products = SegmentedRadRunner._segment_uv_product_fields(runner, rad, measured=False)

    for name in ["chi_broad", "G_CO_diss", "G_H2_diss", "G_C_ion", "G_CO_pdes"]:
        np.testing.assert_allclose(
            products[name].to("dimensionless").magnitude,
            chi.magnitude,
        )
    assert products["F_CO_pdes_photon"].to("1/(cm^2 s)").magnitude.shape == chi.magnitude.shape
    bands = products["F_CO_pdes_photon_bands"].to("1/(cm^2 s)").magnitude
    assert bands.shape[1:] == chi.magnitude.shape
    np.testing.assert_allclose(np.sum(bands, axis=0), products["F_CO_pdes_photon"].magnitude)


def test_measured_segment_requires_all_uv_products():
    rad = RadModel.__new__(RadModel)
    chi = Quantity(np.ones((1, 1, 1)), "dimensionless")
    rad.uv_products = {"chi_broad": chi}
    rad.chi = chi
    rad.model = type("DummyModel", (), {"gas": {}})()

    runner = SegmentedRadRunner.__new__(SegmentedRadRunner)
    with pytest.raises(RuntimeError, match="Measured UV-product segment is missing"):
        SegmentedRadRunner._segment_uv_product_fields(runner, rad, measured=True)

    rad.uv_products = {
        name: (
            Quantity(np.ones((1, 1, 1)), "1/(cm^2 s)")
            if name in {"F_CO_pdes_photon", "F_CO_pdes_photon_bands"}
            else chi
        )
        for name in UV_PRODUCT_MERGED_FIELD_NAMES
    }
    products = SegmentedRadRunner._segment_uv_product_fields(runner, rad, measured=True)
    assert set(products) == set(UV_PRODUCT_MERGED_FIELD_NAMES)
