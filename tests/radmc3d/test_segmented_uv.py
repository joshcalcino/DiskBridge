"""Regression checks for noise-aware segmented UV numerical contracts."""

# db-keywords: uv-products, validation, radmc3d, model, mesh, io
# db-role: validation
# db-scope: test
# db-purpose: Protect paired UV estimator, split, and inherited-source contracts.

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from diskbridge._units import Quantity
from diskbridge._params import DEFAULT_PARAMS_FILE
from diskbridge.model.core import Model
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.model.profiles import (
    find_noise_aware_split,
    paired_radial_uncertainty_metrics,
)
from diskbridge.radmc3d.data import (
    combine_mean_intensity_files,
    radial_shell_mean_intensity,
)
from diskbridge.radmc3d.cache import build_mesh_cache_context
from diskbridge._config import get_config
from diskbridge.radmc3d.model import (
    RADMC_PHOTON_COUNT_MAX,
    RadModel,
    _validate_radmc_photon_count,
)
from diskbridge.radmc3d.segmented import (
    SegmentedRadRunner,
    _calibrate_inherited_spectrum,
    _clip_parent_temperature_to_child,
    _matches_serialized_mcmono_grid,
    _validate_mean_intensity_binary,
    _validate_temperature_binary,
)
from diskbridge.radmc3d.uv_products import (
    default_isrf_path,
    uv_product_schema_hash,
    uv_product_specs_from_config,
)


def test_radmc_photon_count_requires_signed_int32_range() -> None:
    assert (
        _validate_radmc_photon_count(
            RADMC_PHOTON_COUNT_MAX,
            name="nphot_mono",
        )
        == RADMC_PHOTON_COUNT_MAX
    )
    with pytest.raises(ValueError, match="between 1 and 2147483647"):
        _validate_radmc_photon_count(0, name="nphot_mono")
    with pytest.raises(ValueError, match="between 1 and 2147483647"):
        _validate_radmc_photon_count(
            RADMC_PHOTON_COUNT_MAX + 1,
            name="nphot_mono",
        )


def _write_mean_intensity(
    path: Path,
    values: np.ndarray,
    *,
    frequencies: np.ndarray,
) -> None:
    values = np.asarray(values, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.asarray([2, 8, values.shape[1], values.shape[0]], dtype=np.int64).tofile(f)
        np.asarray(frequencies, dtype=np.float64).tofile(f)
        values.tofile(f)


def _read_payload(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int64, count=4)
        np.fromfile(f, dtype=np.float64, count=int(header[3]))
        return np.fromfile(f, dtype=np.float64).reshape((int(header[3]), int(header[2])))


def test_paired_mean_intensity_combines_packet_weighted_binary_payload(tmp_path: Path) -> None:
    frequencies = np.array([1.0e15, 2.0e15])
    first = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    second = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    first_path = tmp_path / "first.bout"
    second_path = tmp_path / "second.bout"
    output_path = tmp_path / "mean_intensity.bout"
    _write_mean_intensity(first_path, first, frequencies=frequencies)
    _write_mean_intensity(second_path, second, frequencies=frequencies)

    metadata = combine_mean_intensity_files(
        first_path,
        second_path,
        output_path,
        first_weight=0.25,
        chunk_values=2,
    )

    np.testing.assert_allclose(_read_payload(output_path), 0.25 * first + 0.75 * second)
    assert metadata["format"] == 2
    assert metadata["ncells"] == 3
    assert metadata["nwavelengths"] == 2


def test_radial_shell_mean_intensity_respects_radmc_fortran_cell_order(tmp_path: Path) -> None:
    frequencies = np.array([1.0e15, 2.0e15])
    payload = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    path = tmp_path / "mean_intensity.bout"
    _write_mean_intensity(path, payload, frequencies=frequencies)
    volumes = np.array([[[1.0, 3.0]], [[2.0, 4.0]]])

    actual_frequencies, shell_mean = radial_shell_mean_intensity(path, volumes)

    np.testing.assert_array_equal(actual_frequencies, frequencies)
    np.testing.assert_allclose(shell_mean[:, 0], [(1.0 + 9.0) / 4.0, (4.0 + 16.0) / 6.0])
    np.testing.assert_allclose(shell_mean[:, 1], [(10.0 + 90.0) / 4.0, (40.0 + 160.0) / 6.0])


def test_paired_radial_metrics_separate_shell_mean_and_cell_p99_noise() -> None:
    first = np.ones((2, 1, 2))
    second = first.copy()
    second[0, 0, 1] = 1.2
    volumes = np.ones_like(first)

    metrics = paired_radial_uncertainty_metrics(
        first,
        second,
        volumes,
        tolerance=0.05,
    )

    assert metrics["shell_fractional"][0] == pytest.approx(0.05 / 1.05)
    assert metrics["cell_fractional_p99"][0] == pytest.approx(0.1 / 1.1)
    assert metrics["failing_volume_fraction"][0] == pytest.approx(0.5)
    assert metrics["shell_fractional"][1] == 0.0


def test_paired_radial_metrics_keep_infinite_zero_signal_uncertainty() -> None:
    first = np.zeros((1, 1, 2))
    second = np.zeros_like(first)
    first[0, 0, 1] = -1.0
    second[0, 0, 1] = 1.0

    metrics = paired_radial_uncertainty_metrics(first, second, np.ones_like(first))

    assert np.isinf(metrics["cell_fractional_p99"][0])
    assert np.isinf(metrics["cell_fractional_max"][0])


def test_inherited_spectrum_calibration_matches_parent_shell() -> None:
    provisional = np.array([2.0, 4.0, 8.0])
    parent = np.array([3.0, 6.0, 12.0])
    child = np.array([2.4, 4.8, 9.6])

    correction, corrected = _calibrate_inherited_spectrum(
        provisional,
        parent,
        child,
    )

    np.testing.assert_allclose(correction, 1.25)
    np.testing.assert_allclose(corrected, provisional * 1.25)


def test_inherited_spectrum_calibration_rejects_unsampled_child() -> None:
    with pytest.raises(ValueError, match="Child comparison spectrum"):
        _calibrate_inherited_spectrum(
            np.ones(2),
            np.ones(2),
            np.array([1.0, 0.0]),
        )


def test_mcmono_grid_rounding_remains_within_parent_grid_check() -> None:
    requested = np.array([0.0912, 0.099879805285527, 0.172335512184473])
    written = np.array([float(f"{value:.6f}") for value in requested])

    assert _matches_serialized_mcmono_grid(written, requested)


def test_boundary_calibration_clips_parent_temperature_to_child_prefix() -> None:
    def spherical_model(r_edges: np.ndarray) -> Model:
        model = Model()
        model.mesh = Mesh.spherical(
            r=Axis(edges=Quantity(r_edges, "au")),
            theta=Axis(edges=Quantity(np.array([0.0, np.pi]), "radian")),
            phi=Axis(edges=Quantity(np.array([0.0, 2.0 * np.pi]), "radian")),
        )
        model.coord_system = "spherical"
        return model

    parent = spherical_model(np.array([1.0, 2.0, 3.0, 4.0]))
    child = spherical_model(np.array([1.0, 2.0, 3.0]))
    temperature = Quantity(np.array([[[10.0]], [[20.0]], [[30.0]]]), "K")

    clipped = _clip_parent_temperature_to_child(temperature, parent, child)

    np.testing.assert_array_equal(clipped.magnitude[:, 0, 0], [10.0, 20.0])


def test_noise_aware_split_returns_explicit_one_shell_join_indices() -> None:
    r_edges = np.arange(6, dtype=float) + 1.0
    reliable = np.array([False, False, True, True, True])
    stellar = np.full((3, 5), 0.001)

    r_split, info = find_noise_aware_split(
        r_edges,
        reliable,
        stellar,
        stellar_fraction_threshold=0.01,
        r_clip_min_au=1.0,
    )

    assert r_split == 4.0
    assert info["outermost_refinement_shell_idx"] == 1
    assert info["comparison_shell_idx"] == 2
    assert info["source_shell_idx"] == 3
    assert info["child_last_shell_idx"] == 2


def test_inherited_external_source_changes_only_uv_values(tmp_path: Path) -> None:
    wavelengths = np.array([0.05, 0.10, 0.20, 1.0, 1000.0])
    original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    output_dir = tmp_path / "radmc3d_inputs"
    output_dir.mkdir()

    configured_lines = ["2", "5"]
    configured_lines.extend(f"{value:13.6e}" for value in wavelengths)
    configured_lines.extend(f"{value:13.6e}" for value in original)
    configured_content = "\n".join(configured_lines) + "\n"

    class DummyWriter:
        def _external_source_content(self, model_dir: Path) -> tuple[Path, str]:
            return output_dir / "external_source.inp", configured_content

    rad = RadModel.__new__(RadModel)
    rad.model_dir = tmp_path
    rad.writer = DummyWriter()
    rad.ensure_inherited_uv_external_source(
        np.array([0.09, 0.10, 0.20, 0.21]),
        np.array([10.0, 20.0, 30.0, 40.0]),
        uv_min=Quantity(91.2, "nm"),
        uv_max=Quantity(206.7, "nm"),
    )

    tokens = (output_dir / "external_source.inp").read_text().split()
    updated = np.asarray(tokens[2 + wavelengths.size :], dtype=float)
    np.testing.assert_array_equal(updated[[0, 3, 4]], original[[0, 3, 4]])
    assert updated[1] == pytest.approx(20.0)
    assert updated[2] == pytest.approx(30.0)

    (output_dir / "external_source.inp").write_text("different spectrum\n")
    with pytest.raises(RuntimeError, match="Remove the segment directory"):
        rad.ensure_inherited_uv_external_source(
            np.array([0.09, 0.10, 0.20, 0.21]),
            np.array([10.0, 20.0, 30.0, 40.0]),
            uv_min=Quantity(91.2, "nm"),
            uv_max=Quantity(206.7, "nm"),
        )


def test_segment_setup_does_not_overwrite_existing_external_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_dir = tmp_path / "segment"
    inputs_dir = work_dir / "radmc3d_inputs"
    inputs_dir.mkdir(parents=True)
    source_path = inputs_dir / "external_source.inp"
    source_path.write_text("stale spectrum\n")

    class DummyWriter:
        def write_all_input_files(self, output_dir: Path) -> None:
            assert Path(output_dir) == work_dir

    class DummyRadModel:
        def __init__(self, model, model_dir: Path):
            self.params = type("Params", (), {"external_uv": True})()
            self.inputs_dir = inputs_dir
            self.writer = DummyWriter()

    monkeypatch.setattr("diskbridge.radmc3d.model.RadModel", DummyRadModel)
    monkeypatch.setattr(
        "diskbridge.radmc3d.utils.link_dustkappa_opacities",
        lambda source, target: None,
    )

    runner = SegmentedRadRunner(base_model=object(), base_model_dir=tmp_path / "base")
    runner._setup_segment(object(), work_dir, tmp_path / "opacities")

    assert source_path.read_text() == "stale spectrum\n"


def _write_temperature_binary(path: Path, *, ncells: int, nspec: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.asarray([1, 8, ncells, nspec], dtype=np.int64).tofile(handle)
        np.zeros(ncells * nspec, dtype=np.float64).tofile(handle)


def _completed_segment_fixture(tmp_path: Path) -> tuple[SegmentedRadRunner, dict, np.ndarray, list]:
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.array([1.0, 2.0, 3.0]), "au")),
        theta=Axis(edges=Quantity(np.array([0.0, np.pi]), "radian")),
        phi=Axis(edges=Quantity(np.array([0.0, 2.0 * np.pi]), "radian")),
    )
    model.coord_system = "spherical"
    model.dust = SimpleNamespace(nbin=2)

    base_dir = tmp_path / "run"
    segment_dir = base_dir / "segments" / "segment_00_full"
    temp_dir = segment_dir / "radmc3d_outputs" / "temperature"
    mono_dir = segment_dir / "radmc3d_outputs" / "mcmono"
    base_dir.mkdir(parents=True)
    params_text = Path(DEFAULT_PARAMS_FILE).read_text()
    (base_dir / "params.txt").write_text(params_text)
    temp_dir.mkdir(parents=True)
    mono_dir.mkdir(parents=True)
    (temp_dir / "params.txt").write_text(params_text)
    (mono_dir / "params.txt").write_text(params_text)

    wavelengths = np.array([0.0912, 0.11, 0.2067], dtype=np.float64)
    ncells = int(np.prod(model.mesh.shape))
    _write_temperature_binary(
        temp_dir / "dust_temperature.bdat",
        ncells=ncells,
        nspec=2,
    )
    _write_mean_intensity(
        mono_dir / "mean_intensity.bout",
        np.zeros((wavelengths.size, ncells)),
        frequencies=np.ones(wavelengths.size),
    )

    specs = uv_product_specs_from_config(
        get_config().get("radmc3d", {}).get("uv_products", {})
    )
    mesh_context = build_mesh_cache_context(model.mesh)
    temperature_context = {
        "nphot": 1000,
        "external_source_enabled": False,
        **mesh_context,
    }
    mono_context = {
        "nphot": 1000,
        "external_source_enabled": False,
        "wavelengths_size": int(wavelengths.size),
        "wavelengths_sha256": hashlib.sha256(wavelengths.tobytes()).hexdigest(),
        "uv_product_schema_sha256": uv_product_schema_hash(specs),
        "uv_product_reference_isrf_hash": hashlib.sha256(
            default_isrf_path().read_bytes()
        ).hexdigest(),
        "uv_products_enabled": True,
        "uv_product_mode": "disc_segment_only",
        **mesh_context,
    }
    (temp_dir / "cache_context.json").write_text(json.dumps(temperature_context))
    (mono_dir / "cache_context.json").write_text(json.dumps(mono_context))

    expected = {
        "mode": "noise_aware_inherited_uv",
        "mcmono_wavelength_source": "uv",
        "segmented_final_nphot_multiplier": 1.0,
        "segmented_tol": 0.01,
        "stellar_fraction_threshold": 0.01,
        "nphot_thermal_nominal": 1000,
        "nphot_mono_nominal": 1000,
        "nphot_thermal_final": 1000,
        "nphot_mono_final": 1000,
        "uv_product_mode": "all_segments_measured",
    }
    manifest = {
        **expected,
        "segments": [
            {
                "level": 0,
                "work_dir": "/relocated/cluster/run/segments/segment_00_full",
                "r_max_au": 3.0,
                "nphot_thermal": 1000,
                "nphot_mono": 1000,
                "is_final": True,
                "terminal_reason": "maximum split level reached",
            }
        ],
    }
    summary_path = base_dir / "segments" / "segmented_rt_summary.json"
    summary_path.write_text(json.dumps(manifest))
    return SegmentedRadRunner(model, base_dir), expected, wavelengths, specs


def test_completed_terminal_products_are_accepted_before_scout_budget(tmp_path: Path) -> None:
    runner, expected, wavelengths, specs = _completed_segment_fixture(tmp_path)

    compatible, reason = runner._validate_completed_segmented_rt(
        expected_summary=expected,
        wavelengths_um=wavelengths,
        product_specs=specs,
    )

    assert compatible, reason


def test_completed_terminal_products_reject_changed_final_budget(tmp_path: Path) -> None:
    runner, expected, wavelengths, specs = _completed_segment_fixture(tmp_path)
    expected["nphot_thermal_final"] = 2000

    compatible, reason = runner._validate_completed_segmented_rt(
        expected_summary=expected,
        wavelengths_um=wavelengths,
        product_specs=specs,
    )

    assert not compatible
    assert "nphot_thermal_final" in reason


def test_completed_terminal_products_reject_truncated_binary(tmp_path: Path) -> None:
    runner, expected, wavelengths, specs = _completed_segment_fixture(tmp_path)
    mean_path = (
        runner.base_model_dir
        / "segments"
        / "segment_00_full"
        / "radmc3d_outputs"
        / "mcmono"
        / "mean_intensity.bout"
    )
    mean_path.write_bytes(mean_path.read_bytes()[:-8])

    compatible, reason = runner._validate_completed_segmented_rt(
        expected_summary=expected,
        wavelengths_um=wavelengths,
        product_specs=specs,
    )

    assert not compatible
    assert "Incomplete mean-intensity payload" in reason


def test_completed_terminal_products_reject_changed_physical_input(tmp_path: Path) -> None:
    runner, expected, wavelengths, specs = _completed_segment_fixture(tmp_path)
    inputs_dir = runner.base_model_dir / "radmc3d_inputs"
    inputs_dir.mkdir()
    density_path = inputs_dir / "dust_density.binp"
    density_path.write_bytes(b"original density")
    summary_path = runner.base_model_dir / "segments" / "segmented_rt_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["rt_input_fingerprints"] = runner._rt_input_fingerprints()
    summary_path.write_text(json.dumps(summary))
    density_path.write_bytes(b"changed density")

    compatible, reason = runner._validate_completed_segmented_rt(
        expected_summary=expected,
        wavelengths_um=wavelengths,
        product_specs=specs,
    )

    assert not compatible
    assert reason == "continuum RT input fingerprints changed"


def test_force_is_the_only_way_to_bypass_completed_product_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import diskbridge

    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.array([1.0, 2.0]), "au")),
        theta=Axis(edges=Quantity(np.array([0.0, np.pi]), "radian")),
        phi=Axis(edges=Quantity(np.array([0.0, 2.0 * np.pi]), "radian")),
    )
    model.coord_system = "spherical"
    monkeypatch.setattr(
        diskbridge,
        "params",
        SimpleNamespace(
            segmented_tol=0.01,
            segmented_r_clip_min=Quantity(1.0, "au"),
            segmented_stop_factor=0.9,
            segmented_nphot_ratio=0.1,
            segmented_final_nphot_multiplier=1.0,
            segmented_max_splits=1,
            nphot_thermal=100,
            nphot_mono=100,
            uv_n_wavelengths=5,
        ),
    )
    runner = SegmentedRadRunner(model, tmp_path)
    monkeypatch.setattr(
        runner,
        "_validate_completed_segmented_rt",
        lambda **kwargs: (True, "compatible"),
    )
    monkeypatch.setattr(
        runner,
        "load_segmented_rt_outputs",
        lambda **kwargs: {"loaded_existing": True},
    )

    assert runner.run_segmented_rt(force=False) == {"loaded_existing": True}

    monkeypatch.setattr(
        "diskbridge.radmc3d.writer.RadWriter.compute_and_write_dust_opacities",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("recompute entered")),
    )
    monkeypatch.setattr(
        runner,
        "_validate_completed_segmented_rt",
        lambda **kwargs: (False, "incompatible"),
    )
    with pytest.raises(RuntimeError, match="recompute entered"):
        runner.run_segmented_rt(force=False)

    called = False

    def fail_if_validated(**kwargs):
        nonlocal called
        called = True
        return True, "compatible"

    monkeypatch.setattr(runner, "_validate_completed_segmented_rt", fail_if_validated)
    with pytest.raises(RuntimeError, match="recompute entered"):
        runner.run_segmented_rt(force=True)
    assert not called


def test_binary_restart_validation_checks_exact_payload_sizes(tmp_path: Path) -> None:
    temperature = tmp_path / "dust_temperature.bdat"
    mean = tmp_path / "mean_intensity.bout"
    _write_temperature_binary(temperature, ncells=2, nspec=3)
    _write_mean_intensity(
        mean,
        np.zeros((4, 2)),
        frequencies=np.ones(4),
    )

    _validate_temperature_binary(temperature, expected_ncells=2, expected_nspec=3)
    _validate_mean_intensity_binary(mean, expected_ncells=2, expected_nwavelengths=4)
