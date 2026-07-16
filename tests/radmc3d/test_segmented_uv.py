"""Regression checks for noise-aware segmented UV numerical contracts."""

# db-keywords: uv-products, validation, radmc3d, model, mesh, io
# db-role: validation
# db-scope: test
# db-purpose: Protect paired UV estimator, split, and inherited-source contracts.

from pathlib import Path

import numpy as np
import pytest

from diskbridge._units import Quantity
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
