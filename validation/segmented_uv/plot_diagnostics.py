#!/usr/bin/env python3
"""Generate diagnostics from saved full-domain and segmented UV runs."""

# db-keywords: uv-products, validation, radmc3d, visualization, plotting
# db-role: validation
# db-scope: validation
# db-purpose: Compare saved segmented UV fields and joins against the full-domain reference.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import diskbridge
from diskbridge._config import get_config
from diskbridge._units import Quantity
from diskbridge.model.core import Model, SubModel
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.model.profiles import (
    compute_cell_volumes,
    compute_volume_weighted_mean_radial_profile,
    weighted_quantile,
)
from diskbridge.model.utils import field_data_as_order
from diskbridge.radmc3d import RadModel
from diskbridge.radmc3d.uv_products import uv_product_specs_from_config


def _read_amr_grid_spherical(amr_grid_path: Path) -> Mesh:
    text = amr_grid_path.read_text().strip().splitlines()
    if len(text) < 7:
        raise ValueError(f"Invalid amr_grid.inp (too few lines): {amr_grid_path}")

    try:
        coordsys_code = int(text[2].strip())
    except Exception as exc:
        raise ValueError(
            f"Failed to parse coordsys from {amr_grid_path}: {exc}"
        ) from exc

    if coordsys_code < 100 or coordsys_code >= 200:
        raise ValueError(
            f"Only spherical grids supported (coordsys={coordsys_code}) in "
            f"{amr_grid_path}"
        )

    dims = [int(x) for x in text[5].split()]
    if len(dims) != 3:
        raise ValueError(f"Invalid dims line in {amr_grid_path}: '{text[5]}'")

    edge_lines = text[6:9]
    if len(edge_lines) != 3:
        raise ValueError(f"Invalid amr_grid.inp (missing edge lines): {amr_grid_path}")

    edges_raw: list[np.ndarray] = []
    for ln in edge_lines:
        vals = [v for v in ln.strip().split() if v]
        edges_raw.append(np.asarray([float(v) for v in vals], dtype=float))

    expected = [dims[0] + 1, dims[1] + 1, dims[2] + 1]
    for i, (arr, exp) in enumerate(zip(edges_raw, expected)):
        if arr.size != exp:
            raise ValueError(
                f"Invalid amr_grid.inp edge count axis={i}: got {arr.size}, "
                f"expected {exp} ({amr_grid_path})"
            )

    r_edges = Quantity(edges_raw[0], "cm")
    theta_edges = Quantity(edges_raw[1], "radian")
    phi_edges = Quantity(edges_raw[2], "radian")

    return Mesh.spherical(
        r=Axis(edges=r_edges),
        theta=Axis(edges=theta_edges),
        phi=Axis(edges=phi_edges),
    )


def _build_model_from_amr_grid(amr_grid_path: Path) -> Model:
    model = Model()
    model.coord_system = "spherical"
    model.mesh = _read_amr_grid_spherical(amr_grid_path)
    model.gas = SubModel(model)
    return model


def _load_temperature(rad: RadModel, path: Path) -> None:
    rad.read_dust_temperature(fname=str(path))


def _load_chi(
    rad: RadModel,
    *,
    mean_intensity_path: Path,
    uv_min: Quantity,
    uv_max: Quantity,
) -> None:
    rad._postprocess_chi(
        mean_intensity_path,
        uv_min,
        uv_max,
        compute_products=True,
    )


def _field_data(model: Model, field_name: str) -> np.ndarray:
    field = field_data_as_order(model.gas[field_name], ("r", "theta", "phi"))
    return np.asarray(field.to_base_units().magnitude)


def write_uv_product_comparison(
    *,
    baseline_model: Model,
    segmented_model: Model,
    output_dir: Path,
) -> dict[str, dict[str, float | int]]:
    """Compare every configured scalar UV product between validation models."""
    import matplotlib.pyplot as plt

    product_specs = uv_product_specs_from_config(
        get_config().get("radmc3d", {}).get("uv_products", {})
    )
    product_names = tuple(spec.field_name for spec in product_specs)
    for model, label in (
        (baseline_model, "baseline"),
        (segmented_model, "segmented"),
    ):
        missing = [name for name in product_names if name not in model.gas]
        if missing:
            raise RuntimeError(
                f"{label} model is missing configured UV products: {', '.join(missing)}"
            )

    volumes = compute_cell_volumes(baseline_model)
    if volumes.shape != tuple(baseline_model.mesh.shape):
        raise ValueError(
            f"cell-volume shape {volumes.shape} does not match mesh {baseline_model.mesh.shape}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float | int]] = {}
    for field_name in product_names:
        r_base, profile_base = compute_volume_weighted_mean_radial_profile(
            baseline_model,
            field_name,
        )
        r_segmented, profile_segmented = compute_volume_weighted_mean_radial_profile(
            segmented_model,
            field_name,
        )
        if r_base.shape != r_segmented.shape or not np.allclose(r_base, r_segmented):
            raise RuntimeError(
                f"baseline and segmented radial grids differ for {field_name}"
            )

        baseline = _field_data(baseline_model, field_name)
        segmented = _field_data(segmented_model, field_name)
        if baseline.shape != segmented.shape or baseline.shape != volumes.shape:
            raise ValueError(
                f"UV-product shape mismatch for {field_name}: "
                f"baseline={baseline.shape}, segmented={segmented.shape}, volumes={volumes.shape}"
            )

        valid = (
            np.isfinite(baseline)
            & np.isfinite(segmented)
            & (baseline > 0.0)
        )
        fractional = np.full_like(baseline, np.nan, dtype=np.float64)
        fractional[valid] = (segmented[valid] - baseline[valid]) / baseline[valid]
        absolute_fractional = np.abs(fractional[valid])
        valid_weights = np.asarray(volumes[valid], dtype=np.float64)
        total_volume = float(np.sum(volumes))
        valid_volume = float(np.sum(valid_weights))

        if valid_weights.size:
            rms = float(
                np.sqrt(
                    np.sum(valid_weights * fractional[valid] ** 2)
                    / valid_volume
                )
            )
            median_absolute = weighted_quantile(
                absolute_fractional,
                valid_weights,
                0.5,
            )
            p99_absolute = weighted_quantile(
                absolute_fractional,
                valid_weights,
                0.99,
            )
            maximum_absolute = float(np.max(absolute_fractional))
        else:
            rms = float("nan")
            median_absolute = float("nan")
            p99_absolute = float("nan")
            maximum_absolute = float("nan")

        metrics[field_name] = {
            "valid_cell_count": int(np.count_nonzero(valid)),
            "total_cell_count": int(valid.size),
            "valid_volume_fraction": (
                float(valid_volume / total_volume) if total_volume > 0.0 else float("nan")
            ),
            "volume_weighted_rms_fractional": rms,
            "volume_weighted_median_absolute_fractional": median_absolute,
            "volume_weighted_p99_absolute_fractional": p99_absolute,
            "maximum_absolute_fractional": maximum_absolute,
        }

        radial_fractional = np.full_like(profile_base, np.nan, dtype=np.float64)
        radial_valid = (
            np.isfinite(profile_base)
            & np.isfinite(profile_segmented)
            & (profile_base > 0.0)
        )
        radial_fractional[radial_valid] = (
            profile_segmented[radial_valid] - profile_base[radial_valid]
        ) / profile_base[radial_valid]

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8.0, 6.5),
            sharex=True,
            constrained_layout=True,
        )
        axes[0].plot(
            r_base,
            np.where(profile_base > 0.0, profile_base, np.nan),
            label="full domain",
        )
        axes[0].plot(
            r_segmented,
            np.where(profile_segmented > 0.0, profile_segmented, np.nan),
            label="segmented",
        )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Normalized UV field")
        axes[0].set_title(field_name)
        axes[0].legend()

        axes[1].plot(r_base, radial_fractional, color="tab:blue")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].axhline(0.01, color="0.45", linestyle="--", linewidth=0.8)
        axes[1].axhline(-0.01, color="0.45", linestyle="--", linewidth=0.8)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("r [au]")
        axes[1].set_ylabel("(segmented - full) / full")
        fig.savefig(output_dir / f"{field_name}.png", dpi=200)
        plt.close(fig)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def write_radial_comparison(
    *,
    baseline_model: Model,
    segmented_model: Model,
    output_path: Path,
    split_radii_au: list[float] | tuple[float, ...] = (),
) -> None:
    """Plot radial dust-temperature and UV-field validation profiles."""
    import matplotlib.pyplot as plt

    r_base, temperature_base = compute_volume_weighted_mean_radial_profile(
        baseline_model,
        "dust_temperature",
    )
    r_segmented, temperature_segmented = compute_volume_weighted_mean_radial_profile(
        segmented_model,
        "dust_temperature",
    )
    _, chi_base = compute_volume_weighted_mean_radial_profile(
        baseline_model,
        "chi",
    )
    _, chi_segmented = compute_volume_weighted_mean_radial_profile(
        segmented_model,
        "chi",
    )
    if r_base.shape != r_segmented.shape or not np.allclose(r_base, r_segmented):
        raise RuntimeError("baseline and segmented radial grids differ")

    temperature_fractional = (
        temperature_segmented - temperature_base
    ) / temperature_base
    chi_fractional = np.full_like(chi_base, np.nan, dtype=float)
    valid_chi = np.isfinite(chi_base) & (chi_base != 0.0)
    chi_fractional[valid_chi] = (
        chi_segmented[valid_chi] - chi_base[valid_chi]
    ) / chi_base[valid_chi]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(r_base, temperature_base, label="full domain")
    axes[0, 0].plot(r_segmented, temperature_segmented, label="segmented")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("T_mean [K]")
    axes[0, 0].legend()

    axes[0, 1].plot(r_base, chi_base, label="full domain")
    axes[0, 1].plot(r_segmented, chi_segmented, label="segmented")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("chi_mean [Draine]")
    axes[0, 1].legend()

    axes[1, 0].plot(r_base, temperature_fractional)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("r [au]")
    axes[1, 0].set_ylabel("(segmented - full) / full")

    axes[1, 1].plot(r_base, chi_fractional)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("r [au]")
    axes[1, 1].set_ylabel("(segmented - full) / full")

    for split_radius_au in split_radii_au:
        for axis in axes.flat:
            axis.axvline(
                float(split_radius_au),
                color="black",
                linestyle="--",
                linewidth=0.8,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    workdir = Path(args.workdir).resolve() if args.workdir is not None else script_dir
    baseline_dir = workdir / "baseline_run"
    segmented_dir = workdir / "segmented_run"
    plots_dir = workdir / "plots"

    params_path = workdir / "params.txt"
    p = diskbridge.read_params(str(params_path))
    diskbridge.params = p

    baseline_grid = baseline_dir / "radmc3d_inputs" / "amr_grid.inp"
    if not baseline_grid.exists():
        raise FileNotFoundError(f"Missing baseline grid: {baseline_grid}")
    baseline_model = _build_model_from_amr_grid(baseline_grid)
    rad_base = RadModel(baseline_model, model_dir=baseline_dir)

    baseline_temperature = (
        baseline_dir
        / "radmc3d_outputs"
        / "temperature"
        / "dust_temperature.bdat"
    )
    if not baseline_temperature.exists():
        raise FileNotFoundError(
            f"Missing baseline dust temperature: {baseline_temperature}"
        )
    _load_temperature(rad_base, baseline_temperature)

    baseline_mean_intensity = (
        baseline_dir
        / "radmc3d_outputs"
        / "mcmono"
        / "mean_intensity.bout"
    )
    if not baseline_mean_intensity.exists():
        raise FileNotFoundError(
            f"Missing baseline mean intensity: {baseline_mean_intensity}"
        )
    _load_chi(
        rad_base,
        mean_intensity_path=baseline_mean_intensity,
        uv_min=p.uv_min,
        uv_max=p.uv_max,
    )

    segmented_grid = (
        segmented_dir
        / "segments"
        / "segment_00_full"
        / "radmc3d_inputs"
        / "amr_grid.inp"
    )
    if not segmented_grid.exists():
        raise FileNotFoundError(f"Missing full segmented grid: {segmented_grid}")
    segmented_model = _build_model_from_amr_grid(segmented_grid)
    rad_segmented = RadModel(segmented_model, model_dir=segmented_dir)
    segmented_result = rad_segmented.load_segmented_rt_outputs(
        diagnostic_plots=True,
        plots_dir=plots_dir / "segmented_rt",
    )

    write_uv_product_comparison(
        baseline_model=baseline_model,
        segmented_model=segmented_model,
        output_dir=plots_dir / "uv_products",
    )
    write_radial_comparison(
        baseline_model=baseline_model,
        segmented_model=segmented_model,
        output_path=plots_dir / "compare_radial_profiles.png",
        split_radii_au=list(segmented_result.get("split_radii_au", [])),
    )

    print(f"wrote: {plots_dir / 'compare_radial_profiles.png'}")
    print(f"wrote: {plots_dir / 'uv_products'}")
    print(f"wrote: {plots_dir / 'segmented_rt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
