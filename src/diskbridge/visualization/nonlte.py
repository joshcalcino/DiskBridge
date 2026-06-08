"""Diagnostic plots for line-transfer and external non-LTE runs."""

# db-keywords: disk-mask, nonlte, line-transfer, units, radmc3d, field, visualization, io
# db-role: entrypoint
# db-scope: package
# db-purpose: Diagnostic plots for line-transfer and external non-LTE runs.

from __future__ import annotations

from pathlib import Path
import json
import warnings
from typing import Any

import numpy as np


def read_line_fits_cube(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read a RADMC-3D line-image FITS cube as ``(nchan, ny, nx)``.

    Returns the cube, velocity axis, and a compact metadata dictionary.
    """

    from astropy.io import fits

    path = Path(path)
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header

    squeezed = np.squeeze(data)
    if squeezed.ndim == 2:
        cube = squeezed[np.newaxis, ...]
    elif squeezed.ndim == 3:
        cube = squeezed
    else:
        raise ValueError(f"Cannot interpret {path} as a line cube; shape={data.shape}")

    nchan = cube.shape[0]
    crval = float(header.get("CRVAL3", 0.0))
    cdelt = float(header.get("CDELT3", 1.0))
    crpix = float(header.get("CRPIX3", 1.0))
    velocities = crval + (np.arange(nchan, dtype=float) + 1.0 - crpix) * cdelt
    meta = {
        "path": str(path),
        "shape": list(cube.shape),
        "bunit": header.get("BUNIT", ""),
        "restfrq_hz": float(header["RESTFRQ"]) if "RESTFRQ" in header else None,
        "velocity_unit": header.get("CUNIT3", "km/s"),
    }
    return cube, velocities, meta


def _safe_stem(value: str) -> str:
    return (
        str(value)
        .replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _finite_limits(values: np.ndarray, lower: float = 1.0, upper: float = 99.0):
    good = np.asarray(values, dtype=float)
    good = good[np.isfinite(good)]
    if good.size == 0:
        return None, None
    return float(np.nanpercentile(good, lower)), float(np.nanpercentile(good, upper))


def _format_colorbar_tick(value: float) -> str:
    if value == 0.0:
        return "0"
    abs_value = abs(float(value))
    if 1.0e-2 <= abs_value < 1.0e4:
        return f"{value:g}"
    return f"{value:.0e}"


def _log_colorbar_ticks(norm) -> tuple[list[float], list[str]]:
    if norm is None:
        return [], []
    vmin = float(norm.vmin)
    vmax = float(norm.vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0.0 or vmax <= vmin:
        return [], []
    emin = int(np.ceil(np.log10(vmin)))
    emax = int(np.floor(np.log10(vmax)))
    ticks = [10.0**exp for exp in range(emin, emax + 1)]
    if len(ticks) > 7:
        step = int(np.ceil(len(ticks) / 7))
        ticks = ticks[::step]
    return ticks, [_format_colorbar_tick(tick) for tick in ticks]


def _apply_log_colorbar_ticks(cbar, norm) -> None:
    ticks, labels = _log_colorbar_ticks(norm)
    if ticks:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)


def _apply_ratio_colorbar_ticks(cbar, norm) -> None:
    if norm is None:
        return
    ticks = [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3]
    ticks = [tick for tick in ticks if float(norm.vmin) <= tick <= float(norm.vmax)]
    if ticks:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([_format_colorbar_tick(tick) for tick in ticks])


def _positive_log_image(values: np.ndarray, *, lower: float = 0.1, upper: float = 100.0):
    """Return masked positive data and a log norm for high-dynamic-range line maps."""

    arr = np.asarray(values, dtype=float)
    return _mask_positive(arr), _positive_log_norm(arr, lower=lower, upper=upper)


def _mask_positive(values: np.ndarray):
    """Mask non-positive values before applying logarithmic image normalization."""

    arr = np.asarray(values, dtype=float)
    return np.ma.masked_where(~np.isfinite(arr) | (arr <= 0.0), arr)


def _positive_log_norm(values: np.ndarray, *, lower: float = 0.1, upper: float = 100.0):
    """Return one log norm for all supplied positive values."""

    from matplotlib.colors import LogNorm

    arr = np.asarray(values, dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0.0)]
    if positive.size == 0:
        return None
    vmin = float(np.nanpercentile(positive, lower))
    # Use the true maximum by default. Percentile clipping made bright channel
    # peaks look artificially saturated in non-LTE diagnostics.
    if upper >= 100.0:
        vmax = float(np.nanmax(positive))
    else:
        vmax = float(np.nanpercentile(positive, upper))
    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = float(np.nanmin(positive))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(positive))
    if vmax <= vmin:
        vmax = vmin * 10.0
    vmin = max(vmin, vmax * 1.0e-3)
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_channel_maps(
    fits_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    max_panels: int = 12,
    cmap: str = "inferno",
) -> Path:
    """Write a grid of representative channel maps."""

    import matplotlib.pyplot as plt

    cube, velocities, meta = read_line_fits_cube(fits_path)
    nchan = cube.shape[0]
    nshow = min(int(max_panels), nchan)
    indices = np.unique(np.linspace(0, nchan - 1, nshow, dtype=int))
    nshow = int(indices.size)
    ncols = min(4, nshow)
    nrows = int(np.ceil(nshow / ncols))
    _, log_norm = _positive_log_image(cube[indices], lower=0.1, upper=100.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    last_im = None
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    for ax, idx in zip(axes.flat, indices):
        image_data, _ = _positive_log_image(cube[idx], lower=0.1, upper=100.0)
        if log_norm is None:
            vmin, vmax = _finite_limits(cube[idx], 1.0, 99.5)
            last_im = ax.imshow(cube[idx], origin="lower", cmap=cmap_obj, vmin=vmin, vmax=vmax)
        else:
            # Channel maps span orders of magnitude; log scaling keeps faint line
            # structure visible without saturating the bright inner disk.
            last_im = ax.imshow(image_data, origin="lower", cmap=cmap_obj, norm=log_norm)
        ax.set_title(f"v={velocities[idx]:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[nshow:]:
        ax.axis("off")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.88)
        label = meta.get("bunit") or "intensity"
        if log_norm is not None:
            label = f"{label} (log color scale)"
        cbar.set_label(label)
        _apply_log_colorbar_ticks(cbar, log_norm)
    if title:
        fig.suptitle(title)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_moment_maps(
    fits_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    cmap: str = "magma",
) -> Path:
    """Write moment-0, peak-channel, and velocity-centroid maps."""

    import matplotlib.pyplot as plt

    cube, velocities, meta = read_line_fits_cube(fits_path)
    dv = float(np.nanmedian(np.diff(velocities))) if velocities.size > 1 else 1.0
    positive = np.where(np.isfinite(cube), cube, 0.0)
    moment0 = np.sum(positive, axis=0) * abs(dv)
    peak = np.nanmax(cube, axis=0)
    denom = np.sum(positive, axis=0)
    centroid = np.divide(
        np.sum(positive * velocities[:, None, None], axis=0),
        denom,
        out=np.full_like(denom, np.nan, dtype=float),
        where=denom > 0.0,
    )

    maps = [
        (moment0, f"moment 0 [{meta.get('bunit', '')} km/s]", cmap, True),
        (peak, f"peak [{meta.get('bunit', '')}]", cmap, True),
        (centroid, "centroid [km/s]", "RdBu_r", False),
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, (arr, label, cm, use_log) in zip(axes, maps):
        vmin, vmax = _finite_limits(arr, 1.0, 99.0)
        if use_log:
            image_data, norm = _positive_log_image(arr, lower=0.1, upper=100.0)
            cmap_obj = plt.get_cmap(cm).copy()
            cmap_obj.set_bad("black")
            im = ax.imshow(image_data, origin="lower", cmap=cmap_obj, norm=norm)
            label = f"{label} (log color scale)"
        elif "centroid" in label:
            vmax_abs = np.nanmax(np.abs([vmin or 0.0, vmax or 0.0]))
            vmin, vmax = -vmax_abs, vmax_abs
            im = ax.imshow(arr, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(arr, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, shrink=0.84)
        if use_log:
            _apply_log_colorbar_ticks(cbar, norm)
    if title:
        fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_integrated_spectrum(
    fits_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    label: str | None = None,
) -> Path:
    """Write the spatially integrated line profile."""

    import matplotlib.pyplot as plt

    cube, velocities, meta = read_line_fits_cube(fits_path)
    spectrum = np.nansum(cube, axis=(1, 2))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(velocities, spectrum, label=label or Path(fits_path).parent.parent.name)
    ax.set_xlabel("velocity [km/s]")
    ax.set_ylabel(f"summed intensity [{meta.get('bunit', '')}]")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_solver_history(manifest_path: str | Path, output_path: str | Path) -> Path | None:
    """Plot convergence and beta ranges from an external-levelpop manifest."""

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text())
    history = payload.get("iteration_history", [])
    if not history:
        return None

    import matplotlib.pyplot as plt

    iteration = np.asarray([row["iteration"] for row in history], dtype=float)
    err = np.asarray([row["err"] for row in history], dtype=float)
    beta_min = np.asarray([row["beta_min"] for row in history], dtype=float)
    beta_max = np.asarray([row["beta_max"] for row in history], dtype=float)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
    axes[0].semilogy(iteration, np.maximum(err, 1e-300), marker="o")
    axes[0].axhline(float(payload.get("convcrit", 0.0)), color="k", alpha=0.45, lw=0.9)
    axes[0].set_ylabel("population change")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].semilogy(iteration, beta_min, marker="o", label="beta min")
    axes[1].semilogy(iteration, beta_max, marker="s", label="beta max")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("escape probability")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()
    fig.suptitle(f"{payload.get('species', '')} external non-LTE solver")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_line_comparison(
    target_fits: str | Path,
    reference_fits: str | Path,
    output_path: str | Path,
    *,
    target_label: str = "external non-LTE",
    reference_label: str = "reference",
) -> Path:
    """Compare spectra and moment-0 maps for two line cubes."""

    import matplotlib.pyplot as plt

    target_cube, target_v, target_meta = read_line_fits_cube(target_fits)
    ref_cube, ref_v, _ = read_line_fits_cube(reference_fits)
    if target_cube.shape != ref_cube.shape:
        raise ValueError(
            f"Cannot compare cubes with different shapes: "
            f"{target_cube.shape} vs {ref_cube.shape}"
        )
    dv = float(np.nanmedian(np.diff(target_v))) if target_v.size > 1 else 1.0
    target_m0 = np.nansum(target_cube, axis=0) * abs(dv)
    ref_m0 = np.nansum(ref_cube, axis=0) * abs(dv)
    ratio = np.divide(
        target_m0,
        ref_m0,
        out=np.full_like(target_m0, np.nan, dtype=float),
        where=ref_m0 != 0.0,
    )
    target_spec = np.nansum(target_cube, axis=(1, 2))
    ref_spec = np.nansum(ref_cube, axis=(1, 2))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    axes[0, 0].plot(target_v, target_spec, label=target_label)
    axes[0, 0].plot(ref_v, ref_spec, label=reference_label, ls="--")
    axes[0, 0].set_xlabel("velocity [km/s]")
    axes[0, 0].set_ylabel(f"summed intensity [{target_meta.get('bunit', '')}]")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    # Compare both methods on exactly the same color scale; otherwise small
    # differences in independent autoscaling can masquerade as morphology.
    shared_moment_norm = _positive_log_norm(
        np.stack([target_m0, ref_m0], axis=0),
        lower=0.1,
        upper=100.0,
    )
    for ax, arr, label in (
        (axes[0, 1], target_m0, f"{target_label} moment 0"),
        (axes[1, 0], ref_m0, f"{reference_label} moment 0"),
    ):
        image_data = _mask_positive(arr)
        cmap_obj = plt.get_cmap("magma").copy()
        cmap_obj.set_bad("black")
        im = ax.imshow(image_data, origin="lower", cmap=cmap_obj, norm=shared_moment_norm)
        ax.set_title(f"{label} (log color scale)")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, shrink=0.84)
        _apply_log_colorbar_ticks(cbar, shared_moment_norm)

    finite = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if finite.size:
        from matplotlib.colors import LogNorm

        low = max(float(np.nanpercentile(finite, 1.0)), 1.0e-30)
        high = float(np.nanpercentile(finite, 99.0))
        limit = min(max(high, 1.0 / low, 1.1), 1.0e3)
        ratio_norm = LogNorm(vmin=1.0 / limit, vmax=limit)
        ratio_image = np.ma.masked_where(~np.isfinite(ratio) | (ratio <= 0.0), ratio)
    else:
        ratio_norm = None
        ratio_image = ratio
    im = axes[1, 1].imshow(ratio_image, origin="lower", cmap="RdBu_r", norm=ratio_norm)
    axes[1, 1].set_title(f"{target_label}/{reference_label} moment 0")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    cbar = fig.colorbar(im, ax=axes[1, 1], shrink=0.84)
    _apply_ratio_colorbar_ticks(cbar, ratio_norm)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_external_nonlte_diagnostic_plots(
    *,
    image_fits: str | Path,
    output_dir: str | Path,
    label: str,
    solver_manifest: str | Path | None = None,
    reference_fits: dict[str, str | Path] | None = None,
) -> list[Path]:
    """Create a standard diagnostic plot set for an external non-LTE image."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_stem(label)
    outputs = [
        plot_channel_maps(
            image_fits,
            output_dir / f"{safe}_channel_maps.png",
            title=f"{label} channel maps",
        ),
        plot_moment_maps(
            image_fits,
            output_dir / f"{safe}_moment_maps.png",
            title=f"{label} moments",
        ),
        plot_integrated_spectrum(
            image_fits,
            output_dir / f"{safe}_spectrum.png",
            title=f"{label} integrated spectrum",
            label=label,
        ),
    ]
    if solver_manifest is not None:
        history_plot = plot_solver_history(
            solver_manifest,
            output_dir / f"{safe}_solver_history.png",
        )
        if history_plot is not None:
            outputs.append(history_plot)

    for ref_label, ref_path in (reference_fits or {}).items():
        ref_path = Path(ref_path)
        if not ref_path.exists():
            continue
        try:
            outputs.append(
                plot_line_comparison(
                    image_fits,
                    ref_path,
                    output_dir / f"{safe}_compare_{_safe_stem(ref_label)}.png",
                    target_label=label,
                    reference_label=ref_label,
                )
            )
        except ValueError as exc:
            warnings.warn(f"Skipping comparison with {ref_label}: {exc}", stacklevel=2)
    return outputs
