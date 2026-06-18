"""Scale and color-limit helpers for scientific plots."""

# db-keywords: visualization, plotting, scales, diagnostics
# db-role: helper
# db-scope: package
# db-purpose: Shared plotting scale helpers for scientific diagnostics.

from __future__ import annotations

import numpy as np


DEFAULT_LOG10_FLOOR: float = 1.0e-30
DEFAULT_LOG10_MAX_DECADES: float = 8.0


def log10_display_values(
    values: np.ndarray,
    *,
    floor: float = DEFAULT_LOG10_FLOOR,
) -> np.ndarray:
    """Return ``log10`` values clipped to a plotting floor.

    Parameters
    ----------
    values : ndarray
        Linear values to display on a log10 scale.
    floor : float, optional
        Smallest positive value used for display clipping. This is a plotting
        floor only; it does not alter source data.

    Returns
    -------
    ndarray
        ``log10(max(values, floor))`` for finite numeric values.
    """

    floor = float(floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be finite and > 0")
    return np.log10(np.maximum(np.asarray(values, dtype=float), floor))


def log10_display_limits(
    *arrays: np.ndarray,
    floor: float = DEFAULT_LOG10_FLOOR,
    lower_percentile: float = 1.0,
    upper_percentile: float = 100.0,
    max_decades: float = DEFAULT_LOG10_MAX_DECADES,
) -> tuple[float, float]:
    """Return robust log10 color limits for positive scientific fields.

    The lower limit never follows numerical zeros down to machine precision.
    It is bounded by ``floor`` and by ``max_decades`` below the upper display
    limit.

    Parameters
    ----------
    *arrays : ndarray
        Linear-value arrays that should share a log10 display scale.
    floor : float, optional
        Positive plotting floor.
    lower_percentile, upper_percentile : float, optional
        Percentiles of finite positive values used for the initial limits.
    max_decades : float, optional
        Maximum displayed dynamic range in dex.

    Returns
    -------
    tuple of float
        ``(vmin, vmax)`` in log10 units.
    """

    floor = float(floor)
    max_decades = float(max_decades)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be finite and > 0")
    if not np.isfinite(max_decades) or max_decades <= 0.0:
        raise ValueError("max_decades must be finite and > 0")
    if not 0.0 <= float(lower_percentile) <= 100.0:
        raise ValueError("lower_percentile must be in [0, 100]")
    if not 0.0 <= float(upper_percentile) <= 100.0:
        raise ValueError("upper_percentile must be in [0, 100]")
    if float(lower_percentile) > float(upper_percentile):
        raise ValueError("lower_percentile must be <= upper_percentile")

    finite_positive: list[np.ndarray] = []
    for array in arrays:
        arr = np.asarray(array, dtype=float)
        vals = arr[np.isfinite(arr) & (arr > 0.0)]
        if vals.size:
            finite_positive.append(vals)

    if not finite_positive:
        vmax = 0.0
        return float(max(np.log10(floor), vmax - max_decades)), float(vmax)

    vals = np.concatenate(finite_positive)
    logs = np.log10(np.maximum(vals, floor))
    finite_logs = logs[np.isfinite(logs)]
    if finite_logs.size == 0:
        vmax = 0.0
        return float(max(np.log10(floor), vmax - max_decades)), float(vmax)

    vmax = float(np.nanpercentile(finite_logs, float(upper_percentile)))
    vmin_percentile = float(np.nanpercentile(finite_logs, float(lower_percentile)))
    vmin = max(vmin_percentile, vmax - max_decades, float(np.log10(floor)))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmax = 0.0
        vmin = max(float(np.log10(floor)), vmax - max_decades)
    if vmin >= vmax:
        vmax = vmin + 1.0

    return float(vmin), float(vmax)
