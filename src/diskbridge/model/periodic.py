# db-keywords: units, model, mesh, coordinates, arrays, plotting
# db-role: canonical
# db-scope: package
# db-purpose: Package module for units, model, mesh, coordinates.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diskbridge._units import Quantity


@dataclass(frozen=True)
class PeriodicAxisReindex:
    """Periodic-axis edge convention plus the matching cell reindex order."""

    edges: Quantity
    index_order: np.ndarray


def reindex_periodic_axis_to_start(
    edges: Quantity,
    *,
    start,
    period,
    atol: float = 1.0e-10,
) -> PeriodicAxisReindex:
    """Move a periodic axis seam to ``start`` and return matching cell order.

    This is intended for hydro readers whose native periodic coordinate starts
    at a different seam than DiskBridge's canonical convention. It only reorders
    whole cells; if ``start`` does not coincide with an existing edge, the reader
    must not silently split cells or invent interpolated data.
    """

    unit = edges.units
    raw = np.asarray(edges.to(unit).magnitude, dtype=np.float64)
    start_value = (
        float(start.to(unit).magnitude) if hasattr(start, "to") else float(start)
    )
    period_value = (
        float(period.to(unit).magnitude) if hasattr(period, "to") else float(period)
    )
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError("Periodic edges must be a one-dimensional edge array")
    if period_value <= 0.0:
        raise ValueError(f"period must be positive, got {period_value}")

    ncell = raw.size - 1
    widths = np.diff(raw)
    if np.any(widths <= 0.0):
        raise ValueError("Periodic edges must be strictly increasing")
    span = raw[-1] - raw[0]
    if not np.isclose(span, period_value, rtol=0.0, atol=atol):
        raise ValueError(
            f"Periodic axis spans {span}, expected period {period_value}"
        )

    # Map each edge into [start, start + period), while allowing the upper
    # closing edge to be represented exactly as start + period.
    rel = np.mod(raw - start_value, period_value)
    rel[np.isclose(rel, period_value, rtol=0.0, atol=atol)] = 0.0
    seam_candidates = np.flatnonzero(np.isclose(rel[:-1], 0.0, rtol=0.0, atol=atol))
    if seam_candidates.size == 0:
        raise ValueError(
            "Cannot reindex periodic axis: requested start does not coincide "
            "with an existing cell edge"
        )
    seam = int(seam_candidates[0])

    index_order = np.concatenate((
        np.arange(seam, ncell, dtype=np.int64),
        np.arange(0, seam, dtype=np.int64),
    ))
    reordered_widths = widths[index_order]
    new_edges = np.empty(ncell + 1, dtype=np.float64)
    new_edges[0] = start_value
    new_edges[1:] = start_value + np.cumsum(reordered_widths)
    if not np.isclose(new_edges[-1], start_value + period_value, rtol=0.0, atol=atol):
        raise ValueError("Reindexed periodic axis does not close at start + period")
    new_edges[-1] = start_value + period_value
    return PeriodicAxisReindex(
        edges=new_edges * unit,
        index_order=np.ascontiguousarray(index_order),
    )


__all__ = [
    "PeriodicAxisReindex",
    "reindex_periodic_axis_to_start",
]
