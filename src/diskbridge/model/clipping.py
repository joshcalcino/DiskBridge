from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from diskbridge._units import Quantity

from .mesh import Axis, Mesh


@dataclass(frozen=True)
class ClipIndexer:
    axis_slices: Dict[str, slice]


def compute_clip_indexer(
    mesh: Mesh,
    bounds: Dict[str, Tuple[Optional[Quantity], Optional[Quantity]]],
) -> tuple[ClipIndexer, Dict[str, Axis]]:
    for axis_name in bounds:
        if axis_name not in mesh.axes:
            raise ValueError(
                f"Cannot clip axis '{axis_name}' for coord_system '{mesh.coord_system}'"
            )

    axis_slices: Dict[str, slice] = {}
    new_axes: Dict[str, Axis] = {}

    for axis_name, axis0 in mesh.axes.items():
        vmin, vmax = bounds.get(axis_name, (None, None))
        if vmin is None and vmax is None:
            new_axes[axis_name] = Axis(edges=axis0.edges, centers=axis0.centers)
            continue

        edges0 = mesh.edges(axis_name)
        centers0 = mesh.centers(axis_name)
        if edges0 is None or centers0 is None:
            raise ValueError(f"Mesh axis '{axis_name}' missing edges or centers")

        c_mag = np.asarray(centers0.magnitude, dtype=float)
        keep = np.ones_like(c_mag, dtype=bool)
        if vmin is not None:
            vmin_use = vmin.to(centers0.units)
            keep &= c_mag >= float(vmin_use.magnitude)
        if vmax is not None:
            vmax_use = vmax.to(centers0.units)
            keep &= c_mag <= float(vmax_use.magnitude)
        if not np.any(keep):
            raise ValueError(f"Clipping removed all cells along axis '{axis_name}'")

        idx = np.flatnonzero(keep)
        i0 = int(idx[0])
        i1 = int(idx[-1])
        axis_slices[axis_name] = slice(i0, i1 + 1)

        new_edges = edges0[i0 : i1 + 2]
        new_axes[axis_name] = Axis(edges=new_edges)

    return ClipIndexer(axis_slices=axis_slices), new_axes
