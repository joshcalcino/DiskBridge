from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

CoordSystem = Literal["spherical", "cylindrical", "cartesian"]


def _centers_from_edges(edges: np.ndarray) -> np.ndarray:
    e = np.asarray(edges, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("edges must be 1D with at least 2 elements")
    return 0.5 * (e[:-1] + e[1:])


def _spherical_r_centers(edges: np.ndarray) -> np.ndarray:
    e = np.asarray(edges, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("edges must be 1D with at least 2 elements")
    num = 2.0 * (e[1:] ** 3 - e[:-1] ** 3)
    den = 3.0 * (e[1:] ** 2 - e[:-1] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmed = num / den
    bad = ~np.isfinite(rmed)
    if np.any(bad):
        rmed[bad] = _centers_from_edges(e)[bad]
    return rmed


@dataclass
class Mesh:
    """
    General mesh container independent of any hydro code.

    - coord_system: one of 'spherical', 'cylindrical', 'cartesian'.
    - Edges (redge, tedge, pedge) define cell interfaces along each dim.
    - Centers (rmed, tmed, pmed) are optional; if absent they are computed.
    - zmed is an optional auxiliary vertical axis for cylindrical workflows.
    """

    coord_system: CoordSystem

    redge: Optional[np.ndarray] = None
    tedge: Optional[np.ndarray] = None
    pedge: Optional[np.ndarray] = None

    rmed: Optional[np.ndarray] = None
    tmed: Optional[np.ndarray] = None
    pmed: Optional[np.ndarray] = None

    zmed: Optional[np.ndarray] = None

    ndims: Optinal[np.int] = None

    def __post_init__(self) -> None:
        # Compute centers if edges are provided and centers are missing
        if self.redge is not None and self.rmed is None:
            if self.coord_system == "spherical":
                self.rmed = _spherical_r_centers(self.redge)
            else:
                self.rmed = _centers_from_edges(self.redge)
        if self.tedge is not None and self.tmed is None:
            self.tmed = _centers_from_edges(self.tedge)
        if self.pedge is not None and self.pmed is None:
            self.pmed = _centers_from_edges(self.pedge)

        # Basic validation
        if self.redge is not None and self.rmed is not None:
            if self.rmed.size != self.redge.size - 1:
                raise ValueError("rmed must have len(redge)-1")
        if self.tedge is not None and self.tmed is not None:
            if self.tmed.size != self.tedge.size - 1:
                raise ValueError("tmed must have len(tedge)-1")
        if self.pedge is not None and self.pmed is not None:
            if self.pmed.size != self.pedge.size - 1:
                raise ValueError("pmed must have len(pedge)-1")
        

    @property
    def nrad(self) -> Optional[int]:
        return None if self.redge is None else int(self.redge.size - 1)

    @property
    def ncol(self) -> Optional[int]:
        return None if self.tedge is None else int(self.tedge.size - 1)

    @property
    def nsec(self) -> Optional[int]:
        return None if self.pedge is None else int(self.pedge.size - 1)
