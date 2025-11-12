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

    # Number of spatial dimensions represented by provided edges
    ndims: Optional[int] = None

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
        
        # Determine dimensionality from provided edges, if not explicitly set
        if self.ndims is None:
            count = 0
            if self.redge is not None:
                count += 1
            if self.pedge is not None:
                count += 1
            if self.tedge is not None:
                count += 1
            self.ndims = count

    @property
    def nrad(self) -> Optional[int]:
        return None if self.redge is None else int(self.redge.size - 1)

    @property
    def ncol(self) -> Optional[int]:
        return None if self.tedge is None else int(self.tedge.size - 1)

    @property
    def nsec(self) -> Optional[int]:
        return None if self.pedge is None else int(self.pedge.size - 1)

    # --- helpers for building 3D meshes from 2D inputs ---
    def with_vertical(self, zmed: np.ndarray, coord_system: CoordSystem = "cylindrical") -> "Mesh":
        """Return a new 3D mesh sharing r,phi, adding a vertical axis zmed.

        coord_system should typically be 'cylindrical' for (z,r,phi) workflows.
        """
        return Mesh(
            coord_system=coord_system,
            redge=self.redge,
            pedge=self.pedge,
            rmed=self.rmed,
            pmed=self.pmed,
            zmed=zmed,
            ndims=3,
        )

    def to_spherical_uniform(self, ncol: int) -> "Mesh":
        tedge = np.linspace(0.0, np.pi, int(ncol) + 1)
        return Mesh(coord_system="spherical", redge=self.redge, pedge=self.pedge, tedge=tedge, ndims=3)

    def to_spherical_by_scale_height(self, ncol: int, aspect_ratio: float, zmax_over_H: float = 5.0, full_disk: bool = True) -> "Mesh":
        ar = float(getattr(aspect_ratio, "magnitude", aspect_ratio))
        thmin = np.pi/2.0 - np.arctan(zmax_over_H * ar)
        thmax = np.pi/2.0
        if full_disk:
            upper = np.linspace(thmin, thmax, int(ncol)//2 + 1)
            lower = np.pi - upper[1:int(ncol)//2 + 1]
            tedge = np.concatenate([lower, upper])
        else:
            tedge = np.linspace(thmin, thmax, int(ncol) + 1)
        return Mesh(coord_system="spherical", redge=self.redge, pedge=self.pedge, tedge=tedge, ndims=3)

    def to_cylindrical_from_spherical(self, nver: int) -> "Mesh":
        if self.rmed is None or self.tmed is None:
            raise ValueError("rmed and tmed required to derive cylindrical zmed from spherical mesh")
        zbuf = -float(np.max(self.rmed)) * np.cos(self.tmed)
        zmed = np.linspace(np.min(zbuf), np.max(zbuf), int(nver))
        return Mesh(coord_system="cylindrical", redge=self.redge, pedge=self.pedge, rmed=self.rmed, pmed=self.pmed, zmed=zmed, ndims=3)
