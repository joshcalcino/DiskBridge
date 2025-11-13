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


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    if c.ndim != 1 or c.size < 1:
        raise ValueError("centers must be 1D with at least 1 element")
    e = np.empty(c.size + 1, dtype=float)
    if c.size == 1:
        d = 1.0
        e[0] = c[0] - 0.5 * d
        e[1] = c[0] + 0.5 * d
        return e
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - (e[1] - c[0])
    e[-1] = c[-1] + (c[-1] - e[-2])
    return e


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

    # Cylindrical/Cartesian vertical edges and centers
    zedge: Optional[np.ndarray] = None
    zmed: Optional[np.ndarray] = None

    # Cartesian edges and centers
    xedge: Optional[np.ndarray] = None
    yedge: Optional[np.ndarray] = None
    xmed: Optional[np.ndarray] = None
    ymed: Optional[np.ndarray] = None

    # Number of spatial dimensions represented by provided edges
    ndims: Optional[int] = None

    def __post_init__(self) -> None:
        # Compute centers if edges are provided and centers are missing, per coord_system
        cs = self.coord_system
        if cs == "spherical":
            if self.redge is not None and self.rmed is None:
                self.rmed = _spherical_r_centers(self.redge)
            if self.tedge is not None and self.tmed is None:
                self.tmed = _centers_from_edges(self.tedge)
            if self.pedge is not None and self.pmed is None:
                self.pmed = _centers_from_edges(self.pedge)
        elif cs == "cylindrical":
            if self.redge is not None and self.rmed is None:
                self.rmed = _centers_from_edges(self.redge)
            if self.pedge is not None and self.pmed is None:
                self.pmed = _centers_from_edges(self.pedge)
            if self.zedge is not None and self.zmed is None:
                self.zmed = _centers_from_edges(self.zedge)
            # reject spherical/cartesian-only arguments
            if self.tedge is not None or self.tmed is not None:
                raise ValueError("theta edges/centers are invalid for cylindrical meshes")
            if any(x is not None for x in (self.xedge, self.yedge, self.xmed, self.ymed)):
                raise ValueError("x/y edges/centers are invalid for cylindrical meshes")
        elif cs == "cartesian":
            if self.xedge is not None and self.xmed is None:
                self.xmed = _centers_from_edges(self.xedge)
            if self.yedge is not None and self.ymed is None:
                self.ymed = _centers_from_edges(self.yedge)
            if self.zedge is not None and self.zmed is None:
                self.zmed = _centers_from_edges(self.zedge)
            # reject spherical/cylindrical-only arguments
            if any(x is not None for x in (self.redge, self.tedge, self.pedge, self.rmed, self.tmed, self.pmed)):
                raise ValueError("r/theta/phi edges/centers are invalid for cartesian meshes")
        else:  # spherical
            # reject cylindrical/cartesian-only arguments
            if any(x is not None for x in (self.zedge, self.zmed, self.xedge, self.yedge, self.xmed, self.ymed)):
                raise ValueError("z/x/y edges/centers are invalid for spherical meshes")

        # Basic validation per provided edges/centers
        if self.redge is not None and self.rmed is not None:
            if self.rmed.size != self.redge.size - 1:
                raise ValueError("rmed must have len(redge)-1")
        if self.tedge is not None and self.tmed is not None:
            if self.tmed.size != self.tedge.size - 1:
                raise ValueError("tmed must have len(tedge)-1")
        if self.pedge is not None and self.pmed is not None:
            if self.pmed.size != self.pedge.size - 1:
                raise ValueError("pmed must have len(pedge)-1")
        if self.zedge is not None and self.zmed is not None:
            if self.zmed.size != self.zedge.size - 1:
                raise ValueError("zmed must have len(zedge)-1")
        if self.xedge is not None and self.xmed is not None:
            if self.xmed.size != self.xedge.size - 1:
                raise ValueError("xmed must have len(xedge)-1")
        if self.yedge is not None and self.ymed is not None:
            if self.ymed.size != self.yedge.size - 1:
                raise ValueError("ymed must have len(yedge)-1")

        # Determine dimensionality if not explicitly set
        if self.ndims is None:
            if cs == "spherical":
                count = 0
                count += 1 if self.redge is not None else 0
                count += 1 if self.tedge is not None else 0
                count += 1 if self.pedge is not None else 0
                self.ndims = count
            elif cs == "cylindrical":
                count = 0
                count += 1 if self.redge is not None else 0
                count += 1 if self.pedge is not None else 0
                # vertical axis can be provided as zedge or zmed
                if self.zedge is not None or self.zmed is not None:
                    count += 1
                self.ndims = count
            elif cs == "cartesian":
                count = 0
                count += 1 if self.xedge is not None else 0
                count += 1 if self.yedge is not None else 0
                count += 1 if self.zedge is not None else 0
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

    def axes(self):
        if self.coord_system == "spherical":
            return ("r", "theta", "phi")
        if self.coord_system == "cylindrical":
            return ("z", "r", "phi")
        if self.coord_system == "cartesian":
            return ("x", "y", "z")
        return ()

    def _assert_axis(self, axis: str):
        axis = str(axis).lower()
        if axis not in self.axes():
            raise ValueError(f"axis '{axis}' is invalid for coord_system '{self.coord_system}'")

    def get_edges(self, axis: str):
        a = axis.lower()
        self._assert_axis(a)
        if self.coord_system == "spherical":
            if a == "r":
                return self.redge
            if a == "theta":
                return self.tedge
            if a == "phi":
                return self.pedge
        if self.coord_system == "cylindrical":
            if a == "z":
                return _edges_from_centers(self.zmed) if self.zmed is not None else None
            if a == "r":
                return self.redge
            if a == "phi":
                return self.pedge
        if self.coord_system == "cartesian":
            if a == "x":
                return getattr(self, "xedge", None)
            if a == "y":
                return getattr(self, "yedge", None)
            if a == "z":
                return getattr(self, "zedge", None)
        return None

    def get_centers(self, axis: str):
        a = axis.lower()
        self._assert_axis(a)
        if self.coord_system == "spherical":
            if a == "r":
                return self.rmed
            if a == "theta":
                return self.tmed
            if a == "phi":
                return self.pmed
        if self.coord_system == "cylindrical":
            if a == "z":
                return self.zmed
            if a == "r":
                return self.rmed
            if a == "phi":
                return self.pmed
        if self.coord_system == "cartesian":
            if a == "x":
                return getattr(self, "xmed", None)
            if a == "y":
                return getattr(self, "ymed", None)
            if a == "z":
                return getattr(self, "zmed", None)
        return None

    def __getattribute__(self, name: str):
        coord_attrs = {"redge","rmed","tedge","tmed","pedge","pmed","zedge","zmed","xedge","xmed","yedge","ymed"}
        if name in coord_attrs:
            cs = object.__getattribute__(self, "coord_system")
            allowed = set()
            if cs == "spherical":
                allowed = {"redge","rmed","tedge","tmed","pedge","pmed"}
            elif cs == "cylindrical":
                allowed = {"redge","rmed","pedge","pmed","zedge","zmed"}
            elif cs == "cartesian":
                allowed = {"xedge","xmed","yedge","ymed","zedge","zmed"}
            if name not in allowed:
                raise AttributeError(f"'{name}' not valid for coord_system '{cs}'")
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str):
        if name == "zedge":
            zmed = object.__getattribute__(self, "zmed")
            return None if zmed is None else _edges_from_centers(zmed)
        if name == "xedge":
            xmed = object.__getattribute__(self, "xmed") if hasattr(self, "xmed") else None
            return None if xmed is None else _edges_from_centers(xmed)
        if name == "yedge":
            ymed = object.__getattribute__(self, "ymed") if hasattr(self, "ymed") else None
            return None if ymed is None else _edges_from_centers(ymed)
        raise AttributeError(name)
