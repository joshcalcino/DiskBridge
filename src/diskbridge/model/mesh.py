from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
import numpy as np
from .._units import units, Quantity

CoordSystem = Literal["spherical", "cylindrical", "cartesian"]


# ---------- utilities ----------
def _centers_from_edges(edges: Quantity) -> Quantity:
    e = np.asarray(edges.magnitude, dtype=float) * edges.units
    if e.ndim != 1 or e.size < 2:
        raise ValueError("edges must be 1D with at least 2 elements")
    return 0.5 * (e[:-1] + e[1:])


def _spherical_r_centers(edges: Quantity) -> Quantity:
    e = np.asarray(edges.magnitude, dtype=float) * edges.units
    if e.ndim != 1 or e.size < 2:
        raise ValueError("edges must be 1D with at least 2 elements")
    num = 2.0 * (e[1:] ** 3 - e[:-1] ** 3)
    den = 3.0 * (e[1:] ** 2 - e[:-1] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmed = num / den
    bad = ~np.isfinite(np.asarray(rmed.magnitude))
    if np.any(bad):
        rmed[bad] = _centers_from_edges(e)[bad]
    return rmed


def _edges_from_centers(centers: Quantity) -> Quantity:
    c_mag = np.asarray(centers.magnitude, dtype=float)
    if c_mag.ndim != 1 or c_mag.size < 1:
        raise ValueError("centers must be 1D with at least 1 element")
    e_mag = np.empty(c_mag.size + 1, dtype=float)
    if c_mag.size == 1:
        d = 1.0
        e_mag[0] = c_mag[0] - 0.5 * d
        e_mag[1] = c_mag[0] + 0.5 * d
        return e_mag * centers.units
    e_mag[1:-1] = 0.5 * (c_mag[:-1] + c_mag[1:])
    e_mag[0] = c_mag[0] - (e_mag[1] - c_mag[0])
    e_mag[-1] = c_mag[-1] + (c_mag[-1] - e_mag[-2])
    return e_mag * centers.units


def _is_strictly_increasing(q: Quantity) -> bool:
    a = np.asarray(q.magnitude, dtype=float)
    return np.all(a[1:] > a[:-1])


# ---------- small containers ----------
@dataclass(frozen=True)
class Axis:
    edges: Optional[Quantity] = None
    centers: Optional[Quantity] = None


def _allowed_axes(cs: CoordSystem) -> Tuple[str, ...]:
    if cs == "spherical":
        return ("r", "theta", "phi")
    if cs == "cylindrical":
        return ("r", "phi", "z")
    if cs == "cartesian":
        return ("x", "y", "z")


def _display_order(cs: CoordSystem) -> Tuple[str, ...]:
    # purely cosmetic order for reporting/iteration
    if cs == "spherical":
        return ("r", "theta", "phi")
    if cs == "cylindrical":
        return ("z", "r", "phi")  # common (z, r, phi) workflow
    if cs == "cartesian":
        return ("x", "y", "z")


# ---------- main Mesh ----------
@dataclass(frozen=True, repr=False)
class Mesh:
    coord_system: CoordSystem
    axes: Dict[str, Axis]
    ndims: int = 0

    def __post_init__(self) -> None:
        allowed = set(_allowed_axes(self.coord_system))
        new_axes: Dict[str, Axis] = {}

        for name, ax in self.axes.items():
            if name not in allowed:
                continue

            edges, centers = ax.edges, ax.centers

            if centers is None and edges is not None:
                centers = (_spherical_r_centers(edges)
                           if (self.coord_system == "spherical" and name == "r")
                           else _centers_from_edges(edges))
            elif edges is None and centers is not None:
                edges = _edges_from_centers(centers)

            # Validate lengths
            if edges is not None and centers is not None:
                if centers.size != edges.size - 1:
                    raise ValueError(
                        f"{name}: centers must have len(edges)-1 "
                        f"({centers.size} vs {edges.size - 1})"
                    )

            # # Validate monotonicity of edges
            # if edges is not None and not _is_strictly_increasing(edges):
            #     print(edges)
            #     raise ValueError(f"{name}: edges must be strictly increasing")

            new_axes[name] = Axis(edges=edges, centers=centers)

        object.__setattr__(self, "axes", new_axes)
        object.__setattr__(
            self,
            "ndims",
            sum(1 for a in new_axes.values() if a.edges is not None or a.centers is not None),
        )

    # ---------- canonical, explicit API ----------
    def axis_names(self) -> Tuple[str, ...]:
        return _display_order(self.coord_system)

    def axis(self, name: str) -> Axis:
        k = name.lower()
        try:
            return self.axes[k]
        except KeyError:
            raise ValueError(f"axis '{name}' invalid for coord_system '{self.coord_system}'")

    def edges(self, name: str) -> Optional[Quantity]:
        return self.axis(name).edges

    def centers(self, name: str) -> Optional[Quantity]:
        return self.axis(name).centers

    def ncell(self, name: str) -> Optional[int]:
        e = self.edges(name)
        return None if e is None else int(e.size - 1)

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of cell counts along all defined axes (in display order)."""
        counts = [self.ncell(a) for a in self.axis_names()]
        return tuple(c for c in counts if c is not None)

    # Small read-only view for dot-access discovery
    @property
    def coords(self):
        """
        Docstring
        """
        class _CoordView:
            __slots__ = ("_mesh",)
            def __init__(self, m: "Mesh"): self._mesh = m
            def __getattr__(self, key: str) -> Axis:
                names = set(_allowed_axes(self._mesh.coord_system))
                if key in names:
                    return self._mesh.axis(key)
                raise AttributeError(key)
            def __dir__(self):
                return list(_allowed_axes(self._mesh.coord_system))
        return _CoordView(self)

    def __repr__(self) -> str:
        cells = ", ".join(f"{a}:{self.ncell(a) or 0}" for a in self.axis_names())
        return f"Mesh(cs='{self.coord_system}', cells={{" + cells + "}})"

    # ---------- convenience constructors ----------
    @classmethod
    def spherical(
        cls,
        r: Axis,
        theta: Optional[Axis] = None,
        phi: Optional[Axis] = None,
    ) -> "Mesh":
        """
        Docstring
        """
        axes = {"r": r}
        if theta is not None: axes["theta"] = theta
        if phi is not None:   axes["phi"] = phi
        return cls("spherical", axes)

    @classmethod
    def cylindrical(
        cls,
        r: Axis,
        phi: Optional[Axis] = None,
        z: Optional[Axis] = None,
    ) -> "Mesh":
        """
        Docstring
        """
        axes = {"r": r}
        if phi is not None: axes["phi"] = phi
        if z is not None:   axes["z"] = z
        return cls("cylindrical", axes)

    @classmethod
    def cartesian(
        cls,
        x: Optional[Axis] = None,
        y: Optional[Axis] = None,
        z: Optional[Axis] = None,
    ) -> "Mesh":
        """
        Docstring
        """
        axes: Dict[str, Axis] = {}
        if x is not None: axes["x"] = x
        if y is not None: axes["y"] = y
        if z is not None: axes["z"] = z
        return cls("cartesian", axes)

    # ---------- pure transforms ----------
    def with_vertical(
        self,
        z_centers: Quantity
    ) -> "Mesh":
        """
        Docstring
        """
        if self.coord_system != "cylindrical":
            raise ValueError("with_vertical is intended for cylindrical workflows")
        new_axes = dict(self.axes)
        new_axes["z"] = Axis(centers=z_centers)
        return Mesh("cylindrical", new_axes)

    def to_spherical_by_scale_height(
        self,
        ncol: int,
        aspect_ratio: float,
        zmax_over_H: float = 5.0
    ) -> "Mesh":
        """
        Docstring
        """
        ar = float(getattr(aspect_ratio, "magnitude", aspect_ratio))
        thmin = np.pi/2.0 - np.arctan(zmax_over_H * ar)
        thmax = np.pi/2.0
        upper = np.linspace(thmin, thmax, int(ncol)//2 + 1)
        lower = np.pi - upper[1:int(ncol)//2 + 1]
        tedge = np.concatenate([lower, upper]) * units.radian

        r_ax = self.axes.get("r", Axis())
        p_ax = self.axes.get("phi", Axis())
        return Mesh.spherical(
            r=Axis(edges=r_ax.edges, centers=r_ax.centers),
            theta=Axis(edges=tedge),
            phi=Axis(edges=p_ax.edges, centers=p_ax.centers)
        )
