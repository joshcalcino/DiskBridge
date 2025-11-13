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


# ---------- small containers ----------
@dataclass
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
    raise ValueError(f"Unknown coord_system: {cs}")


def _display_order(cs: CoordSystem) -> Tuple[str, ...]:
    # purely cosmetic order for reporting/iteration
    if cs == "spherical":
        return ("r", "theta", "phi")
    if cs == "cylindrical":
        return ("z", "r", "phi")  # common (z, r, phi) workflow
    if cs == "cartesian":
        return ("x", "y", "z")
    return ()


def _attr_alias(cs: CoordSystem) -> Dict[str, Tuple[str, str]]:
    """
    Map legacy attribute names (redge, rmed, xedge, ...) -> (axis, kind)
    kind in {"edges","centers"}.
    """
    if cs == "spherical":
        return {
            "redge": ("r", "edges"), "rmed": ("r", "centers"),
            "tedge": ("theta", "edges"), "tmed": ("theta", "centers"),
            "pedge": ("phi", "edges"), "pmed": ("phi", "centers"),
        }
    if cs == "cylindrical":
        return {
            "redge": ("r", "edges"), "rmed": ("r", "centers"),
            "pedge": ("phi", "edges"), "pmed": ("phi", "centers"),
            "zedge": ("z", "edges"), "zmed": ("z", "centers"),
        }
    if cs == "cartesian":
        return {
            "xedge": ("x", "edges"), "xmed": ("x", "centers"),
            "yedge": ("y", "edges"), "ymed": ("y", "centers"),
            "zedge": ("z", "edges"), "zmed": ("z", "centers"),
        }
    return {}


# ---------- main Mesh ----------
@dataclass
class Mesh:
    """
    Compact, extensible mesh container.

    - Stores coordinates in a single dict: self.axes[axis] = Axis(edges, centers)
    - axis names depend on coord_system:
        spherical  -> ("r","theta","phi")
        cylindrical-> ("r","phi","z")
        cartesian  -> ("x","y","z")
    - Backward-compat attribute access via __getattr__:
        mesh.redge, mesh.rmed, mesh.xedge, mesh.pmed, ...
    """

    coord_system: CoordSystem
    axes: Dict[str, Axis]

    def __post_init__(self) -> None:
        allowed = set(_allowed_axes(self.coord_system))
        # prune any unexpected axes in the dict
        self.axes = {k: v for k, v in self.axes.items() if k in allowed}

        # auto-compute missing edges/centers
        for axis_name, ax in self.axes.items():
            if ax.centers is None and ax.edges is not None:
                if self.coord_system == "spherical" and axis_name == "r":
                    ax.centers = _spherical_r_centers(ax.edges)
                else:
                    ax.centers = _centers_from_edges(ax.edges)
            elif ax.edges is None and ax.centers is not None:
                ax.edges = _edges_from_centers(ax.centers)

            # validate lengths if both present
            if ax.edges is not None and ax.centers is not None:
                if ax.centers.size != ax.edges.size - 1:
                    raise ValueError(
                        f"{axis_name}: centers must have len(edges)-1 "
                        f"({ax.centers.size} vs {ax.edges.size - 1})"
                    )

        # infer dimensionality
        self.ndims = sum(
            1 for a in self.axes.values()
            if (a.edges is not None) or (a.centers is not None)
        )

    # ---------- simple API ----------
    def axes_order(self) -> Tuple[str, ...]:
        return _display_order(self.coord_system)

    def get_edges(self, axis: str) -> Optional[Quantity]:
        axis = axis.lower()
        if axis not in self.axes:
            raise ValueError(f"axis '{axis}' invalid for coord_system '{self.coord_system}'")
        return self.axes[axis].edges

    def get_centers(self, axis: str) -> Optional[Quantity]:
        axis = axis.lower()
        if axis not in self.axes:
            raise ValueError(f"axis '{axis}' invalid for coord_system '{self.coord_system}'")
        return self.axes[axis].centers

    # ---------- convenience constructors ----------
    @classmethod
    def spherical(
        cls,
        r: Axis,
        theta: Optional[Axis] = None,
        phi: Optional[Axis] = None,
    ) -> "Mesh":
        axes = {"r": r}
        if theta is not None:
            axes["theta"] = theta
        if phi is not None:
            axes["phi"] = phi
        return cls("spherical", axes)

    @classmethod
    def cylindrical(
        cls,
        r: Axis,
        phi: Optional[Axis] = None,
        z: Optional[Axis] = None,
    ) -> "Mesh":
        axes = {"r": r}
        if phi is not None:
            axes["phi"] = phi
        if z is not None:
            axes["z"] = z
        return cls("cylindrical", axes)

    @classmethod
    def cartesian(
        cls,
        x: Optional[Axis] = None,
        y: Optional[Axis] = None,
        z: Optional[Axis] = None,
    ) -> "Mesh":
        axes: Dict[str, Axis] = {}
        if x is not None:
            axes["x"] = x
        if y is not None:
            axes["y"] = y
        if z is not None:
            axes["z"] = z
        return cls("cartesian", axes)

    # ---------- helpers akin to your original API ----------
    def with_vertical(self, z_centers: Quantity) -> "Mesh":
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

    def to_cylindrical_from_spherical(self, nver: int) -> "Mesh":
        if self.coord_system != "spherical":
            raise ValueError("requires a spherical mesh as input")
        r_c = self.get_centers("r")
        t_c = self.get_centers("theta")
        if r_c is None or t_c is None:
            raise ValueError("r/theta centers required to derive cylindrical z centers")
        r_mag = np.asarray(r_c.magnitude, dtype=float)
        t_mag = np.asarray(t_c.to(units.radian).magnitude, dtype=float)
        zbuf = -np.max(r_mag) * np.cos(t_mag)
        zmed_mag = np.linspace(np.min(zbuf), np.max(zbuf), int(nver))
        zmed = zmed_mag * r_c.units

        # keep r,phi from spherical if present
        r_ax = self.axes.get("r", Axis())
        p_ax = self.axes.get("phi", Axis())
        return Mesh.cylindrical(
            r=Axis(edges=r_ax.edges, centers=r_ax.centers),
            phi=Axis(edges=p_ax.edges, centers=p_ax.centers),
            z=Axis(centers=zmed)
        )

    def __getattr__(self, name: str):
        # route redge/rmed/… to axes dict dynamically
        alias = _attr_alias(self.coord_system).get(name)
        if alias is None:
            raise AttributeError(name)
        axis, kind = alias
        ax = self.axes.get(axis)
        return None if ax is None else getattr(ax, kind)

    def __setattr__(self, name: str, value):
        # allow writing legacy names too (e.g., mesh.redge = arr)
        if name in {"coord_system", "axes", "ndims"}:
            return super().__setattr__(name, value)
        alias = None
        if "coord_system" in self.__dict__:
            alias = _attr_alias(self.coord_system).get(name)
        if alias is None:
            return super().__setattr__(name, value)
        axis, kind = alias
        if axis not in self.axes:
            self.axes[axis] = Axis()
        setattr(self.axes[axis], kind, value)

    # ---------- derived counts ----------
    def ncell(self, axis: str) -> Optional[int]:
        """Return number of cells (len(edges) - 1) along the given axis."""
        ax = self.axes.get(axis.lower())
        if ax is None or ax.edges is None:
            return None
        return int(ax.edges.size - 1)

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of cell counts along all defined axes (in display order)."""
        counts = [self.ncell(a) for a in self.axes_order()]
        return tuple(c for c in counts if c is not None)


    @property
    def nrad(self):  # spherical/cylindrical
        return self.ncell("r")

    @property
    def ncol(self):  # theta
        return self.ncell("theta")

    @property
    def nsec(self):  # phi
        return self.ncell("phi")

    @property
    def nver(self):  # z
        return self.ncell("z")

    @property
    def nx(self):
        return self.ncell("x")

    @property
    def ny(self):
        return self.ncell("y")

    @property
    def nz(self):
        return self.ncell("z")
