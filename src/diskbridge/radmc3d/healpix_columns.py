"""
HEALPix-based CO self-shielding / column integration helpers.

This module provides:
- SphericalHealpixRayTracer: geometric helper for rays through a spherical mesh
- compute_co_shielding_healpix: convenience wrapper that
    * takes n_H, chi, optional n_CO / n_H2
    * integrates N(CO), N(H2) along HEALPix rays from selected cells
    * evaluates Visser+09 CO shielding factors Θ_CO
    * returns Θ_CO and chi_eff = chi * Θ_CO

The functions are independent of RadModel to avoid circular imports.
You can call them from RadModel or from separate post-processing scripts.

Assumptions
-----------
- Mesh coordinate system is spherical, with axes 'r', 'theta', 'phi'.
- nH, chi, etc. are in RADMC / DiskBridge order (nr, ntheta, nphi),
  as produced by RadModel.compute_nH_from_model and compute_mcmono. 
- Column densities are integrated in cm^-2.

Dependencies
------------
- healpy for HEALPix directions (pip install healpy)
- diskbridge._units for unit handling
- diskbridge.radmc3d.visser_shielding for Θ_CO tables.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numba import njit, prange

import healpy as hp

from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from .visser_shielding import VisserShielding


# =============================================================================
# Numba JIT-compiled ray integration kernels
# =============================================================================

@njit(cache=True)
def _integrate_ray_cartesian(
    x0: float, y0: float, z0: float,
    dx: float, dy: float, dz: float,
    ds: float,
    n_field: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    zmin: float, zmax: float,
    max_steps: int = 10000,
) -> float:
    """
    Numba-compiled ray integration for Cartesian mesh.
    
    Integrates n_field along a ray starting at (x0, y0, z0) in direction (dx, dy, dz).
    Returns column density in same units as n_field * ds.
    """
    x, y, z = x0, y0, z0
    N = 0.0
    nx = n_field.shape[0]
    ny = n_field.shape[1]
    nz = n_field.shape[2]
    
    for i in range(max_steps):
        # Check domain bounds
        if x <= xmin or x >= xmax or y <= ymin or y >= ymax or z <= zmin or z >= zmax:
            break
        
        # Find cell indices using binary search
        ix = np.searchsorted(x_edges, x) - 1
        iy = np.searchsorted(y_edges, y) - 1
        iz = np.searchsorted(z_edges, z) - 1
        
        # Check valid index range
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            break
        
        # Accumulate column density
        N += n_field[ix, iy, iz] * ds
        
        # Step along ray
        x += dx * ds
        y += dy * ds
        z += dz * ds
        if i >= max_steps:
            print('not good bro')
    
    return N


@njit(cache=True, parallel=True)
def _integrate_all_rays_cartesian(
    cell_centers: np.ndarray,  # (n_cells, 3)
    directions: np.ndarray,    # (n_dirs, 3)
    ds: float,
    nCO_field: np.ndarray,
    nH2_field: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    zmin: float, zmax: float,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-parallel integration of CO and H2 columns for all cells and directions.
    
    Returns
    -------
    Nco_all : ndarray, shape (n_cells, n_dirs)
        CO column densities for each cell and direction.
    Nh2_all : ndarray, shape (n_cells, n_dirs)
        H2 column densities for each cell and direction.
    """
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]
    
    Nco_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    Nh2_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    
    for i in prange(n_cells):
        x0, y0, z0 = cell_centers[i, 0], cell_centers[i, 1], cell_centers[i, 2]
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j, 0], directions[j, 1], directions[j, 2]
            
            Nco_all[i, j] = _integrate_ray_cartesian(
                x0, y0, z0, dx, dy, dz, ds,
                nCO_field, x_edges, y_edges, z_edges,
                xmin, xmax, ymin, ymax, zmin, zmax, max_steps
            )
            Nh2_all[i, j] = _integrate_ray_cartesian(
                x0, y0, z0, dx, dy, dz, ds,
                nH2_field, x_edges, y_edges, z_edges,
                xmin, xmax, ymin, ymax, zmin, zmax, max_steps
            )
    
    return Nco_all, Nh2_all


@njit(cache=True)
def _integrate_ray_spherical(
    x0: float, y0: float, z0: float,
    dx: float, dy: float, dz: float,
    ds: float,
    n_field: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    rmin: float, rmax: float,
    max_steps: int = 10000,
) -> float:
    """
    Numba-compiled ray integration for spherical mesh.
    """
    x, y, z = x0, y0, z0
    N = 0.0
    nr = n_field.shape[0]
    nt = n_field.shape[1]
    np_ = n_field.shape[2]
    
    two_pi = 2.0 * np.pi
    
    for i in range(max_steps):
        # Convert to spherical
        r = np.sqrt(x*x + y*y + z*z)
        if r < rmin or r > rmax:
            break
        
        th = np.arccos(z / (r + 1e-99))
        ph = np.arctan2(y, x)
        if ph < 0:
            ph += two_pi
        
        # Find indices
        ir = np.searchsorted(r_edges, r) - 1
        it = np.searchsorted(theta_edges, th) - 1
        ip = np.searchsorted(phi_edges, ph) - 1
        
        # Handle phi wraparound
        if ip < 0:
            ip = np_ - 1
        elif ip >= np_:
            ip = 0
        
        if ir < 0 or ir >= nr or it < 0 or it >= nt:
            break
        
        N += n_field[ir, it, ip] * ds
        
        x += dx * ds
        y += dy * ds
        z += dz * ds
        
        if i >= max_steps:
            print('not good bro')
    
    return N


@njit(cache=True, parallel=True)
def _integrate_all_rays_spherical(
    cell_centers: np.ndarray,  # (n_cells, 3) in Cartesian
    directions: np.ndarray,    # (n_dirs, 3)
    ds: float,
    nCO_field: np.ndarray,
    nH2_field: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    rmin: float, rmax: float,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-parallel integration for spherical mesh.
    """
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]
    
    Nco_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    Nh2_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    
    for i in prange(n_cells):
        x0, y0, z0 = cell_centers[i, 0], cell_centers[i, 1], cell_centers[i, 2]
        
        for j in range(n_dirs):
            dx, dy, dz = directions[j, 0], directions[j, 1], directions[j, 2]
            
            Nco_all[i, j] = _integrate_ray_spherical(
                x0, y0, z0, dx, dy, dz, ds,
                nCO_field, r_edges, theta_edges, phi_edges,
                rmin, rmax, max_steps
            )
            Nh2_all[i, j] = _integrate_ray_spherical(
                x0, y0, z0, dx, dy, dz, ds,
                nH2_field, r_edges, theta_edges, phi_edges,
                rmin, rmax, max_steps
            )
    
    return Nco_all, Nh2_all

# Import threshold constant from model module
# Deferred import to avoid circular dependency
_LOG_CHI_OVER_NH_PDISS = None


def _get_log_chi_over_nH_pdiss() -> float:
    """Get the photodissociation threshold, importing lazily to avoid circular imports."""
    global _LOG_CHI_OVER_NH_PDISS
    if _LOG_CHI_OVER_NH_PDISS is None:
        from .model import LOG_CHI_OVER_NH_PDISS
        _LOG_CHI_OVER_NH_PDISS = float(LOG_CHI_OVER_NH_PDISS.magnitude)
    return _LOG_CHI_OVER_NH_PDISS  


def _to_ndarray_cgs(x, target_unit: str) -> np.ndarray:
    """
    Convert Quantity or array-like to plain ndarray in specified CGS unit.
    
    Parameters
    ----------
    x : Quantity or array-like
        Input data.
    target_unit : str
        Target unit string (e.g., 'cm', 'cm^-3').
        
    Returns
    -------
    ndarray
        Plain numpy array in the specified unit.
    """
    if isinstance(x, Quantity):
        return x.to(target_unit).magnitude
    return np.asarray(x, dtype=float)


class SphericalHealpixRayTracer:
    """
    Simple ray tracer on a spherical (r, theta, phi) mesh.

    Geometry helper only; it does not know about chemistry or shielding.

    Parameters
    ----------
    mesh : diskbridge.model.mesh.Mesh
        Mesh from a DiskBridge Model instance (coord_system == 'spherical').
    nside : int
        HEALPix Nside (npix = 12 * nside^2).
    ds_fraction : float, optional
        Step size along rays = ds_fraction * min(Δr).
        Smaller ds -> more accurate, more expensive.

    Notes
    -----
    - Uses simple fixed step-size marching; no attempt at exact
      cell-face intersections. Intended as a first implementation
      that you can optimize / numba-ise later.
    """

    def __init__(self, mesh, nside: int = 4, ds_fraction: float = 0.5) -> None:
        if mesh.coord_system != "spherical":
            raise ValueError(
                f"SphericalHealpixRayTracer requires spherical mesh, "
                f"got {mesh.coord_system!r}"
            )

        self.mesh = mesh
        self.nside = int(nside)

        # Axis edges in CGS / radians
        r_edges_q = mesh.edges("r")  # Quantity
        th_edges_q = mesh.edges("theta")
        ph_edges_q = mesh.edges("phi")

        self.r_edges = r_edges_q.to("cm").magnitude
        self.theta_edges = th_edges_q.to("rad").magnitude
        self.phi_edges = ph_edges_q.to("rad").magnitude

        # Precompute cell centers as plain arrays for speed
        self.r_centers = mesh.axes["r"].centers.to("cm").magnitude
        self.theta_centers = mesh.axes["theta"].centers.to("rad").magnitude
        self.phi_centers = mesh.axes["phi"].centers.to("rad").magnitude

        # Step size: fraction of the minimum Δr
        dr = np.diff(self.r_edges)
        self.ds = float(ds_fraction * dr.min())

        # HEALPix directions (unit vectors on sphere)
        npix = hp.nside2npix(self.nside)
        # hp.pix2vec -> (x, y, z) arrays of length npix
        dirs = hp.pix2vec(self.nside, np.arange(npix))
        self.dirs = np.vstack(dirs).T  # shape (npix, 3)

        # Cache grid shape
        self.nr = self.r_centers.size
        self.nt = self.theta_centers.size
        self.np = self.phi_centers.size

        # Domain for periodic wrapping in phi
        self._phi0 = self.phi_edges[0]
        self._phi_period = self.phi_edges[-1] - self.phi_edges[0]

        logger.info(
            f"Initialized SphericalHealpixRayTracer: "
            f"nr={self.nr}, ntheta={self.nt}, nphi={self.np}, "
            f"nside={self.nside}, npix={npix}, ds={self.ds:.3e} cm"
        )

    # ------------- basic geometry helpers -------------

    def cell_center_xyz(self, ir: int, it: int, ip: int) -> Tuple[float, float, float]:
        """Return Cartesian (x, y, z) position of a cell center in cm."""
        r = self.r_centers[ir]
        th = self.theta_centers[it]
        ph = self.phi_centers[ip]

        st = np.sin(th)
        x = r * st * np.cos(ph)
        y = r * st * np.sin(ph)
        z = r * np.cos(th)
        return float(x), float(y), float(z)

    def _xyz_to_indices(self, x: float, y: float, z: float) -> Optional[Tuple[int, int, int]]:
        """
        Map a Cartesian point to (ir, it, ip) cell indices.

        Returns None if the point lies outside the mesh.
        """
        r = np.sqrt(x * x + y * y + z * z)
        if r <= self.r_edges[0] or r >= self.r_edges[-1]:
            return None

        # polar angle theta ∈ [0, π]
        th = np.arccos(np.clip(z / r, -1.0, 1.0))
        if th <= self.theta_edges[0] or th >= self.theta_edges[-1]:
            return None

        # azimuth phi, wrap into [phi0, phi0 + period)
        ph = np.arctan2(y, x)
        ph = ((ph - self._phi0) % self._phi_period) + self._phi0
        if ph < self.phi_edges[0] or ph >= self.phi_edges[-1]:
            return None

        ir = int(np.searchsorted(self.r_edges, r, side="right") - 1)
        it = int(np.searchsorted(self.theta_edges, th, side="right") - 1)
        ip = int(np.searchsorted(self.phi_edges, ph, side="right") - 1)

        if not (0 <= ir < self.nr and 0 <= it < self.nt and 0 <= ip < self.np):
            return None
        return ir, it, ip

    # ------------- ray integration -------------

    def integrate_column_along_ray(
        self,
        x0: float,
        y0: float,
        z0: float,
        direction: np.ndarray,
        n_field: np.ndarray,
        max_steps: int = 10000,
    ) -> float:
        """
        Integrate column density along a single ray from a starting point.

        Parameters
        ----------
        x0, y0, z0 : float
            Starting position (cm), typically a cell center.
        direction : ndarray, shape (3,)
            Unit vector giving ray direction.
        n_field : ndarray, shape (nr, ntheta, nphi)
            Number density field in cm^-3.
        max_steps : int
            Safety limit on the number of marching steps.

        Returns
        -------
        float
            Column density in cm^-2 along this ray until the ray leaves
            the computational domain.
        """
        ds = self.ds
        dx, dy, dz = map(float, direction / np.linalg.norm(direction))

        x, y, z = x0, y0, z0
        N = 0.0

        for i in range(max_steps):
            idx = self._xyz_to_indices(x, y, z)
            if idx is None:
                break
            ir, it, ip = idx
            n_local = float(n_field[ir, it, ip])
            N += n_local * ds

            x += dx * ds
            y += dy * ds
            z += dz * ds
            if i>= max_steps:
                print('Not good bro.')

        return N


class CartesianHealpixRayTracer:
    """
    Simple ray tracer on a Cartesian (x, y, z) mesh.

    Geometry helper only; it does not know about chemistry or shielding.

    Parameters
    ----------
    mesh : Mesh
        DiskBridge Mesh instance with coord_system == "cartesian"
        and axes "x", "y", "z".
    nside : int, optional
        HEALPix Nside (npix = 12 * nside^2).
    ds_fraction : float, optional
        Step size along rays = ds_fraction * min(Δx, Δy, Δz).
        Smaller ds -> more accurate, more expensive.

    Notes
    -----
    - Uses simple fixed step-size marching; no attempt at exact
      cell-face intersections.
    """

    def __init__(self, mesh, nside: int = 4, ds_fraction: float = 0.5) -> None:
        if mesh.coord_system != "cartesian":
            raise ValueError(
                f"CartesianHealpixRayTracer requires cartesian mesh, "
                f"got {mesh.coord_system!r}"
            )

        self.mesh = mesh
        self.nside = int(nside)

        # Axis edges in CGS
        x_edges_q = mesh.edges("x")  # Quantity
        y_edges_q = mesh.edges("y")
        z_edges_q = mesh.edges("z")

        self.x_edges = x_edges_q.to("cm").magnitude
        self.y_edges = y_edges_q.to("cm").magnitude
        self.z_edges = z_edges_q.to("cm").magnitude

        # Precompute cell centers as plain arrays for speed
        self.x_centers = mesh.axes["x"].centers.to("cm").magnitude
        self.y_centers = mesh.axes["y"].centers.to("cm").magnitude
        self.z_centers = mesh.axes["z"].centers.to("cm").magnitude

        # Step size: fraction of the minimum cell size
        dx_min = np.min(np.diff(self.x_edges))
        dy_min = np.min(np.diff(self.y_edges))
        dz_min = np.min(np.diff(self.z_edges))
        self.ds = float(ds_fraction * min(dx_min, dy_min, dz_min))

        # Precompute HEALPix directions (unit vectors)
        npix = hp.nside2npix(self.nside)
        dirs = hp.pix2vec(self.nside, np.arange(npix))
        self.dirs = np.vstack(dirs).T  # shape (npix, 3)

        # Cache grid shape
        self.nx = self.x_centers.size
        self.ny = self.y_centers.size
        self.nz = self.z_centers.size

        # Domain bounds
        self._xmin = self.x_edges[0]
        self._xmax = self.x_edges[-1]
        self._ymin = self.y_edges[0]
        self._ymax = self.y_edges[-1]
        self._zmin = self.z_edges[0]
        self._zmax = self.z_edges[-1]

        logger.info(
            f"Initialized CartesianHealpixRayTracer: "
            f"nx={self.nx}, ny={self.ny}, nz={self.nz}, "
            f"nside={self.nside}, npix={npix}, ds={self.ds:.3e} cm"
        )

    # ------------- basic geometry helpers -------------

    def cell_center_xyz(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        """Return Cartesian (x, y, z) position of a cell center in cm."""
        x = self.x_centers[ix]
        y = self.y_centers[iy]
        z = self.z_centers[iz]
        return float(x), float(y), float(z)

    def _xyz_to_indices(
        self, x: float, y: float, z: float
    ) -> Optional[Tuple[int, int, int]]:
        """
        Convert Cartesian (x, y, z) in cm to (ix, iy, iz) indices.

        Returns None if (x, y, z) is outside the mesh domain.
        """
        # Quick domain check
        if (
            x <= self._xmin
            or x >= self._xmax
            or y <= self._ymin
            or y >= self._ymax
            or z <= self._zmin
            or z >= self._zmax
        ):
            return None

        ix = int(np.searchsorted(self.x_edges, x, side="right") - 1)
        iy = int(np.searchsorted(self.y_edges, y, side="right") - 1)
        iz = int(np.searchsorted(self.z_edges, z, side="right") - 1)

        if not (0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz):
            return None

        return ix, iy, iz

    # ------------- ray integration -------------

    def integrate_column_along_ray(
        self,
        x0: float,
        y0: float,
        z0: float,
        direction: np.ndarray,
        n_field: np.ndarray,
        max_steps: int = 10000,
    ) -> float:
        """
        Integrate column density along a single ray from a starting point.

        Parameters
        ----------
        x0, y0, z0 : float
            Starting position (cm), typically a cell center.
        direction : ndarray, shape (3,)
            Unit vector giving ray direction.
        n_field : ndarray, shape (nx, ny, nz)
            Number density field in cm^-3.
        max_steps : int
            Safety limit on the number of marching steps.

        Returns
        -------
        float
            Column density in cm^-2 along this ray until the ray leaves
            the computational domain.
        """
        ds = self.ds
        dvec = np.asarray(direction, dtype=float)
        dvec /= np.linalg.norm(dvec)
        dx, dy, dz = map(float, dvec)

        x, y, z = float(x0), float(y0), float(z0)
        N = 0.0

        for i in range(max_steps):
            idx = self._xyz_to_indices(x, y, z)
            if idx is None:
                break
            ix, iy, iz = idx
            n_local = float(n_field[ix, iy, iz])
            N += n_local * ds

            x += dx * ds
            y += dy * ds
            z += dz * ds

            if i >= max_steps - 1:
                print(f'Warning: reached max_steps {max_steps} at ray step {i}')
                break

        return N


def compute_co_shielding_healpix(
    mesh,
    nH,
    chi,
    visser: VisserShielding,
    *,
    nCO=None,
    nH2=None,
    nside: int = 4,
    b_kms: Optional[float] = None,
    candidate_mask: Optional[np.ndarray] = None,
    log_chi_over_nH_pdiss: Optional[float] = None,
    margin_dex: float = 1.0,
    Xco_guess: float = 5e-5,
    XH2_guess: float = 0.5,
    max_cells: Optional[int] = None,
) -> Tuple[Quantity, Quantity]:
    """
    Compute CO self-shielding using HEALPix rays + Visser+09 tables.

    Parameters
    ----------
    mesh : Mesh
        DiskBridge mesh (spherical or cartesian).
    nH : Quantity or ndarray
        H nuclei number density [cm^-3], shape matching mesh.
    chi : Quantity or ndarray
        UV field in Draine units (dust-attenuated, unshielded by molecules).
    visser : VisserShielding
        Loaded Visser+09 shielding interpolator.
    nCO : Quantity or ndarray, optional
        CO number density [cm^-3]. If None, approximated as Xco_guess * nH.
    nH2 : Quantity or ndarray, optional
        H2 number density [cm^-3]. If None, approximated as XH2_guess * nH / 2.
    nside : int, optional
        HEALPix resolution. npix = 12 * nside^2 directions.
    b_kms : float, optional
        Microturbulent Doppler b (km/s). Must match the Visser file if given.
    candidate_mask : ndarray of bool, optional
        Mask of cells to process. If None, constructed from chi/nH threshold.
    log_chi_over_nH_pdiss : float, optional
        Photodissociation threshold log10(chi/nH). Defaults to Pinte+18 value.
    margin_dex : float, optional
        Extra dex below the threshold to include in candidate mask.
    Xco_guess : float, optional
        Default CO abundance relative to nH when nCO is not supplied.
    XH2_guess : float, optional
        Fraction of H nuclei in H2 when nH2 is not supplied.
    max_cells : int, optional
        Limit processed cells (useful for quick tests).

    Returns
    -------
    theta_co : Quantity, dimensionless
        Angle-averaged CO shielding factor (0-1).
    chi_eff : Quantity, dimensionless
        Effective UV field including CO self-shielding: chi * theta_co.

    Notes
    -----
    - Math-heavy sections use plain numpy arrays for numba compatibility.
    - Geometry / ray marching is delegated to SphericalHealpixRayTracer or
      CartesianHealpixRayTracer depending on mesh.coord_system.
    """

    # Convert inputs to plain ndarrays in CGS units
    # These are the math-heavy arrays kept as plain numpy for numba compatibility
    nH_cgs = _to_ndarray_cgs(nH, "cm^-3")
    chi_arr = _to_ndarray_cgs(chi, "dimensionless")  # chi is in Draine units

    if nCO is None:
        nCO_cgs = Xco_guess * nH_cgs
    else:
        nCO_cgs = _to_ndarray_cgs(nCO, "cm^-3")

    if nH2 is None:
        # nH counts nuclei; H2 has 2 H nuclei per molecule
        nH2_cgs = 0.5 * XH2_guess * nH_cgs
    else:
        nH2_cgs = _to_ndarray_cgs(nH2, "cm^-3")

    if nH_cgs.shape != chi_arr.shape:
        raise ValueError(f"nH and chi must have same shape, got {nH_cgs.shape} vs {chi_arr.shape}")
    if nCO_cgs.shape != nH_cgs.shape or nH2_cgs.shape != nH_cgs.shape:
        raise ValueError("nCO and nH2 must match nH shape.")

    # Threshold for "interesting" cells (where simple Pinte would photodissociate)
    if log_chi_over_nH_pdiss is None:
        log_chi_over_nH_pdiss = _get_log_chi_over_nH_pdiss()

    if candidate_mask is None:
        ratio = chi_arr / (nH_cgs + 1e-99)
        log_ratio = np.log10(np.maximum(ratio, 1e-99))
        candidate_mask = log_ratio > (log_chi_over_nH_pdiss - margin_dex)

    candidate_idx = np.argwhere(candidate_mask)
    n_candidates = candidate_idx.shape[0]

    if max_cells is not None and n_candidates > max_cells:
        logger.warning(
            f"Limiting HEALPix shielding to first {max_cells} of {n_candidates} "
            f"candidate cells."
        )
        candidate_idx = candidate_idx[:max_cells]
        n_candidates = max_cells

    logger.info(
        f"compute_co_shielding_healpix: {n_candidates} candidate cells "
        f"(margin={margin_dex:.2f} dex around log10(chi/nH)={log_chi_over_nH_pdiss:.2f})."
    )

    # Prepare ray tracer (spherical or cartesian)
    if mesh.coord_system == "spherical":
        tracer = SphericalHealpixRayTracer(mesh, nside=nside)
    elif mesh.coord_system == "cartesian":
        tracer = CartesianHealpixRayTracer(mesh, nside=nside)
    else:
        raise ValueError(
            f"Unsupported mesh coord_system {mesh.coord_system!r} for "
            "compute_co_shielding_healpix (expected 'spherical' or 'cartesian')."
        )

    dirs = tracer.dirs.astype(np.float64)
    npix = dirs.shape[0]

    # Output arrays
    theta_co = np.ones_like(nH_cgs, dtype=np.float64)

    if n_candidates == 0:
        logger.info("No candidate cells to process.")
        chi_eff = chi_arr * theta_co
        theta_co_q = Quantity(theta_co, "dimensionless")
        chi_eff_q = Quantity(chi_eff, "dimensionless")
        return theta_co_q, chi_eff_q

    # Prepare cell centers array for Numba
    cell_centers = np.zeros((n_candidates, 3), dtype=np.float64)
    for k, idx in enumerate(candidate_idx):
        cell_centers[k] = tracer.cell_center_xyz(*idx)

    # Use Numba-optimized ray integration
    logger.info(f"Running Numba-optimized ray integration ({n_candidates} cells x {npix} directions)...")
    
    if mesh.coord_system == "cartesian":
        Nco_all, Nh2_all = _integrate_all_rays_cartesian(
            cell_centers, dirs, tracer.ds,
            nCO_cgs.astype(np.float64),
            nH2_cgs.astype(np.float64),
            tracer.x_edges.astype(np.float64),
            tracer.y_edges.astype(np.float64),
            tracer.z_edges.astype(np.float64),
            tracer._xmin, tracer._xmax,
            tracer._ymin, tracer._ymax,
            tracer._zmin, tracer._zmax,
        )
    else:  # spherical
        Nco_all, Nh2_all = _integrate_all_rays_spherical(
            cell_centers, dirs, tracer.ds,
            nCO_cgs.astype(np.float64),
            nH2_cgs.astype(np.float64),
            tracer.r_edges.astype(np.float64),
            tracer.theta_edges.astype(np.float64),
            tracer.phi_edges.astype(np.float64),
            tracer.r_edges[0], tracer.r_edges[-1],
        )

    logger.info("Ray integration complete. Computing shielding factors...")

    # Evaluate Visser shielding for all cells (vectorized over rays)
    for k, idx in enumerate(candidate_idx):
        Nco_rays = Nco_all[k, :]
        Nh2_rays = Nh2_all[k, :]
        theta_rays = visser.theta("co", Nco_rays, Nh2_rays, b_kms=b_kms)
        theta_co[tuple(idx)] = float(theta_rays.mean())

    chi_eff = chi_arr * theta_co
    
    # Wrap outputs as Quantities
    theta_co_q = Quantity(theta_co, "dimensionless")
    chi_eff_q = Quantity(chi_eff, "dimensionless")  # Draine units
    
    return theta_co_q, chi_eff_q
