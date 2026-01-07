"""
HEALPix-based CO self-shielding / column integration helpers.

This module provides:
- SphericalHealpixRayTracer: geometric helper for rays through a spherical mesh
- compute_co_shielding_healpix: convenience wrapper that
    * takes n_H, chi, optional n_CO / n_H2
    * integrates N(CO), N(H2) along HEALPix rays from selected cells
    * evaluates Visser+09 CO shielding factors theta_CO
    * returns theta_CO and chi_eff = chi * theta_CO

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
- diskbridge.chemistry.shielding.visser_shielding for theta_CO tables.
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
from diskbridge.chemistry.shielding.healpix_utils import integrate_rays

from diskbridge._constants import LOG_CHI_OVER_NH_PDISS, EPS_CHI


# =============================================================================
# HEALPIX CACHE UTILITIES
# =============================================================================

def _mesh_cache_info(mesh, shape: tuple[int, ...] | list[int]):
    if mesh.coord_system == "cartesian":
        return {
            "coord": "cartesian",
            "shape": list(shape),
            "xmin": float(mesh.edges("x").to("cm").magnitude[0]),
            "xmax": float(mesh.edges("x").to("cm").magnitude[-1]),
            "ymin": float(mesh.edges("y").to("cm").magnitude[0]),
            "ymax": float(mesh.edges("y").to("cm").magnitude[-1]),
            "zmin": float(mesh.edges("z").to("cm").magnitude[0]),
            "zmax": float(mesh.edges("z").to("cm").magnitude[-1]),
        }
    if mesh.coord_system == "spherical":
        return {
            "coord": "spherical",
            "shape": list(shape),
            "rmin": float(mesh.edges("r").to("cm").magnitude[0]),
            "rmax": float(mesh.edges("r").to("cm").magnitude[-1]),
            "thmin": float(mesh.edges("theta").to("rad").magnitude[0]),
            "thmax": float(mesh.edges("theta").to("rad").magnitude[-1]),
            "phmin": float(mesh.edges("phi").to("rad").magnitude[0]),
            "phmax": float(mesh.edges("phi").to("rad").magnitude[-1]),
        }
    return {"coord": mesh.coord_system, "shape": list(shape)}

def _compute_field_hash(arr: np.ndarray, precision: int = 6) -> str:
    """Compute a hash of a numpy array for cache keying.
    
    Parameters
    ----------
    arr : ndarray
        Array to hash.
    precision : int
        Number of decimal places to round to before hashing.
        This allows minor floating point differences to still match.
    
    Returns
    -------
    str
        MD5 hash of the rounded array.
    """
    rounded = np.round(arr.flatten(), precision)
    return hashlib.md5(rounded.tobytes()).hexdigest()[:16]


def _compute_cache_key(
    mesh,
    nside: int,
    nH_cgs: np.ndarray,
    chi_arr: np.ndarray,
    Xco_guess: float,
    XH2_guess: float,
) -> str:
    """Compute a unique cache key for healpix column computation.
    
    The key incorporates:
    - Mesh shape and coordinate system
    - HEALPix nside
    - Hash of nH field (density structure)
    - Hash of chi field (UV field)
    - Abundance guesses and threshold parameters
    
    Returns
    -------
    str
        Cache key string.
    """
    mesh_info = _mesh_cache_info(mesh, nH_cgs.shape)
    
    params = {
        "mesh": mesh_info,
        "nside": nside,
        "nH_hash": _compute_field_hash(nH_cgs),
        "chi_hash": _compute_field_hash(chi_arr),
        "Xco_guess": Xco_guess,
        "XH2_guess": XH2_guess,
        "log_threshold": LOG_CHI_OVER_NH_PDISS,
    }
    
    # Create deterministic JSON string and hash it
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def save_healpix_cache(
    cache_dir: Path | str,
    cache_key: str,
    theta_co: np.ndarray,
    chi_eff: np.ndarray,
    metadata: dict | None = None,
) -> Path:
    """Save healpix computation results to cache.
    
    Parameters
    ----------
    cache_dir : Path or str
        Directory to save cache files.
    cache_key : str
        Unique cache key from _compute_cache_key.
    theta_co : ndarray
        CO shielding factor array.
    chi_eff : ndarray
        Effective UV field array.
    metadata : dict, optional
        Additional metadata to save (e.g., timing info).
    
    Returns
    -------
    Path
        Path to saved cache file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"healpix_cache_{cache_key}.npz"
    
    save_dict = {
        "theta_co": theta_co,
        "chi_eff": chi_eff,
    }
    
    if metadata is not None:
        # Store metadata as a JSON string
        save_dict["metadata_json"] = np.array([json.dumps(metadata)])
    
    np.savez_compressed(cache_file, **save_dict)
    logger.info(f"Saved healpix cache to: {cache_file}")
    return cache_file


def load_healpix_cache(
    cache_dir: Path | str,
    cache_key: str,
) -> tuple[np.ndarray, np.ndarray, dict | None] | None:
    """Load healpix computation results from cache.
    
    Parameters
    ----------
    cache_dir : Path or str
        Directory containing cache files.
    cache_key : str
        Unique cache key from _compute_cache_key.
    
    Returns
    -------
    tuple or None
        (theta_co, chi_eff, metadata) if cache exists, None otherwise.
    """
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"healpix_cache_{cache_key}.npz"
    
    if not cache_file.exists():
        return None
    
    try:
        data = np.load(cache_file)
        theta_co = data["theta_co"]
        chi_eff = data["chi_eff"]
        
        metadata = None
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"][0]))
        
        logger.info(f"Loaded healpix cache from: {cache_file}")
        return theta_co, chi_eff, metadata
    except Exception as e:
        logger.warning(f"Failed to load healpix cache: {e}")
        return None


def find_compatible_cache(
    cache_dir: Path | str,
    mesh,
    nside: int,
    nH_cgs: np.ndarray,
    chi_arr: np.ndarray,
    Xco_guess: float,
    XH2_guess: float,
) -> tuple[np.ndarray, np.ndarray, dict | None] | None:
    """Find and load a compatible cache file if one exists.
    
    This is a convenience function that computes the cache key and
    attempts to load the corresponding cache.
    
    Returns
    -------
    tuple or None
        (theta_co, chi_eff, metadata) if cache exists, None otherwise.
    """
    cache_key = _compute_cache_key(
        mesh, nside, nH_cgs, chi_arr, Xco_guess, XH2_guess,
    )
    return load_healpix_cache(cache_dir, cache_key)  


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


def _compute_mask_hash(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).ravel())
    return hashlib.md5(packed.tobytes()).hexdigest()[:16]


def _compute_nh_cache_key(
    mesh,
    nside: int,
    nH_cgs: np.ndarray,
    candidate_mask: np.ndarray,
) -> str:
    mesh_info = _mesh_cache_info(mesh, nH_cgs.shape)

    params = {
        "mesh": mesh_info,
        "nside": int(nside),
        "nH_hash": _compute_field_hash(nH_cgs),
        "mask_hash": _compute_mask_hash(candidate_mask),
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def _load_nh_rays_cache(
    cache_dir: Path | str,
    cache_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None] | None:
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"healpix_nh_cache_{cache_key}.npz"
    if not cache_file.exists():
        return None

    try:
        data = np.load(cache_file)
        candidate_idx = data["candidate_idx"]
        dirs = data["dirs"]
        N_H_rays = data["N_H_rays"]

        metadata = None
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"][0]))

        logger.info(f"Loaded healpix NH cache from: {cache_file}")
        return candidate_idx, dirs, N_H_rays, metadata
    except Exception as e:
        logger.warning(f"Failed to load healpix NH cache: {e}")
        return None


def _save_nh_rays_cache(
    cache_dir: Path | str,
    cache_key: str,
    candidate_idx: np.ndarray,
    dirs: np.ndarray,
    N_H_rays: np.ndarray,
    metadata: dict | None = None,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"healpix_nh_cache_{cache_key}.npz"

    save_dict = {
        "candidate_idx": candidate_idx,
        "dirs": dirs,
        "N_H_rays": N_H_rays,
    }
    if metadata is not None:
        save_dict["metadata_json"] = np.array([json.dumps(metadata)])

    np.savez_compressed(cache_file, **save_dict)
    logger.info(f"Saved healpix NH cache to: {cache_file}")
    return cache_file


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
        Step size along rays = ds_fraction * min(dr).
        Smaller ds -> more accurate, more expensive.

    Notes
    -----
    - Uses simple fixed step-size marching; no attempt at exact
      cell-face intersections. Intended as a first implementation
      that you can optimize / numba-ise later.
    """

    def __init__(self, mesh, nside: int = 4, ds_fraction: float = 0.5) -> None:
        try:
            import healpy as hp
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "healpy is required for HEALPix shielding (pip install healpy)"
            ) from e

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

        # Step size: fraction of the minimum dr
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

        # Precompute parameters for analytic (log-uniform) indexing
        # r is log-uniform: r_edges = exp(ln_rmin + dlnr * i)
        self._ln_rmin = float(np.log(self.r_edges[0]))
        ln_rmax = float(np.log(self.r_edges[-1]))
        dlnr = (ln_rmax - self._ln_rmin) / self.nr
        self._inv_dlnr = 1.0 / dlnr

        # theta is uniform in [0, pi]
        dtheta = np.pi / self.nt
        self._inv_dtheta = 1.0 / dtheta

        # phi is uniform in [0, 2*pi)
        dphi = 2.0 * np.pi / self.np
        self._inv_dphi = 1.0 / dphi

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


# =============================================================================


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
        Step size along rays = ds_fraction * min(dx, dy, dz).
        Smaller ds -> more accurate, more expensive.

    Notes
    -----
    - Uses simple fixed step-size marching; no attempt at exact
      cell-face intersections.
    """

    def __init__(self, mesh, nside: int = 4, ds_fraction: float = 0.5) -> None:
        try:
            import healpy as hp
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "healpy is required for HEALPix shielding (pip install healpy)"
            ) from e

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


def compute_NH_rays_healpix(
    mesh,
    nH_cgs: np.ndarray,
    *,
    nside: int,
    candidate_mask: np.ndarray,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nH_cgs = np.asarray(nH_cgs, dtype=np.float64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if nH_cgs.shape != candidate_mask.shape:
        raise ValueError(
            f"nH_cgs and candidate_mask must have the same shape, got {nH_cgs.shape} vs {candidate_mask.shape}"
        )

    cache_key = None
    if cache_dir is not None:
        cache_key = _compute_nh_cache_key(mesh, int(nside), nH_cgs, candidate_mask)
        cached = _load_nh_rays_cache(cache_dir, cache_key)
        if cached is not None:
            candidate_idx, dirs, N_H_rays, metadata = cached
            if metadata:
                logger.info(
                    f"Loaded healpix NH rays from cache (computed in {metadata.get('compute_time_s', '?')}s)"
                )
            return candidate_idx, dirs, N_H_rays

    candidate_idx = np.argwhere(candidate_mask)
    n_candidates = candidate_idx.shape[0]

    if mesh.coord_system == "spherical":
        tracer = SphericalHealpixRayTracer(mesh, nside=int(nside))
    elif mesh.coord_system == "cartesian":
        tracer = CartesianHealpixRayTracer(mesh, nside=int(nside))
    else:
        raise ValueError(
            f"Unsupported mesh coord_system {mesh.coord_system!r} for compute_NH_rays_healpix "
            "(expected 'spherical' or 'cartesian')."
        )

    dirs = tracer.dirs.astype(np.float64)
    npix = dirs.shape[0]

    if n_candidates == 0:
        N_H_rays = np.empty((0, npix), dtype=np.float64)
        return candidate_idx, dirs, N_H_rays

    cell_centers = np.zeros((n_candidates, 3), dtype=np.float64)
    for k, idx in enumerate(candidate_idx):
        cell_centers[k] = tracer.cell_center_xyz(*idx)

    _t_start = _time.time()

    if progress_chunks is None or progress_chunks <= 1:
        N_H_rays = integrate_rays(tracer, cell_centers, dirs, nH_cgs)
    else:
        n_chunks = int(progress_chunks)
        if n_chunks <= 0:
            n_chunks = 1
        chunk_size = max(1, n_candidates // n_chunks)

        logger.info(
            f"HEALPix ray tracing progress enabled: {n_candidates} cells "
            f"in {n_chunks} chunks (chunk_size={chunk_size})."
        )

        N_H_rays = np.empty((n_candidates, npix), dtype=np.float64)
        _t_chunk_start = _time.time()
        for ichunk, start in enumerate(range(0, n_candidates, chunk_size), 1):
            end = min(start + chunk_size, n_candidates)
            N_chunk = integrate_rays(
                tracer,
                cell_centers[start:end],
                dirs,
                nH_cgs,
            )
            N_H_rays[start:end] = N_chunk

            frac = end / n_candidates
            t_now = _time.time()
            dt = t_now - _t_chunk_start
            _t_chunk_start = t_now
            n_cells_chunk = end - start
            cells_per_sec = n_cells_chunk / dt if dt > 0.0 else float("inf")
            rays_per_sec = (n_cells_chunk * npix) / dt if dt > 0.0 else float("inf")
            logger.info(
                f"HEALPix rays: {end}/{n_candidates} cells "
                f"({100.0 * frac:.1f}%) done. "
                f"Chunk {ichunk}/{n_chunks}: {n_cells_chunk} cells in {dt:.2f}s "
                f"({cells_per_sec:.1f} cells/s, {rays_per_sec:.1f} rays/s)."
            )

    _t_elapsed = _time.time() - _t_start
    if cache_dir is not None and cache_key is not None:
        metadata = {
            "compute_time_s": round(_t_elapsed, 2),
            "n_candidates": int(n_candidates),
            "nside": int(nside),
            "npix": int(npix),
        }
        _save_nh_rays_cache(cache_dir, cache_key, candidate_idx, dirs, N_H_rays, metadata)

    return candidate_idx, dirs, N_H_rays





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
    Xco_guess: float = 5e-5,
    XH2_guess: float = 0.5,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    return_quantity: bool = True,
) -> Tuple[Quantity, Quantity] | Tuple[np.ndarray, np.ndarray]:
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
    Xco_guess : float, optional
        Default CO abundance relative to nH when nCO is not supplied.
    XH2_guess : float, optional
        Fraction of H nuclei in H2 when nH2 is not supplied.
    progress_chunks : int, optional
        If set to a positive integer, split candidate cells into this many
        chunks, calling integrate_rays on each chunk and logging progress
        after each chunk. None (default) keeps a single fast integrate_rays
        call for maximum performance.
    cache_dir : Path or str, optional
        Directory to cache/load healpix computation results. If provided and
        a matching cache exists, the cached results will be loaded instead of
        recomputing. New results will be saved to cache after computation.

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
    -      CartesianHealpixRayTracer depending on mesh.coord_system.
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
        nH2_cgs = 0.5 * XH2_guess * nH_cgs
    else:
        nH2_cgs = _to_ndarray_cgs(nH2, "cm^-3")

    if nH_cgs.shape != chi_arr.shape:
        raise ValueError(f"nH and chi must have same shape, got {nH_cgs.shape} vs {chi_arr.shape}")
    if nCO_cgs.shape != nH_cgs.shape or nH2_cgs.shape != nH_cgs.shape:
        raise ValueError("nCO and nH2 must match nH shape.")

    Xco_field = nCO_cgs / (nH_cgs + 1.0e-99)
    XH2_field = 2.0 * nH2_cgs / (nH_cgs + 1.0e-99)

    # Check for cached results
    cache_key = None
    if cache_dir is not None:
        cache_key = _compute_cache_key(
            mesh, nside, nH_cgs, chi_arr, Xco_guess, XH2_guess,
        )
        cached = load_healpix_cache(cache_dir, cache_key)
        if cached is not None:
            theta_co_cached, chi_eff_cached, metadata = cached
            logger.info(
                f"Loaded healpix results from cache "
                f"(computed in {metadata.get('compute_time_s', '?')}s)"
                if metadata else "Loaded healpix results from cache"
            )
            if return_quantity:
                theta_co_q = Quantity(theta_co_cached, "dimensionless")
                chi_eff_q = Quantity(chi_eff_cached, "dimensionless")
                return theta_co_q, chi_eff_q
            return theta_co_cached, chi_eff_cached

    if candidate_mask is None:
        ratio = chi_arr / (nH_cgs + 1e-99)
        log_ratio = np.log10(np.maximum(ratio, 1e-99))
        candidate_mask = log_ratio > LOG_CHI_OVER_NH_PDISS

    candidate_idx = np.argwhere(candidate_mask)
    n_candidates = candidate_idx.shape[0]

    logger.info(
        f"compute_co_shielding_healpix: {n_candidates} candidate cells "
        f"(log10(chi/nH) > {LOG_CHI_OVER_NH_PDISS:.2f})."
    )

    _t_start = _time.time()

    _, dirs, Nref_all = compute_NH_rays_healpix(
        mesh,
        nH_cgs,
        nside=int(nside),
        candidate_mask=candidate_mask,
        progress_chunks=progress_chunks,
        cache_dir=cache_dir,
    )
    npix = dirs.shape[0]

    # Output arrays
    theta_co = np.ones_like(nH_cgs, dtype=np.float64)

    if n_candidates == 0:
        logger.info("No candidate cells to process.")
        chi_eff = chi_arr * theta_co
        if return_quantity:
            theta_co_q = Quantity(theta_co, "dimensionless")
            chi_eff_q = Quantity(chi_eff, "dimensionless")
            return theta_co_q, chi_eff_q
        return theta_co, chi_eff

    ci = candidate_idx[:, 0]
    cj = candidate_idx[:, 1]
    ck = candidate_idx[:, 2]
    Xco_cand = Xco_field[ci, cj, ck]
    XH2_cand = XH2_field[ci, cj, ck]

    Nco_all = (Xco_cand[:, None] * Nref_all).astype(np.float64)
    Nh2_all = (0.5 * XH2_cand[:, None] * Nref_all).astype(np.float64)

    logger.info("Ray integration complete. Computing shielding factors...")

    # Evaluate Visser shielding for all cells (vectorized over rays)
    theta_rays_all = visser.theta("co", Nco_all, Nh2_all, b_kms=b_kms)
    theta_mean = theta_rays_all.mean(axis=1)
    for k, idx in enumerate(candidate_idx):
        theta_co[tuple(idx)] = float(theta_mean[k])

    chi_eff = chi_arr * theta_co
    _t_elapsed = _time.time() - _t_start
    
    # Save to cache if cache_dir is specified
    if cache_dir is not None and cache_key is not None:
        metadata = {
            "compute_time_s": round(_t_elapsed, 2),
            "n_candidates": n_candidates,
            "nside": nside,
            "npix": npix,
        }
        save_healpix_cache(cache_dir, cache_key, theta_co, chi_eff, metadata)
    
    if return_quantity:
        theta_co_q = Quantity(theta_co, "dimensionless")
        chi_eff_q = Quantity(chi_eff, "dimensionless")  # Draine units
        return theta_co_q, chi_eff_q
    return theta_co, chi_eff


def compute_pdr_shielding_healpix(
    mesh,
    nH,
    chi,
    *,
    visser: Optional[VisserShielding] = None,
    nCO=None,
    nH2=None,
    nside: int = 4,
    b_kms: Optional[float] = None,
    candidate_mask: Optional[np.ndarray] = None,
    Xco_guess: float = 5e-5,
    XH2_guess: float = 0.5,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    return_quantity: bool = True,
) -> tuple[Quantity, Quantity, Quantity, Quantity] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nH_cgs = _to_ndarray_cgs(nH, "cm^-3")
    chi_arr = _to_ndarray_cgs(chi, "dimensionless")

    if nCO is None:
        nCO_cgs = Xco_guess * nH_cgs
    else:
        nCO_cgs = _to_ndarray_cgs(nCO, "cm^-3")

    if nH2 is None:
        nH2_cgs = 0.5 * XH2_guess * nH_cgs
    else:
        nH2_cgs = _to_ndarray_cgs(nH2, "cm^-3")

    if nH_cgs.shape != chi_arr.shape:
        raise ValueError(f"nH and chi must have same shape, got {nH_cgs.shape} vs {chi_arr.shape}")
    if nCO_cgs.shape != nH_cgs.shape or nH2_cgs.shape != nH_cgs.shape:
        raise ValueError("nCO and nH2 must match nH shape.")

    Xco_field = nCO_cgs / (nH_cgs + 1.0e-99)
    XH2_field = 2.0 * nH2_cgs / (nH_cgs + 1.0e-99)

    if candidate_mask is None:
        ratio = chi_arr / (nH_cgs + 1e-99)
        log_ratio = np.log10(np.maximum(ratio, 1e-99))
        candidate_mask = log_ratio > LOG_CHI_OVER_NH_PDISS

    candidate_idx, dirs, N_H_rays = compute_NH_rays_healpix(
        mesh,
        nH_cgs,
        nside=int(nside),
        candidate_mask=candidate_mask,
        progress_chunks=progress_chunks,
        cache_dir=cache_dir,
    )

    theta_h2 = np.ones_like(nH_cgs, dtype=np.float64)
    theta_co = np.ones_like(nH_cgs, dtype=np.float64)
    theta_c = np.ones_like(nH_cgs, dtype=np.float64)

    n_candidates = candidate_idx.shape[0]
    if n_candidates > 0:
        from diskbridge.chemistry.hydrogen.partition import _h2_self_shielding_db96

        ci = candidate_idx[:, 0]
        cj = candidate_idx[:, 1]
        ck = candidate_idx[:, 2]

        XH2_cand = XH2_field[ci, cj, ck]
        N_H2_rays = (0.5 * XH2_cand[:, None] * N_H_rays).astype(np.float64)
        f_sh_rays = _h2_self_shielding_db96(N_H2_rays, b5=2.0, alpha=-0.75)
        theta_h2_mean = f_sh_rays.mean(axis=1)
        for k, idx in enumerate(candidate_idx):
            theta_h2[tuple(idx)] = float(theta_h2_mean[k])

        if visser is not None:
            Xco_cand = Xco_field[ci, cj, ck]
            Nco_rays = (Xco_cand[:, None] * N_H_rays).astype(np.float64)
            theta_co_rays = visser.theta("co", Nco_rays, N_H2_rays, b_kms=b_kms)
            theta_co_mean = theta_co_rays.mean(axis=1)
            for k, idx in enumerate(candidate_idx):
                theta_co[tuple(idx)] = float(theta_co_mean[k])

    chi_eff_h2 = chi_arr * theta_h2
    chi_eff_co = chi_arr * theta_co
    chi_eff_pdr = chi_arr * theta_h2 * theta_co

    if return_quantity:
        return (
            Quantity(theta_h2, "dimensionless"),
            Quantity(theta_co, "dimensionless"),
            Quantity(theta_c, "dimensionless"),
            Quantity(chi_eff_pdr, "dimensionless"),
        )
    return theta_h2, theta_co, theta_c, chi_eff_pdr
