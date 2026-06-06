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

Dependencies
------------
- healpy for HEALPix directions (pip install healpy)
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
from diskbridge.chemistry.shielding.visser_shielding import N_SHIELD_MIN, VisserShielding
from diskbridge.chemistry.shielding.healpix_utils import (
    integrate_rays_multi,
)

from diskbridge.chemistry.shielding.h2_db96 import h2_self_shielding_db96
import healpy as hp  # type: ignore


DEFAULT_PDR_SHIELDING_MEMORY_BUDGET_GIB = 8.0


def _chunk_size_from_memory_budget(
    *,
    n_candidates: int,
    npix: int,
    dense_ray_map_count: int,
    memory_budget_gib: float | None,
) -> int:
    if n_candidates <= 0:
        return 0
    if memory_budget_gib is None:
        memory_budget_gib = DEFAULT_PDR_SHIELDING_MEMORY_BUDGET_GIB
    budget_bytes = float(memory_budget_gib) * float(2**30)
    if not np.isfinite(budget_bytes) or budget_bytes <= 0.0:
        return int(n_candidates)

    bytes_per_candidate = (
        int(np.dtype(np.float64).itemsize)
        * int(npix)
        * max(1, int(dense_ray_map_count))
    )
    return max(1, min(int(n_candidates), int(budget_bytes // bytes_per_candidate)))


# =============================================================================
# HEALPIX CACHE UTILITIES
# =============================================================================

def _mesh_cache_info(mesh, shape: tuple[int, ...] | list[int]):
    if mesh.coord_system == "cartesian":
        x_edges = mesh.edges_f64("x", "cm")
        y_edges = mesh.edges_f64("y", "cm")
        z_edges = mesh.edges_f64("z", "cm")
        return {
            "coord": "cartesian",
            "shape": list(shape),
            "xmin": float(x_edges[0]),
            "xmax": float(x_edges[-1]),
            "ymin": float(y_edges[0]),
            "ymax": float(y_edges[-1]),
            "zmin": float(z_edges[0]),
            "zmax": float(z_edges[-1]),
        }
    if mesh.coord_system == "spherical":
        r_edges = mesh.edges_f64("r", "cm")
        th_edges = mesh.edges_f64("theta", "rad")
        ph_edges = mesh.edges_f64("phi", "rad")
        return {
            "coord": "spherical",
            "shape": list(shape),
            "rmin": float(r_edges[0]),
            "rmax": float(r_edges[-1]),
            "thmin": float(th_edges[0]),
            "thmax": float(th_edges[-1]),
            "phmin": float(ph_edges[0]),
            "phmax": float(ph_edges[-1]),
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


def _load_healpix_geometry_cache(
    cache_dir: Path | str,
    cache_key: str,
):
    loaded = _load_npz_cache(
        cache_dir,
        prefix="healpix_geom_cache",
        cache_key=cache_key,
        required_keys=("candidate_idx", "dirs", "cell_centers"),
    )
    if loaded is None:
        return None
    arrays, metadata = loaded
    return arrays["candidate_idx"], arrays["dirs"], arrays["cell_centers"], metadata


def _save_healpix_geometry_cache(
    cache_dir: Path | str,
    cache_key: str,
    candidate_idx: np.ndarray,
    dirs: np.ndarray,
    cell_centers: np.ndarray,
    metadata: dict | None = None,
) -> Path:
    return _save_npz_cache(
        cache_dir,
        prefix="healpix_geom_cache",
        cache_key=cache_key,
        arrays={
            "candidate_idx": candidate_idx,
            "dirs": dirs,
            "cell_centers": cell_centers,
        },
        metadata=metadata,
    )


def _cache_file_path(cache_dir: Path | str, prefix: str, cache_key: str) -> Path:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{prefix}_{cache_key}.npz"


def _save_npz_cache(
    cache_dir: Path | str,
    *,
    prefix: str,
    cache_key: str,
    arrays: dict[str, np.ndarray],
    metadata: dict | None = None,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = _cache_file_path(cache_dir, prefix, cache_key)

    save_dict: dict[str, np.ndarray] = dict(arrays)
    if metadata is not None:
        save_dict["metadata_json"] = np.array([json.dumps(metadata)])

    np.savez_compressed(cache_file, **save_dict)
    logger.info(f"Saved healpix cache to: {cache_file}")
    return cache_file


def _load_npz_cache(
    cache_dir: Path | str,
    *,
    prefix: str,
    cache_key: str,
    required_keys: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict | None] | None:
    cache_file = _cache_file_path(cache_dir, prefix, cache_key)

    if not cache_file.exists():
        return None

    try:
        data = np.load(cache_file)
        arrays: dict[str, np.ndarray] = {}
        for key in required_keys:
            arrays[key] = data[key]

        metadata = None
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"][0]))

        logger.info(f"Loaded healpix cache from: {cache_file}")
        return arrays, metadata
    except Exception as e:
        logger.warning(f"Failed to load healpix cache: {e}")
        return None


def _as_f64(name: str, x) -> np.ndarray:
    if hasattr(x, "magnitude") and hasattr(x, "units"):
        raise TypeError(f"{name} must be a float64 numpy array (no unit-carrying objects).")
    a = np.asarray(x, dtype=np.float64)
    if a.dtype == object:
        raise TypeError(f"{name} must be float64; got dtype=object")
    return np.ascontiguousarray(a, dtype=np.float64)


def _scatter_candidates_3d(
    target: np.ndarray,
    candidate_idx: np.ndarray,
    values: np.ndarray,
) -> None:
    """Scatter candidate-reduced values back onto a full 3D array.

    Parameters
    ----------
    target : ndarray
        Full 3D array to write into (modified in place).
    candidate_idx : ndarray
        Integer indices with shape (n_candidates, 3) giving (i0, i1, i2) for
        each candidate cell.
    values : ndarray
        Per-candidate values with shape (n_candidates,).
    """
    idx = np.asarray(candidate_idx, dtype=np.int64)
    if idx.ndim != 2 or int(idx.shape[1]) != 3:
        raise ValueError(
            f"candidate_idx must have shape (n, 3), got {idx.shape}"
        )
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(v.size) != int(idx.shape[0]):
        raise ValueError(
            f"values must have length {int(idx.shape[0])}, got {int(v.size)}"
        )
    i0 = idx[:, 0]
    i1 = idx[:, 1]
    i2 = idx[:, 2]
    target[i0, i1, i2] = v


def _add_weighted_b2_field(
    fields: dict[str, np.ndarray],
    *,
    name: str,
    density: np.ndarray,
    b_grid: Optional[np.ndarray],
    shape: tuple[int, ...],
) -> None:
    """Add ``density * b^2`` to a ray-field map when a b-grid is supplied."""
    if b_grid is None:
        return
    b_arr = _as_f64(f"b_{name}_kms_grid", b_grid)
    if b_arr.shape != tuple(shape):
        raise ValueError(f"b_{name}_kms_grid must match nH shape")
    fields[f"{name}_b2"] = np.ascontiguousarray(density * b_arr * b_arr, dtype=np.float64)


def _effective_b_from_columns(
    b2_col: np.ndarray,
    n_col: np.ndarray,
    *,
    fallback_kms: float,
) -> np.ndarray:
    """Column-weighted effective Doppler b from integrated ``n*b^2``.

    The fallback is used for optically negligible columns so array-valued H2
    shielding never sees b=0 in cells with no H2 along a ray.
    """
    n_arr = np.asarray(n_col, dtype=np.float64)
    fallback2 = float(fallback_kms) * float(fallback_kms)
    b2_eff = np.divide(
        np.asarray(b2_col, dtype=np.float64),
        n_arr,
        out=np.full_like(n_arr, fallback2),
        where=n_arr > N_SHIELD_MIN,
    )
    return np.sqrt(np.maximum(b2_eff, np.finfo(np.float64).tiny))


def _average_rays(theta_rays: np.ndarray, W_rays: Optional[np.ndarray]) -> np.ndarray:
    """Average ray shielding factors using directional weights or uniformly."""
    if W_rays is not None:
        return (W_rays * theta_rays).sum(axis=1)
    return theta_rays.mean(axis=1)



def _compute_mask_hash(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).ravel())
    return hashlib.md5(packed.tobytes()).hexdigest()[:16]


def _compute_healpix_geometry_cache_key(
    mesh,
    nside: int,
    candidate_mask: np.ndarray,
) -> str:
    mesh_info = _mesh_cache_info(mesh, candidate_mask.shape)

    params = {
        "mesh": mesh_info,
        "nside": int(nside),
        "mask_hash": _compute_mask_hash(candidate_mask),
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def _prepare_healpix_geometry(
    mesh,
    *,
    nside: int,
    candidate_mask: np.ndarray,
    cache_dir: Optional[Path | str] = None,
):
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    cache_key = None
    if cache_dir is not None:
        cache_key = _compute_healpix_geometry_cache_key(mesh, int(nside), candidate_mask)
        loaded = _load_healpix_geometry_cache(cache_dir, cache_key)
        if loaded is not None:
            candidate_idx, dirs, cell_centers, _metadata = loaded
            if mesh.coord_system == "spherical":
                tracer = SphericalHealpixRayTracer(mesh, nside=int(nside))
            elif mesh.coord_system == "cartesian":
                tracer = CartesianHealpixRayTracer(mesh, nside=int(nside))
            else:
                raise ValueError(
                    "healpix geometry requires spherical or cartesian mesh, "
                    f"got {mesh.coord_system!r}"
                )
            return (
                tracer,
                _as_f64("dirs", dirs),
                np.asarray(candidate_idx),
                _as_f64("cell_centers", cell_centers),
            )

    if mesh.coord_system == "spherical":
        tracer = SphericalHealpixRayTracer(mesh, nside=int(nside))
    elif mesh.coord_system == "cartesian":
        tracer = CartesianHealpixRayTracer(mesh, nside=int(nside))
    else:
        raise ValueError(
            "healpix geometry requires spherical or cartesian mesh, "
            f"got {mesh.coord_system!r}"
        )

    dirs = _as_f64("dirs", tracer.dirs)
    candidate_idx = np.argwhere(candidate_mask)

    n_candidates = int(candidate_idx.shape[0])
    cell_centers = np.zeros((n_candidates, 3), dtype=np.float64)
    for i in range(n_candidates):
        idx = candidate_idx[i]
        cell_centers[i, :] = tracer.cell_center_xyz(int(idx[0]), int(idx[1]), int(idx[2]))

    if cache_dir is not None and cache_key is not None:
        metadata = {
            "mesh": _mesh_cache_info(mesh, candidate_mask.shape),
            "nside": int(nside),
            "n_candidates": int(n_candidates),
            "npix": int(dirs.shape[0]),
        }
        _save_healpix_geometry_cache(
            cache_dir,
            cache_key,
            np.asarray(candidate_idx),
            np.asarray(dirs),
            np.asarray(cell_centers),
            metadata=metadata,
        )

    return tracer, dirs, candidate_idx, cell_centers


def _compute_healpix_column_cache_key(
    mesh,
    nside: int,
    n_field_cgs: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    field_name: str,
) -> str:
    mesh_info = _mesh_cache_info(mesh, n_field_cgs.shape)

    params = {
        "mesh": mesh_info,
        "nside": int(nside),
        "field": str(field_name),
        "field_hash": _compute_field_hash(n_field_cgs),
        "mask_hash": _compute_mask_hash(candidate_mask),
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def compute_L_geo_from_pathlengths(
    S_all: np.ndarray,
    reduction: str = "percentile_20",
    L_min: float = 1e10,
    L_max: float = 1e20,
) -> np.ndarray:
    """
    Compute geometric escape length per cell from ray path lengths.
    
    Parameters
    ----------
    S_all : ndarray, shape (n_cells, n_dirs)
        Path lengths to boundary for each ray direction.
    reduction : str
        Reduction method:
        - "min": minimum path length (most conservative, more cooling)
        - "percentile_10": 10th percentile
        - "percentile_20": 20th percentile (default, mimics vertical escape)
        - "percentile_30": 30th percentile
        - "harmonic": harmonic mean (penalizes long rays)
    L_min : float
        Minimum allowed L_geo. Prevents tiny values.
    L_max : float
        Maximum allowed L_geo. Caps at domain scale.
    
    Returns
    -------
    L_geo : ndarray, shape (n_cells,)
        Geometric escape length per cell.
    """
    eps = 1e-30
    
    if reduction == "min":
        L_geo = np.min(S_all, axis=1)
    elif reduction.startswith("percentile_"):
        pct = int(reduction.split("_")[1])
        L_geo = np.percentile(S_all, pct, axis=1)
    elif reduction == "harmonic":
        L_geo = 1.0 / np.mean(1.0 / np.maximum(S_all, eps), axis=1)
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")
    
    return np.clip(L_geo, L_min, L_max)


def compute_gradv_nh_weighted(
    NH_rays: np.ndarray,
    dirs: np.ndarray,
    Omega: np.ndarray,
    eR: np.ndarray,
    L_cell: np.ndarray,
    b_kms: float = 0.3,
    q: float = 1.5,
    N0: float = 1e21,
    p: float = 1.0,
    f_corr: float = 4.0,
    gmin: float = 1e-20,
    gmax: float = 1e-8,
) -> np.ndarray:
    """
    Compute per-cell velocity gradient using NH-weighted harmonic mean.
    
    Combines Keplerian shear with turbulent floor, weighted by escape probability.
    
    Parameters
    ----------
    NH_rays : ndarray, shape (n_cells, n_dirs)
        H column density along each HEALPix direction.
    dirs : ndarray, shape (n_dirs, 3)
        Direction vectors.
    Omega : ndarray, shape (n_cells,)
        Angular velocity |v_phi|/R per cell.
    eR : ndarray, shape (n_cells, 3)
        Cylindrical radial unit vector per cell.
    L_cell : ndarray, shape (n_cells,)
        Cell size per cell.
    b_kms : float
        Doppler parameter / microturbulence. Default 0.3.
    q : float
        Shear parameter. Default 1.5 (Keplerian).
    N0 : float
        Column density scale for weighting. Default 1e21.
    p : float
        Power for NH weighting. Default 1.0.
    f_corr : float
        Correlation length factor for turbulence. Default 4.0.
    gmin : float
        Minimum gradv. Default 1e-20.
    gmax : float
        Maximum gradv. Default 1e-8.
    
    Returns
    -------
    gradv : ndarray, shape (n_cells,)
        Effective velocity gradient per cell.
    
    Notes
    -----
    The effective gradient per direction is:
        g_eff = sqrt(g_shear^2 + g_turb^2)
    where:
        g_shear = q * Omega * |n . eR|  (disk shear)
        g_turb = sigma_turb / (f_corr * L_cell)  (turbulent floor)
    
    Directions are weighted by escape probability:
        w = 1 / (NH + N0)^p
    
    Final gradv is the weighted harmonic mean:
        gradv = sum(w) / sum(w / g_eff)
    """
    n_cells = NH_rays.shape[0]
    n_dirs = dirs.shape[0]
    eps = 1e-30
    
    sigma_turb = b_kms * 1e5
    
    w = 1.0 / np.power(NH_rays + N0, p)
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / np.maximum(w_sum, eps)
    
    nR = np.abs(eR @ dirs.T)
    
    g_shear = q * Omega[:, None] * nR
    
    g_turb = sigma_turb / (f_corr * np.maximum(L_cell, eps))
    
    g_eff = np.sqrt(g_shear**2 + g_turb[:, None]**2)
    
    w_over_g = w / np.maximum(g_eff, gmin)
    gradv = w.sum(axis=1) / np.maximum(w_over_g.sum(axis=1), eps)
    
    return np.clip(gradv, gmin, gmax)


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

    def __init__(
        self,
        mesh,
        nside: int,
        ds_fraction: float = 0.5,
    ):
        self.mesh = mesh
        self.nside = int(nside)

        # Axis edges in CGS / radians
        self.r_edges = mesh.edges_f64("r", "cm")
        self.theta_edges = mesh.edges_f64("theta", "rad")
        self.phi_edges = mesh.edges_f64("phi", "rad")

        # Precompute cell centers as plain arrays for speed
        self.r_centers = mesh.centers_f64("r", "cm")
        self.theta_centers = mesh.centers_f64("theta", "rad")
        self.phi_centers = mesh.centers_f64("phi", "rad")

        # Step size: fraction of the minimum dr
        dr = np.diff(self.r_edges)
        self.ds = float(ds_fraction * dr.min())

        # HEALPix directions
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
        """Return Cartesian (x, y, z) position of a cell center."""
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

        if mesh.coord_system != "cartesian":
            raise ValueError(
                f"CartesianHealpixRayTracer requires cartesian mesh, "
                f"got {mesh.coord_system!r}"
            )

        self.mesh = mesh
        self.nside = int(nside)

        # Axis edges in CGS
        self.x_edges = mesh.edges_f64("x", "cm")
        self.y_edges = mesh.edges_f64("y", "cm")
        self.z_edges = mesh.edges_f64("z", "cm")

        # Precompute cell centers as plain arrays for speed
        self.x_centers = mesh.centers_f64("x", "cm")
        self.y_centers = mesh.centers_f64("y", "cm")
        self.z_centers = mesh.centers_f64("z", "cm")

        # Step size: fraction of the minimum cell size
        dx_min = np.min(np.diff(self.x_edges))
        dy_min = np.min(np.diff(self.y_edges))
        dz_min = np.min(np.diff(self.z_edges))
        self.ds = float(ds_fraction * min(dx_min, dy_min, dz_min))

        # Precompute HEALPix directions
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
            f"nside={self.nside}, npix={npix}, ds={self.ds:.3e}"
        )

    # ------------- basic geometry helpers -------------

    def cell_center_xyz(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        """Return Cartesian (x, y, z) position of a cell center."""
        x = self.x_centers[ix]
        y = self.y_centers[iy]
        z = self.z_centers[iz]
        return float(x), float(y), float(z)


def compute_column_rays_healpix(
    mesh,
    fields: dict[str, np.ndarray],
    *,
    nside: int,
    candidate_mask: Optional[np.ndarray] = None,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    tracer: Optional[object] = None,
    dirs: Optional[np.ndarray] = None,
    candidate_idx: Optional[np.ndarray] = None,
    cell_centers: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)

    have_geom = (
        tracer is not None
        and dirs is not None
        and candidate_idx is not None
        and cell_centers is not None
    )

    if not have_geom:
        if candidate_mask is None:
            raise ValueError("candidate_mask is required if geometry is not provided.")
        tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
            mesh,
            nside=int(nside),
            candidate_mask=candidate_mask,
            cache_dir=cache_dir,
        )
    else:
        dirs = _as_f64("dirs", dirs)
        candidate_idx = np.asarray(candidate_idx)
        cell_centers = _as_f64("cell_centers", cell_centers)

    # Disable caching if no candidate_mask (chunked path)
    if candidate_mask is None:
        cache_dir = None

    out: dict[str, np.ndarray] = {}
    missing_names = []
    missing_arrays = []
    missing_cache_keys = []

    for name, field in fields.items():
        n_field_cgs = _as_f64(str(name), field)
        cache_key = None
        if cache_dir is not None and candidate_mask is not None:
            cache_key = _compute_healpix_column_cache_key(
                mesh,
                int(nside),
                n_field_cgs,
                candidate_mask,
                field_name=str(name),
            )
            cached = _load_npz_cache(
                cache_dir,
                prefix="healpix_col_cache",
                cache_key=cache_key,
                required_keys=("N_rays",),
            )
            if cached is not None:
                arrays, _metadata = cached
                out[str(name)] = arrays["N_rays"]
                continue

        missing_names.append(name)
        missing_arrays.append(n_field_cgs)
        missing_cache_keys.append(cache_key)

    if len(missing_names) >= 1:
        fields_stack = np.ascontiguousarray(np.stack(missing_arrays, axis=0), dtype=np.float64)
        n_candidates = int(cell_centers.shape[0])
        npix = int(dirs.shape[0])
        n_fields = int(fields_stack.shape[0])

        if n_candidates == 0:
            N_all = np.zeros((0, npix, n_fields), dtype=np.float64)
        elif progress_chunks is None or progress_chunks <= 1:
            N_all = integrate_rays_multi(
                tracer,
                cell_centers,
                dirs,
                fields_stack,
            )
        else:
            n_chunks = int(progress_chunks)
            if n_chunks <= 0:
                n_chunks = 1
            chunk_size = (n_candidates + n_chunks - 1) // n_chunks
            N_all = np.zeros((n_candidates, npix, n_fields), dtype=np.float64)
            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, n_candidates)
                if start >= end:
                    break
                logger.info(
                    f"Ray marching chunk {i+1}/{n_chunks} ({end-start} cells, multi-field)..."
                )
                N_all[start:end] = integrate_rays_multi(
                    tracer,
                    cell_centers[start:end],
                    dirs,
                    fields_stack,
                )

        for j, name in enumerate(missing_names):
            N_rays = N_all[:, :, j]
            out[str(name)] = N_rays
            if cache_dir is not None and missing_cache_keys[j] is not None:
                metadata = {
                    "field": str(name),
                    "n_candidates": int(cell_centers.shape[0]),
                    "nside": int(nside),
                    "npix": int(dirs.shape[0]),
                }
                _save_npz_cache(
                    cache_dir,
                    prefix="healpix_col_cache",
                    cache_key=missing_cache_keys[j],
                    arrays={"N_rays": N_rays},
                    metadata=metadata,
                )

    return candidate_idx, dirs, out


def compute_co_shielding_healpix(
    mesh,
    nH: np.ndarray,
    chi: np.ndarray,
    visser: VisserShielding,
    *,
    nCO: np.ndarray,
    nH2: np.ndarray,
    nside: int = 4,
    b_CO_kms: Optional[float] = None,
    b_CO_kms_grid: Optional[np.ndarray] = None,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    W_rays: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CO self-shielding factors and effective UV field via HEALPix rays.

    Integrates N(CO) and N(H2) along HEALPix directions from every cell,
    evaluates Visser+09 shielding factors theta_CO per ray, then averages
    over directions using the provided directional UV weights.

    When ``W_rays`` is provided, per-ray shielding factors are averaged as:

    .. math::

        \\langle\\theta\\rangle = \\sum_k W_k \\theta_k

    where :math:`W_k` are the pre-computed directional UV weights from
    :func:`~diskbridge.chemistry.shielding.angular_uv_weights.compute_uv_direction_weights_healpix`.

    When ``W_rays`` is ``None``, a uniform (isotropic) average is used.

    References: Visser et al. 2009 (A&A 503, 323) for shielding tables.

    Parameters
    ----------
    mesh : Mesh
        DiskBridge mesh (spherical or cartesian).
    nH : ndarray
        Total hydrogen number density (cm^-3), 3-D grid.
    chi : ndarray
        Dust-attenuated UV field (dimensionless, Draine units), 3-D grid.
    visser : VisserShielding
        Preloaded Visser+09 shielding table instance.
    nCO : ndarray
        CO number density (cm^-3), 3-D grid.
    nH2 : ndarray
        H2 number density (cm^-3), 3-D grid.
    nside : int, optional
        HEALPix Nside (npix = 12 * nside^2). Default 4.
    b_CO_kms : float or None, optional
        Doppler parameter in km/s for the Visser shielding table.
    progress_chunks : int or None, optional
        If set, ray integration is split into this many chunks with logging.
    cache_dir : Path or str or None, optional
        Directory for caching ray geometry and column results.
    W_rays : ndarray of shape (n_candidates, npix) or None, optional
        Per-direction UV weights. If None, uniform (isotropic) averaging.

    Returns
    -------
    theta_co : ndarray
        CO shielding factor per cell (1.0 = unshielded).
    chi_eff : ndarray
        Effective UV field = chi * theta_co.
    """
    nH_cgs = _as_f64("nH", nH)
    chi_arr = _as_f64("chi", chi)
    nCO_cgs = _as_f64("nCO", nCO)
    nH2_cgs = _as_f64("nH2", nH2)

    if nCO_cgs.shape != nH_cgs.shape or nH2_cgs.shape != nH_cgs.shape:
        raise ValueError("nCO and nH2 must match nH shape.")

    candidate_mask_arr = np.ones(nH_cgs.shape, dtype=bool)

    fields: dict[str, np.ndarray] = {"co": nCO_cgs, "h2": nH2_cgs}
    _add_weighted_b2_field(
        fields,
        name="co",
        density=nCO_cgs,
        b_grid=b_CO_kms_grid,
        shape=nH_cgs.shape,
    )

    candidate_idx, dirs, cols = compute_column_rays_healpix(
        mesh,
        fields=fields,
        nside=int(nside),
        candidate_mask=candidate_mask_arr,
        progress_chunks=progress_chunks,
        cache_dir=cache_dir,
    )

    theta_co = np.ones_like(nH_cgs, dtype=np.float64)
    if candidate_idx.shape[0] > 0:
        N_CO_rays = cols["co"]
        N_H2_rays = cols["h2"]
        if b_CO_kms_grid is not None:
            if b_CO_kms is None:
                raise ValueError("b_CO_kms is required when b_CO_kms_grid is supplied")
            b_eff = _effective_b_from_columns(
                cols["co_b2"], N_CO_rays, fallback_kms=float(b_CO_kms)
            )
            theta_rays = visser.theta_interpolated_b("co", N_CO_rays, N_H2_rays, b_eff)
        else:
            if b_CO_kms is None:
                raise ValueError("b_CO_kms is required for scalar CO shielding")
            theta_rays = visser.theta("co", N_CO_rays, N_H2_rays, b_kms=float(b_CO_kms))

        theta_eff = _average_rays(theta_rays, W_rays)

        _scatter_candidates_3d(theta_co, candidate_idx, theta_eff)

    chi_eff = chi_arr * theta_co
    return theta_co, chi_eff


def compute_pdr_shielding_healpix(
    mesh,
    nH: np.ndarray,
    chi: np.ndarray,
    visser: Optional[VisserShielding] = None,
    *,
    nCO: Optional[np.ndarray] = None,
    nC: np.ndarray,
    nH2: np.ndarray,
    nside: int = 4,
    b_H2_kms: float = 0.3,
    b_CO_kms: float = 0.3,
    b_H2_kms_grid: Optional[np.ndarray] = None,
    b_CO_kms_grid: Optional[np.ndarray] = None,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    W_rays: Optional[np.ndarray] = None,
    chunk_size: Optional[int] = None,
    memory_budget_gib: Optional[float] = None,
) -> tuple[np.ndarray, ...]:
    """Compute PDR shielding factors (H2, CO, C) and effective UV via HEALPix rays.

    Integrates N(H2), N(C), and optionally N(CO) along HEALPix directions,
    evaluates per-species self-shielding factors, then averages over directions
    using the provided directional UV weights.

    When ``W_rays`` is provided, per-ray shielding factors are averaged as:

    .. math::

        \\langle f \\rangle = \\sum_k W_k f_k

    where :math:`W_k` are the pre-computed directional UV weights from
    :func:`~diskbridge.chemistry.shielding.angular_uv_weights.compute_uv_direction_weights_healpix`.

    When ``W_rays`` is ``None``, a uniform (isotropic) average is used.

    References
    ----------
    - Visser et al. 2009, A&A 503, 323 (CO shielding tables)
    - Draine & Bertoldi 1996, ApJ, 468, 269 (H2 self-shielding)

    Parameters
    ----------
    mesh : Mesh
        DiskBridge mesh (spherical or cartesian).
    nH : ndarray
        Total hydrogen number density (cm^-3), 3-D grid.
    chi : ndarray
        Dust-attenuated UV field (dimensionless, Draine units), 3-D grid.
    visser : VisserShielding or None, optional
        Preloaded Visser+09 shielding table instance. If provided, CO
        shielding is computed alongside H2 shielding.
    nCO : ndarray or None, optional
        CO number density (cm^-3). Required if ``visser`` is provided.
    nC : ndarray
        Atomic carbon number density (cm^-3), 3-D grid.
    nH2 : ndarray
        H2 number density (cm^-3), 3-D grid.
    nside : int, optional
        HEALPix Nside. Default 4.
    b_H2_kms, b_CO_kms : float, optional
        Representative Doppler parameters in km/s. These are used directly
        when the corresponding b-grid is absent, and as safe fallbacks for
        zero-column rays when a column-weighted effective b is computed.
    progress_chunks : int or None, optional
        If set, split ray integration into chunks with logging.
    cache_dir : Path or str or None, optional
        Directory for caching ray geometry and column results.
    W_rays : ndarray of shape (n_candidates, npix) or None, optional
        Per-direction UV weights. If None, uniform (isotropic) averaging.
    chunk_size : int or None, optional
        Number of candidate cells to process per shielding chunk. If omitted,
        a chunk size is derived from ``memory_budget_gib``.
    memory_budget_gib : float or None, optional
        Approximate working-memory budget for one shielding chunk. Defaults to
        a conservative internal budget because ``W_rays`` may already be
        resident.
    Returns
    -------
    theta_h2 : ndarray
        H2 self-shielding factor per cell.
    theta_co : ndarray
        CO shielding factor per cell (1.0 if visser is None).
    theta_c : ndarray
        C photoionization shielding factor per cell
        (van Dishoeck & Black 1988; Tielens & Hollenbach 1985).
    theta_pdr : ndarray
        Combined PDR shielding factor = theta_h2 * theta_co.
    chi_eff_pdr : ndarray
        Effective UV field = chi * theta_pdr.
    """
    nH_cgs = _as_f64("nH", nH)
    chi_arr = _as_f64("chi", chi)
    nH2_cgs = _as_f64("nH2", nH2)
    nC_cgs = _as_f64("nC", nC)

    fields: dict[str, np.ndarray] = {"h2": nH2_cgs, "c": nC_cgs}
    _add_weighted_b2_field(
        fields,
        name="h2",
        density=nH2_cgs,
        b_grid=b_H2_kms_grid,
        shape=nH_cgs.shape,
    )
    if visser is not None:
        if nCO is None:
            raise ValueError("nCO is required if visser is provided")
        nCO_cgs = _as_f64("nCO", nCO)
        fields["co"] = nCO_cgs
        _add_weighted_b2_field(
            fields,
            name="co",
            density=nCO_cgs,
            b_grid=b_CO_kms_grid,
            shape=nH_cgs.shape,
        )

    candidate_mask_arr = np.ones(nH_cgs.shape, dtype=bool)
    tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
        mesh,
        nside=int(nside),
        candidate_mask=candidate_mask_arr,
        cache_dir=cache_dir,
    )

    theta_h2 = np.ones_like(nH_cgs, dtype=np.float64)
    theta_c = np.ones_like(nH_cgs, dtype=np.float64)
    theta_co = np.ones_like(nH_cgs, dtype=np.float64)
    theta_pdr = np.ones_like(nH_cgs, dtype=np.float64)

    n_candidates = int(candidate_idx.shape[0])
    npix = int(dirs.shape[0])
    if W_rays is not None:
        W_rays = _as_f64("W_rays", W_rays)
        if W_rays.shape != (n_candidates, npix):
            raise ValueError(
                f"W_rays must have shape ({n_candidates}, {npix}), got {W_rays.shape}"
            )

    if n_candidates > 0:
        # Dense ray maps alive in a chunk: integrated columns, H2 factors, C
        # factors, optional CO/PDR factors, and W_rays multiplication inputs.
        linewidth_field_count = int(b_H2_kms_grid is not None) + int(
            visser is not None and b_CO_kms_grid is not None
        )
        dense_count = (8 if visser is None else 12) + 2 * linewidth_field_count
        if chunk_size is None:
            chunk_size = _chunk_size_from_memory_budget(
                n_candidates=n_candidates,
                npix=npix,
                dense_ray_map_count=dense_count,
                memory_budget_gib=memory_budget_gib,
            )
        else:
            chunk_size = int(chunk_size)
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
        chunk_size = max(1, min(int(chunk_size), n_candidates))
        n_chunks = (n_candidates + chunk_size - 1) // chunk_size
        logger.info(
            "PDR HEALPix shielding: processing %d candidates in %d chunk(s) "
            "(chunk_size=%d, npix=%d, fields=%d)",
            n_candidates,
            n_chunks,
            chunk_size,
            npix,
            len(fields),
        )

    if n_candidates == 0:
        chi_eff_pdr = chi_arr * theta_pdr
        return theta_h2, theta_co, theta_c, theta_pdr, chi_eff_pdr

    for ichunk, start in enumerate(range(0, n_candidates, int(chunk_size)), start=1):
        end = min(start + int(chunk_size), n_candidates)
        candidate_idx_chunk = candidate_idx[start:end]
        cell_centers_chunk = cell_centers[start:end]
        if n_chunks > 1:
            logger.info(
                "PDR HEALPix shielding chunk %d/%d (%d cells)",
                ichunk,
                n_chunks,
                end - start,
            )

        _, _, cols = compute_column_rays_healpix(
            mesh,
            fields=fields,
            nside=int(nside),
            candidate_mask=None,
            progress_chunks=progress_chunks,
            cache_dir=None,
            tracer=tracer,
            dirs=dirs,
            candidate_idx=candidate_idx_chunk,
            cell_centers=cell_centers_chunk,
        )

        N_H2_rays = cols["h2"]
        N_C_rays = cols["c"]

        if b_H2_kms_grid is not None:
            b_H2_eff = _effective_b_from_columns(
                cols["h2_b2"], N_H2_rays, fallback_kms=float(b_H2_kms)
            )
        else:
            b_H2_eff = float(b_H2_kms)
        f_sh_rays = h2_self_shielding_db96(N_H2_rays, b5=b_H2_eff)

        # H2 shielding
        W_chunk = None if W_rays is None else W_rays[start:end]
        theta_h2_mean = _average_rays(f_sh_rays, W_chunk)
        _scatter_candidates_3d(theta_h2, candidate_idx_chunk, theta_h2_mean)

        # C self-shielding 
        # References: van Dishoeck & Black 1988; Tielens & Hollenbach 1985
        # theta_c = exp(-sigma_C * N_C) * exp(-y)/(1+y)
        # where y = AH2 * tau_H2, tau_H2 = sigma_H2 * 2 * N_H2
        AH2 = 1.17e-8
        tau_H2 = 1.2e-14 * 2.0 * N_H2_rays
        y = AH2 * tau_H2
        ry = np.exp(-y) / (1.0 + y)
        rc = np.exp(-1.6e-17 * N_C_rays)
        theta_c_rays = rc * ry

        theta_c_mean = _average_rays(theta_c_rays, W_chunk)
        _scatter_candidates_3d(theta_c, candidate_idx_chunk, theta_c_mean)

        theta_pdr_mean = theta_h2_mean
        _scatter_candidates_3d(theta_pdr, candidate_idx_chunk, theta_h2_mean)

        if visser is not None:
            N_CO_rays = cols["co"]
            if b_CO_kms_grid is not None:
                b_eff = _effective_b_from_columns(
                    cols["co_b2"], N_CO_rays, fallback_kms=float(b_CO_kms)
                )
                theta_co_rays = visser.theta_interpolated_b(
                    "co",
                    N_CO_rays,
                    N_H2_rays,
                    b_eff,
                )
            else:
                theta_co_rays = visser.theta(
                    "co", N_CO_rays, N_H2_rays, b_kms=float(b_CO_kms)
                )

            theta_co_mean = _average_rays(theta_co_rays, W_chunk)
            _scatter_candidates_3d(theta_co, candidate_idx_chunk, theta_co_mean)

            theta_pdr_rays = f_sh_rays * theta_co_rays
            theta_pdr_mean = _average_rays(theta_pdr_rays, W_chunk)
            _scatter_candidates_3d(theta_pdr, candidate_idx_chunk, theta_pdr_mean)

    chi_eff_pdr = chi_arr * theta_pdr
    return theta_h2, theta_co, theta_c, theta_pdr, chi_eff_pdr
