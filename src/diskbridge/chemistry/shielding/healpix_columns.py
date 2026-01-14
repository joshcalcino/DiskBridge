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
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
from diskbridge.chemistry.shielding.healpix_utils import (
    integrate_rays_multi,
)

from diskbridge._constants import LOG_CHI_OVER_NH_PDISS, EPS_CHI


def _get_healpy():
    try:
        import healpy as hp  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "healpy is required for HEALPix shielding. Install with: pip install healpy"
        ) from e
    return hp


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
    a = np.asarray(x, dtype=np.float64)
    return a


def _prepare_candidate_mask(
    nH_cgs: np.ndarray,
    chi_arr: np.ndarray,
    candidate_mask: Optional[np.ndarray],
) -> np.ndarray:
    if candidate_mask is None:
        ratio = np.where(nH_cgs > 0.0, chi_arr / nH_cgs, 0.0)
        log_ratio = np.log10(np.maximum(ratio, EPS_CHI))
        return log_ratio > LOG_CHI_OVER_NH_PDISS
    return np.asarray(candidate_mask, dtype=bool)


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
        Minimum allowed L_geo (cm). Prevents tiny values.
    L_max : float
        Maximum allowed L_geo (cm). Caps at domain scale.
    
    Returns
    -------
    L_geo : ndarray, shape (n_cells,)
        Geometric escape length per cell (cm).
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
        H column density along each HEALPix direction (cm^-2).
    dirs : ndarray, shape (n_dirs, 3)
        Unit direction vectors (Cartesian).
    Omega : ndarray, shape (n_cells,)
        Angular velocity |v_phi|/R (s^-1) per cell.
    eR : ndarray, shape (n_cells, 3)
        Cylindrical radial unit vector per cell.
    L_cell : ndarray, shape (n_cells,)
        Cell size (cm) per cell.
    b_kms : float
        Doppler parameter / microturbulence (km/s). Default 0.3.
    q : float
        Shear parameter. Default 1.5 (Keplerian).
    N0 : float
        Column density scale for weighting (cm^-2). Default 1e21.
    p : float
        Power for NH weighting. Default 1.0.
    f_corr : float
        Correlation length factor for turbulence. Default 4.0.
    gmin : float
        Minimum gradv (s^-1). Default 1e-20.
    gmax : float
        Maximum gradv (s^-1). Default 1e-8.
    
    Returns
    -------
    gradv : ndarray, shape (n_cells,)
        Effective velocity gradient per cell (s^-1).
    
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
        hp = _get_healpy()
        self.mesh = mesh
        self.nside = int(nside)

        # Axis edges in CGS / radians
        r_edges_q = mesh.edges("r")
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

        hp = _get_healpy()

        if mesh.coord_system != "cartesian":
            raise ValueError(
                f"CartesianHealpixRayTracer requires cartesian mesh, "
                f"got {mesh.coord_system!r}"
            )

        self.mesh = mesh
        self.nside = int(nside)

        # Axis edges in CGS
        x_edges_q = mesh.edges("x")
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


def compute_column_rays_healpix(
    mesh,
    fields: dict[str, np.ndarray],
    *,
    nside: int,
    candidate_mask: Optional[np.ndarray] = None,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    self_weight: float = 1.0,
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
                self_weight=float(self_weight),
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
                    self_weight=float(self_weight),
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
    nH,
    chi,
    visser: VisserShielding,
    *,
    nCO=None,
    nH2=None,
    nside: int = 4,
    b_kms: Optional[float] = None,
    candidate_mask: Optional[np.ndarray] = None,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    self_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    nH_cgs = _as_f64("nH", nH)
    chi_arr = _as_f64("chi", chi)  # chi is in Draine units

    if nCO is None:
        raise ValueError("compute_co_shielding_healpix requires nCO")
    nCO_cgs = _as_f64("nCO", nCO)

    if nH2 is None:
        raise ValueError("compute_co_shielding_healpix requires nH2")
    nH2_cgs = _as_f64("nH2", nH2)

    if nCO_cgs.shape != nH_cgs.shape or nH2_cgs.shape != nH_cgs.shape:
        raise ValueError("nCO and nH2 must match nH shape.")

    candidate_mask_arr = _prepare_candidate_mask(nH_cgs, chi_arr, candidate_mask)

    candidate_idx, dirs, cols = compute_column_rays_healpix(
        mesh,
        fields={"co": nCO_cgs, "h2": nH2_cgs},
        nside=int(nside),
        candidate_mask=candidate_mask_arr,
        progress_chunks=progress_chunks,
        cache_dir=cache_dir,
        self_weight=float(self_weight),
    )

    theta_co = np.ones_like(nH_cgs, dtype=np.float64)
    if candidate_idx.shape[0] > 0:
        N_CO_rays = cols["co"]
        N_H2_rays = cols["h2"]
        theta_rays = visser.theta("co", N_CO_rays, N_H2_rays, b_kms=b_kms)
        theta_mean = theta_rays.mean(axis=1)
        _scatter_candidates_3d(theta_co, candidate_idx, theta_mean)

    chi_eff = chi_arr * theta_co
    return theta_co, chi_eff


def compute_pdr_shielding_healpix(
    mesh,
    nH,
    chi,
    visser: Optional[VisserShielding] = None,
    *,
    nCO=None,
    nC=None,
    nH2=None,
    nside: int = 4,
    b_kms: float = 0.3,
    candidate_mask: Optional[np.ndarray] = None,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    self_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nH_cgs = _as_f64("nH", nH)
    chi_arr = _as_f64("chi", chi)

    if nH2 is None:
        raise ValueError("compute_pdr_shielding_healpix requires nH2")
    nH2_cgs = _as_f64("nH2", nH2)

    if nC is None:
        raise ValueError("compute_pdr_shielding_healpix requires nC")
    nC_cgs = _as_f64("nC", nC)

    fields = {"h2": nH2_cgs, "c": nC_cgs}
    if visser is not None:
        if nCO is None:
            raise ValueError("nCO is required if visser is provided")
        fields["co"] = _as_f64("nCO", nCO)

    candidate_mask_arr = _prepare_candidate_mask(nH_cgs, chi_arr, candidate_mask)

    candidate_idx, dirs, cols = compute_column_rays_healpix(
        mesh,
        fields=fields,
        nside=int(nside),
        candidate_mask=candidate_mask_arr,
        progress_chunks=progress_chunks,
        cache_dir=cache_dir,
        self_weight=float(self_weight),
    )

    theta_h2 = np.ones_like(nH_cgs, dtype=np.float64)
    theta_c = np.ones_like(nH_cgs, dtype=np.float64)
    theta_co = np.ones_like(nH_cgs, dtype=np.float64)
    theta_pdr = np.ones_like(nH_cgs, dtype=np.float64)

    if candidate_idx.shape[0] > 0:
        N_H2_rays = cols["h2"]
        N_C_rays = cols["c"]

        f_sh_rays = h2_self_shielding_db96(N_H2_rays, b5=float(b_kms))
        theta_h2_mean = f_sh_rays.mean(axis=1)
        _scatter_candidates_3d(theta_h2, candidate_idx, theta_h2_mean)

        theta_pdr_mean = theta_h2_mean  # baseline
        _scatter_candidates_3d(theta_pdr, candidate_idx, theta_h2_mean)

        if visser is not None:
            N_CO_rays = cols["co"]
            theta_co_rays = visser.theta("co", N_CO_rays, N_H2_rays, b_kms=b_kms)
            theta_co_mean = theta_co_rays.mean(axis=1)
            _scatter_candidates_3d(theta_co, candidate_idx, theta_co_mean)

            theta_pdr_rays = f_sh_rays * theta_co_rays
            theta_pdr_mean = theta_pdr_rays.mean(axis=1)
            _scatter_candidates_3d(theta_pdr, candidate_idx, theta_pdr_mean)

    chi_eff_pdr = chi_arr * theta_pdr
    return theta_h2, theta_co, theta_c, theta_pdr, chi_eff_pdr
