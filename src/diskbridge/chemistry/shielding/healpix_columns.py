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


def save_healpix_cache(
    cache_dir: Path | str,
    cache_key: str,
    theta_co: np.ndarray,
    chi_eff: np.ndarray,
    metadata: dict | None = None,
) -> Path:
    cache_file = _save_npz_cache(
        cache_dir,
        prefix="healpix_cache",
        cache_key=cache_key,
        arrays={
            "theta_co": theta_co,
            "chi_eff": chi_eff,
        },
        metadata=metadata,
    )
    return cache_file


def _load_healpix_geometry_cache(
    cache_dir: Path | str,
    cache_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None] | None:
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


def load_healpix_cache(
    cache_dir: Path | str,
    cache_key: str,
) -> tuple[np.ndarray, np.ndarray, dict | None] | None:
    loaded = _load_npz_cache(
        cache_dir,
        prefix="healpix_cache",
        cache_key=cache_key,
        required_keys=("theta_co", "chi_eff"),
    )
    if loaded is None:
        return None
    arrays, metadata = loaded
    return arrays["theta_co"], arrays["chi_eff"], metadata


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


def _integrate_field_rays(
    tracer,
    cell_centers: np.ndarray,
    dirs: np.ndarray,
    n_field_cgs: np.ndarray,
    *,
    progress_chunks: Optional[int],
) -> np.ndarray:
    n_candidates = cell_centers.shape[0]
    npix = dirs.shape[0]

    if n_candidates == 0:
        return np.empty((0, npix), dtype=np.float64)

    if progress_chunks is None or progress_chunks <= 1:
        return integrate_rays(tracer, cell_centers, dirs, n_field_cgs)

    n_chunks = int(progress_chunks)
    if n_chunks <= 0:
        n_chunks = 1
    chunk_size = max(1, n_candidates // n_chunks)

    logger.info(
        f"HEALPix ray tracing progress enabled: {n_candidates} cells "
        f"in {n_chunks} chunks (chunk_size={chunk_size})."
    )

    N_all = np.empty((n_candidates, npix), dtype=np.float64)
    _t_chunk_start = _time.time()
    for ichunk, start in enumerate(range(0, n_candidates, chunk_size), 1):
        end = min(start + chunk_size, n_candidates)
        N_chunk = integrate_rays(
            tracer,
            cell_centers[start:end],
            dirs,
            n_field_cgs,
        )
        N_all[start:end] = N_chunk

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

    return N_all


def _prepare_healpix_geometry(
    mesh,
    *,
    nside: int,
    candidate_mask: np.ndarray,
    cache_dir: Optional[Path | str],
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    candidate_mask = np.asarray(candidate_mask, dtype=bool)

    geom_cache_key = None
    cached_geom = None
    if cache_dir is not None:
        geom_cache_key = _compute_healpix_geometry_cache_key(mesh, int(nside), candidate_mask)
        cached_geom = _load_healpix_geometry_cache(cache_dir, geom_cache_key)

    if mesh.coord_system == "spherical":
        tracer = SphericalHealpixRayTracer(mesh, nside=int(nside))
    elif mesh.coord_system == "cartesian":
        tracer = CartesianHealpixRayTracer(mesh, nside=int(nside))
    else:
        raise ValueError(
            f"Unsupported mesh coord_system {mesh.coord_system!r} for HEALPix rays "
            "(expected 'spherical' or 'cartesian')."
        )

    if cached_geom is not None:
        candidate_idx, dirs, cell_centers, metadata = cached_geom
        if metadata:
            logger.info(
                f"Loaded healpix geometry from cache (computed in {metadata.get('compute_time_s', '?')}s)"
            )
        return tracer, np.asarray(dirs, dtype=np.float64), np.asarray(candidate_idx), np.asarray(cell_centers, dtype=np.float64)

    candidate_idx = np.argwhere(candidate_mask)
    n_candidates = candidate_idx.shape[0]

    dirs = tracer.dirs.astype(np.float64)
    cell_centers = np.zeros((n_candidates, 3), dtype=np.float64)
    for k, idx in enumerate(candidate_idx):
        cell_centers[k] = tracer.cell_center_xyz(*idx)

    if cache_dir is not None and geom_cache_key is not None:
        metadata = {
            "compute_time_s": 0.0,
            "n_candidates": int(n_candidates),
            "nside": int(nside),
            "npix": int(dirs.shape[0]),
        }
        _save_healpix_geometry_cache(
            cache_dir,
            geom_cache_key,
            candidate_idx,
            dirs,
            cell_centers,
            metadata,
        )

    return tracer, dirs, candidate_idx, cell_centers


def _scatter_candidates_3d(
    grid: np.ndarray,
    candidate_idx: np.ndarray,
    values: np.ndarray,
) -> None:
    ci = candidate_idx[:, 0]
    cj = candidate_idx[:, 1]
    ck = candidate_idx[:, 2]
    grid[ci, cj, ck] = values


def compute_column_rays_healpix(
    mesh,
    fields: dict[str, np.ndarray],
    *,
    nside: int,
    candidate_mask: np.ndarray,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    candidate_mask = np.asarray(candidate_mask, dtype=bool)

    any_field = next(iter(fields.values()))
    if np.shape(any_field) != np.shape(candidate_mask):
        raise ValueError(
            f"field and candidate_mask must have the same shape, got {np.shape(any_field)} vs {candidate_mask.shape}"
        )

    tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
        mesh,
        nside=int(nside),
        candidate_mask=candidate_mask,
        cache_dir=cache_dir,
    )

    out: dict[str, np.ndarray] = {}
    for name, field in fields.items():
        n_field_cgs = np.asarray(field, dtype=np.float64)
        if n_field_cgs.shape != candidate_mask.shape:
            raise ValueError(
                f"field {name!r} must match candidate_mask shape, got {n_field_cgs.shape} vs {candidate_mask.shape}"
            )

        cache_key = None
        if cache_dir is not None:
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

        _t_start = _time.time()
        N_rays = _integrate_field_rays(
            tracer,
            cell_centers,
            dirs,
            n_field_cgs,
            progress_chunks=progress_chunks,
        )
        _t_elapsed = _time.time() - _t_start

        if cache_dir is not None and cache_key is not None:
            metadata = {
                "compute_time_s": round(_t_elapsed, 2),
                "field": str(name),
                "n_candidates": int(cell_centers.shape[0]),
                "nside": int(nside),
                "npix": int(dirs.shape[0]),
            }
            _save_npz_cache(
                cache_dir,
                prefix="healpix_col_cache",
                cache_key=cache_key,
                arrays={"N_rays": N_rays},
                metadata=metadata,
            )

        out[str(name)] = N_rays

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

    nH_cgs = _to_ndarray_cgs(nH, "cm^-3")
    chi_arr = _to_ndarray_cgs(chi, "dimensionless")  # chi is in Draine units

    if nCO is None:
        raise ValueError("compute_co_shielding_healpix requires nCO (no abundance scaling assumptions)")
    nCO_cgs = _to_ndarray_cgs(nCO, "cm^-3")

    if nH2 is None:
        raise ValueError("compute_co_shielding_healpix requires nH2 (no abundance scaling assumptions)")
    nH2_cgs = _to_ndarray_cgs(nH2, "cm^-3")

    if nCO_cgs.shape != nH_cgs.shape or nH2_cgs.shape != nH_cgs.shape:
        raise ValueError("nCO and nH2 must match nH shape.")

    candidate_mask_arr = _prepare_candidate_mask(nH_cgs, chi_arr, candidate_mask)

    # Check for cached results
    cache_key = None
    if cache_dir is not None:
        params = {
            "mesh": _mesh_cache_info(mesh, nH_cgs.shape),
            "nside": int(nside),
            "nH_hash": _compute_field_hash(nH_cgs),
            "chi_hash": _compute_field_hash(chi_arr),
            "nCO_hash": _compute_field_hash(nCO_cgs),
            "nH2_hash": _compute_field_hash(nH2_cgs),
            "b_kms": None if b_kms is None else float(b_kms),
            "mask_hash": _compute_mask_hash(candidate_mask_arr),
            "log_threshold": LOG_CHI_OVER_NH_PDISS,
        }
        cache_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
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

    tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
        mesh,
        nside=int(nside),
        candidate_mask=candidate_mask_arr,
        cache_dir=cache_dir,
    )
    n_candidates = int(candidate_idx.shape[0])

    logger.info(
        f"compute_co_shielding_healpix: {n_candidates} candidate cells "
        f"(log10(chi/nH) > {LOG_CHI_OVER_NH_PDISS:.2f})."
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

    if progress_chunks is None or progress_chunks <= 1:
        chunk_size = n_candidates
    else:
        n_chunks = int(progress_chunks)
        if n_chunks <= 0:
            n_chunks = 1
        chunk_size = max(1, n_candidates // n_chunks)

    for start in range(0, n_candidates, chunk_size):
        end = min(start + chunk_size, n_candidates)
        idx_chunk = candidate_idx[start:end]
        centers_chunk = cell_centers[start:end]

        N_CO_rays = integrate_rays(tracer, centers_chunk, dirs, nCO_cgs)
        N_H2_rays = integrate_rays(tracer, centers_chunk, dirs, nH2_cgs)

        theta_rays = visser.theta("co", N_CO_rays, N_H2_rays, b_kms=b_kms)
        theta_mean = theta_rays.mean(axis=1)
        _scatter_candidates_3d(theta_co, idx_chunk, theta_mean)

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
    nC=None,
    nH2=None,
    nside: int = 4,
    b_kms: Optional[float] = None,
    candidate_mask: Optional[np.ndarray] = None,
    progress_chunks: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    return_quantity: bool = True,
) -> tuple[Quantity, Quantity, Quantity, Quantity] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nH_cgs = _to_ndarray_cgs(nH, "cm^-3")
    chi_arr = _to_ndarray_cgs(chi, "dimensionless")

    if nH2 is None:
        raise ValueError("compute_pdr_shielding_healpix requires nH2 (no abundance scaling assumptions)")
    nH2_cgs = _to_ndarray_cgs(nH2, "cm^-3")

    if nC is None:
        raise ValueError("compute_pdr_shielding_healpix requires nC (no abundance scaling assumptions)")
    nC_cgs = _to_ndarray_cgs(nC, "cm^-3")

    if visser is not None and nCO is None:
        raise ValueError("compute_pdr_shielding_healpix requires nCO when visser is provided")
    nCO_cgs = None if nCO is None else _to_ndarray_cgs(nCO, "cm^-3")
    if nH2_cgs.shape != nH_cgs.shape or nC_cgs.shape != nH_cgs.shape:
        raise ValueError("nC and nH2 must match nH shape.")
    if nCO_cgs is not None and nCO_cgs.shape != nH_cgs.shape:
        raise ValueError("nCO must match nH shape.")

    candidate_mask_arr = _prepare_candidate_mask(nH_cgs, chi_arr, candidate_mask)

    tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
        mesh,
        nside=int(nside),
        candidate_mask=candidate_mask_arr,
        cache_dir=cache_dir,
    )

    theta_h2 = np.ones_like(nH_cgs, dtype=np.float64)
    theta_co = np.ones_like(nH_cgs, dtype=np.float64)
    theta_c = np.ones_like(nH_cgs, dtype=np.float64)

    n_candidates = int(candidate_idx.shape[0])
    if n_candidates > 0:
        from diskbridge.chemistry.hydrogen.partition import _h2_self_shielding_db96

        if b_kms is None:
            raise ValueError("b_kms is required for H2 self-shielding")

        if progress_chunks is None or progress_chunks <= 1:
            chunk_size = n_candidates
        else:
            n_chunks = int(progress_chunks)
            if n_chunks <= 0:
                n_chunks = 1
            chunk_size = max(1, n_candidates // n_chunks)

        for start in range(0, n_candidates, chunk_size):
            end = min(start + chunk_size, n_candidates)
            idx_chunk = candidate_idx[start:end]
            centers_chunk = cell_centers[start:end]

            N_H2_rays = integrate_rays(tracer, centers_chunk, dirs, nH2_cgs)
            f_sh_rays = _h2_self_shielding_db96(N_H2_rays, b5=float(b_kms), alpha=-0.75)
            theta_h2_mean = f_sh_rays.mean(axis=1)
            _scatter_candidates_3d(theta_h2, idx_chunk, theta_h2_mean)

            N_C_rays = integrate_rays(tracer, centers_chunk, dirs, nC_cgs)

            AH2 = 1.17e-8
            tau_H2 = 1.2e-14 * 2.0 * N_H2_rays
            y = AH2 * tau_H2
            ry = np.exp(-y) / (1.0 + y)
            rc = np.exp(-1.6e-17 * N_C_rays)
            theta_c_rays = rc * ry
            theta_c_mean = theta_c_rays.mean(axis=1)
            _scatter_candidates_3d(theta_c, idx_chunk, theta_c_mean)

            if visser is not None:
                N_CO_rays = integrate_rays(tracer, centers_chunk, dirs, nCO_cgs)
                theta_co_rays = visser.theta("co", N_CO_rays, N_H2_rays, b_kms=b_kms)
                theta_co_mean = theta_co_rays.mean(axis=1)
                _scatter_candidates_3d(theta_co, idx_chunk, theta_co_mean)

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
