from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _integrate_ray_cartesian(
    x0: float,
    y0: float,
    z0: float,
    dx: float,
    dy: float,
    dz: float,
    ds: float,
    n_field: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    max_steps: int = 10000,
) -> float:
    x = x0
    y = y0
    z = z0
    N = 0.0
    nx = n_field.shape[0]
    ny = n_field.shape[1]
    nz = n_field.shape[2]

    for i in range(max_steps):
        if x <= xmin or x >= xmax or y <= ymin or y >= ymax or z <= zmin or z >= zmax:
            break

        ix = np.searchsorted(x_edges, x) - 1
        iy = np.searchsorted(y_edges, y) - 1
        iz = np.searchsorted(z_edges, z) - 1

        N += n_field[ix, iy, iz] * ds

        x += dx * ds
        y += dy * ds
        z += dz * ds


    return N


@njit(cache=True, parallel=True)
def _integrate_all_rays_cartesian(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    nCO_field: np.ndarray,
    nH2_field: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    Nco_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    Nh2_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            Nco_all[i, j] = _integrate_ray_cartesian(
                x0,
                y0,
                z0,
                dx,
                dy,
                dz,
                ds,
                nCO_field,
                x_edges,
                y_edges,
                z_edges,
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
                max_steps,
            )
            Nh2_all[i, j] = _integrate_ray_cartesian(
                x0,
                y0,
                z0,
                dx,
                dy,
                dz,
                ds,
                nH2_field,
                x_edges,
                y_edges,
                z_edges,
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
                max_steps,
            )

    return Nco_all, Nh2_all


@njit(cache=True, parallel=True)
def _integrate_all_rays_cartesian_single(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    n_field: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    max_steps: int = 10000,
) -> np.ndarray:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    N_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            N_all[i, j] = _integrate_ray_cartesian(
                x0,
                y0,
                z0,
                dx,
                dy,
                dz,
                ds,
                n_field,
                x_edges,
                y_edges,
                z_edges,
                xmin,
                xmax,
                ymin,
                ymax,
                zmin,
                zmax,
                max_steps,
            )

    return N_all


@njit(cache=True, parallel=True)
def _integrate_all_rays_cartesian_single_v1_inline(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    n_field: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    max_steps: int = 10000,
) -> np.ndarray:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    N_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            x = x0
            y = y0
            z = z0
            N = 0.0

            for _ in range(max_steps):
                if (
                    x <= xmin
                    or x >= xmax
                    or y <= ymin
                    or y >= ymax
                    or z <= zmin
                    or z >= zmax
                ):
                    break

                ix = np.searchsorted(x_edges, x) - 1
                iy = np.searchsorted(y_edges, y) - 1
                iz = np.searchsorted(z_edges, z) - 1

                N += n_field[ix, iy, iz] * ds

                x += dx * ds
                y += dy * ds
                z += dz * ds

            N_all[i, j] = N

    return N_all


@njit(cache=True, parallel=True)
def _integrate_all_rays_cartesian_v1_inline(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    nCO_field: np.ndarray,
    nH2_field: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    Nco_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    Nh2_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            x = x0
            y = y0
            z = z0
            Nco = 0.0
            for _ in range(max_steps):
                if x <= xmin or x >= xmax or y <= ymin or y >= ymax or z <= zmin or z >= zmax:
                    break
                ix = np.searchsorted(x_edges, x) - 1
                iy = np.searchsorted(y_edges, y) - 1
                iz = np.searchsorted(z_edges, z) - 1
                Nco += nCO_field[ix, iy, iz] * ds
                x += dx * ds
                y += dy * ds
                z += dz * ds

            Nco_all[i, j] = Nco

            x = x0
            y = y0
            z = z0
            Nh2 = 0.0
            for _ in range(max_steps):
                if x <= xmin or x >= xmax or y <= ymin or y >= ymax or z <= zmin or z >= zmax:
                    break
                ix = np.searchsorted(x_edges, x) - 1
                iy = np.searchsorted(y_edges, y) - 1
                iz = np.searchsorted(z_edges, z) - 1
                Nh2 += nH2_field[ix, iy, iz] * ds
                x += dx * ds
                y += dy * ds
                z += dz * ds

            Nh2_all[i, j] = Nh2

    return Nco_all, Nh2_all


# =============================================================================
# UNIFORM GRID VARIANTS (no searchsorted - arithmetic index calculation)
# =============================================================================
# These variants are optimized for uniform Cartesian grids where cell spacing
# is constant. They replace np.searchsorted with simple arithmetic:
#   ix = int((x - xmin) * inv_dx)
# This can provide significant speedup since searchsorted has O(log n) cost
# per call while arithmetic indexing is O(1).


@njit(cache=True)
def _integrate_ray_cartesian_uniform(
    x0: float,
    y0: float,
    z0: float,
    dx: float,
    dy: float,
    dz: float,
    ds: float,
    n_field: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    inv_dx: float,
    inv_dy: float,
    inv_dz: float,
    max_steps: int = 10000,
) -> float:
    """Ray integration for uniform Cartesian grids using arithmetic indexing.
    
    Provides ~8-10x speedup over searchsorted-based methods by using O(1)
    arithmetic indexing instead of O(log n) binary search.
    
    KNOWN LIMITATION: Has boundary precision issues when ray positions fall
    exactly on cell edges. This causes up to ~20% max relative error for
    individual rays, though mean error is typically <3% and median <1%.
    
    For production use requiring exact accuracy, prefer the searchsorted-based
    _integrate_ray_cartesian. Use this function when speedup is more important
    than exact boundary accuracy.
    """
    x = x0
    y = y0
    z = z0
    N = 0.0
    nx = n_field.shape[0]
    ny = n_field.shape[1]
    nz = n_field.shape[2]

    for _ in range(max_steps):
        if x <= xmin or x >= xmax or y <= ymin or y >= ymax or z <= zmin or z >= zmax:
            break

        # Compute indices using floor. This is O(1) vs O(log n) for searchsorted.
        # Note: Cannot perfectly match searchsorted when stored edges have
        # floating point noise vs the arithmetic reconstruction xmin + i*dx.
        # Using nextafter helps when edges ARE exactly uniform, but hurts when
        # they're not. We use simple floor for maximum speed.
        ix = int(np.floor((x - xmin) * inv_dx))
        iy = int(np.floor((y - ymin) * inv_dy))
        iz = int(np.floor((z - zmin) * inv_dz))

        # Clamp to valid range (handles edge case at exactly xmax)
        if ix >= nx:
            ix = nx - 1
        if iy >= ny:
            iy = ny - 1
        if iz >= nz:
            iz = nz - 1

        N += n_field[ix, iy, iz] * ds

        x += dx * ds
        y += dy * ds
        z += dz * ds

    return N


@njit(cache=True, parallel=True)
def _integrate_all_rays_cartesian_uniform(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    nCO_field: np.ndarray,
    nH2_field: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    inv_dx: float,
    inv_dy: float,
    inv_dz: float,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Double-field ray integration for uniform Cartesian grids."""
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    Nco_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    Nh2_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            Nco_all[i, j] = _integrate_ray_cartesian_uniform(
                x0, y0, z0, dx, dy, dz, ds, nCO_field,
                xmin, xmax, ymin, ymax, zmin, zmax,
                inv_dx, inv_dy, inv_dz, max_steps,
            )
            Nh2_all[i, j] = _integrate_ray_cartesian_uniform(
                x0, y0, z0, dx, dy, dz, ds, nH2_field,
                xmin, xmax, ymin, ymax, zmin, zmax,
                inv_dx, inv_dy, inv_dz, max_steps,
            )

    return Nco_all, Nh2_all


@njit(cache=True, parallel=True)
def _integrate_all_rays_cartesian_single_uniform(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    n_field: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    inv_dx: float,
    inv_dy: float,
    inv_dz: float,
    max_steps: int = 10000,
) -> np.ndarray:
    """Single-field ray integration for uniform Cartesian grids."""
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    N_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            N_all[i, j] = _integrate_ray_cartesian_uniform(
                x0, y0, z0, dx, dy, dz, ds, n_field,
                xmin, xmax, ymin, ymax, zmin, zmax,
                inv_dx, inv_dy, inv_dz, max_steps,
            )

    return N_all


@njit(cache=True)
def _integrate_ray_spherical(
    x0: float,
    y0: float,
    z0: float,
    dx: float,
    dy: float,
    dz: float,
    ds: float,
    n_field: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    rmin: float,
    rmax: float,
    max_steps: int = 10000,
) -> float:
    x = x0
    y = y0
    z = z0
    N = 0.0
    nr = n_field.shape[0]
    nt = n_field.shape[1]
    np_ = n_field.shape[2]

    two_pi = 2.0 * np.pi

    for i in range(max_steps):
        r = np.sqrt(x * x + y * y + z * z)
        if r < rmin or r > rmax:
            break

        th = np.arccos(z / (r + 1e-99))
        ph = np.arctan2(y, x)
        if ph < 0:
            ph += two_pi

        ir = np.searchsorted(r_edges, r) - 1
        it = np.searchsorted(theta_edges, th) - 1
        ip = np.searchsorted(phi_edges, ph) - 1

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

    return N


@njit(cache=True, parallel=True)
def _integrate_all_rays_spherical(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    nCO_field: np.ndarray,
    nH2_field: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    rmin: float,
    rmax: float,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    Nco_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    Nh2_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            Nco_all[i, j] = _integrate_ray_spherical(
                x0,
                y0,
                z0,
                dx,
                dy,
                dz,
                ds,
                nCO_field,
                r_edges,
                theta_edges,
                phi_edges,
                rmin,
                rmax,
                max_steps,
            )
            Nh2_all[i, j] = _integrate_ray_spherical(
                x0,
                y0,
                z0,
                dx,
                dy,
                dz,
                ds,
                nH2_field,
                r_edges,
                theta_edges,
                phi_edges,
                rmin,
                rmax,
                max_steps,
            )

    return Nco_all, Nh2_all


@njit(cache=True, parallel=True)
def _integrate_all_rays_spherical_single(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    n_field: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    rmin: float,
    rmax: float,
    max_steps: int = 10000,
) -> np.ndarray:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    N_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            N_all[i, j] = _integrate_ray_spherical(
                x0,
                y0,
                z0,
                dx,
                dy,
                dz,
                ds,
                n_field,
                r_edges,
                theta_edges,
                phi_edges,
                rmin,
                rmax,
                max_steps,
            )

    return N_all


@njit(cache=True, parallel=True)
def _integrate_all_rays_spherical_single_v1_inline(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    n_field: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    rmin: float,
    rmax: float,
    max_steps: int = 10000,
) -> np.ndarray:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]
    nr = n_field.shape[0]
    nt = n_field.shape[1]
    np_ = n_field.shape[2]
    two_pi = 2.0 * np.pi
    N_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            x = x0
            y = y0
            z = z0
            N = 0.0

            for _ in range(max_steps):
                r = np.sqrt(x * x + y * y + z * z)
                if r < rmin or r > rmax:
                    break

                th = np.arccos(z / (r + 1e-99))
                ph = np.arctan2(y, x)
                if ph < 0.0:
                    ph += two_pi

                ir = np.searchsorted(r_edges, r) - 1
                it = np.searchsorted(theta_edges, th) - 1
                ip = np.searchsorted(phi_edges, ph) - 1

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

            N_all[i, j] = N

    return N_all


@njit(cache=True, parallel=True)
def _integrate_all_rays_spherical_v1_inline(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    nCO_field: np.ndarray,
    nH2_field: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    rmin: float,
    rmax: float,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]
    nr = nCO_field.shape[0]
    nt = nCO_field.shape[1]
    np_ = nCO_field.shape[2]
    two_pi = 2.0 * np.pi
    Nco_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    Nh2_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            x = x0
            y = y0
            z = z0
            Nco = 0.0

            for _ in range(max_steps):
                r = np.sqrt(x * x + y * y + z * z)
                if r < rmin or r > rmax:
                    break

                th = np.arccos(z / (r + 1e-99))
                ph = np.arctan2(y, x)
                if ph < 0.0:
                    ph += two_pi

                ir = np.searchsorted(r_edges, r) - 1
                it = np.searchsorted(theta_edges, th) - 1
                ip = np.searchsorted(phi_edges, ph) - 1

                if ip < 0:
                    ip = np_ - 1
                elif ip >= np_:
                    ip = 0

                if ir < 0 or ir >= nr or it < 0 or it >= nt:
                    break

                Nco += nCO_field[ir, it, ip] * ds

                x += dx * ds
                y += dy * ds
                z += dz * ds

            Nco_all[i, j] = Nco

            x = x0
            y = y0
            z = z0
            Nh2 = 0.0

            for _ in range(max_steps):
                r = np.sqrt(x * x + y * y + z * z)
                if r < rmin or r > rmax:
                    break

                th = np.arccos(z / (r + 1e-99))
                ph = np.arctan2(y, x)
                if ph < 0.0:
                    ph += two_pi

                ir = np.searchsorted(r_edges, r) - 1
                it = np.searchsorted(theta_edges, th) - 1
                ip = np.searchsorted(phi_edges, ph) - 1

                if ip < 0:
                    ip = np_ - 1
                elif ip >= np_:
                    ip = 0

                if ir < 0 or ir >= nr or it < 0 or it >= nt:
                    break

                Nh2 += nH2_field[ir, it, ip] * ds

                x += dx * ds
                y += dy * ds
                z += dz * ds

            Nh2_all[i, j] = Nh2

    return Nco_all, Nh2_all


# =============================================================================
# LOG-UNIFORM SPHERICAL GRID VARIANTS (no searchsorted - O(1) arithmetic)
# =============================================================================
# These variants are optimized for spherical grids where:
#   - r is log-uniform: ln(r) is uniformly spaced
#   - theta is uniform in [0, pi]
#   - phi is uniform in [0, 2*pi)
#
# They replace np.searchsorted with simple arithmetic:
#   ir = int((ln_r - ln_rmin) * inv_dlnr)
#   it = int(theta * inv_dtheta)
#   ip = int(phi * inv_dphi)
#
# This provides significant speedup since searchsorted has O(log n) cost
# per call while arithmetic indexing is O(1).


@njit(cache=True)
def _integrate_ray_spherical_loguniform(
    x0: float,
    y0: float,
    z0: float,
    dx: float,
    dy: float,
    dz: float,
    ds: float,
    n_field: np.ndarray,
    rmin: float,
    rmax: float,
    ln_rmin: float,
    inv_dlnr: float,
    inv_dtheta: float,
    inv_dphi: float,
    max_steps: int = 10000,
) -> float:
    """Ray integration for log-uniform spherical grids using arithmetic indexing.
    
    Grid assumptions:
    - r is log-uniform: r_edges = exp(ln_rmin + dlnr * i) for i in [0, nr]
    - theta is uniform in [0, pi]: theta_edges = i * (pi/ntheta) for i in [0, ntheta]
    - phi is uniform in [0, 2*pi): phi_edges = i * (2*pi/nphi) for i in [0, nphi]
    
    Parameters
    ----------
    x0, y0, z0 : float
        Starting position in Cartesian coordinates (cm).
    dx, dy, dz : float
        Unit direction vector components.
    ds : float
        Step size (cm).
    n_field : ndarray, shape (nr, ntheta, nphi)
        Number density field to integrate.
    rmin, rmax : float
        Radial bounds (cm).
    ln_rmin : float
        log(rmin), precomputed.
    inv_dlnr : float
        1.0 / dlnr where dlnr = (ln_rmax - ln_rmin) / nr.
    inv_dtheta : float
        ntheta / pi.
    inv_dphi : float
        nphi / (2*pi).
    max_steps : int
        Maximum integration steps.
        
    Returns
    -------
    N : float
        Integrated column density (cm^-2).
    """
    x = x0
    y = y0
    z = z0
    N = 0.0
    nr = n_field.shape[0]
    nt = n_field.shape[1]
    np_ = n_field.shape[2]

    two_pi = 2.0 * np.pi

    for _ in range(max_steps):
        r = np.sqrt(x * x + y * y + z * z)
        if r < rmin or r > rmax:
            break

        # Compute theta and phi
        th = np.arccos(z / (r + 1e-99))
        ph = np.arctan2(y, x)
        if ph < 0.0:
            ph += two_pi

        # Analytic index computation (O(1) instead of O(log n))
        # Radial index: log-uniform grid
        ln_r = np.log(r)
        ir = int((ln_r - ln_rmin) * inv_dlnr)

        # Theta index: uniform in [0, pi]
        it = int(th * inv_dtheta)

        # Phi index: uniform in [0, 2*pi)
        ip = int(ph * inv_dphi)

        # Handle phi wraparound
        if ip < 0:
            ip = np_ - 1
        elif ip >= np_:
            ip = 0

        # Bounds check for r and theta
        if ir < 0 or ir >= nr or it < 0 or it >= nt:
            break

        N += n_field[ir, it, ip] * ds

        x += dx * ds
        y += dy * ds
        z += dz * ds

    return N


@njit(cache=True, parallel=True)
def _integrate_all_rays_spherical_loguniform(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    nCO_field: np.ndarray,
    nH2_field: np.ndarray,
    rmin: float,
    rmax: float,
    ln_rmin: float,
    inv_dlnr: float,
    inv_dtheta: float,
    inv_dphi: float,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Double-field ray integration for log-uniform spherical grids."""
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    Nco_all = np.zeros((n_cells, n_dirs), dtype=np.float64)
    Nh2_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            Nco_all[i, j] = _integrate_ray_spherical_loguniform(
                x0, y0, z0, dx, dy, dz, ds, nCO_field,
                rmin, rmax, ln_rmin, inv_dlnr, inv_dtheta, inv_dphi,
                max_steps,
            )
            Nh2_all[i, j] = _integrate_ray_spherical_loguniform(
                x0, y0, z0, dx, dy, dz, ds, nH2_field,
                rmin, rmax, ln_rmin, inv_dlnr, inv_dtheta, inv_dphi,
                max_steps,
            )

    return Nco_all, Nh2_all


@njit(cache=True, parallel=True)
def _integrate_all_rays_spherical_single_loguniform(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    ds: float,
    n_field: np.ndarray,
    rmin: float,
    rmax: float,
    ln_rmin: float,
    inv_dlnr: float,
    inv_dtheta: float,
    inv_dphi: float,
    max_steps: int = 10000,
) -> np.ndarray:
    """Single-field ray integration for log-uniform spherical grids."""
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]

    N_all = np.zeros((n_cells, n_dirs), dtype=np.float64)

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]

            N_all[i, j] = _integrate_ray_spherical_loguniform(
                x0, y0, z0, dx, dy, dz, ds, n_field,
                rmin, rmax, ln_rmin, inv_dlnr, inv_dtheta, inv_dphi,
                max_steps,
            )

    return N_all


# =============================================================================
# PUBLIC API - Clean dispatcher for ray integration
# =============================================================================
# These functions provide a unified interface for ray integration, hiding the
# complexity of the various Numba implementations behind a simple API.
#
# Usage:
#   from diskbridge.radmc3d.healpix_utils import integrate_rays
#   N_all = integrate_rays(tracer, cell_centers, directions, n_field)
#   Nco, Nh2 = integrate_rays(tracer, cell_centers, directions, nCO, nH2)


def integrate_rays(
    tracer,
    cell_centers: np.ndarray,
    directions: np.ndarray,
    *fields: np.ndarray,
    max_steps: int = 10000,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Unified ray integration dispatcher.
    
    Automatically selects the optimal Numba implementation based on the tracer
    type (Cartesian vs Spherical) and uses fast O(1) arithmetic indexing where
    available.
    
    Parameters
    ----------
    tracer : CartesianHealpixRayTracer or SphericalHealpixRayTracer
        Ray tracer instance with precomputed grid parameters.
    cell_centers : ndarray, shape (n_cells, 3)
        Starting positions in Cartesian (x, y, z) coordinates.
    directions : ndarray, shape (n_dirs, 3)
        Unit direction vectors.
    *fields : ndarray
        One or two density fields to integrate:
        - Single field: returns N_all of shape (n_cells, n_dirs)
        - Two fields (nCO, nH2): returns (Nco_all, Nh2_all)
    max_steps : int, optional
        Maximum integration steps per ray.
        
    Returns
    -------
    N_all : ndarray or tuple of ndarrays
        Integrated column densities, shape (n_cells, n_dirs).
        
    Examples
    --------
    Single field integration:
    >>> N_all = integrate_rays(tracer, centers, dirs, nH_field)
    
    Double field integration:
    >>> Nco, Nh2 = integrate_rays(tracer, centers, dirs, nCO_field, nH2_field)
    """
    if len(fields) == 0:
        raise ValueError("At least one density field is required")
    if len(fields) > 2:
        raise ValueError("At most two density fields supported")
    
    # Determine tracer type from attributes
    is_cartesian = hasattr(tracer, 'x_edges')
    is_spherical = hasattr(tracer, 'r_edges')
    
    if not (is_cartesian or is_spherical):
        raise TypeError(
            f"Unknown tracer type: {type(tracer).__name__}. "
            "Expected CartesianHealpixRayTracer or SphericalHealpixRayTracer."
        )
    
    ds = float(tracer.ds)
    
    if len(fields) == 1:
        # Single field integration
        n_field = fields[0].astype(np.float64)
        
        if is_cartesian:
            return _integrate_all_rays_cartesian_single_v1_inline(
                cell_centers, directions, ds, n_field,
                tracer.x_edges.astype(np.float64),
                tracer.y_edges.astype(np.float64),
                tracer.z_edges.astype(np.float64),
                tracer._xmin, tracer._xmax,
                tracer._ymin, tracer._ymax,
                tracer._zmin, tracer._zmax,
                max_steps,
            )
        else:  # spherical - use fast loguniform indexing
            return _integrate_all_rays_spherical_single_loguniform(
                cell_centers, directions, ds, n_field,
                tracer.r_edges[0], tracer.r_edges[-1],
                tracer._ln_rmin, tracer._inv_dlnr,
                tracer._inv_dtheta, tracer._inv_dphi,
                max_steps,
            )
    
    else:
        # Double field integration (nCO, nH2)
        nCO_field = fields[0].astype(np.float64)
        nH2_field = fields[1].astype(np.float64)
        
        if is_cartesian:
            return _integrate_all_rays_cartesian(
                cell_centers, directions, ds, nCO_field, nH2_field,
                tracer.x_edges.astype(np.float64),
                tracer.y_edges.astype(np.float64),
                tracer.z_edges.astype(np.float64),
                tracer._xmin, tracer._xmax,
                tracer._ymin, tracer._ymax,
                tracer._zmin, tracer._zmax,
                max_steps,
            )
        else:  # spherical - use fast loguniform indexing
            return _integrate_all_rays_spherical_loguniform(
                cell_centers, directions, ds, nCO_field, nH2_field,
                tracer.r_edges[0], tracer.r_edges[-1],
                tracer._ln_rmin, tracer._inv_dlnr,
                tracer._inv_dtheta, tracer._inv_dphi,
                max_steps,
            )
