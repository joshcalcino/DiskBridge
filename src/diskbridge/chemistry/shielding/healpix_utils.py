from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit, prange

from diskbridge._constants import HEALPIX_SELF_WEIGHT


def _as_f64(name: str, x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a


@njit(cache=True)
def integrate_ray_cartesian_dda_3d(
    x0: float,
    y0: float,
    z0: float,
    vx: float,
    vy: float,
    vz: float,
    density: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    max_steps: int = 100000,
) -> Tuple[float, float]:
    """DDA-style ray integration returning both column and path length."""
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    if norm == 0.0:
        return 0.0, 0.0
    ux = vx / norm
    uy = vy / norm
    uz = vz / norm

    nx = density.shape[0]
    ny = density.shape[1]
    nz = density.shape[2]

    xmin = x_edges[0]
    xmax = x_edges[-1]
    ymin = y_edges[0]
    ymax = y_edges[-1]
    zmin = z_edges[0]
    zmax = z_edges[-1]

    eps = 1e-12
    if x0 <= xmin:
        x0 = xmin + eps
    if x0 >= xmax:
        x0 = xmax - eps
    if y0 <= ymin:
        y0 = ymin + eps
    if y0 >= ymax:
        y0 = ymax - eps
    if z0 <= zmin:
        z0 = zmin + eps
    if z0 >= zmax:
        z0 = zmax - eps

    ix = np.searchsorted(x_edges, x0) - 1
    iy = np.searchsorted(y_edges, y0) - 1
    iz = np.searchsorted(z_edges, z0) - 1

    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return 0.0, 0.0

    dx_cell = x_edges[1] - x_edges[0]
    dy_cell = y_edges[1] - y_edges[0]
    dz_cell = z_edges[1] - z_edges[0]

    if ux > 0.0:
        step_x = 1
        next_x = x_edges[ix + 1]
        tMaxX = (next_x - x0) / ux
        tDeltaX = dx_cell / ux
    elif ux < 0.0:
        step_x = -1
        next_x = x_edges[ix]
        tMaxX = (next_x - x0) / ux
        tDeltaX = dx_cell / (-ux)
    else:
        step_x = 0
        tMaxX = np.inf
        tDeltaX = np.inf

    if uy > 0.0:
        step_y = 1
        next_y = y_edges[iy + 1]
        tMaxY = (next_y - y0) / uy
        tDeltaY = dy_cell / uy
    elif uy < 0.0:
        step_y = -1
        next_y = y_edges[iy]
        tMaxY = (next_y - y0) / uy
        tDeltaY = dy_cell / (-uy)
    else:
        step_y = 0
        tMaxY = np.inf
        tDeltaY = np.inf

    if uz > 0.0:
        step_z = 1
        next_z = z_edges[iz + 1]
        tMaxZ = (next_z - z0) / uz
        tDeltaZ = dz_cell / uz
    elif uz < 0.0:
        step_z = -1
        next_z = z_edges[iz]
        tMaxZ = (next_z - z0) / uz
        tDeltaZ = dz_cell / (-uz)
    else:
        step_z = 0
        tMaxZ = np.inf
        tDeltaZ = np.inf

    col = 0.0
    s_total = 0.0
    t_curr = 0.0
    is_first = True

    for _ in range(max_steps):
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            break

        # Use a tie-aware DDA step. If the ray hits an edge/corner (multiple
        # boundaries at the same parametric distance), we must step all
        # involved axes to avoid systematic grid-aligned bias.
        t_next = tMaxX
        if tMaxY < t_next:
            t_next = tMaxY
        if tMaxZ < t_next:
            t_next = tMaxZ

        tol = 1e-12 * (1.0 + np.abs(t_next))
        hit_x = np.abs(tMaxX - t_next) <= tol
        hit_y = np.abs(tMaxY - t_next) <= tol
        hit_z = np.abs(tMaxZ - t_next) <= tol

        ds_loc = t_next - t_curr
        if ds_loc <= 0.0 or not np.isfinite(ds_loc):
            break

        if is_first:
            col += HEALPIX_SELF_WEIGHT * density[ix, iy, iz] * ds_loc
            is_first = False
        else:
            col += density[ix, iy, iz] * ds_loc
        s_total += ds_loc
        t_curr = t_next

        if hit_x:
            ix += step_x
            tMaxX += tDeltaX
        if hit_y:
            iy += step_y
            tMaxY += tDeltaY
        if hit_z:
            iz += step_z
            tMaxZ += tDeltaZ

        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            break

    return col, s_total


@njit(cache=True)
def _next_positive_quadratic_root(a: float, b: float, c: float) -> float:
    if a == 0.0:
        if b == 0.0:
            return np.inf
        t = -c / b
        if t > 1e-12:
            return t
        return np.inf

    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return np.inf

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    t_min = np.inf
    if t1 > 1e-12 and t1 < t_min:
        t_min = t1
    if t2 > 1e-12 and t2 < t_min:
        t_min = t2

    return t_min


@njit(cache=True)
def _t_to_radius_boundary(
    x: float,
    y: float,
    z: float,
    dx: float,
    dy: float,
    dz: float,
    r_edge: float,
) -> float:
    a = dx * dx + dy * dy + dz * dz
    b = 2.0 * (x * dx + y * dy + z * dz)
    c = x * x + y * y + z * z - r_edge * r_edge
    return _next_positive_quadratic_root(a, b, c)


@njit(cache=True)
def _t_to_theta_boundary(
    x: float,
    y: float,
    z: float,
    dx: float,
    dy: float,
    dz: float,
    theta_edge: float,
) -> float:
    c = np.cos(theta_edge)
    c2 = c * c
    s2 = 1.0 - c2

    if s2 < 1e-12:
        return np.inf

    A = s2 * dz * dz - c2 * dx * dx - c2 * dy * dy
    B = 2.0 * (s2 * z * dz - c2 * x * dx - c2 * y * dy)
    C = s2 * z * z - c2 * x * x - c2 * y * y

    return _next_positive_quadratic_root(A, B, C)


@njit(cache=True)
def _t_to_phi_boundary(
    x: float,
    y: float,
    dx: float,
    dy: float,
    phi_edge: float,
) -> float:
    c = np.cos(phi_edge)
    s = np.sin(phi_edge)

    denom = dy * c - dx * s
    if np.abs(denom) < 1e-14:
        return np.inf

    num = -(y * c - x * s)
    t = num / denom
    if t > 1e-12:
        return t
    return np.inf


@njit(cache=True)
def integrate_ray_spherical_dda_3d(
    x0: float,
    y0: float,
    z0: float,
    vx: float,
    vy: float,
    vz: float,
    density: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    max_steps: int = 200000,
) -> Tuple[float, float]:
    """Spherical DDA ray integration returning both column and path length."""
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    if norm == 0.0:
        return 0.0, 0.0
    dx = vx / norm
    dy = vy / norm
    dz = vz / norm

    nr = density.shape[0]
    nt = density.shape[1]
    nphi = density.shape[2]

    rmin = r_edges[0]
    rmax = r_edges[-1]

    two_pi = 2.0 * np.pi

    x = x0
    y = y0
    z = z0

    r = np.sqrt(x * x + y * y + z * z)
    if r <= rmin or r >= rmax:
        return 0.0, 0.0

    mu = z / r
    if mu > 1.0:
        mu = 1.0
    elif mu < -1.0:
        mu = -1.0
    theta = np.arccos(mu)
    phi = np.arctan2(y, x)
    if phi < 0.0:
        phi += two_pi

    ir = np.searchsorted(r_edges, r, side="right") - 1
    it = np.searchsorted(theta_edges, theta, side="right") - 1
    ip = np.searchsorted(phi_edges, phi, side="right") - 1

    if ir < 0 or ir >= nr or it < 0 or it >= nt or ip < 0 or ip >= nphi:
        return 0.0, 0.0

    col = 0.0
    s_total = 0.0
    is_first = True

    for _ in range(max_steps):
        if ir < 0 or ir >= nr or it < 0 or it >= nt:
            break

        r = np.sqrt(x * x + y * y + z * z)
        if r <= rmin or r >= rmax:
            break

        t_min = np.inf
        hit_dim = -1
        hit_side = -1

        if ir > 0:
            t_r_lo = _t_to_radius_boundary(x, y, z, dx, dy, dz, r_edges[ir])
            if t_r_lo < t_min:
                t_min = t_r_lo
                hit_dim = 0
                hit_side = 0
        if ir < nr:
            t_r_hi = _t_to_radius_boundary(x, y, z, dx, dy, dz, r_edges[ir + 1])
            if t_r_hi < t_min:
                t_min = t_r_hi
                hit_dim = 0
                hit_side = 1

        if it > 0:
            th_lo = theta_edges[it]
            t_th_lo = _t_to_theta_boundary(x, y, z, dx, dy, dz, th_lo)
            if t_th_lo < t_min:
                t_min = t_th_lo
                hit_dim = 1
                hit_side = 0
        if it < nt - 1:
            th_hi = theta_edges[it + 1]
            t_th_hi = _t_to_theta_boundary(x, y, z, dx, dy, dz, th_hi)
            if t_th_hi < t_min:
                t_min = t_th_hi
                hit_dim = 1
                hit_side = 1

        phi_lo = phi_edges[ip]
        t_phi_lo = _t_to_phi_boundary(x, y, dx, dy, phi_lo)
        if t_phi_lo < t_min:
            t_min = t_phi_lo
            hit_dim = 2
            hit_side = 0

        phi_hi = phi_edges[ip + 1]
        t_phi_hi = _t_to_phi_boundary(x, y, dx, dy, phi_hi)
        if t_phi_hi < t_min:
            t_min = t_phi_hi
            hit_dim = 2
            hit_side = 1

        if not np.isfinite(t_min) or t_min <= 0.0:
            break

        ds = t_min
        if is_first:
            col += HEALPIX_SELF_WEIGHT * density[ir, it, ip] * ds
            is_first = False
        else:
            col += density[ir, it, ip] * ds
        s_total += ds

        x += dx * t_min
        y += dy * t_min
        z += dz * t_min

        if hit_dim == 0:
            if hit_side == 0:
                ir -= 1
            else:
                ir += 1
        elif hit_dim == 1:
            if hit_side == 0:
                it -= 1
            else:
                it += 1
        elif hit_dim == 2:
            if hit_side == 0:
                ip -= 1
                if ip < 0:
                    ip = nphi - 1
            else:
                ip += 1
                if ip >= nphi:
                    ip = 0

    return col, s_total


@njit(cache=True, parallel=True)
def _integrate_all_rays_spherical_dda_multi(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    fields_stack: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    max_steps: int = 200000,
) -> np.ndarray:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]
    n_fields = fields_stack.shape[0]

    nr = fields_stack.shape[1]
    nt = fields_stack.shape[2]
    nphi = fields_stack.shape[3]

    N_all = np.zeros((n_cells, n_dirs, n_fields), dtype=np.float64)

    rmin = r_edges[0]
    rmax = r_edges[-1]

    two_pi = 2.0 * np.pi

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            vx = directions[j, 0]
            vy = directions[j, 1]
            vz = directions[j, 2]

            norm = np.sqrt(vx * vx + vy * vy + vz * vz)
            if norm == 0.0:
                continue
            dx = vx / norm
            dy = vy / norm
            dz = vz / norm

            x = x0
            y = y0
            z = z0

            r = np.sqrt(x * x + y * y + z * z)
            if r <= rmin or r >= rmax:
                continue

            mu = z / r
            if mu > 1.0:
                mu = 1.0
            elif mu < -1.0:
                mu = -1.0
            theta = np.arccos(mu)
            phi = np.arctan2(y, x)
            if phi < 0.0:
                phi += two_pi

            ir = np.searchsorted(r_edges, r, side="right") - 1
            it = np.searchsorted(theta_edges, theta, side="right") - 1
            ip = np.searchsorted(phi_edges, phi, side="right") - 1

            if ir < 0 or ir >= nr or it < 0 or it >= nt or ip < 0 or ip >= nphi:
                continue

            is_first = True

            for _ in range(max_steps):
                if ir < 0 or ir >= nr or it < 0 or it >= nt:
                    break

                r = np.sqrt(x * x + y * y + z * z)
                if r <= rmin or r >= rmax:
                    break

                t_min = np.inf
                hit_dim = -1
                hit_side = -1

                if ir > 0:
                    t_r_lo = _t_to_radius_boundary(x, y, z, dx, dy, dz, r_edges[ir])
                    if t_r_lo < t_min:
                        t_min = t_r_lo
                        hit_dim = 0
                        hit_side = 0
                if ir < nr:
                    t_r_hi = _t_to_radius_boundary(x, y, z, dx, dy, dz, r_edges[ir + 1])
                    if t_r_hi < t_min:
                        t_min = t_r_hi
                        hit_dim = 0
                        hit_side = 1

                if it > 0:
                    th_lo = theta_edges[it]
                    t_th_lo = _t_to_theta_boundary(x, y, z, dx, dy, dz, th_lo)
                    if t_th_lo < t_min:
                        t_min = t_th_lo
                        hit_dim = 1
                        hit_side = 0
                if it < nt - 1:
                    th_hi = theta_edges[it + 1]
                    t_th_hi = _t_to_theta_boundary(x, y, z, dx, dy, dz, th_hi)
                    if t_th_hi < t_min:
                        t_min = t_th_hi
                        hit_dim = 1
                        hit_side = 1

                phi_lo = phi_edges[ip]
                t_phi_lo = _t_to_phi_boundary(x, y, dx, dy, phi_lo)
                if t_phi_lo < t_min:
                    t_min = t_phi_lo
                    hit_dim = 2
                    hit_side = 0

                phi_hi = phi_edges[ip + 1]
                t_phi_hi = _t_to_phi_boundary(x, y, dx, dy, phi_hi)
                if t_phi_hi < t_min:
                    t_min = t_phi_hi
                    hit_dim = 2
                    hit_side = 1

                if not np.isfinite(t_min) or t_min <= 0.0:
                    break

                ds = t_min
                if is_first:
                    for k in range(n_fields):
                        N_all[i, j, k] += HEALPIX_SELF_WEIGHT * fields_stack[k, ir, it, ip] * ds
                    is_first = False
                else:
                    for k in range(n_fields):
                        N_all[i, j, k] += fields_stack[k, ir, it, ip] * ds

                x += dx * t_min
                y += dy * t_min
                z += dz * t_min

                if hit_dim == 0:
                    if hit_side == 0:
                        ir -= 1
                    else:
                        ir += 1
                elif hit_dim == 1:
                    if hit_side == 0:
                        it -= 1
                    else:
                        it += 1
                elif hit_dim == 2:
                    if hit_side == 0:
                        ip -= 1
                        if ip < 0:
                            ip = nphi - 1
                    else:
                        ip += 1
                        if ip >= nphi:
                            ip = 0

    return N_all


@njit(cache=True, parallel=True)
def _integrate_all_rays_cartesian_dda_multi(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    fields_stack: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    max_steps: int = 100000,
) -> np.ndarray:
    n_cells = cell_centers.shape[0]
    n_dirs = directions.shape[0]
    n_fields = fields_stack.shape[0]

    nx = fields_stack.shape[1]
    ny = fields_stack.shape[2]
    nz = fields_stack.shape[3]

    N_all = np.zeros((n_cells, n_dirs, n_fields), dtype=np.float64)

    xmin = x_edges[0]
    xmax = x_edges[-1]
    ymin = y_edges[0]
    ymax = y_edges[-1]
    zmin = z_edges[0]
    zmax = z_edges[-1]

    dx_cell = x_edges[1] - x_edges[0]
    dy_cell = y_edges[1] - y_edges[0]
    dz_cell = z_edges[1] - z_edges[0]

    eps = 1e-12

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        for j in range(n_dirs):
            vx = directions[j, 0]
            vy = directions[j, 1]
            vz = directions[j, 2]

            norm = np.sqrt(vx * vx + vy * vy + vz * vz)
            if norm == 0.0:
                continue
            ux = vx / norm
            uy = vy / norm
            uz = vz / norm

            x = x0
            y = y0
            z = z0

            if x <= xmin:
                x = xmin + eps
            if x >= xmax:
                x = xmax - eps
            if y <= ymin:
                y = ymin + eps
            if y >= ymax:
                y = ymax - eps
            if z <= zmin:
                z = zmin + eps
            if z >= zmax:
                z = zmax - eps

            ix = np.searchsorted(x_edges, x) - 1
            iy = np.searchsorted(y_edges, y) - 1
            iz = np.searchsorted(z_edges, z) - 1

            if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                continue

            if ux > 0.0:
                step_x = 1
                next_x = x_edges[ix + 1]
                tMaxX = (next_x - x) / ux
                tDeltaX = dx_cell / ux
            elif ux < 0.0:
                step_x = -1
                next_x = x_edges[ix]
                tMaxX = (next_x - x) / ux
                tDeltaX = dx_cell / (-ux)
            else:
                step_x = 0
                tMaxX = np.inf
                tDeltaX = np.inf

            if uy > 0.0:
                step_y = 1
                next_y = y_edges[iy + 1]
                tMaxY = (next_y - y) / uy
                tDeltaY = dy_cell / uy
            elif uy < 0.0:
                step_y = -1
                next_y = y_edges[iy]
                tMaxY = (next_y - y) / uy
                tDeltaY = dy_cell / (-uy)
            else:
                step_y = 0
                tMaxY = np.inf
                tDeltaY = np.inf

            if uz > 0.0:
                step_z = 1
                next_z = z_edges[iz + 1]
                tMaxZ = (next_z - z) / uz
                tDeltaZ = dz_cell / uz
            elif uz < 0.0:
                step_z = -1
                next_z = z_edges[iz]
                tMaxZ = (next_z - z) / uz
                tDeltaZ = dz_cell / (-uz)
            else:
                step_z = 0
                tMaxZ = np.inf
                tDeltaZ = np.inf

            t_curr = 0.0
            is_first = True

            for _ in range(max_steps):
                if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                    break

                # Use a tie-aware DDA step. If the ray hits an edge/corner (multiple
                # boundaries at the same parametric distance), we must step all
                # involved axes to avoid systematic grid-aligned bias.
                t_next = tMaxX
                if tMaxY < t_next:
                    t_next = tMaxY
                if tMaxZ < t_next:
                    t_next = tMaxZ

                tol = 1e-12 * (1.0 + np.abs(t_next))
                hit_x = np.abs(tMaxX - t_next) <= tol
                hit_y = np.abs(tMaxY - t_next) <= tol
                hit_z = np.abs(tMaxZ - t_next) <= tol

                ds_loc = t_next - t_curr
                if ds_loc <= 0.0 or not np.isfinite(ds_loc):
                    break

                if is_first:
                    for k in range(n_fields):
                        N_all[i, j, k] += HEALPIX_SELF_WEIGHT * fields_stack[k, ix, iy, iz] * ds_loc
                    is_first = False
                else:
                    for k in range(n_fields):
                        N_all[i, j, k] += fields_stack[k, ix, iy, iz] * ds_loc

                t_curr = t_next

                if hit_x:
                    ix += step_x
                    tMaxX += tDeltaX
                if hit_y:
                    iy += step_y
                    tMaxY += tDeltaY
                if hit_z:
                    iz += step_z
                    tMaxZ += tDeltaZ

                if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                    break

    return N_all


def _tracer_kind(tracer) -> str:
    if hasattr(tracer, "x_edges"):
        return "cartesian"
    if hasattr(tracer, "r_edges"):
        return "spherical"
    raise TypeError(
        f"Unknown tracer type: {type(tracer).__name__}. "
        "Expected CartesianHealpixRayTracer or SphericalHealpixRayTracer."
    )


def _tracer_edges_float64(tracer, kind: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == "cartesian":
        return (
            np.asarray(tracer.x_edges, dtype=np.float64),
            np.asarray(tracer.y_edges, dtype=np.float64),
            np.asarray(tracer.z_edges, dtype=np.float64),
        )
    return (
        np.asarray(tracer.r_edges, dtype=np.float64),
        np.asarray(tracer.theta_edges, dtype=np.float64),
        np.asarray(tracer.phi_edges, dtype=np.float64),
    )


def integrate_rays(
    tracer,
    cell_centers: np.ndarray,
    directions: np.ndarray,
    n_field: np.ndarray,
    max_steps: int = 10000,
) -> np.ndarray:
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
        Direction vectors.
    n_field : ndarray
        Density field to integrate.
    max_steps : int, optional
        Maximum integration steps per ray.
    
    Returns
    -------
    N_all : ndarray
        Integrated column densities, shape (n_cells, n_dirs).
    """
    cell_centers = _as_f64("cell_centers", cell_centers)
    directions = _as_f64("directions", directions)
    n_field = _as_f64("n_field", n_field)

    fields_stack = n_field[None, ...]  # (1, nx, ny, nz)
    N_all = integrate_rays_multi(
        tracer,
        cell_centers,
        directions,
        fields_stack,
        max_steps=max_steps,
    )  # (n_cells, n_dirs, 1)
    return N_all[:, :, 0]


def integrate_rays_multi(
    tracer,
    cell_centers: np.ndarray,
    directions: np.ndarray,
    fields_stack: np.ndarray,
    max_steps: int = 10000,
) -> np.ndarray:
    kind = _tracer_kind(tracer)
    cell_centers = _as_f64("cell_centers", cell_centers)
    directions = _as_f64("directions", directions)
    fields_stack = _as_f64("fields_stack", fields_stack)
    edges = _tracer_edges_float64(tracer, kind)

    if kind == "cartesian":
        return _integrate_all_rays_cartesian_dda_multi(
            cell_centers, directions, fields_stack, *edges, max_steps
        )
    return _integrate_all_rays_spherical_dda_multi(
        cell_centers, directions, fields_stack, *edges, max_steps
    )


def integrate_rays_with_pathlength(
    tracer,
    cell_centers: np.ndarray,
    directions: np.ndarray,
    n_field: np.ndarray,
    max_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    cell_centers = _as_f64("cell_centers", cell_centers)
    directions = _as_f64("directions", directions)
    n_field = _as_f64("n_field", n_field)

    ones_field = np.ones_like(n_field, dtype=np.float64)
    fields_stack = np.ascontiguousarray(np.stack((n_field, ones_field), axis=0), dtype=np.float64)

    N_all = integrate_rays_multi(
        tracer,
        cell_centers,
        directions,
        fields_stack,
        max_steps=max_steps,
    )
    return N_all[:, :, 0], N_all[:, :, 1]


# =============================================================================
# STARWARD RAY INTEGRATION (one inward ray per cell toward origin)
# =============================================================================


@njit(parallel=True, cache=True)
def _integrate_starward_radial_spherical(
    fields_stack: np.ndarray,
    r_edges: np.ndarray,
    r_centers: np.ndarray,
    candidate_idx: np.ndarray,
) -> np.ndarray:
    """Integrate fields radially inward (toward origin) for spherical meshes.

    For a spherical mesh centered on the star, the star direction is purely
    radial inward. The column from cell (ir, itheta, iphi) to the inner
    boundary is a simple radial sum with no angular displacement.

    Parameters
    ----------
    fields_stack : ndarray, shape (n_fields, nr, ntheta, nphi)
        Density fields to integrate.
    r_edges : ndarray, shape (nr+1,)
        Radial cell edges in cm.
    r_centers : ndarray, shape (nr,)
        Radial cell centers in cm.
    candidate_idx : ndarray, shape (n_cand, 3)
        Integer indices (ir, itheta, iphi) per candidate cell.
    Returns
    -------
    cols : ndarray, shape (n_cand, n_fields)
        Integrated column density per field per candidate cell.
    """
    n_cand = candidate_idx.shape[0]
    n_fields = fields_stack.shape[0]

    cols = np.zeros((n_cand, n_fields), dtype=np.float64)

    for i in prange(n_cand):
        ir_cell = candidate_idx[i, 0]
        it = candidate_idx[i, 1]
        ip = candidate_idx[i, 2]

        # Self-cell: from cell center inward to inner edge of cell
        ds_self = r_centers[ir_cell] - r_edges[ir_cell]
        if ds_self > 0.0:
            for k in range(n_fields):
                cols[i, k] += HEALPIX_SELF_WEIGHT * fields_stack[k, ir_cell, it, ip] * ds_self

        # Inner cells: full radial extent of each cell
        for j in range(ir_cell - 1, -1, -1):
            ds = r_edges[j + 1] - r_edges[j]
            for k in range(n_fields):
                cols[i, k] += fields_stack[k, j, it, ip] * ds

    return cols


@njit(parallel=True, cache=True)
def _integrate_starward_cartesian_dda_multi(
    cell_centers: np.ndarray,
    fields_stack: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    """Integrate fields along rays toward origin for cartesian meshes.

    Each cell gets its own direction vector pointing toward the origin.
    Uses 3-D DDA ray marching identical to the boundary ray integrator.

    Parameters
    ----------
    cell_centers : ndarray, shape (n_cells, 3)
        Starting positions (x, y, z) in cm.
    fields_stack : ndarray, shape (n_fields, nx, ny, nz)
        Density fields to integrate.
    x_edges, y_edges, z_edges : ndarray
        Cell edge arrays in cm.
    max_steps : int
        Maximum DDA steps per ray.
    Returns
    -------
    cols : ndarray, shape (n_cells, n_fields)
        Integrated column density per field per cell.
    """
    n_cells = cell_centers.shape[0]
    n_fields = fields_stack.shape[0]
    nx = fields_stack.shape[1]
    ny = fields_stack.shape[2]
    nz = fields_stack.shape[3]

    cols = np.zeros((n_cells, n_fields), dtype=np.float64)

    xmin = x_edges[0]
    xmax = x_edges[-1]
    ymin = y_edges[0]
    ymax = y_edges[-1]
    zmin = z_edges[0]
    zmax = z_edges[-1]

    dx_cell = x_edges[1] - x_edges[0]
    dy_cell = y_edges[1] - y_edges[0]
    dz_cell = z_edges[1] - z_edges[0]

    eps = 1e-12

    for i in prange(n_cells):
        x0 = cell_centers[i, 0]
        y0 = cell_centers[i, 1]
        z0 = cell_centers[i, 2]

        # Direction toward origin
        r_mag = np.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
        if r_mag < 1e-30:
            continue
        ux = -x0 / r_mag
        uy = -y0 / r_mag
        uz = -z0 / r_mag

        x = x0
        y = y0
        z = z0

        if x <= xmin:
            x = xmin + eps
        if x >= xmax:
            x = xmax - eps
        if y <= ymin:
            y = ymin + eps
        if y >= ymax:
            y = ymax - eps
        if z <= zmin:
            z = zmin + eps
        if z >= zmax:
            z = zmax - eps

        ix = np.searchsorted(x_edges, x) - 1
        iy = np.searchsorted(y_edges, y) - 1
        iz = np.searchsorted(z_edges, z) - 1

        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            continue

        if ux > 0.0:
            step_x = 1
            tMaxX = (x_edges[ix + 1] - x) / ux
            tDeltaX = dx_cell / ux
        elif ux < 0.0:
            step_x = -1
            tMaxX = (x_edges[ix] - x) / ux
            tDeltaX = dx_cell / (-ux)
        else:
            step_x = 0
            tMaxX = np.inf
            tDeltaX = np.inf

        if uy > 0.0:
            step_y = 1
            tMaxY = (y_edges[iy + 1] - y) / uy
            tDeltaY = dy_cell / uy
        elif uy < 0.0:
            step_y = -1
            tMaxY = (y_edges[iy] - y) / uy
            tDeltaY = dy_cell / (-uy)
        else:
            step_y = 0
            tMaxY = np.inf
            tDeltaY = np.inf

        if uz > 0.0:
            step_z = 1
            tMaxZ = (z_edges[iz + 1] - z) / uz
            tDeltaZ = dz_cell / uz
        elif uz < 0.0:
            step_z = -1
            tMaxZ = (z_edges[iz] - z) / uz
            tDeltaZ = dz_cell / (-uz)
        else:
            step_z = 0
            tMaxZ = np.inf
            tDeltaZ = np.inf

        t_curr = 0.0
        is_first = True

        for _ in range(max_steps):
            if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                break

            t_next = tMaxX
            if tMaxY < t_next:
                t_next = tMaxY
            if tMaxZ < t_next:
                t_next = tMaxZ

            tol = 1e-12 * (1.0 + np.abs(t_next))
            hit_x = np.abs(tMaxX - t_next) <= tol
            hit_y = np.abs(tMaxY - t_next) <= tol
            hit_z = np.abs(tMaxZ - t_next) <= tol

            ds_loc = t_next - t_curr
            if ds_loc <= 0.0 or not np.isfinite(ds_loc):
                break

            if is_first:
                for k in range(n_fields):
                    cols[i, k] += HEALPIX_SELF_WEIGHT * fields_stack[k, ix, iy, iz] * ds_loc
                is_first = False
            else:
                for k in range(n_fields):
                    cols[i, k] += fields_stack[k, ix, iy, iz] * ds_loc

            t_curr = t_next

            if hit_x:
                ix += step_x
                tMaxX += tDeltaX
            if hit_y:
                iy += step_y
                tMaxY += tDeltaY
            if hit_z:
                iz += step_z
                tMaxZ += tDeltaZ

            if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                break

    return cols


def integrate_starward_rays_multi(
    tracer,
    cell_centers: np.ndarray,
    fields_stack: np.ndarray,
    candidate_idx: np.ndarray = None,
) -> np.ndarray:
    """Integrate fields along rays from each cell toward the origin (star).

    For spherical meshes, uses an efficient radial sum (exact, no DDA needed).
    Cartesian starward integration is intentionally unsupported until the
    ray marcher has a correct finite stop at the stellar surface or configured
    inner source radius.

    Parameters
    ----------
    tracer : SphericalHealpixRayTracer or CartesianHealpixRayTracer
        Ray tracer instance with precomputed grid edges.
    cell_centers : ndarray, shape (n_cells, 3)
        Cell center positions in Cartesian (x, y, z) coordinates, cm.
    fields_stack : ndarray, shape (n_fields, dim0, dim1, dim2)
        Density fields to integrate (e.g. dust bin densities in g/cm^3).
    candidate_idx : ndarray, shape (n_cells, 3), optional
        Grid indices per cell. Required for spherical meshes (ir, itheta, iphi).
        If None and spherical, raises ValueError.

    Returns
    -------
    cols : ndarray, shape (n_cells, n_fields)
        Integrated column per field per cell (e.g. g/cm^2 for dust density).
    """
    kind = _tracer_kind(tracer)
    fields_stack = _as_f64("fields_stack", fields_stack)
    cell_centers = _as_f64("cell_centers", cell_centers)

    if kind == "spherical":
        if candidate_idx is None:
            raise ValueError(
                "candidate_idx is required for spherical starward integration"
            )
        candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
        edges = _tracer_edges_float64(tracer, kind)
        r_centers = np.asarray(tracer.r_centers, dtype=np.float64)
        return _integrate_starward_radial_spherical(
            fields_stack, edges[0], r_centers, candidate_idx,
        )

    raise NotImplementedError(
        "Direct stellar UV attenuation is not supported for Cartesian meshes. "
        "The previous Cartesian starward DDA path marched toward the origin "
        "without a correct stop at the star/inner radius, so it is disabled. "
        "Use a spherical star-centered mesh for direct stellar UV weights, "
        "or set star_uv_luminosity_erg_s=0.0 to build weights without the "
        "direct stellar component."
    )
