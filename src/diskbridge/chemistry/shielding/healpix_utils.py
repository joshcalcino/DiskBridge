# db-keywords: shielding, healpix-columns, chemistry, mesh, field, coordinates
# db-role: helper
# db-scope: package
# db-purpose: Package module for shielding, healpix-columns, chemistry, mesh.

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import get_num_threads, njit, prange


def _as_f64(name: str, x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a


@njit(cache=True, parallel=True)
def _mean_rays_parallel(theta_rays: np.ndarray) -> np.ndarray:
    """Reduce shielding factors uniformly without a ray-sized temporary."""
    n_cells, n_rays = theta_rays.shape
    out = np.empty(n_cells, dtype=np.float64)
    for i in prange(n_cells):
        total = 0.0
        for j in range(n_rays):
            total += theta_rays[i, j]
        out[i] = total / n_rays
    return out


@njit(cache=True, parallel=True)
def _weighted_mean_rays_parallel(
    theta_rays: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Reduce weighted shielding factors without materializing their product."""
    n_cells, n_rays = theta_rays.shape
    out = np.empty(n_cells, dtype=np.float64)
    for i in prange(n_cells):
        total = 0.0
        for j in range(n_rays):
            total += weights[i, j] * theta_rays[i, j]
        out[i] = total
    return out


@njit(cache=True, parallel=True)
def _mean_product_rays_parallel(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Uniformly reduce a per-ray product without materializing it."""
    n_cells, n_rays = left.shape
    out = np.empty(n_cells, dtype=np.float64)
    for i in prange(n_cells):
        total = 0.0
        for j in range(n_rays):
            total += left[i, j] * right[i, j]
        out[i] = total / n_rays
    return out


@njit(cache=True, parallel=True)
def _weighted_mean_product_rays_parallel(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weight and reduce a per-ray product without materializing it."""
    n_cells, n_rays = left.shape
    out = np.empty(n_cells, dtype=np.float64)
    for i in prange(n_cells):
        total = 0.0
        for j in range(n_rays):
            total += weights[i, j] * (left[i, j] * right[i, j])
        out[i] = total
    return out


@njit(cache=True, parallel=True)
def _effective_b_rays_inplace_parallel(
    weighted_b2_column: np.ndarray,
    density_column: np.ndarray,
    fallback2: float,
    minimum_column: float,
    tiny: float,
) -> None:
    """Replace a 2-D integrated ``n*b^2`` column with effective ``b``."""
    n_cells, n_rays = weighted_b2_column.shape
    for i in prange(n_cells):
        for j in range(n_rays):
            if density_column[i, j] > minimum_column:
                value = weighted_b2_column[i, j] / density_column[i, j]
            else:
                value = fallback2
            if value < tiny:
                value = tiny
            weighted_b2_column[i, j] = np.sqrt(value)


@njit(cache=True, parallel=True)
def _effective_b_cells_inplace_parallel(
    weighted_b2_column: np.ndarray,
    density_column: np.ndarray,
    fallback2: float,
    minimum_column: float,
    tiny: float,
) -> None:
    """Replace a 1-D integrated ``n*b^2`` column with effective ``b``."""
    for i in prange(weighted_b2_column.size):
        if density_column[i] > minimum_column:
            value = weighted_b2_column[i] / density_column[i]
        else:
            value = fallback2
        if value < tiny:
            value = tiny
        weighted_b2_column[i] = np.sqrt(value)


@njit(cache=True, parallel=True)
def _c_shielding_rays_parallel(
    carbon_column: np.ndarray,
    h2_column: np.ndarray,
) -> np.ndarray:
    """Evaluate C shielding over a 2-D cell/ray map in parallel."""
    n_cells, n_rays = carbon_column.shape
    out = np.empty((n_cells, n_rays), dtype=np.float64)
    for i in prange(n_cells):
        for j in range(n_rays):
            tau_h2 = 1.2e-14 * 2.0 * h2_column[i, j]
            y = 1.17e-8 * tau_h2
            ry = np.exp(-y) / (1.0 + y)
            rc = np.exp(-1.6e-17 * carbon_column[i, j])
            out[i, j] = rc * ry
    return out


@njit(cache=True, parallel=True)
def _c_shielding_cells_parallel(
    carbon_column: np.ndarray,
    h2_column: np.ndarray,
) -> np.ndarray:
    """Evaluate C shielding over a 1-D cell array in parallel."""
    out = np.empty(carbon_column.size, dtype=np.float64)
    for i in prange(carbon_column.size):
        tau_h2 = 1.2e-14 * 2.0 * h2_column[i]
        y = 1.17e-8 * tau_h2
        ry = np.exp(-y) / (1.0 + y)
        rc = np.exp(-1.6e-17 * carbon_column[i])
        out[i] = rc * ry
    return out


@njit(cache=True, parallel=True)
def _stellar_weight_correction_parallel(
    theta_mean: np.ndarray,
    theta_rays: np.ndarray,
    theta_star: np.ndarray,
    star_pixel: np.ndarray,
    star_weight: np.ndarray,
) -> np.ndarray:
    """Replace a direct-stellar ray contribution in parallel by cell."""
    out = np.empty(theta_mean.size, dtype=np.float64)
    for i in prange(theta_mean.size):
        value = theta_mean[i]
        if star_weight[i] > 0.0:
            value += star_weight[i] * (
                theta_star[i] - theta_rays[i, star_pixel[i]]
            )
        out[i] = value
    return out


@njit(cache=True, parallel=True)
def _stellar_product_weight_correction_parallel(
    theta_mean: np.ndarray,
    left_rays: np.ndarray,
    right_rays: np.ndarray,
    theta_star: np.ndarray,
    star_pixel: np.ndarray,
    star_weight: np.ndarray,
) -> np.ndarray:
    """Replace a direct-stellar product contribution without a product map."""
    out = np.empty(theta_mean.size, dtype=np.float64)
    for i in prange(theta_mean.size):
        value = theta_mean[i]
        if star_weight[i] > 0.0:
            ray_value = left_rays[i, star_pixel[i]] * right_rays[i, star_pixel[i]]
            value += star_weight[i] * (theta_star[i] - ray_value)
        out[i] = value
    return out


@njit(cache=True, parallel=True)
def _scatter_candidates_3d_parallel(
    target: np.ndarray,
    candidate_idx: np.ndarray,
    values: np.ndarray,
) -> None:
    """Scatter values at unique 3-D candidate indices in parallel."""
    for i in prange(values.size):
        target[
            candidate_idx[i, 0],
            candidate_idx[i, 1],
            candidate_idx[i, 2],
        ] = values[i]


@njit(cache=True, parallel=True)
def _multiply_3d_parallel(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Multiply equally shaped 3-D fields in parallel over the first axis."""
    n0, n1, n2 = left.shape
    out = np.empty((n0, n1, n2), dtype=np.float64)
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                out[i, j, k] = left[i, j, k] * right[i, j, k]
    return out


@njit(cache=True, parallel=True)
def _weighted_b2_3d_parallel(
    density: np.ndarray,
    linewidth: np.ndarray,
) -> np.ndarray:
    """Form ``density * linewidth**2`` without an intermediate field."""
    n0, n1, n2 = density.shape
    out = np.empty((n0, n1, n2), dtype=np.float64)
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                b_value = linewidth[i, j, k]
                out[i, j, k] = density[i, j, k] * b_value * b_value
    return out


@njit(cache=True)
def _positive_quadratic_roots(
    a: float,
    b: float,
    c: float,
    minimum_t: float,
) -> tuple[float, float]:
    """Return the positive real roots in ascending order."""
    if a == 0.0:
        if b == 0.0:
            return np.inf, np.inf
        t = -c / b
        if t > minimum_t:
            return t, np.inf
        return np.inf, np.inf

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return np.inf, np.inf

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    if t1 > t2:
        temporary = t1
        t1 = t2
        t2 = temporary
    if t1 <= minimum_t:
        t1 = np.inf
    if t2 <= minimum_t:
        t2 = np.inf
    if t2 < t1:
        temporary = t1
        t1 = t2
        t2 = temporary
    return t1, t2


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
    radius = np.sqrt(x * x + y * y + z * z)
    minimum_t = 1.0e-10 * max(radius, 1.0)
    t1, _ = _positive_quadratic_roots(a, b, c, minimum_t)
    return t1


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

    radius = np.sqrt(x * x + y * y + z * z)
    minimum_t = 1.0e-10 * max(radius, 1.0)
    t1, t2 = _positive_quadratic_roots(A, B, C, minimum_t)

    # The quadratic describes both halves of a cone.  A theta boundary is
    # only the half with the same sign of z as cos(theta_edge).
    for t in (t1, t2):
        if not np.isfinite(t):
            continue
        x_hit = x + dx * t
        y_hit = y + dy * t
        z_hit = z + dz * t
        r_hit = np.sqrt(x_hit * x_hit + y_hit * y_hit + z_hit * z_hit)
        if r_hit == 0.0:
            continue
        branch_tolerance = 1.0e-10 * r_hit
        if z_hit * c >= -branch_tolerance:
            return t
    return np.inf


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
    radius = np.sqrt(x * x + y * y)
    minimum_t = 1.0e-10 * max(radius, 1.0)
    if t > minimum_t:
        x_hit = x + dx * t
        y_hit = y + dy * t
        # A constant-phi boundary is a half-plane.  Reject the opposite
        # azimuth, which lies on the same infinite Cartesian plane.
        radial_projection = x_hit * c + y_hit * s
        hit_radius = np.sqrt(x_hit * x_hit + y_hit * y_hit)
        if radial_projection >= -1.0e-10 * max(hit_radius, 1.0):
            return t
    return np.inf


@njit(cache=True, parallel=True)
def _integrate_all_rays_spherical_dda_multi(
    cell_centers: np.ndarray,
    directions: np.ndarray,
    fields_stack: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    max_steps: int = 200000,
    n_workers: int = 1,
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

    # Transpose the conceptual (worker, cell) work grid so every static
    # worker block samples the complete native cell order instead of one
    # contiguous spatial band. Output rows remain in native order.
    cells_per_worker = (n_cells + n_workers - 1) // n_workers
    padded_work = n_workers * cells_per_worker
    for work_index in prange(padded_work):
        worker_block = work_index // cells_per_worker
        block_offset = work_index - worker_block * cells_per_worker
        i = block_offset * n_workers + worker_block
        if i >= n_cells:
            continue
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

            theta = np.arctan2(np.sqrt(x * x + y * y), z)
            phi = np.arctan2(y, x)
            if phi < 0.0:
                phi += two_pi

            ir = np.searchsorted(r_edges, r, side="right") - 1
            it = np.searchsorted(theta_edges, theta, side="right") - 1
            ip = np.searchsorted(phi_edges, phi, side="right") - 1

            if ir < 0 or ir >= nr or it < 0 or it >= nt or ip < 0 or ip >= nphi:
                continue

            for _ in range(max_steps):
                if ir < 0 or ir >= nr or it < 0 or it >= nt:
                    break

                r = np.sqrt(x * x + y * y + z * z)
                if r <= rmin or r >= rmax:
                    break

                t_min = np.inf
                t_r_lo = _t_to_radius_boundary(
                    x, y, z, dx, dy, dz, r_edges[ir]
                )
                if t_r_lo < t_min:
                    t_min = t_r_lo
                if ir < nr:
                    t_r_hi = _t_to_radius_boundary(x, y, z, dx, dy, dz, r_edges[ir + 1])
                    if t_r_hi < t_min:
                        t_min = t_r_hi

                if it > 0:
                    th_lo = theta_edges[it]
                    t_th_lo = _t_to_theta_boundary(x, y, z, dx, dy, dz, th_lo)
                    if t_th_lo < t_min:
                        t_min = t_th_lo
                if it < nt - 1:
                    th_hi = theta_edges[it + 1]
                    t_th_hi = _t_to_theta_boundary(x, y, z, dx, dy, dz, th_hi)
                    if t_th_hi < t_min:
                        t_min = t_th_hi

                phi_lo = phi_edges[ip]
                t_phi_lo = _t_to_phi_boundary(x, y, dx, dy, phi_lo)
                if t_phi_lo < t_min:
                    t_min = t_phi_lo

                phi_hi = phi_edges[ip + 1]
                t_phi_hi = _t_to_phi_boundary(x, y, dx, dy, phi_hi)
                if t_phi_hi < t_min:
                    t_min = t_phi_hi

                if (
                    not np.isfinite(t_min)
                    or t_min <= 0.0
                    or t_min > 2.0 * rmax * (1.0 + 1.0e-10)
                ):
                    break

                ds = t_min
                for k in range(n_fields):
                    N_all[i, j, k] += fields_stack[k, ir, it, ip] * ds

                x += dx * t_min
                y += dy * t_min
                z += dz * t_min

                # Resolve all tied crossings from a point just beyond the
                # boundary.  Keeping the integration point at the exact hit
                # avoids dropping the probe segment from the column.
                probe = 1.0e-9 * max(np.sqrt(x * x + y * y + z * z), 1.0)
                x_probe = x + dx * probe
                y_probe = y + dy * probe
                z_probe = z + dz * probe
                r_probe = np.sqrt(
                    x_probe * x_probe + y_probe * y_probe + z_probe * z_probe
                )
                if r_probe <= rmin or r_probe >= rmax:
                    break
                theta_probe = np.arctan2(
                    np.sqrt(x_probe * x_probe + y_probe * y_probe), z_probe
                )
                phi_probe = np.arctan2(y_probe, x_probe)
                if phi_probe < 0.0:
                    phi_probe += two_pi
                ir = np.searchsorted(r_edges, r_probe, side="right") - 1
                it = np.searchsorted(theta_edges, theta_probe, side="right") - 1
                ip = np.searchsorted(phi_edges, phi_probe, side="right") - 1

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
    n_workers: int = 1,
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

    # Use the same no-copy cyclic assignment as the spherical DDA kernel.
    cells_per_worker = (n_cells + n_workers - 1) // n_workers
    padded_work = n_workers * cells_per_worker
    for work_index in prange(padded_work):
        worker_block = work_index // cells_per_worker
        block_offset = work_index - worker_block * cells_per_worker
        i = block_offset * n_workers + worker_block
        if i >= n_cells:
            continue
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
            cell_centers,
            directions,
            fields_stack,
            *edges,
            max_steps,
            int(get_num_threads()),
        )
    return _integrate_all_rays_spherical_dda_multi(
        cell_centers,
        directions,
        fields_stack,
        *edges,
        max_steps,
        int(get_num_threads()),
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
    stop_radius_cm: float,
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

        r_stop = stop_radius_cm
        if r_stop < r_edges[0]:
            r_stop = r_edges[0]
        if r_centers[ir_cell] <= r_stop:
            continue

        # Self-cell: from cell center inward to the larger of the cell inner
        # edge and the stellar/inner source radius.
        self_inner = r_edges[ir_cell]
        if self_inner < r_stop:
            self_inner = r_stop
        ds_self = r_centers[ir_cell] - self_inner
        if ds_self > 0.0:
            for k in range(n_fields):
                cols[i, k] += fields_stack[k, ir_cell, it, ip] * ds_self

        # Inner cells: full or partial radial extents down to r_stop.
        for j in range(ir_cell - 1, -1, -1):
            if r_edges[j + 1] <= r_stop:
                break
            r_lo = r_edges[j]
            if r_lo < r_stop:
                r_lo = r_stop
            ds = r_edges[j + 1] - r_lo
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
    stop_radius_cm: float,
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
    stop_radius_cm : float
        Inner stellar/source radius where the ray stops. Use 0 to stop at the
        coordinate origin.
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

        t_stop = r_mag - max(stop_radius_cm, 0.0)
        if t_stop <= 0.0:
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
        for _ in range(max_steps):
            if t_curr >= t_stop:
                break
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

            t_seg = t_next
            if t_seg > t_stop:
                t_seg = t_stop

            ds_loc = t_seg - t_curr
            if ds_loc <= 0.0 or not np.isfinite(ds_loc):
                break

            for k in range(n_fields):
                cols[i, k] += fields_stack[k, ix, iy, iz] * ds_loc

            t_curr = t_seg

            if t_curr >= t_stop:
                break

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
    stop_radius_cm: float = 0.0,
) -> np.ndarray:
    """Integrate fields along rays from each cell toward the origin (star).

    For spherical meshes, uses an efficient radial sum (exact, no DDA needed).
    For Cartesian meshes, uses the multi-field DDA marcher with a finite stop
    at ``stop_radius_cm`` (or the coordinate origin when zero).

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
    stop_radius_cm : float, optional
        Inner stellar/source radius where starward rays stop. Use 0 to stop at
        the coordinate origin.

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
            fields_stack,
            edges[0],
            r_centers,
            candidate_idx,
            float(stop_radius_cm),
        )

    edges = _tracer_edges_float64(tracer, kind)
    return _integrate_starward_cartesian_dda_multi(
        cell_centers,
        fields_stack,
        edges[0],
        edges[1],
        edges[2],
        100000,
        float(stop_radius_cm),
    )
