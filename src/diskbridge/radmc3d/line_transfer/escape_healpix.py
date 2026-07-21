"""HEALPix velocity-coherent escape-probability kernel (Numba, parallel).

For each candidate cell, this module integrates the direction-dependent
optical depth along each HEALPix ray out to the simulation boundary, with a
Doppler overlap factor that suppresses contributions from cells whose
projected velocity is offset from the absorbing cell by more than the line
width. Per-direction beta values are then direction-averaged.

The DDA logic mirrors
``diskbridge.chemistry.shielding.healpix_utils`` but accumulates tau per line
instead of a scalar molecular column. Integration starts at the cell center
and includes the geometric first segment to the next face once.
"""

# db-keywords: shielding, healpix-columns, gow17, line-transfer, radmc3d, mesh, field, coordinates
# db-role: canonical
# db-scope: package
# db-purpose: HEALPix velocity-coherent escape-probability kernel (Numba, parallel).

from __future__ import annotations

from typing import Tuple
import time

import numpy as np
from numba import njit, prange

from diskbridge._logging import logger
from diskbridge.chemistry.shielding.healpix_utils import (
    _t_to_phi_boundary,
    _t_to_radius_boundary,
    _t_to_theta_boundary,
)


def _unit_directions(dirs: np.ndarray) -> np.ndarray:
    """Normalize ray directions once before entering the per-cell kernels."""

    directions = np.ascontiguousarray(dirs, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1)
    nonzero = norms > 0.0
    normalized = directions.copy()
    normalized[nonzero] /= norms[nonzero, None]
    return np.ascontiguousarray(normalized)


@njit(cache=True, inline="always")
def _beta_of_tau(tau: float) -> float:
    """Stable (1 - exp(-tau)) / tau with small-tau series."""
    if tau <= 0.0:
        return 1.0
    if tau < 1.0e-4:
        return 1.0 - 0.5 * tau + tau * tau / 6.0
    return (1.0 - np.exp(-tau)) / tau


@njit(parallel=True, cache=True, fastmath=False)
def _integrate_velocity_coherent_cartesian(
    candidate_idx: np.ndarray,        # (Ncand, 3) int64 (ix, iy, iz)
    cell_centers: np.ndarray,         # (Ncand, 3) f64
    dirs: np.ndarray,                 # (Npix, 3) f64
    alpha0_stack: np.ndarray,         # (nlin, nx, ny, nz) f64, cm^-1
    velocity_xyz: np.ndarray,         # (nx, ny, nz, 3) f64, cm/s
    a_line: np.ndarray,               # (nx, ny, nz) f64, cm/s
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    max_ray_steps: int,
    reduction: int,
) -> np.ndarray:
    """Integrate coherent fields, returning rays or mean escape probability."""

    n_cand = candidate_idx.shape[0]
    n_dirs = dirs.shape[0]
    nlin = alpha0_stack.shape[0]
    nx = alpha0_stack.shape[1]
    ny = alpha0_stack.shape[2]
    nz = alpha0_stack.shape[3]

    n_reduced = n_dirs if reduction == 0 else 1
    out = np.zeros((n_cand, n_reduced, nlin), dtype=np.float64)

    xmin = x_edges[0]
    xmax = x_edges[-1]
    ymin = y_edges[0]
    ymax = y_edges[-1]
    zmin = z_edges[0]
    zmax = z_edges[-1]

    dx_cell = x_edges[1] - x_edges[0]
    dy_cell = y_edges[1] - y_edges[0]
    dz_cell = z_edges[1] - z_edges[0]

    eps = 1.0e-12
    inv_npix = 1.0 / float(n_dirs)

    for c in prange(n_cand):
        ix0 = candidate_idx[c, 0]
        iy0 = candidate_idx[c, 1]
        iz0 = candidate_idx[c, 2]

        v0x = velocity_xyz[ix0, iy0, iz0, 0]
        v0y = velocity_xyz[ix0, iy0, iz0, 1]
        v0z = velocity_xyz[ix0, iy0, iz0, 2]

        # Per-cell scratch (thread-local).
        tau = np.zeros(nlin, dtype=np.float64)
        beta_sum = np.zeros(nlin, dtype=np.float64)

        for k in range(n_dirs):
            ux = dirs[k, 0]
            uy = dirs[k, 1]
            uz = dirs[k, 2]
            if ux == 0.0 and uy == 0.0 and uz == 0.0:
                if reduction == 1:
                    for m in range(nlin):
                        beta_sum[m] += 1.0
                continue

            x = cell_centers[c, 0]
            y = cell_centers[c, 1]
            z = cell_centers[c, 2]
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
                if reduction == 1:
                    for m in range(nlin):
                        beta_sum[m] += 1.0
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

            for m in range(nlin):
                tau[m] = 0.0
            t_curr = 0.0

            for _step in range(max_ray_steps):
                if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                    break
                t_next = tMaxX
                if tMaxY < t_next:
                    t_next = tMaxY
                if tMaxZ < t_next:
                    t_next = tMaxZ
                tol = 1.0e-12 * (1.0 + np.abs(t_next))
                hit_x = np.abs(tMaxX - t_next) <= tol
                hit_y = np.abs(tMaxY - t_next) <= tol
                hit_z = np.abs(tMaxZ - t_next) <= tol
                ds = t_next - t_curr
                if ds <= 0.0 or not np.isfinite(ds):
                    break

                dv = (
                    (velocity_xyz[ix, iy, iz, 0] - v0x) * ux
                    + (velocity_xyz[ix, iy, iz, 1] - v0y) * uy
                    + (velocity_xyz[ix, iy, iz, 2] - v0z) * uz
                )
                aL = a_line[ix, iy, iz]
                if aL <= 0.0:
                    overlap = 0.0
                elif dv == 0.0:
                    overlap = 1.0
                else:
                    arg = dv / aL
                    arg_sq = arg * arg
                    if arg_sq > 745.1332191019412:
                        # exp(-arg_sq) rounds to zero in float64 beyond this
                        # point, so avoid an expensive libm call without
                        # introducing a physical overlap cutoff.
                        overlap = 0.0
                    else:
                        overlap = np.exp(-arg_sq)

                for m in range(nlin):
                    tau[m] += alpha0_stack[m, ix, iy, iz] * overlap * ds

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

            if reduction == 0:
                for m in range(nlin):
                    out[c, k, m] = tau[m]
            else:
                for m in range(nlin):
                    beta_sum[m] += _beta_of_tau(tau[m])

        if reduction == 1:
            for m in range(nlin):
                out[c, 0, m] = beta_sum[m] * inv_npix

    return out


@njit(parallel=True, cache=True, fastmath=False)
def _integrate_velocity_coherent_spherical(
    candidate_idx: np.ndarray,        # (Ncand, 3) int64 (ir, it, ip)
    cell_centers: np.ndarray,         # (Ncand, 3) f64
    dirs: np.ndarray,                 # (Npix, 3) f64
    alpha0_stack: np.ndarray,         # (nlin, nr, nt, nphi) f64, cm^-1
    velocity_xyz: np.ndarray,         # (nr, nt, nphi, 3) f64, cm/s
    a_line: np.ndarray,               # (nr, nt, nphi) f64, cm/s
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    max_ray_steps: int,
    reduction: int,
) -> np.ndarray:
    """Integrate coherent fields, returning rays or mean escape probability."""

    n_cand = candidate_idx.shape[0]
    n_dirs = dirs.shape[0]
    nlin = alpha0_stack.shape[0]
    nr = alpha0_stack.shape[1]
    nt = alpha0_stack.shape[2]
    nphi = alpha0_stack.shape[3]

    n_reduced = n_dirs if reduction == 0 else 1
    out = np.zeros((n_cand, n_reduced, nlin), dtype=np.float64)

    rmin = r_edges[0]
    rmax = r_edges[-1]
    inv_npix = 1.0 / float(n_dirs)

    for c in prange(n_cand):
        ir0 = candidate_idx[c, 0]
        it0 = candidate_idx[c, 1]
        ip0 = candidate_idx[c, 2]

        v0x = velocity_xyz[ir0, it0, ip0, 0]
        v0y = velocity_xyz[ir0, it0, ip0, 1]
        v0z = velocity_xyz[ir0, it0, ip0, 2]

        tau = np.zeros(nlin, dtype=np.float64)
        beta_sum = np.zeros(nlin, dtype=np.float64)

        for k in range(n_dirs):
            ux = dirs[k, 0]
            uy = dirs[k, 1]
            uz = dirs[k, 2]
            if ux == 0.0 and uy == 0.0 and uz == 0.0:
                if reduction == 1:
                    for m in range(nlin):
                        beta_sum[m] += 1.0
                continue

            x = cell_centers[c, 0]
            y = cell_centers[c, 1]
            z = cell_centers[c, 2]

            r = np.sqrt(x * x + y * y + z * z)
            if r <= rmin or r >= rmax:
                if reduction == 1:
                    for m in range(nlin):
                        beta_sum[m] += 1.0
                continue
            # The caller already knows the containing cell. Recomputing it from
            # atan2() can be wrong for meshes whose phi edges are [-pi, pi],
            # which would make negative-phi starting cells escape immediately.
            ir = ir0
            it = it0
            ip = ip0
            if ir < 0 or ir >= nr or it < 0 or it >= nt or ip < 0 or ip >= nphi:
                if reduction == 1:
                    for m in range(nlin):
                        beta_sum[m] += 1.0
                continue

            for m in range(nlin):
                tau[m] = 0.0

            for _step in range(max_ray_steps):
                if ir < 0 or ir >= nr or it < 0 or it >= nt:
                    break
                r = np.sqrt(x * x + y * y + z * z)
                if r <= rmin or r >= rmax:
                    break

                t_min = np.inf
                hit_dim = -1
                hit_side = -1

                if ir > 0:
                    t_r_lo = _t_to_radius_boundary(x, y, z, ux, uy, uz, r_edges[ir])
                    if t_r_lo < t_min:
                        t_min = t_r_lo
                        hit_dim = 0
                        hit_side = 0
                if ir < nr:
                    t_r_hi = _t_to_radius_boundary(x, y, z, ux, uy, uz, r_edges[ir + 1])
                    if t_r_hi < t_min:
                        t_min = t_r_hi
                        hit_dim = 0
                        hit_side = 1
                if it > 0:
                    th_lo = theta_edges[it]
                    t_th_lo = _t_to_theta_boundary(x, y, z, ux, uy, uz, th_lo)
                    if t_th_lo < t_min:
                        t_min = t_th_lo
                        hit_dim = 1
                        hit_side = 0
                if it < nt - 1:
                    th_hi = theta_edges[it + 1]
                    t_th_hi = _t_to_theta_boundary(x, y, z, ux, uy, uz, th_hi)
                    if t_th_hi < t_min:
                        t_min = t_th_hi
                        hit_dim = 1
                        hit_side = 1
                phi_lo = phi_edges[ip]
                t_phi_lo = _t_to_phi_boundary(x, y, ux, uy, phi_lo)
                if t_phi_lo < t_min:
                    t_min = t_phi_lo
                    hit_dim = 2
                    hit_side = 0
                phi_hi = phi_edges[ip + 1]
                t_phi_hi = _t_to_phi_boundary(x, y, ux, uy, phi_hi)
                if t_phi_hi < t_min:
                    t_min = t_phi_hi
                    hit_dim = 2
                    hit_side = 1

                if not np.isfinite(t_min) or t_min <= 0.0:
                    break

                ds = t_min
                dv = (
                    (velocity_xyz[ir, it, ip, 0] - v0x) * ux
                    + (velocity_xyz[ir, it, ip, 1] - v0y) * uy
                    + (velocity_xyz[ir, it, ip, 2] - v0z) * uz
                )
                aL = a_line[ir, it, ip]
                if aL <= 0.0:
                    overlap = 0.0
                elif dv == 0.0:
                    overlap = 1.0
                else:
                    arg = dv / aL
                    arg_sq = arg * arg
                    if arg_sq > 745.1332191019412:
                        # This is the float64 underflow limit, not a physical
                        # truncation of the Gaussian profile.
                        overlap = 0.0
                    else:
                        overlap = np.exp(-arg_sq)
                for m in range(nlin):
                    tau[m] += alpha0_stack[m, ir, it, ip] * overlap * ds

                x += ux * t_min
                y += uy * t_min
                z += uz * t_min

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

            if reduction == 0:
                for m in range(nlin):
                    out[c, k, m] = tau[m]
            else:
                for m in range(nlin):
                    beta_sum[m] += _beta_of_tau(tau[m])

        if reduction == 1:
            for m in range(nlin):
                out[c, 0, m] = beta_sum[m] * inv_npix

    return out


@njit(parallel=True, cache=True, fastmath=False)
def _integrate_velocity_coherent_spherical_columns(
    candidate_idx: np.ndarray,
    cell_centers: np.ndarray,
    dirs: np.ndarray,
    field: np.ndarray,
    velocity_xyz: np.ndarray,
    a_line: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    max_ray_steps: int,
) -> np.ndarray:
    """Integrate one coherent field with direction-major parallel work."""

    n_cand = candidate_idx.shape[0]
    n_dirs = dirs.shape[0]
    nr, nt, nphi = field.shape
    out = np.zeros((n_dirs, n_cand), dtype=np.float64)
    rmin = r_edges[0]
    rmax = r_edges[-1]

    # Direction-major ordering distributes the complete set of source-cell
    # path lengths across each worker instead of assigning a radial block of
    # cells to one worker. Each (cell, direction) ray remains independent.
    for q in prange(n_cand * n_dirs):
        k = q // n_cand
        c = q - k * n_cand

        ir0 = candidate_idx[c, 0]
        it0 = candidate_idx[c, 1]
        ip0 = candidate_idx[c, 2]
        if ir0 < 0 or ir0 >= nr or it0 < 0 or it0 >= nt or ip0 < 0 or ip0 >= nphi:
            continue

        ux = dirs[k, 0]
        uy = dirs[k, 1]
        uz = dirs[k, 2]
        if ux == 0.0 and uy == 0.0 and uz == 0.0:
            continue

        x = cell_centers[c, 0]
        y = cell_centers[c, 1]
        z = cell_centers[c, 2]
        radius = np.sqrt(x * x + y * y + z * z)
        if radius <= rmin or radius >= rmax:
            continue

        ir = ir0
        it = it0
        ip = ip0
        v0x = velocity_xyz[ir0, it0, ip0, 0]
        v0y = velocity_xyz[ir0, it0, ip0, 1]
        v0z = velocity_xyz[ir0, it0, ip0, 2]
        column = 0.0

        for _step in range(max_ray_steps):
            if ir < 0 or ir >= nr or it < 0 or it >= nt:
                break
            radius = np.sqrt(x * x + y * y + z * z)
            if radius <= rmin or radius >= rmax:
                break

            t_min = np.inf
            hit_dim = -1
            hit_side = -1

            if ir > 0:
                value = _t_to_radius_boundary(
                    x, y, z, ux, uy, uz, r_edges[ir]
                )
                if value < t_min:
                    t_min = value
                    hit_dim = 0
                    hit_side = 0
            if ir < nr:
                value = _t_to_radius_boundary(
                    x, y, z, ux, uy, uz, r_edges[ir + 1]
                )
                if value < t_min:
                    t_min = value
                    hit_dim = 0
                    hit_side = 1
            if it > 0:
                value = _t_to_theta_boundary(
                    x, y, z, ux, uy, uz, theta_edges[it]
                )
                if value < t_min:
                    t_min = value
                    hit_dim = 1
                    hit_side = 0
            if it < nt - 1:
                value = _t_to_theta_boundary(
                    x, y, z, ux, uy, uz, theta_edges[it + 1]
                )
                if value < t_min:
                    t_min = value
                    hit_dim = 1
                    hit_side = 1

            value = _t_to_phi_boundary(x, y, ux, uy, phi_edges[ip])
            if value < t_min:
                t_min = value
                hit_dim = 2
                hit_side = 0
            value = _t_to_phi_boundary(x, y, ux, uy, phi_edges[ip + 1])
            if value < t_min:
                t_min = value
                hit_dim = 2
                hit_side = 1

            if not np.isfinite(t_min) or t_min <= 0.0:
                break

            dv = (
                (velocity_xyz[ir, it, ip, 0] - v0x) * ux
                + (velocity_xyz[ir, it, ip, 1] - v0y) * uy
                + (velocity_xyz[ir, it, ip, 2] - v0z) * uz
            )
            aL = a_line[ir, it, ip]
            if aL <= 0.0:
                overlap = 0.0
            elif dv == 0.0:
                overlap = 1.0
            else:
                arg = dv / aL
                arg_sq = arg * arg
                if arg_sq > 745.1332191019412:
                    overlap = 0.0
                else:
                    overlap = np.exp(-arg_sq)
            column += field[ir, it, ip] * overlap * t_min

            x += ux * t_min
            y += uy * t_min
            z += uz * t_min

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

        out[k, c] = column

    return out


def compute_beta_cartesian(
    candidate_idx,
    cell_centers,
    dirs,
    alpha0_stack,
    velocity_xyz,
    a_line,
    x_edges,
    y_edges,
    z_edges,
    max_ray_steps,
):
    """Return direction-averaged Cartesian escape probabilities."""

    return _integrate_velocity_coherent_cartesian(
        candidate_idx,
        cell_centers,
        _unit_directions(dirs),
        alpha0_stack,
        velocity_xyz,
        a_line,
        x_edges,
        y_edges,
        z_edges,
        max_ray_steps,
        1,
    )[:, 0, :]


def compute_beta_spherical(
    candidate_idx,
    cell_centers,
    dirs,
    alpha0_stack,
    velocity_xyz,
    a_line,
    r_edges,
    theta_edges,
    phi_edges,
    max_ray_steps,
):
    """Return direction-averaged spherical escape probabilities."""

    return _integrate_velocity_coherent_spherical(
        candidate_idx,
        cell_centers,
        _unit_directions(dirs),
        alpha0_stack,
        velocity_xyz,
        a_line,
        r_edges,
        theta_edges,
        phi_edges,
        max_ray_steps,
        1,
    )[:, 0, :]


def compute_velocity_coherent_columns_healpix(
    *,
    mesh,
    tracer,
    candidate_idx: np.ndarray,
    cell_centers: np.ndarray,
    dirs: np.ndarray,
    density_over_width: np.ndarray,
    velocity_xyz: np.ndarray,
    a_line: np.ndarray,
    max_ray_steps: int = 200_000,
) -> np.ndarray:
    """Integrate direction-dependent velocity-coherent columns.

    ``density_over_width`` is normally ``n_CO / a_CO``. The returned array has
    shape ``(ncells, npix)`` and units ``cm^-2 / (cm s^-1)``.
    """

    candidate_idx = np.ascontiguousarray(candidate_idx, dtype=np.int64)
    cell_centers = np.ascontiguousarray(cell_centers, dtype=np.float64)
    dirs = _unit_directions(dirs)
    field_stack = np.ascontiguousarray(
        np.asarray(density_over_width, dtype=np.float64)[None, ...]
    )
    velocity_xyz = np.ascontiguousarray(velocity_xyz, dtype=np.float64)
    a_line = np.ascontiguousarray(a_line, dtype=np.float64)
    if mesh.coord_system == "cartesian":
        out = _integrate_velocity_coherent_cartesian(
            candidate_idx,
            cell_centers,
            dirs,
            field_stack,
            velocity_xyz,
            a_line,
            np.ascontiguousarray(tracer.x_edges, dtype=np.float64),
            np.ascontiguousarray(tracer.y_edges, dtype=np.float64),
            np.ascontiguousarray(tracer.z_edges, dtype=np.float64),
            int(max_ray_steps),
            0,
        )
    elif mesh.coord_system == "spherical":
        phi_edges = np.ascontiguousarray(tracer.phi_edges, dtype=np.float64)
        direction_major = _integrate_velocity_coherent_spherical_columns(
            candidate_idx,
            cell_centers,
            dirs,
            field_stack[0],
            velocity_xyz,
            a_line,
            np.ascontiguousarray(tracer.r_edges, dtype=np.float64),
            np.ascontiguousarray(tracer.theta_edges, dtype=np.float64),
            phi_edges,
            int(max_ray_steps),
        )
        return np.ascontiguousarray(direction_major.T)
    else:
        raise ValueError(
            "Velocity-coherent HEALPix integration requires spherical or "
            f"Cartesian mesh, got {mesh.coord_system!r}"
        )
    return out[:, :, 0]


def compute_escape_probabilities_healpix(
    *,
    mesh,
    tracer,
    candidate_idx: np.ndarray,
    cell_centers: np.ndarray,
    dirs: np.ndarray,
    alpha0_stack: np.ndarray,
    velocity_xyz: np.ndarray,
    a_line: np.ndarray,
    max_ray_steps: int = 200_000,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Compute direction-averaged escape probability per candidate cell, per line.

    Parameters
    ----------
    mesh : diskbridge.model.mesh.Mesh
    tracer : SphericalHealpixRayTracer or CartesianHealpixRayTracer
    candidate_idx : ndarray, shape (Ncand, 3)
    cell_centers : ndarray, shape (Ncand, 3)
    dirs : ndarray, shape (Npix, 3)
    alpha0_stack : ndarray, shape (nlin, n0, n1, n2)
        Line-center opacities in cm^-1, in native mesh order.
    velocity_xyz : ndarray, shape (n0, n1, n2, 3)
        Cartesian gas velocities at cell centers, cm/s, native mesh order.
    a_line : ndarray, shape (n0, n1, n2)
        Total line broadening parameter (thermal + microturbulence) in cm/s.
    max_ray_steps : int
    chunk_size : int or None
        If not None, candidates are processed in chunks of this size.

    Returns
    -------
    beta : ndarray, shape (Ncand, nlin)
    """

    candidate_idx = np.ascontiguousarray(candidate_idx, dtype=np.int64)
    cell_centers = np.ascontiguousarray(cell_centers, dtype=np.float64)
    dirs = _unit_directions(dirs)
    alpha0_stack = np.ascontiguousarray(alpha0_stack, dtype=np.float64)
    velocity_xyz = np.ascontiguousarray(velocity_xyz, dtype=np.float64)
    a_line = np.ascontiguousarray(a_line, dtype=np.float64)
    n_cand = candidate_idx.shape[0]
    nlin = alpha0_stack.shape[0]
    if n_cand == 0:
        return np.zeros((0, nlin), dtype=np.float64)

    if chunk_size is None or chunk_size >= n_cand:
        return _dispatch_compute_beta(
            mesh=mesh,
            tracer=tracer,
            candidate_idx=candidate_idx,
            cell_centers=cell_centers,
            dirs=dirs,
            alpha0_stack=alpha0_stack,
            velocity_xyz=velocity_xyz,
            a_line=a_line,
            max_ray_steps=int(max_ray_steps),
        )

    out = np.empty((n_cand, nlin), dtype=np.float64)
    for start in range(0, n_cand, int(chunk_size)):
        end = min(start + int(chunk_size), n_cand)
        t0 = time.perf_counter()
        logger.info(
            "HEALPix escape beta chunk: cells %d:%d / %d",
            start,
            end,
            n_cand,
        )
        out[start:end] = _dispatch_compute_beta(
            mesh=mesh,
            tracer=tracer,
            candidate_idx=candidate_idx[start:end],
            cell_centers=cell_centers[start:end],
            dirs=dirs,
            alpha0_stack=alpha0_stack,
            velocity_xyz=velocity_xyz,
            a_line=a_line,
            max_ray_steps=int(max_ray_steps),
        )
        logger.info(
            "HEALPix escape beta chunk done: cells %d:%d / %d (%.2f s)",
            start,
            end,
            n_cand,
            time.perf_counter() - t0,
        )
    return out


def _dispatch_compute_beta(
    *,
    mesh,
    tracer,
    candidate_idx,
    cell_centers,
    dirs,
    alpha0_stack,
    velocity_xyz,
    a_line,
    max_ray_steps,
):
    if mesh.coord_system == "cartesian":
        return compute_beta_cartesian(
            candidate_idx,
            cell_centers,
            dirs,
            alpha0_stack,
            velocity_xyz,
            a_line,
            np.ascontiguousarray(tracer.x_edges, dtype=np.float64),
            np.ascontiguousarray(tracer.y_edges, dtype=np.float64),
            np.ascontiguousarray(tracer.z_edges, dtype=np.float64),
            max_ray_steps,
        )
    if mesh.coord_system == "spherical":
        return compute_beta_spherical(
            candidate_idx,
            cell_centers,
            dirs,
            alpha0_stack,
            velocity_xyz,
            a_line,
            np.ascontiguousarray(tracer.r_edges, dtype=np.float64),
            np.ascontiguousarray(tracer.theta_edges, dtype=np.float64),
            np.ascontiguousarray(tracer.phi_edges, dtype=np.float64),
            max_ray_steps,
        )
    raise ValueError(
        "HEALPix escape kernel requires spherical or cartesian mesh, "
        f"got {mesh.coord_system!r}"
    )


__all__ = [
    "compute_escape_probabilities_healpix",
    "compute_velocity_coherent_columns_healpix",
    "compute_beta_cartesian",
    "compute_beta_spherical",
    "_beta_of_tau",
]
