"""Shared gas-velocity, velocity-gradient, and molecular line-width helpers."""

# db-keywords: model, field, coordinates, line-transfer, gow17, gas-temperature, kinematics, velocity-gradient, co-cooling
# db-role: canonical
# db-scope: package
# db-purpose: Shared Cartesian gas velocity and molecular Doppler-width conventions.

from __future__ import annotations

import numpy as np
from numba import njit, prange

from diskbridge._constants import K_B, M_H


def _field_f64(rad, name: str, unit: str) -> np.ndarray:
    if rad.model.gas is None or name not in rad.model.gas:
        raise KeyError(f"model.gas[{name!r}] is required")
    value = rad.model.gas[name]
    value = value.data if hasattr(value, "data") else value
    if hasattr(value, "to"):
        value = value.to(unit).magnitude
    return np.ascontiguousarray(value, dtype=np.float64)


def cartesian_velocity_cm_s(rad, *, basis_mesh=None) -> np.ndarray:
    """Return Cartesian gas velocity on the native mesh in cm/s.

    Parameters
    ----------
    rad : diskbridge.radmc3d.model.RadModel
        Model wrapper containing the gas velocity fields.
    basis_mesh : diskbridge.model.mesh.Mesh, optional
        Spherical basis used for conversion. The model mesh is used by
        default.

    Returns
    -------
    numpy.ndarray
        Array with shape ``mesh.shape + (3,)``.
    """

    mesh = rad.model.mesh
    basis_mesh = basis_mesh or mesh
    if mesh.coord_system == "cartesian":
        vx = _field_f64(rad, "vx", "cm/s")
        vy = _field_f64(rad, "vy", "cm/s")
        vz = _field_f64(rad, "vz", "cm/s")
    elif mesh.coord_system == "spherical":
        vr = _field_f64(rad, "vr", "cm/s")
        vtheta = _field_f64(rad, "vtheta", "cm/s")
        vphi = _field_f64(rad, "vphi", "cm/s")
        vx, vy, vz = basis_mesh.spherical_vector_components_to_cartesian(
            vr,
            vtheta,
            vphi,
            axis_order=mesh.axis_names(),
        )
    else:
        raise ValueError(
            "Velocity-coherent HEALPix integration requires a spherical or "
            f"Cartesian mesh, got {mesh.coord_system!r}"
        )

    out = np.empty(np.shape(vx) + (3,), dtype=np.float64)
    out[..., 0] = vx
    out[..., 1] = vy
    out[..., 2] = vz
    return np.ascontiguousarray(out)


def microturbulence_cm_s(rad) -> np.ndarray:
    """Return the RADMC-3D turbulent Doppler parameter in cm/s."""

    return _field_f64(rad, "microturbulence", "cm/s")


def molecular_doppler_width_cm_s(
    temperature_K: np.ndarray,
    microturbulence: np.ndarray,
    molecular_weight: float,
) -> np.ndarray:
    """Return ``sqrt(a_turb**2 + 2 k_B T / m_mol)`` in cm/s."""

    temperature = np.asarray(temperature_K, dtype=np.float64)
    a_turb = np.asarray(microturbulence, dtype=np.float64)
    mass = float(molecular_weight) * M_H
    if mass <= 0.0:
        raise ValueError("molecular_weight must be positive")
    thermal_sq = 2.0 * K_B * temperature / mass
    return np.sqrt(np.maximum(a_turb * a_turb + thermal_sq, 0.0))


def _axis_derivative(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    axis: int,
    periodic_period: float | None = None,
) -> np.ndarray:
    """Differentiate one mesh axis, including uniform periodic azimuths."""

    arr = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    size = int(arr.shape[axis])
    if coords.shape != (size,):
        raise ValueError("coordinate length must match the differentiated axis")
    if size == 1:
        return np.zeros_like(arr)
    if periodic_period is None:
        return np.gradient(
            arr,
            coords,
            axis=axis,
            edge_order=2 if size >= 3 else 1,
        )

    spacing = float(periodic_period) / float(size)
    wrapped_steps = np.mod(np.diff(coords), float(periodic_period))
    if not np.allclose(wrapped_steps, spacing, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("periodic spherical phi centers must be uniformly spaced")
    return (
        np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)
    ) / (2.0 * spacing)


def symmetric_velocity_gradient_s1(rad) -> np.ndarray:
    """Return the symmetric physical velocity-gradient tensor in ``s^-1``.

    The final tensor axis stores ``(11, 22, 33, 12, 13, 23)``. For a
    Cartesian mesh these are Cartesian components. For a spherical mesh they
    are components in the local orthonormal ``(r, theta, phi)`` basis and
    include the covariant metric terms.

    Parameters
    ----------
    rad : diskbridge.radmc3d.model.RadModel
        Model wrapper containing the native gas velocity fields.

    Returns
    -------
    numpy.ndarray
        Symmetric tensor with shape ``mesh.shape + (6,)`` in ``s^-1``.
    """

    mesh = rad.model.mesh
    if mesh.coord_system == "cartesian":
        coordinates = [mesh.centers_f64(name, "cm") for name in ("x", "y", "z")]
        velocity = [_field_f64(rad, name, "cm/s") for name in ("vx", "vy", "vz")]
        gradient = [
            [
                _axis_derivative(velocity[component], coordinates[axis], axis=axis)
                for axis in range(3)
            ]
            for component in range(3)
        ]
        out = np.empty(mesh.shape + (6,), dtype=np.float64)
        out[..., 0] = gradient[0][0]
        out[..., 1] = gradient[1][1]
        out[..., 2] = gradient[2][2]
        out[..., 3] = 0.5 * (gradient[0][1] + gradient[1][0])
        out[..., 4] = 0.5 * (gradient[0][2] + gradient[2][0])
        out[..., 5] = 0.5 * (gradient[1][2] + gradient[2][1])
        return np.ascontiguousarray(out)

    if mesh.coord_system != "spherical":
        raise ValueError(
            "velocity gradients require a spherical or Cartesian mesh, got "
            f"{mesh.coord_system!r}"
        )

    r = mesh.centers_f64("r", "cm")
    theta = mesh.centers_f64("theta", "rad")
    phi = mesh.centers_f64("phi", "rad")
    phi_edges = mesh.edges_f64("phi", "rad")
    phi_span = float(phi_edges[-1] - phi_edges[0])
    periodic_phi = (
        2.0 * np.pi if np.isclose(phi_span, 2.0 * np.pi, rtol=1.0e-10, atol=1.0e-12) else None
    )

    vr = _field_f64(rad, "vr", "cm/s")
    vt = _field_f64(rad, "vtheta", "cm/s")
    vp = _field_f64(rad, "vphi", "cm/s")
    expected = mesh.shape
    if vr.shape != expected or vt.shape != expected or vp.shape != expected:
        raise ValueError("spherical velocity fields must match mesh.shape")

    dvr_dr = _axis_derivative(vr, r, axis=0)
    dvr_dt = _axis_derivative(vr, theta, axis=1)
    dvr_dp = _axis_derivative(vr, phi, axis=2, periodic_period=periodic_phi)
    dvt_dr = _axis_derivative(vt, r, axis=0)
    dvt_dt = _axis_derivative(vt, theta, axis=1)
    dvt_dp = _axis_derivative(vt, phi, axis=2, periodic_period=periodic_phi)
    dvp_dr = _axis_derivative(vp, r, axis=0)
    dvp_dt = _axis_derivative(vp, theta, axis=1)
    dvp_dp = _axis_derivative(vp, phi, axis=2, periodic_period=periodic_phi)

    radius = r[:, None, None]
    angle = theta[None, :, None]
    sin_t = np.sin(angle)
    if np.any(np.abs(sin_t) <= 1.0e-14):
        raise ValueError("spherical velocity gradients are undefined at polar cell centers")
    inv_r = 1.0 / radius
    inv_r_sin = inv_r / sin_t
    cot_t_over_r = np.cos(angle) * inv_r_sin

    out = np.empty(mesh.shape + (6,), dtype=np.float64)
    out[..., 0] = dvr_dr
    out[..., 1] = inv_r * dvt_dt + vr * inv_r
    out[..., 2] = inv_r_sin * dvp_dp + vr * inv_r + vt * cot_t_over_r
    out[..., 3] = 0.5 * (inv_r * dvr_dt + dvt_dr - vt * inv_r)
    out[..., 4] = 0.5 * (inv_r_sin * dvr_dp + dvp_dr - vp * inv_r)
    out[..., 5] = 0.5 * (
        inv_r_sin * dvt_dp + inv_r * dvp_dt - vp * cot_t_over_r
    )
    if not np.all(np.isfinite(out)):
        raise ValueError("velocity-gradient tensor contains non-finite values")
    return np.ascontiguousarray(out)


@njit(parallel=True, cache=True)
def _project_tensor_cartesian(
    candidate_idx: np.ndarray,
    directions: np.ndarray,
    tensor: np.ndarray,
) -> np.ndarray:
    n_cells = candidate_idx.shape[0]
    n_dirs = directions.shape[0]
    out = np.empty((n_cells, n_dirs), dtype=np.float64)
    for q in prange(n_cells * n_dirs):
        cell = q // n_dirs
        direction = q - cell * n_dirs
        i = candidate_idx[cell, 0]
        j = candidate_idx[cell, 1]
        k = candidate_idx[cell, 2]
        nx = directions[direction, 0]
        ny = directions[direction, 1]
        nz = directions[direction, 2]
        value = (
            tensor[i, j, k, 0] * nx * nx
            + tensor[i, j, k, 1] * ny * ny
            + tensor[i, j, k, 2] * nz * nz
            + 2.0 * tensor[i, j, k, 3] * nx * ny
            + 2.0 * tensor[i, j, k, 4] * nx * nz
            + 2.0 * tensor[i, j, k, 5] * ny * nz
        )
        out[cell, direction] = abs(value)
    return out


@njit(parallel=True, cache=True)
def _project_tensor_spherical(
    candidate_idx: np.ndarray,
    directions: np.ndarray,
    tensor: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    n_cells = candidate_idx.shape[0]
    n_dirs = directions.shape[0]
    out = np.empty((n_cells, n_dirs), dtype=np.float64)
    for q in prange(n_cells * n_dirs):
        cell = q // n_dirs
        direction = q - cell * n_dirs
        ir = candidate_idx[cell, 0]
        it = candidate_idx[cell, 1]
        ip = candidate_idx[cell, 2]
        st = np.sin(theta[it])
        ct = np.cos(theta[it])
        sp = np.sin(phi[ip])
        cp = np.cos(phi[ip])
        nx = directions[direction, 0]
        ny = directions[direction, 1]
        nz = directions[direction, 2]
        nr = st * cp * nx + st * sp * ny + ct * nz
        nt = ct * cp * nx + ct * sp * ny - st * nz
        np_ = -sp * nx + cp * ny
        value = (
            tensor[ir, it, ip, 0] * nr * nr
            + tensor[ir, it, ip, 1] * nt * nt
            + tensor[ir, it, ip, 2] * np_ * np_
            + 2.0 * tensor[ir, it, ip, 3] * nr * nt
            + 2.0 * tensor[ir, it, ip, 4] * nr * np_
            + 2.0 * tensor[ir, it, ip, 5] * nt * np_
        )
        out[cell, direction] = abs(value)
    return out


def projected_velocity_gradient_s1(
    mesh,
    tensor_s1: np.ndarray,
    candidate_idx: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Contract a symmetric velocity gradient with unit ray directions.

    Parameters
    ----------
    mesh : diskbridge.model.mesh.Mesh
        Native spherical or Cartesian mesh.
    tensor_s1 : ndarray
        Packed symmetric tensor with shape ``mesh.shape + (6,)``.
    candidate_idx : ndarray
        Integer cell indices with shape ``(ncells, 3)``.
    directions : ndarray
        Cartesian ray directions with shape ``(ndirections, 3)``.

    Returns
    -------
    numpy.ndarray
        Absolute projected gradients with shape ``(ncells, ndirections)`` in
        ``s^-1``.
    """

    tensor = np.ascontiguousarray(tensor_s1, dtype=np.float64)
    if tensor.shape != mesh.shape + (6,):
        raise ValueError("tensor_s1 must have shape mesh.shape + (6,)")
    indices = np.ascontiguousarray(candidate_idx, dtype=np.int64)
    dirs = np.ascontiguousarray(directions, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("candidate_idx must have shape (ncells, 3)")
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("directions must have shape (ndirections, 3)")
    norms = np.linalg.norm(dirs, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("directions must be finite and non-zero")
    dirs = np.ascontiguousarray(dirs / norms[:, None])

    if mesh.coord_system == "cartesian":
        return _project_tensor_cartesian(indices, dirs, tensor)
    if mesh.coord_system == "spherical":
        return _project_tensor_spherical(
            indices,
            dirs,
            tensor,
            mesh.centers_f64("theta", "rad"),
            mesh.centers_f64("phi", "rad"),
        )
    raise ValueError(
        "projected velocity gradients require a spherical or Cartesian mesh, "
        f"got {mesh.coord_system!r}"
    )


__all__ = [
    "cartesian_velocity_cm_s",
    "microturbulence_cm_s",
    "molecular_doppler_width_cm_s",
    "symmetric_velocity_gradient_s1",
    "projected_velocity_gradient_s1",
]
