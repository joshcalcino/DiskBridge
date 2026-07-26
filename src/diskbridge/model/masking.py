# db-keywords: disk-mask, line-transfer, units, model, mesh, field
# db-role: helper
# db-scope: package
# db-purpose: Package module for disk-mask, line-transfer, units, model.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING, Tuple, Union

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d

from diskbridge._units import Quantity, units
from diskbridge._logging import logger
from diskbridge.model.field import Field
from diskbridge.model.coords import (
    spherical_grids,
    cylindrical_from_spherical,
    cartesian_from_spherical,
)
from diskbridge.model.disk import scale_height
from .core import SubModel

if TYPE_CHECKING:
    from .core import Model


@dataclass
class DiskFrameData:
    k_hat: np.ndarray
    R_d: np.ndarray
    z_d: np.ndarray
    vR_d: Quantity
    vphi_d: Quantity
    vz_d: Quantity
    theta_from_midplane: np.ndarray
    phi_d: Optional[np.ndarray] = None


@dataclass(frozen=True)
class JoosCriterionData:
    """Dimensionless Joos criterion margins evaluated on the native grid."""
    valid: np.ndarray
    q_pol: np.ndarray
    q_rot: np.ndarray
    q_rho: np.ndarray
    hard_pass: np.ndarray


def _compute_disk_orientation(
    model: "Model",
    rho: Quantity,
    dV: Quantity,
    r_grid: Quantity,
    theta_grid: Quantity,
    phi_grid: Quantity,
    vr: Quantity,
    vphi: Quantity,
    vtheta: Quantity,
    rho_core_min: Quantity,
    r_max_for_axis: Quantity,
) -> np.ndarray:
    axis_limit = np.asarray(
        r_max_for_axis.to(r_grid.units).magnitude,
        dtype=float,
    )
    if axis_limit.ndim != 0 or not np.isfinite(axis_limit) or axis_limit <= 0.0:
        raise ValueError("r_max_for_axis must be a finite positive scalar length")

    core_mask = (rho >= rho_core_min) & (r_grid <= r_max_for_axis)

    x, y, z = cartesian_from_spherical(r_grid, phi_grid, theta_grid)
    # Reuse the mesh-owned spherical basis conversion so disk masking and
    # line-transfer diagnostics cannot silently diverge in velocity geometry.
    vx, vy, vz = model.mesh.spherical_vector_components_to_cartesian(
        vr,
        vtheta,
        vphi,
    )

    w = (rho * dV) * core_mask
    if not np.any(core_mask):
        raise ValueError(
            "No cells satisfy the disk-axis density and radial support; "
            "adjust rho_core_min or r_max_for_axis explicitly"
        )

    Lx = np.sum(w * (y * vz - z * vy))
    Ly = np.sum(w * (z * vx - x * vz))
    Lz = np.sum(w * (x * vy - y * vx))
    Lnorm = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    Lnorm_mag = float(Lnorm.magnitude)
    if Lnorm_mag == 0.0 or not np.isfinite(Lnorm_mag):
        raise ValueError(
            "Disk-axis angular momentum is zero or non-finite within the "
            "requested density and radial support"
        )
    return np.array([
        (Lx / Lnorm).to("dimensionless").magnitude,
        (Ly / Lnorm).to("dimensionless").magnitude,
        (Lz / Lnorm).to("dimensionless").magnitude,
    ])


def _transform_to_disk_frame(
    mesh,
    r_grid: Quantity,
    theta_grid: Quantity,
    phi_grid: Quantity,
    vr: Quantity,
    vphi: Quantity,
    vtheta: Quantity,
    k_hat: np.ndarray,
) -> DiskFrameData:
    x, y, z = cartesian_from_spherical(r_grid, phi_grid, theta_grid)
    vx, vy, vz = mesh.spherical_vector_components_to_cartesian(
        vr,
        vtheta,
        vphi,
    )

    z_d = x * k_hat[0] + y * k_hat[1] + z * k_hat[2]
    rx = x - z_d * k_hat[0]
    ry = y - z_d * k_hat[1]
    rz = z - z_d * k_hat[2]
    R_d = np.sqrt(rx * rx + ry * ry + rz * rz)

    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(k_hat[2])) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    e1 = np.cross(k_hat, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(k_hat, e1)
    x_d = x * e1[0] + y * e1[1] + z * e1[2]
    y_d = x * e2[0] + y * e2[1] + z * e2[2]
    phi_d = np.arctan2(y_d.magnitude, x_d.magnitude)

    invR = np.zeros_like(R_d.magnitude) / R_d.units
    nz = R_d.magnitude > 0.0
    invR[nz] = 1.0 / R_d[nz]
    rhatx = rx * invR
    rhaty = ry * invR
    rhatz = rz * invR

    phix = k_hat[1] * rhatz - k_hat[2] * rhaty
    phiy = k_hat[2] * rhatx - k_hat[0] * rhatz
    phiz = k_hat[0] * rhaty - k_hat[1] * rhatx

    vR_d = vx * rhatx + vy * rhaty + vz * rhatz
    vphi_d = vx * phix + vy * phiy + vz * phiz
    vz_d = vx * k_hat[0] + vy * k_hat[1] + vz * k_hat[2]

    ok = r_grid.magnitude > 0.0
    theta_from_midplane = np.zeros_like(z_d.magnitude, dtype=float)
    if np.any(ok):
        cos_theta = np.clip((z_d / r_grid)[ok].to("dimensionless").magnitude, -1.0, 1.0)
        theta_d = np.arccos(cos_theta)
        theta_from_midplane[ok] = theta_d - (0.5 * np.pi)

    return DiskFrameData(
        k_hat=k_hat,
        R_d=R_d.magnitude,
        z_d=z_d.magnitude,
        vR_d=vR_d,
        vphi_d=vphi_d,
        vz_d=vz_d,
        theta_from_midplane=theta_from_midplane,
        phi_d=phi_d,
    )


def _compute_thermal_pressure(model: "Model", rho: Quantity) -> Quantity:
    if "pressure" in model.gas:
        Pth = model.gas["pressure"].data
    elif "temperature" in model.gas:
        if not hasattr(model, "variables") or "MU" not in model.variables:
            raise ValueError(
                "Cannot compute thermal pressure from temperature without mean molecular weight. "
                "Provide gas['pressure'] or ensure the reader sets model.variables['MU']."
            )
        T = model.gas["temperature"].data
        mu_val = float(getattr(model.variables["MU"], "magnitude", model.variables["MU"]))
        Pth = rho.to("g/cm^3") * units("k_B") * T.to("K") / (mu_val * units("m_H"))
    else:
        raise KeyError("Need either gas['pressure'] or gas['temperature'] to evaluate thermal support")
    return Pth.to_base_units()


def _setup_binning(
    mesh,
    n_r_bins: Optional[int],
    n_theta_bins: Optional[int],
    n_bins_native: int,
    disk_frame: DiskFrameData,
    r_grid: Quantity,
) -> Tuple[Quantity, np.ndarray, int]:
    r_edges_native = mesh.edges("r")
    if r_edges_native is None:
        raise ValueError("Mesh is missing radial edges")
    r_edges_native = r_edges_native.to_base_units()

    nR_native = len(r_edges_native) - 1
    if nR_native < 1:
        raise ValueError("No radial bins available")

    if n_r_bins is None:
        r_edges = r_edges_native
    else:
        if int(n_r_bins) != n_r_bins or n_r_bins <= 0:
            raise ValueError("n_r_bins must be a positive integer")
        if n_r_bins > nR_native:
            raise ValueError(
                f"n_r_bins={n_r_bins} exceeds native radial bin count nR={nR_native}. "
                "Refining radial bins is not supported."
            )
        if n_r_bins == nR_native:
            r_edges = r_edges_native
        else:
            idx = np.array([(i * nR_native) // n_r_bins for i in range(n_r_bins + 1)], dtype=int)
            r_edges = r_edges_native[idx]

    nR = len(r_edges) - 1
    if n_theta_bins is None:
        n_theta_bins = n_bins_native

    ok = r_grid.to_base_units() > (0.0 * r_grid.units)
    theta_max = float(np.max(np.abs(disk_frame.theta_from_midplane[ok])))
    if theta_max == 0.0 or not np.isfinite(theta_max):
        raise ValueError("Invalid disk-frame theta extent; cannot build theta bins")

    theta_edges = np.linspace(-theta_max, theta_max, n_theta_bins + 1)
    return r_edges, theta_edges, n_theta_bins


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return numerator / denominator with disk-mask friendly zero handling."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    ratio = np.zeros_like(num, dtype=float)
    finite_num = np.isfinite(num)
    finite_den = np.isfinite(den)
    ok = finite_num & finite_den & (den > 0.0)
    np.divide(num, den, out=ratio, where=ok)
    ratio[finite_num & finite_den & (den == 0.0) & (num > 0.0)] = np.inf
    ratio[~finite_num | ~finite_den | (num < 0.0)] = 0.0
    ratio[np.isnan(ratio)] = 0.0
    return ratio


def _compute_joos_criterion_data(
    rho: Quantity,
    disk_frame: DiskFrameData,
    Pth: Quantity,
    valid: np.ndarray,
    rho_disk_min: Quantity,
    max_poloidal_mach: float,
    rotational_support_factor: float,
) -> JoosCriterionData:
    """Evaluate the settled rotating-disc criteria on individual cells."""
    if not np.isfinite(max_poloidal_mach) or max_poloidal_mach <= 0.0:
        raise ValueError("max_poloidal_mach must be positive and finite")
    if not np.isfinite(rotational_support_factor) or rotational_support_factor <= 0.0:
        raise ValueError("rotational_support_factor must be positive and finite")

    vphi_abs = np.abs(disk_frame.vphi_d.to_base_units().magnitude)
    vR_abs = np.abs(disk_frame.vR_d.to_base_units().magnitude)
    vz_abs = np.abs(disk_frame.vz_d.to_base_units().magnitude)
    rho_base = rho.to_base_units().magnitude
    rot = (0.5 * rho * (disk_frame.vphi_d.to_base_units() ** 2)).to_base_units().magnitude
    P = Pth.to_base_units().magnitude
    rho_min = float(rho_disk_min.to_base_units().magnitude)

    sound_speed = np.sqrt(_safe_ratio(P, rho_base))
    q_pol = _safe_ratio(
        max_poloidal_mach * sound_speed,
        np.hypot(vR_abs, vz_abs),
    )
    q_rot = _safe_ratio(rot, rotational_support_factor * P)
    q_rho = _safe_ratio(rho_base, np.full_like(rho_base, rho_min, dtype=float))
    for q in (q_pol, q_rot, q_rho):
        q[~valid] = 0.0

    hard_pass = (
        valid
        & (q_pol > 1.0)
        & (q_rot > 1.0)
        & (q_rho > 1.0)
    )
    return JoosCriterionData(
        valid=valid,
        q_pol=q_pol,
        q_rot=q_rot,
        q_rho=q_rho,
        hard_pass=hard_pass,
    )


def _apply_hard_midplane_connectivity(
    cell_pass: np.ndarray,
    disk_frame: DiskFrameData,
    valid: np.ndarray,
) -> np.ndarray:
    """Keep the contiguous hard-pass chain around the closest hard-pass midplane cell."""
    connected = np.zeros_like(cell_pass, dtype=bool)
    nr, _ntheta, nphi = cell_pass.shape
    for ir in range(nr):
        for iphi in range(nphi):
            populated = np.flatnonzero(valid[ir, :, iphi])
            if populated.size == 0:
                continue

            pass_idx = populated[cell_pass[ir, populated, iphi]]
            if pass_idx.size == 0:
                continue

            theta_col = disk_frame.theta_from_midplane[ir, populated, iphi]
            order_local = np.argsort(theta_col)
            pop_order = populated[order_local]
            seed = int(pass_idx[np.argmin(np.abs(disk_frame.theta_from_midplane[ir, pass_idx, iphi]))])
            seed_pos = np.flatnonzero(pop_order == seed)
            if seed_pos.size == 0:
                continue

            k = int(seed_pos[0])
            while k < pop_order.size and cell_pass[ir, pop_order[k], iphi]:
                connected[ir, pop_order[k], iphi] = True
                k += 1

            k = int(seed_pos[0]) - 1
            while k >= 0 and cell_pass[ir, pop_order[k], iphi]:
                connected[ir, pop_order[k], iphi] = True
                k -= 1

    return connected


def _apply_radial_midplane_connectivity(
    values: np.ndarray,
    disk_frame: DiskFrameData,
    valid: np.ndarray,
    *,
    seed_min: float,
    floor: float,
) -> np.ndarray:
    """Keep the outward radial chain connected to the first disc-like column.

    Each azimuth is treated independently. The closest valid cell to the disc
    midplane represents its radial column. Starting at the first column whose
    midplane value reaches ``seed_min``, the connected value is capped by the
    weakest midplane value encountered at smaller radii. Propagation stops when
    a midplane value reaches ``floor``, so detached outer material cannot
    re-enter the disc.
    """
    if not 0.0 <= floor < seed_min <= 1.0:
        raise ValueError("radial connectivity requires 0 <= floor < seed_min <= 1")

    array = np.asarray(values, dtype=float)
    connected = np.zeros_like(array, dtype=float)
    nr, _ntheta, nphi = array.shape

    for iphi in range(nphi):
        mid_indices = np.full(nr, -1, dtype=int)
        mid_values = np.zeros(nr, dtype=float)
        for ir in range(nr):
            populated = np.flatnonzero(valid[ir, :, iphi])
            if populated.size == 0:
                continue
            mid = int(
                populated[
                    np.argmin(
                        np.abs(
                            disk_frame.theta_from_midplane[
                                ir, populated, iphi
                            ]
                        )
                    )
                ]
            )
            mid_indices[ir] = mid
            mid_values[ir] = float(array[ir, mid, iphi])

        seeds = np.flatnonzero(mid_values >= seed_min)
        if seeds.size == 0:
            continue

        running = float(mid_values[int(seeds[0])])
        for ir in range(int(seeds[0]), nr):
            if mid_indices[ir] < 0 or mid_values[ir] <= floor:
                break
            running = min(running, float(mid_values[ir]))
            populated = valid[ir, :, iphi]
            connected[ir, populated, iphi] = np.minimum(
                array[ir, populated, iphi],
                running,
            )

    return np.clip(connected, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _soft_cut_from_ratio(q: np.ndarray, delta: float, floor: float) -> np.ndarray:
    """Map a hard Joos margin q > 1 to a continuous score in [0, 1]."""
    if delta <= 0.0 or not np.isfinite(delta):
        raise ValueError("soft Joos delta values must be positive and finite")
    if floor < 0.0 or floor >= 0.5:
        raise ValueError("weight_floor must satisfy 0 <= weight_floor < 0.5")
    q = np.asarray(q, dtype=float)
    x = np.full_like(q, -np.inf, dtype=float)
    positive = q > 0.0
    x[positive] = np.log(q[positive]) / float(delta)
    score = _sigmoid(x)
    if floor > 0.0:
        score = np.where(score <= floor, 0.0, score)
    score[~np.isfinite(score)] = 0.0
    upper_open = np.nextafter(1.0, 0.0)
    return np.clip(score, 0.0, upper_open)


def _soft_delta_value(
    soft_delta: Union[float, Mapping[str, float]],
    key: str,
) -> float:
    if isinstance(soft_delta, Mapping):
        if key in soft_delta:
            return float(soft_delta[key])
        if "default" in soft_delta:
            return float(soft_delta["default"])
        raise KeyError(f"soft_delta mapping is missing {key!r} and 'default'")
    return float(soft_delta)


def _compute_local_soft_joos_weight(
    criteria: JoosCriterionData,
    soft_delta: Union[float, Mapping[str, float]],
    weight_floor: float,
) -> np.ndarray:
    """Compute the local fuzzy-AND of the Joos criterion scores."""
    s_pol = _soft_cut_from_ratio(
        criteria.q_pol,
        _soft_delta_value(soft_delta, "mach"),
        weight_floor,
    )
    s_rot = _soft_cut_from_ratio(
        criteria.q_rot,
        _soft_delta_value(soft_delta, "rot"),
        weight_floor,
    )
    s_rho = _soft_cut_from_ratio(
        criteria.q_rho,
        _soft_delta_value(soft_delta, "rho"),
        weight_floor,
    )
    w_local = np.minimum.reduce([s_pol, s_rot, s_rho])
    w_local[~criteria.valid] = 0.0
    return np.clip(w_local, 0.0, 1.0)


def _smooth_disk_frame_kinematics(
    rho: Quantity,
    disk_frame: DiskFrameData,
    smoothing_bins: Tuple[float, float, float],
) -> DiskFrameData:
    """Return density-weighted bulk velocities smoothed on the native mesh."""
    sigma = np.asarray(smoothing_bins, dtype=float)
    if sigma.shape != (3,) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("kinematic_smoothing_bins must contain three positive values")

    rho_values = np.asarray(rho.to_base_units().magnitude, dtype=float)
    smooth_rho = gaussian_filter(
        rho_values,
        sigma=tuple(sigma),
        mode=("nearest", "nearest", "wrap"),
    )
    density_floor = np.finfo(float).tiny

    def smooth(component: Quantity) -> Quantity:
        momentum = rho_values * np.asarray(component.magnitude, dtype=float)
        gaussian_filter(
            momentum,
            sigma=tuple(sigma),
            mode=("nearest", "nearest", "wrap"),
            output=momentum,
        )
        np.divide(momentum, np.maximum(smooth_rho, density_floor), out=momentum)
        return Quantity(momentum, component.units)

    return DiskFrameData(
        k_hat=disk_frame.k_hat,
        R_d=disk_frame.R_d,
        z_d=disk_frame.z_d,
        vR_d=smooth(disk_frame.vR_d),
        vphi_d=smooth(disk_frame.vphi_d),
        vz_d=smooth(disk_frame.vz_d),
        theta_from_midplane=disk_frame.theta_from_midplane,
        phi_d=disk_frame.phi_d,
    )


def _disk_phi_bins(disk_frame: DiskFrameData, nphi: int) -> np.ndarray:
    """Return disc-frame azimuth bins for every native cell."""
    if disk_frame.phi_d is None:
        native = np.arange(nphi, dtype=np.int32)[None, None, :]
        return np.broadcast_to(native, disk_frame.R_d.shape)
    fraction = np.mod(disk_frame.phi_d + np.pi, 2.0 * np.pi) / (2.0 * np.pi)
    return np.minimum((fraction * nphi).astype(np.int32), nphi - 1)


def _fill_periodic_midplane_holes(
    values: np.ndarray,
    populated: np.ndarray,
) -> np.ndarray:
    """Fill empty disc-plane bins from their nearest populated periodic neighbour."""
    if np.all(populated):
        return values
    if not np.any(populated):
        raise ValueError("No populated disc-frame midplane bins")

    nphi = values.shape[1]
    values_pad = np.concatenate([values, values, values], axis=1)
    populated_pad = np.concatenate([populated, populated, populated], axis=1)
    nearest = distance_transform_edt(
        ~populated_pad,
        return_distances=False,
        return_indices=True,
    )
    filled_pad = values_pad[tuple(nearest)]
    return filled_pad[:, nphi : 2 * nphi]


def _cylindrical_midplane_support(
    w_local: np.ndarray,
    disk_frame: DiskFrameData,
    valid: np.ndarray,
    r_edges: Quantity,
    *,
    seed_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cap cells by the disc score at their cylindrical midplane footpoint.

    Unlike the former fixed-spherical-radius weakest-link walk, this maps each
    cell to the disc midplane at the same disc-frame cylindrical radius and
    azimuth. A column is rejected when its midplane footpoint is not disc-like.
    """
    if not 0.0 <= seed_min <= 1.0:
        raise ValueError("weight_m0/seed_min must lie in [0, 1]")
    if disk_frame.R_d.shape != w_local.shape or valid.shape != w_local.shape:
        raise ValueError("disk-frame geometry, validity, and weight shapes must match")

    nr = len(r_edges) - 1
    nphi = w_local.shape[2]
    edges = np.asarray(r_edges.to_base_units().magnitude, dtype=float)
    phi_bins = _disk_phi_bins(disk_frame, nphi)
    support = np.zeros((nr, nphi), dtype=float)
    populated_grid = np.zeros((nr, nphi), dtype=bool)

    for ir in range(w_local.shape[0]):
        for iphi in range(nphi):
            populated = np.flatnonzero(valid[ir, :, iphi])
            if populated.size == 0:
                continue
            mid = int(
                populated[
                    np.argmin(
                        np.abs(
                            disk_frame.theta_from_midplane[ir, populated, iphi]
                        )
                    )
                ]
            )
            r_bin = int(np.digitize(disk_frame.R_d[ir, mid, iphi], edges) - 1)
            if r_bin < 0 or r_bin >= nr:
                continue
            p_bin = int(phi_bins[ir, mid, iphi])
            support[r_bin, p_bin] = max(
                support[r_bin, p_bin],
                float(w_local[ir, mid, iphi]),
            )
            populated_grid[r_bin, p_bin] = True

    support = _fill_periodic_midplane_holes(support, populated_grid)
    connected_support = np.where(support >= seed_min, support, 0.0)
    connected = np.zeros_like(w_local, dtype=float)
    for ir in range(w_local.shape[0]):
        r_bin = np.digitize(disk_frame.R_d[ir], edges).astype(np.int32) - 1
        in_grid = valid[ir] & (r_bin >= 0) & (r_bin < nr)
        if not np.any(in_grid):
            continue
        r_use = np.clip(r_bin, 0, nr - 1)
        footpoint = connected_support[r_use, phi_bins[ir]]
        connected[ir, in_grid] = np.minimum(
            w_local[ir, in_grid],
            footpoint[in_grid],
        )

    return np.clip(connected, 0.0, 1.0), support


def _midplane_density_support(
    rho: Quantity,
    disk_frame: DiskFrameData,
    valid: np.ndarray,
    r_edges: Quantity,
    rho_midplane_min: Quantity,
    *,
    softness_dex: float,
    smoothing_bins: float,
    floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth annular midplane-density support score.

    The density nearest the disc-frame midplane is accumulated on the existing
    cylindrical ``(R, phi)`` grid. Its azimuthal median is smoothed in log
    density before being compared with ``rho_midplane_min``. The resulting
    radial score is also mapped back to every native cell at the same
    disc-frame cylindrical radius so the condition can cap a complete vertical
    column without imposing a local density cut on the disc atmosphere.
    """
    threshold = np.asarray(
        rho_midplane_min.to(rho.units).magnitude,
        dtype=float,
    )
    if threshold.ndim != 0 or not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("rho_midplane_min must be a finite positive scalar density")
    if not np.isfinite(softness_dex) or softness_dex <= 0.0:
        raise ValueError("midplane_density_softness_dex must be positive and finite")
    if not np.isfinite(smoothing_bins) or smoothing_bins <= 0.0:
        raise ValueError("midplane_density_smoothing_bins must be positive and finite")
    if disk_frame.R_d.shape != rho.shape or valid.shape != rho.shape:
        raise ValueError("density, disk-frame geometry, and validity must match")

    edges = np.asarray(r_edges.to_base_units().magnitude, dtype=float)
    nr = edges.size - 1
    nphi = rho.shape[2]
    phi_bins = _disk_phi_bins(disk_frame, nphi)
    rho_values = np.asarray(rho.to_base_units().magnitude, dtype=float)
    midplane = np.full((nr, nphi), -np.inf, dtype=float)

    for ir in range(rho.shape[0]):
        for iphi in range(nphi):
            populated = np.flatnonzero(valid[ir, :, iphi])
            if populated.size == 0:
                continue
            mid = int(
                populated[
                    np.argmin(
                        np.abs(
                            disk_frame.theta_from_midplane[ir, populated, iphi]
                        )
                    )
                ]
            )
            value = float(rho_values[ir, mid, iphi])
            r_bin = int(np.digitize(disk_frame.R_d[ir, mid, iphi], edges) - 1)
            if r_bin < 0 or r_bin >= nr or not np.isfinite(value) or value <= 0.0:
                continue
            p_bin = int(phi_bins[ir, mid, iphi])
            midplane[r_bin, p_bin] = max(midplane[r_bin, p_bin], value)

    midplane[midplane == -np.inf] = np.nan
    profile = np.full(nr, np.nan, dtype=float)
    populated_rows = np.any(np.isfinite(midplane), axis=1)
    profile[populated_rows] = np.nanmedian(midplane[populated_rows], axis=1)
    populated_radius = np.flatnonzero(np.isfinite(profile) & (profile > 0.0))
    if populated_radius.size == 0:
        raise ValueError("No positive disc-frame midplane densities were found")
    missing_radius = np.flatnonzero(~np.isfinite(profile) | (profile <= 0.0))
    if missing_radius.size:
        profile[missing_radius] = np.interp(
            missing_radius,
            populated_radius,
            profile[populated_radius],
        )

    log_profile = gaussian_filter1d(
        np.log10(profile),
        sigma=float(smoothing_bins),
        mode="nearest",
    )
    smooth_profile = np.power(10.0, log_profile)
    radial_score = _soft_cut_from_ratio(
        smooth_profile / float(threshold),
        float(softness_dex) * np.log(10.0),
        floor,
    )

    cell_score = np.zeros_like(rho_values, dtype=np.float32)
    for ir in range(rho.shape[0]):
        r_bin = np.digitize(disk_frame.R_d[ir], edges).astype(np.int32) - 1
        in_grid = valid[ir] & (r_bin >= 0) & (r_bin < nr)
        if np.any(in_grid):
            cell_score[ir, in_grid] = radial_score[
                np.clip(r_bin[in_grid], 0, nr - 1)
            ]

    return cell_score, smooth_profile, radial_score


def _apply_connected_midplane_coherence(
    values: np.ndarray,
    midplane_support: np.ndarray,
    disk_frame: DiskFrameData,
    valid: np.ndarray,
    r_edges: Quantity,
    *,
    seed_min: float,
    floor: float,
    soft_delta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply radial weakest-link and annular coherence to midplane support.

    Each azimuth is propagated outward from its first disc-like seed with a
    running minimum, so a detached island cannot re-enter after a failed
    column. The phi-mean of that connected support is smoothed, mapped through
    the same fuzzy threshold used by the radial Joos criterion, and propagated
    monotonically outward. An azimuthally narrow streamer therefore decays
    toward zero rather than defining the outer disc. Both continuous scores cap
    the local 3-D weight.
    """
    if not 0.0 <= floor < seed_min <= 1.0:
        raise ValueError(
            "connected coherence requires 0 <= floor < seed_min <= 1"
        )
    if not np.isfinite(soft_delta) or soft_delta <= 0.0:
        raise ValueError("connected coherence soft_delta must be positive")

    edges = np.asarray(r_edges.to_base_units().magnitude, dtype=float)
    support = np.asarray(midplane_support, dtype=float)
    if support.ndim != 2:
        raise ValueError("midplane_support must have shape (radius, phi)")
    if support.shape[0] != edges.size - 1:
        raise ValueError("midplane support and radial edges do not match")
    if values.shape != disk_frame.R_d.shape or values.shape != valid.shape:
        raise ValueError("values, disk-frame geometry, and validity must match")

    support = np.clip(support, 0.0, 1.0)
    radial_support = np.zeros_like(support, dtype=float)
    for iphi in range(support.shape[1]):
        seeds = np.flatnonzero(support[:, iphi] >= seed_min)
        if seeds.size == 0:
            continue
        start = int(seeds[0])
        running = float(support[start, iphi])
        for ir in range(start, support.shape[0]):
            running = min(running, float(support[ir, iphi]))
            if running <= floor:
                break
            radial_support[ir, iphi] = running

    coverage = np.mean(radial_support, axis=1)
    coverage_smooth = gaussian_filter1d(coverage, sigma=2.0, mode="nearest")
    coherence_score = _soft_cut_from_ratio(
        coverage_smooth / seed_min,
        soft_delta,
        floor,
    )
    coherence = np.zeros_like(coherence_score)
    coherent_seeds = np.flatnonzero(coherence_score >= seed_min)
    if coherent_seeds.size:
        start = int(coherent_seeds[0])
        coherence[start:] = np.minimum.accumulate(coherence_score[start:])

    coherent_support = np.minimum(radial_support, coherence[:, None])
    phi_bins = _disk_phi_bins(disk_frame, support.shape[1])
    radial_values = np.zeros_like(values, dtype=float)
    coherent_values = np.zeros_like(values, dtype=float)
    for ir in range(values.shape[0]):
        r_bin = np.digitize(disk_frame.R_d[ir], edges).astype(np.int32) - 1
        in_grid = valid[ir] & (r_bin >= 0) & (r_bin < support.shape[0])
        if not np.any(in_grid):
            continue
        r_use = np.clip(r_bin, 0, support.shape[0] - 1)
        radial_cap = radial_support[r_use, phi_bins[ir]]
        coherent_cap = coherent_support[r_use, phi_bins[ir]]
        radial_values[ir, in_grid] = np.minimum(
            values[ir, in_grid],
            radial_cap[in_grid],
        )
        coherent_values[ir, in_grid] = np.minimum(
            values[ir, in_grid],
            coherent_cap[in_grid],
        )

    radial_values[radial_values <= floor] = 0.0
    coherent_values[coherent_values <= floor] = 0.0
    return (
        np.clip(coherent_values, 0.0, 1.0),
        np.clip(radial_values, 0.0, 1.0),
        radial_support,
        coverage,
        coherence,
    )


def _apply_vertical_connectivity(
    values: np.ndarray,
    disk_frame: DiskFrameData,
    valid: np.ndarray,
    r_edges: Quantity,
    theta_edges: np.ndarray,
    *,
    floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply vertical weakest-link connectivity in the disc frame.

    Local scores are accumulated on a cylindrical ``(R, theta_d, phi_d)``
    grid. Each side is propagated independently away from the midplane with a
    running minimum, preventing material from re-entering the disc above a
    failed layer without imposing a height or density cutoff.
    """
    if not 0.0 <= floor < 1.0:
        raise ValueError("vertical connectivity requires 0 <= floor < 1")
    if values.shape != disk_frame.R_d.shape or values.shape != valid.shape:
        raise ValueError("values, disk-frame geometry, and validity must match")

    radial_edges = np.asarray(r_edges.to_base_units().magnitude, dtype=float)
    vertical_edges = np.asarray(theta_edges, dtype=float)
    if vertical_edges.ndim != 1 or vertical_edges.size < 3:
        raise ValueError("theta_edges must be a one-dimensional bin-edge array")
    if np.any(~np.isfinite(vertical_edges)) or np.any(np.diff(vertical_edges) <= 0.0):
        raise ValueError("theta_edges must be finite and strictly increasing")

    nr = radial_edges.size - 1
    ntheta = vertical_edges.size - 1
    nphi = values.shape[2]
    support = np.zeros((nr, ntheta, nphi), dtype=np.float32)
    populated = np.zeros_like(support, dtype=bool)
    support_flat = support.reshape(-1)
    populated_flat = populated.reshape(-1)
    phi_bins = _disk_phi_bins(disk_frame, nphi)

    for ir in range(values.shape[0]):
        r_bin = np.digitize(disk_frame.R_d[ir], radial_edges).astype(np.int32) - 1
        theta_bin = (
            np.digitize(disk_frame.theta_from_midplane[ir], vertical_edges)
            .astype(np.int32)
            - 1
        )
        in_grid = (
            valid[ir]
            & (r_bin >= 0)
            & (r_bin < nr)
            & (theta_bin >= 0)
            & (theta_bin < ntheta)
        )
        if not np.any(in_grid):
            continue
        flat_index = (
            (r_bin[in_grid] * ntheta + theta_bin[in_grid]) * nphi
            + phi_bins[ir][in_grid]
        )
        np.maximum.at(
            support_flat,
            flat_index,
            np.asarray(values[ir][in_grid], dtype=np.float32),
        )
        populated_flat[flat_index] = True

    row_counts = np.sum(populated, axis=2)
    partial_rows = np.argwhere((row_counts > 0) & (row_counts < nphi))
    for ir, itheta in partial_rows:
        row_populated = populated[ir, itheta]
        known = np.flatnonzero(row_populated)
        missing = np.flatnonzero(~row_populated)
        clockwise = np.mod(missing[:, None] - known[None, :], nphi)
        counterclockwise = np.mod(known[None, :] - missing[:, None], nphi)
        nearest = known[np.argmin(np.minimum(clockwise, counterclockwise), axis=1)]
        support[ir, itheta, missing] = support[ir, itheta, nearest]
        populated[ir, itheta, missing] = True

    theta_centers = 0.5 * (vertical_edges[:-1] + vertical_edges[1:])
    negative = np.flatnonzero(theta_centers < 0.0)[::-1]
    positive = np.flatnonzero(theta_centers >= 0.0)
    for indices in (negative, positive):
        if indices.size:
            support[:, indices, :] = np.minimum.accumulate(
                support[:, indices, :],
                axis=1,
            )

    vertical_values = np.zeros_like(values, dtype=float)
    for ir in range(values.shape[0]):
        r_bin = np.digitize(disk_frame.R_d[ir], radial_edges).astype(np.int32) - 1
        theta_bin = (
            np.digitize(disk_frame.theta_from_midplane[ir], vertical_edges)
            .astype(np.int32)
            - 1
        )
        in_grid = (
            valid[ir]
            & (r_bin >= 0)
            & (r_bin < nr)
            & (theta_bin >= 0)
            & (theta_bin < ntheta)
        )
        if not np.any(in_grid):
            continue
        r_use = np.clip(r_bin, 0, nr - 1)
        theta_use = np.clip(theta_bin, 0, ntheta - 1)
        vertical_cap = support[r_use, theta_use, phi_bins[ir]]
        vertical_values[ir, in_grid] = np.minimum(
            values[ir, in_grid],
            vertical_cap[in_grid],
        )

    vertical_values[vertical_values <= floor] = 0.0
    return np.clip(vertical_values, 0.0, 1.0), support


def _build_disk_weight(
    hard_mask: np.ndarray,
    criteria: JoosCriterionData,
    disk_frame: DiskFrameData,
    *,
    weight_mode: str,
    soft_delta: Union[float, Mapping[str, float]],
    weight_m0: float,
    weight_floor: float,
    r_edges: Optional[Quantity] = None,
    theta_edges: Optional[np.ndarray] = None,
    midplane_density_score: Optional[np.ndarray] = None,
) -> np.ndarray:
    mode = str(weight_mode).lower()
    if mode in {"cell", "binary", "hard"}:
        return hard_mask.astype(float)
    if mode in {"soft", "soft_joos", "soft-connected", "soft_connected"}:
        w_local = _compute_local_soft_joos_weight(
            criteria,
            soft_delta,
            weight_floor,
        )
        if mode in {"soft", "soft_joos"}:
            return w_local
        if r_edges is None or theta_edges is None:
            raise ValueError(
                "soft_connected mode requires radial and disc-frame theta edges"
            )
        _w_cylindrical, midplane_support = _cylindrical_midplane_support(
            w_local,
            disk_frame,
            criteria.valid,
            r_edges,
            seed_min=float(weight_m0),
        )
        if midplane_density_score is not None:
            density_score = np.asarray(midplane_density_score, dtype=float)
            if density_score.shape != (len(r_edges) - 1,):
                raise ValueError(
                    "midplane_density_score must match the cylindrical radial bins"
                )
            midplane_support = np.minimum(
                midplane_support,
                density_score[:, None],
            )
        w_midplane, _w_radial, _radial_support, _coverage, _coherence = (
            _apply_connected_midplane_coherence(
                w_local,
                midplane_support,
                disk_frame,
                criteria.valid,
                r_edges,
                seed_min=float(weight_m0),
                floor=float(weight_floor),
                soft_delta=_soft_delta_value(soft_delta, "mach"),
            )
        )
        w_connected, _vertical_support = _apply_vertical_connectivity(
            w_midplane,
            disk_frame,
            criteria.valid,
            r_edges,
            theta_edges,
            floor=float(weight_floor),
        )
        return w_connected
    raise ValueError(
        "weight_mode must be one of 'none', 'cell', 'soft', or 'soft_connected'"
    )


# db-keywords: disk-mask
# db-role: canonical
def set_mask_from_joos_disk(
    model: "Model",
    rho_disk_min: Quantity,
    *,
    max_poloidal_mach: float = 1.0,
    rotational_support_factor: float = 2.0,
    rho_core_min: Optional[Quantity] = None,
    r_max_for_axis: Quantity,
    n_r_bins: Optional[int] = None,
    n_theta_bins: Optional[int] = None,
    weight_mode: str = "soft_connected",
    weight_m0: float = 0.50,
    weight_floor: float = 1e-4,
    soft_delta: Union[float, Mapping[str, float]] = 0.20,
    kinematic_smoothing_bins: Tuple[float, float, float] = (2.0, 1.0, 2.0),
    rho_midplane_min: Optional[Quantity] = None,
    midplane_density_softness_dex: float = 0.30,
    midplane_density_smoothing_bins: float = 2.0,
) -> "SubModel":
    """Create a disk region mask using the Joos et al. (2012) kinematic/pressure criteria.

    This function:

    1) Estimates the disk angular-momentum axis and transforms velocities into a disk frame.
    2) Evaluates Joos-style criteria directly on individual cells:
       - poloidal motion is subsonic
       - rotational support dominates over thermal pressure
       - density exceeds rho_disk_min
    3) For ``soft_connected``, extracts density-weighted bulk kinematics,
       requires radial connection to a disc-like cylindrical midplane
       footpoint, caps narrow outer sectors by their annular coherence, and
       optionally requires a dense azimuthally coherent midplane reservoir,
       and prevents vertical re-entry with a weakest-link connection to the
       midplane on each side.
    4) Registers the boolean mask as model.gas["disk_mask"].
    5) Returns a SubModel with the same boolean mask.

    By default, it registers the connected continuous material split as
    ``model.gas["disk_weight"]`` and defines the boolean disc mask at weight
    0.5. Use ``weight_mode="cell"`` for a binary criterion mask or
    ``weight_mode="soft"`` for local scores without connectivity.

    Parameters
    ----------
    model : Model
        DiskBridge model with a spherical mesh and gas fields.
    rho_disk_min : Quantity
        Minimum gas density to be considered disc-like.
    max_poloidal_mach : float, optional
        Maximum settled-flow poloidal Mach number. The sound speed is
        ``sqrt(P/rho)`` and the poloidal speed is ``hypot(v_R,d, v_z,d)`` in
        the measured disc frame.
    rotational_support_factor : float, optional
        Minimum ratio denominator for rotational kinetic pressure versus
        thermal pressure. A cell passes when
        ``0.5 rho v_phi,d**2 > rotational_support_factor P``.
    rho_core_min : Quantity or None, optional
        Density defining the axis-estimation core. Defaults to ten times
        ``rho_disk_min``.
    r_max_for_axis : Quantity
        Finite radius supporting only the angular-momentum axis estimate. It
        does not truncate the material mask or weight.
    n_r_bins, n_theta_bins : int or None, optional
        Optional radial downsampling and number of disc-frame polar bins.
    weight_mode : str, optional
        ``"none"`` skips ``disk_weight``; ``"cell"`` uses the binary mask;
        ``"soft"`` uses local Joos margins; and the default
        ``"soft_connected"`` adds
        cylindrical radial and vertical connectivity plus a midplane annular-
        coherence cap.
    weight_m0 : float, optional
        Minimum midplane seed weight for connected classification.
    weight_floor : float, optional
        Scores at or below this value are truncated to zero.
    soft_delta : float or mapping, optional
        Shared or criterion-specific softness in log-ratio space. Mapping keys
        are ``"mach"``, ``"rot"``, ``"rho"``, or ``"default"``.
    kinematic_smoothing_bins : tuple of float, optional
        Gaussian sigma in native ``(r, theta, phi)`` cells for the
        density-weighted velocity background used by ``soft_connected``.
    rho_midplane_min : Quantity or None, optional
        Midplane gas density giving a support score of 0.5. When provided for
        ``soft_connected``, the smoothed azimuthal-median density at each
        disc-frame cylindrical radius caps the full vertical column. This is
        separate from the cell-local ``rho_disk_min`` criterion.
    midplane_density_softness_dex : float, optional
        Logistic transition width in dex around ``rho_midplane_min``.
    midplane_density_smoothing_bins : float, optional
        Gaussian smoothing width in cylindrical radial bins for the
        azimuthal-median log midplane-density profile.

    Returns
    -------
    SubModel
        Disc region with a boolean mask.

    Raises
    ------
    ValueError
        If the mesh, axis support, binning, or weight parameters are invalid.
    KeyError
        If required gas fields are missing.
    """
    mesh = model.mesh
    if mesh.coord_system != "spherical":
        raise ValueError(
            f"set_mask_from_joos_disk only supports spherical coordinates, got {mesh.coord_system}"
        )

    r_c = mesh.centers("r")
    theta_c = mesh.centers("theta")
    phi_c = mesh.centers("phi")
    r_e = mesh.edges("r")
    theta_e = mesh.edges("theta")
    phi_e = mesh.edges("phi")
    if r_c is None or theta_c is None or phi_c is None:
        raise ValueError("Mesh is missing one or more spherical center axes")
    if r_e is None or theta_e is None or phi_e is None:
        raise ValueError("Mesh is missing one or more spherical edge axes")

    n_bins_native = len(theta_c)

    r3 = (r_e[1:] ** 3 - r_e[:-1] ** 3) / 3.0
    theta_e_rad = theta_e.to("radian").magnitude
    dcos = np.cos(theta_e_rad[:-1]) - np.cos(theta_e_rad[1:])
    phi_e_rad = phi_e.to("radian").magnitude
    dphi = phi_e_rad[1:] - phi_e_rad[:-1]
    dV = r3[:, None, None] * dcos[None, :, None] * dphi[None, None, :]
    rho = model.gas["density"].data.to_base_units()
    vr = model.gas["vr"].data.to_base_units()
    vphi = model.gas["vphi"].data.to_base_units()
    vtheta = model.gas["vtheta"].data.to_base_units()

    r_grid, theta_grid, phi_grid = spherical_grids(
        r_c.to_base_units(),
        theta_c.to("radian"),
        phi_c.to("radian"),
    )

    if rho_core_min is None:
        rho_core_min = 10.0 * rho_disk_min

    k_hat = _compute_disk_orientation(
        model, rho, dV, r_grid, theta_grid, phi_grid,
        vr, vphi, vtheta, rho_core_min, r_max_for_axis
    )

    disk_frame = _transform_to_disk_frame(
        mesh, r_grid, theta_grid, phi_grid, vr, vphi, vtheta, k_hat
    )

    Pth = _compute_thermal_pressure(model, rho)

    r_edges, theta_edges, n_theta_bins = _setup_binning(
        mesh, n_r_bins, n_theta_bins, n_bins_native, disk_frame, r_grid
    )
    nR = len(r_edges) - 1

    r_bin = np.digitize(r_grid.to_base_units().magnitude, r_edges.magnitude) - 1
    t_bin = np.digitize(disk_frame.theta_from_midplane, theta_edges) - 1
    valid = (r_bin >= 0) & (r_bin < nR) & (t_bin >= 0) & (t_bin < n_theta_bins)

    criteria = _compute_joos_criterion_data(
        rho,
        disk_frame,
        Pth,
        valid,
        rho_disk_min,
        float(max_poloidal_mach),
        float(rotational_support_factor),
    )

    mask_vertical = _apply_hard_midplane_connectivity(
        criteria.hard_pass,
        disk_frame,
        valid,
    )
    mask = _apply_radial_midplane_connectivity(
        mask_vertical,
        disk_frame,
        valid,
        seed_min=1.0,
        floor=0.0,
    ) > 0.5

    mode = str(weight_mode).lower()
    connected_mode = mode in {"soft-connected", "soft_connected"}
    if rho_midplane_min is not None and not connected_mode:
        raise ValueError(
            "rho_midplane_min is only supported with weight_mode='soft_connected'"
        )
    if mode != "none":
        weight_frame = disk_frame
        weight_criteria = criteria
        if connected_mode:
            weight_frame = _smooth_disk_frame_kinematics(
                rho,
                disk_frame,
                kinematic_smoothing_bins,
            )
            weight_criteria = _compute_joos_criterion_data(
                rho,
                weight_frame,
                Pth,
                valid,
                rho_disk_min,
                float(max_poloidal_mach),
                float(rotational_support_factor),
            )
        midplane_density_cell_score = None
        midplane_density_profile = None
        midplane_density_score = None
        if rho_midplane_min is not None:
            (
                midplane_density_cell_score,
                midplane_density_profile,
                midplane_density_score,
            ) = _midplane_density_support(
                rho,
                weight_frame,
                valid,
                r_edges,
                rho_midplane_min,
                softness_dex=float(midplane_density_softness_dex),
                smoothing_bins=float(midplane_density_smoothing_bins),
                floor=float(weight_floor),
            )
        w_disk = _build_disk_weight(
            mask,
            weight_criteria,
            weight_frame,
            weight_mode=weight_mode,
            soft_delta=soft_delta,
            weight_m0=weight_m0,
            weight_floor=weight_floor,
            r_edges=r_edges,
            theta_edges=theta_edges,
            midplane_density_score=midplane_density_score,
        )
        if connected_mode:
            mask = w_disk >= 0.5
        if midplane_density_cell_score is not None:
            model.gas_register(
                "disk_midplane_density_support",
                Field(
                    data=Quantity(
                        np.clip(midplane_density_cell_score, 0.0, 1.0),
                        "dimensionless",
                    ),
                    quantity="mask",
                    axis_order=mesh.axis_names(),
                    attrs={
                        "source": "disc_frame_midplane_density",
                        "rho_midplane_min_g_cm3": float(
                            rho_midplane_min.to("g/cm^3").magnitude
                        ),
                        "softness_dex": float(midplane_density_softness_dex),
                        "smoothing_bins": float(midplane_density_smoothing_bins),
                        "azimuthal_statistic": "median",
                        "radial_profile_min_g_cm3": float(
                            Quantity(midplane_density_profile, rho.units)
                            .to("g/cm^3")
                            .magnitude.min()
                        ),
                        "radial_profile_max_g_cm3": float(
                            Quantity(midplane_density_profile, rho.units)
                            .to("g/cm^3")
                            .magnitude.max()
                        ),
                        "radial_centres_au": tuple(
                            float(value)
                            for value in (
                                0.5 * (r_edges[:-1] + r_edges[1:])
                            ).to("au").magnitude
                        ),
                        "radial_profile_g_cm3": tuple(
                            float(value)
                            for value in Quantity(
                                midplane_density_profile,
                                rho.units,
                            ).to("g/cm^3").magnitude
                        ),
                        "radial_support_score": tuple(
                            float(value) for value in midplane_density_score
                        ),
                    },
                ),
            )
        w_field = Field(
            data=Quantity(np.clip(w_disk, 0.0, 1.0), "dimensionless"),
            quantity="mask",
            axis_order=mesh.axis_names(),
            attrs={
                "source": "joos_disk",
                "disk_axis_cartesian": tuple(float(value) for value in k_hat),
                "r_max_for_axis_au": float(r_max_for_axis.to("au").magnitude),
                "max_poloidal_mach": float(max_poloidal_mach),
                "rotational_support_factor": float(rotational_support_factor),
                "weight_mode": str(weight_mode),
                "soft_delta": (
                    dict(soft_delta)
                    if isinstance(soft_delta, Mapping)
                    else float(soft_delta)
                ),
                "weight_m0": float(weight_m0),
                "weight_floor": float(weight_floor),
                "kinematic_smoothing_bins": tuple(
                    float(value) for value in kinematic_smoothing_bins
                ),
                "midplane_connectivity": "radial_weakest_link",
                "azimuthal_coherence": "phi_mean_running_minimum",
                "vertical_connectivity": "midplane_weakest_link",
                "rho_midplane_min_g_cm3": (
                    None
                    if rho_midplane_min is None
                    else float(rho_midplane_min.to("g/cm^3").magnitude)
                ),
                "midplane_density_softness_dex": float(
                    midplane_density_softness_dex
                ),
                "midplane_density_smoothing_bins": float(
                    midplane_density_smoothing_bins
                ),
            },
        )
        model.gas_register("disk_weight", w_field)

    disk_region = model.set_mask_from_array(mask, is_a_disk=True)
    disk_region.mask = Field(
        data=disk_region.mask.data,
        quantity=disk_region.mask.quantity,
        axis_order=disk_region.mask.axis_order,
        attrs={
            "source": "joos_disk",
            "disk_axis_cartesian": tuple(float(value) for value in k_hat),
        },
    )
    model.gas_register("disk_mask", disk_region.mask)
    return disk_region


def set_mask_from_geometry(
    model: "Model",
    r_min: Optional[Quantity] = None,
    r_max: Optional[Quantity] = None,
    theta_min: Optional[Quantity] = None,
    theta_max: Optional[Quantity] = None,
    honrmax: Optional[float] = None,
    is_a_disk: bool = False,
) -> "SubModel":
    from .core import SubModel
    
    mesh = model.mesh
    if mesh is None:
        raise ValueError("Model has no mesh")
        
    if mesh.coord_system != 'spherical':
        raise ValueError(
            f"set_mask_from_geometry only supports spherical coordinates, "
            f"got {mesh.coord_system}"
        )
    
    if is_a_disk and model.disk is not None:
        target_region = model.disk
    else:
        target_region = SubModel(model)
    
    r = mesh.centers('r')
    theta = mesh.centers('theta')
    phi = mesh.centers('phi')
    
    r_grid, theta_grid, phi_grid = spherical_grids(r, theta, phi)
    
    mask = np.ones_like(r_grid, dtype=bool)
    
    if r_min is not None:
        mask &= (r_grid >= r_min)
    if r_max is not None:
        mask &= (r_grid <= r_max)
    
    if honrmax is not None and model.disk is not None:
        R_cyl, z_cyl = cylindrical_from_spherical(r_grid, theta_grid)
        
        h0 = model.disk.parameters["aspectratio"]
        fl = model.disk.parameters["flaringindex"]
        r0 = model.disk.parameters["r0"]
        
        H = scale_height(R_cyl, h0, r0, fl)
        
        z_max = honrmax * H
        mask &= (np.abs(z_cyl) <= z_max)
    
    if theta_min is not None:
        mask &= (theta_grid >= theta_min)
    if theta_max is not None:
        mask &= (theta_grid <= theta_max)
    
    axis_order = mesh.axis_names()
    
    mask_quantity = Quantity(mask, 'dimensionless')
    mask_field = Field(
        data=mask_quantity,
        quantity='mask',
        axis_order=axis_order,
    )
    
    target_region.mask = mask_field
    
    if is_a_disk:
        target_region.is_disk_region = True
    
    logger.info(
        f"{target_region.__class__.__name__} mask set: {np.sum(mask)} / {mask.size} cells "
        f"({100*np.sum(mask)/mask.size:.1f}%)"
    )
    
    return target_region
