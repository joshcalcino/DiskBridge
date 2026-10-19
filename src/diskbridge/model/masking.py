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
    max_poloidal_mach: float,
    rotational_support_factor: float,
) -> JoosCriterionData:
    """Evaluate the two local settled rotating-disc kinematic margins."""
    if not np.isfinite(max_poloidal_mach) or max_poloidal_mach <= 0.0:
        raise ValueError("max_poloidal_mach must be positive and finite")
    if not np.isfinite(rotational_support_factor) or rotational_support_factor <= 0.0:
        raise ValueError("rotational_support_factor must be positive and finite")
    vR_abs = np.abs(disk_frame.vR_d.to_base_units().magnitude)
    vz_abs = np.abs(disk_frame.vz_d.to_base_units().magnitude)
    rho_base = rho.to_base_units().magnitude
    rot = (0.5 * rho * (disk_frame.vphi_d.to_base_units() ** 2)).to_base_units().magnitude
    P = Pth.to_base_units().magnitude
    sound_speed = np.sqrt(_safe_ratio(P, rho_base))
    q_pol = _safe_ratio(max_poloidal_mach * sound_speed, np.hypot(vR_abs, vz_abs))
    q_rot = _safe_ratio(rot, rotational_support_factor * P)
    for q in (q_pol, q_rot):
        q[~valid] = 0.0
    return JoosCriterionData(valid=valid, q_pol=q_pol, q_rot=q_rot)


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


def _connectivity_gate(score: np.ndarray, seed_min: float) -> np.ndarray:
    """Return a C1 gate that saturates at the existing passing seed score."""
    if not np.isfinite(seed_min) or not 0.0 < seed_min <= 1.0:
        raise ValueError("connectivity seed_min must lie in (0, 1]")
    scaled = np.clip(np.asarray(score, dtype=float) / seed_min, 0.0, 1.0)
    return scaled**2 * (3.0 - 2.0 * scaled)


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
    w_local = np.minimum(s_pol, s_rot)
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
) -> np.ndarray:
    """Collect local midplane scores on the disc-frame cylindrical grid."""
    if disk_frame.R_d.shape != w_local.shape or valid.shape != w_local.shape:
        raise ValueError("disk-frame geometry, validity, and weight shapes must match")
    nr, nphi = len(r_edges) - 1, w_local.shape[2]
    edges = np.asarray(r_edges.to_base_units().magnitude, dtype=float)
    phi_bins = _disk_phi_bins(disk_frame, nphi)
    support = np.zeros((nr, nphi), dtype=float)
    populated_grid = np.zeros((nr, nphi), dtype=bool)
    for ir in range(w_local.shape[0]):
        for iphi in range(nphi):
            populated = np.flatnonzero(valid[ir, :, iphi])
            if populated.size == 0:
                continue
            mid = int(populated[np.argmin(np.abs(
                disk_frame.theta_from_midplane[ir, populated, iphi]
            ))])
            r_bin = int(np.digitize(disk_frame.R_d[ir, mid, iphi], edges) - 1)
            if not 0 <= r_bin < nr:
                continue
            p_bin = int(phi_bins[ir, mid, iphi])
            support[r_bin, p_bin] = max(support[r_bin, p_bin], float(w_local[ir, mid, iphi]))
            populated_grid[r_bin, p_bin] = True
    return _fill_periodic_midplane_holes(support, populated_grid)


def _apply_connected_midplane_coherence(
    values: np.ndarray,
    midplane_support: np.ndarray,
    disk_frame: DiskFrameData,
    valid: np.ndarray,
    r_edges: Quantity,
    *,
    seed_min: float,
    floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply radial weakest-link and annular coherence to midplane support.

    Raw midplane scores locate each azimuth's first disc-like seed. Their
    saturating gates are propagated outward with a running minimum, so a
    detached island cannot re-enter after a failed column. The phi-mean of
    that connected gate support is smoothed and mapped to
    a coherence cap with ``C**2 * (3 - 2*C)``. This preserves zero, half, and
    full support, including unity at full support. The cap is propagated
    monotonically outward, suppressing narrow streamers. Both continuous
    gates cap the supplied 3-D connectivity gates.
    """
    if not 0.0 <= floor < seed_min <= 1.0:
        raise ValueError(
            "connected coherence requires 0 <= floor < seed_min <= 1"
        )
    edges = np.asarray(r_edges.to_base_units().magnitude, dtype=float)
    support = np.asarray(midplane_support, dtype=float)
    if support.ndim != 2:
        raise ValueError("midplane_support must have shape (radius, phi)")
    if support.shape[0] != edges.size - 1:
        raise ValueError("midplane support and radial edges do not match")
    if values.shape != disk_frame.R_d.shape or values.shape != valid.shape:
        raise ValueError("values, disk-frame geometry, and validity must match")

    support = np.clip(support, 0.0, 1.0)
    gate_support = _connectivity_gate(support, seed_min)
    radial_support = np.zeros_like(support, dtype=float)
    for iphi in range(support.shape[1]):
        seeds = np.flatnonzero(support[:, iphi] >= seed_min)
        if seeds.size == 0:
            continue
        start = int(seeds[0])
        running = float(gate_support[start, iphi])
        for ir in range(start, support.shape[0]):
            running = min(running, float(gate_support[ir, iphi]))
            if running <= floor:
                break
            radial_support[ir, iphi] = running

    coverage = np.mean(radial_support, axis=1)
    coverage_smooth = gaussian_filter1d(coverage, sigma=2.0, mode="nearest")
    coverage_smooth = np.clip(coverage_smooth, 0.0, 1.0)
    coherence_score = coverage_smooth**2 * (3.0 - 2.0 * coverage_smooth)
    coherence_score[coherence_score <= floor] = 0.0
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

    Connectivity gates are accumulated on a cylindrical ``(R, theta_d, phi_d)``
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
    criteria: JoosCriterionData,
    disk_frame: DiskFrameData,
    *,
    soft_delta: Union[float, Mapping[str, float]],
    weight_m0: float,
    weight_floor: float,
    r_edges: Quantity,
    theta_edges: np.ndarray,
) -> np.ndarray:
    """Cap local kinematic weights by the accepted connected support gates."""
    w_local = _compute_local_soft_joos_weight(criteria, soft_delta, weight_floor)
    midplane_support = _cylindrical_midplane_support(
        w_local, disk_frame, criteria.valid, r_edges,
    )
    local_gate = _connectivity_gate(w_local, float(weight_m0))
    gate_midplane, _, _, _, _ = _apply_connected_midplane_coherence(
        local_gate, midplane_support, disk_frame, criteria.valid, r_edges,
        seed_min=float(weight_m0), floor=float(weight_floor),
    )
    gate_connected, _ = _apply_vertical_connectivity(
        gate_midplane, disk_frame, criteria.valid, r_edges, theta_edges,
        floor=float(weight_floor),
    )
    return np.minimum(w_local, gate_connected)


# db-keywords: disk-mask
# db-role: canonical
def set_mask_from_joos_disk(
    model: "Model",
    *,
    max_poloidal_mach: float = 1.0,
    rotational_support_factor: float = 2.0,
    rho_core_min: Quantity,
    r_max_for_axis: Quantity,
    n_r_bins: Optional[int] = None,
    n_theta_bins: Optional[int] = None,
    weight_m0: float = 0.50,
    weight_floor: float = 1e-4,
    soft_delta: Union[float, Mapping[str, float]] = 0.20,
    kinematic_smoothing_bins: Tuple[float, float, float] = (2.0, 1.0, 2.0),
) -> "SubModel":
    """Construct the smoothed Joos-style disk weight with saturating connectivity.

    Density-weighted smoothed velocities define the local poloidal-motion and
    rotational-support scores. Radial, vertical and annular support gates cap
    their minimum. The function registers ``disk_weight`` and its boolean
    selection ``disk_mask`` at weight 0.5.

    Parameters
    ----------
    model : Model
        Model with a spherical mesh and gas fields.
    max_poloidal_mach : float, optional
        Maximum poloidal Mach number, using ``hypot(v_R, v_z)/sqrt(P/rho)``.
    rotational_support_factor : float, optional
        Required rotational-to-thermal ratio: ``0.5*rho*v_phi**2 > factor*P``.
    rho_core_min : Quantity
        Gas density defining the independent angular-momentum axis estimator.
    r_max_for_axis : Quantity
        Finite axis-estimation aperture [length], not a material cutoff.
    n_r_bins, n_theta_bins : int or None, optional
        Number of cylindrical radial and disc-frame polar bins. Native sizes
        are used when omitted.
    weight_m0 : float, optional
        Raw midplane seed score and saturation point of the support gates.
    weight_floor : float, optional
        Numerical score/gate truncation floor.
    soft_delta : float or mapping, optional
        Local sigmoid widths in log-ratio space. Mapping keys are ``"mach"``,
        ``"rot"`` and ``"default"``.
    kinematic_smoothing_bins : tuple of float, optional
        Density-weighted Gaussian velocity smoothing sigma in native
        ``(r, theta, phi)`` cells.

    Returns
    -------
    SubModel
        Disk region registered on the model, with the boolean weight-0.5 mask.

    Raises
    ------
    ValueError
        If geometry, axis support, physical thresholds or weighting controls
        are invalid.
    KeyError
        If required gas fields or criterion softness values are missing.
    """
    mesh = model.mesh
    if mesh.coord_system != "spherical":
        raise ValueError(
            f"set_mask_from_joos_disk only supports spherical coordinates, got {mesh.coord_system}"
        )
    r_c, theta_c, phi_c = (mesh.centers(axis) for axis in ("r", "theta", "phi"))
    r_e, theta_e, phi_e = (mesh.edges(axis) for axis in ("r", "theta", "phi"))
    if any(value is None for value in (r_c, theta_c, phi_c, r_e, theta_e, phi_e)):
        raise ValueError("Mesh is missing spherical coordinate axes")
    r3 = (r_e[1:] ** 3 - r_e[:-1] ** 3) / 3.0
    dcos = np.cos(theta_e.to("radian").magnitude[:-1]) - np.cos(theta_e.to("radian").magnitude[1:])
    dphi = np.diff(phi_e.to("radian").magnitude)
    dV = r3[:, None, None] * dcos[None, :, None] * dphi[None, None, :]
    rho = model.gas["density"].data.to_base_units()
    vr = model.gas["vr"].data.to_base_units()
    vphi = model.gas["vphi"].data.to_base_units()
    vtheta = model.gas["vtheta"].data.to_base_units()
    r_grid, theta_grid, phi_grid = spherical_grids(
        r_c.to_base_units(), theta_c.to("radian"), phi_c.to("radian"),
    )
    k_hat = _compute_disk_orientation(
        model, rho, dV, r_grid, theta_grid, phi_grid,
        vr, vphi, vtheta, rho_core_min, r_max_for_axis,
    )
    disk_frame = _transform_to_disk_frame(
        mesh, r_grid, theta_grid, phi_grid, vr, vphi, vtheta, k_hat,
    )
    r_edges, theta_edges, ntheta = _setup_binning(
        mesh, n_r_bins, n_theta_bins, len(theta_c), disk_frame, r_grid,
    )
    r_bin = np.digitize(r_grid.to_base_units().magnitude, r_edges.magnitude) - 1
    t_bin = np.digitize(disk_frame.theta_from_midplane, theta_edges) - 1
    valid = (r_bin >= 0) & (r_bin < len(r_edges)-1) & (t_bin >= 0) & (t_bin < ntheta)
    weight_frame = _smooth_disk_frame_kinematics(rho, disk_frame, kinematic_smoothing_bins)
    criteria = _compute_joos_criterion_data(
        rho, weight_frame, _compute_thermal_pressure(model, rho), valid,
        float(max_poloidal_mach), float(rotational_support_factor),
    )
    w_disk = _build_disk_weight(
        criteria, weight_frame, soft_delta=soft_delta, weight_m0=weight_m0,
        weight_floor=weight_floor, r_edges=r_edges, theta_edges=theta_edges,
    )
    model.gas_register("disk_weight", Field(
        data=Quantity(np.clip(w_disk, 0.0, 1.0), "dimensionless"),
        quantity="mask", axis_order=mesh.axis_names(),
        attrs={
            "source": "joos_disk", "method": "saturating_connected",
            "disk_axis_cartesian": tuple(float(value) for value in k_hat),
            "rho_core_min_g_cm3": float(rho_core_min.to("g/cm^3").magnitude),
            "r_max_for_axis_au": float(r_max_for_axis.to("au").magnitude),
            "max_poloidal_mach": float(max_poloidal_mach),
            "rotational_support_factor": float(rotational_support_factor),
            "soft_delta": dict(soft_delta) if isinstance(soft_delta, Mapping) else float(soft_delta),
            "weight_m0": float(weight_m0), "weight_floor": float(weight_floor),
            "kinematic_smoothing_bins": tuple(float(value) for value in kinematic_smoothing_bins),
            "midplane_connectivity": "radial_weakest_link",
            "azimuthal_coherence": "phi_mean_running_minimum",
            "coherence_score": "smoothstep",
            "connectivity_gate": "smoothstep_saturating_at_weight_m0",
            "vertical_connectivity": "midplane_weakest_link",
        },
    ))
    disk_region = model.set_mask_from_array(w_disk >= 0.5, is_a_disk=True)
    disk_region.mask = Field(
        data=disk_region.mask.data, quantity=disk_region.mask.quantity,
        axis_order=disk_region.mask.axis_order,
        attrs={"source": "joos_disk", "disk_axis_cartesian": tuple(float(value) for value in k_hat)},
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
