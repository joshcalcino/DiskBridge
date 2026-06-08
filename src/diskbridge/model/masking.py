# db-keywords: disk-mask, line-transfer, units, model, mesh, field
# db-role: helper
# db-scope: package
# db-purpose: Package module for disk-mask, line-transfer, units, model.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Dict, Optional, TYPE_CHECKING, Tuple, Union

import numpy as np

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


@dataclass(frozen=True)
class JoosCriterionData:
    """Dimensionless Joos criterion margins evaluated on the native grid."""
    valid: np.ndarray
    q_vr: np.ndarray
    q_vz: np.ndarray
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
    rho_disk_min: Quantity,
    r_max_for_axis: Optional[Quantity],
) -> np.ndarray:
    core_mask = rho >= rho_core_min
    if r_max_for_axis is not None:
        core_mask &= (r_grid <= r_max_for_axis)

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
        core_mask = rho >= rho_disk_min
        w = (rho * dV) * core_mask

    Lx = np.sum(w * (y * vz - z * vy))
    Ly = np.sum(w * (z * vx - x * vz))
    Lz = np.sum(w * (x * vy - y * vx))
    Lnorm = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    Lnorm_mag = float(Lnorm.magnitude)
    if Lnorm_mag == 0.0 or not np.isfinite(Lnorm_mag):
        k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        k_hat = np.array([
            (Lx / Lnorm).to("dimensionless").magnitude,
            (Ly / Lnorm).to("dimensionless").magnitude,
            (Lz / Lnorm).to("dimensionless").magnitude
        ])
    return k_hat


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
    r_max: Optional[Quantity],
    n_r_bins: Optional[int],
    n_theta_bins: Optional[int],
    n_bins_native: int,
    disk_frame: DiskFrameData,
    r_grid: Quantity,
    theta_sel: np.ndarray,
) -> Tuple[Quantity, np.ndarray, int]:
    r_edges_native = mesh.edges("r")
    if r_edges_native is None:
        raise ValueError("Mesh is missing radial edges")
    r_edges_native = r_edges_native.to_base_units()

    if r_max is not None:
        r_max_base = r_max.to_base_units()
        r_centers_native = mesh.centers("r")
        if r_centers_native is None:
            raise ValueError("Mesh is missing radial centers")
        keep = np.where(r_centers_native.to_base_units() <= r_max_base)[0]
        if keep.size == 0:
            raise ValueError("r_max is too small; no radial bins remain")
        r_edges_native = r_edges_native[: int(keep[-1]) + 2]

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
    theta_extent_sel = theta_sel & ok
    if np.any(theta_extent_sel):
        theta_max = float(np.max(np.abs(disk_frame.theta_from_midplane[theta_extent_sel])))
    else:
        theta_max = float(np.max(np.abs(disk_frame.theta_from_midplane[ok])))
    if theta_max == 0.0 or not np.isfinite(theta_max):
        raise ValueError("Invalid disk-frame theta extent; cannot build theta bins")

    theta_edges = np.linspace(-theta_max, theta_max, n_theta_bins + 1)
    return r_edges, theta_edges, n_theta_bins


def _evaluate_threshold_params(
    fthres: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]],
    fthres_vr: Optional[Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]],
    fthres_vr_inner: Optional[float],
    r_edges: Quantity,
    nR: int,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    if callable(fthres):
        r_mid = 0.5 * (r_edges[:-1].to("au").magnitude + r_edges[1:].to("au").magnitude)
        fthres_eval = np.asarray(fthres(r_mid), dtype=float)
        if fthres_eval.ndim != 1:
            raise ValueError("fthres callable must return a 1D array of thresholds")
        if fthres_eval.size != nR:
            raise ValueError(
                f"fthres callable returned {fthres_eval.size} values, expected nR={nR}"
            )
        fthres_use: Union[float, np.ndarray] = fthres_eval[:, None]
    elif np.isscalar(fthres):
        fthres_use = float(fthres)
    else:
        fthres_arr = np.asarray(fthres, dtype=float)
        if fthres_arr.ndim != 1:
            raise ValueError("fthres must be a scalar or a 1D array of length nR")
        if fthres_arr.size != nR:
            raise ValueError(f"fthres array length {fthres_arr.size} does not match nR={nR}")
        fthres_use = fthres_arr[:, None]

    if fthres_vr is not None and fthres_vr_inner is not None:
        raise ValueError("Specify either fthres_vr or fthres_vr_inner, not both")

    if fthres_vr_inner is not None:
        r_mid = 0.5 * (r_edges[:-1].to("au").magnitude + r_edges[1:].to("au").magnitude)
        r_min = float(np.min(r_mid[r_mid > 0.0]))
        r_max = float(r_edges[-1].to("au").magnitude)
        if r_max <= r_min:
            x = np.zeros_like(r_mid)
        else:
            x = np.log(np.clip(r_mid, r_min, r_max) / r_min) / np.log(r_max / r_min)

        if np.isscalar(fthres_use):
            fthres_target = np.full_like(r_mid, float(fthres_use), dtype=float)
        else:
            fthres_target = np.asarray(fthres_use, dtype=float).reshape(-1)

        fthres_vr_use = (
            float(fthres_vr_inner) + (fthres_target - float(fthres_vr_inner)) * x
        )[:, None]
    elif fthres_vr is None:
        fthres_vr_use: Union[float, np.ndarray] = fthres_use
    elif callable(fthres_vr):
        r_mid = 0.5 * (r_edges[:-1].to("au").magnitude + r_edges[1:].to("au").magnitude)
        fthres_vr_eval = np.asarray(fthres_vr(r_mid), dtype=float)
        if fthres_vr_eval.ndim != 1:
            raise ValueError("fthres_vr callable must return a 1D array of thresholds")
        if fthres_vr_eval.size != nR:
            raise ValueError(
                f"fthres_vr callable returned {fthres_vr_eval.size} values, expected nR={nR}"
            )
        fthres_vr_use = fthres_vr_eval[:, None]
    elif np.isscalar(fthres_vr):
        fthres_vr_use = float(fthres_vr)
    else:
        fthres_vr_arr = np.asarray(fthres_vr, dtype=float)
        if fthres_vr_arr.ndim != 1:
            raise ValueError("fthres_vr must be a scalar or a 1D array of length nR")
        if fthres_vr_arr.size != nR:
            raise ValueError(
                f"fthres_vr array length {fthres_vr_arr.size} does not match nR={nR}"
            )
        fthres_vr_use = fthres_vr_arr[:, None]

    return fthres_use, fthres_vr_use


def _threshold_for_cells(
    threshold: Union[float, np.ndarray],
    r_bin: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    if np.isscalar(threshold):
        return np.full_like(r_bin, float(threshold), dtype=float)

    arr = np.asarray(threshold, dtype=float).reshape(-1)
    out = np.full_like(r_bin, np.nan, dtype=float)
    out[valid] = arr[r_bin[valid]]
    return out


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
    r_bin: np.ndarray,
    valid: np.ndarray,
    rho_disk_min: Quantity,
    fthres_use: Union[float, np.ndarray],
    fthres_vr_use: Union[float, np.ndarray],
) -> JoosCriterionData:
    """Evaluate Joos criterion margins on individual cells."""
    fthres_grid = _threshold_for_cells(fthres_use, r_bin, valid)
    fthres_vr_grid = _threshold_for_cells(fthres_vr_use, r_bin, valid)

    vphi_abs = np.abs(disk_frame.vphi_d.to_base_units().magnitude)
    vR_abs = np.abs(disk_frame.vR_d.to_base_units().magnitude)
    vz_abs = np.abs(disk_frame.vz_d.to_base_units().magnitude)
    rho_base = rho.to_base_units().magnitude
    rot = (0.5 * rho * (disk_frame.vphi_d.to_base_units() ** 2)).to_base_units().magnitude
    P = Pth.to_base_units().magnitude
    rho_min = float(rho_disk_min.to_base_units().magnitude)

    q_vr = _safe_ratio(vphi_abs, fthres_vr_grid * vR_abs)
    q_vz = _safe_ratio(vphi_abs, fthres_grid * vz_abs)
    q_rot = _safe_ratio(rot, fthres_grid * P)
    q_rho = _safe_ratio(rho_base, np.full_like(rho_base, rho_min, dtype=float))
    for q in (q_vr, q_vz, q_rot, q_rho):
        q[~valid] = 0.0

    hard_pass = (
        valid
        & (q_vr > 1.0)
        & (q_vz > 1.0)
        & (q_rot > 1.0)
        & (q_rho > 1.0)
    )
    return JoosCriterionData(
        valid=valid,
        q_vr=q_vr,
        q_vz=q_vz,
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
        score = np.where(score >= 1.0 - floor, 1.0, score)
    score[~np.isfinite(score)] = 0.0
    return np.clip(score, 0.0, 1.0)


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
    s_vr = _soft_cut_from_ratio(
        criteria.q_vr,
        _soft_delta_value(soft_delta, "vr"),
        weight_floor,
    )
    s_vz = _soft_cut_from_ratio(
        criteria.q_vz,
        _soft_delta_value(soft_delta, "vz"),
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
    w_local = np.minimum.reduce([s_vr, s_vz, s_rot, s_rho])
    w_local[~criteria.valid] = 0.0
    return np.clip(w_local, 0.0, 1.0)


def _apply_soft_midplane_connectivity(
    w_local: np.ndarray,
    disk_frame: DiskFrameData,
    valid: np.ndarray,
    *,
    seed_min: float,
    floor: float,
) -> np.ndarray:
    """Propagate soft disk weight from the true midplane by a weakest-link rule.

    The connected weight of a cell is the minimum local Joos score along the
    vertical path from the column midplane to that cell. Columns whose midplane
    cell is below ``seed_min`` are assigned zero weight.
    """
    if seed_min < 0.0 or seed_min > 1.0:
        raise ValueError("weight_m0/seed_min must lie in [0, 1]")
    w_conn = np.zeros_like(w_local, dtype=float)
    nr, _ntheta, nphi = w_local.shape
    for ir in range(nr):
        for iphi in range(nphi):
            populated = np.flatnonzero(valid[ir, :, iphi])
            if populated.size == 0:
                continue
            theta_col = disk_frame.theta_from_midplane[ir, populated, iphi]
            pop_order = populated[np.argsort(theta_col)]
            mid_pos = int(
                np.argmin(
                    np.abs(disk_frame.theta_from_midplane[ir, pop_order, iphi])
                )
            )
            mid = int(pop_order[mid_pos])
            mid_weight = float(w_local[ir, mid, iphi])
            if mid_weight < seed_min or mid_weight <= floor:
                continue

            running = mid_weight
            for k in range(mid_pos, pop_order.size):
                j = int(pop_order[k])
                running = min(running, float(w_local[ir, j, iphi]))
                if running <= floor:
                    break
                w_conn[ir, j, iphi] = running

            running = mid_weight
            for k in range(mid_pos - 1, -1, -1):
                j = int(pop_order[k])
                running = min(running, float(w_local[ir, j, iphi]))
                if running <= floor:
                    break
                w_conn[ir, j, iphi] = running

    return np.clip(w_conn, 0.0, 1.0)


def _build_disk_weight(
    hard_mask: np.ndarray,
    criteria: JoosCriterionData,
    disk_frame: DiskFrameData,
    *,
    weight_mode: str,
    soft_delta: Union[float, Mapping[str, float]],
    weight_m0: float,
    weight_floor: float,
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
        return _apply_soft_midplane_connectivity(
            w_local,
            disk_frame,
            criteria.valid,
            seed_min=float(weight_m0),
            floor=float(weight_floor),
        )
    raise ValueError(
        "weight_mode must be one of 'none', 'cell', 'soft', or 'soft_connected'"
    )


def _compute_cell_criteria_mask(
    rho: Quantity,
    disk_frame: DiskFrameData,
    Pth: Quantity,
    r_bin: np.ndarray,
    t_bin: np.ndarray,
    valid: np.ndarray,
    rho_disk_min: Quantity,
    fthres_use: Union[float, np.ndarray],
    fthres_vr_use: Union[float, np.ndarray],
) -> np.ndarray:
    """Evaluate Joos criteria per cell and keep midplane-connected theta columns."""
    del t_bin
    criteria = _compute_joos_criterion_data(
        rho,
        disk_frame,
        Pth,
        r_bin,
        valid,
        rho_disk_min,
        fthres_use,
        fthres_vr_use,
    )
    return _apply_hard_midplane_connectivity(criteria.hard_pass, disk_frame, valid)


# db-keywords: disk-mask
# db-role: canonical
def set_mask_from_joos_disk(
    model: "Model",
    rho_disk_min: Quantity,
    *,
    fthres: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 2.0,
    fthres_vr: Optional[Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]] = None,
    fthres_vr_inner: Optional[float] = None,
    rho_core_min: Optional[Quantity] = None,
    r_max_for_axis: Optional[Quantity] = None,
    n_r_bins: Optional[int] = None,
    n_theta_bins: Optional[int] = None,
    r_max: Optional[Quantity] = None,
    weight_mode: str = "cell",
    weight_delta_bins: float = 3.0,
    weight_m0: float = 0.25,
    weight_floor: float = 1e-4,
    soft_delta: Union[float, Mapping[str, float]] = 0.20,
) -> "SubModel":
    """Create a disk region mask using the Joos et al. (2012) kinematic/pressure criteria.

    This function:

    1) Estimates the disk angular-momentum axis and transforms velocities into a disk frame.
    2) Evaluates Joos-style criteria directly on individual cells:
       - vphi dominates over vR and vz (with thresholds fthres / fthres_vr)
       - rotational support dominates over thermal pressure
       - density exceeds rho_disk_min
    3) Enforces connectivity to the midplane.
    4) Registers the boolean mask as model.gas["disk_mask"].
    5) Returns a SubModel with the same boolean mask.

    By default, it also registers the same binary mask as model.gas["disk_weight"].
    Use ``weight_mode="soft"`` or ``"soft_connected"`` to register a continuous
    disk/ISM material split derived from Joos criterion margins.

    Args:
        model: DiskBridge Model with spherical mesh and gas fields.
        rho_disk_min: Minimum gas density to be considered disk-like.
        fthres: Threshold(s) for vphi vs vz and rotational vs thermal support. May be a
            scalar, a length-nR array, or a callable f(R_au)->array.
        fthres_vr: Threshold(s) for vphi vs vR. If None, defaults to fthres.
        fthres_vr_inner: Inner vphi-vs-vR threshold. If set, fthres_vr ramps
            logarithmically from this value at the inner radius to fthres at r_max.
        rho_core_min: Density used to define the core region when estimating the disk axis.
            If None, defaults to 10 * rho_disk_min.
        r_max_for_axis: If provided, restricts the axis-estimation core region to r <= this.
        n_r_bins: Optional downsampling of radial bins (cannot refine beyond native).
        n_theta_bins: Number of theta_from_midplane bins used for ring averages.
        r_max: Optional maximum radius included in the final boolean mask.
        weight_mode: "none" skips disk_weight, "cell" registers the binary cell-wise
            disk mask, "soft" registers the local soft Joos score, and
            "soft_connected" registers the soft score with weakest-link midplane
            connectivity.
        weight_delta_bins: Retained for API compatibility; ignored by the cell-wise mask.
        weight_m0: Minimum midplane seed weight for soft_connected mode.
        weight_floor: Scores at or below this value are truncated to zero.
        soft_delta: Shared or criterion-specific softness in log-ratio space. Mapping
            keys are "vr", "vz", "rot", "rho", or "default".

    Returns:
        SubModel representing the disk region. The returned SubModel.mask is boolean.

    Raises:
        ValueError: If mesh is not spherical, binning yields no valid cells, or weight
            parameters are invalid.
        KeyError: If required gas fields are missing.
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
        vr, vphi, vtheta, rho_core_min, rho_disk_min, r_max_for_axis
    )

    disk_frame = _transform_to_disk_frame(
        mesh, r_grid, theta_grid, phi_grid, vr, vphi, vtheta, k_hat
    )

    Pth = _compute_thermal_pressure(model, rho)

    theta_sel = np.ones_like(disk_frame.z_d, dtype=bool)
    if r_max is not None:
        theta_sel &= (r_grid <= r_max)

    r_edges, theta_edges, n_theta_bins = _setup_binning(
        mesh, r_max, n_r_bins, n_theta_bins, n_bins_native, disk_frame, r_grid, theta_sel
    )
    nR = len(r_edges) - 1

    fthres_use, fthres_vr_use = _evaluate_threshold_params(
        fthres, fthres_vr, fthres_vr_inner, r_edges, nR
    )

    r_bin = np.digitize(r_grid.to_base_units().magnitude, r_edges.magnitude) - 1
    t_bin = np.digitize(disk_frame.theta_from_midplane, theta_edges) - 1
    valid = (r_bin >= 0) & (r_bin < nR) & (t_bin >= 0) & (t_bin < n_theta_bins)

    criteria = _compute_joos_criterion_data(
        rho,
        disk_frame,
        Pth,
        r_bin,
        valid,
        rho_disk_min,
        fthres_use,
        fthres_vr_use,
    )

    mask = _apply_hard_midplane_connectivity(
        criteria.hard_pass,
        disk_frame,
        valid,
    )
    inside_rmax = None
    if r_max is not None:
        inside_rmax = r_grid <= r_max
        mask &= inside_rmax

    if str(weight_mode).lower() != "none":
        w_disk = _build_disk_weight(
            mask,
            criteria,
            disk_frame,
            weight_mode=weight_mode,
            soft_delta=soft_delta,
            weight_m0=weight_m0,
            weight_floor=weight_floor,
        )
        if inside_rmax is not None:
            w_disk = np.where(inside_rmax, w_disk, 0.0)
        w_field = Field(
            data=Quantity(np.clip(w_disk, 0.0, 1.0), "dimensionless"),
            quantity="mask",
            axis_order=mesh.axis_names(),
            attrs={
                "source": "joos_disk",
                "weight_mode": str(weight_mode),
                "soft_delta": (
                    dict(soft_delta)
                    if isinstance(soft_delta, Mapping)
                    else float(soft_delta)
                ),
                "weight_m0": float(weight_m0),
                "weight_floor": float(weight_floor),
            },
        )
        model.gas_register("disk_weight", w_field)

    disk_region = model.set_mask_from_array(mask, is_a_disk=True)
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
