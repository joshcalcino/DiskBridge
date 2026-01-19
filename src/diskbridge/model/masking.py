from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING, Tuple, Union

import numpy as np

from diskbridge._units import Quantity, units
from diskbridge._logging import logger
from diskbridge.model.field import Field
from diskbridge.model.coords import spherical_grids, cylindrical_from_spherical
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


def _compute_disk_orientation(
    model: "Model",
    rho: Quantity,
    dV: Quantity,
    r_grid: Quantity,
    theta_grid_mag: np.ndarray,
    phi_grid_mag: np.ndarray,
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

    sin_t = np.sin(theta_grid_mag)
    cos_t = np.cos(theta_grid_mag)
    cos_p = np.cos(phi_grid_mag)
    sin_p = np.sin(phi_grid_mag)

    er_x = sin_t * cos_p
    er_y = sin_t * sin_p
    er_z = cos_t

    et_x = cos_t * cos_p
    et_y = cos_t * sin_p
    et_z = -sin_t

    ep_x = -sin_p
    ep_y = cos_p
    ep_z = 0.0

    x = r_grid * er_x
    y = r_grid * er_y
    z = r_grid * er_z

    vx = vr * er_x + vtheta * et_x + vphi * ep_x
    vy = vr * er_y + vtheta * et_y + vphi * ep_y
    vz = vr * er_z + vtheta * et_z + vphi * ep_z

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
    r_grid: Quantity,
    theta_grid_mag: np.ndarray,
    phi_grid_mag: np.ndarray,
    vr: Quantity,
    vphi: Quantity,
    vtheta: Quantity,
    k_hat: np.ndarray,
) -> DiskFrameData:
    sin_t = np.sin(theta_grid_mag)
    cos_t = np.cos(theta_grid_mag)
    cos_p = np.cos(phi_grid_mag)
    sin_p = np.sin(phi_grid_mag)

    er_x = sin_t * cos_p
    er_y = sin_t * sin_p
    er_z = cos_t

    et_x = cos_t * cos_p
    et_y = cos_t * sin_p
    et_z = -sin_t

    ep_x = -sin_p
    ep_y = cos_p
    ep_z = 0.0

    x = r_grid * er_x
    y = r_grid * er_y
    z = r_grid * er_z

    vx = vr * er_x + vtheta * et_x + vphi * ep_x
    vy = vr * er_y + vtheta * et_y + vphi * ep_y
    vz = vr * er_z + vtheta * et_z + vphi * ep_z

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
        r_edges_native = r_edges_native[r_edges_native <= r_max_base]
        if len(r_edges_native) < 2:
            raise ValueError("r_max is too small; no radial bins remain")

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

    if fthres_vr is None:
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


def _compute_ring_criteria(
    rho: Quantity,
    dV_mag: np.ndarray,
    disk_frame: DiskFrameData,
    Pth: Quantity,
    r_grid: Quantity,
    r_edges: Quantity,
    theta_edges: np.ndarray,
    nR: int,
    n_theta_bins: int,
    rho_disk_min: Quantity,
    fthres_use: Union[float, np.ndarray],
    fthres_vr_use: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_bin = np.digitize(r_grid.to_base_units().magnitude, r_edges.magnitude) - 1
    t_bin = np.digitize(disk_frame.theta_from_midplane, theta_edges) - 1
    valid = (r_bin >= 0) & (r_bin < nR) & (t_bin >= 0) & (t_bin < n_theta_bins)

    ring_index = (r_bin * n_theta_bins + t_bin).astype(np.int64)
    ring_index_flat = ring_index[valid].ravel()
    dV_w = dV_mag[valid].ravel()
    if dV_w.size == 0:
        raise ValueError("No valid cells for ring binning (check mesh/edges)")

    nbins = nR * n_theta_bins
    sum_w = np.bincount(ring_index_flat, weights=dV_w, minlength=nbins)

    def ring_avg(q_mag_flat: np.ndarray) -> np.ndarray:
        num = np.bincount(ring_index_flat, weights=(q_mag_flat * dV_w), minlength=nbins)
        out = np.zeros(nbins, dtype=float)
        ok_w = sum_w > 0.0
        out[ok_w] = num[ok_w] / sum_w[ok_w]
        return out.reshape(nR, n_theta_bins)

    vphi_avg = ring_avg(np.abs(disk_frame.vphi_d.to_base_units().magnitude)[valid].ravel())
    vR_avg = ring_avg(np.abs(disk_frame.vR_d.to_base_units().magnitude)[valid].ravel())
    vz_avg = ring_avg(np.abs(disk_frame.vz_d.to_base_units().magnitude)[valid].ravel())
    rho_avg_base = ring_avg(rho.to_base_units().magnitude[valid].ravel())

    rot = (0.5 * rho * (disk_frame.vphi_d.to_base_units() ** 2)).to_base_units()
    rot_avg = ring_avg(rot.magnitude[valid].ravel())
    P_avg = ring_avg(Pth.to_base_units().magnitude[valid].ravel())

    c1 = vphi_avg > (fthres_vr_use * vR_avg)
    c2 = vphi_avg > (fthres_use * vz_avg)
    c3 = rot_avg > (fthres_use * P_avg)
    c5 = rho_avg_base > rho_disk_min.to_base_units().magnitude
    ring_pass = c1 & c2 & c3 & c5

    return ring_pass, sum_w.reshape(nR, n_theta_bins), theta_edges


def _apply_connectivity(
    ring_pass: np.ndarray,
    sum_w_2d: np.ndarray,
    theta_edges: np.ndarray,
    nR: int,
    n_theta_bins: int,
) -> np.ndarray:
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    connected = np.zeros_like(ring_pass, dtype=bool)
    for i in range(nR):
        populated_idx = np.flatnonzero(sum_w_2d[i] > 0.0)
        if populated_idx.size == 0:
            continue

        seed_candidates = populated_idx[ring_pass[i, populated_idx]]
        if seed_candidates.size == 0:
            continue

        mid_t = int(seed_candidates[np.argmin(np.abs(theta_centers[seed_candidates]))])

        pop_order = populated_idx[np.argsort(theta_centers[populated_idx])]
        k0 = int(np.flatnonzero(pop_order == mid_t)[0])

        k = k0
        while k < pop_order.size and ring_pass[i, pop_order[k]]:
            connected[i, pop_order[k]] = True
            k += 1

        k = k0 - 1
        while k >= 0 and ring_pass[i, pop_order[k]]:
            connected[i, pop_order[k]] = True
            k -= 1

    return connected


def set_mask_from_joos_disk(
    model: "Model",
    rho_disk_min: Quantity,
    *,
    fthres: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 2.0,
    fthres_vr: Optional[Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]] = None,
    rho_core_min: Optional[Quantity] = None,
    r_max_for_axis: Optional[Quantity] = None,
    n_r_bins: Optional[int] = None,
    n_theta_bins: Optional[int] = None,
    r_max: Optional[Quantity] = None,
) -> "SubModel":
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
    dV_mag = dV.to_base_units().magnitude

    rho = model.gas["density"].data.to_base_units()
    vr = model.gas["vr"].data.to_base_units()
    vphi = model.gas["vphi"].data.to_base_units()
    vtheta = model.gas["vtheta"].data.to_base_units()

    r_grid_mag, theta_grid_mag, phi_grid_mag = np.meshgrid(
        r_c.to_base_units().magnitude,
        theta_c.to("radian").magnitude,
        phi_c.to("radian").magnitude,
        indexing="ij",
    )
    r_grid = r_grid_mag * r_c.to_base_units().units

    if rho_core_min is None:
        rho_core_min = 10.0 * rho_disk_min

    k_hat = _compute_disk_orientation(
        model, rho, dV, r_grid, theta_grid_mag, phi_grid_mag,
        vr, vphi, vtheta, rho_core_min, rho_disk_min, r_max_for_axis
    )

    disk_frame = _transform_to_disk_frame(
        r_grid, theta_grid_mag, phi_grid_mag, vr, vphi, vtheta, k_hat
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
        fthres, fthres_vr, r_edges, nR
    )

    ring_pass, sum_w_2d, theta_edges = _compute_ring_criteria(
        rho, dV_mag, disk_frame, Pth, r_grid, r_edges, theta_edges,
        nR, n_theta_bins, rho_disk_min, fthres_use, fthres_vr_use
    )

    connected = _apply_connectivity(ring_pass, sum_w_2d, theta_edges, nR, n_theta_bins)

    r_bin = np.digitize(r_grid.to_base_units().magnitude, r_edges.magnitude) - 1
    t_bin = np.digitize(disk_frame.theta_from_midplane, theta_edges) - 1
    valid = (r_bin >= 0) & (r_bin < nR) & (t_bin >= 0) & (t_bin < n_theta_bins)
    ring_index = (r_bin * n_theta_bins + t_bin).astype(np.int64)
    ring_index_flat = ring_index[valid].ravel()

    connected_flat = connected.reshape(-1)
    mask = np.zeros_like(r_grid_mag, dtype=bool)
    mask_valid = connected_flat[ring_index_flat]
    mask[valid] = mask_valid

    if r_max is not None:
        mask &= (r_grid <= r_max)

    return model.set_mask_from_array(mask, is_a_disk=True)


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
