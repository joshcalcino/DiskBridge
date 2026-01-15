from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from diskbridge.model.core import Model


def compute_cell_volumes(model: "Model") -> np.ndarray:
    mesh = model.mesh
    if mesh.coord_system != 'spherical':
        raise ValueError(f"Only spherical meshes supported, got {mesh.coord_system}")

    r_edges = mesh.edges('r').to('cm')
    theta_edges = mesh.edges('theta').to('radian')
    phi_edges = mesh.edges('phi').to('radian')

    dr3 = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
    dcos_theta = np.cos(theta_edges[:-1].magnitude) - np.cos(theta_edges[1:].magnitude)
    dphi = np.diff(phi_edges.magnitude)

    volumes = (dr3[:, None, None] * dcos_theta[None, :, None] * dphi[None, None, :]).magnitude
    return volumes


def compute_volume_weighted_mean_radial_profile(
    model: "Model",
    field_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mesh = model.mesh
    r = mesh.centers('r').to('au').magnitude

    if field_name not in model.gas:
        raise KeyError(f"Field '{field_name}' not found in model.gas")

    field = model.gas[field_name]
    data = field.data
    if hasattr(data, 'magnitude'):
        data = data.magnitude

    if getattr(mesh, 'coord_system', None) != 'spherical':
        raise ValueError(
            "compute_volume_weighted_mean_radial_profile supports spherical meshes only"
        )

    axis_order = getattr(field, 'axis_order', None)
    if axis_order is None:
        raise ValueError(f"Field '{field_name}' is missing axis_order")
    if axis_order != mesh.axis_names():
        raise ValueError(
            f"Field '{field_name}' has axis_order={axis_order}; expected canonical {mesh.axis_names()}"
        )

    nr, _, _ = data.shape
    volumes_use = compute_cell_volumes(model)
    if volumes_use.shape != data.shape:
        raise ValueError(
            f"Field '{field_name}' data shape {data.shape} does not match volume shape {volumes_use.shape}"
        )

    profile = np.zeros(nr)
    for i_r in range(nr):
        vals = data[i_r, :, :].ravel()
        wts = volumes_use[i_r, :, :].ravel()
        wt_sum = float(np.sum(wts))
        if wt_sum <= 0.0:
            raise ValueError(f"Non-positive volume sum at radial index {i_r}")
        profile[i_r] = float(np.sum(vals * wts) / wt_sum)

    return r, profile


def find_r_split(
    r_au: np.ndarray,
    r_edges_au: np.ndarray,
    chi_profile: np.ndarray,
    T_profile: np.ndarray,
    tol_chi: float = 0.01,
    tol_T: float = 0.01,
    window_fraction: float = 0.1,
    r_clip_min_au: float = 1.0,
) -> Tuple[float, dict]:
    nr = int(len(r_au))
    if nr == 0:
        raise ValueError("r_au is empty")

    if chi_profile.shape != (nr,) or T_profile.shape != (nr,):
        raise ValueError("chi_profile and T_profile must match r_au shape")

    r_max = float(r_au[-1])
    r_min = float(r_au[0])
    window_r_min = r_max - float(window_fraction) * (r_max - r_min)
    window_mask = r_au >= window_r_min

    if not np.any(window_mask):
        raise ValueError(f"Asymptote window is empty (window_r_min={window_r_min:.2f} AU)")

    chi_asymptote = float(np.mean(chi_profile[window_mask]))
    T_asymptote = float(np.mean(T_profile[window_mask]))

    if chi_asymptote <= 0.0:
        raise ValueError(f"Invalid chi_asymptote={chi_asymptote}")
    if T_asymptote <= 0.0:
        raise ValueError(f"Invalid T_asymptote={T_asymptote}")

    chi_dev = np.abs(chi_profile - chi_asymptote) / chi_asymptote
    T_dev = np.abs(T_profile - T_asymptote) / T_asymptote

    within_tol = (chi_dev <= float(tol_chi)) & (T_dev <= float(tol_T))

    r_split_idx = None
    for i in range(nr - 2, -1, -1):
        if within_tol[i]:
            r_split_idx = i
        else:
            break

    if r_split_idx is None:
        for i in range(nr - 1, -1, -1):
            if not within_tol[i]:
                r_split_idx = i + 1
                break
        if r_split_idx is None or r_split_idx >= nr:
            raise ValueError("Could not find valid R_split: profiles never reach asymptote")

    r_split_au = float(r_edges_au[int(r_split_idx)])

    if r_split_au < float(r_clip_min_au):
        raise ValueError(
            f"R_split={r_split_au:.2f} AU < r_clip_min={float(r_clip_min_au):.2f} AU. "
            "The stellar radiation dominates too far out for segmented RT."
        )

    info = {
        "chi_asymptote": float(chi_asymptote),
        "T_asymptote": float(T_asymptote),
        "r_split_cell_idx": int(r_split_idx),
        "window_r_min": float(window_r_min),
        "chi_dev_at_split": float(chi_dev[int(r_split_idx)]) if int(r_split_idx) < nr else np.nan,
        "T_dev_at_split": float(T_dev[int(r_split_idx)]) if int(r_split_idx) < nr else np.nan,
    }

    return r_split_au, info

