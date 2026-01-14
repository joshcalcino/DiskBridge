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

