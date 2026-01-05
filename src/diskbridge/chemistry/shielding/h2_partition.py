from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from diskbridge._logging import logger
from diskbridge._units import Quantity
from diskbridge._constants import (
    H2P_K0_DISS,
    H2P_N_ITER,
    H2P_NSIDE,
    H2P_R_FORM,
)
from diskbridge.chemistry.shielding.healpix_columns import (
    CartesianHealpixRayTracer,
    SphericalHealpixRayTracer,
)
from diskbridge.chemistry.shielding.healpix_utils import integrate_rays

if TYPE_CHECKING:
    from diskbridge.model.mesh import Mesh


def _h2_self_shielding_db96(N_H2: np.ndarray, b5: float, alpha: float) -> np.ndarray:
    x = N_H2 / 5.0e14
    x = np.maximum(x, 0.0)

    term1 = (1.0 + x / b5) ** alpha
    term2 = np.exp(-5.0e-4 * np.sqrt(1.0 + x))
    f_shield = term1 * term2

    f_shield = np.clip(f_shield, 0.0, 1.0)
    return f_shield


def compute_h2_partition(
    rad,
    mesh: "Mesh",
    nH: Quantity,
    chi_dust: Quantity,
) -> Tuple[Quantity, Quantity]:
    nside = H2P_NSIDE
    n_iter = H2P_N_ITER
    R_form = H2P_R_FORM
    k0_diss = H2P_K0_DISS

    # Draine & Bertoldi 1996 fit parameters
    b5 = 2.0
    alpha = -0.75

    if mesh.coord_system == "spherical":
        tracer = SphericalHealpixRayTracer(mesh, nside=nside)
    elif mesh.coord_system == "cartesian":
        tracer = CartesianHealpixRayTracer(mesh, nside=nside)
    else:
        raise ValueError(
            f"Unsupported mesh coord_system {mesh.coord_system!r} for H2 partition "
            "(expected 'spherical' or 'cartesian')."
        )

    dirs = tracer.dirs.astype(np.float64)
    npix = dirs.shape[0]

    nH_cgs = nH.to("cm^-3").magnitude
    chi_arr = chi_dust.to("dimensionless").magnitude

    if nH_cgs.shape != chi_arr.shape:
        raise ValueError(f"nH and chi_dust must have same shape, got {nH_cgs.shape} vs {chi_arr.shape}")

    shape = nH_cgs.shape
    idx_all = np.argwhere(np.ones(shape, dtype=bool))
    n_cells = idx_all.shape[0]

    cell_centers = np.zeros((n_cells, 3), dtype=np.float64)
    for k, idx in enumerate(idx_all):
        cell_centers[k] = tracer.cell_center_xyz(*idx)

    nH_flat = nH_cgs.reshape(-1)
    chi_flat = chi_arr.reshape(-1)

    denom0 = 2.0 * R_form * nH_flat + k0_diss * chi_flat
    fH2 = np.where(denom0 > 0.0, (2.0 * R_form * nH_flat) / denom0, 0.0)
    fH2 = np.clip(fH2, 0.0, 1.0)

    reduce_fn = np.mean

    logger.info(
        f"Computing H2 partition with HEALPix columns: nside={nside}, npix={npix}, n_iter={n_iter}"
    )

    for it in range(n_iter):
        nH2_cgs = 0.5 * fH2.reshape(shape) * nH_cgs
        f_sh_eff = np.zeros(n_cells, dtype=np.float64)

        # Integrate rays in slabs along the first axis to limit peak allocations.
        axis0 = idx_all[:, 0]
        n0 = int(shape[0])
        for i0 in range(n0):
            mask = axis0 == i0
            if not np.any(mask):
                continue
            centers_chunk = cell_centers[mask]
            N_H2_rays = integrate_rays(
                tracer,
                centers_chunk,
                dirs,
                nH2_cgs,
            )
            f_sh_rays = _h2_self_shielding_db96(N_H2_rays, b5=b5, alpha=alpha)
            f_sh_eff[mask] = reduce_fn(f_sh_rays, axis=1)

        k_diss_eff = k0_diss * chi_flat * f_sh_eff
        denom = 2.0 * R_form * nH_flat + k_diss_eff
        fH2_new = np.where(denom > 0.0, (2.0 * R_form * nH_flat) / denom, 0.0)
        fH2_new = np.clip(fH2_new, 0.0, 1.0)

        fH2 = fH2_new

    nH2_out = Quantity((0.5 * fH2.reshape(shape) * nH_cgs), "cm^-3")
    nH_atom_out = Quantity(((1.0 - fH2.reshape(shape)) * nH_cgs), "cm^-3")

    return nH2_out, nH_atom_out
