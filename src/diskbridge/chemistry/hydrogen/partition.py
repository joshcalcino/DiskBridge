from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from diskbridge._logging import logger
from diskbridge._constants import (
    H2P_K0_DISS,
    H2P_N_ITER,
    H2P_NSIDE,
    H2P_R_FORM,
)
from diskbridge.chemistry.shielding.columns_1d import (
    column_to_outer_boundary_1d,
    effective_1d_axis,
    is_effectively_1d,
)
from diskbridge.chemistry.shielding.h2_db96 import h2_self_shielding_db96
from diskbridge.chemistry.shielding.healpix_columns import (
    _prepare_healpix_geometry,
    compute_column_rays_healpix,
)


if TYPE_CHECKING:
    from diskbridge.model.mesh import Mesh


def _as_f64(name: str, x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return np.ascontiguousarray(a, dtype=np.float64)


def compute_h2_partition(
    rad,
    mesh: "Mesh",
    nH_cm3: np.ndarray,
    chi_dust: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    nside = H2P_NSIDE
    n_iter = H2P_N_ITER
    R_form = H2P_R_FORM
    k0_diss = H2P_K0_DISS

    # Draine & Bertoldi 1996 fit parameters
    b5 = 2.0

    nH_cgs = _as_f64("nH_cm3", nH_cm3)
    chi_arr = _as_f64("chi_dust", chi_dust)

    if nH_cgs.shape != chi_arr.shape:
        raise ValueError(f"nH and chi_dust must have same shape, got {nH_cgs.shape} vs {chi_arr.shape}")

    shape = nH_cgs.shape
    nH_flat = nH_cgs.reshape(-1)
    chi_flat = chi_arr.reshape(-1)

    denom0 = 2.0 * R_form * nH_flat + k0_diss * chi_flat
    fH2 = np.where(denom0 > 0.0, (2.0 * R_form * nH_flat) / denom0, 0.0)
    fH2 = np.clip(fH2, 0.0, 1.0)

    if is_effectively_1d(mesh, shape):
        axis_name, axis_index = effective_1d_axis(mesh, shape)
        for _ in range(n_iter):
            nH2_cgs = 0.5 * fH2.reshape(shape) * nH_cgs
            N_H2 = column_to_outer_boundary_1d(
                mesh,
                nH2_cgs,
                axis_name=axis_name,
                axis_index=axis_index,
                outer="min",
            )
            f_sh_eff = h2_self_shielding_db96(N_H2, b5=b5).reshape(-1)

            k_diss_eff = k0_diss * chi_flat * f_sh_eff
            denom = 2.0 * R_form * nH_flat + k_diss_eff
            fH2_new = np.where(denom > 0.0, (2.0 * R_form * nH_flat) / denom, 0.0)
            fH2_new = np.clip(fH2_new, 0.0, 1.0)
            fH2 = fH2_new

        nH2_out = np.ascontiguousarray((0.5 * fH2.reshape(shape) * nH_cgs), dtype=np.float64)
        nH_atom_out = np.ascontiguousarray(((1.0 - fH2.reshape(shape)) * nH_cgs), dtype=np.float64)
        return nH2_out, nH_atom_out

    # HEALPix path
    candidate_mask_all = np.ones(shape, dtype=bool)
    tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
        mesh,
        nside=int(nside),
        candidate_mask=candidate_mask_all,
        cache_dir=None,
    )

    n_cells = candidate_idx.shape[0]
    reduce_fn = np.mean

    logger.info(
        f"Computing H2 partition with HEALPix columns: nside={nside}, npix={dirs.shape[0]}, n_iter={n_iter}"
    )

    for it in range(n_iter):
        nH2_cgs = 0.5 * fH2.reshape(shape) * nH_cgs
        f_sh_eff = np.zeros(n_cells, dtype=np.float64)

        # Integrate rays in slabs along the first axis to limit peak allocations.
        axis0 = candidate_idx[:, 0]
        n0 = int(shape[0])
        for i0 in range(n0):
            mask = axis0 == i0
            if not np.any(mask):
                continue
            
            _, _, cols = compute_column_rays_healpix(
                mesh,
                fields={"h2": nH2_cgs},
                nside=int(nside),
                candidate_mask=None,
                progress_chunks=None,
                cache_dir=None,
                self_weight=1.0,
                tracer=tracer,
                dirs=dirs,
                candidate_idx=candidate_idx[mask],
                cell_centers=cell_centers[mask],
            )
            N_H2_rays = cols["h2"]
            f_sh_rays = h2_self_shielding_db96(N_H2_rays, b5=b5)
            f_sh_eff[mask] = reduce_fn(f_sh_rays, axis=1)

        k_diss_eff = k0_diss * chi_flat * f_sh_eff
        denom = 2.0 * R_form * nH_flat + k_diss_eff
        fH2_new = np.where(denom > 0.0, (2.0 * R_form * nH_flat) / denom, 0.0)
        fH2_new = np.clip(fH2_new, 0.0, 1.0)
        fH2 = fH2_new

    nH2_out = np.ascontiguousarray((0.5 * fH2.reshape(shape) * nH_cgs), dtype=np.float64)
    nH_atom_out = np.ascontiguousarray(((1.0 - fH2.reshape(shape)) * nH_cgs), dtype=np.float64)

    return nH2_out, nH_atom_out
