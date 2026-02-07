from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import diskbridge
from diskbridge._logging import logger
from diskbridge._units import Quantity
from diskbridge._constants import T_FRZ, EPS_FRZ, CD_THRESHOLD_PDES, CD_THRESHOLD_PDISS
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.models.pinte_switches import compute_freezeout_factor
from diskbridge.chemistry.shielding.healpix_columns import (
    SphericalHealpixRayTracer,
    CartesianHealpixRayTracer,
)
from diskbridge.chemistry.shielding.healpix_utils import integrate_rays_multi


def run_abundance_switches(rad: "RadModel", config: dict) -> ChemistryResult:
    molecule = config.get("molecule", "co")
    X0 = config.get("X0", float(diskbridge.params.abundance))
    photodissociation = config.get(
        "photodissociation", diskbridge.params.photodissociation
    )
    freezeout = config.get("freezeout", diskbridge.params.freezeout)
    photodesorption = config.get("photodesorption", diskbridge.params.photodesorption)

    T = rad.ensure_temperature()
    nH = rad.ensure_nH()

    T_K = T.to("K").magnitude
    nH_cm3 = nH.to("cm^-3").magnitude

    mesh = rad.model.mesh

    X, n_mol = compute_abundance_layered_column(
        T_K=T_K,
        nH_cm3=nH_cm3,
        mesh=mesh,
        X0=float(X0),
        photodissociation=bool(photodissociation),
        freezeout=bool(freezeout),
        photodesorption=bool(photodesorption),
    )

    mol_lower = str(molecule).lower()

    Xq = Quantity(X, "dimensionless")
    nq = Quantity(n_mol, "cm^-3")

    return ChemistryResult(
        abundances={mol_lower: Xq},
        number_densities={mol_lower: nq},
        fields={},
        meta={"model": "abundance_switches"},
    )


def compute_vertical_cd_cm2(mesh, nH_cm3: np.ndarray) -> np.ndarray:
    if mesh.coord_system == "spherical":
        tracer = SphericalHealpixRayTracer(mesh, nside=1)
    elif mesh.coord_system == "cartesian":
        tracer = CartesianHealpixRayTracer(mesh, nside=1)
    else:
        raise ValueError(
            "vertical column requires spherical or cartesian mesh, "
            f"got {mesh.coord_system!r}"
        )

    shape = nH_cm3.shape
    if len(shape) != 3:
        raise ValueError(f"nH_cm3 must be 3D, got shape={shape}")

    nx, ny, nz = shape

    cell_centers = np.zeros((nx * ny * nz, 3), dtype=np.float64)
    flat = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cell_centers[flat, :] = tracer.cell_center_xyz(ix, iy, iz)
                flat += 1

    dirs = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)

    fields_stack = np.asarray(nH_cm3[None, ...], dtype=np.float64)

    N_all = integrate_rays_multi(
        tracer,
        cell_centers,
        dirs,
        fields_stack,
        self_weight=1.0,
    )

    CD = N_all[:, 0, 0].reshape(shape)
    return CD


def compute_abundance_layered_column(
    *,
    T_K: np.ndarray,
    nH_cm3: np.ndarray,
    mesh,
    X0: float,
    photodissociation: bool,
    freezeout: bool,
    photodesorption: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if T_K.shape != nH_cm3.shape:
        raise ValueError("T_K and nH_cm3 must have the same shape")

    T_vals = T_K

    X = np.full_like(T_vals, float(X0), dtype=float)

    if freezeout:
        freeze_factor, mask_frz = compute_freezeout_factor(
            T_vals,
            float(T_FRZ),
            float(EPS_FRZ),
            0.0,
        )
        X *= freeze_factor
        n_frz = int(np.sum(mask_frz))
        logger.info(f"Freeze-out: {n_frz} cells ({100*n_frz/X.size:.1f}%)")
    else:
        mask_frz = np.zeros_like(T_vals, dtype=bool)

    needs_cd = bool(photodissociation or (photodesorption and freezeout))
    if needs_cd:
        CD_cm2 = compute_vertical_cd_cm2(mesh, nH_cm3)

    mask_pdes = np.zeros_like(T_vals, dtype=bool)
    if photodesorption and freezeout:
        mask_pdes = mask_frz & (CD_cm2 < float(CD_THRESHOLD_PDES))
        X[mask_pdes] = float(X0)
        n_pdes = int(np.sum(mask_pdes))
        logger.info(f"Photodesorption: {n_pdes} cells ({100*n_pdes/X.size:.1f}%)")

    if photodissociation:
        mask_pdiss = CD_cm2 < float(CD_THRESHOLD_PDISS)
        X[mask_pdiss] = 0.0
        n_pdiss = int(np.sum(mask_pdiss))
        logger.info(
            f"Photodissociation: {n_pdiss} cells ({100*n_pdiss/X.size:.1f}%)"
        )

    n_mol = X * nH_cm3
    X_mean = float(np.mean(X))
    n_mean = float(np.mean(n_mol))
    logger.info(f"Computed abundance: X_mean={X_mean:.2e}, n_mean={n_mean:.2e}")

    return X, n_mol
