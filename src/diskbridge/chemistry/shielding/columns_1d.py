from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from diskbridge._units import Quantity
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding


def _to_ndarray_cgs(x, unit: str) -> np.ndarray:
    if isinstance(x, Quantity):
        return np.asarray(x.to(unit).magnitude, dtype=np.float64)
    if hasattr(x, "to"):
        return np.asarray(x.to(unit).magnitude, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def effective_1d_axis(mesh, shape: tuple[int, ...]) -> Tuple[str, int]:
    active_axes = [a for a in mesh.axis_names() if mesh.ncell(a) is not None]
    if len(active_axes) != len(shape):
        raise ValueError(
            f"mesh active axes do not match array shape: axes={active_axes}, shape={shape}"
        )

    non_singleton = [i for i, n in enumerate(shape) if int(n) > 1]
    if len(non_singleton) != 1:
        raise ValueError(
            f"mesh is not effectively 1D: shape={shape} (non-singleton axes={non_singleton})"
        )

    axis_index = int(non_singleton[0])
    axis_name = str(active_axes[axis_index])
    return axis_name, axis_index


def is_effectively_1d(mesh, shape: tuple[int, ...]) -> bool:
    active_axes = [a for a in mesh.axis_names() if mesh.ncell(a) is not None]
    if len(active_axes) != len(shape):
        return False
    n_non_singleton = sum(int(n) > 1 for n in shape)
    return n_non_singleton == 1


def column_to_outer_boundary_1d(
    mesh,
    n_field: np.ndarray,
    *,
    axis_name: str,
    axis_index: int,
    outer: str = "max",
) -> np.ndarray:
    n_field = np.asarray(n_field, dtype=np.float64)
    if np.shape(n_field) != tuple(mesh.shape):
        raise ValueError(
            f"n_field must match mesh.shape, got {np.shape(n_field)} vs {tuple(mesh.shape)}"
        )

    edges_q = mesh.edges(axis_name)
    if edges_q is None:
        raise ValueError(f"mesh axis {axis_name!r} has no edges")
    e_cm = np.asarray(edges_q.to("cm").magnitude, dtype=np.float64)
    dx = np.diff(e_cm)

    if n_field.ndim == 1:
        line = n_field
    else:
        idx = [0] * n_field.ndim
        idx[axis_index] = slice(None)
        line = n_field[tuple(idx)]

    if line.size != dx.size:
        raise ValueError(
            f"axis length mismatch: line.size={line.size}, dx.size={dx.size}"
        )

    contrib = line * dx

    if outer == "max":
        cum = np.cumsum(contrib[::-1])[::-1]
        N_line = cum - 0.5 * contrib
    elif outer == "min":
        cum = np.cumsum(contrib)
        N_line = cum - 0.5 * contrib
    else:
        raise ValueError(f"outer must be 'max' or 'min', got {outer!r}")

    if n_field.ndim == 1:
        return np.ascontiguousarray(N_line, dtype=np.float64)

    reshape = [1] * n_field.ndim
    reshape[axis_index] = int(N_line.size)
    N = np.broadcast_to(N_line.reshape(reshape), n_field.shape)
    return np.ascontiguousarray(N, dtype=np.float64)


def compute_pdr_shielding_1d(
    mesh,
    nH,
    chi,
    *,
    visser: Optional[VisserShielding] = None,
    nCO=None,
    nC=None,
    nH2=None,
    b_kms: Optional[float] = None,
    outer: str = "max",
    return_quantity: bool = True,
):
    nH_cgs = _to_ndarray_cgs(nH, "cm^-3")
    chi_arr = _to_ndarray_cgs(chi, "dimensionless")

    if nH2 is None:
        raise ValueError("compute_pdr_shielding_1d requires nH2")
    nH2_cgs = _to_ndarray_cgs(nH2, "cm^-3")

    if nC is None:
        raise ValueError("compute_pdr_shielding_1d requires nC")
    nC_cgs = _to_ndarray_cgs(nC, "cm^-3")

    if visser is not None and nCO is None:
        raise ValueError("compute_pdr_shielding_1d requires nCO when visser is provided")
    nCO_cgs = None if nCO is None else _to_ndarray_cgs(nCO, "cm^-3")

    if nH2_cgs.shape != nH_cgs.shape or nC_cgs.shape != nH_cgs.shape:
        raise ValueError("nC and nH2 must match nH shape")
    if nCO_cgs is not None and nCO_cgs.shape != nH_cgs.shape:
        raise ValueError("nCO must match nH shape")

    shape = tuple(nH_cgs.shape)
    axis_name, axis_index = effective_1d_axis(mesh, shape)

    N_H2 = column_to_outer_boundary_1d(
        mesh, nH2_cgs, axis_name=axis_name, axis_index=axis_index, outer=outer
    )
    N_C = column_to_outer_boundary_1d(
        mesh, nC_cgs, axis_name=axis_name, axis_index=axis_index, outer=outer
    )

    from diskbridge.chemistry.shielding.h2_db96 import h2_self_shielding_db96

    if b_kms is None:
        raise ValueError("b_kms is required for H2 self-shielding")

    theta_h2 = h2_self_shielding_db96(N_H2, b5=float(b_kms))

    theta_co = np.ones_like(nH_cgs, dtype=np.float64)
    if visser is not None:
        N_CO = column_to_outer_boundary_1d(
            mesh, nCO_cgs, axis_name=axis_name, axis_index=axis_index, outer=outer
        )
        theta_co = visser.theta("co", N_CO, N_H2, b_kms=float(b_kms))

    AH2 = 1.17e-8
    tau_H2 = 1.2e-14 * 2.0 * N_H2
    y = AH2 * tau_H2
    ry = np.exp(-y) / (1.0 + y)
    rc = np.exp(-1.6e-17 * N_C)
    theta_c = rc * ry

    theta_pdr = theta_h2 * theta_co
    chi_eff_pdr = chi_arr * theta_pdr

    if return_quantity:
        return (
            Quantity(theta_h2, "dimensionless"),
            Quantity(theta_co, "dimensionless"),
            Quantity(theta_c, "dimensionless"),
            Quantity(theta_pdr, "dimensionless"),
            Quantity(chi_eff_pdr, "dimensionless"),
        )

    return theta_h2, theta_co, theta_c, theta_pdr, chi_eff_pdr
