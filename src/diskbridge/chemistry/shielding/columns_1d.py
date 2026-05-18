from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from diskbridge.chemistry.shielding.visser_shielding import N_SHIELD_MIN, VisserShielding


def _as_f64(name: str, x) -> np.ndarray:
    if hasattr(x, "magnitude") and hasattr(x, "units"):
        raise TypeError(f"{name} must be a float64 numpy array (no unit-carrying objects).")
    a = np.asarray(x, dtype=np.float64)
    if a.dtype == object:
        raise TypeError(f"{name} must be float64; got dtype=object")
    return np.ascontiguousarray(a, dtype=np.float64)


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
    n_field = _as_f64("n_field", n_field)
    if np.shape(n_field) != tuple(mesh.shape):
        raise ValueError(
            f"n_field must match mesh.shape, got {np.shape(n_field)} vs {tuple(mesh.shape)}"
        )

    e_cm = mesh.edges_f64(axis_name, "cm")
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
    elif outer == "both":
        cum_max = np.cumsum(contrib[::-1])[::-1]
        N_max = cum_max - 0.5 * contrib

        cum_min = np.cumsum(contrib)
        N_min = cum_min - 0.5 * contrib

        N_line = np.minimum(N_min, N_max)
    else:
        raise ValueError(f"outer must be 'max', 'min', or 'both', got {outer!r}")

    if n_field.ndim == 1:
        return np.ascontiguousarray(N_line, dtype=np.float64)

    reshape = [1] * n_field.ndim
    reshape[axis_index] = int(N_line.size)
    N = np.broadcast_to(N_line.reshape(reshape), n_field.shape)
    return np.ascontiguousarray(N, dtype=np.float64)


def compute_pdr_shielding_1d(
    mesh,
    nH: np.ndarray,
    chi: np.ndarray,
    *,
    visser: Optional[VisserShielding] = None,
    nCO: Optional[np.ndarray] = None,
    nC: np.ndarray,
    nH2: np.ndarray,
    b_kms: Optional[float] = None,
    b_CO_kms_grid: Optional[np.ndarray] = None,
    outer: str = "max",
):
    nH_cgs = _as_f64("nH", nH)
    chi_arr = _as_f64("chi", chi)

    nH2_cgs = _as_f64("nH2", nH2)
    nC_cgs = _as_f64("nC", nC)

    if visser is not None and nCO is None:
        raise ValueError("compute_pdr_shielding_1d requires nCO when visser is provided")
    nCO_cgs = None if nCO is None else _as_f64("nCO", nCO)
    b_CO_grid = None if b_CO_kms_grid is None else _as_f64("b_CO_kms_grid", b_CO_kms_grid)

    if nH2_cgs.shape != nH_cgs.shape or nC_cgs.shape != nH_cgs.shape:
        raise ValueError("nC and nH2 must match nH shape")
    if nCO_cgs is not None and nCO_cgs.shape != nH_cgs.shape:
        raise ValueError("nCO must match nH shape")
    if b_CO_grid is not None and b_CO_grid.shape != nH_cgs.shape:
        raise ValueError("b_CO_kms_grid must match nH shape")

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
        if b_CO_grid is not None:
            N_CO_b2 = column_to_outer_boundary_1d(
                mesh,
                nCO_cgs * b_CO_grid * b_CO_grid,
                axis_name=axis_name,
                axis_index=axis_index,
                outer=outer,
            )
            b_eff = np.sqrt(N_CO_b2 / np.maximum(N_CO, N_SHIELD_MIN))
            theta_co = visser.theta_interpolated_b("co", N_CO, N_H2, b_eff)
        else:
            theta_co = visser.theta("co", N_CO, N_H2, b_kms=float(b_kms))

    AH2 = 1.17e-8
    tau_H2 = 1.2e-14 * 2.0 * N_H2
    y = AH2 * tau_H2
    ry = np.exp(-y) / (1.0 + y)
    rc = np.exp(-1.6e-17 * N_C)
    theta_c = rc * ry

    theta_pdr = theta_h2 * theta_co
    chi_eff_pdr = chi_arr * theta_pdr
    return theta_h2, theta_co, theta_c, theta_pdr, chi_eff_pdr


def compute_co_shielding_1d(
    mesh,
    nH: np.ndarray,
    chi: np.ndarray,
    *,
    visser: VisserShielding,
    nCO: np.ndarray,
    nH2: np.ndarray,
    b_kms: float,
    b_CO_kms_grid: Optional[np.ndarray] = None,
    outer: str = "max",
):
    nH_cgs = _as_f64("nH", nH)
    chi_arr = _as_f64("chi", chi)
    nH2_cgs = _as_f64("nH2", nH2)
    nCO_cgs = _as_f64("nCO", nCO)
    b_CO_grid = None if b_CO_kms_grid is None else _as_f64("b_CO_kms_grid", b_CO_kms_grid)

    if nH2_cgs.shape != nH_cgs.shape or nCO_cgs.shape != nH_cgs.shape or chi_arr.shape != nH_cgs.shape:
        raise ValueError("nH, chi, nH2, and nCO must have the same shape")
    if b_CO_grid is not None and b_CO_grid.shape != nH_cgs.shape:
        raise ValueError("b_CO_kms_grid must match nH shape")

    shape = tuple(nH_cgs.shape)
    axis_name, axis_index = effective_1d_axis(mesh, shape)

    N_H2 = column_to_outer_boundary_1d(
        mesh, nH2_cgs, axis_name=axis_name, axis_index=axis_index, outer=outer
    )
    N_CO = column_to_outer_boundary_1d(
        mesh, nCO_cgs, axis_name=axis_name, axis_index=axis_index, outer=outer
    )

    if b_CO_grid is not None:
        N_CO_b2 = column_to_outer_boundary_1d(
            mesh,
            nCO_cgs * b_CO_grid * b_CO_grid,
            axis_name=axis_name,
            axis_index=axis_index,
            outer=outer,
        )
        b_eff = np.sqrt(N_CO_b2 / np.maximum(N_CO, N_SHIELD_MIN))
        theta_co = visser.theta_interpolated_b("co", N_CO, N_H2, b_eff)
    else:
        theta_co = visser.theta("co", N_CO, N_H2, b_kms=float(b_kms))
    chi_eff = chi_arr * theta_co
    return theta_co, chi_eff
