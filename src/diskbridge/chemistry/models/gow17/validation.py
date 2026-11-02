"""GOW17 abundance-budget projection and solver diagnostics."""

# db-keywords: gow17, chemistry, arrays
# db-role: canonical
# db-scope: package
# db-purpose: Validate and project GOW17 network states and summarize solver diagnostics.

from __future__ import annotations

import numpy as np
from numba import njit, prange


def gow17_budget_diagnostics(
    y: np.ndarray,
    xCtot,
    xOtot=None,
    rtol: float = 1e-2,
    logger=None,
) -> dict:
    """Warn-only budget diagnostics for the GOW17 chemistry model.

    Computes hydrogen and carbon budget-violation magnitudes from the
    solved abundance vector ``y`` (shape ``(..., N_Y)``) and logs
    warnings when the violation exceeds ``rtol``.

    The hydrogen budget violation is defined as::

        max(0, -min(xH_atom_raw), max(xH_accounted) - 1)

    where ``xH_atom_raw = 1 - xH_accounted`` is the residual atomic-H
    fraction.  A positive violation means either the residual went
    negative (over-accounting) or the accounted fraction exceeded unity.

    The carbon budget violation is defined analogously::

        max(0, -min(xC_neutral_raw), max(xC_accounted) - max(xCtot))

    Parameters
    ----------
    y : np.ndarray
        GOW17 abundance array with species along the last axis.
        Uses the same index layout as ``diskbridge.chemistry.models.gow17._gow17``.
    xCtot : float or np.ndarray
        Total carbon abundance per H nucleus (e.g. ``Zg * XC_STD``).
        May be a scalar (applied to all cells) or an array broadcastable
        to the cell shape ``y.shape[:-1]``.
    rtol : float
        Relative tolerance; a warning is issued when the violation
        exceeds this value.
    logger : logging.Logger, optional
        Logger instance. If *None*, warnings are printed to stdout.

    Returns
    -------
    dict
        Keys: ``h_xH_atom_min``, ``h_xH_accounted_max``,
        ``c_xC_neutral_min``, ``c_xC_accounted_max``,
        ``h_budget_violation``, ``c_budget_violation``.
    """
    import diskbridge.chemistry.models.gow17._gow17 as _g

    xH2 = y[..., _g.I_H2]
    xHp = y[..., _g.I_HP]
    xH2p = y[..., _g.I_H2P]
    xH3p = y[..., _g.I_H3P]
    xHCOp = y[..., _g.I_HCOP]
    xCHx = y[..., _g.I_CHX]
    xOHx = y[..., _g.I_OHX]
    xCO = y[..., _g.I_CO]
    xCplus = y[..., _g.I_CP]
    xCOice = y[..., _g.I_CO_ICE]

    xCtot = np.asarray(xCtot, dtype=np.float64)
    xOtot_arr = None if xOtot is None else np.asarray(xOtot, dtype=np.float64)

    xH_accounted = xOHx + xCHx + xHCOp + 3.0 * xH3p + 2.0 * xH2p + xHp + 2.0 * xH2
    xH_atom_raw = 1.0 - xH_accounted

    xC_accounted = xHCOp + xCHx + xCO + xCOice + xCplus
    xC_neutral_raw = xCtot - xC_accounted

    if xOtot_arr is not None:
        xOplus = y[..., _g.I_OP]
        xO_accounted = xOHx + xHCOp + xCO + xCOice + xOplus
        xO_neutral_raw = xOtot_arr - xO_accounted
        o_xO_neutral_min = float(np.min(xO_neutral_raw))
        o_xO_accounted_max = float(np.max(xO_accounted))
        o_budget_violation = float(max(
            0.0,
            -o_xO_neutral_min,
            float(np.max(xO_accounted - xOtot_arr)),
        ))
    else:
        o_xO_neutral_min = float("nan")
        o_xO_accounted_max = float("nan")
        o_budget_violation = float("nan")

    h_xH_atom_min = float(np.min(xH_atom_raw))
    h_xH_accounted_max = float(np.max(xH_accounted))
    c_xC_neutral_min = float(np.min(xC_neutral_raw))
    c_xC_accounted_max = float(np.max(xC_accounted))

    # Budget-violation magnitudes: positive means budget is broken.
    h_budget_violation = float(max(
        0.0,
        -h_xH_atom_min,
        h_xH_accounted_max - 1.0,
    ))
    c_budget_violation = float(max(
        0.0,
        -c_xC_neutral_min,
        float(np.max(xC_accounted - xCtot)),
    ))

    def _warn(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")

    if h_budget_violation > rtol:
        _warn(
            f"gow17 H budget violation = {h_budget_violation:.3e} > {rtol:.1e} "
            f"(min xH_atom={h_xH_atom_min:.3e}, max xH_accounted={h_xH_accounted_max:.3e})"
        )
    if c_budget_violation > rtol:
        _warn(
            f"gow17 C budget violation = {c_budget_violation:.3e} > {rtol:.1e} "
            f"(min xC_neutral={c_xC_neutral_min:.3e}, max xC_accounted={c_xC_accounted_max:.3e})"
        )
    if xOtot_arr is not None and o_budget_violation > rtol:
        _warn(
            f"gow17 O budget violation = {o_budget_violation:.3e} > {rtol:.1e} "
            f"(min xO_neutral={o_xO_neutral_min:.3e}, max xO_accounted={o_xO_accounted_max:.3e})"
        )

    return {
        "h_xH_atom_min": h_xH_atom_min,
        "h_xH_accounted_max": h_xH_accounted_max,
        "c_xC_neutral_min": c_xC_neutral_min,
        "c_xC_accounted_max": c_xC_accounted_max,
        "o_xO_neutral_min": o_xO_neutral_min,
        "o_xO_accounted_max": o_xO_accounted_max,
        "h_budget_violation": h_budget_violation,
        "c_budget_violation": c_budget_violation,
        "o_budget_violation": o_budget_violation,
    }


@njit(cache=True, parallel=True)
def _project_gow17_state_to_budgets_parallel(
    y: np.ndarray,
    xCtot: np.ndarray,
    xOtot: np.ndarray,
    carbon_species: np.ndarray,
    oxygen_species: np.ndarray,
    carbon_budget_scalar: bool,
    oxygen_budget_scalar: bool,
    project_oxygen: bool,
) -> np.ndarray:
    """Project flattened states independently across cells."""
    n_cells, n_species = y.shape
    y_proj = np.empty((n_cells, n_species), dtype=np.float64)
    for i in prange(n_cells):
        for j in range(n_species):
            value = y[i, j]
            y_proj[i, j] = 0.0 if value <= 0.0 else value

        c_sum = 0.0
        for j in range(carbon_species.size):
            c_sum += y_proj[i, carbon_species[j]]
        xC_cell = xCtot[0] if carbon_budget_scalar else xCtot[i]
        if c_sum > xC_cell and c_sum > 0.0:
            c_scale = xC_cell / c_sum
            for j in range(carbon_species.size):
                index = carbon_species[j]
                y_proj[i, index] *= c_scale

        if project_oxygen:
            xO_cell = xOtot[0] if oxygen_budget_scalar else xOtot[i]
            o_sum = 0.0
            for j in range(oxygen_species.size):
                o_sum += y_proj[i, oxygen_species[j]]
            if o_sum > xO_cell and o_sum > 0.0:
                o_scale = xO_cell / o_sum
                for j in range(oxygen_species.size):
                    index = oxygen_species[j]
                    y_proj[i, index] *= o_scale
    return y_proj


def project_gow17_state_to_budgets(
    y: np.ndarray,
    *,
    xCtot,
    xOtot=None,
) -> np.ndarray:
    """Project a GOW17 state onto non-negative C/O abundance budgets.

    The projection is intentionally conservative: it only rescales explicitly
    carbon-bearing and oxygen-bearing species when their summed abundance
    exceeds the supplied elemental budget. Hydrogen bookkeeping is left to the
    native network state.
    """
    import diskbridge.chemistry.models.gow17._gow17 as _g

    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 0 or y_arr.shape[-1] != int(_g.N_Y):
        raise ValueError(f"y must have {int(_g.N_Y)} species on its last axis")
    leading_shape = y_arr.shape[:-1]
    y_flat = np.ascontiguousarray(y_arr.reshape(-1, y_arr.shape[-1]))
    xCtot_arr = np.asarray(xCtot, dtype=np.float64)
    carbon_budget_scalar = xCtot_arr.ndim == 0
    xCtot_flat = (
        xCtot_arr.reshape(1)
        if carbon_budget_scalar
        else np.ascontiguousarray(
            np.broadcast_to(xCtot_arr, leading_shape).reshape(-1)
        )
    )
    carbon_species = np.asarray([
        _g.I_HCOP,
        _g.I_CHX,
        _g.I_CO,
        _g.I_CO_ICE,
        _g.I_CP,
    ], dtype=np.int64)

    if xOtot is not None:
        xOtot_arr = np.asarray(xOtot, dtype=np.float64)
        oxygen_budget_scalar = xOtot_arr.ndim == 0
        xOtot_flat = (
            xOtot_arr.reshape(1)
            if oxygen_budget_scalar
            else np.ascontiguousarray(
                np.broadcast_to(xOtot_arr, leading_shape).reshape(-1)
            )
        )
        oxygen_species = np.asarray([
            _g.I_OHX,
            _g.I_HCOP,
            _g.I_CO,
            _g.I_CO_ICE,
            _g.I_OP,
        ], dtype=np.int64)
        project_oxygen = True
    else:
        xOtot_flat = np.empty(0, dtype=np.float64)
        oxygen_species = np.empty(0, dtype=np.int64)
        oxygen_budget_scalar = False
        project_oxygen = False

    projected = _project_gow17_state_to_budgets_parallel(
        y_flat,
        xCtot_flat,
        xOtot_flat,
        carbon_species,
        oxygen_species,
        carbon_budget_scalar,
        oxygen_budget_scalar,
        project_oxygen,
    )
    return projected.reshape(y_arr.shape)


def _gow17_validation_summary(meta: dict) -> dict:
    """Return a compact summary for GOW17 validation runs."""
    diag = meta.get("gow17_diagnostics", {})
    time_cvode_failure_cells_total = int(sum(diag.get("time_cvode_failure_cells_hist", [])))
    time_exception_failure_cells_total = int(sum(diag.get("time_exception_failure_cells_hist", [])))
    time_negative_abundance_corrections_total = int(
        sum(diag.get("time_negative_abundance_corrections_hist", []))
    )
    return {
        "n_fail": int(meta.get("n_fail", -1)),
        "max_status": int(meta.get("max_status", 0)),
        "temperature_mode": meta.get("temperature_mode"),
        "enable_co_phase": bool(meta.get("enable_co_phase", False)),
        "b_CO_requested_scalar_kms": diag.get("b_CO_requested_scalar_kms"),
        "b_CO_table_kms": diag.get("b_CO_table_kms"),
        "b_CO_scalar_approximation": bool(diag.get("b_CO_scalar_approximation", False)),
        "rhs_status": int(diag.get("gow17_rhs_status", 0)),
        "rhs_residual_max_global": float(diag.get("gow17_rhs_residual_max_global", float("nan"))),
        "thermal_balance_residual_max_abs": float(
            diag.get("thermal_balance_residual_max_abs", float("nan"))
        ),
        "tevol_max_cells_total": int(diag.get("tevol_max_cells_total", 0)),
        "negative_abundance_corrections_total": int(
            diag.get("negative_abundance_corrections_total", 0)
        ),
        "cvode_failure_cells_total": int(diag.get("cvode_failure_cells_total", 0)),
        "exception_failure_cells_total": int(diag.get("exception_failure_cells_total", 0)),
        "time_cvode_failure_cells_total": time_cvode_failure_cells_total,
        "time_exception_failure_cells_total": time_exception_failure_cells_total,
        "time_negative_abundance_corrections_total": time_negative_abundance_corrections_total,
    }
