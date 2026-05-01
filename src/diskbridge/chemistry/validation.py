"""Chemistry state validation functions.

Ensures consistency between computed abundances and conservation laws.
"""

from __future__ import annotations

import numpy as np

from diskbridge._units import Quantity


def check_hydrogen_conservation(
    nH2: Quantity,
    nHI: Quantity,
    nH: Quantity,
    rtol: float = 1e-4,
) -> None:
    """Check H nuclei conservation: nHI + 2*nH2 == nH.

    Parameters
    ----------
    nH2 : Quantity
        H2 density [cm^-3]
    nHI : Quantity
        Atomic H density [cm^-3]
    nH : Quantity
        Total H nuclei density [cm^-3]
    rtol : float
        Relative tolerance

    Raises
    ------
    ValueError
        If conservation is violated beyond tolerance
    """
    nH2_flat = nH2.to('cm^-3').magnitude.flatten()
    nHI_flat = nHI.to('cm^-3').magnitude.flatten()
    nH_flat = nH.to('cm^-3').magnitude.flatten()

    computed = nHI_flat + 2.0 * nH2_flat
    rel_err = np.abs(computed - nH_flat) / np.maximum(nH_flat, 1e-30)
    max_err = np.max(rel_err)

    if max_err > rtol:
        idx = np.argmax(rel_err)
        raise ValueError(
            f"H conservation violated: max relative error = {max_err:.2e} "
            f"at index {idx} (nHI={nHI_flat[idx]:.2e}, nH2={nH2_flat[idx]:.2e}, "
            f"nH={nH_flat[idx]:.2e}, computed={computed[idx]:.2e})"
        )


def check_carbon_budget(
    nC: Quantity,
    nCplus: Quantity,
    nco_total: Quantity,
    nH: Quantity,
    X_C_tot: float = None,
    rtol: float = 1e-4,
) -> None:
    """Check carbon budget: nC + nCplus + nCO <= X_C_tot * nH.

    Parameters
    ----------
    nC : Quantity
        Neutral C density [cm^-3]
    nCplus : Quantity
        C+ density [cm^-3]
    nco_total : Quantity
        Total CO density [cm^-3]
    nH : Quantity
        H nuclei density [cm^-3]
    X_C_tot : float, optional
        Total carbon abundance. If None, uses value from _constants.
    rtol : float
        Relative tolerance

    Raises
    ------
    ValueError
        If carbon budget is violated
    """
    from diskbridge._constants import X_C_TOT as _X_C_TOT

    if X_C_tot is None:
        X_C_tot = _X_C_TOT

    nC_flat = nC.to('cm^-3').magnitude.flatten()
    nCplus_flat = nCplus.to('cm^-3').magnitude.flatten()
    nco_flat = nco_total.to('cm^-3').magnitude.flatten()
    nH_flat = nH.to('cm^-3').magnitude.flatten()

    n_C_total = nC_flat + nCplus_flat + nco_flat
    n_C_max = X_C_tot * nH_flat

    excess = n_C_total - n_C_max
    rel_excess = excess / np.maximum(n_C_max, 1e-30)
    max_excess = np.max(rel_excess)

    if max_excess > rtol:
        idx = np.argmax(rel_excess)
        raise ValueError(
            f"Carbon budget violated: nC + nCplus + nCO exceeds X_C*nH by "
            f"{max_excess:.2e} (relative) at index {idx}. "
            f"nC={nC_flat[idx]:.2e}, nCplus={nCplus_flat[idx]:.2e}, "
            f"nCO={nco_flat[idx]:.2e}, X_C*nH={n_C_max[idx]:.2e}"
        )


def check_non_negative(
    *quantities: tuple[Quantity, str],
) -> None:
    """Check that all quantities are non-negative.

    Parameters
    ----------
    *quantities : tuple[Quantity, str]
        Pairs of (quantity, name) to check

    Raises
    ------
    ValueError
        If any quantity has negative values
    """
    for qty, name in quantities:
        arr = qty.magnitude.flatten()
        min_val = np.min(arr)
        if min_val < 0:
            idx = np.argmin(arr)
            raise ValueError(
                f"Negative {name} found: min={min_val:.2e} at index {idx}"
            )


def check_photodesorption_diagnostics(
    n_ice_act_max: Quantity,
    n_ice_act: Quantity,
    k_pd_surf: Quantity,
    R_pd: Quantity,
    rtol: float = 1e-8,
) -> None:
    nmax = n_ice_act_max.to('cm^-3').magnitude.flatten()
    nact = n_ice_act.to('cm^-3').magnitude.flatten()
    ksurf = k_pd_surf.to('1/s').magnitude.flatten()
    rpd = R_pd.to('cm^-3/s').magnitude.flatten()

    if nmax.shape != nact.shape or nmax.shape != rpd.shape:
        raise ValueError(
            f"Photodesorption diagnostic shapes must match: "
            f"n_ice_act_max={nmax.shape}, n_ice_act={nact.shape}, R_pd={rpd.shape}"
        )

    if np.any(nmax < 0.0) or np.any(nact < 0.0) or np.any(rpd < 0.0) or np.any(ksurf < 0.0):
        check_non_negative(
            (n_ice_act_max, 'n_ice_act_max'),
            (n_ice_act, 'n_ice_act'),
            (k_pd_surf, 'k_pd_surf'),
            (R_pd, 'R_pd'),
        )

    if np.any(nact > nmax * (1.0 + float(rtol))):
        idx = int(np.argmax(nact - nmax))
        raise ValueError(
            f"n_ice_act exceeds n_ice_act_max at index {idx}: "
            f"n_ice_act={nact[idx]:.3e}, n_ice_act_max={nmax[idx]:.3e}"
        )

    rpd_max = ksurf * nmax
    if np.any(rpd > rpd_max * (1.0 + float(rtol))):
        idx = int(np.argmax(rpd - rpd_max))
        raise ValueError(
            f"R_pd exceeds k_pd_surf*n_ice_act_max at index {idx}: "
            f"R_pd={rpd[idx]:.3e}, limit={rpd_max[idx]:.3e}"
        )


def gow17_budget_diagnostics(
    y: np.ndarray,
    xCtot,
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
        Uses the same index layout as ``diskbridge._gow17``.
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
    import diskbridge._gow17 as _g

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

    xH_accounted = xOHx + xCHx + xHCOp + 3.0 * xH3p + 2.0 * xH2p + xHp + 2.0 * xH2
    xH_atom_raw = 1.0 - xH_accounted

    xC_accounted = xHCOp + xCHx + xCO + xCOice + xCplus
    xC_neutral_raw = xCtot - xC_accounted

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

    return {
        "h_xH_atom_min": h_xH_atom_min,
        "h_xH_accounted_max": h_xH_accounted_max,
        "c_xC_neutral_min": c_xC_neutral_min,
        "c_xC_accounted_max": c_xC_accounted_max,
        "h_budget_violation": h_budget_violation,
        "c_budget_violation": c_budget_violation,
    }


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
    import diskbridge._gow17 as _g

    y_proj = np.asarray(y, dtype=np.float64).copy()
    y_proj[...] = np.maximum(y_proj, 0.0)

    xCtot_arr = np.asarray(xCtot, dtype=np.float64)
    carbon_species = [
        _g.I_HCOP,
        _g.I_CHX,
        _g.I_CO,
        _g.I_CO_ICE,
        _g.I_CP,
    ]
    c_sum = np.zeros(y_proj.shape[:-1], dtype=np.float64)
    for idx in carbon_species:
        c_sum += y_proj[..., idx]
    c_scale = np.divide(
        xCtot_arr,
        c_sum,
        out=np.ones_like(c_sum, dtype=np.float64),
        where=(c_sum > xCtot_arr) & (c_sum > 0.0),
    )
    for idx in carbon_species:
        y_proj[..., idx] *= c_scale

    if xOtot is not None:
        xOtot_arr = np.asarray(xOtot, dtype=np.float64)
        oxygen_species = [
            _g.I_OHX,
            _g.I_HCOP,
            _g.I_CO,
            _g.I_CO_ICE,
            _g.I_OP,
        ]
        o_sum = np.zeros(y_proj.shape[:-1], dtype=np.float64)
        for idx in oxygen_species:
            o_sum += y_proj[..., idx]
        o_scale = np.divide(
            xOtot_arr,
            o_sum,
            out=np.ones_like(o_sum, dtype=np.float64),
            where=(o_sum > xOtot_arr) & (o_sum > 0.0),
        )
        for idx in oxygen_species:
            y_proj[..., idx] *= o_scale

    return y_proj


def validate_chemistry_state(
    nH: Quantity,
    nH2: Quantity = None,
    nHI: Quantity = None,
    nC: Quantity = None,
    nCplus: Quantity = None,
    nco_total: Quantity = None,
    check_h: bool = True,
    check_c: bool = True,
    check_pd: bool = False,
    n_ice_act_max: Quantity = None,
    n_ice_act: Quantity = None,
    k_pd_surf: Quantity = None,
    R_pd: Quantity = None,
    rtol: float = 1e-4,
) -> None:
    """Validate chemistry state for consistency.

    Parameters
    ----------
    nH : Quantity
        Total H nuclei density
    nH2 : Quantity, optional
        H2 density
    nHI : Quantity, optional
        Atomic H density
    nC : Quantity, optional
        Neutral C density
    nCplus : Quantity, optional
        C+ density
    nco_total : Quantity, optional
        Total CO density
    check_h : bool
        Check H conservation (requires nH2 and nHI)
    check_c : bool
        Check C budget (requires nC, nCplus, nco_total)
    rtol : float
        Relative tolerance

    Raises
    ------
    ValueError
        If any validation fails
    """
    to_check = [(nH, 'nH')]
    if nH2 is not None:
        to_check.append((nH2, 'nH2'))
    if nHI is not None:
        to_check.append((nHI, 'nHI'))
    if nC is not None:
        to_check.append((nC, 'nC'))
    if nCplus is not None:
        to_check.append((nCplus, 'nCplus'))
    if nco_total is not None:
        to_check.append((nco_total, 'nco_total'))

    check_non_negative(*to_check)

    if check_h and nH2 is not None and nHI is not None:
        check_hydrogen_conservation(nH2, nHI, nH, rtol=rtol)

    if check_c and nC is not None and nCplus is not None and nco_total is not None:
        check_carbon_budget(nC, nCplus, nco_total, nH, rtol=rtol)

    if check_pd:
        if n_ice_act_max is None or n_ice_act is None or k_pd_surf is None or R_pd is None:
            raise ValueError("check_pd=True requires n_ice_act_max, n_ice_act, k_pd_surf, and R_pd")
        check_photodesorption_diagnostics(
            n_ice_act_max=n_ice_act_max,
            n_ice_act=n_ice_act,
            k_pd_surf=k_pd_surf,
            R_pd=R_pd,
            rtol=rtol,
        )
