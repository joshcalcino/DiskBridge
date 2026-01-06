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
