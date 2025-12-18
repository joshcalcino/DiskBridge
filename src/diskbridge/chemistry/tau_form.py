from __future__ import annotations

from typing import Optional

import numpy as np

from diskbridge._units import Quantity


def compute_tau_form_co(
    *,
    nH: Optional[Quantity] = None,
    nH2: Optional[Quantity] = None,
    fH2: Optional[float | np.ndarray | Quantity] = None,
    n0: Quantity = Quantity(1.0e4, "cm^-3"),
    tau0: Quantity = Quantity(1.0e5, "yr"),
    tau_min: Quantity = Quantity(1.0e3, "yr"),
    alpha: float = 1.0,
) -> Quantity:
    """Compute an effective CO formation timescale field.

    This tau_form is effective (not a full network).
    tau_form ~ 1/n is NL97-style reduced chemistry motivated.
    In very dense gas, formation is capped because CO formation is not the bottleneck.
    """

    if fH2 is not None:
        if nH is None:
            raise ValueError("fH2 was provided but nH is None; provide nH when using fH2")
        if isinstance(fH2, Quantity):
            fH2_mag = np.asarray(fH2.to("dimensionless").magnitude)
        else:
            fH2_mag = np.asarray(fH2)
        nH_mag = np.asarray(nH.to("cm^-3").magnitude)
        n_eff = nH_mag * np.asarray(fH2_mag, dtype=nH_mag.dtype)
        n_eff = Quantity(n_eff, "cm^-3")
    elif nH2 is not None:
        n_eff = nH2
    elif nH is not None:
        n_eff = nH
    else:
        raise ValueError("Must provide nH and/or nH2")

    n_eff_cm3 = n_eff.to("cm^-3")
    n_eff_mag = np.asarray(n_eff_cm3.magnitude)
    if np.any(n_eff_mag <= 0.0):
        raise ValueError("n_eff must be > 0 everywhere to compute a formation timescale")

    dtype = n_eff_mag.dtype
    n0_mag = np.asarray(n0.to("cm^-3").magnitude, dtype=dtype)
    tau0_mag = np.asarray(tau0.to("yr").magnitude, dtype=dtype)
    tau_min_mag = np.asarray(tau_min.to("yr").magnitude, dtype=dtype)
    alpha_mag = np.asarray(float(alpha), dtype=dtype)

    # CO formation timescale model:
    # - Motivated by reduced CO networks (Nelson & Langer 1997-style),
    #   where CO formation is effectively two-body and scales ~1/n in molecular gas.
    #   See summary/comparison in Glover & Clark 2012, MNRAS, 421, 116.  (NL97 discussed there)
    # - We cap the fastest allowed formation (tau_min) so that in very dense gas (e.g. disks)
    #   formation is effectively instantaneous compared to other processes.
    #   This is chosen shorter than typical CO freeze-out timescales in dense/cold gas
    #   (often <= 1e4 yr; see e.g. Hocuk et al. 2016, MNRAS 456, 2586; Flower et al. 2005, A&A 436, 933).

    tau_mag = tau0_mag * np.power(n0_mag / n_eff_mag, alpha_mag)
    tau_mag = np.maximum(tau_mag, tau_min_mag)
    return Quantity(tau_mag, "yr")
