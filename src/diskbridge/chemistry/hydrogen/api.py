from __future__ import annotations

from typing import Tuple

import numpy as np

from diskbridge._units import Quantity
from diskbridge.chemistry.hydrogen.partition import compute_h2_partition


def ensure_h2_partition(
    rad,
    *,
    nH: Quantity,
    chi_dust: Quantity,
    force: bool = False,
    h2_self_shielding: bool = True,
) -> Tuple[Quantity, Quantity]:
    """Ensure H2/HI partition is computed and stored on ``rad``.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper.
    nH : Quantity
        Total hydrogen number density.
    chi_dust : Quantity
        Dust-attenuated UV field (Draine units).
    force : bool, optional
        If True, recompute even if nH2/nH_atom already exist.
    h2_self_shielding : bool, optional
        If True (default), include H2 self-shielding iterations
        (Draine & Bertoldi 1996). If False, use unshielded balance.

    Returns
    -------
    nH2 : Quantity
        H2 number density.
    nH_atom : Quantity
        Atomic hydrogen number density.
    """
    if (not force) and getattr(rad, "nH2", None) is not None and getattr(rad, "nH_atom", None) is not None:
        return rad.nH2, rad.nH_atom

    nH_cm3 = np.ascontiguousarray(nH.to("cm^-3").magnitude, dtype=np.float64)
    chi_dim = np.ascontiguousarray(chi_dust.to("dimensionless").magnitude, dtype=np.float64)

    nH2_cm3, nH_atom_cm3 = compute_h2_partition(
        rad=rad,
        mesh=rad.model.mesh,
        nH_cm3=nH_cm3,
        chi_dust=chi_dim,
        h2_self_shielding=bool(h2_self_shielding),
    )

    rad.nH2 = Quantity(nH2_cm3, "cm^-3")
    rad.nH_atom = Quantity(nH_atom_cm3, "cm^-3")
    return rad.nH2, rad.nH_atom
