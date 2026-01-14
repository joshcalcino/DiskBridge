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
) -> Tuple[Quantity, Quantity]:
    if (not force) and getattr(rad, "nH2", None) is not None and getattr(rad, "nH_atom", None) is not None:
        return rad.nH2, rad.nH_atom

    nH_cm3 = np.ascontiguousarray(nH.to("cm^-3").magnitude, dtype=np.float64)
    chi_dim = np.ascontiguousarray(chi_dust.to("dimensionless").magnitude, dtype=np.float64)

    nH2_cm3, nH_atom_cm3 = compute_h2_partition(
        rad=rad,
        mesh=rad.model.mesh,
        nH_cm3=nH_cm3,
        chi_dust=chi_dim,
    )

    rad.nH2 = Quantity(nH2_cm3, "cm^-3")
    rad.nH_atom = Quantity(nH_atom_cm3, "cm^-3")
    return rad.nH2, rad.nH_atom
