from __future__ import annotations

from typing import Tuple

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

    nH2, nH_atom = compute_h2_partition(
        rad=rad,
        mesh=rad.model.mesh,
        nH=nH,
        chi_dust=chi_dust,
    )

    rad.nH2 = nH2
    rad.nH_atom = nH_atom
    return nH2, nH_atom
