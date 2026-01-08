from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel
    from diskbridge.chemistry.types import ChemistryResult

from diskbridge.chemistry.models import pinte_switches, carbon_reduced
from diskbridge.chemistry.models import gow17_pdr
from diskbridge.chemistry.models import gow17


REGISTRY: dict[str, Callable[['RadModel', dict], 'ChemistryResult']] = {
    'carbon_reduced': carbon_reduced.run_carbon_reduced,
    'pinte_switches': pinte_switches.run_pinte_switches,
    'gow17_pdr': gow17_pdr.run_gow17_pdr,
    'gow17': gow17.run_gow17,
}
