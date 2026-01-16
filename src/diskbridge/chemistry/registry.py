from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel
    from diskbridge.chemistry.types import ChemistryResult

from diskbridge.chemistry.models import pinte_switches
from diskbridge.chemistry.models import gow17
from diskbridge.chemistry.models import carbon_reduced


REGISTRY: dict[str, Callable[['RadModel', dict], 'ChemistryResult']] = {
    'pinte_switches': pinte_switches.run_pinte_switches,
    'gow17': gow17.run_gow17,
    'carbon_reduced': carbon_reduced.run_carbon_reduced,
}
