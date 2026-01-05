from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel
    from diskbridge.chemistry.types import ChemistryResult

from diskbridge.chemistry.models import pinte_switches, co_two_phase


REGISTRY: dict[str, Callable[['RadModel', dict], 'ChemistryResult']] = {
    'co_two_phase': co_two_phase.run_co_two_phase,
    'pinte_switches': pinte_switches.run_pinte_switches,
}
