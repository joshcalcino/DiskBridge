from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel
    from diskbridge.chemistry.types import ChemistryResult

from diskbridge.chemistry.models import pinte, co_two_phase


REGISTRY: dict[str, Callable[['RadModel', dict], 'ChemistryResult']] = {
    'pinte_switches': pinte.run,
    'co_two_phase_steady': co_two_phase.run_steady,
    'co_two_phase_time': co_two_phase.run_time_dependent,
}
