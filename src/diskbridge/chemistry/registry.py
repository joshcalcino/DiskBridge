from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel
    from diskbridge.chemistry.types import ChemistryResult

_MODEL_SPECS: dict[str, tuple[str, str]] = {
    "pinte_switches": ("diskbridge.chemistry.models.pinte_switches", "run_pinte_switches"),
    "layered_column_switches": ("diskbridge.chemistry.models.abundance_switches", "run_abundance_switches"),
    "gow17": ("diskbridge.chemistry.models.gow17", "run_gow17"),
    "carbon_reduced": ("diskbridge.chemistry.models.carbon_reduced", "run_carbon_reduced"),
}


def available_models() -> list[str]:
    return sorted(_MODEL_SPECS.keys())


def get_model_callable(model: str) -> Callable[["RadModel", dict], "ChemistryResult"]:
    model_lower = str(model).lower()
    if model_lower not in _MODEL_SPECS:
        available = ", ".join(available_models())
        raise ValueError(f"Unknown chemistry model: {model!r}. Available models: {available}")

    module_path, fn_name = _MODEL_SPECS[model_lower]
    mod = import_module(module_path)
    fn = getattr(mod, fn_name)
    return fn


__all__ = [
    "available_models",
    "get_model_callable",
]
