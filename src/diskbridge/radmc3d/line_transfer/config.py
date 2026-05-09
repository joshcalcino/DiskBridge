from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from diskbridge.radmc3d.colliders import gow17_lamda_colliders


H2OPRMode = Literal["fixed_3_to_1"]
ColliderPolicy = Literal["gow17_lamda"]


def _normalize_collider_name(name: str) -> str:
    key = str(name).strip().lower().replace("_", "-")
    aliases = {
        "ph2": "p-h2",
        "para-h2": "p-h2",
        "oh2": "o-h2",
        "ortho-h2": "o-h2",
        "electron": "e",
        "e-": "e",
        "hp": "h+",
        "hplus": "h+",
        "h-atom": "h",
    }
    return aliases.get(key, key)


@dataclass(frozen=True)
class SpeciesLineConfig:
    """Line-transfer settings for one emitting species."""

    species: str
    line_mode: int = 3
    colliders: list[str] = field(default_factory=list)
    transition: int | None = None
    collider_policy: ColliderPolicy = "gow17_lamda"

    def __post_init__(self) -> None:
        """Require the single strict GOW17/LAMDA collider policy."""

        species = str(self.species).lower().strip()
        object.__setattr__(self, "species", species)
        if self.collider_policy != "gow17_lamda":
            raise ValueError("Only collider_policy='gow17_lamda' is supported")

        if abs(int(self.line_mode)) not in {3, 4}:
            if self.colliders:
                raise ValueError("LTE line modes must not specify non-LTE colliders")
            return

        expected = gow17_lamda_colliders(species)
        requested = [_normalize_collider_name(c) for c in self.colliders]
        if not requested:
            object.__setattr__(self, "colliders", expected)
            return
        if requested != expected:
            raise ValueError(
                "Non-LTE GOW17 line transfer supports only the strict "
                f"GOW17/LAMDA collider order for {species}: {expected}; "
                f"got {requested}."
            )
        object.__setattr__(self, "colliders", expected)


@dataclass(frozen=True)
class NonLTELineTransferConfig:
    """Minimal non-LTE line-transfer configuration for smoke runs."""

    use_gow17_tgas: bool = True
    tgas_eq_tdust: bool = False
    require_gas_velocity: bool = True
    h2_opr_mode: H2OPRMode = "fixed_3_to_1"
    lines_nonlte_maxiter: int = 300
    lines_nonlte_convcrit: float = 1.0e-4
    lines_slowlvg_as_alternative: bool = False
    incl_dust: int = 1
    itempdecoup: int = 1
    rto_style: int = 3
    line_tgas_min_K: float = 2.0
    line_tgas_max_K: float = 2999.0
