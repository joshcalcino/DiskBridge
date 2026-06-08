# db-keywords: config, units, field, paths, opacity, dust
# db-role: canonical
# db-scope: package
# db-purpose: Package module for config, units, field, paths.

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np

from diskbridge._units import Quantity


SPECIES_REGISTRY = {
    "mix_2species_ice70": {
        "lnk": "mix_2species_ice70.lnk",
        "grain_density": 1.26,
    },
    "mix_2species_porous_ice70": {
        "lnk": "mix_2species_porous_ice70.lnk",
        "grain_density": 0.1,
    },
    "mix_2species_60silicates_40carbons": {
        "lnk": "mix_2species_60silicates_40carbons.lnk",
        "grain_density": 2.7,
    },
}


def normalize_species_name(species: str | Path | list[str]) -> str:
    if isinstance(species, list):
        if not species:
            raise ValueError("Dust species list must not be empty")
        species = species[0]
    name = Path(str(species)).name
    if name.endswith(".lnk"):
        name = name[:-4]
    return name


def is_known_species(species: str | Path) -> bool:
    return normalize_species_name(species) in SPECIES_REGISTRY


def species_lnk_filename(species: str | Path) -> str:
    name = normalize_species_name(species)
    entry = SPECIES_REGISTRY.get(name)
    if entry is not None:
        return str(entry["lnk"])
    return f"{name}.lnk"


def species_optconst_path(species: str | Path, opacity_dir: str | Path) -> Path:
    return Path(opacity_dir) / species_lnk_filename(species)


def _grain_density_to_float(value: Any) -> float:
    if isinstance(value, Quantity):
        return float(value.to("g/cm^3").magnitude)
    return float(value)


def _as_grain_density_quantity(value: Any) -> Quantity:
    if isinstance(value, Quantity):
        return value.to("g/cm^3")
    return Quantity(float(value), "g/cm^3")


def resolve_species_grain_density(
    species: str | Path,
    user_grain_density: Any | None = None,
) -> Quantity:
    """Resolve the intrinsic material density for a dust opacity species."""

    name = normalize_species_name(species)
    entry = SPECIES_REGISTRY.get(name)
    if entry is not None:
        rho_s = float(entry["grain_density"])
        if user_grain_density is not None:
            user_rho_s = _grain_density_to_float(user_grain_density)
            if not np.isclose(user_rho_s, rho_s, rtol=0.0, atol=1.0e-12):
                warnings.warn(
                    "grain_density was provided, but species "
                    f"'{name}' has a fixed intrinsic density "
                    f"rho_s={rho_s:g} g/cm^3. Using the species density for "
                    "settling and opacities.",
                    UserWarning,
                    stacklevel=2,
                )
        return Quantity(rho_s, "g/cm^3")

    if user_grain_density is None:
        raise ValueError(
            f"Unknown dust species '{name}' requires grain_density to be specified."
        )
    return _as_grain_density_quantity(user_grain_density)
