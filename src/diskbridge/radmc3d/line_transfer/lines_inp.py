# db-keywords: line-transfer, config, radmc3d, io, paths, molecule-data
# db-role: canonical
# db-scope: package
# db-purpose: Package module for line-transfer, config, radmc3d, io.

from __future__ import annotations

from pathlib import Path

from .config import SpeciesLineConfig


_COLLIDER_ALIASES = {
    "h": "h",
    "h_atom": "h",
    "e": "e",
    "e-": "e",
    "electron": "e",
    "h+": "h+",
    "hp": "h+",
    "hplus": "h+",
    "he": "he",
    "h2": "h2",
    "p-h2": "p-h2",
    "ph2": "p-h2",
    "para-h2": "p-h2",
    "o-h2": "o-h2",
    "oh2": "o-h2",
    "ortho-h2": "o-h2",
}


def normalize_collider_name(name: str) -> str:
    """Return the RADMC-3D collider label used by DiskBridge."""

    key = str(name).strip().lower().replace("_", "-")
    return _COLLIDER_ALIASES.get(key, key)


def lines_inp_content(species_config: SpeciesLineConfig) -> str:
    """Return the single-species RADMC-3D ``lines.inp`` content."""

    species = str(species_config.species).strip().lower()
    colliders = [normalize_collider_name(c) for c in species_config.colliders]
    lines = [
        "2",
        "1",
        f"{species}    leiden    0    0    {len(colliders)}",
        *colliders,
    ]
    return "\n".join(lines) + "\n"


def write_lines_inp(path: str | Path, species_config: SpeciesLineConfig) -> Path:
    """Write a single-species RADMC-3D ``lines.inp`` file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lines_inp_content(species_config))
    return path


expected_lines_inp_content = lines_inp_content
