from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict
from pathlib import Path

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

from diskbridge._units import Quantity


def write_numberdens(
    rad: 'RadModel',
    species: str,
    ndens: Quantity,
    output_dir: Optional[Path] = None,
) -> None:
    """Write a single number density field to RADMC-3D format.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    species : str
        Molecule/species name
    ndens : Quantity
        Number density in cm^-3
    output_dir : Path, optional
        Output directory (default: rad.model_dir)
    """
    if output_dir is None:
        output_dir = rad.model_dir
    
    rad.writer.write_number_density(species, ndens, output_dir=output_dir)


def write_many(
    rad: 'RadModel',
    number_densities: Dict[str, Quantity],
    output_dir: Optional[Path] = None,
) -> None:
    """Write multiple number density fields to RADMC-3D format.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    number_densities : dict[str, Quantity]
        Dictionary mapping species names to number densities
    output_dir : Path, optional
        Output directory (default: rad.model_dir)
    """
    for species, ndens in number_densities.items():
        write_numberdens(rad, species, ndens, output_dir)
