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


def get_gas_temperature(rad: 'RadModel') -> Optional[Quantity]:
    """Return the best available gas temperature field on a RadModel."""
    tgas = getattr(rad, 'Tgas_gow17', None)
    if tgas is not None:
        return tgas

    tgas = getattr(rad, 'gas_temperature', None)
    if tgas is not None:
        return tgas

    if hasattr(rad, 'ensure_gas_temperature'):
        return rad.ensure_gas_temperature()

    return None


def write_gas_temperature(
    rad: 'RadModel',
    output_dir: Optional[Path] = None,
    *,
    binary: bool = True,
) -> Path:
    """Write the current gas temperature field to RADMC-3D format."""
    if output_dir is None:
        output_dir = rad.model_dir

    tgas = get_gas_temperature(rad)
    if tgas is None:
        raise RuntimeError("No gas temperature field is available to write")

    base_dir = Path(output_dir)
    if getattr(rad.writer, 'organize_files', False):
        output_path = base_dir / getattr(rad.writer, 'inputs_dir', 'radmc3d_inputs')
    else:
        output_path = base_dir

    rad.writer.write_gas_temperature(tgas, output_dir=output_path, binary=binary)
    suffix = "binp" if binary else "inp"
    return output_path / f"gas_temperature.{suffix}"
