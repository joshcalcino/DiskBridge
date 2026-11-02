# db-keywords: gow17, line-transfer, gas-temperature, units, radmc3d, chemistry, model, mesh
# db-role: canonical
# db-scope: package
# db-purpose: Package module for gow17, line-transfer, gas-temperature, units.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict
from pathlib import Path

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

from diskbridge._units import Quantity
from diskbridge.model.field import Field


def attach_chemistry_result_to_model(rad: 'RadModel', result) -> None:
    """Attach chemistry outputs to ``rad.model.gas`` as realized Fields."""

    model = getattr(rad, 'model', None)
    if model is None or getattr(model, 'gas', None) is None or getattr(model, 'mesh', None) is None:
        return

    axis_order = model.mesh.axis_names()
    if hasattr(rad, '_chem_axis_order'):
        axis_order = rad._chem_axis_order()

    tgas = getattr(rad, 'Tgas_gow17', None)
    if tgas is None:
        tgas = getattr(rad, 'gas_temperature', None)
    if tgas is None and hasattr(rad, 'ensure_gas_temperature'):
        tgas = rad.ensure_gas_temperature()

    if tgas is not None:
        model.gas_register(
            'gas_temperature',
            Field(
                quantity='gas_temperature',
                data=tgas,
                axis_order=axis_order,
                attrs={'source': 'rad'},
            ),
        )

    for sp, x in result.abundances.items():
        model.gas_register(
            f'abundance_{sp}',
            Field(
                quantity=f'abundance_{sp}',
                data=x,
                axis_order=axis_order,
                attrs={'source': 'chemistry', 'kind': 'abundance', 'species': sp},
            ),
        )

    for sp, n in result.number_densities.items():
        model.gas_register(
            f'number_density_{sp}',
            Field(
                quantity=f'number_density_{sp}',
                data=n,
                axis_order=axis_order,
                attrs={'source': 'chemistry', 'kind': 'number_density', 'species': sp},
            ),
        )

    for name, q in result.fields.items():
        model.gas_register(
            f'chem_{name}',
            Field(
                quantity=f'chem_{name}',
                data=q,
                axis_order=axis_order,
                attrs={'source': 'chemistry', 'kind': 'diagnostic', 'name': name},
            ),
        )


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
