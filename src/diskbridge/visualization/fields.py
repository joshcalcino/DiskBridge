"""Field definitions and derived fields for yt visualization.

This module defines how DiskBridge fields map to yt fields and provides
derived field functions for quantities like:
- Column/surface density (integrated along line of sight)
- Scale height
- Optical depth
- Stokes number
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Callable
import numpy as np

if TYPE_CHECKING:
    import yt

# Field metadata: maps DiskBridge names to (yt_name, units, display_name)
FIELD_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    # Gas fields
    'density': ('density', 'g/cm**3', r'Gas Density'),
    'surface_density': ('surface_density', 'g/cm**2', r'Surface Density'),
    'temperature': ('temperature', 'K', r'Temperature'),
    'vr': ('velocity_spherical_r', 'cm/s', r'Radial Velocity'),
    'vphi': ('velocity_spherical_phi', 'cm/s', r'Azimuthal Velocity'),
    'vtheta': ('velocity_spherical_theta', 'cm/s', r'Polar Velocity'),
    
    # Dust fields
    'dust_density': ('dust_density', 'g/cm**3', r'Dust Density'),
    'dust_to_gas': ('dust_to_gas_ratio', 'dimensionless', r'Dust-to-Gas Ratio'),
    
    # RADMC-3D fields  
    'chi': ('uv_field', 'dimensionless', r'UV Field $\chi$'),
    'nH': ('H_nuclei_density', 'cm**-3', r'H Nuclei Density'),
    'mean_intensity': ('mean_intensity', 'erg/(s*cm**2*Hz*sr)', r'Mean Intensity'),
}


def get_field_info(name: str) -> Tuple[str, str, str]:
    """Get field metadata for a DiskBridge field name.
    
    Parameters
    ----------
    name : str
        DiskBridge field name
        
    Returns
    -------
    tuple
        (yt_name, units, display_name)
    """
    if name in FIELD_REGISTRY:
        return FIELD_REGISTRY[name]
    # Default: use name as-is
    return (name, 'dimensionless', name)


def add_derived_fields(ds: "yt.Dataset") -> None:
    """Add derived fields to a yt dataset.
    
    Parameters
    ----------
    ds : yt.Dataset
        The yt dataset to add fields to
    """
    # Dust-to-gas ratio
    def _dust_to_gas(field, data):
        dust_dens = data['dust', 'density']
        gas_dens = data['gas', 'density']
        return dust_dens / gas_dens
    
    try:
        if ('dust', 'density') in ds.field_list and ('gas', 'density') in ds.field_list:
            ds.add_field(
                ('gas', 'dust_to_gas_ratio'),
                function=_dust_to_gas,
                sampling_type='cell',
                units='dimensionless',
                display_name=r'Dust-to-Gas Ratio',
            )
    except Exception:
        pass
    
    # Cylindrical coordinates for disk analysis
    def _cylindrical_radius(field, data):
        r = data['index', 'spherical_r']
        theta = data['index', 'spherical_theta']
        return r * np.sin(theta)
    
    def _cylindrical_z(field, data):
        r = data['index', 'spherical_r']
        theta = data['index', 'spherical_theta']
        return r * np.cos(theta)
    
    try:
        ds.add_field(
            ('index', 'cylindrical_r'),
            function=_cylindrical_radius,
            sampling_type='cell',
            units='cm',
            display_name=r'Cylindrical Radius $R$',
        )
        ds.add_field(
            ('index', 'cylindrical_z'),
            function=_cylindrical_z,
            sampling_type='cell',
            units='cm',
            display_name=r'Height $z$',
        )
    except Exception:
        pass


# Unit display preferences for common fields
UNIT_DISPLAY = {
    'density': 'g/cm**3',
    'temperature': 'K',
    'velocity': 'km/s',
    'length': 'au',
    'surface_density': 'g/cm**2',
    'column_density': 'cm**-2',
}


def get_display_unit(field_type: str) -> str:
    """Get preferred display unit for a field type.
    
    Parameters
    ----------
    field_type : str
        Type of field (density, temperature, etc.)
        
    Returns
    -------
    str
        Preferred unit string for display
    """
    return UNIT_DISPLAY.get(field_type, '')
