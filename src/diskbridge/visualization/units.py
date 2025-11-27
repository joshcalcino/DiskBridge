"""Unit conversion utilities for DiskBridge visualization.

This module provides functions to convert between Pint (used by DiskBridge)
and unyt (used by yt-project) unit systems.

The key functions are:
- pint_to_unyt: Convert Pint Quantity to unyt array
- unyt_to_pint: Convert unyt array back to Pint Quantity
- get_unyt_unit: Get unyt unit from a string
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union
import numpy as np

if TYPE_CHECKING:
    from diskbridge._units import Quantity

import unyt
from unyt import unyt_array, unyt_quantity

HAS_UNYT = True

def check_unyt_available() -> None:
    return None

# Import pint from diskbridge
from diskbridge._units import units as pint_units, Quantity as PintQuantity


def pint_to_unyt(quantity: "Quantity") -> "unyt_array":
    """Convert a Pint Quantity to a unyt array.
    
    Parameters
    ----------
    quantity : Pint Quantity
        The Pint quantity to convert. Can be a scalar or array.
        
    Returns
    -------
    unyt_array
        The equivalent unyt array with converted units.
        
    Notes
    -----
    Uses unyt's built-in from_pint() method which handles most
    common unit conversions automatically.
    
    Examples
    --------
    >>> from diskbridge._units import Quantity
    >>> q = Quantity([1, 2, 3], 'au')
    >>> u = pint_to_unyt(q)
    >>> print(u.units)
    AU
    """
    
    # Convert to base units first to ensure clean conversion
    q_base = quantity.to_base_units()
    
    # Use unyt's from_pint method
    return unyt_array.from_pint(q_base)


def pint_to_unyt_cgs(quantity: "Quantity") -> "unyt_array":
    """Convert a Pint Quantity to a unyt array in CGS units.
    
    This is preferred for RADMC-3D compatibility where CGS is standard.
    
    Parameters
    ----------
    quantity : Pint Quantity
        The Pint quantity to convert.
        
    Returns
    -------
    unyt_array
        The equivalent unyt array in CGS units.
    """
    
    # Get the magnitude in CGS
    try:
        cgs_mag = quantity.to_base_units().magnitude
    except Exception:
        cgs_mag = quantity.magnitude
    
    # Determine the CGS unit string
    dims = quantity.dimensionality
    
    # Map common dimensionalities to CGS units
    cgs_unit = _dimensionality_to_cgs(dims)
    
    return unyt_array(cgs_mag, cgs_unit)


def unyt_to_pint(arr: "unyt_array", target_units: str = None) -> "Quantity":
    """Convert a unyt array back to a Pint Quantity.
    
    Parameters
    ----------
    arr : unyt_array
        The unyt array to convert.
    target_units : str, optional
        Target Pint unit string. If None, uses the unyt units directly.
        
    Returns
    -------
    Quantity
        The equivalent Pint Quantity.
        
    Examples
    --------
    >>> from unyt import unyt_array
    >>> u = unyt_array([1, 2, 3], 'AU')
    >>> q = unyt_to_pint(u, 'au')
    >>> print(q.units)
    astronomical_unit
    """
    
    # Use unyt's to_pint method
    pint_q = arr.to_pint()
    
    # Convert to target units if specified
    if target_units is not None:
        pint_q = pint_q.to(target_units)
    
    return pint_q


def get_unyt_unit(unit_str: str) -> "unyt.Unit":
    """Get a unyt Unit object from a string.
    
    Parameters
    ----------
    unit_str : str
        Unit string (e.g., 'g/cm**3', 'K', 'cm/s')
        
    Returns
    -------
    unyt.Unit
        The unyt Unit object.
    """
    return unyt.Unit(unit_str)


def _dimensionality_to_cgs(dims: dict) -> str:
    """Map Pint dimensionality dict to CGS unit string.
    
    Parameters
    ----------
    dims : dict
        Pint dimensionality dictionary like {'[length]': 1, '[mass]': -3}
        
    Returns
    -------
    str
        CGS unit string like 'g/cm**3'
    """
    # Extract powers from dimensionality
    length_pow = float(dims.get('[length]', 0))
    mass_pow = float(dims.get('[mass]', 0))
    time_pow = float(dims.get('[time]', 0))
    temp_pow = float(dims.get('[temperature]', 0))
    
    # Build CGS unit string
    parts = []
    
    if mass_pow != 0:
        if mass_pow == 1:
            parts.append('g')
        else:
            parts.append(f'g**{int(mass_pow)}')
    
    if length_pow != 0:
        if length_pow == 1:
            parts.append('cm')
        else:
            parts.append(f'cm**{int(length_pow)}')
    
    if time_pow != 0:
        if time_pow == 1:
            parts.append('s')
        else:
            parts.append(f's**{int(time_pow)}')
    
    if temp_pow != 0:
        if temp_pow == 1:
            parts.append('K')
        else:
            parts.append(f'K**{int(temp_pow)}')
    
    if not parts:
        return 'dimensionless'
    
    # Join with multiplication, handle division for negative powers
    result = '*'.join(parts)
    return result if result else 'dimensionless'


# Common unit mappings for convenience
DISKBRIDGE_TO_YT_UNITS = {
    'density': 'g/cm**3',
    'surface_density': 'g/cm**2',
    'temperature': 'K',
    'velocity': 'cm/s',
    'length': 'cm',
    'mass': 'g',
    'time': 's',
}

YT_FIELD_UNITS = {
    ('gas', 'density'): 'g/cm**3',
    ('gas', 'temperature'): 'K',
    ('gas', 'velocity_r'): 'cm/s',
    ('gas', 'velocity_phi'): 'cm/s',
    ('gas', 'velocity_theta'): 'cm/s',
    ('dust', 'density'): 'g/cm**3',
    ('radmc', 'uv_field'): 'dimensionless',
    ('radmc', 'mean_intensity'): 'erg/(s*cm**2*Hz*sr)',
}
