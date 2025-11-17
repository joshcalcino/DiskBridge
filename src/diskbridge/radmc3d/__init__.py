"""RADMC-3D interface module for DiskBridge.

This module provides functionality to:
- Compute dust opacities using Mie scattering
- Write RADMC-3D input files from Model instances
- Read RADMC-3D output files (future)

The module handles unit conversion from Pint Quantities (used in Model)
to CGS units (required by RADMC-3D).
"""

from .opacities import DustOpacityCalculator
from .writer import RADMC3DWriter

__all__ = [
    'DustOpacityCalculator',
    'RADMC3DWriter',
]
