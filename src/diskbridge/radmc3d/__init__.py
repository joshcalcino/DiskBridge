"""RADMC-3D writer and utilities for DiskBridge Models.

This submodule provides tools to write RADMC-3D input files from DiskBridge
Model objects, handling unit conversions and data formatting automatically.

Main components:
- RADMC3DWriter: Write RADMC-3D input files (AMR grid, dust density, opacity, etc.)
- RADMC3DModel: Extended model class for molecular line radiative transfer
- DustOpacityCalculator: Compute dust opacities using Mie theory
"""

from .writer import RADMC3DWriter
from .model import RADMC3DModel
from .opacities import DustOpacityCalculator

__all__ = [
    'RADMC3DWriter',
    'RADMC3DModel',
    'DustOpacityCalculator',
]
