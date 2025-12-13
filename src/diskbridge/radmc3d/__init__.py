"""RADMC-3D interface for DiskBridge Models.

This submodule provides a complete interface to RADMC-3D, mirroring the
structure of radmc3dPy with clear separation of concerns:

Main components:
- writer: Write RADMC-3D input files (AMR grid, dust density, etc.)
- data: Read RADMC-3D output files (temperature, intensity, etc.)
- model: High-level model wrapper for photochemistry and workflows
- molecule: Molecular data handling (Leiden LAMDA format)
- image: Read RADMC-3D images and write FITS files
- opacities: Dust opacity calculations using Mie theory

Architecture:
- RadWriter: Writes input files from DiskBridge models
- RadData: Reads RADMC-3D output files
- RadModel: High-level wrapper for molecular RT workflows
- RadMolecule: Molecular data from molecule_*.inp files
- RadImage: Reads images and writes FITS files
- DustOpacityCalculator: Dust opacity computations

This separation ensures clean interfaces:
- Input (writer) vs Output (data, image)
- Low-level I/O (data, writer, image) vs High-level workflows (model)
- Model building (Model class) vs RADMC-3D operations (RadModel)
"""

from .writer import RadWriter
from .data import RadData
from .model import RadModel
from .molecule import RadMolecule
from .image import RadImage
from .opacities import DustOpacityCalculator

__all__ = [
    'RadWriter',
    'RadData',
    'RadModel',
    'RadMolecule',
    'RadImage',
    'DustOpacityCalculator',
]
