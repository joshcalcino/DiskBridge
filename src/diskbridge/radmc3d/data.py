"""RADMC-3D data reader class.

This module provides the radmc3dData class for reading RADMC-3D output files,
handling both ASCII and binary formats. It mirrors the structure of radmc3dPy.data
but adapted for DiskBridge's Model class.

The class is separate from the writer functionality, providing clean separation
between input (writer) and output (data reader) operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from diskbridge.model.model import Model

from diskbridge._logging import logger
from diskbridge._units import Quantity, units


class RadData:
    """RADMC-3D data reader for output files.
    
    This class handles reading RADMC-3D output files (dust temperature, 
    gas temperature, dust density, mean intensity, etc.) in both ASCII 
    and binary formats. It works with a DiskBridge Model instance to
    ensure consistency with the mesh structure.
    
    The class does NOT handle model building or writing input files - 
    those are handled by RADMC3DWriter. This maintains clear separation
    of read (data) and write (writer) functionality.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance with mesh definition
    model_dir : str or Path, optional
        Directory containing RADMC-3D files (default: current directory)
        
    Attributes
    ----------
    model : Model
        Reference to the DiskBridge model
    model_dir : Path
        Directory containing RADMC-3D files
    rhodust : Quantity or None
        Dust density in g/cm^3
    dusttemp : Quantity or None
        Dust temperature in K
    rhogas : Quantity or None
        Gas density in g/cm^3
    gastemp : Quantity or None
        Gas temperature in K
    gasvel : Quantity or None
        Gas velocity in cm/s
    vturb : Quantity or None
        Microturbulence in cm/s
    ndens_mol : Quantity or None
        Molecular number density in molecules/cm^3
    mean_intensity : Quantity or None
        Mean intensity field in erg/s/cm^2/Hz/sr
        
    Examples
    --------
    >>> from diskbridge import Model
    >>> from diskbridge.radmc3d import RadData
    >>> 
    >>> model = Model()
    >>> # ... load model ...
    >>> 
    >>> # Create data reader
    >>> data = RadData(model, model_dir='.')
    >>> 
    >>> # Read dust temperature
    >>> data.readDustTemp()
    >>> 
    >>> # Read gas temperature  
    >>> data.readGasTemp()
    """
    
    def __init__(self, model: 'Model', model_dir: str | Path = '.'):
        """Initialize the data reader.
        
        Parameters
        ----------
        model : Model
            DiskBridge Model instance
        model_dir : str or Path, optional
            Directory with RADMC-3D files (default: '.')
        """
        self.model = model
        self.model_dir = Path(model_dir)
        
        # Storage for RADMC-3D output data
        self.rhodust: Optional[Quantity] = None
        self.dusttemp: Optional[Quantity] = None
        self.rhogas: Optional[Quantity] = None
        self.gastemp: Optional[Quantity] = None
        self.gasvel: Optional[Quantity] = None
        self.vturb: Optional[Quantity] = None
        self.ndens_mol: Optional[Quantity] = None
        self.mean_intensity: Optional[Quantity] = None
    
    def _isBinary(self, fname: str | Path) -> bool:
        """Check if a file is in binary format.
        
        Parameters
        ----------
        fname : str or Path
            File path to check
            
        Returns
        -------
        bool
            True if binary format, False otherwise
        """
        fname_str = str(fname)
        ext = fname_str[-5:] if len(fname_str) >= 5 else fname_str
        return ext in ['.binp', '.bdat', '.bout']
    
    def _findDataFile(self, basename: str) -> Optional[Path]:
        """Find a data file with various extensions.
        
        Parameters
        ----------
        basename : str
            Base filename without extension
            
        Returns
        -------
        Path or None
            Path to found file, or None if not found
        """
        exts = ['.inp', '.binp', '.out', '.bout', '.dat', '.bdat']
        found_files = []
        
        for ext in exts:
            fpath = self.model_dir / (basename + ext)
            if fpath.exists():
                found_files.append(fpath)
        
        if len(found_files) > 1:
            logger.warning(f"Multiple files for basename '{basename}' found. Using {found_files[0]}")
        
        return found_files[0] if found_files else None
    
    def _getMeshShape(self) -> Tuple[int, int, int]:
        """Get mesh dimensions in RADMC-3D order.
        
        Returns
        -------
        tuple of int
            (nx, ny, nz) dimensions in RADMC-3D convention
        """
        mesh = self.model.mesh
        if mesh is None:
            raise ValueError("Model has no mesh defined")
        
        if mesh.coord_system == 'spherical':
            # RADMC-3D: (r, theta, phi)
            nr = len(mesh.axes['r'].centers)
            ntheta = len(mesh.axes['theta'].centers)
            nphi = len(mesh.axes['phi'].centers)
            return (nr, ntheta, nphi)
        elif mesh.coord_system == 'cylindrical':
            # RADMC-3D: (r, phi, z)
            nr = len(mesh.axes['r'].centers)
            nphi = len(mesh.axes['phi'].centers)
            nz = len(mesh.axes['z'].centers)
            return (nr, nphi, nz)
        elif mesh.coord_system == 'cartesian':
            # RADMC-3D: (x, y, z)
            nx = len(mesh.axes['x'].centers)
            ny = len(mesh.axes['y'].centers)
            nz = len(mesh.axes['z'].centers)
            return (nx, ny, nz)
        else:
            raise ValueError(f"Unsupported coordinate system: {mesh.coord_system}")
    
    def readDustTemp(self, fname: Optional[str | Path] = None, ispec: int = 0) -> Quantity:
        """Read dust temperature from RADMC-3D output.
        
        Tries binary format first (.bdat), then ASCII format (.dat).
        
        Parameters
        ----------
        fname : str or Path, optional
            Filename to read (default: auto-detect dust_temperature)
        ispec : int, optional
            Dust species index (default: 0)
            
        Returns
        -------
        Quantity
            Dust temperature in Kelvin
            
        Raises
        ------
        FileNotFoundError
            If no dust temperature file is found
        """
        if fname is None:
            fpath = self._findDataFile('dust_temperature')
            if fpath is None:
                raise FileNotFoundError("No dust_temperature file found")
        else:
            fpath = Path(fname)
            if not fpath.exists():
                raise FileNotFoundError(f"File not found: {fpath}")
        
        if self._isBinary(fpath):
            data = self._readScalarFieldBinary(fpath)
        else:
            data = self._readScalarFieldASCII(fpath)
        
        # Extract the requested species
        logger.debug(f"Dust temperature data shape: {data.shape}, dtype: {data.dtype}, ispec: {ispec}, type(ispec): {type(ispec)}")
        
        if len(data.shape) == 2:
            # Multiple species: (nspec, ncells)
            if not isinstance(ispec, (int, np.integer)):
                ispec = int(ispec)
            temp = data[ispec, :]
        elif len(data.shape) == 1:
            # Single species: just use the data directly
            temp = data
        else:
            # Unexpected shape
            raise ValueError(f"Unexpected dust temperature data shape: {data.shape}")
        
        # Reshape to mesh
        nx, ny, nz = self._getMeshShape()
        temp = temp.reshape((nx, ny, nz), order='F')  # Fortran order
        
        self.dusttemp = Quantity(temp, 'K')
        
        logger.info(f"Read dust temperature from {fpath}: shape={temp.shape}, "
                   f"T_min={np.min(temp):.2f} K, T_max={np.max(temp):.2f} K")
        
        return self.dusttemp
    
    def readGasTemp(self, fname: Optional[str | Path] = None) -> Quantity:
        """Read gas temperature from RADMC-3D output or input.
        
        Parameters
        ----------
        fname : str or Path, optional
            Filename to read (default: auto-detect gas_temperature)
            
        Returns
        -------
        Quantity
            Gas temperature in Kelvin
            
        Raises
        ------
        FileNotFoundError
            If no gas temperature file is found
        """
        if fname is None:
            fpath = self._findDataFile('gas_temperature')
            if fpath is None:
                raise FileNotFoundError("No gas_temperature file found")
        else:
            fpath = Path(fname)
            if not fpath.exists():
                raise FileNotFoundError(f"File not found: {fpath}")
        
        if self._isBinary(fpath):
            data = self._readScalarFieldBinary(fpath)
        else:
            data = self._readScalarFieldASCII(fpath)
        
        # Reshape to mesh
        nx, ny, nz = self._getMeshShape()
        temp = data.reshape((nx, ny, nz), order='F')  # Fortran order
        
        self.gastemp = Quantity(temp, 'K')
        
        logger.info(f"Read gas temperature from {fpath}: shape={temp.shape}, "
                   f"T_min={np.min(temp):.2f} K, T_max={np.max(temp):.2f} K")
        
        return self.gastemp
    
    def readDustDens(self, fname: Optional[str | Path] = None, ispec: int = 0) -> Quantity:
        """Read dust density from RADMC-3D input file.
        
        Parameters
        ----------
        fname : str or Path, optional
            Filename to read (default: auto-detect dust_density)
        ispec : int, optional
            Dust species index (default: 0)
            
        Returns
        -------
        Quantity
            Dust density in g/cm^3
            
        Raises
        ------
        FileNotFoundError
            If no dust density file is found
        """
        if fname is None:
            fpath = self._findDataFile('dust_density')
            if fpath is None:
                raise FileNotFoundError("No dust_density file found")
        else:
            fpath = Path(fname)
            if not fpath.exists():
                raise FileNotFoundError(f"File not found: {fpath}")
        
        if self._isBinary(fpath):
            data = self._readScalarFieldBinary(fpath)
        else:
            data = self._readScalarFieldASCII(fpath)
        
        # Extract the requested species
        if len(data.shape) == 2:
            # Multiple species: (nspec, ncells)
            dens = data[ispec, :]
        else:
            # Single species
            dens = data
        
        # Reshape to mesh
        nx, ny, nz = self._getMeshShape()
        dens = dens.reshape((nx, ny, nz), order='F')  # Fortran order
        
        self.rhodust = Quantity(dens, 'g/cm**3')
        
        logger.info(f"Read dust density from {fpath}: shape={dens.shape}")
        
        return self.rhodust
    
    def readGasVel(self, fname: Optional[str | Path] = None) -> Quantity:
        """Read gas velocity from RADMC-3D input file.
        
        Parameters
        ----------
        fname : str or Path, optional
            Filename to read (default: auto-detect gas_velocity)
            
        Returns
        -------
        Quantity
            Gas velocity components in cm/s, shape (nx, ny, nz, 3)
            
        Raises
        ------
        FileNotFoundError
            If no gas velocity file is found
        """
        if fname is None:
            fpath = self._findDataFile('gas_velocity')
            if fpath is None:
                raise FileNotFoundError("No gas_velocity file found")
        else:
            fpath = Path(fname)
            if not fpath.exists():
                raise FileNotFoundError(f"File not found: {fpath}")
        
        if self._isBinary(fpath):
            data = self._readVectorFieldBinary(fpath)
        else:
            data = self._readVectorFieldASCII(fpath)
        
        # Reshape to mesh
        nx, ny, nz = self._getMeshShape()
        vel = data.reshape((nx, ny, nz, 3), order='F')  # Fortran order
        
        self.gasvel = Quantity(vel, 'cm/s')
        
        logger.info(f"Read gas velocity from {fpath}: shape={vel.shape}")
        
        return self.gasvel
    
    def readVTurb(self, fname: Optional[str | Path] = None) -> Quantity:
        """Read turbulent velocity from RADMC-3D input file.
        
        Parameters
        ----------
        fname : str or Path, optional
            Filename to read (default: auto-detect microturbulence)
            
        Returns
        -------
        Quantity
            Turbulent velocity in cm/s
            
        Raises
        ------
        FileNotFoundError
            If no turbulent velocity file is found
        """
        if fname is None:
            fpath = self._findDataFile('microturbulence')
            if fpath is None:
                raise FileNotFoundError("No microturbulence file found")
        else:
            fpath = Path(fname)
            if not fpath.exists():
                raise FileNotFoundError(f"File not found: {fpath}")
        
        if self._isBinary(fpath):
            data = self._readScalarFieldBinary(fpath)
        else:
            data = self._readScalarFieldASCII(fpath)
        
        # Reshape to mesh
        nx, ny, nz = self._getMeshShape()
        vturb = data.reshape((nx, ny, nz), order='F')  # Fortran order
        
        self.vturb = Quantity(vturb, 'cm/s')
        
        logger.info(f"Read turbulent velocity from {fpath}: shape={vturb.shape}")
        
        return self.vturb
    
    def readGasDens(self, fname: Optional[str | Path] = None, ispec: str = '') -> Quantity:
        """Read gas/molecular number density from RADMC-3D input file.
        
        Parameters
        ----------
        fname : str or Path, optional
            Filename to read (default: auto-detect numberdens_<ispec>)
        ispec : str, optional
            Molecule name for filename (default: '')
            
        Returns
        -------
        Quantity
            Molecular number density in molecules/cm^3
            
        Raises
        ------
        FileNotFoundError
            If no number density file is found
        """
        if fname is None:
            basename = f'numberdens_{ispec}' if ispec else 'numberdens'
            fpath = self._findDataFile(basename)
            if fpath is None:
                raise FileNotFoundError(f"No {basename} file found")
        else:
            fpath = Path(fname)
            if not fpath.exists():
                raise FileNotFoundError(f"File not found: {fpath}")
        
        if self._isBinary(fpath):
            data = self._readScalarFieldBinary(fpath)
        else:
            data = self._readScalarFieldASCII(fpath)
        
        # Reshape to mesh
        nx, ny, nz = self._getMeshShape()
        ndens = data.reshape((nx, ny, nz), order='F')  # Fortran order
        
        self.ndens_mol = Quantity(ndens, '1/cm**3')
        
        logger.info(f"Read molecular number density from {fpath}: shape={ndens.shape}")
        
        return self.ndens_mol
    
    def readMeanIntensity(self, fname: Optional[str | Path] = None) -> Quantity:
        """Read mean intensity from RADMC-3D output.
        
        Parameters
        ----------
        fname : str or Path, optional
            Filename to read (default: auto-detect mean_intensity)
            
        Returns
        -------
        Quantity
            Mean intensity in erg/s/cm^2/Hz/sr
            
        Raises
        ------
        FileNotFoundError
            If no mean intensity file is found
        """
        if fname is None:
            fpath = self._findDataFile('mean_intensity')
            if fpath is None:
                raise FileNotFoundError("No mean_intensity file found")
        else:
            fpath = Path(fname)
            if not fpath.exists():
                raise FileNotFoundError(f"File not found: {fpath}")
        
        if self._isBinary(fpath):
            data = self._readScalarFieldBinary(fpath)
        else:
            data = self._readScalarFieldASCII(fpath)
        
        # Reshape to mesh
        nx, ny, nz = self._getMeshShape()
        intensity = data.reshape((nx, ny, nz), order='F')  # Fortran order
        
        self.mean_intensity = Quantity(intensity, 'erg/(s*cm**2*Hz*sr)')
        
        logger.info(f"Read mean intensity from {fpath}: shape={intensity.shape}")
        
        return self.mean_intensity
    
    def _readScalarFieldASCII(self, fname: Path) -> np.ndarray:
        """Read a scalar field from ASCII file.
        
        Parameters
        ----------
        fname : Path
            File to read
            
        Returns
        -------
        np.ndarray
            Data array
        """
        with open(fname, 'r') as f:
            # Read header
            iformat = int(f.readline().strip())
            
            # Read number of cells
            ncells = int(f.readline().strip())
            
            # Check for number of species (dust only)
            line = f.readline().strip()
            try:
                nspec = int(line)
                # Multiple species
                data = np.loadtxt(f)
                if data.size == ncells * nspec:
                    data = data.reshape((nspec, ncells))
            except ValueError:
                # Single species, line already contains data
                first_val = float(line)
                rest = np.loadtxt(f)
                data = np.concatenate([[first_val], rest])
        
        return data
    
    def _readScalarFieldBinary(self, fname: Path) -> np.ndarray:
        """Read a scalar field from binary file.
        
        Parameters
        ----------
        fname : Path
            File to read
            
        Returns
        -------
        np.ndarray
            Data array
        """
        with open(fname, 'rb') as f:
            # Read header (format, precision, ncells, [nspec])
            hdr3 = np.fromfile(f, dtype=np.int64, count=3)
            if hdr3.size < 3:
                raise ValueError(f"Binary scalar field {fname} has incomplete header")
            iformat = hdr3[0]
            prec = hdr3[1]  # 8 for double, 4 for single
            ncells = hdr3[2]
            
            total_bytes = fname.stat().st_size
            header_bytes_min = 3 * 8
            remaining_bytes = total_bytes - header_bytes_min
            nspec = 1
            item_bytes = 8 if prec == 8 else 4

            if remaining_bytes > item_bytes * ncells:
                extra = np.fromfile(f, dtype=np.int64, count=1)
                if extra.size == 1:
                    candidate_nspec = int(extra[0])
                    remaining_after_nspec = remaining_bytes - 8
                    if candidate_nspec > 0 and remaining_after_nspec == item_bytes * ncells * candidate_nspec:
                        nspec = candidate_nspec
                    else:
                        f.seek(-8, 1)
            
            dtype = np.float64 if prec == 8 else np.float32
            data = np.fromfile(f, dtype=dtype)
            
            if nspec > 1:
                if data.size != ncells * nspec:
                    raise ValueError(f"Binary scalar field {fname} has size {data.size}, expected {ncells * nspec}")
                data = data.reshape((nspec, ncells))
        
        return data
    
    def _readVectorFieldASCII(self, fname: Path) -> np.ndarray:
        """Read a vector field from ASCII file.
        
        Parameters
        ----------
        fname : Path
            File to read
            
        Returns
        -------
        np.ndarray
            Data array with shape (ncells * 3,)
        """
        with open(fname, 'r') as f:
            # Read header
            iformat = int(f.readline().strip())
            ncells = int(f.readline().strip())
            
            # Read velocity data (3 components per cell)
            data = np.loadtxt(f)
            
        return data.flatten()  # Returns (ncells * 3,) array
    
    def _readVectorFieldBinary(self, fname: Path) -> np.ndarray:
        """Read a vector field from binary file.
        
        Parameters
        ----------
        fname : Path
            File to read
            
        Returns
        -------
        np.ndarray
            Data array with shape (ncells * 3,)
        """
        with open(fname, 'rb') as f:
            # Read header
            hdr = np.fromfile(f, dtype=np.int64, count=3)
            iformat = hdr[0]
            prec = hdr[1]
            ncells = hdr[2]
            
            # Read data
            dtype = np.float64 if prec == 8 else np.float32
            data = np.fromfile(f, dtype=dtype, count=ncells * 3)
        
        return data
