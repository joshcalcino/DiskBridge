"""RADMC-3D data reader class.

This module provides the radmc3dData class for reading RADMC-3D output files,
handling both ASCII and binary formats. It mirrors the structure of radmc3dPy.data
but adapted for DiskBridge's Model class.

The class is separate from the writer functionality, providing clean separation
between input (writer) and output (data reader) operations.
"""

# db-keywords: gas-temperature, units, radmc3d, model, mesh, field, io, files
# db-role: helper
# db-scope: package
# db-purpose: RADMC-3D data reader class.

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from diskbridge.model.core import Model

from diskbridge._logging import logger
from diskbridge._units import Quantity, units


def combine_mean_intensity_files(
    first: str | Path,
    second: str | Path,
    output: str | Path,
    *,
    first_weight: float = 0.5,
    chunk_values: int = 1_000_000,
) -> dict[str, int | float]:
    """Write the weighted mean of two binary RADMC-3D intensity files.

    The payload is combined in chunks so a full production radiation field is
    never duplicated in memory.

    Parameters
    ----------
    first, second : str or pathlib.Path
        Input ``mean_intensity.bout`` files with identical headers and
        frequency grids.
    output : str or pathlib.Path
        Destination ``mean_intensity.bout`` file.
    first_weight : float, optional
        Weight assigned to ``first``. The second weight is its complement.
    chunk_values : int, optional
        Maximum number of intensity values combined at once.

    Returns
    -------
    dict
        Binary format, precision, cell count, wavelength count, and weights.
    """
    first = Path(first)
    second = Path(second)
    output = Path(output)
    if output == first or output == second:
        raise ValueError("output must differ from both mean-intensity inputs")
    temporary_output = output.with_name(output.name + ".tmp")
    weight1 = float(first_weight)
    weight2 = 1.0 - weight1
    if not 0.0 <= weight1 <= 1.0:
        raise ValueError("first_weight must be in [0, 1]")
    if int(chunk_values) < 1:
        raise ValueError("chunk_values must be positive")

    with first.open("rb") as f1, second.open("rb") as f2:
        header1 = np.fromfile(f1, dtype=np.int64, count=4)
        header2 = np.fromfile(f2, dtype=np.int64, count=4)
        if header1.size != 4 or header2.size != 4:
            raise ValueError("Mean-intensity input has an incomplete binary header")
        if not np.array_equal(header1, header2):
            raise ValueError("Mean-intensity binary headers do not match")
        if int(header1[0]) != 2:
            raise ValueError(f"Unsupported mean-intensity format {int(header1[0])}")

        precision = int(header1[1])
        dtype = dtype_from_radmc_precision(first, precision)
        ncells = int(header1[2])
        nwav = int(header1[3])
        frequencies1 = np.fromfile(f1, dtype=np.float64, count=nwav)
        frequencies2 = np.fromfile(f2, dtype=np.float64, count=nwav)
        if frequencies1.size != nwav or frequencies2.size != nwav:
            raise ValueError("Mean-intensity input has an incomplete frequency grid")
        if not np.array_equal(frequencies1, frequencies2):
            raise ValueError("Mean-intensity frequency grids do not match")

        output.parent.mkdir(parents=True, exist_ok=True)
        remaining = int(ncells * nwav)
        with temporary_output.open("wb") as fout:
            header1.tofile(fout)
            frequencies1.tofile(fout)
            while remaining:
                count = min(remaining, int(chunk_values))
                values1 = np.fromfile(f1, dtype=dtype, count=count)
                values2 = np.fromfile(f2, dtype=dtype, count=count)
                if values1.size != count or values2.size != count:
                    raise ValueError("Mean-intensity input has an incomplete data payload")
                combined = weight1 * values1.astype(np.float64) + weight2 * values2.astype(
                    np.float64
                )
                if not np.all(np.isfinite(combined)):
                    raise ValueError("Combined mean intensity contains non-finite values")
                combined.astype(dtype).tofile(fout)
                remaining -= count

        if f1.read(1) or f2.read(1):
            raise ValueError("Mean-intensity input has trailing payload data")
    temporary_output.replace(output)

    return {
        "format": int(header1[0]),
        "precision": precision,
        "ncells": ncells,
        "nwavelengths": nwav,
        "first_weight": weight1,
        "second_weight": weight2,
    }


def radial_shell_mean_intensity(
    path: str | Path,
    volumes: np.ndarray,
    *,
    wavelength_chunk: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Read volume-weighted radial spectra from a binary intensity file.

    Parameters
    ----------
    path : str or pathlib.Path
        Binary ``mean_intensity.bout`` file.
    volumes : ndarray
        Spherical cell volumes with shape ``(nr, ntheta, nphi)``.
    wavelength_chunk : int, optional
        Number of wavelengths reduced per memory-mapped chunk.

    Returns
    -------
    frequencies_hz, shell_mean : tuple of ndarray
        Frequency grid and radial shell spectra with shape ``(nr, nwav)``.
    """
    path = Path(path)
    volumes = np.asarray(volumes, dtype=np.float64)
    if volumes.ndim != 3:
        raise ValueError("volumes must have shape (nr, ntheta, nphi)")
    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int64, count=4)
        if header.size != 4:
            raise ValueError(f"{path} has an incomplete binary header")
        if int(header[0]) != 2:
            raise ValueError(f"{path} has unsupported format {int(header[0])}")
        dtype = dtype_from_radmc_precision(path, int(header[1]))
        ncells = int(header[2])
        nwav = int(header[3])
        if ncells != int(volumes.size):
            raise ValueError(
                f"{path} contains {ncells} cells but volumes contain {volumes.size}"
            )
        frequencies = np.fromfile(f, dtype=np.float64, count=nwav)
        if frequencies.size != nwav:
            raise ValueError(f"{path} has an incomplete frequency grid")
    offset = 4 * np.dtype(np.int64).itemsize + nwav * np.dtype(np.float64).itemsize
    payload = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(nwav, ncells))
    radial_volume = np.sum(volumes, axis=(1, 2))
    if np.any(radial_volume <= 0.0):
        raise ValueError("Radial shell volumes must be positive")
    result = np.empty((volumes.shape[0], nwav), dtype=np.float64)
    for start in range(0, nwav, int(wavelength_chunk)):
        stop = min(start + int(wavelength_chunk), nwav)
        block = np.asarray(payload[start:stop], dtype=np.float64)
        spatial = block.T.reshape(volumes.shape + (stop - start,), order="F")
        result[:, start:stop] = np.sum(spatial * volumes[..., None], axis=(1, 2)) / radial_volume[:, None]
    return frequencies, result


def read_amr_grid_cell_count(path: str | Path) -> int:
    """Return the regular-grid cell count declared by ``amr_grid.inp``."""

    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 6:
        raise ValueError(f"{path} is too short to contain RADMC-3D grid dimensions")
    dims = lines[5].split()
    if len(dims) < 3:
        raise ValueError(f"{path} does not contain three RADMC-3D grid dimensions")
    try:
        nx, ny, nz = (int(float(value)) for value in dims[:3])
    except ValueError as exc:
        raise ValueError(f"{path} has invalid RADMC-3D grid dimensions: {lines[5]!r}") from exc
    return int(nx * ny * nz)


def dtype_from_radmc_precision(path: str | Path, precision: int) -> np.dtype:
    """Return the NumPy dtype for a RADMC-3D binary precision code."""

    path = Path(path)
    if precision == 8:
        return np.dtype(np.float64)
    if precision == 4:
        return np.dtype(np.float32)
    raise ValueError(f"{path.name} has unsupported precision {precision}; expected 4 or 8")


def read_radmc_binp_field(path: str | Path, *, components: int) -> tuple[np.ndarray, int]:
    """Read a scalar/vector RADMC-3D ``.binp`` field as flat float64 values."""

    path = Path(path)
    components = int(components)
    if components <= 0:
        raise ValueError("components must be positive")
    if path.suffix != ".binp":
        raise ValueError(f"{path.name} must be a .binp RADMC field file")

    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int64, count=3)
        if header.size < 3:
            raise ValueError(f"{path.name} has an incomplete binary header")
        iformat = int(header[0])
        precision = int(header[1])
        ncells = int(header[2])
        if iformat != 1:
            raise ValueError(f"{path.name} has unsupported format {iformat}; expected 1")
        dtype = dtype_from_radmc_precision(path, precision)
        expected = ncells * components
        values = np.fromfile(f, dtype=dtype, count=expected).astype(
            np.float64,
            copy=False,
        )
        if values.size != expected:
            raise ValueError(
                f"{path.name} has {values.size} values, expected {expected} "
                f"({ncells} cells x {components} components)"
            )
    return values, ncells


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
            nr = len(mesh.axes['r'].centers)
            ntheta = len(mesh.axes['theta'].centers)
            nphi = len(mesh.axes['phi'].centers)
            return (nr, ntheta, nphi)
        elif mesh.coord_system == 'cylindrical':
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

    def _expected_ncells(self) -> int:
        nx, ny, nz = self._getMeshShape()
        return int(nx * ny * nz)

    def _validate_cell_count(self, fpath: Path, actual_ncells: int) -> None:
        """Refuse to read RADMC products from a different grid."""

        actual_ncells = int(actual_ncells)
        expected = self._expected_ncells()
        if actual_ncells != expected:
            raise ValueError(
                f"RADMC-3D grid mismatch for {fpath}: file has {actual_ncells} cells, "
                f"model mesh expects {expected}. Regenerate stale cached products."
            )

        amr = self.model_dir / "radmc3d_inputs" / "amr_grid.inp"
        amr_ncells = read_amr_grid_cell_count(amr) if amr.exists() else expected
        if amr_ncells != expected:
            raise ValueError(
                f"RADMC-3D grid mismatch: model mesh expects {expected} cells, "
                f"but {amr} expects {amr_ncells}."
            )

    def _resolve_data_file(
        self,
        fname: Optional[str | Path],
        basename: str,
        missing_message: str,
    ) -> Path:
        if fname is None:
            fpath = self._findDataFile(basename)
            if fpath is None:
                raise FileNotFoundError(missing_message)
            return fpath

        fpath = Path(fname)
        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")
        return fpath

    def _read_scalar_data(self, fpath: Path) -> np.ndarray:
        return self._readScalarFieldBinary(fpath)

    def _read_vector_data(self, fpath: Path) -> np.ndarray:
        return self._readVectorFieldBinary(fpath)

    def read_binary_field(self, fname: str | Path, *, components: int) -> tuple[np.ndarray, int]:
        """Read a RADMC-3D ``.binp`` field without reshaping it."""

        return read_radmc_binp_field(fname, components=components)

    def _reshape_scalar_to_mesh(self, data: np.ndarray) -> np.ndarray:
        nx, ny, nz = self._getMeshShape()
        return data.reshape((nx, ny, nz), order='F')

    def _reshape_vector_to_mesh(self, data: np.ndarray) -> np.ndarray:
        nx, ny, nz = self._getMeshShape()
        # RADMC-3D vector binp files are cell-major triples:
        #   (v1_cell0, v2_cell0, v3_cell0), (v1_cell1, ...)
        # The component axis is not part of the Fortran cell ordering, so each
        # component must be reshaped back to the mesh separately.
        values = np.asarray(data)
        ncells = nx * ny * nz
        if values.size != 3 * ncells:
            raise ValueError(
                f"Vector field has {values.size} values, expected {3 * ncells} "
                f"({ncells} cells x 3 components)"
            )
        by_cell = values.reshape((ncells, 3), order='C')
        return np.stack(
            [by_cell[:, i].reshape((nx, ny, nz), order='F') for i in range(3)],
            axis=-1,
        )

    def _select_dust_species(self, data: np.ndarray, ispec: int, field_name: str) -> np.ndarray:
        if data.ndim == 2:
            if not isinstance(ispec, (int, np.integer)):
                ispec = int(ispec)
            return data[ispec, :]
        if data.ndim == 1:
            return data
        raise ValueError(f"Unexpected {field_name} data shape: {data.shape}")
    
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
        fpath = self._resolve_data_file(
            fname=fname,
            basename='dust_temperature',
            missing_message="No dust_temperature file found",
        )

        data = self._read_scalar_data(fpath)
        
        # Extract the requested species
        logger.debug(f"Dust temperature data shape: {data.shape}, dtype: {data.dtype}, ispec: {ispec}, type(ispec): {type(ispec)}")
        
        temp_1d = self._select_dust_species(data, ispec=ispec, field_name='dust temperature')
        temp = self._reshape_scalar_to_mesh(temp_1d)
        
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
        fpath = self._resolve_data_file(
            fname=fname,
            basename='gas_temperature',
            missing_message="No gas_temperature file found",
        )

        data = self._read_scalar_data(fpath)
        temp = self._reshape_scalar_to_mesh(data)
        
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
        fpath = self._resolve_data_file(
            fname=fname,
            basename='dust_density',
            missing_message="No dust_density file found",
        )

        data = self._read_scalar_data(fpath)
        dens_1d = self._select_dust_species(data, ispec=ispec, field_name='dust density')
        dens = self._reshape_scalar_to_mesh(dens_1d)
        
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
        fpath = self._resolve_data_file(
            fname=fname,
            basename='gas_velocity',
            missing_message="No gas_velocity file found",
        )

        data = self._read_vector_data(fpath)
        vel = self._reshape_vector_to_mesh(data)
        
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
        fpath = self._resolve_data_file(
            fname=fname,
            basename='microturbulence',
            missing_message="No microturbulence file found",
        )

        data = self._read_scalar_data(fpath)
        vturb = self._reshape_scalar_to_mesh(data)
        
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
        basename = f'numberdens_{ispec}' if ispec else 'numberdens'
        fpath = self._resolve_data_file(
            fname=fname,
            basename=basename,
            missing_message=f"No {basename} file found",
        )

        data = self._read_scalar_data(fpath)
        ndens = self._reshape_scalar_to_mesh(data)
        
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
        fpath = self._resolve_data_file(
            fname=fname,
            basename='mean_intensity',
            missing_message="No mean_intensity file found",
        )

        data = self._read_scalar_data(fpath)
        intensity = self._reshape_scalar_to_mesh(data)
        
        self.mean_intensity = Quantity(intensity, 'erg/(s*cm**2*Hz*sr)')
        
        logger.info(f"Read mean intensity from {fpath}: shape={intensity.shape}")
        
        return self.mean_intensity

    @staticmethod
    def _read_n_floats_from_text(f, n: int) -> np.ndarray:
        vals: list[float] = []
        while len(vals) < n:
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if not parts:
                continue
            for p in parts:
                vals.append(float(p))
                if len(vals) >= n:
                    break
        if len(vals) != n:
            raise ValueError(f"Expected {n} floats, got {len(vals)}")
        return np.asarray(vals, dtype=float)

    def read_mean_intensity_file(self, path: str | Path) -> tuple[Quantity, Quantity]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mean intensity file not found: {path}")

        if path.suffix == '.bout':
            with open(path, 'rb') as f:
                hdr4 = np.fromfile(f, dtype=np.int64, count=4)
                if hdr4.size < 4:
                    raise ValueError(f"{path} has incomplete header")
                prec = int(hdr4[1])
                nrcells = int(hdr4[2])
                nwav = int(hdr4[3])
                self._validate_cell_count(path, nrcells)
                freq_hz = np.fromfile(f, dtype=np.float64, count=nwav) * units('Hz')
                dtype = np.float64 if prec == 8 else np.float32
                j_flat = np.fromfile(f, dtype=dtype, count=nwav * nrcells).astype(np.float64, copy=False)
            j_lambda = j_flat.reshape((nwav, nrcells)).T * units('erg/(s*cm^2*Hz*sr)')
            return freq_hz, j_lambda

        with open(path, 'r') as f:
            _ = int(f.readline().strip())
            nrcells = int(f.readline().strip())
            nwav = int(f.readline().strip())
            self._validate_cell_count(path, nrcells)
            freq_vals = self._read_n_floats_from_text(f, nwav)
            freq_hz = freq_vals * units('Hz')
            j_vals = self._read_n_floats_from_text(f, nwav * nrcells)
            j_lambda = j_vals.reshape((nwav, nrcells)).T * units('erg/(s*cm^2*Hz*sr)')
            return freq_hz, j_lambda

    def _findDataFile(self, basename: str) -> Optional[Path]:
        """Find a data file with strict binary-only extension mapping.

        Parameters
        ----------
        basename : str
            Base filename without extension

        Returns
        -------
        Path or None
            Path to found file, or None if not found

        Raises
        ------
        RuntimeError
            If basename is not in the allowed list or if non-binary file is found
        """
        # Strict mapping: basename → allowed binary extensions only
        _ALLOWED_FORMATS = {
            'dust_temperature': ['.bdat'],
            'mean_intensity':   ['.bout'],
            'image':            ['.bout'],
            'dust_density':     ['.binp'],
            'gas_velocity':     ['.binp'],
            'microturbulence':  ['.binp'],
            'gas_temperature':  ['.binp'],
        }

        # Extract base name (e.g., 'numberdens_co' -> 'numberdens')
        base_pattern = basename.split('_')[0] if '_' in basename else basename

        # Handle numberdens_* pattern
        if base_pattern == 'numberdens':
            allowed_exts = ['.binp']
        elif basename in _ALLOWED_FORMATS:
            allowed_exts = _ALLOWED_FORMATS[basename]
        else:
            # Unknown basename - raise error
            raise RuntimeError(
                f"DiskBridge is configured for binary RADMC-3D files only. "
                f"Unknown or unsupported file basename: '{basename}'"
            )

        found_files = []
        for ext in allowed_exts:
            fpath = self.model_dir / (basename + ext)
            if fpath.exists():
                found_files.append(fpath)

        if len(found_files) > 1:
            logger.warning(f"Multiple binary files for basename '{basename}' found. Using {found_files[0]}")

        return found_files[0] if found_files else None

    def _readScalarFieldBinary(self, fname: Path) -> np.ndarray:
        """Read a scalar field from binary file.
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
            if iformat != 1:
                raise ValueError(
                    f"Binary scalar field {fname} has unsupported format {iformat}; expected 1"
                )
            self._validate_cell_count(fname, int(ncells))
            
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
                    raise ValueError(
                        f"Binary scalar field {fname} has size {data.size}, "
                        f"expected {ncells * nspec}"
                    )
                data = data.reshape((nspec, ncells))
        
        return data
    
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
        data, ncells = read_radmc_binp_field(fname, components=3)
        self._validate_cell_count(fname, ncells)
        return data
