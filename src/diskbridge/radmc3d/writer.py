"""RADMC-3D input file writer for DiskBridge Models.

This module provides the RADMC3DWriter class that takes a Model instance
and writes the necessary RADMC-3D input files, handling unit conversion
from Pint Quantities to CGS units.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Tuple
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from diskbridge.model.model import Model

from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from .opacities import DustOpacityCalculator


class RADMC3DWriter:
    """Write RADMC-3D input files from a DiskBridge Model.
    
    This class handles conversion from Pint Quantities to CGS units
    and writes all necessary RADMC-3D input files.
    
    Example:
        model = Model()
        model.load_model('path/to/data', reader='fargo')
        
        writer = RADMC3DWriter(model)
        writer.write_amr_grid(output_dir='.')
        writer.write_dust_density(output_dir='.')
        writer.write_wavelength_grid(output_dir='.')
    """
    
    def __init__(self, model: 'Model'):
        """Initialize the writer with a Model instance.
        
        Args:
            model: DiskBridge Model instance containing the hydro data
        """
        self.model = model
        self.opacity_calculator = DustOpacityCalculator()
        
    def _to_cgs(self, quantity: Quantity) -> np.ndarray:
        """Convert Pint Quantity to CGS magnitude array.
        
        Args:
            quantity: Pint Quantity to convert
            
        Returns:
            NumPy array of CGS values
        """
        return quantity.to_base_units().magnitude
    
    def write_amr_grid(
        self,
        output_dir: str | Path = '.',
        coordsys_code: int = 101,
    ) -> None:
        """Write amr_grid.inp file.
        
        Args:
            output_dir: Directory to write the file
            coordsys_code: RADMC-3D coordinate system code
                - 0-99: Cartesian
                - 100-199: Spherical
                - 200-299: Cylindrical
                Default: 101 (spherical)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'amr_grid.inp'
        
        mesh = self.model.mesh
        if mesh is None:
            raise ValueError("Model has no mesh defined")
        
        # Determine coordinate system
        if mesh.coord_system == 'spherical':
            if coordsys_code < 100 or coordsys_code >= 200:
                coordsys_code = 101
            axis_names = ('r', 'theta', 'phi')
        elif mesh.coord_system == 'cylindrical':
            if coordsys_code < 200 or coordsys_code >= 300:
                coordsys_code = 201
            axis_names = ('r', 'phi', 'z')
        elif mesh.coord_system == 'cartesian':
            if coordsys_code >= 100:
                coordsys_code = 1
            axis_names = ('x', 'y', 'z')
        else:
            raise ValueError(f"Unsupported coordinate system: {mesh.coord_system}")
        
        # Get grid dimensions
        dims = []
        active_dims = [0, 0, 0]
        
        for i, name in enumerate(axis_names):
            ncell = mesh.ncell(name)
            if ncell is not None and ncell > 0:
                dims.append(ncell)
                active_dims[i] = 1
            else:
                dims.append(1)  # Inactive dimension has size 1
        
        with open(filepath, 'w') as f:
            f.write('1\n')  # iformat
            f.write('0\n')  # AMR grid style (0=regular)
            f.write(f'{coordsys_code}\n')  # Coordinate system
            f.write('0\n')  # gridinfo
            f.write(f'{active_dims[0]} {active_dims[1]} {active_dims[2]}\n')
            f.write(f'{dims[0]} {dims[1]} {dims[2]}\n')
            
            # Write grid edges for each dimension (in cm)
            for name in axis_names:
                edges = mesh.edges(name)
                if edges is not None:
                    edges_cgs = self._to_cgs(edges)
                    for val in edges_cgs:
                        f.write(f'{val:13.6e} ')
                else:
                    # Write dummy edges for inactive dimension
                    f.write('0.0 1.0 ')
                f.write('\n')
        
        logger.info(f"Wrote AMR grid file: {filepath}")
    
    def write_wavelength_grid(
        self,
        output_dir: str | Path = '.',
        wmin_micron: float = 0.1,
        wmax_micron: float = 10000.0,
        nwav: int = 150,
    ) -> None:
        """Write wavelength_micron.inp file.
        
        Args:
            output_dir: Directory to write the file
            wmin_micron: Minimum wavelength in microns
            wmax_micron: Maximum wavelength in microns
            nwav: Number of wavelength points
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'wavelength_micron.inp'
        
        # Create logarithmically spaced wavelength grid
        Pw = (wmax_micron / wmin_micron) ** (1.0 / (nwav - 1))
        waves = np.zeros(nwav)
        waves[0] = wmin_micron
        for i in range(1, nwav):
            waves[i] = wmin_micron * Pw ** i
        
        with open(filepath, 'w') as f:
            f.write(f'{nwav}\n')
            for w in waves:
                f.write(f'{w:13.6e}\n')
        
        logger.info(f"Wrote wavelength grid file: {filepath}")
    
    def write_stars(
        self,
        output_dir: str | Path = '.',
        rstar: Optional[float] = 2.0,
        tstar: Optional[float] = 4000.0,
        mstar: Optional[float] = 1.0,
        position: Tuple[float, float, float] = (0., 0., 0.),
    ) -> None:
        """Write stars.inp file for stellar radiation source.
        
        Args:
            output_dir: Directory to write the file
            rstar: Stellar radius in solar radii (default: 2.0)
            tstar: Stellar effective temperature in K (default: 4000.0)
            mstar: Stellar mass in solar masses (default: 1.0)
            position: (x, y, z) position in AU (default: origin)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'stars.inp'
        
        # Convert to CGS using Pint constants
        rstar_cgs = (rstar * units('solar_radius')).to_base_units().magnitude  # Solar radii to cm
        mstar_cgs = (mstar * units('solar_mass')).to_base_units().magnitude  # Solar masses to g
        pos_cgs = [(p * units('astronomical_unit')).to_base_units().magnitude for p in position]  # AU to cm
        
        with open(filepath, 'w') as f:
            f.write('2\n')  # Format number
            f.write('1 1\n')  # 1 star, take spectral template from next
            f.write(f'{rstar_cgs:13.6e} {mstar_cgs:13.6e} ')
            f.write(f'{pos_cgs[0]:13.6e} {pos_cgs[1]:13.6e} {pos_cgs[2]:13.6e}\n')
            f.write(f'{tstar:13.6e}\n')
        
        logger.info(f"Wrote stars file: {filepath}")
    
    def write_radmc3d_inp(
        self,
        output_dir: str | Path = '.',
        incl_dust: int = 1,
        incl_lines: int = 0,
        nphot: int = 1000000,
        nphot_scat: int = 1000000,
        scattering_mode_max: int = 0,
        modified_random_walk: int = 1,
        setthreads: int = 1,
    ) -> None:
        """Write radmc3d.inp control file.
        
        Args:
            output_dir: Directory to write the file
            incl_dust: Include dust (1=yes, 0=no)
            incl_lines: Include lines (1=yes, 0=no)
            nphot: Number of photons for thermal Monte Carlo
            nphot_scat: Number of photons for scattering Monte Carlo
            scattering_mode_max: Scattering mode (0=no scat, 1=isotropic, etc.)
            modified_random_walk: Use modified random walk (1=yes, 0=no)
            setthreads: Number of OpenMP threads
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'radmc3d.inp'
        
        with open(filepath, 'w') as f:
            f.write(f'incl_dust = {incl_dust}\n')
            f.write(f'incl_lines = {incl_lines}\n')
            f.write(f'nphot = {nphot}\n')
            f.write(f'nphot_scat = {nphot_scat}\n')
            f.write(f'scattering_mode_max = {scattering_mode_max}\n')
            f.write(f'modified_random_walk = {modified_random_walk}\n')
            f.write(f'setthreads = {setthreads}\n')
        
        logger.info(f"Wrote radmc3d.inp control file: {filepath}")
    
    def write_dust_density(
        self,
        output_dir: str | Path = '.',
        binary: bool = True,
    ) -> None:
        """Write dust_density.inp or dust_density.binp file.
        
        Args:
            output_dir: Directory to write the file
            binary: Write binary format (True) or ASCII (False)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model.dust is None:
            raise ValueError("Model has no dust data")
        
        # Get number of dust bins
        nbin = self.model.dust.nbin
        if nbin == 0:
            raise ValueError("Model has no dust bins defined")
        
        # Get mesh shape
        mesh = self.model.mesh
        if mesh is None:
            raise ValueError("Model has no mesh defined")
        
        shape = mesh.shape
        ncells = int(np.prod(shape))
        
        if binary:
            filepath = output_dir / 'dust_density.binp'
            self._write_dust_density_binary(filepath, nbin, ncells)
        else:
            filepath = output_dir / 'dust_density.inp'
            self._write_dust_density_ascii(filepath, nbin, ncells)
        
        logger.info(f"Wrote dust density file: {filepath}")
    
    def _write_dust_density_binary(
        self,
        filepath: Path,
        nbin: int,
        ncells: int,
    ) -> None:
        """Write binary dust density file."""
        with open(filepath, 'wb') as f:
            # Header
            header = np.array([1, ncells, nbin], dtype=np.int64)
            header.tofile(f)
            
            # Write density for each bin
            for ibin in range(nbin):
                bin_data = self.model.dust.bins[ibin]
                rho = bin_data['density']  # Pint Quantity
                rho_cgs = self._to_cgs(rho).flatten()
                rho_cgs.astype(np.float64).tofile(f)
    
    def _write_dust_density_ascii(
        self,
        filepath: Path,
        nbin: int,
        ncells: int,
    ) -> None:
        """Write ASCII dust density file."""
        with open(filepath, 'w') as f:
            # Header
            f.write('1\n')  # Format number
            f.write(f'{ncells}\n')
            f.write(f'{nbin}\n')
            
            # Write density for each bin
            for ibin in range(nbin):
                bin_data = self.model.dust.bins[ibin]
                rho = bin_data['density']  # Pint Quantity
                rho_cgs = self._to_cgs(rho).flatten()
                
                for val in rho_cgs:
                    f.write(f'{val:13.6e}\n')
    
    def write_dustopac(
        self,
        output_dir: str | Path = '.',
        species_names: Optional[List[str]] = None,
        scattering_mode: int = 0,
    ) -> None:
        """Write dustopac.inp file.
        
        Args:
            output_dir: Directory to write the file
            species_names: List of species names (default: bin_0, bin_1, ...)
            scattering_mode: Scattering mode (0=no scat, >=3=full matrix)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / 'dustopac.inp'
        
        if self.model.dust is None:
            raise ValueError("Model has no dust data")
        
        nbin = self.model.dust.nbin
        if species_names is None:
            species_names = [f"bin_{i}" for i in range(nbin)]
        
        # Determine input style based on scattering mode
        inputstyle = 10 if scattering_mode >= 3 else 1
        
        with open(filepath, 'w') as f:
            f.write('2\n')  # Format number
            f.write(f'{nbin}\n')
            f.write('-' * 77 + '\n')
            
            for i, name in enumerate(species_names):
                f.write(f'{inputstyle}\n')
                f.write('0\n')  # 0=thermal grains
                f.write(f'{name}\n')
                f.write('-' * 77 + '\n')
        
        logger.info(f"Wrote dustopac.inp file: {filepath}")
    
    def compute_and_write_dust_opacities(
        self,
        output_dir: str | Path = '.',
        optconst_file: str | Path = None,
        grain_density: float = 2.7,
        wmin_micron: float = 0.1,
        wmax_micron: float = 10000.0,
        nwav: int = 200,
        ntheta: int = 181,
        scattering_mode: int = 0,
        logawidth: float = 0.05,
        na: int = 20,
    ) -> None:
        """Compute and write dust opacity files for all dust bins.
        
        Args:
            output_dir: Directory to write files
            optconst_file: Path to optical constants file (.lnk format)
            grain_density: Grain material density in g/cm^3
            wmin_micron: Minimum wavelength in microns
            wmax_micron: Maximum wavelength in microns
            nwav: Number of wavelength points
            ntheta: Number of scattering angles
            scattering_mode: Scattering mode (>=3 for full matrix)
            logawidth: Width parameter for size distribution smoothing
            na: Number of size samples for smoothing
        """
        if self.model.dust is None:
            raise ValueError("Model has no dust data")
        
        if optconst_file is None:
            raise ValueError("Must provide optical constants file")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create wavelength grid in cm
        Pw = (wmax_micron / wmin_micron) ** (1.0 / (nwav - 1))
        waves_micron = wmin_micron * Pw ** np.arange(nwav)
        waves_cm = waves_micron * 1e-4
        
        # Create angular grid if needed
        theta = np.linspace(0., 180., ntheta) if scattering_mode >= 3 else None
        
        # Compute opacity for each dust bin
        for ibin in range(self.model.dust.nbin):
            bin_data = self.model.dust.bins[ibin]
            
            # Get grain size in CGS
            grain_size_cgs = self._to_cgs(bin_data.size)
            
            logger.info(f"Computing opacity for bin {ibin}: "
                       f"size = {grain_size_cgs:.6e} cm")
            
            # Compute opacity
            opac = self.opacity_calculator.compute_opacity(
                optconst_file=optconst_file,
                grain_density=grain_density,
                grain_size=grain_size_cgs,
                wavelengths=waves_cm,
                theta=theta,
                logawidth=logawidth,
                na=na,
                extrapolate=True,
            )
            
            # Write to file
            output_path = output_dir / f"bin_{ibin}"
            self.opacity_calculator.write_radmc3d_opacity_file(
                opacity_data=opac,
                output_path=output_path,
                scattering_matrix=(scattering_mode >= 3),
            )
        
        logger.info(f"Wrote {self.model.dust.nbin} opacity files")
    
    def write_all_input_files(
        self,
        output_dir: str | Path = '.',
        optconst_file: Optional[str | Path] = None,
        grain_density: float = 2.7,
        rstar: float = 2.0,
        tstar: float = 4000.0,
        mstar: float = 1.0,
        scattering_mode: int = 0,
        nphot: int = 1000000,
        **kwargs,
    ) -> None:
        """Write all RADMC-3D input files in one call.
        
        Args:
            output_dir: Directory to write all files
            optconst_file: Path to optical constants file
            grain_density: Grain material density in g/cm^3
            rstar: Stellar radius in solar radii
            tstar: Stellar temperature in K
            mstar: Stellar mass in solar masses
            scattering_mode: Scattering mode (0=none, >=3=full matrix)
            nphot: Number of photons for Monte Carlo
            **kwargs: Additional arguments passed to individual writers
        """
        logger.info(f"Writing all RADMC-3D input files to {output_dir}")
        
        # Write grid files
        self.write_amr_grid(output_dir)
        self.write_wavelength_grid(output_dir)
        
        # Write stellar source
        self.write_stars(output_dir, rstar=rstar, tstar=tstar, mstar=mstar)
        
        # Write dust data
        if self.model.dust is not None and self.model.dust.nbin > 0:
            self.write_dust_density(output_dir)
            self.write_dustopac(output_dir, scattering_mode=scattering_mode)
            
            if optconst_file is not None:
                self.compute_and_write_dust_opacities(
                    output_dir,
                    optconst_file=optconst_file,
                    grain_density=grain_density,
                    scattering_mode=scattering_mode,
                    **kwargs,
                )
        
        # Write control file
        self.write_radmc3d_inp(
            output_dir,
            scattering_mode_max=scattering_mode,
            nphot=nphot,
            nphot_scat=nphot,
        )
        
        logger.info("All RADMC-3D input files written successfully")
