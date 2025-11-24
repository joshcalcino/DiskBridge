"""RADMC-3D input file writer for DiskBridge Models.

This module provides the RadWriter class that takes a Model instance
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
import diskbridge
from .opacities import DustOpacityCalculator

G_CGS = 6.67430e-8
SIGMA_SB = 5.670374419e-5


class RadWriter:
    """Write RADMC-3D input files from a DiskBridge Model.
    
    This class handles conversion from Pint Quantities to CGS units
    and writes all necessary RADMC-3D input files.
    
    Example:
        model = Model()
        model.load_model('path/to/data', reader='fargo')
        
        writer = RadWriter(model)
        writer.write_amr_grid(output_dir='.')
        writer.write_dust_density(output_dir='.')
        writer.write_wavelength_grid(output_dir='.')
    """
    
    def __init__(self, model: 'Model', organize_files: bool = True):
        """Initialize the writer with a Model instance.
        
        Args:
            model: DiskBridge Model instance containing the hydro data
            organize_files: If True, organize files into subdirectories (default: True)
        """
        self.model = model
        self.params = diskbridge.params  # Reference to global params (updated dynamically)
        self.opacity_calculator = DustOpacityCalculator()
        self.organize_files = organize_files
        
        # Define subdirectories for different file types
        self.grid_dir = 'input_grids'
        self.opacity_dir = 'input_opacities'
        self.star_dir = 'input_stars'
        self.dust_dir = 'input_dust'
        self.gas_dir = 'input_gas'
        self.config_dir = 'input_config'
        
        # Track written files for symlink management
        self.written_files = {}
        
    def _get_output_dir(self, base_dir: Path, file_type: str) -> Path:
        """Get output directory for a file type.
        
        Args:
            base_dir: Base output directory
            file_type: Type of file ('grid', 'opacity', 'star', 'dust', 'gas', 'config')
            
        Returns:
            Path to output directory
        """
        if not self.organize_files:
            return base_dir
        
        subdir_map = {
            'grid': self.grid_dir,
            'opacity': self.opacity_dir,
            'star': self.star_dir,
            'dust': self.dust_dir,
            'gas': self.gas_dir,
            'config': self.config_dir,
        }
        
        if file_type not in subdir_map:
            raise ValueError(f"Unknown file type: {file_type}")
        
        output_dir = base_dir / subdir_map[file_type]
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
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
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'grid')
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
            
            # Write grid edges for each dimension (in cm or radians)
            for name in axis_names:
                edges = mesh.edges(name)
                if edges is not None:
                    edges_cgs = self._to_cgs(edges)
                    
                    # RADMC-3D expects phi from 0 to 2π
                    # FARGO uses -π to +π, so we create a uniform grid for RADMC-3D
                    if name == 'phi':
                        nphi = len(edges_cgs)
                        edges_cgs = np.linspace(0.0, 2.0 * np.pi, nphi)
                    
                    for val in edges_cgs:
                        f.write(f'{val:13.6e} ')
                else:
                    # Write dummy edges for inactive dimension
                    f.write('0.0 1.0 ')
                f.write('\n')
        
        self.written_files['amr_grid.inp'] = filepath
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
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'grid')
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
        
        self.written_files['wavelength_micron.inp'] = filepath
        logger.info(f"Wrote wavelength grid file: {filepath}")
    
    def write_stars(
        self,
        output_dir: str | Path = '.',
        rstar: Optional[float] = 2.0,
        tstar: Optional[float] = 4000.0,
        mstar: Optional[float] = 1.0,
        position: Tuple[float, float, float] = (0., 0., 0.),
        wmin_micron: float = 0.1,
        wmax_micron: float = 10000.0,
        nwav: int = 150,
    ) -> None:
        """Write stars.inp file for stellar radiation source.
        
        This uses format 2 with wavelength-dependent spectrum (blackbody).
        
        Args:
            output_dir: Directory to write the file
            rstar: Stellar radius in solar radii (default: 2.0)
            tstar: Stellar effective temperature in K (default: 4000.0)
            mstar: Stellar mass in solar masses (default: 1.0)
            position: (x, y, z) position in AU (default: origin)
            wmin_micron: Minimum wavelength in microns
            wmax_micron: Maximum wavelength in microns
            nwav: Number of wavelength points
        """
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'star')
        filepath = output_dir / 'stars.inp'
        
        # Convert to CGS using Pint constants
        rstar_cgs = (rstar * units('solar_radius')).to_base_units().magnitude
        mstar_cgs = (mstar * units('solar_mass')).to_base_units().magnitude
        pos_cgs = [(p * units('astronomical_unit')).to_base_units().magnitude for p in position]

        # Build wavelength grid (matches wavelength_micron.inp)
        Pw = (wmax_micron / wmin_micron) ** (1.0 / (nwav - 1))
        waves_micron = wmin_micron * Pw ** np.arange(nwav)

        stars = []
        stars.append({'R_cm': rstar_cgs, 'M_g': mstar_cgs, 'T_K': float(tstar)})

        mdot_msun_per_yr = getattr(self.params, 'mdot', 0.0)
        if mdot_msun_per_yr > 0.0:
            f_fill = getattr(self.params, 'accretion_fill_factor', 0.01)
            f_fill = max(min(f_fill, 1.0), 1e-6)
            m_sun_cgs = (1.0 * units('solar_mass')).to_base_units().magnitude
            mdot_g_per_s = mdot_msun_per_yr * m_sun_cgs / (365.25 * 24.0 * 3600.0)
            Lacc = G_CGS * mstar_cgs * mdot_g_per_s / rstar_cgs
            r_acc = (f_fill ** 0.5) * rstar_cgs
            Tacc = (Lacc / (4.0 * np.pi * SIGMA_SB * r_acc * r_acc)) ** 0.25
            if Tacc > 0.0:
                stars.append({'R_cm': r_acc, 'M_g': mstar_cgs, 'T_K': float(Tacc)})

        with open(filepath, 'w') as f:
            f.write('2\n')
            f.write(f'{len(stars)} {nwav}\n')
            for s in stars:
                f.write(f"{s['R_cm']:13.6e} {s['M_g']:13.6e} ")
                f.write(f"{pos_cgs[0]:13.6e} {pos_cgs[1]:13.6e} {pos_cgs[2]:13.6e}\n")
            for wav in waves_micron:
                f.write(f'{wav:13.6e}\n')
            for s in stars:
                f.write(f"{-s['T_K']:13.6e}\n")
        
        self.written_files['stars.inp'] = filepath
        logger.info(f"Wrote stars file: {filepath}")
    
    def write_radmc3d_inp(
        self,
        output_dir: str | Path = '.',
        incl_dust: int = 1,
        incl_lines: int = 0,
        nphot: Optional[int] = None,
        nphot_scat: Optional[int] = None,
        scattering_mode_max: Optional[int] = None,
        modified_random_walk: int = 1,
        setthreads: Optional[int] = None,
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
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'config')
        filepath = output_dir / 'radmc3d.inp'

        if nphot is None:
            nphot = int(self.params.n_thermal)
        if nphot_scat is None:
            nphot_scat = int(self.params.n_scat)
        if scattering_mode_max is None:
            scattering_mode_max = self.params.scat_mode
        if setthreads is None:
            setthreads = self.params.nbcores
        
        with open(filepath, 'w') as f:
            f.write(f'incl_dust = {incl_dust}\n')
            f.write(f'incl_lines = {incl_lines}\n')
            f.write(f'nphot = {nphot}\n')
            f.write(f'nphot_scat = {nphot_scat}\n')
            f.write(f'scattering_mode_max = {scattering_mode_max}\n')
            f.write(f'modified_random_walk = {modified_random_walk}\n')
            f.write(f'setthreads = {setthreads}\n')
        
        self.written_files['radmc3d.inp'] = filepath
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
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'dust')
        
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
            self.written_files['dust_density.binp'] = filepath
        else:
            filepath = output_dir / 'dust_density.inp'
            self._write_dust_density_ascii(filepath, nbin, ncells)
            self.written_files['dust_density.inp'] = filepath
        
        logger.info(f"Wrote dust density file: {filepath}")
    
    def _write_dust_density_binary(
        self,
        filepath: Path,
        nbin: int,
        ncells: int,
    ) -> None:
        """Write binary dust density file.
        
        RADMC-3D expects data in (nsec, ncol, nrad) order for spherical grids.
        DiskBridge stores data in (r, phi, theta) = (nrad, nsec, ncol) order.
        We need to transpose before writing.
        """
        with open(filepath, 'wb') as f:
            # Header: format_number, precision(8=double), ncells, nbin
            header = np.array([1, 8, ncells, nbin], dtype=np.int64)
            header.tofile(f)
            
            # Write density for each bin
            for ibin in range(nbin):
                bin_data = self.model.dust.bins[f"bin_{ibin}"]
                rho_field = bin_data['density']  # Field object
                rho_cgs = self._to_cgs(rho_field.data)
                
                # Transpose from DiskBridge order (nrad, nsec, ncol) 
                # to RADMC-3D order (nsec, ncol, nrad)
                if rho_field.axis_order == ('r', 'phi', 'theta'):
                    rho_cgs = np.transpose(rho_cgs, (1, 2, 0))  # (nrad, nsec, ncol) -> (nsec, ncol, nrad)
                
                rho_cgs.flatten().astype(np.float64).tofile(f)
    
    def _write_dust_density_ascii(
        self,
        filepath: Path,
        nbin: int,
        ncells: int,
    ) -> None:
        """Write ASCII dust density file.
        
        RADMC-3D expects data in (nsec, ncol, nrad) order for spherical grids.
        DiskBridge stores data in (r, phi, theta) = (nrad, nsec, ncol) order.
        We need to transpose before writing.
        """
        with open(filepath, 'w') as f:
            # Header
            f.write('1\n')  # Format number
            f.write(f'{ncells}\n')
            f.write(f'{nbin}\n')
            
            # Write density for each bin
            for ibin in range(nbin):
                bin_data = self.model.dust.bins[f"bin_{ibin}"]
                rho_field = bin_data['density']  # Field object
                rho_cgs = self._to_cgs(rho_field.data)
                
                # Transpose from DiskBridge order (nrad, nsec, ncol) 
                # to RADMC-3D order (nsec, ncol, nrad)
                if rho_field.axis_order == ('r', 'phi', 'theta'):
                    rho_cgs = np.transpose(rho_cgs, (1, 2, 0))  # (nrad, nsec, ncol) -> (nsec, ncol, nrad)
                
                rho_flat = rho_cgs.flatten()
                for val in rho_flat:
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
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'opacity')
        filepath = output_dir / 'dustopac.inp'
        
        if self.model.dust is None:
            raise ValueError("Model has no dust data")
        
        nbin = self.model.dust.nbin
        if species_names is None:
            base_species = getattr(self.params, 'species', None)
            if base_species is not None:
                species_names = [f"{base_species}{i}" for i in range(nbin)]
            else:
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
        
        self.written_files['dustopac.inp'] = filepath
        logger.info(f"Wrote dustopac.inp file: {filepath}")
    
    def compute_and_write_dust_opacities(
        self,
        output_dir: str | Path = '.',
        optconst_file: Optional[str | Path] = None,
        grain_density: Optional[float] = None,
        wmin_micron: float = 0.1,
        wmax_micron: float = 10000.0,
        nwav: int = 200,
        ntheta: int = 181,
        scattering_mode: Optional[int] = None,
        logawidth: float = 0.05,
        na: int = 20,
    ) -> None:
        """Compute and write dust opacity files for all dust bins.
        
        Uses Mie theory with parameters matching fargo2radmc3d defaults:
        - logawidth = 0.05 (5% grain size smearing)
        - na = 20 (number of size samples)
        - chopforward = 1.0 degree (forward scattering removal)
        
        Args:
            output_dir: Directory to write files
            optconst_file: Path to optical constants file (.lnk format)
            grain_density: Grain material density in g/cm^3 (default: species-dependent)
            wmin_micron: Minimum wavelength in microns
            wmax_micron: Maximum wavelength in microns
            nwav: Number of wavelength points
            ntheta: Number of scattering angles
            scattering_mode: Scattering mode (>=3 for full matrix)
            logawidth: Width parameter for size distribution smoothing (default: 0.05)
            na: Number of size samples for smoothing (default: 20)
        """
        if self.model.dust is None:
            raise ValueError("Model has no dust data")

        # Derive optical constants file from global parameters if not provided
        if optconst_file is None:
            opacity_dir = self.params.opacity_dir
            species = self.params.species
            optconst_file = Path(opacity_dir) / f"{species}.lnk"
        else:
            # Extract species name from optconst_file for density lookup
            species = Path(optconst_file).stem
        
        # Set grain density based on species (matching fargo2radmc3d defaults)
        if grain_density is None:
            species_lower = species.lower()
            if 'ice70' in species_lower or species_lower == 'mix_2species_ice70':
                grain_density = 1.26  # g/cm³ for mix_2species_ice70
            elif 'porous' in species_lower:
                grain_density = 0.1   # g/cm³ for porous species
            elif species_lower in ['mix_2species', 'mix_2species_60silicates_40ice']:
                grain_density = 1.7   # g/cm³
            elif '60silicates_40carbons' in species_lower:
                grain_density = 2.7   # g/cm³
            else:
                grain_density = 2.7   # g/cm³ default
                logger.warning(f"Unknown species '{species}', using default grain_density = {grain_density} g/cm³")

        # Default scattering mode from global parameters if not provided
        if scattering_mode is None:
            scattering_mode = self.params.scat_mode
        
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'opacity')
        
        species_base = species
        
        # Create wavelength grid in cm
        Pw = (wmax_micron / wmin_micron) ** (1.0 / (nwav - 1))
        waves_micron = wmin_micron * Pw ** np.arange(nwav)
        waves_cm = waves_micron * 1e-4
        
        # Create angular grid for scattering.
        # To match fargo2radmc3d's makedustopac.py, always use a full theta grid
        # (0..180 degrees, ntheta points) and apply a small forward-scattering
        # chop when computing opacities. RADMC-3D will still only use the full
        # scattering matrix if scattering_mode>=3.
        theta = np.linspace(0., 180., ntheta)
        
        # Compute opacity for each dust bin
        for ibin in range(self.model.dust.nbin):
            bin_data = self.model.dust.bins[f"bin_{ibin}"]
            
            # Get grain size in CGS
            grain_size_cgs = self._to_cgs(bin_data.size_min)
            
            logger.debug(f"Computing opacity for bin {ibin}: "
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
                chopforward=1.0,
                extrapolate=True,
            )
            
            # Write to file
            species_name = f"{species_base}{ibin}"
            output_path = output_dir / species_name
            self.opacity_calculator.write_radmc3d_opacity_file(
                opacity_data=opac,
                output_path=output_path,
                scattering_matrix=(scattering_mode >= 3),
            )
            
            # Track the opacity file
            opacity_file = f"dustkappa_{species_name}.inp"
            self.written_files[opacity_file] = output_dir / opacity_file
        
        logger.info(f"Wrote {self.model.dust.nbin} opacity files")
    
    def write_all_input_files(
        self,
        output_dir: str | Path = '.',
        optconst_file: Optional[str | Path] = None,
        grain_density: Optional[float] = None,
        rstar: Optional[float] = None,
        tstar: Optional[float] = None,
        mstar: Optional[float] = None,
        scattering_mode: Optional[int] = None,
        nphot: Optional[int] = None,
        nphot_scat: Optional[int] = None,
        setthreads: Optional[int] = None,
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
            nphot_scat: Number of photons for scattering Monte Carlo
            setthreads: Number of OpenMP threads
            **kwargs: Additional arguments passed to individual writers
        """
        logger.info(f"Writing all RADMC-3D input files to {output_dir}")

        # Fill defaults from global parameter set if not explicitly given
        if grain_density is None:
            grain_density = self.params.grain_density
        if rstar is None:
            rstar = self.params.rstar_rsun
        if tstar is None:
            tstar = self.params.teff_K
        if mstar is None:
            mstar = self.params.mstar_msun
        if scattering_mode is None:
            scattering_mode = self.params.scat_mode
        if nphot is None:
            nphot = int(self.params.n_thermal)
        if nphot_scat is None:
            nphot_scat = int(self.params.n_scat)
        if setthreads is None:
            setthreads = self.params.nbcores
        
        # Determine wavelength grid from kwargs or global parameters
        wmin = kwargs.get('wmin_micron', self.params.lambda_min_micron)
        wmax = kwargs.get('wmax_micron', self.params.lambda_max_micron)
        nwav = kwargs.get('nwav', self.params.n_lambda)

        # Write grid files
        self.write_amr_grid(output_dir)
        self.write_wavelength_grid(output_dir, wmin_micron=wmin, wmax_micron=wmax, nwav=nwav)
        
        # Write stellar source
        wmin = wmin
        wmax = wmax
        nwav = nwav
        self.write_stars(output_dir, rstar=rstar, tstar=tstar, mstar=mstar,
                        wmin_micron=wmin, wmax_micron=wmax, nwav=nwav)
        
        # Write dust data
        if self.model.dust is not None and self.model.dust.nbin > 0:
            self.write_dust_density(output_dir)
            self.write_dustopac(output_dir, scattering_mode=scattering_mode)
            
            if optconst_file is not None:
                op_kwargs = {k: v for k, v in kwargs.items() 
                             if k in ['wmin_micron', 'wmax_micron', 'nwav', 'ntheta', 'logawidth', 'na']}
                op_kwargs.setdefault('wmin_micron', wmin)
                op_kwargs.setdefault('wmax_micron', wmax)
                self.compute_and_write_dust_opacities(
                    output_dir,
                    optconst_file=optconst_file,
                    grain_density=grain_density,
                    scattering_mode=scattering_mode,
                    **op_kwargs,
                )
        
        # Write control file
        self.write_radmc3d_inp(
            output_dir,
            scattering_mode_max=scattering_mode,
            nphot=nphot,
            nphot_scat=nphot_scat,
            setthreads=setthreads,
        )
        
        logger.info("All RADMC-3D input files written successfully")
    
    def write_gas_temperature(self, temperature: Quantity, output_dir: str | Path = '.') -> None:
        """Write gas temperature to gas_temperature.inp.
        
        Parameters
        ----------
        temperature : Quantity
            Gas temperature field in K, shape (nx, ny, nz)
        output_dir : str or Path, optional
            Directory to write file (default: '.')
            
        Notes
        -----
        Writes ASCII format gas_temperature.inp file for RADMC-3D LTE line transfer.
        Temperature should be in Kelvin.
        
        RADMC-3D expects cells in the same order as the grid.
        For spherical grids, cells are ordered as (nsec, ncol, nrad) = (phi, theta, r).
        DiskBridge stores data in (r, phi, theta) order, so we transpose before writing.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fpath = output_dir / 'gas_temperature.inp'
        
        # Convert to K and CGS
        temp = self._to_cgs(temperature.to('K'))
        
        # Transpose from DiskBridge order to RADMC-3D order
        # This matches the approach used in write_dust_density and write_gas_velocity
        mesh = self.model.mesh
        if mesh.coord_system == 'spherical':
            # DiskBridge: (r, phi, theta) = (nrad, nsec, ncol)
            # RADMC-3D:   (phi, theta, r) = (nsec, ncol, nrad)
            # Transpose: (nrad, nsec, ncol) -> (nsec, ncol, nrad)
            temp = np.transpose(temp, (1, 2, 0))
        
        # Flatten in C order (row-major) to match RADMC-3D cell ordering
        temp_flat = temp.flatten()
        ncells = temp_flat.size
        
        logger.info(f"Writing gas temperature to {fpath}: {ncells} cells, "
                   f"T_range=[{temp.min():.1f}, {temp.max():.1f}] K")
        
        with open(fpath, 'w') as f:
            f.write('1\n')  # Format number
            f.write(f'{ncells}\n')
            for T in temp_flat:
                f.write(f'{T:.6e}\n')
                
        logger.info(f"Wrote {fpath}")
        
    def write_gas_velocity(
        self, 
        vr: Quantity, 
        vtheta: Quantity, 
        vphi: Quantity, 
        output_dir: str | Path = '.',
        binary: bool = True
    ) -> None:
        """Write gas velocity to gas_velocity.binp or gas_velocity.inp.
        
        Parameters
        ----------
        vr : Quantity
            Radial velocity component, shape (nx, ny, nz)
        vtheta : Quantity
            Theta velocity component, shape (nx, ny, nz)
        vphi : Quantity
            Phi velocity component, shape (nx, ny, nz)
        output_dir : str or Path, optional
            Directory to write file (default: '.')
        binary : bool, optional
            Write binary format (default: True)
            
        Notes
        -----
        Writes gas velocity in cm/s for RADMC-3D LTE line transfer. 
        Binary format is recommended for large grids. Velocities should 
        be in the same coordinate system as the grid (spherical, cylindrical, 
        or Cartesian).
        
        RADMC-3D expects velocities in cell order matching the grid.
        For spherical grids, cells are ordered as (nsec, ncol, nrad) = (phi, theta, r).
        DiskBridge stores data in (r, phi, theta) order, so we transpose before writing.
        """
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'gas')
        
        # Convert to cm/s
        vr_cgs = self._to_cgs(vr.to('cm/s'))
        vtheta_cgs = self._to_cgs(vtheta.to('cm/s'))
        vphi_cgs = self._to_cgs(vphi.to('cm/s'))
        
        # Transpose from DiskBridge order to RADMC-3D order
        # This matches the approach used in write_dust_density
        mesh = self.model.mesh
        if mesh.coord_system == 'spherical':
            # DiskBridge: (r, phi, theta) = (nrad, nsec, ncol)
            # RADMC-3D:   (phi, theta, r) = (nsec, ncol, nrad)
            # Transpose: (nrad, nsec, ncol) -> (nsec, ncol, nrad)
            vr_cgs = np.transpose(vr_cgs, (1, 2, 0))
            vtheta_cgs = np.transpose(vtheta_cgs, (1, 2, 0))
            vphi_cgs = np.transpose(vphi_cgs, (1, 2, 0))
        
        # Flatten in C order (row-major) to match RADMC-3D cell ordering
        vr_flat = vr_cgs.flatten()
        vtheta_flat = vtheta_cgs.flatten()
        vphi_flat = vphi_cgs.flatten()
        ncells = vr_flat.size
        
        if binary:
            fpath = output_dir / 'gas_velocity.binp'
            logger.info(f"Writing gas velocity (binary) to {fpath}: {ncells} cells")
            
            with open(fpath, 'wb') as f:
                # Write header: format, precision, ncells
                np.array([1], dtype=np.int64).tofile(f)  # Format
                np.array([8], dtype=np.int64).tofile(f)  # Precision (8 bytes for float64)
                np.array([ncells], dtype=np.int64).tofile(f)  # Number of cells
                
                # Write velocities (vr, vtheta, vphi for each cell)
                for i in range(ncells):
                    np.array([vr_flat[i], vtheta_flat[i], vphi_flat[i]], 
                            dtype=np.float64).tofile(f)
        else:
            fpath = output_dir / 'gas_velocity.inp'
            logger.info(f"Writing gas velocity (ASCII) to {fpath}: {ncells} cells")
            
            with open(fpath, 'w') as f:
                f.write('1\n')  # Format number
                f.write(f'{ncells}\n')
                for i in range(ncells):
                    f.write(f'{vr_flat[i]:.6e} {vtheta_flat[i]:.6e} {vphi_flat[i]:.6e}\n')
        
        # Track the written file
        self.written_files[fpath.name] = fpath
        logger.info(f"Wrote {fpath}")
