"""RADMC-3D input file writer for DiskBridge Models.

This module provides the RadWriter class that takes a Model instance
and writes the necessary RADMC-3D input files, handling unit conversion
from Pint Quantities to CGS units.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Tuple
from pathlib import Path
import hashlib
import numpy as np

if TYPE_CHECKING:
    from diskbridge.model.core import Model

from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from diskbridge._constants import (
    C_LIGHT as C_CGS,
    EXTERNAL_CMB,
    EXTERNAL_IR_BETA,
    EXTERNAL_IR_DUST,
    EXTERNAL_IR_REFERENCE_WAVELENGTH_MICRON,
    EXTERNAL_IR_T0,
    EXTERNAL_IR_TAU_REF,
    H_PLANCK as H_CGS,
    K_B as K_B_CGS,
    T_CMB,
)
import diskbridge
from .opacities import DustOpacityCalculator
from diskbridge.model.utils import transpose_to_axis_order
from diskbridge.model.microturbulence import MICROTURBULENCE_FIELD, ensure_microturbulence_field

# Physical constants from config
G_CGS = units('G')
SIGMA_SB = units('sigma_SB')

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
        
        # Define single input directory for all radmc3d input files
        self.inputs_dir = 'radmc3d_inputs'
        
        # Track written files for symlink management
        self.written_files = {}

    @staticmethod
    def radmc_spherical_phi_edges_rad(mesh) -> np.ndarray:
        """Return the phi edges DiskBridge writes to RADMC-3D spherical grids."""

        if mesh.coord_system != 'spherical':
            raise ValueError(
                f"radmc_spherical_phi_edges_rad requires a spherical mesh, got {mesh.coord_system}"
            )
        edges = mesh.edges_f64("phi", "rad")
        tol = 1.0e-10
        if edges[0] >= -tol and edges[-1] <= 2.0 * np.pi + tol:
            out = np.array(edges, copy=True)
            out[0] = max(out[0], 0.0)
            out[-1] = min(out[-1], 2.0 * np.pi)
            return out
        # Compatibility for older saved FARGO snapshots. New readers should
        # canonicalize phi at load time so this fallback is not normally used.
        nphi = int(mesh.ncell("phi") or 0)
        return np.linspace(0.0, 2.0 * np.pi, nphi + 1)

    @staticmethod
    def radmc_spherical_phi_centers_rad(mesh) -> np.ndarray:
        edges = RadWriter.radmc_spherical_phi_edges_rad(mesh)
        return 0.5 * (edges[:-1] + edges[1:])

    @staticmethod
    def flatten_scalar_to_radmc_order(mesh, arr3d: np.ndarray) -> np.ndarray:
        """Flatten a native mesh-order scalar field in RADMC-3D cell order."""

        arr = np.asarray(arr3d)
        if mesh.coord_system == 'spherical':
            arr = transpose_to_axis_order(
                arr,
                from_order=mesh.axis_names(),
                to_order=('phi', 'theta', 'r'),
            )
        elif mesh.coord_system != 'cartesian':
            raise ValueError(f"Unsupported coordinate system: {mesh.coord_system}")
        return np.ascontiguousarray(arr).reshape(-1, order='C')

    @staticmethod
    def flatten_vector_to_radmc_order(mesh, components: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """Return interleaved RADMC-3D vector rows, shape ``(ncells, 3)``."""

        flats = [
            RadWriter.flatten_scalar_to_radmc_order(mesh, np.asarray(component))
            for component in components
        ]
        return np.ascontiguousarray(np.stack(flats, axis=1), dtype=np.float64)

    @staticmethod
    def gas_velocity_binp_sha256(mesh, components: tuple[np.ndarray, np.ndarray, np.ndarray]) -> str:
        """Hash the exact binary payload written by ``write_gas_velocity``."""

        vectors = RadWriter.flatten_vector_to_radmc_order(mesh, components)
        digest = hashlib.sha256()
        digest.update(np.asarray([1, 8, vectors.shape[0]], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(vectors, dtype=np.float64).tobytes())
        return digest.hexdigest()

    @staticmethod
    def write_levelpop_dat(
        path: str | Path,
        *,
        levelpop_cm3_radmc_order: np.ndarray,
        level_numbers_1based: np.ndarray,
    ) -> Path:
        """Write a RADMC-3D ``levelpop_<species>.dat`` file."""

        path = Path(path)
        pop = np.ascontiguousarray(levelpop_cm3_radmc_order, dtype=np.float64)
        if pop.ndim != 2:
            raise ValueError(
                f"levelpop_cm3_radmc_order must be 2D, got shape {pop.shape}"
            )
        n_cells, n_levels = pop.shape
        levels = np.ascontiguousarray(level_numbers_1based, dtype=np.int64)
        if levels.ndim != 1 or levels.size != n_levels:
            raise ValueError(
                f"level_numbers_1based shape {levels.shape} inconsistent with "
                f"n_levels={n_levels}"
            )
        if not np.all(np.isfinite(pop)):
            raise ValueError("levelpop_cm3_radmc_order contains non-finite values")
        if np.any(pop < 0.0):
            raise ValueError("levelpop_cm3_radmc_order contains negative values")

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            f.write('1\n')
            f.write(f'{n_cells}\n')
            f.write(f'{n_levels}\n')
            f.write(' '.join(str(int(x)) for x in levels) + '\n')
            for i in range(n_cells):
                row = pop[i]
                f.write(' '.join(f'{x:.16e}' for x in row) + '\n')
        return path

    def write_levelpop(
        self,
        species: str,
        levelpop_cm3: np.ndarray,
        output_dir: str | Path = '.',
        level_numbers_1based: np.ndarray | None = None,
    ) -> None:
        """Write native mesh-order level populations to ``levelpop_<species>.dat``."""

        mesh = self.model.mesh
        if mesh is None:
            raise ValueError('Model has no mesh defined')
        pop = np.asarray(levelpop_cm3, dtype=np.float64)
        if pop.ndim != 4:
            raise ValueError(f"levelpop_cm3 must have shape (n0, n1, n2, nlev), got {pop.shape}")
        n_levels = pop.shape[-1]
        if level_numbers_1based is None:
            level_numbers_1based = np.arange(1, n_levels + 1, dtype=np.int64)

        pop_radmc_order = np.stack(
            [
                self.flatten_scalar_to_radmc_order(mesh, pop[..., k])
                for k in range(n_levels)
            ],
            axis=1,
        )
        output_dir = self._get_output_dir(Path(output_dir), 'molecule')
        species = str(species).lower().strip()
        fpath = output_dir / f'levelpop_{species}.dat'
        self.write_levelpop_dat(
            fpath,
            levelpop_cm3_radmc_order=pop_radmc_order,
            level_numbers_1based=level_numbers_1based,
        )
        self.written_files[fpath.name] = fpath
        logger.info(f'Wrote {fpath}')
    
    def _get_output_dir(self, base_dir: Path, file_type: str) -> Path:
        """Get output directory for a file type.
        
        Args:
            base_dir: Base output directory
            file_type: Type of file (all input files go to radmc3d_inputs)
            
        Returns:
            Path to output directory
        """
        if not self.organize_files:
            return base_dir
        
        # All input files now go to the same directory
        output_dir = base_dir / self.inputs_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    
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
                active_dims[i] = 0 if (
                    mesh.coord_system == 'spherical' and name == 'phi' and int(ncell) == 1
                ) else 1
            else:
                dims.append(1)  # Inactive dimension has size 1
        
        with open(filepath, 'w') as f:
            f.write('1\n')  # iformat
            f.write('0\n')  # AMR grid style (0=regular)
            f.write(f'{coordsys_code}\n')  # Coordinate system
            f.write('0\n')  # gridinfo
            f.write(f'{active_dims[0]} {active_dims[1]} {active_dims[2]}\n')
            f.write(f'{dims[0]} {dims[1]} {dims[2]}\n')
            
            # Write grid edges for each dimension (in cm or radians) as plain floats
            for name in axis_names:
                edges = mesh.edges(name)
                if edges is not None:
                    # Convert to base units and drop units so we write pure numbers
                    edges_cgs = edges.to_base_units().magnitude
                    
                    # RADMC-3D requires spherical phi in [0, 2*pi]. Hydro
                    # readers should canonicalize meshes to this convention at
                    # load time; the helper keeps old saved snapshots readable.
                    if name == 'phi':
                        edges_cgs = self.radmc_spherical_phi_edges_rad(mesh)
                    
                    for val in edges_cgs:
                        f.write(f'{val:.16e} ')
                else:
                    # Write dummy edges for inactive dimension
                    f.write('0.0 1.0 ')
                f.write('\n')
        
        self.written_files['amr_grid.inp'] = filepath
        logger.info(f"Wrote AMR grid file: {filepath}")
    
    def write_wavelength_grid(
        self,
        output_dir: str | Path = '.',
    ) -> None:
        """Write wavelength_micron.inp file.
        
        Args:
            output_dir: Directory to write the file
        """
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'grid')
        filepath = output_dir / 'wavelength_micron.inp'
        
        # Get parameters from params
        wmin_micron = self.params.lambda_min
        wmax_micron = self.params.lambda_max
        nwav = self.params.n_lambda
        
        # Convert to microns
        wmin = wmin_micron.to('micron').magnitude
        wmax = wmax_micron.to('micron').magnitude
        
        # Create logarithmically spaced wavelength grid
        Pw = (wmax / wmin) ** (1.0 / (nwav - 1))
        waves = np.zeros(nwav)
        waves[0] = wmin
        for i in range(1, nwav):
            waves[i] = wmin * Pw ** i

        waves[0] = wmin
        waves[-1] = wmax
        
        with open(filepath, 'w') as f:
            f.write(f'{nwav}\n')
            for w in waves:
                f.write(f'{w:13.6e}\n')
        
        self.written_files['wavelength_micron.inp'] = filepath
        logger.info(f"Wrote wavelength grid file: {filepath}")
    
    def write_stars(
        self,
        output_dir: str | Path = '.',
    ) -> None:
        """Write stars.inp file for stellar radiation source.
        
        Args:
            output_dir: Directory to write the file
        """
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'star')
        filepath = output_dir / 'stars.inp'
        
        # Get parameters from params and convert to CGS / microns
        rstar_cgs = self.params.rstar.to('cm').magnitude
        mstar_cgs = self.params.mstar.to('g').magnitude
        tstar_K = self.params.teff.to('K').magnitude
        pos0 = Quantity(0.0, 'au').to('cm').magnitude
        pos_cgs = [pos0, pos0, pos0]  # Always at origin
        wmin = self.params.lambda_min.to('micron').magnitude
        wmax = self.params.lambda_max.to('micron').magnitude
        nwav = self.params.n_lambda
        
        # Build wavelength grid (matches wavelength_micron.inp)
        Pw = (wmax / wmin) ** (1.0 / (nwav - 1))
        waves_micron = wmin * Pw ** np.arange(nwav)

        stars = []
        stars.append({'R_cm': rstar_cgs, 'M_g': mstar_cgs, 'T_K': tstar_K})

        mdot_msun_per_yr = getattr(self.params, 'mdot', 0.0)
        if mdot_msun_per_yr > 0.0:
            f_fill = getattr(self.params, 'accretion_fill_factor', 0.01)
            f_fill = max(min(f_fill, 1.0), 1e-6)
            # Convert constants to CGS
            g_cgs = G_CGS.to_base_units().magnitude
            sigma_cgs = SIGMA_SB.to_base_units().magnitude
            m_sun_g = units('solar_mass').to('g').magnitude
            # Compute accretion luminosity
            mdot_cgs = mdot_msun_per_yr * m_sun_g / (365.25 * 24.0 * 3600.0)
            Lacc_cgs = g_cgs * mstar_cgs * mdot_cgs / rstar_cgs
            r_acc_cgs = (f_fill ** 0.5) * rstar_cgs
            Tacc_K = (Lacc_cgs / (4.0 * np.pi * sigma_cgs * r_acc_cgs * r_acc_cgs)) ** 0.25
            if Tacc_K > 0.0:
                stars.append({'R_cm': r_acc_cgs, 'M_g': mstar_cgs, 'T_K': Tacc_K})

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
        nphot_mono: Optional[int] = None,
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
            nphot = int(self.params.nphot_thermal)
        if nphot_scat is None:
            nphot_scat = int(self.params.nphot_scat)
        if nphot_mono is None:
            nphot_mono = int(getattr(self.params, 'nphot_mono', self.params.nphot_thermal))
        if scattering_mode_max is None:
            scattering_mode_max = self.params.scat_mode
        if setthreads is None:
            setthreads = self.params.nbcores
        line_params = diskbridge.canonicalize_line_params(self.params)
        line_modes = {int(mode) for mode in line_params["line_mode"]}
        if len(line_modes) != 1:
            raise ValueError(
                "RADMC-3D uses one global lines_mode per run; got "
                f"{sorted(line_modes)} from params."
            )
        line_mode = next(iter(line_modes))
        
        with open(filepath, 'w') as f:
            f.write(f'incl_dust = {incl_dust}\n')
            f.write(f'incl_lines = {incl_lines}\n')
            f.write(f'lines_mode = {line_mode}\n')
            f.write(f'nphot = {nphot}\n')
            f.write(f'nphot_scat = {nphot_scat}\n')
            f.write(f'nphot_mono = {nphot_mono}\n')
            f.write(f'scattering_mode_max = {scattering_mode_max}\n')
            f.write(f'modified_random_walk = {modified_random_walk}\n')
            f.write(f'setthreads = {setthreads}\n')
            f.write('rto_style = 3\n')
        
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
        DiskBridge stores fields in the canonical mesh.axis_names() order.
        We reorder explicitly via Field.axis_order before writing.
        """
        with open(filepath, 'wb') as f:
            # Header: format_number, precision(8=double), ncells, nbin
            header = np.array([1, 8, ncells, nbin], dtype=np.int64)
            header.tofile(f)
            
            # Write density for each bin
            for ibin in range(nbin):
                bin_data = self.model.dust.bins[f"bin_{ibin}"]
                rho_field = bin_data['density']  # Field object
                rho_cgs = rho_field.data.to_base_units().magnitude

                if self.model.mesh is None:
                    raise ValueError("Model has no mesh defined")

                if self.model.mesh.coord_system == 'spherical':
                    rho_out = transpose_to_axis_order(
                        np.asarray(rho_cgs),
                        from_order=rho_field.axis_order,
                        to_order=('phi', 'theta', 'r'),
                    )
                    rho_flat = np.ravel(rho_out)
                elif self.model.mesh.coord_system == 'cartesian':
                    rho_out = transpose_to_axis_order(
                        np.asarray(rho_cgs),
                        from_order=rho_field.axis_order,
                        to_order=('x', 'y', 'z'),
                    )
                    rho_flat = np.ravel(rho_out, order='F')
                else:
                    raise ValueError(
                        f"dust density writer supports spherical or cartesian only, got {self.model.mesh.coord_system}"
                    )

                rho_flat.astype(np.float64).tofile(f)
    
    def _write_dust_density_ascii(
        self,
        filepath: Path,
        nbin: int,
        ncells: int,
    ) -> None:
        """Write ASCII dust density file.
        
        RADMC-3D expects data in (nsec, ncol, nrad) order for spherical grids.
        DiskBridge stores fields in the canonical mesh.axis_names() order.
        We reorder explicitly via Field.axis_order before writing.
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
                rho_cgs = rho_field.data.to_base_units().magnitude

                if self.model.mesh is None:
                    raise ValueError("Model has no mesh defined")

                if self.model.mesh.coord_system == 'spherical':
                    rho_out = transpose_to_axis_order(
                        np.asarray(rho_cgs),
                        from_order=rho_field.axis_order,
                        to_order=('phi', 'theta', 'r'),
                    )
                    rho_flat = np.ravel(rho_out)
                elif self.model.mesh.coord_system == 'cartesian':
                    rho_out = transpose_to_axis_order(
                        np.asarray(rho_cgs),
                        from_order=rho_field.axis_order,
                        to_order=('x', 'y', 'z'),
                    )
                    rho_flat = np.ravel(rho_out, order='F')
                else:
                    raise ValueError(
                        f"dust density writer supports spherical or cartesian only, got {self.model.mesh.coord_system}"
                    )

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
                grain_density = 1.26  # g/cm^3 for mix_2species_ice70
            elif 'porous' in species_lower:
                grain_density = 0.1   # g/cm^3 for porous species
            elif species_lower in ['mix_2species', 'mix_2species_60silicates_40ice']:
                grain_density = 1.7   # g/cm^3
            elif '60silicates_40carbons' in species_lower:
                grain_density = 2.7   # g/cm^3
            else:
                grain_density = 2.7   # g/cm^3 default
                logger.warning(f"Unknown species '{species}', using default grain_density = {grain_density} g/cm^3")

        # Default scattering mode from global parameters if not provided
        if scattering_mode is None:
            scattering_mode = self.params.scat_mode
        
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'opacity')
        
        species_base = species
        
        # Get wavelength parameters from params
        wmin_micron = self.params.lambda_min.to('micron').magnitude
        wmax_micron = self.params.lambda_max.to('micron').magnitude
        nwav = self.params.n_lambda
        
        # Create wavelength grid in cm
        Pw = (wmax_micron / wmin_micron) ** (1.0 / (nwav - 1))
        waves_micron = wmin_micron * Pw ** np.arange(nwav)
        waves_micron[0] = wmin_micron
        waves_micron[-1] = wmax_micron
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
            grain_size_cgs = bin_data.size_min.to_base_units().magnitude
            
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
        if scattering_mode is None:
            scattering_mode = self.params.scat_mode
        if nphot is None:
            nphot = int(self.params.nphot_thermal)
        if nphot_scat is None:
            nphot_scat = int(self.params.nphot_scat)
        if setthreads is None:
            setthreads = self.params.nbcores
        
        # Write grid files
        self.write_amr_grid(output_dir)
        self.write_wavelength_grid(output_dir)
        
        # Write stellar source
        self.write_stars(output_dir)
        
        # Write dust data
        if self.model.dust is not None and self.model.dust.nbin > 0:
            self.write_dust_density(output_dir)
            self.write_dustopac(output_dir, scattering_mode=scattering_mode)
            
            if optconst_file is not None:
                op_kwargs = {k: v for k, v in kwargs.items() 
                             if k in ['ntheta', 'logawidth', 'na']}
                self.compute_and_write_dust_opacities(
                    output_dir,
                    optconst_file=optconst_file,
                    grain_density=grain_density,
                    scattering_mode=scattering_mode,
                    **op_kwargs,
                )

        ensure_microturbulence_field(self.model, self.params)
        if self.model.gas is not None and MICROTURBULENCE_FIELD in self.model.gas:
            self.write_microturbulence(output_dir)
        
        # Write control file
        self.write_radmc3d_inp(
            output_dir,
            scattering_mode_max=scattering_mode,
            nphot=nphot,
            nphot_scat=nphot_scat,
            setthreads=setthreads,
        )

        logger.info("All RADMC-3D input files written successfully")

    def write_model_inputs(
        self,
        output_dir: str | Path = '.',
        *,
        compute_opacities: bool = True,
        include_gas_velocity: bool = True,
        include_external_source: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Write the complete RADMC-3D input set for the active model."""
        self.write_all_input_files(output_dir, **kwargs)

        if include_gas_velocity:
            self.write_gas_velocity(output_dir)

        if include_external_source is None:
            include_external_source = bool(getattr(self.params, "external_uv", False))
        if include_external_source:
            self.write_external_source(output_dir)

        if compute_opacities and self.model.dust is not None and self.model.dust.nbin > 0:
            self.compute_and_write_dust_opacities(output_dir)
    
    def _ensure_isrf_file(self, path: Path) -> Path:
        if path.is_file():
            return path
        raise FileNotFoundError(
            "ISRF.dat not found. DiskBridge does not download this file at runtime. "
            f"Expected to find it at: {path}"
        )
    
    def _planck_B_nu(self, nu_hz: np.ndarray, T: float) -> np.ndarray:
        nu = np.asarray(nu_hz, dtype=float)
        x = H_CGS * nu / (K_B_CGS * float(T))
        x = np.clip(x, 1.0e-10, 1.0e3)
        prefac = 2.0 * H_CGS * nu**3 / C_CGS**2
        return prefac / np.expm1(x)

    def _load_leiden_draine_i_nu(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        data = np.loadtxt(str(path), comments="#")
        lam_nm = data[:, 0]
        photon_flux_nm = data[:, 1]
        lam_cm = lam_nm * 1.0e-7
        e_ph = H_CGS * C_CGS / lam_cm
        # table is photons s^-1 cm^-2 nm^-1
        # convert nm^-1 -> cm^-1, then divide by 4pi for isotropic intensity
        i_lambda = photon_flux_nm * e_ph / 1.0e-7 / (4.0 * np.pi)
        # I_nu = I_lambda * lambda^2 / c
        i_nu = i_lambda * lam_cm**2 / C_CGS
        order = np.argsort(lam_cm)
        return lam_cm[order], i_nu[order]

    def _make_external_i_nu(
        self,
        lam_cm: np.ndarray,
        chi: float,
        isrf_path: Path,
        include_ir_dust: bool = True,
        T0_ir: float = 18.0,
        beta_ir: float = 1.7,
        lambda_ref_um: float = 250.0,
        tau_ref: float = 1.0,
        include_cmb: bool = True,
    ) -> np.ndarray:
        """Build external RADMC-3D boundary intensity I_nu.

        The Draine/Leiden UV-optical-NIR ISRF is scaled by ``chi``. The same
        scale sets the greybody IR temperature as
        T_IR = T0_ir * chi**(1 / (4 + beta_ir)); ``tau_ref`` normalizes the
        greybody at ``lambda_ref_um``. The CMB component is unscaled.
        """
        lam_cm = np.asarray(lam_cm, dtype=float)
        nu = C_CGS / lam_cm
        lam_tab_cm, i_nu_tab = self._load_leiden_draine_i_nu(isrf_path)
        i_draine = np.zeros_like(lam_cm)
        m = (lam_cm >= lam_tab_cm.min()) & (lam_cm <= lam_tab_cm.max())
        if np.any(m):
            good = i_nu_tab > 0.0
            i_draine[m] = np.exp(
                np.interp(
                    np.log(lam_cm[m]),
                    np.log(lam_tab_cm[good]),
                    np.log(i_nu_tab[good]),
                )
            )
        i_total = float(chi) * i_draine
        if include_ir_dust:
            beta = float(beta_ir)
            T_ir = float(T0_ir) * float(chi) ** (1.0 / (4.0 + beta))
            lambda_ref_cm = float(lambda_ref_um) * 1.0e-4
            nu_ref = C_CGS / lambda_ref_cm
            i_ir = (
                float(tau_ref)
                * (nu / nu_ref) ** beta
                * self._planck_B_nu(nu, T_ir)
            )
            i_total += i_ir
        if include_cmb:
            i_total += self._planck_B_nu(nu, T_CMB)
        i_total = np.where(np.isfinite(i_total), i_total, 0.0)
        i_total = np.maximum(i_total, 0.0)
        return i_total

    def _read_wavelength_grid_from_file(self, filepath: Path) -> np.ndarray:
        if not filepath.is_file():
            raise FileNotFoundError(f"wavelength grid file not found: {filepath}")
        with open(filepath, 'r') as f:
            first = f.readline().strip()
            try:
                n = int(first)
                vals = [float(f.readline().strip()) for _ in range(n)]
            except ValueError:
                raise RuntimeError("Invalid wavelength_micron.inp format")
        return np.array(vals, dtype=float)

    def write_external_source(
        self,
        output_dir: str | Path = '.',
        chi: Optional[float] = None,
        isrf_path: str | Path = 'ISRF.dat',
    ) -> None:
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'external')
        data_dir = Path(__file__).resolve().parents[3] / 'data'
        isrf_name = Path(isrf_path).name
        isrf_path = data_dir / isrf_name
        isrf_file = self._ensure_isrf_file(isrf_path)
        wav_file = output_dir / 'wavelength_micron.inp'
        lam_um = self._read_wavelength_grid_from_file(wav_file)
        lam_cm = lam_um * 1.0e-4
        if chi is None:
            chi_val = float(getattr(self.params, 'external_uv_chi', 1.0))
        else:
            chi_val = float(chi)
        i_nu_scaled = self._make_external_i_nu(
            lam_cm,
            chi=chi_val,
            isrf_path=isrf_file,
            include_ir_dust=EXTERNAL_IR_DUST,
            T0_ir=EXTERNAL_IR_T0,
            beta_ir=EXTERNAL_IR_BETA,
            lambda_ref_um=EXTERNAL_IR_REFERENCE_WAVELENGTH_MICRON,
            tau_ref=EXTERNAL_IR_TAU_REF,
            include_cmb=EXTERNAL_CMB,
        )
        filepath = output_dir / 'external_source.inp'
        with open(filepath, 'w') as f:
            # RADMC-3D manual (sec-ext-src-inp): format 2, then nlam, then
            # lambda[i] in micron (identical to wavelength_micron.inp), then
            # intensity[i] in erg/cm^2/s/Hz/sr.
            f.write('2\n')
            f.write(f"{lam_um.size}\n")
            for w in lam_um:
                f.write(f"{w:13.6e}\n")
            for val in i_nu_scaled:
                f.write(f"{val:13.6e}\n")
        self.written_files['external_source.inp'] = filepath
        logger.info(f"Wrote external_source.inp file: {filepath}")
    def write_gas_temperature(
        self,
        temperature: Quantity,
        output_dir: str | Path = '.',
        binary: bool = False,
    ) -> None:
        """Write gas temperature to gas_temperature.inp or gas_temperature.binp.
        
        Parameters
        ----------
        temperature : Quantity
            Gas temperature field in K, shape (nx, ny, nz)
        output_dir : str or Path, optional
            Directory to write file (default: '.')
        binary : bool, optional
            Write RADMC-3D binary format (default: False)
            
        Notes
        -----
        Writes gas_temperature for RADMC-3D line transfer.
        Temperature should be in Kelvin.
        
        RADMC-3D expects cells in the same order as the grid.
        For spherical grids, cells are ordered as (nsec, ncol, nrad) = (phi, theta, r).
        DiskBridge stores fields in the canonical mesh.axis_names() order.
        We reorder explicitly before writing.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fpath = output_dir / ('gas_temperature.binp' if binary else 'gas_temperature.inp')
        
        temp = temperature.to('K').magnitude

        mesh = self.model.mesh
        if mesh is None:
            raise ValueError('Model has no mesh defined')
        if mesh.coord_system == 'spherical':
            temp = transpose_to_axis_order(
                np.asarray(temp),
                from_order=mesh.axis_names(),
                to_order=('phi', 'theta', 'r'),
            )
        elif mesh.coord_system != 'cartesian':
            raise ValueError(f'Unsupported coordinate system: {mesh.coord_system}')
        
        # Flatten in C order (row-major) to match RADMC-3D cell ordering
        temp_flat = np.asarray(temp).flatten()
        ncells = temp_flat.size
        
        logger.info(
            f"Writing gas temperature to {fpath}: {ncells} cells, "
            f"T_range=[{temp.min():.1f}, {temp.max():.1f}] K"
        )

        if binary:
            with open(fpath, 'wb') as f:
                header = np.array([1, 8, ncells], dtype=np.int64)
                header.tofile(f)
                temp_flat.astype(np.float64).tofile(f)
        else:
            with open(fpath, 'w') as f:
                f.write('1\n')  # Format number
                f.write(f'{ncells}\n')
                for T in temp_flat:
                    f.write(f'{T:.6e}\n')
                
        self.written_files[fpath.name] = fpath
        logger.info(f"Wrote {fpath}")

    def write_dust_temperature(
        self,
        temperature: Quantity,
        output_dir: str | Path = '.',
        nspec: int = 1,
    ) -> None:
        """Write dust temperature to dust_temperature.bdat.

        Parameters
        ----------
        temperature : Quantity
            Dust temperature field in K, shape (nx, ny, nz)
        output_dir : str or Path, optional
            Directory to write file (default: '.')
        nspec : int, optional
            Number of dust species in the file header (default: 1)

        Notes
        -----
        Writes binary format dust_temperature.bdat. The file format matches what
        RadData._readScalarFieldBinary() expects: header [iformat=1, prec=8, ncells, nspec], then values.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fpath = output_dir / 'dust_temperature.bdat'

        temp = np.asarray(temperature.to('K').magnitude)

        mesh = self.model.mesh
        if mesh is None:
            raise ValueError('Model has no mesh defined')

        nspec_i = int(nspec)
        if nspec_i <= 0:
            raise ValueError(f'nspec must be > 0, got {nspec_i}')

        if temp.ndim not in (3, 4):
            raise ValueError(
                f"dust_temperature must be 3D (single species) or 4D (nspec, ...), got shape={temp.shape}"
            )

        if temp.ndim == 4 and int(temp.shape[0]) != nspec_i:
            raise ValueError(
                f"dust_temperature nspec mismatch: nspec={nspec_i}, temperature.shape[0]={int(temp.shape[0])}"
            )

        if mesh.coord_system == 'spherical':
            if temp.ndim == 3:
                temp = transpose_to_axis_order(
                    np.asarray(temp),
                    from_order=mesh.axis_names(),
                    to_order=('phi', 'theta', 'r'),
                )
                temp_flat = np.asarray(temp).flatten(order='C')
            else:
                temp_flat = None
        elif mesh.coord_system == 'cartesian':
            if temp.ndim == 3:
                temp = transpose_to_axis_order(
                    np.asarray(temp),
                    from_order=mesh.axis_names(),
                    to_order=('x', 'y', 'z'),
                )
                temp_flat = np.asarray(temp).flatten(order='F')
            else:
                temp_flat = None
        else:
            raise ValueError(f'Unsupported coordinate system: {mesh.coord_system}')

        if temp.ndim == 3:
            if temp_flat is None:
                raise ValueError('Internal error: missing flattened temperature array')
            ncells = int(temp_flat.size)
        else:
            if mesh.coord_system == 'spherical':
                nr = len(mesh.axes['r'].centers)
                ntheta = len(mesh.axes['theta'].centers)
                nphi = len(mesh.axes['phi'].centers)
                ncells = int(nr * ntheta * nphi)
            elif mesh.coord_system == 'cartesian':
                nx = len(mesh.axes['x'].centers)
                ny = len(mesh.axes['y'].centers)
                nz = len(mesh.axes['z'].centers)
                ncells = int(nx * ny * nz)
            else:
                raise ValueError(f'Unsupported coordinate system: {mesh.coord_system}')

        logger.info(
            f"Writing dust temperature to {fpath}: {ncells} cells, nspec={nspec_i}, "
            f"T_range=[{float(np.min(temp)):.1f}, {float(np.max(temp)):.1f}] K"
        )

        # Write binary format: header [iformat=1, prec=8, ncells, nspec], then data
        with open(fpath, 'wb') as f:
            # Write header: format=1, precision=8 (float64), ncells, nspec
            header = np.array([1, 8, ncells, nspec_i], dtype=np.int64)
            header.tofile(f)

            # Write temperature data
            if temp.ndim == 3:
                if temp_flat is None:
                    raise ValueError('Internal error: missing flattened temperature array')
                # Write same temperature for all species
                for _ in range(nspec_i):
                    temp_flat.astype(np.float64).tofile(f)
            else:
                # Write each species separately
                for ispec in range(nspec_i):
                    temp_s = temp[ispec, ...]

                    if mesh.coord_system == 'spherical':
                        temp_s = transpose_to_axis_order(
                            np.asarray(temp_s),
                            from_order=mesh.axis_names(),
                            to_order=('phi', 'theta', 'r'),
                        )
                        temp_s_flat = np.asarray(temp_s).flatten(order='C')
                    elif mesh.coord_system == 'cartesian':
                        temp_s = transpose_to_axis_order(
                            np.asarray(temp_s),
                            from_order=mesh.axis_names(),
                            to_order=('x', 'y', 'z'),
                        )
                        temp_s_flat = np.asarray(temp_s).flatten(order='F')
                    else:
                        raise ValueError(f'Unsupported coordinate system: {mesh.coord_system}')

                    if int(temp_s_flat.size) != ncells:
                        raise ValueError(
                            f"dust_temperature ncells mismatch: got {int(temp_s_flat.size)}, expected {ncells}"
                        )

                    temp_s_flat.astype(np.float64).tofile(f)

        self.written_files[fpath.name] = fpath
        logger.info(f"Wrote {fpath}")

    def write_number_density(
        self,
        molecule: str,
        number_density: Quantity,
        output_dir: str | Path = '.',
        binary: bool = True,
    ) -> None:
        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'molecule')

        mol_lower = str(molecule).lower()
        n_dens = number_density.to('cm^-3').magnitude

        mesh = self.model.mesh
        if mesh is None:
            raise ValueError('Model has no mesh defined')

        if mesh.coord_system == 'spherical':
            n_dens = transpose_to_axis_order(
                np.asarray(n_dens),
                from_order=mesh.axis_names(),
                to_order=('phi', 'theta', 'r'),
            )
        elif mesh.coord_system != 'cartesian':
            raise ValueError(f'Unsupported coordinate system: {mesh.coord_system}')

        ncells = int(n_dens.size)

        if binary:
            fpath = output_dir / f'numberdens_{mol_lower}.binp'
            with open(fpath, 'wb') as f:
                header = np.array([1, 8, ncells], dtype=np.int64)
                header.tofile(f)
                n_dens.flatten().astype(np.float64).tofile(f)
        else:
            fpath = output_dir / f'numberdens_{mol_lower}.inp'
            with open(fpath, 'w') as f:
                f.write('1\n')
                f.write(f'{ncells}\n')
                n_dens.flatten().tofile(f, sep='\n')
                f.write('\n')

        self.written_files[fpath.name] = fpath
        logger.info(f'Wrote {fpath}')

    def write_microturbulence(
        self,
        vturb: Optional[Quantity | str | Path] = None,
        output_dir: str | Path = '.',
    ) -> None:
        """Write RADMC-3D microturbulent linewidth to ``microturbulence.binp``.

        If ``vturb`` is omitted, ``model.gas["microturbulence"]`` is used.
        Values are written in cm/s.
        """
        if isinstance(vturb, (str, Path)) and output_dir == '.':
            output_dir = vturb
            vturb = None

        if vturb is None:
            if self.model.gas is None or MICROTURBULENCE_FIELD not in self.model.gas:
                raise KeyError(
                    "Cannot infer microturbulence field; "
                    "model.gas['microturbulence'] is missing"
                )
            vturb = self.model.gas[MICROTURBULENCE_FIELD].data

        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'gas')

        vturb_cgs = vturb.to('cm/s').magnitude

        mesh = self.model.mesh
        if mesh is None:
            raise ValueError('Model has no mesh defined')
        if mesh.coord_system == 'spherical':
            vturb_cgs = transpose_to_axis_order(
                np.asarray(vturb_cgs),
                from_order=mesh.axis_names(),
                to_order=('phi', 'theta', 'r'),
            )
        elif mesh.coord_system != 'cartesian':
            raise ValueError(f'Unsupported coordinate system: {mesh.coord_system}')

        vturb_flat = np.asarray(vturb_cgs, dtype=np.float64).flatten()
        ncells = int(vturb_flat.size)

        fpath = output_dir / 'microturbulence.binp'
        logger.info(f"Writing microturbulence (binary) to {fpath}: {ncells} cells")
        with open(fpath, 'wb') as f:
            np.array([1, 8, ncells], dtype=np.int64).tofile(f)
            vturb_flat.astype(np.float64).tofile(f)

        self.written_files[fpath.name] = fpath
        logger.info(f"Wrote {fpath}")
        
    def write_gas_velocity(
        self,
        vr: Optional[Quantity | str | Path] = None,
        vtheta: Optional[Quantity] = None,
        vphi: Optional[Quantity] = None,
        output_dir: str | Path = '.',
        binary: bool = True,
    ) -> None:
        """Write gas velocity to gas_velocity.binp or gas_velocity.inp.
        
        Parameters
        ----------
        vr : Quantity or str or Path, optional
            Radial velocity component, shape (nx, ny, nz). If omitted, this is
            read from ``self.model.gas['vr']``. For convenience,
            ``write_gas_velocity(output_dir)`` is also accepted.
        vtheta : Quantity, optional
            Theta velocity component, shape (nx, ny, nz). If omitted, this is
            read from ``self.model.gas['vtheta']``.
        vphi : Quantity, optional
            Phi velocity component, shape (nx, ny, nz). If omitted, this is
            read from ``self.model.gas['vphi']``.
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
        DiskBridge stores fields in the canonical mesh.axis_names() order.
        We reorder explicitly before writing.
        """
        if (
            isinstance(vr, (str, Path))
            and vtheta is None
            and vphi is None
            and output_dir == '.'
        ):
            output_dir = vr
            vr = None

        mesh = self.model.mesh
        if mesh is None:
            raise ValueError('Model has no mesh defined')

        if vr is None and vtheta is None and vphi is None:
            if self.model.gas is None:
                raise ValueError(
                    "Model has no gas submodel; cannot infer gas velocity fields"
                )
            if mesh.coord_system == "spherical":
                component_names = ("vr", "vtheta", "vphi")
            elif mesh.coord_system == "cartesian":
                component_names = ("vx", "vy", "vz")
            else:
                raise ValueError(f'Unsupported coordinate system: {mesh.coord_system}')
            missing = [
                name
                for name in component_names
                if name not in self.model.gas
            ]
            if missing:
                raise KeyError(
                    "Cannot infer gas velocity fields from model.gas; missing "
                    + ", ".join(repr(name) for name in missing)
                )
            vr = self.model.gas[component_names[0]].data
            vtheta = self.model.gas[component_names[1]].data
            vphi = self.model.gas[component_names[2]].data
        elif vr is None or vtheta is None or vphi is None:
            raise ValueError(
                "Provide all three velocity components, "
                "or omit all of them to use model.gas fields"
            )

        base_dir = Path(output_dir)
        output_dir = self._get_output_dir(base_dir, 'gas')
        
        # Convert to cm/s and drop units so we work with plain floats
        vr_cgs = vr.to('cm/s').magnitude
        vtheta_cgs = vtheta.to('cm/s').magnitude
        vphi_cgs = vphi.to('cm/s').magnitude
        
        vectors = self.flatten_vector_to_radmc_order(
            mesh,
            (np.asarray(vr_cgs), np.asarray(vtheta_cgs), np.asarray(vphi_cgs)),
        )
        ncells = vectors.shape[0]
        
        if binary:
            fpath = output_dir / 'gas_velocity.binp'
            logger.info(f"Writing gas velocity (binary) to {fpath}: {ncells} cells")
            
            with open(fpath, 'wb') as f:
                # Write header: format, precision, ncells
                np.array([1], dtype=np.int64).tofile(f)  # Format
                np.array([8], dtype=np.int64).tofile(f)  # Precision (8 bytes for float64)
                np.array([ncells], dtype=np.int64).tofile(f)  # Number of cells
                
                vectors.tofile(f)
        else:
            fpath = output_dir / 'gas_velocity.inp'
            logger.info(f"Writing gas velocity (ASCII) to {fpath}: {ncells} cells")
            
            with open(fpath, 'w') as f:
                f.write('1\n')  # Format number
                f.write(f'{ncells}\n')
                for i in range(ncells):
                    f.write(
                        f'{vectors[i, 0]:.6e} {vectors[i, 1]:.6e} {vectors[i, 2]:.6e}\n'
                    )
        
        # Track the written file
        self.written_files[fpath.name] = fpath
        logger.info(f"Wrote {fpath}")
