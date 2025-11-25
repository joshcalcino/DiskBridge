"""RADMC-3D model wrapper for molecular line radiative transfer.

This module provides the RADMC3DModel class that wraps a DiskBridge Model
and provides high-level operations for RADMC-3D workflows:
- Reading RADMC-3D output files (via radmc3dData)
- Computing UV fields and photochemistry
- Computing molecular abundances with photodissociation/freeze-out
- Writing molecular number density files

This class does NOT handle model building or writing RADMC-3D input files -
those are handled by RADMC3DWriter. This maintains clean separation between
input (writer) and output (data, model) functionality.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple
from pathlib import Path
import numpy as np
import subprocess
import shutil
import datetime

if TYPE_CHECKING:
    from diskbridge.model.model import Model

from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from .data import RadData
from .utils import _extract_radmc_errors, create_symlinks_for_file_map, cleanup_symlinks
import diskbridge 

# Physical constants (CGS)
C_LIGHT = 2.99792458e10  # cm/s
M_H = 1.6735575e-24  # g
MU_H = 2.3  # mean molecular weight for H2 + He (protoplanetary disk standard)
SIGMA_SB = 5.670374419e-5  # Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)
U_DRAINE = 9.0e-14

# Default Pinte+18 thresholds
T_FRZ_DEFAULT = 21.0  # K
EPS_DEFAULT = 8e-5  # freeze-out survival fraction
LOG_CHI_OVER_NH_PDISS = -6.0  # photodissociation threshold
LOG_CHI_OVER_NH_PDES = -7.0  # photodesorption threshold


class RadModel:
    """High-level wrapper for RADMC-3D molecular line radiative transfer.
    
    This class provides a high-level interface for RADMC-3D workflows by
    wrapping a DiskBridge Model and a radmc3dData reader. It focuses on:
    - Reading RADMC-3D output (via radmc3dData)
    - Computing UV fields from mean intensity
    - Applying photochemistry prescriptions (Pinte et al. 2018)
    - Writing molecular number density files
    
    This class does NOT build models or write RADMC-3D input files - use
    RADMC3DWriter for that. This separation mirrors the radmc3dPy structure
    where data (reading), setup (writing), and models (high-level) are separate.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance with mesh and gas data
    model_dir : str or Path, optional
        Directory containing RADMC-3D files (default: current directory)
        
    Attributes
    ----------
    model : Model
        Reference to the DiskBridge model
    data : radmc3dData
        Data reader for RADMC-3D output files
    model_dir : Path
        Directory containing RADMC-3D files
    temperature : Quantity or None
        Temperature field from RADMC-3D
    chi : Quantity or None
        UV field in Draine units
    nH : Quantity or None
        H nuclei number density
        
    Examples
    --------
    >>> import diskbridge
    >>> from diskbridge.radmc3d import RadModel
    >>> 
    >>> # Load a model
    >>> model = diskbridge.load_model('data/')
    >>> 
    >>> # Create RADMC3D model wrapper
    >>> radmc = RadModel(model, model_dir='.')
    >>> 
    >>> # Read temperature from RADMC-3D output
    >>> radmc.read_temperature()
    >>> 
    >>> # Apply Pinte 2018 photodissociation for CO
    >>> radmc.compute_pinte2018_abundance(
    ...     molecule='co',
    ...     X0=5e-5,
    ...     write_output=True
    ... )
    """
    
    def __init__(self, model: 'Model', model_dir: str | Path = '.'):
        """Initialize RADMC3DModel.
        
        Parameters
        ----------
        model : Model
            DiskBridge Model instance
        model_dir : str or Path, optional
            Directory with RADMC-3D files (default: '.')
        """
        self.model = model
        self.model_dir = Path(model_dir)
        
        # Initialize data reader
        self.data = RadData(model, model_dir)
        self.params = diskbridge.params
        
        # Storage for computed fields
        self.temperature: Optional[Quantity] = None
        self.chi: Optional[Quantity] = None  # UV field in Draine units
        self.nH: Optional[Quantity] = None  # H nuclei number density
        
        # Subdirectory paths for organized file storage
        self.inputs_dir = self.model_dir / 'radmc3d_inputs'
        self.outputs_dir = self.model_dir / 'radmc3d_outputs'
        
        # Track active symlinks for cleanup
        self._active_symlinks: list[Path] = []
        
    def _to_cgs(self, quantity: Quantity) -> np.ndarray:
        """Convert Pint Quantity to CGS magnitude array.
        
        Parameters
        ----------
        quantity : Quantity
            Pint Quantity to convert
            
        Returns
        -------
        np.ndarray
            NumPy array of CGS values
        """
        return quantity.to_base_units().magnitude
    
    def create_symlinks(self) -> None:
        """Create symlinks to organized input files in the model directory.
        
        RADMC-3D expects input files in the directory where it runs. This method
        creates symlinks from the model directory to the organized subdirectories.
        """
        # Define files to symlink from each subdirectory
        # All input files come from the inputs directory
        input_files = [
            'amr_grid.inp', 'wavelength_micron.inp',
            'stars.inp', 'dustopac.inp',
            'dust_density.binp', 'dust_density.inp',
            'gas_velocity.binp', 'gas_velocity.inp',
            'radmc3d.inp'
        ]
        
        file_map = {self.inputs_dir: input_files}

        if self.inputs_dir.exists():
            opacity_files = list(self.inputs_dir.glob('dustkappa_*.inp'))
            for opac_file in opacity_files:
                file_map[self.inputs_dir].append(opac_file.name)

        create_symlinks_for_file_map(self.model_dir, file_map, self._active_symlinks)
    
    def cleanup_symlinks(self) -> None:
        """Remove all symlinks created by create_symlinks().
        
        This ensures the model directory stays clean after RADMC-3D runs.
        Only removes symlinks that were tracked by this instance.
        """
        cleanup_symlinks(self._active_symlinks)
    
    def read_gas_temperature(self) -> Quantity:
        """Read gas_temperature file using radmc3dData.
        
        Returns
        -------
        Quantity
            Temperature field in Kelvin with shape matching model mesh
            
        Raises
        ------
        FileNotFoundError
            If gas_temperature file is not found
        """
        self.temperature = self.data.readGasTemp()
        return self.temperature
    
    def read_dust_temperature(self, fname: Optional[str | Path] = None, ispec: int = 0) -> Quantity:
        """Read dust temperature from RADMC-3D output using radmc3dData.
        
        Parameters
        ----------
        fname : str or Path, optional
            Path to temperature file (if None, uses default in model_dir)
        ispec : int, optional
            Dust species index to read (default: 0)
            
        Returns
        -------
        Quantity
            Temperature field in Kelvin
            
        Raises
        ------
        FileNotFoundError
            If no dust temperature file is found
        """
        self.temperature = self.data.readDustTemp(fname=fname, ispec=ispec)
        return self.temperature
    
    def read_temperature(self, source: str = 'auto', ispec: int = 0) -> Quantity:
        """Read temperature from RADMC-3D output.
        
        Parameters
        ----------
        source : str, optional
            Temperature source: 'auto', 'gas', or 'dust' (default: 'auto')
        ispec : int, optional
            Dust species for dust temperature (default: 0)
            
        Returns
        -------
        Quantity
            Temperature field in Kelvin
        """
        if source == 'auto':
            # Try gas first, then dust
            if (self.model_dir / 'gas_temperature.inp').exists():
                return self.read_gas_temperature()
            else:
                return self.read_dust_temperature(ispec=ispec)
        elif source == 'gas':
            return self.read_gas_temperature()
        elif source == 'dust':
            return self.read_dust_temperature(ispec=ispec)
        else:
            raise ValueError(f"Invalid temperature source: {source}")
    
    def read_mean_intensity_uv(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read mean intensity from RADMC-3D mcmono output.
        
        Returns
        -------
        lam_cm : np.ndarray
            Wavelengths in cm
        J_lambda : np.ndarray
            Mean intensity [erg/s/cm^2/Hz/sr] with shape matching grid
            
        Raises
        ------
        FileNotFoundError
            If no mean intensity file found
        """
        # Try different file names
        candidates = [
            'mean_intensity_lambda.binp',
            'mean_intensity_lambda.out',
            'meanint_lambda.binp',
            'meanint_lambda.out',
            'mean_intensity.out',
        ]
        
        for filename in candidates:
            filepath = self.model_dir / filename
            if not filepath.exists():
                continue
            
            if filepath.suffix == '.binp':
                # Binary format
                with open(filepath, 'rb') as f:
                    try:
                        # Read header
                        hdr_i = np.fromfile(f, dtype='>i4', count=1)
                        if hdr_i.size == 0:
                            continue
                        iformat = int(hdr_i[0])
                        dims = np.fromfile(f, dtype='>i4', count=4)
                        nx, ny, nz, nw = map(int, dims)
                        
                        # Read wavelengths (in microns)
                        lam_micron = np.fromfile(f, dtype='>f8', count=nw)
                        
                        # Read data
                        data = np.fromfile(f, dtype='>f8', count=nx*ny*nz*nw)
                    except Exception as e:
                        logger.warning(f"Failed to read binary file {filepath}: {e}")
                        continue
                
                try:
                    J = data.reshape((nx, ny, nz, nw))
                    lam_cm = lam_micron * 1e-4  # micron to cm
                    
                    logger.info(f"Read mean intensity from {filepath}: "
                               f"grid=({nx},{ny},{nz}), nwave={nw}")
                    return lam_cm, J
                except Exception as e:
                    logger.warning(f"Failed to reshape data from {filepath}: {e}")
                    continue
            else:
                # ASCII format
                with open(filepath, 'r') as f:
                    iformat = int(f.readline().strip())
                    
                    if iformat == 1:
                        # Format 1: grid dims on one line
                        try:
                            head = f.readline().split()
                            nx, ny, nz, nw = map(int, head)
                            lam_micron = np.fromfile(f, count=nw, sep='\n')
                            data = np.fromfile(f, count=nx*ny*nz*nw, sep='\n')
                            
                            J = data.reshape((nx, ny, nz, nw))
                            lam_cm = lam_micron * 1e-4
                            
                            logger.info(f"Read mean intensity from {filepath}: "
                                       f"grid=({nx},{ny},{nz}), nwave={nw}")
                            return lam_cm, J
                        except Exception as e:
                            logger.warning(f"Failed to read ASCII file format 1 {filepath}: {e}")
                            continue
                            
                    elif iformat == 2:
                        # Format 2: ncells and nwav on separate lines
                        # Wavelengths come from separate file, frequencies in this file
                        try:
                            ncells = int(f.readline().strip())
                            nw = int(f.readline().strip())
                            
                            # Read frequencies (ignore them, use wavelengths from mcmono file)
                            frequencies = np.fromfile(f, count=nw, sep='\n')
                            
                            # Read wavelengths from mcmono_wavelength_micron.inp
                            wl_file = self.model_dir / 'mcmono_wavelength_micron.inp'
                            if not wl_file.exists():
                                raise FileNotFoundError(f"mcmono_wavelength_micron.inp not found")
                            
                            with open(wl_file, 'r') as wf:
                                nw_file = int(wf.readline().strip())
                                if nw_file != nw:
                                    raise ValueError(f"Wavelength count mismatch: {nw_file} vs {nw}")
                                lam_micron = np.fromfile(wf, count=nw, sep='\n')
                            
                            # Read intensity data
                            data = np.fromfile(f, count=ncells*nw, sep='\n')
                            
                            # Get mesh shape from model
                            mesh = self.model.mesh
                            if mesh.coord_system == 'spherical':
                                ncol = len(mesh.axes['theta'].centers)
                                nrad = len(mesh.axes['r'].centers)
                                nsec = len(mesh.axes['phi'].centers)
                            else:
                                raise ValueError(f"Unsupported coordinate system for format 2: {mesh.coord_system}")
                            
                            # Reshape: RADMC-3D uses Fortran order (column-major)
                            J = data.reshape((nw, nrad, ncol, nsec), order='F')
                            # Transpose to match DiskBridge data order (nrad, ncol, nsec, nw)
                            # This matches the temperature order from readDustTemp: (nr, ntheta, nphi)
                            J = np.transpose(J, (1, 2, 3, 0))
                            
                            lam_cm = lam_micron * 1e-4
                            
                            logger.info(f"Read mean intensity from {filepath}: "
                                       f"grid=({nrad},{ncol},{nsec}), nwave={nw}")
                            return lam_cm, J
                        except Exception as e:
                            logger.warning(f"Failed to read ASCII file format 2 {filepath}: {e}")
                            continue
                    else:
                        logger.warning(f"Unsupported format {iformat} in {filepath}")
                        continue
        
        raise FileNotFoundError(
            "No mean intensity file found. Run 'radmc3d mcmono' with UV wavelengths first."
        )
    
    def compute_nH_from_model(self) -> Quantity:
        """Compute H nuclei number density from model gas density.
        
        Returns
        -------
        Quantity
            Number density of H nuclei in cm^-3
            
        Raises
        ------
        KeyError
            If model has no density field
        """
        if 'density' not in self.model.gas:
            raise KeyError("Model has no gas density field")
        
        # Get gas density in CGS
        rho_gas = self.model.gas['density'].data.to('g/cm^3')
        MU_HNUC = 1.4  # mass per H nucleus including He
        nH_cgs = rho_gas.magnitude / (MU_HNUC * M_H)
        
        # Model gas density has axis_order ('r', 'phi', 'theta') for spherical coords
        # but RADMC-3D/RadData uses ('r', 'theta', 'phi')
        # Need to transpose if necessary
        field = self.model.gas['density']
        if hasattr(field, 'axis_order'):
            axis_order = field.axis_order
            # For spherical: convert (r, phi, theta) -> (r, theta, phi)
            if axis_order == ('r', 'phi', 'theta'):
                nH_cgs = np.transpose(nH_cgs, (0, 2, 1))  # (r, phi, theta) -> (r, theta, phi)
        
        self.nH = Quantity(nH_cgs, 'cm^-3')
        
        logger.info(f"Computed nH from gas density: "
                   f"min={np.min(nH_cgs):.2e} cm^-3, max={np.max(nH_cgs):.2e} cm^-3")
        
        return self.nH
    
    def compute_abundance(
        self,
        molecule: str = 'co',
        X0: float = 5e-5,
        eps: float = EPS_DEFAULT,
        Tfrz: float = T_FRZ_DEFAULT,
        photodissociation: Optional[bool] = None,
        freezeout: Optional[bool] = None,
        photodesorption: Optional[bool] = None,
        write_output: bool = True,
    ) -> Tuple[Quantity, Quantity]:
        """Compute molecular abundance with photochemistry (Pinte et al. 2018).
        
        Applies three processes:
        1. Photodissociation: X = 0 where log10(chi/nH) > -6
        2. Freeze-out: X *= eps where T < Tfrz
        3. Photodesorption (optional): skip freeze-out where log10(chi/nH) > -7
        
        Parameters
        ----------
        molecule : str, optional
            Molecule name (default: 'co')
        X0 : float, optional
            Initial abundance X/nH (default: 5e-5)
        eps : float, optional
            Depletion factor for freeze-out (default: 1e-3)
        Tfrz : float, optional
            Freeze-out temperature in K (default: 20 K)
        photodissociation : bool, optional
            Apply photodissociation (default: True)
        freezeout : bool, optional
            Apply freeze-out (default: True)
        photodesorption : bool, optional
            Apply photodesorption (default: False)
        write_output : bool, optional
            Write numberdens file (default: True)
            
        Returns
        -------
        X : Quantity
            Abundance X/nH (dimensionless)
        n_mol : Quantity
            Number density in cm^-3
            
        References
        ----------
        Pinte et al. (2018), A&A 609, A47
        """
        # Resolve photochemistry flags from params if not explicitly set
        if photodissociation is None:
            photodissociation = self.params.photodissociation
        if freezeout is None:
            freezeout = self.params.freezeout
        if photodesorption is None:
            photodesorption = self.params.photodesorption

        # Ensure we have temperature
        if self.temperature is None:
            self.read_temperature()
        
        # Ensure we have nH
        if self.nH is None:
            self.compute_nH_from_model()
        
        # Ensure we have chi from mcmono only if needed
        # chi is required for photodissociation and photodesorption,
        # but not for pure freeze-out cases.
        chi = None
        if photodissociation or photodesorption:
            if self.chi is None:
                self.compute_mcmono()
            chi = self.chi.magnitude
        
        # Get arrays in correct units
        T = self.temperature.to('K').magnitude
        nH = self.nH.to('cm^-3').magnitude
        
        # Initialize abundance
        X = np.full_like(T, float(X0), dtype=float)
        
        # 1. Photodissociation (kill molecule)
        if photodissociation:
            mask_pdiss = (np.log10(chi / (nH + 1e-99)) > LOG_CHI_OVER_NH_PDISS)
            X[mask_pdiss] = 0.0
            n_pdiss = np.sum(mask_pdiss)
            logger.info(f"Photodissociation: {n_pdiss} cells ({100*n_pdiss/X.size:.1f}%)")
        
        # 2. Photodesorption escape (optional)
        if photodesorption:
            mask_pdes = (np.log10(chi / (nH + 1e-99)) > LOG_CHI_OVER_NH_PDES)
            n_pdes = np.sum(mask_pdes)
            logger.info(f"Photodesorption: {n_pdes} cells ({100*n_pdes/X.size:.1f}%)")
        else:
            mask_pdes = np.zeros_like(T, dtype=bool)
        
        # 3. Freeze-out (reduce abundance)
        if freezeout:
            mask_frz = (T < Tfrz) & ~mask_pdes  # Don't freeze where photodesorption occurs
            X[mask_frz] *= eps
            n_frz = np.sum(mask_frz)
            logger.info(f"Freeze-out: {n_frz} cells ({100*n_frz/X.size:.1f}%)")
        
        # Compute number density
        n_mol = X * nH
        
        # Store as Quantities
        X_qty = Quantity(X, 'dimensionless')
        n_mol_qty = Quantity(n_mol, 'cm^-3')
        
        logger.info(f"Computed {molecule} abundance: "
                   f"X_mean={np.mean(X):.2e}, n_mean={np.mean(n_mol):.2e} cm^-3")
        
        # Write output file
        if write_output:
            self.write_numberdens(molecule, n_mol_qty)
        
        return X_qty, n_mol_qty
    
    def write_numberdens(
        self,
        molecule: str,
        number_density: Quantity,
        binary: bool = True,
    ) -> None:
        """Write numberdens file for RADMC-3D.
        
        Parameters
        ----------
        molecule : str
            Molecule name (lowercase)
        number_density : Quantity
            Number density in cm^-3
        binary : bool, optional
            Write binary format (default: True)
        """
        mol_lower = molecule.lower()
        
        # Convert to CGS
        n_cgs = number_density.to('cm^-3').magnitude
        
        # Get mesh shape
        mesh = self.model.mesh
        if mesh.coord_system == 'spherical':
            ncol = len(mesh.axes['theta'].centers)
            nrad = len(mesh.axes['r'].centers)
            nsec = len(mesh.axes['phi'].centers)
        elif mesh.coord_system == 'cylindrical':
            nrad = len(mesh.axes['r'].centers)
            nsec = len(mesh.axes['phi'].centers)
            ncol = len(mesh.axes['z'].centers)
        else:
            raise ValueError(f"Unsupported coordinate system: {mesh.coord_system}")
        
        # Create inputs_dir if it doesn't exist
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        
        n_radmc = np.transpose(n_cgs, (2, 1, 0))
        if binary:
            filepath = self.inputs_dir / f'numberdens_{mol_lower}.binp'
            
            with open(filepath, 'wb') as f:
                # Header: format, precision, ncells
                header = np.array([1, 8, nrad * nsec * ncol], dtype=np.int64)
                header.tofile(f)
                n_radmc.flatten().astype(np.float64).tofile(f)
            
            logger.info(f"Wrote {filepath}")
        else:
            filepath = self.inputs_dir / f'numberdens_{mol_lower}.inp'
            
            with open(filepath, 'w') as f:
                f.write('1\n')
                f.write(f'{ncol} {nrad} {nsec}\n')
                n_radmc.ravel(order='C').tofile(f, sep='\n')
                f.write('\n')
            
            logger.info(f"Wrote {filepath}")
    
    def compute_temperature(
        self,
        nphot: int = None,
        output_dir: Optional[str | Path] = None,
        force: bool = False,
    ) -> Quantity:
        """Run RADMC-3D thermal Monte Carlo to compute dust temperature.
        
        Parameters
        ----------
        nphot : int, optional
            Number of photon packages (uses nphot_thermal from params if None)
        output_dir : str or Path, optional
            Output directory (default: 'temperature/')
        force : bool, optional
            Force recomputation even if output exists (default: False)
            
        Returns
        -------
        Quantity
            Dust temperature field in K
            
        Examples
        --------
        >>> rad = RadModel(model)
        >>> temperature = rad.compute_temperature()  # Uses params defaults
        >>> temperature = rad.compute_temperature(nphot=1000000, force=True)
        """
        if nphot is None:
            nphot = int(self.params.nphot_thermal)
        
        # Set output directory
        if output_dir is None:
            output_dir = self.outputs_dir
        else:
            output_dir = Path(output_dir)
        
        # Check if already computed (check for any temperature file)
        temp_files = [output_dir / 'dust_temperature.dat', 
                     output_dir / 'dust_temperature.bdat',
                     output_dir / 'dust_temperature.binp']
        existing_file = next((f for f in temp_files if f.exists()), None)
        if existing_file is not None and not force:
            logger.info(f"Temperature already computed at {existing_file}")
            self.read_dust_temperature(fname=str(existing_file))
            return self.temperature
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Create symlinks to input files
        self.create_symlinks()
        
        try:
            # Run mctherm (RADMC-3D gets nphot and setthreads from radmc3d.inp)
            logger.info(f"Running RADMC-3D mctherm with {nphot} photons...")
            cmd = ['radmc3d', 'mctherm']
            
            result = subprocess.run(
                cmd,
                cwd=str(self.model_dir),
                capture_output=True,
                text=True
            )
            
            log_path = self.model_dir / 'radmc3d.out'
            try:
                with open(log_path, 'a') as f:
                    f.write('\n--- mctherm ---\n')
                    if result.stdout:
                        f.write(result.stdout)
                    if result.stderr:
                        f.write('\n[stderr]\n')
                        f.write(result.stderr)
            except Exception:
                pass
            
            if result.returncode != 0:
                logger.error(f"RADMC-3D mctherm failed (see {log_path})")
                errors = _extract_radmc_errors(log_path)
                if errors:
                    logger.error(f"RADMC-3D errors:\n{errors}")
                    raise RuntimeError(f"mctherm failed with RADMC-3D errors:\n{errors}")
                raise RuntimeError("mctherm failed")
            
            logger.info(f"mctherm completed (log written to {log_path})")
        finally:
            # Clean up symlinks
            self.cleanup_symlinks()
        
        # Organize output
        self._organize_output(
            output_dir,
            ['dust_temperature.dat', 'dust_temperature.bdat', 'dust_temperature.binp'],
            f'radmc3d mctherm nphot={nphot}'
        )
        
        # Read temperature from the output directory
        # Find the temperature file that was moved
        for suffix in ['.bdat', '.dat', '.binp']:
            temp_file = output_dir / f'dust_temperature{suffix}'
            if temp_file.exists():
                self.read_dust_temperature(fname=str(temp_file))
                break
        
        return self.temperature
    
    def compute_mcmono(
        self,
        nphot: int = None,
        output_dir: Optional[str | Path] = None,
        force: bool = False,
        uv_min_nm: float = None,
        uv_max_nm: float = None,
        n_wavelengths: int = None
    ) -> Quantity:
        """Run RADMC-3D monochromatic Monte Carlo for UV field.
        
        Parameters
        ----------
        nphot : int, optional
            Number of photon packages (uses nphot_mono from params if None)
        output_dir : str or Path, optional
            Output directory (default: 'mcmono/')
        force : bool, optional
            Force recomputation even if output exists (default: False)
        uv_min_nm : float, optional
            UV range lower bound in nm (default: 91.2 nm, from params)
        uv_max_nm : float, optional
            UV range upper bound in nm (default: 205.0 nm, from params)
        n_wavelengths : int, optional
            Number of wavelengths for integration (default: 10, from params)
            
        Returns
        -------
        Quantity
            UV field chi in Draine units
            
        Examples
        --------
        >>> rad = RadModel(model)
        >>> chi = rad.compute_mcmono()
        """
        if nphot is None:
            nphot = self.params.nphot_mono
        if uv_min_nm is None:
            uv_min_nm = self.params.uv_min_nm
        if uv_max_nm is None:
            uv_max_nm = self.params.uv_max_nm
        if n_wavelengths is None:
            n_wavelengths = self.params.uv_n_wavelengths
        
        # Validate UV range
        uv_min_cm = uv_min_nm * 1e-7
        uv_max_cm = uv_max_nm * 1e-7
        u_draine = U_DRAINE
        
        logger.info(f"UV field configuration: {uv_min_nm:.1f}-{uv_max_nm:.1f} nm, {n_wavelengths} wavelengths")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.outputs_dir
        else:
            output_dir = Path(output_dir)
        
        # Check if already computed
        mean_intensity_file = output_dir / 'mean_intensity.out'
        if mean_intensity_file.exists() and not force:
            logger.info(f"Mean intensity already computed at {mean_intensity_file}")
            # Parse the existing file with multi-wavelength format
            with open(mean_intensity_file, 'r') as f:
                iformat = int(f.readline().strip())
                nrcells = int(f.readline().strip())
                nwav = int(f.readline().strip())
                
                # Read wavelengths (all on one line, space-separated, in Hz)
                freq_hz = np.array([float(x) for x in f.readline().split()])
                
                # Convert frequency (Hz) to wavelength (cm): lambda = c / nu
                lam_cm = C_LIGHT / freq_hz
                
                # Read mean intensity values
                j_flat = np.array([float(f.readline().strip()) for _ in range(nwav * nrcells)])
                j_lambda = j_flat.reshape((nrcells, nwav))
            
            nx, ny, nz = self.data._getMeshShape()
            self.mean_intensity = j_lambda
            
            # Compute chi by integrating over UV band in frequency space
            uv_mask = (lam_cm >= uv_min_cm) & (lam_cm <= uv_max_cm)
            if not np.any(uv_mask):
                raise ValueError(
                    f'No UV wavelengths ({uv_min_nm:.1f}-{uv_max_nm:.1f} nm) in existing mean intensity file. '
                    f'Wavelength range: {lam_cm.min()*1e7:.1f}-{lam_cm.max()*1e7:.1f} nm'
                )
            
            j_uv = j_lambda[:, uv_mask]

            nu_uv = freq_hz[uv_mask]
            u_nu = 4.0 * np.pi * j_uv / C_LIGHT
            sort_idx = np.argsort(nu_uv)
            nu_sorted = nu_uv[sort_idx]
            u_nu_sorted = u_nu[:, sort_idx]
            u_band = np.trapz(u_nu_sorted, nu_sorted, axis=1)
            chi_flat = u_band / u_draine
            chi_3d = chi_flat.reshape((nx, ny, nz), order='F')
            self.chi = Quantity(chi_3d, 'dimensionless')
            
            logger.info(f"Loaded chi from {np.sum(uv_mask)} UV wavelengths: "
                       f"min={np.min(chi_3d):.2e}, max={np.max(chi_3d):.2e}")
            return self.chi
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Create mcmono_wavelength_micron.inp with UV wavelength range
        mcmono_wav_file = self.model_dir / 'mcmono_wavelength_micron.inp'
        uv_lam_nm = np.linspace(uv_min_nm, uv_max_nm, n_wavelengths)
        uv_lam_micron = uv_lam_nm / 1000.0  # Convert to microns
        
        with open(mcmono_wav_file, 'w') as f:
            f.write(f'{len(uv_lam_micron)}\n')  # Number of wavelengths
            for lam in uv_lam_micron:
                f.write(f'{lam:.6f}\n')
        logger.debug(f"Wrote {mcmono_wav_file} with {len(uv_lam_micron)} UV wavelengths")
        
        # Create symlinks to input files
        self.create_symlinks()
        
        # RADMC-3D mcmono also needs temperature file - create symlink if it exists in outputs subdirectory
        for suffix in ['.bdat', '.dat', '.binp']:
            src_temp = self.outputs_dir / f'dust_temperature{suffix}'
            dst_temp = self.model_dir / f'dust_temperature{suffix}'
            if src_temp.exists() and not dst_temp.exists():
                import os
                os.symlink(src_temp, dst_temp)
                self._active_symlinks.append(dst_temp)
                logger.debug(f"Created symlink: {dst_temp} -> {src_temp}")
                break
        
        try:
            # Run mcmono at UV wavelengths
            setthreads = self.params.nbcores
            logger.info(f"Running RADMC-3D mcmono at {n_wavelengths} UV wavelengths ({uv_min_nm:.1f}-{uv_max_nm:.1f} nm)...")
            cmd = ['radmc3d', 'mcmono', 'setthreads', str(setthreads)]

            # Preserve any existing radmc3d.out log (e.g. from mctherm)
            log_path = self.model_dir / 'radmc3d.out'
            previous_log = ""
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        previous_log = f.read()
                except Exception:
                    previous_log = ""
            
            result = subprocess.run(
                cmd,
                cwd=str(self.model_dir),
                capture_output=True,
                text=True
            )

            # Rebuild radmc3d.out to contain both the previous log and the new mcmono output
            combined_log = previous_log
            combined_log += '\n--- mcmono ---\n'
            if result.stdout:
                combined_log += result.stdout
            if result.stderr:
                combined_log += '\n[stderr]\n'
                combined_log += result.stderr

            try:
                with open(log_path, 'w') as f:
                    f.write(combined_log)
            except Exception:
                pass
            
            if result.returncode != 0:
                logger.error(f"RADMC-3D mcmono failed (see {log_path})")
                errors = _extract_radmc_errors(log_path)
                if errors:
                    logger.error(f"RADMC-3D errors:\n{errors}")
                    raise RuntimeError(f"mcmono failed with RADMC-3D errors:\n{errors}")
                raise RuntimeError("mcmono failed")
            
            logger.info(f"mcmono completed (log written to {log_path})")
        finally:
            # Clean up symlinks
            self.cleanup_symlinks()
        
        # Organize output (move mean_intensity file and mcmono_wavelength_micron.inp to mcmono/ directory)
        self._organize_output(
            output_dir,
            ['mean_intensity.out', 'mcmono_wavelength_micron.inp'],
            f'radmc3d mcmono UV_range_{uv_min_nm:.1f}-{uv_max_nm:.1f}nm_{n_wavelengths}wavelengths'
        )
        
        # Read mean intensity directly from the mcmono output directory
        mean_intensity_path = output_dir / 'mean_intensity.out'
        if not mean_intensity_path.exists():
            raise FileNotFoundError(f"Mean intensity file not found at {mean_intensity_path}")
        
        # Parse the mean intensity file
        with open(mean_intensity_path, 'r') as f:
            # Format: iformat (line 1), nrcells (line 2), nwav (line 3)
            # Line 4: all wavelengths (space-separated)
            # Remaining lines: mean intensity values (one per line, nwav * nrcells total)
            nrcells = int(f.readline().strip())
            nwav = int(f.readline().strip())
            
            # Read wavelengths (all on one line, space-separated, in Hz)
            freq_hz = np.array([float(x) for x in f.readline().split()])
            
            # Convert frequency (Hz) to wavelength (cm): lambda = c / nu
            lam_cm = C_LIGHT / freq_hz
            
            # Read mean intensity values (nwav * nrcells values, one per line)
            # Data is ordered: all wavelengths for cell 0, then all for cell 1, etc.
            j_flat = np.array([float(f.readline().strip()) for _ in range(nwav * nrcells)])
            
            # Reshape to (nrcells, nwav)
            j_lambda = j_flat.reshape((nrcells, nwav))
        
        # Store wavelengths and mean intensity
        nx, ny, nz = self.data._getMeshShape()
        self.mean_intensity = j_lambda  # Keep full spectral dimension
        
        # Compute chi by integrating over UV band in frequency space
        # Following Pinte et al. 2018 and fargo2radmc3d implementation
        uv_mask = (lam_cm >= uv_min_cm) & (lam_cm <= uv_max_cm)
        
        if not np.any(uv_mask):
            raise ValueError(
                f'No UV wavelengths ({uv_min_nm:.1f}-{uv_max_nm:.1f} nm) found in mean intensity file. '
                f'Wavelength range: {lam_cm.min()*1e7:.1f}-{lam_cm.max()*1e7:.1f} nm'
            )
        
        j_uv = j_lambda[:, uv_mask]  # Shape: (nrcells, n_uv_wavelengths)

        nu_uv = freq_hz[uv_mask]
        u_nu = 4.0 * np.pi * j_uv / C_LIGHT
        sort_idx = np.argsort(nu_uv)
        nu_sorted = nu_uv[sort_idx]
        u_nu_sorted = u_nu[:, sort_idx]
        u_band = np.trapz(u_nu_sorted, nu_sorted, axis=1)
        chi_flat = u_band / u_draine  # Shape: (nrcells,)
        
        # Reshape to 3D grid
        chi_3d = chi_flat.reshape((nx, ny, nz), order='F')
        self.chi = Quantity(chi_3d, 'dimensionless')
        
        logger.info(f"Computed chi from {np.sum(uv_mask)} UV wavelengths ({uv_min_nm:.1f}-{uv_max_nm:.1f} nm): "
                   f"min={np.min(chi_3d):.2e}, max={np.max(chi_3d):.2e}")
        
        return self.chi
    
    def _organize_output(
        self,
        output_dir: Path,
        files: list[str],
        command: str
    ) -> None:
        """Organize RADMC-3D output files into a directory.
        
        Parameters
        ----------
        output_dir : Path
            Target directory
        files : list of str
            Files to move
        command : str
            Command that was run (for metadata)
        """
        # Move files
        for fname in files:
            src = self.model_dir / fname
            if src.exists():
                dst = output_dir / fname
                shutil.move(str(src), str(dst))
                logger.debug(f"Moved {fname} -> {output_dir}")
        
        # Copy params file if it exists
        params_file = self.model_dir / 'params.txt'
        if params_file.exists():
            dst_params = output_dir / 'params.txt'
            shutil.copy2(str(params_file), str(dst_params))
            
            # Append metadata
            with open(dst_params, 'a') as f:
                f.write('\n# --- Run metadata (auto-generated) ---\n')
                f.write(f'timestamp = {datetime.datetime.now().isoformat()}\n')
                f.write(f'radmc3d_command = {command}\n')
            
            logger.debug(f"Copied params.txt -> {output_dir}")
