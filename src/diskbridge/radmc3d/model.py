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

if TYPE_CHECKING:
    from diskbridge.model.model import Model

from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from .data import RadData


# Physical constants (CGS)
C_LIGHT = 2.99792458e10  # cm/s
M_H = 1.6735575e-24  # g
MU_H = 1.4  # mass per H nucleus in units of m_H
SIGMA_SB = 5.670374419e-5  # Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)

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
        
        # Storage for computed fields
        self.temperature: Optional[Quantity] = None
        self.chi: Optional[Quantity] = None  # UV field in Draine units
        self.nH: Optional[Quantity] = None  # H nuclei number density
        
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
    
    def read_dust_temperature(self, ispec: int = 0) -> Quantity:
        """Read dust temperature from RADMC-3D output using radmc3dData.
        
        Parameters
        ----------
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
        self.temperature = self.data.readDustTemp(ispec=ispec)
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
    
    def compute_chi_from_mean_intensity(
        self,
        lam_cm: Optional[np.ndarray] = None,
        J_lambda: Optional[np.ndarray] = None,
    ) -> Quantity:
        """Compute UV field from mean intensity.
        
        Integrates mean intensity over UV wavelengths (91.2-205 nm) to get chi.
        
        Parameters
        ----------
        lam_cm : np.ndarray, optional
            Wavelengths in cm (if None, reads from file)
        J_lambda : np.ndarray, optional
            Mean intensity (if None, reads from file)
            
        Returns
        -------
        Quantity
            UV field in Draine units (dimensionless)
        """
        if lam_cm is None or J_lambda is None:
            lam_cm, J_lambda = self.read_mean_intensity_uv()
        
        # UV wavelength range: 91.2-205 nm
        uv_mask = (lam_cm >= 91.2e-7) & (lam_cm <= 205e-7)
        
        if not np.any(uv_mask):
            raise ValueError(
                'No UV wavelengths (91.2-205 nm) found in mean intensity file. '
                'Run mcmono with UV wavelengths.'
            )
        
        lam_uv = lam_cm[uv_mask]
        J_uv = J_lambda[..., uv_mask]
        
        # Radiation energy density: u_lambda = 4π J_lambda / c
        u_lambda = 4.0 * np.pi * J_uv / C_LIGHT
        
        # Integrate over UV wavelengths
        chi_array = np.trapz(u_lambda, lam_uv, axis=-1)
        
        self.chi = Quantity(chi_array, 'dimensionless')
        
        logger.info(f"Computed chi from mean intensity: "
                   f"min={np.min(chi_array):.2e}, max={np.max(chi_array):.2e}")
        
        return self.chi
    
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
        nH_cgs = rho_gas.magnitude / (MU_H * M_H)
        
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
    
    def compute_pinte2018_abundance(
        self,
        molecule: str = 'co',
        X0: float = 5e-5,
        eps: float = EPS_DEFAULT,
        Tfrz: float = T_FRZ_DEFAULT,
        apply_photodissociation: bool = True,
        apply_freezeout: bool = True,
        apply_photodesorption: bool = False,
        write_output: bool = True,
    ) -> Tuple[Quantity, Quantity]:
        """Compute molecular abundance using Pinte et al. (2018) prescription.
        
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
            Freeze-out survival fraction (default: 8e-5)
        Tfrz : float, optional
            Freeze-out temperature threshold in K (default: 21.0)
        apply_photodissociation : bool, optional
            Apply photodissociation (default: True)
        apply_freezeout : bool, optional
            Apply freeze-out (default: True)
        apply_photodesorption : bool, optional
            Apply photodesorption escape (default: False)
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
        # Ensure we have temperature
        if self.temperature is None:
            self.read_temperature()
        
        # Ensure we have nH
        if self.nH is None:
            self.compute_nH_from_model()
        
        # Ensure we have chi from mcmono
        if self.chi is None:
            self.compute_chi_from_mean_intensity()
        
        # Get arrays in correct units
        T = self.temperature.to('K').magnitude
        nH = self.nH.to('cm^-3').magnitude
        chi = self.chi.magnitude
        
        # Initialize abundance
        X = np.full_like(T, float(X0), dtype=float)
        
        # 1. Photodissociation (kill molecule)
        if apply_photodissociation:
            mask_pdiss = (np.log10(chi / (nH + 1e-99)) > LOG_CHI_OVER_NH_PDISS)
            X[mask_pdiss] = 0.0
            n_pdiss = np.sum(mask_pdiss)
            logger.info(f"Photodissociation: {n_pdiss} cells ({100*n_pdiss/X.size:.1f}%)")
        
        # 2. Photodesorption escape (optional)
        if apply_photodesorption:
            mask_pdes = (np.log10(chi / (nH + 1e-99)) > LOG_CHI_OVER_NH_PDES)
            n_pdes = np.sum(mask_pdes)
            logger.info(f"Photodesorption: {n_pdes} cells ({100*n_pdes/X.size:.1f}%)")
        else:
            mask_pdes = np.zeros_like(T, dtype=bool)
        
        # 3. Freeze-out (reduce abundance)
        if apply_freezeout:
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
        
        if binary:
            filepath = self.model_dir / f'numberdens_{mol_lower}.binp'
            
            # RADMC-3D expects (nsec, ncol, nrad) order
            # DiskBridge uses (nrad, nsec, ncol/nz) order
            # Need to transpose: (nrad, nsec, ncol) -> (nsec, ncol, nrad)
            n_radmc = np.transpose(n_cgs, (1, 2, 0))
            
            with open(filepath, 'wb') as f:
                # Header: format, precision, ncells
                header = np.array([1, 8, nrad * nsec * ncol], dtype=np.int64)
                header.tofile(f)
                n_radmc.flatten().astype(np.float64).tofile(f)
            
            logger.info(f"Wrote {filepath}")
        else:
            filepath = self.model_dir / f'numberdens_{mol_lower}.inp'
            
            # Transpose for RADMC-3D
            n_radmc = np.transpose(n_cgs, (1, 2, 0))
            
            with open(filepath, 'w') as f:
                f.write('1\n')
                f.write(f'{ncol} {nrad} {nsec}\n')
                n_radmc.ravel(order='C').tofile(f, sep='\n')
                f.write('\n')
            
            logger.info(f"Wrote {filepath}")
