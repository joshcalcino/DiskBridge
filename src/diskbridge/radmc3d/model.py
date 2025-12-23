"""RADMC-3D model wrapper for molecular line radiative transfer.

This module provides the RADMC3DModel class that wraps a DiskBridge Model
and provides high-level operations for RADMC-3D workflows:
- Reading RADMC-3D output files (via radmc3dData)
- Computing UV fields and photochemistry
- Computing molecular abundances with photodissociation/freeze-out
- Writing molecular number density files
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Sequence
from pathlib import Path
import numpy as np
import shutil
import datetime

if TYPE_CHECKING:
    from diskbridge.model.core import Model

from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from diskbridge.model.field import Field
from .data import RadData
from .utils import (
    _extract_radmc_errors,
    create_radmc3d_symlinks,
    cleanup_symlink_paths,
    run_radmc3d_and_log,
    _read_params_snapshot,
    _params_signature,
)
from .writer import RadWriter
import diskbridge 

# Physical constants from config
C_LIGHT = units('c')  # Speed of light
M_H = units('m_H')  # Hydrogen mass
SIGMA_SB = units('sigma_SB')  # Stefan-Boltzmann constant

from diskbridge.chemistry.constants import eps_chi, LOG_CHI_OVER_NH_PDISS

# Draine (1978) UV field constant
U_DRAINE = Quantity(9.0e-14, 'erg/cm^3')
 
_MCTHERM_PARAM_KEYS = (
    'nphot_thermal',
    'n_lambda',
    'lambda_min',
    'lambda_max',
    'scat_mode',
    'amin',
    'amax',
    'pindex',
    'dust_to_gas_ratio',
    'nbins',
    'grain_density',
    'species',
    'opacity_dir',
    'rstar',
    'teff',
    'mstar',
    'mdot',
    'accretion_fill_factor',
    'secondorder',
    'external_uv',
    'external_uv_chi',
)

_MCMONO_EXTRA_PARAM_KEYS = (
    'nphot_mono',
    'uv_min',
    'uv_max',
    'uv_n_wavelengths',
    'external_uv',
    'external_uv_chi',
)


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
    >>> # Apply Pinte 2018-style photochemistry for CO
    >>> radmc.compute_abundance(
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
        self.writer = RadWriter(model)
        self.params = diskbridge.params
        
        # Storage for computed fields
        self.temperature: Optional[Quantity] = None
        self.chi: Optional[Quantity] = None  # UV field in Draine units
        self.nH: Optional[Quantity] = None  # H nuclei number density
        self.theta_co: Optional[Quantity] = None
        self.chi_eff: Optional[Quantity] = None
        self.k_diss_co: Optional[Quantity] = None
        self.tau_diss_co: Optional[Quantity] = None
        
        # Two-phase CO chemistry fields
        self.nco_gas: Optional[Quantity] = None  # Gas-phase CO number density [cm^-3]
        self.nco_ice: Optional[Quantity] = None  # CO ice number density [cm^-3]
        
        # Subdirectory paths for organized file storage
        self.inputs_dir = self.model_dir / 'radmc3d_inputs'
        self.outputs_dir = self.model_dir / 'radmc3d_outputs'
        
        # Track active symlinks for cleanup
        self._active_symlinks: list[Path] = []
    
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
            'radmc3d.inp', 'external_source.inp',
            'numberdens_*.inp', 'numberdens_*.binp'
        ]

        create_radmc3d_symlinks(
            model_dir=self.model_dir,
            inputs_dir=self.inputs_dir,
            input_files=input_files,
            active_symlinks=self._active_symlinks,
        )
    
    def cleanup_symlinks(self) -> None:
        """Remove all symlinks created by create_symlinks().
        
        This ensures the model directory stays clean after RADMC-3D runs.
        Only removes symlinks that were tracked by this instance.
        """
        cleanup_symlink_paths(self._active_symlinks)

    def _ensure_cntdump_ge_countwrite(self, countwrite: int, nphot: int) -> None:
        radmc_inp_path = self.inputs_dir / 'radmc3d.inp'
        if not radmc_inp_path.exists():
            raise FileNotFoundError(
                f"Missing required input file: {radmc_inp_path}. "
                "Generate RADMC-3D inputs (including radmc3d.inp) before running mctherm/mcmono."
            )

        int32_max = int(np.iinfo(np.int32).max)

        countwrite = int(countwrite)
        if countwrite > int32_max:
            logger.warning(
                "countwrite too large; clamping to int32 max (%d)" % int32_max
            )
            countwrite = int32_max

        desired_cntdump = max(int(nphot), int(countwrite))
        if desired_cntdump > int32_max:
            logger.warning(
                "cntdump too large; clamping to int32 max (%d)" % int32_max
            )
            desired_cntdump = int32_max

        try:
            lines = radmc_inp_path.read_text().splitlines(True)
        except Exception as e:
            raise RuntimeError(f"Failed to read {radmc_inp_path}: {e}")

        updated_lines: list[str] = []
        saw_cntdump = False

        for line in lines:
            stripped = line.strip()
            if (not stripped) or stripped.startswith('#') or ('=' not in line):
                updated_lines.append(line)
                continue

            name, value = line.split('=', 1)
            key = name.strip()

            if key != 'cntdump':
                updated_lines.append(line)
                continue

            saw_cntdump = True
            try:
                current = int(float(value.strip()))
            except Exception:
                current = None

            if current is not None and current > int32_max:
                logger.warning(
                    "cntdump too large; clamping to int32 max (%d)" % int32_max
                )
                updated_lines.append(f'cntdump = {int32_max}\n')
            elif current is not None and current >= countwrite:
                updated_lines.append(line)
            else:
                updated_lines.append(f'cntdump = {desired_cntdump}\n')

        if not saw_cntdump:
            if updated_lines and not updated_lines[-1].endswith('\n'):
                updated_lines[-1] = updated_lines[-1] + '\n'
            updated_lines.append(f'cntdump = {desired_cntdump}\n')

        try:
            radmc_inp_path.write_text(''.join(updated_lines))
        except Exception as e:
            raise RuntimeError(f"Failed to update {radmc_inp_path}: {e}")
    
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

        # RadData returns spherical fields in RADMC-3D order (r, theta, phi).
        # DiskBridge model convention is (r, phi, theta).
        if self.model.mesh.coord_system == 'spherical':
            self.temperature = np.transpose(self.temperature, (0, 2, 1))

        # Register temperature as a Field on the model
        axis_order = ('r', 'phi', 'theta') if self.model.mesh.coord_system == 'spherical' else self.model.mesh.axis_names()
        self.model.gas_register(
            'temperature',
            Field(
                quantity='temperature',
                data=self.temperature,
                axis_order=axis_order,
            ),
        )
        
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
            'mean_intensity.bout',
            'mean_intensity.out',
        ]
        
        for filename in candidates:
            filepath = self.model_dir / filename
            if not filepath.exists():
                continue
            
            if filepath.suffix == '.bout':
                try:
                    with open(filepath, 'rb') as f:
                        hdr4 = np.fromfile(f, dtype=np.int64, count=4)
                        if hdr4.size < 4:
                            continue
                        iformat = int(hdr4[0])
                        prec = int(hdr4[1])
                        ncells = int(hdr4[2])
                        nw = int(hdr4[3])

                        freq_hz = np.fromfile(f, dtype=np.float64, count=nw) * units('Hz')
                        dtype = np.float64 if prec == 8 else np.float32
                        data = np.fromfile(f, dtype=dtype, count=ncells * nw)

                    j_flat = data.astype(np.float64, copy=False)
                    j_lambda = j_flat.reshape((nw, ncells)).T * units('erg/(s*cm^2*Hz*sr)')

                    nx, ny, nz = self.data._getMeshShape()
                    if ncells != nx * ny * nz:
                        raise ValueError(
                            f"mean_intensity.bout ncells ({ncells}) != mesh cells ({nx*ny*nz})"
                        )

                    J = np.empty((nx, ny, nz, nw), dtype=float)
                    for iw in range(nw):
                        J[..., iw] = j_lambda[:, iw].magnitude.reshape((nx, ny, nz), order='F')

                    lam_cm = (C_LIGHT / freq_hz).to('cm').magnitude

                    logger.info(
                        f"Read mean intensity from {filepath}: grid=({nx},{ny},{nz}), nwave={nw}"
                    )
                    return lam_cm, J
                except Exception as e:
                    logger.warning(f"Failed to read binary file {filepath}: {e}")
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
        
        # Get gas density
        rho_gas = self.model.gas['density'].data.to('g/cm^3')
        # Mass per H nucleus including He - should be read from simulation data
        # For now using typical protoplanetary disk value
        # TODO: Read from simulation data or params
        MU_HNUC = 1.4  # mass per H nucleus including He
        nH = rho_gas / (MU_HNUC * M_H)
        
        # Model gas density has axis_order ('r', 'phi', 'theta') for spherical coords
        # but RADMC-3D/RadData uses ('r', 'theta', 'phi')
        # Need to transpose if necessary
        field = self.model.gas['density']
        if hasattr(field, 'axis_order'):
            axis_order = field.axis_order
            # For spherical: convert (r, phi, theta) -> (r, theta, phi)
            if axis_order == ('r', 'phi', 'theta'):
                # Transpose - pint quantities can be transposed directly
                nH = np.transpose(nH, (0, 2, 1))  # (r, phi, theta) -> (r, theta, phi)
        
        self.nH = nH.to('cm^-3')
        
        logger.info(f"Computed nH from gas density: "
                   f"min={np.min(self.nH):.2e}, max={np.max(self.nH):.2e}")
        
        return self.nH
    
    def summarize_chi_over_nH(
        self,
        log_min: float = -8.0,
        log_max: float =  0.0,
        nbins: int = 50
    ) -> dict:

        if self.nH is None:
            self.compute_nH_from_model()
        if self.chi is None:
            raise RuntimeError(
                "chi field is not computed; run compute_mcmono() before "
                "summarize_chi_over_nH()."
            )

        nH = self.nH.to('cm^-3').magnitude
        chi = self.chi.to('dimensionless').magnitude

        ratio = chi / (nH + eps_chi)
        log_ratio = np.log10(np.maximum(ratio, eps_chi))

        hist, edges = np.histogram(log_ratio, bins=nbins, range=(log_min, log_max))
        total = log_ratio.size

        logger.info(
            "log10(chi/nH): min=%.2f, max=%.2f, median=%.2f"
            % (
                float(log_ratio.min()),
                float(log_ratio.max()),
                float(np.median(log_ratio)),
            )
        )

        log_thr = float(LOG_CHI_OVER_NH_PDISS.magnitude)
        candidate_counts = []
        candidate_fractions = []

        return {
            "bin_edges": edges,
            "hist": hist,
            "total_cells": int(total),
            "candidate_counts": np.array(candidate_counts, dtype=int),
            "candidate_fractions": np.array(candidate_fractions, dtype=float),
        }
    
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
        use_params_nphot = nphot is None
        if nphot is None:
            nphot = int(self.params.nphot_thermal)
        countwrite = max(1, int(nphot // 100))
        int32_max = int(np.iinfo(np.int32).max)
        if countwrite > int32_max:
            logger.warning(
                "countwrite too large; clamping to int32 max (%d)" % int32_max
            )
            countwrite = int32_max
        
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
            use_cache = False
            params_path_current = self.model_dir / 'params.txt'
            params_path_saved = output_dir / 'params.txt'
            if (
                use_params_nphot
                and params_path_current.exists()
                and params_path_saved.exists()
            ):
                try:
                    params_current = _read_params_snapshot(params_path_current)
                    params_saved = _read_params_snapshot(params_path_saved)
                    sig_current = _params_signature(params_current, _MCTHERM_PARAM_KEYS)
                    sig_saved = _params_signature(params_saved, _MCTHERM_PARAM_KEYS)
                    if sig_current == sig_saved:
                        use_cache = True
                except Exception:
                    use_cache = False

            if use_cache:
                logger.info(f"Temperature already computed at {existing_file}")
                self.read_dust_temperature(fname=str(existing_file))
                return self.temperature
            else:
                logger.info(
                    "Existing dust_temperature output found but parameters "
                    "have changed or nphot override used; recomputing mctherm."
                )
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_cntdump_ge_countwrite(countwrite, nphot)
        
        # If an external UV field is requested, ensure external_source.inp
        # is (re)generated on the current continuum wavelength grid before
        # creating symlinks and running mctherm.
        if getattr(self.params, 'external_uv', False):
            external_source_path = self.inputs_dir / 'external_source.inp'
            if not external_source_path.exists():
                from .writer import RadWriter
                writer = RadWriter(self.model, organize_files=True)
                writer.write_external_source(self.model_dir)
        
        # Create symlinks to input files
        self.create_symlinks()
        
        try:
            # Run mctherm (RADMC-3D gets nphot and setthreads from radmc3d.inp)
            logger.info(f"Running RADMC-3D mctherm with {nphot} photons (countwrite={countwrite})...")
            cmd = ['radmc3d', 'mctherm', 'countwrite', str(countwrite)]
            log_path = self.model_dir / 'radmc3d.out'
            returncode, stdout, stderr, _, _ = run_radmc3d_and_log(
                cmd,
                self.model_dir,
                section='mctherm',
                log_path=log_path,
                preserve_existing=False,
            )
            
            if returncode != 0:
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

    def _compute_chi_from_mean_intensity(
        self,
        j_lambda: Quantity,
        freq_hz: Quantity,
        uv_min: Quantity,
        uv_max: Quantity,
        mesh_shape: tuple[int, int, int],
        u_draine: Quantity,
        source: str,
    ) -> tuple[Quantity, int]:
        # Convert frequency to wavelength: lambda = c / nu
        lam = C_LIGHT / freq_hz

        # Compute chi by integrating over UV band in frequency space
        # Following Pinte et al. 2018 and fargo2radmc3d implementation
        uv_mask = (lam >= uv_min) & (lam <= uv_max)
        if not np.any(uv_mask):
            raise ValueError(
                f"No UV wavelengths ({uv_min:~P}-{uv_max:~P}) found in {source}. "
                f"Wavelength range: {lam.min():~P}-{lam.max():~P}"
            )

        j_uv = j_lambda[:, uv_mask]
        nu_uv = freq_hz[uv_mask]
        u_nu = 4.0 * np.pi * j_uv / C_LIGHT.to_base_units()

        # Sort by frequency for integration
        sort_idx = np.argsort(nu_uv)
        nu_sorted = nu_uv[sort_idx]
        u_nu_sorted = u_nu[:, sort_idx]

        # Integrate energy density over frequency
        u_band = np.trapz(u_nu_sorted, nu_sorted, axis=1)

        # Compute chi (dimensionless)
        chi_flat = (u_band / u_draine).to('dimensionless')

        # Reshape to 3D grid
        chi_3d = chi_flat.reshape(mesh_shape, order='F')

        # For spherical grids, RadData/RADMC-3D uses (r, theta, phi) but DiskBridge uses (r, phi, theta)
        if self.model.mesh.coord_system == 'spherical':
            chi_3d = np.transpose(chi_3d, (0, 2, 1))

        return chi_3d, int(np.count_nonzero(uv_mask))
    
    def compute_mcmono(
        self,
        nphot: int = None,
        output_dir: Optional[str | Path] = None,
        force: bool = False,
        wavelengths_um: Optional[np.ndarray] = None,
        uv_min: Quantity = None,
        uv_max: Quantity = None,
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
        uv_min : Quantity, optional
            UV range lower bound (uses params if None)
        uv_max : Quantity, optional
            UV range upper bound (uses params if None)
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
        use_params_nphot = nphot is None
        use_params_uv_min = uv_min is None
        use_params_uv_max = uv_max is None
        use_params_nw = n_wavelengths is None
        use_params_wavelengths = wavelengths_um is None

        if nphot is None:
            nphot = self.params.nphot_mono
        countwrite = max(1, int(nphot // 100))
        int32_max = int(np.iinfo(np.int32).max)
        if countwrite > int32_max:
            logger.warning(
                "countwrite too large; clamping to int32 max (%d)" % int32_max
            )
            countwrite = int32_max
        if uv_min is None:
            uv_min = self.params.uv_min
        if uv_max is None:
            uv_max = self.params.uv_max
        if n_wavelengths is None:
            n_wavelengths = self.params.uv_n_wavelengths

        if uv_min < self.params.lambda_min or uv_max > self.params.lambda_max:
            raise ValueError(
                "mcmono UV wavelength range is outside the global wavelength grid: "
                f"uv=[{uv_min:~P},{uv_max:~P}], "
                f"grid=[{self.params.lambda_min:~P},{self.params.lambda_max:~P}]"
            )
        
        # Validate UV range
        u_draine = U_DRAINE
        
        logger.info(
            "UV field configuration: "
            f"uv=[{uv_min:~P},{uv_max:~P}], "
            f"n_wavelengths={n_wavelengths}"
        )
        
        # Set output directory
        if output_dir is None:
            output_dir = self.outputs_dir
        else:
            output_dir = Path(output_dir)

        # Check if already computed
        mean_intensity_candidates = [
            output_dir / 'mean_intensity.bout',
            output_dir / 'mean_intensity.out',
        ]
        mean_intensity_file = next((p for p in mean_intensity_candidates if p.exists()), None)
        if mean_intensity_file is not None and not force:
            use_cache = False
            params_path_current = self.model_dir / 'params.txt'
            params_path_saved = output_dir / 'params.txt'
            overrides = not (
                use_params_nphot
                and use_params_uv_min
                and use_params_uv_max
                and use_params_nw
            )
            if (
                not overrides
                and params_path_current.exists()
                and params_path_saved.exists()
            ):
                try:
                    params_current = _read_params_snapshot(params_path_current)
                    params_saved = _read_params_snapshot(params_path_saved)
                    names = _MCTHERM_PARAM_KEYS + _MCMONO_EXTRA_PARAM_KEYS
                    sig_current = _params_signature(params_current, names)
                    sig_saved = _params_signature(params_saved, names)
                    if sig_current == sig_saved:
                        use_cache = True
                except Exception:
                    use_cache = False

            if use_cache:
                logger.info(f"Mean intensity already computed at {mean_intensity_file}")
                freq_hz, j_lambda = self.data.read_mean_intensity_file(mean_intensity_file)

                nx, ny, nz = self.data._getMeshShape()
                self.mean_intensity = j_lambda

                chi_3d, n_uv = self._compute_chi_from_mean_intensity(
                    j_lambda=j_lambda,
                    freq_hz=freq_hz,
                    uv_min=uv_min,
                    uv_max=uv_max,
                    mesh_shape=(nx, ny, nz),
                    u_draine=u_draine,
                    source="existing mean_intensity file",
                )
                self.chi = chi_3d

                logger.info(
                    f"Loaded chi from {n_uv} UV wavelengths: "
                    f"min={np.min(chi_3d):.2e}, max={np.max(chi_3d):.2e}"
                )

                try:
                    self.summarize_chi_over_nH()
                except Exception as e:
                    logger.warning(f"summarize_chi_over_nH failed after loading chi: {e}")

                return self.chi
            else:
                logger.info(
                    "Existing mean_intensity output found but parameters "
                    "have changed or mcmono overrides used; recomputing mcmono."
                )
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_cntdump_ge_countwrite(countwrite, nphot)
        
        # Create mcmono_wavelength_micron.inp
        mcmono_wav_file = self.model_dir / 'mcmono_wavelength_micron.inp'
        if wavelengths_um is not None:
            wav = np.asarray(wavelengths_um, dtype=float)
            if wav.ndim != 1:
                raise ValueError("wavelengths_um must be a 1D array")
            if wav.size < 2:
                raise ValueError("wavelengths_um must contain at least 2 wavelengths")
            if not np.all(np.isfinite(wav)):
                raise ValueError("wavelengths_um contains non-finite values")
            if not np.all(wav[1:] > wav[:-1]):
                raise ValueError("wavelengths_um must be strictly increasing")
            if wav[0] < self.params.lambda_min.to('micron').magnitude or wav[-1] > self.params.lambda_max.to('micron').magnitude:
                raise ValueError(
                    "mcmono wavelength grid is outside the global wavelength grid: "
                    f"mcmono=[{wav[0]:.6g},{wav[-1]:.6g}] micron, "
                    f"grid=[{self.params.lambda_min.to('micron').magnitude:.6g},{self.params.lambda_max.to('micron').magnitude:.6g}] micron"
                )
            mcmono_lam_um = wav
        else:
            uv_lam = np.linspace(uv_min, uv_max, n_wavelengths).to('micron')
            mcmono_lam_um = np.asarray(uv_lam.magnitude, dtype=float)

        with open(mcmono_wav_file, 'w') as f:
            f.write(f'{mcmono_lam_um.size}\n')
            for lam in mcmono_lam_um:
                f.write(f'{lam:.6f}\n')
        logger.debug(f"Wrote {mcmono_wav_file} with {mcmono_lam_um.size} wavelengths")
        
        # Create symlinks to input files
        self.create_symlinks()
        
        # If an external UV field is requested, ensure external_source.inp exists
        if getattr(self.params, 'external_uv', False):
            external_source_path = self.inputs_dir / 'external_source.inp'
            if not external_source_path.exists():
                from .writer import RadWriter
                writer = RadWriter(self.model, organize_files=True)
                writer.write_external_source(self.model_dir)

        # RADMC-3D mcmono also needs temperature file - create symlink if it exists in outputs subdirectory
        for suffix in ['.bdat', '.dat', '.binp']:
            dst_temp = self.model_dir / f'dust_temperature{suffix}'
            if dst_temp.exists():
                continue

            src_candidates = [
                self.outputs_dir / f'dust_temperature{suffix}',
                self.outputs_dir / 'temperature' / f'dust_temperature{suffix}',
                Path(output_dir).parent / 'temperature' / f'dust_temperature{suffix}',
                Path(output_dir) / f'dust_temperature{suffix}',
            ]
            src_temp = next((p for p in src_candidates if p.exists()), None)
            if src_temp is None:
                continue

            import os
            os.symlink(src_temp, dst_temp)
            self._active_symlinks.append(dst_temp)

            logger.debug(f"Created symlink: {dst_temp} -> {src_temp}")
            break
        
        try:
            # Run mcmono at UV wavelengths
            setthreads = self.params.nbcores
            logger.info(
                f"Running RADMC-3D mcmono at {mcmono_lam_um.size} wavelengths "
                f"({mcmono_lam_um[0]:.6g}-{mcmono_lam_um[-1]:.6g} micron) with {nphot} photons "
                f"(countwrite={countwrite})..."
            )
            cmd = ['radmc3d', 'mcmono', 'setthreads', str(setthreads), 'countwrite', str(countwrite)]

            log_path = self.model_dir / 'radmc3d.out'
            returncode, stdout, stderr, combined_log, _ = run_radmc3d_and_log(
                cmd,
                self.model_dir,
                section='mcmono',
                log_path=log_path,
                preserve_existing=True,
            )

            mcmono_text = combined_log.split('--- mcmono ---', 1)[-1]
            error_lines = [line.strip() for line in mcmono_text.splitlines() if 'ERROR' in line.upper()]
            if error_lines:
                errors = "\n".join(error_lines[-10:])
                raise RuntimeError(f"mcmono failed with RADMC-3D errors:\n{errors}")
            
            if returncode != 0:
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
            ['mean_intensity.out', 'mean_intensity.bout', 'mcmono_wavelength_micron.inp'],
            f'radmc3d mcmono range_{mcmono_lam_um[0]:.6g}-{mcmono_lam_um[-1]:.6g}micron_{mcmono_lam_um.size}wavelengths'
        )
        
        # Read mean intensity directly from the mcmono output directory
        mean_intensity_path = next(
            (
                p
                for p in (
                    output_dir / 'mean_intensity.bout',
                    output_dir / 'mean_intensity.out',
                )
                if p.exists()
            ),
            None,
        )
        if mean_intensity_path is None:
            raise FileNotFoundError(f"Mean intensity file not found in {output_dir}")

        freq_hz, j_lambda = self.data.read_mean_intensity_file(mean_intensity_path)
        
        # Store wavelengths and mean intensity
        nx, ny, nz = self.data._getMeshShape()
        self.mean_intensity = j_lambda  # Keep full spectral dimension

        chi_3d, n_uv = self._compute_chi_from_mean_intensity(
            j_lambda=j_lambda,
            freq_hz=freq_hz,
            uv_min=uv_min,
            uv_max=uv_max,
            mesh_shape=(nx, ny, nz),
            u_draine=u_draine,
            source="mean_intensity file",
        )
        self.chi = chi_3d

        axis_order = ('r', 'phi', 'theta') if self.model.mesh.coord_system == 'spherical' else self.model.mesh.axis_names()
        self.model.gas_register(
            'chi',
            Field(
                quantity='chi',
                data=self.chi,
                axis_order=axis_order,
            ),
        )
        
        logger.info(
            f"Computed chi from {n_uv} UV wavelengths ({uv_min:~P}-{uv_max:~P}): "
            f"min={np.min(chi_3d):.2e}, max={np.max(chi_3d):.2e}"
        )

        try:
            self.summarize_chi_over_nH()
        except Exception as e:
            logger.warning(f"summarize_chi_over_nH failed after computing chi: {e}")
        
        return self.chi
    
    def extract_shell_spectrum(
        self,
        r_split_au: float,
        shell_ncells: int = 3,
        mcmono_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract volume-weighted mean spectrum from a shell at R_split.
        
        This extracts the mean intensity from cells in a radial shell around
        R_split to create an effective external spectrum for the inner run.
        
        Parameters
        ----------
        r_split_au : float
            Split radius in AU
        shell_ncells : int
            Number of radial cells to include in the shell (default: 3)
        mcmono_dir : Path, optional
            Directory containing mcmono output (default: self.outputs_dir)
            
        Returns
        -------
        wavelengths_um : ndarray
            Wavelengths in microns
        spectrum : ndarray
            Volume-weighted mean intensity in erg/s/cm^2/Hz/sr
            
        Raises
        ------
        ValueError
            If mean_intensity not available or R_split outside grid
        """
        if mcmono_dir is None:
            mcmono_dir = self.outputs_dir
        mcmono_dir = Path(mcmono_dir)
        
        # Read mean intensity if not already loaded
        mean_intensity_path = mcmono_dir / 'mean_intensity.bout'
        if not mean_intensity_path.exists():
            mean_intensity_path = mcmono_dir / 'mean_intensity.out'
        
        if not mean_intensity_path.exists():
            raise FileNotFoundError(f"No mean_intensity file in {mcmono_dir}")

        freq_hz_q, j_lambda_q = self.data.read_mean_intensity_file(mean_intensity_path)
        freq_hz = np.asarray(freq_hz_q.to('Hz').magnitude, dtype=float)
        j_lambda = np.asarray(j_lambda_q.to('erg/(s*cm^2*Hz*sr)').magnitude, dtype=float)
        nwav = int(freq_hz.size)
        
        # Get mesh info
        mesh = self.model.mesh
        r_edges_au = mesh.edges('r').to('au').magnitude
        r_centers_au = mesh.centers('r').to('au').magnitude
        nr = len(r_centers_au)
        nphi = len(mesh.centers('phi'))
        ntheta = len(mesh.centers('theta'))
        
        # Find radial index closest to R_split
        r_split_idx = np.searchsorted(r_edges_au, r_split_au)
        if r_split_idx <= 0 or r_split_idx >= nr:
            raise ValueError(f"R_split={r_split_au} AU outside grid range")
        
        # Define shell indices (cells just outside R_split)
        shell_start = r_split_idx
        shell_end = min(r_split_idx + shell_ncells, nr)
        
        if shell_end <= shell_start:
            raise ValueError(f"Shell is empty: r_split_idx={r_split_idx}, nr={nr}")
        
        logger.info(
            f"Extracting shell spectrum at R_split={r_split_au:.2f} AU, "
            f"radial cells [{shell_start}:{shell_end}]"
        )
        
        # Compute cell volumes for weighting
        from diskbridge.model.profiles import compute_cell_volumes
        volumes = compute_cell_volumes(self.model)  # (nr, nphi, ntheta)
        
        # Mean intensity is stored in RADMC-3D order (r, theta, phi) flattened
        # Reshape to (nr, ntheta, nphi)
        j_3d = j_lambda.reshape((nr, ntheta, nphi, nwav), order='F')
        
        # Extract shell and compute volume-weighted average
        shell_volumes = volumes[shell_start:shell_end, :, :]  # (n_shell, nphi, ntheta)
        shell_j = j_3d[shell_start:shell_end, :, :, :]  # (n_shell, ntheta, nphi, nwav)

        # Transpose volumes to match j_3d order (swap phi and theta)
        shell_volumes_reordered = shell_volumes.transpose((0, 2, 1))  # (n_shell, ntheta, nphi)
        
        # Compute weighted average over spatial dimensions
        total_volume = np.sum(shell_volumes_reordered)
        spectrum = np.zeros(nwav)
        for iw in range(nwav):
            weighted_sum = np.sum(shell_j[:, :, :, iw] * shell_volumes_reordered)
            spectrum[iw] = weighted_sum / total_volume
        
        # Convert frequency to wavelength
        c_cgs = 2.99792458e10  # cm/s
        wavelengths_um = (c_cgs / freq_hz) * 1e4  # cm to micron
        
        logger.info(
            f"Shell spectrum: {nwav} wavelengths, "
            f"J_mean range [{spectrum.min():.2e}, {spectrum.max():.2e}] erg/s/cm^2/Hz/sr"
        )
        
        return wavelengths_um, spectrum
    
    def write_effective_external_source(
        self,
        wavelengths_um: np.ndarray,
        spectrum: np.ndarray,
        output_dir: Path,
        require_coverage: bool = True,
    ) -> Path:
        """Write effective external source file from shell spectrum.
        
        Parameters
        ----------
        wavelengths_um : ndarray
            Wavelengths in microns
        spectrum : ndarray
            Mean intensity in erg/s/cm^2/Hz/sr
        output_dir : Path
            Directory to write the file
            
        Returns
        -------
        Path
            Path to the written file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / 'external_source.inp'

        wav_file = output_dir / 'wavelength_micron.inp'
        if not wav_file.is_file():
            raise FileNotFoundError(f"wavelength grid file not found: {wav_file}")

        with open(wav_file, 'r') as f:
            n_global = int(f.readline().strip())
            wavelengths_global = np.array([float(f.readline().strip()) for _ in range(n_global)], dtype=float)

        sort_idx = np.argsort(wavelengths_um)
        w_src = np.asarray(wavelengths_um, dtype=float)[sort_idx]
        s_src = np.asarray(spectrum, dtype=float)[sort_idx]

        if require_coverage:
            if w_src.size == 0:
                raise ValueError("Effective external spectrum is empty")
            if float(w_src[0]) > float(np.min(wavelengths_global)) or float(w_src[-1]) < float(np.max(wavelengths_global)):
                raise ValueError(
                    "Effective external spectrum does not cover the full external-field wavelength grid: "
                    f"effective=[{w_src[0]:.6g},{w_src[-1]:.6g}] micron, "
                    f"grid=[{float(np.min(wavelengths_global)):.6g},{float(np.max(wavelengths_global)):.6g}] micron"
                )

        if np.any(s_src <= 0.0):
            raise ValueError("loglog interpolation requires strictly positive spectrum values")
        intensity = np.exp(
            np.interp(
                np.log(wavelengths_global),
                np.log(w_src),
                np.log(s_src),
            )
        )

        with open(filepath, 'w') as f:
            f.write('2\n')
            f.write(f'{len(wavelengths_global)}\n')
            for w in wavelengths_global:
                f.write(f'{w:13.6e}\n')
            for val in intensity:
                f.write(f'{val:13.6e}\n')
        
        logger.info(f"Wrote effective external source: {filepath}")
        return filepath
    
    def compute_segmented_rt(
        self,
        nphot_therm: Optional[int] = None,
        nphot_mono: Optional[int] = None,
        mcmono_n_wavelengths: Optional[int] = None,
        mcmono_uv_n_wavelengths: Optional[int] = None,
        mcmono_wavelength_source: str = 'external',
        mcmono_wavelength_spacing: str = 'log',
        mcmono_wavelengths_um: Optional[np.ndarray] = None,
        force: bool = False,
        max_splits: Optional[int] = None,
    ) -> dict:
        """Run segmented RT using the driver layer.

        Segmentation is a workflow/orchestration concern, not a property of the
        hydro model. This method remains as a convenience wrapper.
        """
        from diskbridge.radmc3d.segmented import SegmentedRadmcRunner

        runner = SegmentedRadmcRunner(base_model=self.model, base_model_dir=self.model_dir)
        result = runner.run_segmented_rt(
            nphot_therm=nphot_therm,
            nphot_mono=nphot_mono,
            mcmono_n_wavelengths=mcmono_n_wavelengths,
            mcmono_uv_n_wavelengths=mcmono_uv_n_wavelengths,
            mcmono_wavelength_source=mcmono_wavelength_source,
            mcmono_wavelength_spacing=mcmono_wavelength_spacing,
            mcmono_wavelengths_um=mcmono_wavelengths_um,
            force=force,
            max_splits=max_splits,
        )

        # Keep RadModel instance in sync with merged fields
        self.temperature = result['temperature']
        self.chi = result['chi']
        return result
    
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
