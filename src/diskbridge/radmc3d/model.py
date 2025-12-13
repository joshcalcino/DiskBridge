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
    from diskbridge.model.model import Model

from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from .data import RadData
from diskbridge.chemistry.models.pinte_switches import (
    apply_photodissociation,
    apply_photodesorption_escape,
    compute_freezeout_factor,
)
from .utils import (
    _extract_radmc_errors,
    create_symlinks_for_file_map,
    cleanup_symlinks,
    run_radmc3d_command,
    _read_params_snapshot,
    _params_signature,
)
import diskbridge 

# Physical constants from config
C_LIGHT = units('c')  # Speed of light
M_H = units('m_H')  # Hydrogen mass
SIGMA_SB = units('sigma_SB')  # Stefan-Boltzmann constant

# Draine (1978) UV field constant
U_DRAINE = Quantity(9.0e-14, 'erg/cm^3')

# Default Pinte+18 thresholds (model-specific, as Quantities)
T_FRZ_DEFAULT = Quantity(21.0, 'K')  # freeze-out temperature
EPS_DEFAULT = Quantity(8e-5, 'dimensionless')  # freeze-out survival fraction
LOG_CHI_OVER_NH_PDISS = Quantity(-6.0, 'dimensionless')  # photodissociation threshold
LOG_CHI_OVER_NH_PDES = Quantity(-7.0, 'dimensionless')  # photodesorption threshold
eps_chi = 1.0e-99

# --- Two-phase CO chemistry defaults (Heays+2017, Visser+2009 style) ---
# These can be overridden via method arguments
XCO_TOT_DEFAULT = 1.0e-4           # total CO abundance if all CO is present (per H)
TAU_CO_FORM_DEFAULT = Quantity(1.0e5, "yr")   # CO formation timescale
E_BIND_CO_DEFAULT = 855.0          # binding energy in K
NU0_CO_DEFAULT = 1.0e12            # vibrational frequency [1/s] for thermal desorption
SIGMA_D_PER_H_DEFAULT = 1.0e-21    # grain cross-section per H [cm^2 per H]
ALPHA_PD_ICE_DEFAULT = 1.0e-12     # photodesorption rate per chi=1 [1/s]
K0_CO_DEFAULT = Quantity(2.0e-10, "1/s")  # unshielded Heays rate for chi=1 Draine
M_CO_CGS = (28.0 * units('m_H')).to('g').magnitude       # CO mass in g (28 * m_H)
K_BOLTZ_CGS = units('k_B').to('erg/K').magnitude          # Boltzmann constant in erg/K
 
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


def compute_co_photodissociation_rate(
    chi,
    theta_co,
    k0_co,
    candidate_mask=None,
    min_rate: float = 1.0e-30,
):
    if isinstance(chi, Quantity):
        chi_arr = chi.to("dimensionless").magnitude
    else:
        chi_arr = np.asarray(chi, dtype=float)

    if theta_co is None:
        theta_arr = np.ones_like(chi_arr, dtype=float)
    elif isinstance(theta_co, Quantity):
        theta_arr = theta_co.to("dimensionless").magnitude
    else:
        theta_arr = np.asarray(theta_co, dtype=float)

    if candidate_mask is not None:
        mask = np.asarray(candidate_mask, dtype=bool)
        if mask.shape != theta_arr.shape:
            raise ValueError(
                f"candidate_mask shape {mask.shape} does not match theta_co shape {theta_arr.shape}"
            )
        theta_arr = np.where(mask, theta_arr, 1.0)

    theta_arr = np.clip(theta_arr, 0.0, 1.0)

    if isinstance(k0_co, Quantity):
        k0_val = k0_co.to("1/s").magnitude
    else:
        k0_val = float(k0_co)

    k_diss_arr = k0_val * chi_arr * theta_arr
    k_diss_co = Quantity(k_diss_arr, "1/s")

    tau_diss_co = None
    if min_rate is not None:
        rate_floor = float(min_rate)
        if rate_floor > 0.0:
            k_safe = np.maximum(k_diss_arr, rate_floor)
            tau_arr = 1.0 / k_safe
            tau_diss_co = Quantity(tau_arr, "s")

    return k_diss_co, tau_diss_co


def co_freezeout_rate(
    T: Quantity,
    nH: Quantity,
    sigma_d_per_H: float = SIGMA_D_PER_H_DEFAULT,
) -> Quantity:
    """
    Compute CO freeze-out rate coefficient k_fo [1/s] for each cell.
    
    Uses k_fo = S * (sigma_d_per_H * nH) * v_th where S ~ 1 (sticking coefficient).
    
    Parameters
    ----------
    T : Quantity
        Gas temperature in K
    nH : Quantity
        H nuclei number density in cm^-3
    sigma_d_per_H : float
        Grain cross-section per H atom [cm^2 per H]
        
    Returns
    -------
    Quantity
        Freeze-out rate coefficient [1/s]
    """
    T_K = T.to("K").magnitude
    nH_cm3 = nH.to("cm^-3").magnitude
    
    # thermal speed of CO: v_th = sqrt(8 k_B T / (pi m_CO))
    v_th = np.sqrt(8.0 * K_BOLTZ_CGS * T_K / (np.pi * M_CO_CGS))  # cm/s
    
    sigma_nd = sigma_d_per_H * nH_cm3  # cm^-1
    S = 1.0  # sticking coefficient (constant for simplicity)
    
    k_fo = S * sigma_nd * v_th  # s^-1
    return Quantity(k_fo, "1/s")


def co_thermal_desorption_rate(
    T_d: Quantity,
    E_bind: float = E_BIND_CO_DEFAULT,
    nu0: float = NU0_CO_DEFAULT,
) -> Quantity:
    """
    Compute thermal desorption rate coefficient k_td [1/s].
    
    Uses Arrhenius form: k_td = nu0 * exp(-E_bind / T_d)
    
    Parameters
    ----------
    T_d : Quantity
        Dust temperature in K
    E_bind : float
        Binding energy in K
    nu0 : float
        Vibrational frequency [1/s]
        
    Returns
    -------
    Quantity
        Thermal desorption rate coefficient [1/s]
    """
    T_K = T_d.to("K").magnitude
    k_td = nu0 * np.exp(-E_bind / np.maximum(T_K, 1e-6))
    return Quantity(k_td, "1/s")


def co_photodesorption_rate(
    chi: Quantity,
    alpha_pd: float = ALPHA_PD_ICE_DEFAULT,
) -> Quantity:
    """
    Compute photodesorption rate coefficient from ice k_pd_ice [1/s].
    
    Simple linear scaling: k_pd_ice = alpha_pd * chi
    
    Parameters
    ----------
    chi : Quantity
        UV field in Draine units (dimensionless)
    alpha_pd : float
        Photodesorption rate per chi=1 [1/s]
        
    Returns
    -------
    Quantity
        Photodesorption rate coefficient [1/s]
    """
    chi_val = chi.to("dimensionless").magnitude
    k_pd_ice = alpha_pd * chi_val
    return Quantity(k_pd_ice, "1/s")


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
            'radmc3d.inp', 'external_source.inp'
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
        nbins: int = 50,
        margins: Optional[Sequence[float]] = None,
    ) -> dict:
        if margins is None:
            margins = (0.0, 0.5, 1.0, 2.0)

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

        for margin in margins:
            upper = log_thr + float(margin)
            mask = (log_ratio > log_thr) & (log_ratio <= upper)
            n_cand = int(mask.sum())
            frac = 100.0 * n_cand / total if total > 0 else 0.0
            candidate_counts.append(n_cand)
            candidate_fractions.append(frac)
            logger.info(
                "margin_dex=%.2f: candidates=%d (%.2f%%) for %.2f < log10(chi/nH) <= %.2f"
                % (float(margin), n_cand, frac, log_thr, upper)
            )

        return {
            "bin_edges": edges,
            "hist": hist,
            "total_cells": int(total),
            "margins": tuple(float(m) for m in margins),
            "candidate_counts": np.array(candidate_counts, dtype=int),
            "candidate_fractions": np.array(candidate_fractions, dtype=float),
        }

    def compute_abundance(
        self,
        molecule: str = 'co',
        X0: float = 5e-5,
        eps: float = EPS_DEFAULT,
        Tfrz: Quantity = T_FRZ_DEFAULT,
        photodissociation: Optional[bool] = None,
        freezeout: Optional[bool] = None,
        photodesorption: Optional[bool] = None,
        self_shielding: Optional[bool] = None,
        nside: int = 4,
        b_kms: float = 0.3,
        XH2_guess: float = 0.5,
        margin_dex: float = 1.5,
        write_output: bool = True,
        progress_chunks: Optional[int] = None,
        smooth_log_chi_nH_dex: float = 0.0,
        smooth_Tfrz_K: float = 0.0,
    ) -> Tuple[Quantity, Quantity]:
        """Compute molecular abundance with photochemistry (Pinte et al. 2018).
        
        Applies up to four processes:
        1. Self-shielding (optional): compute effective chi using Visser+09 tables
        2. Photodissociation: X = 0 where log10(chi_eff/nH) > -6
        3. Freeze-out: X *= eps where T < Tfrz
        4. Photodesorption (optional): skip freeze-out where log10(chi_eff/nH) > -7
        
        Parameters
        ----------
        molecule : str, optional
            Molecule name (default: 'co')
        X0 : float, optional
            Initial abundance X/nH (default: 5e-5)
        eps : float, optional
            Depletion factor for freeze-out (default: 1e-3)
        Tfrz : Quantity, optional
            Freeze-out temperature (default: 21 K)
        photodissociation : bool, optional
            Apply photodissociation (default: from params)
        freezeout : bool, optional
            Apply freeze-out (default: from params)
        photodesorption : bool, optional
            Apply photodesorption (default: from params)
        self_shielding : bool, optional
            Apply CO self-shielding via Visser+09 tables and HEALPix ray 
            tracing (default: False). When enabled, chi is replaced by
            chi_eff = chi * theta_CO for photodissociation/photodesorption.
        nside : int, optional
            HEALPix Nside for self-shielding (npix = 12 * nside^2, default: 4)
        b_kms : float, optional
            Doppler b parameter for Visser tables (default: 0.3 km/s)
        XH2_guess : float, optional
            Assumed H2 fraction for self-shielding when nH2 not provided
            (default: 0.5, i.e., fully molecular)
        margin_dex : float, optional
            Extra dex below threshold to include in candidate mask (default: 1.5)
        progress_chunks : int, optional
            If set to a positive integer, split HEALPix candidate cells into
            this many chunks when computing self-shielding, logging progress
            after each chunk. None (default) keeps a single fast call for
            maximum performance.
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
        Visser et al. (2009), A&A 503, 323 (CO shielding functions)
        Heays et al. (2017), A&A 602, A105 (photodissociation rates)
        """
        # Resolve photochemistry flags from params if not explicitly set
        if photodissociation is None:
            photodissociation = self.params.photodissociation
        if freezeout is None:
            freezeout = self.params.freezeout
        if photodesorption is None:
            photodesorption = self.params.photodesorption

        # Resolve self-shielding flags (params provide defaults, function args override)
        if molecule.lower() == "co":
            if self_shielding is None:
                self_shielding_flag = bool(getattr(self.params, "co_self_shielding", False))
            else:
                self_shielding_flag = bool(self_shielding)
        else:
            self_shielding_flag = bool(self_shielding) if self_shielding is not None else False

        # Ensure we have temperature
        if self.temperature is None:
            self.read_temperature()
        
        # Ensure we have nH
        if self.nH is None:
            self.compute_nH_from_model()
        
        # Ensure we have chi from mcmono only if needed
        # chi is required for photodissociation, photodesorption, and self-shielding
        if photodissociation or photodesorption or self_shielding_flag:
            if self.chi is None:
                self.compute_mcmono()
        
        # Get temperature and nH as Quantities
        T = self.temperature.to('K')
        nH = self.nH.to('cm^-3')
        Tfrz_K = Tfrz.to('K') if isinstance(Tfrz, Quantity) else Quantity(Tfrz, 'K')

        T_vals = T.to('K').magnitude
        Tfrz_val = float(Tfrz_K.magnitude)

        if isinstance(eps, Quantity):
            eps_q = eps.to('dimensionless')
        else:
            eps_q = Quantity(eps, 'dimensionless')
        eps_val = float(eps_q.magnitude)
        
        # Compute effective chi (with self-shielding if enabled)
        if self_shielding_flag and (photodissociation or photodesorption):
            from diskbridge.chemistry.shielding.healpix_columns import compute_co_shielding_healpix
            from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
            
            logger.info(f"Computing CO self-shielding (nside={nside}, b={b_kms} km/s)...")
            visser = VisserShielding(b_kms=b_kms)
            theta_co, chi_eff = compute_co_shielding_healpix(
                mesh=self.model.mesh,
                nH=nH,
                chi=self.chi,
                visser=visser,
                nCO=None,  # Will use X0 * nH as guess
                nH2=None,  # Will use XH2_guess * nH / 2
                nside=nside,
                b_kms=b_kms,
                Xco_guess=float(X0),
                XH2_guess=XH2_guess,
                margin_dex=margin_dex,
                progress_chunks=progress_chunks,
            )
            logger.info(f"Self-shielding: mean(theta_CO)={float(theta_co.magnitude.mean()):.3f}")
            logger.info(f"Self-shielding: max(theta_CO)={float(theta_co.magnitude.max()):.3f}")
            logger.info(f"Self-shielding: min(theta_CO)={float(theta_co.magnitude.min()):.3f}")
            self.theta_co = theta_co
            self.chi_eff = chi_eff
        else:
            chi_eff = self.chi
            self.theta_co = None
            self.chi_eff = chi_eff
        
        # Initialize abundance (dimensionless Quantity)
        X = Quantity(np.full_like(T, float(X0), dtype=float), 'dimensionless')
        
        # 1. Freeze-out (reduce abundance) with optional smooth transition
        if freezeout:
            freeze_factor, mask_frz = compute_freezeout_factor(
                T_vals,
                Tfrz_val,
                eps_val,
                smooth_Tfrz_K,
            )
            X *= freeze_factor
            n_frz = int(np.sum(mask_frz))
            logger.info(f"Freeze-out: {n_frz} cells ({100*n_frz/X.size:.1f}%)")
        else:
            freeze_factor = np.ones_like(T_vals, dtype=float)
            mask_frz = np.zeros_like(T_vals, dtype=bool)
        
        # Precompute log10(chi_eff / nH) if needed
        chi_over_nH = None
        if photodissociation or photodesorption:
            chi_vals = chi_eff.magnitude
            nH_vals = nH.magnitude
            ratio = chi_vals / (nH_vals + eps_chi)
            chi_over_nH = np.log10(np.maximum(ratio, eps_chi))
        
        # 2. Photodesorption escape (optional)
        mask_pdes = np.zeros_like(T_vals, dtype=bool)
        if photodesorption and chi_over_nH is not None:
            log_thr_pdes = float(LOG_CHI_OVER_NH_PDES.magnitude)
            if smooth_log_chi_nH_dex > 0.0:
                X, mask_pdes = apply_photodesorption_escape(
                    X,
                    X0=float(X0),
                    freezeout=bool(freezeout),
                    freeze_factor=freeze_factor,
                    mask_frz=mask_frz,
                    chi_over_nH=chi_over_nH,
                    log_thr_pdes=log_thr_pdes,
                    smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
                )
            else:
                chi_over_nH = np.log10(chi_eff.magnitude / (nH.magnitude + eps_chi))
                X, mask_pdes = apply_photodesorption_escape(
                    X,
                    X0=float(X0),
                    freezeout=bool(freezeout),
                    freeze_factor=freeze_factor,
                    mask_frz=mask_frz,
                    chi_over_nH=chi_over_nH,
                    log_thr_pdes=log_thr_pdes,
                    smooth_log_chi_nH_dex=0.0,
                )

            n_pdes = int(np.sum(mask_pdes))
            logger.info(f"Photodesorption: {n_pdes} cells ({100*n_pdes/X.size:.1f}%)")
        
        # 3. Photodissociation (kill molecule) with optional smooth transition
        if photodissociation and chi_over_nH is not None:
            log_thr_pdiss = float(LOG_CHI_OVER_NH_PDISS.magnitude)
            if smooth_log_chi_nH_dex > 0.0:
                X, mask_pdiss = apply_photodissociation(
                    X,
                    chi_over_nH=chi_over_nH,
                    log_thr_pdiss=log_thr_pdiss,
                    smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
                )
            else:
                chi_over_nH = np.log10(chi_eff.magnitude / (nH.magnitude + eps_chi))
                X, mask_pdiss = apply_photodissociation(
                    X,
                    chi_over_nH=chi_over_nH,
                    log_thr_pdiss=log_thr_pdiss,
                    smooth_log_chi_nH_dex=0.0,
                )

            n_pdiss = int(np.sum(mask_pdiss))
            logger.info(f"Photodissociation: {n_pdiss} cells ({100*n_pdiss/X.size:.1f}%)")
        
        # Compute number density
        n_mol = X * nH
        
        logger.info(f"Computed {molecule} abundance: "
                   f"X_mean={np.mean(X):.2e}, n_mean={np.mean(n_mol):.2e}")
        
        # Write output file
        if write_output:
            self.write_numberdens(molecule, n_mol)
        
        return X, n_mol

    def compute_co_photodissociation_rate_field(
        self,
        k0_co,
        candidate_mask: Optional[np.ndarray] = None,
        min_rate: float = 1.0e-30,
    ) -> Quantity:
        if self.chi is None:
            self.compute_mcmono()

        theta = self.theta_co
        k_diss_co, tau_diss_co = compute_co_photodissociation_rate(
            self.chi,
            theta,
            k0_co,
            candidate_mask=candidate_mask,
            min_rate=min_rate,
        )

        self.k_diss_co = k_diss_co
        self.tau_diss_co = tau_diss_co

        return self.k_diss_co
    
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
        
        # Get number density in cm^-3 (pint will handle units)
        n_dens = number_density.to('cm^-3')
        
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
        
        # Transpose for RADMC-3D order and write magnitudes to file
        n_radmc = np.transpose(n_dens, (2, 1, 0))
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
        use_params_nphot = nphot is None
        if nphot is None:
            nphot = int(self.params.nphot_thermal)
        countwrite = max(1, int(nphot // 100))
        
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
        output_dir.mkdir(exist_ok=True)
        
        # If an external UV field is requested, ensure external_source.inp
        # is (re)generated on the current continuum wavelength grid before
        # creating symlinks and running mctherm.
        if getattr(self.params, 'external_uv', False):
            from .writer import RadWriter
            writer = RadWriter(self.model, organize_files=True)
            writer.write_external_source(self.model_dir)
        
        # Create symlinks to input files
        self.create_symlinks()
        
        try:
            # Run mctherm (RADMC-3D gets nphot and setthreads from radmc3d.inp)
            logger.info(f"Running RADMC-3D mctherm with {nphot} photons (countwrite={countwrite})...")
            cmd = ['radmc3d', 'mctherm', 'countwrite', str(countwrite)]
            
            returncode, stdout, stderr = run_radmc3d_command(cmd, self.model_dir)
            
            log_path = self.model_dir / 'radmc3d.out'
            try:
                with open(log_path, 'a') as f:
                    f.write('\n--- mctherm ---\n')
                    if stdout:
                        f.write(stdout)
                    if stderr:
                        f.write('\n[stderr]\n')
                        f.write(stderr)
            except Exception:
                pass
            
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
    
    def compute_mcmono(
        self,
        nphot: int = None,
        output_dir: Optional[str | Path] = None,
        force: bool = False,
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

        if nphot is None:
            nphot = self.params.nphot_mono
        countwrite = max(1, int(nphot // 100))
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
        mean_intensity_file = output_dir / 'mean_intensity.out'
        if mean_intensity_file.exists() and not force:
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
                # Parse the existing file with multi-wavelength format
                with open(mean_intensity_file, 'r') as f:
                    iformat = int(f.readline().strip())
                    nrcells = int(f.readline().strip())
                    nwav = int(f.readline().strip())

                    # Read wavelengths (all on one line, space-separated, in Hz)
                    freq_hz = np.array([float(x) for x in f.readline().split()]) * units('Hz')

                    # Convert frequency to wavelength: lambda = c / nu
                    lam = C_LIGHT / freq_hz

                    # Read mean intensity values
                    # RADMC-3D stores data as: all cells for wavelength 0, then all cells for wavelength 1, etc.
                    j_flat = np.array([float(f.readline().strip()) for _ in range(nwav * nrcells)])
                    j_lambda = j_flat.reshape((nwav, nrcells)).T * units('erg/(s*cm^2*Hz*sr)')

                nx, ny, nz = self.data._getMeshShape()
                self.mean_intensity = j_lambda

                # Compute chi by integrating over UV band in frequency space
                uv_mask = (lam >= uv_min) & (lam <= uv_max)
                if not np.any(uv_mask):
                    raise ValueError(
                        f"No UV wavelengths ({uv_min:~P}-{uv_max:~P}) in existing mean intensity file. "
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
                chi_3d = chi_flat.reshape((nx, ny, nz), order='F')
                self.chi = chi_3d

                logger.info(
                    f"Loaded chi from {np.sum(uv_mask)} UV wavelengths: "
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
        output_dir.mkdir(exist_ok=True)
        
        # Create mcmono_wavelength_micron.inp with UV wavelength range
        mcmono_wav_file = self.model_dir / 'mcmono_wavelength_micron.inp'
        uv_lam = np.linspace(uv_min, uv_max, n_wavelengths).to(self.params.lambda_min.units)
        
        with open(mcmono_wav_file, 'w') as f:
            f.write(f'{len(uv_lam)}\n')  # Number of wavelengths
            for lam in uv_lam:
                f.write(f'{lam:.6f}\n')
        logger.debug(f"Wrote {mcmono_wav_file} with {len(uv_lam)} UV wavelengths")
        
        # Create symlinks to input files
        self.create_symlinks()
        
        # If an external UV field is requested, ensure external_source.inp exists
        if getattr(self.params, 'external_uv', False):
            from .writer import RadWriter
            writer = RadWriter(self.model, organize_files=True)
            writer.write_external_source(self.model_dir)

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
            logger.info(
                f"Running RADMC-3D mcmono at {n_wavelengths} UV wavelengths "
                f"({uv_min:~P}-{uv_max:~P}) with {nphot} photons "
                f"(countwrite={countwrite})..."
            )
            cmd = ['radmc3d', 'mcmono', 'setthreads', str(setthreads), 'countwrite', str(countwrite)]

            # Preserve any existing radmc3d.out log (e.g. from mctherm)
            log_path = self.model_dir / 'radmc3d.out'
            previous_log = ""
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        previous_log = f.read()
                except Exception:
                    previous_log = ""
            
            returncode, stdout, stderr = run_radmc3d_command(cmd, self.model_dir)

            # Rebuild radmc3d.out to contain both the previous log and the new mcmono output
            combined_log = previous_log
            combined_log += '\n--- mcmono ---\n'
            if stdout:
                combined_log += stdout
            if stderr:
                combined_log += '\n[stderr]\n'
                combined_log += stderr

            try:
                with open(log_path, 'w') as f:
                    f.write(combined_log)
            except Exception:
                pass

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
            ['mean_intensity.out', 'mcmono_wavelength_micron.inp'],
            f'radmc3d mcmono UV_range_{uv_min:~P}-{uv_max:~P}_{n_wavelengths}wavelengths'
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
            iformat = int(f.readline().strip())
            nrcells = int(f.readline().strip())
            nwav = int(f.readline().strip())
            
            # Read wavelengths (all on one line, space-separated, in Hz)
            freq_hz = np.array([float(x) for x in f.readline().split()]) * units('Hz')
            
            # Convert frequency to wavelength: lambda = c / nu
            lam = C_LIGHT / freq_hz
            
            # Read mean intensity values (nwav * nrcells values, one per line)
            # RADMC-3D stores data as: all cells for wavelength 0, then all cells for wavelength 1, etc.
            j_flat = np.array([float(f.readline().strip()) for _ in range(nwav * nrcells)])
            
            # Reshape to (nrcells, nwav) - data is wavelength-major, so reshape and transpose
            j_lambda = j_flat.reshape((nwav, nrcells)).T * units('erg/(s*cm^2*Hz*sr)')  # Attach units
        
        # Store wavelengths and mean intensity
        nx, ny, nz = self.data._getMeshShape()
        self.mean_intensity = j_lambda  # Keep full spectral dimension
        
        # Compute chi by integrating over UV band in frequency space
        # Following Pinte et al. 2018 and fargo2radmc3d implementation
        uv_mask = (lam >= uv_min) & (lam <= uv_max)
        
        if not np.any(uv_mask):
            raise ValueError(
                f"No UV wavelengths ({uv_min:~P}-{uv_max:~P}) found in mean intensity file. "
                f"Wavelength range: {lam.min():~P}-{lam.max():~P}"
            )
        
        j_uv = j_lambda[:, uv_mask]  # Shape: (nrcells, n_uv_wavelengths)

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
        chi_3d = chi_flat.reshape((nx, ny, nz), order='F')
        self.chi = chi_3d
        
        logger.info(
            f"Computed chi from {np.sum(uv_mask)} UV wavelengths ({uv_min:~P}-{uv_max:~P}): "
            f"min={np.min(chi_3d):.2e}, max={np.max(chi_3d):.2e}"
        )

        try:
            self.summarize_chi_over_nH()
        except Exception as e:
            logger.warning(f"summarize_chi_over_nH failed after computing chi: {e}")
        
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
    
    def compute_co_steady_state(
        self,
        Xco_tot: float = XCO_TOT_DEFAULT,
        tau_form: Quantity = TAU_CO_FORM_DEFAULT,
        k0_co: Quantity = K0_CO_DEFAULT,
        sigma_d_per_H: float = SIGMA_D_PER_H_DEFAULT,
        E_bind: float = E_BIND_CO_DEFAULT,
        nu0: float = NU0_CO_DEFAULT,
        alpha_pd_ice: float = ALPHA_PD_ICE_DEFAULT,
        use_dust_temp: bool = True,
        write_output: bool = True,
        skip_shielding: bool = False,
        nside: int = 4,
        b_kms: float = 0.3,
        margin_dex: float = 1.5,
    ) -> Tuple[Quantity, Quantity, Quantity]:
        """
        Solve two-phase CO chemistry in steady state using the analytic solution.
        
        This method computes the steady-state gas-phase and ice CO densities by
        solving the rate equations analytically. It uses:
        - Photodissociation from Heays+2017 with optional Visser+2009 shielding
        - Freeze-out onto grains
        - Thermal desorption from ice
        - Photodesorption from ice
        - Formation with a characteristic timescale
        
        This method does NOT use the Pinte chi/nH threshold switches from
        compute_abundance(). It is a separate, rate-based chemistry approach.
        
        Parameters
        ----------
        Xco_tot : float
            Total CO abundance if all CO is present (per H)
        tau_form : Quantity
            CO formation timescale
        k0_co : Quantity
            Unshielded photodissociation rate for chi=1 Draine field [1/s]
        sigma_d_per_H : float
            Grain cross-section per H atom [cm^2 per H]
        E_bind : float
            CO binding energy in K
        nu0 : float
            Vibrational frequency for thermal desorption [1/s]
        alpha_pd_ice : float
            Photodesorption rate per chi=1 [1/s]
        use_dust_temp : bool
            If True, use dust temperature for freeze-out/desorption. If False,
            use gas temperature.
        write_output : bool
            If True, write numberdens_co.inp file
        skip_shielding : bool
            If True, skip CO self-shielding calculation (theta_co = 1)
        nside : int
            HEALPix resolution for shielding calculation
        b_kms : float
            Doppler b parameter for Visser shielding [km/s]
        margin_dex : float
            Margin in dex for candidate cell selection in shielding calc
            
        Returns
        -------
        X_co : Quantity
            Gas-phase CO abundance (dimensionless)
        nco_gas : Quantity
            Gas-phase CO number density [cm^-3]
        nco_ice : Quantity
            CO ice number density [cm^-3]
        """
        # Ensure required fields exist
        if self.temperature is None:
            logger.info("Temperature not loaded, attempting to read from file...")
            try:
                self.read_temperature()
            except FileNotFoundError:
                raise RuntimeError(
                    "Temperature field required. Run mctherm or load temperature first."
                )
        
        if self.nH is None:
            logger.info("nH not computed, computing from gas density...")
            self.compute_nH_from_model()
        
        if self.chi is None:
            logger.info("chi not computed, attempting to compute from mcmono output...")
            try:
                self.compute_mcmono()
            except FileNotFoundError:
                raise RuntimeError(
                    "chi field required. Run compute_mcmono() first."
                )
        
        # Compute CO self-shielding if not skipped
        if skip_shielding:
            theta_co = Quantity(np.ones_like(self.chi.magnitude), "dimensionless")
            logger.info("Skipping CO self-shielding (theta_co = 1)")
        elif self.theta_co is not None:
            theta_co = self.theta_co
            logger.info("Using existing theta_co from previous calculation")
        else:
            # Import shielding machinery
            from diskbridge.chemistry.shielding.healpix_columns import compute_co_shielding_healpix
            from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
            
            logger.info("Computing CO self-shielding with HEALPix rays...")
            visser = VisserShielding(b_kms=b_kms)
            theta_co, chi_eff = compute_co_shielding_healpix(
                mesh=self.model.mesh,
                nH=self.nH,
                chi=self.chi,
                visser=visser,
                nside=nside,
                b_kms=b_kms,
                margin_dex=margin_dex,
            )
            self.theta_co = theta_co
            self.chi_eff = chi_eff
        
        # Compute photodissociation rate
        k_pd, tau_pd = compute_co_photodissociation_rate(
            chi=self.chi,
            theta_co=theta_co,
            k0_co=k0_co,
            candidate_mask=None,
            min_rate=1.0e-30,
        )
        self.k_diss_co = k_pd
        self.tau_diss_co = tau_pd
        
        # Choose temperature for freeze-out/desorption
        T = self.temperature
        
        # Compute rate coefficients
        k_fo = co_freezeout_rate(T, self.nH, sigma_d_per_H)
        k_td = co_thermal_desorption_rate(T, E_bind, nu0)
        k_pd_ice = co_photodesorption_rate(self.chi, alpha_pd_ice)
        
        # Compute eta = k_fo / (k_td + k_pd_ice)
        k_td_plus = k_td.to("1/s").magnitude + k_pd_ice.to("1/s").magnitude
        k_td_plus_safe = np.maximum(k_td_plus, 1e-60)
        eta = k_fo.to("1/s").magnitude / k_td_plus_safe
        
        # Compute nco_max and tau_form in consistent units
        nH_cm3 = self.nH.to("cm^-3").magnitude
        nco_max = Xco_tot * nH_cm3  # cm^-3
        tau_s = tau_form.to("s").magnitude
        
        # Compute steady-state solution:
        # nco_gas = nco_max / ((1 + eta) + k_pd * tau_form)
        k_pd_s = k_pd.to("1/s").magnitude
        denom = (1.0 + eta) + k_pd_s * tau_s
        nco_gas_arr = nco_max / denom
        nco_ice_arr = eta * nco_gas_arr
        
        # Wrap as Quantities
        self.nco_gas = Quantity(nco_gas_arr, "cm^-3")
        self.nco_ice = Quantity(nco_ice_arr, "cm^-3")
        
        # Compute gas-phase abundance
        X_co = Quantity(nco_gas_arr / (nH_cm3 + 1e-99), "dimensionless")
        
        # Log summary
        logger.info(
            f"CO steady-state: nco_gas min={np.min(nco_gas_arr):.2e}, "
            f"max={np.max(nco_gas_arr):.2e} cm^-3"
        )
        logger.info(
            f"CO steady-state: nco_ice min={np.min(nco_ice_arr):.2e}, "
            f"max={np.max(nco_ice_arr):.2e} cm^-3"
        )
        ice_frac = np.sum(nco_ice_arr) / (np.sum(nco_gas_arr) + np.sum(nco_ice_arr) + 1e-99)
        logger.info(f"CO ice fraction (by number): {ice_frac:.3f}")
        
        # Write output if requested
        if write_output:
            self.data.write_numberdens('co', self.nco_gas)
            logger.info("Wrote numberdens_co.inp")
        
        return X_co, self.nco_gas, self.nco_ice
    
    def evolve_co_time_dependent(
        self,
        t_end: Quantity,
        dt: Optional[Quantity] = None,
        Xco_tot: float = XCO_TOT_DEFAULT,
        tau_form: Quantity = TAU_CO_FORM_DEFAULT,
        k0_co: Quantity = K0_CO_DEFAULT,
        sigma_d_per_H: float = SIGMA_D_PER_H_DEFAULT,
        E_bind: float = E_BIND_CO_DEFAULT,
        nu0: float = NU0_CO_DEFAULT,
        alpha_pd_ice: float = ALPHA_PD_ICE_DEFAULT,
        Xco_gas_init: Optional[float] = None,
        Xco_ice_init: Optional[float] = None,
        use_dust_temp: bool = True,
        write_output: bool = True,
        skip_shielding: bool = False,
        nside: int = 4,
        b_kms: float = 0.3,
        margin_dex: float = 1.5,
    ) -> Tuple[Quantity, Quantity, Quantity]:
        """
        Time-dependent evolution of two-phase CO chemistry up to t_end.
        
        Integrates the coupled ODEs for gas-phase and ice CO densities:
        
        dnCO_gas/dt = R_form + k_td*nCO_ice + k_pd_ice*nCO_ice 
                      - k_fo*nCO_gas - k_pd*nCO_gas
        dnCO_ice/dt = k_fo*nCO_gas - k_td*nCO_ice - k_pd_ice*nCO_ice
        
        where R_form = (nCO_max - nCO_gas - nCO_ice) / tau_form
        
        This method does NOT use the Pinte chi/nH threshold switches.
        
        Parameters
        ----------
        t_end : Quantity
            Total evolution time
        dt : Quantity, optional
            Timestep. If None, automatically chosen based on rate timescales.
        Xco_tot : float
            Total CO abundance if all CO is present (per H)
        tau_form : Quantity
            CO formation timescale
        k0_co : Quantity
            Unshielded photodissociation rate for chi=1 Draine field [1/s]
        sigma_d_per_H : float
            Grain cross-section per H atom [cm^2 per H]
        E_bind : float
            CO binding energy in K
        nu0 : float
            Vibrational frequency for thermal desorption [1/s]
        alpha_pd_ice : float
            Photodesorption rate per chi=1 [1/s]
        Xco_gas_init : float, optional
            Initial gas-phase CO abundance. If None, starts with all CO in gas.
        Xco_ice_init : float, optional
            Initial ice CO abundance. If None, starts with no ice.
        use_dust_temp : bool
            If True, use dust temperature for freeze-out/desorption.
        write_output : bool
            If True, write numberdens_co.inp file at the end
        skip_shielding : bool
            If True, skip CO self-shielding calculation (theta_co = 1)
        nside : int
            HEALPix resolution for shielding calculation
        b_kms : float
            Doppler b parameter for Visser shielding [km/s]
        margin_dex : float
            Margin in dex for candidate cell selection
            
        Returns
        -------
        X_co : Quantity
            Final gas-phase CO abundance (dimensionless)
        nco_gas : Quantity
            Final gas-phase CO number density [cm^-3]
        nco_ice : Quantity
            Final CO ice number density [cm^-3]
        """
        # Ensure required fields exist (same as steady-state)
        if self.temperature is None:
            logger.info("Temperature not loaded, attempting to read from file...")
            try:
                self.read_temperature()
            except FileNotFoundError:
                raise RuntimeError(
                    "Temperature field required. Run mctherm or load temperature first."
                )
        
        if self.nH is None:
            logger.info("nH not computed, computing from gas density...")
            self.compute_nH_from_model()
        
        if self.chi is None:
            logger.info("chi not computed, attempting to compute from mcmono output...")
            try:
                self.compute_mcmono()
            except FileNotFoundError:
                raise RuntimeError(
                    "chi field required. Run compute_mcmono() first."
                )
        
        # Compute CO self-shielding if not skipped
        if skip_shielding:
            theta_co = Quantity(np.ones_like(self.chi.magnitude), "dimensionless")
            logger.info("Skipping CO self-shielding (theta_co = 1)")
        elif self.theta_co is not None:
            theta_co = self.theta_co
            logger.info("Using existing theta_co from previous calculation")
        else:
            from diskbridge.chemistry.shielding.healpix_columns import compute_co_shielding_healpix
            from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
            
            logger.info("Computing CO self-shielding with HEALPix rays...")
            visser = VisserShielding(b_kms=b_kms)
            theta_co, chi_eff = compute_co_shielding_healpix(
                mesh=self.model.mesh,
                nH=self.nH,
                chi=self.chi,
                visser=visser,
                nside=nside,
                b_kms=b_kms,
                margin_dex=margin_dex,
            )
            self.theta_co = theta_co
            self.chi_eff = chi_eff
        
        # Compute photodissociation rate
        k_pd, tau_pd = compute_co_photodissociation_rate(
            chi=self.chi,
            theta_co=theta_co,
            k0_co=k0_co,
            candidate_mask=None,
            min_rate=1.0e-30,
        )
        self.k_diss_co = k_pd
        self.tau_diss_co = tau_pd
        
        # Choose temperature
        T = self.temperature
        
        # Compute rate coefficients (assumed constant during evolution)
        k_fo = co_freezeout_rate(T, self.nH, sigma_d_per_H)
        k_td = co_thermal_desorption_rate(T, E_bind, nu0)
        k_pd_ice = co_photodesorption_rate(self.chi, alpha_pd_ice)
        
        # Extract magnitudes for fast array operations
        k_fo_s = k_fo.to("1/s").magnitude
        k_td_s = k_td.to("1/s").magnitude
        k_pd_s = k_pd.to("1/s").magnitude
        k_pd_ice_s = k_pd_ice.to("1/s").magnitude
        tau_s = tau_form.to("s").magnitude
        t_end_s = t_end.to("s").magnitude
        
        # Compute nco_max
        nH_cm3 = self.nH.to("cm^-3").magnitude
        nco_max = Xco_tot * nH_cm3
        
        # Initial conditions
        if Xco_gas_init is None:
            nco_gas = nco_max.copy()  # Start with all CO in gas phase
        else:
            nco_gas = Xco_gas_init * nH_cm3
        
        if Xco_ice_init is None:
            nco_ice = np.zeros_like(nco_gas)  # Start with no ice
        else:
            nco_ice = Xco_ice_init * nH_cm3
        
        # Choose timestep if not provided
        if dt is None:
            # Estimate minimum timescale from all rates
            # tau ~ 1 / sum_of_rates
            total_rate = k_fo_s + k_td_s + k_pd_s + k_pd_ice_s + 1.0 / tau_s
            min_timescale = 1.0 / np.maximum(total_rate, 1e-60)
            dt_s = 0.1 * np.nanmin(min_timescale)
            # Limit to reasonable range
            dt_s = np.clip(dt_s, 1e-10 * t_end_s, 0.01 * t_end_s)
            logger.info(f"Auto-selected timestep: dt = {dt_s:.2e} s")
        else:
            dt_s = dt.to("s").magnitude
        
        # Time integration (explicit Euler, vectorized over all cells)
        t = 0.0
        n_steps = 0
        max_steps = int(1e8)  # Safety limit
        
        logger.info(
            f"Starting time-dependent CO evolution: t_end={t_end_s:.2e} s, dt={dt_s:.2e} s"
        )
        
        while t < t_end_s and n_steps < max_steps:
            dt_step = min(dt_s, t_end_s - t)
            
            # Formation rate
            R_form = (nco_max - nco_gas - nco_ice) / tau_s
            
            # Rate of change for gas-phase CO
            dnco_gas_dt = (
                R_form
                + k_td_s * nco_ice
                + k_pd_ice_s * nco_ice
                - k_fo_s * nco_gas
                - k_pd_s * nco_gas
            )
            
            # Rate of change for ice CO
            dnco_ice_dt = (
                k_fo_s * nco_gas
                - k_td_s * nco_ice
                - k_pd_ice_s * nco_ice
            )
            
            # Update
            nco_gas = nco_gas + dt_step * dnco_gas_dt
            nco_ice = nco_ice + dt_step * dnco_ice_dt
            
            # Ensure non-negative (numerical stability)
            nco_gas = np.maximum(nco_gas, 0.0)
            nco_ice = np.maximum(nco_ice, 0.0)
            
            t += dt_step
            n_steps += 1
        
        if n_steps >= max_steps:
            logger.warning(f"Time evolution reached max_steps={max_steps}")
        
        logger.info(f"Time evolution complete after {n_steps} steps")
        
        # Store results
        self.nco_gas = Quantity(nco_gas, "cm^-3")
        self.nco_ice = Quantity(nco_ice, "cm^-3")
        
        # Compute gas-phase abundance
        X_co = Quantity(nco_gas / (nH_cm3 + 1e-99), "dimensionless")
        
        # Log summary
        logger.info(
            f"CO time-dependent final: nco_gas min={np.min(nco_gas):.2e}, "
            f"max={np.max(nco_gas):.2e} cm^-3"
        )
        logger.info(
            f"CO time-dependent final: nco_ice min={np.min(nco_ice):.2e}, "
            f"max={np.max(nco_ice):.2e} cm^-3"
        )
        ice_frac = np.sum(nco_ice) / (np.sum(nco_gas) + np.sum(nco_ice) + 1e-99)
        logger.info(f"CO ice fraction (by number): {ice_frac:.3f}")
        
        # Write output if requested
        if write_output:
            self.data.write_numberdens('co', self.nco_gas)
            logger.info("Wrote numberdens_co.inp")
        
        return X_co, self.nco_gas, self.nco_ice
