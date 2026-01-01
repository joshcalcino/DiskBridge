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
from .cache import should_use_cache, find_cached_output
from .run import SymlinkContext, run_radmc3d, organize_outputs, ensure_temperature_symlink
from .wavelengths import build_wavelength_grid, write_wavelength_file, validate_wavelength_array, check_wavelength_range
from .writer import RadWriter
import diskbridge 

C_LIGHT = units('c')
M_H = units('m_H')
SIGMA_SB = units('sigma_SB')

from diskbridge._constants import EPS_CHI, LOG_CHI_OVER_NH_PDISS

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
        
        self.data = RadData(model, model_dir)
        self.writer = RadWriter(model)
        self.params = diskbridge.params
        
        self.dust_temperature: Optional[Quantity] = None
        self.gas_temperature: Optional[Quantity] = None
        self.chi: Optional[Quantity] = None
        self.nH: Optional[Quantity] = None
        self.theta_co: Optional[Quantity] = None
        self.chi_eff: Optional[Quantity] = None
        self.k_diss_co: Optional[Quantity] = None
        self.tau_diss_co: Optional[Quantity] = None
        
        self.nco_gas: Optional[Quantity] = None
        self.nco_ice: Optional[Quantity] = None
        
        self.inputs_dir = self.model_dir / 'radmc3d_inputs'
        self.outputs_dir = self.model_dir / 'radmc3d_outputs'
        
        self._active_symlinks: list[Path] = []
    
    def _get_input_files(self) -> list[str]:
        """Get list of input files to symlink."""
        return [
            'amr_grid.inp', 'wavelength_micron.inp',
            'stars.inp', 'dustopac.inp',
            'dust_density.binp', 'dust_density.inp',
            'gas_velocity.binp', 'gas_velocity.inp',
            'radmc3d.inp', 'external_source.inp',
            'numberdens_*.inp', 'numberdens_*.binp'
        ]

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
        """
        self.gas_temperature = self.data.readGasTemp()
        
        axis_order = ('r', 'phi', 'theta') if self.model.mesh.coord_system == 'spherical' else self.model.mesh.axis_names()
        self.model.gas_register(
            'gas_temperature',
            Field(
                quantity='gas_temperature',
                data=self.gas_temperature,
                axis_order=axis_order,
            ),
        )
        
        return self.gas_temperature
    
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
        """
        self.dust_temperature = self.data.readDustTemp(fname=fname, ispec=ispec)

        if self.model.mesh.coord_system == 'spherical':
            self.dust_temperature = np.transpose(self.dust_temperature, (0, 2, 1))

        axis_order = ('r', 'phi', 'theta') if self.model.mesh.coord_system == 'spherical' else self.model.mesh.axis_names()
        self.model.gas_register(
            'dust_temperature',
            Field(
                quantity='dust_temperature',
                data=self.dust_temperature,
                axis_order=axis_order,
            ),
        )
        
        return self.dust_temperature
    
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
    
    def compute_nH_from_model(self) -> Quantity:
        """Compute H nuclei number density from model gas density.
        
        Returns
        -------
        Quantity
            Number density of H nuclei in cm^-3
        """
        if 'density' not in self.model.gas:
            raise KeyError("Model has no gas density field")
        
        rho_gas = self.model.gas['density'].data.to('g/cm^3')
        MU_HNUC = 1.4
        nH = rho_gas / (MU_HNUC * M_H)
        
        field = self.model.gas['density']
        if hasattr(field, 'axis_order'):
            axis_order = field.axis_order
            if axis_order == ('r', 'phi', 'theta'):
                nH = np.transpose(nH, (0, 2, 1))
        
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

        ratio = chi / (nH + EPS_CHI)
        log_ratio = np.log10(np.maximum(ratio, EPS_CHI))

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
        """
        use_params_nphot = nphot is None
        if nphot is None:
            nphot = int(self.params.nphot_thermal)
        
        countwrite = max(1, int(nphot // 100))
        countwrite = min(countwrite, int(np.iinfo(np.int32).max))
        
        if output_dir is None:
            output_dir = self.outputs_dir
        else:
            output_dir = Path(output_dir)

        use_cache, cached_file = should_use_cache(
            output_dir=output_dir,
            candidate_files=['dust_temperature.bdat', 'dust_temperature.dat', 'dust_temperature.binp'],
            current_params_path=self.model_dir / 'params.txt',
            param_keys=_MCTHERM_PARAM_KEYS,
            force=force,
            use_params_nphot=use_params_nphot,
        )
        
        if use_cache and cached_file:
            self.read_dust_temperature(fname=str(cached_file))
            return self.dust_temperature
        
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_cntdump_ge_countwrite(countwrite, nphot)
        
        if getattr(self.params, 'external_uv', False):
            external_source_path = self.inputs_dir / 'external_source.inp'
            if not external_source_path.exists():
                writer = RadWriter(self.model, organize_files=True)
                writer.write_external_source(self.model_dir)
        
        with SymlinkContext(
            model_dir=self.model_dir,
            inputs_dir=self.inputs_dir,
            input_files=self._get_input_files(),
        ):
            logger.info(f"Running RADMC-3D mctherm with {nphot} photons (countwrite={countwrite})...")
            run_radmc3d(
                command=['mctherm', 'countwrite', str(countwrite)],
                model_dir=self.model_dir,
                log_section='mctherm',
                preserve_log=False,
            )
        
        organize_outputs(
            output_dir=output_dir,
            model_dir=self.model_dir,
            output_files=['dust_temperature.dat', 'dust_temperature.bdat', 'dust_temperature.binp'],
            description=f'radmc3d mctherm nphot={nphot}',
        )
        
        temp_file = find_cached_output(
            output_dir,
            ['dust_temperature.bdat', 'dust_temperature.dat', 'dust_temperature.binp'],
        )
        if temp_file:
            self.read_dust_temperature(fname=str(temp_file))
        
        return self.dust_temperature

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
        lam = C_LIGHT / freq_hz

        uv_mask = (lam >= uv_min) & (lam <= uv_max)
        if not np.any(uv_mask):
            raise ValueError(
                f"No UV wavelengths ({uv_min:~P}-{uv_max:~P}) found in {source}. "
                f"Wavelength range: {lam.min():~P}-{lam.max():~P}"
            )

        j_uv = j_lambda[:, uv_mask]
        nu_uv = freq_hz[uv_mask]
        u_nu = 4.0 * np.pi * j_uv / C_LIGHT.to_base_units()

        sort_idx = np.argsort(nu_uv)
        nu_sorted = nu_uv[sort_idx]
        u_nu_sorted = u_nu[:, sort_idx]

        u_band = np.trapz(u_nu_sorted, nu_sorted, axis=1)

        chi_flat = (u_band / u_draine).to('dimensionless')

        chi_3d = chi_flat.reshape(mesh_shape, order='F')

        if self.model.mesh.coord_system == 'spherical':
            chi_3d = np.transpose(chi_3d, (0, 2, 1))

        return chi_3d, int(np.count_nonzero(uv_mask))
    
    def _postprocess_chi(
        self,
        mean_intensity_file: Path,
        uv_min: Quantity,
        uv_max: Quantity,
    ) -> Quantity:
        """Load mean intensity, compute chi, register as field.
        
        This unifies postprocessing for both cache-hit and recompute paths.
        """
        freq_hz, Jnu_flat = self.data.read_mean_intensity_file(mean_intensity_file)
        self.mean_intensity = Jnu_flat
        nx, ny, nz = self.data._getMeshShape()
        
        chi_3d, n_uv = self._compute_chi_from_mean_intensity(
            j_lambda=Jnu_flat,
            freq_hz=freq_hz,
            uv_min=uv_min,
            uv_max=uv_max,
            mesh_shape=(nx, ny, nz),
            u_draine=U_DRAINE,
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
            logger.warning(f"summarize_chi_over_nH failed: {e}")
        
        return self.chi
    
    def _resolve_mcmono_config(
        self,
        nphot: Optional[int],
        uv_min: Optional[Quantity],
        uv_max: Optional[Quantity],
        n_wavelengths: Optional[int],
        wavelengths_um: Optional[np.ndarray],
    ) -> Tuple[int, Quantity, Quantity, int, bool]:
        """Resolve mcmono configuration from params and arguments.
        
        Returns
        -------
        tuple
            (nphot, uv_min, uv_max, n_wavelengths, all_params_used)
        """
        use_params_nphot = nphot is None
        use_params_uv_min = uv_min is None
        use_params_uv_max = uv_max is None
        use_params_nw = n_wavelengths is None
        use_params_wavelengths = wavelengths_um is None
        
        if nphot is None:
            nphot = self.params.nphot_mono
        if uv_min is None:
            uv_min = self.params.uv_min
        if uv_max is None:
            uv_max = self.params.uv_max
        if n_wavelengths is None:
            n_wavelengths = self.params.uv_n_wavelengths
        
        if uv_min < self.params.lambda_min or uv_max > self.params.lambda_max:
            raise ValueError(
                f"mcmono UV range [{uv_min:~P},{uv_max:~P}] outside global grid "
                f"[{self.params.lambda_min:~P},{self.params.lambda_max:~P}]"
            )
        
        logger.info(f"UV field: [{uv_min:~P},{uv_max:~P}], n={n_wavelengths}")
        
        all_params_used = (
            use_params_nphot and use_params_uv_min and 
            use_params_uv_max and use_params_nw and use_params_wavelengths
        )
        
        return nphot, uv_min, uv_max, n_wavelengths, all_params_used
    
    def _prepare_mcmono_wavelengths(
        self,
        wavelengths_um: Optional[np.ndarray],
        uv_min: Quantity,
        uv_max: Quantity,
        n_wavelengths: int,
    ) -> np.ndarray:
        """Prepare mcmono wavelength grid.
        
        Returns
        -------
        ndarray
            Wavelength grid in microns
        """
        if wavelengths_um is not None:
            validate_wavelength_array(wavelengths_um, "wavelengths_um")
            mcmono_lam_um = np.asarray(wavelengths_um, dtype=float)
            check_wavelength_range(
                mcmono_lam_um,
                self.params.lambda_min,
                self.params.lambda_max,
                "mcmono wavelength grid",
            )
        else:
            mcmono_lam_um = build_wavelength_grid(
                wmin=uv_min,
                wmax=uv_max,
                n_wavelengths=n_wavelengths,
                spacing='linear',
            )
        
        mcmono_wav_file = self.model_dir / 'mcmono_wavelength_micron.inp'
        write_wavelength_file(mcmono_wav_file, mcmono_lam_um, file_format='mcmono')
        return mcmono_lam_um
    
    def _prepare_mcmono_run(self, output_dir: Path) -> None:
        """Prepare environment for mcmono run (external source, temperature)."""
        if getattr(self.params, 'external_uv', False):
            external_source_path = self.inputs_dir / 'external_source.inp'
            if not external_source_path.exists():
                writer = RadWriter(self.model, organize_files=True)
                writer.write_external_source(self.model_dir)
    
    def _run_mcmono(
        self,
        output_dir: Path,
        mcmono_lam_um: np.ndarray,
        nphot: int,
        countwrite: int,
    ) -> None:
        """Run RADMC-3D mcmono and organize outputs."""
        with SymlinkContext(
            model_dir=self.model_dir,
            inputs_dir=self.inputs_dir,
            input_files=self._get_input_files(),
        ) as ctx:
            temp_symlink = ensure_temperature_symlink(
                model_dir=self.model_dir,
                outputs_dir=self.outputs_dir,
                output_dir=output_dir,
            )
            if temp_symlink:
                ctx._active_symlinks.append(temp_symlink)
            
            setthreads = self.params.nbcores
            logger.info(
                f"Running mcmono: {mcmono_lam_um.size} wavelengths "
                f"({mcmono_lam_um[0]:.6g}-{mcmono_lam_um[-1]:.6g} µm), "
                f"{nphot} photons"
            )
            run_radmc3d(
                command=['mcmono', 'setthreads', str(setthreads), 'countwrite', str(countwrite)],
                model_dir=self.model_dir,
                log_section='mcmono',
                preserve_log=True,
            )
        
        organize_outputs(
            output_dir=output_dir,
            model_dir=self.model_dir,
            output_files=['mean_intensity.out', 'mean_intensity.bout', 'mcmono_wavelength_micron.inp'],
            description=f'radmc3d mcmono range_{mcmono_lam_um[0]:.6g}-{mcmono_lam_um[-1]:.6g}micron_{mcmono_lam_um.size}wavelengths',
        )
    
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
            Number of photon packages
        output_dir : str or Path, optional
            Output directory (default: outputs_dir)
        force : bool, optional
            Force recomputation
        wavelengths_um : ndarray, optional
            Custom wavelength grid in microns
        uv_min, uv_max : Quantity, optional
            UV range bounds
        n_wavelengths : int, optional
            Number of wavelengths
            
        Returns
        -------
        Quantity
            UV field chi in Draine units
        """
        nphot, uv_min, uv_max, n_wavelengths, all_params_used = self._resolve_mcmono_config(
            nphot, uv_min, uv_max, n_wavelengths, wavelengths_um
        )
        
        if output_dir is None:
            output_dir = self.outputs_dir
        else:
            output_dir = Path(output_dir)
        
        use_cache, cached_file = should_use_cache(
            output_dir=output_dir,
            candidate_files=['mean_intensity.bout', 'mean_intensity.out'],
            current_params_path=self.model_dir / 'params.txt',
            param_keys=_MCTHERM_PARAM_KEYS + _MCMONO_EXTRA_PARAM_KEYS,
            force=force,
            use_params_nphot=all_params_used,
        )
        
        if use_cache and cached_file:
            return self._postprocess_chi(cached_file, uv_min, uv_max)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        countwrite = max(1, min(int(nphot // 100), int(np.iinfo(np.int32).max)))
        self._ensure_cntdump_ge_countwrite(countwrite, nphot)
        
        mcmono_lam_um = self._prepare_mcmono_wavelengths(
            wavelengths_um, uv_min, uv_max, n_wavelengths
        )
        self._prepare_mcmono_run(output_dir)
        self._run_mcmono(output_dir, mcmono_lam_um, nphot, countwrite)
        
        mean_intensity_file = find_cached_output(
            output_dir,
            ['mean_intensity.bout', 'mean_intensity.out'],
        )
        if not mean_intensity_file:
            raise FileNotFoundError(f"Mean intensity file not found in {output_dir}")
        
        return self._postprocess_chi(mean_intensity_file, uv_min, uv_max)
    
    def ensure_dust_temperature(self, force: bool = False) -> Quantity:
        """Ensure dust temperature field exists, reading or computing as needed.
        
        Parameters
        ----------
        force : bool, optional
            Force recomputation even if temperature exists (default: False)
            
        Returns
        -------
        Quantity
            Dust temperature field in Kelvin
            
        Raises
        ------
        RuntimeError
            If dust temperature cannot be obtained
        """
        if self.dust_temperature is not None and not force:
            return self.dust_temperature
        
        try:
            self.read_dust_temperature()
        except Exception:
            self.compute_temperature(force=force)
        
        if self.dust_temperature is None:
            raise RuntimeError('Dust temperature not available after ensure_dust_temperature')
        
        return self.dust_temperature
    
    def ensure_gas_temperature(self) -> Optional[Quantity]:
        """Ensure gas temperature field exists if available.
        
        Returns
        -------
        Quantity or None
            Gas temperature field in Kelvin if available, else None
            
        Notes
        -----
        Unlike ensure_dust_temperature, this does not compute if missing.
        Gas temperature must be read from file or set by a thermal solver.
        """
        if self.gas_temperature is not None:
            return self.gas_temperature
        
        try:
            self.read_gas_temperature()
        except Exception:
            pass
        
        return self.gas_temperature
    
    def ensure_temperature(self, force: bool = False) -> Quantity:
        """Ensure temperature field exists (alias to ensure_dust_temperature).
        
        Parameters
        ----------
        force : bool, optional
            Force recomputation even if temperature exists (default: False)
            
        Returns
        -------
        Quantity
            Dust temperature field in Kelvin
            
        Raises
        ------
        RuntimeError
            If dust temperature cannot be obtained
            
        Notes
        -----
        This is kept as an alias to ensure_dust_temperature for backward compatibility.
        For new code, prefer ensure_dust_temperature or ensure_gas_temperature explicitly.
        """
        return self.ensure_dust_temperature(force=force)
    
    def ensure_nH(self) -> Quantity:
        """Ensure H nuclei number density exists, computing from gas density if needed.
        
        Returns
        -------
        Quantity
            Number density of H nuclei in cm^-3
            
        Raises
        ------
        RuntimeError
            If nH cannot be computed
        """
        if self.nH is None:
            self.compute_nH_from_model()
        
        if self.nH is None:
            raise RuntimeError('nH not available after ensure_nH')
        
        return self.nH
    
    def ensure_chi(
        self,
        force: bool = False,
        uv_min: Optional[Quantity] = None,
        uv_max: Optional[Quantity] = None,
        n_wavelengths: Optional[int] = None,
    ) -> Quantity:
        """Ensure UV field exists, computing with mcmono if needed.
        
        Parameters
        ----------
        force : bool, optional
            Force recomputation even if chi exists (default: False)
        uv_min : Quantity, optional
            Minimum UV wavelength (uses params if None)
        uv_max : Quantity, optional
            Maximum UV wavelength (uses params if None)
        n_wavelengths : int, optional
            Number of wavelengths (uses params if None)
            
        Returns
        -------
        Quantity
            UV field in Draine units
            
        Raises
        ------
        RuntimeError
            If chi cannot be computed
        """
        if self.chi is not None and not force:
            return self.chi
        
        self.compute_mcmono(
            force=force,
            uv_min=uv_min,
            uv_max=uv_max,
            n_wavelengths=n_wavelengths,
        )
        
        if self.chi is None:
            raise RuntimeError('chi not available after ensure_chi')
        
        return self.chi
    
    def _organize_output(
        self,
        output_dir: Path,
        files: list[str],
        command: str
    ) -> None:
        """Organize RADMC-3D output files into a directory."""
        for fname in files:
            src = self.model_dir / fname
            if src.exists():
                dst = output_dir / fname
                shutil.move(str(src), str(dst))
                logger.debug(f"Moved {fname} -> {output_dir}")
        
        params_file = self.model_dir / 'params.txt'
        if params_file.exists():
            dst_params = output_dir / 'params.txt'
            shutil.copy2(str(params_file), str(dst_params))
            
            with open(dst_params, 'a') as f:
                f.write('\n# --- Run metadata (auto-generated) ---\n')
                f.write(f'timestamp = {datetime.datetime.now().isoformat()}\n')
                f.write(f'radmc3d_command = {command}\n')
            
            logger.debug(f"Copied params.txt -> {output_dir}")
