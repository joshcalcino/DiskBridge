"""RADMC-3D model wrapper for molecular line radiative transfer.

This module provides the RADMC3DModel class that wraps a DiskBridge Model
and provides high-level operations for RADMC-3D workflows:
- Reading RADMC-3D output files (via radmc3dData)
- Computing UV fields from mean intensity
- Applying photochemistry prescriptions (Pinte et al. 2018)
- Writing molecular number density files
    
This class does NOT build models or write RADMC-3D input files - use
RADMC3DWriter for that. This separation mirrors the radmc3dPy structure
where data (reading), setup (writing), and models (high-level) are separate.
"""

# db-keywords: uv-products, line-transfer, gas-temperature, config, units, radmc3d, model, mesh, field
# db-role: canonical
# db-scope: package
# db-purpose: RADMC-3D model wrapper for molecular line radiative transfer.

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Sequence
from pathlib import Path
import hashlib
import numpy as np
import shutil
import datetime

if TYPE_CHECKING:
    from diskbridge.model.core import Model

from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from diskbridge.model.field import Field
from diskbridge._config import get_config
from diskbridge.model.utils import field_data_as_order
from .data import RadData
from .cache import build_mesh_cache_context, should_use_cache, find_cached_output
from .run import SymlinkContext, run_radmc3d, organize_outputs, ensure_temperature_symlink
from .wavelengths import build_mcmono_wavelengths, write_wavelength_file, validate_wavelength_array, check_wavelength_range
from .uv_products import (
    UV_PRODUCT_MERGED_FIELD_NAMES,
    compute_chi_broad,
    compute_uv_products,
    default_isrf_path,
    file_sha256,
    uv_product_schema,
    uv_product_schema_hash,
    uv_product_edges_from_specs,
    uv_product_specs_from_config,
)
from .writer import RadWriter
import diskbridge 

C_LIGHT = units('c')
M_H = units('m_H')
SIGMA_SB = units('sigma_SB')

from diskbridge._constants import (
    EPS_CHI,
    LOG_CHI_OVER_NH_PDISS,
    MMP83_IR_TABLE,
)

U_DRAINE = Quantity(9.0e-14, 'erg/cm^3')
RADMC_PHOTON_COUNT_MAX = int(np.iinfo(np.int32).max)


def _validate_radmc_photon_count(value: int, *, name: str) -> int:
    """Return a RADMC-3D photon count that fits its signed 32-bit counter."""
    count = int(value)
    if count < 1 or count > RADMC_PHOTON_COUNT_MAX:
        raise ValueError(
            f"{name} must be between 1 and {RADMC_PHOTON_COUNT_MAX}; got {count}"
        )
    return count
 
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
    'external_ir_background',
    'external_ir_Tback',
    'external_cmb',
)

_MCMONO_EXTRA_PARAM_KEYS = (
    'nphot_mono',
    'uv_min',
    'uv_max',
    'uv_n_wavelengths',
    'external_uv',
    'external_uv_chi',
    'external_ir_background',
    'external_ir_Tback',
    'external_cmb',
)


def _external_source_cache_context(params) -> dict[str, object]:
    if not bool(getattr(params, "external_uv", False)):
        return {"external_source_enabled": False}
    ir_path = Path(__file__).resolve().parents[3] / "data" / MMP83_IR_TABLE
    return {
        "external_source_enabled": True,
        "external_mmp83_ir_table_hash": file_sha256(ir_path) if ir_path.is_file() else "",
    }


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
        self.uv_products: dict[str, Quantity] = {}
        self.radiation_mode: Optional[str] = None
        self.nH: Optional[Quantity] = None
        self.theta_co: Optional[Quantity] = None
        self.chi_eff: Optional[Quantity] = None
        self.k_diss_co: Optional[Quantity] = None
        self.tau_diss_co: Optional[Quantity] = None
        
        self.nco_gas: Optional[Quantity] = None
        self.nco_ice: Optional[Quantity] = None

        # Hydrogen partition (from chemistry, not thermal)
        self.nH2: Optional[Quantity] = None
        self.nH_atom: Optional[Quantity] = None
        
        # Carbon closure products (from chemistry, not thermal)
        self.nCplus: Optional[Quantity] = None
        self.nC: Optional[Quantity] = None
        self.ne: Optional[Quantity] = None
        
        self.inputs_dir = self.model_dir / 'radmc3d_inputs'
        self.outputs_dir = self.model_dir / 'radmc3d_outputs'
        
        self._active_symlinks: list[Path] = []
        self._inherit_canonical_rt_fields()

    def _inherit_canonical_rt_fields(self) -> None:
        """Initialize cached RT fields from canonical model fields.

        Returns
        -------
        None
            Updates ``dust_temperature`` and ``chi`` in place when the wrapped
            model already contains canonical gas fields with those names.
        """
        gas = getattr(self.model, "gas", None)
        if gas is None:
            return

        if self.dust_temperature is None and "dust_temperature" in gas:
            self.dust_temperature = gas["dust_temperature"].data
        if self.chi is None and "chi" in gas:
            self.chi = gas["chi"].data
        for name in UV_PRODUCT_MERGED_FIELD_NAMES:
            if name in gas:
                self.uv_products[name] = gas[name].data
        if all(name in self.uv_products for name in UV_PRODUCT_MERGED_FIELD_NAMES):
            if self.chi is None:
                self.chi = self.uv_products["chi_broad"]
            self.radiation_mode = "local_uv_products"
        elif self.chi is None and "chi_broad" in self.uv_products:
            self.chi = self.uv_products["chi_broad"]
            self.radiation_mode = "local_chi"
        elif self.chi is not None:
            self.radiation_mode = "local_chi"
    
    def _get_input_files(self) -> list[str]:
        """Get list of input files to symlink."""
        return [
            'amr_grid.inp', 'wavelength_micron.inp',
            'stars.inp', 'dustopac.inp',
            'dust_density.binp',
            'gas_velocity.binp',
            'gas_temperature.*',
            'levelpop_*.dat',
            'radmc3d.inp', 'external_source.inp',
            'numberdens_*.binp',
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

    def extract_shell_spectrum(
        self,
        r_split_au: float,
        shell_ncells: int,
        mcmono_dir: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        mcmono_dir = Path(mcmono_dir)
        wl_path = mcmono_dir / 'mcmono_wavelength_micron.inp'
        if not wl_path.exists():
            raise FileNotFoundError(f"Missing mcmono wavelength file: {wl_path}")

        with open(wl_path, 'r') as f:
            n = int(f.readline().strip())
            wavelengths_um = np.array([float(f.readline().strip()) for _ in range(n)], dtype=float)

        mean_candidates = [mcmono_dir / 'mean_intensity.bout']
        mean_path = None
        for p in mean_candidates:
            if p.exists():
                mean_path = p
                break
        if mean_path is None:
            raise FileNotFoundError(
                f"Missing mean intensity output in {mcmono_dir} (expected mean_intensity.bout or mean_intensity.out)"
            )

        _freq_hz, Jnu_flat = self.data.read_mean_intensity_file(mean_path)

        mesh = self.model.mesh
        if mesh is None:
            raise ValueError("Model has no mesh")
        if mesh.coord_system != 'spherical':
            raise ValueError(f"extract_shell_spectrum requires spherical mesh, got {mesh.coord_system}")

        r_edges_au = mesh.edges('r').to('au').magnitude
        nr = int(len(mesh.centers('r')))
        ntheta = int(len(mesh.centers('theta')))
        nphi = int(len(mesh.centers('phi')))
        ncells_expected = nr * ntheta * nphi
        if int(Jnu_flat.magnitude.shape[0]) != ncells_expected:
            raise ValueError(
                "Mean intensity cell count mismatch: got %d, expected %d" % (int(Jnu_flat.magnitude.shape[0]), int(ncells_expected))
            )

        r_split_idx = int(np.searchsorted(r_edges_au, float(r_split_au), side='left'))
        if r_split_idx <= 0 or r_split_idx >= nr:
            raise ValueError(f"r_split_au={r_split_au} AU outside grid range")

        shell_ncells = int(shell_ncells)
        if shell_ncells < 1:
            raise ValueError("shell_ncells must be >= 1")

        shell_start = r_split_idx
        shell_end = min(r_split_idx + shell_ncells, nr)
        if shell_end <= shell_start:
            raise ValueError(
                "Shell is empty: r_split_idx=%d, nr=%d" % (int(r_split_idx), int(nr))
            )

        ir_sel = np.arange(shell_start, shell_end, dtype=np.int64)

        itheta = np.arange(ntheta, dtype=np.int64)[None, :, None]
        iphi = np.arange(nphi, dtype=np.int64)[None, None, :]
        ir = ir_sel[:, None, None]
        idx = ir + nr * itheta + (nr * ntheta) * iphi
        idx_flat = idx.reshape(-1)

        J = np.asarray(Jnu_flat.magnitude, dtype=float)
        from diskbridge.model.profiles import compute_cell_volumes
        volumes = compute_cell_volumes(self.model)
        volumes_flat = np.asarray(volumes, dtype=float).reshape(-1, order='F')
        w = volumes_flat[idx_flat]
        wsum = float(np.sum(w))
        if wsum <= 0.0:
            raise ValueError("Non-positive shell volume sum")

        shell_spec = np.sum(J[idx_flat, :] * w[:, None], axis=0) / wsum

        if shell_spec.shape[0] != wavelengths_um.shape[0]:
            raise ValueError(
                "Mean intensity wavelength count mismatch: got %d, expected %d" % (int(shell_spec.shape[0]), int(wavelengths_um.shape[0]))
            )

        return wavelengths_um, shell_spec

    def write_effective_external_source(
        self,
        wavelengths_um: np.ndarray,
        spectrum: np.ndarray,
        output_dir: Path,
        require_coverage: bool = True,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        wav_path = output_dir / 'wavelength_micron.inp'
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing wavelength grid file: {wav_path}")

        with open(wav_path, 'r') as f:
            n = int(f.readline().strip())
            wav_um = np.array([float(f.readline().strip()) for _ in range(n)], dtype=float)

        wav_in = np.asarray(wavelengths_um, dtype=float)
        spec_in = np.asarray(spectrum, dtype=float)
        if wav_in.ndim != 1 or spec_in.ndim != 1:
            raise ValueError("wavelengths_um and spectrum must be 1D")
        if wav_in.size != spec_in.size:
            raise ValueError("wavelengths_um and spectrum must have same length")
        if wav_in.size < 2:
            raise ValueError("wavelengths_um must contain at least 2 points")

        sort_idx = np.argsort(wav_in)
        wav_in = wav_in[sort_idx]
        spec_in = spec_in[sort_idx]

        if require_coverage:
            if wav_in[0] > wav_um[0] or wav_in[-1] < wav_um[-1]:
                raise ValueError(
                    "Effective external spectrum does not cover wavelength_micron.inp: "
                    "spec=[%.6g, %.6g] um, grid=[%.6g, %.6g] um" % (float(wav_in[0]), float(wav_in[-1]), float(wav_um[0]), float(wav_um[-1]))
                )

        if np.any(spec_in <= 0.0):
            raise ValueError("loglog interpolation requires strictly positive spectrum values")

        if wav_in.size == wav_um.size and np.allclose(wav_in, wav_um, rtol=0.0, atol=0.0):
            spec_out = spec_in
        else:
            spec_out = np.exp(
                np.interp(
                    np.log(wav_um),
                    np.log(wav_in),
                    np.log(spec_in),
                )
            )

        ext_path = output_dir / 'external_source.inp'
        with open(ext_path, 'w') as f:
            f.write('2\n')
            f.write(f"{wav_um.size}\n")
            for w in wav_um:
                f.write(f"{w:13.6e}\n")
            for val in spec_out:
                f.write(f"{float(val):13.6e}\n")

        logger.info(f"Wrote external_source.inp file: {ext_path}")
        return ext_path

    def write_inherited_uv_external_source(
        self,
        wavelengths_um: np.ndarray,
        spectrum: np.ndarray,
        output_dir: Path,
        *,
        uv_min: Quantity,
        uv_max: Quantity,
    ) -> Path:
        """Replace only the UV interval of the configured external spectrum.

        Parameters
        ----------
        wavelengths_um, spectrum : ndarray
            Parent-shell UV wavelength grid and mean intensity.
        output_dir : pathlib.Path
            Child RADMC-3D input directory.
        uv_min, uv_max : Quantity
            Inclusive wavelength interval replaced by the inherited spectrum.

        Returns
        -------
        pathlib.Path
            Updated ``external_source.inp`` path.
        """
        output_dir = Path(output_dir)
        ext_path = output_dir / "external_source.inp"
        if not ext_path.exists():
            self.writer.write_external_source(self.model_dir)
        lines = ext_path.read_text().split()
        if len(lines) < 2:
            raise ValueError(f"Invalid external source file: {ext_path}")
        if int(lines[0]) != 2:
            raise ValueError(f"Unsupported external source format in {ext_path}")
        nlam = int(lines[1])
        if len(lines) != 2 + 2 * nlam:
            raise ValueError(f"Invalid external source value count in {ext_path}")
        target_wavelengths = np.asarray(lines[2 : 2 + nlam], dtype=np.float64)
        original = np.asarray(lines[2 + nlam :], dtype=np.float64)

        source_wavelengths = np.asarray(wavelengths_um, dtype=np.float64)
        source_spectrum = np.asarray(spectrum, dtype=np.float64)
        if source_wavelengths.ndim != 1 or source_spectrum.shape != source_wavelengths.shape:
            raise ValueError("Inherited wavelengths and spectrum must be matching 1-D arrays")
        order = np.argsort(source_wavelengths)
        source_wavelengths = source_wavelengths[order]
        source_spectrum = source_spectrum[order]
        if np.any(source_spectrum <= 0.0) or not np.all(np.isfinite(source_spectrum)):
            raise ValueError("Inherited UV spectrum must be positive and finite")

        lo = float(uv_min.to("micron").magnitude)
        hi = float(uv_max.to("micron").magnitude)
        uv_mask = (target_wavelengths >= lo) & (target_wavelengths <= hi)
        if not np.any(uv_mask):
            raise ValueError("External wavelength grid does not contain the inherited UV band")
        uv_target = target_wavelengths[uv_mask]
        if source_wavelengths[0] > uv_target[0] or source_wavelengths[-1] < uv_target[-1]:
            raise ValueError("Inherited spectrum does not cover the child UV wavelength interval")

        inherited = np.exp(
            np.interp(
                np.log(uv_target),
                np.log(source_wavelengths),
                np.log(source_spectrum),
            )
        )
        combined = original.copy()
        combined[uv_mask] = inherited
        with ext_path.open("w") as f:
            f.write("2\n")
            f.write(f"{nlam}\n")
            for wavelength in target_wavelengths:
                f.write(f"{wavelength:13.6e}\n")
            for value in combined:
                f.write(f"{value:13.6e}\n")
        logger.info("Wrote inherited UV external source: %s", ext_path)
        return ext_path
    
    def _update_radmc3d_inp_int_params(self, updates: dict[str, int]) -> None:
        radmc_inp_path = self.inputs_dir / 'radmc3d.inp'
        if not radmc_inp_path.exists():
            raise FileNotFoundError(
                f"Missing required input file: {radmc_inp_path}. "
                "Generate RADMC-3D inputs (including radmc3d.inp) before running mctherm/mcmono."
            )

        try:
            lines = radmc_inp_path.read_text().splitlines(True)
        except Exception as e:
            raise RuntimeError(f"Failed to read {radmc_inp_path}: {e}")

        pending = {str(k): int(v) for k, v in updates.items()}
        updated_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if (not stripped) or stripped.startswith('#') or ('=' not in line):
                updated_lines.append(line)
                continue

            name, _value = line.split('=', 1)
            key = name.strip()
            if key not in pending:
                updated_lines.append(line)
                continue

            updated_lines.append(f'{key} = {pending.pop(key)}\n')

        if pending:
            if updated_lines and not updated_lines[-1].endswith('\n'):
                updated_lines[-1] = updated_lines[-1] + '\n'
            for key, val in pending.items():
                updated_lines.append(f'{key} = {val}\n')

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

        axis_order = self.model.mesh.axis_names()
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

        axis_order = self.model.mesh.axis_names()
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

        self.nH = nH.to('cm^-3')
        
        logger.info(f"Computed nH from gas density: "
                   f"min={np.min(self.nH):.2e}, max={np.max(self.nH):.2e}")
        
        return self.nH

    def _chem_axis_order(self) -> Tuple[str, ...]:
        if 'density' in self.model.gas:
            return self.model.gas['density'].axis_order
        return self.model.mesh.axis_names()

    def compute_sigma_d_per_H_from_dust(self) -> Quantity:
        if self.model.dust is None:
            raise RuntimeError("Model has no dust submodel; cannot compute sigma_d_per_H")

        target = self._chem_axis_order()

        nH = self.ensure_nH().to('cm^-3')
        nH_cm3 = np.asarray(nH.magnitude)

        A_d = None
        for i in range(self.model.dust.nbin):
            bin_obj = self.model.dust[f'bin_{i}']
            dust_density_field = bin_obj['density']
            rho_d_i = field_data_as_order(dust_density_field, target).to('g/cm^3')
            a_i = bin_obj.size.to('cm').magnitude
            rho_s = bin_obj.density_material.to('g/cm^3').magnitude

            if float(a_i) <= 0.0:
                raise ValueError(f"Dust bin {i} has non-positive grain size a={a_i}")
            if float(rho_s) <= 0.0:
                raise ValueError(f"Dust bin {i} has non-positive material density rho_s={rho_s}")

            rho = np.asarray(rho_d_i.magnitude)
            A_i = (3.0 * rho) / (4.0 * float(a_i) * float(rho_s))
            if A_d is None:
                A_d = np.asarray(A_i, dtype=float)
            else:
                A_d = A_d + np.asarray(A_i, dtype=float)

        if A_d is None:
            raise RuntimeError("No dust bins available to compute sigma_d_per_H")

        sigma = np.divide(A_d, nH_cm3, out=np.zeros_like(A_d), where=(nH_cm3 > 0.0))
        sigma_q = Quantity(sigma, 'cm^2')

        self.model.gas_register(
            'sigma_d_per_H',
            Field(
                quantity='sigma_d_per_H',
                data=sigma_q,
                axis_order=target,
            ),
        )
        return sigma_q

    def _dust_temperature_bins_for_coupling(self, target: Tuple[str, ...], nbin: int) -> np.ndarray:
        """Return dust temperatures as ``(nbin, *mesh_shape)`` in K."""
        if self.dust_temperature is not None:
            temp = np.asarray(self.dust_temperature.to('K').magnitude, dtype=np.float64)
            if temp.shape == tuple(np.asarray(self.ensure_nH().magnitude).shape):
                return np.broadcast_to(temp, (nbin,) + temp.shape).copy()
            if temp.ndim == len(target) + 1 and temp.shape[0] >= nbin:
                return np.ascontiguousarray(temp[:nbin], dtype=np.float64)

        try:
            fpath = self.data._resolve_data_file(
                fname=None,
                basename='dust_temperature',
                missing_message="No dust_temperature file found",
            )
        except FileNotFoundError:
            temp_q = self.ensure_dust_temperature()
            temp = np.asarray(temp_q.to('K').magnitude, dtype=np.float64)
            if temp.shape != tuple(np.asarray(self.ensure_nH().magnitude).shape):
                raise
            return np.broadcast_to(temp, (nbin,) + temp.shape).copy()

        raw = self.data._read_scalar_data(fpath)
        if raw.ndim == 1:
            temps = [self.data._reshape_scalar_to_mesh(raw)]
        elif raw.ndim == 2:
            temps = [self.data._reshape_scalar_to_mesh(raw[i, :]) for i in range(raw.shape[0])]
        else:
            raise ValueError(f"Unexpected dust temperature data shape: {raw.shape}")
        temp_arr = np.asarray(temps, dtype=np.float64)
        if temp_arr.shape[0] == 1:
            return np.broadcast_to(temp_arr[0], (nbin,) + temp_arr.shape[1:]).copy()
        if temp_arr.shape[0] < nbin:
            raise ValueError(
                f"dust_temperature has {temp_arr.shape[0]} species, but dust model has {nbin} bins"
            )
        return np.ascontiguousarray(temp_arr[:nbin], dtype=np.float64)

    def compute_gas_dust_surface_area_coupling(self) -> tuple[Quantity, Quantity]:
        """Compute total projected dust area per H and surface-area-weighted Tdust."""
        if self.model.dust is None:
            raise RuntimeError("Model has no dust submodel; cannot compute gas-dust coupling")

        target = self._chem_axis_order()
        nH = self.ensure_nH().to('cm^-3')
        nH_cm3 = np.asarray(nH.magnitude, dtype=np.float64)
        nbin = int(self.model.dust.nbin)
        temp_bins = self._dust_temperature_bins_for_coupling(target, nbin)

        sigma_total = np.zeros_like(nH_cm3, dtype=np.float64)
        weighted_temp = np.zeros_like(nH_cm3, dtype=np.float64)

        for i in range(nbin):
            bin_obj = self.model.dust[f'bin_{i}']
            dust_density_field = bin_obj['density']
            rho_d_i = field_data_as_order(dust_density_field, target).to('g/cm^3')
            a_i = bin_obj.size.to('cm').magnitude
            rho_s = bin_obj.density_material.to('g/cm^3').magnitude

            if float(a_i) <= 0.0:
                raise ValueError(f"Dust bin {i} has non-positive grain size a={a_i}")
            if float(rho_s) <= 0.0:
                raise ValueError(f"Dust bin {i} has non-positive material density rho_s={rho_s}")

            rho = np.asarray(rho_d_i.magnitude, dtype=np.float64)
            area_density = (3.0 * rho) / (4.0 * float(a_i) * float(rho_s))
            sigma_i = np.divide(
                area_density,
                nH_cm3,
                out=np.zeros_like(area_density, dtype=np.float64),
                where=(nH_cm3 > 0.0),
            )
            sigma_i = np.where(np.isfinite(sigma_i) & (sigma_i > 0.0), sigma_i, 0.0)
            sigma_total += sigma_i
            weighted_temp += sigma_i * temp_bins[i]

        Tdust_gd = np.divide(
            weighted_temp,
            sigma_total,
            out=np.full_like(weighted_temp, 10.0, dtype=np.float64),
            where=(sigma_total > 0.0),
        )
        Tdust_gd = np.where(np.isfinite(Tdust_gd), Tdust_gd, 10.0)
        return Quantity(sigma_total, 'cm^2'), Quantity(Tdust_gd, 'K')

    def compute_h2_formation_surface_area(self) -> Quantity:
        """Projected ordinary-grain area per H for H2 formation."""
        if self.model.dust is None or int(self.model.dust.nbin) == 0:
            raise RuntimeError("No ordinary dust bins available for H2 surface area")

        target = self._chem_axis_order()
        nH = self.ensure_nH().to('cm^-3')
        nH_cm3 = np.asarray(nH.magnitude, dtype=np.float64)
        sigma_total = np.zeros_like(nH_cm3, dtype=np.float64)

        for i in range(int(self.model.dust.nbin)):
            bin_obj = self.model.dust[f'bin_{i}']
            if getattr(bin_obj, "role", "ordinary_dust") == "pah":
                continue
            rho_d_i = field_data_as_order(
                bin_obj['density'], target
            ).to('g/cm^3')
            a_i = float(bin_obj.size.to('cm').magnitude)
            rho_s = float(bin_obj.density_material.to('g/cm^3').magnitude)
            if a_i <= 0.0:
                raise ValueError(f"Dust bin {i} has non-positive size")
            if rho_s <= 0.0:
                raise ValueError(f"Dust bin {i} has non-positive material density")

            rho = np.asarray(rho_d_i.magnitude, dtype=np.float64)
            area_density = 3.0 * rho / (4.0 * a_i * rho_s)
            sigma_i = np.divide(
                area_density,
                nH_cm3,
                out=np.zeros_like(area_density, dtype=np.float64),
                where=(nH_cm3 > 0.0),
            )
            sigma_i = np.where(
                np.isfinite(sigma_i) & (sigma_i > 0.0),
                sigma_i,
                0.0,
            )
            sigma_total += sigma_i

        return Quantity(sigma_total, 'cm^2')

    def ensure_sigma_d_per_H(self, force: bool = False) -> Quantity:
        target = self._chem_axis_order()

        if (not force) and ('sigma_d_per_H' in self.model.gas):
            f = self.model.gas['sigma_d_per_H']
            return field_data_as_order(f, target).to('cm^2')

        return self.compute_sigma_d_per_H_from_dust()
    
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
        if nphot is None:
            nphot = int(self.params.nphot_thermal)
        nphot = _validate_radmc_photon_count(nphot, name="nphot_thermal")
        
        cache_context = {
            'nphot': int(nphot),
        }
        cache_context.update(build_mesh_cache_context(self.model.mesh))
        cache_context.update(_external_source_cache_context(self.params))
        
        countwrite = max(1, int(nphot // 100))
        countwrite = min(countwrite, int(np.iinfo(np.int32).max))
        
        if output_dir is None:
            output_dir = self.outputs_dir
        else:
            output_dir = Path(output_dir)

        use_cache, cached_file = should_use_cache(
            output_dir=output_dir,
            candidate_files=['dust_temperature.bdat'],
            current_params_path=self.model_dir / 'params.txt',
            param_keys=_MCTHERM_PARAM_KEYS,
            force=force,
            **cache_context,
        )
        
        if use_cache and cached_file:
            self.read_dust_temperature(fname=str(cached_file))
            return self.dust_temperature
        
        output_dir.mkdir(parents=True, exist_ok=True)
        self._update_radmc3d_inp_int_params({'nphot': int(nphot)})
        self._ensure_cntdump_ge_countwrite(countwrite, nphot)
        
        if getattr(self.params, 'external_uv', False):
            external_source_path = self.inputs_dir / 'external_source.inp'
            if not external_source_path.exists():
                writer = RadWriter(self.model, organize_files=True)
                writer.write_external_source(self.model_dir)

        stale = self.model_dir / 'dust_temperature.bdat'
        if stale.is_symlink():
            stale.unlink()
        
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
            output_files=['dust_temperature.bdat'],
            description=f'radmc3d mctherm nphot={nphot}',
            cache_context=cache_context,
        )
        
        temp_file = find_cached_output(
            output_dir,
            ['dust_temperature.bdat'],
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

        u_band = np.trapezoid(u_nu_sorted, nu_sorted, axis=1)

        chi_flat = (u_band / u_draine).to('dimensionless')

        chi_3d = chi_flat.reshape(mesh_shape, order='F')

        return chi_3d, int(np.count_nonzero(uv_mask))
    
    def _postprocess_chi(
        self,
        mean_intensity_file: Path,
        uv_min: Quantity,
        uv_max: Quantity,
        compute_products: bool = False,
    ) -> Quantity:
        """Load mean intensity and register UV radiation fields.

        Parameters
        ----------
        mean_intensity_file : pathlib.Path
            RADMC-3D mean-intensity output file.
        uv_min : Quantity
            Lower wavelength bound for legacy broad ``chi``.
        uv_max : Quantity
            Upper wavelength bound for legacy broad ``chi``.
        compute_products : bool, optional
            Compute and register process-specific UV products. When enabled,
            ``chi`` is set to ``chi_broad``.

        Returns
        -------
        Quantity
            Broad UV field in Draine units.
        """
        freq_hz, Jnu_flat = self.data.read_mean_intensity_file(mean_intensity_file)
        self.mean_intensity = Jnu_flat
        nx, ny, nz = self.data._getMeshShape()
        mesh_shape = (nx, ny, nz)
        axis_order = self.model.mesh.axis_names()
        uv_cfg = get_config().get("radmc3d", {}).get("uv_products", {})
        uv_specs = uv_product_specs_from_config(uv_cfg)

        if compute_products:
            products = compute_uv_products(
                freq_hz=freq_hz,
                Jnu_flat=Jnu_flat,
                mesh_shape=mesh_shape,
                specs=uv_specs,
                isrf_path=default_isrf_path(),
            )
            self.uv_products = products
            self.chi = products["chi_broad"]
            self.radiation_mode = "local_uv_products"

            self.model.gas_register(
                'chi',
                Field(
                    quantity='chi',
                    data=self.chi,
                    axis_order=axis_order,
                ),
            )
            for name, arr in products.items():
                if name == "F_CO_pdes_photon_bands":
                    continue
                self.model.gas_register(
                    name,
                    Field(
                        quantity=name,
                        data=arr,
                        axis_order=axis_order,
                    ),
                )

            logger.info(
                "Computed UV products from mean_intensity: "
                "chi_broad min=%.2e max=%.2e, G_CO_diss min=%.2e max=%.2e",
                float(np.nanmin(products["chi_broad"].magnitude)),
                float(np.nanmax(products["chi_broad"].magnitude)),
                float(np.nanmin(products["G_CO_diss"].magnitude)),
                float(np.nanmax(products["G_CO_diss"].magnitude)),
            )

            try:
                self.summarize_chi_over_nH()
            except Exception as e:
                logger.warning(f"summarize_chi_over_nH failed: {e}")

            return self.chi
        
        self.chi = compute_chi_broad(
            freq_hz=freq_hz,
            Jnu_flat=Jnu_flat,
            mesh_shape=mesh_shape,
            specs=uv_specs,
            isrf_path=default_isrf_path(),
        )
        self.uv_products = {"chi_broad": self.chi}
        self.radiation_mode = "local_chi"
        
        self.model.gas_register(
            'chi',
            Field(
                quantity='chi',
                data=self.chi,
                axis_order=axis_order,
            ),
        )
        self.model.gas_register(
            'chi_broad',
            Field(
                quantity='chi_broad',
                data=self.chi,
                axis_order=axis_order,
            ),
        )
        
        logger.info(
            "Computed canonical chi_broad from mean_intensity "
            f"(91.2-206.7 nm): min={np.min(self.chi):.2e}, max={np.max(self.chi):.2e}"
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
    
    def _build_mcmono_wavelengths(
        self,
        wavelengths_um: Optional[np.ndarray],
        uv_min: Quantity,
        uv_max: Quantity,
        n_wavelengths: int,
        compute_uv_products: bool,
    ) -> np.ndarray:
        """Build mcmono wavelength grid.
        
        Returns
        -------
        ndarray
            Wavelength grid in microns
        """
        extra_edges_um = None
        if compute_uv_products:
            uv_cfg = get_config().get("radmc3d", {}).get("uv_products", {})
            uv_specs = uv_product_specs_from_config(uv_cfg)
            # This contract is still a bit confusing: uv_n_wavelengths is the
            # base grid count, but UV product boundaries are appended exactly.
            extra_edges_um = uv_product_edges_from_specs(uv_specs) * 1.0e-3

        if wavelengths_um is not None:
            mcmono_lam_um = build_mcmono_wavelengths(
                wavelength_source="uv",
                wavelength_file=None,
                uv_min_um=uv_min.to("micron").magnitude,
                uv_max_um=uv_max.to("micron").magnitude,
                n_wavelengths=n_wavelengths,
                n_uv_enforce=0,
                spacing="log",
                provided_wavelengths=wavelengths_um,
                extra_enforced_wavelengths_um=extra_edges_um,
            )
        else:
            mcmono_lam_um = build_mcmono_wavelengths(
                wavelength_source="uv",
                wavelength_file=None,
                uv_min_um=uv_min.to("micron").magnitude,
                uv_max_um=uv_max.to("micron").magnitude,
                n_wavelengths=n_wavelengths,
                n_uv_enforce=0,
                spacing="log",
                extra_enforced_wavelengths_um=extra_edges_um,
            )

        check_wavelength_range(
            mcmono_lam_um,
            self.params.lambda_min,
            self.params.lambda_max,
            "mcmono wavelength grid",
        )
        return mcmono_lam_um

    def _write_mcmono_wavelengths(self, mcmono_lam_um: np.ndarray) -> None:
        """Write mcmono wavelength grid."""
        validate_wavelength_array(mcmono_lam_um, "mcmono wavelength grid")
        
        mcmono_wav_file = self.model_dir / 'mcmono_wavelength_micron.inp'
        write_wavelength_file(mcmono_wav_file, mcmono_lam_um, file_format='mcmono')
    
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
        setthreads: Optional[int] = None,
        cache_context: Optional[dict] = None,
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
            if not temp_symlink:
                raise RuntimeError(
                    "mcmono requires dust_temperature.dat or dust_temperature.bdat, but no temperature file was found."
                )

            ctx._active_symlinks.append(temp_symlink)
            
            if setthreads is None:
                setthreads = self.params.nbcores
            setthreads = int(setthreads)
            logger.info(
                f"Running mcmono: {mcmono_lam_um.size} wavelengths "
                f"({mcmono_lam_um[0]:.6g}-{mcmono_lam_um[-1]:.6g} um), "
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
            output_files=['mean_intensity.bout', 'mcmono_wavelength_micron.inp'],
            description=f'radmc3d mcmono range_{mcmono_lam_um[0]:.6g}-{mcmono_lam_um[-1]:.6g}micron_{mcmono_lam_um.size}wavelengths',
            cache_context=cache_context,
        )
    
    def compute_mcmono(
        self,
        nphot: int = None,
        output_dir: Optional[str | Path] = None,
        force: bool = False,
        wavelengths_um: Optional[np.ndarray] = None,
        uv_min: Quantity = None,
        uv_max: Quantity = None,
        n_wavelengths: int = None,
        setthreads: Optional[int] = None,
        compute_uv_products: bool = False,
        iseed: Optional[int] = None,
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
        compute_uv_products : bool, optional
            Compute process-specific UV products from this mcmono spectrum.
        iseed : int, optional
            Deterministic RADMC-3D random seed.
            
        Returns
        -------
        Quantity
            UV field chi in Draine units
        """
        nphot, uv_min, uv_max, n_wavelengths, all_params_used = self._resolve_mcmono_config(
            nphot, uv_min, uv_max, n_wavelengths, wavelengths_um
        )
        nphot = _validate_radmc_photon_count(nphot, name="nphot_mono")
        
        mcmono_lam_um = self._build_mcmono_wavelengths(
            wavelengths_um,
            uv_min,
            uv_max,
            n_wavelengths,
            compute_uv_products,
        )

        cache_context = {
            'nphot': int(nphot),
            'uv_min_um': float(uv_min.to('micron').magnitude),
            'uv_max_um': float(uv_max.to('micron').magnitude),
            'n_wavelengths': int(n_wavelengths),
            'wavelengths_sha256': hashlib.sha256(
                np.asarray(mcmono_lam_um, dtype=np.float64).tobytes()
            ).hexdigest(),
            'wavelengths_size': int(mcmono_lam_um.size),
            'wavelengths_min_um': float(np.min(mcmono_lam_um)),
            'wavelengths_max_um': float(np.max(mcmono_lam_um)),
        }
        cache_context.update(build_mesh_cache_context(self.model.mesh))
        if iseed is not None:
            iseed = int(iseed)
            if iseed <= 0:
                raise ValueError("iseed must be positive")
            cache_context["iseed"] = iseed
        cache_context.update(_external_source_cache_context(self.params))

        if compute_uv_products:
            isrf_path = default_isrf_path()
            uv_cfg = get_config().get("radmc3d", {}).get("uv_products", {})
            uv_specs = uv_product_specs_from_config(uv_cfg)
            schema = uv_product_schema(uv_specs)
            cache_context.update(
                {
                    'uv_products_enabled': True,
                    'uv_product_mode': 'disc_segment_only',
                    'uv_product_schema_sha256': uv_product_schema_hash(uv_specs),
                    'uv_product_partition_edges_nm': schema["partition_edges_nm"],
                    'uv_product_band_edges_nm': [
                        [p["lam_min_nm"], p["lam_max_nm"]]
                        for p in schema["products"]
                    ],
                    'uv_product_weight_types': [
                        p["weight"] for p in schema["products"]
                    ],
                    'uv_product_reference_isrf_hash': file_sha256(isrf_path),
                }
            )

        if output_dir is None:
            output_dir = self.outputs_dir
        else:
            output_dir = Path(output_dir)
        
        use_cache, cached_file = should_use_cache(
            output_dir=output_dir,
            candidate_files=['mean_intensity.bout'],
            current_params_path=self.model_dir / 'params.txt',
            param_keys=_MCTHERM_PARAM_KEYS + _MCMONO_EXTRA_PARAM_KEYS,
            force=force,
            **cache_context,
        )
        
        if use_cache and cached_file:
            return self._postprocess_chi(
                cached_file,
                uv_min,
                uv_max,
                compute_products=compute_uv_products,
            )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        countwrite = max(1, min(int(nphot // 100), int(np.iinfo(np.int32).max)))

        inp_updates = {'nphot_mono': int(nphot)}
        if iseed is not None:
            inp_updates['iseed'] = iseed
        self._update_radmc3d_inp_int_params(inp_updates)
        self._ensure_cntdump_ge_countwrite(countwrite, nphot)
        
        self._write_mcmono_wavelengths(mcmono_lam_um)
        self._prepare_mcmono_run(output_dir)
        self._run_mcmono(
            output_dir,
            mcmono_lam_um,
            nphot,
            countwrite,
            setthreads=setthreads,
            cache_context=cache_context,
        )
        
        mean_intensity_file = find_cached_output(
            output_dir,
            ['mean_intensity.bout'],
        )
        if not mean_intensity_file:
            raise FileNotFoundError(f"Mean intensity file not found in {output_dir}")
        
        return self._postprocess_chi(
            mean_intensity_file,
            uv_min,
            uv_max,
            compute_products=compute_uv_products,
        )

    def compute_segmented_rt(
        self,
        nphot_therm: Optional[int] = None,
        nphot_mono: Optional[int] = None,
        mcmono_n_wavelengths: Optional[int] = None,
        mcmono_uv_n_wavelengths: Optional[int] = None,
        mcmono_wavelength_spacing: str = 'log',
        mcmono_wavelengths_um: Optional[np.ndarray] = None,
        max_splits: Optional[int] = None,
        segmented_final_nphot_multiplier: Optional[float] = None,
        force: bool = False,
        diagnostic_plots: bool = False,
        plots_dir: Optional[str | Path] = None,
    ) -> dict:
        """Run segmented dust-temperature and UV-field RADMC-3D calculations.

        Parameters
        ----------
        nphot_therm : int, optional
            Photon count for thermal Monte Carlo runs.
        nphot_mono : int, optional
            Photon count for monochromatic Monte Carlo runs.
        mcmono_n_wavelengths : int, optional
            Number of wavelengths to use for monochromatic transfer.
        mcmono_uv_n_wavelengths : int, optional
            Number of UV wavelengths to enforce.
        mcmono_wavelength_spacing : str, optional
            Wavelength spacing mode.
        mcmono_wavelengths_um : ndarray, optional
            Explicit monochromatic wavelengths in micron.
        max_splits : int, optional
            Maximum number of radial split updates.
        segmented_final_nphot_multiplier : float, optional
            Multiplier for the terminal segment photon count.
        force : bool, optional
            Recompute outputs even when cached outputs are available.
        diagnostic_plots : bool, optional
            Whether to write diagnostic plots after the merged fields are
            built. When enabled, writes merged plots plus per-segment plots
            under ``plots_dir / "segments"``.
        plots_dir : str or pathlib.Path, optional
            Diagnostic plot directory. Defaults to
            ``model_dir / "plots" / "segmented_rt"``.

        Returns
        -------
        dict
            Segmented-run metadata and merged ``temperature``/``chi`` fields.
        """
        from diskbridge.radmc3d.segmented import SegmentedRadRunner

        runner = SegmentedRadRunner(self.model, self.model_dir)
        out = runner.run_segmented_rt(
            nphot_therm=nphot_therm,
            nphot_mono=nphot_mono,
            mcmono_n_wavelengths=mcmono_n_wavelengths,
            mcmono_uv_n_wavelengths=mcmono_uv_n_wavelengths,
            mcmono_wavelength_spacing=mcmono_wavelength_spacing,
            mcmono_wavelengths_um=mcmono_wavelengths_um,
            max_splits=max_splits,
            segmented_final_nphot_multiplier=segmented_final_nphot_multiplier,
            force=force,
            diagnostic_plots=diagnostic_plots,
            plots_dir=plots_dir,
        )

        self.dust_temperature = out.get('temperature')
        self.chi = out.get('chi')
        if out.get('uv_products') is not None:
            self.uv_products = out['uv_products']
            self.radiation_mode = "local_uv_products"
        elif self.chi is not None:
            self.radiation_mode = "local_chi"
        split_radii = list(out.get('split_radii_au', []))
        isotropic_outside = float(split_radii[-1]) if split_radii else None
        self.isotropic_weight_outside_r_au = isotropic_outside
        self.model.segmented_isotropic_weight_outside_r_au = isotropic_outside
        out['isotropic_weight_outside_r_au'] = isotropic_outside
        return out

    def load_segmented_rt_outputs(
        self,
        *,
        diagnostic_plots: bool = False,
        plots_dir: Optional[str | Path] = None,
    ) -> dict:
        """Load existing segmented RT outputs without running RADMC-3D.

        Parameters
        ----------
        diagnostic_plots : bool, optional
            Whether to write diagnostic plots after the merged fields are
            built. When enabled, writes merged plots plus per-segment plots
            under ``plots_dir / "segments"``.
        plots_dir : str or pathlib.Path, optional
            Diagnostic plot directory. Defaults to
            ``model_dir / "plots" / "segmented_rt"``.

        Returns
        -------
        dict
            Segmented-run metadata and merged ``temperature``/``chi`` fields.
        """
        from diskbridge.radmc3d.segmented import SegmentedRadRunner

        runner = SegmentedRadRunner(self.model, self.model_dir)
        out = runner.load_segmented_rt_outputs(
            diagnostic_plots=diagnostic_plots,
            plots_dir=plots_dir,
        )

        self.dust_temperature = out.get("temperature")
        self.chi = out.get("chi")
        if out.get("uv_products") is not None:
            self.uv_products = out["uv_products"]
            self.radiation_mode = "local_uv_products"
        elif self.chi is not None:
            self.radiation_mode = "local_chi"
        split_radii = list(out.get("split_radii_au", []))
        isotropic_outside = float(split_radii[-1]) if split_radii else None
        self.isotropic_weight_outside_r_au = isotropic_outside
        self.model.segmented_isotropic_weight_outside_r_au = isotropic_outside
        out["isotropic_weight_outside_r_au"] = isotropic_outside
        return out

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

    def has_uv_product(self, name: str) -> bool:
        """Return whether a UV product is available.

        Parameters
        ----------
        name : str
            UV product field name.

        Returns
        -------
        bool
            ``True`` when the product is cached on this wrapper or registered
            on the gas model.
        """
        if name in self.uv_products:
            return True
        gas = getattr(self.model, "gas", None)
        return bool(gas is not None and name in gas)

    def set_local_uv_products(self, products: dict[str, Quantity]) -> None:
        """Register local RADMC-like UV products on this wrapper."""
        missing = [name for name in UV_PRODUCT_MERGED_FIELD_NAMES if name not in products]
        if missing:
            raise KeyError(f"Missing UV product(s): {missing}")
        self.uv_products = {name: products[name] for name in UV_PRODUCT_MERGED_FIELD_NAMES}
        self.chi = self.uv_products["chi_broad"]
        self.radiation_mode = "local_uv_products"

    def set_incident_uv(self, *, chi: Quantity, Av: Quantity) -> None:
        """Register an incident 1D slab UV field and dust-depth profile."""
        self.chi = chi
        self.Av = Av
        self.uv_products = {}
        self.radiation_mode = "incident_slab_chi"

    def set_incident_uv_products(
        self,
        *,
        products: dict[str, Quantity],
        Av: Quantity,
    ) -> None:
        """Register incident 1D slab UV products and dust-depth profile."""
        missing = [name for name in UV_PRODUCT_MERGED_FIELD_NAMES if name not in products]
        if missing:
            raise KeyError(f"Missing UV product(s): {missing}")
        self.uv_products = {name: products[name] for name in UV_PRODUCT_MERGED_FIELD_NAMES}
        self.chi = self.uv_products["chi_broad"]
        self.Av = Av
        self.radiation_mode = "incident_slab_uv_products"

    def ensure_uv_product(
        self,
        name: str,
        fallback_to_chi: bool = False,
    ) -> Quantity:
        """Return a UV product, optionally falling back to ``chi``.

        Parameters
        ----------
        name : str
            UV product field name.
        fallback_to_chi : bool, optional
            Return ``chi`` when the requested product is unavailable.

        Returns
        -------
        Quantity
            UV product field.

        Raises
        ------
        KeyError
            If the product is unavailable and fallback is disabled.
        """
        if name in self.uv_products:
            return self.uv_products[name]

        gas = getattr(self.model, "gas", None)
        if gas is not None and name in gas:
            product = gas[name].data
            self.uv_products[name] = product
            return product

        if name == "chi_broad" and self.chi is not None:
            self.uv_products[name] = self.chi
            return self.chi

        if fallback_to_chi:
            return self.ensure_chi()

        raise KeyError(name)
    
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
