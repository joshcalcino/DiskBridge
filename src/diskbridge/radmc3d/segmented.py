from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from diskbridge._logging import logger
from diskbridge._units import units, Quantity
from diskbridge.model.model import Model
from diskbridge.model.field import Field
from diskbridge.model.clipping import ClipIndexer, compute_clip_indexer

from diskbridge.model.profiles import (
    compute_volume_weighted_mean_radial_profile,
    find_r_split,
)


@dataclass(frozen=True)
class SegmentDefinition:
    name: str
    bounds: Dict[str, Tuple[Optional[Quantity], Optional[Quantity]]]
    work_dir: Path


class SegmentedRadmcRunner:
    def __init__(self, base_model: Model, base_model_dir: Path):
        self.base_model = base_model
        self.base_model_dir = Path(base_model_dir)

    def _build_segment_model_and_indexer(
        self,
        segment: SegmentDefinition,
    ) -> tuple[Model, ClipIndexer]:
        mesh0 = self.base_model.mesh
        if mesh0 is None:
            raise ValueError("Base model has no mesh")

        indexer, _ = compute_clip_indexer(mesh0, segment.bounds)
        clipped = self.base_model.clip_mesh(
            **{f"{ax}_min": v[0] for ax, v in segment.bounds.items() if v[0] is not None},
            **{f"{ax}_max": v[1] for ax, v in segment.bounds.items() if v[1] is not None},
        )
        return clipped, indexer

    def _merge_field(
        self,
        merged: Quantity,
        child: Quantity,
        indexer: ClipIndexer,
        axis_order: Tuple[str, ...],
    ) -> Quantity:
        merged_mag = np.asarray(merged.magnitude)
        child_mag = np.asarray(child.magnitude)

        slicer = []
        for ax in axis_order:
            slicer.append(indexer.axis_slices.get(ax, slice(None)))

        merged_mag[tuple(slicer)] = child_mag
        return Quantity(merged_mag, merged.units)

    def run_segmented_rt(
        self,
        nphot_therm: Optional[int] = None,
        nphot_mono: Optional[int] = None,
        mcmono_n_wavelengths: Optional[int] = None,
        mcmono_uv_n_wavelengths: Optional[int] = None,
        mcmono_wavelength_source: str = 'external',
        mcmono_wavelength_spacing: str = 'log',
        mcmono_wavelengths_um: Optional[np.ndarray] = None,
        effective_external_interpolation: str = 'loglog',
        max_splits: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        from diskbridge.radmc3d.model import RadModel

        import diskbridge
        params = diskbridge.params

        tol_T = params.segmented_tol_T
        tol_chi = params.segmented_tol_chi
        window_fraction = params.segmented_window_fraction
        shell_ncells = params.segmented_shell_ncells
        r_clip_min_au = params.segmented_r_clip_min.to('au').magnitude

        if max_splits is None:
            max_splits = int(params.segmented_max_splits)
        max_splits = int(max_splits)
        if max_splits < 0:
            raise ValueError("max_splits must be >= 0")

        mesh = self.base_model.mesh
        if mesh is None:
            raise ValueError("Base model has no mesh")
        axis_order = ('r', 'phi', 'theta') if mesh.coord_system == 'spherical' else mesh.axis_names()

        segments: list[dict[str, Any]] = []
        split_radii_au: list[float] = []
        r_edges_base_au = self.base_model.mesh.edges('r').to('au').magnitude  # type: ignore[union-attr]
        current_outer_rmax_au = float(np.max(r_edges_base_au))
        outer_rad: Optional[RadModel] = None
        merged_T: Optional[Quantity] = None
        merged_chi: Optional[Quantity] = None

        mcmono_wav_um_use: Optional[np.ndarray] = None

        for level in range(max_splits + 1):
            if level == 0:
                seg_bounds: Dict[str, Tuple[Optional[Quantity], Optional[Quantity]]] = {}
                seg_work_dir = self.base_model_dir / 'segments' / f'segment_{level:02d}_full'
            else:
                seg_bounds = {'r': (None, current_outer_rmax_au * units('au'))}
                seg_work_dir = self.base_model_dir / 'segments' / f'segment_{level:02d}_rmax_{current_outer_rmax_au:.6g}au'

            segment = SegmentDefinition(
                name=f'segment_{level:02d}',
                bounds=seg_bounds,
                work_dir=seg_work_dir,
            )

            if level == 0:
                seg_model = self.base_model
                seg_indexer = None
            else:
                seg_model, seg_indexer = self._build_segment_model_and_indexer(segment)

            segment.work_dir.mkdir(parents=True, exist_ok=True)
            rad = RadModel(seg_model, model_dir=segment.work_dir)
            rad.writer.write_all_input_files(segment.work_dir)
            rad.writer.compute_and_write_dust_opacities(segment.work_dir)

            temp_dir = rad.outputs_dir / 'temperature'
            mcmono_dir = rad.outputs_dir / 'mcmono'

            if mcmono_wav_um_use is None:
                if mcmono_wavelengths_um is not None:
                    mcmono_wav_um_use = mcmono_wavelengths_um
                else:
                    if mcmono_wavelength_source == 'uv':
                        lam_min = params.uv_min.to('micron').magnitude
                        lam_max = params.uv_max.to('micron').magnitude
                        n_use = params.uv_n_wavelengths if mcmono_n_wavelengths is None else int(mcmono_n_wavelengths)
                        if n_use < 2:
                            raise ValueError("mcmono_n_wavelengths must be >= 2")
                        if mcmono_wavelength_spacing == 'linear':
                            mcmono_wav_um_use = np.linspace(lam_min, lam_max, n_use)
                        elif mcmono_wavelength_spacing == 'log':
                            mcmono_wav_um_use = np.exp(np.linspace(np.log(lam_min), np.log(lam_max), n_use))
                        else:
                            raise ValueError(
                                "mcmono_wavelength_spacing must be one of: 'linear', 'log'"
                            )
                    elif mcmono_wavelength_source == 'external':
                        wav_file = rad.inputs_dir / 'wavelength_micron.inp'
                        if not wav_file.is_file():
                            raise FileNotFoundError(f"wavelength grid file not found: {wav_file}")
                        with open(wav_file, 'r') as f:
                            n_global = int(f.readline().strip())
                            wav_global = np.array([float(f.readline().strip()) for _ in range(n_global)], dtype=float)
                        if mcmono_n_wavelengths is None:
                            mcmono_wav_um_use = wav_global
                        else:
                            lam_min = float(np.min(wav_global))
                            lam_max = float(np.max(wav_global))
                            n_use = int(mcmono_n_wavelengths)
                            if n_use < 2:
                                raise ValueError("mcmono_n_wavelengths must be >= 2")
                            if mcmono_wavelength_spacing == 'linear':
                                mcmono_wav_um_use = np.linspace(lam_min, lam_max, n_use)
                            elif mcmono_wavelength_spacing == 'log':
                                mcmono_wav_um_use = np.exp(np.linspace(np.log(lam_min), np.log(lam_max), n_use))
                            else:
                                raise ValueError(
                                    "mcmono_wavelength_spacing must be one of: 'linear', 'log'"
                                )
                    else:
                        raise ValueError(
                            "mcmono_wavelength_source must be one of: 'uv', 'external'"
                        )

                n_uv = params.uv_n_wavelengths if mcmono_uv_n_wavelengths is None else int(mcmono_uv_n_wavelengths)
                if n_uv == 1:
                    raise ValueError("mcmono_uv_n_wavelengths must be >= 2 (or 0 to disable UV enforcement)")
                if n_uv > 0:
                    uv_min_um = float(params.uv_min.to('micron').magnitude)
                    uv_max_um = float(params.uv_max.to('micron').magnitude)
                    uv_grid = np.linspace(uv_min_um, uv_max_um, n_uv)
                    mcmono_wav_um_use = np.unique(
                        np.concatenate([np.asarray(mcmono_wav_um_use, dtype=float), uv_grid])
                    )

            if level > 0 and outer_rad is not None:
                wavelengths_um, shell_spectrum = outer_rad.extract_shell_spectrum(
                    r_split_au=current_outer_rmax_au,
                    shell_ncells=shell_ncells,
                    mcmono_dir=outer_rad.outputs_dir / 'mcmono',
                )
                rad.write_effective_external_source(
                    wavelengths_um=wavelengths_um,
                    spectrum=shell_spectrum,
                    output_dir=rad.inputs_dir,
                    interpolation=effective_external_interpolation,
                    require_coverage=True,
                )

            rad.compute_temperature(nphot=nphot_therm, output_dir=temp_dir, force=force)
            rad.compute_mcmono(
                nphot=nphot_mono,
                output_dir=mcmono_dir,
                force=force,
                wavelengths_um=mcmono_wav_um_use,
            )

            if merged_T is None or merged_chi is None:
                merged_T = rad.temperature
                merged_chi = rad.chi
            else:
                if seg_indexer is None:
                    raise ValueError("Internal error: missing indexer for inner segment")
                merged_T = self._merge_field(
                    merged=merged_T,
                    child=rad.temperature,
                    indexer=seg_indexer,
                    axis_order=axis_order,
                )
                merged_chi = self._merge_field(
                    merged=merged_chi,
                    child=rad.chi,
                    indexer=seg_indexer,
                    axis_order=axis_order,
                )

            segments.append(
                {
                    'level': int(level),
                    'work_dir': str(segment.work_dir),
                    'r_max_au': float(current_outer_rmax_au),
                }
            )

            if level >= max_splits:
                outer_rad = rad
                break

            r_au, chi_profile = compute_volume_weighted_mean_radial_profile(seg_model, 'chi')
            _, T_profile = compute_volume_weighted_mean_radial_profile(seg_model, 'temperature')
            r_edges_au = seg_model.mesh.edges('r').to('au').magnitude  # type: ignore[union-attr]

            try:
                r_split_au, r_split_info = find_r_split(
                    r_au=r_au,
                    r_edges_au=r_edges_au,
                    chi_profile=chi_profile,
                    T_profile=T_profile,
                    tol_chi=tol_chi,
                    tol_T=tol_T,
                    window_fraction=window_fraction,
                    r_clip_min_au=float(r_clip_min_au),
                )
            except ValueError:
                outer_rad = rad
                break

            split_radii_au.append(float(r_split_au))

            if float(r_split_au) >= float(current_outer_rmax_au):
                outer_rad = rad
                break

            outer_rad = rad
            current_outer_rmax_au = float(r_split_au)

        if merged_T is None or merged_chi is None:
            raise ValueError("Segmented RT produced no results")

        self.base_model.gas_register(
            'temperature',
            Field(quantity='temperature', data=merged_T, axis_order=axis_order),
        )
        self.base_model.gas_register(
            'chi',
            Field(quantity='chi', data=merged_chi, axis_order=axis_order),
        )

        logger.info(
            f"Segmented RT complete. splits={len(split_radii_au)} max_splits={max_splits}"
        )

        return {
            'split_radii_au': split_radii_au,
            'segments': segments,
            'temperature': merged_T,
            'chi': merged_chi,
        }
