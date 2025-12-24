from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from diskbridge._logging import logger
from diskbridge._units import units, Quantity
from diskbridge.model import Model
from diskbridge.model.field import Field
from diskbridge.model.clipping import ClipIndexer, compute_clip_indexer

from diskbridge.model.profiles import (
    compute_volume_weighted_mean_radial_profile,
    find_r_split,
)
from .wavelengths import build_mcmono_wavelengths


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
    
    def _setup_segment(
        self,
        seg_model: Model,
        work_dir: Path,
        base_opacity_dir: Path,
    ) -> 'RadModel':
        """Setup segment: create RadModel, write inputs, link opacities."""
        from diskbridge.radmc3d.model import RadModel
        from diskbridge.radmc3d.utils import link_dustkappa_opacities
        
        work_dir.mkdir(parents=True, exist_ok=True)
        rad = RadModel(seg_model, model_dir=work_dir)
        rad.writer.write_all_input_files(work_dir)
        link_dustkappa_opacities(base_opacity_dir, rad.inputs_dir)
        return rad
    
    def _inherit_external_source(
        self,
        outer_rad: 'RadModel',
        inner_rad: 'RadModel',
        r_split_au: float,
        shell_ncells: int,
    ) -> None:
        """Extract shell spectrum from outer and write as external source for inner."""
        wavelengths_um, shell_spectrum = outer_rad.extract_shell_spectrum(
            r_split_au=r_split_au,
            shell_ncells=shell_ncells,
            mcmono_dir=outer_rad.outputs_dir / 'mcmono',
        )
        inner_rad.write_effective_external_source(
            wavelengths_um=wavelengths_um,
            spectrum=shell_spectrum,
            output_dir=inner_rad.inputs_dir,
            require_coverage=True,
        )
    
    def _run_segment_rt(
        self,
        rad: 'RadModel',
        nphot_therm: int,
        nphot_mono: int,
        mcmono_wav_um: np.ndarray,
        force: bool,
    ) -> None:
        """Run temperature and mcmono for a segment."""
        temp_dir = rad.outputs_dir / 'temperature'
        mcmono_dir = rad.outputs_dir / 'mcmono'
        rad.compute_temperature(nphot=nphot_therm, output_dir=temp_dir, force=force)
        rad.compute_mcmono(
            nphot=nphot_mono,
            output_dir=mcmono_dir,
            force=force,
            wavelengths_um=mcmono_wav_um,
        )

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
        max_splits: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        from diskbridge.radmc3d.model import RadModel
        from diskbridge.radmc3d.utils import link_dustkappa_opacities

        import diskbridge
        params = diskbridge.params

        tol_T = params.segmented_tol_T
        tol_chi = params.segmented_tol_chi
        window_fraction = params.segmented_window_fraction
        shell_ncells = params.segmented_shell_ncells
        r_clip_min_au = params.segmented_r_clip_min.to('au').magnitude
        stop_factor = float(params.segmented_stop_factor)
        if stop_factor <= 0.0 or stop_factor >= 1.0:
            raise ValueError("segmented_stop_factor must be in (0, 1)")

        nphot_therm_intermediate = int(params.segmented_nphot_thermal)
        nphot_mono_intermediate = int(params.segmented_nphot_mono)

        nphot_therm_final = int(params.nphot_thermal) if nphot_therm is None else int(nphot_therm)
        nphot_mono_final = int(params.nphot_mono) if nphot_mono is None else int(nphot_mono)

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

        base_opacity_dir = self.base_model_dir / 'radmc3d_inputs'
        base_opacity_dir.mkdir(parents=True, exist_ok=True)

        def _ensure_opacities_available() -> None:
            from diskbridge.radmc3d.writer import RadWriter

            if list(base_opacity_dir.glob('dustkappa_*.inp')):
                return
            writer = RadWriter(self.base_model)
            writer.compute_and_write_dust_opacities(self.base_model_dir)

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

            _ensure_opacities_available()
            rad = self._setup_segment(seg_model, segment.work_dir, base_opacity_dir)

            if mcmono_wav_um_use is None:
                n_uv = params.uv_n_wavelengths if mcmono_uv_n_wavelengths is None else int(mcmono_uv_n_wavelengths)
                mcmono_wav_um_use = build_mcmono_wavelengths(
                    wavelength_source=mcmono_wavelength_source,
                    wavelength_file=rad.inputs_dir / 'wavelength_micron.inp',
                    uv_min_um=params.uv_min.to('micron').magnitude,
                    uv_max_um=params.uv_max.to('micron').magnitude,
                    n_wavelengths=mcmono_n_wavelengths,
                    n_uv_enforce=n_uv,
                    spacing=mcmono_wavelength_spacing,
                    provided_wavelengths=mcmono_wavelengths_um,
                )

            if level > 0 and outer_rad is not None:
                self._inherit_external_source(outer_rad, rad, current_outer_rmax_au, shell_ncells)

            self._run_segment_rt(rad, nphot_therm_intermediate, nphot_mono_intermediate, mcmono_wav_um_use, force)

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

            if merged_T is None or merged_chi is None:
                raise ValueError("Internal error: missing merged fields")

            self.base_model.gas_register(
                'temperature',
                Field(quantity='temperature', data=merged_T, axis_order=axis_order),
            )
            self.base_model.gas_register(
                'chi',
                Field(quantity='chi', data=merged_chi, axis_order=axis_order),
            )

            r_au, chi_profile = compute_volume_weighted_mean_radial_profile(self.base_model, 'chi')
            _, T_profile = compute_volume_weighted_mean_radial_profile(self.base_model, 'temperature')
            r_edges_au = self.base_model.mesh.edges('r').to('au').magnitude  # type: ignore[union-attr]

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

            if float(r_split_au) >= float(current_outer_rmax_au):
                outer_rad = rad
                break

            if float(r_split_au) >= stop_factor * float(current_outer_rmax_au):
                final_work_dir = self.base_model_dir / 'segments' / f'segment_{level + 1:02d}_rmax_{current_outer_rmax_au:.6g}au_final'
                final_seg_bounds = {'r': (None, current_outer_rmax_au * units('au'))}
                final_segment = SegmentDefinition(
                    name=f'segment_{level + 1:02d}_final',
                    bounds=final_seg_bounds,
                    work_dir=final_work_dir,
                )
                final_model, final_indexer = self._build_segment_model_and_indexer(final_segment)
                
                _ensure_opacities_available()
                final_rad = self._setup_segment(final_model, final_work_dir, base_opacity_dir)
                
                if outer_rad is not None:
                    self._inherit_external_source(outer_rad, final_rad, current_outer_rmax_au, shell_ncells)
                
                self._run_segment_rt(final_rad, nphot_therm_final, nphot_mono_final, mcmono_wav_um_use, force)

                merged_T = self._merge_field(
                    merged=merged_T,
                    child=final_rad.temperature,
                    indexer=final_indexer,
                    axis_order=axis_order,
                )
                merged_chi = self._merge_field(
                    merged=merged_chi,
                    child=final_rad.chi,
                    indexer=final_indexer,
                    axis_order=axis_order,
                )
                segments.append(
                    {
                        'level': int(level + 1),
                        'work_dir': str(final_segment.work_dir),
                        'r_max_au': float(current_outer_rmax_au),
                    }
                )

                outer_rad = final_rad
                break

            split_radii_au.append(float(r_split_au))

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
