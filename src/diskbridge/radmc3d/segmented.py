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
    compute_volume_weighted_median_radial_profile,
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

        # ===== Phase 1: Outer run (full domain) =====
        outer_dir = self.base_model_dir / 'outer_run'
        outer_dir.mkdir(parents=True, exist_ok=True)

        outer_writer_dir = outer_dir  # writer will create radmc3d_inputs within
        outer_rad = RadModel(self.base_model, model_dir=outer_writer_dir)
        outer_rad.writer.write_all_input_files(outer_writer_dir)
        outer_rad.writer.compute_and_write_dust_opacities(outer_writer_dir)

        outer_temp_dir = outer_rad.outputs_dir / 'temperature'
        outer_mcmono_dir = outer_rad.outputs_dir / 'mcmono'

        outer_rad.compute_temperature(nphot=nphot_therm, output_dir=outer_temp_dir, force=force)
        outer_rad.compute_mcmono(nphot=nphot_mono, output_dir=outer_mcmono_dir, force=force)

        # ===== Phase 2: Determine R_split =====
        r_au, chi_profile = compute_volume_weighted_median_radial_profile(self.base_model, 'chi')
        _, T_profile = compute_volume_weighted_median_radial_profile(self.base_model, 'temperature')
        r_edges_au = self.base_model.mesh.edges('r').to('au').magnitude  # type: ignore[union-attr]

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

        # ===== Phase 3: Shell spectrum extraction =====
        wavelengths_um, shell_spectrum = outer_rad.extract_shell_spectrum(
            r_split_au=r_split_au,
            shell_ncells=shell_ncells,
            mcmono_dir=outer_mcmono_dir,
        )

        # ===== Phase 4: Inner run =====
        inner_bounds: Dict[str, Tuple[Optional[Quantity], Optional[Quantity]]] = {
            'r': (None, r_split_au * units('au')),
        }
        inner_segment = SegmentDefinition(
            name='inner',
            bounds=inner_bounds,
            work_dir=self.base_model_dir / 'inner_run',
        )

        inner_model, inner_indexer = self._build_segment_model_and_indexer(inner_segment)

        inner_segment.work_dir.mkdir(parents=True, exist_ok=True)
        inner_rad = RadModel(inner_model, model_dir=inner_segment.work_dir)
        inner_rad.writer.write_all_input_files(inner_segment.work_dir)
        inner_rad.writer.compute_and_write_dust_opacities(inner_segment.work_dir)

        # Override external source with effective spectrum
        inner_rad.write_effective_external_source(
            wavelengths_um=wavelengths_um,
            spectrum=shell_spectrum,
            output_dir=inner_rad.inputs_dir,
        )

        inner_temp_dir = inner_rad.outputs_dir / 'temperature'
        inner_mcmono_dir = inner_rad.outputs_dir / 'mcmono'
        inner_rad.compute_temperature(nphot=nphot_therm, output_dir=inner_temp_dir, force=force)
        inner_rad.compute_mcmono(nphot=nphot_mono, output_dir=inner_mcmono_dir, force=force)

        # ===== Phase 5: Merge =====
        mesh = self.base_model.mesh
        if mesh is None:
            raise ValueError("Base model has no mesh")
        axis_order = ('r', 'phi', 'theta') if mesh.coord_system == 'spherical' else mesh.axis_names()

        merged_T = self._merge_field(
            merged=outer_rad.temperature,
            child=inner_rad.temperature,
            indexer=inner_indexer,
            axis_order=axis_order,
        )
        merged_chi = self._merge_field(
            merged=outer_rad.chi,
            child=inner_rad.chi,
            indexer=inner_indexer,
            axis_order=axis_order,
        )

        # Register fields back on base model
        self.base_model.gas_register(
            'temperature',
            Field(quantity='temperature', data=merged_T, axis_order=axis_order),
        )
        self.base_model.gas_register(
            'chi',
            Field(quantity='chi', data=merged_chi, axis_order=axis_order),
        )

        # seam diagnostics (mean of last inner + first outer radial cells)
        seam_idx = int(r_split_info['r_split_cell_idx'])
        t_mag = merged_T.to('K').magnitude
        chi_mag = merged_chi.to('dimensionless').magnitude

        if seam_idx - 1 >= 0 and seam_idx < t_mag.shape[0]:
            T_inner = float(np.mean(t_mag[seam_idx - 1, :, :]))
            T_outer = float(np.mean(t_mag[seam_idx, :, :]))
            chi_inner = float(np.mean(chi_mag[seam_idx - 1, :, :]))
            chi_outer = float(np.mean(chi_mag[seam_idx, :, :]))

            if T_outer == 0.0:
                raise ValueError("Invalid seam diagnostic: T_outer_seam is zero")
            if chi_outer == 0.0:
                raise ValueError("Invalid seam diagnostic: chi_outer_seam is zero")

            seam = {
                'T_inner_seam': T_inner,
                'T_outer_seam': T_outer,
                'T_discontinuity_frac': abs(T_outer - T_inner) / abs(T_outer),
                'chi_inner_seam': chi_inner,
                'chi_outer_seam': chi_outer,
                'chi_discontinuity_frac': abs(chi_outer - chi_inner) / abs(chi_outer),
            }
        else:
            seam = {'error': 'Invalid seam indices'}

        logger.info(f"Segmented RT complete. R_split={r_split_au:.3f} AU")

        return {
            'r_split_au': float(r_split_au),
            'r_split_info': r_split_info,
            'seam_diagnostics': seam,
            'temperature': merged_T,
            'chi': merged_chi,
            'outer_rad': outer_rad,
            'inner_rad': inner_rad,
            'inner_indexer': inner_indexer,
        }
