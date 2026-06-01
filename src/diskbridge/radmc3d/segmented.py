from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json
import re

import numpy as np
import shutil

from diskbridge._logging import logger
from diskbridge._config import get_config
from diskbridge._units import units, Quantity
from diskbridge.model import Model
from diskbridge.model.field import Field
from diskbridge.model.clipping import ClipIndexer, compute_clip_indexer

from diskbridge.model.profiles import (
    compute_volume_weighted_mean_radial_profile,
    find_r_split,
)
from diskbridge.chemistry.shielding.angular_uv_weights import compute_star_uv_source_strength
from diskbridge.radmc3d.uv_products import (
    UV_PRODUCT_MERGED_FIELD_NAMES,
    draine_references_for_product_partitions,
    draine_reference_for_product,
    uv_product_edges_from_specs,
    uv_product_specs_from_config,
    validate_uv_chemistry_config,
)
from .wavelengths import build_mcmono_wavelengths


@dataclass(frozen=True)
class SegmentDefinition:
    name: str
    bounds: Dict[str, Tuple[Optional[Quantity], Optional[Quantity]]]
    work_dir: Path


class SegmentedRadRunner:
    def __init__(self, base_model: Model, base_model_dir: Path):
        self.base_model = base_model
        self.base_model_dir = Path(base_model_dir)

    def _uv_products_enabled(self) -> bool:
        """Return whether segmented UV products are enabled in configuration.

        Returns
        -------
        bool
            ``True`` when ``[radmc3d.uv_products].enabled`` is enabled.
        """
        cfg = get_config()
        return validate_uv_chemistry_config(cfg, segmented=True).products_enabled

    def _stellar_product_threshold_radius_au(
        self,
        product_specs,
        threshold: float,
    ) -> Optional[float]:
        """Return radius where stellar UV reaches the configured product fraction."""
        import diskbridge

        params = diskbridge.params
        threshold = float(threshold)
        if threshold <= 0.0:
            return None

        chi_ext0 = 0.0
        if getattr(params, "external_uv", False):
            chi_ext0 = float(getattr(params, "external_uv_chi", 0.0))
        chi_ext0 = max(chi_ext0, np.finfo(np.float64).tiny)

        c_cgs = float(units("c").to("cm/s").magnitude)
        radii_cm: list[float] = []
        for spec in product_specs:
            lam_min_cm = float(spec.band.lam_min_nm) * 1.0e-7
            lam_max_cm = float(spec.band.lam_max_nm) * 1.0e-7
            source_strength = compute_star_uv_source_strength(
                params,
                lam_min_cm,
                lam_max_cm,
                weighting=spec.band.weight,
            )
            if source_strength <= 0.0:
                continue

            if spec.band.weight == "photon":
                ref_flux = float(
                    draine_reference_for_product(spec.field_name, product_specs)
                    .to("1/(cm^2 s)")
                    .magnitude
                )
            else:
                ref_flux = c_cgs * float(
                    draine_reference_for_product(spec.field_name, product_specs)
                    .to("erg/cm^3")
                    .magnitude
                )
            if ref_flux <= 0.0:
                continue

            radii_cm.append(
                float(np.sqrt(source_strength / (4.0 * np.pi * ref_flux * chi_ext0 * threshold)))
            )

        if not radii_cm:
            return None
        return max(radii_cm) * units("cm").to("au").magnitude

    def _parse_segment_dir(self, segment_dir: Path) -> tuple[int, float | None, bool]:
        """Parse a segmented-run directory name.

        Parameters
        ----------
        segment_dir : pathlib.Path
            Segment working directory.

        Returns
        -------
        tuple
            Segment level, outer radius in au if present, and whether this is
            a final full-photon segment.

        Raises
        ------
        ValueError
            If the directory name is not a segmented-run directory.
        """
        name = segment_dir.name
        full_match = re.fullmatch(r"segment_(\d+)_full", name)
        if full_match:
            return int(full_match.group(1)), None, False

        rmax_match = re.fullmatch(
            r"segment_(\d+)_rmax_([0-9.eE+-]+)au(_final)?",
            name,
        )
        if rmax_match:
            return (
                int(rmax_match.group(1)),
                float(rmax_match.group(2)),
                bool(rmax_match.group(3)),
            )

        raise ValueError(f"Not a segmented-run directory: {segment_dir}")

    def _read_segment_cache_int(self, segment_dir: Path, subdir: str, key: str) -> int | None:
        """Read an integer value from a segment cache-context file.

        Parameters
        ----------
        segment_dir : pathlib.Path
            Segment working directory.
        subdir : str
            RADMC-3D output subdirectory name.
        key : str
            Cache-context key to read.

        Returns
        -------
        int or None
            Cached integer value if present.
        """
        path = segment_dir / "radmc3d_outputs" / subdir / "cache_context.json"
        if not path.exists():
            return None
        with path.open("r") as f:
            cache = json.load(f)
        value = cache.get(key)
        if value is None:
            return None
        return int(value)

    def load_segmented_rt_outputs(
        self,
        *,
        diagnostic_plots: bool = False,
        plots_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Load existing segmented RT outputs without running RADMC-3D.

        Parameters
        ----------
        diagnostic_plots : bool, optional
            Whether to write diagnostic plots after merged fields are built.
        plots_dir : str or pathlib.Path, optional
            Diagnostic plot directory. Defaults to
            ``base_model_dir / "plots" / "segmented_rt"``.

        Returns
        -------
        dict
            Segmented-run metadata and merged ``temperature``/``chi`` fields.

        Raises
        ------
        FileNotFoundError
            If required saved segment outputs are missing.
        """
        from diskbridge.radmc3d.model import RadModel

        segments_root = self.base_model_dir / "segments"
        if not segments_root.exists():
            raise FileNotFoundError(f"No segmented RT directory found: {segments_root}")

        parsed_segments: list[tuple[int, float | None, bool, Path]] = []
        summary_path = self.base_model_dir / "plots" / "segmented_rt" / "diagnostic_summary.json"
        if summary_path.exists():
            try:
                with summary_path.open("r") as f:
                    summary = json.load(f)
                manifest_segments = summary.get("segmented_rt", {}).get("segments", [])
            except Exception as exc:
                logger.warning(
                    "Could not read segmented RT manifest %s; falling back to directory scan: %s",
                    summary_path,
                    exc,
                )
                manifest_segments = []
            for entry in manifest_segments:
                work_dir = entry.get("work_dir")
                if not work_dir:
                    continue
                segment_dir = Path(work_dir)
                if not segment_dir.is_absolute():
                    segment_dir = self.base_model_dir / segment_dir
                if not segment_dir.exists():
                    logger.warning("Skipping missing segmented RT manifest entry: %s", segment_dir)
                    continue
                try:
                    level, rmax_au, parsed_is_final = self._parse_segment_dir(segment_dir)
                except ValueError:
                    logger.warning("Skipping invalid segmented RT manifest entry: %s", segment_dir)
                    continue
                parsed_segments.append(
                    (
                        level,
                        rmax_au,
                        bool(entry.get("is_final", parsed_is_final)),
                        segment_dir,
                    )
                )

        if not parsed_segments:
            for segment_dir in segments_root.iterdir():
                if not segment_dir.is_dir():
                    continue
                try:
                    level, rmax_au, is_final = self._parse_segment_dir(segment_dir)
                except ValueError:
                    continue
                parsed_segments.append((level, rmax_au, is_final, segment_dir))

        parsed_segments.sort(key=lambda item: (item[0], item[2]))
        if not parsed_segments:
            raise FileNotFoundError(f"No segmented RT outputs found in {segments_root}")

        mesh = self.base_model.mesh
        if mesh is None:
            raise ValueError("Base model has no mesh")
        axis_order = mesh.axis_names()

        merged_T: Optional[Quantity] = None
        merged_chi: Optional[Quantity] = None
        merged_uv_products: Optional[dict[str, Quantity]] = None
        uv_product_measured_mask: Optional[np.ndarray] = None
        segment_id: Optional[np.ndarray] = None
        cfg = get_config()
        runtime_mode = validate_uv_chemistry_config(cfg, segmented=True)
        uv_products_enabled = runtime_mode.products_enabled
        uv_cfg = cfg.get("radmc3d", {}).get("uv_products", {})
        active_specs = uv_product_specs_from_config(uv_cfg)
        broad_spec = next(spec for spec in active_specs if spec.field_name == "chi_broad")
        disc_uv_min = Quantity(float(broad_spec.band.lam_min_nm), "nm")
        disc_uv_max = Quantity(float(broad_spec.band.lam_max_nm), "nm")
        segments: list[dict[str, Any]] = []
        split_radii_au: list[float] = []

        for level, rmax_au, is_final, segment_dir in parsed_segments:
            temp_path = segment_dir / "radmc3d_outputs" / "temperature" / "dust_temperature.bdat"
            mean_path = segment_dir / "radmc3d_outputs" / "mcmono" / "mean_intensity.bout"
            if not temp_path.exists():
                raise FileNotFoundError(f"Missing saved segment dust temperature: {temp_path}")
            if not mean_path.exists():
                raise FileNotFoundError(f"Missing saved segment mean intensity: {mean_path}")

            if level == 0 and rmax_au is None:
                seg_model = self.base_model
                seg_indexer = None
            else:
                if rmax_au is None:
                    raise ValueError(f"Segment {segment_dir} is missing an outer radius")
                segment = SegmentDefinition(
                    name=f"segment_{level:02d}",
                    bounds={"r": (None, rmax_au * units("au"))},
                    work_dir=segment_dir,
                )
                seg_model, seg_indexer = self._build_segment_model_and_indexer(segment)
                split_radii_au.append(float(rmax_au))

            rad = RadModel(seg_model, model_dir=segment_dir)
            rad.read_dust_temperature(fname=temp_path)
            rad._postprocess_chi(
                mean_path,
                disc_uv_min if is_final else rad.params.uv_min,
                disc_uv_max if is_final else rad.params.uv_max,
                compute_products=bool(uv_products_enabled and is_final),
            )

            merged_T, merged_chi = self._merge_segment_fields(
                merged_T=merged_T,
                merged_chi=merged_chi,
                rad=rad,
                indexer=seg_indexer,
                axis_order=axis_order,
            )
            if uv_products_enabled:
                merged_uv_products, uv_product_measured_mask, segment_id = (
                    self._merge_segment_uv_products(
                        merged_uv_products=merged_uv_products,
                        uv_product_measured_mask=uv_product_measured_mask,
                        segment_id=segment_id,
                        rad=rad,
                        indexer=seg_indexer,
                        axis_order=axis_order,
                        segment_level=level,
                        measured=bool(is_final),
                    )
                )

            segments.append(
                {
                    "level": int(level),
                    "work_dir": str(segment_dir),
                    "r_max_au": None if rmax_au is None else float(rmax_au),
                    "nphot_thermal": self._read_segment_cache_int(segment_dir, "temperature", "nphot"),
                    "nphot_mono": self._read_segment_cache_int(segment_dir, "mcmono", "nphot"),
                    "has_temperature": True,
                    "has_mcmono": True,
                    "is_final": bool(is_final),
                    "loaded_existing": True,
                }
            )

        if merged_T is None or merged_chi is None:
            raise ValueError("Segmented RT saved outputs produced no merged fields")

        self.base_model.gas_register(
            "dust_temperature",
            Field(quantity="dust_temperature", data=merged_T, axis_order=axis_order),
        )
        self.base_model.gas_register(
            "chi",
            Field(quantity="chi", data=merged_chi, axis_order=axis_order),
        )
        if merged_uv_products is not None:
            for name, product in merged_uv_products.items():
                self.base_model.gas_register(
                    name,
                    Field(quantity=name, data=product, axis_order=axis_order),
                )
        if uv_product_measured_mask is not None:
            self.base_model.gas_register(
                "uv_product_measured_mask",
                Field(
                    quantity="uv_product_measured_mask",
                    data=Quantity(uv_product_measured_mask, "dimensionless"),
                    axis_order=axis_order,
                ),
            )
        if segment_id is not None:
            self.base_model.gas_register(
                "segment_id",
                Field(
                    quantity="segment_id",
                    data=Quantity(segment_id, "dimensionless"),
                    axis_order=axis_order,
                ),
            )
        self.base_model.validate_canonical_axis_orders(include_dust=False)

        base_rad = RadModel(self.base_model, model_dir=self.base_model_dir)
        base_rad.dust_temperature = merged_T
        base_rad.chi = merged_chi
        if merged_uv_products is not None:
            base_rad.uv_products = merged_uv_products
            base_rad.radiation_mode = "local_uv_products"
        else:
            base_rad.radiation_mode = "local_chi"

        out = {
            "loaded_existing": True,
            "split_radii_au": split_radii_au,
            "segments": segments,
            "scout_runs": [],
            "temperature": merged_T,
            "chi": merged_chi,
            "uv_products": merged_uv_products,
            "uv_product_mode": "disc_segment_only",
            "outer_product_policy": runtime_mode.outer_product_policy,
            "uv_product_measured_mask": uv_product_measured_mask,
            "segment_id": segment_id,
        }
        if diagnostic_plots:
            from diskbridge.visualization.diagnostics import make_segmented_rt_diagnostic_plots

            plot_output_dir = (
                Path(plots_dir)
                if plots_dir is not None
                else self.base_model_dir / "plots" / "segmented_rt"
            )
            made = make_segmented_rt_diagnostic_plots(base_rad, out, plot_output_dir)
            out["diagnostic_plots"] = [str(path) for path in made]
        return out

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

        base_params = self.base_model_dir / 'params.txt'
        if base_params.exists():
            seg_params = work_dir / 'params.txt'
            if seg_params.exists():
                seg_params.unlink()
            shutil.copy2(str(base_params), str(seg_params))

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
        *,
        compute_uv_products: bool = False,
        uv_min: Optional[Quantity] = None,
        uv_max: Optional[Quantity] = None,
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
            uv_min=uv_min,
            uv_max=uv_max,
            compute_uv_products=compute_uv_products,
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

    def _merge_segment_fields(
        self,
        *,
        merged_T: Optional[Quantity],
        merged_chi: Optional[Quantity],
        rad: 'RadModel',
        indexer: Optional[ClipIndexer],
        axis_order: Tuple[str, ...],
    ) -> tuple[Quantity, Quantity]:
        if merged_T is None or merged_chi is None:
            return rad.dust_temperature, rad.chi

        if indexer is None:
            return rad.dust_temperature, rad.chi

        return (
            self._merge_field(
                merged=merged_T,
                child=rad.dust_temperature,
                indexer=indexer,
                axis_order=axis_order,
            ),
            self._merge_field(
                merged=merged_chi,
                child=rad.chi,
                indexer=indexer,
                axis_order=axis_order,
            ),
        )

    def _segment_uv_product_fields(
        self,
        rad: 'RadModel',
        *,
        measured: bool,
    ) -> dict[str, Quantity]:
        """Return UV products for measured or Draine-equivalent segments.

        Parameters
        ----------
        rad : RadModel
            Segment RADMC-3D wrapper.
        measured : bool
            Whether process-specific products were measured in this segment.

        Returns
        -------
        dict
            UV product quantities on the segment grid.
        """
        chi_broad = rad.ensure_uv_product("chi_broad", fallback_to_chi=False)
        if measured:
            missing = [
                name
                for name in UV_PRODUCT_MERGED_FIELD_NAMES
                if not rad.has_uv_product(name)
            ]
            if missing:
                raise RuntimeError(
                    "Measured UV-product segment is missing required fields: "
                    + ", ".join(missing)
                )
            return {
                name: rad.ensure_uv_product(name, fallback_to_chi=False)
                for name in UV_PRODUCT_MERGED_FIELD_NAMES
            }

        products: dict[str, Quantity] = {"chi_broad": chi_broad}
        uv_cfg = get_config().get("radmc3d", {}).get("uv_products", {})
        specs = uv_product_specs_from_config(uv_cfg)
        f_pdes_ref = (
            draine_reference_for_product("F_CO_pdes_photon", specs)
            .to("1/(cm^2 s)")
            .magnitude
        )
        f_pdes_band_ref = (
            draine_references_for_product_partitions("F_CO_pdes_photon", specs)
            .to("1/(cm^2 s)")
            .magnitude
        )
        chi_broad_arr = np.asarray(chi_broad.to("dimensionless").magnitude, dtype=np.float64)
        for name in UV_PRODUCT_MERGED_FIELD_NAMES:
            if name == "chi_broad":
                continue
            if name == "F_CO_pdes_photon":
                products[name] = Quantity(
                    chi_broad_arr * float(f_pdes_ref),
                    "1/(cm^2 s)",
                )
            elif name == "F_CO_pdes_photon_bands":
                products[name] = Quantity(
                    f_pdes_band_ref.reshape((int(f_pdes_band_ref.size),) + (1,) * chi_broad_arr.ndim)
                    * chi_broad_arr[None, ...],
                    "1/(cm^2 s)",
                )
            else:
                products[name] = chi_broad
        return products

    def _merge_segment_uv_products(
        self,
        *,
        merged_uv_products: Optional[dict[str, Quantity]],
        uv_product_measured_mask: Optional[np.ndarray],
        segment_id: Optional[np.ndarray],
        rad: 'RadModel',
        indexer: Optional[ClipIndexer],
        axis_order: Tuple[str, ...],
        segment_level: int,
        measured: bool,
    ) -> tuple[dict[str, Quantity], np.ndarray, np.ndarray]:
        """Merge segment UV products into base-grid arrays.

        Parameters
        ----------
        merged_uv_products : dict or None
            Existing merged product fields.
        uv_product_measured_mask : ndarray or None
            Existing mask of cells with measured process-specific products.
        segment_id : ndarray or None
            Existing segment identifier field.
        rad : RadModel
            Segment RADMC-3D wrapper.
        indexer : ClipIndexer or None
            Indexer for merging clipped segment data.
        axis_order : tuple of str
            Canonical axis order.
        segment_level : int
            Segment level identifier to write.
        measured : bool
            Whether process-specific products were measured in this segment.

        Returns
        -------
        tuple
            Merged products, measured mask, and segment id arrays.
        """
        products = self._segment_uv_product_fields(rad, measured=measured)

        if merged_uv_products is None:
            merged_uv_products = {
                name: Quantity(
                    np.array(q.magnitude, copy=True),
                    q.units,
                )
                for name, q in products.items()
            }
            shape = np.asarray(rad.chi.magnitude).shape
            uv_product_measured_mask = np.full(shape, bool(measured), dtype=bool)
            segment_id = np.full(shape, int(segment_level), dtype=np.int32)
            return merged_uv_products, uv_product_measured_mask, segment_id

        if uv_product_measured_mask is None or segment_id is None:
            raise ValueError("UV product merge state is incomplete")

        if indexer is None:
            merged_uv_products = {
                name: Quantity(np.array(q.magnitude, copy=True), q.units)
                for name, q in products.items()
            }
            shape = np.asarray(rad.chi.magnitude).shape
            uv_product_measured_mask = np.full(shape, bool(measured), dtype=bool)
            segment_id = np.full(shape, int(segment_level), dtype=np.int32)
            return merged_uv_products, uv_product_measured_mask, segment_id

        for name, q in products.items():
            merged_uv_products[name] = self._merge_field(
                merged=merged_uv_products[name],
                child=q,
                indexer=indexer,
                axis_order=axis_order,
            )

        mask_child = Quantity(
            np.full(np.asarray(rad.chi.magnitude).shape, bool(measured), dtype=bool),
            "dimensionless",
        )
        seg_child = Quantity(
            np.full(np.asarray(rad.chi.magnitude).shape, int(segment_level), dtype=np.int32),
            "dimensionless",
        )
        uv_product_measured_mask = np.asarray(
            self._merge_field(
                merged=Quantity(uv_product_measured_mask, "dimensionless"),
                child=mask_child,
                indexer=indexer,
                axis_order=axis_order,
            ).magnitude,
            dtype=bool,
        )
        segment_id = np.asarray(
            self._merge_field(
                merged=Quantity(segment_id, "dimensionless"),
                child=seg_child,
                indexer=indexer,
                axis_order=axis_order,
            ).magnitude,
            dtype=np.int32,
        )
        return merged_uv_products, uv_product_measured_mask, segment_id

    def _run_final_segment_rt(
        self,
        *,
        rad: 'RadModel',
        indexer: Optional[ClipIndexer],
        axis_order: Tuple[str, ...],
        merged_T: Optional[Quantity],
        merged_chi: Optional[Quantity],
        nphot_therm_final: int,
        nphot_mono_final: int,
        mcmono_wav_um: np.ndarray,
        force: bool,
        compute_uv_products: bool = False,
        uv_min: Optional[Quantity] = None,
        uv_max: Optional[Quantity] = None,
    ) -> tuple[Quantity, Quantity]:
        self._run_segment_rt(
            rad,
            nphot_therm_final,
            nphot_mono_final,
            mcmono_wav_um,
            force,
            compute_uv_products=compute_uv_products,
            uv_min=uv_min,
            uv_max=uv_max,
        )
        return self._merge_segment_fields(
            merged_T=merged_T,
            merged_chi=merged_chi,
            rad=rad,
            indexer=indexer,
            axis_order=axis_order,
        )

    def run_segmented_rt(
        self,
        nphot_therm: Optional[int] = None,
        nphot_mono: Optional[int] = None,
        mcmono_n_wavelengths: Optional[int] = None,
        mcmono_uv_n_wavelengths: Optional[int] = None,
        mcmono_wavelength_spacing: str = 'log',
        mcmono_wavelengths_um: Optional[np.ndarray] = None,
        max_splits: Optional[int] = None,
        segmented_external_source_mode: Optional[str] = None,
        segmented_final_nphot_multiplier: Optional[float] = None,
        force: bool = False,
        diagnostic_plots: bool = False,
        plots_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run segmented RADMC-3D and optionally make diagnostics.

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
        segmented_external_source_mode : str, optional
            External source mode for inner segments.
        segmented_final_nphot_multiplier : float, optional
            Multiplier for the terminal segment photon count.
        force : bool, optional
            Recompute outputs even when cached outputs are available.
        diagnostic_plots : bool, optional
            Whether to write diagnostic plots after merged fields are built.
        plots_dir : str or pathlib.Path, optional
            Diagnostic plot directory. Defaults to
            ``base_model_dir / "plots" / "segmented_rt"``.

        Returns
        -------
        dict
            Segmented-run metadata and merged ``temperature``/``chi`` fields.
        """
        from diskbridge.radmc3d.model import RadModel
        from diskbridge.radmc3d.writer import RadWriter
        
        import diskbridge
        params = diskbridge.params

        tol = params.segmented_tol
        window_fraction = params.segmented_window_fraction
        shell_ncells = params.segmented_shell_ncells
        r_clip_min_au = params.segmented_r_clip_min.to('au').magnitude
        external_source_mode = (
            getattr(params, "segmented_external_source_mode", "shell")
            if segmented_external_source_mode is None
            else segmented_external_source_mode
        )
        external_source_mode = str(external_source_mode).strip().lower()
        if external_source_mode not in {"shell", "initial"}:
            raise ValueError(
                "segmented_external_source_mode must be 'shell' or 'initial', "
                f"got {external_source_mode!r}"
            )

        wavelength_source_use = "uv" if external_source_mode == "initial" else "external"
        stop_factor = float(params.segmented_stop_factor)
        if stop_factor <= 0.0 or stop_factor >= 1.0:
            raise ValueError("segmented_stop_factor must be in (0, 1)")

        nphot_therm_nominal = int(params.nphot_thermal) if nphot_therm is None else int(nphot_therm)
        nphot_mono_nominal = int(params.nphot_mono) if nphot_mono is None else int(nphot_mono)

        final_nphot_multiplier = float(
            getattr(params, "segmented_final_nphot_multiplier", 1.0)
            if segmented_final_nphot_multiplier is None
            else segmented_final_nphot_multiplier
        )
        if final_nphot_multiplier < 1.0:
            raise ValueError("segmented_final_nphot_multiplier must be >= 1")

        nphot_therm_final = int(final_nphot_multiplier * nphot_therm_nominal)
        nphot_mono_final = int(final_nphot_multiplier * nphot_mono_nominal)
        if nphot_therm_final < 1:
            raise ValueError(
                f"segmented_final_nphot_multiplier={final_nphot_multiplier} yields "
                f"nphot_therm_final={nphot_therm_final}. Increase the multiplier or nphot_thermal."
            )
        if nphot_mono_final < 1:
            raise ValueError(
                f"segmented_final_nphot_multiplier={final_nphot_multiplier} yields "
                f"nphot_mono_final={nphot_mono_final}. Increase the multiplier or nphot_mono."
            )

        nphot_ratio = float(params.segmented_nphot_ratio)
        if nphot_ratio <= 0.0:
            raise ValueError("segmented_nphot_ratio must be > 0")

        nphot_therm_intermediate = int(nphot_ratio * nphot_therm_nominal)
        nphot_mono_intermediate = int(nphot_ratio * nphot_mono_nominal)
        if nphot_therm_intermediate < 1:
            raise ValueError(
                f"segmented_nphot_ratio={nphot_ratio} yields nphot_therm_intermediate={nphot_therm_intermediate}. "
                "Increase segmented_nphot_ratio or nphot_thermal."
            )
        if nphot_mono_intermediate < 1:
            raise ValueError(
                f"segmented_nphot_ratio={nphot_ratio} yields nphot_mono_intermediate={nphot_mono_intermediate}. "
                "Increase segmented_nphot_ratio or nphot_mono."
            )

        logger.info(
            "Segmented RT photons: ratio=%.6g, final_multiplier=%.6g, "
            "mctherm=%d/%d/%d, mcmono=%d/%d/%d"
            % (
                float(nphot_ratio),
                float(final_nphot_multiplier),
                int(nphot_therm_intermediate),
                int(nphot_therm_nominal),
                int(nphot_therm_final),
                int(nphot_mono_intermediate),
                int(nphot_mono_nominal),
                int(nphot_mono_final),
            )
        )

        if max_splits is None:
            max_splits = int(params.segmented_max_splits)
        max_splits = int(max_splits)
        if max_splits < 0:
            raise ValueError("max_splits must be >= 0")

        mesh = self.base_model.mesh
        if mesh is None:
            raise ValueError("Base model has no mesh")
        axis_order = mesh.axis_names()

        segments: list[dict[str, Any]] = []
        scout_runs: list[dict[str, Any]] = []
        split_radii_au: list[float] = []
        r_edges_base_au = self.base_model.mesh.edges('r').to('au').magnitude  # type: ignore[union-attr]
        current_outer_rmax_au = float(np.max(r_edges_base_au))
        outer_rad: Optional[RadModel] = None
        merged_T: Optional[Quantity] = None
        merged_chi: Optional[Quantity] = None
        merged_uv_products: Optional[dict[str, Quantity]] = None
        uv_product_measured_mask: Optional[np.ndarray] = None
        segment_id: Optional[np.ndarray] = None
        mcmono_wav_um_use: Optional[np.ndarray] = None
        mcmono_wav_um_disc: Optional[np.ndarray] = None
        cfg = get_config()
        runtime_mode = validate_uv_chemistry_config(cfg, segmented=True)
        uv_products_enabled = runtime_mode.products_enabled
        uv_cfg = cfg.get("radmc3d", {}).get("uv_products", {})
        active_specs = uv_product_specs_from_config(uv_cfg)
        active_edges_nm = uv_product_edges_from_specs(active_specs)
        broad_spec = next(spec for spec in active_specs if spec.field_name == "chi_broad")
        disc_uv_min = Quantity(float(broad_spec.band.lam_min_nm), "nm")
        disc_uv_max = Quantity(float(broad_spec.band.lam_max_nm), "nm")
        product_threshold_r_au = (
            self._stellar_product_threshold_radius_au(
                active_specs,
                runtime_mode.stellar_fraction_threshold,
            )
            if uv_products_enabled
            else None
        )
        if product_threshold_r_au is not None:
            logger.info(
                "UV product stellar-fraction threshold radius: %.6g au "
                "(threshold=%.3g)",
                float(product_threshold_r_au),
                float(runtime_mode.stellar_fraction_threshold),
            )

        base_opacity_dir = self.base_model_dir / 'radmc3d_inputs'
        base_opacity_dir.mkdir(parents=True, exist_ok=True)

        if not list(base_opacity_dir.glob('dustkappa_*.inp')):
            RadWriter(self.base_model, organize_files=True).compute_and_write_dust_opacities(
                self.base_model_dir
            )
            if not list(base_opacity_dir.glob('dustkappa_*.inp')):
                raise RuntimeError(
                    f"No dustkappa_*.inp files found in {base_opacity_dir} after computing dust opacities"
                )

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

            rad = self._setup_segment(seg_model, segment.work_dir, base_opacity_dir)

            if mcmono_wav_um_use is None:
                n_uv = params.uv_n_wavelengths if mcmono_uv_n_wavelengths is None else int(mcmono_uv_n_wavelengths)
                n_wavelengths_use = mcmono_n_wavelengths
                if wavelength_source_use == "uv" and n_wavelengths_use is None:
                    n_wavelengths_use = n_uv
                n_uv_enforce_use = 0 if wavelength_source_use == "uv" else n_uv
                mcmono_wav_um_use = build_mcmono_wavelengths(
                    wavelength_source=wavelength_source_use,
                    wavelength_file=rad.inputs_dir / 'wavelength_micron.inp',
                    uv_min_um=params.uv_min.to('micron').magnitude,
                    uv_max_um=params.uv_max.to('micron').magnitude,
                    n_wavelengths=n_wavelengths_use,
                    n_uv_enforce=n_uv_enforce_use,
                    spacing=mcmono_wavelength_spacing,
                    provided_wavelengths=mcmono_wavelengths_um,
                )
                mcmono_wav_um_disc = build_mcmono_wavelengths(
                    wavelength_source=wavelength_source_use,
                    wavelength_file=rad.inputs_dir / 'wavelength_micron.inp',
                    uv_min_um=disc_uv_min.to('micron').magnitude,
                    uv_max_um=disc_uv_max.to('micron').magnitude,
                    n_wavelengths=n_wavelengths_use,
                    n_uv_enforce=n_uv_enforce_use,
                    spacing=mcmono_wavelength_spacing,
                    provided_wavelengths=mcmono_wavelengths_um,
                    extra_enforced_wavelengths_um=active_edges_nm * 1.0e-3,
                )

            if external_source_mode == "shell" and level > 0 and outer_rad is not None:
                self._inherit_external_source(outer_rad, rad, current_outer_rmax_au, shell_ncells)

            self._run_segment_rt(rad, nphot_therm_intermediate, nphot_mono_intermediate, mcmono_wav_um_use, force)

            merged_T, merged_chi = self._merge_segment_fields(
                merged_T=merged_T,
                merged_chi=merged_chi,
                rad=rad,
                indexer=seg_indexer,
                axis_order=axis_order,
            )
            if uv_products_enabled:
                merged_uv_products, uv_product_measured_mask, segment_id = (
                    self._merge_segment_uv_products(
                        merged_uv_products=merged_uv_products,
                        uv_product_measured_mask=uv_product_measured_mask,
                        segment_id=segment_id,
                        rad=rad,
                        indexer=seg_indexer,
                        axis_order=axis_order,
                        segment_level=level,
                        measured=False,
                    )
                )

            segments.append(
                {
                    'level': int(level),
                    'work_dir': str(segment.work_dir),
                    'r_max_au': float(current_outer_rmax_au),
                    'nphot_thermal': int(nphot_therm_intermediate),
                    'nphot_mono': int(nphot_mono_intermediate),
                    'has_temperature': True,
                    'has_mcmono': True,
                    'is_final': False,
                }
            )

            force_product_final = (
                product_threshold_r_au is not None
                and float(current_outer_rmax_au) <= float(product_threshold_r_au)
            )
            if force_product_final:
                logger.info(
                    "UV product threshold reached at level=%d "
                    "(segment r_max=%.6g au <= threshold radius %.6g au); "
                    "rerunning as product-measured final segment",
                    int(level),
                    float(current_outer_rmax_au),
                    float(product_threshold_r_au),
                )

            if level >= max_splits or force_product_final:
                if (
                    nphot_therm_intermediate != nphot_therm_final
                    or nphot_mono_intermediate != nphot_mono_final
                    or uv_products_enabled
                ):
                    if (
                        nphot_therm_intermediate != nphot_therm_final
                        or nphot_mono_intermediate != nphot_mono_final
                    ):
                        scout_runs.append(dict(segments[-1]))
                    merged_T, merged_chi = self._run_final_segment_rt(
                        rad=rad,
                        indexer=seg_indexer,
                        axis_order=axis_order,
                        merged_T=merged_T,
                        merged_chi=merged_chi,
                        nphot_therm_final=nphot_therm_final,
                        nphot_mono_final=nphot_mono_final,
                        mcmono_wav_um=mcmono_wav_um_disc,
                        force=force,
                        compute_uv_products=uv_products_enabled,
                        uv_min=disc_uv_min,
                        uv_max=disc_uv_max,
                    )
                    if uv_products_enabled:
                        merged_uv_products, uv_product_measured_mask, segment_id = (
                            self._merge_segment_uv_products(
                                merged_uv_products=merged_uv_products,
                                uv_product_measured_mask=uv_product_measured_mask,
                                segment_id=segment_id,
                                rad=rad,
                                indexer=seg_indexer,
                                axis_order=axis_order,
                                segment_level=level,
                                measured=True,
                            )
                        )
                segments[-1]['nphot_thermal'] = int(nphot_therm_final)
                segments[-1]['nphot_mono'] = int(nphot_mono_final)
                segments[-1]['is_final'] = True
                outer_rad = rad
                break

            if merged_T is None or merged_chi is None:
                raise ValueError("Internal error: missing merged fields")

            rad.model.gas_register(
                'temperature',
                Field(quantity='temperature', data=rad.dust_temperature, axis_order=rad.model.mesh.axis_names()),
            )
            rad.model.gas_register(
                'chi',
                Field(quantity='chi', data=rad.chi, axis_order=rad.model.mesh.axis_names()),
            )
            rad.model.validate_canonical_axis_orders(include_dust=False)

            r_au, chi_profile = compute_volume_weighted_mean_radial_profile(rad.model, 'chi')
            _, T_profile = compute_volume_weighted_mean_radial_profile(rad.model, 'temperature')
            r_edges_au = rad.model.mesh.edges('r').to('au').magnitude

            try:
                r_split_au, r_split_info = find_r_split(
                    r_au=r_au,
                    r_edges_au=r_edges_au,
                    chi_profile=chi_profile,
                    T_profile=T_profile,
                    tol_chi=tol,
                    tol_T=tol,
                    window_fraction=window_fraction,
                    r_clip_min_au=float(r_clip_min_au),
                )
                if product_threshold_r_au is not None and r_split_au > product_threshold_r_au:
                    logger.info(
                        "UV product threshold caps r_split from %.6g au to %.6g au",
                        float(r_split_au),
                        float(product_threshold_r_au),
                    )
                    r_split_au = max(float(product_threshold_r_au), float(r_clip_min_au))
                    r_split_info = dict(r_split_info)
                    r_split_info["uv_product_threshold_r_au"] = float(product_threshold_r_au)
                logger.info(
                    "find_r_split: level=%d current_outer_rmax_au=%.6g r_split_au=%.6g "
                    "r_split_cell_idx=%s window_r_min=%.6g chi_asymptote=%.6g T_asymptote=%.6g "
                    "chi_dev_at_split=%.6g T_dev_at_split=%.6g"
                    % (
                        int(level),
                        float(current_outer_rmax_au),
                        float(r_split_au),
                        str(r_split_info.get('r_split_cell_idx')),
                        float(r_split_info.get('window_r_min', np.nan)),
                        float(r_split_info.get('chi_asymptote', np.nan)),
                        float(r_split_info.get('T_asymptote', np.nan)),
                        float(r_split_info.get('chi_dev_at_split', np.nan)),
                        float(r_split_info.get('T_dev_at_split', np.nan)),
                    )
                )
            except ValueError:
                if (
                    nphot_therm_intermediate != nphot_therm_final
                    or nphot_mono_intermediate != nphot_mono_final
                    or uv_products_enabled
                ):
                    if (
                        nphot_therm_intermediate != nphot_therm_final
                        or nphot_mono_intermediate != nphot_mono_final
                    ):
                        scout_runs.append(dict(segments[-1]))
                    merged_T, merged_chi = self._run_final_segment_rt(
                        rad=rad,
                        indexer=seg_indexer,
                        axis_order=axis_order,
                        merged_T=merged_T,
                        merged_chi=merged_chi,
                        nphot_therm_final=nphot_therm_final,
                        nphot_mono_final=nphot_mono_final,
                        mcmono_wav_um=mcmono_wav_um_disc,
                        force=force,
                        compute_uv_products=uv_products_enabled,
                        uv_min=disc_uv_min,
                        uv_max=disc_uv_max,
                    )
                    if uv_products_enabled:
                        merged_uv_products, uv_product_measured_mask, segment_id = (
                            self._merge_segment_uv_products(
                                merged_uv_products=merged_uv_products,
                                uv_product_measured_mask=uv_product_measured_mask,
                                segment_id=segment_id,
                                rad=rad,
                                indexer=seg_indexer,
                                axis_order=axis_order,
                                segment_level=level,
                                measured=True,
                            )
                        )
                segments[-1]['nphot_thermal'] = int(nphot_therm_final)
                segments[-1]['nphot_mono'] = int(nphot_mono_final)
                segments[-1]['is_final'] = True
                outer_rad = rad
                break

            if float(r_split_au) >= float(current_outer_rmax_au):
                if (
                    nphot_therm_intermediate != nphot_therm_final
                    or nphot_mono_intermediate != nphot_mono_final
                    or uv_products_enabled
                ):
                    if (
                        nphot_therm_intermediate != nphot_therm_final
                        or nphot_mono_intermediate != nphot_mono_final
                    ):
                        scout_runs.append(dict(segments[-1]))
                    merged_T, merged_chi = self._run_final_segment_rt(
                        rad=rad,
                        indexer=seg_indexer,
                        axis_order=axis_order,
                        merged_T=merged_T,
                        merged_chi=merged_chi,
                        nphot_therm_final=nphot_therm_final,
                        nphot_mono_final=nphot_mono_final,
                        mcmono_wav_um=mcmono_wav_um_disc,
                        force=force,
                        compute_uv_products=uv_products_enabled,
                        uv_min=disc_uv_min,
                        uv_max=disc_uv_max,
                    )
                    if uv_products_enabled:
                        merged_uv_products, uv_product_measured_mask, segment_id = (
                            self._merge_segment_uv_products(
                                merged_uv_products=merged_uv_products,
                                uv_product_measured_mask=uv_product_measured_mask,
                                segment_id=segment_id,
                                rad=rad,
                                indexer=seg_indexer,
                                axis_order=axis_order,
                                segment_level=level,
                                measured=True,
                            )
                        )
                segments[-1]['nphot_thermal'] = int(nphot_therm_final)
                segments[-1]['nphot_mono'] = int(nphot_mono_final)
                segments[-1]['is_final'] = True
                outer_rad = rad
                break

            if float(r_split_au) >= stop_factor * float(current_outer_rmax_au):
                if (
                    nphot_therm_intermediate == nphot_therm_final
                    and nphot_mono_intermediate == nphot_mono_final
                    and not uv_products_enabled
                ):
                    segments[-1]['is_final'] = True
                    outer_rad = rad
                    break

                if (
                    nphot_therm_intermediate != nphot_therm_final
                    or nphot_mono_intermediate != nphot_mono_final
                ):
                    scout_runs.append(dict(segments.pop()))
                else:
                    segments.pop()
                final_work_dir = self.base_model_dir / 'segments' / f'segment_{level + 1:02d}_rmax_{current_outer_rmax_au:.6g}au_final'
                final_seg_bounds = {'r': (None, current_outer_rmax_au * units('au'))}
                final_segment = SegmentDefinition(
                    name=f'segment_{level + 1:02d}_final',
                    bounds=final_seg_bounds,
                    work_dir=final_work_dir,
                )
                final_model, final_indexer = self._build_segment_model_and_indexer(final_segment)
                
                final_rad = self._setup_segment(final_model, final_work_dir, base_opacity_dir)
                
                if external_source_mode == "shell" and outer_rad is not None:
                    self._inherit_external_source(outer_rad, final_rad, current_outer_rmax_au, shell_ncells)
                
                self._run_segment_rt(
                    final_rad,
                    nphot_therm_final,
                    nphot_mono_final,
                    mcmono_wav_um_disc,
                    force,
                    compute_uv_products=uv_products_enabled,
                    uv_min=disc_uv_min,
                    uv_max=disc_uv_max,
                )

                merged_T = self._merge_field(
                    merged=merged_T,
                    child=final_rad.dust_temperature,
                    indexer=final_indexer,
                    axis_order=axis_order,
                )
                merged_chi = self._merge_field(
                    merged=merged_chi,
                    child=final_rad.chi,
                    indexer=final_indexer,
                    axis_order=axis_order,
                )
                if uv_products_enabled:
                    merged_uv_products, uv_product_measured_mask, segment_id = (
                        self._merge_segment_uv_products(
                            merged_uv_products=merged_uv_products,
                            uv_product_measured_mask=uv_product_measured_mask,
                            segment_id=segment_id,
                            rad=final_rad,
                            indexer=final_indexer,
                            axis_order=axis_order,
                            segment_level=level + 1,
                            measured=True,
                        )
                    )
                segments.append(
                    {
                        'level': int(level + 1),
                        'work_dir': str(final_segment.work_dir),
                        'r_max_au': float(current_outer_rmax_au),
                        'nphot_thermal': int(nphot_therm_final),
                        'nphot_mono': int(nphot_mono_final),
                        'has_temperature': True,
                        'has_mcmono': True,
                        'is_final': True,
                    }
                )

                outer_rad = final_rad
                break

            split_radii_au.append(float(r_split_au))

            outer_rad = rad
            current_outer_rmax_au = float(r_split_au)

        if merged_T is None or merged_chi is None:
            raise ValueError("Segmented RT produced no results")

        # Finalize: attach merged fields to the *base* model in canonical names.
        # Downstream chemistry and imaging should not need any manual plumbing.
        self.base_model.gas_register(
            'dust_temperature',
            Field(quantity='dust_temperature', data=merged_T, axis_order=axis_order),
        )
        self.base_model.gas_register(
            'chi',
            Field(quantity='chi', data=merged_chi, axis_order=axis_order),
        )

        if uv_products_enabled:
            if merged_uv_products is None:
                raise ValueError("Segmented RT produced no merged UV products")
            if uv_product_measured_mask is None or segment_id is None:
                raise ValueError("Segmented RT produced incomplete UV product metadata")

            for name, product in merged_uv_products.items():
                self.base_model.gas_register(
                    name,
                    Field(quantity=name, data=product, axis_order=axis_order),
                )
            self.base_model.gas_register(
                'uv_product_measured_mask',
                Field(
                    quantity='uv_product_measured_mask',
                    data=Quantity(uv_product_measured_mask, "dimensionless"),
                    axis_order=axis_order,
                ),
            )
            self.base_model.gas_register(
                'segment_id',
                Field(
                    quantity='segment_id',
                    data=Quantity(segment_id, "dimensionless"),
                    axis_order=axis_order,
                ),
            )

        self.base_model.validate_canonical_axis_orders(include_dust=False)

        # Write merged dust temperature into the base model_dir outputs so RadImage
        # (and any external RADMC-3D calls) can find dust_temperature.* without
        # workflow-level custom file writing.
        base_rad = RadModel(self.base_model, model_dir=self.base_model_dir)
        base_rad.dust_temperature = merged_T
        if uv_products_enabled and merged_uv_products is not None:
            base_rad.uv_products = merged_uv_products
            base_rad.chi = merged_uv_products["chi_broad"]
            base_rad.radiation_mode = "local_uv_products"
        else:
            base_rad.uv_products = {"chi_broad": merged_chi}
            base_rad.chi = merged_chi
            base_rad.radiation_mode = "local_chi"
        base_rad.outputs_dir.mkdir(parents=True, exist_ok=True)

        if self.base_model.dust is None:
            raise ValueError("Segmented RT requires dust model to write dust_temperature")
        nspec = int(self.base_model.dust.nbin)
        if nspec <= 0:
            raise ValueError(f"Invalid dust nbin={nspec}")

        dustopac_path = base_rad.inputs_dir / 'dustopac.inp'
        if dustopac_path.exists():
            with open(dustopac_path, 'r') as f:
                _fmt = f.readline().strip()
                nbin_line = f.readline().strip()
            try:
                nbin_file = int(nbin_line)
            except Exception as e:
                raise ValueError(f"Could not parse dustopac.inp nbin from {dustopac_path}: {e}")
            if nbin_file != nspec:
                raise ValueError(
                    "dust species mismatch: model.dust.nbin=%d but dustopac.inp declares %d" % (int(nspec), int(nbin_file))
                )

        base_rad.writer.write_dust_temperature(
            merged_T,
            output_dir=base_rad.outputs_dir,
            nspec=nspec,
        )

        logger.info(
            f"Segmented RT complete. splits={len(split_radii_au)} max_splits={max_splits}"
        )

        out = {
            'mode': external_source_mode,
            'segmented_external_source_mode': external_source_mode,
            'mcmono_wavelength_source': wavelength_source_use,
            'segmented_final_nphot_multiplier': float(final_nphot_multiplier),
            'nphot_thermal_nominal': int(nphot_therm_nominal),
            'nphot_mono_nominal': int(nphot_mono_nominal),
            'nphot_thermal_final': int(nphot_therm_final),
            'nphot_mono_final': int(nphot_mono_final),
            'split_radii_au': split_radii_au,
            'segments': segments,
            'scout_runs': scout_runs,
            'temperature': merged_T,
            'chi': base_rad.chi,
            'uv_products': merged_uv_products,
            'uv_product_mode': 'disc_segment_only',
            'outer_product_policy': runtime_mode.outer_product_policy,
            'uv_product_measured_mask': uv_product_measured_mask,
            'segment_id': segment_id,
        }
        if diagnostic_plots:
            from diskbridge.visualization.diagnostics import make_segmented_rt_diagnostic_plots

            plot_output_dir = (
                Path(plots_dir)
                if plots_dir is not None
                else self.base_model_dir / "plots" / "segmented_rt"
            )
            made = make_segmented_rt_diagnostic_plots(base_rad, out, plot_output_dir)
            out["diagnostic_plots"] = [str(path) for path in made]
        return out
