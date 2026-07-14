# db-keywords: shielding, uv-products, config, units, radmc3d, model, field, coordinates
# db-role: entrypoint
# db-scope: package
# db-purpose: Package module for shielding, uv-products, config, units.

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
from diskbridge.serialization import jsonable
from diskbridge.utils import sha256_file

from diskbridge.model.profiles import (
    compute_cell_volumes,
    compute_volume_weighted_mean_radial_profile,
    find_noise_aware_split,
    paired_radial_uncertainty_metrics,
    weighted_quantile,
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
from .data import combine_mean_intensity_files, radial_shell_mean_intensity


@dataclass(frozen=True)
class SegmentDefinition:
    name: str
    bounds: Dict[str, Tuple[Optional[Quantity], Optional[Quantity]]]
    work_dir: Path


@dataclass
class PairedScoutResult:
    """In-memory paired-estimator products needed by split and join checks."""

    metadata: dict[str, Any]
    products: dict[str, dict[str, np.ndarray]]
    wavelength_metrics: dict[str, np.ndarray]
    reliable_shells: np.ndarray
    estimator_dirs: tuple[Path, Path]


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

    def _parse_segment_dir(self, segment_dir: Path) -> tuple[int, float | None]:
        """Parse a segmented-run directory name.

        Parameters
        ----------
        segment_dir : pathlib.Path
            Segment working directory.

        Returns
        -------
        tuple
            Segment level and outer radius in au if present.

        Raises
        ------
        ValueError
            If the directory name is not a segmented-run directory.
        """
        name = segment_dir.name
        full_match = re.fullmatch(r"segment_(\d+)_full", name)
        if full_match:
            return int(full_match.group(1)), None

        rmax_match = re.fullmatch(
            r"segment_(\d+)_rmax_([0-9.eE+-]+)au",
            name,
        )
        if rmax_match:
            return int(rmax_match.group(1)), float(rmax_match.group(2))

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

    def _resolve_segment_work_dir(self, segment_entry: dict[str, Any]) -> Path:
        """Return an absolute segment working directory from metadata."""
        work_dir = segment_entry.get("work_dir")
        if not work_dir:
            raise ValueError(f"Segment metadata is missing work_dir: {segment_entry}")
        segment_dir = Path(str(work_dir))
        if not segment_dir.is_absolute():
            segment_dir = self.base_model_dir / segment_dir
        return segment_dir

    def _segment_uv_band_for_diagnostics(self) -> tuple[Quantity, Quantity]:
        """Return the broad UV-product band used by final segment diagnostics."""
        cfg = get_config()
        uv_cfg = cfg.get("radmc3d", {}).get("uv_products", {})
        active_specs = uv_product_specs_from_config(uv_cfg)
        broad_spec = next(spec for spec in active_specs if spec.field_name == "chi_broad")
        return (
            Quantity(float(broad_spec.band.lam_min_nm), "nm"),
            Quantity(float(broad_spec.band.lam_max_nm), "nm"),
        )

    def _load_segment_rad_for_diagnostics(
        self,
        segment_entry: dict[str, Any],
        *,
        disc_uv_min: Quantity,
        disc_uv_max: Quantity,
    ) -> tuple["RadModel", dict[str, Any]]:
        """Load one saved segment into an isolated ``RadModel`` for plotting.

        Parameters
        ----------
        segment_entry : dict
            Segment metadata emitted by the segmented RT runner.
        disc_uv_min, disc_uv_max : Quantity
            UV band used for final/product segment diagnostics.

        Returns
        -------
        tuple
            Segment ``RadModel`` and JSON-ready metadata.
        """
        from diskbridge.radmc3d.model import RadModel

        segment_dir = self._resolve_segment_work_dir(segment_entry)
        parsed_level, parsed_rmax_au = self._parse_segment_dir(segment_dir)
        level = int(segment_entry.get("level", parsed_level))
        is_final = bool(segment_entry.get("is_final", False))
        rmax_au = segment_entry.get("r_max_au", parsed_rmax_au)
        if rmax_au is None:
            rmax_au = parsed_rmax_au

        if level == 0 and parsed_rmax_au is None:
            seg_model = self.base_model.clip_mesh()
        else:
            if rmax_au is None:
                raise ValueError(f"Segment {segment_dir} is missing an outer radius")
            segment = SegmentDefinition(
                name=f"segment_{level:02d}",
                bounds={"r": (None, float(rmax_au) * units("au"))},
                work_dir=segment_dir,
            )
            seg_model, _ = self._build_segment_model_and_indexer(segment)

        rad = RadModel(seg_model, model_dir=segment_dir)
        temp_path = segment_dir / "radmc3d_outputs" / "temperature" / "dust_temperature.bdat"
        mean_path = segment_dir / "radmc3d_outputs" / "mcmono" / "mean_intensity.bout"
        if not temp_path.exists():
            raise FileNotFoundError(f"Missing saved segment dust temperature: {temp_path}")
        if not mean_path.exists():
            raise FileNotFoundError(f"Missing saved segment mean intensity: {mean_path}")

        rad.read_dust_temperature(fname=temp_path)
        rad._postprocess_chi(
            mean_path,
            disc_uv_min if is_final else rad.params.uv_min,
            disc_uv_max if is_final else rad.params.uv_max,
        )
        rad.model.validate_canonical_axis_orders(include_dust=False)

        mesh = rad.model.mesh
        r_min_au: float | None = None
        r_max_au_actual: float | None = None
        if mesh is not None and "r" in mesh.axes:
            r_edges_au = mesh.edges("r").to("au").magnitude
            r_min_au = float(np.min(r_edges_au))
            r_max_au_actual = float(np.max(r_edges_au))

        metadata = {
            "segment_name": segment_dir.name,
            "work_dir": str(segment_dir),
            "level": int(level),
            "is_final": bool(is_final),
            "r_min_au": r_min_au,
            "r_max_au": r_max_au_actual,
            "mesh_shape": list(mesh.shape) if mesh is not None else None,
            "nphot_thermal": segment_entry.get(
                "nphot_thermal",
                self._read_segment_cache_int(segment_dir, "temperature", "nphot"),
            ),
            "nphot_mono": segment_entry.get(
                "nphot_mono",
                self._read_segment_cache_int(segment_dir, "mcmono", "nphot"),
            ),
            "has_temperature": True,
            "has_mcmono": True,
            "temperature_file": str(temp_path),
            "mean_intensity_file": str(mean_path),
            "scout_metrics_file": str(
                segment_dir / "radmc3d_outputs" / "mcmono" / "uv_scout_metrics.npz"
            ),
            "scout": segment_entry.get("scout"),
            "split": segment_entry.get("split"),
            "join": segment_entry.get("join"),
            "stellar_fraction_threshold": segment_entry.get("stellar_fraction_threshold"),
        }
        return rad, metadata

    def _make_segmented_rt_diagnostics(
        self,
        base_rad: "RadModel",
        segmented_result: dict[str, Any],
        plot_output_dir: Path,
    ) -> list[Path]:
        """Write merged and per-segment segmented RT diagnostics."""
        from diskbridge.visualization.diagnostics import (
            make_segmented_rt_diagnostic_plots,
            make_segmented_rt_segment_diagnostic_plots,
        )

        plot_output_dir = Path(plot_output_dir)
        made = list(make_segmented_rt_diagnostic_plots(base_rad, segmented_result, plot_output_dir))
        disc_uv_min, disc_uv_max = self._segment_uv_band_for_diagnostics()

        segment_summaries: list[dict[str, Any]] = []
        for segment_entry in list(segmented_result.get("segments", [])):
            try:
                segment_dir = self._resolve_segment_work_dir(segment_entry)
                segment_rad, metadata = self._load_segment_rad_for_diagnostics(
                    segment_entry,
                    disc_uv_min=disc_uv_min,
                    disc_uv_max=disc_uv_max,
                )
                segment_plot_dir = plot_output_dir / "segments" / segment_dir.name
                summary = make_segmented_rt_segment_diagnostic_plots(
                    segment_rad,
                    metadata,
                    segment_plot_dir,
                )
                segment_summaries.append(summary)
                made.extend(Path(path) for path in summary.get("plots", []))
            except Exception as exc:
                segment_name = str(segment_entry.get("work_dir", "unknown"))
                logger.warning("Failed to make segmented RT segment diagnostics for %s: %s", segment_name, exc)
                segment_summaries.append(
                    {
                        "segment": jsonable(segment_entry),
                        "plots": [],
                        "error": str(exc),
                    }
                )

        manifest = {
            "plots_dir": str(plot_output_dir / "segments"),
            "segments": segment_summaries,
        }
        (plot_output_dir / "segments_summary.json").write_text(
            json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n"
        )
        return made

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
        summary_path = segments_root / "segmented_rt_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing segmented RT manifest: {summary_path}")

        with summary_path.open("r") as f:
            summary = json.load(f)
        summary_mode = summary.get("mode")
        if summary_mode != "noise_aware_inherited_uv":
            raise ValueError(
                "Unsupported segmented RT manifest mode "
                f"{summary_mode!r}; expected 'noise_aware_inherited_uv'"
            )
        manifest_segments = summary.get("segments")
        if not isinstance(manifest_segments, list) or not manifest_segments:
            raise ValueError(f"Segmented RT manifest has no segments: {summary_path}")

        manifest_by_level: dict[int, dict[str, Any]] = {}
        for entry in manifest_segments:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid segmented RT manifest entry: {entry!r}")
            segment_dir = self._resolve_segment_work_dir(entry)
            if not segment_dir.exists():
                raise FileNotFoundError(f"Missing segmented RT output directory: {segment_dir}")
            parsed_level, rmax_au = self._parse_segment_dir(segment_dir)
            level = int(entry.get("level", parsed_level))
            if level != parsed_level:
                raise ValueError(
                    f"Segment level mismatch for {segment_dir}: "
                    f"manifest={level}, directory={parsed_level}"
                )
            if level in manifest_by_level:
                raise ValueError(f"Duplicate segment level in {summary_path}: {level}")
            manifest_by_level[level] = entry
            parsed_segments.append(
                (level, rmax_au, bool(entry.get("is_final", False)), segment_dir)
            )

        parsed_segments.sort(key=lambda item: item[0])

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
            measured_segment = bool(uv_products_enabled)
            rad._postprocess_chi(
                mean_path,
                disc_uv_min if is_final else rad.params.uv_min,
                disc_uv_max if is_final else rad.params.uv_max,
                compute_products=measured_segment,
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
                        measured=measured_segment,
                    )
                )

            loaded_entry = dict(manifest_by_level.get(int(level), {}))
            loaded_entry.update(
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
            segments.append(loaded_entry)

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
            self._register_mesh_shaped_uv_products(merged_uv_products, axis_order)
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
            "scout_runs": summary.get("scout_runs", []),
            "joins": summary.get("joins", []),
            "temperature": merged_T,
            "chi": merged_chi,
            "uv_products": merged_uv_products,
            "mode": summary_mode,
            "uv_product_mode": "all_segments_measured",
            "outer_product_policy": None,
            "uv_product_measured_mask": uv_product_measured_mask,
            "segment_id": segment_id,
        }
        if diagnostic_plots:
            plot_output_dir = (
                Path(plots_dir)
                if plots_dir is not None
                else self.base_model_dir / "plots" / "segmented_rt"
            )
            made = self._make_segmented_rt_diagnostics(base_rad, out, plot_output_dir)
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
        uv_min: Quantity,
        uv_max: Quantity,
    ) -> None:
        """Inherit one parent source shell while retaining original non-UV flux."""
        wavelengths_um, shell_spectrum = outer_rad.extract_shell_spectrum(
            r_split_au=r_split_au,
            shell_ncells=1,
            mcmono_dir=outer_rad.outputs_dir / 'mcmono',
        )
        inner_rad.write_inherited_uv_external_source(
            wavelengths_um=wavelengths_um,
            spectrum=shell_spectrum,
            output_dir=inner_rad.inputs_dir,
            uv_min=uv_min,
            uv_max=uv_max,
        )

    @staticmethod
    def _copy_uv_products(rad: "RadModel") -> dict[str, Quantity]:
        """Copy scalar UV products before the next estimator overwrites them."""
        return {
            name: Quantity(np.array(value.magnitude, copy=True), value.units)
            for name, value in rad.uv_products.items()
            if np.asarray(value.magnitude).ndim == 3
        }

    def _run_paired_uv_scout(
        self,
        rad: "RadModel",
        *,
        nphot_total: int,
        wavelengths_um: np.ndarray,
        uv_min: Quantity,
        uv_max: Quantity,
        noise_tolerance: float,
        level: int,
        force: bool,
    ) -> PairedScoutResult:
        """Run two independent UV estimators and create the canonical mean field."""
        nphot_total = int(nphot_total)
        if nphot_total < 2:
            raise ValueError("Paired mcmono scouts require at least two photon packets")
        counts = (nphot_total // 2, nphot_total - nphot_total // 2)
        seeds = (104729 + 2 * int(level), 104730 + 2 * int(level))
        estimator_dirs = (
            rad.outputs_dir / "mcmono_estimator_1",
            rad.outputs_dir / "mcmono_estimator_2",
        )
        estimator_products: list[dict[str, Quantity]] = []
        hashes: list[str] = []
        for count, seed, output_dir in zip(counts, seeds, estimator_dirs):
            rad.compute_mcmono(
                nphot=count,
                output_dir=output_dir,
                force=force,
                wavelengths_um=wavelengths_um,
                uv_min=uv_min,
                uv_max=uv_max,
                compute_uv_products=True,
                iseed=seed,
            )
            estimator_products.append(self._copy_uv_products(rad))
            intensity_path = output_dir / "mean_intensity.bout"
            hashes.append(sha256_file(intensity_path))

        canonical_dir = rad.outputs_dir / "mcmono"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        first_weight = counts[0] / float(sum(counts))
        binary_metadata = combine_mean_intensity_files(
            estimator_dirs[0] / "mean_intensity.bout",
            estimator_dirs[1] / "mean_intensity.bout",
            canonical_dir / "mean_intensity.bout",
            first_weight=first_weight,
        )
        shutil.copy2(
            estimator_dirs[0] / "mcmono_wavelength_micron.inp",
            canonical_dir / "mcmono_wavelength_micron.inp",
        )
        rad._postprocess_chi(
            canonical_dir / "mean_intensity.bout",
            uv_min,
            uv_max,
            compute_products=True,
        )
        volumes = compute_cell_volumes(rad.model)
        product_metrics: dict[str, dict[str, np.ndarray]] = {}
        for name in sorted(set(estimator_products[0]) & set(estimator_products[1])):
            first = estimator_products[0][name]
            second = estimator_products[1][name].to(first.units)
            metrics = paired_radial_uncertainty_metrics(
                first.magnitude,
                second.magnitude,
                volumes,
                first_weight=first_weight,
                tolerance=noise_tolerance,
            )
            metrics.pop("sigma")
            if name != "chi_broad":
                metrics.pop("mean")
            product_metrics[name] = metrics

        frequencies1, shell_j1 = radial_shell_mean_intensity(
            estimator_dirs[0] / "mean_intensity.bout", volumes
        )
        frequencies2, shell_j2 = radial_shell_mean_intensity(
            estimator_dirs[1] / "mean_intensity.bout", volumes
        )
        if not np.array_equal(frequencies1, frequencies2):
            raise ValueError("Paired scout frequency grids differ")
        shell_j_mean = first_weight * shell_j1 + (1.0 - first_weight) * shell_j2
        shell_j_sigma = 0.5 * np.abs(shell_j1 - shell_j2)
        shell_j_fractional = np.divide(
            shell_j_sigma,
            np.abs(shell_j_mean),
            out=np.where(shell_j_sigma == 0.0, 0.0, np.inf),
            where=np.abs(shell_j_mean) > 0.0,
        )
        wavelength_metrics = {
            "frequencies_hz": frequencies1,
            "shell_mean": shell_j_mean,
            "shell_sigma": shell_j_sigma,
            "shell_fractional": shell_j_fractional,
            "shell_fractional_max": np.max(shell_j_fractional, axis=1),
        }

        reliable = np.all(shell_j_fractional <= float(noise_tolerance), axis=1)
        for metrics in product_metrics.values():
            reliable &= metrics["shell_fractional"] <= float(noise_tolerance)
            reliable &= metrics["cell_fractional_p99"] <= float(noise_tolerance)

        metadata = {
            "seeds": list(seeds),
            "nphot_estimators": list(counts),
            "nphot_total": int(sum(counts)),
            "estimator_sha256": hashes,
            "canonical_scout_sha256": sha256_file(
                canonical_dir / "mean_intensity.bout"
            ),
            "first_weight": float(first_weight),
            "binary": binary_metadata,
            "noise_tolerance": float(noise_tolerance),
        }
        (canonical_dir / "paired_scout.json").write_text(
            json.dumps(jsonable(metadata), indent=2, sort_keys=True) + "\n"
        )
        return PairedScoutResult(
            metadata=metadata,
            products=product_metrics,
            wavelength_metrics=wavelength_metrics,
            reliable_shells=reliable,
            estimator_dirs=estimator_dirs,
        )

    @staticmethod
    def _remove_scout_estimators(scout: PairedScoutResult) -> None:
        """Remove temporary paired intensity outputs after diagnostics are saved."""
        for directory in scout.estimator_dirs:
            shutil.rmtree(directory, ignore_errors=True)

    def _stellar_screen_profiles(
        self,
        rad: "RadModel",
        scout: PairedScoutResult,
        product_specs,
    ) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        """Return unattenuated direct-stellar fractions of measured UV products."""
        import diskbridge

        radii_cm = np.asarray(rad.model.mesh.centers("r").to("cm").magnitude, dtype=np.float64)
        c_cgs = float(units("c").to("cm/s").magnitude)
        profiles: dict[str, dict[str, np.ndarray]] = {}
        volumes = compute_cell_volumes(rad.model)
        for spec in product_specs:
            source_strength = compute_star_uv_source_strength(
                diskbridge.params,
                float(spec.band.lam_min_nm) * 1.0e-7,
                float(spec.band.lam_max_nm) * 1.0e-7,
                weighting=spec.band.weight,
            )
            if spec.band.weight == "photon":
                reference_flux = float(
                    draine_reference_for_product(spec.field_name, product_specs)
                    .to("1/(cm^2 s)")
                    .magnitude
                )
            else:
                reference_flux = c_cgs * float(
                    draine_reference_for_product(spec.field_name, product_specs)
                    .to("erg/cm^3")
                    .magnitude
                )
            stellar_product = source_strength / (4.0 * np.pi * radii_cm**2 * reference_flux)
            measured = scout.products[spec.field_name]["shell_mean"]
            shell_fraction = np.divide(
                stellar_product,
                measured,
                out=np.full_like(stellar_product, np.inf),
                where=measured > 0.0,
            )
            measured_cells = np.asarray(
                rad.uv_products[spec.field_name].to("dimensionless").magnitude,
                dtype=np.float64,
            )
            cell_fraction = np.divide(
                stellar_product[:, None, None],
                measured_cells,
                out=np.full_like(measured_cells, np.inf),
                where=measured_cells > 0.0,
            )
            cell_p99 = np.empty(radii_cm.size, dtype=np.float64)
            cell_max = np.empty(radii_cm.size, dtype=np.float64)
            for ir in range(radii_cm.size):
                cell_p99[ir] = weighted_quantile(cell_fraction[ir], volumes[ir], 0.99)
                cell_max[ir] = float(np.nanmax(cell_fraction[ir]))
            profiles[spec.field_name] = {
                "shell_mean": shell_fraction,
                "cell_p99": cell_p99,
                "cell_max": cell_max,
            }
        return (
            np.stack([profiles[spec.field_name]["shell_mean"] for spec in product_specs]),
            profiles,
        )

    def _write_scout_diagnostics(
        self,
        rad: "RadModel",
        scout: PairedScoutResult,
        stellar_profiles: dict[str, dict[str, np.ndarray]],
    ) -> Path:
        """Persist paired-estimator radial diagnostics without full 3-D duplicates."""
        output = rad.outputs_dir / "mcmono" / "uv_scout_metrics.npz"
        arrays: dict[str, np.ndarray] = {
            "r_au": np.asarray(rad.model.mesh.centers("r").to("au").magnitude),
            "reliable_shells": scout.reliable_shells,
            "frequency_hz": scout.wavelength_metrics["frequencies_hz"],
            "j_shell_mean": scout.wavelength_metrics["shell_mean"],
            "j_shell_sigma": scout.wavelength_metrics["shell_sigma"],
            "j_shell_fractional": scout.wavelength_metrics["shell_fractional"],
        }
        for name, metrics in scout.products.items():
            for metric_name in (
                "shell_mean",
                "shell_fractional",
                "cell_fractional_p99",
                "cell_fractional_max",
                "cell_fractional_max_flat_index",
                "failing_volume_fraction",
            ):
                arrays[f"{name}__{metric_name}"] = metrics[metric_name]
        for name, metrics in stellar_profiles.items():
            arrays[f"{name}__stellar_fraction"] = metrics["shell_mean"]
            arrays[f"{name}__stellar_fraction_p99"] = metrics["cell_p99"]
            arrays[f"{name}__stellar_fraction_max"] = metrics["cell_max"]
        np.savez_compressed(output, **arrays)
        return output

    def _join_diagnostics(
        self,
        parent: PairedScoutResult,
        child: PairedScoutResult,
        *,
        parent_comparison_idx: int,
        child_comparison_idx: int,
        volumes: np.ndarray,
        tolerance: float,
    ) -> dict[str, Any]:
        """Compare one shared parent-child shell and warn on scientific mismatch."""
        tiny = np.finfo(np.float64).tiny

        def discrepancy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            denominator = np.abs(a) + np.abs(b)
            return np.divide(
                2.0 * np.abs(a - b),
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator > tiny,
            )

        parent_j = parent.wavelength_metrics["shell_mean"][parent_comparison_idx]
        child_j = child.wavelength_metrics["shell_mean"][child_comparison_idx]
        parent_j_sigma = parent.wavelength_metrics["shell_sigma"][parent_comparison_idx]
        child_j_sigma = child.wavelength_metrics["shell_sigma"][child_comparison_idx]
        j_delta = discrepancy(parent_j, child_j)
        j_sigma_rel = np.divide(
            2.0 * np.sqrt(parent_j_sigma**2 + child_j_sigma**2),
            np.abs(parent_j) + np.abs(child_j),
            out=np.zeros_like(parent_j),
            where=(np.abs(parent_j) + np.abs(child_j)) > tiny,
        )

        products: dict[str, dict[str, float]] = {}
        warning = bool(np.any(j_delta > float(tolerance)))
        common_products = sorted(set(parent.products) & set(child.products))
        for name in common_products:
            parent_mean = float(parent.products[name]["shell_mean"][parent_comparison_idx])
            child_mean = float(child.products[name]["shell_mean"][child_comparison_idx])
            delta = float(discrepancy(np.asarray(parent_mean), np.asarray(child_mean)))
            parent_sigma = abs(parent_mean) * float(
                parent.products[name]["shell_fractional"][parent_comparison_idx]
            )
            child_sigma = abs(child_mean) * float(
                child.products[name]["shell_fractional"][child_comparison_idx]
            )
            denominator = abs(parent_mean) + abs(child_mean)
            sigma_rel = (
                2.0 * np.sqrt(parent_sigma**2 + child_sigma**2) / denominator
                if denominator > tiny
                else 0.0
            )
            products[name] = {"delta": delta, "sigma_rel": float(sigma_rel)}
            warning |= delta > float(tolerance)

        parent_cells = parent.products["chi_broad"]["mean"][parent_comparison_idx]
        child_cells = child.products["chi_broad"]["mean"][child_comparison_idx]
        if parent_cells.shape != child_cells.shape:
            raise ValueError("Parent-child comparison shells have incompatible angular grids")
        cell_delta = discrepancy(parent_cells, child_cells)
        shell_weights = np.asarray(volumes[child_comparison_idx], dtype=np.float64)
        cell_p99 = weighted_quantile(cell_delta, shell_weights, 0.99)
        result = {
            "parent_comparison_shell_idx": int(parent_comparison_idx),
            "child_comparison_shell_idx": int(child_comparison_idx),
            "tolerance": float(tolerance),
            "warning": bool(warning),
            "j_fractional_max": float(np.max(j_delta)),
            "j_sigma_rel_max": float(np.max(j_sigma_rel)),
            "frequency_hz": parent.wavelength_metrics["frequencies_hz"],
            "parent_j": parent_j,
            "child_j": child_j,
            "parent_j_sigma": parent_j_sigma,
            "child_j_sigma": child_j_sigma,
            "j_fractional": j_delta,
            "j_sigma_rel": j_sigma_rel,
            "chi_cell_fractional_p99": float(cell_p99),
            "products": products,
        }
        if warning:
            logger.warning(
                "Segmented UV join exceeds tolerance %.3g: max J discrepancy=%.3g, "
                "chi cell P99=%.3g; continuing with the child field",
                float(tolerance),
                result["j_fractional_max"],
                result["chi_cell_fractional_p99"],
            )
        return result
    
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
        merged_mag = np.array(merged.magnitude, copy=True)
        child_mag = np.asarray(child.magnitude)

        spatial_ndim = len(axis_order)
        if merged_mag.ndim < spatial_ndim or child_mag.ndim < spatial_ndim:
            raise ValueError(
                f"Cannot merge field with shape {child_mag.shape} into {merged_mag.shape} "
                f"for axis order {axis_order}"
            )

        leading_ndim = merged_mag.ndim - spatial_ndim
        if child_mag.ndim != merged_mag.ndim:
            raise ValueError(
                f"Cannot merge field with shape {child_mag.shape} into {merged_mag.shape}: "
                "segment and merged field must have the same number of axes"
            )
        if child_mag.shape[:leading_ndim] != merged_mag.shape[:leading_ndim]:
            raise ValueError(
                f"Cannot merge field with leading shape {child_mag.shape[:leading_ndim]} "
                f"into {merged_mag.shape[:leading_ndim]}"
            )

        slicer = [slice(None)] * leading_ndim
        for ax in axis_order:
            slicer.append(indexer.axis_slices.get(ax, slice(None)))

        target_shape = merged_mag[tuple(slicer)].shape
        if child_mag.shape != target_shape:
            raise ValueError(
                f"Cannot merge field with shape {child_mag.shape} into target slice "
                f"with shape {target_shape}"
            )

        merged_mag[tuple(slicer)] = child_mag
        return Quantity(merged_mag, merged.units)

    def _register_mesh_shaped_uv_products(
        self,
        uv_products: Dict[str, Quantity],
        axis_order: Tuple[str, ...],
    ) -> None:
        mesh_shape = tuple(self.base_model.mesh.shape)
        for name, product in uv_products.items():
            if np.asarray(product.magnitude).shape != mesh_shape:
                continue
            self.base_model.gas_register(
                name,
                Field(quantity=name, data=product, axis_order=axis_order),
            )

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

    def run_segmented_rt(
        self,
        nphot_therm: Optional[int] = None,
        nphot_mono: Optional[int] = None,
        mcmono_n_wavelengths: Optional[int] = None,
        mcmono_uv_n_wavelengths: Optional[int] = None,
        mcmono_wavelength_spacing: str = "log",
        mcmono_wavelengths_um: Optional[np.ndarray] = None,
        max_splits: Optional[int] = None,
        segmented_final_nphot_multiplier: Optional[float] = None,
        force: bool = False,
        diagnostic_plots: bool = False,
        plots_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run noise-aware segmented UV radiative transfer.

        Scout UV calculations use two independent estimators whose total
        packet count is the configured intermediate budget. Each child inherits
        the parent source-shell UV spectrum while retaining the configured IR
        and CMB spectrum. The selected terminal domain is then run once at the
        configured final thermal and monochromatic packet budgets.

        Parameters
        ----------
        nphot_therm, nphot_mono : int, optional
            Nominal thermal and monochromatic packet counts.
        mcmono_n_wavelengths, mcmono_uv_n_wavelengths : int, optional
            UV wavelength-grid controls. The UV-specific value takes
            precedence when both are supplied.
        mcmono_wavelength_spacing : {"log", "linear"}, optional
            UV wavelength spacing.
        mcmono_wavelengths_um : ndarray, optional
            Explicit UV wavelengths in micron.
        max_splits : int, optional
            Maximum number of child boundaries.
        segmented_final_nphot_multiplier : float, optional
            Packet multiplier for the terminal science calculation.
        force : bool, optional
            Recompute cached RADMC-3D outputs.
        diagnostic_plots : bool, optional
            Write merged and per-segment diagnostics.
        plots_dir : str or pathlib.Path, optional
            Diagnostic output directory.

        Returns
        -------
        dict
            Segment metadata, diagnostics, and merged radiation fields.
        """
        from diskbridge.radmc3d.model import RadModel, _validate_radmc_photon_count
        from diskbridge.radmc3d.writer import RadWriter
        import diskbridge

        params = diskbridge.params
        tolerance = float(params.segmented_tol)
        if tolerance <= 0.0:
            raise ValueError("segmented_tol must be positive")
        r_clip_min_au = float(params.segmented_r_clip_min.to("au").magnitude)
        stop_factor = float(params.segmented_stop_factor)
        if not 0.0 < stop_factor < 1.0:
            raise ValueError("segmented_stop_factor must be in (0, 1)")

        nominal_thermal = int(params.nphot_thermal if nphot_therm is None else nphot_therm)
        nominal_mono = int(params.nphot_mono if nphot_mono is None else nphot_mono)
        ratio = float(params.segmented_nphot_ratio)
        if ratio <= 0.0:
            raise ValueError("segmented_nphot_ratio must be positive")
        scout_thermal = int(ratio * nominal_thermal)
        scout_mono = int(ratio * nominal_mono)
        if scout_thermal < 1 or scout_mono < 2:
            raise ValueError(
                "Resolved scout budgets require at least one thermal and two mcmono packets"
            )
        final_multiplier = float(
            getattr(params, "segmented_final_nphot_multiplier", 1.0)
            if segmented_final_nphot_multiplier is None
            else segmented_final_nphot_multiplier
        )
        if final_multiplier < 1.0:
            raise ValueError("segmented_final_nphot_multiplier must be >= 1")
        final_thermal = int(final_multiplier * nominal_thermal)
        final_mono = int(final_multiplier * nominal_mono)
        if final_thermal < 1 or final_mono < 1:
            raise ValueError("Resolved final packet budgets must be positive")
        _validate_radmc_photon_count(scout_thermal, name="segmented scout nphot_thermal")
        scout_counts = (scout_mono // 2, scout_mono - scout_mono // 2)
        for estimator, count in enumerate(scout_counts, start=1):
            _validate_radmc_photon_count(
                count,
                name=f"segmented scout estimator {estimator} nphot_mono",
            )
        _validate_radmc_photon_count(final_thermal, name="segmented final nphot_thermal")
        _validate_radmc_photon_count(final_mono, name="segmented final nphot_mono")

        max_splits = int(params.segmented_max_splits if max_splits is None else max_splits)
        if max_splits < 0:
            raise ValueError("max_splits must be non-negative")
        mesh = self.base_model.mesh
        if mesh is None or mesh.coord_system != "spherical":
            raise ValueError("Segmented RT requires a spherical base mesh")
        axis_order = mesh.axis_names()

        cfg = get_config()
        runtime_mode = validate_uv_chemistry_config(cfg, segmented=True)
        product_specs = uv_product_specs_from_config(
            cfg.get("radmc3d", {}).get("uv_products", {})
        )
        product_edges_nm = uv_product_edges_from_specs(product_specs)
        broad_spec = next(spec for spec in product_specs if spec.field_name == "chi_broad")
        uv_min = Quantity(float(broad_spec.band.lam_min_nm), "nm")
        uv_max = Quantity(float(broad_spec.band.lam_max_nm), "nm")
        n_uv = int(
            params.uv_n_wavelengths
            if mcmono_uv_n_wavelengths is None
            else mcmono_uv_n_wavelengths
        )
        n_wavelengths = (
            int(mcmono_uv_n_wavelengths)
            if mcmono_uv_n_wavelengths is not None
            else int(mcmono_n_wavelengths)
            if mcmono_n_wavelengths is not None
            else n_uv
        )
        wavelengths_um = build_mcmono_wavelengths(
            wavelength_source="uv",
            wavelength_file=None,
            uv_min_um=float(uv_min.to("micron").magnitude),
            uv_max_um=float(uv_max.to("micron").magnitude),
            n_wavelengths=n_wavelengths,
            n_uv_enforce=0,
            spacing=mcmono_wavelength_spacing,
            provided_wavelengths=mcmono_wavelengths_um,
            extra_enforced_wavelengths_um=product_edges_nm * 1.0e-3,
        )

        logger.info(
            "Noise-aware segmented RT: scout mctherm=%d, paired mcmono total=%d, "
            "final mctherm=%d, final mcmono=%d, segmented_tol=%.3g",
            scout_thermal,
            scout_mono,
            final_thermal,
            final_mono,
            tolerance,
        )

        base_opacity_dir = self.base_model_dir / "radmc3d_inputs"
        base_opacity_dir.mkdir(parents=True, exist_ok=True)
        if not list(base_opacity_dir.glob("dustkappa_*.inp")):
            RadWriter(self.base_model, organize_files=True).compute_and_write_dust_opacities(
                self.base_model_dir
            )
        if not list(base_opacity_dir.glob("dustkappa_*.inp")):
            raise RuntimeError(f"No dustkappa_*.inp files found in {base_opacity_dir}")

        merged_temperature: Optional[Quantity] = None
        merged_chi: Optional[Quantity] = None
        merged_uv_products: Optional[dict[str, Quantity]] = None
        measured_mask: Optional[np.ndarray] = None
        segment_id: Optional[np.ndarray] = None
        segments: list[dict[str, Any]] = []
        split_radii_au: list[float] = []
        joins: list[dict[str, Any]] = []
        scout_runs: list[dict[str, Any]] = []

        current_outer_rmax_au = float(np.max(mesh.edges("r").to("au").magnitude))
        parent_rad: Optional[RadModel] = None
        parent_scout: Optional[PairedScoutResult] = None
        parent_split_info: Optional[dict[str, Any]] = None

        for level in range(max_splits + 1):
            if level == 0:
                bounds: Dict[str, Tuple[Optional[Quantity], Optional[Quantity]]] = {}
                work_dir = self.base_model_dir / "segments" / "segment_00_full"
                segment_model = self.base_model
                segment_indexer = None
            else:
                bounds = {"r": (None, current_outer_rmax_au * units("au"))}
                work_dir = (
                    self.base_model_dir
                    / "segments"
                    / f"segment_{level:02d}_rmax_{current_outer_rmax_au:.6g}au"
                )
                definition = SegmentDefinition(f"segment_{level:02d}", bounds, work_dir)
                segment_model, segment_indexer = self._build_segment_model_and_indexer(definition)

            rad = self._setup_segment(segment_model, work_dir, base_opacity_dir)
            if level > 0:
                if parent_rad is None:
                    raise RuntimeError("Missing parent segment for inherited UV source")
                self._inherit_external_source(
                    parent_rad,
                    rad,
                    current_outer_rmax_au,
                    uv_min,
                    uv_max,
                )

            rad.compute_temperature(
                nphot=scout_thermal,
                output_dir=rad.outputs_dir / "temperature",
                force=force,
            )
            scout = self._run_paired_uv_scout(
                rad,
                nphot_total=scout_mono,
                wavelengths_um=wavelengths_um,
                uv_min=uv_min,
                uv_max=uv_max,
                noise_tolerance=tolerance,
                level=level,
                force=force,
            )
            stellar_array, stellar_profiles = self._stellar_screen_profiles(
                rad, scout, product_specs
            )
            metrics_file = self._write_scout_diagnostics(rad, scout, stellar_profiles)

            join: Optional[dict[str, Any]] = None
            if parent_scout is not None and parent_split_info is not None:
                child_comparison_idx = int(rad.model.mesh.shape[0] - 1)
                join = self._join_diagnostics(
                    parent_scout,
                    scout,
                    parent_comparison_idx=int(parent_split_info["comparison_shell_idx"]),
                    child_comparison_idx=child_comparison_idx,
                    volumes=compute_cell_volumes(rad.model),
                    tolerance=tolerance,
                )
                (rad.outputs_dir / "mcmono" / "join_diagnostics.json").write_text(
                    json.dumps(jsonable(join), indent=2, sort_keys=True) + "\n"
                )
                joins.append(join)

            self._remove_scout_estimators(scout)

            split_info: Optional[dict[str, Any]] = None
            terminal_reason: Optional[str] = None
            if level >= max_splits:
                terminal_reason = "maximum_splits"
            else:
                try:
                    split_radius, raw_split_info = find_noise_aware_split(
                        rad.model.mesh.edges("r").to("au").magnitude,
                        scout.reliable_shells,
                        stellar_array,
                        stellar_fraction_threshold=runtime_mode.stellar_fraction_threshold,
                        r_clip_min_au=r_clip_min_au,
                    )
                    split_info = {
                        key: value
                        for key, value in raw_split_info.items()
                        if key.endswith("_idx")
                    }
                    split_info["r_split_au"] = float(split_radius)
                    radial_centers_au = np.asarray(
                        rad.model.mesh.centers("r").to("au").magnitude,
                        dtype=np.float64,
                    )
                    split_info["comparison_shell_r_au"] = float(
                        radial_centers_au[int(split_info["comparison_shell_idx"])]
                    )
                    split_info["source_shell_r_au"] = float(
                        radial_centers_au[int(split_info["source_shell_idx"])]
                    )
                    if float(split_radius) >= stop_factor * current_outer_rmax_au:
                        terminal_reason = "insufficient_radial_reduction"
                except ValueError as exc:
                    terminal_reason = str(exc)

            segment_entry: dict[str, Any] = {
                "level": int(level),
                "work_dir": str(work_dir),
                "r_max_au": float(current_outer_rmax_au),
                "nphot_thermal": int(scout_thermal),
                "nphot_mono": int(scout_mono),
                "scout": jsonable(scout.metadata),
                "scout_metrics_file": str(metrics_file),
                "noise_reliable_shell_count": int(np.count_nonzero(scout.reliable_shells)),
                "noise_shell_count": int(scout.reliable_shells.size),
                "split": split_info,
                "join": join,
                "stellar_fraction_threshold": float(
                    runtime_mode.stellar_fraction_threshold
                ),
                "has_temperature": True,
                "has_mcmono": True,
                "is_final": terminal_reason is not None,
            }
            scout_runs.append(dict(segment_entry))

            if terminal_reason is not None:
                self._run_segment_rt(
                    rad,
                    final_thermal,
                    final_mono,
                    wavelengths_um,
                    True,
                    compute_uv_products=True,
                    uv_min=uv_min,
                    uv_max=uv_max,
                )
                segment_entry["nphot_thermal"] = int(final_thermal)
                segment_entry["nphot_mono"] = int(final_mono)
                segment_entry["terminal_reason"] = terminal_reason
                segment_entry["scout_canonical_replaced_by_final"] = True
            else:
                split_radius = float(split_info["r_split_au"])
                split_radii_au.append(split_radius)

            merged_temperature, merged_chi = self._merge_segment_fields(
                merged_T=merged_temperature,
                merged_chi=merged_chi,
                rad=rad,
                indexer=segment_indexer,
                axis_order=axis_order,
            )
            if runtime_mode.products_enabled:
                merged_uv_products, measured_mask, segment_id = self._merge_segment_uv_products(
                    merged_uv_products=merged_uv_products,
                    uv_product_measured_mask=measured_mask,
                    segment_id=segment_id,
                    rad=rad,
                    indexer=segment_indexer,
                    axis_order=axis_order,
                    segment_level=level,
                    measured=True,
                )
            segments.append(segment_entry)

            if terminal_reason is not None:
                break
            parent_rad = rad
            parent_scout = scout
            parent_split_info = split_info
            current_outer_rmax_au = float(split_info["r_split_au"])

        if merged_temperature is None or merged_chi is None:
            raise RuntimeError("Segmented RT produced no merged fields")
        self.base_model.gas_register(
            "dust_temperature",
            Field("dust_temperature", merged_temperature, axis_order=axis_order),
        )
        self.base_model.gas_register(
            "chi", Field("chi", merged_chi, axis_order=axis_order)
        )
        if runtime_mode.products_enabled:
            if merged_uv_products is None or measured_mask is None or segment_id is None:
                raise RuntimeError("Segmented RT produced incomplete UV-product state")
            self._register_mesh_shaped_uv_products(merged_uv_products, axis_order)
            self.base_model.gas_register(
                "uv_product_measured_mask",
                Field(
                    "uv_product_measured_mask",
                    Quantity(measured_mask, "dimensionless"),
                    axis_order=axis_order,
                ),
            )
            self.base_model.gas_register(
                "segment_id",
                Field("segment_id", Quantity(segment_id, "dimensionless"), axis_order=axis_order),
            )
        self.base_model.validate_canonical_axis_orders(include_dust=False)

        base_rad = RadModel(self.base_model, model_dir=self.base_model_dir)
        base_rad.dust_temperature = merged_temperature
        base_rad.chi = merged_uv_products["chi_broad"] if merged_uv_products else merged_chi
        base_rad.uv_products = merged_uv_products or {"chi_broad": merged_chi}
        base_rad.radiation_mode = (
            "local_uv_products" if runtime_mode.products_enabled else "local_chi"
        )
        if self.base_model.dust is None or int(self.base_model.dust.nbin) < 1:
            raise ValueError("Segmented RT requires at least one dust species")
        base_rad.outputs_dir.mkdir(parents=True, exist_ok=True)
        base_rad.writer.write_dust_temperature(
            merged_temperature,
            output_dir=base_rad.outputs_dir,
            nspec=int(self.base_model.dust.nbin),
        )

        result = {
            "mode": "noise_aware_inherited_uv",
            "mcmono_wavelength_source": "uv",
            "segmented_final_nphot_multiplier": final_multiplier,
            "segmented_tol": tolerance,
            "stellar_fraction_threshold": runtime_mode.stellar_fraction_threshold,
            "nphot_thermal_nominal": nominal_thermal,
            "nphot_mono_nominal": nominal_mono,
            "nphot_thermal_final": final_thermal,
            "nphot_mono_final": final_mono,
            "split_radii_au": split_radii_au,
            "segments": segments,
            "scout_runs": scout_runs,
            "joins": joins,
            "temperature": merged_temperature,
            "chi": base_rad.chi,
            "uv_products": merged_uv_products,
            "uv_product_mode": "all_segments_measured",
            "outer_product_policy": None,
            "uv_product_measured_mask": measured_mask,
            "segment_id": segment_id,
        }
        summary_path = self.base_model_dir / "segments" / "segmented_rt_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                jsonable(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in {"temperature", "chi", "uv_products", "uv_product_measured_mask", "segment_id"}
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if diagnostic_plots:
            plot_output_dir = (
                Path(plots_dir)
                if plots_dir is not None
                else self.base_model_dir / "plots" / "segmented_rt"
            )
            made = self._make_segmented_rt_diagnostics(base_rad, result, plot_output_dir)
            result["diagnostic_plots"] = [str(path) for path in made]
        return result
