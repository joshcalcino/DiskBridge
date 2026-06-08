"""Run the 2D outer-disk CO phase validation."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

import diskbridge
from diskbridge._params import read_params
from diskbridge._units import Quantity
from diskbridge.chemistry.io import attach_chemistry_result_to_model
from diskbridge.chemistry.models.gow17 import run_gow17
from diskbridge.model.field import Field
from diskbridge.model.utils import field_data_as_order
from diskbridge.radmc3d.model import RadModel
from diskbridge.radmc3d.uv_products import UV_PRODUCT_MERGED_FIELD_NAMES

try:
    from .build_model import (
        CONFIG_FILE,
        ROOT,
        build_outer_disk_model,
        field_array,
        load_validation_config,
        region_masks,
        resolve_run_settings,
        summarize_regions,
        variant_configs,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from build_model import (  # type: ignore
        CONFIG_FILE,
        ROOT,
        build_outer_disk_model,
        field_array,
        load_validation_config,
        region_masks,
        resolve_run_settings,
        summarize_regions,
        variant_configs,
        write_json,
    )


PARAMS_FILE = ROOT / "params.txt"
OUTPUT_DIR = ROOT / "outputs" / "default"
VALIDATION_UV_INTENSITY_FLOOR = 1.0e-300
UV_PRODUCTS_GAS_FIELD_NAMES = tuple(
    name for name in UV_PRODUCT_MERGED_FIELD_NAMES if name != "F_CO_pdes_photon_bands"
)


def _install_validation_uv_floor():
    """Install a validation-local floor for exact zero RADMC UV samples."""

    from diskbridge.radmc3d import uv_products

    original = uv_products._loglinear_interp_strict
    stats: dict[str, Any] = {
        "enabled": True,
        "floor_value": VALIDATION_UV_INTENSITY_FLOOR,
        "floor_unit": "erg/(s*cm^2*Hz*sr)",
        "floored_values": 0,
        "cells_with_any_floor": 0,
        "cells_with_all_floor": 0,
        "chunks_with_floor": 0,
    }

    def _flooring_loglinear_interp(lam_sample, J_sample, lam_quad):
        J = np.asarray(J_sample, dtype=np.float64)
        if np.any(~np.isfinite(J)) or np.any(J < 0.0):
            raise ValueError(
                "UV mean intensity contains negative or non-finite values. "
                "The validation-local floor only handles exact zero samples."
            )
        mask = J <= 0.0
        if np.any(mask):
            stats["floored_values"] += int(np.count_nonzero(mask))
            stats["cells_with_any_floor"] += int(np.count_nonzero(np.any(mask, axis=1)))
            stats["cells_with_all_floor"] += int(np.count_nonzero(np.all(mask, axis=1)))
            stats["chunks_with_floor"] += 1
            J = J.copy()
            J[mask] = VALIDATION_UV_INTENSITY_FLOOR
        return original(lam_sample, J, lam_quad)

    uv_products._loglinear_interp_strict = _flooring_loglinear_interp
    return stats, original


def _jsonable_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, Quantity):
            mag = np.asarray(value.magnitude)
            out[key] = {
                "unit": str(value.units),
                "magnitude": mag.tolist() if mag.ndim else float(mag),
            }
        elif isinstance(value, dict):
            out[key] = _jsonable_meta(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [
                _jsonable_meta(v) if isinstance(v, dict) else v.item() if isinstance(v, np.generic) else v
                for v in value
            ]
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _configure_diskbridge(settings, *, external_uv: bool) -> None:
    del settings
    diskbridge.params = read_params(PARAMS_FILE)
    diskbridge.params.external_uv = bool(external_uv)
    diskbridge.params.external_uv_chi = 1.0 if external_uv else 0.0
    diskbridge.params.microturbulence = Quantity(0.2, "km/s")


def _cell_count(settings) -> int:
    return (
        int(settings.grid["nr"])
        * int(settings.grid["ntheta_upper"])
        * 2
        * int(settings.grid["nphi"])
    )


def _photon_budget(settings) -> dict[str, Any]:
    ncells = _cell_count(settings)
    thermal = int(diskbridge.params.nphot_thermal)
    mono = int(diskbridge.params.nphot_mono)
    thermal_per_cell = thermal / float(ncells)
    mono_per_cell = mono / float(ncells)
    return {
        "ncells": ncells,
        "nphot_thermal": thermal,
        "nphot_mono": mono,
        "thermal_packets_per_cell": float(thermal_per_cell),
        "mono_packets_per_cell": float(mono_per_cell),
        "thermal_poisson_floor": float(1.0 / np.sqrt(thermal_per_cell)),
        "mono_poisson_floor": float(1.0 / np.sqrt(mono_per_cell)),
    }


def _register_quantity(model, name: str, q: Quantity) -> None:
    model.gas_register(
        name,
        Field(
            quantity=name,
            data=q,
            axis_order=model.mesh.axis_names(),
        ),
    )


def _register_uv_product_fields(model, products: dict[str, Quantity]) -> None:
    for name in UV_PRODUCTS_GAS_FIELD_NAMES:
        _register_quantity(model, name, products[name])


def _save_uv_products(path: Path, products: dict[str, Quantity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(q.magnitude) for name, q in products.items()}
    units = {name: str(q.units) for name, q in products.items()}
    arrays["__units_json__"] = np.asarray(json.dumps(units))
    np.savez_compressed(path, **arrays)


def _mean_intensity_zero_stats(rad: RadModel) -> dict[str, Any]:
    J = np.asarray(
        rad.mean_intensity.to("erg/(s*cm^2*Hz*sr)").magnitude,
        dtype=np.float64,
    )
    positive = J[J > 0.0]
    return {
        "total_values": int(J.size),
        "zero_values": int(np.count_nonzero(J == 0.0)),
        "negative_values": int(np.count_nonzero(J < 0.0)),
        "nonfinite_values": int(np.count_nonzero(~np.isfinite(J))),
        "zero_fraction": float(np.count_nonzero(J == 0.0) / J.size),
        "cells_with_any_zero": int(np.count_nonzero(np.any(J <= 0.0, axis=1))),
        "cells_with_all_zero": int(np.count_nonzero(np.all(J <= 0.0, axis=1))),
        "positive_min": float(np.min(positive)) if positive.size else None,
        "positive_max": float(np.max(positive)) if positive.size else None,
    }


def _run_radmc_products(
    settings,
    out_dir: Path,
    *,
    external_uv: bool,
    label: str,
) -> tuple[dict[str, Quantity], Quantity, dict[str, Any]]:
    _configure_diskbridge(settings, external_uv=external_uv)
    photon_budget = _photon_budget(settings)
    model, grid_meta = build_outer_disk_model(settings)
    rad_dir = out_dir / label
    rad_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PARAMS_FILE, rad_dir / "params.txt")

    rad = RadModel(model, model_dir=rad_dir)
    rad.writer.write_model_inputs(rad_dir, include_gas_velocity=False)
    start = time.perf_counter()
    tdust = rad.compute_temperature(
        nphot=int(diskbridge.params.nphot_thermal),
        force=True,
    )
    from diskbridge.radmc3d import uv_products

    uv_floor_stats, original_uv_interp = _install_validation_uv_floor()
    skipped_product_registrations: list[dict[str, str]] = []
    original_gas_register = rad.model.gas_register

    def _gas_register_mesh_products_only(name, field):
        if name == "F_CO_pdes_photon_bands":
            skipped_product_registrations.append(
                {
                    "field": name,
                    "reason": "per-band UV product has an extra band axis",
                }
            )
            return
        original_gas_register(name, field)

    rad.model.gas_register = _gas_register_mesh_products_only
    try:
        chi = rad.compute_mcmono(
            nphot=int(diskbridge.params.nphot_mono),
            n_wavelengths=int(diskbridge.params.uv_n_wavelengths),
            force=True,
            compute_uv_products=True,
        )
    finally:
        rad.model.gas_register = original_gas_register
        uv_products._loglinear_interp_strict = original_uv_interp
    elapsed = time.perf_counter() - start
    products = {name: rad.uv_products[name] for name in UV_PRODUCT_MERGED_FIELD_NAMES}
    products_path = out_dir / ("uv_products.npz" if external_uv else "uv_products_no_external.npz")
    _save_uv_products(products_path, products)
    meta = {
        "elapsed_s": elapsed,
        "grid": grid_meta,
        "external_uv": bool(diskbridge.params.external_uv),
        "external_uv_chi": float(diskbridge.params.external_uv_chi),
        "nphot_thermal": int(diskbridge.params.nphot_thermal),
        "nphot_mono": int(diskbridge.params.nphot_mono),
        "uv_n_wavelengths": int(diskbridge.params.uv_n_wavelengths),
        "scattering_mode_max": int(diskbridge.params.scat_mode),
        "photon_budget": photon_budget,
        "validation_uv_floor": {
            "source_mean_intensity": _mean_intensity_zero_stats(rad),
            "interpolation_applications": uv_floor_stats,
        },
        "skipped_model_registrations": skipped_product_registrations,
        "chi_min": float(np.nanmin(chi.magnitude)),
        "chi_max": float(np.nanmax(chi.magnitude)),
        "uv_products": {
            name: {
                "unit": str(products[name].units),
                "min": float(np.nanmin(products[name].magnitude)),
                "max": float(np.nanmax(products[name].magnitude)),
            }
            for name in UV_PRODUCT_MERGED_FIELD_NAMES
        },
        "uv_products_path": str(products_path),
        "radmc3d_dir": str(rad_dir),
    }
    summary_name = "radmc3d_uv_summary.json" if external_uv else "radmc3d_uv_no_external_summary.json"
    write_json(out_dir / summary_name, meta)
    return products, tdust, meta


def _grid_sanity_summary(settings, out_dir: Path, products: dict[str, Quantity]) -> dict[str, Any]:
    model, grid_meta = build_outer_disk_model(settings)
    _register_uv_product_fields(model, products)
    _register_quantity(model, "chi", products["chi_broad"])
    fields = {
        "nH": "number_density_H",
        "Tdust": "dust_temperature",
        "chi": "chi_broad",
    }
    summary = {
        "grid": grid_meta,
        "photon_budget": _photon_budget(settings),
        "regions": summarize_regions(model, settings, fields),
        "region_cell_counts": {
            name: int(np.count_nonzero(mask))
            for name, mask in region_masks(model, settings, fields["chi"]).items()
        },
    }
    write_json(out_dir / "grid_summary.json", summary)
    return summary


def _run_variant(
    name: str,
    overrides: dict[str, Any],
    settings,
    out_dir: Path,
    products: dict[str, Quantity],
    tdust: Quantity,
    *,
    radiation_context: dict[str, Any],
) -> dict[str, Any]:
    model, grid_meta = build_outer_disk_model(settings)
    _register_quantity(model, "dust_temperature", tdust)
    _register_quantity(model, "temperature", tdust)
    _register_uv_product_fields(model, products)
    _register_quantity(model, "chi", products["chi_broad"])

    variant_dir = out_dir / "variants" / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    rad = RadModel(model, model_dir=variant_dir / "rad")
    rad.dust_temperature = field_data_as_order(
        model.gas["dust_temperature"], model.mesh.axis_names()
    ).to("K")
    rad.gas_temperature = field_data_as_order(
        model.gas["gas_temperature"], model.mesh.axis_names()
    ).to("K")
    rad.set_local_uv_products(products)

    start = time.perf_counter()
    result = run_gow17(rad, overrides)
    elapsed = time.perf_counter() - start
    attach_chemistry_result_to_model(rad, result)

    snapshot = variant_dir / "snapshot.h5"
    model.save_hdf5(snapshot, overwrite=True)
    write_json(variant_dir / "gow17_meta.json", _jsonable_meta(result.meta))

    fields = {
        "nH": "number_density_H",
        "Tdust": "dust_temperature",
        "Tgas": "gas_temperature",
        "chi_broad": "chi_broad",
        "chi_eff": "chem_chi_eff",
        "CO": "abundance_co",
        "CO_ice": "abundance_co_ice",
        "Cplus": "abundance_c+",
        "F_CO_pdes_photon_total": "chem_F_CO_pdes_photon_total",
    }
    region_summary = summarize_regions(model, settings, fields)
    status = field_array(model, "chem_status", "dimensionless") if model.gas and "chem_status" in model.gas else None
    summary = {
        "variant": name,
        "elapsed_s": elapsed,
        "snapshot": str(snapshot),
        "grid": grid_meta,
        "config_overrides": overrides,
        "radiation_context": radiation_context,
        "regions": region_summary,
        "status": {
            "min": int(np.nanmin(status)) if status is not None else None,
            "max": int(np.nanmax(status)) if status is not None else None,
            "nonzero_cells": int(np.count_nonzero(status)) if status is not None else None,
        },
        "gow17_meta": str(variant_dir / "gow17_meta.json"),
    }
    write_json(variant_dir / "summary.json", _jsonable_meta(summary))
    return summary


# db-keywords: validation, gow17, photodesorption, uv-products
# db-role: validation
def run_validation() -> Path:
    """Run the outer-disk CO phase validation with defaults from config files."""

    config = load_validation_config(CONFIG_FILE)
    settings = resolve_run_settings("default", config)
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    products, tdust, radmc_meta = _run_radmc_products(
        settings,
        out_dir,
        external_uv=True,
        label="radmc3d_uv",
    )
    grid_summary = _grid_sanity_summary(settings, out_dir, products=products)
    summaries = []
    variants = variant_configs(settings.chemistry)
    for variant, overrides in variants.items():
        summaries.append(
            _run_variant(
                variant,
                overrides,
                settings,
                out_dir,
                products,
                tdust,
                radiation_context={
                    "external_uv": True,
                    "radmc3d_summary": "radmc3d_uv_summary.json",
                    "uv_products": "uv_products.npz",
                },
            )
        )

    products_no_external, tdust_no_external, radmc_no_external_meta = _run_radmc_products(
        settings,
        out_dir,
        external_uv=False,
        label="radmc3d_uv_no_external",
    )
    no_external_variant = "co_ice_full_no_external_uv"
    summaries.append(
        _run_variant(
            no_external_variant,
            variants["co_ice_full_photodesorption"],
            settings,
            out_dir,
            products_no_external,
            tdust_no_external,
            radiation_context={
                "external_uv": False,
                "radmc3d_summary": "radmc3d_uv_no_external_summary.json",
                "uv_products": "uv_products_no_external.npz",
            },
        )
    )

    comparison = {
        "run": settings.name,
        "radmc3d": radmc_meta,
        "radmc3d_no_external_uv": radmc_no_external_meta,
        "external_field_comparison": {
            "external_uv_variant": "co_ice_full_photodesorption",
            "no_external_uv_variant": no_external_variant,
        },
        "grid": grid_summary,
        "variants": summaries,
    }
    write_json(out_dir / "comparison" / "summary.json", _jsonable_meta(comparison))

    try:
        from .plot_diagnostics import write_all_plots
    except ImportError:  # pragma: no cover
        from plot_diagnostics import write_all_plots  # type: ignore

    write_all_plots(out_dir)
    return out_dir


def main() -> None:
    out_dir = run_validation()
    print(out_dir)


if __name__ == "__main__":
    main()
