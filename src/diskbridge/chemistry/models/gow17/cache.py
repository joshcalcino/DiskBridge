"""Dependency identities for cached GOW17 chemistry results."""

# db-keywords: gow17, chemistry, hashing, cache, units
# db-role: canonical
# db-scope: package
# db-purpose: Physical dependency context for GOW17 result reuse.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from diskbridge.chemistry.shielding.angular_uv_weights import SHIELDING_ANGULAR_WEIGHTING
from diskbridge.chemistry.shielding.healpix_utils import SHIELDING_RAY_GEOMETRY
from diskbridge.model.field import Field
from diskbridge.model.microturbulence import ensure_microturbulence_field
from diskbridge.radmc3d.cache import stable_json_hash
from diskbridge.radmc3d.model import DUST_TEMPERATURE_WEIGHTING
from diskbridge.radmc3d.uv_products import UV_PRODUCT_MERGED_FIELD_NAMES
from diskbridge.utils.serialization import jsonable
from diskbridge.utils import sha256_file

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel


def _quantity_context(q, unit: str | None = None) -> dict[str, Any]:
    data = q.to(unit) if unit is not None else q
    arr = np.ascontiguousarray(np.asarray(data.magnitude, dtype=np.float64))
    return {
        "unit": str(data.units),
        "shape": list(arr.shape),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }



def _field_context(field: Field, unit: str | None = None) -> dict[str, Any]:
    out = _quantity_context(field.data, unit=unit)
    out["axis_order"] = list(field.axis_order)
    return out



def _file_hashes(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                out[str(path.relative_to(root))] = sha256_file(path)
    return out



def _gow17_cache_context(rad: "RadModel", config: dict) -> dict[str, Any]:
    """Build the explicit dependency context for GOW17 chemistry."""
    config = {
        key: value
        for key, value in dict(config).items()
        if key not in {"checkpoint", "_output_dir"}
    }
    model = rad.model
    if model.gas is None:
        raise ValueError("GOW17 cache context requires model.gas")

    ensure_microturbulence_field(model, rad.params)

    gas_units = {
        "density": "g/cm^3",
        "dust_temperature": "K",
        "microturbulence": "km/s",
        "chi": "dimensionless",
        "chi_broad": "dimensionless",
        "chi_h2": "dimensionless",
        "chi_co": "dimensionless",
        "chi_c": "dimensionless",
        "G_CO_pdes": "dimensionless",
        "F_CO_pdes_photon": "1/(cm^2 s)",
        "F_CO_pdes_photon_bands": "1/(cm^2 s)",
        "uv_product_measured_mask": "dimensionless",
        "segment_id": "dimensionless",
    }
    gas_fields: dict[str, Any] = {}
    required = ("density", "dust_temperature", "microturbulence")
    optional = (
        "chi",
        "uv_product_measured_mask",
        "segment_id",
        *UV_PRODUCT_MERGED_FIELD_NAMES,
    )
    for name in (*required, *optional):
        if name in model.gas:
            gas_fields[name] = _field_context(model.gas[name], unit=gas_units.get(name))
        elif name in required:
            raise KeyError(f"GOW17 cache context requires gas field {name!r}")

    dust_context: dict[str, Any] = {"present": model.dust is not None}
    if model.dust is not None:
        bins: dict[str, Any] = {}
        for bin_name, dust_bin in sorted(model.dust.bins.items()):
            bins[bin_name] = {
                "size": jsonable(dust_bin.size),
                "size_min": jsonable(dust_bin.size_min),
                "size_max": jsonable(dust_bin.size_max),
                "mass_fraction": float(dust_bin.mass_fraction),
                "density_material": jsonable(dust_bin.density_material),
                "opacity": dust_bin.opacity.to_dict(),
                "opacity_species": dust_bin.opacity_species,
                "density": _field_context(dust_bin["density"], unit="g/cm^3"),
                "temperature": _field_context(
                    model.dust[f"temperature_{bin_name}"], unit="K",
                ),
            }
        dust_context.update(
            {
                "nbin": int(model.dust.nbin),
                "bins": bins,
            }
        )

    model_dir = Path(rad.model_dir)
    upstream_contexts = _file_hashes(
        model_dir,
        (
            "radmc3d_outputs/**/cache_context.json",
            "segments/**/cache_context.json",
        ),
    )
    opacity_inputs = _file_hashes(
        Path(rad.inputs_dir),
        (
            "dustopac.inp",
            "dustkappa_*.inp",
            "dustkapscatmat_*.inp",
            "optool_*.json",
        ),
    )

    return {
        "kind": "gow17",
        "shielding_ray_geometry": SHIELDING_RAY_GEOMETRY,
        "shielding_angular_weighting": SHIELDING_ANGULAR_WEIGHTING,
        "dust_temperature_weighting": DUST_TEMPERATURE_WEIGHTING,
        "config_sha256": stable_json_hash(config),
        "params": {
            "nside": int(rad.params.nside),
        },
        "model": {
            "mesh_shape": list(model.mesh.shape) if model.mesh is not None else None,
            "mesh_axes": list(model.mesh.axis_names()) if model.mesh is not None else None,
            "gas_fields": gas_fields,
            "dust": dust_context,
        },
        "radmc": {
            "upstream_cache_contexts": upstream_contexts,
            "opacity_inputs": opacity_inputs,
        },
    }

