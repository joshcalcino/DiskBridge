"""HDF5 snapshot I/O for DiskBridge Model objects.

This module provides a portable cache format for expensive derived fields
(chemistry, shielding, thermal diagnostics, etc.), so you can restore a full
Model state later without re-running the upstream solvers.

What is stored
--------------
* Model metadata: coord_system, variables, compile_options, macros, file_units
* Mesh axes: edges/centers with units
* Gas fields: realized fields in `model.gas._fields` (units + axis_order)
* Disk parameters (if present): `model.disk.parameters` (supports Quantities)
* Dust components (if present): multi-component configuration created via
  `model.dust.add_component_from_mask(...)`, including each component's
  distribution, parameters, and mask/weight field.

Notes
-----
* Lazy fields are not serialized (only realized values).
* Field attrs are stored as JSON with a small encoder that supports Pint
  Quantities and numpy scalars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union
import json

import numpy as np
import h5py  # type: ignore

from diskbridge._units import Quantity
from diskbridge._logging import logger

from .mesh import Mesh, Axis
from .field import Field


SCHEMA_VERSION = 1


def _as_str(x: Any) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return str(x)


# -----------------------------------------------------------------------------
# JSON helpers (attrs + disk parameters)
# -----------------------------------------------------------------------------


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, Quantity):
            mag = np.asarray(obj.magnitude)
            return {
                "__quantity__": True,
                "unit": str(obj.units),
                "magnitude": mag.tolist() if mag.ndim > 0 else float(mag),
            }

        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return repr(obj)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, cls=_JSONEncoder, ensure_ascii=True)


def _json_loads(s: str) -> Any:
    def hook(d: Dict[str, Any]) -> Any:
        if d.get("__quantity__") is True:
            return Quantity(np.asarray(d["magnitude"]), d["unit"])
        return d

    return json.loads(s, object_hook=hook)


# -----------------------------------------------------------------------------
# Quantity/Field I/O primitives
# -----------------------------------------------------------------------------


def _write_quantity(group: "h5py.Group", name: str, q: Quantity) -> None:
    arr = np.asarray(q.magnitude)
    dset = group.create_dataset(name, data=arr)
    dset.attrs["unit"] = str(q.units)


def _read_quantity(group: "h5py.Group", name: str) -> Quantity:
    dset = group[name]
    unit = _as_str(dset.attrs["unit"])
    return Quantity(np.asarray(dset[...]), unit)


def _write_field(group: "h5py.Group", name: str, field: Field) -> None:
    fg = group.create_group(name)
    fg.attrs["quantity"] = field.quantity
    fg.attrs["axis_order_json"] = _json_dumps(list(field.axis_order))
    fg.attrs["attrs_json"] = _json_dumps(field.attrs)

    _write_quantity(fg, "data", field.data)


def _read_field(group: "h5py.Group", name: str) -> Field:
    fg = group[name]
    quantity_tag = _as_str(fg.attrs["quantity"])
    axis_order = tuple(_json_loads(_as_str(fg.attrs["axis_order_json"])))
    attrs = _json_loads(_as_str(fg.attrs.get("attrs_json", "{}")))
    data = _read_quantity(fg, "data")
    return Field(quantity=quantity_tag, data=data, axis_order=axis_order, attrs=attrs)


# -----------------------------------------------------------------------------
# Mesh I/O
# -----------------------------------------------------------------------------


def _write_mesh(group: "h5py.Group", mesh: Mesh) -> None:
    group.attrs["coord_system"] = mesh.coord_system

    axes_g = group.create_group("axes")

    for ax_name in mesh.axis_names():
        if ax_name not in mesh.axes:
            continue
        ax = mesh.axes[ax_name]
        ag = axes_g.create_group(ax_name)
        if ax.edges is not None:
            _write_quantity(ag, "edges", ax.edges)
        if ax.centers is not None:
            _write_quantity(ag, "centers", ax.centers)


def _read_mesh(group: "h5py.Group") -> Mesh:
    cs = _as_str(group.attrs["coord_system"])
    axes_g = group["axes"]
    axes: Dict[str, Axis] = {}

    for ax_name in axes_g.keys():
        ag = axes_g[ax_name]
        edges = _read_quantity(ag, "edges") if "edges" in ag else None
        centers = _read_quantity(ag, "centers") if "centers" in ag else None
        axes[str(ax_name)] = Axis(edges=edges, centers=centers)

    return Mesh(coord_system=cs, axes=axes)


# -----------------------------------------------------------------------------
# Dust I/O
# -----------------------------------------------------------------------------


def _write_dust(group: "h5py.Group", model: Any) -> None:
    dust = getattr(model, "dust", None)
    if dust is None:
        return

    dg = group.create_group("dust")

    comps = getattr(dust, "_components", [])
    dg.attrs["n_components"] = int(len(comps))
    dg.attrs["has_region_components"] = bool(getattr(dust, "_has_region_components", False))

    comps_g = dg.create_group("components")
    for i, comp in enumerate(comps):
        cg = comps_g.create_group(f"component_{i}")

        cg.attrs["component_json"] = _json_dumps(
            {
                "mode": comp.mode,
                "dust_to_gas_ratio": float(comp.dust_to_gas_ratio),
                "alpha": comp.alpha,
                "delta": comp.delta,
                "mean_molecular_weight": float(comp.mean_molecular_weight),
                "species_base": str(comp.species_base),
                "component_index": int(comp.component_index),
            }
        )

        dist = comp.distribution
        dist_g = cg.create_group("distribution")
        dist_g.attrs["power_index"] = float(dist.power_index)
        _write_quantity(dist_g, "amin", dist.amin)
        _write_quantity(dist_g, "amax", dist.amax)
        _write_quantity(dist_g, "grain_density", dist.grain_density)
        _write_quantity(dist_g, "bin_edges", dist.bin_edges)
        _write_quantity(dist_g, "bin_centers", dist.bin_centers)
        dist_g.create_dataset(
            "mass_fractions",
            data=np.asarray(dist.mass_fractions, dtype=np.float64),
        )

        if comp.mask is not None:
            _write_field(cg, "mask", comp.mask)


def _read_dust(group: "h5py.Group", model: Any) -> None:
    if "dust" not in group:
        return

    from .dust import Dust, DustDistribution, DustComponent, DustBin

    dg = group["dust"]

    dust = getattr(model, "dust", None)
    if dust is None:
        dust = Dust(model)
        model.dust = dust

    comps: list[DustComponent] = []
    comps_g = dg.get("components", None)
    if comps_g is None:
        return

    for name in sorted(comps_g.keys()):
        cg = comps_g[name]

        comp_meta = _json_loads(_as_str(cg.attrs["component_json"]))
        mode = str(comp_meta["mode"])
        dust_to_gas_ratio = float(comp_meta["dust_to_gas_ratio"])
        alpha = comp_meta.get("alpha", None)
        delta = comp_meta.get("delta", None)
        mmw = float(comp_meta.get("mean_molecular_weight", 2.3))
        species_base = str(comp_meta.get("species_base", "dust"))
        component_index = int(comp_meta.get("component_index", 0))

        dist_g = cg["distribution"]
        power_index = float(dist_g.attrs["power_index"])
        amin = _read_quantity(dist_g, "amin")
        amax = _read_quantity(dist_g, "amax")
        grain_density = _read_quantity(dist_g, "grain_density")

        bin_edges = _read_quantity(dist_g, "bin_edges")
        nbin = int(np.asarray(bin_edges.magnitude).size - 1)

        dist = DustDistribution(
            amin=amin,
            amax=amax,
            nbin=nbin,
            power_index=power_index,
            grain_density=grain_density,
        )

        dist.bin_edges = bin_edges
        dist.bin_centers = _read_quantity(dist_g, "bin_centers")
        dist.mass_fractions = np.asarray(dist_g["mass_fractions"][...], dtype=np.float64)

        mask = _read_field(cg, "mask") if "mask" in cg else None

        comp = DustComponent(
            distribution=dist,
            dust_to_gas_ratio=dust_to_gas_ratio,
            mode=mode,  # type: ignore[arg-type]
            mask=mask,
            alpha=alpha,
            delta=delta,
            mean_molecular_weight=mmw,
            species_base=species_base,
            component_index=component_index,
        )
        comps.append(comp)

    dust._components = comps
    dust._has_region_components = bool(dg.attrs.get("has_region_components", False))

    dust._global_bins = {}
    dust._bins = {}

    global_bin_idx = 0
    for comp_idx, comp in enumerate(dust._components):
        dist = comp.distribution
        for local_idx in range(dist.nbin):
            bin_name = f"bin_{global_bin_idx}"
            dust._global_bins[bin_name] = (comp_idx, local_idx)
            dust._bins[bin_name] = DustBin(
                parent_dust=dust,
                bin_index=global_bin_idx,
                size=dist.bin_centers[local_idx],
                size_min=dist.bin_edges[local_idx],
                size_max=dist.bin_edges[local_idx + 1],
                mass_fraction=float(dist.mass_fractions[local_idx]),
                density_material=dist.grain_density,
            )
            global_bin_idx += 1


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def save_model_hdf5(
    model: Any,
    path: Union[str, Path],
    *,
    include_disk: bool = True,
    include_dust: bool = True,
    overwrite: bool = True,
) -> Path:
    """Save a DiskBridge Model snapshot to HDF5."""

    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(str(p))

    if model.mesh is None:
        raise ValueError("Model has no mesh; cannot snapshot")
    if model.gas is None:
        raise ValueError("Model has no gas SubModel; cannot snapshot")

    with h5py.File(p, "w") as f:
        f.attrs["schema_version"] = int(SCHEMA_VERSION)

        meta = f.create_group("meta")
        meta.attrs["meta_json"] = _json_dumps(
            {
                "coord_system": model.coord_system,
                "file_units": model.file_units,
                "directory": model.directory,
                "n_file": model.n_file,
                "filename": model.filename,
                "variables": model.variables,
                "compile_options": model.compile_options,
                "macros": model.macros,
            }
        )

        _write_mesh(f.create_group("mesh"), model.mesh)

        gas_g = f.create_group("gas")
        fields_g = gas_g.create_group("fields")
        for name, field in model.gas.items():
            _write_field(fields_g, name, field)

        if include_disk and getattr(model, "disk", None) is not None:
            dg = f.create_group("disk")
            dg.attrs["parameters_json"] = _json_dumps(getattr(model.disk, "parameters", {}))

        if include_dust and getattr(model, "dust", None) is not None:
            _write_dust(f, model)

    logger.info("Saved HDF5 model snapshot -> %s", str(p))
    return p


def load_model_hdf5(path: Union[str, Path]) -> Any:
    """Load a DiskBridge Model snapshot from HDF5."""

    from .core import Model, SubModel

    p = Path(path)
    with h5py.File(p, "r") as f:
        ver = int(f.attrs.get("schema_version", 0))
        if ver != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported snapshot schema_version={ver} (expected {SCHEMA_VERSION})"
            )

        model = Model()

        meta = _json_loads(_as_str(f["meta"].attrs["meta_json"]))
        model.coord_system = meta.get("coord_system", None)
        model.file_units = meta.get("file_units", None)
        model.directory = meta.get("directory", None)
        model.n_file = meta.get("n_file", None)
        model.filename = meta.get("filename", None)
        model.variables = meta.get("variables", {})
        model.compile_options = meta.get("compile_options", {})
        model.macros = meta.get("macros", {})

        model.mesh = _read_mesh(f["mesh"])

        model.gas = SubModel(model)
        fields_g = f["gas"]["fields"]
        for name in fields_g.keys():
            field = _read_field(fields_g, str(name))
            model.gas.register(str(name), field)

        if "dust" in f:
            from .dust import Dust

            model.dust = Dust(model)
            _read_dust(f, model)

        if "disk" in f:
            from .disk import Disk

            params = _json_loads(_as_str(f["disk"].attrs.get("parameters_json", "{}")))
            model.disk = Disk(model, params)

        if model.gas is not None:
            model.gas.mesh = model.mesh
        if model.dust is not None:
            model.dust.mesh = model.mesh
        if model.disk is not None:
            model.disk.mesh = model.mesh

    logger.info("Loaded HDF5 model snapshot <- %s", str(p))
    return model
