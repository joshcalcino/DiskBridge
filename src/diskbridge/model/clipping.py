# db-keywords: units, model, mesh, field, coordinates, serialization
# db-role: canonical
# db-scope: package
# db-purpose: Package module for units, model, mesh, field.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from diskbridge._units import Quantity

from .mesh import Axis, Mesh
from .field import Field

if TYPE_CHECKING:
    from .core import Model
    from .disk import Disk


@dataclass(frozen=True)
class ClipIndexer:
    axis_slices: Dict[str, slice]


def compute_clip_indexer(
    mesh: Mesh,
    bounds: Dict[str, Tuple[Optional[Quantity], Optional[Quantity]]],
) -> tuple[ClipIndexer, Dict[str, Axis]]:
    for axis_name in bounds:
        if axis_name not in mesh.axes:
            raise ValueError(
                f"Cannot clip axis '{axis_name}' for coord_system '{mesh.coord_system}'"
            )

    axis_slices: Dict[str, slice] = {}
    new_axes: Dict[str, Axis] = {}

    for axis_name, axis0 in mesh.axes.items():
        vmin, vmax = bounds.get(axis_name, (None, None))
        if vmin is None and vmax is None:
            new_axes[axis_name] = Axis(edges=axis0.edges, centers=axis0.centers)
            continue

        edges0 = mesh.edges(axis_name)
        centers0 = mesh.centers(axis_name)
        if edges0 is None or centers0 is None:
            raise ValueError(f"Mesh axis '{axis_name}' missing edges or centers")

        c_mag = np.asarray(centers0.magnitude, dtype=float)
        keep = np.ones_like(c_mag, dtype=bool)
        if vmin is not None:
            vmin_use = vmin.to(centers0.units)
            keep &= c_mag >= float(vmin_use.magnitude)
        if vmax is not None:
            vmax_use = vmax.to(centers0.units)
            keep &= c_mag <= float(vmax_use.magnitude)
        if not np.any(keep):
            raise ValueError(f"Clipping removed all cells along axis '{axis_name}'")

        idx = np.flatnonzero(keep)
        i0 = int(idx[0])
        i1 = int(idx[-1])
        axis_slices[axis_name] = slice(i0, i1 + 1)

        new_edges = edges0[i0 : i1 + 2]
        new_axes[axis_name] = Axis(edges=new_edges)

    return ClipIndexer(axis_slices=axis_slices), new_axes


def clip_model(
    model: "Model",
    *,
    r_min: Optional[Quantity] = None,
    r_max: Optional[Quantity] = None,
    theta_min: Optional[Quantity] = None,
    theta_max: Optional[Quantity] = None,
    phi_min: Optional[Quantity] = None,
    phi_max: Optional[Quantity] = None,
    x_min: Optional[Quantity] = None,
    x_max: Optional[Quantity] = None,
    y_min: Optional[Quantity] = None,
    y_max: Optional[Quantity] = None,
    z_min: Optional[Quantity] = None,
    z_max: Optional[Quantity] = None,
) -> "Model":
    from .core import Model, SubModel
    
    mesh0 = model.mesh
    if mesh0 is None:
        raise ValueError("Model has no mesh")

    bounds: Dict[str, tuple[Optional[Quantity], Optional[Quantity]]] = {
        "r": (r_min, r_max),
        "theta": (theta_min, theta_max),
        "phi": (phi_min, phi_max),
        "x": (x_min, x_max),
        "y": (y_min, y_max),
        "z": (z_min, z_max),
    }
    bounds = {
        k: (vmin, vmax)
        for k, (vmin, vmax) in bounds.items()
        if vmin is not None or vmax is not None
    }

    for axis_name in bounds:
        if axis_name not in mesh0.axes:
            raise ValueError(
                f"Cannot clip axis '{axis_name}' for coord_system '{mesh0.coord_system}'"
            )

    indexer, new_axes = compute_clip_indexer(mesh0, bounds)
    axis_slices = indexer.axis_slices

    mesh1 = Mesh(mesh0.coord_system, new_axes)

    def _slice_field(field0: Field) -> Field:
        slicer = []
        for ax in field0.axis_order:
            slicer.append(axis_slices.get(ax, slice(None)))
        mag0 = np.asarray(field0.data.magnitude)
        mag1 = mag0[tuple(slicer)]
        data1 = Quantity(mag1, field0.data.units)
        return Field(
            data=data1,
            quantity=field0.quantity,
            axis_order=field0.axis_order,
            attrs=field0.attrs,
        )

    new = Model()
    new.coord_system = mesh1.coord_system
    new.variables = dict(model.variables)
    new.compile_options = dict(model.compile_options)
    new.macros = dict(model.macros)
    new.mesh = mesh1
    new.file_units = model.file_units
    new.directory = model.directory
    new.n_file = model.n_file
    new.filename = model.filename

    for attr in ("length_scale", "mass_scale", "time_scale", "velocity_scale"):
        if hasattr(model, attr):
            setattr(new, attr, getattr(model, attr))

    new.gas = SubModel(new)
    if getattr(model.gas, "_lazy", None):
        for name in list(model.gas._lazy.keys()):
            _ = model.gas[name]
    for name, field0 in list(model.gas.items()):
        new.gas_register(name, _slice_field(field0))

    if getattr(model, "disk", None) is not None:
        from .disk import Disk
        new.disk = Disk(new, dict(model.disk.parameters))
        new.disk.is_disk_region = bool(getattr(model.disk, "is_disk_region", False))
        if model.disk.mask is not None:
            new.disk.mask = _slice_field(model.disk.mask)

    if getattr(model, "dust", None) is not None:
        from .dust import Dust, DustComponent, DustBin

        old_dust = model.dust
        new.dust = Dust(new)
        new_dust = new.dust

        new_dust.mask = _slice_field(old_dust.mask) if old_dust.mask is not None else None
        new_dust.is_disk_region = bool(getattr(old_dust, "is_disk_region", False))

        new_dust.distribution = old_dust.distribution
        new_dust.dust_to_gas_ratio = old_dust.dust_to_gas_ratio
        new_dust.mode = old_dust.mode
        new_dust.alpha = old_dust.alpha
        new_dust.delta = old_dust.delta
        new_dust.mean_molecular_weight = old_dust.mean_molecular_weight

        new_dust._has_region_components = bool(getattr(old_dust, "_has_region_components", False))
        new_dust._components = []
        for comp0 in getattr(old_dust, "_components", []):
            comp_mask = _slice_field(comp0.mask) if comp0.mask is not None else None
            new_dust._components.append(
                DustComponent(
                    distribution=comp0.distribution,
                    dust_to_gas_ratio=comp0.dust_to_gas_ratio,
                    mode=comp0.mode,
                    mask=comp_mask,
                    alpha=comp0.alpha,
                    delta=comp0.delta,
                    mean_molecular_weight=comp0.mean_molecular_weight,
                    species_base=comp0.species_base,
                    component_index=comp0.component_index,
                )
            )

        new_dust._global_bins = {}
        new_dust._bins = {}
        for comp_idx, comp in enumerate(new_dust._components):
            nbin = int(comp.distribution.nbin)
            for local_idx in range(nbin):
                global_bin_idx = len(new_dust._global_bins)
                bin_name = f"bin_{global_bin_idx}"
                new_dust._global_bins[bin_name] = (comp_idx, local_idx)
                new_dust._bins[bin_name] = DustBin(
                    parent_dust=new_dust,
                    bin_index=global_bin_idx,
                    size=comp.distribution.bin_centers[local_idx],
                    size_min=comp.distribution.bin_edges[local_idx],
                    size_max=comp.distribution.bin_edges[local_idx + 1],
                    mass_fraction=comp.distribution.mass_fractions[local_idx],
                    density_material=comp.grain_density,
                )

        new_dust._dust_fields = {}
        for name, field0 in getattr(old_dust, "_dust_fields", {}).items():
            new_dust._dust_fields[name] = _slice_field(field0)

    return new
