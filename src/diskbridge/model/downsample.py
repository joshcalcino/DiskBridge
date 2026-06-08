# db-keywords: units, model, mesh, field, coordinates, serialization
# db-role: entrypoint
# db-scope: package
# db-purpose: Package module for units, model, mesh, field.

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import numpy as np

from diskbridge._units import Quantity

from .core import Model, SubModel
from .field import Field
from .mesh import Axis, Mesh
from .profiles import compute_cell_volumes
from .utils import field_data_as_order


def _axis_factors(mesh: Mesh, factor: int | Mapping[str, int]) -> dict[str, int]:
    if isinstance(factor, Mapping):
        factors = {axis: int(factor.get(axis, 1)) for axis in mesh.axis_names()}
    else:
        factors = {axis: int(factor) for axis in mesh.axis_names()}

    for axis, f in factors.items():
        if f < 1:
            raise ValueError(f"Downsample factor for axis {axis!r} must be >= 1")
    return factors


def _coarsened_edges(
    mesh: Mesh,
    factors: Mapping[str, int],
) -> tuple[dict[str, Quantity], dict[str, slice]]:
    edges_out: dict[str, Quantity] = {}
    retained_slices: dict[str, slice] = {}

    for axis in mesh.axis_names():
        edges = mesh.edges(axis)
        if edges is None:
            raise ValueError(f"Mesh axis {axis!r} has no edges")

        ncell = int(edges.size - 1)
        f = int(factors[axis])
        remainder = ncell % f
        nuse = ncell - remainder
        if nuse <= 0:
            raise ValueError(
                f"Axis {axis!r} has too few cells ({ncell}) for factor {f}"
            )

        edge_idx = np.arange(0, nuse + 1, f, dtype=int)
        edges_out[axis] = edges[edge_idx]
        retained_slices[axis] = slice(0, nuse)

    return edges_out, retained_slices


def _make_mesh_like(mesh: Mesh, edges: Mapping[str, Quantity]) -> Mesh:
    if mesh.coord_system == "spherical":
        return Mesh.spherical(
            r=Axis(edges=edges["r"]),
            theta=Axis(edges=edges["theta"]),
            phi=Axis(edges=edges["phi"]),
        )
    if mesh.coord_system == "polar":
        return Mesh.polar(
            r=Axis(edges=edges["r"]),
            phi=Axis(edges=edges["phi"]),
        )
    if mesh.coord_system == "cartesian":
        return Mesh.cartesian(
            x=Axis(edges=edges["x"]),
            y=Axis(edges=edges["y"]),
            z=Axis(edges=edges["z"]),
        )
    raise ValueError(f"Unsupported coordinate system: {mesh.coord_system!r}")


def _reshape_blocks(arr: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    shape: list[int] = []
    for n, f in zip(arr.shape, factors):
        if n % f:
            raise ValueError(f"Array shape {arr.shape} is not divisible by {factors}")
        shape.extend([n // f, f])
    return arr.reshape(tuple(shape))


def _block_sum(arr: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    blocked = _reshape_blocks(arr, factors)
    block_axes = tuple(range(1, 2 * arr.ndim, 2))
    return np.sum(blocked, axis=block_axes)


def _block_weighted_mean(
    arr: np.ndarray,
    weights: np.ndarray,
    factors: tuple[int, ...],
) -> np.ndarray:
    numerator = _block_sum(arr * weights, factors)
    denominator = _block_sum(weights, factors)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def _field_weight(
    *,
    name: str,
    volumes: np.ndarray,
    density: np.ndarray | None,
) -> np.ndarray:
    if name in {"vr", "vtheta", "vphi", "vx", "vy", "vz"} and density is not None:
        return volumes * np.maximum(density, 0.0)
    return volumes


def downsample_model(
    model: Model,
    factor: int | Mapping[str, int] = 2,
    *,
    fields: tuple[str, ...] | None = None,
) -> Model:
    """Coarsen a model by merging fixed-size blocks of neighboring cells.

    This helper does not interpolate. It merges complete blocks of neighboring
    cells. If an axis length is not divisible by ``factor``, incomplete leftover
    cells are dropped from the high-index edge of that axis before coarsening.
    For the spherical radial axis, this preserves the inner domain and drops the
    outermost leftover radial cells. On spherical meshes, scalar fields are
    volume-weighted. Velocity fields are density-weighted when a gas density
    field is available.

    Parameters
    ----------
    model
        Source model to coarsen.
    factor
        Integer factor for all axes, or a mapping from axis name to factor.
    fields
        Optional gas field names to downsample. Defaults to all registered gas
        fields.
    """
    if model.mesh is None:
        raise ValueError("Model has no mesh")
    if model.gas is None:
        raise ValueError("Model has no gas submodel")
    if model.mesh.coord_system != "spherical":
        raise ValueError(
            "downsample_model currently supports spherical meshes only"
        )

    source_mesh = model.mesh
    axis_names = source_mesh.axis_names()
    factors_by_axis = _axis_factors(source_mesh, factor)
    new_edges, retained_slices_by_axis = _coarsened_edges(
        source_mesh,
        factors_by_axis,
    )
    new_mesh = _make_mesh_like(source_mesh, new_edges)

    retained_indexer = tuple(retained_slices_by_axis[axis] for axis in axis_names)
    block_factors = tuple(factors_by_axis[axis] for axis in axis_names)

    tmp_edges = {
        axis: source_mesh.edges(axis)[0 : retained_slices_by_axis[axis].stop + 1]
        for axis in axis_names
    }
    tmp_model = Model()
    tmp_model.coord_system = model.coord_system
    tmp_model.mesh = _make_mesh_like(source_mesh, tmp_edges)

    # The temporary model is only used for its cell volumes over the retained
    # source-region cells.
    volumes = compute_cell_volumes(tmp_model)

    density_arr: np.ndarray | None = None
    if "density" in model.gas:
        density_q = field_data_as_order(model.gas["density"], axis_names)
        density_arr = np.asarray(density_q.to("g/cm^3").magnitude, dtype=float)[
            retained_indexer
        ]

    out = Model()
    out.coord_system = model.coord_system
    out.variables = deepcopy(model.variables)
    out.compile_options = deepcopy(model.compile_options)
    out.macros = deepcopy(model.macros)
    out.mesh = new_mesh
    out.file_units = model.file_units
    out.directory = model.directory
    out.n_file = model.n_file
    out.filename = model.filename
    out.gas = SubModel(out)

    from .dust import Dust

    out.dust = Dust(out)

    if model.disk is not None:
        from .disk import Disk

        out.disk = Disk(out, deepcopy(model.disk.parameters))

    names = tuple(model.gas.keys()) if fields is None else fields
    for name in names:
        if name not in model.gas:
            raise KeyError(f"Gas field {name!r} not found")

        field = model.gas[name]
        q = field_data_as_order(field, axis_names)
        arr = np.asarray(q.magnitude, dtype=float)[retained_indexer]
        weights = _field_weight(name=name, volumes=volumes, density=density_arr)
        reduced = _block_weighted_mean(arr, weights, block_factors)

        attrs = dict(field.attrs)
        attrs["downsampled_from"] = {
            "factor": {axis: int(factors_by_axis[axis]) for axis in axis_names},
            "source_shape": list(np.asarray(q.magnitude).shape),
            "retained_shape": [
                int(retained_slices_by_axis[axis].stop) for axis in axis_names
            ],
            "dropped_high_index_cells": {
                axis: int(
                    source_mesh.ncell(axis) - retained_slices_by_axis[axis].stop
                )
                for axis in axis_names
            },
        }
        out.gas_register(
            name,
            Field(
                quantity=field.quantity,
                data=Quantity(reduced, str(q.units)),
                axis_order=axis_names,
                attrs=attrs,
            ),
        )

    return out
