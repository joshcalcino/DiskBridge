# db-keywords: units, model, mesh, field, arrays, plotting
# db-role: canonical
# db-scope: package
# db-purpose: Package module for units, model, mesh, field.

import numpy as np
from .mesh import Mesh
from .field import Field
from diskbridge._units import Quantity


def validate_field_against_mesh(field: Field, mesh: Mesh) -> None:
    """Validate that a field's shape matches the mesh dimensions."""
    names = set(mesh.axis_names())
    missing = [a for a in field.axis_order if a not in names]
    if missing:
        raise ValueError(f"{field.quantity}: axis {missing} not in mesh axes {mesh.axis_names()}")
    expected = tuple(mesh.ncell(a) for a in field.axis_order)
    actual = tuple(np.asarray(field.data.magnitude).shape)
    if actual != expected:
        raise ValueError(
            f"{field.quantity}: data shape {actual} != expected {expected} for {field.axis_order}"
        )


def transpose_to_axis_order(
    arr: np.ndarray,
    from_order: tuple[str, ...],
    to_order: tuple[str, ...],
) -> np.ndarray:
    if from_order == to_order:
        return arr
    if set(from_order) != set(to_order):
        raise ValueError(f"Cannot reorder {from_order} -> {to_order}: different axis sets")
    perm = tuple(from_order.index(ax) for ax in to_order)
    return np.transpose(arr, perm)


def field_data_as_order(field: Field, to_order: tuple[str, ...]) -> Quantity:
    data = field.data
    arr = np.asarray(data.magnitude)
    out = transpose_to_axis_order(arr, field.axis_order, to_order)
    return Quantity(out, str(data.units))
