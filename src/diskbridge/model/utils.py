import numpy as np
from .mesh import Mesh
from .field import Field


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
