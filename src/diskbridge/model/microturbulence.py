# db-keywords: config, units, model, mesh, field, arrays
# db-role: canonical
# db-scope: package
# db-purpose: Package module for config, units, model, mesh.

from __future__ import annotations

from typing import Any

import numpy as np

from diskbridge._units import Quantity
from diskbridge.model.field import Field


MICROTURBULENCE_FIELD = "microturbulence"
DISK_WEIGHT_FIELD = "disk_weight"
SPATIALLY_CONSTANT_ATTR = "spatially_constant"


def microturbulence_spatially_constant(model) -> bool:
    """Return whether the canonical microturbulence field is spatially constant."""
    if model.gas is None or MICROTURBULENCE_FIELD not in model.gas:
        raise KeyError("model.gas['microturbulence'] is missing")

    value = model.gas[MICROTURBULENCE_FIELD].attrs.get(SPATIALLY_CONSTANT_ATTR)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "model.gas['microturbulence'] is missing boolean "
            f"{SPATIALLY_CONSTANT_ATTR!r} metadata"
        )
    return bool(value)


def ensure_microturbulence_field(
    model,
    params_obj: Any,
    *,
    weight_field: str = DISK_WEIGHT_FIELD,
) -> bool:
    """Create the canonical microturbulence field from params when requested.

    ``params_obj.microturbulence`` may be either a scalar velocity or a
    two-element list ``[disk, ism]``. A scalar fills the whole grid. A list uses
    ``model.gas[weight_field]`` as the disk weight and ``1 - weight`` as the
    ISM weight. If the parameter is ``None``, no field is created.
    """
    if model.gas is None:
        raise ValueError("Model has no gas submodel")
    if MICROTURBULENCE_FIELD in model.gas:
        return True

    value = getattr(params_obj, MICROTURBULENCE_FIELD, None)
    if value is None:
        return False

    axis_order = model.mesh.axis_names()
    shape = model.mesh.shape

    def _as_velocity(q, label: str):
        try:
            return q.to("cm/s")
        except Exception as exc:
            raise ValueError(f"{label} must have velocity units") from exc

    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(
                "microturbulence must be a scalar velocity or a two-value "
                "list [disk, ism]"
            )
        if weight_field not in model.gas:
            raise KeyError(
                f"microturbulence=[disk, ism] requires model.gas[{weight_field!r}]"
            )
        disk_value = _as_velocity(value[0], "microturbulence disk value")
        ism_value = _as_velocity(value[1], "microturbulence ISM value")
        w_disk = model.gas[weight_field].data.to("dimensionless")
        w_disk_arr = np.clip(np.asarray(w_disk.magnitude, dtype=float), 0.0, 1.0)
        w_disk_q = Quantity(w_disk_arr, "dimensionless")
        data = w_disk_q * disk_value + (1.0 - w_disk_q) * ism_value
        attrs = {
            "mode": "disk_ism_weighted",
            SPATIALLY_CONSTANT_ATTR: False,
            "weight_field": str(weight_field),
            "disk_value": str(disk_value),
            "ism_value": str(ism_value),
        }
    else:
        velocity = _as_velocity(value, "microturbulence")
        data = Quantity(np.ones(shape, dtype=float), "dimensionless") * velocity
        attrs = {
            "mode": "constant",
            SPATIALLY_CONSTANT_ATTR: True,
            "value": str(velocity),
        }

    model.gas_register(
        MICROTURBULENCE_FIELD,
        Field(
            quantity=MICROTURBULENCE_FIELD,
            data=data,
            axis_order=axis_order,
            attrs=attrs,
        ),
    )
    return True
