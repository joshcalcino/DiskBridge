from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .mesh import Mesh
from diskbridge import units
from .._units import Quantity

@dataclass
class Field:
    """
    A physical quantity sampled on a Mesh.

    - data: ndarray whose shape/order matches axis_order
    - unit: string (interpreted by the units backend at higher levels)
    - quantity: semantic tag (e.g., 'gas_density', 'temperature')
    - axis_order: tuple of axis names in data order (e.g., ('r','phi') or ('theta','r','phi'))
    """

    data: Quantity
    mesh: Mesh
    quantity: str
    axis_order: Tuple[str, ...]

    def __post_init__(self):
        # Fields store Pint Quantities in `data`; units live on the Quantity itself.
        # For convenience only, expose a .unit reference if available.
        self.unit = getattr(self.data, "units", None)

    # Support multiplication by scalars or Pint units, returning a new Field
    def _scaled(self, factor):
        try:
            new_data = self.data * factor
        except Exception:
            new_data = self.data
        return Field(
            data=new_data,
            mesh=self.mesh,
            quantity=self.quantity,
            axis_order=self.axis_order,
        )

    def __mul__(self, other):
        return self._scaled(other)

    def __rmul__(self, other):
        return self._scaled(other)
