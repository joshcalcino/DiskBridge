from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

from .._units import Quantity
from dataclasses import field


@dataclass(frozen=True)
class Field:
    """
    A physical quantity sampled on a Mesh.

    - quantity: semantic tag (e.g., 'gas_density', 'temperature')
    - data: Pint Quantity
    - axis_order: tuple of axis names in data order (e.g., ('r','phi') or ('theta','r','phi'))
    """

    quantity: str
    data: Quantity
    axis_order: Tuple[str, ...]
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def units(self):
        return getattr(self.data, "units")   

    # lightweight conveniences (read-only)
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self.data.magnitude).shape)

    @property
    def ndim(self) -> int:
        return np.asarray(self.data.magnitude).ndim

    def __repr__(self) -> str:
        return (f"Field('{self.quantity}', units='{self.units}', "
                f"axis_order={self.axis_order}, shape={self.shape}, "
                f"ndim={self.ndim})")

    def _scaled(self, factor):
        new_data = self.data * factor
        return Field(
            data=new_data,
            quantity=self.quantity,
            axis_order=self.axis_order,
            attrs=self.attrs,
        )
        
    def __mul__(self, other):
        return self._scaled(other)

    def __rmul__(self, other):
        return self._scaled(other)