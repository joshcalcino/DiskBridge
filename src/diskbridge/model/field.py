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
        self.unit = units(self.data.unit)
