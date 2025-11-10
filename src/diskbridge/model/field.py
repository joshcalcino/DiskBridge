from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .mesh import Mesh


@dataclass
class Field:
    """
    A physical quantity sampled on a Mesh.

    - data: ndarray whose shape/order matches axis_order
    - unit: string (interpreted by the units backend at higher levels)
    - quantity: semantic tag (e.g., 'gas_density', 'temperature')
    - axis_order: tuple of axis names in data order (e.g., ('r','phi') or ('theta','r','phi'))
    """

    data: np.ndarray
    mesh: Mesh
    unit: str
    quantity: str
    axis_order: Tuple[str, ...]
