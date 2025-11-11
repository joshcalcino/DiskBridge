from __future__ import annotations

from pathlib import Path
from typing import Union

from ._units import units, Quantity, add_units, array_units, array_quantities, generate_array_code_units
from ._logging import logger_init as _logger_init
from .model.model import Model
from .model import load_model 

__version__ = '0.1.0'

add_units()
_logger_init(__version__)

__all__ = [
    "units",
    "Quantity",
    "add_units",
    "array_units",
    "array_quantities",
    "generate_array_code_units",
    "Model",
    "load_model"
    ]
