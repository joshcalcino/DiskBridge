from __future__ import annotations

from ._units import units, Quantity, add_units, array_units, array_quantities, generate_array_code_units
from ._logging import logger_init as _logger_init
from .model.model import Model
from .model import load_model, puff_up_model, extend_disk_inwards
from . import _params as _params_module
from ._params import read_params


__version__ = '0.1.0'

# Add custom units from config before initializing params
add_units()

# Initialize params now that units are loaded
_params_module.params = read_params(None)
params = _params_module.params

_logger_init(__version__)

__all__ = [
    "units",
    "Quantity",
    "add_units",
    "array_units",
    "array_quantities",
    "generate_array_code_units",
    "read_params",
    "params",
    "Model",
    "load_model",
    "puff_up_model",
    "extend_disk_inwards",
    ]
