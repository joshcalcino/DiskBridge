# db-keywords: config, units, model, io, serialization, arrays
# db-role: helper
# db-scope: package
# db-purpose: Package module for config, units, model, io.

from __future__ import annotations

import sys
from types import ModuleType

from ._units import units, Quantity, add_units, array_units, array_quantities, generate_array_code_units
from ._logging import logger_init as _logger_init
from .model import Model
from .model import downsample_model, load_model, puff_up_model
from .serialization import jsonable
from .provenance import write_run_manifest, build_run_manifest
from . import _params as _params_module
from ._params import canonicalize_line_params, read_params


__version__ = '0.1.0'

# Add custom units from config before initializing params
add_units()

# Initialize params now that units are loaded
_params_module.params = read_params(None)
params = _params_module.params


class _DiskBridgeModule(ModuleType):
    """Module type that keeps ``params`` synchronized with ``_params``."""

    def __getattribute__(self, name):
        """Return a package attribute.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        object
            Attribute value.
        """
        if name == "params":
            return _params_module.params
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        """Set a package attribute.

        Parameters
        ----------
        name : str
            Attribute name.
        value : object
            Attribute value.
        """
        if name == "params":
            _params_module.params = value
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _DiskBridgeModule

_logger_init(__version__)

__all__ = [
    "units",
    "Quantity",
    "add_units",
    "array_units",
    "array_quantities",
    "generate_array_code_units",
    "read_params",
    "canonicalize_line_params",
    "params",
    "Model",
    "downsample_model",
    "load_model",
    "puff_up_model",
    "jsonable",
    "write_run_manifest",
    "build_run_manifest",
    ]
