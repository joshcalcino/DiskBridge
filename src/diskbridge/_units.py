from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import pint

from ._config import read_config

# Global registry
units = pint.UnitRegistry(system='cgs')
Quantity = units.Quantity


# db-keywords: units, config
# db-role: canonical
def add_units(config: Union[str, Path, None] = None) -> None:
    """Add units to the unit registry from a config file.

    Parameters
    ----------
    config : optional
        Path to a TOML config. If None, use the package default.
    """
    conf = read_config(filename=config)
    units_conf = conf.get("units", {})

    # Custom unit definitions
    for unit, definition in units_conf.get("definitions", {}).items():
        units.define(f"{unit} = {definition}")

    # Dimensionful constants defined in config.toml under [units.constants]
    for const, definition in units_conf.get("constants", {}).items():
        units.define(f"{const} = {definition}")


def array_units(config: Union[str, Path, None] = None) -> Dict[str, str]:
    """Return a mapping of array names to default unit strings.

    This maps array dimensionalities to the default base units set in config.
    """
    conf = read_config(filename=config)
    d: Dict[str, str] = {}
    arr_dims = conf.get("arrays", {}).get("dimensions", {})
    defaults = conf.get("units", {}).get("defaults", {})
    for key, dim_str in arr_dims.items():
        dim = _convert_dim_string(dim_str)
        if dim == {"[angle]": 1.0}:
            d[key] = "radian"
        elif dim == {}:
            d[key] = "dimensionless"
        else:
            # choose the first default whose dimensionality matches
            for _, default_unit in defaults.items():
                if _dimensionality_comparison(dict(units(default_unit).dimensionality), dim):
                    d[key] = default_unit
                    break
    return d


def array_quantities(config: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Return a mapping of array names to dimensionality dicts."""
    conf = read_config(filename=config)
    arrays = conf.get("arrays", {}).get("dimensions", {})
    dim: Dict[str, Any] = {}
    for key, val in arrays.items():
        dim[key] = _convert_dim_string(val)
    return dim


def generate_array_code_units(code_units: Dict[str, Any]) -> Dict[str, Any]:
    """Generate array code units dictionary as pint Quantity units.

    Parameters
    ----------
    code_units : mapping like {'mass': Unit, 'length': Unit, 'time': Unit, ...}
    """
    _units: Dict[str, Any] = {}
    _array_quantities = array_quantities()
    for arr, unit in _array_quantities.items():
        _units[arr] = _get_code_unit(unit, code_units)
    return _units


# -----------------
# helpers
# -----------------

def _convert_dim_string(string: str):
    if string == "":
        return {}
    list_dims = [dim.strip() for dim in string.split(",")]
    keys = []
    vals = []
    for dim in list_dims:
        key, val = dim.split(": ")
        val_f = float(val)
        keys.append(key)
        vals.append(val_f)
    return {"[" + key + "]": val for key, val in zip(keys, vals)}


def _get_code_unit(dim_dict, code_units):
    unit = 1.0 * units["dimensionless"]
    for d in dim_dict:
        if d == "[angle]":
            unit *= units["radian"]
        else:
            base = code_units[d[1:-1]]  # strip [..]
            unit *= base ** dim_dict[d]
    return unit


def _dimensionality_comparison(dim1, dim2):
    _dim1 = {key: float(val) for key, val in dim1.items()}
    _dim2 = {key: float(val) for key, val in dim2.items()}
    return _dim1 == _dim2
