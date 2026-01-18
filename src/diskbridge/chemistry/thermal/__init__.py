"""Thermal balance calculations for gas temperature.

This subpackage provides modular thermal balance solvers with:
- Pluggable heating/cooling terms
- Carbon ionization closure
- Clean API following chemistry pattern

Examples
--------
>>> from diskbridge.chemistry.thermal import run_thermal
>>> 
>>> result = run_thermal(rad, model="thermal_balance")
>>> print(f"Gas temperature: {result.tgas.to('K')}")
"""

from diskbridge.chemistry.thermal.api import run_thermal
from diskbridge.chemistry.thermal.types import ThermalState, ThermalResult
from diskbridge.chemistry.thermal.registry import THERMAL_REGISTRY, get_thermal_model

__all__ = [
    "run_thermal",
    "ThermalState",
    "ThermalResult",
    "THERMAL_REGISTRY",
    "get_thermal_model",
]
