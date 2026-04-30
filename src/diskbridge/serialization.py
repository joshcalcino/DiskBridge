"""Serialization helpers for DiskBridge metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """Convert nested values to JSON-compatible objects.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Any
        JSON-compatible representation of ``value``.
    """
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        arr = np.asarray(value.magnitude, dtype=float)
        return {
            "unit": str(value.units),
            "shape": list(arr.shape),
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
            "mean": float(np.nanmean(arr)) if arr.size else None,
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
