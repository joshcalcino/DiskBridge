# db-keywords: shielding, chemistry, arrays
# db-role: canonical
# db-scope: package
# db-purpose: Shared float64 coercion/validation for shielding array inputs.

from __future__ import annotations

import numpy as np


def _as_f64(name: str, x) -> np.ndarray:
    """Coerce ``x`` to a contiguous float64 array, rejecting unit-carrying inputs.

    This is the canonical strict coercion used by shielding column/ray code
    that feeds Numba kernels. It refuses Pint quantities (``magnitude``/``units``
    attributes) and object-dtype arrays so unitless, contiguous float64 inputs
    are guaranteed at kernel boundaries.
    """

    if hasattr(x, "magnitude") and hasattr(x, "units"):
        raise TypeError(f"{name} must be a float64 numpy array (no unit-carrying objects).")
    a = np.asarray(x, dtype=np.float64)
    if a.dtype == object:
        raise TypeError(f"{name} must be float64; got dtype=object")
    return np.ascontiguousarray(a, dtype=np.float64)
