from __future__ import annotations

import numpy as np


def h2_self_shielding_db96(N_H2: np.ndarray, *, b5) -> np.ndarray:
    """Draine & Bertoldi H2 shielding for scalar or ray-wise Doppler b.

    ``b5`` is the Doppler parameter in units of 1e5 cm/s. Numerically this is
    equal to ``b`` in km/s, so callers may pass either a scalar representative
    linewidth or an array of per-ray effective linewidths.
    """
    b5_arr = np.asarray(b5, dtype=float)
    bad = (~np.isfinite(b5_arr)) | (b5_arr <= 0.0)
    if np.any(bad):
        raise ValueError("b5 must contain only finite positive values")

    x = np.maximum(np.asarray(N_H2, dtype=float) / 5.0e14, 0.0)
    x, b5_arr = np.broadcast_arrays(x, b5_arr)

    p1 = 0.965 / (1.0 + x / b5_arr) ** 2
    p2 = 0.035 / np.sqrt(1.0 + x) * np.exp(-8.5e-4 * np.sqrt(1.0 + x))
    f_shield = p1 + p2

    f_shield = np.clip(f_shield, 0.0, 1.0)
    return f_shield
