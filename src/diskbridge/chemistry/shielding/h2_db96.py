from __future__ import annotations

import numpy as np


def h2_self_shielding_db96(N_H2: np.ndarray, *, b5: float) -> np.ndarray:
    if float(b5) <= 0.0:
        raise ValueError("b5 must be > 0")

    x = N_H2 / 5.0e14
    x = np.maximum(x, 0.0)

    p1 = 0.965 / (1.0 + x / float(b5)) ** 2
    p2 = 0.035 / np.sqrt(1.0 + x) * np.exp(-8.5e-4 * np.sqrt(1.0 + x))
    f_shield = p1 + p2

    f_shield = np.clip(f_shield, 0.0, 1.0)
    return f_shield
