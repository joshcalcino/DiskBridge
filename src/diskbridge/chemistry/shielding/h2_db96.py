from __future__ import annotations

import numpy as np


def h2_self_shielding_db96(N_H2: np.ndarray, *, b5: float, alpha: float) -> np.ndarray:
    x = N_H2 / 5.0e14
    x = np.maximum(x, 0.0)

    term1 = (1.0 + x / b5) ** alpha
    term2 = np.exp(-5.0e-4 * np.sqrt(1.0 + x))
    f_shield = term1 * term2

    f_shield = np.clip(f_shield, 0.0, 1.0)
    return f_shield
