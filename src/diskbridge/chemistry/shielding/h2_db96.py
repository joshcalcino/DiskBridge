from __future__ import annotations

import numpy as np


def h2_self_shielding_db96(N_H2: np.ndarray, *, b5) -> np.ndarray:
    """Return the H2 self-shielding factor from the Draine & Bertoldi formula.

    Evaluates the analytic self-shielding function for Lyman-Werner photons as a
    function of H2 column density and Doppler broadening. ``b5`` and ``N_H2``
    broadcast against each other, so a scalar representative linewidth or an
    array of per-ray effective linewidths may be passed.

    Parameters
    ----------
    N_H2 : ndarray
        H2 column density [cm^-2]. Negative values are treated as zero.
    b5 : float or ndarray
        Doppler parameter in units of 1e5 cm/s, equal numerically to ``b`` in
        km/s. Must be finite and positive; broadcasts against ``N_H2``.

    Returns
    -------
    f_shield : ndarray
        Self-shielding factor in [0, 1] (1.0 = unshielded), broadcast to the
        common shape of ``N_H2`` and ``b5``.

    Raises
    ------
    ValueError
        If ``b5`` contains a non-finite or non-positive value.

    References
    ----------
    Draine & Bertoldi 1996, ApJ 468, 269; see the shielding guide.
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
