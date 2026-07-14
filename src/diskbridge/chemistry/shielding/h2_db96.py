# db-keywords: shielding, co-shielding, config, units, chemistry, arrays
# db-role: canonical
# db-scope: package
# db-purpose: Package module for shielding, co-shielding, config, units.

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _h2_self_shielding_value(column: float, b5: float) -> float:
    """Evaluate one Draine--Bertoldi shielding factor."""
    x = column / 5.0e14
    if x < 0.0:
        x = 0.0
    root = np.sqrt(1.0 + x)
    p1 = 0.965 / ((1.0 + x / b5) ** 2)
    p2 = 0.035 / root * np.exp(-8.5e-4 * root)
    value = p1 + p2
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@njit(cache=True, parallel=True)
def _h2_self_shielding_1d_parallel(
    column: np.ndarray,
    b5: np.ndarray,
) -> np.ndarray:
    out = np.empty(column.size, dtype=np.float64)
    for i in prange(column.size):
        out[i] = _h2_self_shielding_value(column[i], b5[i])
    return out


@njit(cache=True, parallel=True)
def _h2_self_shielding_1d_scalar_b_parallel(
    column: np.ndarray,
    b5: float,
) -> np.ndarray:
    out = np.empty(column.size, dtype=np.float64)
    for i in prange(column.size):
        out[i] = _h2_self_shielding_value(column[i], b5)
    return out


@njit(cache=True, parallel=True)
def _h2_self_shielding_2d_parallel(
    column: np.ndarray,
    b5: np.ndarray,
) -> np.ndarray:
    n_rows, n_columns = column.shape
    out = np.empty((n_rows, n_columns), dtype=np.float64)
    for i in prange(n_rows):
        for j in range(n_columns):
            out[i, j] = _h2_self_shielding_value(column[i, j], b5[i, j])
    return out


@njit(cache=True, parallel=True)
def _h2_self_shielding_2d_scalar_b_parallel(
    column: np.ndarray,
    b5: float,
) -> np.ndarray:
    n_rows, n_columns = column.shape
    out = np.empty((n_rows, n_columns), dtype=np.float64)
    for i in prange(n_rows):
        for j in range(n_columns):
            out[i, j] = _h2_self_shielding_value(column[i, j], b5)
    return out


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
    b5_arr = np.asarray(b5, dtype=np.float64)
    bad = (~np.isfinite(b5_arr)) | (b5_arr <= 0.0)
    if np.any(bad):
        raise ValueError("b5 must contain only finite positive values")

    column_arr = np.asarray(N_H2, dtype=np.float64)
    if b5_arr.ndim == 0:
        b5_scalar = float(b5_arr)
        if column_arr.ndim == 2:
            return _h2_self_shielding_2d_scalar_b_parallel(
                column_arr,
                b5_scalar,
            )
        shape = column_arr.shape
        column_flat = np.ascontiguousarray(
            column_arr.reshape(-1),
            dtype=np.float64,
        )
        return _h2_self_shielding_1d_scalar_b_parallel(
            column_flat,
            b5_scalar,
        ).reshape(shape)

    column, b5_arr = np.broadcast_arrays(
        column_arr,
        b5_arr,
    )
    if column.ndim == 2:
        return _h2_self_shielding_2d_parallel(column, b5_arr)

    shape = column.shape
    column_flat = np.ascontiguousarray(column.reshape(-1), dtype=np.float64)
    b5_flat = np.ascontiguousarray(b5_arr.reshape(-1), dtype=np.float64)
    return _h2_self_shielding_1d_parallel(column_flat, b5_flat).reshape(shape)
