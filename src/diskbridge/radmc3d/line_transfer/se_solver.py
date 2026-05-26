"""Statistical-equilibrium solver for non-LTE populations under HEALPix
escape-probability closure.

For each cell:

    M f = b,    sum f = 1,    f >= 0,

where ``M`` is the rate matrix built from radiative transitions (with escape
probability ``beta`` and CMB background) plus collisions. The last row of
``M`` is replaced by the normalization constraint.

Compute-heavy assembly (alpha0, collision rates, rate-matrix construction) is
Numba-parallelized over cells. The linear solve uses
``np.linalg.solve`` on the stacked matrices.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from diskbridge._constants import C_LIGHT, H_PLANCK, K_B
from .molecular_rates import (
    MoleculeData,
)


@njit(parallel=True, cache=True)
def compute_line_center_opacity(
    fracpop: np.ndarray,        # (Ncand, nlev)
    n_species_cand: np.ndarray, # (Ncand,)
    a_line_cand: np.ndarray,    # (Ncand,)
    iup: np.ndarray,            # (nlin,) int64
    ilow: np.ndarray,           # (nlin,) int64
    aud: np.ndarray,            # (nlin,)
    freq_hz: np.ndarray,        # (nlin,)
    weight: np.ndarray,         # (nlev,)
) -> np.ndarray:
    """Line-center opacity per candidate cell per line, in cm^-1.

    Returns shape (Ncand, nlin).
    """
    n_cand = fracpop.shape[0]
    nlin = iup.shape[0]
    out = np.zeros((n_cand, nlin), dtype=np.float64)
    pref_const = C_LIGHT * C_LIGHT * C_LIGHT / (8.0 * (np.pi ** 1.5))
    for i in prange(n_cand):
        n_sp = n_species_cand[i]
        aL = a_line_cand[i]
        if n_sp <= 0.0 or aL <= 0.0:
            continue
        inv_aL = 1.0 / aL
        for m in range(nlin):
            u = iup[m]
            l = ilow[m]
            ratio = weight[u] / weight[l]
            delta = ratio * fracpop[i, l] - fracpop[i, u]
            nu = freq_hz[m]
            coeff = pref_const * aud[m] / (nu * nu * nu) * inv_aL
            out[i, m] = coeff * n_sp * delta
    return out


@njit(parallel=True, cache=True)
def compute_collisional_rates(
    Tgas_cand: np.ndarray,         # (Ncand,)
    collider_dens_cand: np.ndarray, # (Ncand, n_colliders)
    tgrids: np.ndarray,            # (n_colliders, max_nT)
    ntemps: np.ndarray,            # (n_colliders,) int64
    tables: np.ndarray,            # (n_colliders, max_nT, nlev, nlev)
    weight: np.ndarray,            # (nlev,)
    energy_erg: np.ndarray,        # (nlev,)
) -> np.ndarray:
    """Per-cell collisional rate matrix ``C[c, i, j]`` (s^-1).

    Downward rates from interpolated tables, upward rates by detailed balance.
    """
    n_cand = Tgas_cand.shape[0]
    n_col = tables.shape[0]
    nlev = weight.shape[0]
    C = np.zeros((n_cand, nlev, nlev), dtype=np.float64)
    for c in prange(n_cand):
        T = Tgas_cand[c]
        if T <= 0.0:
            continue
        kT = K_B * T
        for k in range(n_col):
            n_p = collider_dens_cand[c, k]
            if n_p <= 0.0:
                continue
            nt = ntemps[k]
            if T <= tgrids[k, 0]:
                it = 0
                eps = 0.0
            elif T >= tgrids[k, nt - 1]:
                it = nt - 2
                eps = 1.0
            else:
                it = 0
                for j in range(nt - 1):
                    if tgrids[k, j] <= T and T < tgrids[k, j + 1]:
                        it = j
                        break
                eps = (T - tgrids[k, it]) / (tgrids[k, it + 1] - tgrids[k, it])
            for u in range(1, nlev):
                for l in range(u):
                    gamma = (1.0 - eps) * tables[k, it, u, l] + eps * tables[k, it + 1, u, l]
                    C[c, u, l] += n_p * gamma
        for u in range(1, nlev):
            for l in range(u):
                C_ul = C[c, u, l]
                if C_ul <= 0.0:
                    continue
                ratio = weight[u] / weight[l]
                C[c, l, u] = C_ul * ratio * np.exp(-(energy_erg[u] - energy_erg[l]) / kT)
    return C


@njit(parallel=True, cache=True)
def build_rate_matrix_and_rhs(
    beta: np.ndarray,         # (Ncand, nlin)
    C: np.ndarray,            # (Ncand, nlev, nlev)
    iup: np.ndarray,          # (nlin,) int64
    ilow: np.ndarray,         # (nlin,) int64
    aud: np.ndarray,          # (nlin,)
    freq_hz: np.ndarray,      # (nlin,)
    weight: np.ndarray,       # (nlev,)
    tbg_K: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct ``M f = b`` per candidate cell.

    The last row of ``M`` is replaced with the normalization constraint and
    ``b`` is set to zero except the last entry which is 1.

    Returns
    -------
    M : ndarray, shape (Ncand, nlev, nlev)
    b : ndarray, shape (Ncand, nlev)
    """
    n_cand = beta.shape[0]
    nlin = beta.shape[1]
    nlev = C.shape[1]

    M = np.zeros((n_cand, nlev, nlev), dtype=np.float64)
    b = np.zeros((n_cand, nlev), dtype=np.float64)

    two_h_over_c2 = 2.0 * H_PLANCK / (C_LIGHT * C_LIGHT)

    for c in prange(n_cand):
        # First, fill the rate matrix M where M[i, j] = R_{j -> i} for i != j,
        # M[i, i] = -sum_{j != i} R_{i -> j}.
        # Radiative contributions (one per radiative transition).
        for m in range(nlin):
            u = iup[m]
            l = ilow[m]
            nu = freq_hz[m]
            be = beta[c, m]
            # CMB mean intensity, B_nu(T_bg)
            x = H_PLANCK * nu / (K_B * tbg_K)
            if x > 700.0:
                Jbg = 0.0
            else:
                Jbg = two_h_over_c2 * nu * nu * nu / (np.exp(x) - 1.0)
            # Einstein coefficients
            A_ul = aud[m]
            B_ul = A_ul * (C_LIGHT * C_LIGHT) / (2.0 * H_PLANCK * nu * nu * nu)
            B_lu = B_ul * weight[u] / weight[l]
            R_ul = A_ul * be + B_ul * be * Jbg
            R_lu = B_lu * be * Jbg
            # off-diagonals
            M[c, l, u] += R_ul   # rate into l from u
            M[c, u, l] += R_lu   # rate into u from l
            # accumulate sinks into the diagonal
            M[c, u, u] -= R_ul
            M[c, l, l] -= R_lu

        # Collisional contributions. C[i, j] is the rate from i to j (s^-1).
        # M[j, i] += C[i, j]   (rate into j from i)
        # M[i, i] -= sum_j C[i, j]   (total sink from i)
        for i in range(nlev):
            sink = 0.0
            for j in range(nlev):
                if i == j:
                    continue
                C_ij = C[c, i, j]
                sink += C_ij
                M[c, j, i] += C_ij
            M[c, i, i] -= sink

        # Replace last row with normalization.
        for j in range(nlev):
            M[c, nlev - 1, j] = 1.0
        b[c, nlev - 1] = 1.0

    return M, b


def solve_statistical_equilibrium(
    *,
    molecule: MoleculeData,
    Tgas_cand: np.ndarray,
    n_species_cand: np.ndarray,
    a_line_cand: np.ndarray,
    collider_dens_cand: np.ndarray,
    beta: np.ndarray,
    tbg_K: float,
    tgrids: np.ndarray,
    ntemps: np.ndarray,
    tables: np.ndarray,
    chunk_size: int | None = None,
    pop_floor: float = -1.0e-14,
) -> np.ndarray:
    """Solve SE for all candidate cells.

    Returns
    -------
    fracpop : ndarray, shape (Ncand, nlev)
    """
    n_cand = beta.shape[0]
    nlev = molecule.nlev
    if n_cand == 0:
        return np.zeros((0, nlev), dtype=np.float64)
    if chunk_size is None:
        chunk_size = n_cand
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = n_cand

    fracpop = np.empty((n_cand, nlev), dtype=np.float64)

    for start in range(0, n_cand, chunk_size):
        end = min(start + chunk_size, n_cand)
        sl = slice(start, end)
        C_chunk = compute_collisional_rates(
            np.ascontiguousarray(Tgas_cand[sl]),
            np.ascontiguousarray(collider_dens_cand[sl]),
            tgrids,
            ntemps,
            tables,
            molecule.weight,
            molecule.energy_erg,
        )
        M_chunk, b_chunk = build_rate_matrix_and_rhs(
            np.ascontiguousarray(beta[sl]),
            C_chunk,
            molecule.iup,
            molecule.ilow,
            molecule.aud,
            molecule.freq_hz,
            molecule.weight,
            float(tbg_K),
        )
        try:
            f_chunk = np.linalg.solve(M_chunk, b_chunk[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Statistical-equilibrium solve failed for chunk "
                f"[{start}:{end}]: {exc}"
            ) from exc

        # Clean roundoff and renormalize.
        min_val = float(np.min(f_chunk))
        if min_val < pop_floor:
            n_bad = int(np.count_nonzero(f_chunk < pop_floor))
            raise RuntimeError(
                f"Negative populations beyond roundoff tolerance: "
                f"{n_bad} entries below {pop_floor} (min {min_val:.3e})."
            )
        f_chunk = np.where(f_chunk < 0.0, 0.0, f_chunk)
        sums = f_chunk.sum(axis=1, keepdims=True)
        sums = np.where(sums <= 0.0, 1.0, sums)
        f_chunk = f_chunk / sums
        fracpop[sl] = f_chunk

    return fracpop


def population_change_diagnostics(
    f_new: np.ndarray,
    f_old: np.ndarray,
    *,
    level_floor: float = 1.0e-12,
    all_level_floor: float = 1.0e-20,
) -> dict[str, float]:
    """Measure population changes without letting empty high levels dominate.

    The stopping criterion is based on total population variation per cell plus
    the relative change in meaningfully populated levels. The all-level relative
    maximum is still reported because it is useful for diagnosing numerical
    floor noise, but it is not a physically useful convergence gate.
    """

    f_new = np.asarray(f_new, dtype=np.float64)
    f_old = np.asarray(f_old, dtype=np.float64)
    if f_new.shape != f_old.shape:
        raise ValueError(
            f"Population arrays must have matching shapes, got {f_new.shape} and {f_old.shape}"
        )
    if f_new.size == 0:
        return {
            "population_tv_change_max": 0.0,
            "population_abs_change_max": 0.0,
            "population_rel_important_max": 0.0,
            "population_rel_all_levels_max": 0.0,
            "population_convergence_error": 0.0,
        }

    absdiff = np.abs(f_new - f_old)
    tv = 0.5 * np.sum(absdiff, axis=1)
    scale = np.maximum(np.abs(f_new), np.abs(f_old))
    active = scale > float(level_floor)

    rel_important = (
        float(np.max(absdiff[active] / scale[active]))
        if np.any(active)
        else 0.0
    )
    denom_all = np.maximum(scale, float(all_level_floor))
    rel_all = float(np.max(absdiff / denom_all))
    tv_max = float(np.max(tv)) if tv.size else 0.0
    abs_max = float(np.max(absdiff))
    return {
        "population_tv_change_max": tv_max,
        "population_abs_change_max": abs_max,
        "population_rel_important_max": rel_important,
        "population_rel_all_levels_max": rel_all,
        "population_convergence_error": max(tv_max, rel_important),
    }


def convergence_error(
    f_new: np.ndarray,
    f_old: np.ndarray,
    *,
    level_floor: float = 1.0e-12,
) -> float:
    return population_change_diagnostics(
        f_new,
        f_old,
        level_floor=level_floor,
    )["population_convergence_error"]


__all__ = [
    "compute_line_center_opacity",
    "compute_collisional_rates",
    "build_rate_matrix_and_rhs",
    "solve_statistical_equilibrium",
    "convergence_error",
    "population_change_diagnostics",
]
