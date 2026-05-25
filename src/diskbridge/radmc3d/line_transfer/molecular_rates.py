"""LAMDA parser and collision-rate helpers for external-population non-LTE.

This module provides a clean, 3D-native, Numba-friendly description of one
molecular species:

- Energy levels and statistical weights (CGS, 0-indexed).
- Radiative transitions (Einstein A coefficients, frequencies).
- Collision rate tables per partner (downward rates only).

Collision blocks are read in the order they appear in the molecule file. The
expected order is enforced upstream by
``diskbridge.radmc3d.colliders.install_validated_molecule_file``.

Reference: ``other_codes/lines.py:radmc3dMolecule`` is consulted for formula
shape only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

from diskbridge.radmc3d.colliders import LAMDA_COLLIDER_ID_TO_NAME


_H_CGS = 6.62606957e-27           # erg s
_C_CGS = 2.99792458e10            # cm s^-1
_KB_CGS = 1.380648813e-16         # erg K^-1
_HC_CGS = _H_CGS * _C_CGS         # erg cm
_M_P_CGS = 1.6726218e-24          # g (proton mass)


@dataclass(frozen=True)
class MoleculeData:
    """Numba-friendly molecular data for one species.

    Attributes
    ----------
    name : str
        Species name from the LAMDA file (e.g. ``"co"``).
    molweight : float
        Molecular weight in proton-mass units.
    nlev : int
        Number of energy levels.
    nlin : int
        Number of radiative transitions.
    energy_erg : ndarray, shape (nlev,)
        Level energies in erg.
    weight : ndarray, shape (nlev,)
        Statistical weights.
    iup, ilow : ndarray, shape (nlin,), int64
        Upper / lower level indices (0-indexed).
    aud : ndarray, shape (nlin,), float64
        Einstein A coefficients in s^-1.
    freq_hz : ndarray, shape (nlin,), float64
        Transition rest frequencies in Hz.
    collider_names : list[str]
        Collider labels in file order (e.g. ``["p-h2", "o-h2"]``).
    collider_tgrid_K : list[np.ndarray]
        Per-collider temperature grids in K.
    collider_down_rates : list[np.ndarray]
        Per-collider downward collision-rate tables ``gamma[t, u, l]`` in
        cm^3 s^-1, with ``u > l``. Upper-triangle entries are zero.
    """

    name: str
    molweight: float
    nlev: int
    nlin: int
    energy_erg: np.ndarray
    weight: np.ndarray
    iup: np.ndarray
    ilow: np.ndarray
    aud: np.ndarray
    freq_hz: np.ndarray
    collider_names: tuple[str, ...]
    collider_tgrid_K: tuple[np.ndarray, ...]
    collider_down_rates: tuple[np.ndarray, ...]

    @property
    def m_mol_g(self) -> float:
        return float(self.molweight) * _M_P_CGS

    def collider_temperature_bounds(self) -> list[tuple[float, float]]:
        return [(float(t[0]), float(t[-1])) for t in self.collider_tgrid_K]


def parse_lamda_molecule_file(path: str | Path) -> MoleculeData:
    """Parse a Leiden LAMDA molecule file into a ``MoleculeData``.

    Collider blocks are read in file order. Use
    ``install_validated_molecule_file`` upstream to guarantee the order matches
    the strict GOW17/LAMDA policy.
    """

    path = Path(path)
    with path.open("r") as f:
        _ = f.readline()  # comment
        name = f.readline().split()[0]
        _ = f.readline()
        molweight = float(f.readline().strip())
        _ = f.readline()
        nlev = int(f.readline().strip())
        _ = f.readline()
        energy_cm = np.zeros(nlev, dtype=np.float64)
        weight = np.zeros(nlev, dtype=np.float64)
        for i in range(nlev):
            parts = f.readline().split()
            energy_cm[i] = float(parts[1])
            weight[i] = float(parts[2])
            if i > 0 and energy_cm[i] < energy_cm[i - 1] - 1e-12:
                raise ValueError(
                    f"Energy levels in {path} are not monotonically increasing"
                )
        energy_erg = energy_cm * _HC_CGS

        _ = f.readline()
        nlin = int(f.readline().strip())
        _ = f.readline()
        iup = np.zeros(nlin, dtype=np.int64)
        ilow = np.zeros(nlin, dtype=np.int64)
        aud = np.zeros(nlin, dtype=np.float64)
        freq_hz = np.zeros(nlin, dtype=np.float64)
        for i in range(nlin):
            parts = f.readline().split()
            iup[i] = int(parts[1]) - 1
            ilow[i] = int(parts[2]) - 1
            aud[i] = float(parts[3])
            freq_hz[i] = float(parts[4]) * 1.0e9

        collider_names: list[str] = []
        collider_tgrid_K: list[np.ndarray] = []
        collider_down_rates: list[np.ndarray] = []

        _ = f.readline()
        line = f.readline()
        if line == "":
            ncolp = 0
        else:
            ncolp = int(line.strip())
        for _kp in range(ncolp):
            _ = f.readline()  # comment
            id_line = f.readline().strip().split()
            collider_id = int(id_line[0])
            try:
                cname = LAMDA_COLLIDER_ID_TO_NAME[collider_id]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown LAMDA collider id {collider_id} in {path}"
                ) from exc
            collider_names.append(cname)

            _ = f.readline()
            ncolr = int(f.readline().strip())
            _ = f.readline()
            ntemp = int(f.readline().strip())
            _ = f.readline()
            tgrid = np.asarray(f.readline().split(), dtype=np.float64)
            if tgrid.size != ntemp:
                raise ValueError(
                    f"Collider {cname} in {path}: temperature grid length "
                    f"{tgrid.size} != ntemp {ntemp}"
                )
            collider_tgrid_K.append(tgrid)

            _ = f.readline()
            colr = np.zeros((ntemp, nlev, nlev), dtype=np.float64)
            for _r in range(ncolr):
                parts = f.readline().split()
                u = int(parts[1]) - 1
                l = int(parts[2]) - 1
                values = np.asarray(parts[3:3 + ntemp], dtype=np.float64)
                colr[:, u, l] = values
            collider_down_rates.append(colr)

    return MoleculeData(
        name=name,
        molweight=molweight,
        nlev=int(nlev),
        nlin=int(nlin),
        energy_erg=energy_erg,
        weight=weight,
        iup=iup,
        ilow=ilow,
        aud=aud,
        freq_hz=freq_hz,
        collider_names=tuple(collider_names),
        collider_tgrid_K=tuple(collider_tgrid_K),
        collider_down_rates=tuple(collider_down_rates),
    )


@njit(cache=True)
def _interp_table(t: float, tgrid: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Linear interpolation of a (nT, nlev, nlev) table at temperature t.

    Returns a fresh (nlev, nlev) f64 array.
    """
    nt = tgrid.shape[0]
    nlev = table.shape[1]
    out = np.zeros((nlev, nlev), dtype=np.float64)
    if t <= tgrid[0]:
        for u in range(nlev):
            for l in range(nlev):
                out[u, l] = table[0, u, l]
        return out
    if t >= tgrid[nt - 1]:
        for u in range(nlev):
            for l in range(nlev):
                out[u, l] = table[nt - 1, u, l]
        return out
    it = 0
    for k in range(nt - 1):
        if tgrid[k] <= t and t < tgrid[k + 1]:
            it = k
            break
    eps = (t - tgrid[it]) / (tgrid[it + 1] - tgrid[it])
    for u in range(nlev):
        for l in range(nlev):
            out[u, l] = (1.0 - eps) * table[it, u, l] + eps * table[it + 1, u, l]
    return out


def stack_collider_tables(
    molecule: MoleculeData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack collider tables into Numba-friendly arrays.

    Returns
    -------
    tgrids : ndarray, shape (n_colliders, max_ntemp)
        Temperature grids, right-padded with the last value.
    ntemps : ndarray, shape (n_colliders,), int64
        Real grid lengths.
    tables : ndarray, shape (n_colliders, max_ntemp, nlev, nlev)
        Downward rate tables, padded along the temperature axis with the last
        slab so that interpolation past the real grid is well-defined.
    """
    nc = len(molecule.collider_names)
    if nc == 0:
        return (
            np.zeros((0, 1), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 1, molecule.nlev, molecule.nlev), dtype=np.float64),
        )
    ntemps = np.asarray(
        [t.size for t in molecule.collider_tgrid_K], dtype=np.int64
    )
    max_nt = int(ntemps.max())
    tgrids = np.zeros((nc, max_nt), dtype=np.float64)
    tables = np.zeros((nc, max_nt, molecule.nlev, molecule.nlev), dtype=np.float64)
    for k in range(nc):
        nt = int(ntemps[k])
        tgrids[k, :nt] = molecule.collider_tgrid_K[k]
        tgrids[k, nt:] = molecule.collider_tgrid_K[k][-1]
        tables[k, :nt, :, :] = molecule.collider_down_rates[k]
        tables[k, nt:, :, :] = molecule.collider_down_rates[k][-1]
    return tgrids, ntemps, tables


def lte_populations(molecule: MoleculeData, Tgas: np.ndarray) -> np.ndarray:
    """Boltzmann fractional populations at gas temperature ``Tgas``.

    Parameters
    ----------
    molecule : MoleculeData
    Tgas : ndarray
        Gas temperature in K. Any shape.

    Returns
    -------
    f : ndarray, shape Tgas.shape + (nlev,)
        Fractional populations.
    """
    T = np.asarray(Tgas, dtype=np.float64)
    out_shape = T.shape + (molecule.nlev,)
    T_flat = T.reshape(-1)
    f_flat = _lte_populations_flat(
        T_flat,
        molecule.energy_erg.astype(np.float64),
        molecule.weight.astype(np.float64),
    )
    return f_flat.reshape(out_shape)


@njit(parallel=True, cache=True)
def _lte_populations_flat(
    T: np.ndarray,
    energy_erg: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    n = T.shape[0]
    nlev = energy_erg.shape[0]
    out = np.zeros((n, nlev), dtype=np.float64)
    for i in prange(n):
        if T[i] <= 0.0:
            out[i, 0] = 1.0
            continue
        kT = _KB_CGS * T[i]
        e0 = energy_erg[0]
        z = 0.0
        for k in range(nlev):
            v = weight[k] * np.exp(-(energy_erg[k] - e0) / kT)
            out[i, k] = v
            z += v
        if z > 0.0:
            inv = 1.0 / z
            for k in range(nlev):
                out[i, k] *= inv
        else:
            out[i, 0] = 1.0
    return out


def assert_temperature_in_collision_range(
    molecule: MoleculeData,
    Tgas: np.ndarray,
    candidate_mask: np.ndarray,
) -> None:
    """Strict temperature-bound policy.

    Raises if any candidate cell's temperature lies outside the tabulated
    temperature range of any required collider.
    """
    T = np.asarray(Tgas, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if T.shape != mask.shape:
        raise ValueError(
            f"Tgas shape {T.shape} does not match candidate_mask shape {mask.shape}"
        )
    if not np.any(mask):
        return
    T_cand = T[mask]
    for name, tgrid in zip(molecule.collider_names, molecule.collider_tgrid_K):
        lo, hi = float(tgrid[0]), float(tgrid[-1])
        below = int(np.count_nonzero(T_cand < lo))
        above = int(np.count_nonzero(T_cand > hi))
        if below or above:
            raise ValueError(
                f"Temperature out of collision-rate range for collider {name!r} "
                f"({lo:.3f} K - {hi:.3f} K): "
                f"{below} candidate cells below, {above} cells above; "
                f"T range in candidates is "
                f"[{float(T_cand.min()):.3f}, {float(T_cand.max()):.3f}] K."
            )


__all__ = [
    "MoleculeData",
    "parse_lamda_molecule_file",
    "stack_collider_tables",
    "lte_populations",
    "assert_temperature_in_collision_range",
    "_KB_CGS",
    "_H_CGS",
    "_C_CGS",
    "_HC_CGS",
    "_M_P_CGS",
]
