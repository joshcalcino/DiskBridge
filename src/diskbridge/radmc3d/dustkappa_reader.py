"""
Reader and UV-band averaging for native RADMC-3D opacity tables.

This module reads scalar ``dustkappa_*.inp`` and matrix
``dustkapscatmat_*.inp`` tables and computes band-averaged
extinction opacities for use in LOS UV optical depth calculations.

The band-averaged extinction opacity is defined as:

.. math::

    \\langle \\kappa_{\\rm ext} \\rangle
    = \\frac{\\int_{\\ln\\lambda_{\\min}}^{\\ln\\lambda_{\\max}}
             \\kappa_{\\rm ext}(\\lambda)\\,d\\ln\\lambda}
           {\\int_{\\ln\\lambda_{\\min}}^{\\ln\\lambda_{\\max}} d\\ln\\lambda}

where :math:`\\kappa_{\\rm ext} = \\kappa_{\\rm abs} + \\kappa_{\\rm scat}`
and the integration uses linear interpolation in :math:`\\log\\lambda`.

References
----------
- RADMC-3D manual, Sect. 7.4 (dustkappa file format)
- Weingartner & Draine (2001) for ISM dust opacity context

Functions
---------
read_dustkappa
    Read the opacity columns of one native scalar or matrix table.
read_dust_opacity_paths
    Resolve native table paths in the species order of ``dustopac.inp``.
band_average_kext
    Compute band-averaged extinction opacity over a UV wavelength range.
load_kext_uv_for_bins
    Load band-averaged kext_uv for all dust bins in a directory.
"""

# db-keywords: uv-products, config, units, radmc3d, io, serialization, paths
# db-role: canonical
# db-scope: package
# db-purpose: Native scalar/matrix opacity inventory and UV-band extinction.

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from diskbridge._logging import logger
from diskbridge.radmc3d.uv_products import C_CGS, H_CGS, default_isrf_path, load_draine_reference


# ---------------------------------------------------------------------------
# In-memory cache keyed by file, band, and weighting settings.
# ---------------------------------------------------------------------------
_KEXT_CACHE: dict[Tuple[str, float, float, float, str, str], float] = {}


def _opacity_lines(path: Path):
    """Yield native data rows, skipping RADMC comment and separator lines."""
    with path.open() as stream:
        for line in stream:
            value = line.strip()
            if value and not value.startswith(("#", ";", "!", "=", "-")):
                yield value


def read_dust_opacity_paths(radmc_inputs_dir: str | Path) -> list[Path]:
    """Return native opacity tables in the order declared by ``dustopac.inp``.

    Parameters
    ----------
    radmc_inputs_dir : str or pathlib.Path
        Directory containing ``dustopac.inp`` and its referenced native tables.

    Returns
    -------
    list of pathlib.Path
        One path per dust species, preserving the inventory order. Input style
        one selects a scalar table; style ten selects a scattering-matrix table.

    Raises
    ------
    FileNotFoundError
        If the inventory or any referenced table is missing.
    ValueError
        If the inventory format, species names, or input styles are invalid.
    """
    directory = Path(radmc_inputs_dir)
    inventory = directory / "dustopac.inp"
    # RADMC's list-directed inventory rows permit explanatory text after
    # their first value (as used by its official problem_setup examples).
    lines = [line.split()[0] for line in _opacity_lines(inventory)]
    try:
        fmt, nspecies = int(lines[0]), int(lines[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed opacity inventory: {inventory}") from exc
    if fmt != 2 or nspecies < 1 or len(lines) != 2 + 3 * nspecies:
        raise ValueError(f"Expected format-2 dust opacity inventory: {inventory}")
    paths = []
    for index in range(nspecies):
        style, thermal, name = lines[2+3*index:5+3*index]
        if style not in {"1", "10"} or thermal not in {"0", "1"}:
            raise ValueError(f"Unsupported dust opacity entry in {inventory}: {style}, {thermal}")
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid dust species name in {inventory}: {name!r}")
        prefix = "dustkappa" if style == "1" else "dustkapscatmat"
        path = directory / f"{prefix}_{name}.inp"
        if not path.is_file():
            raise FileNotFoundError(f"Referenced dust opacity table is missing: {path}")
        paths.append(path)
    return paths


def read_dustkappa(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read opacity columns from a native RADMC-3D scalar or matrix table.

    Parameters
    ----------
    path : str or pathlib.Path
        ``dustkappa_*.inp`` (native formats one, two, or three), or
        ``dustkapscatmat_*.inp`` (native format one). Matrix-table filenames
        identify their layout; only their wavelength/opacity columns are read.

    Returns
    -------
    lam_um, kabs, kscat, g : tuple of ndarray, each shape (N,)
        Wavelength in microns, absorption and scattering opacity in cm^2/g of
        dust, and asymmetry parameter. Missing scalar scattering columns are
        zero for absorption-only or isotropic native tables.

    Raises
    ------
    FileNotFoundError
        If the table is missing.
    ValueError
        If its native header, column count, or opacity values are invalid.
    """
    path = Path(path)
    rows = iter(_opacity_lines(path))
    try:
        fmt = int(next(rows).split()[0])
        nlam = int(next(rows).split()[0])
        matrix = path.name.startswith("dustkapscatmat_")
        if matrix:
            nang = int(next(rows).split()[0])
            if fmt != 1 or nang < 5:
                raise ValueError("Matrix opacity requires format one and at least five angles")
            ncolumns = 4
        else:
            if fmt not in (1, 2, 3):
                raise ValueError("Scalar opacity requires formats 1, 2, and 3")
            ncolumns = fmt + 1
        if nlam < 1:
            raise ValueError("Opacity wavelength count must be positive")
        table = np.zeros((nlam, 4), dtype=np.float64)
        for index in range(nlam):
            values = next(rows).split()
            if len(values) != ncolumns:
                raise ValueError(f"Expected exactly {ncolumns} opacity columns, got {len(values)}")
            table[index, :ncolumns] = [float(value) for value in values]
        if not matrix and next(rows, None) is not None:
            raise ValueError("Unexpected data beyond the declared scalar wavelength count")
    except (StopIteration, ValueError) as exc:
        raise ValueError(f"Malformed native opacity table {path}: {exc}") from exc
    if (not np.all(np.isfinite(table)) or np.any(table[:, 0] <= 0)
            or np.any(np.diff(table[:, 0]) <= 0) or np.any(table[:, 1:3] < 0)
            or np.any(np.abs(table[:, 3]) > 1)):
        raise ValueError(f"Invalid wavelength, opacity, or asymmetry values in {path}")
    return tuple(table[:, index] for index in range(4))


def band_average_kext(
    dustkappa_path: str | Path,
    uv_min_um: float,
    uv_max_um: float,
    *,
    weighting: str = "log",
    reference_spectrum: str | None = None,
) -> float:
    """Compute band-averaged extinction opacity over a UV wavelength range.

    Uses flat-in-log-lambda weighting:

    .. math::

        \\langle \\kappa_{\\rm ext} \\rangle
        = \\frac{\\int_{\\ln\\lambda_{\\min}}^{\\ln\\lambda_{\\max}}
                 \\kappa_{\\rm ext}(\\lambda)\\,d\\ln\\lambda}
               {\\ln(\\lambda_{\\max}/\\lambda_{\\min})}

    The extinction opacity is interpolated linearly in log(lambda) space
    from the tabulated values.

    Results are cached in memory keyed by ``(file_path, mtime, uv_min, uv_max)``.

    Parameters
    ----------
    dustkappa_path : str or Path
        Path to a native scalar or scattering-matrix opacity table.
    uv_min_um : float
        Minimum UV wavelength in microns (e.g. 0.0912 for Lyman limit).
    uv_max_um : float
        Maximum UV wavelength in microns (e.g. 0.2 for FUV band).
    weighting : {"log", "energy", "photon"}, optional
        Averaging weight. ``"log"`` uses a flat-in-log
        wavelength average. ``"energy"`` and ``"photon"`` require
        ``reference_spectrum="draine"``.
    reference_spectrum : {"draine", None}, optional
        Reference spectrum used for energy/photon weighting.

    Returns
    -------
    float
        Band-averaged extinction opacity kext_uv in cm^2/g (per gram of dust).

    Raises
    ------
    ValueError
        If the UV band does not overlap with the tabulated wavelength range.
    """
    dustkappa_path = Path(dustkappa_path)
    mtime = os.path.getmtime(dustkappa_path)
    weighting = str(weighting).lower()
    reference_key = "none" if reference_spectrum is None else str(reference_spectrum).lower()

    cache_key = (
        str(dustkappa_path),
        mtime,
        float(uv_min_um),
        float(uv_max_um),
        weighting,
        reference_key,
    )
    if cache_key in _KEXT_CACHE:
        return _KEXT_CACHE[cache_key]

    lam_um, kabs, kscat, _g = read_dustkappa(dustkappa_path)
    kext = kabs + kscat

    # Check overlap
    if uv_max_um < lam_um[0] or uv_min_um > lam_um[-1]:
        raise ValueError(
            f"UV band [{uv_min_um:.4g}, {uv_max_um:.4g}] um does not overlap "
            f"with dustkappa range [{lam_um[0]:.4g}, {lam_um[-1]:.4g}] um "
            f"in {dustkappa_path}"
        )

    # Clamp integration limits to tabulated range
    lam_lo = max(uv_min_um, lam_um[0])
    lam_hi = min(uv_max_um, lam_um[-1])

    if weighting == "log":
        # Integrate kext(ln lambda) d(ln lambda) via trapezoidal rule.
        ln_lo = np.log(lam_lo)
        ln_hi = np.log(lam_hi)
        n_interp = max(200, 2 * len(lam_um))
        ln_lam_grid = np.linspace(ln_lo, ln_hi, n_interp)
        lam_grid = np.exp(ln_lam_grid)
        kext_interp = np.interp(np.log(lam_grid), np.log(lam_um), kext)
        kext_avg = float(np.trapezoid(kext_interp, ln_lam_grid) / (ln_hi - ln_lo))
    elif weighting in {"energy", "photon"} and reference_key == "draine":
        lam_ref_nm, photon_ref_nm = load_draine_reference(default_isrf_path())
        lo_nm = lam_lo * 1.0e3
        hi_nm = lam_hi * 1.0e3
        in_ref = lam_ref_nm[(lam_ref_nm > lo_nm) & (lam_ref_nm < hi_nm)]
        in_kappa = (lam_um * 1.0e3)[(lam_um > lam_lo) & (lam_um < lam_hi)]
        lam_grid_nm = np.unique(
            np.concatenate(([lo_nm], in_ref, in_kappa, [hi_nm])).astype(np.float64)
        )
        kext_interp = np.interp(
            np.log(lam_grid_nm * 1.0e-3),
            np.log(lam_um),
            kext,
        )
        photon = np.interp(lam_grid_nm, lam_ref_nm, photon_ref_nm)
        if weighting == "energy":
            weights = photon * H_CGS * C_CGS / (lam_grid_nm * 1.0e-7)
        else:
            weights = photon
        denom = float(np.trapezoid(weights, lam_grid_nm))
        if denom <= 0.0:
            raise ValueError("Draine reference has non-positive weight over UV band")
        kext_avg = float(np.trapezoid(kext_interp * weights, lam_grid_nm) / denom)
    else:
        raise ValueError(
            "weighting must be 'log', or 'energy'/'photon' with "
            "reference_spectrum='draine'"
        )

    _KEXT_CACHE[cache_key] = kext_avg

    logger.debug(
        "Band-averaged kext_uv = %.4e cm^2/g over [%.4g, %.4g] um from %s",
        kext_avg, uv_min_um, uv_max_um, dustkappa_path.name,
    )
    return kext_avg


def load_kext_uv_for_bins(
    radmc_inputs_dir: str | Path,
    nbin: int,
    uv_min_um: float,
    uv_max_um: float,
    *,
    weighting: str = "log",
    reference_spectrum: str | None = None,
) -> Optional[np.ndarray]:
    """Load band-averaged UV extinction opacity for each dust bin.

    Read the ordered scalar/matrix inventory from ``dustopac.inp``; its
    species order must match the dust-density bin order.

    Parameters
    ----------
    radmc_inputs_dir : str or Path
        Directory containing the native opacity inventory and tables.
    nbin : int
        Number of dust bins.
    uv_min_um : float
        Minimum UV wavelength in microns.
    uv_max_um : float
        Maximum UV wavelength in microns.
    weighting : {"log", "energy", "photon"}, optional
        Band-average weighting passed to :func:`band_average_kext`.
    reference_spectrum : {"draine", None}, optional
        Reference spectrum used for weighted averages.

    Returns
    -------
    kext_uv : ndarray of shape (nbin,) or None
        Band-averaged extinction opacity per bin in cm^2/g, or ``None`` if
        any required file is missing.
    """
    radmc_inputs_dir = Path(radmc_inputs_dir)
    try:
        paths = read_dust_opacity_paths(radmc_inputs_dir)
    except FileNotFoundError:
        logger.debug("Required native opacity inventory/table missing in %s", radmc_inputs_dir)
        return None
    if len(paths) != nbin:
        raise ValueError(f"Opacity inventory has {len(paths)} species, expected {nbin} dust bins")
    kext_uv = np.asarray([
        band_average_kext(path, uv_min_um, uv_max_um,
                          weighting=weighting, reference_spectrum=reference_spectrum)
        for path in paths
    ], dtype=np.float64)

    logger.info(
        "Loaded kext_uv for %d bins from %s: %s cm^2/g",
        nbin, radmc_inputs_dir, kext_uv,
    )
    return kext_uv
