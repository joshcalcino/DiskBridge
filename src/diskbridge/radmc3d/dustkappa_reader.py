"""
Reader and UV-band averaging for RADMC-3D dustkappa opacity files.

This module reads ``dustkappa_*.inp`` files (format 1) written by
DiskBridge's :class:`DustOpacityCalculator` and computes band-averaged
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
    Parse a single ``dustkappa_*.inp`` file.
band_average_kext
    Compute band-averaged extinction opacity over a UV wavelength range.
load_kext_uv_for_bins
    Load band-averaged kext_uv for all dust bins in a directory.
"""

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


def read_dustkappa(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a RADMC-3D ``dustkappa_*.inp`` file (format 1).

    The file format written by DiskBridge is::

        # comment lines
        1                          <-- format number (must be 1)
        N                          <-- number of wavelength points
        lambda[um]  kabs  kscat  g <-- N data rows

    Parameters
    ----------
    path : str or Path
        Path to the ``dustkappa_*.inp`` file.

    Returns
    -------
    lam_um : ndarray, shape (N,)
        Wavelengths in microns.
    kabs : ndarray, shape (N,)
        Absorption opacity in cm^2/g (per gram of dust).
    kscat : ndarray, shape (N,)
        Scattering opacity in cm^2/g.
    g : ndarray, shape (N,)
        Asymmetry parameter (Henyey-Greenstein g).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the format number is not 1 or the file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dustkappa file not found: {path}")

    with open(path, "r") as f:
        # Skip comment lines, read format number
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fmt = int(stripped)
            break
        else:
            raise ValueError(f"Empty or comment-only dustkappa file: {path}")

        if fmt != 1:
            raise ValueError(
                f"Only dustkappa format 1 is supported, got {fmt} in {path}"
            )

        # Read number of wavelengths
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            nlam = int(stripped)
            break
        else:
            raise ValueError(f"Could not read nlam from {path}")

        # Read data rows
        lam_um = np.empty(nlam, dtype=np.float64)
        kabs = np.empty(nlam, dtype=np.float64)
        kscat = np.empty(nlam, dtype=np.float64)
        g = np.empty(nlam, dtype=np.float64)

        count = 0
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                raise ValueError(
                    f"Expected 4 columns in dustkappa data row, got {len(parts)} "
                    f"at line: {stripped!r}"
                )
            lam_um[count] = float(parts[0])
            kabs[count] = float(parts[1])
            kscat[count] = float(parts[2])
            g[count] = float(parts[3])
            count += 1
            if count >= nlam:
                break

        if count < nlam:
            raise ValueError(
                f"Expected {nlam} data rows but only found {count} in {path}"
            )

    logger.debug(
        "Read dustkappa file %s: %d wavelengths, %.3g-%.3g um",
        path.name, nlam, lam_um[0], lam_um[-1],
    )
    return lam_um, kabs, kscat, g


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
        Path to the ``dustkappa_*.inp`` file.
    uv_min_um : float
        Minimum UV wavelength in microns (e.g. 0.0912 for Lyman limit).
    uv_max_um : float
        Maximum UV wavelength in microns (e.g. 0.2 for FUV band).
    weighting : {"log", "energy", "photon"}, optional
        Averaging weight. ``"log"`` preserves the historical flat-in-log
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
    species_base: str,
    nbin: int,
    uv_min_um: float,
    uv_max_um: float,
    *,
    weighting: str = "log",
    reference_spectrum: str | None = None,
) -> Optional[np.ndarray]:
    """Load band-averaged UV extinction opacity for each dust bin.

    Looks for files named ``dustkappa_{species_base}{ibin}.inp`` for
    ``ibin`` in ``range(nbin)`` in the given directory.

    Parameters
    ----------
    radmc_inputs_dir : str or Path
        Directory containing the dustkappa files.
    species_base : str
        Species base name (e.g. ``"silicate"``), matching the naming convention
        used by :meth:`RadWriter.compute_and_write_dust_opacities`.
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
    kext_uv = np.empty(nbin, dtype=np.float64)

    for ibin in range(nbin):
        fname = f"dustkappa_{species_base}{ibin}.inp"
        fpath = radmc_inputs_dir / fname
        if not fpath.exists():
            logger.debug(
                "dustkappa file missing for bin %d: %s -- cannot use dustkappa mode",
                ibin, fpath,
            )
            return None
        kext_uv[ibin] = band_average_kext(
            fpath,
            uv_min_um,
            uv_max_um,
            weighting=weighting,
            reference_spectrum=reference_spectrum,
        )

    logger.info(
        "Loaded kext_uv for %d bins from %s: %s cm^2/g",
        nbin, radmc_inputs_dir, kext_uv,
    )
    return kext_uv
