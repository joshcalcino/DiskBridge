"""Numerical calibration and display helpers for synthetic observations."""

# db-keywords: visualization, plotting, units, arrays
# db-role: helper
# db-scope: package
# db-purpose: Count-rate calibration, fixed logarithmic display, and effective PSF kernels.

from __future__ import annotations

import numpy as np


def log_stretch(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Apply the fixed logarithmic image mapping used by the HST notebook.

    Parameters
    ----------
    values : ndarray
        Linear image values, with any shape. Source values are not modified.
    vmin, vmax : float
        Finite lower and upper display limits in the same units as ``values``.
        The upper limit must exceed the lower limit.

    Returns
    -------
    ndarray
        Floating display values in [0, 1], with the input shape. The mapping is
        ``log10(1000 * clip((values-vmin)/(vmax-vmin), 0, 1) + 1) / log10(1001)``.
        NaN becomes black, positive infinity white, and negative infinity black.

    Raises
    ------
    ValueError
        If the display limits are nonfinite or not increasing.
    """
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("vmin and vmax must be finite with vmax > vmin")
    scaled = np.clip((np.asarray(values, dtype=float) - vmin) / (vmax - vmin), 0, 1)
    return np.nan_to_num(np.log10(1000 * scaled + 1) / np.log10(1001))


def jy_pixel_to_count_rate(
    values: np.ndarray,
    *,
    pixel_area_arcsec2: float,
    hst_pixel_area_arcsec2: float,
    pivot_angstrom: float,
    photflam: float,
) -> np.ndarray:
    """Convert mean band flux density into count rate per HST source pixel.

    Parameters
    ----------
    values : ndarray
        Finite mean band flux densities in Jy per input pixel, with any shape.
        The band mean must use the photon response ``T(lambda)/lambda`` for
        F_nu. Signed values are retained and the input is not modified.
    pixel_area_arcsec2 : float
        Positive input-pixel angular area in square arcseconds.
    hst_pixel_area_arcsec2 : float
        Positive original HST source-pixel area in square arcseconds. This is
        the normalization retained when count rates are later interpolated
        onto a common display grid.
    pivot_angstrom : float
        Positive filter pivot wavelength in angstrom, from ``PHOTPLAM``.
    photflam : float
        Positive inverse sensitivity from ``PHOTFLAM``, in
        erg s^-1 cm^-2 angstrom^-1 per electron/s.

    Returns
    -------
    ndarray
        Electrons/s per HST source pixel, with the input shape. The conversion
        uses surface brightness and the HST source-pixel area; it performs no
        spatial resampling, exposure multiplication, background subtraction,
        clipping, or noise addition.

    Raises
    ------
    ValueError
        If calibration values or pixel areas are nonfinite or nonpositive,
        or if an input flux density is nonfinite.

    Notes
    -----
    The conversion is ``values * (hst_area/input_area) * 1e-23 * c_A /
    (pivot_angstrom**2 * photflam)``, where ``c_A = 2.99792458e18`` angstrom/s.
    """
    parameters = {
        "pixel_area_arcsec2": float(pixel_area_arcsec2),
        "hst_pixel_area_arcsec2": float(hst_pixel_area_arcsec2),
        "pivot_angstrom": float(pivot_angstrom),
        "photflam": float(photflam),
    }
    for name, value in parameters.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    flux = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(flux)):
        raise ValueError("values must contain only finite flux densities")
    area_ratio = parameters["hst_pixel_area_arcsec2"] / parameters["pixel_area_arcsec2"]
    sensitivity = 1e-23 * 2.99792458e18 / (
        parameters["pivot_angstrom"] ** 2 * parameters["photflam"]
    )
    return flux * area_ratio * sensitivity


def moffat_kernel(
    *,
    alpha_arcsec: float,
    beta: float,
    pixel_scale_arcsec: float,
    radius_arcsec: float = 2.0,
) -> np.ndarray:
    """Sample and normalize a circular Moffat effective-PSF kernel.

    Parameters
    ----------
    alpha_arcsec : float
        Positive Moffat radial scale in arcseconds.
    beta : float
        Finite power-law index greater than one, so the untruncated profile
        has a finite integral.
    pixel_scale_arcsec : float
        Positive square sampling-pixel width in arcseconds.
    radius_arcsec : float, optional
        Positive circular support radius in arcseconds. The sampled kernel
        is truncated at this radius and normalized to unit sum.

    Returns
    -------
    ndarray
        Nonnegative kernel with odd square shape ``(2*n+1, 2*n+1)``, where
        ``n = ceil(radius_arcsec/pixel_scale_arcsec)``. Its central pixel is
        the profile peak. A unit-sum convolution preserves constant surface
        brightness away from boundaries and total flux on a padded image.

    Raises
    ------
    ValueError
        If scales or radius are nonfinite or nonpositive, or beta <= 1.

    Notes
    -----
    The profile is ``(1 + (r/alpha_arcsec)**2)**(-beta)``. It samples a smooth
    effective PSF that already includes the measured detector/drizzle core;
    no additional pixel-response integration or box kernel is applied.
    """
    alpha_arcsec = float(alpha_arcsec)
    beta = float(beta)
    pixel_scale_arcsec = float(pixel_scale_arcsec)
    radius_arcsec = float(radius_arcsec)
    for name, value in (
        ("alpha_arcsec", alpha_arcsec),
        ("pixel_scale_arcsec", pixel_scale_arcsec),
        ("radius_arcsec", radius_arcsec),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(beta) or beta <= 1:
        raise ValueError("beta must be finite and greater than one")
    half_size = int(np.ceil(radius_arcsec / pixel_scale_arcsec))
    offsets = np.arange(-half_size, half_size + 1) * pixel_scale_arcsec
    radius = np.hypot(offsets[:, None], offsets[None, :])
    with np.errstate(over="ignore"):
        kernel = (1 + (radius / alpha_arcsec) ** 2) ** (-beta)
    kernel[radius > radius_arcsec] = 0
    return kernel / kernel.sum()
