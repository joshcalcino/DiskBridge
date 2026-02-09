"""
Dust UV optical depth computation for HEALPix directional shielding.

This module provides:

- **resolve_uv_tau_mode**: decide whether to use dustkappa-based extinction
  or the simple ``SIGMA_DUST * N_H`` fallback.
- **compute_tau_uv_from_dust_columns**: compute per-ray UV optical depth
  from dust mass columns and band-averaged extinction opacities.
- **prepare_dust_density_fields**: extract per-bin dust density arrays from
  a DiskBridge model for ray integration.

The dustkappa-based mode uses the *actual* absorption + scattering opacities
written to ``dustkappa_*.inp`` files by DiskBridge, giving:

.. math::

    \\tau_{\\rm UV}(k) = \\sum_{b} \\kappa^{(b)}_{\\rm ext,UV}
                         \\,\\Sigma^{(b)}_{\\rm dust}(k)

where :math:`\\Sigma^{(b)}_{\\rm dust}(k) = \\int \\rho^{(b)}_{\\rm dust}\\,ds`
is the dust mass column along ray direction *k* for bin *b*, and
:math:`\\kappa^{(b)}_{\\rm ext,UV}` is the band-averaged extinction opacity
from the dustkappa file.

References
----------
- RADMC-3D manual, Sect. 7.4 (dustkappa file format)
- Weingartner & Draine (2001) for ISM grain opacity context

Functions
---------
resolve_uv_tau_mode
    Determine which UV optical depth method to use.
compute_tau_uv_from_dust_columns
    Compute tau_uv per ray from dust mass columns and kext_uv.
prepare_dust_density_fields
    Extract per-bin dust density arrays from a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from diskbridge._logging import logger
from diskbridge.radmc3d.dustkappa_reader import load_kext_uv_for_bins


def resolve_uv_tau_mode(
    model,
    radmc_inputs_dir: Optional[str | Path],
    species_base: str,
    uv_min_um: float,
    uv_max_um: float,
) -> Tuple[str, Optional[np.ndarray]]:
    """Determine whether to use dustkappa-based or SIGMA_DUST UV extinction.

    Uses dustkappa mode only if **all** of the following are true:

    1. ``radmc_inputs_dir`` exists and contains ``dustkappa_{species_base}{ibin}.inp``
       for every dust bin.
    2. ``model.dust`` is not None with ``nbin > 0`` and densities available.
    3. The dustkappa file set matches the bin naming convention DiskBridge writes.

    If any condition fails, falls back to the ``"sigma_dust"`` mode which uses
    ``tau = SIGMA_DUST * N_H``.

    Parameters
    ----------
    model : diskbridge.model.Model
        DiskBridge model instance.
    radmc_inputs_dir : str, Path, or None
        Directory containing ``dustkappa_*.inp`` files. If None, always
        falls back to sigma_dust mode.
    species_base : str
        Species base name used in dustkappa file naming (e.g. ``"silicate"``).
    uv_min_um : float
        Minimum UV wavelength in microns.
    uv_max_um : float
        Maximum UV wavelength in microns.

    Returns
    -------
    mode : str
        ``"dustkappa"`` or ``"sigma_dust"``.
    kext_uv : ndarray of shape (nbin,) or None
        Band-averaged extinction opacities per bin (cm^2/g) if mode is
        ``"dustkappa"``, else ``None``.
    """
    # Condition 1: directory must exist
    if radmc_inputs_dir is None:
        logger.info("UV tau mode: sigma_dust (no radmc_inputs_dir provided)")
        return "sigma_dust", None

    radmc_inputs_dir = Path(radmc_inputs_dir)
    if not radmc_inputs_dir.is_dir():
        logger.info(
            "UV tau mode: sigma_dust (radmc_inputs_dir %s does not exist)",
            radmc_inputs_dir,
        )
        return "sigma_dust", None

    # Condition 2: model must have dust with bins
    if model.dust is None:
        logger.info("UV tau mode: sigma_dust (model has no dust)")
        return "sigma_dust", None

    nbin = model.dust.nbin
    if nbin == 0:
        logger.info("UV tau mode: sigma_dust (model has 0 dust bins)")
        return "sigma_dust", None

    # Condition 3: all dustkappa files must exist and be readable
    kext_uv = load_kext_uv_for_bins(
        radmc_inputs_dir, species_base, nbin, uv_min_um, uv_max_um,
    )
    if kext_uv is None:
        logger.info(
            "UV tau mode: sigma_dust (missing dustkappa files for species=%s, nbin=%d)",
            species_base, nbin,
        )
        return "sigma_dust", None

    logger.info(
        "UV tau mode: dustkappa (nbin=%d, species=%s, kext_uv=%s cm^2/g)",
        nbin, species_base, kext_uv,
    )
    return "dustkappa", kext_uv


def prepare_dust_density_fields(
    model,
) -> list[np.ndarray]:
    """Extract per-bin dust density arrays (CGS, g/cm^3) from a model.

    Parameters
    ----------
    model : diskbridge.model.Model
        DiskBridge model with ``model.dust`` populated.

    Returns
    -------
    list of ndarray
        One 3-D array per dust bin, each in CGS (g/cm^3) with axis order
        matching ``model.mesh.axis_names()``.

    Raises
    ------
    ValueError
        If ``model.dust`` is None or has no bins.
    """
    if model.dust is None:
        raise ValueError("Model has no dust data")
    nbin = model.dust.nbin
    if nbin == 0:
        raise ValueError("Model has no dust bins defined")

    rho_bins = []
    for ibin in range(nbin):
        bin_data = model.dust.bins[f"bin_{ibin}"]
        rho_field = bin_data["density"]  # Field object
        rho_cgs = np.asarray(
            rho_field.data.to_base_units().magnitude, dtype=np.float64
        )
        rho_bins.append(np.ascontiguousarray(rho_cgs))

    return rho_bins


def compute_tau_uv_from_dust_columns(
    dust_cols: dict[str, np.ndarray],
    kext_uv: np.ndarray,
    nbin: int,
) -> np.ndarray:
    """Compute per-ray UV optical depth from dust mass columns.

    .. math::

        \\tau_{\\rm UV}(\\text{cell}, \\text{pix})
        = \\sum_{b=0}^{N_{\\rm bin}-1}
          \\Sigma_{\\rm dust}^{(b)}(\\text{cell}, \\text{pix})
          \\;\\kappa^{(b)}_{\\rm ext,UV}

    Parameters
    ----------
    dust_cols : dict[str, ndarray]
        Column integration results keyed by ``"dust_bin_0"``, ``"dust_bin_1"``,
        etc. Each value has shape ``(n_candidates, npix)`` in units of g/cm^2.
    kext_uv : ndarray, shape (nbin,)
        Band-averaged UV extinction opacity per bin in cm^2/g.
    nbin : int
        Number of dust bins.

    Returns
    -------
    tau_uv : ndarray, shape (n_candidates, npix)
        UV optical depth per ray.
    """
    tau_uv = None
    for ibin in range(nbin):
        key = f"dust_bin_{ibin}"
        sigma_dust = dust_cols[key]  # (n_candidates, npix), g/cm^2
        contribution = sigma_dust * kext_uv[ibin]
        if tau_uv is None:
            tau_uv = contribution.copy()
        else:
            tau_uv += contribution

    if tau_uv is None:
        raise ValueError("No dust bin columns found (nbin=0?)")

    return tau_uv
