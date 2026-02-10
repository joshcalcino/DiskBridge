"""
Cache and reuse directional UV weights (W_rays) across shielding iterations.

W_rays depends only on dust densities, dust opacities, chi_radmc, nside,
chi_ext0, and star_uv_luminosity -- NOT on gas-phase abundances. It can
therefore be computed once per radiation+dust state and reused for all
subsequent shielding recomputations within an iterative chemistry loop.

This module provides:

- :func:`ensure_W_rays`: low-level cache helper that takes explicit dust/UV
  parameters, computes W_rays if not cached, and stores on ``rad``.
- :func:`maybe_ensure_W_rays`: high-level convenience that resolves all
  prerequisites (dust bins, kext_uv, chi_ext0, star UV luminosity) from
  ``rad`` and ``diskbridge.params``, then delegates to :func:`ensure_W_rays`.
  Silently returns ``None`` when prerequisites are unavailable (e.g. 1-D mesh
  or missing dustkappa files).

Functions
---------
ensure_W_rays
    Compute W_rays once and cache on rad; return cached on subsequent calls.
maybe_ensure_W_rays
    Resolve prerequisites from rad/params and call ensure_W_rays if possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

import diskbridge
from diskbridge._logging import logger
from diskbridge.chemistry.shielding.angular_uv_weights import (
    compute_star_uv_luminosity,
    compute_uv_direction_weights_healpix,
)
from diskbridge.chemistry.shielding.columns_1d import is_effectively_1d
from diskbridge.chemistry.shielding.dust_uv_tau import (
    prepare_dust_density_fields,
    resolve_uv_tau_mode,
)

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel


def _build_cache_key(
    nside: int,
    chi_ext0: float,
    star_uv_luminosity_erg_s: float,
    self_weight: float,
) -> tuple:
    """Build a lightweight hashable key for W_rays invalidation.

    Parameters
    ----------
    nside : int
        HEALPix Nside used for the weight computation.
    chi_ext0 : float
        External UV field strength (Draine units).
    star_uv_luminosity_erg_s : float
        Stellar UV luminosity (erg/s).
    self_weight : float
        Starting-cell weight in ray integration.

    Returns
    -------
    tuple
        Hashable key. If any element changes, W_rays must be recomputed.

    Notes
    -----
    Dust density arrays and kext_uv are not included because their identity
    (object id) is ephemeral and unreliable across calls. Callers must clear
    ``rad.W_rays`` explicitly when dust or chi_radmc changes (e.g. after a
    RADMC-3D rerun).
    """
    return (int(nside), float(chi_ext0), float(star_uv_luminosity_erg_s),
            float(self_weight))


def ensure_W_rays(
    rad: "RadModel",
    *,
    nside: int,
    dust_rho_bins: list[np.ndarray],
    kext_uv: np.ndarray,
    chi_ext0: float,
    star_uv_luminosity_erg_s: float,
    self_weight: float = 1.0,
    cache_dir: str | None = None,
) -> np.ndarray:
    """Compute and cache ``rad.W_rays``; return existing cache on repeat calls.

    W_rays (directional UV weights) depends on dust and chi_radmc, not on
    gas-phase abundances. This function computes it once per radiation+dust
    state and stores the result on ``rad`` so that subsequent shielding calls
    within an iterative chemistry loop can reuse it without recomputing.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper. ``rad.W_rays`` is set as a side effect.
    nside : int
        HEALPix Nside (npix = 12 * nside**2).
    dust_rho_bins : list of ndarray
        Per-bin dust mass density arrays (g/cm^3).
    kext_uv : ndarray, shape (nbin,)
        Band-averaged UV extinction opacity per bin (cm^2/g).
    chi_ext0 : float
        External UV field strength in Draine units.
    star_uv_luminosity_erg_s : float
        Total stellar UV luminosity in erg/s.
    self_weight : float, optional
        Starting-cell weight in ray integration. Default 1.0.
    cache_dir : str or None, optional
        Directory for HEALPix geometry / column caching (passed through to
        ``compute_uv_direction_weights_healpix``).

    Returns
    -------
    W_rays : ndarray, shape (n_cells, npix)
        Normalised directional UV weights per cell.
    """
    key = _build_cache_key(nside, chi_ext0, star_uv_luminosity_erg_s,
                           self_weight)

    existing = getattr(rad, "W_rays", None)
    existing_key = getattr(rad, "W_rays_key", None)
    if existing is not None and existing_key == key:
        logger.info("W_rays cache hit (nside=%d); reusing existing array", nside)
        return existing

    # --- Memory estimate ---
    chi_radmc = rad.ensure_chi()
    chi_arr = np.asarray(chi_radmc.to("dimensionless").magnitude, dtype=np.float64)
    n_cells = chi_arr.size
    npix = 12 * int(nside) ** 2
    nbytes = n_cells * npix * 8  # float64
    gib = nbytes / (2**30)

    logger.info(
        "Allocating W_rays: shape=(%d, %d) float64 => %.2f GiB",
        n_cells, npix, gib,
    )

    # --- Compute ---
    W_rays, _candidate_idx, _dirs, _cell_centers, _debug = (
        compute_uv_direction_weights_healpix(
            rad.model.mesh,
            chi_radmc=chi_arr,
            nside=int(nside),
            dust_rho_bins=dust_rho_bins,
            kext_uv=kext_uv,
            chi_ext0=float(chi_ext0),
            star_uv_luminosity_erg_s=float(star_uv_luminosity_erg_s),
            self_weight=float(self_weight),
            cache_dir=cache_dir,
        )
    )

    # Discard debug dict to save memory; keep only W_rays.
    del _debug, _candidate_idx, _dirs, _cell_centers

    rad.W_rays = W_rays
    rad.W_rays_key = key

    logger.info(
        "W_rays computed and cached on rad (nside=%d, %.2f GiB)",
        nside, gib,
    )
    return W_rays


def maybe_ensure_W_rays(
    rad: "RadModel",
    *,
    nside: int,
) -> Optional[np.ndarray]:
    """Resolve W_rays prerequisites from ``rad`` / ``diskbridge.params`` and cache.

    This is the recommended entry point for chemistry models. It:

    1. Checks whether the mesh is effectively 1-D (W_rays is only meaningful
       for multi-D HEALPix integration); returns ``None`` for 1-D meshes.
    2. Resolves dust opacity mode via :func:`resolve_uv_tau_mode`. If
       dustkappa files are unavailable, returns ``None`` (shielding will
       fall back to isotropic averaging).
    3. Extracts ``chi_ext0`` from ``diskbridge.params.external_uv_chi``
       (0.0 if external UV is disabled).
    4. Computes stellar UV luminosity from ``diskbridge.params`` star/accretion
       fields and the UV wavelength band.
    5. Delegates to :func:`ensure_W_rays` which handles caching and the
       memory estimate.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper with ``rad.model``, ``rad.inputs_dir``, and
        ``rad.ensure_chi()`` available.
    nside : int
        HEALPix Nside (npix = 12 * nside**2).

    Returns
    -------
    W_rays : ndarray or None
        Directional UV weights if all prerequisites are met, else ``None``.
        When ``None``, downstream shielding functions use uniform (isotropic)
        averaging.
    """
    params = diskbridge.params

    # 1. Skip for 1-D meshes (HEALPix not used)
    chi = rad.ensure_chi()
    chi_arr = np.asarray(chi.to("dimensionless").magnitude, dtype=np.float64)
    if is_effectively_1d(rad.model.mesh, chi_arr.shape):
        logger.debug("W_rays: skipping for effectively 1-D mesh")
        return getattr(rad, "W_rays", None)

    # 2. Resolve dust opacity mode
    species_base = params.species
    if isinstance(species_base, list):
        species_base = species_base[0]

    uv_min_um = float(params.uv_min.to("um").magnitude)
    uv_max_um = float(params.uv_max.to("um").magnitude)

    mode, kext_uv = resolve_uv_tau_mode(
        rad.model,
        radmc_inputs_dir=rad.inputs_dir,
        species_base=species_base,
        uv_min_um=uv_min_um,
        uv_max_um=uv_max_um,
    )
    if mode != "dustkappa" or kext_uv is None:
        logger.info(
            "W_rays: dustkappa mode unavailable (mode=%s); "
            "shielding will use isotropic averaging",
            mode,
        )
        return getattr(rad, "W_rays", None)

    # 3. Dust density fields
    try:
        dust_rho_bins = prepare_dust_density_fields(rad.model)
    except (ValueError, KeyError) as exc:
        logger.info("W_rays: cannot prepare dust density fields (%s); skipping", exc)
        return getattr(rad, "W_rays", None)

    # 4. External UV
    chi_ext0 = 0.0
    if getattr(params, "external_uv", False):
        chi_ext0 = float(getattr(params, "external_uv_chi", 0.0))

    # 5. Stellar UV luminosity
    uv_min_cm = uv_min_um * 1.0e-4  # um -> cm
    uv_max_cm = uv_max_um * 1.0e-4
    star_uv_lum = compute_star_uv_luminosity(params, uv_min_cm, uv_max_cm)

    # 6. Delegate to ensure_W_rays (handles caching + memory check)
    return ensure_W_rays(
        rad,
        nside=nside,
        dust_rho_bins=dust_rho_bins,
        kext_uv=kext_uv,
        chi_ext0=chi_ext0,
        star_uv_luminosity_erg_s=star_uv_lum,
    )
