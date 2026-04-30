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
from diskbridge._units import units
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
    isotropic_outside_r_au: float | None,
    outer_weight_mode: str | None,
    uv_product: str = "chi",
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
    return (
        int(nside),
        float(chi_ext0),
        float(star_uv_luminosity_erg_s),
        None if isotropic_outside_r_au is None else float(isotropic_outside_r_au),
        None if outer_weight_mode is None else str(outer_weight_mode),
        str(uv_product),
    )


def _apply_uniform_weights_outside_radius(
    W_rays: np.ndarray,
    cell_centers: np.ndarray,
    *,
    isotropic_outside_r_cm: float,
) -> np.ndarray:
    """Force uniform HEALPix weights outside a spherical radius."""
    if W_rays.ndim != 2:
        raise ValueError(f"W_rays must be 2-D, got shape {W_rays.shape}")
    if cell_centers.shape != (W_rays.shape[0], 3):
        raise ValueError(
            f"cell_centers must have shape ({W_rays.shape[0]}, 3), got {cell_centers.shape}"
        )

    radii_cm = np.sqrt(np.sum(np.asarray(cell_centers, dtype=float) ** 2, axis=1))
    outer_mask = radii_cm >= float(isotropic_outside_r_cm)
    if not np.any(outer_mask):
        return W_rays

    W_out = np.asarray(W_rays, dtype=float).copy()
    W_out[outer_mask, :] = 1.0 / float(W_out.shape[1])
    return W_out


def _apply_tau_weights_outside_radius(
    W_rays: np.ndarray,
    tau_ext_rays: np.ndarray,
    cell_centers: np.ndarray,
    *,
    isotropic_outside_r_cm: float,
) -> np.ndarray:
    """Force boundary-tau directional weights outside a spherical radius."""
    if W_rays.ndim != 2:
        raise ValueError(f"W_rays must be 2-D, got shape {W_rays.shape}")
    if tau_ext_rays.shape != W_rays.shape:
        raise ValueError(
            f"tau_ext_rays must match W_rays shape {W_rays.shape}, got {tau_ext_rays.shape}"
        )
    if cell_centers.shape != (W_rays.shape[0], 3):
        raise ValueError(
            f"cell_centers must have shape ({W_rays.shape[0]}, 3), got {cell_centers.shape}"
        )

    radii_cm = np.sqrt(np.sum(np.asarray(cell_centers, dtype=float) ** 2, axis=1))
    outer_mask = radii_cm >= float(isotropic_outside_r_cm)
    if not np.any(outer_mask):
        return W_rays

    W_out = np.asarray(W_rays, dtype=float).copy()
    outer_idx = np.where(outer_mask)[0]
    tau_outer = np.asarray(tau_ext_rays[outer_mask], dtype=float)
    tau_outer = np.clip(tau_outer, 0.0, 700.0)
    raw = np.exp(-tau_outer)
    sums = np.sum(raw, axis=1, keepdims=True)
    valid = sums[:, 0] > 0.0
    if np.any(valid):
        W_out[outer_idx[valid], :] = raw[valid, :] / sums[valid, :]
    if np.any(~valid):
        W_out[outer_idx[~valid], :] = 1.0 / float(W_out.shape[1])
    return W_out


def _apply_outer_background_weights(
    W_rays: np.ndarray,
    *,
    tau_ext_rays: np.ndarray,
    cell_centers: np.ndarray,
    isotropic_outside_r_cm: float,
    mode: str,
) -> np.ndarray:
    """Apply the configured background-dominated outer weighting mode."""
    mode = str(mode).strip().lower()
    if mode == "uniform":
        return _apply_uniform_weights_outside_radius(
            W_rays,
            cell_centers,
            isotropic_outside_r_cm=isotropic_outside_r_cm,
        )
    if mode == "tau":
        return _apply_tau_weights_outside_radius(
            W_rays,
            tau_ext_rays,
            cell_centers,
            isotropic_outside_r_cm=isotropic_outside_r_cm,
        )
    raise ValueError(
        "segmented_outer_weight_mode must be 'tau' or 'uniform', "
        f"got {mode!r}"
    )


def ensure_W_rays(
    rad: "RadModel",
    *,
    nside: int,
    dust_rho_bins: list[np.ndarray],
    kext_uv: np.ndarray,
    chi_ext0: float,
    star_uv_luminosity_erg_s: float,
    cache_dir: str | None = None,
    isotropic_outside_r_au: float | None = None,
    outer_weight_mode: str | None = None,
    uv_product: str = "chi",
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
    cache_dir : str or None, optional
        Directory for HEALPix geometry / column caching (passed through to
        ``compute_uv_direction_weights_healpix``).

    Returns
    -------
    W_rays : ndarray, shape (n_cells, npix)
        Normalised directional UV weights per cell.
    """
    key = _build_cache_key(
        nside,
        chi_ext0,
        star_uv_luminosity_erg_s,
        isotropic_outside_r_au,
        outer_weight_mode,
        uv_product,
    )

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
            cache_dir=cache_dir,
        )
    )

    if isotropic_outside_r_au is not None:
        W_rays = _apply_outer_background_weights(
            W_rays,
            tau_ext_rays=np.asarray(_debug["tau_ext_rays"], dtype=float),
            cell_centers=_cell_centers,
            isotropic_outside_r_cm=float(isotropic_outside_r_au) * units("au").to("cm").magnitude,
            mode="tau" if outer_weight_mode is None else str(outer_weight_mode),
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
    uv_product: str = "G_CO_diss",
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
    uv_product : str, optional
        UV product used for directional shielding weights.

    Returns
    -------
    W_rays : ndarray or None
        Directional UV weights if all prerequisites are met, else ``None``.
        When ``None``, downstream shielding functions use uniform (isotropic)
        averaging.
    """
    params = diskbridge.params

    # 1. Skip for 1-D meshes (HEALPix not used)
    chi = rad.ensure_uv_product(uv_product, fallback_to_chi=True)
    chi_arr = np.asarray(chi.to("dimensionless").magnitude, dtype=np.float64)
    if is_effectively_1d(rad.model.mesh, chi_arr.shape):
        logger.debug("W_rays: skipping for effectively 1-D mesh")
        return getattr(rad, "W_rays", None)

    # 2. Resolve dust opacity mode
    species_base = params.species
    if isinstance(species_base, list):
        species_base = species_base[0]

    if uv_product == "G_C_ion":
        uv_min_um = 0.0912
        uv_max_um = 0.1101
    elif uv_product in {"G_CO_diss", "G_H2_diss"}:
        uv_min_um = 0.0912
        uv_max_um = 0.1118
    else:
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

    isotropic_outside_r_au = getattr(
        rad,
        "isotropic_weight_outside_r_au",
        getattr(rad.model, "segmented_isotropic_weight_outside_r_au", None),
    )
    outer_weight_mode = str(
        getattr(params, "segmented_outer_weight_mode", "tau")
    ).strip().lower()

    # 6. Delegate to ensure_W_rays (handles caching + memory check)
    return ensure_W_rays(
        rad,
        nside=nside,
        dust_rho_bins=dust_rho_bins,
        kext_uv=kext_uv,
        chi_ext0=chi_ext0,
        star_uv_luminosity_erg_s=star_uv_lum,
        isotropic_outside_r_au=isotropic_outside_r_au,
        outer_weight_mode=outer_weight_mode,
        uv_product=uv_product,
    )
