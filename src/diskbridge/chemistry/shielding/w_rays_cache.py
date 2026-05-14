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
from diskbridge._config import get_config
from diskbridge._units import units
from diskbridge.chemistry.shielding.angular_uv_weights import (
    _chunk_size_from_memory_budget,
    compute_star_uv_source_strength,
    compute_uv_direction_weights_healpix,
)
from diskbridge.chemistry.shielding.columns_1d import is_effectively_1d
from diskbridge.chemistry.shielding.dust_uv_tau import (
    prepare_dust_density_fields,
    resolve_uv_tau_mode,
)
from diskbridge.radmc3d.uv_products import (
    draine_reference_for_product,
    uv_product_specs_from_config,
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
    uv_product_fingerprint: tuple | None = None,
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
        uv_product_fingerprint,
    )


def _bytes_to_gib(nbytes: int | float) -> float:
    return float(nbytes) / float(2**30)


def _estimate_w_rays_memory_bytes(
    *,
    n_cells: int,
    npix: int,
    nbin: int,
    include_star: bool,
) -> dict[str, int | float]:
    """Estimate peak memory for building directional HEALPix UV weights.

    The old estimate only counted the final ``W_rays`` array. The build path
    also keeps dense ray maps for dust columns, tau, source contributions, the
    total contribution map, and normalization temporaries.
    """
    n_cells = int(n_cells)
    npix = int(npix)
    nbin = int(nbin)
    if n_cells < 0 or npix < 0 or nbin < 0:
        raise ValueError("n_cells, npix, and nbin must be non-negative")

    float64_bytes = np.dtype(np.float64).itemsize
    bool_bytes = np.dtype(bool).itemsize
    index_bytes = np.dtype(np.intp).itemsize

    ray_map_bytes = n_cells * npix * float64_bytes
    cell_scalar_bytes = n_cells * float64_bytes

    # Dense ray-sized arrays alive near the normalization step:
    # dust columns (nbin), tau_ext, uv_ext, uv_star, uv_iso, uv_total, W_rays,
    # plus two conservative temporaries from masked advanced-index division.
    dense_ray_map_count = nbin + 8
    dense_ray_peak_bytes = dense_ray_map_count * ray_map_bytes

    non_ray_bytes = (
        cell_scalar_bytes  # chi_arr
        + nbin * cell_scalar_bytes  # prepared dust_rho_bins held by caller
        + nbin * cell_scalar_bytes  # fields_stack inside column integration
        + n_cells * bool_bytes  # candidate_mask
        + n_cells * 3 * index_bytes  # candidate_idx
        + n_cells * 3 * float64_bytes  # cell_centers
    )
    if include_star:
        non_ray_bytes += (
            nbin * cell_scalar_bytes  # dust_fields_stack for starward rays
            + nbin * cell_scalar_bytes  # star_cols
            + 6 * cell_scalar_bytes  # tau/r/F/chi/star-dir helper arrays
            + n_cells * index_bytes  # k_star
        )

    estimated_peak_bytes = dense_ray_peak_bytes + non_ray_bytes

    return {
        "ray_map_bytes": ray_map_bytes,
        "w_rays_final_bytes": ray_map_bytes,
        "dust_column_bytes": nbin * ray_map_bytes,
        "uv_working_ray_bytes": 5 * ray_map_bytes,
        "normalization_temp_bytes": 2 * ray_map_bytes,
        "dense_ray_map_count": dense_ray_map_count,
        "dense_ray_peak_bytes": dense_ray_peak_bytes,
        "non_ray_bytes": non_ray_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
    }


def _uv_product_band_um(uv_product: str) -> tuple[float, float]:
    """Return wavelength limits for a UV product in micron.

    Parameters
    ----------
    uv_product : str
        UV product name.

    Returns
    -------
    tuple of float
        Lower and upper wavelength limits in micron.
    """
    specs = uv_product_specs_from_config(get_config().get("radmc3d", {}).get("uv_products", {}))
    for spec in specs:
        if spec.field_name == uv_product:
            return float(spec.band.lam_min_nm) * 1.0e-3, float(spec.band.lam_max_nm) * 1.0e-3
    return (
        float(diskbridge.params.uv_min.to("um").magnitude),
        float(diskbridge.params.uv_max.to("um").magnitude),
    )


def _uv_product_weighting(uv_product: str) -> str:
    """Return weighting for a UV product.

    Parameters
    ----------
    uv_product : str
        UV product name.

    Returns
    -------
    str
        ``"energy"`` or ``"photon"``.
    """
    specs = uv_product_specs_from_config(get_config().get("radmc3d", {}).get("uv_products", {}))
    for spec in specs:
        if spec.field_name == uv_product:
            return spec.band.weight
    return "energy"


def _uv_product_draine_reference(uv_product: str, weighting: str) -> float:
    """Return the Draine reference for a directional UV product.

    Parameters
    ----------
    uv_product : str
        UV product name.
    weighting : {"energy", "photon"}
        Product weighting.

    Returns
    -------
    float
        Energy density in erg cm^-3 or photon flux in cm^-2 s^-1.
    """
    specs = uv_product_specs_from_config(get_config().get("radmc3d", {}).get("uv_products", {}))
    unit = "1/(cm^2 s)" if str(weighting).lower() == "photon" else "erg/cm^3"
    product_name = "chi_broad" if str(uv_product) == "chi" else uv_product
    return float(draine_reference_for_product(product_name, specs).to(unit).magnitude)


def _ensure_weight_field(rad: "RadModel", uv_product: str):
    """Return the scalar field used for directional weights without hidden fallback."""
    if str(uv_product) == "chi":
        return rad.ensure_chi()
    return rad.ensure_uv_product(uv_product, fallback_to_chi=False)


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
    chunk_size: int | None = None,
    memory_budget_gib: float | None = None,
    keep_debug_arrays: bool | None = None,
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
    chunk_size : int or None, optional
        Number of candidate cells per UV-weight build chunk.
    memory_budget_gib : float or None, optional
        Approximate working-memory budget per chunk, excluding final W_rays.
    keep_debug_arrays : bool or None, optional
        Keep full diagnostic ray maps. Defaults to false in
        ``compute_uv_direction_weights_healpix``.

    Returns
    -------
    W_rays : ndarray, shape (n_cells, npix)
        Normalised directional UV weights per cell.
    """
    uv_weighting = _uv_product_weighting(uv_product)
    uv_band_um = _uv_product_band_um(uv_product)
    uv_reference = _uv_product_draine_reference(uv_product, uv_weighting)
    uv_product_fingerprint = (
        float(uv_band_um[0]),
        float(uv_band_um[1]),
        str(uv_weighting),
        float(uv_reference),
    )
    key = _build_cache_key(
        nside,
        chi_ext0,
        star_uv_luminosity_erg_s,
        isotropic_outside_r_au,
        outer_weight_mode,
        uv_product,
        uv_product_fingerprint,
    )

    existing = getattr(rad, "W_rays", None)
    existing_key = getattr(rad, "W_rays_key", None)
    if existing is not None and existing_key == key:
        logger.info("W_rays cache hit (nside=%d); reusing existing array", nside)
        return existing

    # --- Memory estimate ---
    chi_radmc = _ensure_weight_field(rad, uv_product)
    chi_arr = np.asarray(chi_radmc.to("dimensionless").magnitude, dtype=np.float64)
    n_cells = chi_arr.size
    npix = 12 * int(nside) ** 2
    nbin = len(dust_rho_bins)
    if memory_budget_gib is None:
        memory_budget_gib = getattr(diskbridge.params, "w_rays_memory_budget_gib", None)
    if chunk_size is None:
        chunk_size = getattr(diskbridge.params, "w_rays_chunk_size", None)
    if chunk_size is not None:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("w_rays_chunk_size must be positive")
    resolved_chunk_size = (
        max(1, min(int(chunk_size), n_cells))
        if chunk_size is not None
        else _chunk_size_from_memory_budget(
            n_candidates=n_cells,
            npix=npix,
            nbin=nbin,
            memory_budget_gib=memory_budget_gib,
        )
    )
    memory = _estimate_w_rays_memory_bytes(
        n_cells=n_cells,
        npix=npix,
        nbin=nbin,
        include_star=float(star_uv_luminosity_erg_s) > 0.0,
    )
    gib = _bytes_to_gib(int(memory["w_rays_final_bytes"]))
    peak_gib = _bytes_to_gib(int(memory["estimated_peak_bytes"]))

    logger.info(
        "Allocating W_rays: shape=(%d, %d) float64, nbin=%d => %.2f GiB final; "
        "unchunked estimated build peak %.2f GiB "
        "(dense ray maps=%d, dust columns %.2f GiB, UV maps %.2f GiB, "
        "normalization temps %.2f GiB, geometry/input %.2f GiB)",
        n_cells,
        npix,
        nbin,
        gib,
        peak_gib,
        int(memory["dense_ray_map_count"]),
        _bytes_to_gib(int(memory["dust_column_bytes"])),
        _bytes_to_gib(int(memory["uv_working_ray_bytes"])),
        _bytes_to_gib(int(memory["normalization_temp_bytes"])),
        _bytes_to_gib(int(memory["non_ray_bytes"])),
    )
    chunk_memory = _estimate_w_rays_memory_bytes(
        n_cells=resolved_chunk_size,
        npix=npix,
        nbin=nbin,
        include_star=float(star_uv_luminosity_erg_s) > 0.0,
    )
    chunk_peak_gib = _bytes_to_gib(
        int(memory["w_rays_final_bytes"]) + int(chunk_memory["estimated_peak_bytes"])
    )
    logger.info(
        "Chunked W_rays build: chunk_size=%d (%d chunk(s)); estimated peak %.2f GiB "
        "including final W_rays and one chunk",
        resolved_chunk_size,
        (n_cells + resolved_chunk_size - 1) // resolved_chunk_size,
        chunk_peak_gib,
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
            star_uv_reference_energy_density=uv_reference,
            star_uv_weighting=uv_weighting,
            chunk_size=resolved_chunk_size,
            memory_budget_gib=memory_budget_gib,
            keep_debug_arrays=keep_debug_arrays,
            isotropic_outside_r_cm=(
                None
                if isotropic_outside_r_au is None
                else float(isotropic_outside_r_au) * units("au").to("cm").magnitude
            ),
            outer_weight_mode="tau" if outer_weight_mode is None else str(outer_weight_mode),
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
        UV product accessors available.
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
    chi = _ensure_weight_field(rad, uv_product)
    chi_arr = np.asarray(chi.to("dimensionless").magnitude, dtype=np.float64)
    if is_effectively_1d(rad.model.mesh, chi_arr.shape):
        logger.debug("W_rays: skipping for effectively 1-D mesh")
        return getattr(rad, "W_rays", None)

    # 2. Resolve dust opacity mode
    species_base = params.species
    if isinstance(species_base, list):
        species_base = species_base[0]

    uv_min_um, uv_max_um = _uv_product_band_um(uv_product)
    uv_weighting = _uv_product_weighting(uv_product)

    mode, kext_uv = resolve_uv_tau_mode(
        rad.model,
        radmc_inputs_dir=rad.inputs_dir,
        species_base=species_base,
        uv_min_um=uv_min_um,
        uv_max_um=uv_max_um,
        weighting=uv_weighting,
        reference_spectrum="draine",
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
    star_uv_lum = compute_star_uv_source_strength(
        params,
        uv_min_cm,
        uv_max_cm,
        weighting=uv_weighting,
    )

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
