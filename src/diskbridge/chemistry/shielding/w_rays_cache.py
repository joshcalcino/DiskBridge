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
  Retains an already supplied W-ray map when automatic prerequisites are
  unavailable; otherwise returns ``None`` (e.g. on a 1-D mesh or without
  dustkappa files).

Functions
---------
ensure_W_rays
    Compute W_rays once and cache on rad; return cached on subsequent calls.
maybe_ensure_W_rays
    Resolve prerequisites from rad/params and call ensure_W_rays if possible.
"""

# db-keywords: shielding, healpix-columns, uv-products, config, units, radmc3d, chemistry, model, mesh
# db-role: helper
# db-scope: package
# db-purpose: Cache and reuse directional UV weights (W_rays) across shielding iterations.

from __future__ import annotations

import os
from pathlib import Path
import resource
import sys
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
    star_inner_radius_cm: float,
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
    star_inner_radius_cm : float
        Inner radius where direct-star rays stop.
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
        float(star_inner_radius_cm),
        None if isotropic_outside_r_au is None else float(isotropic_outside_r_au),
        None if outer_weight_mode is None else str(outer_weight_mode),
        str(uv_product),
        uv_product_fingerprint,
    )


def _bytes_to_gib(nbytes: int | float) -> float:
    return float(nbytes) / float(2**30)


def _current_process_rss_bytes() -> int:
    """Return current RSS on Linux and a conservative high-water value elsewhere."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _estimate_w_rays_memory_bytes(
    *,
    n_cells: int,
    npix: int,
    nbin: int,
    include_star: bool,
    chunk_cells: int | None = None,
    keep_debug_arrays: bool = False,
    keep_closure_diagnostics: bool = False,
) -> dict[str, int | float]:
    """Estimate peak memory for building directional HEALPix UV weights.

    The estimate separates arrays retained for the full build from one chunk's
    incremental working storage. It counts the opacity-weighted scalar mesh
    field and one optical-depth/attenuation ray map; the optimized builder does
    not materialize per-bin dust columns or normalization copies.
    """
    n_cells = int(n_cells)
    npix = int(npix)
    nbin = int(nbin)
    if n_cells < 0 or npix < 0 or nbin < 0:
        raise ValueError("n_cells, npix, and nbin must be non-negative")
    if chunk_cells is None:
        chunk_cells = n_cells
    chunk_cells = max(0, min(int(chunk_cells), n_cells))

    float64_bytes = np.dtype(np.float64).itemsize
    bool_bytes = np.dtype(bool).itemsize
    index_bytes = np.dtype(np.intp).itemsize

    ray_map_bytes = n_cells * npix * float64_bytes
    chunk_ray_map_bytes = chunk_cells * npix * float64_bytes
    cell_scalar_bytes = n_cells * float64_bytes
    chunk_scalar_bytes = chunk_cells * float64_bytes

    caller_retained_bytes = (
        cell_scalar_bytes  # chi_arr retained by the caller
        + nbin * cell_scalar_bytes  # dust_rho_bins retained by the caller
    )
    builder_nonray_persistent_bytes = (
        cell_scalar_bytes  # alpha_uv opacity-weighted mesh field
        + n_cells * bool_bytes  # candidate_mask
        + n_cells * 3 * index_bytes  # candidate_idx
        + n_cells * 3 * float64_bytes  # cell_centers
        + npix * 3 * float64_bytes  # HEALPix directions
    )
    if include_star:
        builder_nonray_persistent_bytes += (
            n_cells * index_bytes  # k_star
            + cell_scalar_bytes  # w_star
            + n_cells * bool_bytes  # valid_star
        )

    builder_persistent_bytes = ray_map_bytes + builder_nonray_persistent_bytes
    persistent_input_bytes = caller_retained_bytes + builder_nonray_persistent_bytes
    persistent_bytes = caller_retained_bytes + builder_persistent_bytes

    # One external attenuation map plus conservative per-cell scalar/index
    # storage used while constructing stellar and closure contributions.
    chunk_float_scalar_count = 10 + (7 if include_star else 0)
    chunk_incremental_bytes = (
        chunk_ray_map_bytes
        + chunk_float_scalar_count * chunk_scalar_bytes
        + chunk_cells * index_bytes
        + 2 * chunk_cells * bool_bytes
    )

    retained_debug_ray_map_count = 5 if keep_debug_arrays else 0
    retained_debug_cell_scalar_count = 7 if keep_debug_arrays else 0
    retained_closure_cell_scalar_count = 15 if keep_closure_diagnostics else 0
    retained_diagnostics_bytes = (
        retained_debug_ray_map_count * ray_map_bytes
        + retained_debug_cell_scalar_count * cell_scalar_bytes
        + retained_closure_cell_scalar_count * cell_scalar_bytes
    )
    # Final concatenation temporarily coexists with the per-chunk diagnostic
    # parts. Count that peak explicitly even though production disables it.
    diagnostic_concatenation_bytes = retained_diagnostics_bytes

    estimated_peak_bytes = (
        persistent_bytes
        + chunk_incremental_bytes
        + retained_diagnostics_bytes
        + diagnostic_concatenation_bytes
    )
    builder_peak_increment_bytes = (
        builder_persistent_bytes
        + chunk_incremental_bytes
        + retained_diagnostics_bytes
        + diagnostic_concatenation_bytes
    )

    return {
        "ray_map_bytes": ray_map_bytes,
        "chunk_ray_map_bytes": chunk_ray_map_bytes,
        "w_rays_final_bytes": ray_map_bytes,
        "alpha_uv_bytes": cell_scalar_bytes,
        "dust_input_bytes": nbin * cell_scalar_bytes,
        "caller_retained_bytes": caller_retained_bytes,
        "builder_nonray_persistent_bytes": builder_nonray_persistent_bytes,
        "builder_persistent_bytes": builder_persistent_bytes,
        "builder_peak_increment_bytes": builder_peak_increment_bytes,
        "dust_column_bytes": 0,
        "uv_working_ray_bytes": chunk_ray_map_bytes,
        "normalization_temp_bytes": 0,
        "persistent_input_bytes": persistent_input_bytes,
        "persistent_bytes": persistent_bytes,
        "chunk_incremental_bytes": chunk_incremental_bytes,
        "retained_diagnostics_bytes": retained_diagnostics_bytes,
        "diagnostic_concatenation_bytes": diagnostic_concatenation_bytes,
        "retained_debug_ray_map_count": retained_debug_ray_map_count,
        "retained_debug_cell_scalar_count": retained_debug_cell_scalar_count,
        "retained_closure_cell_scalar_count": retained_closure_cell_scalar_count,
        "dense_ray_map_count": 1,
        "dense_ray_peak_bytes": chunk_ray_map_bytes,
        "non_ray_bytes": persistent_input_bytes,
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
    keep_closure_diagnostics: bool | None = None,
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
    keep_closure_diagnostics : bool or None, optional
        Keep scalar UV closure diagnostics on ``rad`` as
        ``rad.W_rays_closure_diagnostics``. Defaults to false.

    Returns
    -------
    W_rays : ndarray, shape (n_cells, npix)
        Normalised directional UV weights per cell.
    """
    uv_weighting = _uv_product_weighting(uv_product)
    uv_band_um = _uv_product_band_um(uv_product)
    uv_reference = _uv_product_draine_reference(uv_product, uv_weighting)
    star_inner_radius_cm = 0.0
    if float(star_uv_luminosity_erg_s) > 0.0:
        star_inner_radius_cm = float(diskbridge.params.rstar.to("cm").magnitude)
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
        star_inner_radius_cm,
        isotropic_outside_r_au,
        outer_weight_mode,
        uv_product,
        uv_product_fingerprint,
    )

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
    if keep_debug_arrays is None:
        keep_debug_arrays = getattr(diskbridge.params, "w_rays_keep_debug_arrays", None)
    if keep_closure_diagnostics is None:
        keep_closure_diagnostics = getattr(
            diskbridge.params,
            "w_rays_keep_closure_diagnostics",
            False,
        )
    keep_debug_arrays = bool(keep_debug_arrays)
    keep_closure_diagnostics = bool(keep_closure_diagnostics)

    existing = getattr(rad, "W_rays", None)
    existing_key = getattr(rad, "W_rays_key", None)
    existing_closure = getattr(rad, "W_rays_closure_diagnostics", None)
    existing_stellar = getattr(rad, "W_rays_stellar_metadata", None)
    needs_stellar = float(star_uv_luminosity_erg_s) > 0.0
    if existing is not None and existing_key == key:
        has_requested_closure = (not keep_closure_diagnostics) or existing_closure is not None
        has_requested_stellar = (not needs_stellar) or existing_stellar is not None
        if has_requested_closure and has_requested_stellar:
            logger.info("W_rays cache hit (nside=%d); reusing existing array", nside)
            return existing
        if not has_requested_closure:
            logger.info(
                "W_rays cache hit lacks requested closure diagnostics; recomputing diagnostics"
            )
        if not has_requested_stellar:
            logger.info("W_rays cache hit lacks stellar metadata; recomputing weights")
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
        chunk_cells=resolved_chunk_size,
        keep_debug_arrays=keep_debug_arrays,
        keep_closure_diagnostics=keep_closure_diagnostics,
    )
    gib = _bytes_to_gib(int(memory["w_rays_final_bytes"]))
    peak_gib = _bytes_to_gib(int(memory["estimated_peak_bytes"]))
    rss_before_bytes = _current_process_rss_bytes()
    projected_process_peak_bytes = rss_before_bytes + int(
        memory["builder_peak_increment_bytes"]
    )

    logger.info(
        "Allocating W_rays: shape=(%d, %d) float64, nbin=%d => %.2f GiB final; "
        "known-array peak %.2f GiB "
        "(persistent %.2f GiB, alpha_uv %.2f GiB, one chunk %.2f GiB, "
        "chunk ray map %.2f GiB, retained diagnostics %.2f GiB)",
        n_cells,
        npix,
        nbin,
        gib,
        peak_gib,
        _bytes_to_gib(int(memory["persistent_bytes"])),
        _bytes_to_gib(int(memory["alpha_uv_bytes"])),
        _bytes_to_gib(int(memory["chunk_incremental_bytes"])),
        _bytes_to_gib(int(memory["chunk_ray_map_bytes"])),
        _bytes_to_gib(int(memory["retained_diagnostics_bytes"])),
    )
    logger.info(
        "Chunked W_rays build: chunk_size=%d (%d chunk(s)); current RSS %.2f GiB, "
        "projected process peak %.2f GiB (current RSS + builder-owned arrays)",
        resolved_chunk_size,
        (n_cells + resolved_chunk_size - 1) // resolved_chunk_size,
        _bytes_to_gib(rss_before_bytes),
        _bytes_to_gib(projected_process_peak_bytes),
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
            star_inner_radius_cm=star_inner_radius_cm,
            chunk_size=resolved_chunk_size,
            memory_budget_gib=memory_budget_gib,
            keep_debug_arrays=keep_debug_arrays,
            keep_closure_diagnostics=keep_closure_diagnostics,
            isotropic_outside_r_cm=(
                None
                if isotropic_outside_r_au is None
                else float(isotropic_outside_r_au) * units("au").to("cm").magnitude
            ),
            outer_weight_mode="tau" if outer_weight_mode is None else str(outer_weight_mode),
            cache_dir=cache_dir,
        )
    )

    closure_diagnostics = _debug.get("closure_diagnostics")
    stellar_metadata = _debug.get("stellar")

    # Discard debug dict to save memory; keep only W_rays and optional
    # lightweight scalar closure diagnostics and stellar component metadata.
    del _debug, _candidate_idx, _dirs, _cell_centers

    rad.W_rays = W_rays
    rad.W_rays_key = key
    rad.W_rays_stellar_metadata = stellar_metadata
    if keep_closure_diagnostics and closure_diagnostics is not None:
        rad.W_rays_closure_diagnostics = closure_diagnostics
    elif hasattr(rad, "W_rays_closure_diagnostics"):
        rad.W_rays_closure_diagnostics = None

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
       for multi-D HEALPix integration); retains a supplied map if present and
       otherwise returns ``None`` for 1-D meshes.
    2. Resolves dust opacity mode via :func:`resolve_uv_tau_mode`. If
       dustkappa files are unavailable, retains a precomputed W-ray map when
       present; otherwise shielding falls back to isotropic averaging.
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
        existing = getattr(rad, "W_rays", None)
        if existing is None:
            logger.info(
                "W_rays: dustkappa mode unavailable (mode=%s); "
                "shielding will use isotropic averaging",
                mode,
            )
        else:
            logger.info(
                "W_rays: dustkappa mode unavailable (mode=%s); "
                "retaining the precomputed directional weights",
                mode,
            )
        return existing

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
