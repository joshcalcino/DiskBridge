"""
Directional UV weighting for HEALPix-based shielding averages.

For each cell, builds a HEALPix map ``uv_total_contrib_k`` representing the
relative UV contribution from direction ``k``:

.. math::

    uv\\_total\\_contrib_k =
    uv\\_ext\\_contrib_k + uv\\_star\\_contrib_k + uv\\_iso\\_contrib_k

Components
----------
- **External direct**:
  :math:`uv\\_ext\\_contrib_k = \\chi_{\\rm ext,0} \\exp(-\\tau^{\\rm ext}_k)`
  where :math:`\\tau^{\\rm ext}_k` is the dust UV optical depth along ray k
  to the domain boundary.

- **Stellar direct** (point source):
  :math:`uv\\_star\\_contrib_{k} = N_{\\rm pix} \\chi^{\\star}_{\\rm dir}
  e^{-\\tau^\\star}` in the single pixel ``k_star`` containing the star
  direction; zero elsewhere. The :math:`N_{\\rm pix}` factor ensures the
  pixel-mean equals the correct scalar stellar contribution independent of
  NSIDE.

- **Isotropic** (RADMC residual):
  :math:`\\chi_{\\rm iso} = \\max(\\chi_{\\rm RADMC} - \\chi_{\\rm ext,dir}
  - \\chi_{\\star,\\rm dir,att},\\; 0)`,
  broadcast uniformly to all pixels.

Weights are then:

.. math::

    W_k = \\frac{uv\\_total\\_contrib_k}{\\sum_j uv\\_total\\_contrib_j}

Any per-ray shielding factor :math:`f_k` is averaged as
:math:`\\langle f \\rangle = \\sum_k W_k f_k`.

References
----------
- Draine 1978, ApJS, 36, 595 (ISRF normalization)
- Visser et al. 2009, A&A 503, 323 (CO shielding tables)
- Draine & Bertoldi 1996, ApJ, 468, 269 (H2 self-shielding)

Functions
---------
planck_band_luminosity
    Integrate Planck function over a wavelength band for a blackbody sphere.
compute_star_uv_luminosity
    Total stellar + accretion UV band luminosity from DiskBridge params.
compute_uv_direction_weights_healpix
    Build per-direction UV weights for HEALPix shielding averaging.
"""

# db-keywords: shielding, co-shielding, healpix-columns, uv-products, disk-mask, config, radmc3d, chemistry, mesh, ray-tracing
# db-role: canonical
# db-scope: package
# db-purpose: Directional UV weighting for HEALPix-based shielding averages.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import healpy as hp
from numba import njit, prange

from diskbridge._logging import logger
from diskbridge._constants import (
    H_PLANCK,
    C_LIGHT,
    K_B,
    SIGMA_SB,
    G_GRAV,
    SOLAR_MASS,
    U_DRAINE,
)
from diskbridge.chemistry.shielding.healpix_columns import (
    _as_f64,
    _prepare_healpix_geometry,
)
from diskbridge.chemistry.shielding.healpix_utils import (
    integrate_rays_multi,
    integrate_starward_rays_multi,
)


DEFAULT_W_RAYS_MEMORY_BUDGET_GIB = 64.0


def _chunk_size_from_memory_budget(
    *,
    n_candidates: int,
    npix: int,
    nbin: int,
    memory_budget_gib: float | None,
) -> int:
    if n_candidates <= 0:
        return 0
    if memory_budget_gib is None:
        memory_budget_gib = DEFAULT_W_RAYS_MEMORY_BUDGET_GIB
    budget_bytes = float(memory_budget_gib) * float(2**30)
    if not np.isfinite(budget_bytes) or budget_bytes <= 0.0:
        return int(n_candidates)

    # Preserve the established production chunk cap until same-node scaling
    # validates a larger value. The optimized builder holds only one dense
    # chunk ray map, so this deliberately conservative formula no longer
    # describes its actual peak; ``w_rays_cache`` reports the live-array
    # estimate separately. The final W_rays array remains outside this budget.
    bytes_per_candidate = int(np.dtype(np.float64).itemsize) * int(npix) * (int(nbin) + 4)
    return max(1, min(int(n_candidates), int(budget_bytes // max(bytes_per_candidate, 1))))


@njit(cache=True, parallel=True)
def _fill_scaled_field_parallel(
    output: np.ndarray,
    field: np.ndarray,
    scale: float,
) -> None:
    """Fill ``output`` with a scaled field without a mesh-sized temporary."""
    output_flat = output.reshape(output.size)
    field_flat = field.reshape(field.size)
    for i in prange(output_flat.size):
        output_flat[i] = scale * field_flat[i]


@njit(cache=True, parallel=True)
def _add_scaled_field_parallel(
    output: np.ndarray,
    field: np.ndarray,
    scale: float,
) -> None:
    """Accumulate a scaled field without a mesh-sized temporary."""
    output_flat = output.reshape(output.size)
    field_flat = field.reshape(field.size)
    for i in prange(output_flat.size):
        output_flat[i] += scale * field_flat[i]


def _build_uv_extinction_field(
    dust_rho_bins: list[np.ndarray],
    kext_uv: np.ndarray,
    mesh_shape: tuple[int, ...],
) -> np.ndarray:
    """Return ``sum(kext_uv * rho_dust)`` as one contiguous ``cm^-1`` field."""
    if len(dust_rho_bins) != int(kext_uv.size):
        raise ValueError(
            "kext_uv must contain one opacity per dust density bin; "
            f"got {kext_uv.size} opacities and {len(dust_rho_bins)} bins"
        )

    alpha_uv = None
    for ibin, rho_bin in enumerate(dust_rho_bins):
        rho = _as_f64(f"dust_rho_bins[{ibin}]", rho_bin)
        if rho.shape != mesh_shape:
            raise ValueError(
                "each dust density field must match chi_radmc shape; "
                f"dust_rho_bins[{ibin}] has {rho.shape}, expected {mesh_shape}"
            )
        if alpha_uv is None:
            alpha_uv = np.empty(mesh_shape, dtype=np.float64)
            _fill_scaled_field_parallel(alpha_uv, rho, float(kext_uv[ibin]))
        else:
            _add_scaled_field_parallel(alpha_uv, rho, float(kext_uv[ibin]))

    if alpha_uv is None:
        raise ValueError("dust_rho_bins is required (list of per-bin density arrays)")
    return alpha_uv


def _integrate_extinction_rays(
    tracer,
    cell_centers: np.ndarray,
    directions: np.ndarray,
    alpha_uv_stack: np.ndarray,
    progress_chunks: int | None,
) -> np.ndarray:
    """Integrate one extinction field, optionally preserving progress chunks."""
    n_cells = int(cell_centers.shape[0])
    if progress_chunks is None or int(progress_chunks) <= 1:
        return integrate_rays_multi(
            tracer,
            cell_centers,
            directions,
            alpha_uv_stack,
        )[:, :, 0]

    n_chunks = max(1, int(progress_chunks))
    chunk_size = (n_cells + n_chunks - 1) // n_chunks
    tau_ext_rays = np.empty((n_cells, directions.shape[0]), dtype=np.float64)
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_cells)
        if start >= end:
            break
        logger.info(
            "UV extinction ray subchunk %d/%d (%d cells)",
            i + 1,
            n_chunks,
            end - start,
        )
        tau_ext_rays[start:end] = integrate_rays_multi(
            tracer,
            cell_centers[start:end],
            directions,
            alpha_uv_stack,
        )[:, :, 0]
    return tau_ext_rays


@njit(cache=True, parallel=True)
def _attenuation_and_external_mean_inplace_parallel(
    tau_or_attenuation: np.ndarray,
    chi_ext0: float,
    outer_mask: np.ndarray,
    outer_tau_mode: bool,
) -> np.ndarray:
    """Replace optical depth by attenuation and return mean external strength."""
    n_cells, npix = tau_or_attenuation.shape
    chi_ext_dir = np.empty(n_cells, dtype=np.float64)
    for i in prange(n_cells):
        attenuation_sum = 0.0
        for j in range(npix):
            tau = tau_or_attenuation[i, j]
            attenuation = np.exp(-tau)
            attenuation_sum += attenuation
            if outer_tau_mode and outer_mask[i]:
                clipped_tau = min(max(tau, 0.0), 700.0)
                tau_or_attenuation[i, j] = np.exp(-clipped_tau)
            else:
                tau_or_attenuation[i, j] = attenuation
        chi_ext_dir[i] = chi_ext0 * attenuation_sum / float(npix)
    return chi_ext_dir


@njit(cache=True, parallel=True)
def _fill_uv_weights_parallel(
    W_chunk: np.ndarray,
    attenuation_rays: np.ndarray,
    chi_ext0: float,
    chi_ext_dir: np.ndarray,
    chi_iso: np.ndarray,
    chi_star_dir_att: np.ndarray,
    valid_star: np.ndarray,
    k_star: np.ndarray,
    outer_mask: np.ndarray,
    outer_mode_code: int,
    w_star_chunk: np.ndarray,
) -> None:
    """Fill every W-ray element once without advanced-indexing temporaries."""
    n_cells, npix = W_chunk.shape
    uniform = 1.0 / float(npix)
    for i in prange(n_cells):
        w_star_chunk[i] = 0.0
        if outer_mask[i]:
            if outer_mode_code == 1:
                for j in range(npix):
                    W_chunk[i, j] = uniform
            else:
                attenuation_sum = 0.0
                for j in range(npix):
                    attenuation_sum += attenuation_rays[i, j]
                if attenuation_sum > 0.0:
                    for j in range(npix):
                        W_chunk[i, j] = attenuation_rays[i, j] / attenuation_sum
                else:
                    for j in range(npix):
                        W_chunk[i, j] = uniform
            continue

        sum_uv_contrib = float(npix) * (
            chi_ext_dir[i] + chi_iso[i] + chi_star_dir_att[i]
        )
        if sum_uv_contrib > 0.0:
            for j in range(npix):
                W_chunk[i, j] = (
                    chi_ext0 * attenuation_rays[i, j] + chi_iso[i]
                ) / sum_uv_contrib
            if valid_star[i]:
                w_star = (
                    float(npix) * chi_star_dir_att[i] / sum_uv_contrib
                )
                w_star_chunk[i] = w_star
                W_chunk[i, k_star[i]] += w_star
        else:
            for j in range(npix):
                W_chunk[i, j] = uniform


def planck_band_luminosity(
    R_cm: float,
    T_K: float,
    lam_min_cm: float,
    lam_max_cm: float,
    n_points: int = 500,
) -> float:
    """Compute blackbody luminosity in a wavelength band.

    .. math::

        L_{\\rm band} = 4\\pi^2 R^2
        \\int_{\\lambda_{\\min}}^{\\lambda_{\\max}} B_\\lambda(T)\\,d\\lambda

    where :math:`B_\\lambda` is the Planck specific intensity.

    Parameters
    ----------
    R_cm : float
        Radius of the emitting sphere in cm.
    T_K : float
        Temperature in Kelvin.
    lam_min_cm : float
        Minimum wavelength in cm.
    lam_max_cm : float
        Maximum wavelength in cm.
    n_points : int, optional
        Number of integration points. Default 500.

    Returns
    -------
    float
        Band luminosity in erg/s.
    """
    if T_K <= 0.0 or R_cm <= 0.0:
        return 0.0

    lam = np.linspace(lam_min_cm, lam_max_cm, n_points)
    lam = np.clip(lam, 1.0e-12, None)
    x = H_PLANCK * C_LIGHT / (lam * K_B * T_K)
    x = np.clip(x, 1.0e-10, 700.0)
    B_lam = 2.0 * H_PLANCK * C_LIGHT**2 / lam**5 / np.expm1(x)

    integral = float(np.trapezoid(B_lam, lam))
    return 4.0 * np.pi**2 * R_cm**2 * integral


def planck_band_photon_luminosity(
    R_cm: float,
    T_K: float,
    lam_min_cm: float,
    lam_max_cm: float,
    n_points: int = 500,
) -> float:
    """Compute blackbody photon luminosity in a wavelength band.

    Parameters
    ----------
    R_cm : float
        Radius of the emitting sphere in cm.
    T_K : float
        Temperature in Kelvin.
    lam_min_cm : float
        Minimum wavelength in cm.
    lam_max_cm : float
        Maximum wavelength in cm.
    n_points : int, optional
        Number of integration points.

    Returns
    -------
    float
        Photon luminosity in photons s^-1.
    """
    if T_K <= 0.0 or R_cm <= 0.0:
        return 0.0

    lam = np.linspace(lam_min_cm, lam_max_cm, n_points)
    lam = np.clip(lam, 1.0e-12, None)
    x = H_PLANCK * C_LIGHT / (lam * K_B * T_K)
    x = np.clip(x, 1.0e-10, 700.0)
    B_lam = 2.0 * H_PLANCK * C_LIGHT**2 / lam**5 / np.expm1(x)
    photon_B_lam = B_lam * lam / (H_PLANCK * C_LIGHT)

    integral = float(np.trapezoid(photon_B_lam, lam))
    return 4.0 * np.pi**2 * R_cm**2 * integral


def compute_star_uv_luminosity(
    params,
    lam_min_cm: float,
    lam_max_cm: float,
) -> float:
    """Compute total stellar UV band luminosity from DiskBridge params.

    Includes the photosphere and an optional accretion hotspot (when
    ``params.mdot > 0``).  The accretion luminosity and hotspot temperature
    follow the same prescription as :meth:`RadWriter.write_stars`.

    Parameters
    ----------
    params : diskbridge._params.Params
        DiskBridge params object with star/accretion fields.
    lam_min_cm : float
        Minimum UV wavelength in cm.
    lam_max_cm : float
        Maximum UV wavelength in cm.

    Returns
    -------
    float
        Total UV luminosity in erg/s (photosphere + accretion hotspot).
    """
    rstar_cm = float(params.rstar.to("cm").magnitude)
    teff_K = float(params.teff.to("K").magnitude)

    L_uv = planck_band_luminosity(rstar_cm, teff_K, lam_min_cm, lam_max_cm)

    mdot_msun_per_yr = float(getattr(params, "mdot", 0.0))
    if mdot_msun_per_yr > 0.0:
        f_fill = float(getattr(params, "accretion_fill_factor", 0.01))
        f_fill = max(min(f_fill, 1.0), 1e-6)
        mstar_g = float(params.mstar.to("g").magnitude)
        mdot_cgs = mdot_msun_per_yr * SOLAR_MASS / (365.25 * 24.0 * 3600.0)
        Lacc_cgs = G_GRAV * mstar_g * mdot_cgs / rstar_cm
        r_acc_cm = (f_fill**0.5) * rstar_cm
        Tacc_K = (Lacc_cgs / (4.0 * np.pi * SIGMA_SB * r_acc_cm**2)) ** 0.25
        if Tacc_K > 0.0:
            L_uv += planck_band_luminosity(r_acc_cm, Tacc_K, lam_min_cm, lam_max_cm)
            logger.debug(
                "Accretion hotspot: T_acc=%.0f K, r_acc=%.3e cm, L_uv_acc=%.3e erg/s",
                Tacc_K,
                r_acc_cm,
                planck_band_luminosity(r_acc_cm, Tacc_K, lam_min_cm, lam_max_cm),
            )

    logger.info("Star UV luminosity (%.4g-%.4g cm): %.3e erg/s", lam_min_cm, lam_max_cm, L_uv)
    return L_uv


def compute_star_uv_source_strength(
    params,
    lam_min_cm: float,
    lam_max_cm: float,
    *,
    weighting: str,
) -> float:
    """Compute stellar UV source strength for an energy or photon product.

    Parameters
    ----------
    params : diskbridge._params.Params
        DiskBridge params object with star/accretion fields.
    lam_min_cm : float
        Minimum UV wavelength in cm.
    lam_max_cm : float
        Maximum UV wavelength in cm.
    weighting : {"energy", "photon"}
        Source weighting to compute.

    Returns
    -------
    float
        Energy luminosity in erg/s or photon luminosity in photons/s.
    """
    weighting = str(weighting).lower()
    if weighting == "energy":
        return compute_star_uv_luminosity(params, lam_min_cm, lam_max_cm)
    if weighting != "photon":
        raise ValueError("weighting must be 'energy' or 'photon'")

    rstar_cm = float(params.rstar.to("cm").magnitude)
    teff_K = float(params.teff.to("K").magnitude)
    L_uv = planck_band_photon_luminosity(rstar_cm, teff_K, lam_min_cm, lam_max_cm)

    mdot_msun_per_yr = float(getattr(params, "mdot", 0.0))
    if mdot_msun_per_yr > 0.0:
        f_fill = float(getattr(params, "accretion_fill_factor", 0.01))
        f_fill = max(min(f_fill, 1.0), 1e-6)
        mstar_g = float(params.mstar.to("g").magnitude)
        mdot_cgs = mdot_msun_per_yr * SOLAR_MASS / (365.25 * 24.0 * 3600.0)
        Lacc_cgs = G_GRAV * mstar_g * mdot_cgs / rstar_cm
        r_acc_cm = (f_fill**0.5) * rstar_cm
        Tacc_K = (Lacc_cgs / (4.0 * np.pi * SIGMA_SB * r_acc_cm**2)) ** 0.25
        if Tacc_K > 0.0:
            L_uv += planck_band_photon_luminosity(r_acc_cm, Tacc_K, lam_min_cm, lam_max_cm)

    logger.info(
        "Star UV photon luminosity (%.4g-%.4g cm): %.3e photons/s",
        lam_min_cm,
        lam_max_cm,
        L_uv,
    )
    return L_uv


def compute_uv_direction_weights_healpix(
    mesh,
    *,
    chi_radmc: np.ndarray,
    nside: int,
    candidate_mask: np.ndarray | None = None,
    dust_rho_bins: list[np.ndarray],
    kext_uv: np.ndarray,
    chi_ext0: float,
    star_uv_luminosity_erg_s: float,
    star_uv_reference_energy_density: float = U_DRAINE,
    star_uv_weighting: str = "energy",
    star_inner_radius_cm: float = 0.0,
    progress_chunks: int | None = None,
    chunk_size: int | None = None,
    memory_budget_gib: float | None = None,
    keep_debug_arrays: bool | None = None,
    keep_closure_diagnostics: bool | None = None,
    isotropic_outside_r_cm: float | None = None,
    outer_weight_mode: str = "tau",
    cache_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute per-direction UV weights for HEALPix shielding averages.

    Builds a HEALPix map of relative UV contributions per direction for each
    candidate cell, combining external direct, stellar direct, and isotropic
    (RADMC residual) components.  Returns normalised weights that can be used
    to average *any* per-ray shielding factor.

    Parameters
    ----------
    mesh : Mesh
        DiskBridge mesh (spherical or cartesian).
    chi_radmc : ndarray
        Full scalar chi from RADMC mean intensity (Draine units), 3-D grid.
    nside : int
        HEALPix Nside (npix = 12 * nside**2).
    candidate_mask : ndarray of bool or None, optional
        Mesh-shaped mask selecting cells for which weights are built. If
        omitted, weights are built for every cell. Selected rows follow
        ``np.argwhere(candidate_mask)`` ordering.
    dust_rho_bins : list of ndarray
        Per-bin dust mass density arrays (g/cm^3), each matching mesh shape.
    kext_uv : ndarray, shape (nbin,)
        Band-averaged UV extinction opacity per dust bin (cm^2/g).
    chi_ext0 : float
        External (boundary) UV field strength in Draine units.
        Set to 0.0 if no external UV is enabled.
    star_uv_luminosity_erg_s : float
        Total stellar UV band luminosity in erg/s.
        Set to 0.0 if no central star.
    star_uv_reference_energy_density : float, optional
        Draine reference energy density or photon flux for the same stellar
        UV band.
    star_uv_weighting : {"energy", "photon"}, optional
        Weighting for the stellar source strength.
    star_inner_radius_cm : float, optional
        Stellar/source radius where direct-star attenuation rays stop. Use 0
        to stop at the coordinate origin.
    progress_chunks : int or None, optional
        If set, split ray integration into chunks with logging.
    chunk_size : int or None, optional
        Number of candidate cells to process per UV-weight chunk. If omitted,
        a chunk size is derived from ``memory_budget_gib``.
    memory_budget_gib : float or None, optional
        Approximate working-memory budget for one chunk, excluding the final
        ``W_rays`` array. Defaults to a conservative internal budget.
    keep_debug_arrays : bool or None, optional
        Keep full-size diagnostic arrays in the returned debug dict. By
        default this is disabled for normal runs.
    keep_closure_diagnostics : bool or None, optional
        Keep scalar per-cell UV closure diagnostics in the returned debug dict.
        This is much smaller than ``keep_debug_arrays`` and is disabled by
        default.
    isotropic_outside_r_cm : float or None, optional
        If set, replace weights outside this radius using ``outer_weight_mode``.
    outer_weight_mode : {"tau", "uniform"}, optional
        Outer-radius weighting mode.
    cache_dir : Path or str or None, optional
        Directory for caching HEALPix geometry and column results.

    Returns
    -------
    W_rays : ndarray, shape (n_candidates, npix)
        Normalised directional weights per candidate cell.
    candidate_idx : ndarray, shape (n_candidates, 3)
        Grid indices per candidate cell.
    dirs : ndarray, shape (npix, 3)
        HEALPix direction vectors.
    cell_centers : ndarray, shape (n_candidates, 3)
        Cartesian cell center positions (cm).
    debug : dict
        Diagnostic arrays when ``keep_debug_arrays`` is true. In normal runs
        only small scalar summaries are returned. When a stellar source is
        present, ``debug["stellar"]`` contains compact per-cell metadata for
        the direct stellar contribution.

    Raises
    ------
    ValueError
        If ``dust_rho_bins`` or ``kext_uv`` are missing/empty.
    """
    if dust_rho_bins is None or len(dust_rho_bins) == 0:
        raise ValueError("dust_rho_bins is required (list of per-bin density arrays)")
    kext_uv = np.asarray(kext_uv, dtype=np.float64)
    if kext_uv.size == 0:
        raise ValueError("kext_uv is required (per-bin UV extinction opacity)")
    if kext_uv.ndim != 1:
        raise ValueError(f"kext_uv must be one-dimensional; got shape {kext_uv.shape}")

    chi_arr = _as_f64("chi_radmc", chi_radmc)
    nbin = len(dust_rho_bins)

    if candidate_mask is None:
        candidate_mask_arr = np.ones(chi_arr.shape, dtype=bool)
    else:
        candidate_mask_arr = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask_arr.shape != chi_arr.shape:
            raise ValueError(
                "candidate_mask must match chi_radmc shape; "
                f"got {candidate_mask_arr.shape} and {chi_arr.shape}"
            )

    # -- 1. Prepare HEALPix geometry -----------------------------------------
    tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
        mesh,
        nside=int(nside),
        candidate_mask=candidate_mask_arr,
        cache_dir=cache_dir,
    )

    n_candidates = int(candidate_idx.shape[0])
    npix = int(dirs.shape[0])

    if n_candidates == 0:
        W_rays = np.full((0, npix), 1.0 / npix, dtype=np.float64)
        return W_rays, candidate_idx, dirs, cell_centers, {
            "keep_debug_arrays": False,
            "keep_closure_diagnostics": False,
        }

    # -- 2. Prepare one opacity-weighted extinction field --------------------
    alpha_uv = _build_uv_extinction_field(
        dust_rho_bins,
        kext_uv,
        tuple(int(v) for v in chi_arr.shape),
    )
    alpha_uv_stack = alpha_uv[np.newaxis, ...]

    if keep_debug_arrays is None:
        keep_debug_arrays = False
    if keep_closure_diagnostics is None:
        keep_closure_diagnostics = False

    if chunk_size is None:
        chunk_size = _chunk_size_from_memory_budget(
            n_candidates=n_candidates,
            npix=npix,
            nbin=nbin,
            memory_budget_gib=memory_budget_gib,
        )
    else:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
    chunk_size = max(1, min(int(chunk_size), n_candidates))

    n_chunks = (n_candidates + chunk_size - 1) // chunk_size
    W_rays = np.empty((n_candidates, npix), dtype=np.float64)

    has_star = float(star_uv_luminosity_erg_s) > 0.0
    star_k_all = None
    star_w_all = None
    star_valid_all = None
    if has_star:
        star_k_all = np.zeros(n_candidates, dtype=np.intp)
        star_w_all = np.zeros(n_candidates, dtype=np.float64)
        star_valid_all = np.zeros(n_candidates, dtype=bool)

    outer_mode_code = 0
    if isotropic_outside_r_cm is not None:
        mode = str(outer_weight_mode).strip().lower()
        if mode == "uniform":
            outer_mode_code = 1
        elif mode == "tau":
            outer_mode_code = 2
        else:
            raise ValueError(
                "segmented_outer_weight_mode must be 'tau' or 'uniform', "
                f"got {mode!r}"
            )

    debug_chunks: dict[str, list[np.ndarray]] = {}
    if keep_debug_arrays:
        for name in (
            "chi_ext_dir",
            "chi_star_dir_att",
            "chi_iso",
            "chi_radmc_cand",
            "tau_ext_rays",
            "uv_ext_contrib",
            "uv_star_contrib",
            "uv_iso_contrib",
            "uv_total_contrib",
            "tau_star",
            "chi_star_unatt",
            "k_star",
        ):
            debug_chunks[name] = []

    closure_chunks: dict[str, list[np.ndarray]] = {}
    if keep_closure_diagnostics:
        for name in (
            "chi_radmc",
            "chi_ext_dir",
            "chi_star_dir_att",
            "chi_iso",
            "chi_direct",
            "chi_reconstructed",
            "closure_residual",
            "direct_excess",
            "direct_to_radmc",
            "f_ext",
            "f_star",
            "f_iso",
            "tau_star",
            "chi_star_unatt",
            "outer_weight_overridden",
        ):
            closure_chunks[name] = []

    chi_iso_medians: list[float] = []
    max_chi_star_unatt = 0.0
    max_tau_star = 0.0

    logger.info(
        "UV direction weights: processing %d candidates in %d chunk(s) "
        "(chunk_size=%d, npix=%d, keep_debug_arrays=%s)",
        n_candidates,
        n_chunks,
        chunk_size,
        npix,
        bool(keep_debug_arrays),
    )

    for ichunk, start in enumerate(range(0, n_candidates, chunk_size), start=1):
        end = min(start + chunk_size, n_candidates)
        candidate_idx_chunk = candidate_idx[start:end]
        cell_centers_chunk = cell_centers[start:end]
        n_chunk = end - start

        if n_chunks > 1:
            logger.info(
                "UV direction weights chunk %d/%d (%d cells)",
                ichunk,
                n_chunks,
                n_chunk,
            )

        attenuation_rays = _integrate_extinction_rays(
            tracer,
            cell_centers_chunk,
            dirs,
            alpha_uv_stack,
            progress_chunks,
        )
        tau_ext_debug = attenuation_rays.copy() if keep_debug_arrays else None

        if isotropic_outside_r_cm is None:
            outer_mask = np.zeros(n_chunk, dtype=bool)
        else:
            radii_cm = np.sqrt(np.sum(cell_centers_chunk**2, axis=1))
            outer_mask = radii_cm >= float(isotropic_outside_r_cm)

        chi_ext_dir = _attenuation_and_external_mean_inplace_parallel(
            attenuation_rays,
            float(chi_ext0),
            outer_mask,
            outer_mode_code == 2,
        )

        chi_star_dir_att = np.zeros(n_chunk, dtype=np.float64)
        chi_star_unatt = np.zeros(n_chunk, dtype=np.float64)
        tau_star = np.zeros(n_chunk, dtype=np.float64)
        k_star = np.zeros(n_chunk, dtype=np.intp)
        valid_star = np.zeros(n_chunk, dtype=bool)

        if has_star:
            tau_star = integrate_starward_rays_multi(
                tracer,
                cell_centers_chunk,
                alpha_uv_stack,
                candidate_idx=candidate_idx_chunk,
                stop_radius_cm=float(star_inner_radius_cm),
            )[:, 0]

            r_raw = np.sqrt(np.sum(cell_centers_chunk**2, axis=1))
            valid_star = r_raw > max(float(star_inner_radius_cm), 0.0)
            r_cell = np.maximum(r_raw[valid_star], 1e-30)
            F_uv = np.zeros(n_chunk, dtype=np.float64)
            F_uv[valid_star] = float(star_uv_luminosity_erg_s) / (4.0 * np.pi * r_cell**2)
            ref = max(float(star_uv_reference_energy_density), np.finfo(np.float64).tiny)
            if str(star_uv_weighting).lower() == "photon":
                chi_star_unatt = F_uv / ref
            else:
                chi_star_unatt = F_uv / (C_LIGHT * ref)
            chi_star_dir_att = chi_star_unatt * np.exp(-tau_star)

            if np.any(valid_star):
                star_dir_x = -cell_centers_chunk[valid_star, 0] / r_cell
                star_dir_y = -cell_centers_chunk[valid_star, 1] / r_cell
                star_dir_z = -cell_centers_chunk[valid_star, 2] / r_cell
                k_star[valid_star] = hp.vec2pix(
                    int(nside),
                    star_dir_x,
                    star_dir_y,
                    star_dir_z,
                )

            max_chi_star_unatt = max(max_chi_star_unatt, float(np.max(chi_star_unatt)))
            max_tau_star = max(max_tau_star, float(np.max(tau_star)))

        chi_cand = chi_arr[
            candidate_idx_chunk[:, 0],
            candidate_idx_chunk[:, 1],
            candidate_idx_chunk[:, 2],
        ]
        chi_iso = np.maximum(chi_cand - chi_ext_dir - chi_star_dir_att, 0.0)
        chi_iso_medians.append(float(np.median(chi_iso)))

        if keep_closure_diagnostics:
            chi_direct = chi_ext_dir + chi_star_dir_att
            chi_reconstructed = chi_direct + chi_iso
            closure_residual = chi_cand - chi_reconstructed
            direct_excess = np.maximum(chi_direct - chi_cand, 0.0)
            direct_to_radmc = np.divide(
                chi_direct,
                chi_cand,
                out=np.full_like(chi_direct, np.nan, dtype=np.float64),
                where=chi_cand > 0.0,
            )
            component_total = np.maximum(
                chi_reconstructed,
                np.finfo(np.float64).tiny,
            )
            f_ext = chi_ext_dir / component_total
            f_star = chi_star_dir_att / component_total
            f_iso = chi_iso / component_total
            outer_weight_overridden = outer_mask.astype(np.float64)

            closure_chunks["chi_radmc"].append(chi_cand.copy())
            closure_chunks["chi_ext_dir"].append(chi_ext_dir.copy())
            closure_chunks["chi_star_dir_att"].append(chi_star_dir_att.copy())
            closure_chunks["chi_iso"].append(chi_iso.copy())
            closure_chunks["chi_direct"].append(chi_direct.copy())
            closure_chunks["chi_reconstructed"].append(chi_reconstructed.copy())
            closure_chunks["closure_residual"].append(closure_residual.copy())
            closure_chunks["direct_excess"].append(direct_excess.copy())
            closure_chunks["direct_to_radmc"].append(direct_to_radmc.copy())
            closure_chunks["f_ext"].append(f_ext.copy())
            closure_chunks["f_star"].append(f_star.copy())
            closure_chunks["f_iso"].append(f_iso.copy())
            closure_chunks["tau_star"].append(tau_star.copy())
            closure_chunks["chi_star_unatt"].append(chi_star_unatt.copy())
            closure_chunks["outer_weight_overridden"].append(outer_weight_overridden)

        w_star_chunk = np.empty(n_chunk, dtype=np.float64)
        W_chunk = W_rays[start:end]
        _fill_uv_weights_parallel(
            W_chunk,
            attenuation_rays,
            float(chi_ext0),
            chi_ext_dir,
            chi_iso,
            chi_star_dir_att,
            valid_star,
            k_star,
            outer_mask,
            outer_mode_code,
            w_star_chunk,
        )

        if star_k_all is not None and star_w_all is not None and star_valid_all is not None:
            star_k_all[start:end] = k_star
            star_w_all[start:end] = w_star_chunk
            star_valid_all[start:end] = valid_star & (w_star_chunk > 0.0)

        if keep_debug_arrays:
            assert tau_ext_debug is not None
            uv_ext_contrib = float(chi_ext0) * np.exp(-tau_ext_debug)
            uv_star_contrib = np.zeros((n_chunk, npix), dtype=np.float64)
            if has_star:
                uv_star_contrib[np.arange(n_chunk, dtype=np.intp), k_star] = (
                    float(npix) * chi_star_dir_att
                )
            uv_iso_contrib = np.broadcast_to(chi_iso[:, None], (n_chunk, npix)).copy()
            uv_total_contrib = uv_ext_contrib + uv_star_contrib + uv_iso_contrib
            debug_chunks["chi_ext_dir"].append(chi_ext_dir.copy())
            debug_chunks["chi_star_dir_att"].append(chi_star_dir_att.copy())
            debug_chunks["chi_iso"].append(chi_iso.copy())
            debug_chunks["chi_radmc_cand"].append(chi_cand.copy())
            debug_chunks["tau_ext_rays"].append(tau_ext_debug)
            debug_chunks["uv_ext_contrib"].append(uv_ext_contrib)
            debug_chunks["uv_star_contrib"].append(uv_star_contrib)
            debug_chunks["uv_iso_contrib"].append(uv_iso_contrib)
            debug_chunks["uv_total_contrib"].append(uv_total_contrib)
            debug_chunks["tau_star"].append(tau_star.copy())
            debug_chunks["chi_star_unatt"].append(chi_star_unatt.copy())
            debug_chunks["k_star"].append(k_star.copy())

        del attenuation_rays

    debug: dict = {
        "keep_debug_arrays": bool(keep_debug_arrays),
        "keep_closure_diagnostics": bool(keep_closure_diagnostics),
        "chunk_size": int(chunk_size),
        "n_chunks": int(n_chunks),
        "median_chi_iso": float(np.median(chi_iso_medians)) if chi_iso_medians else 0.0,
    }
    if star_k_all is not None and star_w_all is not None and star_valid_all is not None:
        debug["stellar"] = {
            "k_star": star_k_all,
            "w_star": star_w_all,
            "valid_star": star_valid_all,
            "shape": tuple(int(v) for v in chi_arr.shape),
            "nside": int(nside),
            "npix": int(npix),
            "star_inner_radius_cm": float(star_inner_radius_cm),
        }
    if keep_debug_arrays:
        debug.update({
            name: np.concatenate(parts, axis=0)
            for name, parts in debug_chunks.items()
            if parts
        })
    if keep_closure_diagnostics:
        closure = {
            name: np.concatenate(parts, axis=0)
            for name, parts in closure_chunks.items()
            if parts
        }
        if closure:
            direct_to_radmc_finite = closure["direct_to_radmc"][
                np.isfinite(closure["direct_to_radmc"])
            ]
            max_direct_to_radmc = (
                float(np.max(direct_to_radmc_finite))
                if direct_to_radmc_finite.size
                else float("nan")
            )
            debug["closure_diagnostics"] = {
                "shape": tuple(int(v) for v in chi_arr.shape),
                "nside": int(nside),
                "npix": int(npix),
                "uv_product_weighting": str(star_uv_weighting).lower(),
                "outer_weight_mode": str(outer_weight_mode),
                "isotropic_outside_r_cm": (
                    None
                    if isotropic_outside_r_cm is None
                    else float(isotropic_outside_r_cm)
                ),
                "fields": closure,
                "summary": {
                    "max_direct_excess": float(np.nanmax(closure["direct_excess"])),
                    "max_direct_to_radmc": max_direct_to_radmc,
                    "max_abs_closure_residual": float(
                        np.nanmax(np.abs(closure["closure_residual"]))
                    ),
                    "median_f_ext": float(np.nanmedian(closure["f_ext"])),
                    "median_f_star": float(np.nanmedian(closure["f_star"])),
                    "median_f_iso": float(np.nanmedian(closure["f_iso"])),
                    "outer_weight_overridden_fraction": float(
                        np.nanmean(closure["outer_weight_overridden"])
                    ),
                },
            }

    if has_star:
        logger.info(
            "Stellar direct: max(chi_star_unatt)=%.3e, max(tau_star)=%.3e",
            max_chi_star_unatt,
            max_tau_star,
        )

    logger.info(
        "UV direction weights: n_cand=%d, npix=%d, "
        "chi_ext0=%.2e, L_star_uv=%.2e, "
        "median(chi_iso)=%.2e",
        n_candidates,
        npix,
        float(chi_ext0),
        float(star_uv_luminosity_erg_s),
        float(debug["median_chi_iso"]),
    )

    return W_rays, candidate_idx, dirs, cell_centers, debug
