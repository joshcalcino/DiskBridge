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

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import healpy as hp

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
from diskbridge.chemistry.shielding.dust_uv_tau import compute_tau_uv_from_dust_columns
from diskbridge.chemistry.shielding.healpix_columns import (
    _as_f64,
    _prepare_healpix_geometry,
    compute_column_rays_healpix,
)
from diskbridge.chemistry.shielding.healpix_utils import integrate_starward_rays_multi


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

    # Per-cell, per-pixel dense temporaries held inside one chunk:
    # dust columns for each bin, tau_ext, uv_ext, and a little headroom for
    # normalization/copy temporaries.  The final W_rays array is allocated once
    # outside this budget.
    bytes_per_candidate = int(np.dtype(np.float64).itemsize) * int(npix) * (int(nbin) + 4)
    return max(1, min(int(n_candidates), int(budget_bytes // max(bytes_per_candidate, 1))))


def _apply_outer_weights_to_chunk(
    W_chunk: np.ndarray,
    tau_ext_rays: np.ndarray,
    cell_centers_chunk: np.ndarray,
    *,
    isotropic_outside_r_cm: float,
    mode: str,
) -> None:
    radii_cm = np.sqrt(np.sum(np.asarray(cell_centers_chunk, dtype=float) ** 2, axis=1))
    outer_mask = radii_cm >= float(isotropic_outside_r_cm)
    if not np.any(outer_mask):
        return

    mode = str(mode).strip().lower()
    if mode == "uniform":
        W_chunk[outer_mask, :] = 1.0 / float(W_chunk.shape[1])
        return
    if mode != "tau":
        raise ValueError(
            "segmented_outer_weight_mode must be 'tau' or 'uniform', "
            f"got {mode!r}"
        )

    tau_outer = np.asarray(tau_ext_rays[outer_mask], dtype=float)
    tau_outer = np.clip(tau_outer, 0.0, 700.0)
    raw = np.exp(-tau_outer)
    sums = np.sum(raw, axis=1, keepdims=True)
    outer_idx = np.where(outer_mask)[0]
    valid = sums[:, 0] > 0.0
    if np.any(valid):
        W_chunk[outer_idx[valid], :] = raw[valid, :] / sums[valid, :]
    if np.any(~valid):
        W_chunk[outer_idx[~valid], :] = 1.0 / float(W_chunk.shape[1])


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
    dust_rho_bins: list[np.ndarray],
    kext_uv: np.ndarray,
    chi_ext0: float,
    star_uv_luminosity_erg_s: float,
    star_uv_reference_energy_density: float = U_DRAINE,
    star_uv_weighting: str = "energy",
    progress_chunks: int | None = None,
    chunk_size: int | None = None,
    memory_budget_gib: float | None = None,
    keep_debug_arrays: bool | None = None,
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
        only small scalar summaries are returned.

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

    chi_arr = _as_f64("chi_radmc", chi_radmc)
    nbin = len(dust_rho_bins)

    candidate_mask = np.ones(chi_arr.shape, dtype=bool)

    # -- 1. Prepare HEALPix geometry -----------------------------------------
    tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
        mesh,
        nside=int(nside),
        candidate_mask=candidate_mask,
        cache_dir=cache_dir,
    )

    n_candidates = int(candidate_idx.shape[0])
    npix = int(dirs.shape[0])

    if n_candidates == 0:
        W_rays = np.full((0, npix), 1.0 / npix, dtype=np.float64)
        return W_rays, candidate_idx, dirs, cell_centers, {"keep_debug_arrays": False}

    # -- 2. Integrate dust density along HEALPix rays (external tau) ----------
    dust_fields: dict[str, np.ndarray] = {}
    for ibin, rho_bin in enumerate(dust_rho_bins):
        dust_fields[f"dust_bin_{ibin}"] = _as_f64(f"dust_bin_{ibin}", rho_bin)

    if keep_debug_arrays is None:
        keep_debug_arrays = False

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
    uniform = 1.0 / float(npix)
    W_rays = np.full((n_candidates, npix), uniform, dtype=np.float64)

    dust_fields_stack = None
    if float(star_uv_luminosity_erg_s) > 0.0:
        dust_fields_stack = np.ascontiguousarray(
            np.stack(
                [_as_f64(f"rho_{i}", rho) for i, rho in enumerate(dust_rho_bins)],
                axis=0,
            ),
            dtype=np.float64,
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

        _, _, dust_cols = compute_column_rays_healpix(
            mesh,
            fields=dust_fields,
            nside=int(nside),
            candidate_mask=None,
            progress_chunks=progress_chunks,
            cache_dir=None,
            tracer=tracer,
            dirs=dirs,
            candidate_idx=candidate_idx_chunk,
            cell_centers=cell_centers_chunk,
        )

        tau_ext_rays = compute_tau_uv_from_dust_columns(dust_cols, kext_uv, nbin=nbin)
        uv_ext_contrib = float(chi_ext0) * np.exp(-tau_ext_rays)
        chi_ext_dir = uv_ext_contrib.mean(axis=1)

        chi_star_dir_att = np.zeros(n_chunk, dtype=np.float64)
        chi_star_unatt = np.zeros(n_chunk, dtype=np.float64)
        tau_star = np.zeros(n_chunk, dtype=np.float64)
        k_star = np.zeros(n_chunk, dtype=np.intp)

        if dust_fields_stack is not None:
            star_cols = integrate_starward_rays_multi(
                tracer,
                cell_centers_chunk,
                dust_fields_stack,
                candidate_idx=candidate_idx_chunk,
            )

            for ibin in range(nbin):
                tau_star += kext_uv[ibin] * star_cols[:, ibin]

            r_cell = np.sqrt(np.sum(cell_centers_chunk**2, axis=1))
            r_cell = np.maximum(r_cell, 1e-30)
            F_uv = float(star_uv_luminosity_erg_s) / (4.0 * np.pi * r_cell**2)
            ref = max(float(star_uv_reference_energy_density), np.finfo(np.float64).tiny)
            if str(star_uv_weighting).lower() == "photon":
                chi_star_unatt = F_uv / ref
            else:
                chi_star_unatt = F_uv / (C_LIGHT * ref)
            chi_star_dir_att = chi_star_unatt * np.exp(-tau_star)

            star_dir_x = -cell_centers_chunk[:, 0] / r_cell
            star_dir_y = -cell_centers_chunk[:, 1] / r_cell
            star_dir_z = -cell_centers_chunk[:, 2] / r_cell
            k_star = hp.vec2pix(int(nside), star_dir_x, star_dir_y, star_dir_z)

            max_chi_star_unatt = max(max_chi_star_unatt, float(np.max(chi_star_unatt)))
            max_tau_star = max(max_tau_star, float(np.max(tau_star)))

        chi_cand = chi_arr[
            candidate_idx_chunk[:, 0],
            candidate_idx_chunk[:, 1],
            candidate_idx_chunk[:, 2],
        ]
        chi_iso = np.maximum(chi_cand - chi_ext_dir - chi_star_dir_att, 0.0)
        chi_iso_medians.append(float(np.median(chi_iso)))

        sum_uv_contrib = (
            uv_ext_contrib.sum(axis=1)
            + float(npix) * chi_iso
            + float(npix) * chi_star_dir_att
        )
        mask_nonzero = sum_uv_contrib > 0.0

        W_chunk = W_rays[start:end]
        W_chunk[:, :] = uniform
        if np.any(mask_nonzero):
            rows_nonzero = np.where(mask_nonzero)[0]
            W_chunk[rows_nonzero, :] = uv_ext_contrib[rows_nonzero]
            W_chunk[rows_nonzero, :] += chi_iso[rows_nonzero, None]
            W_chunk[rows_nonzero, :] /= sum_uv_contrib[rows_nonzero, None]
            if dust_fields_stack is not None:
                W_chunk[rows_nonzero, k_star[rows_nonzero]] += (
                    float(npix)
                    * chi_star_dir_att[rows_nonzero]
                    / sum_uv_contrib[rows_nonzero]
                )

        if isotropic_outside_r_cm is not None:
            _apply_outer_weights_to_chunk(
                W_chunk,
                tau_ext_rays,
                cell_centers_chunk,
                isotropic_outside_r_cm=float(isotropic_outside_r_cm),
                mode=outer_weight_mode,
            )

        if keep_debug_arrays:
            uv_star_contrib = np.zeros((n_chunk, npix), dtype=np.float64)
            if dust_fields_stack is not None:
                uv_star_contrib[np.arange(n_chunk, dtype=np.intp), k_star] = (
                    float(npix) * chi_star_dir_att
                )
            uv_iso_contrib = np.broadcast_to(chi_iso[:, None], (n_chunk, npix)).copy()
            uv_total_contrib = uv_ext_contrib + uv_star_contrib + uv_iso_contrib
            debug_chunks["chi_ext_dir"].append(chi_ext_dir.copy())
            debug_chunks["chi_star_dir_att"].append(chi_star_dir_att.copy())
            debug_chunks["chi_iso"].append(chi_iso.copy())
            debug_chunks["chi_radmc_cand"].append(chi_cand.copy())
            debug_chunks["tau_ext_rays"].append(tau_ext_rays.copy())
            debug_chunks["uv_ext_contrib"].append(uv_ext_contrib.copy())
            debug_chunks["uv_star_contrib"].append(uv_star_contrib)
            debug_chunks["uv_iso_contrib"].append(uv_iso_contrib)
            debug_chunks["uv_total_contrib"].append(uv_total_contrib)
            debug_chunks["tau_star"].append(tau_star.copy())
            debug_chunks["chi_star_unatt"].append(chi_star_unatt.copy())
            debug_chunks["k_star"].append(k_star.copy())

    debug: dict = {
        "keep_debug_arrays": bool(keep_debug_arrays),
        "chunk_size": int(chunk_size),
        "n_chunks": int(n_chunks),
        "median_chi_iso": float(np.median(chi_iso_medians)) if chi_iso_medians else 0.0,
    }
    if keep_debug_arrays:
        debug.update({
            name: np.concatenate(parts, axis=0)
            for name, parts in debug_chunks.items()
            if parts
        })

    if dust_fields_stack is not None:
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
