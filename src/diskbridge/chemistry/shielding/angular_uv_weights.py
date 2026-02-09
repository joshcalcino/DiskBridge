"""
Directional UV weighting for HEALPix-based shielding averages.

For each cell, builds a HEALPix map C_k representing the relative UV
contribution from direction k:

.. math::

    C_k = C^{\\rm ext}_k + C^{\\star}_k + C^{\\rm iso}

Components
----------
- **External direct**: :math:`C^{\\rm ext}_k = \\chi_{\\rm ext,0} \\exp(-\\tau^{\\rm ext}_k)`
  where :math:`\\tau^{\\rm ext}_k` is the dust UV optical depth along ray k
  to the domain boundary.

- **Stellar direct** (point source):
  :math:`C^{\\star}_{k} = N_{\\rm pix} \\chi^{\\star}_{\\rm dir} e^{-\\tau^\\star}`
  in the single pixel k_star containing the star direction; zero elsewhere.
  The :math:`N_{\\rm pix}` factor ensures the pixel-mean equals the correct
  scalar stellar contribution independent of NSIDE.

- **Isotropic** (RADMC residual):
  :math:`\\chi_{\\rm iso} = \\max(\\chi_{\\rm RADMC} - \\chi_{\\rm ext,dir}
  - \\chi_{\\star,\\rm dir,att},\\; 0)`,
  broadcast uniformly to all pixels.

Weights are then:

.. math::

    W_k = \\frac{C_k}{\\sum_j C_j}

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


def compute_uv_direction_weights_healpix(
    mesh,
    *,
    chi_radmc: np.ndarray,
    nside: int,
    dust_rho_bins: list[np.ndarray],
    kext_uv: np.ndarray,
    chi_ext0: float,
    star_uv_luminosity_erg_s: float,
    self_weight: float = 1.0,
    progress_chunks: int | None = None,
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
    self_weight : float, optional
        Weight of starting cell in ray integration. Default 1.0.
    progress_chunks : int or None, optional
        If set, split ray integration into chunks with logging.
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
        Diagnostic arrays: chi_ext_dir, chi_star_dir_att, chi_iso,
        chi_radmc_cand, tau_ext_rays, C_ext, C_star, C_iso, and
        optionally tau_star, chi_star_unatt.

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
        return W_rays, candidate_idx, dirs, cell_centers, {}

    # -- 2. Integrate dust density along HEALPix rays (external tau) ----------
    dust_fields: dict[str, np.ndarray] = {}
    for ibin, rho_bin in enumerate(dust_rho_bins):
        dust_fields[f"dust_bin_{ibin}"] = _as_f64(f"dust_bin_{ibin}", rho_bin)

    _, _, dust_cols = compute_column_rays_healpix(
        mesh,
        fields=dust_fields,
        nside=int(nside),
        candidate_mask=candidate_mask,
        progress_chunks=progress_chunks,
        cache_dir=cache_dir,
        self_weight=float(self_weight),
        tracer=tracer,
        dirs=dirs,
        candidate_idx=candidate_idx,
        cell_centers=cell_centers,
    )

    # -- 3. Compute tau_ext per ray -------------------------------------------
    tau_ext_rays = compute_tau_uv_from_dust_columns(dust_cols, kext_uv, nbin=nbin)

    # -- 4. External direct component -----------------------------------------
    C_ext = float(chi_ext0) * np.exp(-tau_ext_rays)  # (n_candidates, npix)
    chi_ext_dir = C_ext.mean(axis=1)  # scalar per cell

    # -- 5. Stellar direct component ------------------------------------------
    chi_star_dir_att = np.zeros(n_candidates, dtype=np.float64)
    C_star = np.zeros((n_candidates, npix), dtype=np.float64)

    debug: dict = {}

    if float(star_uv_luminosity_erg_s) > 0.0:
        # 5.1 Integrate dust along star-ward rays
        dust_fields_stack = np.ascontiguousarray(
            np.stack(
                [_as_f64(f"rho_{i}", rho) for i, rho in enumerate(dust_rho_bins)],
                axis=0,
            ),
            dtype=np.float64,
        )  # (nbin, dim0, dim1, dim2)

        star_cols = integrate_starward_rays_multi(
            tracer,
            cell_centers,
            dust_fields_stack,
            self_weight=float(self_weight),
            candidate_idx=candidate_idx,
        )  # (n_candidates, nbin)

        tau_star = np.zeros(n_candidates, dtype=np.float64)
        for ibin in range(nbin):
            tau_star += kext_uv[ibin] * star_cols[:, ibin]

        # 5.2 Compute unattenuated stellar chi per cell
        r_cell = np.sqrt(np.sum(cell_centers**2, axis=1))  # cm
        r_cell = np.maximum(r_cell, 1e-30)
        F_uv = float(star_uv_luminosity_erg_s) / (4.0 * np.pi * r_cell**2)
        chi_star_unatt = F_uv / (C_LIGHT * U_DRAINE)

        # 5.3 Attenuate
        chi_star_dir_att = chi_star_unatt * np.exp(-tau_star)

        # 5.4 Inject delta function into star pixel
        # Star direction from cell: toward origin = -cell_center / |cell_center|
        star_dir_x = -cell_centers[:, 0] / r_cell
        star_dir_y = -cell_centers[:, 1] / r_cell
        star_dir_z = -cell_centers[:, 2] / r_cell
        k_star = hp.vec2pix(int(nside), star_dir_x, star_dir_y, star_dir_z)

        # Vectorised injection: C_star[i, k_star[i]] = npix * chi_star_dir_att[i]
        idx_cells = np.arange(n_candidates)
        C_star[idx_cells, k_star] = npix * chi_star_dir_att

        debug["tau_star"] = tau_star
        debug["chi_star_unatt"] = chi_star_unatt
        debug["k_star"] = k_star

        logger.info(
            "Stellar direct: max(chi_star_unatt)=%.3e, max(tau_star)=%.3e",
            float(np.max(chi_star_unatt)),
            float(np.max(tau_star)),
        )

    # -- 6. Isotropic component (RADMC residual) ------------------------------
    chi_cand = chi_arr[
        candidate_idx[:, 0], candidate_idx[:, 1], candidate_idx[:, 2]
    ]
    chi_iso = np.maximum(chi_cand - chi_ext_dir - chi_star_dir_att, 0.0)
    C_iso = np.broadcast_to(chi_iso[:, None], (n_candidates, npix)).copy()

    # -- 7. Build total and normalise to weights ------------------------------
    C_total = C_ext + C_star + C_iso
    sumC = C_total.sum(axis=1)  # (n_candidates,)

    uniform = 1.0 / float(npix)
    mask_nonzero = sumC > 0.0
    W_rays = np.full((n_candidates, npix), uniform, dtype=np.float64)
    W_rays[mask_nonzero] = C_total[mask_nonzero] / sumC[mask_nonzero, None]

    # -- 8. Diagnostics -------------------------------------------------------
    debug.update({
        "chi_ext_dir": chi_ext_dir,
        "chi_star_dir_att": chi_star_dir_att,
        "chi_iso": chi_iso,
        "chi_radmc_cand": chi_cand,
        "tau_ext_rays": tau_ext_rays,
        "C_ext": C_ext,
        "C_star": C_star,
        "C_iso": C_iso,
    })

    logger.info(
        "UV direction weights: n_cand=%d, npix=%d, "
        "chi_ext0=%.2e, L_star_uv=%.2e, "
        "median(chi_iso)=%.2e",
        n_candidates,
        npix,
        float(chi_ext0),
        float(star_uv_luminosity_erg_s),
        float(np.median(chi_iso)),
    )

    return W_rays, candidate_idx, dirs, cell_centers, debug
