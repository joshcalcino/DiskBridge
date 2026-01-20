from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import diskbridge
from diskbridge._logging import logger
from diskbridge._units import Quantity
from diskbridge._constants import (
    T_FRZ, EPS_FRZ, LOG_CHI_OVER_NH_PDISS, LOG_CHI_OVER_NH_PDES, EPS_CHI
)
from diskbridge.chemistry.types import ChemistryResult


def run_pinte_switches(rad: 'RadModel', config: dict) -> ChemistryResult:
    molecule = config.get('molecule', 'co')
    X0 = config.get('X0', float(diskbridge.params.abundance))
    photodissociation = config.get('photodissociation', diskbridge.params.photodissociation)
    freezeout = config.get('freezeout', diskbridge.params.freezeout)
    photodesorption = config.get('photodesorption', diskbridge.params.photodesorption)
    t_frz = config.get('t_frz', T_FRZ)
    eps_frz = config.get('eps_frz', EPS_FRZ)
    eps_chi = config.get('eps_chi', EPS_CHI)
    smooth_log_chi_nH_dex = config.get('smooth_log_chi_nH_dex', 0.0)
    smooth_Tfrz_K = config.get('smooth_Tfrz_K', 0.0)

    T = rad.ensure_temperature()
    nH = rad.ensure_nH()

    needs_chi = bool(photodissociation or photodesorption)
    chi = rad.ensure_chi() if needs_chi else None

    T_K = T.to('K').magnitude
    nH_cm3 = nH.to('cm^-3').magnitude
    chi_dim = chi.to('dimensionless').magnitude if chi is not None else None

    X, n_mol = compute_abundance_pinte(
        molecule=str(molecule).lower(),
        T_K=T_K,
        nH_cm3=nH_cm3,
        chi=chi_dim,
        chi_eff=None,
        X0=float(X0),
        photodissociation=bool(photodissociation),
        freezeout=bool(freezeout),
        photodesorption=bool(photodesorption),
        t_frz=float(t_frz),
        eps_frz=float(eps_frz),
        eps_chi=float(eps_chi),
        smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
        smooth_Tfrz_K=float(smooth_Tfrz_K),
    )

    mol_lower = str(molecule).lower()

    Xq = Quantity(X, 'dimensionless')
    nq = Quantity(n_mol, 'cm^-3')

    return ChemistryResult(
        abundances={mol_lower: Xq},
        number_densities={mol_lower: nq},
        fields={},
        meta={'model': 'pinte_switches'},
    )


def compute_abundance_pinte(
    *,
    molecule: str,
    T_K: np.ndarray,
    nH_cm3: np.ndarray,
    chi: Optional[np.ndarray],
    chi_eff: Optional[np.ndarray],
    X0: float,
    photodissociation: bool,
    freezeout: bool,
    photodesorption: bool,
    t_frz: float,
    eps_frz: float,
    eps_chi: float,
    smooth_log_chi_nH_dex: float = 0.0,
    smooth_Tfrz_K: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pinte-style abundance switches using constants from config.

    Assumes prerequisites are already satisfied:
    - Temperature `T_K` in K
    - nH field `nH_cm3` in cm^-3
    - UV field `chi` (dimensionless) if photodissociation or photodesorption is enabled
    - Effective UV field `chi_eff` (dimensionless) if enabled by the caller
    """
    T_vals = T_K

    if T_vals.shape != nH_cm3.shape:
        raise ValueError('T_K and nH_cm3 must have the same shape')

    X = np.full_like(T_vals, X0, dtype=float)

    if freezeout:
        freeze_factor, mask_frz = compute_freezeout_factor(
            T_vals, float(t_frz), float(eps_frz), float(smooth_Tfrz_K),
        )
        X *= freeze_factor
        n_frz = int(np.sum(mask_frz))
        logger.info(f'Freeze-out: {n_frz} cells ({100*n_frz/X.size:.1f}%)')
    else:
        freeze_factor = np.ones_like(T_vals, dtype=float)
        mask_frz = np.zeros_like(T_vals, dtype=bool)

    chi_over_nH = None
    if photodissociation or photodesorption:
        chi_eff_use = chi_eff if chi_eff is not None else chi
    else:
        chi_eff_use = None

    if chi_eff_use is not None and (photodissociation or photodesorption):
        if chi_eff_use.shape != nH_cm3.shape:
            raise ValueError('chi/chi_eff must have the same shape as nH_cm3')
        ratio = chi_eff_use / (nH_cm3 + float(eps_chi))
        chi_over_nH = np.log10(np.maximum(ratio, float(eps_chi)))

    mask_pdes = np.zeros_like(T_vals, dtype=bool)
    if photodesorption and chi_over_nH is not None:
        X, mask_pdes = apply_photodesorption_escape(
            X,
            X0=X0,
            freezeout=bool(freezeout),
            freeze_factor=freeze_factor, 
            mask_frz=mask_frz,
            chi_over_nH=chi_over_nH,
            log_thr_pdes=LOG_CHI_OVER_NH_PDES,
            smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
        )
        n_pdes = int(np.sum(mask_pdes))
        logger.info(f'Photodesorption: {n_pdes} cells ({100*n_pdes/X.size:.1f}%)')

    if photodissociation and chi_over_nH is not None:
        X, mask_pdiss = apply_photodissociation(
            X,
            chi_over_nH=chi_over_nH,
            log_thr_pdiss=LOG_CHI_OVER_NH_PDISS,
            smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
        )
        n_pdiss = int(np.sum(mask_pdiss))
        logger.info(f'Photodissociation: {n_pdiss} cells ({100*n_pdiss/X.size:.1f}%)')

    n_mol = X * nH_cm3
    X_mean = float(np.mean(X))
    n_mean = float(np.mean(n_mol))
    logger.info(f'Computed {molecule} abundance: X_mean={X_mean:.2e}, n_mean={n_mean:.2e}')
    return X, n_mol


####################################################
# COMPONENTS OF THE PINTE ET AL. 2017 SWITCH MODEL #
####################################################

def compute_freezeout_factor(
    T_vals: np.ndarray,
    Tfrz: float,
    eps: float,
    smooth_Tfrz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    freeze_factor = np.ones_like(T_vals, dtype=float)

    if smooth_Tfrz > 0.0:
        half_T = 0.5 * smooth_Tfrz
        T_low = Tfrz - half_T
        T_high = Tfrz + half_T

        mask_cold = T_vals <= T_low
        mask_warm = T_vals >= T_high
        mask_mid = (~mask_cold) & (~mask_warm)

        freeze_factor[mask_cold] = eps
        freeze_factor[mask_warm] = 1.0
        if np.any(mask_mid):
            t_mid = (T_vals[mask_mid] - T_low) / (2.0 * half_T)
            s_mid = _smoothstep01(t_mid)
            freeze_factor[mask_mid] = eps + (1.0 - eps) * s_mid

        mask_frz = freeze_factor < 1.0 - 1e-12
    else:
        mask_frz = T_vals < Tfrz
        freeze_factor[mask_frz] = eps

    return freeze_factor, mask_frz


def apply_photodesorption_escape(
    X: np.ndarray,
    *,
    X0: float,
    freezeout: bool,
    freeze_factor: Optional[np.ndarray],
    mask_frz: Optional[np.ndarray],
    chi_over_nH: np.ndarray,
    log_thr_pdes: float,
    smooth_log_chi_nH_dex: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask_pdes = np.zeros_like(chi_over_nH, dtype=bool)

    if smooth_log_chi_nH_dex > 0.0:
        half_dex = 0.5 * smooth_log_chi_nH_dex
        lower = log_thr_pdes - half_dex
        width = max(smooth_log_chi_nH_dex, 1e-30)
        t = (chi_over_nH - lower) / width
        w = _smoothstep01(t)

        if freezeout:
            if freeze_factor is None:
                raise ValueError("freeze_factor is required when freezeout=True")

            has_freeze = freeze_factor < 1.0 - 1e-12
            if np.any(has_freeze):
                w_eff = w * has_freeze
                F = freeze_factor
                F_new = F + w_eff * (1.0 - F)
                ratio_F = np.ones_like(F, dtype=float)
                positive = F > 0.0
                ratio_F[positive] = F_new[positive] / F[positive]
                X *= ratio_F

        mask_pdes = chi_over_nH > log_thr_pdes
    else:
        mask_pdes = chi_over_nH > log_thr_pdes
        if freezeout:
            if mask_frz is None:
                raise ValueError("mask_frz is required when freezeout=True")
            unfreeze_mask = mask_pdes & mask_frz
            X[unfreeze_mask] = float(X0)

    return X, mask_pdes


def apply_photodissociation(
    X: np.ndarray,
    *,
    chi_over_nH: np.ndarray,
    log_thr_pdiss: float,
    smooth_log_chi_nH_dex: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if smooth_log_chi_nH_dex > 0.0:
        half_dex = 0.5 * smooth_log_chi_nH_dex
        lower = log_thr_pdiss - half_dex
        width = max(smooth_log_chi_nH_dex, 1e-30)
        t = (chi_over_nH - lower) / width
        w_pdiss = _smoothstep01(t)
        kill_factor = 1.0 - w_pdiss
        X *= kill_factor
        mask_pdiss = chi_over_nH > log_thr_pdiss
    else:
        mask_pdiss = chi_over_nH > log_thr_pdiss
        X[mask_pdiss] = 0.0

    return X, mask_pdiss

# UTIL FUNCTIONS 

def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, 0.0, 1.0)
    return x_clipped * x_clipped * (3.0 - 2.0 * x_clipped)
