from __future__ import annotations

from typing import Any, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import diskbridge
from diskbridge._units import Quantity
from diskbridge._logging import logger
from diskbridge._constants import (
    T_FRZ, EPS_FRZ, LOG_CHI_OVER_NH_PDISS, LOG_CHI_OVER_NH_PDES, EPS_CHI
)
from diskbridge.chemistry.types import ChemistryResult


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, 0.0, 1.0)
    return x_clipped * x_clipped * (3.0 - 2.0 * x_clipped)


def compute_freezeout_factor(
    T_vals_K: np.ndarray,
    Tfrz_K: float,
    eps: float,
    smooth_Tfrz_K: float,
) -> Tuple[np.ndarray, np.ndarray]:
    freeze_factor = np.ones_like(T_vals_K, dtype=float)

    if smooth_Tfrz_K > 0.0:
        half_T = 0.5 * float(smooth_Tfrz_K)
        T_low = float(Tfrz_K) - half_T
        T_high = float(Tfrz_K) + half_T

        mask_cold = T_vals_K <= T_low
        mask_warm = T_vals_K >= T_high
        mask_mid = (~mask_cold) & (~mask_warm)

        freeze_factor[mask_cold] = float(eps)
        freeze_factor[mask_warm] = 1.0
        if np.any(mask_mid):
            t_mid = (T_vals_K[mask_mid] - T_low) / (2.0 * half_T)
            s_mid = _smoothstep01(t_mid)
            freeze_factor[mask_mid] = float(eps) + (1.0 - float(eps)) * s_mid

        mask_frz = freeze_factor < 1.0 - 1e-12
    else:
        mask_frz = T_vals_K < float(Tfrz_K)
        freeze_factor[mask_frz] = float(eps)

    return freeze_factor, mask_frz


def apply_photodesorption_escape(
    X: Any,
    *,
    X0: float,
    freezeout: bool,
    freeze_factor: Optional[np.ndarray],
    mask_frz: Optional[np.ndarray],
    chi_over_nH: np.ndarray,
    log_thr_pdes: float,
    smooth_log_chi_nH_dex: float,
) -> Tuple[Any, np.ndarray]:
    mask_pdes = np.zeros_like(chi_over_nH, dtype=bool)

    if smooth_log_chi_nH_dex > 0.0:
        half_dex = 0.5 * float(smooth_log_chi_nH_dex)
        lower = float(log_thr_pdes) - half_dex
        width = max(float(smooth_log_chi_nH_dex), 1e-30)
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

        mask_pdes = chi_over_nH > float(log_thr_pdes)
    else:
        mask_pdes = chi_over_nH > float(log_thr_pdes)
        if freezeout:
            if mask_frz is None:
                raise ValueError("mask_frz is required when freezeout=True")
            unfreeze_mask = mask_pdes & mask_frz
            X[unfreeze_mask] = float(X0)

    return X, mask_pdes


def apply_photodissociation(
    X: Any,
    *,
    chi_over_nH: np.ndarray,
    log_thr_pdiss: float,
    smooth_log_chi_nH_dex: float,
) -> Tuple[Any, np.ndarray]:
    if smooth_log_chi_nH_dex > 0.0:
        half_dex = 0.5 * float(smooth_log_chi_nH_dex)
        lower = float(log_thr_pdiss) - half_dex
        width = max(float(smooth_log_chi_nH_dex), 1e-30)
        t = (chi_over_nH - lower) / width
        w_pdiss = _smoothstep01(t)
        kill_factor = 1.0 - w_pdiss
        X *= kill_factor
        mask_pdiss = chi_over_nH > float(log_thr_pdiss)
    else:
        mask_pdiss = chi_over_nH > float(log_thr_pdiss)
        X[mask_pdiss] = 0.0

    return X, mask_pdiss


def compute_abundance_pinte(
    *,
    molecule: str,
    T: Quantity,
    nH: Quantity,
    chi: Optional[Quantity],
    chi_eff: Optional[Quantity],
    X0: float,
    photodissociation: bool,
    freezeout: bool,
    photodesorption: bool,
    smooth_log_chi_nH_dex: float = 0.0,
    smooth_Tfrz_K: float = 0.0,
) -> Tuple[Quantity, Quantity]:
    """Pinte-style abundance switches using constants from config.

    Assumes prerequisites are already satisfied:
    - Temperature `T`
    - nH field `nH`
    - UV field `chi` if photodissociation or photodesorption is enabled
    - Effective UV field `chi_eff` if self-shielding is enabled by the caller
    """
    T_K = T.to('K')
    nH_cm3 = nH.to('cm^-3')
    T_vals = T_K.magnitude

    if photodissociation or photodesorption:
        chi_eff_use = chi_eff if chi_eff is not None else chi
    else:
        chi_eff_use = None

    X = Quantity(np.full_like(T_vals, float(X0), dtype=float), 'dimensionless')

    if freezeout:
        freeze_factor, mask_frz = compute_freezeout_factor(
            T_vals, T_FRZ, EPS_FRZ, float(smooth_Tfrz_K),
        )
        X *= freeze_factor
        n_frz = int(np.sum(mask_frz))
        logger.info(f'Freeze-out: {n_frz} cells ({100*n_frz/X.size:.1f}%)')
    else:
        freeze_factor = np.ones_like(T_vals, dtype=float)
        mask_frz = np.zeros_like(T_vals, dtype=bool)

    chi_over_nH = None
    if chi_eff_use is not None and (photodissociation or photodesorption):
        ratio = chi_eff_use.to('dimensionless').magnitude / (nH_cm3.magnitude + EPS_CHI)
        chi_over_nH = np.log10(np.maximum(ratio, EPS_CHI))

    mask_pdes = np.zeros_like(T_vals, dtype=bool)
    if photodesorption and chi_over_nH is not None:
        X, mask_pdes = apply_photodesorption_escape(
            X,
            X0=float(X0),
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
    X_mean = float(X.to('dimensionless').magnitude.mean())
    n_mean = float(n_mol.to('cm^-3').magnitude.mean())
    logger.info(f'Computed {molecule} abundance: X_mean={X_mean:.2e}, n_mean={n_mean:.2e}')
    return X, n_mol


def run_pinte_switches(rad: 'RadModel', config: dict) -> ChemistryResult:
    molecule = config.get('molecule', 'co')
    X0 = config.get('X0', float(diskbridge.params.abundance))
    photodissociation = config.get('photodissociation', diskbridge.params.photodissociation)
    freezeout = config.get('freezeout', diskbridge.params.freezeout)
    photodesorption = config.get('photodesorption', diskbridge.params.photodesorption)
    smooth_log_chi_nH_dex = config.get('smooth_log_chi_nH_dex', 0.0)
    smooth_Tfrz_K = config.get('smooth_Tfrz_K', 0.0)

    T = rad.ensure_temperature()
    nH = rad.ensure_nH()

    needs_chi = bool(photodissociation or photodesorption)
    chi = rad.ensure_chi() if needs_chi else None

    X, n_mol = compute_abundance_pinte(
        molecule=str(molecule).lower(),
        T=T,
        nH=nH,
        chi=chi,
        chi_eff=None,
        X0=float(X0),
        photodissociation=bool(photodissociation),
        freezeout=bool(freezeout),
        photodesorption=bool(photodesorption),
        smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
        smooth_Tfrz_K=float(smooth_Tfrz_K),
    )

    mol_lower = str(molecule).lower()

    return ChemistryResult(
        abundances={mol_lower: X},
        number_densities={mol_lower: n_mol},
        fields={},
        meta={'model': 'pinte_switches'},
    )


__all__ = [
    'run_pinte_switches',
]
