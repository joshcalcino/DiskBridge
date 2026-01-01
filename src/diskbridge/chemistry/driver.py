from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from diskbridge._units import Quantity
from diskbridge._logging import logger
from diskbridge._config import resolve_model_config
from diskbridge.chemistry.models.pinte_switches import (
    apply_photodissociation,
    apply_photodesorption_escape,
    compute_freezeout_factor,
)
from diskbridge.chemistry.processes import (
    compute_co_photodissociation_rate,
    evolve_co_two_phase_time_dependent,
    solve_co_two_phase_steady_state,
)


WriteNumberDensityFn = Callable[[str, Quantity], None]


def compute_abundance_pinte(
    *,
    molecule: str,
    T: Quantity,
    nH: Quantity,
    chi: Optional[Quantity],
    chi_eff: Optional[Quantity],
    X0: float,
    eps: float | Quantity,
    Tfrz: Quantity,
    photodissociation: bool,
    freezeout: bool,
    photodesorption: bool,
    log_chi_over_nh_pdiss: float,
    log_chi_over_nh_pdes: float,
    eps_chi: float,
    smooth_log_chi_nH_dex: float = 0.0,
    smooth_Tfrz_K: float = 0.0,
    write_number_density: Optional[WriteNumberDensityFn] = None,
) -> Tuple[Quantity, Quantity]:
    """Pinte-style abundance switches.

    Assumes prerequisites are already satisfied:
    - Temperature `T`
    - nH field `nH`
    - UV field `chi` if photodissociation or photodesorption is enabled
    - Effective UV field `chi_eff` if self-shielding is enabled by the caller
    """

    T_K = T.to('K')
    nH_cm3 = nH.to('cm^-3')

    T_vals = T_K.magnitude
    Tfrz_val = float(Tfrz.to('K').magnitude)

    if isinstance(eps, Quantity):
        eps_val = float(eps.to('dimensionless').magnitude)
    else:
        eps_val = float(eps)

    if photodissociation or photodesorption:
        if chi_eff is None:
            chi_eff_use = chi
        else:
            chi_eff_use = chi_eff
    else:
        chi_eff_use = None

    X = Quantity(np.full_like(T_vals, float(X0), dtype=float), 'dimensionless')

    if freezeout:
        freeze_factor, mask_frz = compute_freezeout_factor(
            T_vals,
            Tfrz_val,
            eps_val,
            float(smooth_Tfrz_K),
        )
        X *= freeze_factor
        n_frz = int(np.sum(mask_frz))
        logger.info(f'Freeze-out: {n_frz} cells ({100*n_frz/X.size:.1f}%)')
    else:
        freeze_factor = np.ones_like(T_vals, dtype=float)
        mask_frz = np.zeros_like(T_vals, dtype=bool)

    chi_over_nH = None
    if chi_eff_use is not None and (photodissociation or photodesorption):
        ratio = chi_eff_use.to('dimensionless').magnitude / (nH_cm3.magnitude + eps_chi)
        chi_over_nH = np.log10(np.maximum(ratio, eps_chi))

    mask_pdes = np.zeros_like(T_vals, dtype=bool)
    if photodesorption and chi_over_nH is not None:
        log_thr_pdes = float(log_chi_over_nh_pdes)
        X, mask_pdes = apply_photodesorption_escape(
            X,
            X0=float(X0),
            freezeout=bool(freezeout),
            freeze_factor=freeze_factor,
            mask_frz=mask_frz,
            chi_over_nH=chi_over_nH,
            log_thr_pdes=log_thr_pdes,
            smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
        )
        n_pdes = int(np.sum(mask_pdes))
        logger.info(f'Photodesorption: {n_pdes} cells ({100*n_pdes/X.size:.1f}%)')

    if photodissociation and chi_over_nH is not None:
        log_thr_pdiss = float(log_chi_over_nh_pdiss)
        X, mask_pdiss = apply_photodissociation(
            X,
            chi_over_nH=chi_over_nH,
            log_thr_pdiss=log_thr_pdiss,
            smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
        )
        n_pdiss = int(np.sum(mask_pdiss))
        logger.info(f'Photodissociation: {n_pdiss} cells ({100*n_pdiss/X.size:.1f}%)')

    n_mol = X * nH_cm3
    logger.info(
        f'Computed {molecule} abundance: X_mean={np.mean(X):.2e}, n_mean={np.mean(n_mol):.2e}'
    )

    if write_number_density is not None:
        write_number_density(str(molecule).lower(), n_mol)

    return X, n_mol


def compute_co_photodissociation_rate_field(
    *,
    chi: Quantity,
    theta_co,
    k0_co,
    candidate_mask: Optional[np.ndarray] = None,
    min_rate: float = 1.0e-30,
) -> Tuple[Quantity, Optional[Quantity]]:
    return compute_co_photodissociation_rate(
        chi,
        theta_co,
        k0_co,
        candidate_mask=candidate_mask,
        min_rate=min_rate,
    )


def compute_co_steady_state(
    *,
    nH: Quantity,
    T: Quantity,
    chi: Quantity,
    theta_co,
    Xco_tot: float,
    tau_form: Quantity,
    k0_co: Quantity,
    sigma_d_per_H: float,
    E_bind: float,
    nu0: float,
    alpha_pd_ice: float,
    write_number_density: Optional[WriteNumberDensityFn] = None,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Optional[Quantity]]:
    X_co, nco_gas, nco_ice, k_pd, tau_pd = solve_co_two_phase_steady_state(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        Xco_tot=Xco_tot,
        tau_form=tau_form,
        k0_co=k0_co,
        sigma_d_per_H=sigma_d_per_H,
        E_bind=E_bind,
        nu0=nu0,
        alpha_pd_ice=alpha_pd_ice,
        min_rate=1.0e-30,
    )

    if write_number_density is not None:
        write_number_density('co', nco_gas)

    return X_co, nco_gas, nco_ice, k_pd, tau_pd


def evolve_co_time_dependent(
    *,
    nH: Quantity,
    T: Quantity,
    chi: Quantity,
    theta_co,
    t_end: Quantity,
    dt: Optional[Quantity] = None,
    Xco_tot: float,
    tau_form: Quantity,
    k0_co: Quantity,
    sigma_d_per_H: float,
    E_bind: float,
    nu0: float,
    alpha_pd_ice: float,
    Xco_gas_init: Optional[float] = None,
    Xco_ice_init: Optional[float] = None,
    write_number_density: Optional[WriteNumberDensityFn] = None,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Optional[Quantity]]:
    X_co, nco_gas, nco_ice, k_pd, tau_pd = evolve_co_two_phase_time_dependent(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        t_end=t_end,
        dt=dt,
        Xco_tot=Xco_tot,
        tau_form=tau_form,
        k0_co=k0_co,
        sigma_d_per_H=sigma_d_per_H,
        E_bind=E_bind,
        nu0=nu0,
        alpha_pd_ice=alpha_pd_ice,
        Xco_gas_init=Xco_gas_init,
        Xco_ice_init=Xco_ice_init,
        min_rate=1.0e-30,
    )

    if write_number_density is not None:
        write_number_density('co', nco_gas)

    return X_co, nco_gas, nco_ice, k_pd, tau_pd

