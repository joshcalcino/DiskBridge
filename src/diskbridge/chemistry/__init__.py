from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Dict

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np

import diskbridge
from diskbridge._units import Quantity
from diskbridge._logging import logger
from diskbridge.chemistry.api import run_chemistry, run_thermochemistry
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.shielding.uv_boundary import find_uv_boundary_radius
from diskbridge.chemistry.tracers import compute_chem_age
from diskbridge.chemistry.models.co_two_phase import compute_boundary_co_ic
from diskbridge.chemistry.constants import (
    K0_CO_DEFAULT,
    TAU_CO_FORM_DEFAULT,
    SIGMA_D_PER_H_DEFAULT,
    E_BIND_CO_DEFAULT,
    NU0_CO_DEFAULT,
    ALPHA_PD_ICE_DEFAULT,
)
from diskbridge.chemistry.tau_form import compute_tau_form_co
from diskbridge.chemistry.processes import evolve_co_two_phase_time_dependent
from diskbridge.model.profiles import compute_volume_weighted_mean_radial_profile, find_r_split


def evolve_co_time_dependent_infall_age(
    rad: "RadModel",
    *,
    outer_age: Quantity = Quantity(1e5, "yr"),
    disc_age: Quantity = Quantity(1e6, "yr"),
    tol_chi: float = 0.01,
    window_fraction: float = 0.10,
    shell_cells: int = 2,
    disc_mask: Optional[np.ndarray] = None,
    vr_field: str = "vr",
    dt: None = None,
    skip_shielding: bool = False,
    nside: int = 4,
    b_kms: float = 0.3,
    Xco_tot: float = 1.0e-4,
    tau_form: Optional[Quantity] = None,
    k0_co: Quantity = K0_CO_DEFAULT,
    sigma_d_per_H: float = SIGMA_D_PER_H_DEFAULT,
    E_bind: float = E_BIND_CO_DEFAULT,
    nu0: float = NU0_CO_DEFAULT,
    alpha_pd_ice: float = ALPHA_PD_ICE_DEFAULT,
    eps_vr: float = 1e-30,
) -> Tuple[Quantity, Quantity, Quantity, Dict]:
    """One-call CO two-phase chemistry with UV boundary and synthetic infall ages.
    
    Complete workflow that:
    - Finds UV boundary radius where stellar UV exceeds background by tol_chi
    - Computes boundary initial conditions by solving chemistry at outer_age
    - Builds per-cell ages: outer (fixed), disc (fixed), infall (radial fall time)
    - Runs time-dependent two-phase CO chemistry with per-cell ages and ICs
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper with radiative transfer outputs and velocity fields
    outer_age : Quantity, optional
        Fixed age for outer region (r >= R_boundary), default 1e5 yr
    disc_age : Quantity, optional
        Fixed age for disc region, default 1e6 yr
    tol_chi : float, optional
        UV boundary tolerance (fractional), default 0.01 (1%)
    window_fraction : float, optional
        Fraction of domain for background UV window, default 0.10
    shell_cells : int, optional
        Number of cells for boundary IC shell, default 2
    disc_mask : ndarray, optional
        Boolean disc mask. If None, uses rad.model.disk.mask
    vr_field : str, optional
        Radial velocity field name, default "vr"
    dt : None
        Must be None (closed-form solver with per-cell t_end)
    skip_shielding : bool, optional
        Skip CO shielding computation, default False
    nside : int, optional
        HEALPix nside for shielding, default 4
    b_kms : float, optional
        Doppler parameter in km/s, default 0.3
    Xco_tot : float, optional
        Total CO abundance, default 1e-4
    tau_form : Quantity, optional
        Formation timescale (auto-computed if None)
    k0_co : Quantity, optional
        Photodissociation rate coefficient
    sigma_d_per_H : float, optional
        Dust cross-section per H
    E_bind : float, optional
        Binding energy in K
    nu0 : float, optional
        Attempt frequency in Hz
    alpha_pd_ice : float, optional
        Photodesorption yield
    eps_vr : float, optional
        Velocity floor in cm/s, default 1e-30
        
    Returns
    -------
    X_co : Quantity
        Total CO abundance field
    nco_gas : Quantity
        CO gas number density
    nco_ice : Quantity
        CO ice number density
    meta : dict
        Metadata with keys:
        - 'R_boundary': Boundary radius (Quantity)
        - 'R_boundary_info': Segmentation info dict
        - 'Xco_gas0': Boundary gas abundance (float)
        - 'Xco_ice0': Boundary ice abundance (float)
        - 'masks': Dict with 'outer', 'disc', 'infall', 'shell' masks
        - 't_end': Per-cell age field (Quantity)
        
    Raises
    ------
    ValueError
        If dt is not None, or if disc_mask unavailable
    KeyError
        If velocity field missing
        
    Examples
    --------
    >>> from diskbridge.chemistry import evolve_co_time_dependent_infall_age
    >>> from diskbridge._units import Quantity
    >>> 
    >>> disc_mask = rad.model.disk.mask.data.magnitude.astype(bool)
    >>> X_co, nco_gas, nco_ice, meta = evolve_co_time_dependent_infall_age(
    ...     rad,
    ...     disc_mask=disc_mask,
    ...     outer_age=Quantity(1e5, "yr"),
    ...     disc_age=Quantity(1e6, "yr"),
    ... )
    >>> print("UV boundary:", meta["R_boundary"])
    """
    if dt is not None:
        raise ValueError("This API uses per-cell t_end; dt must be None (closed-form solver).")
    
    logger.info("=" * 60)
    logger.info("CO two-phase with infall age workflow")
    logger.info("=" * 60)
    
    T = rad.ensure_temperature()
    nH = rad.ensure_nH()
    chi = rad.ensure_chi()
    
    if skip_shielding:
        theta_co = Quantity(np.ones_like(chi.magnitude), "dimensionless")
    else:
        if rad.theta_co is None:
            from diskbridge.chemistry.shielding.healpix_columns import compute_co_shielding_healpix
            from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
            
            logger.info("Computing CO shielding...")
            visser = VisserShielding(b_kms=float(b_kms))
            theta_co, chi_eff = compute_co_shielding_healpix(
                mesh=rad.model.mesh,
                nH=nH,
                chi=chi,
                visser=visser,
                nCO=None,
                nH2=None,
                nside=int(nside),
                b_kms=float(b_kms),
                Xco_guess=float(Xco_tot),
                XH2_guess=0.5,
                progress_chunks=None,
            )
            rad.theta_co = theta_co
            rad.chi_eff = chi_eff
        else:
            theta_co = rad.theta_co
    
    if disc_mask is None:
        if getattr(rad.model, "disk", None) is not None and getattr(rad.model.disk, "mask", None) is not None:
            disc_mask = rad.model.disk.mask.data.magnitude.astype(bool)
            logger.info("Using rad.model.disk.mask")
        else:
            raise ValueError("disc_mask not provided and rad.model.disk.mask is missing.")
    
    logger.info(f"Disc cells: {np.sum(disc_mask)} / {disc_mask.size}")
    
    logger.info("Finding UV boundary radius...")
    r_au, chi_prof = compute_volume_weighted_mean_radial_profile(rad.model, "chi")
    _, T_prof = compute_volume_weighted_mean_radial_profile(rad.model, "temperature")
    r_edges_au = rad.model.mesh.edges("r").to("au").magnitude
    
    Rb_au, Rb_info = find_r_split(
        r_au=r_au,
        r_edges_au=r_edges_au,
        chi_profile=chi_prof,
        T_profile=T_prof,
        tol_chi=float(tol_chi),
        tol_T=1e9,
        window_fraction=float(window_fraction),
        r_clip_min_au=0.1,
    )
    Rb = Quantity(Rb_au, "au")
    logger.info(f"  UV boundary: R_b = {Rb:.3g}")
    
    logger.info("Computing boundary initial conditions...")
    r_centers_au = rad.model.mesh.centers("r").to("au").magnitude
    dr_edges_au = np.diff(r_edges_au)
    i_rb = int(np.argmin(np.abs(r_edges_au - Rb_au)))
    shell_dr_au = float(shell_cells) * float(dr_edges_au[max(i_rb - 1, 0)])
    shell_rmask = np.abs(r_centers_au - Rb_au) < shell_dr_au
    
    nH_cm3 = nH.to("cm^-3").magnitude
    shell_mask = np.zeros_like(nH_cm3, dtype=bool)
    shell_mask[shell_rmask, ...] = True
    
    if tau_form is None:
        model = str(diskbridge.params.co_tau_form_model).lower()
        if model == "density_capped":
            tau_form_use = compute_tau_form_co(
                nH=nH,
                n0=diskbridge.params.co_n0,
                tau0=diskbridge.params.co_tau0,
                tau_min=diskbridge.params.co_tau_min,
                alpha=float(diskbridge.params.co_alpha),
            )
        elif model == "off":
            tau_form_use = TAU_CO_FORM_DEFAULT
        else:
            raise ValueError(f"Unknown co_tau_form_model={diskbridge.params.co_tau_form_model!r}")
    else:
        tau_form_use = tau_form
    
    T_shell = Quantity(T.to("K").magnitude[shell_mask], "K")
    nH_shell = Quantity(nH_cm3[shell_mask], "cm^-3")
    chi_shell = Quantity(chi.magnitude[shell_mask], "dimensionless")
    theta_shell = Quantity(theta_co.magnitude[shell_mask], "dimensionless")
    
    if hasattr(tau_form_use, 'magnitude'):
        tau_form_shell = Quantity(tau_form_use.magnitude[shell_mask], tau_form_use.units)
    else:
        tau_form_shell = tau_form_use
    
    _, nco_gas_shell, nco_ice_shell, _, _ = evolve_co_two_phase_time_dependent(
        nH=nH_shell,
        T=T_shell,
        chi=chi_shell,
        theta_co=theta_shell,
        t_end=outer_age,
        dt=None,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form_shell,
        k0_co=k0_co,
        sigma_d_per_H=float(sigma_d_per_H),
        E_bind=float(E_bind),
        nu0=float(nu0),
        alpha_pd_ice=float(alpha_pd_ice),
        Xco_gas_init=None,
        Xco_ice_init=None,
        min_rate=1.0e-30,
    )
    
    Xco_gas0 = float(np.median(nco_gas_shell.to("cm^-3").magnitude / nH_shell.to("cm^-3").magnitude))
    Xco_ice0 = float(np.median(nco_ice_shell.to("cm^-3").magnitude / nH_shell.to("cm^-3").magnitude))
    logger.info(f"  Boundary IC: Xco_gas={Xco_gas0:.3e}, Xco_ice={Xco_ice0:.3e}")
    
    logger.info("Building per-cell age field...")
    if vr_field not in rad.model.gas:
        raise KeyError(f"Missing velocity field {vr_field!r} in model.gas")
    
    vr = rad.model.gas[vr_field].data.to("cm/s").magnitude
    
    r_full_cm = (rad.model.mesh.centers("r").to("cm").magnitude)[:, None, None] * np.ones_like(nH_cm3)
    Rb_cm = Rb.to("cm").magnitude
    
    outer_mask = (r_full_cm >= Rb_cm)
    infall_mask = (~outer_mask) & (~disc_mask)
    
    t_end_s = np.zeros_like(nH_cm3, dtype=float)
    t_end_s[outer_mask] = outer_age.to("s").magnitude
    t_end_s[disc_mask] = disc_age.to("s").magnitude
    
    t_fall_s = (Rb_cm - r_full_cm) / np.maximum(np.abs(vr), float(eps_vr))
    t_fall_s = np.maximum(t_fall_s, 0.0)
    t_end_s[infall_mask] = t_fall_s[infall_mask]
    
    t_end = Quantity(t_end_s, "s")
    
    logger.info(f"  Outer: {np.sum(outer_mask)} cells")
    logger.info(f"  Disc: {np.sum(disc_mask)} cells")
    logger.info(f"  Infall: {np.sum(infall_mask)} cells")
    
    logger.info("Building initial condition arrays...")
    Xco_gas_init = np.full_like(nH_cm3, Xco_gas0, dtype=float)
    Xco_ice_init = np.full_like(nH_cm3, Xco_ice0, dtype=float)
    
    logger.info("Running time-dependent CO chemistry...")
    X_co, nco_gas, nco_ice, k_pd, tau_pd = evolve_co_two_phase_time_dependent(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        t_end=t_end,
        dt=None,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form_use,
        k0_co=k0_co,
        sigma_d_per_H=float(sigma_d_per_H),
        E_bind=float(E_bind),
        nu0=float(nu0),
        alpha_pd_ice=float(alpha_pd_ice),
        Xco_gas_init=Xco_gas_init,
        Xco_ice_init=Xco_ice_init,
        min_rate=1.0e-30,
    )
    
    meta = {
        "R_boundary": Rb,
        "R_boundary_info": Rb_info,
        "Xco_gas0": Xco_gas0,
        "Xco_ice0": Xco_ice0,
        "masks": {"outer": outer_mask, "disc": disc_mask, "infall": infall_mask, "shell": shell_mask},
        "t_end": t_end,
    }
    
    logger.info("=" * 60)
    logger.info("Workflow complete!")
    logger.info(f"  UV boundary: {Rb:.3g}")
    logger.info(f"  Boundary IC: Xco_gas={Xco_gas0:.3e}, Xco_ice={Xco_ice0:.3e}")
    logger.info("=" * 60)
    
    return X_co, nco_gas, nco_ice, meta


__all__ = [
    "run_chemistry",
    "run_thermochemistry",
    "ChemistryResult",
    "find_uv_boundary_radius",
    "compute_chem_age",
    "compute_boundary_co_ic",
    "evolve_co_time_dependent_infall_age",
]
