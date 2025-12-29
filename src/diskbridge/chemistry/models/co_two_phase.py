from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np

import diskbridge
from diskbridge._units import Quantity
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.constants import K0_CO_DEFAULT, TAU_CO_FORM_DEFAULT
from diskbridge.chemistry.driver import compute_co_steady_state, evolve_co_time_dependent
from diskbridge.chemistry.tau_form import compute_tau_form_co


def run_steady(rad: 'RadModel', config: dict) -> ChemistryResult:
    """Run steady-state CO two-phase chemistry.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    config : dict
        Configuration with keys:
        - Xco_tot: float (default: from diskbridge.params.abundance)
        - tau_form: Quantity (default: from diskbridge.params.co_tau_form_model)
        - k0_co: Quantity (default: K0_CO_DEFAULT)
        - skip_shielding: bool (default: False)
        - nside: int (default: 4)
        - b_kms: float (default: 0.3)
        
    Returns
    -------
    ChemistryResult
        Result with abundances, number_densities, and diagnostic fields
    """
    Xco_tot = config.get('Xco_tot', float(diskbridge.params.abundance))
    k0_co = config.get('k0_co', K0_CO_DEFAULT)
    skip_shielding = config.get('skip_shielding', False)
    nside = config.get('nside', 4)
    b_kms = config.get('b_kms', 0.3)
    
    if 'tau_form' in config:
        tau_form = config['tau_form']
    else:
        model = str(diskbridge.params.co_tau_form_model).lower()
        if model == "density_capped":
            nH = rad.ensure_nH()
            tau_form = compute_tau_form_co(
                nH=nH,
                n0=diskbridge.params.co_n0,
                tau0=diskbridge.params.co_tau0,
                tau_min=diskbridge.params.co_tau_min,
                alpha=float(diskbridge.params.co_alpha),
            )
        elif model == "off":
            tau_form = TAU_CO_FORM_DEFAULT
        else:
            raise ValueError(f"Unknown co_tau_form_model={diskbridge.params.co_tau_form_model!r}")
    
    T = rad.ensure_temperature()
    nH = rad.ensure_nH()
    chi = rad.ensure_chi()
    
    if skip_shielding:
        theta_co = Quantity(np.ones_like(chi.magnitude), 'dimensionless')
    else:
        if rad.theta_co is None:
            from diskbridge.chemistry.shielding.healpix_columns import compute_co_shielding_healpix
            from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
            
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
    
    X_co, nco_gas, nco_ice, k_pd, tau_pd = compute_co_steady_state(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form,
        k0_co=k0_co,
        write_number_density=None,
    )
    
    return ChemistryResult(
        abundances={'co': X_co},
        number_densities={'co_gas': nco_gas, 'co_ice': nco_ice, 'co': nco_gas},
        fields={'k_diss_co': k_pd, 'tau_diss_co': tau_pd, 'theta_co': theta_co},
        meta={'model': 'co_two_phase_steady'},
    )


def run_time_dependent(rad: 'RadModel', config: dict) -> ChemistryResult:
    """Run time-dependent CO two-phase chemistry.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    config : dict
        Configuration with keys:
        - t_end: Quantity (required)
        - dt: Quantity (optional)
        - Xco_tot: float (default: from diskbridge.params.abundance)
        - tau_form: Quantity (default: from diskbridge.params.co_tau_form_model)
        - k0_co: Quantity (default: K0_CO_DEFAULT)
        - skip_shielding: bool (default: False)
        - nside: int (default: 4)
        - b_kms: float (default: 0.3)
        - Xco_gas_init: float (optional)
        - Xco_ice_init: float (optional)
        
    Returns
    -------
    ChemistryResult
        Result with abundances, number_densities, and diagnostic fields
    """
    if 't_end' not in config:
        raise ValueError("t_end is required for time-dependent CO chemistry")
    
    t_end = config['t_end']
    dt = config.get('dt', None)
    Xco_tot = config.get('Xco_tot', float(diskbridge.params.abundance))
    k0_co = config.get('k0_co', K0_CO_DEFAULT)
    skip_shielding = config.get('skip_shielding', False)
    nside = config.get('nside', 4)
    b_kms = config.get('b_kms', 0.3)
    Xco_gas_init = config.get('Xco_gas_init', None)
    Xco_ice_init = config.get('Xco_ice_init', None)
    
    if 'tau_form' in config:
        tau_form = config['tau_form']
    else:
        model = str(diskbridge.params.co_tau_form_model).lower()
        if model == "density_capped":
            nH = rad.ensure_nH()
            tau_form = compute_tau_form_co(
                nH=nH,
                n0=diskbridge.params.co_n0,
                tau0=diskbridge.params.co_tau0,
                tau_min=diskbridge.params.co_tau_min,
                alpha=float(diskbridge.params.co_alpha),
            )
        elif model == "off":
            tau_form = TAU_CO_FORM_DEFAULT
        else:
            raise ValueError(f"Unknown co_tau_form_model={diskbridge.params.co_tau_form_model!r}")
    
    T = rad.ensure_temperature()
    nH = rad.ensure_nH()
    chi = rad.ensure_chi()
    
    if skip_shielding:
        theta_co = Quantity(np.ones_like(chi.magnitude), 'dimensionless')
    else:
        if rad.theta_co is None:
            from diskbridge.chemistry.shielding.healpix_columns import compute_co_shielding_healpix
            from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
            
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
    
    X_co, nco_gas, nco_ice, k_pd, tau_pd = evolve_co_time_dependent(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        t_end=t_end,
        dt=dt,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form,
        k0_co=k0_co,
        Xco_gas_init=Xco_gas_init,
        Xco_ice_init=Xco_ice_init,
        write_number_density=None,
    )
    
    return ChemistryResult(
        abundances={'co': X_co},
        number_densities={'co_gas': nco_gas, 'co_ice': nco_ice, 'co': nco_gas},
        fields={'k_diss_co': k_pd, 'tau_diss_co': tau_pd, 'theta_co': theta_co},
        meta={'model': 'co_two_phase_time_dependent'},
    )
