from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np

import diskbridge
from diskbridge._units import Quantity
from diskbridge._constants import (
    TAU_CO_FORM, K0_CO, SIGMA_D_PER_H, E_BIND_CO, NU0_CO, ALPHA_PD_ICE,
    TAU_FORM_N0, TAU_FORM_TAU0, TAU_FORM_TAU_MIN, TAU_FORM_ALPHA
)
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.models._co_two_phase_math import (
    solve_co_two_phase_steady_state,
    evolve_co_two_phase_time_dependent,
)
from diskbridge.chemistry.tau_form import compute_tau_form_co
from diskbridge.chemistry.closures.carbon import compute_carbon_closure
from diskbridge._logging import logger


def run_steady(rad: 'RadModel', config: dict) -> ChemistryResult:
    """Run steady-state CO two-phase chemistry.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    config : dict
        Configuration with keys:
        - Xco_tot: float (default: from diskbridge.params.abundance)
        - tau_form: Quantity (default: from config)
        - k0_co: float in s^-1 (default: from config)
        - skip_shielding: bool (default: False)
        - nside: int (default: 4)
        - b_kms: float (default: 0.3)
        
    Returns
    -------
    ChemistryResult
        Result with abundances, number_densities, and diagnostic fields
    """
    Xco_tot = config.get('Xco_tot', float(diskbridge.params.abundance))
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
                n0=Quantity(TAU_FORM_N0, 'cm^-3'),
                tau0=Quantity(TAU_FORM_TAU0, 's'),
                tau_min=Quantity(TAU_FORM_TAU_MIN, 's'),
                alpha=TAU_FORM_ALPHA,
            )
        elif model == "off":
            tau_form = Quantity(TAU_CO_FORM, 's')
        else:
            raise ValueError(f"Unknown co_tau_form_model={diskbridge.params.co_tau_form_model!r}")
    
    T = rad.ensure_dust_temperature()
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
                nH2=rad.nH2,
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
    
    X_co, nco_gas, nco_ice, k_pd, tau_pd = solve_co_two_phase_steady_state(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form,
    )
    
    # Compute carbon closure using dust-attenuated UV (NOT CO-shielded chi_eff)
    nco_total = Quantity(
        nco_gas.to('cm^-3').magnitude + nco_ice.to('cm^-3').magnitude,
        'cm^-3'
    )
    
    # Use gas temperature if available, else dust temperature
    if rad.gas_temperature is not None:
        Tg = rad.gas_temperature
        logger.info("Carbon closure using gas temperature")
    else:
        Tg = T
        logger.info("Carbon closure using dust temperature (gas T not available)")
    
    # IMPORTANT: Use chi (dust-attenuated), NOT chi_eff (CO-shielded)
    # Carbon photoionization is in FUV continuum, attenuated by dust but not CO
    nCplus, nC, ne = compute_carbon_closure(nH, chi, Tg, nco_total)
    logger.info(f"Carbon closure: max(nCplus)={np.max(nCplus.magnitude):.2e} cm^-3")
    
    return ChemistryResult(
        abundances={'co': X_co},
        number_densities={
            'co_gas': nco_gas,
            'co_ice': nco_ice,
            'co': nco_gas,
            'cplus': nCplus,
            'c': nC,
            'e': ne,
        },
        fields={'k_diss_co': k_pd, 'tau_diss_co': tau_pd, 'theta_co': theta_co},
        meta={'model': 'co_two_phase'},
    )


def run_time_dependent(rad: 'RadModel', config: dict) -> ChemistryResult:
    """Run time-dependent CO two-phase chemistry.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    config : dict
        Configuration with keys:
        - t_end: Quantity (required, scalar or array for per-cell time)
        - dt: Quantity (optional)
        - Xco_tot: float (default: from diskbridge.params.abundance)
        - tau_form: Quantity (default: from diskbridge.params.co_tau_form_model)
        - k0_co: Quantity (default: K0_CO_DEFAULT)
        - skip_shielding: bool (default: False)
        - nside: int (default: 4)
        - b_kms: float (default: 0.3)
        - Xco_gas_init: float or array (optional, scalar or per-cell IC)
        - Xco_ice_init: float or array (optional, scalar or per-cell IC)
        
    Returns
    -------
    ChemistryResult
        Result with abundances, number_densities, and diagnostic fields
    """
    if 't_end' not in config:
        raise ValueError("t_end is required for time-dependent CO chemistry")
    
    t_end = config['t_end']
    Xco_tot = config.get('Xco_tot', float(diskbridge.params.abundance))
    skip_shielding = config.get('skip_shielding', False)
    nside = config.get('nside', 4)
    b_kms = config.get('b_kms', 0.3)
    
    Xco_gas_init = config.get('Xco_gas_init', None)
    if Xco_gas_init is not None and hasattr(Xco_gas_init, 'magnitude'):
        Xco_gas_init = Xco_gas_init.magnitude
    
    Xco_ice_init = config.get('Xco_ice_init', None)
    if Xco_ice_init is not None and hasattr(Xco_ice_init, 'magnitude'):
        Xco_ice_init = Xco_ice_init.magnitude
    
    if 'tau_form' in config:
        tau_form = config['tau_form']
    else:
        model = str(diskbridge.params.co_tau_form_model).lower()
        if model == "density_capped":
            nH = rad.ensure_nH()
            tau_form = compute_tau_form_co(
                nH=nH,
                n0=Quantity(TAU_FORM_N0, 'cm^-3'),
                tau0=Quantity(TAU_FORM_TAU0, 's'),
                tau_min=Quantity(TAU_FORM_TAU_MIN, 's'),
                alpha=TAU_FORM_ALPHA,
            )
        elif model == "off":
            tau_form = Quantity(TAU_CO_FORM, 's')
        else:
            raise ValueError(f"Unknown co_tau_form_model={diskbridge.params.co_tau_form_model!r}")
    
    T = rad.ensure_dust_temperature()
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
                nH2=rad.nH2,
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
    
    X_co, nco_gas, nco_ice, k_pd, tau_pd = evolve_co_two_phase_time_dependent(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        t_end=t_end,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form,
        Xco_gas_init=Xco_gas_init,
        Xco_ice_init=Xco_ice_init,
    )
    
    # Compute carbon closure using dust-attenuated UV (NOT CO-shielded chi_eff)
    nco_total = Quantity(
        nco_gas.to('cm^-3').magnitude + nco_ice.to('cm^-3').magnitude,
        'cm^-3'
    )
    
    # Use gas temperature if available, else dust temperature
    if rad.gas_temperature is not None:
        Tg = rad.gas_temperature
        logger.info("Carbon closure using gas temperature")
    else:
        Tg = T
        logger.info("Carbon closure using dust temperature (gas T not available)")
    
    # IMPORTANT: Use chi (dust-attenuated), NOT chi_eff (CO-shielded)
    # Carbon photoionization is in FUV continuum, attenuated by dust but not CO
    nCplus, nC, ne = compute_carbon_closure(nH, chi, Tg, nco_total)
    logger.info(f"Carbon closure: max(nCplus)={np.max(nCplus.magnitude):.2e} cm^-3")
    
    return ChemistryResult(
        abundances={'co': X_co},
        number_densities={
            'co_gas': nco_gas,
            'co_ice': nco_ice,
            'co': nco_gas,
            'cplus': nCplus,
            'c': nC,
            'e': ne,
        },
        fields={'k_diss_co': k_pd, 'tau_diss_co': tau_pd, 'theta_co': theta_co},
        meta={'model': 'co_two_phase'},
    )


def compute_boundary_co_ic(
    rad: 'RadModel',
    *,
    r_boundary: Quantity,
    shell_dr: Optional[Quantity] = None,
    age: Quantity,
    base_config: Optional[dict] = None,
    reducer: str = "median",
) -> tuple[float, float]:
    """Compute boundary CO initial conditions from outer shell chemistry.
    
    Solves time-dependent chemistry in a thin shell near r_boundary for the
    specified age, then reduces to scalar abundances for use as ICs.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    r_boundary : Quantity
        Boundary radius
    shell_dr : Quantity, optional
        Shell thickness (default: 2 * dr at boundary)
    age : Quantity
        Evolution time for boundary chemistry
    base_config : dict, optional
        Base chemistry config (default: {})
    reducer : str, optional
        "median" or "mean" (default: "median")
        
    Returns
    -------
    xco_gas0 : float
        Boundary gas abundance (dimensionless)
    xco_ice0 : float
        Boundary ice abundance (dimensionless)
        
    Examples
    --------
    >>> xco_gas0, xco_ice0 = compute_boundary_co_ic(
    ...     rad,
    ...     r_boundary=Quantity(100, "au"),
    ...     age=Quantity(1e5, "yr"),
    ... )
    """
    from diskbridge._logging import logger
    
    if reducer not in ("median", "mean"):
        raise ValueError(f"Invalid reducer='{reducer}'")
    
    mesh = rad.model.mesh
    r_centers = mesh.centers('r')
    r_edges = mesh.edges('r')
    
    r_au = r_centers.to('au').magnitude
    r_edges_au = r_edges.to('au').magnitude
    r_boundary_au = r_boundary.to('au').magnitude
    
    r_idx = np.argmin(np.abs(r_au - r_boundary_au))
    
    if shell_dr is None:
        if r_idx < len(r_edges_au) - 1:
            dr = r_edges_au[r_idx + 1] - r_edges_au[r_idx]
        else:
            dr = r_edges_au[r_idx] - r_edges_au[r_idx - 1]
        shell_dr = Quantity(2.0 * dr, 'au')
    
    shell_dr_au = shell_dr.to('au').magnitude
    shell_mask = np.abs(r_au - r_boundary_au) <= shell_dr_au / 2.0
    shell_mask_3d = shell_mask[:, None, None]
    
    if not np.any(shell_mask):
        raise ValueError(f"No cells in shell at r={r_boundary_au:.3g} AU")
    
    if base_config is None:
        base_config = {}
    
    config = base_config.copy()
    config['t_end'] = age
    config['dt'] = None
    
    result = run_time_dependent(rad, config)
    
    nH = rad.ensure_nH()
    if hasattr(nH, 'magnitude'):
        nH_cm3 = nH.to('cm^-3').magnitude
    else:
        nH_cm3 = nH
    
    nco_gas = result.number_densities['co_gas']
    if hasattr(nco_gas, 'magnitude'):
        nco_gas_cm3 = nco_gas.to('cm^-3').magnitude
    else:
        nco_gas_cm3 = nco_gas
    
    nco_ice = result.number_densities['co_ice']
    if hasattr(nco_ice, 'magnitude'):
        nco_ice_cm3 = nco_ice.to('cm^-3').magnitude
    else:
        nco_ice_cm3 = nco_ice
    
    xco_gas = nco_gas_cm3 / (nH_cm3 + 1e-99)
    xco_ice = nco_ice_cm3 / (nH_cm3 + 1e-99)
    
    xco_gas_shell = xco_gas[shell_mask_3d]
    xco_ice_shell = xco_ice[shell_mask_3d]
    
    if reducer == "median":
        xco_gas0 = float(np.median(xco_gas_shell))
        xco_ice0 = float(np.median(xco_ice_shell))
    else:
        xco_gas0 = float(np.mean(xco_gas_shell))
        xco_ice0 = float(np.mean(xco_ice_shell))
    
    logger.info(
        f"Boundary IC: r={r_boundary_au:.3g} AU, "
        f"Xco_gas={xco_gas0:.3e}, Xco_ice={xco_ice0:.3e}"
    )
    
    return xco_gas0, xco_ice0


def run_co_two_phase(rad: 'RadModel', config: dict) -> ChemistryResult:
    if 't_end' in config:
        return run_time_dependent(rad, config)
    return run_steady(rad, config)


__all__ = [
    'run_co_two_phase',
]
