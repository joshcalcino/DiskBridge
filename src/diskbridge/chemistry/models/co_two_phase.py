from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np
from numba import njit, prange

import diskbridge
from diskbridge._units import Quantity
from diskbridge._constants import (
    TAU_CO_FORM,
    TAU_FORM_N0, TAU_FORM_TAU0, TAU_FORM_TAU_MIN, TAU_FORM_ALPHA
)
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.models._co_two_phase_math import (
    solve_co_two_phase_steady_state,
    evolve_co_two_phase_time_dependent,
)
from diskbridge.chemistry.tau_form import compute_tau_form_co
from diskbridge.chemistry.validation import validate_chemistry_state
from diskbridge._logging import logger


@njit(parallel=True, fastmath=True, cache=True)
def _carbon_closure_kernel(
    nH: np.ndarray,
    chi: np.ndarray,
    Tg: np.ndarray,
    nco_total: np.ndarray,
    X_C_tot: float,
    Gamma_C0: float,
    alpha_rec_c0: float,
    T_rec_exp: float,
    nCplus_out: np.ndarray,
    nC_out: np.ndarray,
    ne_out: np.ndarray,
) -> None:
    ncells = nH.size
    for i in prange(ncells):
        n_C_tot = X_C_tot * nH[i]
        n_C_available = max(n_C_tot - nco_total[i], 0.0)

        Gamma_photo = Gamma_C0 * chi[i]
        alpha = alpha_rec_c0 * (Tg[i] / 300.0) ** T_rec_exp

        if Gamma_photo <= 0.0 or n_C_available <= 0.0:
            nCplus = 0.0
        elif alpha <= 0.0:
            nCplus = n_C_available
        else:
            disc = Gamma_photo * Gamma_photo + 4.0 * alpha * Gamma_photo * n_C_available
            sqrt_disc = np.sqrt(disc)
            nCplus = (2.0 * Gamma_photo * n_C_available) / (Gamma_photo + sqrt_disc)
            if nCplus < 0.0:
                nCplus = 0.0
            elif nCplus > n_C_available:
                nCplus = n_C_available

        nC = n_C_available - nCplus

        nCplus_out[i] = nCplus
        nC_out[i] = nC
        ne_out[i] = nCplus


def _carbon_closure(
    *,
    nH: Quantity,
    chi: Quantity,
    Tg: Quantity,
    nco_total: Quantity,
    X_C_tot: float = None,
    Gamma_C0: float = None,
    alpha_rec_c0: float = None,
    T_rec_exp: float = None,
) -> tuple[Quantity, Quantity, Quantity]:
    from diskbridge._constants import (
        X_C_TOT as _X_C_TOT,
        GAMMA_C0 as _GAMMA_C0,
        ALPHA_REC_C0 as _ALPHA_REC_C0,
        T_REC_EXP as _T_REC_EXP,
    )

    if X_C_tot is None:
        X_C_tot = _X_C_TOT
    if Gamma_C0 is None:
        Gamma_C0 = _GAMMA_C0
    if alpha_rec_c0 is None:
        alpha_rec_c0 = _ALPHA_REC_C0
    if T_rec_exp is None:
        T_rec_exp = _T_REC_EXP

    orig_shape = nH.magnitude.shape
    ncells = nH.magnitude.size

    nH_flat = np.ascontiguousarray(nH.to('cm^-3').magnitude.reshape(ncells), dtype=np.float64)
    chi_flat = np.ascontiguousarray(chi.to('dimensionless').magnitude.reshape(ncells), dtype=np.float64)
    Tg_flat = np.ascontiguousarray(Tg.to('K').magnitude.reshape(ncells), dtype=np.float64)
    nco_flat = np.ascontiguousarray(nco_total.to('cm^-3').magnitude.reshape(ncells), dtype=np.float64)

    nCplus_flat = np.zeros(ncells, dtype=np.float64)
    nC_flat = np.zeros(ncells, dtype=np.float64)
    ne_flat = np.zeros(ncells, dtype=np.float64)

    _carbon_closure_kernel(
        nH_flat,
        chi_flat,
        Tg_flat,
        nco_flat,
        float(X_C_tot),
        float(Gamma_C0),
        float(alpha_rec_c0),
        float(T_rec_exp),
        nCplus_flat,
        nC_flat,
        ne_flat,
    )

    nCplus = Quantity(nCplus_flat.reshape(orig_shape), 'cm^-3')
    nC = Quantity(nC_flat.reshape(orig_shape), 'cm^-3')
    ne = Quantity(ne_flat.reshape(orig_shape), 'cm^-3')

    return nCplus, nC, ne


def _resolve_tau_form(rad: 'RadModel', *, config: dict, nH: Quantity) -> Quantity:
    if 'tau_form' in config:
        return config['tau_form']

    model = str(diskbridge.params.co_tau_form_model).lower()
    if model == "density_capped":
        return compute_tau_form_co(
            nH=nH,
            n0=Quantity(TAU_FORM_N0, 'cm^-3'),
            tau0=Quantity(TAU_FORM_TAU0, 's'),
            tau_min=Quantity(TAU_FORM_TAU_MIN, 's'),
            alpha=TAU_FORM_ALPHA,
        )
    if model == "off":
        return Quantity(TAU_CO_FORM, 's')
    raise ValueError(f"Unknown co_tau_form_model={diskbridge.params.co_tau_form_model!r}")


def _resolve_theta_co(
    rad: 'RadModel',
    *,
    nH: Quantity,
    chi: Quantity,
    Xco_tot: float,
    skip_shielding: bool,
    nside: int,
    b_kms: float,
) -> Quantity:
    if skip_shielding:
        return Quantity(np.ones_like(chi.magnitude), 'dimensionless')

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
        return theta_co

    return rad.theta_co


def _run_carbon_closure(
    *,
    rad: 'RadModel',
    nH: Quantity,
    chi: Quantity,
    Tdust: Quantity,
    nco_gas: Quantity,
    nco_ice: Quantity,
) -> tuple[Quantity, Quantity, Quantity, Quantity]:
    nco_total = Quantity(
        nco_gas.to('cm^-3').magnitude + nco_ice.to('cm^-3').magnitude,
        'cm^-3'
    )

    if rad.gas_temperature is not None:
        Tg = rad.gas_temperature
        logger.info("Carbon closure using gas temperature")
    else:
        Tg = Tdust
        logger.info("Carbon closure using dust temperature (gas T not available)")

    nCplus, nC, ne = _carbon_closure(nH=nH, chi=chi, Tg=Tg, nco_total=nco_total)
    logger.info(f"Carbon closure: max(nCplus)={np.max(nCplus.magnitude):.2e} cm^-3")
    return nCplus, nC, ne, nco_total


def _prepare_common(rad: 'RadModel', config: dict) -> tuple[Quantity, Quantity, Quantity, Quantity, Quantity, float]:
    Xco_tot = float(config.get('Xco_tot', float(diskbridge.params.abundance)))
    skip_shielding = bool(config.get('skip_shielding', False))
    nside = int(config.get('nside', 4))
    b_kms = float(config.get('b_kms', 0.3))

    nH = rad.ensure_nH()
    tau_form = _resolve_tau_form(rad, config=config, nH=nH)

    Tdust = rad.ensure_dust_temperature()
    chi = rad.ensure_chi()

    theta_co = _resolve_theta_co(
        rad,
        nH=nH,
        chi=chi,
        Xco_tot=float(Xco_tot),
        skip_shielding=skip_shielding,
        nside=nside,
        b_kms=b_kms,
    )

    return nH, Tdust, chi, theta_co, tau_form, Xco_tot


def run_steady(rad: 'RadModel', config: dict) -> ChemistryResult:
    """Run steady-state CO two-phase chemistry.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    config : dict
        Configuration with keys:
        - Xco_tot: float (default: from diskbridge.params.abundance)
        - tau_form: Quantity (optional; if absent uses diskbridge.params.co_tau_form_model)
        - skip_shielding: bool (default: False)
        - nside: int (default: 4)
        - b_kms: float (default: 0.3)
        
    Returns
    -------
    ChemistryResult
        Result with abundances, number_densities, and diagnostic fields
    """
    nH, Tdust, chi, theta_co, tau_form, Xco_tot = _prepare_common(rad, config)

    Xco_gas, nco_gas, nco_ice, k_pd, tau_pd = solve_co_two_phase_steady_state(
        nH=nH,
        T=Tdust,
        chi=chi,
        theta_co=theta_co,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form,
    )

    nCplus, nC, ne, nco_total = _run_carbon_closure(
        rad=rad,
        nH=nH,
        chi=chi,
        Tdust=Tdust,
        nco_gas=nco_gas,
        nco_ice=nco_ice,
    )

    validate_chemistry_state(
        nH=nH,
        nC=nC,
        nCplus=nCplus,
        nco_total=nco_total,
        check_h=False,
        check_c=True,
    )
    
    return ChemistryResult(
        abundances={'co': Xco_gas},
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
        - tau_form: Quantity (optional; if absent uses diskbridge.params.co_tau_form_model)
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
    dt = config.get('dt', None)
    
    Xco_gas_init = config.get('Xco_gas_init', None)
    if Xco_gas_init is not None and hasattr(Xco_gas_init, 'magnitude'):
        Xco_gas_init = Xco_gas_init.magnitude
    
    Xco_ice_init = config.get('Xco_ice_init', None)
    if Xco_ice_init is not None and hasattr(Xco_ice_init, 'magnitude'):
        Xco_ice_init = Xco_ice_init.magnitude
    
    nH, Tdust, chi, theta_co, tau_form, Xco_tot = _prepare_common(rad, config)

    Xco_gas, nco_gas, nco_ice, k_pd, tau_pd = evolve_co_two_phase_time_dependent(
        nH=nH,
        T=Tdust,
        chi=chi,
        theta_co=theta_co,
        t_end=t_end,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form,
        Xco_gas_init=Xco_gas_init,
        Xco_ice_init=Xco_ice_init,
        dt=dt,
    )

    nCplus, nC, ne, nco_total = _run_carbon_closure(
        rad=rad,
        nH=nH,
        chi=chi,
        Tdust=Tdust,
        nco_gas=nco_gas,
        nco_ice=nco_ice,
    )

    validate_chemistry_state(
        nH=nH,
        nC=nC,
        nCplus=nCplus,
        nco_total=nco_total,
        check_h=False,
        check_c=True,
    )
    
    return ChemistryResult(
        abundances={'co': Xco_gas},
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
