from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

import diskbridge
from diskbridge._units import Quantity
from diskbridge.chemistry.constants import EPS_DEFAULT, K0_CO_DEFAULT, TAU_CO_FORM_DEFAULT, T_FRZ_DEFAULT
from diskbridge.chemistry.driver import (
    compute_abundance_pinte,
    compute_co_photodissociation_rate_field as _chem_compute_kdiss,
    compute_co_steady_state as _chem_co_steady,
    evolve_co_time_dependent as _chem_co_evolve,
)


def _ensure_temperature(rad) -> Quantity:
    if rad.temperature is None:
        try:
            rad.read_temperature()
        except Exception:
            rad.compute_temperature(force=False)
    if rad.temperature is None:
        raise RuntimeError('Temperature not available after ensure_temperature')
    return rad.temperature


def _ensure_nH(rad) -> Quantity:
    if rad.nH is None:
        rad.compute_nH_from_model()
    if rad.nH is None:
        raise RuntimeError('nH not available after ensure_nH')
    return rad.nH


def _ensure_chi(rad) -> Quantity:
    if rad.chi is None:
        rad.compute_mcmono(
            force=False,
            uv_min=diskbridge.params.uv_min,
            uv_max=diskbridge.params.uv_max,
            n_wavelengths=diskbridge.params.uv_n_wavelengths,
        )
    if rad.chi is None:
        raise RuntimeError('chi not available after ensure_chi')
    return rad.chi


def _ensure_co_shielding(
    rad,
    *,
    nside: int,
    b_kms: float,
    margin_dex: float,
    Xco_guess: float,
    XH2_guess: float,
    progress_chunks: Optional[int],
) -> Tuple[Quantity, Quantity]:
    if rad.theta_co is not None and rad.chi_eff is not None:
        return rad.theta_co, rad.chi_eff

    from diskbridge.chemistry.shielding.healpix_columns import compute_co_shielding_healpix
    from diskbridge.chemistry.shielding.visser_shielding import VisserShielding

    chi = _ensure_chi(rad)
    nH = _ensure_nH(rad)

    visser = VisserShielding(b_kms=b_kms)
    theta_co, chi_eff = compute_co_shielding_healpix(
        mesh=rad.model.mesh,
        nH=nH,
        chi=chi,
        visser=visser,
        nCO=None,
        nH2=None,
        nside=nside,
        b_kms=b_kms,
        Xco_guess=float(Xco_guess),
        XH2_guess=float(XH2_guess),
        margin_dex=margin_dex,
        progress_chunks=progress_chunks,
    )

    rad.theta_co = theta_co
    rad.chi_eff = chi_eff
    return theta_co, chi_eff


def write_number_density_to_model_dir(rad, molecule: str, number_density: Quantity) -> None:
    rad.writer.write_number_density(molecule, number_density, output_dir=rad.model_dir)


def compute_abundance(
    rad,
    *,
    molecule: str = 'co',
    X0: float = 5e-5,
    eps: float | Quantity | None = None,
    Tfrz: Quantity | None = None,
    photodissociation: Optional[bool] = None,
    freezeout: Optional[bool] = None,
    photodesorption: Optional[bool] = None,
    self_shielding: Optional[bool] = None,
    nside: int = 4,
    b_kms: float = 0.3,
    XH2_guess: float = 0.5,
    margin_dex: float = 1.5,
    write_output: bool = True,
    progress_chunks: Optional[int] = None,
    smooth_log_chi_nH_dex: float = 0.0,
    smooth_Tfrz_K: float = 0.0,
) -> Tuple[Quantity, Quantity]:
    if eps is None:
        eps = EPS_DEFAULT
    if Tfrz is None:
        Tfrz = T_FRZ_DEFAULT

    if self_shielding is not None and bool(self_shielding):
        raise ValueError(
            "self_shielding is only supported for the CO two-phase chemistry models; "
            "it is not supported for the Pinte-switch abundance model"
        )
    _ = nside
    _ = b_kms
    _ = XH2_guess
    _ = margin_dex
    _ = progress_chunks

    if photodissociation is None:
        photodissociation = diskbridge.params.photodissociation
    if freezeout is None:
        freezeout = diskbridge.params.freezeout
    if photodesorption is None:
        photodesorption = diskbridge.params.photodesorption

    mol_lower = str(molecule).lower()

    T = _ensure_temperature(rad)
    nH = _ensure_nH(rad)

    needs_chi = bool(photodissociation or photodesorption)
    chi = _ensure_chi(rad) if needs_chi else None

    write_cb = None
    if write_output:
        write_cb = lambda mol, dens: write_number_density_to_model_dir(rad, mol, dens)

    return compute_abundance_pinte(
        molecule=mol_lower,
        T=T,
        nH=nH,
        chi=chi,
        chi_eff=None,
        X0=float(X0),
        eps=eps,
        Tfrz=Tfrz,
        photodissociation=bool(photodissociation),
        freezeout=bool(freezeout),
        photodesorption=bool(photodesorption),
        smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
        smooth_Tfrz_K=float(smooth_Tfrz_K),
        write_number_density=write_cb,
    )


def run_chemistry(
    rad,
    *,
    chemistry_model: Optional[str] = None,
    **kwargs,
):
    model = None if chemistry_model is None else str(chemistry_model).lower()

    if model in (None, 'pinte', 'switches', 'abundance'):
        return compute_abundance(rad, **kwargs)

    if model in ('co_steady_state', 'co-steady-state', 'steady_state', 'steady-state'):
        return compute_co_steady_state(rad, **kwargs)

    if model in (
        'co_time_dependent',
        'co-time-dependent',
        'time_dependent',
        'time-dependent',
    ):
        return evolve_co_time_dependent(rad, **kwargs)

    raise ValueError(f"Unknown chemistry_model={chemistry_model!r}")


def compute_co_photodissociation_rate_field(
    rad,
    *,
    k0_co: Quantity = K0_CO_DEFAULT,
    candidate_mask: Optional[np.ndarray] = None,
    min_rate: float = 1.0e-30,
) -> Quantity:
    chi = _ensure_chi(rad)

    theta = rad.theta_co
    if theta is None and bool(getattr(diskbridge.params, 'co_self_shielding', False)):
        theta, _ = _ensure_co_shielding(
            rad,
            nside=4,
            b_kms=0.3,
            margin_dex=1.5,
            Xco_guess=float(diskbridge.params.abundance),
            XH2_guess=0.5,
            progress_chunks=None,
        )

    k_diss, tau = _chem_compute_kdiss(
        chi=chi,
        theta_co=theta,
        k0_co=k0_co,
        candidate_mask=candidate_mask,
        min_rate=min_rate,
    )

    rad.k_diss_co = k_diss
    rad.tau_diss_co = tau

    return k_diss


def compute_co_steady_state(
    rad,
    *,
    write_output: bool = True,
    skip_shielding: bool = False,
    nside: int = 4,
    b_kms: float = 0.3,
    margin_dex: float = 1.5,
    Xco_tot: float = 1.0e-4,
    tau_form: Quantity = TAU_CO_FORM_DEFAULT,
    k0_co: Quantity = K0_CO_DEFAULT,
) -> Tuple[Quantity, Quantity, Quantity]:
    T = _ensure_temperature(rad)
    nH = _ensure_nH(rad)
    chi = _ensure_chi(rad)

    if skip_shielding:
        theta_co = Quantity(np.ones_like(chi.magnitude), 'dimensionless')
        rad.theta_co = theta_co
        rad.chi_eff = chi
    else:
        theta_co, _ = _ensure_co_shielding(
            rad,
            nside=nside,
            b_kms=b_kms,
            margin_dex=margin_dex,
            Xco_guess=float(diskbridge.params.abundance),
            XH2_guess=0.5,
            progress_chunks=None,
        )

    write_cb = None
    if write_output:
        write_cb = lambda mol, dens: write_number_density_to_model_dir(rad, mol, dens)

    X_co, nco_gas, nco_ice, k_pd, tau_pd = _chem_co_steady(
        nH=nH,
        T=T,
        chi=chi,
        theta_co=theta_co,
        Xco_tot=float(Xco_tot),
        tau_form=tau_form,
        k0_co=k0_co,
        write_number_density=write_cb,
    )

    rad.nco_gas = nco_gas
    rad.nco_ice = nco_ice
    rad.k_diss_co = k_pd
    rad.tau_diss_co = tau_pd

    return X_co, nco_gas, nco_ice


def evolve_co_time_dependent(
    rad,
    *,
    t_end: Quantity,
    dt: Optional[Quantity] = None,
    write_output: bool = True,
    skip_shielding: bool = False,
    nside: int = 4,
    b_kms: float = 0.3,
    margin_dex: float = 1.5,
    Xco_tot: float = 1.0e-4,
    tau_form: Quantity = TAU_CO_FORM_DEFAULT,
    k0_co: Quantity = K0_CO_DEFAULT,
    Xco_gas_init: Optional[float] = None,
    Xco_ice_init: Optional[float] = None,
) -> Tuple[Quantity, Quantity, Quantity]:
    T = _ensure_temperature(rad)
    nH = _ensure_nH(rad)
    chi = _ensure_chi(rad)

    if skip_shielding:
        theta_co = Quantity(np.ones_like(chi.magnitude), 'dimensionless')
        rad.theta_co = theta_co
        rad.chi_eff = chi
    else:
        theta_co, _ = _ensure_co_shielding(
            rad,
            nside=nside,
            b_kms=b_kms,
            margin_dex=margin_dex,
            Xco_guess=float(diskbridge.params.abundance),
            XH2_guess=0.5,
            progress_chunks=None,
        )

    write_cb = None
    if write_output:
        write_cb = lambda mol, dens: write_number_density_to_model_dir(rad, mol, dens)

    X_co, nco_gas, nco_ice, k_pd, tau_pd = _chem_co_evolve(
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
        write_number_density=write_cb,
    )

    rad.nco_gas = nco_gas
    rad.nco_ice = nco_ice
    rad.k_diss_co = k_pd
    rad.tau_diss_co = tau_pd

    return X_co, nco_gas, nco_ice
