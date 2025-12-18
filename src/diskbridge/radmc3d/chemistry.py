from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from diskbridge._units import Quantity
from diskbridge.chemistry.constants import K0_CO_DEFAULT, TAU_CO_FORM_DEFAULT
from diskbridge.chemistry import (
    compute_abundance as _chem_compute_abundance,
    compute_co_photodissociation_rate_field as _chem_compute_co_kdiss_field,
    compute_co_steady_state as _chem_compute_co_steady_state,
    evolve_co_time_dependent as _chem_evolve_co_time_dependent,
    run_chemistry as _chem_run_chemistry,
)


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
    write_output: bool = True,
    progress_chunks: Optional[int] = None,
    smooth_log_chi_nH_dex: float = 0.0,
    smooth_Tfrz_K: float = 0.0,
    chemistry_model: Optional[str] = None,
) -> Tuple[Quantity, Quantity]:
    if chemistry_model is not None:
        raise ValueError(
            "chemistry_model is not supported by compute_abundance(). "
            "Use run_chemistry(rad, chemistry_model=...) for parts B/C."
        )
    return _chem_compute_abundance(
        rad,
        molecule=molecule,
        X0=X0,
        eps=eps,
        Tfrz=Tfrz,
        photodissociation=photodissociation,
        freezeout=freezeout,
        photodesorption=photodesorption,
        self_shielding=self_shielding,
        nside=nside,
        b_kms=b_kms,
        XH2_guess=XH2_guess,
        write_output=write_output,
        progress_chunks=progress_chunks,
        smooth_log_chi_nH_dex=smooth_log_chi_nH_dex,
        smooth_Tfrz_K=smooth_Tfrz_K,
    )


def compute_co_photodissociation_rate_field(
    rad,
    *,
    k0_co: Quantity = K0_CO_DEFAULT,
    candidate_mask: Optional[np.ndarray] = None,
    min_rate: float = 1.0e-30,
) -> Quantity:
    return _chem_compute_co_kdiss_field(
        rad,
        k0_co=k0_co,
        candidate_mask=candidate_mask,
        min_rate=min_rate,
    )


def compute_co_steady_state(
    rad,
    *,
    write_output: bool = True,
    skip_shielding: bool = False,
    nside: int = 4,
    b_kms: float = 0.3,
    Xco_tot: float = 1.0e-4,
    tau_form: Quantity = TAU_CO_FORM_DEFAULT,
    k0_co: Quantity = K0_CO_DEFAULT,
) -> Tuple[Quantity, Quantity, Quantity]:
    return _chem_compute_co_steady_state(
        rad,
        write_output=write_output,
        skip_shielding=skip_shielding,
        nside=nside,
        b_kms=b_kms,
        Xco_tot=Xco_tot,
        tau_form=tau_form,
        k0_co=k0_co,
    )


def evolve_co_time_dependent(
    rad,
    *,
    t_end: Quantity,
    dt: Optional[Quantity] = None,
    write_output: bool = True,
    skip_shielding: bool = False,
    nside: int = 4,
    b_kms: float = 0.3,
    Xco_tot: float = 1.0e-4,
    tau_form: Quantity = TAU_CO_FORM_DEFAULT,
    k0_co: Quantity = K0_CO_DEFAULT,
    Xco_gas_init: Optional[float] = None,
    Xco_ice_init: Optional[float] = None,
) -> Tuple[Quantity, Quantity, Quantity]:
    return _chem_evolve_co_time_dependent(
        rad,
        t_end=t_end,
        dt=dt,
        write_output=write_output,
        skip_shielding=skip_shielding,
        nside=nside,
        b_kms=b_kms,
        Xco_tot=Xco_tot,
        tau_form=tau_form,
        k0_co=k0_co,
        Xco_gas_init=Xco_gas_init,
        Xco_ice_init=Xco_ice_init,
    )


def run_chemistry(rad, *, chemistry_model: Optional[str] = None, **kwargs):
    return _chem_run_chemistry(rad, chemistry_model=chemistry_model, **kwargs)
