
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from diskbridge._units import Quantity
from diskbridge.chemistry.constants import EPS_DEFAULT, K0_CO_DEFAULT, T_FRZ_DEFAULT


def compute_abundance(
    rad,
    *,
    molecule: str = 'co',
    X0: float = 5e-5,
    eps: float | Quantity = EPS_DEFAULT,
    Tfrz: Quantity = T_FRZ_DEFAULT,
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
    from diskbridge.radmc3d.chemistry import compute_abundance as _compute

    return _compute(
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
        margin_dex=margin_dex,
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
    from diskbridge.radmc3d.chemistry import compute_co_photodissociation_rate_field as _compute

    return _compute(
        rad,
        k0_co=k0_co,
        candidate_mask=candidate_mask,
        min_rate=min_rate,
    )


def compute_co_steady_state(rad, **kwargs):
    from diskbridge.radmc3d.chemistry import compute_co_steady_state as _compute

    return _compute(rad, **kwargs)


def evolve_co_time_dependent(rad, **kwargs):
    from diskbridge.radmc3d.chemistry import evolve_co_time_dependent as _compute

    return _compute(rad, **kwargs)

