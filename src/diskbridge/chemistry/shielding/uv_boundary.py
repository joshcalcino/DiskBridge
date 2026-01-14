from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diskbridge.model.core import Model

from diskbridge.model.profiles import (
    compute_volume_weighted_mean_radial_profile,
    find_r_split,
)


def find_uv_boundary_radius(
    model: "Model",
    *,
    tol_chi: float = 0.01,
    window_fraction: float = 0.10,
    r_min_au: Optional[float] = None,
    r_max_au: Optional[float] = None,
) -> float:
    """Find boundary radius where stellar UV increases chi above background.
    
    Returns the innermost radius where volume-weighted mean chi(r) first
    exceeds (1 + tol_chi) * chi_background. Background chi is the mean
    in the outer window_fraction of the radial domain.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model with 'chi' field
    tol_chi : float, optional
        Fractional tolerance above background (default: 0.01 = 1%)
    window_fraction : float, optional
        Fraction of domain for background window (default: 0.10)
    r_min_au : float, optional
        Minimum allowed boundary radius (au)
    r_max_au : float, optional
        Maximum allowed boundary radius (au)
        
    Returns
    -------
    r_boundary : float
        Boundary radius in au
        
    Raises
    ------
    ValueError
        If r_boundary outside allowed range or no valid boundary found
    KeyError
        If 'chi' field not in model
        
    Examples
    --------
    >>> r_b = find_uv_boundary_radius(model, tol_chi=0.01)
    """
    if 'chi' not in model.gas:
        raise KeyError("Field 'chi' not found in model.gas")
    
    r_au, chi_profile = compute_volume_weighted_mean_radial_profile(model, 'chi')
    r_edges_au = model.mesh.edges_f64('r', 'au')
    
    T_profile = chi_profile.copy()
    
    r_clip_min_au = 0.1
    if r_min_au is not None:
        r_clip_min_au = float(r_min_au)
    
    try:
        r_boundary_au, info = find_r_split(
            r_au=r_au,
            r_edges_au=r_edges_au,
            chi_profile=chi_profile,
            T_profile=T_profile,
            tol_chi=tol_chi,
            tol_T=1e10,
            window_fraction=window_fraction,
            r_clip_min_au=r_clip_min_au,
        )
    except ValueError as e:
        raise ValueError(
            f"Could not find UV boundary: {e}. "
            "Stellar UV may dominate entire domain."
        ) from e
    
    r_boundary = float(r_boundary_au)

    if r_max_au is not None and r_boundary > float(r_max_au):
        raise ValueError(
            f"UV boundary {r_boundary:.2f} exceeds max {float(r_max_au):.2f}"
        )

    return r_boundary
