from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from diskbridge.model.core import Model

import numpy as np

from diskbridge._units import Quantity


def compute_chem_age(
    model: "Model",
    *,
    r_boundary: Quantity,
    disc_mask: np.ndarray,
    outer_age: Quantity,
    disc_age: Quantity,
    v_r_field: str = "vr",
    max_age: Quantity = None,
    mode: str = "radial",
) -> tuple[Quantity, Dict[str, np.ndarray]]:
    """Compute per-cell chemistry ages for tracer-based workflows.
    
    Assigns ages based on region:
    - Outer (r >= r_boundary): fixed outer_age
    - Disc (disc_mask): fixed disc_age
    - Infall: synthetic age = (r_boundary - r) / |vr|
    
    Parameters
    ----------
    model : Model
        DiskBridge Model with velocity field
    r_boundary : Quantity
        Boundary radius separating outer from disc/infall
    disc_mask : ndarray
        Boolean mask identifying disc cells
    outer_age : Quantity
        Fixed age for outer region
    disc_age : Quantity
        Fixed age for disc region
    v_r_field : str, optional
        Radial velocity field name (default: "vr")
    max_age : Quantity, optional
        Maximum age cap for infall cells
    mode : str, optional
        "radial" for simple backtracking (default)
        "streamline" for full integration (not implemented)
        
    Returns
    -------
    chem_age : Quantity
        Per-cell age in seconds
    masks : dict
        Region masks: 'outer', 'disc', 'infall'
        
    Examples
    --------
    >>> age, masks = compute_chem_age(
    ...     model,
    ...     r_boundary=Quantity(100, "au"),
    ...     disc_mask=joos_mask,
    ...     outer_age=Quantity(1e5, "yr"),
    ...     disc_age=Quantity(1e6, "yr"),
    ... )
    """
    if mode == "streamline":
        raise NotImplementedError(
            "Streamline mode not implemented. Use mode='radial'."
        )
    elif mode != "radial":
        raise ValueError(f"Unknown mode='{mode}'")
    
    mesh = model.mesh
    if mesh.coord_system != 'spherical':
        raise ValueError(f"Requires spherical mesh, got {mesh.coord_system}")
    
    r_centers = mesh.centers('r')
    r = r_centers.magnitude
    
    r_boundary_val = r_boundary.to(r_centers.units).magnitude
    outer_age_s = outer_age.to('s').magnitude
    disc_age_s = disc_age.to('s').magnitude
    
    if max_age is not None:
        max_age_s = max_age.to('s').magnitude
    else:
        max_age_s = 1e20
    
    shape = model.gas.density.data.shape
    age_arr = np.zeros(shape, dtype=float)
    
    outer_mask = r[:, None, None] >= r_boundary_val
    infall_mask = (~outer_mask) & (~disc_mask)
    
    age_arr[outer_mask] = outer_age_s
    age_arr[disc_mask] = disc_age_s
    
    if np.any(infall_mask):
        if v_r_field not in model.gas:
            raise ValueError(f"Velocity field '{v_r_field}' not found")
        
        vr_field = model.gas[v_r_field]
        vr = vr_field.data
        if hasattr(vr, 'magnitude'):
            vr_cms = vr.to('cm/s').magnitude
        else:
            vr_cms = vr
        
        r_cm = r_centers.to('cm').magnitude
        r_boundary_cm = r_boundary.to('cm').magnitude
        
        dr_fall = r_boundary_cm - r_cm[:, None, None]
        
        vr_infall = np.abs(vr_cms[infall_mask])
        dr_infall = dr_fall[infall_mask]
        
        vr_safe = np.maximum(vr_infall, 1e-10)
        t_fall = dr_infall / vr_safe
        
        outflow_mask_infall = vr_cms[infall_mask] >= 0
        t_fall[outflow_mask_infall] = 0.0
        
        t_fall_clipped = np.clip(t_fall, 0.0, max_age_s)
        age_arr[infall_mask] = t_fall_clipped
    
    chem_age = Quantity(age_arr, 's')
    
    masks = {
        'outer': outer_mask,
        'disc': disc_mask,
        'infall': infall_mask,
    }
    
    return chem_age, masks
