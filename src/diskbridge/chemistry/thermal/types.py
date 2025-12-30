from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diskbridge.model.mesh import Mesh

from diskbridge._units import Quantity


@dataclass
class ThermalState:
    """State container for thermal balance calculations.
    
    Contains all physical fields needed to evaluate heating/cooling terms.
    
    Required Attributes
    -------------------
    nH : Quantity
        H nuclei number density [cm^-3]
    Tdust : Quantity
        Dust temperature [K]
    chi_eff : Quantity
        Effective UV field (dimensionless, normalized to Draine)
    mesh : Mesh
        Model mesh for geometry
        
    Optional Attributes
    -------------------
    Tgas : Quantity, optional
        Gas temperature [K] (set during iteration)
    nco_gas : Quantity, optional
        CO gas number density [cm^-3]
    nco_ice : Quantity, optional
        CO ice number density [cm^-3]
    nCplus : Quantity, optional
        C+ ion number density [cm^-3]
    nC : Quantity, optional
        Neutral carbon number density [cm^-3]
    ne : Quantity, optional
        Electron number density [cm^-3]
    nH2 : Quantity, optional
        H2 number density [cm^-3]
    nH_atom : Quantity, optional
        Atomic H number density [cm^-3]
    theta_co : Quantity, optional
        CO shielding factor (dimensionless)
    chi : Quantity, optional
        Unshielded UV field (dimensionless)
        
    Notes
    -----
    Design principle: each heating/cooling term should only use fields in
    this state object and parameters from a dict. This makes terms modular
    and easy to add without modifying the solver.
    """
    nH: Quantity
    Tdust: Quantity
    chi_eff: Quantity
    mesh: "Mesh"
    
    Tgas: Optional[Quantity] = None
    nco_gas: Optional[Quantity] = None
    nco_ice: Optional[Quantity] = None
    nCplus: Optional[Quantity] = None
    nC: Optional[Quantity] = None
    ne: Optional[Quantity] = None
    nH2: Optional[Quantity] = None
    nH_atom: Optional[Quantity] = None
    theta_co: Optional[Quantity] = None
    chi: Optional[Quantity] = None


@dataclass
class ThermalResult:
    """Result container for thermal balance calculation.
    
    Attributes
    ----------
    tgas : Quantity
        Solved gas temperature field [K]
    fields : dict[str, Quantity]
        Auxiliary fields computed during solve (nCplus, ne, per-term rates)
    meta : dict
        Metadata including convergence info, solver config, etc.
        
    Examples
    --------
    >>> result = ThermalResult(
    ...     tgas=Quantity(np.array([...]), "K"),
    ...     fields={"nCplus": ..., "ne": ..., "heating_pe": ...},
    ...     meta={"n_iter": 3, "converged": True}
    ... )
    """
    tgas: Quantity
    fields: dict[str, Quantity] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
