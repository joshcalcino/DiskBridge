from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict

from diskbridge._units import Quantity


@dataclass
class ChemistryResult:
    """Result from a chemistry computation.
    
    Attributes
    ----------
    abundances : dict[str, Quantity]
        Molecular abundances (fraction relative to H nuclei)
    number_densities : dict[str, Quantity]
        Number densities in cm^-3
    fields : dict[str, Quantity]
        Additional diagnostic fields (e.g., k_diss_co, tau_diss_co, theta_co, chi_eff)
    meta : dict
        Metadata about the computation (iterations, timings, etc.)
    """
    abundances: Dict[str, Quantity] = field(default_factory=dict)
    number_densities: Dict[str, Quantity] = field(default_factory=dict)
    fields: Dict[str, Quantity] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)
    
    def get(self, name: str, default=None) -> Optional[Quantity]:
        """Get a value by name, searching abundances, number_densities, then fields.
        
        Parameters
        ----------
        name : str
            Field name to retrieve
        default : optional
            Default value if not found
            
        Returns
        -------
        Quantity or default
            The requested field or default value
        """
        if name in self.abundances:
            return self.abundances[name]
        if name in self.number_densities:
            return self.number_densities[name]
        if name in self.fields:
            return self.fields[name]
        return default
