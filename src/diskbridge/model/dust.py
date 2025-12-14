"""Dust grain submodel for DiskBridge.

This module provides dust grain distribution modeling with power-law size distributions,
memory-efficient storage for gas-proportional dust, and integration with radmc3d.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Callable, List, Literal
import numpy as np

from .field import Field
from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from diskbridge._params import params, canonicalize_dust_params
from .mesh import Mesh, Axis

if TYPE_CHECKING:
    from .model import Model

# Import SubModel for inheritance
from .model import SubModel

k_B = units('k_B')  # Boltzmann constant
m_H = units('m_H')  # Hydrogen mass
G = units('G')      # Gravitational constant
solar_mass = units('solar_mass')  # Solar mass

class DustBin:
    """Represents a single dust size bin with properties and density field.
    
    This class provides access to an individual dust bin's properties and 
    can compute its density field on-the-fly if dust is proportional to gas.
    
    Attributes:
        name: Bin identifier (e.g., 'bin_0')
        size: Grain size (geometric mean for this bin)
        density_material: Material density of dust grains (e.g., 2.7 g/cm^3)
        mass_fraction: Fraction of total dust mass in this bin
        size_min: Minimum size of this bin
        size_max: Maximum size of this bin
    """
    
    def __init__(
        self,
        parent_dust: 'Dust',
        bin_index: int,
        size: Quantity,
        size_min: Quantity,
        size_max: Quantity,
        mass_fraction: float,
        density_material: Quantity,
    ):
        """Initialize a DustBin.
        
        Args:
            parent_dust: Parent Dust SubModel
            bin_index: Index of this bin
            size: Grain size (geometric mean)
            size_min: Minimum size in this bin
            size_max: Maximum size in this bin
            mass_fraction: Fraction of total dust mass
            density_material: Material density of grains
        """
        self.parent_dust = parent_dust
        self.bin_index = bin_index
        self.name = f"bin_{bin_index}"
        self.size = size
        self.size_min = size_min
        self.size_max = size_max
        self.mass_fraction = mass_fraction
        self.density_material = density_material
        self._stored_fields: Dict[str, Field] = {}
        
    def __getitem__(self, key: str) -> Field:
        """Access fields for this bin (e.g., bin['density']).
        
        Args:
            key: Field name (e.g., 'density')
            
        Returns:
            Field object for this bin
        """
        if key == 'density':
            return self._get_density()
        elif key in self._stored_fields:
            return self._stored_fields[key]
        else:
            raise KeyError(f"Field '{key}' not available for dust bin {self.name}")
            
    def _get_density(self) -> Field:
        """Get dust density field for this bin."""
        return self.parent_dust._compute_bin_density(self.bin_index)
        
    def __repr__(self) -> str:
        return (
            f"DustBin(name='{self.name}', size={self.size}, "
            f"mass_fraction={self.mass_fraction:.4f})"
        )
        
    def __str__(self) -> str:
        return self.__repr__()


class DustDistribution:
    """Manages dust grain size distribution with power-law.
    
    This class handles the grain size bins and computes mass fractions
    following the same power-law formalism as fargo2radmc3d:
    
    frac[ibin] = (a[i+1]^(4-p) - a[i]^(4-p)) / (amax^(4-p) - amin^(4-p))
    
    where p is the power-law index (typically 3.5 for MRN distribution).
    """
    
    def __init__(
        self,
        amin: Quantity,
        amax: Quantity,
        nbin: int,
        power_index: float = 3.5,
        grain_density: Quantity = 2.7 * units('g/cm^3'),
    ):
        """Initialize dust grain size distribution.
        
        Args:
            amin: Minimum grain size
            amax: Maximum grain size
            nbin: Number of size bins
            power_index: Power-law exponent (3.5 for MRN)
            grain_density: Material density of grains
        """
        self.amin = amin
        self.amax = amax
        self.nbin = nbin
        self.power_index = power_index
        self.grain_density = grain_density
        
        # Create logarithmically spaced bin edges
        amin_val = amin.to('m').magnitude
        amax_val = amax.to('m').magnitude
        bin_edges = np.logspace(
            np.log10(amin_val),
            np.log10(amax_val),
            nbin + 1
        ) * units('m')
        
        self.bin_edges = bin_edges
        self.bin_centers = self._compute_bin_centers()
        self.mass_fractions = self._compute_mass_fractions()
        
    def _compute_bin_centers(self) -> Quantity:
        """Compute geometric mean grain size for each bin."""
        edges = self.bin_edges.magnitude
        centers = 10 ** (0.5 * (np.log10(edges[:-1]) + np.log10(edges[1:])))
        return centers * self.bin_edges.units
        
    def _compute_mass_fractions(self) -> np.ndarray:
        """Compute mass fraction in each size bin using power-law.
        
        Following fargo2radmc3d formalism:
        frac[i] = (a[i+1]^(4-p) - a[i]^(4-p)) / (amax^(4-p) - amin^(4-p))
        """
        edges = self.bin_edges.magnitude
        p = self.power_index
        
        # Compute normalization factor
        norm = edges[-1]**(4.0 - p) - edges[0]**(4.0 - p)
        
        # Compute fraction for each bin
        fractions = np.zeros(self.nbin)
        for i in range(self.nbin):
            fractions[i] = (edges[i+1]**(4.0 - p) - edges[i]**(4.0 - p)) / norm
            
        # Verify normalization
        assert np.abs(np.sum(fractions) - 1.0) < 1e-10, "Mass fractions don't sum to 1"
        
        return fractions
        
    def create_species_list(self, name_prefix: str = "dust") -> List[DustSpecies]:
        """Create list of DustSpecies objects for each bin.
        
        Args:
            name_prefix: Prefix for species names
            
        Returns:
            List of DustSpecies objects
        """
        species = []
        for i in range(self.nbin):
            spec = DustSpecies(
                name=f"{name_prefix}_{i}",
                grain_size=self.bin_centers[i],
                grain_density=self.grain_density,
                abundance_fraction=self.mass_fractions[i],
            )
            species.append(spec)
        return species


class DustComponent:
    """Represents a single dust component with its own distribution and spatial region.
    
    Attributes:
        distribution: DustDistribution for this component
        dust_to_gas_ratio: Dust-to-gas mass ratio for this component
        mode: 'proportional' or 'settling'
        mask: Optional Field defining where this component exists
        alpha: Turbulent viscosity parameter (for settling mode)
        delta: Turbulent diffusion parameter (for settling mode)
        mean_molecular_weight: Mean molecular weight for gas
        species_base: Base name for dust species (for opacity files)
        component_index: Index of this component in the parent Dust
    """
    
    def __init__(
        self,
        distribution: DustDistribution,
        dust_to_gas_ratio: float,
        mode: Literal['proportional', 'settling'] = 'proportional',
        mask: Optional[Field] = None,
        alpha: Optional[float] = None,
        delta: Optional[float] = None,
        mean_molecular_weight: float = 2.3,
        species_base: str = "dust",
        component_index: int = 0,
    ):
        """Initialize a dust component.
        
        Args:
            distribution: DustDistribution instance
            dust_to_gas_ratio: Dust-to-gas mass ratio
            mode: 'proportional' or 'settling'
            mask: Optional mask Field (True = included)
            alpha: Turbulent viscosity parameter
            delta: Turbulent diffusion parameter
            mean_molecular_weight: Mean molecular weight
            species_base: Base name for opacity files
            component_index: Component index (for naming)
        """
        self.distribution = distribution
        self.dust_to_gas_ratio = dust_to_gas_ratio
        self.mode = mode
        self.mask = mask
        self.alpha = alpha
        self.delta = delta if delta is not None else alpha
        self.mean_molecular_weight = mean_molecular_weight
        self.species_base = species_base
        self.component_index = component_index
        
    def __repr__(self) -> str:
        return (
            f"DustComponent(index={self.component_index}, "
            f"nbin={self.distribution.nbin}, "
            f"size_range={self.distribution.amin} to {self.distribution.amax}, "
            f"dust/gas={self.dust_to_gas_ratio:.3e}, mode='{self.mode}', "
            f"species='{self.species_base}')"
        )


class Dust(SubModel):
    """SubModel for dust grains in a disk model.
    
    This class provides memory-efficient storage of dust distributions.
    Supports two modes:
    1. 'proportional': Dust density proportional to gas (simple scaling)
    2. 'settling': Vertical settling-diffusion equilibrium (Youdin & Lithwick 2007)
    
    Access patterns:
        model.dust['density']  # Total dust density
        model.dust['bin_0']    # Access individual bin (returns DustBin object)
        model.dust['bin_0']['density']  # Density of specific bin
        model.dust['bin_0'].size  # Grain size of bin
    """
    
    def __init__(self, parent_model: Model):
        """Initialize dust submodel.
        
        Args:
            parent_model: Parent Model instance
        """
        # Call SubModel.__init__
        super().__init__(parent_model)
        
        # Dust-specific fields
        self._dust_fields: Dict[str, Field] = {}
        self._lazy_builders: Dict[str, Callable[[], Field]] = {}
        self._bins: Dict[str, DustBin] = {}
        
        # Multi-component support
        self._components: List[DustComponent] = []
        self._global_bins: Dict[str, tuple[int, int]] = {}  # bin_name -> (comp_idx, local_bin_idx)
        self._has_region_components: bool = False
        
        # Legacy single-component attributes (for backward compatibility)
        # These now proxy to the first component when available
        self.distribution: Optional[DustDistribution] = None
        self.dust_to_gas_ratio: Optional[float] = None
        self.mode: Literal['proportional', 'settling'] = 'proportional'
        
        # Settling-diffusion parameters (legacy)
        self.alpha: Optional[float] = None  # Turbulent viscosity parameter
        self.delta: Optional[float] = None  # Turbulent diffusion parameter (default: = alpha)
        self.mean_molecular_weight: float = 2.3  # Mean molecular weight (for H2)
        
        
    def set_distribution(
        self,
        amin: Optional[Quantity] = None,
        amax: Optional[Quantity] = None,
        nbin: Optional[int] = None,
        power_index: Optional[float] = None,
        grain_density: Optional[Quantity] = None,
        dust_to_gas_ratio: Optional[float] = None,
        mode: Literal['proportional', 'settling'] = 'proportional',
        alpha: Optional[float] = None,
        delta: Optional[float] = None,
        mean_molecular_weight: float = 2.3,
    ) -> None:
        """Set dust grain size distribution.
        
        All parameters are optional and will be pulled from diskbridge.params if not provided.
        
        Args:
            amin: Minimum grain size (default: from params)
            amax: Maximum grain size (default: from params)
            nbin: Number of size bins (default: from params)
            power_index: Power-law exponent (default: from params, typically 3.5 for MRN)
            grain_density: Material density of grains (default: from params)
            dust_to_gas_ratio: Global dust-to-gas mass ratio (default: from params)
            mode: Distribution mode ('proportional' or 'settling')
            alpha: Turbulent viscosity parameter (required for 'settling' mode)
            delta: Turbulent diffusion parameter (default: = alpha, assumes Sc ~ 1)
            mean_molecular_weight: Mean molecular weight for gas (default: 2.3 for H2)
        """
        if getattr(self, "_has_region_components", False) and self._components:
            raise ValueError(
                "Cannot use global dust.set_distribution after region-based dust components "
                "have been configured. Use region.dust.set_distribution() instead."
            )

        # Pull defaults from params if not provided
        if amin is None:
            # params.amin is already a Quantity in microns
            amin = params.amin
        if amax is None:
            # params.amax is already a Quantity in microns
            amax = params.amax
        if nbin is None:
            nbin = params.nbins
        if power_index is None:
            power_index = params.pindex
        if grain_density is None:
            # params.grain_density is already a Quantity in g/cm^3
            grain_density = params.grain_density
        if dust_to_gas_ratio is None:
            dust_to_gas_ratio = params.dust_to_gas_ratio
        
        distribution = DustDistribution(
            amin=amin,
            amax=amax,
            nbin=nbin,
            power_index=power_index,
            grain_density=grain_density,
        )
        
        # Set turbulence parameters for settling mode
        if mode == 'settling':
            if alpha is None:
                raise ValueError("alpha parameter required for settling mode (pass as argument)")
            alpha = alpha
            delta = delta if delta is not None else alpha  # Assume Sc ~ 1
            logger.info(f"Settling mode: alpha={alpha}, delta={delta}")
        
        # Get species base name from params
        species_base = params.species if isinstance(params.species, str) else params.species[0]
        
        # Create a single component (for backward compatibility)
        component = DustComponent(
            distribution=distribution,
            dust_to_gas_ratio=dust_to_gas_ratio,
            mode=mode,
            mask=self.mask,  # Use dust-level mask if set
            alpha=alpha,
            delta=delta,
            mean_molecular_weight=mean_molecular_weight,
            species_base=species_base,
            component_index=0,
        )
        
        # Set as the only component
        self._components = [component]
        
        # Rebuild global bins mapping
        self._global_bins = {}
        self._bins = {}
        global_bin_idx = 0
        for i in range(nbin):
            # Map global bin to (component_index, local_bin_index)
            self._global_bins[f"bin_{global_bin_idx}"] = (0, i)
            
            # Create DustBin for backward compatibility
            dust_bin = DustBin(
                parent_dust=self,
                bin_index=global_bin_idx,
                size=distribution.bin_centers[i],
                size_min=distribution.bin_edges[i],
                size_max=distribution.bin_edges[i + 1],
                mass_fraction=distribution.mass_fractions[i],
                density_material=grain_density,
            )
            self._bins[f"bin_{global_bin_idx}"] = dust_bin
            global_bin_idx += 1
        
        # Update legacy attributes to proxy to first component
        self.distribution = distribution
        self.dust_to_gas_ratio = dust_to_gas_ratio
        self.mode = mode
        self.alpha = alpha
        self.delta = delta if delta is not None else alpha
        self.mean_molecular_weight = mean_molecular_weight
        
        logger.info(
            f"Dust distribution set: {nbin} bins from {amin.to('um')} to {amax.to('um')}, "
            f"dust/gas={dust_to_gas_ratio:.3e}, power_index={power_index}, mode={mode}"
        )
    
    def add_component_from_mask(
        self,
        mask: Field,
        mode: Literal['proportional', 'settling'] = 'proportional',
        amin: Optional[Quantity] = None,
        amax: Optional[Quantity] = None,
        nbin: Optional[int] = None,
        power_index: Optional[float] = None,
        grain_density: Optional[Quantity] = None,
        dust_to_gas_ratio: Optional[float] = None,
        alpha: Optional[float] = None,
        delta: Optional[float] = None,
        mean_molecular_weight: float = 2.3,
    ) -> None:
        """Add a new dust component with specific mask and mode.
        
        This allows multiple dust distributions in different spatial regions.
        
        Args:
            mask: Field defining where this component exists
            mode: 'proportional' or 'settling'
            amin: Minimum grain size (default: from canonicalized params)
            amax: Maximum grain size (default: from canonicalized params)
            nbin: Number of size bins (default: from canonicalized params)
            power_index: Power-law exponent (default: from canonicalized params)
            grain_density: Material density of grains (default: from canonicalized params)
            dust_to_gas_ratio: Dust-to-gas mass ratio (default: from canonicalized params)
            alpha: Turbulent viscosity parameter (for settling)
            delta: Turbulent diffusion parameter (default: = alpha)
            mean_molecular_weight: Mean molecular weight (default: 2.3)
        """
        # Determine this component's index
        self._has_region_components = True
        component_index = len(self._components)

        # Pull canonical dust parameters from global params
        canon = canonicalize_dust_params(params)
        ncomp = canon['ncomp']
        if component_index >= ncomp:
            raise ValueError(
                f"Requested dust component index {component_index}, "
                f"but canonicalized params define only {ncomp} components."
            )

        # Use canonical per-component values if not explicitly provided
        if amin is None:
            amin = canon['amin'][component_index]
        if amax is None:
            amax = canon['amax'][component_index]
        if nbin is None:
            nbin = int(canon['nbins'][component_index])
        if power_index is None:
            power_index = float(canon['pindex'][component_index])
        if grain_density is None:
            grain_density = canon['grain_density'][component_index]
        if dust_to_gas_ratio is None:
            dust_to_gas_ratio = float(canon['dust_to_gas_ratio'][component_index])
        
        # Create distribution for this component
        distribution = DustDistribution(
            amin=amin,
            amax=amax,
            nbin=nbin,
            power_index=power_index,
            grain_density=grain_density,
        )
        
        # Validate settling mode
        if mode == 'settling':
            if alpha is None:
                raise ValueError("alpha parameter required for settling mode")
            delta = delta if delta is not None else alpha
            logger.info(f"Settling mode component: alpha={alpha}, delta={delta}")
        
        # Get species base name for this component
        canon_species = canon['species']
        if isinstance(canon_species, list):
            species_base = canon_species[component_index]
        else:
            species_base = str(canon_species)
        
        # Create component
        component = DustComponent(
            distribution=distribution,
            dust_to_gas_ratio=dust_to_gas_ratio,
            mode=mode,
            mask=mask,
            alpha=alpha,
            delta=delta,
            mean_molecular_weight=mean_molecular_weight,
            species_base=species_base,
            component_index=component_index,
        )
        
        # Add to components list
        self._components.append(component)
        
        # Update global bins mapping
        global_bin_start = len(self._global_bins)
        for local_idx in range(nbin):
            global_bin_idx = global_bin_start + local_idx
            bin_name = f"bin_{global_bin_idx}"
            self._global_bins[bin_name] = (component_index, local_idx)
            
            # Create DustBin for backward compatibility
            dust_bin = DustBin(
                parent_dust=self,
                bin_index=global_bin_idx,
                size=distribution.bin_centers[local_idx],
                size_min=distribution.bin_edges[local_idx],
                size_max=distribution.bin_edges[local_idx + 1],
                mass_fraction=distribution.mass_fractions[local_idx],
                density_material=grain_density,
            )
            self._bins[bin_name] = dust_bin
        
        logger.info(
            f"Added dust component {component_index}: {nbin} bins, "
            f"{amin.to('um')} to {amax.to('um')}, "
            f"dust/gas={dust_to_gas_ratio:.3e}, mode={mode}"
        )
        
    def _compute_stokes_number(
        self,
        grain_size: Quantity,
        gas_density: Quantity,
        gas_temperature: Quantity,
        keplerian_frequency: Quantity,
        grain_density: Quantity,
        mean_molecular_weight: float,
    ) -> Quantity:
        """Compute Stokes number for Epstein drag regime.
        
        St = Omega_K * t_s
        where t_s = (rho_s * a) / (rho_g * c_s) for Epstein regime
        
        Args:
            grain_size: Grain radius
            gas_density: Gas density
            gas_temperature: Gas temperature
            keplerian_frequency: Keplerian frequency Omega_K
            grain_density: Material density of dust grains
            mean_molecular_weight: Mean molecular weight of gas
            
        Returns:
            Stokes number (dimensionless)
        """
        # Material density
        rho_s = grain_density
        
        # Sound speed: c_s = sqrt(k_B * T / (mu * m_H))
        mu = mean_molecular_weight
        
        c_s = np.sqrt(k_B * gas_temperature / (mu * m_H))
        
        # Stopping time (Epstein regime)
        t_s = (rho_s * grain_size) / (gas_density * c_s)
        
        # Stokes number
        St = keplerian_frequency * t_s
        
        return St.to('dimensionless')
        
    def _compute_dust_scale_height(
        self,
        gas_scale_height: Quantity,
        stokes_number: Quantity,
    ) -> Quantity:
        """Compute dust scale height from settling-diffusion equilibrium.
        
        H_d / H_g = sqrt(delta / (St + delta))
        
        Following Youdin & Lithwick (2007), with turbulent diffusion parameter delta.
        For Schmidt number Sc ~ 1, delta ~ alpha.
        
        Args:
            gas_scale_height: Gas scale height H_g
            stokes_number: Stokes number
            
        Returns:
            Dust scale height H_d
        """
        if self.delta is None:
            raise ValueError("delta parameter not set (required for settling mode)")
            
        delta = self.delta
        
        # Handle Quantity arrays
        if isinstance(stokes_number, Quantity):
            St_mag = stokes_number.magnitude
        else:
            St_mag = stokes_number
            
        # Compute scale height ratio
        ratio = np.sqrt(delta / (St_mag + delta))
        
        H_d = gas_scale_height * ratio
        
        return H_d
        
    def _compute_gas_scale_height(self) -> Quantity:
        """Compute gas scale height H = c_s / Omega_K.
        
        Returns:
            Gas scale height as function of radius
        """
        mesh = self.parent.mesh
        
        # Get radial coordinates
        r = mesh.centers('r')
        
        # Option 1: Use disk parameters if available
        if self.parent.disk is not None:
            h0 = self.parent.disk.parameters.get('aspectratio')
            fl = self.parent.disk.parameters.get('flaringindex', 0.0)
            r0 = self.parent.disk.parameters.get('r0')
            
            if h0 is not None and r0 is not None:
                H = h0 * r0 * (r / r0) ** fl
                return H
                
        # Option 2: Compute from temperature and stellar mass
        if 'temperature' in self.parent.gas:
            temp = self.parent.gas['temperature'].data
            
            # Get stellar mass
            if hasattr(self.parent, 'variables'):
                if 'mstar' in self.parent.variables:
                    M_star = self.parent.variables['mstar']
                    
            # Keplerian frequency (1D array)
            Omega_K_1d = np.sqrt(G * M_star / r**3)
            
            # Sound speed (3D array)
            mu = self.mean_molecular_weight
            c_s = np.sqrt(k_B * temp / (mu * m_H))
            
            # Broadcast Omega_K to match c_s shape (spherical coordinates only)
            if mesh.coord_system != 'spherical':
                raise ValueError(f"Only spherical coordinates supported, got: {mesh.coord_system}")
            phi = mesh.centers('phi')
            theta = mesh.centers('theta')
            Omega_K = np.meshgrid(Omega_K_1d, phi, theta, indexing='ij')[0]
            
            # Scale height (3D array)
            H = c_s / Omega_K
            return H
            
        raise ValueError("Cannot compute scale height: no disk parameters or temperature")
    
    def _get_keplerian_frequency(self) -> Quantity:
        """Get Keplerian frequency Omega_K = sqrt(GM/r^3).
        
        Returns:
            Keplerian frequency array
        """
        mesh = self.parent.mesh
        r = mesh.centers('r')
        
        # Get stellar mass
        M_star = units('solar_mass').to('g')
        if hasattr(self.parent, 'variables'):
            if 'mstar' in self.parent.variables:
                M_star = self.parent.variables['mstar']
                
        G = units('G')
        Omega_K = np.sqrt(G * M_star / r**3)
        
        return Omega_K
        
    def _compute_bin_density(self, bin_index: int) -> Field:
        """Compute dust density field for a specific size bin.
        
        Component-aware: delegates to the appropriate component's logic.
        
        Args:
            bin_index: Global index of the size bin (0 to total_nbin-1)
            
        Returns:
            Field containing dust density for this bin
        """
        # Check for stored field first
        key = f"density_bin_{bin_index}"
        if key in self._dust_fields:
            return self._dust_fields[key]
        
        # Look up which component this bin belongs to
        bin_name = f"bin_{bin_index}"
        if bin_name not in self._global_bins:
            raise KeyError(f"Bin {bin_name} not found in global bins mapping")
        
        comp_idx, local_bin_idx = self._global_bins[bin_name]
        component = self._components[comp_idx]
        
        # Get gas density
        if 'density' in self.parent.gas:
            gas_density = self.parent.gas['density']
        elif 'surface_density' in self.parent.gas:
            # For 2D models, use surface density (proportional mode only)
            if component.mode == 'settling':
                raise ValueError("Settling mode requires 3D model with density field")
            gas_density = self.parent.gas['surface_density']
        else:
            raise KeyError("No gas density or surface_density field found")
        
        # Delegate to component-specific computation
        return self._compute_component_bin_density(
            component, local_bin_idx, gas_density
        )
    
    def _compute_component_bin_density(
        self, component: DustComponent, local_bin_idx: int, gas_density: Field
    ) -> Field:
        """Compute dust density for a specific bin within a component.
        
        Args:
            component: DustComponent instance
            local_bin_idx: Index within the component's bins
            gas_density: Gas density field
            
        Returns:
            Field containing dust density for this bin
        """
        # Mode 1: Proportional (simple scaling)
        if component.mode == 'proportional':
            dust_density_data = (
                component.dust_to_gas_ratio * 
                component.distribution.mass_fractions[local_bin_idx] *
                gas_density.data
            )
            
            # Apply component mask if set
            if component.mask is not None:
                mask_array = component.mask.data.magnitude.astype(bool)
                dust_density_data = dust_density_data * mask_array
                
            dust_field = Field(
                data=dust_density_data,
                quantity=gas_density.quantity,
                axis_order=gas_density.axis_order,
            )
            
            return dust_field
            
        # Mode 2: Settling-diffusion equilibrium
        elif component.mode == 'settling':
            return self._compute_settling_density(component, local_bin_idx, gas_density)
            
        else:
            raise ValueError(f"Unknown mode: {component.mode}")
            
    def _compute_settling_density(self, component: DustComponent, local_bin_idx: int, gas_density: Field) -> Field:
        """Compute dust settling-diffusion equilibrium density for a component bin.
        
        Implements the settling-diffusion equilibrium model where dust settles
        toward the midplane with a scale height H_d that depends on the Stokes number:
        
            H_d / H_g = sqrt(delta / (St + delta))
        
        where St is the Stokes number and delta is the turbulent diffusion parameter.
        
        Args:
            component: DustComponent instance
            local_bin_idx: Dust bin index within the component
            gas_density: Gas density field
            
        Returns:
            Dust density field with vertical settling profile
        """
        mesh = self.parent.mesh
        
        if mesh.coord_system != 'spherical':
            raise ValueError(f"Settling only supports spherical coordinates, got: {mesh.coord_system}")
        
        return self._compute_settling_density_spherical(component, local_bin_idx, gas_density)
    
    def _compute_settling_density_spherical(self, component: DustComponent, local_bin_idx: int, gas_density: Field) -> Field:
        """Compute dust settling in spherical coordinates with mass conservation.
        
        Computes settled dust density directly in spherical coordinates by:
        1. Computing z = r*cos(theta) for each spherical cell
        2. Evaluating the Gaussian vertical profile at that height
        3. Normalizing each radial column to conserve surface density
        
        Args:
            component: DustComponent instance
            local_bin_idx: Dust bin index within component
            gas_density: Gas density field in spherical coordinates
            
        Returns:
            Dust density field with vertical settling profile
        """
        mesh = self.parent.mesh
        grain_size = component.distribution.bin_centers[local_bin_idx]
        mass_fraction = component.distribution.mass_fractions[local_bin_idx]
        
        # Need temperature to compute Stokes number
        if 'temperature' not in self.parent.gas:
            raise ValueError("Settling mode requires temperature field")
        gas_temp = self.parent.gas['temperature'].data
        
        logger.debug(
            f"Computing dust settling directly in spherical coordinates for "
            f"component {component.component_index} bin {local_bin_idx} (mass-conserving)"
        )
        
        # Get spherical grid coordinates
        r = mesh.centers('r')
        theta = mesh.centers('theta')
        phi = mesh.centers('phi')
        
        r_edges = mesh.edges('r')
        theta_edges = mesh.edges('theta')
        
        # Create 3D grids - use magnitudes to avoid pint meshgrid issues, then add units
        # Use indexing='ij' to get shape (n_r, n_phi, n_theta)
        r_mag = r.magnitude
        phi_mag = phi.magnitude
        theta_mag = theta.magnitude
        r_units = r.units
        
        r_grid_mag, phi_grid_mag, theta_grid_mag = np.meshgrid(r_mag, phi_mag, theta_mag, indexing='ij')
        r_grid = r_grid_mag * r_units
        
        # Compute cylindrical coordinates for each spherical cell
        # z inherits units from r_grid
        z_cyl = r_grid * np.cos(theta_grid_mag)  # Height above midplane
        
        # Find midplane index (theta closest to pi/2)
        theta_mid_idx = np.argmin(np.abs(theta_mag - np.pi/2))
        
        # Get midplane properties
        rho_g0 = gas_density.data[:, :, theta_mid_idx]
        T0 = gas_temp[:, :, theta_mid_idx]
        
        # Compute Keplerian frequency at r (same as cylindrical method)
        Omega_K_1d = self._get_keplerian_frequency()
        Omega_K = Omega_K_1d[:, None]
        
        # Sound speed at midplane
        mu = component.mean_molecular_weight
        c_s0 = np.sqrt(k_B * T0 / (mu * m_H))
        
        # Gas scale height at midplane: H_g = c_s / Omega_K
        # Convert to same units as r for consistent length units
        H_g0 = (c_s0 / Omega_K).to(r_units)
        
        # Stokes number at midplane
        St0 = self._compute_stokes_number(
            grain_size=grain_size,
            gas_density=rho_g0,
            gas_temperature=T0,
            keplerian_frequency=Omega_K,
            grain_density=component.distribution.grain_density,
            mean_molecular_weight=component.mean_molecular_weight,
        )
        
        # Dust scale height from midplane Stokes number
        delta = component.delta
        H_d0 = H_g0 * np.sqrt(delta / (St0 + delta))
        
        # Expand to 3D (H_d is function of r only, constant in theta)
        n_theta = len(theta)
        H_d0_mag = H_d0.magnitude
        H_d_mag = np.broadcast_to(H_d0_mag[:, :, None], (*H_d0_mag.shape, n_theta))
        H_d = H_d_mag * H_d0.units
        
        # Compute vertical profile: exp(-z^2 / (2 * H_d^2))
        # Now z_cyl and H_d have the same units (r_units), so z^2/H_d^2 is dimensionless
        vertical_profile = np.exp(-(z_cyl**2 / (2 * H_d**2)).to('dimensionless').magnitude)
        
        # Compute dust density directly from analytic settling profile, without
        # additional column renormalization. This makes the dust density a
        # simple function of local gas density, dtg, and the vertical settling
        # profile, and is useful for inspecting the raw behaviour.

        dust_to_gas_local = component.dust_to_gas_ratio * mass_fraction
        prefactor = dust_to_gas_local * (H_g0[:, :, None] / H_d)
        dust_density_data = rho_g0[:, :, None] * prefactor * vertical_profile

        # Apply mask if set
        if component.mask is not None:
            mask_array = component.mask.data.magnitude.astype(bool)
            dust_density_data = dust_density_data * mask_array

        # Ensure numerical stability
        dust_density_mag = dust_density_data.magnitude
        dust_density_mag[~np.isfinite(dust_density_mag)] = 0.0
        dust_density_mag[dust_density_mag < 0.0] = 0.0
        dust_density_data = dust_density_mag * dust_density_data.units
        
        axis_order = gas_density.axis_order
        
        dust_field = Field(
            data=dust_density_data,
            quantity=gas_density.quantity,
            axis_order=axis_order,
        )
        
        logger.debug(
            f"Dust settling (direct spherical) completed for component {component.component_index} "
            f"bin {local_bin_idx}"
        )
        
        return dust_field
        
    def _get_total_dust_density(self) -> Field:
        """Get total dust density summed over all size bins.
        
        Returns:
            Field containing total dust density
        """
        # Sum over all bins (works for both proportional and settling modes)
        total_data = None
        first_field = None
        for i in range(self.distribution.nbin):
            bin_density = self._compute_bin_density(i)
            if total_data is None:
                total_data = bin_density.data.copy()
                first_field = bin_density
            else:
                total_data = total_data + bin_density.data
        
        return Field(
            data=total_data,
            quantity=first_field.quantity,
            axis_order=first_field.axis_order,
        )
            
    def register(self, name: str, field: Field) -> None:
        """Register a dust field (for explicitly stored dust fields).
        
        Args:
            name: Field name
            field: Field object
        """
        self._dust_fields[name] = field
        
    def __getitem__(self, key: str):
        """Access dust fields or bins by name.
        
        Args:
            key: Field name ('density' for total) or bin name ('bin_0', 'bin_1', etc.)
            
        Returns:
            Field object for 'density', or DustBin object for 'bin_X'
        """
        # Special case: total density
        if key == 'density':
            return self._get_total_dust_density()
            
        # Check for bin access
        if key.startswith("bin_") or key.startswith("bin"):
            # Normalize key (handle both 'bin_0' and 'bin0')
            if "_" not in key:
                key = f"bin_{key[3:]}"
            if key in self._bins:
                return self._bins[key]
                
        # Check explicit fields
        if key in self._dust_fields:
            return self._dust_fields[key]
            
        # Check lazy builders
        if key in self._lazy_builders:
            field = self._lazy_builders[key]()
            self._dust_fields[key] = field
            return field
                
        raise KeyError(f"Dust field or bin '{key}' not found")
        
    def __contains__(self, key: str) -> bool:
        """Check if a dust field or bin exists."""
        if key == 'density' and self.distribution is not None:
            return True
        if key.startswith("bin"):
            # Normalize key
            if "_" not in key:
                key = f"bin_{key[3:]}"
            return key in self._bins
        return (
            key in self._dust_fields or
            key in self._lazy_builders
        )
        
    def keys(self):
        """Return available dust field and bin names."""
        keys = list(self._dust_fields.keys()) + list(self._lazy_builders.keys())
        if self.distribution is not None:
            keys.append('density')  # Total density always available
            keys.extend(self._bins.keys())
        return keys
        
    @property
    def bins(self) -> Dict[str, DustBin]:
        """Get all dust bins as a dictionary."""
        return self._bins
        
    @property
    def nbin(self) -> int:
        """Total number of dust bins across all components."""
        return len(self._global_bins)
        
    def __repr__(self) -> str:
        if self.distribution is None:
            return "Dust(no distribution set)"
        repr_str = (
            f"Dust(nbin={self.nbin}, "
            f"size_range={self.distribution.amin} to {self.distribution.amax}, "
            f"dust/gas={self.dust_to_gas_ratio:.3e}, mode='{self.mode}'"
        )
        if self.mode == 'settling' and self.alpha is not None:
            repr_str += f", alpha={self.alpha}"
        repr_str += ")"
        return repr_str
        
    def __str__(self) -> str:
        return self.__repr__()
