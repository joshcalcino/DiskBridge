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

if TYPE_CHECKING:
    from .model import Model

# Import SubModel for inheritance
from .model import SubModel


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
        
        # Use base for consistency with SubModel
        self.parent = parent_model  # Keep for backwards compatibility
        
        # Dust-specific fields
        self._dust_fields: Dict[str, Field] = {}
        self._lazy_builders: Dict[str, Callable[[], Field]] = {}
        self._bins: Dict[str, DustBin] = {}
        
        # Dust distribution parameters
        self.distribution: Optional[DustDistribution] = None
        self.dust_to_gas_ratio: Optional[float] = None
        self.mode: Literal['proportional', 'settling'] = 'proportional'
        
        # Settling-diffusion parameters
        self.alpha: Optional[float] = None  # Turbulent viscosity parameter
        self.delta: Optional[float] = None  # Turbulent diffusion parameter (default: = alpha)
        self.mean_molecular_weight: float = 2.3  # Mean molecular weight (for H2)
        
    def set_distribution(
        self,
        amin: Quantity,
        amax: Quantity,
        nbin: int,
        power_index: float = 3.5,
        grain_density: Quantity = 2.7 * units('g/cm^3'),
        dust_to_gas_ratio: float = 0.01,
        mode: Literal['proportional', 'settling'] = 'proportional',
        alpha: Optional[float] = None,
        delta: Optional[float] = None,
        mean_molecular_weight: float = 2.3,
    ) -> None:
        """Set dust grain size distribution.
        
        Args:
            amin: Minimum grain size
            amax: Maximum grain size  
            nbin: Number of size bins
            power_index: Power-law exponent (3.5 for MRN)
            grain_density: Material density of grains
            dust_to_gas_ratio: Global dust-to-gas mass ratio
            mode: Distribution mode ('proportional' or 'settling')
            alpha: Turbulent viscosity parameter (required for 'settling' mode)
            delta: Turbulent diffusion parameter (default: = alpha, assumes Sc ~ 1)
            mean_molecular_weight: Mean molecular weight for gas (default: 2.3 for H2)
        """
        self.distribution = DustDistribution(
            amin=amin,
            amax=amax,
            nbin=nbin,
            power_index=power_index,
            grain_density=grain_density,
        )
        self.dust_to_gas_ratio = dust_to_gas_ratio
        self.mode = mode
        self.mean_molecular_weight = mean_molecular_weight
        
        # Set turbulence parameters for settling mode
        if mode == 'settling':
            if alpha is None:
                raise ValueError("alpha parameter required for settling mode")
            self.alpha = alpha
            self.delta = delta if delta is not None else alpha  # Assume Sc ~ 1
            logger.info(f"Settling mode: alpha={self.alpha}, delta={self.delta}")
        else:
            self.alpha = alpha
            self.delta = delta
        
        # Create DustBin objects for each bin
        self._bins = {}
        for i in range(nbin):
            dust_bin = DustBin(
                parent_dust=self,
                bin_index=i,
                size=self.distribution.bin_centers[i],
                size_min=self.distribution.bin_edges[i],
                size_max=self.distribution.bin_edges[i + 1],
                mass_fraction=self.distribution.mass_fractions[i],
                density_material=grain_density,
            )
            self._bins[f"bin_{i}"] = dust_bin
        
        logger.info(
            f"Dust distribution set: {nbin} bins from {amin} to {amax}, "
            f"dust/gas={dust_to_gas_ratio:.3e}, power_index={power_index}, mode={mode}"
        )
        
    def _compute_stokes_number(
        self,
        grain_size: Quantity,
        gas_density: Quantity,
        gas_temperature: Quantity,
        keplerian_frequency: Quantity,
    ) -> Quantity:
        """Compute Stokes number for Epstein drag regime.
        
        St = Omega_K * t_s
        where t_s = (rho_s * a) / (rho_g * c_s) for Epstein regime
        
        Args:
            grain_size: Grain radius
            gas_density: Gas density
            gas_temperature: Gas temperature
            keplerian_frequency: Keplerian frequency Omega_K
            
        Returns:
            Stokes number (dimensionless)
        """
        # Material density from distribution
        rho_s = self.distribution.grain_density
        
        # Sound speed: c_s = sqrt(k_B * T / (mu * m_H))
        k_B = 1.380649e-16 * units('erg/K')  # Boltzmann constant
        m_H = 1.6737236e-24 * units('g')      # Hydrogen mass
        mu = self.mean_molecular_weight
        
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
            
            # Get stellar mass (default 1 M_sun if not available)
            M_star = 1.0 * 1.989e33 * units('g')  # Solar mass
            if hasattr(self.parent, 'variables'):
                if 'mstar' in self.parent.variables:
                    M_star = self.parent.variables['mstar']
                    
            # Keplerian frequency
            G = 6.674e-8 * units('cm^3 / (g * s^2)')
            Omega_K = np.sqrt(G * M_star / r**3)
            
            # Sound speed
            k_B = 1.380649e-16 * units('erg/K')
            m_H = 1.6737236e-24 * units('g')
            mu = self.mean_molecular_weight
            c_s = np.sqrt(k_B * temp / (mu * m_H))
            
            # Scale height
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
        M_star = 1.0 * 1.989e33 * units('g')  # Default: 1 M_sun
        if hasattr(self.parent, 'variables'):
            if 'mstar' in self.parent.variables:
                M_star = self.parent.variables['mstar']
                
        G = 6.674e-8 * units('cm^3 / (g * s^2)')
        Omega_K = np.sqrt(G * M_star / r**3)
        
        return Omega_K
        
    def _compute_bin_density(self, bin_index: int) -> Field:
        """Compute dust density field for a specific size bin.
        
        Handles two modes:
        - 'proportional': Simple scaling of gas density
        - 'settling': Settling-diffusion equilibrium with vertical structure
        
        Args:
            bin_index: Index of the size bin (0 to nbin-1)
            
        Returns:
            Field containing dust density for this bin
        """
        # Check for stored field first
        key = f"density_bin_{bin_index}"
        if key in self._dust_fields:
            return self._dust_fields[key]
            
        if self.distribution is None:
            raise RuntimeError("Dust distribution not set")
            
        # Get gas density
        if 'density' in self.parent.gas:
            gas_density = self.parent.gas['density']
        elif 'surface_density' in self.parent.gas:
            # For 2D models, use surface density (proportional mode only)
            if self.mode == 'settling':
                raise ValueError("Settling mode requires 3D model with density field")
            gas_density = self.parent.gas['surface_density']
        else:
            raise KeyError("No gas density or surface_density field found")
            
        # Mode 1: Proportional (simple scaling)
        if self.mode == 'proportional':
            dust_density_data = (
                self.dust_to_gas_ratio * 
                self.distribution.mass_fractions[bin_index] *
                gas_density.data
            )
            
            # Apply mask if set
            if self.mask is not None:
                mask_array = self.mask.data.magnitude.astype(bool)
                dust_density_data = dust_density_data * mask_array
                
            dust_field = Field(
                data=dust_density_data,
                quantity=gas_density.quantity,
                axis_order=gas_density.axis_order,
            )
            
            return dust_field
            
        # Mode 2: Settling-diffusion equilibrium
        elif self.mode == 'settling':
            return self._compute_settling_density(bin_index, gas_density)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def _compute_settling_density(self, bin_index: int, gas_density: Field) -> Field:
        """Compute dust density with settling-diffusion vertical structure.
        
        Args:
            bin_index: Dust bin index
            gas_density: Gas density field
            
        Returns:
            Dust density field with vertical settling profile
        """
        mesh = self.parent.mesh
        grain_size = self.distribution.bin_centers[bin_index]
        mass_fraction = self.distribution.mass_fractions[bin_index]
        
        # Need temperature to compute Stokes number
        if 'temperature' not in self.parent.gas:
            raise ValueError("Settling mode requires temperature field")
        gas_temp = self.parent.gas['temperature'].data
        
        # Get coordinates
        if mesh.coord_system == 'spherical':
            r = mesh.centers('r')
            phi = mesh.centers('phi')
            theta = mesh.centers('theta')
            
            # Create grids
            r_grid, phi_grid, theta_grid = np.meshgrid(r, phi, theta, indexing='ij')
            
            # Compute vertical coordinate z = r * |theta - pi/2|
            # (distance from midplane at theta = pi/2)
            z_grid = r_grid * np.abs(theta_grid - np.pi/2)
            
        elif mesh.coord_system == 'cylindrical':
            r = mesh.centers('r')
            phi = mesh.centers('phi')
            z = mesh.centers('z')
            
            r_grid, phi_grid, z_grid = np.meshgrid(r, phi, z, indexing='ij')
        else:
            raise ValueError(f"Unsupported coordinate system: {mesh.coord_system}")
            
        # Compute Keplerian frequency (function of r only)
        Omega_K_1d = self._get_keplerian_frequency()
        Omega_K_grid = np.meshgrid(Omega_K_1d, phi, 
                                    theta if mesh.coord_system == 'spherical' else z, 
                                    indexing='ij')[0]
        
        # Compute Stokes number at each cell
        St = self._compute_stokes_number(
            grain_size=grain_size,
            gas_density=gas_density.data,
            gas_temperature=gas_temp,
            keplerian_frequency=Omega_K_grid,
        )
        
        # Compute gas scale height (function of r only)
        H_g_1d = self._compute_gas_scale_height()
        H_g_grid = np.meshgrid(H_g_1d, phi,
                               theta if mesh.coord_system == 'spherical' else z,
                               indexing='ij')[0]
        
        # Compute dust scale height
        H_d = self._compute_dust_scale_height(H_g_grid, St)
        
        # Compute vertical profile: exp(-z^2 / (2 * H_d^2))
        vertical_profile = np.exp(-z_grid**2 / (2 * H_d**2))
        
        # Normalize to preserve column density
        # The midplane density enhancement factor ensures that
        # integrating rho_d over z gives the desired surface density
        normalization = 1.0 / (np.sqrt(2 * np.pi) * H_d / H_g_grid)
        
        # Base dust-to-gas ratio scaled by mass fraction
        dust_to_gas_local = self.dust_to_gas_ratio * mass_fraction
        
        # Final dust density
        dust_density_data = (
            gas_density.data * 
            dust_to_gas_local * 
            vertical_profile * 
            normalization
        )
        
        # Apply mask if set
        if self.mask is not None:
            mask_array = self.mask.data.magnitude.astype(bool)
            dust_density_data = dust_density_data * mask_array
            
        axis_order = gas_density.axis_order
        
        dust_field = Field(
            data=dust_density_data,
            quantity=gas_density.quantity,
            axis_order=axis_order,
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
        """Number of dust bins."""
        if self.distribution is None:
            return 0
        return self.distribution.nbin
        
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
