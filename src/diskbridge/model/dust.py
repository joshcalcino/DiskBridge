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
from diskbridge._params import params
from .utils import _interp_sph_to_cyl, _interp_cyl_to_sph
from .mesh import Mesh, Axis

if TYPE_CHECKING:
    from .model import Model

# Import SubModel for inheritance
from .model import SubModel

k_B = 1.380649e-16 * units('erg/K')  # Boltzmann constant
m_H = 1.6737236e-24 * units('g')      # Hydrogen mass

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
        spherical_interp_r_factor: Optional[float] = None,
        spherical_interp_z_factor: Optional[float] = None,
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
            spherical_interp_r_factor: Resolution multiplier for radial direction when 
                converting from spherical to cylindrical coordinates. If None (default),
                automatically determined based on mesh resolution. Typical range: 1-3.
            spherical_interp_z_factor: Resolution multiplier for vertical direction when
                converting from spherical to cylindrical coordinates. If None (default),
                automatically determined based on mesh resolution. Typical range: 1-5.
        """
        # Pull defaults from params if not provided
        if amin is None:
            amin = params.get_dust_size_m('amin') * units('m')
        if amax is None:
            amax = params.get_dust_size_m('amax') * units('m')
        if nbin is None:
            nbin = params.getint('dust_sizes', 'nbins')
        if power_index is None:
            power_index = params.getfloat('dust_sizes', 'pindex')
        if grain_density is None:
            grain_density = params.getfloat('dust_sizes', 'grain_density') * units('g/cm^3')
        if dust_to_gas_ratio is None:
            dust_to_gas_ratio = params.getfloat('dust_sizes', 'dust_to_gas_ratio')
        
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
        
        # Store spherical interpolation resolution factors (will be auto-determined if None)
        self.spherical_interp_r_factor = spherical_interp_r_factor
        self.spherical_interp_z_factor = spherical_interp_z_factor
        
        # Set turbulence parameters for settling mode
        if mode == 'settling':
            if alpha is None:
                # Try to read from params
                alpha = params.getfloat('dust_sizes', 'alpha', fallback=None)
                if alpha is None:
                    raise ValueError("alpha parameter required for settling mode (add to params.txt or pass as argument)")
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
            f"Dust distribution set: {nbin} bins from {amin.to('um')} to {amax.to('um')}, "
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
                    
            # Keplerian frequency (1D array)
            G = 6.674e-8 * units('cm^3 / (g * s^2)')
            Omega_K_1d = np.sqrt(G * M_star / r**3)
            
            # Sound speed (3D array)
            mu = self.mean_molecular_weight
            c_s = np.sqrt(k_B * temp / (mu * m_H))
            
            # Broadcast Omega_K to match c_s shape
            if mesh.coord_system == 'cylindrical':
                phi = mesh.centers('phi')
                z = mesh.centers('z')
                Omega_K = np.meshgrid(Omega_K_1d, phi, z, indexing='ij')[0]
            elif mesh.coord_system == 'spherical':
                phi = mesh.centers('phi')
                theta = mesh.centers('theta')
                Omega_K = np.meshgrid(Omega_K_1d, phi, theta, indexing='ij')[0]
            else:
                raise ValueError(f"Unsupported coordinate system: {mesh.coord_system}")
            
            # Scale height (3D array)
            H = c_s / Omega_K
            return H
            
        raise ValueError("Cannot compute scale height: no disk parameters or temperature")
        
    def _get_spherical_interp_factors(self) -> tuple[float, float]:
        """Determine resolution factors for spherical to cylindrical interpolation.
        
        The factors are automatically determined based on current mesh resolution:
        - Low resolution meshes (<30 cells): Use higher factors (2-3×)
        - Medium resolution meshes (30-60 cells): Use moderate factors (1.5-2×)
        - High resolution meshes (>60 cells): Use minimal factors (1-1.5×)
        
        Returns:
            Tuple of (r_factor, z_factor) for radial and vertical resolution
        """
        # Use user-provided factors if available
        if self.spherical_interp_r_factor is not None and self.spherical_interp_z_factor is not None:
            return self.spherical_interp_r_factor, self.spherical_interp_z_factor
        
        mesh = self.parent.mesh
        
        # Get current mesh resolution
        n_r = mesh.ncell('r')
        n_theta = mesh.ncell('theta') if mesh.coord_system == 'spherical' else mesh.ncell('z')
        
        # Determine radial factor
        if self.spherical_interp_r_factor is not None:
            r_factor = self.spherical_interp_r_factor
        else:
            if n_r < 30:
                r_factor = 2.5
            elif n_r < 60:
                r_factor = 1.5
            else:
                r_factor = 1.0
        
        # Determine vertical factor (typically needs higher resolution than radial)
        if self.spherical_interp_z_factor is not None:
            z_factor = self.spherical_interp_z_factor
        else:
            if n_theta < 30:
                z_factor = 3.5
            elif n_theta < 60:
                z_factor = 2.0
            else:
                z_factor = 1.5
        
        logger.debug(
            f"Spherical interpolation factors (adaptive): "
            f"r_factor={r_factor:.1f}, z_factor={z_factor:.1f} "
            f"(mesh: {n_r}×{n_theta} cells)"
        )
        
        return r_factor, z_factor
    
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
        
        For spherical coordinate systems, this method converts to cylindrical coordinates,
        performs the settling calculation in cylindrical coordinates (where the physics
        naturally applies), and then interpolates back to spherical coordinates.
        
        Args:
            bin_index: Dust bin index
            gas_density: Gas density field
            
        Returns:
            Dust density field with vertical settling profile
        """
        mesh = self.parent.mesh
        
        # For spherical coordinates, use the conversion method
        if mesh.coord_system == 'spherical':
            return self._compute_settling_density_spherical(bin_index, gas_density)
        elif mesh.coord_system == 'cylindrical':
            return self._compute_settling_density_cylindrical(bin_index, gas_density)
        else:
            raise ValueError(f"Unsupported coordinate system: {mesh.coord_system}")
    
    def _compute_settling_density_cylindrical(self, bin_index: int, gas_density: Field) -> Field:
        """Compute dust settling in cylindrical coordinates (native implementation).
        
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
        
        # Get cylindrical coordinates
        r = mesh.centers('r')
        phi = mesh.centers('phi')
        z = mesh.centers('z')
        
        r_grid, phi_grid, z_grid = np.meshgrid(r, phi, z, indexing='ij')
        
        # Compute Keplerian frequency (function of r only)
        Omega_K_1d = self._get_keplerian_frequency()
        Omega_K = Omega_K_1d[:, None]

        # Midplane index (closest to z = 0)
        k0 = int(np.argmin(np.abs(z.magnitude)))

        # Midplane gas density and temperature
        rho_g0 = gas_density.data[:, :, k0]
        T0 = gas_temp[:, :, k0]

        # Midplane sound speed and gas scale height
        mu = self.mean_molecular_weight
        c_s0 = np.sqrt(k_B * T0 / (mu * m_H))
        H_g0 = c_s0 / Omega_K

        # Stokes number at midplane only (one value per column)
        St0 = self._compute_stokes_number(
            grain_size=grain_size,
            gas_density=rho_g0,
            gas_temperature=T0,
            keplerian_frequency=Omega_K,
        )
        St0 = St0[:, :, None]

        # Dust scale height from midplane Stokes number
        delta = self.delta
        H_d = H_g0[:, :, None] * np.sqrt(delta / (St0 + delta))
        
        # Compute vertical profile: exp(-z^2 / (2 * H_d^2))
        vertical_profile = np.exp(-z_grid**2 / (2 * H_d**2))
        
        # Normalize to preserve column density
        # The midplane density enhancement factor ensures that
        # integrating rho_d over z gives the desired surface density
        dust_to_gas_local = self.dust_to_gas_ratio * mass_fraction
        prefactor = dust_to_gas_local * (H_g0[:, :, None] / H_d)

        # Final dust density
        dust_density_data = rho_g0[:, :, None] * prefactor * vertical_profile

        # Ensure numerical stability: clamp NaNs and negative values to zero
        dust_density_mag = dust_density_data.magnitude
        dust_density_mag[~np.isfinite(dust_density_mag)] = 0.0
        dust_density_mag[dust_density_mag < 0.0] = 0.0
        dust_density_data = dust_density_mag * dust_density_data.units

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
    
    def _compute_settling_density_spherical(self, bin_index: int, gas_density: Field) -> Field:
        """Compute dust settling for spherical coordinates via cylindrical conversion.
        
        This method:
        1. Converts gas properties from spherical to cylindrical coordinates
        2. Performs settling calculation in cylindrical coordinates
        3. Interpolates the result back to spherical coordinates
        
        Args:
            bin_index: Dust bin index
            gas_density: Gas density field in spherical coordinates
            
        Returns:
            Dust density field with vertical settling profile in spherical coordinates
        """

        
        mesh_sph = self.parent.mesh
        grain_size = self.distribution.bin_centers[bin_index]
        mass_fraction = self.distribution.mass_fractions[bin_index]
        
        # Need temperature to compute Stokes number
        if 'temperature' not in self.parent.gas:
            raise ValueError("Settling mode requires temperature field")
        gas_temp_sph = self.parent.gas['temperature'].data
        
        logger.debug(f"Computing dust settling in spherical coordinates for bin {bin_index} (converting to cylindrical)")
        
        # Get spherical grid coordinates
        r_sph = mesh_sph.centers('r')
        theta_sph = mesh_sph.centers('theta')
        phi_sph = mesh_sph.centers('phi')
        
        # Get resolution factors (adaptive or user-specified)
        r_factor, z_factor = self._get_spherical_interp_factors()
        
        # Create cylindrical mesh covering the same domain
        # Use higher resolution to reduce interpolation smoothing
        r_sph_min = r_sph.magnitude.min()
        r_sph_max = r_sph.magnitude.max()
        theta_range = theta_sph.magnitude
        
        # Calculate exact extent needed in cylindrical coordinates
        # For spherical (r, theta), cylindrical coords are:
        #   R = r * sin(theta)
        #   Z = r * cos(theta)
        
        # Cylindrical r: maximum occurs at largest r and theta closest to pi/2
        r_cyl_max = r_sph_max * np.sin(theta_range).max()
        n_r_cyl = int(len(r_sph) * r_factor)
        r_cyl_edges = np.linspace(r_sph_min * 0.1, r_cyl_max, n_r_cyl + 1) * r_sph.units
        
        # Cylindrical z: maximum |z| occurs at largest r and theta at extremes
        # Calculate the actual max and min z values from spherical grid corners
        z_at_max_r_min_theta = r_sph_max * np.cos(theta_range.min())  # Upper hemisphere
        z_at_max_r_max_theta = r_sph_max * np.cos(theta_range.max())  # Lower hemisphere
        
        # Take the maximum absolute value to ensure full coverage
        z_max = max(abs(z_at_max_r_min_theta), abs(z_at_max_r_max_theta))
        
        # Add small buffer (5%) to avoid edge effects
        z_max = z_max * 1.05
        
        n_z = int(len(theta_sph) * z_factor)
        z_edges = np.linspace(-z_max, z_max, n_z + 1) * r_sph.units
        
        # Phi: same as spherical
        phi_edges = mesh_sph.edges('phi')
        
        mesh_cyl = Mesh.cylindrical(
            r=Axis(edges=r_cyl_edges),
            phi=Axis(edges=phi_edges),
            z=Axis(edges=z_edges)
        )
        
        r_cyl = mesh_cyl.centers('r')
        phi_cyl = mesh_cyl.centers('phi')
        z_cyl = mesh_cyl.centers('z')
        
        # Transpose arrays to match utils format
        # From (r, phi, theta) to (theta, r, phi) for spherical
        gas_dens_sph_transposed = np.transpose(gas_density.data.magnitude, (2, 0, 1))
        gas_temp_sph_transposed = np.transpose(gas_temp_sph.magnitude, (2, 0, 1))
        
        # Interpolate to cylindrical coordinates
        # utils expect: sph (ntheta, nrad, nsec), output cyl (nz, nrad, nsec)
        gas_dens_cyl_transposed = _interp_sph_to_cyl(
            gas_dens_sph_transposed,
            r_sph.magnitude, theta_sph.magnitude,
            r_cyl.magnitude, z_cyl.magnitude
        )
        gas_temp_cyl_transposed = _interp_sph_to_cyl(
            gas_temp_sph_transposed,
            r_sph.magnitude, theta_sph.magnitude,
            r_cyl.magnitude, z_cyl.magnitude
        )
        
        # Transpose back to (r, phi, z) format
        gas_dens_cyl = np.transpose(gas_dens_cyl_transposed, (1, 2, 0)) * gas_density.data.units
        gas_temp_cyl = np.transpose(gas_temp_cyl_transposed, (1, 2, 0)) * gas_temp_sph.units
        
        # Create grids for cylindrical coordinates
        r_cyl_grid, phi_cyl_grid, z_cyl_grid = np.meshgrid(r_cyl, phi_cyl, z_cyl, indexing='ij')
        
        # Get stellar mass
        M_star = 1.0 * 1.989e33 * units('g')  # Default: 1 M_sun
        if hasattr(self.parent, 'variables'):
            if 'mstar' in self.parent.variables:
                M_star = self.parent.variables['mstar']
        
        # Compute Keplerian frequency in cylindrical coordinates
        G = 6.674e-8 * units('cm^3 / (g * s^2)')
        Omega_K_1d_cyl = np.sqrt(G * M_star / r_cyl**3)
        Omega_K = Omega_K_1d_cyl[:, None]

        # Midplane index (closest to z = 0 in cylindrical grid)
        k0_cyl = int(np.argmin(np.abs(z_cyl.magnitude)))

        # Midplane gas density and temperature in cylindrical coordinates
        rho_g0_cyl = gas_dens_cyl[:, :, k0_cyl]
        T0_cyl = gas_temp_cyl[:, :, k0_cyl]

        # Midplane sound speed and gas scale height
        mu = self.mean_molecular_weight
        c_s0_cyl = np.sqrt(k_B * T0_cyl / (mu * m_H))
        H_g0_cyl = c_s0_cyl / Omega_K

        # Stokes number at midplane only (one value per cylindrical column)
        St0_cyl = self._compute_stokes_number(
            grain_size=grain_size,
            gas_density=rho_g0_cyl,
            gas_temperature=T0_cyl,
            keplerian_frequency=Omega_K,
        )
        St0_cyl = St0_cyl[:, :, None]

        # Dust scale height from midplane Stokes number
        delta = self.delta
        H_d_cyl = H_g0_cyl[:, :, None] * np.sqrt(delta / (St0_cyl + delta))

        # Compute vertical profile: single Gaussian with scale height H_d_cyl
        vertical_profile_cyl = np.exp(-z_cyl_grid**2 / (2 * H_d_cyl**2))

        # Prefactor ensures desired dust surface density per bin
        dust_to_gas_local = self.dust_to_gas_ratio * mass_fraction
        prefactor_cyl = dust_to_gas_local * (H_g0_cyl[:, :, None] / H_d_cyl)

        # Final dust density in cylindrical coordinates
        dust_dens_cyl = rho_g0_cyl[:, :, None] * prefactor_cyl * vertical_profile_cyl

        # Ensure numerical stability: clamp NaNs and negative values to zero
        dust_dens_cyl_mag = dust_dens_cyl.magnitude
        dust_dens_cyl_mag[~np.isfinite(dust_dens_cyl_mag)] = 0.0
        dust_dens_cyl_mag[dust_dens_cyl_mag < 0.0] = 0.0
        dust_dens_cyl = dust_dens_cyl_mag * dust_dens_cyl.units

        # Transpose for interpolation back to spherical
        # From (r, phi, z) to (z, r, phi) for cylindrical
        dust_dens_cyl_transposed = np.transpose(dust_dens_cyl.magnitude, (2, 0, 1))
        
        # Interpolate back to spherical coordinates
        # utils expect: cyl (nz, nrad, nsec), output sph (ntheta, nrad, nsec)
        dust_dens_sph_transposed = _interp_cyl_to_sph(
            dust_dens_cyl_transposed,
            r_cyl.magnitude, z_cyl.magnitude,
            r_sph.magnitude, theta_sph.magnitude
        )
        
        # Transpose back to (r, phi, theta) format
        dust_dens_sph = np.transpose(dust_dens_sph_transposed, (1, 2, 0)) * dust_dens_cyl.units
        
        # Apply mask if set
        if self.mask is not None:
            mask_array = self.mask.data.magnitude.astype(bool)
            dust_dens_sph = dust_dens_sph * mask_array
        
        axis_order = gas_density.axis_order
        
        dust_field = Field(
            data=dust_dens_sph,
            quantity=gas_density.quantity,
            axis_order=axis_order,
        )
        
        logger.debug(f"Dust settling calculation completed for bin {bin_index} in spherical coordinates")
        
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
