"""Dust grain submodel for DiskBridge.

This module provides dust grain distribution modeling with power-law size distributions,
memory-efficient storage for gas-proportional dust, and integration with radmc3d.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Callable, List, Literal, Union
import numpy as np

from diskbridge._units import Quantity, units
from diskbridge._logging import logger
from diskbridge.model.field import Field
from diskbridge.model.disk import scale_height, keplerian_frequency
from diskbridge._params import canonicalize_dust_params

if TYPE_CHECKING:
    from .core import Model

# Import SubModel for inheritance
from .core import SubModel

k_B = units('k_B')  # Boltzmann constant
m_H = units('m_H')  # Hydrogen mass
G = units('G')      # Gravitational constant

PAH_N_C_DEFAULT = 54
PAH_N_H_DEFAULT = 18
PAH_X_ISM_DEFAULT = 3.0e-7 * (50.0 / PAH_N_C_DEFAULT)
AMU_G = 1.66053906660e-24
PAH_MASS_G_DEFAULT = (
    PAH_N_C_DEFAULT * 12.011 + PAH_N_H_DEFAULT * 1.008
) * AMU_G


def _current_params():
    import diskbridge

    return diskbridge.params


def stokes_number(
    grain_size: Quantity,
    gas_density: Quantity,
    gas_temperature: Quantity,
    keplerian_freq: Quantity,
    grain_density: Quantity,
    mean_molecular_weight: float,
) -> Quantity:
    """Compute Stokes number for Epstein drag regime.
    
    St = Omega_K * t_s where t_s = (rho_s * a) / (rho_g * c_s)
    
    Parameters
    ----------
    grain_size : Quantity
        Grain radius [length]
    gas_density : Quantity
        Gas density [mass/volume]
    gas_temperature : Quantity
        Gas temperature [temperature]
    keplerian_freq : Quantity
        Keplerian frequency Omega_K [1/time]
    grain_density : Quantity
        Material density of dust grains [mass/volume]
    mean_molecular_weight : float
        Mean molecular weight of gas (dimensionless)
        
    Returns
    -------
    Quantity
        Stokes number (dimensionless)
        
    Notes
    -----
    Valid in the Epstein drag regime where grain size << gas mean free path.
    """
    c_s = np.sqrt(k_B * gas_temperature / (mean_molecular_weight * m_H))
    t_s = (grain_density * grain_size) / (gas_density * c_s)
    St = keplerian_freq * t_s
    
    return St

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

        current_params = _current_params()

        # Pull defaults from params if not provided
        if amin is None:
            # params.amin is already a Quantity in microns
            amin = current_params.amin
        if amax is None:
            # params.amax is already a Quantity in microns
            amax = current_params.amax
        if nbin is None:
            nbin = current_params.nbins
        if power_index is None:
            power_index = current_params.pindex
        if grain_density is None:
            # params.grain_density is already a Quantity in g/cm^3
            grain_density = current_params.grain_density
        if dust_to_gas_ratio is None:
            dust_to_gas_ratio = current_params.dust_to_gas_ratio
        
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
                if self.parent.disk is not None:
                    alpha_q = self.parent.disk.parameters.get("alphavisocity")
                else:
                    alpha_q = None
                if alpha_q is None:
                    raise ValueError("alpha parameter required for settling mode")
                alpha = float(getattr(alpha_q, "magnitude", alpha_q))
            delta = delta if delta is not None else alpha  # Assume Sc ~ 1
            logger.info(f"Settling mode: alpha={alpha}, delta={delta}")
        
        # Get species base name from params
        species_base = (
            current_params.species
            if isinstance(current_params.species, str)
            else current_params.species[0]
        )
        
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
    
    def _resolve_mask_field(
        self,
        mask: Union[Field, str],
        *,
        complement: bool = False,
    ) -> Field:
        if isinstance(mask, str):
            if self.parent.gas is None or mask not in self.parent.gas:
                raise KeyError(f"Gas mask/weight field {mask!r} not found")
            source = self.parent.gas[mask]
            source_name = mask
        else:
            source = mask
            source_name = None

        from .utils import validate_field_against_mesh, field_data_as_order

        validate_field_against_mesh(source, self.parent.mesh)  # type: ignore[arg-type]
        target = self.parent.mesh.axis_names()  # type: ignore[union-attr]
        data = field_data_as_order(source, target).to("dimensionless")
        values = np.clip(np.asarray(data.magnitude, dtype=float), 0.0, 1.0)
        attrs = dict(source.attrs)

        if complement:
            values = 1.0 - values
            if source_name is not None:
                attrs["complement_of"] = source_name

        return Field(
            data=Quantity(values, "dimensionless"),
            quantity=source.quantity,
            axis_order=target,
            attrs=attrs,
        )

    def add_component_from_mask(
        self,
        mask: Union[Field, str],
        mode: Literal['proportional', 'settling'] = 'proportional',
        complement: bool = False,
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
            mask: Field or gas field name defining where this component exists.
            mode: 'proportional' or 'settling'
            complement: If True, use 1 - mask after clipping mask values to [0, 1].
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
        mask = self._resolve_mask_field(mask, complement=complement)

        # Determine this component's index
        self._has_region_components = True
        component_index = len(self._components)

        # Pull canonical dust parameters from global params
        canon = canonicalize_dust_params(_current_params())
        ncomp = canon['ncomp']
        if component_index >= ncomp:
            if ncomp == 1:
                param_index = 0
            else:
                raise ValueError(
                    f"Requested dust component index {component_index}, "
                    f"but canonicalized params define only {ncomp} components."
                )
        else:
            param_index = component_index

        # Use canonical per-component values if not explicitly provided
        if amin is None:
            amin = canon['amin'][param_index]
        if amax is None:
            amax = canon['amax'][param_index]
        if nbin is None:
            nbin = int(canon['nbins'][param_index])
        if power_index is None:
            power_index = float(canon['pindex'][param_index])
        if grain_density is None:
            grain_density = canon['grain_density'][param_index]
        if dust_to_gas_ratio is None:
            dust_to_gas_ratio = float(canon['dust_to_gas_ratio'][param_index])
        
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
                if self.parent.disk is not None:
                    alpha_q = self.parent.disk.parameters.get("alphavisocity")
                else:
                    alpha_q = None
                if alpha_q is None:
                    raise ValueError("alpha parameter required for settling mode")
                alpha = float(getattr(alpha_q, "magnitude", alpha_q))
            delta = delta if delta is not None else alpha
            logger.info(f"Settling mode component: alpha={alpha}, delta={delta}")
        
        # Get species base name for this component
        canon_species = canon['species']
        if isinstance(canon_species, list):
            species_base = canon_species[param_index]
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

    def add_to_disk(
        self,
        mode: Optional[Literal['proportional', 'settling']] = None,
        *,
        include_ism: bool = True,
        alpha: Optional[float] = None,
    ) -> None:
        """Add the canonical disk dust component and, by default, ISM dust."""
        if self.parent.gas is None or "disk_mask" not in self.parent.gas:
            raise KeyError("Disk mask not found; call set_mask_from_joos_disk first")

        current_params = _current_params()
        disk_mode = mode if mode is not None else current_params.disk_dust_mode
        if disk_mode not in ("proportional", "settling"):
            raise ValueError("disk_dust_mode must be 'proportional' or 'settling'")

        disk_alpha = alpha
        if disk_mode == "settling" and disk_alpha is None:
            disk_alpha = current_params.disk_dust_settling_alpha

        self.add_component_from_mask(
            mask="disk_mask",
            mode=disk_mode,
            alpha=disk_alpha,
        )

        if include_ism:
            self.add_component_from_mask(
                mask="disk_mask",
                complement=True,
                mode="proportional",
            )

    def add_pah_to_disk(
        self,
        *,
        f_pah_disk: float = 0.01,
        include_ism: bool = True,
        disk_weight_field: str = "disk_weight",
        disk_mask_field: str = "disk_mask",
    ) -> None:
        """Construct PAH abundance fields from the disk/ISM material split.

        PAHs are registered as gas-side chemistry/material fields, not as
        ordinary dust bins, so they do not contribute to dust surface-area
        moments used for gas-dust coupling or CO freeze-out.
        """
        if self.parent.gas is None:
            raise KeyError("Model has no gas submodel")
        if "density" not in self.parent.gas:
            raise KeyError("Need gas['density'] to construct PAH abundance")
        if f_pah_disk < 0.0:
            raise ValueError("f_pah_disk must be non-negative")

        if disk_weight_field in self.parent.gas:
            w_field = self.parent.gas[disk_weight_field]
        elif disk_mask_field in self.parent.gas:
            w_field = self.parent.gas[disk_mask_field]
        else:
            raise KeyError("Need disk_weight or disk_mask to construct disk PAH field")

        target = self.parent.mesh.axis_names()  # type: ignore[union-attr]
        from .utils import field_data_as_order

        rho_gas = field_data_as_order(
            self.parent.gas["density"], target
        ).to("g/cm^3")
        w_disk_q = field_data_as_order(w_field, target).to("dimensionless")
        w_disk = np.clip(np.asarray(w_disk_q.magnitude, dtype=float), 0.0, 1.0)
        w_ism = 1.0 - w_disk

        # Match the nH convention used by the RADMC/GOW17 bridge.
        nH = rho_gas / (1.4 * m_H)
        if include_ism:
            D_pah = w_ism + f_pah_disk * w_disk
        else:
            D_pah = f_pah_disk * w_disk
        D_pah = np.where(np.isfinite(D_pah) & (D_pah >= 0.0), D_pah, 0.0)
        rho_pah = (
            nH
            * Quantity(PAH_X_ISM_DEFAULT, "dimensionless")
            * Quantity(PAH_MASS_G_DEFAULT, "g")
            * Quantity(D_pah, "dimensionless")
        ).to("g/cm^3")
        attrs = {
            "role": "pah",
            "pah_model": "C54H18",
            "N_C": PAH_N_C_DEFAULT,
            "N_H": PAH_N_H_DEFAULT,
            "x_PAH_ISM": PAH_X_ISM_DEFAULT,
            "m_PAH_g": PAH_MASS_G_DEFAULT,
            "f_PAH_disk": float(f_pah_disk),
            "include_ism": bool(include_ism),
            "source": "disk_mask_mixture",
        }
        self.parent.gas_register(
            "pah_abundance_rel_ism",
            Field(
                quantity="pah_abundance_rel_ism",
                data=Quantity(D_pah, "dimensionless"),
                axis_order=target,
                attrs=attrs,
            ),
        )
        self.parent.gas_register(
            "pah_density",
            Field(
                quantity="pah_density",
                data=rho_pah,
                axis_order=target,
                attrs=attrs,
            ),
        )
        
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
        # FLAG THIS FOR FUTURE CONSIDERATION
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
                H = scale_height(r, h0, r0, fl)
                return H
                
        # Option 2: Compute from temperature and stellar mass
        if 'temperature' in self.parent.gas:
            temp = self.parent.gas['temperature'].data
            
            # Get stellar mass
            M_star = units('solar_mass').to('g')
            if hasattr(self.parent, 'variables'):
                if 'mstar' in self.parent.variables:
                    M_star = self.parent.variables['mstar']
                    
            # Keplerian frequency (1D array)
            Omega_K_1d = keplerian_frequency(r, M_star)
            
            # Sound speed (3D array)
            mu = self.mean_molecular_weight
            c_s = np.sqrt(k_B * temp / (mu * m_H))
            
            # Broadcast Omega_K to match c_s shape 
            phi = mesh.centers('phi')
            theta = mesh.centers('theta')
            Omega_K = np.meshgrid(Omega_K_1d, theta, phi, indexing='ij')[0]
            
            # Scale height (3D array)
            H = c_s / Omega_K
            return H
            
        raise ValueError("Cannot compute scale height: no disk parameters or temperature")
    
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
                w = np.asarray(component.mask.data.to("dimensionless").magnitude, dtype=float)
                w = np.clip(w, 0.0, 1.0)
                dust_density_data = dust_density_data * w
                
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
        """Compute dust density using an analytic settling profile in spherical coordinates.

        This routine builds a simple settling-diffusion equilibrium scaling using midplane
        gas properties to compute H_g and H_d, and applies a Gaussian profile
        exp(-z^2 / (2 H_d^2)).

        Note:
            This routine does not renormalize columns to enforce exact mass conservation.

        If component.mask is set, the resulting density is multiplied by the mask field.
        The mask may be boolean or float weights; values are clipped to [0, 1].

        Args:
            component: DustComponent instance.
            local_bin_idx: Dust bin index within component.
            gas_density: Gas density field in spherical coordinates.

        Returns:
            Dust density field with vertical settling profile.
        """
        mesh = self.parent.mesh
        grain_size = component.distribution.bin_centers[local_bin_idx]
        mass_fraction = component.distribution.mass_fractions[local_bin_idx]
        
        has_temperature = 'temperature' in self.parent.gas
        has_pressure = 'pressure' in self.parent.gas
        if not has_temperature and not has_pressure:
            raise ValueError("Settling mode requires temperature or pressure field")
        gas_temp = self.parent.gas['temperature'].data if has_temperature else None
        gas_pressure = self.parent.gas['pressure'].data if has_pressure else None
        
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
        
        # Create 3D grids
        r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi, indexing='ij')
        
        # Compute cylindrical coordinates for each spherical cell
        z_cyl = r_grid * np.cos(theta_grid)
        
        # Find midplane index (theta closest to pi/2)
        theta_mid_idx = np.argmin(np.abs(theta.to("radian") - np.pi/2))
        
        # Get midplane properties
        rho_g0 = gas_density.data[:, theta_mid_idx, :]
        T0 = gas_temp[:, theta_mid_idx, :] if gas_temp is not None else None
        P0 = gas_pressure[:, theta_mid_idx, :] if gas_pressure is not None else None

        mu = component.mean_molecular_weight
        if T0 is None:
            T0 = (mu * m_H / k_B) * (P0 / rho_g0)
        
        # Get stellar mass
        M_star = units('solar_mass')
        if hasattr(self.parent, 'variables') and 'mstar' in self.parent.variables:
            M_star = self.parent.variables['mstar']
        
        # Compute Keplerian frequency at r
        r = self.parent.mesh.centers('r')
        Omega_K_1d = keplerian_frequency(r, M_star)
        Omega_K = Omega_K_1d[:, None]
        
        # Sound speed at midplane
        c_s0 = np.sqrt(k_B * T0 / (mu * m_H))
        
        # Gas scale height at midplane: H_g = c_s / Omega_K
        # Convert to same units as r for consistent length units
        H_g0 = (c_s0 / Omega_K).to(r.units)
        
        # Stokes number at midplane
        St0 = stokes_number(
            grain_size=grain_size,
            gas_density=rho_g0,
            gas_temperature=T0,
            keplerian_freq=Omega_K,
            grain_density=component.distribution.grain_density,
            mean_molecular_weight=mu,
        )
        
        # Dust scale height from midplane Stokes number
        delta = component.delta
        H_d0 = H_g0 * np.sqrt(delta / (St0 + delta))
        
        # Expand to 3D (midplane values expanded over theta)
        n_theta = len(theta)
        n_phi = len(phi)
        H_d = np.broadcast_to(H_d0[:, None, :], (H_d0.shape[0], n_theta, n_phi))
        
        # Compute vertical profile: exp(-z^2 / (2 * H_d^2))
        expo = (-(z_cyl**2) / (2 * H_d**2)).to("dimensionless").magnitude
        vertical_profile = np.exp(expo)
        
        # Compute dust density directly from analytic settling profile, without
        # additional column renormalization. This makes the dust density a
        # simple function of local gas density, dtg, and the vertical settling
        # profile, and is useful for inspecting the raw behaviour.

        dust_to_gas_local = component.dust_to_gas_ratio * mass_fraction
        prefactor = dust_to_gas_local * (H_g0[:, None, :] / H_d)
        dust_density_data = rho_g0[:, None, :] * prefactor * vertical_profile

        # Apply mask if set
        if component.mask is not None:
            w = np.asarray(component.mask.data.to("dimensionless").magnitude, dtype=float)
            w = np.clip(w, 0.0, 1.0)
            dust_density_data = dust_density_data * w

        # Ensure numerical stability
        dust_density_data = np.where(np.isfinite(dust_density_data), dust_density_data, 0.0)
        dust_density_data = np.where(dust_density_data >= 0.0, dust_density_data, 0.0)
        
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
        for i in range(self.nbin):
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
        from .utils import validate_field_against_mesh, field_data_as_order

        validate_field_against_mesh(field, self.mesh)  # type: ignore[arg-type]
        target = self.mesh.axis_names()  # type: ignore[union-attr]
        if field.axis_order != target:
            logger.warning(
                "dust.register canonicalizing field '%s' axis_order %s -> %s",
                name,
                field.axis_order,
                target,
            )
        data = field_data_as_order(field, target)
        self._dust_fields[name] = Field(
            quantity=field.quantity,
            data=data,
            axis_order=target,
            attrs=field.attrs,
        )
        
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
