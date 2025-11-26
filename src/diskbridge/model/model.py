from __future__ import annotations

from dataclasses import dataclass, field as dcfield
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import numpy as np
import pickle

from .mesh import Mesh, Axis
from .field import Field
from diskbridge._logging import logger
from diskbridge._units import Quantity
from .utils import validate_field_against_mesh, _interp_cyl_to_sph


class Model:

    """
    Add in documentation
    """

    def __init__(self):
        self.coord_system: Optional[str] = None
        self.variables: Dict[str, Any] = {}
        self.compile_options: Dict[str, Optional[bool]] = {}
        self.macros: Dict[str, float] = {}
        self.mesh: Optional[Mesh] = None
        self.file_units: Optional[str] = None # 'kms', 'cgs', or 'code'
        self.directory: Optional[str] = None
        self.n_file: Optional[int] = None
        self.filename: Optional[str] = None 
        
        # Initialize submodels
        self.gas: SubModel = None
        self.disk: Disk = None
        self.dust: 'Dust' = None  # Dust submodel
        
    def get_variables(self) -> Dict[str, Any]:
        return dict(self.variables)

    def gas_register(self, name: str, field: Field) -> None:
        # validate against the model's single mesh
        validate_field_against_mesh(field, self.mesh)  # type: ignore[arg-type]
        self.gas.register(name, field)

    def gas_register_lazy(self, name: str, builder: Callable[[], Field]) -> None:
        self.gas.register_lazy(name, builder)

    def _apply_rescaling(self, length_scale=None, mass_scale=None) -> None:
        """
        Internal method to apply length and mass rescaling during initialization.
        
        Parameters
        ----------
        length_scale : float, optional
            Dimensionless multiplier for length (e.g., 10 means 1 code_length = 10 au)
        mass_scale : float, optional
            Dimensionless multiplier for mass (e.g., 2 means 1 code_mass = 2 M_sun)
            
        Notes
        -----
        Code units are already defined as:
        - code_length = 1 au
        - code_mass = 1 M_sun
        - code_time = sqrt(au^3 / (G * M_sun))

        """
        # Get dimensionless scaling factors (default to 1 if not provided)
        length_factor = float(length_scale) if length_scale is not None else 1.0
        mass_factor = float(mass_scale) if mass_scale is not None else 1.0
        
        # Derived scaling factors based on Keplerian dynamics
        # Time: T² ∝ L³/M → time_factor = sqrt(length_factor³ / mass_factor)
        time_factor = np.sqrt(length_factor**3 / mass_factor)
        
        # Velocity: V = L/T → velocity_factor = length_factor / time_factor = sqrt(mass_factor / length_factor)
        velocity_factor = np.sqrt(mass_factor / length_factor)
        
        # Apply rescaling only if at least one scale is provided
        if length_scale is not None or mass_scale is not None:
            # Rescale mesh coordinates
            if self.mesh is not None:
                self.mesh = self.mesh.rescale_length(length_factor)
            
            # Rescale gas fields
            if self.gas is not None:
                for name, field in list(self.gas.items()):
                    new_data = field.data
                    
                    if 'density' in name or 'surface_density' in name:
                        # Volume density: ρ = M/L³ → scale by mass_factor/length_factor³
                        # Surface density: Σ = M/L² → scale by mass_factor/length_factor²
                        if 'surface' in name:
                            new_data = field.data * (mass_factor / length_factor**2)
                        else:
                            new_data = field.data * (mass_factor / length_factor**3)
                    
                    elif name in ['vr', 'vphi', 'vz', 'vtheta']:
                        # Velocity: V = L/T → scale by sqrt(M/L)
                        new_data = field.data * velocity_factor
                    
                    elif name == 'temperature':
                        # Temperature: cutemp ∝ M/L, so T → T × (M/L)
                        # When length increases, temperature decreases
                        temp_factor = mass_factor / length_factor
                        new_data = field.data * temp_factor
                    
                    # Replace field with rescaled version
                    if new_data is not field.data:
                        self.gas[name] = Field(
                            data=new_data,
                            quantity=field.quantity,
                            axis_order=field.axis_order,
                            attrs=field.attrs
                        )
            
            # Store the scaling factors
            self.length_scale = length_factor
            self.mass_scale = mass_factor
            self.time_scale = time_factor
            self.velocity_scale = velocity_factor

    def load_model(
        self,
        path: Union[str, Path],
        reader: str = "fargo",
        file_n: int = 0,
        file_units: str = "code",
        length_scale: Optional[float] = None,
        mass_scale: Optional[float] = None,
    ) -> "Model":
        """
        Load a hydro snapshot and populate this Model, then return self.
        
        Parameters
        ----------
        path : str or Path
            Path to the simulation data directory
        reader : str, optional
            Reader type (default: "fargo")
        file_n : int, optional
            File number to load (default: 0)
        file_units : str, optional
            Unit system of the files: 'code', 'cgs', or 'kms' (default: 'code')
        length_scale : float, optional
            Dimensionless multiplier for length units. For code units (1 au), 
            this rescales by this factor (e.g., 10 means 1 code_length = 10 au).
            Time is also rescaled following Keplerian dynamics.
        mass_scale : float, optional
            Dimensionless multiplier for mass units. For code units (1 M_sun),
            this rescales by this factor (e.g., 2 means 1 code_mass = 2 M_sun).
            Velocities are rescaled as V -> V*sqrt(mass_scale/length_scale).
        
        Returns
        -------
        Model
            The populated model instance
            
        Notes
        -----
        Rescaling follows Keplerian dynamics where T² ∝ L³/M:
        - Lengths scale by length_scale
        - Masses scale by mass_scale  
        - Times scale by sqrt(length_scale³/mass_scale)
        - Velocities scale by sqrt(mass_scale/length_scale)
        - Densities scale by mass_scale/length_scale³
        """
        p = Path(path)
        if reader.lower() == "fargo":
            from .readers.fargo import read_fargo_snapshot

            snap = read_fargo_snapshot(p, file_n=file_n, file_units=file_units)
        else:
            raise ValueError(f"Unsupported reader: {reader}")

        # Populate instance
        self.coord_system = snap["coord_system"]
        self.variables = snap["variables"]
        self.compile_options = snap.get("compile_options", {})
        self.macros = snap.get("macros", {})
        self.mesh = snap["mesh"]
        self.file_units = file_units 
        self.directory = str(p)
        self.n_file = file_n
        self.filename = None
        
        # Initialize submodels
        self.gas = SubModel(self)
        
        # Initialize Dust submodel (lazy import to avoid circular dependency)
        from .dust import Dust
        self.dust = Dust(self)

        # Initialize Disk parameters if present
        if "disk_parameters" in snap:
            self.disk = Disk(self, snap["disk_parameters"])

        # Register gas fields
        for name, field in snap["gas_fields"].items():
            self.gas_register(name, field)
        
        # Apply rescaling if requested
        if length_scale is not None or mass_scale is not None:
            self._apply_rescaling(length_scale=length_scale, mass_scale=mass_scale)
            # Update SubModel mesh references after rescaling
            if self.gas is not None:
                self.gas.mesh = self.mesh
            if self.dust is not None:
                self.dust.mesh = self.mesh
            if hasattr(self, 'disk') and self.disk is not None:
                self.disk.mesh = self.mesh

        return self
    
    def set_mask_from_geometry(
        self,
        r_min: Optional[Quantity] = None,
        r_max: Optional[Quantity] = None,
        theta_min: Optional[Quantity] = None,
        theta_max: Optional[Quantity] = None,
        honrmax: Optional[float] = None,
        is_a_disk: bool = False,
    ) -> SubModel:
        """Define a masked region on the model.
        
        If is_a_disk=True and self.disk exists, configure self.disk's mask.
        Otherwise, create and return a new SubModel with the requested mask.
        
        Args:
            r_min: Minimum radius
            r_max: Maximum radius
            theta_min: Minimum colatitude (spherical)
            theta_max: Maximum colatitude (spherical)
            honrmax: Number of pressure scale heights (disk-specific, requires self.disk)
            is_a_disk: Mark region as disk (enables settling mode)
            
        Returns:
            SubModel (or Disk) with mask applied
        """
        mesh = self.mesh
        if mesh is None:
            raise ValueError("Model has no mesh")
            
        if mesh.coord_system != 'spherical':
            raise ValueError(
                f"set_mask_from_geometry only supports spherical coordinates, "
                f"got {mesh.coord_system}"
            )
        
        # Determine target region
        if is_a_disk and self.disk is not None:
            target_region = self.disk
        else:
            target_region = SubModel(self)
        
        # Get coordinate arrays
        r = mesh.centers('r')
        theta = mesh.centers('theta')
        phi = mesh.centers('phi')
        
        r_grid, phi_grid, theta_grid = np.meshgrid(r, phi, theta, indexing='ij')
        
        # Start with all True
        mask = np.ones_like(r_grid, dtype=bool)
        
        # Apply radial constraints
        if r_min is not None:
            mask &= (r_grid.magnitude >= r_min.to(r.units).magnitude)
        if r_max is not None:
            mask &= (r_grid.magnitude <= r_max.to(r.units).magnitude)
        
        # If honrmax is provided and we have disk parameters, compute hydrostatic mask
        if honrmax is not None and self.disk is not None:
            # Convert spherical to cylindrical for height calculation
            R_cyl = r_grid * np.sin(theta_grid)
            z_cyl = r_grid * np.cos(theta_grid)
            
            # Get disk parameters
            h0 = self.disk.parameters["aspectratio"]
            fl = self.disk.parameters["flaringindex"]
            r0 = self.disk.parameters["r0"]
            
            # h(R_cyl) = h0 * (R_cyl/r0)^fl, dimensionless
            h = h0 * (R_cyl / r0.to(R_cyl.units)) ** fl
            
            # H(R_cyl) = h * R_cyl, scale height with units
            H = h * R_cyl
            
            # Mask: |z| <= honrmax * H(R_cyl)
            z_max = honrmax * H
            mask &= (np.abs(z_cyl) <= z_max)
        
        # Apply theta constraints if given
        if theta_min is not None:
            mask &= (theta_grid.magnitude >= theta_min.to(theta.units).magnitude)
        if theta_max is not None:
            mask &= (theta_grid.magnitude <= theta_max.to(theta.units).magnitude)
        
        axis_order = ('r', 'phi', 'theta')
        
        mask_quantity = Quantity(mask, 'dimensionless')
        mask_field = Field(
            data=mask_quantity,
            quantity='mask',
            axis_order=axis_order,
        )
        
        target_region.mask = mask_field
        
        # Mark as disk region if requested
        if is_a_disk:
            target_region.is_disk_region = True
        
        logger.info(
            f"{target_region.__class__.__name__} mask set: {np.sum(mask)} / {mask.size} cells "
            f"({100*np.sum(mask)/mask.size:.1f}%)"
        )
        
        return target_region

    def puff_up_model(
        self, n: int, 
        coordinates: str = "cylindrical", 
        zmax_over_H: float = 5.0
    ) -> "Model":
        self.disk.puff_up_disk(n=n, coordinates=coordinates, zmax_over_H=zmax_over_H)
        return self
        

class SubModel:
    """Component that holds/modifies fields in a region."""

    def __init__(self, parent: Model):
        self.parent = parent
        self.mesh = parent.mesh
        self.coord_system = parent.coord_system
        self._fields: Dict[str, Field] = {}
        self._lazy: Dict[str, Callable[[], Field]] = {}

        # SubModel specific properties
        self.mask: Optional[Field] = None
        self.is_disk_region: bool = False  # Marks if this region allows settling

    # Registry API
    def register(self, name: str, field: Field) -> None:
        validate_field_against_mesh(field, self.mesh)  # type: ignore[arg-type]
        self._fields[name] = field

    def register_lazy(self, name: str, builder: Callable[[], Field]) -> None:
        self._lazy[name] = builder

    # Mapping-like API
    def __getitem__(self, key: str) -> Field:
        # Guard: request for volume density requires 3D mesh
        if key == "density":
            try:
                nd = getattr(self.mesh, "ndims", None)
            except Exception:
                nd = None
            if nd is not None and nd != 3:
                logger.error("'density' is unavailable: simulation is %sd (use 'surface_density')", nd)
                raise KeyError("density not available: simulation is 2D; use 'surface_density'")

        if key in self._fields:
            return self._fields[key]
        if key in self._lazy:
            field = self._lazy[key]()
            self._fields[key] = field
            return field
        raise KeyError(key)

    def __setitem__(self, key: str, value: Field) -> None:
        self._fields[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._fields or key in self._lazy

    def keys(self):
        return self._fields.keys()

    def items(self):
        return self._fields.items()

    def clear(self) -> None:
        self._fields.clear()
    
    @property
    def dust(self):
        """Access to dust with region-aware configuration."""
        if not hasattr(self.parent, 'dust') or self.parent.dust is None:
            raise AttributeError("Parent model has no dust submodel")
        return _RegionDustConfigurator(
            global_dust=self.parent.dust,
            mask=self.mask,
            is_disk_region=self.is_disk_region,
        )

    def set_mask_from_density(
        self,
        density_threshold: Quantity,
        r_min: Optional[Quantity] = None,
        r_max: Optional[Quantity] = None,
    ) -> Field:
        """Create mask based on density threshold.
        
        Args:
            density_threshold: Minimum density for inclusion
            r_min: Optional minimum radius
            r_max: Optional maximum radius
            
        Returns:
            Field containing boolean mask
        """
        
        # Get density from the submodel
        if 'density' in self:
            density = self['density'].data
        else:
            raise KeyError("No density field found in submodel")
        
        mesh = self.mesh
        if mesh.coord_system != 'spherical':
            raise ValueError(
                f"set_mask_from_density only supports spherical coordinates, "
                f"got {mesh.coord_system}"
            )
            
        threshold_val = density_threshold.to(density.units).magnitude
        mask = (density.magnitude >= threshold_val)
        
        # Apply radial constraints if given
        if r_min is not None or r_max is not None:
            r = mesh.centers('r')
            phi = mesh.centers('phi')
            theta = mesh.centers('theta')
            r_grid, _, _ = np.meshgrid(r, phi, theta, indexing='ij')
            
            if r_min is not None:
                mask &= (r_grid.magnitude >= r_min.to(r.units).magnitude)
            if r_max is not None:
                mask &= (r_grid.magnitude <= r_max.to(r.units).magnitude)
        
        axis_order = ('r', 'phi', 'theta')
                
        mask_quantity = Quantity(mask, 'dimensionless')
        mask_field = Field(
            data=mask_quantity,
            quantity='mask',
            axis_order=axis_order,
        )
        
        self.mask = mask_field
        
        logger.info(
            f"{self.__class__.__name__} mask set (density): {np.sum(mask)} / {mask.size} cells "
            f"({100*np.sum(mask)/mask.size:.1f}%)"
        )
        
        return mask_field
        
    def set_mask_from_array(self, mask_array: np.ndarray) -> Field:
        """Set mask from custom boolean array.
        
        Args:
            mask_array: Boolean array matching grid shape
            
        Returns:
            Field containing mask
        """
        
        mesh = self.mesh
        
        # Verify shape
        if 'density' in self:
            expected_shape = self['density'].data.shape
        else:
            # Get shape from mesh
            if mesh.coord_system == 'spherical':
                expected_shape = (len(mesh.axes['r'].centers),
                                len(mesh.axes['phi'].centers),
                                len(mesh.axes['theta'].centers))
            elif mesh.coord_system == 'cylindrical':
                expected_shape = (len(mesh.axes['r'].centers),
                                len(mesh.axes['phi'].centers),
                                len(mesh.axes['z'].centers))
            else:
                raise ValueError(f"Cannot determine shape for {mesh.coord_system}")
            
        if mask_array.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask_array.shape} doesn't match grid shape {expected_shape}"
            )
            
        if mesh.coord_system == 'spherical':
            axis_order = ('r', 'phi', 'theta')
        elif mesh.coord_system == 'cylindrical':
            axis_order = ('r', 'phi', 'z')
        else:
            axis_order = None
            
        mask_quantity = Quantity(mask_array.astype(bool), 'dimensionless')
        mask_field = Field(
            data=mask_quantity,
            quantity='mask',
            axis_order=axis_order,
        )
        
        self.mask = mask_field
        
        logger.info(
            f"{self.__class__.__name__} mask set: {np.sum(mask_array)} / {mask_array.size} cells "
            f"({100*np.sum(mask_array)/mask_array.size:.1f}%)"
        )
        
        return mask_field
    
    def anti_mask(self) -> "SubModel":
        """Create a new SubModel with the complement of this submodel's mask.
        
        Returns:
            SubModel with inverted mask
        """
        if self.mask is None:
            raise ValueError("This SubModel has no mask, cannot create an anti-mask")
        
        # Compute boolean complement
        mask_bool = self.mask.data.magnitude.astype(bool)
        other_bool = ~mask_bool
        
        # Create new SubModel for the complement region
        other = SubModel(self.parent)
        other.set_mask_from_array(other_bool)
        
        # Explicitly mark as not a disk region
        other.is_disk_region = False
        
        logger.info(
            f"Created complement region: {np.sum(other_bool)} / {other_bool.size} cells "
            f"({100*np.sum(other_bool)/other_bool.size:.1f}%)"
        )
        
        return other


class _RegionDustConfigurator:
    """Internal helper that provides region-aware dust configuration."""
    
    def __init__(self, global_dust, mask: Optional[Field], is_disk_region: bool):
        self._dust = global_dust
        self._mask = mask
        self._is_disk_region = is_disk_region
    
    def set_distribution(self, mode: str = "proportional", **kwargs):
        """Configure dust distribution for this region.
        
        Args:
            mode: 'proportional' or 'settling'
            **kwargs: Additional parameters (amin, amax, nbin, etc.)
        """
        if self._mask is None:
            raise ValueError(
                "Region has no mask defined; call set_mask_from_geometry first"
            )
        
        if mode == "settling" and not self._is_disk_region:
            raise ValueError(
                "Settling mode can only be used on regions defined as disk. "
                "Use is_a_disk=True when creating the mask."
            )
        
        self._dust.add_component_from_mask(
            mask=self._mask,
            mode=mode,
            **kwargs,
        )


class Disk(SubModel):
    """Disk component with disk-specific operations."""

    _required_parameters = [
        "aspectratio",
        "flaringindex",
        "r0"
    ]

    def __init__(self, parent: Model, parameters: Dict[str, Any]):
        super().__init__(parent)
        for key in self._required_parameters:
            if key not in parameters:
                raise ValueError(f"Missing required parameter in Disk class: {key}")
        self.parameters: Dict[str, Any] = parameters
    
    @property
    def gas(self) -> SubModel:
        """Access to gas fields (convenience accessor to parent.gas)."""
        return self.parent.gas
    
    @property
    def dust(self):
        """Access to dust with region-aware configuration."""
        return _RegionDustConfigurator(
            global_dust=self.parent.dust,
            mask=self.mask,
            is_disk_region=self.is_disk_region,
        )

    def _puff_density(self, nz: int = 64, zmax_scale: float = 5.0):
        """
        Puff 2D surface density to 3D volume density in cylindrical coordinates.
        
        Follows fargo2radmc3d's approach with proper normalization including
        the radial coordinate factor for cylindrical geometry and mass conservation.
        """
        if "surface_density" not in self.parent.gas:
            raise KeyError("Missing gas field 'surface_density'")

        Sigma = self.parent.gas["surface_density"].data  # Quantity (nrad, nsec)
        r     = self.parent.mesh.centers("r")            # Quantity (nrad,)

        h0 = self.parameters["aspectratio"]
        fl = self.parameters["flaringindex"]
        r0 = self.parameters["r0"]

        # Compute dimensionless aspect ratio h(r) = h₀ * (r/r₀)^β
        h = h0 * (r / r0) ** fl  # dimensionless
        
        # Build vertical grid based on maximum scale height
        H_max = np.max(h * r)  # maximum scale height with units
        zmed = _build_z_grid(nz, zmax_scale, H_max)  # (nz,) with units
        
        # Puff using fargo2radmc3d gas formula (all Quantities)
        rho3d = _puff_gaussian(Sigma, r, h, zmed)  # (nz, nrad, nsec) Quantity

        # 3D cylindrical mesh: reuse r/phi, add z centers
        new_mesh = self.parent.mesh.with_vertical(zmed)
        
        # Mass conservation renormalization
        rho3d = _renormalize_mass_conservation(rho3d, Sigma, new_mesh)

        # Return density (meshless Field) and new mesh
        dens_field = Field(
            data=rho3d,
            quantity="density",
            axis_order=("z", "r", "phi"),
        )
        return dens_field, new_mesh

    def _puff_velocity(self, nz: int = 64, zmed=None):
        vr2d   = self.parent.gas['vr'].data   # (nrad, nsec) Quantity
        vphi2d = self.parent.gas['vphi'].data # (nrad, nsec) Quantity

        if zmed is None:
            h0 = self.parameters["aspectratio"]
            fl = self.parameters["flaringindex"]
            r0 = self.parameters["r0"]
            H  = _scale_height(self.parent.mesh.centers("r"), h0, fl, r0)
            zmed = _build_z_grid(nz, 5.0, H)

        nrad = vr2d.shape[0]
        nsec = vr2d.shape[1]

        ones  = np.ones((len(zmed), 1, 1))
        vr3d   = vr2d.reshape(1, nrad, nsec)  * ones
        vphi3d = vphi2d.reshape(1, nrad, nsec) * ones

        return {"vr3d_cyl": vr3d, "vphi3d_cyl": vphi3d, "zmed": zmed}
    
    def puff_up_disk(
        self,
        n: int,
        coordinates: str = "spherical",
        zmax_over_H: float = 5.0,
    ) -> "Model":
        r  = self.parent.mesh.centers("r")
        h0 = self.parameters["aspectratio"]
        fl = self.parameters["flaringindex"]
        r0 = self.parameters["r0"]

        if coordinates == "spherical":
            # Puff in cylindrical coordinates first, then interpolate to spherical
            # This matches fargo2radmc3d's approach (gas_density.py lines 130-207)
            nz_cyl = max(3, int(0.5 * n))
            dens_field_cyl, cyl_mesh = self._puff_density(nz=nz_cyl, zmax_scale=zmax_over_H)
            vel = self._puff_velocity(nz=nz_cyl, zmed=cyl_mesh.centers("z"))
            zmed = cyl_mesh.centers("z")

            # Build spherical mesh with n polar cells
            sph_mesh = cyl_mesh.to_spherical_by_scale_height(
                ncol=n, aspect_ratio=h0, zmax_over_H=zmax_over_H
            )

            # Interpolate cyl -> sph on centers
            r_cyl = cyl_mesh.centers("r")
            r_sph = sph_mesh.centers("r")
            t_sph = sph_mesh.centers("theta")

            # Extract units before interpolation and reattach after
            rho_units = getattr(dens_field_cyl.data, "units", None)
            vr_units = getattr(vel["vr3d_cyl"], "units", None) if "vr" in self.parent.gas else None
            vphi_units = getattr(vel["vphi3d_cyl"], "units", None) if "vphi" in self.parent.gas else None

            rho_sph_raw = _interp_cyl_to_sph(dens_field_cyl.data, r_cyl, zmed, r_sph, t_sph)
            vr_sph_raw = _interp_cyl_to_sph(vel["vr3d_cyl"], r_cyl, zmed, r_sph, t_sph) if "vr" in self.parent.gas else None
            vphi_sph_raw = _interp_cyl_to_sph(vel["vphi3d_cyl"], r_cyl, zmed, r_sph, t_sph) if "vphi" in self.parent.gas else None

            # Reattach units
            rho_sph = rho_sph_raw * rho_units if rho_units else rho_sph_raw
            vr_sph = vr_sph_raw * vr_units if vr_units and vr_sph_raw is not None else None
            vphi_sph = vphi_sph_raw * vphi_units if vphi_units and vphi_sph_raw is not None else None

            
            # vtheta = 0 with proper unit
            vunit = (getattr(self.parent.gas["vr"].data, "units", None)
                    if "vr" in self.parent.gas else
                    getattr(self.parent.gas["vphi"].data, "units", None) if "vphi" in self.parent.gas else None)
            ntheta = int(sph_mesh.ncell("theta") or 0)
            nr = int(sph_mesh.ncell("r") or 0)
            nphi = int(sph_mesh.ncell("phi") or 0)
            vtheta = np.zeros((ntheta, nr, nphi))
            if vunit is not None:
                vtheta = vtheta * vunit

            # Commit spherical mesh + fields
            self.parent.mesh = sph_mesh
            self.mesh = sph_mesh
            if hasattr(self.parent, "gas") and self.parent.gas is not None:
                self.parent.gas.mesh = sph_mesh
            self.coord_system = "spherical"

            self.parent.gas.clear()
            self.parent.gas_register("density", Field(quantity="density", data=rho_sph, axis_order=("theta","r","phi")))
            if vr_sph is not None:
                self.parent.gas_register("vr", Field(quantity="vr", data=vr_sph, axis_order=("theta","r","phi")))
            if vphi_sph is not None:
                self.parent.gas_register("vphi", Field(quantity="vphi", data=vphi_sph, axis_order=("theta","r","phi")))
            self.parent.gas_register("vtheta", Field(quantity="vtheta", data=vtheta, axis_order=("theta","r","phi")))
            return self

        else:
            raise ValueError(f"coordinates must be 'spherical', got '{coordinates}'")


def _build_z_grid(nver: int, zmax_scale: float, H_max: Quantity) -> Quantity:
    """
    Build vertical grid for cylindrical coordinates.
    
    Parameters
    ----------
    nver : int
        Number of vertical cells
    zmax_scale : float
        Vertical extent in units of scale height
    H_max : Quantity
        Maximum scale height with units
        
    Returns
    -------
    z : Quantity array (nver,)
        Vertical grid centers with same units as H_max
    """
    zmax = float(zmax_scale) * H_max
    z_mag = np.linspace(-zmax.magnitude, zmax.magnitude, int(nver))
    return z_mag * H_max.units


def _puff_gaussian(Sigma: Quantity, r: Quantity, h: Quantity, zmed: Quantity) -> Quantity:
    """
    Gaussian vertical puffing matching fargo2radmc3d's gas formula.
    
    Formula: rho(z) = Sigma / (sqrt(2π) * r * h) * exp(-0.5*(z/(h*r))²)
    
    Parameters
    ----------
    Sigma : Quantity (nrad, nsec)
        Surface density
    r : Quantity (nrad,)
        Radial centers
    h : Quantity (nrad,)
        Dimensionless aspect ratio h(r)
    zmed : Quantity (nver,)
        Vertical grid centers
        
    Returns
    -------
    rho : Quantity (nver, nrad, nsec)
        3D volume density
    """
    nrad, nsec = Sigma.shape
    nver = len(zmed)
    
    # Work directly with Quantities - Pint will handle unit conversions
    # Broadcast arrays: z(nver,1,1), r(1,nrad,1), h(1,nrad,1), Sigma(1,nrad,nsec)
    Z = zmed.magnitude.reshape(nver, 1, 1)
    R = r.magnitude.reshape(1, nrad, 1)
    H = h.magnitude.reshape(1, nrad, 1)
    S = Sigma.magnitude.reshape(1, nrad, nsec)
    
    # Exponential factor: exp(-0.5 * (z/(h*r))²) - dimensionless
    expo = np.exp(-0.5 * (Z / (H * R)) ** 2)
    
    # Normalization: 1 / (sqrt(2π) * r * h) - units: 1/length
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * R * H)
    
    # Calculate density magnitude
    rho_mag = S * norm * expo
    
    # Reattach units: Sigma/r = surface_density/length = volume_density
    rho = rho_mag * (Sigma.units / r.units)
    
    return rho


def _renormalize_mass_conservation(rho3d: Quantity, Sigma: Quantity, mesh: Mesh) -> Quantity:
    """
    Renormalize 3D density to conserve mass, following fargo2radmc3d's approach.
    
    Parameters
    ----------
    rho3d : Quantity (nz, nrad, nsec)
        3D volume density
    Sigma : Quantity (nrad, nsec)
        2D surface density
    mesh : Mesh
        Cylindrical mesh with z, r, phi axes
        
    Returns
    -------
    rho3d_norm : Quantity (nz, nrad, nsec)
        Renormalized 3D volume density
    """
    # Get mesh edges  
    redge = mesh.edges('r')
    zedge = mesh.edges('z')
    phiedge = mesh.edges('phi')
    
    # Build meshgrid for cell edges (z, r, phi) - work with magnitudes
    Zedge, Redge, Phiedge = np.meshgrid(zedge.magnitude, redge.magnitude, phiedge.magnitude, indexing='ij')
    
    # Cell dimensions
    dz = Zedge[1:, :-1, :-1] - Zedge[:-1, :-1, :-1]
    dr = Redge[:-1, 1:, :-1] - Redge[:-1, :-1, :-1]
    dphi = Phiedge[:-1, :-1, 1:] - Phiedge[:-1, :-1, :-1]
    
    # Cell volumes in cylindrical coords: dV = R * dR * dphi * dz  
    R_centers = Redge[:-1, :-1, :-1] + 0.5 * dr
    cell_vol = R_centers * dr * dphi * dz  # dimensionless magnitude
    cell_vol_units = redge.units ** 2 * zedge.units  # r² * z
    
    # Total mass in 3D grid
    total_mass_3d = np.sum(rho3d.magnitude * cell_vol) * (rho3d.units * cell_vol_units)
    
    # Expected mass from 2D surface density (cell areas: dA = R * dR * dphi)
    Redge_2d, Phiedge_2d = np.meshgrid(redge.magnitude, phiedge.magnitude, indexing='ij')
    dr_2d = Redge_2d[1:, :-1] - Redge_2d[:-1, :-1]
    dphi_2d = Phiedge_2d[:-1, 1:] - Phiedge_2d[:-1, :-1]
    R_centers_2d = Redge_2d[:-1, :-1] + 0.5 * dr_2d
    cell_area = R_centers_2d * dr_2d * dphi_2d  # dimensionless magnitude  
    cell_area_units = redge.units ** 2
    
    expected_mass = np.sum(Sigma.magnitude * cell_area) * (Sigma.units * cell_area_units)
    
    # Normalization factor (dimensionless)
    norm_factor = (expected_mass / total_mass_3d).to_base_units().magnitude
    
    return rho3d * norm_factor




def puff_up_model(
    model: "Model",
    n: int,
    coordinates: str = "spherical",
    zmax_over_H: float = 5.0,
) -> "Model":

    new = Model()
    new.coord_system    = model.mesh.coord_system  # type: ignore[union-attr]
    new.variables       = dict(model.variables)
    new.compile_options = dict(model.compile_options)
    new.macros          = dict(model.macros)

    cs = model.mesh.coord_system  # type: ignore[union-attr]

    if cs == "cylindrical":
        # clone ONLY r/phi; z will be added by the puff step
        new.mesh = Mesh.cylindrical(
            r=Axis(edges=model.mesh.edges("r"),   centers=model.mesh.centers("r")),
            phi=Axis(edges=model.mesh.edges("phi"), centers=model.mesh.centers("phi")),
        )
    elif cs == "spherical":
        # clone r/(theta?)/phi exactly as present
        new.mesh = Mesh.spherical(
            r=Axis(edges=model.mesh.edges("r"), centers=model.mesh.centers("r")),
            theta=(
                Axis(edges=model.mesh.edges("theta"), centers=model.mesh.centers("theta"))
                if model.mesh.ncell("theta") else None
            ),
            phi=Axis(edges=model.mesh.edges("phi"), centers=model.mesh.centers("phi")),
        )
    else:
        raise ValueError("Cartesian meshes are not supported for puffing.")

    new.file_units = model.file_units
    new.directory  = model.directory
    new.n_file     = model.n_file
    new.filename   = model.filename

    # submodels
    new.gas  = SubModel(new)
    if getattr(model, "disk", None) is not None:
        new.disk = Disk(new, model.disk.parameters)

    # copy gas fields (mesh-less Fields; share data)
    for name, f in model.gas.items():
        new.gas_register(name, Field(data=f.data, quantity=f.quantity, axis_order=f.axis_order))

    # perform the puff on the clone
    new.disk.puff_up_disk(n, coordinates=coordinates, zmax_over_H=zmax_over_H)
    return new
