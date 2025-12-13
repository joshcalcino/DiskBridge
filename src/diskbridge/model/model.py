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
from .utils import validate_field_against_mesh


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
        # Time: T^2 ~ L^3/M -> time_factor = sqrt(length_factor^3 / mass_factor)
        time_factor = np.sqrt(length_factor**3 / mass_factor)
        
        # Velocity: V = L/T -> velocity_factor = length_factor / time_factor = sqrt(mass_factor / length_factor)
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
                        # Volume density: rho = M/L^3 -> scale by mass_factor/length_factor^3
                        # Surface density: Sigma = M/L^2 -> scale by mass_factor/length_factor^2
                        if 'surface' in name:
                            new_data = field.data * (mass_factor / length_factor**2)
                        else:
                            new_data = field.data * (mass_factor / length_factor**3)
                    
                    elif name in ['vr', 'vphi', 'vz', 'vtheta']:
                        # Velocity: V = L/T -> scale by sqrt(M/L)
                        new_data = field.data * velocity_factor
                    
                    elif name == 'temperature':
                        # Temperature: cutemp ~ M/L, so T -> T * (M/L)
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
        Rescaling follows Keplerian dynamics where T^2 ~ L^3/M:
        - Lengths scale by length_scale
        - Masses scale by mass_scale  
        - Times scale by sqrt(length_scale^3/mass_scale)
        - Velocities scale by sqrt(mass_scale/length_scale)
        - Densities scale by mass_scale/length_scale^3
        """
        if length_scale is None or mass_scale is None:
            try:
                from diskbridge import params as _global_params
            except Exception:
                _global_params = None
            if _global_params is not None:
                if length_scale is None and hasattr(_global_params, "length_scale"):
                    length_scale = float(_global_params.length_scale)
                if mass_scale is None and hasattr(_global_params, "mass_scale"):
                    mass_scale = float(_global_params.mass_scale)
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
    
    def set_mask_from_density(
        self,
        density_threshold: Quantity,
        r_min: Optional[Quantity] = None,
        r_max: Optional[Quantity] = None,
        is_a_disk: bool = False,
    ) -> SubModel:
        """Define a masked region based on density threshold.
        
        Args:
            density_threshold: Minimum density for inclusion
            r_min: Optional minimum radius
            r_max: Optional maximum radius
            is_a_disk: Mark region as disk (enables settling mode)
            
        Returns:
            SubModel (or Disk) with mask applied
        """
        mesh = self.mesh
        if mesh is None:
            raise ValueError("Model has no mesh")
        
        if mesh.coord_system != 'spherical':
            raise ValueError(
                f"set_mask_from_density only supports spherical coordinates, "
                f"got {mesh.coord_system}"
            )
        
        # Determine target region
        if is_a_disk and self.disk is not None:
            target_region = self.disk
        else:
            target_region = SubModel(self)
        
        # Get density from gas
        if 'density' not in self.gas:
            raise KeyError("No density field found in gas")
        
        density = self.gas['density'].data
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
        
        target_region.mask = mask_field
        
        # Mark as disk region if requested
        if is_a_disk:
            target_region.is_disk_region = True
        
        logger.info(
            f"{target_region.__class__.__name__} mask set (density): {np.sum(mask)} / {mask.size} cells "
            f"({100*np.sum(mask)/mask.size:.1f}%)"
        )
        
        return target_region
    
    def set_mask_from_array(
        self,
        mask_array: np.ndarray,
        is_a_disk: bool = False,
    ) -> SubModel:
        """Define a masked region from a custom boolean array.
        
        Args:
            mask_array: Boolean array matching grid shape
            is_a_disk: Mark region as disk (enables settling mode)
            
        Returns:
            SubModel (or Disk) with mask applied
        """
        mesh = self.mesh
        if mesh is None:
            raise ValueError("Model has no mesh")
        
        if mesh.coord_system != 'spherical':
            raise ValueError(
                f"set_mask_from_array only supports spherical coordinates, "
                f"got {mesh.coord_system}"
            )
        
        # Determine target region
        if is_a_disk and self.disk is not None:
            target_region = self.disk
        else:
            target_region = SubModel(self)
        
        # Verify shape
        expected_shape = (
            len(mesh.axes['r'].centers),
            len(mesh.axes['phi'].centers),
            len(mesh.axes['theta'].centers)
        )
        
        if mask_array.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask_array.shape} doesn't match grid shape {expected_shape}"
            )
        
        axis_order = ('r', 'phi', 'theta')
        
        mask_quantity = Quantity(mask_array.astype(bool), 'dimensionless')
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
            f"{target_region.__class__.__name__} mask set: {np.sum(mask_array)} / {mask_array.size} cells "
            f"({100*np.sum(mask_array)/mask_array.size:.1f}%)"
        )
        
        return target_region

    def puff_up_model(
        self, n: int, 
        zmax_over_H: float = 5.0
    ) -> "Model":
        """Puff 2D polar model to 3D spherical coordinates."""
        self.disk.puff_up_disk(n=n, zmax_over_H=zmax_over_H)
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
        
        # Use parent model's set_mask_from_array method
        other = self.parent.set_mask_from_array(other_bool, is_a_disk=False)
        
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

    def puff_up_disk(
        self,
        n: int,
        zmax_over_H: float = 5.0,
    ) -> "Model":
        """Puff 2D surface density directly to 3D spherical coordinates.
        
        Computes the Gaussian vertical profile directly in spherical coordinates
        without intermediate cylindrical conversion, ensuring mass conservation.
        """

        r = self.parent.mesh.centers("r")
        h0 = self.parameters["aspectratio"]
        fl = self.parameters["flaringindex"]
        r0 = self.parameters["r0"]
        
        # Get 2D surface density
        if "surface_density" not in self.parent.gas:
            raise KeyError("Missing gas field 'surface_density'")
        Sigma = self.parent.gas["surface_density"].data  # (nrad, nsec)
        
        # Build spherical mesh
        sph_mesh = self.parent.mesh.to_spherical_by_scale_height(
            ncol=n, aspect_ratio=h0, zmax_over_H=zmax_over_H
        )
        
        r_sph = sph_mesh.centers("r")
        theta_sph = sph_mesh.centers("theta")
        phi_sph = sph_mesh.centers("phi")
        
        # Create 3D grids
        r_mag = r_sph.magnitude
        theta_mag = theta_sph.magnitude
        phi_mag = phi_sph.magnitude
        r_units = r_sph.units
        
        # Shape: (n_r, n_phi, n_theta) matching typical axis order
        r_grid, phi_grid, theta_grid = np.meshgrid(r_mag, phi_mag, theta_mag, indexing='ij')
        r_grid = r_grid * r_units
        
        # Compute z = r * cos(theta) for each spherical cell
        z_cyl = r_grid * np.cos(theta_grid)
        
        # Scale height H(r) = h(r) * r where h(r) = h0 * (r/r0)^fl
        h_r = h0 * (r_sph / r0) ** fl  # dimensionless
        H = h_r * r_sph  # scale height with units
        H_3d = H[:, None, None] * np.ones_like(theta_grid)  # broadcast to 3D
        
        # Expand Sigma to 3D: (nrad, nsec) -> (nrad, nsec, ntheta)
        # Note: Sigma is (r, phi), we need (r, phi, theta)
        Sigma_3d = Sigma[:, :, None] * np.ones(len(theta_sph))
        
        # Compute Gaussian density: rho = Sigma / (sqrt(2*pi) * H) * exp(-z^2 / (2*H^2))
        rho_3d = Sigma_3d / (np.sqrt(2 * np.pi) * H_3d) * np.exp(-z_cyl**2 / (2 * H_3d**2))
        
        # Handle velocities - expand 2D to 3D (constant in theta)
        ntheta = len(theta_sph)
        nr = len(r_sph)
        nphi = len(phi_sph)
        
        vr_sph = None
        vphi_sph = None
        if "vr" in self.parent.gas:
            vr_2d = self.parent.gas["vr"].data
            vr_sph = vr_2d[:, :, None] * np.ones(ntheta)
        if "vphi" in self.parent.gas:
            vphi_2d = self.parent.gas["vphi"].data
            vphi_sph = vphi_2d[:, :, None] * np.ones(ntheta)
        
        # vtheta = 0
        vunit = None
        if "vr" in self.parent.gas:
            vunit = getattr(self.parent.gas["vr"].data, "units", None)
        elif "vphi" in self.parent.gas:
            vunit = getattr(self.parent.gas["vphi"].data, "units", None)
        vtheta = np.zeros((nr, nphi, ntheta))
        if vunit is not None:
            vtheta = vtheta * vunit
        
        # Commit spherical mesh + fields
        self.parent.mesh = sph_mesh
        self.mesh = sph_mesh
        if hasattr(self.parent, "gas") and self.parent.gas is not None:
            self.parent.gas.mesh = sph_mesh
        self.coord_system = "spherical"
        
        self.parent.gas.clear()
        self.parent.gas_register("density", Field(quantity="density", data=rho_3d, axis_order=("r", "phi", "theta")))
        if vr_sph is not None:
            self.parent.gas_register("vr", Field(quantity="vr", data=vr_sph, axis_order=("r", "phi", "theta")))
        if vphi_sph is not None:
            self.parent.gas_register("vphi", Field(quantity="vphi", data=vphi_sph, axis_order=("r", "phi", "theta")))
        self.parent.gas_register("vtheta", Field(quantity="vtheta", data=vtheta, axis_order=("r", "phi", "theta")))
        return self


def puff_up_model(
    model: "Model",
    n: int,
    zmax_over_H: float = 5.0,
) -> "Model":
    """Puff a 2D polar model to 3D spherical coordinates."""
    new = Model()
    new.coord_system = model.mesh.coord_system
    new.variables = dict(model.variables)
    new.compile_options = dict(model.compile_options)
    new.macros = dict(model.macros)

    cs = model.mesh.coord_system

    if cs == "polar":
        new.mesh = Mesh.polar(
            r=Axis(edges=model.mesh.edges("r"), centers=model.mesh.centers("r")),
            phi=Axis(edges=model.mesh.edges("phi"), centers=model.mesh.centers("phi")),
        )
    elif cs == "spherical":
        new.mesh = Mesh.spherical(
            r=Axis(edges=model.mesh.edges("r"), centers=model.mesh.centers("r")),
            theta=(
                Axis(edges=model.mesh.edges("theta"), centers=model.mesh.centers("theta"))
                if model.mesh.ncell("theta") else None
            ),
            phi=Axis(edges=model.mesh.edges("phi"), centers=model.mesh.centers("phi")),
        )
    else:
        raise ValueError(f"Unsupported coordinate system for puffing: {cs}")

    new.file_units = model.file_units
    new.directory = model.directory
    new.n_file = model.n_file
    new.filename = model.filename

    new.gas = SubModel(new)
    if getattr(model, "disk", None) is not None:
        new.disk = Disk(new, model.disk.parameters)

    for name, f in model.gas.items():
        new.gas_register(name, Field(data=f.data, quantity=f.quantity, axis_order=f.axis_order))

    new.disk.puff_up_disk(n, zmax_over_H=zmax_over_H)
    return new
