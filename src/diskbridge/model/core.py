from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union, TYPE_CHECKING
from pathlib import Path
import numpy as np

from .mesh import Mesh, Axis
from .field import Field
from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from .utils import validate_field_against_mesh
from .clipping import compute_clip_indexer

if TYPE_CHECKING:
    from .disk_component import Disk

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
        - Densities scale by mass_scale/length_factor^3
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
            from .disk_component import Disk
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
    
    def set_mask_from_joos_disk(
        self,
        rho_disk_min: Quantity,
        *,
        fthres: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 2.0,
        fthres_vr: Optional[Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]] = None,
        rho_core_min: Optional[Quantity] = None,
        r_max_for_axis: Optional[Quantity] = None,
        n_r_bins: Optional[int] = None,
        n_theta_bins: Optional[int] = None,
        r_max: Optional[Quantity] = None,
    ) -> SubModel:
        from .masking import set_mask_from_joos_disk as _set_mask_from_joos_disk

        return _set_mask_from_joos_disk(
            self,
            rho_disk_min,
            fthres=fthres,
            fthres_vr=fthres_vr,
            rho_core_min=rho_core_min,
            r_max_for_axis=r_max_for_axis,
            n_r_bins=n_r_bins,
            n_theta_bins=n_theta_bins,
            r_max=r_max,
        )

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

    def extend_spherical_grid_inwards(
        self,
        r_min: Quantity,
        *,
        spacing: Optional[str] = None,
        density_match: Optional[str] = None,
    ) -> "Model":
        from .mesh_extend import extend_spherical_grid_inwards as _extend_spherical_grid_inwards

        return _extend_spherical_grid_inwards(
            self,
            r_min,
            spacing=spacing,
            density_match=density_match,
        )

    def clip_mesh(
        self,
        *,
        r_min: Optional[Quantity] = None,
        r_max: Optional[Quantity] = None,
        theta_min: Optional[Quantity] = None,
        theta_max: Optional[Quantity] = None,
        phi_min: Optional[Quantity] = None,
        phi_max: Optional[Quantity] = None,
        x_min: Optional[Quantity] = None,
        x_max: Optional[Quantity] = None,
        y_min: Optional[Quantity] = None,
        y_max: Optional[Quantity] = None,
        z_min: Optional[Quantity] = None,
        z_max: Optional[Quantity] = None,
    ) -> "Model":
        mesh0 = self.mesh
        if mesh0 is None:
            raise ValueError("Model has no mesh")

        bounds: Dict[str, tuple[Optional[Quantity], Optional[Quantity]]] = {
            "r": (r_min, r_max),
            "theta": (theta_min, theta_max),
            "phi": (phi_min, phi_max),
            "x": (x_min, x_max),
            "y": (y_min, y_max),
            "z": (z_min, z_max),
        }
        bounds = {
            k: (vmin, vmax)
            for k, (vmin, vmax) in bounds.items()
            if vmin is not None or vmax is not None
        }

        for axis_name in bounds:
            if axis_name not in mesh0.axes:
                raise ValueError(
                    f"Cannot clip axis '{axis_name}' for coord_system '{mesh0.coord_system}'"
                )

        indexer, new_axes = compute_clip_indexer(mesh0, bounds)
        axis_slices = indexer.axis_slices

        mesh1 = Mesh(mesh0.coord_system, new_axes)

        def _slice_field(field0: Field) -> Field:
            slicer = []
            for ax in field0.axis_order:
                slicer.append(axis_slices.get(ax, slice(None)))
            mag0 = np.asarray(field0.data.magnitude)
            mag1 = mag0[tuple(slicer)]
            data1 = Quantity(mag1, field0.data.units)
            return Field(
                data=data1,
                quantity=field0.quantity,
                axis_order=field0.axis_order,
                attrs=field0.attrs,
            )

        new = Model()
        new.coord_system = mesh1.coord_system
        new.variables = dict(self.variables)
        new.compile_options = dict(self.compile_options)
        new.macros = dict(self.macros)
        new.mesh = mesh1
        new.file_units = self.file_units
        new.directory = self.directory
        new.n_file = self.n_file
        new.filename = self.filename

        for attr in ("length_scale", "mass_scale", "time_scale", "velocity_scale"):
            if hasattr(self, attr):
                setattr(new, attr, getattr(self, attr))

        new.gas = SubModel(new)
        if getattr(self.gas, "_lazy", None):
            for name in list(self.gas._lazy.keys()):
                _ = self.gas[name]
        for name, field0 in list(self.gas.items()):
            new.gas_register(name, _slice_field(field0))

        if getattr(self, "disk", None) is not None:
            from .disk_component import Disk

            new.disk = Disk(new, dict(self.disk.parameters))
            new.disk.is_disk_region = bool(getattr(self.disk, "is_disk_region", False))
            if self.disk.mask is not None:
                new.disk.mask = _slice_field(self.disk.mask)

        if getattr(self, "dust", None) is not None:
            from .dust import Dust, DustComponent, DustBin

            old_dust = self.dust
            new.dust = Dust(new)
            new_dust = new.dust

            new_dust.mask = _slice_field(old_dust.mask) if old_dust.mask is not None else None
            new_dust.is_disk_region = bool(getattr(old_dust, "is_disk_region", False))

            new_dust.distribution = old_dust.distribution
            new_dust.dust_to_gas_ratio = old_dust.dust_to_gas_ratio
            new_dust.mode = old_dust.mode
            new_dust.alpha = old_dust.alpha
            new_dust.delta = old_dust.delta
            new_dust.mean_molecular_weight = old_dust.mean_molecular_weight

            new_dust._has_region_components = bool(getattr(old_dust, "_has_region_components", False))
            new_dust._components = []
            for comp0 in getattr(old_dust, "_components", []):
                comp_mask = _slice_field(comp0.mask) if comp0.mask is not None else None
                new_dust._components.append(
                    DustComponent(
                        distribution=comp0.distribution,
                        dust_to_gas_ratio=comp0.dust_to_gas_ratio,
                        mode=comp0.mode,
                        mask=comp_mask,
                        alpha=comp0.alpha,
                        delta=comp0.delta,
                        mean_molecular_weight=comp0.mean_molecular_weight,
                        species_base=comp0.species_base,
                        component_index=comp0.component_index,
                    )
                )

            new_dust._global_bins = {}
            new_dust._bins = {}
            for comp_idx, comp in enumerate(new_dust._components):
                nbin = int(comp.distribution.nbin)
                for local_idx in range(nbin):
                    global_bin_idx = len(new_dust._global_bins)
                    bin_name = f"bin_{global_bin_idx}"
                    new_dust._global_bins[bin_name] = (comp_idx, local_idx)
                    new_dust._bins[bin_name] = DustBin(
                        parent_dust=new_dust,
                        bin_index=global_bin_idx,
                        size=comp.distribution.bin_centers[local_idx],
                        size_min=comp.distribution.bin_edges[local_idx],
                        size_max=comp.distribution.bin_edges[local_idx + 1],
                        mass_fraction=comp.distribution.mass_fractions[local_idx],
                        density_material=comp.distribution.grain_density,
                    )

            new_dust._dust_fields = {}
            for name, field0 in getattr(old_dust, "_dust_fields", {}).items():
                new_dust._dust_fields[name] = _slice_field(field0)

        return new
        

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
        from .disk_component import _RegionDustConfigurator

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
        from .disk_component import Disk

        new.disk = Disk(new, model.disk.parameters)

    for name, f in model.gas.items():
        new.gas_register(name, Field(data=f.data, quantity=f.quantity, axis_order=f.axis_order))

    new.disk.puff_up_disk(n, zmax_over_H=zmax_over_H)
    return new


def extend_disk_inwards(
    model: "Model",
    *,
    r_min: Optional[Quantity] = None,
    r_min_factor: Optional[float] = None,
    spacing: Optional[str] = None,
    density_match: Optional[str] = None,
) -> int:
    if (r_min is None) == (r_min_factor is None):
        raise ValueError("Provide exactly one of r_min or r_min_factor")

    mesh0 = model.mesh
    if mesh0 is None:
        raise ValueError("Model mesh is not set")
    r_edges0 = mesh0.edges("r")
    if r_edges0 is None:
        raise ValueError("Model radial edges are not set")

    if r_min is None:
        r_min = float(r_min_factor) * r_edges0[0]

    model.extend_spherical_grid_inwards(r_min, spacing=spacing, density_match=density_match)

    mesh1 = model.mesh
    if mesh1 is None:
        raise ValueError("Model mesh is not set after extension")
    r_edges1 = mesh1.edges("r")
    if r_edges1 is None:
        raise ValueError("Model radial edges are not set after extension")

    return int(r_edges1.size - r_edges0.size)
