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
    from .disk import Disk

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
        self.gas: Optional['SubModel'] = None
        self.disk: Optional['Disk'] = None
        self.dust: Optional['Dust'] = None
        
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

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        reader: str = "fargo",
        file_n: int = 0,
        file_units: str = "code",
        length_scale: Optional[float] = None,
        mass_scale: Optional[float] = None,
    ) -> "Model":
        """
        Load a hydro snapshot or saved model.
        
        Parameters
        ----------
        path : str or Path
            Path to simulation data directory or saved model file
        reader : str, optional
            Reader type (default: "fargo"). Ignored for saved models.
        file_n : int, optional
            File number to load (default: 0). Ignored for saved models.
        file_units : str, optional
            Unit system: 'code', 'cgs', or 'kms' (default: 'code'). Ignored for saved models.
        length_scale : float, optional
            Length units multiplier (e.g., 10 means 1 code_length = 10 au). Ignored for saved models.
        mass_scale : float, optional
            Mass units multiplier (e.g., 2 means 1 code_mass = 2 M_sun). Ignored for saved models.
        
        Returns
        -------
        Model
            The loaded model instance
        """
        import pickle
        
        p = Path(path)
        
        if p.is_file():
            with open(p, 'rb') as f:
                return pickle.load(f)
        
        if length_scale is None or mass_scale is None:
            try:
                from diskbridge import params as _global_params
            except (ImportError, AttributeError):
                _global_params = None
            if _global_params is not None:
                if length_scale is None and hasattr(_global_params, "length_scale"):
                    length_scale = float(_global_params.length_scale)
                if mass_scale is None and hasattr(_global_params, "mass_scale"):
                    mass_scale = float(_global_params.mass_scale)
        
        if reader.lower() == "fargo":
            from .readers.fargo import read_fargo_snapshot
            snap = read_fargo_snapshot(p, file_n=file_n, file_units=file_units)
        else:
            raise ValueError(f"Unsupported reader: {reader}")

        model = cls()
        model.coord_system = snap["coord_system"]
        model.variables = snap["variables"]
        model.compile_options = snap.get("compile_options", {})
        model.macros = snap.get("macros", {})
        model.mesh = snap["mesh"]
        model.file_units = file_units 
        model.directory = str(p)
        model.n_file = file_n
        model.filename = None
        
        model.gas = SubModel(model)
        
        from .dust import Dust
        model.dust = Dust(model)

        if "disk_parameters" in snap:
            from .disk import Disk
            model.disk = Disk(model, snap["disk_parameters"])

        for name, field in snap["gas_fields"].items():
            model.gas_register(name, field)
        
        if length_scale is not None or mass_scale is not None:
            model._apply_rescaling(length_scale=length_scale, mass_scale=mass_scale)
            if model.gas is not None:
                model.gas.mesh = model.mesh
            if model.dust is not None:
                model.dust.mesh = model.mesh
            if hasattr(model, 'disk') and model.disk is not None:
                model.disk.mesh = model.mesh

        return model
    
    def set_mask_from_geometry(
        self,
        r_min: Optional[Quantity] = None,
        r_max: Optional[Quantity] = None,
        theta_min: Optional[Quantity] = None,
        theta_max: Optional[Quantity] = None,
        honrmax: Optional[float] = None,
        is_a_disk: bool = False,
    ) -> SubModel:
        from .masking import set_mask_from_geometry as _set_mask_from_geometry
        
        return _set_mask_from_geometry(
            self,
            r_min=r_min,
            r_max=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            honrmax=honrmax,
            is_a_disk=is_a_disk,
        )
    
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
        from .clipping import clip_model
        
        return clip_model(
            self,
            r_min=r_min,
            r_max=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        

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
            except AttributeError:
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
        from .disk import Disk
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
