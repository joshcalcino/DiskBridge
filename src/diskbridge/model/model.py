from __future__ import annotations

from dataclasses import dataclass, field as dcfield
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import numpy as np

from .mesh import Mesh, Axis
from .field import Field
from diskbridge._logging import logger


def _get_param(model: "Model", key: str, fallback=None, default=None):
    v = None
    if isinstance(getattr(model, "variables", None), dict):
        v = model.variables.get(key)
    if v is None:
        v = fallback
    if v is None:
        v = default
    return v


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
        
        # Initialize properties of the simulation
        self.gas: SubModel = None
        self.disk: Disk = None
        
    def get_variables(self) -> Dict[str, Any]:
        return dict(self.variables)

    def gas_register(self, name: str, field: Field) -> None:
        self.gas.register(name, field)

    def gas_register_lazy(self, name: str, builder: Callable[[], Field]) -> None:
        self.gas.register_lazy(name, builder)

    def rescale_length_units(self, factor_or_unit) -> None:
        # Placeholder: record intent only
        # Future: apply scaling to mesh edges/centers and to velocity fields
        setattr(self, "_length_scale", factor_or_unit)

    def rescale_mass_units(self, factor_or_unit) -> None:
        # Placeholder: record intent only
        setattr(self, "_mass_scale", factor_or_unit)

    def load_model(
        self,
        path: Union[str, Path],
        reader: str = "fargo",
        file_n: int = 0,
        file_units: str = "code",
    ) -> "Model":
        """
        Load a hydro snapshot and populate this Model, then return self.
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
        self.disk = Disk(self)

        # Initialize Disk parameters if present
        if "disk_parameters" in snap:
            self.disk.parameters.clear()
            self.disk.parameters.update(snap["disk_parameters"])

        # Register gas fields
        for name, field in snap["gas_fields"].items():
            self.gas_register(name, field)

        return self

    def puff_up_model(
        self,
        n: int,
        coordinates: str = "cylindrical",
        zmax_over_H: float = 5.0,
        full_disk: bool = True,
    ) -> "Model":
        # Build 3D mesh and fields and mutate this model

        Sigma = self.gas["surface_density"].data
        r = self.mesh.rmed

        h0 = _get_param(self, "ASPECTRATIO", fallback=getattr(getattr(self, "disk", None), "parameters", {}).get("honr"), default=0.05)
        fl = _get_param(self, "FLARINGINDEX", fallback=getattr(getattr(self, "disk", None), "parameters", {}).get("flaringindex"), default=0.0)
        r0 = _get_param(self, "R0", fallback=getattr(getattr(self, "disk", None), "parameters", {}).get("r0"), default=1.0)

        H = _scale_height(r, h0, fl, r0)

        if coordinates == "cylindrical":
            zmed = _build_z_grid(n, 5.0, H)
            rho3d = _puff_gaussian(Sigma, H, zmed)
            new_mesh = self.mesh.with_vertical(zmed)

            # velocities
            vr3d = None
            vphi3d = None
            vz3d = None
            if "vr" in self.gas:
                vr2d = self.gas["vr"].data
                vr3d = vr2d.reshape(1, vr2d.shape[0], vr2d.shape[1]) * np.ones((len(zmed), 1, 1))
            if "vphi" in self.gas:
                vphi2d = self.gas["vphi"].data
                vphi3d = vphi2d.reshape(1, vphi2d.shape[0], vphi2d.shape[1]) * np.ones((len(zmed), 1, 1))
            # vertical velocity default 0
            try:
                vunit = getattr(self.gas["vr"].data, "units", None) or getattr(self.gas["vphi"].data, "units", None)
            except Exception:
                vunit = None
            zeros = np.zeros((len(zmed), new_mesh.nrad, new_mesh.nsec))
            vz3d = zeros if vunit is None else zeros * vunit

            # mutate model
            self.mesh = new_mesh
            self.coord_system = "cylindrical"
            new_fields: Dict[str, Field] = {}
            new_fields["density"] = Field(data=rho3d, mesh=new_mesh, quantity="density", axis_order=("z", "r", "phi"))
            if vr3d is not None:
                new_fields["vr"] = Field(data=vr3d, mesh=new_mesh, quantity="vr", axis_order=("z", "r", "phi"))
            if vphi3d is not None:
                new_fields["vphi"] = Field(data=vphi3d, mesh=new_mesh, quantity="vphi", axis_order=("z", "r", "phi"))
            new_fields["vz"] = Field(data=vz3d, mesh=new_mesh, quantity="vz", axis_order=("z", "r", "phi"))

            # replace gas registry
            self.gas.clear()
            for k, f in new_fields.items():
                self.gas_register(k, f)
            return self

        elif coordinates == "spherical":
            zmed = _build_z_grid(max(3, int(0.5*n)), 5.0, H)
            rho3d_cyl = _puff_gaussian(Sigma, H, zmed)
            # velocities to cylindrical first
            vr3d_cyl = None
            vphi3d_cyl = None
            if "vr" in self.gas:
                vr2d = self.gas["vr"].data
                vr3d_cyl = vr2d.reshape(1, vr2d.shape[0], vr2d.shape[1]) * np.ones((len(zmed), 1, 1))
            if "vphi" in self.gas:
                vphi2d = self.gas["vphi"].data
                vphi3d_cyl = vphi2d.reshape(1, vphi2d.shape[0], vphi2d.shape[1]) * np.ones((len(zmed), 1, 1))

            new_mesh = self.mesh.to_spherical_by_scale_height(n, aspect_ratio=h0, zmax_over_H=zmax_over_H, full_disk=full_disk)

            # Interpolate cylindrical -> spherical on centers
            rho3d_sph = _interp_cyl_to_sph(rho3d_cyl, self.mesh.rmed, zmed, new_mesh.rmed, new_mesh.tmed)
            vr3d_sph = _interp_cyl_to_sph(vr3d_cyl, self.mesh.rmed, zmed, new_mesh.rmed, new_mesh.tmed) if vr3d_cyl is not None else None
            vphi3d_sph = _interp_cyl_to_sph(vphi3d_cyl, self.mesh.rmed, zmed, new_mesh.rmed, new_mesh.tmed) if vphi3d_cyl is not None else None
            # vtheta zeros
            try:
                vunit = getattr(self.gas["vr"].data, "units", None) or getattr(self.gas["vphi"].data, "units", None)
            except Exception:
                vunit = None
            zeros = np.zeros((new_mesh.ncol, new_mesh.nrad, new_mesh.nsec))
            vtheta3d = zeros if vunit is None else zeros * vunit

            self.mesh = new_mesh
            self.coord_system = "spherical"
            self.gas.clear()
            self.gas_register("density", Field(data=rho3d_sph, mesh=new_mesh, quantity="density", axis_order=("theta", "r", "phi")))
            if vr3d_sph is not None:
                self.gas_register("vr", Field(data=vr3d_sph, mesh=new_mesh, quantity="vr", axis_order=("theta", "r", "phi")))
            if vphi3d_sph is not None:
                self.gas_register("vphi", Field(data=vphi3d_sph, mesh=new_mesh, quantity="vphi", axis_order=("theta", "r", "phi")))
            self.gas_register("vtheta", Field(data=vtheta3d, mesh=new_mesh, quantity="vtheta", axis_order=("theta", "r", "phi")))
            return self

        else:
            raise ValueError("coordinates must be 'cylindrical' or 'spherical'")


class SubModel(Model):

    def __init__(self, base: Model):
        self.base = base
        self.mesh = base.mesh
        self.disk = base.disk
        self.coord_system = base.coord_system
        self.variables = {}
        self.compile_options = {}
        self.macros = {}
        self.file_units = base.file_units 
        self.directory = None
        self.n_file = None
        self.filename = None

        # Internal field storage and lazy builders
        self._fields: Dict[str, Field] = {}
        self._lazy: Dict[str, Callable[[], Field]] = {}

        # SubModel specific properties
        self.mask: Optional[Field] = None

    # Registry API
    def register(self, name: str, field: Field) -> None:
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

    def _get_mask(self):
        pass 

    def set_mask(self):
        pass



class Disk(SubModel):

    def __init__(self, base: Model):
        super().__init__(base)
        self.parameters: Dict[str, Any] = {}

    def puff_density(
        self,
        nz: int = 64,
        zmax_scale: float = 5.0,
        surface_density_key: str = "surface_density",
    ):
        # Validate mesh and required 2D field
        if self.base.mesh is None:
            raise ValueError("mesh is not loaded")
        if surface_density_key not in self.base.gas:
            raise KeyError(f"Missing gas field '{surface_density_key}'")

        # Input data with units (Quantities)
        Sigma = self.base.gas[surface_density_key].data  # (nrad, nsec)
        r = self.base.mesh.rmed  # (nrad,)

        # Disk geometry parameters (dimensionless except r0 has length)
        h0 = _get_param(self, "ASPECTRATIO", fallback=self.parameters.get("honr"), default=0.05)
        fl = _get_param(self, "FLARINGINDEX", fallback=self.parameters.get("flaringindex"), default=0.0)
        r0 = _get_param(self, "R0", fallback=self.parameters.get("r0"), default=1.0)

        # Build vertical grid and puff Gaussian with unit-aware math
        H = _scale_height(r, h0, fl, r0)  # (nrad,)
        zmed = _build_z_grid(nz, zmax_scale, H)  # (nz,)
        rho3d = _puff_gaussian(Sigma, H, zmed)   # (nz, nrad, nsec)

        # Promote model to 3D cylindrical: new Mesh shares r,phi; adds z and ndims=3
        new_mesh = Mesh(
            coord_system="cylindrical",
            redge=self.base.mesh.redge,
            pedge=self.base.mesh.pedge,
            rmed=self.base.mesh.rmed,
            pmed=self.base.mesh.pmed,
            zmed=zmed,
            ndims=3,
        )

        # Update base model and all submodels to share the same Mesh
        self.base.mesh = new_mesh
        self.mesh = new_mesh
        if hasattr(self.base, "gas") and self.base.gas is not None:
            self.base.gas.mesh = new_mesh
            # Ensure all existing fields reference the new mesh for consistency
            for _, f in list(self.base.gas.items()):
                f.mesh = new_mesh

        # Register 3D density field with correct quantity and axis order
        dens_field = Field(
            data=rho3d,
            mesh=new_mesh,
            quantity="density",
            axis_order=("z", "r", "phi"),
        )
        self.base.gas.register("density", dens_field)

        # Return for convenience
        return dens_field

    def puff_velocity(
        self,
        nz: int = 64,
        zmed=None,
        vr_key: str = "vr",
        vphi_key: str = "vphi",
    ):
        if self.mesh is None:
            raise ValueError("mesh is not loaded")
        if vr_key not in self.base.gas or vphi_key not in self.base.gas:
            missing = [k for k in (vr_key, vphi_key) if k not in self.base.gas]
            raise KeyError(f"Missing gas field(s): {missing}")

        vr2d = self.base.gas[vr_key].data  # (nrad, nsec)
        vphi2d = self.base.gas[vphi_key].data  # (nrad, nsec)

        if zmed is None:
            # Build a symmetric z grid using a default scale height estimate
            h0 = _get_param(self, "ASPECTRATIO", fallback=self.parameters.get("honr"), default=0.05)
            fl = _get_param(self, "FLARINGINDEX", fallback=self.parameters.get("flaringindex"), default=0.0)
            r0 = _get_param(self, "R0", fallback=self.parameters.get("r0"), default=1.0)
            H = _scale_height(self.base.mesh.rmed, h0, fl, r0)
            zmed = _build_z_grid(nz, 5.0, H)

        nrad = vr2d.shape[0]
        nsec = vr2d.shape[1]

        vr3d = vr2d.reshape(1, nrad, nsec) * np.ones((len(zmed), 1, 1))
        vphi3d = vphi2d.reshape(1, nrad, nsec) * np.ones((len(zmed), 1, 1))

        return {"vr3d_cyl": vr3d, "vphi3d_cyl": vphi3d, "zmed": zmed}


def _scale_height(r: np.ndarray, h0: float, flaringindex: float, r0: float) -> np.ndarray:
    return (h0 * (r / r0) ** flaringindex) * r


def _build_z_grid(nver: int, zmax_scale: float, H: np.ndarray) -> np.ndarray:
    # Keep units if H is a Quantity by working on magnitude and reattaching units
    try:
        zmax_mag = float(zmax_scale) * np.max(getattr(H, "magnitude", H))
        units = getattr(H, "units", None)
        z = np.linspace(-zmax_mag, zmax_mag, int(nver))
        if units is not None:
            return z * units
        return z
    except Exception:
        zmax = float(zmax_scale) * float(np.max(H))
        return np.linspace(-zmax, zmax, int(nver))


def _puff_gaussian(Sigma: np.ndarray, H: np.ndarray, zmed: np.ndarray) -> np.ndarray:
    # Unit-aware Gaussian that integrates to Sigma over z
    nrad, nsec = Sigma.shape
    nver = zmed.size
    H2 = H.reshape(1, nrad, 1)
    Z = zmed.reshape(nver, 1, 1)
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * H2)
    expo = np.exp(-0.5 * (Z / H2) ** 2)
    rho = Sigma.reshape(1, nrad, nsec) * norm * expo
    return rho


def _interp_cyl_to_sph(cyl: Optional[np.ndarray], r_cyl: np.ndarray, zmed: np.ndarray, r_sph: np.ndarray, tmed_sph: np.ndarray) -> Optional[np.ndarray]:
    if cyl is None:
        return None
    nz, nrad, nsec = cyl.shape
    nt = len(tmed_sph)
    out = []
    for j in range(nt):
        theta = tmed_sph[j]
        R = r_sph * np.sin(theta)
        Z = r_sph * np.cos(theta)
        slice_j = np.zeros((nrad, nsec))
        for i in range(nrad):
            Ri = R[i]
            Zi = Z[i]
            ir = int(np.clip(np.searchsorted(r_cyl, Ri) - 1, 0, nrad - 2))
            iz = int(np.clip(np.searchsorted(zmed, Zi) - 1, 0, nz - 2))
            r0 = r_cyl[ir]
            r1 = r_cyl[ir + 1]
            z0 = zmed[iz]
            z1 = zmed[iz + 1]
            dr = (Ri - r0) / (r1 - r0) if r1 != r0 else 0.0
            dz = (Zi - z0) / (z1 - z0) if z1 != z0 else 0.0
            c00 = cyl[iz, ir, :]
            c01 = cyl[iz, ir + 1, :]
            c10 = cyl[iz + 1, ir, :]
            c11 = cyl[iz + 1, ir + 1, :]
            c0 = c00 * (1 - dr) + c01 * dr
            c1 = c10 * (1 - dr) + c11 * dr
            slice_j[i, :] = c0 * (1 - dz) + c1 * dz
        out.append(slice_j)
    return np.stack(out, axis=0)


def puff_up_model(
    model: "Model",
    n: int,
    coordinates: str = "cylindrical",
    zmax_over_H: float = 5.0,
    full_disk: bool = True,
) -> "Model":

    new = Model()
    new.coord_system = model.mesh.coord_system
    new.variables = dict(model.variables)
    new.compile_options = dict(model.compile_options)
    new.macros = dict(model.macros)
    new.mesh = Mesh(coord_system=model.mesh.coord_system, 
                    axes={"r": Axis(edges=model.mesh.redge), 
                          "phi": Axis(edges=model.mesh.pedge), 
                          "z": Axis(edges=model.mesh.zmed)})
    
    new.file_units = model.file_units
    new.directory = model.directory
    new.n_file = model.n_file
    new.filename = model.filename
    new.gas = SubModel(new)
    # copy gas fields to the new model (share data, update mesh reference)
    try:
        for name, f in model.gas.items():
            new_field = Field(data=f.data, mesh=new.mesh, quantity=f.quantity, axis_order=f.axis_order)
            new.gas_register(name, new_field)
    except Exception:
        pass
    try:
        if hasattr(model, "disk") and model.disk is not None and hasattr(model.disk, "parameters"):
            new.disk = Disk(new)
            new.disk.parameters = dict(model.disk.parameters)
    except Exception:
        pass
    # perform in-place puff on the new model
    return new.puff_up_model(
        n,
        coordinates=coordinates,
        zmax_over_H=zmax_over_H,
        full_disk=full_disk,
    )

