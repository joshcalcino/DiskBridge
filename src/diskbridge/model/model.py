from __future__ import annotations

from dataclasses import dataclass, field as dcfield
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import numpy as np

from .mesh import Mesh, Axis
from .field import Field
from diskbridge._logging import logger
from diskbridge._units import Quantity
from .utils import validate_field_against_mesh  # <— add


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
        # validate against the model's single mesh
        validate_field_against_mesh(field, self.mesh)  # type: ignore[arg-type]
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

        # Initialize Disk parameters if present
        if "disk_parameters" in snap:
            self.disk = Disk(self, snap["disk_parameters"])

        # Register gas fields
        for name, field in snap["gas_fields"].items():
            self.gas_register(name, field)

        return self

    def puff_up_model(
        self, n: int, 
        coordinates: str = "cylindrical", 
        zmax_over_H: float = 5.0
    ) -> "Model":
        self.disk.puff_up_disk(n=n, coordinates=coordinates, zmax_over_H=zmax_over_H)
        return self


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

    def _get_mask(self):
        pass 

    def set_mask(self):
        pass



class Disk(SubModel):

    _required_parameters = [
        "aspectratio",
        "flaringindex",
        "r0"
    ]

    def __init__(self, base: Model, parameters: Dict[str, Any]):
        super().__init__(base)
        for key in self._required_parameters:
            if key not in parameters:
                raise ValueError(f"Missing required parameter in Disk class: {key}")
        self.parameters: Dict[str, Any] = parameters

    def _puff_density(self, nz: int = 64, zmax_scale: float = 5.0):
        if "surface_density" not in self.base.gas:
            raise KeyError("Missing gas field 'surface_density'")

        Sigma = self.base.gas["surface_density"].data  # Quantity (nrad, nsec)
        r     = self.base.mesh.centers("r")            # Quantity (nrad,)   <— changed

        h0 = self.parameters["aspectratio"]
        fl = self.parameters["flaringindex"]
        r0 = self.parameters["r0"]

        H     = _scale_height(r, h0, fl, r0)             # (nrad,)
        zmed  = _build_z_grid(nz, zmax_scale, H)         # (nz,)
        rho3d = _puff_gaussian(Sigma, H, zmed)           # (nz, nrad, nsec) Quantity

        # 3D cylindrical mesh: reuse r/phi, add z centers
        new_mesh = self.base.mesh.with_vertical(zmed)     # <— pure builder

        # Return density (meshless Field) and new mesh
        dens_field = Field(
            data=rho3d,
            quantity="density",
            axis_order=("z", "r", "phi"),
        )
        return dens_field, new_mesh

    def _puff_velocity(self, nz: int = 64, zmed=None):
        vr2d   = self.base.gas['vr'].data   # (nrad, nsec) Quantity
        vphi2d = self.base.gas['vphi'].data # (nrad, nsec) Quantity

        if zmed is None:
            h0 = self.parameters["aspectratio"]
            fl = self.parameters["flaringindex"]
            r0 = self.parameters["r0"]
            H  = _scale_height(self.base.mesh.centers("r"), h0, fl, r0)
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
    coordinates: str = "cylindrical",
    zmax_over_H: float = 5.0,
) -> "Model":
        r  = self.base.mesh.centers("r")
        h0 = self.parameters["aspectratio"]
        fl = self.parameters["flaringindex"]
        r0 = self.parameters["r0"]

        if coordinates == "cylindrical":
            dens_field, new_mesh = self._puff_density(nz=n, zmax_scale=zmax_over_H)
            vel = self._puff_velocity(nz=n, zmed=new_mesh.centers("z"))

            # 3) Update all mesh references in the model
            self.base.mesh = new_mesh
            self.mesh = new_mesh
            if hasattr(self.base, "gas") and self.base.gas is not None:
                self.base.gas.mesh = new_mesh

            new_fields: Dict[str, Field] = {}
            new_fields["density"] = dens_field

            if "vr" in self.base.gas:
                new_fields["vr"] = Field(
                    quantity="vr",
                    data=vel["vr3d_cyl"],
                    axis_order=("z", "r", "phi"),
                )
            if "vphi" in self.base.gas:
                new_fields["vphi"] = Field(
                    quantity="vphi",
                    data=vel["vphi3d_cyl"],
                    axis_order=("z", "r", "phi"),
                )

            # vz = 0 with velocity units if available
            vunit = (getattr(self.base.gas["vr"].data, "units", None)
                    if "vr" in self.base.gas else
                    getattr(self.base.gas["vphi"].data, "units", None) if "vphi" in self.base.gas else None)
            nr = int(new_mesh.ncell("r") or 0)
            nphi = int(new_mesh.ncell("phi") or 0)
            vz = np.zeros((len(new_mesh.centers("z")), nr, nphi))
            if vunit is not None:
                vz = vz * vunit
            new_fields["vz"] = Field(quantity="vz", data=vz, axis_order=("z", "r", "phi"))

            self.base.gas.clear()
            for k, f in new_fields.items():
                self.base.gas_register(k, f)
            self.coord_system = "cylindrical"
            return self

        elif coordinates == "spherical":
            # Puff in cyl at half the target polar resolution for economy
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
            vr_units = getattr(vel["vr3d_cyl"], "units", None) if "vr" in self.base.gas else None
            vphi_units = getattr(vel["vphi3d_cyl"], "units", None) if "vphi" in self.base.gas else None

            rho_sph_raw = _interp_cyl_to_sph(dens_field_cyl.data, r_cyl, zmed, r_sph, t_sph)
            vr_sph_raw = _interp_cyl_to_sph(vel["vr3d_cyl"], r_cyl, zmed, r_sph, t_sph) if "vr" in self.base.gas else None
            vphi_sph_raw = _interp_cyl_to_sph(vel["vphi3d_cyl"], r_cyl, zmed, r_sph, t_sph) if "vphi" in self.base.gas else None

            # Reattach units
            rho_sph = rho_sph_raw * rho_units if rho_units else rho_sph_raw
            vr_sph = vr_sph_raw * vr_units if vr_units and vr_sph_raw is not None else None
            vphi_sph = vphi_sph_raw * vphi_units if vphi_units and vphi_sph_raw is not None else None

            # vtheta = 0 with proper unit
            vunit = (getattr(self.base.gas["vr"].data, "units", None)
                    if "vr" in self.base.gas else
                    getattr(self.base.gas["vphi"].data, "units", None) if "vphi" in self.base.gas else None)
            ntheta = int(sph_mesh.ncell("theta") or 0)
            nr = int(sph_mesh.ncell("r") or 0)
            nphi = int(sph_mesh.ncell("phi") or 0)
            vtheta = np.zeros((ntheta, nr, nphi))
            if vunit is not None:
                vtheta = vtheta * vunit

            # Commit spherical mesh + fields
            self.base.mesh = sph_mesh
            self.mesh = sph_mesh
            if hasattr(self.base, "gas") and self.base.gas is not None:
                self.base.gas.mesh = sph_mesh
            self.coord_system = "spherical"

            self.base.gas.clear()
            self.base.gas_register("density", Field(quantity="density", data=rho_sph, axis_order=("theta","r","phi")))
            if vr_sph is not None:
                self.base.gas_register("vr", Field(quantity="vr", data=vr_sph, axis_order=("theta","r","phi")))
            if vphi_sph is not None:
                self.base.gas_register("vphi", Field(quantity="vphi", data=vphi_sph, axis_order=("theta","r","phi")))
            self.base.gas_register("vtheta", Field(quantity="vtheta", data=vtheta, axis_order=("theta","r","phi")))
            return self

        else:
            raise ValueError("coordinates must be 'cylindrical' or 'spherical'")


def _scale_height(r: Quantity, h0: Quantity, flaringindex: Quantity, r0: Quantity) -> Quantity:
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
