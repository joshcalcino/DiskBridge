from __future__ import annotations

from dataclasses import dataclass, field as dcfield
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import numpy as np

from .mesh import Mesh
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
        """Load a hydro snapshot and populate this Model, then return self.

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
        
        # Initialize Disk parameters of one is present
        if "disk_parameters" in snap:
            self.disk = Disk()
            self.disk.parameters.clear()
            self.disk.parameters.update(snap["disk_parameters"])

        # Register gas fields
        for name, field in snap["gas_fields"].items():
            self.gas = SubModel(self)
            self.gas_register(name, field)

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

    def __init__(self):
        self.parameters: Dict[str, Any] = {}

    def puff_density(
        self,
        nz: int = 64,
        zmax_scale: float = 5.0,
        surface_density_key: str = "surface_density",
    ):
        if self.mesh is None:
            raise ValueError("mesh is not loaded")
        if surface_density_key not in self.gas:
            raise KeyError(f"Missing gas field '{surface_density_key}'")

        Sigma = self.gas[surface_density_key].data  # (nrad, nsec)
        r = self.mesh.rmed  # (nrad,)

        h0 = _get_param(self, "ASPECTRATIO", fallback=self.disk.aspectratio, default=0.05)
        fl = _get_param(self, "FLARINGINDEX", fallback=self.disk.flaringindex, default=0.0)
        r0 = _get_param(self, "R0", fallback=self.disk.r0, default=1.0)

        H = _scale_height(r, h0, fl, r0)  # (nrad,)
        zmed = _build_z_grid(nz, zmax_scale, H)

        rho3d = _puff_gaussian(Sigma, H, zmed)
        return rho3d, zmed

    def puff_velocity(
        self,
        nz: int = 64,
        zmed=None,
        vr_key: str = "vr",
        vphi_key: str = "vphi",
    ):
        if self.mesh is None:
            raise ValueError("mesh is not loaded")
        if vr_key not in self.gas or vphi_key not in self.gas:
            missing = [k for k in (vr_key, vphi_key) if k not in self.gas]
            raise KeyError(f"Missing gas field(s): {missing}")

        vr2d = self.gas[vr_key].data  # (nrad, nsec)
        vphi2d = self.gas[vphi_key].data  # (nrad, nsec)

        if zmed is None:
            # Build a symmetric z grid using a default scale height estimate
            h0 = _get_param(self, "ASPECTRATIO", fallback=self.disk.aspectratio, default=0.05)
            fl = _get_param(self, "FLARINGINDEX", fallback=self.disk.flaringindex, default=0.0)
            r0 = _get_param(self, "R0", fallback=self.disk.r0, default=1.0)
            H = _scale_height(self.mesh.rmed, h0, fl, r0)
            zmed = _build_z_grid(nz, 5.0, H)

        nrad = vr2d.shape[0]
        nsec = vr2d.shape[1]

        vr3d = vr2d.reshape(1, nrad, nsec) * np.ones((len(zmed), 1, 1))
        vphi3d = vphi2d.reshape(1, nrad, nsec) * np.ones((len(zmed), 1, 1))

        return {"vr3d_cyl": vr3d, "vphi3d_cyl": vphi3d, "zmed": zmed}


    def _scale_height(r: np.ndarray, h0: float, flaringindex: float, r0: float) -> np.ndarray:
        return (h0 * (r / r0) ** flaringindex) * r


    def _build_z_grid(nver: int, zmax_scale: float, H: np.ndarray) -> np.ndarray:
        zmax = float(zmax_scale) * float(np.max(H))
        return np.linspace(-zmax, zmax, int(nver))


    def _puff_gaussian(Sigma: np.ndarray, H: np.ndarray, zmed: np.ndarray) -> np.ndarray:
        nrad, nsec = Sigma.shape
        nver = zmed.size
        H2 = H.reshape(1, nrad, 1)
        Z = zmed.reshape(nver, 1, 1)
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * H2)
        rho = Sigma.reshape(1, nrad, nsec) * norm * np.exp(-0.5 * (Z / H2) ** 2)
        return rho

