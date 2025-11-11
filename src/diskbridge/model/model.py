from __future__ import annotations

from dataclasses import dataclass, field as dcfield
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import numpy as np

from .mesh import Mesh
from .field import Field


@dataclass
class GasDiskParameters:
    alpha: Optional[float] = None
    nu: Optional[float] = None
    aspectratio: Optional[float] = None
    aspect: Optional[Union[float, str]] = None
    flaringindex: Optional[float] = None
    sigma0: Optional[float] = None
    sigmaslope: Optional[float] = None
    gamma: Optional[float] = None
    cs: Optional[float] = None
    coordinates: Optional[str] = None
    r0: Optional[float] = None
    parameters: Dict[str, Any] = dcfield(default_factory=dict)


@dataclass
class Model:
    coord_system: Optional[str] = None
    variables: Dict[str, Any] = dcfield(default_factory=dict)
    compile_options: Dict[str, Optional[bool]] = dcfield(default_factory=dict)
    macros: Dict[str, float] = dcfield(default_factory=dict)
    disk_parameters: Dict[str, Any] = dcfield(default_factory=dict)
    mesh: Optional[Mesh] = None
    file_units: str = "code" # 'kms', 'cgs', or 'code'
    directory: Optional[str] = None
    n_file: Optional[int] = None
    filename: Optional[str] = None 

    gas: SubModel = dcfield(default_factory=SubModel)
    disk: GasDiskParameters = dcfield(default_factory=GasDiskParameters)

    def __post_init__(self) -> None:
        # Surface disk parameters for example access pattern model.disk.parameters[...]
        if self.disk_parameters:
            self.disk.parameters.update(self.disk_parameters)
        if not isinstance(self.gas, SubModel):
            self.gas = SubModel(self)
        else:
            self.gas._model = self

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
        self.disk_parameters = snap["disk_parameters"]
        self.mesh = snap["mesh"]
        self.file_units = file_units
        self.directory = str(p)
        # reflect into disk.parameters
        self.disk.parameters.clear()
        self.disk.parameters.update(self.disk_parameters)

        # Overwrite Disk defaults from variables.par
        self.disk.alpha = self.variables.get("ALPHA")
        self.disk.nu = self.variables.get("NU")
        self.disk.aspectratio = self.variables.get("ASPECTRATIO")
        self.disk.aspect = self.variables.get("ASPECT")
        self.disk.flaringindex = self.variables.get("FLARINGINDEX")
        self.disk.sigma0 = self.variables.get("SIGMA0")
        self.disk.sigmaslope = self.variables.get("SIGMASLOPE")
        self.disk.gamma = self.variables.get("GAMMA")
        self.disk.cs = self.variables.get("CS")
        self.disk.coordinates = self.variables.get("COORDINATES")

        # Register gas fields
        self.gas.clear()
        for name, field in snap["gas_fields"].items():
            self.gas_register(name, field)


        return self


class SubModel(Model):


    def __init__(self, base: Model):

        super().__init__()

        self.base = base
        self.mesh = self.base.mesh 
        self.disk = self.base.disk
        
        # self._lazy: Dict[str, Callable[[], Field]] = {}

        # # Attributes same as Model
        # self.data_source = self.base.data_source
        # self.file_path = self.base.file_path
        # self._code_units = self.base._code_units
        # self._default_units = self.base._default_units
        # self._properties = self.base._properties
        # self._array_code_units = self.base._array_code_units
        # self._array_registry = self.base._array_registry
        # self._arrays = self.base._arrays


    def register(self, name: str, field: Field) -> None:
        self[name] = field

    def register_lazy(self, name: str, builder: Callable[[], Field]) -> None:
        self._lazy[name] = builder

    def __getitem__(self, key: str) -> Field:
        if key in self.keys():
            return super().__getitem__(key)
        if key in self._lazy:
            field = self._lazy[key]()
            self[key] = field
            return field
        return super().__getitem__(key)
