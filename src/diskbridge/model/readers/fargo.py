from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..mesh import Mesh
from ..field import Field
from .._units import Quantity 
from diskbridge import units  


# -----------------
# helpers
# -----------------

G_phys = Quantity(6.674e-11, "m^3 / (kg s^2)")

# Derived code units
T0 = (units("au")**3 / (G_phys * units("solar_mass")))**0.5
V0 = units("au") / T0
rho0 = units("solar_mass") / units("au")**3
Sigma0 = units("solar_mass") / units("au")**2
nu0 = units("au")**2 / T0
Omega0 = 1 / T0

# Define custom units in Pint
units.define(f"code_length = {R0_phys.to('m').magnitude} * meter")
units.define(f"code_mass = {Mstar_phys.to('kg').magnitude} * kilogram")
units.define(f"code_time = {T0.to('s').magnitude} * second")
units.define(f"code_velocity = {V0.to('m/s').magnitude} * meter / second")
units.define(f"code_density = {rho0.to('kg/m^3').magnitude} * kg / meter ** 3")
units.define(f"code_surface_density = {Sigma0.to('kg/m^2').magnitude} * kg / meter ** 2")
units.define(f"code_viscosity = {nu0.to('m^2/s').magnitude} * meter ** 2 / second")
units.define(f"code_omega = {Omega0.to('1/s').magnitude} / second")


def _get_units(file_units: str):
    if file_units.lower() == "cgs":
        unit_length = units("cm")
        unit_time = units("s")
        unit_mass = units("g")
        unit_velocity = units("cm/s")
        unit_surface_density = units("g/cm^2")
        unit_density = units("g/cm^3")
    elif file_units.lower() == "kms":
        unit_length = units("m")
        unit_time = units("s")
        unit_mass = units("kg")
        unit_velocity = units("m/s")
        unit_surface_density = units("kg/m^2")
        unit_density = units("kg/m^3")
    else:
        unit_length = units("code_length")
        unit_time = units("code_time")
        unit_mass = units("code_mass")
        unit_velocity = units("code_velocity")
        unit_surface_density = units("code_surface_density")
        unit_density = units("code_density")
    unit_dict = {
        "unit_length": unit_length,
        "unit_time": unit_time,
        "unit_mass": unit_mass,
        "unit_velocity": unit_velocity,
        "unit_surface_density": unit_surface_density,
        "unit_density": unit_density,
    }
    return unit_dict

def _read_variables_par(path: Path) -> Dict[str, Any]:
    p = path / "variables.par" if path.is_dir() else path
    if not p.exists():
        raise FileNotFoundError(f"variables.par not found at {p}")
    out: Dict[str, Any] = {}
    with p.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "!", "%")):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            key = parts[0].upper()
            val = parts[1]
            # int -> float -> str
            try:
                out[key] = int(val)
            except ValueError:
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
    return out


def _read_dims(path: Path) -> Optional[Tuple[int, int]]:
    p = path / "dims.dat"
    try:
        vals = np.loadtxt(p)
        flat = np.ravel(vals)
        if flat.size >= 8:
            # dims format: ... nrad nsec in last two columns in our examples
            nrad = int(flat[-2])
            nsec = int(flat[-1])
            return nrad, nsec
    except Exception:
        pass
    return None


def _read_used_rad(path: Path) -> Optional[np.ndarray]:
    p = path / "used_rad.dat"
    try:
        return np.loadtxt(p)
    except Exception:
        return None


def _build_pedge(nsec: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, nsec + 1)


# -----------------
# main entry
# -----------------

def read_fargo_snapshot(directory: Path, file_n: int, file_units: str = "code") -> Dict[str, Any]:
    """Read a FARGO/FARGO3D snapshot (2D or 3D auto-detected).

    Returns a dict with:
    - coord_system: 'cylindrical' for 2D, 'spherical' for 3D (future)
    - variables: dict
    - disk_parameters: dict with curated values
    - mesh: Mesh instance (minimal for 2D)
    - gas_fields: dict[str, Field]
    """

    unit_dict = _get_units(file_units)

    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    variables = _read_variables_par(directory)

    # Auto-detect dimensionality
    nz = int(variables.get("NZ", 1))
    is_3d = nz and nz > 1

    # Dimensions
    dims = _read_dims(directory)
    if dims is None:
        # Fall back to variables (FARGO uses NY=radial, NX=azimuthal)
        nrad = int(variables.get("NY", 0))
        nsec = int(variables.get("NX", 0))
        if nrad <= 0 or nsec <= 0:
            raise RuntimeError("Could not determine grid dimensions from dims.dat or variables.par")
    else:
        nrad, nsec = dims

    # Radial edges
    redge = _read_used_rad(directory)*unit_dict['unit_length']
    if redge is None:
        # For simplicity, require used_rad.dat in our examples; inference could be added later
        raise FileNotFoundError("used_rad.dat not found; cannot build radial edges")

    # Azimuth edges
    pedge = _build_pedge(nsec) * units('radians')

    # Build mesh (2D cylindrical for now)
    if not is_3d:
        mesh = Mesh(coord_system="cylindrical", redge=redge, pedge=pedge)
        coord_system = "cylindrical"
    else:
        # Placeholder for 3D: would also set tedge from domain_z.dat
        mesh = Mesh(coord_system="spherical", redge=redge, pedge=pedge)
        coord_system = "spherical"

    # Load fields present on disk for the snapshot
    gas_fields: Dict[str, Field] = {}

    def _read_field(filename: str, quantity: str) -> Optional[Field]:
        p = directory / filename
        if not p.exists():
            return None
        arr = np.fromfile(p, dtype="float64").reshape(nrad, nsec)
        return Field(
            data=arr,
            mesh=mesh,
            quantity=quantity,
            axis_order=("r", "phi"),
        )

    # FARGO 2D canonical names
    f_density = _read_field(f"gasdens{file_n}.dat", "density")
    if f_density is not None:
        gas_fields["density"] = f_density*unit_dict['unit_surface_density']

    f_vr = _read_field(f"gasvx{file_n}.dat", "vr")
    if f_vr is not None:
        gas_fields["vr"] = f_vr*unit_dict['unit_velocity']

    f_vphi = _read_field(f"gasvy{file_n}.dat", "vphi")
    if f_vphi is not None:
        gas_fields["vphi"] = f_vphi*unit_dict['unit_velocity']

    # Curate disk parameters used by downstream steps
    disk_parameters: Dict[str, Any] = {}
    disk_parameters["alphavisocity"] = variables.get("ALPHA") * units('dimensionless')
    disk_parameters["honr"] = variables.get("ASPECTRATIO") * units('dimensionless')
    disk_parameters["flaringindex"] = variables.get("FLARINGINDEX") * units('dimensionless')
    disk_parameters["sigma0"] = variables.get("SIGMA0") * unit_dict['unit_surface_density']
    disk_parameters["sigmaslope"] = variables.get("SIGMASLOPE") * units('dimensionless')
    disk_parameters["gamma"] = variables.get("GAMMA") * units('dimensionless')
    disk_parameters["cs"] = variables.get("CS") * unit_dict['unit_velocity']

    norm_units = (file_units or "code").lower()
    if norm_units in ("cgs", "kms"):
        disk_parameters["r0"] = 5.2 * units('au')
    else:
        disk_parameters["r0"] = 1.0 * units('code_length')

    return {
        "coord_system": coord_system,
        "variables": variables,
        "disk_parameters": disk_parameters,
        "mesh": mesh,
        "gas_fields": gas_fields,
        "output_number": file_n,
    }
