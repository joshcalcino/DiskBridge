from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..mesh import Mesh
from ..field import Field
from diskbridge import units  


# -----------------
# helpers
# -----------------

def _set_units(file_units: str):
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
        u_v = "code_velocity"
        u_sigma = "code_surface_density"
    return u_v, u_sigma


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
    redge = _read_used_rad(directory)
    if redge is None:
        # For simplicity, require used_rad.dat in our examples; inference could be added later
        raise FileNotFoundError("used_rad.dat not found; cannot build radial edges")

    # Azimuth edges
    pedge = _build_pedge(nsec)

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

    def _read_field(filename: str, quantity: str, unit: str) -> Optional[Field]:
        p = directory / filename
        if not p.exists():
            return None
        arr = np.fromfile(p, dtype="float64").reshape(nrad, nsec)
        return Field(
            data=arr,
            mesh=mesh,
            unit=unit,
            quantity=quantity,
            axis_order=("r", "phi"),
        )

    # Choose unit strings using the units registry (to keep parseable names)
    norm_units = (file_units or "code").lower()
    if norm_units == "cgs":
        u_v = "cm/s"  # units("cm/s") is valid
        u_sigma = "g/cm^2"
    elif norm_units == "kms":
        u_v = "km/s"  # units("km/s") is valid
        # In KMS mode, density remains in cgs mass/area
        u_sigma = "g/cm^2"
    else:
        # code units: leave as code_* tags that the unit system can map externally
        u_v = "code_velocity"
        u_sigma = "code_surface_density"

    # FARGO 2D canonical names
    f_density = _read_field(f"gasdens{file_n}.dat", "density", unit=u_sigma)
    if f_density is not None:
        gas_fields["density"] = f_density

    f_vr = _read_field(f"gasvx{file_n}.dat", "vr", unit=u_v)
    if f_vr is not None:
        gas_fields["vr"] = f_vr

    f_vphi = _read_field(f"gasvy{file_n}.dat", "vphi", unit=u_v)
    if f_vphi is not None:
        gas_fields["vphi"] = f_vphi

    # Curate disk parameters used by downstream steps
    disk_parameters: Dict[str, Any] = {}
    disk_parameters["alphavisocity"] = variables.get("ALPHA")
    disk_parameters["honr"] = variables.get("ASPECTRATIO")
    disk_parameters["flaringindex"] = variables.get("FLARINGINDEX")

    norm_units = (file_units or "code").lower()
    if norm_units in ("cgs", "kms"):
        disk_parameters["r0"] = 5.2
    else:
        disk_parameters["r0"] = 1.0

    return {
        "coord_system": coord_system,
        "variables": variables,
        "disk_parameters": disk_parameters,
        "mesh": mesh,
        "gas_fields": gas_fields,
        "output_number": file_n,
    }
