from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..mesh import Mesh, Axis
from ..field import Field
from diskbridge._units import Quantity 
from diskbridge._logging import logger
from diskbridge import units  

# -----------------
# helpers
# -----------------

G_phys = Quantity(6.674e-11, "m^3 / (kg s^2)")

au = 1.0*units('au')
solar_mass = 1.0*units('solar_mass')

au = units('au')
solar_mass = units('solar_mass')

# Derived code units
T0 = (au**3 / (G_phys * solar_mass))**0.5
V0 = au / T0
rho0 = solar_mass / au**3
Sigma0 = solar_mass / au**2
nu0 = au**2 / T0
Omega0 = 1 / T0

# Define custom units in Pint
units.define(f"code_length = {au.to('m').magnitude} * meter")
units.define(f"code_mass = {solar_mass.to('kg').magnitude} * kilogram")
units.define(f"code_time = {T0.to('s').magnitude} * second")
units.define(f"code_velocity = {V0.to('m/s').magnitude} * meter / second")
units.define(f"code_density = {rho0.to('kg/m^3').magnitude} * kg / meter ** 3")
units.define(f"code_surface_density = {Sigma0.to('kg/m^2').magnitude} * kg / meter ** 2")
units.define(f"code_viscosity = {nu0.to('m^2/s').magnitude} * meter ** 2 / second")
units.define(f"code_omega = {Omega0.to('1/s').magnitude} / second")


def _get_units(file_units: str):
    """
    Build the units dictionary based on the file units
    """
    if file_units.lower() == "cgs":
        unit_length = units("cm")
        unit_time = units("s")
        unit_mass = units("g")
        unit_velocity = units("cm/s")
        unit_surface_density = units("g/cm^2")
        unit_density = units("g/cm^3")
        unit_viscosity = units("g^2/s")
    elif file_units.lower() == "kms":
        unit_length = units("m")
        unit_time = units("s")
        unit_mass = units("kg")
        unit_velocity = units("m/s")
        unit_surface_density = units("kg/m^2")
        unit_density = units("kg/m^3")
        unit_viscosity = units("m^2/s")
    else:
        unit_length = units("code_length")
        unit_time = units("code_time")
        unit_mass = units("code_mass")
        unit_velocity = units("code_velocity")
        unit_surface_density = units("code_surface_density")
        unit_density = units("code_density")
        unit_viscosity = units("code_viscosity")
    
    unit_dict = {
        "unit_length": unit_length,
        "unit_time": unit_time,
        "unit_mass": unit_mass,
        "unit_velocity": unit_velocity,
        "unit_surface_density": unit_surface_density,
        "unit_density": unit_density,
        "unit_viscosity": unit_viscosity
    }
    return unit_dict

def _read_variables_par(path: Path) -> Dict[str, Any]:
    p = path / "variables.par" if path.is_dir() else path
    if not p.exists():
        logger.warning(f"variables.par not found at {p}; proceeding with empty variables")
        return {}
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
        logger.info(f"Could not read dims from {p}; will try fallbacks")
    return None


def _read_used_rad(path: Path) -> Optional[np.ndarray]:
    p = path / "used_rad.dat"
    try:
        return np.loadtxt(p)
    except Exception:
        logger.info(f"Could not read radial edges from {p}; will try fallbacks")
        return None


def _build_pedge(nsec: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, nsec + 1)


def _read_summary(directory: Path, file_n: int) -> Tuple[Dict[str, Optional[bool]], Dict[str, float]]:
    candidates = [directory / f"summary{file_n}.dat", directory / "summary.dat"]
    p = None
    for c in candidates:
        if c.exists():
            p = c
            break
    if p is None:
        out_flags: Dict[str, Optional[bool]] = {
            "NFLUIDS": None,
            "ISOTHERMAL": None,
            "CYLINDRICAL": None,
            "VISCOSITY": None,
        }
        return out_flags, {}

    text = p.read_text(errors="ignore")
    lines = [ln.rstrip() for ln in text.splitlines()]

    in_comp = False
    flags_found: Dict[str, bool] = {}
    for ln in lines:
        if "COMPILATION OPTION SECTION" in ln:
            in_comp = True
            continue
        if in_comp:
            if not ln or ln.startswith("=") or ln.startswith("Ghost layer") or "SECTION:" in ln:
                in_comp = False
                continue
            parts = ln.split()
            for tok in parts:
                if tok.startswith("-D"):
                    name_val = tok[2:]
                    name = name_val.split("=")[0]
                    if name:
                        flags_found[name] = True

    out_flags_bool: Dict[str, Optional[bool]] = {
        "NFLUIDS": True if flags_found.get("NFLUIDS") else False,
        "ISOTHERMAL": True if flags_found.get("ISOTHERMAL") else False,
        "CYLINDRICAL": True if flags_found.get("CYLINDRICAL") else False,
        "VISCOSITY": True if flags_found.get("VISCOSITY") else False,
    }
    for k in flags_found.keys():
        if k not in out_flags_bool:
            out_flags_bool[k] = True

    in_mac = False
    macros: Dict[str, float] = {}
    for ln in lines:
        if "PREPROCESSOR MACROS SECTION" in ln:
            in_mac = True
            continue
        if in_mac:
            if not ln or ln.startswith("=") or "SECTION:" in ln:
                in_mac = False
                continue
            if "=" in ln:
                left, _, right = ln.partition("=")
                name = left.strip()
                last = ln.split("=")[-1].strip()
                try:
                    val = float(last)
                    macros[name] = val
                except ValueError:
                    pass

    return out_flags_bool, macros


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
    compile_options, macros = _read_summary(directory, file_n)

    # Auto-detect dimensionality
    nz_val = variables.get("NZ", None)
    if nz_val is not None:
        try:
            nz = int(nz_val)
        except Exception:
            nz = 0
        is_3d = nz > 1
    else:
        nz = 0
        # Only fall back to domain_z.dat presence when NZ absent
        if (directory / "domain_z.dat").exists():
            is_3d = True
            logger.info("Detected 3D run from domain_z.dat; NZ missing in variables.par")
        else:
            is_3d = False

    # Dimensions and edges
    if not is_3d:
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
        redge = _read_used_rad(directory) * unit_dict['unit_length']
        if redge is None:
            # Try to build from variables
            try:
                rmin = float(variables.get("RMIN"))
                rmax = float(variables.get("RMAX"))
                if rmax <= rmin:
                    raise ValueError
                redge = np.linspace(rmin, rmax, nrad + 1) * unit_dict['unit_length']
                logger.warning("used_rad.dat missing; built redge from RMIN/RMAX in variables.par")
            except Exception:
                logger.error("used_rad.dat missing and could not infer redge from variables.par")
                raise FileNotFoundError("used_rad.dat not found and no valid RMIN/RMAX; cannot build radial edges")

        # Azimuth edges
        pedge = _build_pedge(nsec) * units('radians')

        mesh = Mesh.cylindrical(r=Axis(edges=redge), 
                                phi=Axis(edges=pedge))
        coord_system = "cylindrical"
        ncol = 1
    else:
        # For 3D, prefer edges from domain files (FARGO3D)
        px = directory / "domain_x.dat"
        py = directory / "domain_y.dat"
        pz = directory / "domain_z.dat"
        pedge = None
        redge = None
        tedge = None
        if px.exists():
            try:
                pedge = np.loadtxt(px) * units('radians')
            except Exception:
                logger.warning(f"Failed reading {px}; will try building phi edges from NX")
        else:
            logger.info(f"{px} not found; will try building phi edges from NX")
        if py.exists():
            try:
                redge = np.loadtxt(py) * unit_dict['unit_length']
            except Exception:
                logger.warning(f"Failed reading {py}; will try used_rad.dat or variables")
        else:
            logger.info(f"{py} not found; will try used_rad.dat or variables")
        if pz.exists():
            try:
                tedge = np.loadtxt(pz) * units('radians')
            except Exception:
                logger.warning(f"Failed reading {pz}; will try building theta edges from NZ")
        else:
            logger.info(f"{pz} not found; will try building theta edges from NZ")

        # Fallbacks
        if redge is None:
            rr = _read_used_rad(directory)
            if rr is not None:
                redge = rr * unit_dict['unit_length']
                logger.warning("domain_y.dat missing; using used_rad.dat for radial edges")
        if redge is None:
            # variables RMIN/RMAX and NY
            try:
                nrad = int(variables.get("NY"))
                rmin = float(variables.get("RMIN"))
                rmax = float(variables.get("RMAX"))
                if nrad > 0 and rmax > rmin:
                    redge = np.linspace(rmin, rmax, nrad + 1) * unit_dict['unit_length']
                    logger.warning("domain_y.dat missing; built redge from variables.par (RMIN/RMAX/NY)")
            except Exception:
                pass

        if pedge is None:
            try:
                nsec = int(variables.get("NX"))
                if nsec > 0:
                    pedge = _build_pedge(nsec) * units('radians')
                    logger.warning("domain_x.dat missing; built phi edges from NX")
            except Exception:
                pass

        if tedge is None:
            if nz > 1:
                ncol = nz
                tedge = np.linspace(0.0, np.pi, ncol + 1) * units('radians')
                logger.warning("domain_z.dat missing; built theta edges uniformly from NZ between 0 and pi")

        # Final checks
        if redge is None or pedge is None or tedge is None:
            logger.error("Insufficient information to construct 3D mesh (need domain files or valid fallbacks)")
            raise FileNotFoundError("Cannot construct 3D mesh: missing domain edges and fallbacks")

        # Cell counts
        # Note: FARGO3D domain files include ghost zones (3 on each side) for radial and vertical directions
        # The azimuthal direction is periodic and does not have ghost zones
        nsec_raw = int(pedge.size - 1)
        nrad_raw = int(redge.size - 1)
        ncol_raw = int(tedge.size - 1)
        
        # Account for 6 ghost zones (3 on each side) in radial and colatitude
        # No ghost zones in azimuthal (periodic boundary)
        nsec = nsec_raw
        nrad = nrad_raw - 6
        ncol = ncol_raw - 6
        
        # Adjust edges to exclude ghost zones
        # FARGO3D domain files: first 3 are lower ghost, last 3 are upper ghost
        redge = redge[3:-3]
        tedge = tedge[3:-3]
        # pedge does not need adjustment (no ghost zones in azimuthal)

        mesh = Mesh.spherical(r=Axis(edges=redge), 
                              phi=Axis(edges=pedge), 
                              theta=Axis(edges=tedge))
        coord_system = "spherical"

    # Load fields present on disk for the snapshot
    gas_fields: Dict[str, Field] = {}

    def _read_field(filename: str, quantity: str, units: Quantity) -> Optional[Field]:
        p = directory / filename
        if not p.exists():
            logger.info(f"Field file missing: {p}")
            return None
        arr = np.fromfile(p, dtype="float64")
        if is_3d:
            # FARGO3D stores 3D data in (ncol, nrad, nsec) order (colatitude, radius, azimuth)
            # This is the native output format from FARGO3D simulations
            try:
                arr = arr.reshape(ncol, nrad, nsec)
            except Exception:
                logger.warning(f"Reshape to (ncol,nrad,nsec) failed for {p}; attempting auto-infer of dimensions")
                if nrad > 0 and nsec > 0 and arr.size % (nrad * nsec) == 0:
                    inferred_ncol = arr.size // (nrad * nsec)
                    arr = arr.reshape(inferred_ncol, nrad, nsec)
                    logger.info(f"Inferred ncol={inferred_ncol} from file size for {p}")
                else:
                    raise
            
            # Apply azimuthal roll by nsec/2 (FARGO3D convention: origin at x-axis, needs shift to align with standard coordinates)
            arr = np.roll(arr, shift=int(nsec//2), axis=2)
            
            # Transpose to (r, phi, theta) order for consistency with mesh
            arr = np.transpose(arr, (1, 2, 0))  # (ncol, nrad, nsec) -> (nrad, nsec, ncol)
            axes = ("r", "phi", "theta")
        else:
            arr = arr.reshape(nrad, nsec)
            # For FARGO3D runs (indicated by presence of variables.par), apply azimuthal roll
            if (directory / "variables.par").exists():
                arr = np.roll(arr, shift=int(nsec//2), axis=1)
            axes = ("r", "phi")
        arr = arr * units
        return Field(
            data=arr,
            quantity=quantity,
            axis_order=axes,
        )

    # FARGO 2D canonical names
    if is_3d:
        gas_fields["density"] = _read_field(f"gasdens{file_n}.dat", 
                                            "density", 
                                            units=unit_dict['unit_density'])
    else:
        gas_fields["surface_density"] = _read_field(f"gasdens{file_n}.dat", 
                                                    "surface_density", 
                                                    units=unit_dict['unit_surface_density'])

    gas_fields["vr"] = _read_field(f"gasvx{file_n}.dat", 
                                   "vr", 
                                   units=unit_dict['unit_velocity'])

    gas_fields["vphi"] = _read_field(f"gasvy{file_n}.dat", 
                                     "vphi", 
                                     units=unit_dict['unit_velocity'])

    if is_3d:
        gas_fields["vtheta"] = _read_field(f"gasvz{file_n}.dat", 
                                            "vtheta", 
                                            units=unit_dict['unit_velocity'])

    # Curate disk parameters used by downstream steps
    disk_parameters: Dict[str, Any] = {}
    disk_parameters["alphavisocity"] = variables.get("ALPHA") * units('dimensionless')
    disk_parameters["kinematicviscosity"] = variables.get("NU") * unit_dict['unit_viscosity']
    disk_parameters["aspectratio"] = variables.get("ASPECTRATIO") * units('dimensionless')
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
        "compile_options": compile_options,
        "macros": macros,
        "disk_parameters": disk_parameters,
        "mesh": mesh,
        "gas_fields": gas_fields,
        "output_number": file_n,
    }
