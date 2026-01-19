from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..mesh import Mesh, Axis
from ..field import Field
from diskbridge._units import Quantity 
from diskbridge._logging import logger
from diskbridge import units


@dataclass
class DimensionInfo:
    is_3d: bool
    nrad: int
    nsec: int
    ncol: int  

# -----------------
# helpers
# -----------------

G_phys = units('G').to("m^3 / (kg s^2)")

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
        unit_viscosity = units("cm^2/s")
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
    skip_next_equals = False
    flags_found: Dict[str, bool] = {}
    for i, ln in enumerate(lines):
        if "COMPILATION OPTION SECTION" in ln:
            in_comp = True
            skip_next_equals = True  # Skip the ===== line right after section header
            continue
        if in_comp:
            # Skip the first ===== line after section header
            if skip_next_equals and ln.startswith("="):
                skip_next_equals = False
                continue
            # End section on empty line, subsequent ===, or new SECTION
            if not ln or (ln.startswith("=") and not skip_next_equals) or "SECTION:" in ln:
                in_comp = False
                continue
            # Split on "Ghost layer" if present (it can be on the same line as flags)
            if "Ghost layer" in ln:
                ln = ln.split("Ghost layer")[0]
                in_comp = False  # End section after this line
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


def _detect_dimensionality(
    directory: Path, variables: Dict[str, Any]
) -> DimensionInfo:
    nz_val = variables.get("NZ", None)
    if nz_val is not None:
        try:
            nz = int(nz_val)
        except Exception:
            nz = 0
        is_3d = nz > 1
    else:
        nz = 0
        if (directory / "domain_z.dat").exists():
            is_3d = True
            logger.info("Detected 3D run from domain_z.dat; NZ missing in variables.par")
        else:
            is_3d = False
    
    return DimensionInfo(is_3d=is_3d, nrad=0, nsec=0, ncol=nz if is_3d else 1)


def _load_edges_2d(
    directory: Path, variables: Dict[str, Any], unit_dict: Dict[str, Quantity]
) -> Tuple[Quantity, Quantity, int, int]:
    dims = _read_dims(directory)
    if dims is None:
        nrad = int(variables.get("NY", 0))
        nsec = int(variables.get("NX", 0))
        if nrad <= 0 or nsec <= 0:
            raise RuntimeError("Could not determine grid dimensions from dims.dat or variables.par")
    else:
        nrad, nsec = dims

    redge = _read_used_rad(directory)
    if redge is None:
        try:
            rmin = float(variables.get("RMIN"))
            rmax = float(variables.get("RMAX"))
            if rmax <= rmin:
                raise ValueError
            redge = np.linspace(rmin, rmax, nrad + 1)
            logger.warning("used_rad.dat missing; built redge from RMIN/RMAX in variables.par")
        except Exception:
            logger.error("used_rad.dat missing and could not infer redge from variables.par")
            raise FileNotFoundError("used_rad.dat not found and no valid RMIN/RMAX; cannot build radial edges")
    
    redge = redge * unit_dict['unit_length']
    pedge = _build_pedge(nsec) * units('radian')
    return redge, pedge, nrad, nsec


def _load_edges_3d(
    directory: Path, variables: Dict[str, Any], unit_dict: Dict[str, Quantity], nz: int
) -> Tuple[Quantity, Quantity, Quantity, int, int, int]:
    px = directory / "domain_x.dat"
    py = directory / "domain_y.dat"
    pz = directory / "domain_z.dat"
    pedge = None
    redge = None
    tedge = None
    
    if px.exists():
        try:
            pedge = np.loadtxt(px) * units('radian')
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
            tedge = np.loadtxt(pz) * units('radian')
        except Exception:
            logger.warning(f"Failed reading {pz}; will try building theta edges from NZ")
    else:
        logger.info(f"{pz} not found; will try building theta edges from NZ")

    if redge is None:
        rr = _read_used_rad(directory)
        if rr is not None:
            redge = rr * unit_dict['unit_length']
            logger.warning("domain_y.dat missing; using used_rad.dat for radial edges")
    
    if redge is None:
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
                pedge = _build_pedge(nsec) * units('radian')
                logger.warning("domain_x.dat missing; built phi edges from NX")
        except Exception:
            pass

    if tedge is None:
        if nz > 1:
            ncol = nz
            tedge = np.linspace(0.0, np.pi, ncol + 1) * units('radian')
            logger.warning("domain_z.dat missing; built theta edges uniformly from NZ between 0 and pi")

    if redge is None or pedge is None or tedge is None:
        logger.error("Insufficient information to construct 3D mesh (need domain files or valid fallbacks)")
        raise FileNotFoundError("Cannot construct 3D mesh: missing domain edges and fallbacks")

    nsec_raw = int(pedge.size - 1)
    nrad_raw = int(redge.size - 1)
    ncol_raw = int(tedge.size - 1)
    
    nsec = nsec_raw
    nrad = nrad_raw - 6
    ncol = ncol_raw - 6
    
    redge = redge[3:-3]
    tedge = tedge[3:-3]
    
    return redge, pedge, tedge, nrad, nsec, ncol


def _load_fields(
    directory: Path,
    file_n: int,
    is_3d: bool,
    nrad: int,
    nsec: int,
    ncol: int,
    unit_dict: Dict[str, Quantity],
) -> Dict[str, Field]:
    gas_fields: Dict[str, Field] = {}
    
    def _read_field(filename: str, quantity: str, units: Quantity) -> Optional[Field]:
        p = directory / filename
        if not p.exists():
            logger.info(f"Field file missing: {p}")
            return None
        arr = np.fromfile(p, dtype="float64")
        if is_3d:
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
            
            arr = np.roll(arr, shift=int(nsec//2), axis=2)
            arr = np.transpose(arr, (1, 2, 0))
            axes = ("r", "phi", "theta")
        else:
            arr = arr.reshape(nrad, nsec)
            if (directory / "variables.par").exists():
                arr = np.roll(arr, shift=int(nsec//2), axis=1)
            axes = ("r", "phi")
        arr = arr * units
        return Field(
            data=arr,
            quantity=quantity,
            axis_order=axes,
        )

    if is_3d:
        gas_fields["density"] = _read_field(f"gasdens{file_n}.dat", 
                                            "density", 
                                            units=unit_dict['unit_density'])
    else:
        gas_fields["surface_density"] = _read_field(f"gasdens{file_n}.dat", 
                                                    "surface_density", 
                                                    units=unit_dict['unit_surface_density'])

    gas_fields["vr"] = _read_field(f"gasvy{file_n}.dat", 
                                   "vr", 
                                   units=unit_dict['unit_velocity'])

    gas_fields["vphi"] = _read_field(f"gasvx{file_n}.dat", 
                                     "vphi", 
                                     units=unit_dict['unit_velocity'])

    if is_3d:
        gas_fields["vtheta"] = _read_field(f"gasvz{file_n}.dat", 
                                            "vtheta", 
                                            units=unit_dict['unit_velocity'])
    
    return gas_fields


def _apply_frame_corrections(
    gas_fields: Dict[str, Field],
    variables: Dict[str, Any],
    redge: Quantity,
    is_3d: bool,
    unit_dict: Dict[str, Quantity],
) -> None:
    omegaframe = variables.get("OMEGAFRAME", 0.0)
    if omegaframe == 0.0 or gas_fields.get("vphi") is None:
        return
    
    logger.info(f"Applying corotating frame correction: OMEGAFRAME = {omegaframe}")
    
    vphi_data = gas_fields["vphi"].data
    r_centers = 0.5 * (redge[:-1] + redge[1:])
    r_centers = np.asarray(r_centers, dtype=float)
    vphi_mag = np.asarray(vphi_data.to(unit_dict['unit_velocity']).magnitude, dtype=float)
    
    if is_3d:
        r_grid = r_centers[:, np.newaxis, np.newaxis]
    else:
        r_grid = r_centers[:, np.newaxis]
    
    vphi_corrected_mag = vphi_mag + r_grid * omegaframe
    vphi_corrected = Quantity(vphi_corrected_mag, unit_dict['unit_velocity'])
    
    gas_fields["vphi"] = Field(
        data=vphi_corrected,
        quantity="vphi",
        axis_order=gas_fields["vphi"].axis_order,
    )
    
    logger.info(f"Corotating frame correction applied to vphi")


def _derive_temperature(
    directory: Path,
    file_n: int,
    gas_fields: Dict[str, Field],
    variables: Dict[str, Any],
    compile_options: Dict[str, Optional[bool]],
    is_3d: bool,
    unit_dict: Dict[str, Quantity],
    norm_units: str,
) -> None:
    gasenergy_field = gas_fields.get("gasenergy")
    if gasenergy_field is None:
        p = directory / f"gasenergy{file_n}.dat"
        if not p.exists():
            return
        arr = np.fromfile(p, dtype="float64")
        if is_3d:
            nrad = gas_fields["density"].data.shape[0]
            nsec = gas_fields["density"].data.shape[1]
            ncol = gas_fields["density"].data.shape[2]
            try:
                arr = arr.reshape(ncol, nrad, nsec)
            except Exception:
                if nrad > 0 and nsec > 0 and arr.size % (nrad * nsec) == 0:
                    inferred_ncol = arr.size // (nrad * nsec)
                    arr = arr.reshape(inferred_ncol, nrad, nsec)
                else:
                    raise
            arr = np.roll(arr, shift=int(nsec//2), axis=2)
            arr = np.transpose(arr, (1, 2, 0))
            axes = ("r", "phi", "theta")
        else:
            nrad = gas_fields["surface_density"].data.shape[0]
            nsec = gas_fields["surface_density"].data.shape[1]
            arr = arr.reshape(nrad, nsec)
            if (directory / "variables.par").exists():
                arr = np.roll(arr, shift=int(nsec//2), axis=1)
            axes = ("r", "phi")
        gasenergy_field = Field(
            data=Quantity(arr, 'dimensionless'),
            quantity="gasenergy",
            axis_order=axes,
        )
    
    is_isothermal_opt = compile_options.get('ISOTHERMAL', None)
    if is_isothermal_opt is None:
        write_energy = variables.get('WRITEENERGY', None)
        try:
            write_energy_val = int(write_energy) if write_energy is not None else None
        except Exception:
            write_energy_val = None
        is_isothermal = (write_energy_val == 0)
    else:
        is_isothermal = bool(is_isothermal_opt)
    
    temp_data = None
    
    if is_isothermal:
        c_s = Quantity(gasenergy_field.data.magnitude, unit_dict['unit_velocity'])
        rho = gas_fields.get("density")
        if rho is not None:
            gas_fields["pressure"] = Field(
                data=rho.data * c_s ** 2,
                quantity="pressure",
                axis_order=rho.axis_order,
            )

        temp_data = c_s.to(unit_dict['unit_velocity']).magnitude ** 2
        logger.debug(
            f"Isothermal: c_s range = {c_s.magnitude.min():.3e} - {c_s.magnitude.max():.3e}"
        )
        logger.debug(f"T_code range = {temp_data.min():.3e} - {temp_data.max():.3e}")
    else:
        gamma = variables.get("GAMMA")
        if is_3d:
            rho_field = gas_fields.get("density")
            if rho_field is None:
                return
            rho = rho_field.data.magnitude
            e = gasenergy_field.data.magnitude
            temp_data = (gamma - 1.0) * e / rho

            e_density = Quantity(gasenergy_field.data.magnitude, unit_dict['unit_density'] * unit_dict['unit_velocity'] ** 2)
            gas_fields["pressure"] = Field(
                data=(gamma - 1.0) * e_density,
                quantity="pressure",
                axis_order=gasenergy_field.axis_order,
            )
        else:
            logger.warning("Temperature calculation for 2D non-isothermal requires vertical puffing")

    if temp_data is not None and "MU" in variables:
        if norm_units == "code":
            code_mass_kg = (1.0 * solar_mass).to('kg').magnitude
            code_length_m = (1.0 * au).to('m').magnitude
            mu = float(variables["MU"])
            cutemp = mu * 8.0841643e-15 * code_mass_kg / code_length_m
        elif norm_units == "cgs":
            mu = float(variables["MU"])
            cutemp = (mu * units('m_H') / units('k_B')).to('K*s^2/cm^2').magnitude
        elif norm_units == "kms":
            mu = float(variables["MU"])
            cutemp = (mu * units('m_H') / units('k_B')).to('K*s^2/m^2').magnitude
        else:
            cutemp = 1.0

        temp_data_K = temp_data * cutemp
        temp_field = Field(
            data=Quantity(temp_data_K, 'K'),
            quantity="temperature",
            axis_order=gasenergy_field.axis_order,
        )
        gas_fields["temperature"] = temp_field


def _build_disk_parameters(
    variables: Dict[str, Any], unit_dict: Dict[str, Quantity], norm_units: str
) -> Dict[str, Any]:
    disk_parameters: Dict[str, Any] = {}
    disk_parameters["alphavisocity"] = variables.get("ALPHA") * units('dimensionless')
    disk_parameters["kinematicviscosity"] = variables.get("NU") * unit_dict['unit_viscosity']
    disk_parameters["aspectratio"] = variables.get("ASPECTRATIO") * units('dimensionless')
    disk_parameters["flaringindex"] = variables.get("FLARINGINDEX") * units('dimensionless')
    disk_parameters["sigma0"] = variables.get("SIGMA0") * unit_dict['unit_surface_density']
    disk_parameters["sigmaslope"] = variables.get("SIGMASLOPE") * units('dimensionless')
    disk_parameters["gamma"] = variables.get("GAMMA") * units('dimensionless')
    disk_parameters["cs"] = variables.get("CS") * unit_dict['unit_velocity']

    if norm_units in ("cgs", "kms"):
        disk_parameters["r0"] = 5.2 * units('au')
    else:
        disk_parameters["r0"] = 1.0 * units('code_length')
    
    return disk_parameters


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

    norm_units = (file_units or "code").lower()

    dim_info = _detect_dimensionality(directory, variables)
    
    if not dim_info.is_3d:
        redge, pedge, nrad, nsec = _load_edges_2d(directory, variables, unit_dict)
        mesh = Mesh.polar(r=Axis(edges=redge), phi=Axis(edges=pedge))
        coord_system = "polar"
        ncol = 1
    else:
        redge, pedge, tedge, nrad, nsec, ncol = _load_edges_3d(
            directory, variables, unit_dict, dim_info.ncol
        )
        mesh = Mesh.spherical(r=Axis(edges=redge), 
                              phi=Axis(edges=pedge), 
                              theta=Axis(edges=tedge))
        coord_system = "spherical"

    gas_fields = _load_fields(
        directory, file_n, dim_info.is_3d, nrad, nsec, ncol, unit_dict
    )
    
    _apply_frame_corrections(gas_fields, variables, redge, dim_info.is_3d, unit_dict)
    _derive_temperature(
        directory,
        file_n,
        gas_fields,
        variables,
        compile_options,
        dim_info.is_3d,
        unit_dict,
        norm_units,
    )
    disk_parameters = _build_disk_parameters(variables, unit_dict, norm_units)

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
