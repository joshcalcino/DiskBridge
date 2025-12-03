from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union, List, Optional, Dict, get_origin, get_args, get_type_hints

from ._units import Quantity, units


# Default params file shipped with the package
DEFAULT_PARAMS_FILE = Path(__file__).parent / "params.txt"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Typed parameter container 
# ---------------------------------------------------------------------------

@dataclass
class Params:
    # simulation / photons
    nbcores: int
    nphot_thermal: int
    nphot_scat: int
    nphot_mono: int

    # global rescaling (dimensionless)
    length_scale: float
    mass_scale: float

    # wavelengths
    n_lambda: int
    lambda_min: Quantity
    lambda_max: Quantity

    # map
    nbpixels: int
    mapsize: Quantity
    distance: Quantity
    inclination: Union[float, List[float]]
    posangle: float

    # dust_rt
    scat_mode: int

    # dust_sizes (can be scalar or list for multi-component dust)
    amin: Union[Quantity, List[Quantity]]
    amax: Union[Quantity, List[Quantity]]
    pindex: Union[float, List[float]]
    dust_to_gas_ratio: Union[float, List[float]]
    nbins: Union[int, List[int]]
    grain_density: Union[Quantity, List[Quantity]]

    # opacity (can be scalar or list for multi-component dust)
    species: Union[str, List[str]]
    opacity_dir: str

    # gas_rt
    gasspecies: str
    iline: int
    abundance: float
    width: Quantity
    nline: int
    turbvel: Quantity
    photodissociation: bool
    freezeout: bool
    photodesorption: bool
    uv_min: Quantity
    uv_max: Quantity
    uv_n_wavelengths: int

    external_uv: bool
    external_uv_chi: float
    co_self_shielding: bool
    co_self_shielding_method: str

    # star
    rstar: Quantity
    teff: Quantity
    mstar: Quantity

    mdot: float
    accretion_fill_factor: float

    # radmc
    secondorder: bool
    noscat: bool

    # naming
    prepend_name: str
    append_name: str



# ---------------------------------------------------------------------------
# Simple "key = value" parser
# ---------------------------------------------------------------------------

def _parse_param_file(path: Path) -> Dict[str, str]:
    """Return {key: raw_value_string} from a flat key=value file."""
    result: Dict[str, str] = {}
    if not path.exists():
        return result

    with path.open("r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()

    return result


# ---------------------------------------------------------------------------
# Parameter to unit mapping
# ---------------------------------------------------------------------------

# Map parameter names to their expected units
PARAM_UNITS = {
    'lambda_min': 'micron',
    'lambda_max': 'micron',
    'mapsize': 'au',
    'distance': 'pc',
    'amin': 'micron',
    'amax': 'micron',
    'grain_density': 'g/cm^3',
    'width': 'km/s',
    'turbvel': 'm/s',
    'uv_min': 'nm',
    'uv_max': 'nm',
    'rstar': 'solar_radius',
    'teff': 'K',
    'mstar': 'solar_mass',
}

# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

def _parse_bool(raw: str) -> bool:
    raw = raw.strip().lower()
    return raw in ("1", "true", "t", "yes", "y")


def _parse_scalar(raw: str, target_type, param_name: str = ''):
    raw = raw.strip()
    if target_type is bool:
        return _parse_bool(raw)
    if target_type is int:
        return int(float(raw))  # handles 1e6, etc.
    if target_type is float:
        return float(raw)
    if target_type is str:
        return raw
    # Handle Quantity type
    if target_type is Quantity or (hasattr(target_type, '__origin__') and target_type.__origin__ is Quantity):
        # Get the unit for this parameter
        unit_str = PARAM_UNITS.get(param_name, 'dimensionless')
        value = float(raw)
        return Quantity(value, unit_str)
    return raw


def _parse_value(raw: str, target_type, param_name: str = ''):
    raw = raw.split("#", 1)[0].strip()
    origin = get_origin(target_type)

    # Union[T, List[T]] support (e.g. Union[Quantity, List[Quantity]])
    if origin is Union:
        args = get_args(target_type)
        list_type = next((t for t in args if get_origin(t) is list or t is list), None)
        if list_type:
            # Check if raw value is a list
            if "," in raw or "[" in raw:
                # Parse as list
                if get_origin(list_type) is list:
                    inner = get_args(list_type)[0]
                else:
                    # Fallback for older Python or edge cases
                    scalar_type = next((t for t in args if t is not list_type), None)
                    inner = scalar_type if scalar_type else str
                
                vals = raw.strip("[]")
                if not vals:
                    return []
                return [_parse_scalar(part.strip(), inner, param_name) for part in vals.split(",")]
            else:
                # Parse as scalar - find the non-list type in the Union
                scalar_type = next((t for t in args if get_origin(t) is not list and t is not list), None)
                if scalar_type:
                    return _parse_scalar(raw, scalar_type, param_name)
                # Fallback
                return raw

    # List[T]
    if origin is list:
        inner = get_args(target_type)[0]
        vals = raw.strip("[]")
        if not vals:
            return []
        return [_parse_scalar(part.strip(), inner, param_name) for part in vals.split(",")]

    return _parse_scalar(raw, target_type, param_name)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

params: Params  # global


def read_params(filename: Optional[Union[str, Path]] = None) -> Params:
    """Load defaults from DEFAULT_PARAMS_FILE, overlay user overrides, and build Params."""
    global params

    # 1. Read defaults (the *real* default source)
    values = _parse_param_file(DEFAULT_PARAMS_FILE)

    # 2. Overlay user file
    if filename is not None:
        user_vals = _parse_param_file(Path(filename))
        values.update(user_vals)

    # 3. Build typed Params instance
    # Use get_type_hints to resolve string annotations to actual types
    type_hints = get_type_hints(Params)
    
    kwargs = {}
    for field in fields(Params):
        key = field.name
        if key not in values:
            raise KeyError(
                f"Missing required parameter '{key}' in params.txt (no default provided)."
            )
        raw = values[key]
        # Use the resolved type from type_hints, not field.type which may be a string
        field_type = type_hints[key]
        parsed = _parse_value(raw, field_type, param_name=key)
        kwargs[key] = parsed

    params = Params(**kwargs)
    opacity_path = Path(params.opacity_dir)
    if not opacity_path.is_absolute():
        params.opacity_dir = str((REPO_ROOT / opacity_path).resolve())
    return params


def canonicalize_dust_params(params_obj: Params) -> dict:
    """Canonicalize dust parameters to per-component lists with broadcasting.
    
    Rules:
    - Convert all dust params to lists
    - Determine ncomp from the longest list
    - Broadcast scalars (length-1 lists) to all components
    - Validate all lists have length 1 or ncomp
    
    Returns:
        dict with keys:
            - 'ncomp': int, number of dust components
            - 'amin': list of floats
            - 'amax': list of floats
            - 'pindex': list of floats
            - 'dust_to_gas_ratio': list of floats
            - 'nbins': list of ints
            - 'grain_density': list of floats
            - 'species': list of strings
    """
    # Dust parameters to canonicalize
    dust_param_names = [
        'amin', 'amax', 'pindex', 'dust_to_gas_ratio',
        'nbins', 'grain_density', 'species'
    ]
    
    # Convert each to list
    def to_list(x):
        return x if isinstance(x, list) else [x]
    
    lists = {k: to_list(getattr(params_obj, k)) for k in dust_param_names}
    
    # Determine number of components
    ncomp = max(len(v) for v in lists.values())
    
    # Broadcast scalars and validate lengths
    canonical = {'ncomp': ncomp}
    for k, v in lists.items():
        if len(v) == 1 and ncomp > 1:
            # Broadcast scalar to all components
            canonical[k] = v * ncomp
        elif len(v) == ncomp:
            # Already correct length
            canonical[k] = v
        else:
            raise ValueError(
                f"Dust parameter '{k}' has length {len(v)}, but other parameters "
                f"imply {ncomp} components. Each parameter must have length 1 or {ncomp}."
            )
    
    return canonical


# Params will be initialized in __init__.py after add_units() is called
params = None  # type: ignore
