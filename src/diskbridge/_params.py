from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union, List, Optional, Dict, get_origin, get_args, get_type_hints


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
    n_thermal: float
    n_scat: float

    # wavelengths
    n_lambda: int
    lambda_min_micron: float
    lambda_max_micron: float

    # map
    nbpixels: int
    size_au: float
    distance_pc: float
    inclination: Union[float, List[float]]
    posangle: float

    # dust_rt
    scat_mode: int

    # dust_sizes
    amin: float
    amax: float
    pindex: float
    dust_to_gas_ratio: float
    nbins: int
    grain_density: float

    # opacity
    species: str
    opacity_dir: str

    # gas_rt
    gasspecies: str
    iline: int
    abundance: float
    widthkms: float
    nline: int
    turbvel_ms: float
    photodissociation: bool
    freezeout: bool
    photodesorption: bool
    uv_min_nm: float
    uv_max_nm: float
    uv_n_wavelengths: int

    # star
    rstar_rsun: float
    teff_K: float
    mstar_msun: float

    mdot: float
    accretion_fill_factor: float

    # radmc
    secondorder: bool



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
# Type casting
# ---------------------------------------------------------------------------

def _parse_bool(raw: str) -> bool:
    raw = raw.strip().lower()
    return raw in ("1", "true", "t", "yes", "y")


def _parse_scalar(raw: str, target_type):
    raw = raw.strip()
    if target_type is bool:
        return _parse_bool(raw)
    if target_type is int:
        return int(float(raw))  # handles 1e6, etc.
    if target_type is float:
        return float(raw)
    if target_type is str:
        return raw
    return raw


def _parse_value(raw: str, target_type):
    raw = raw.split("#", 1)[0].strip()
    origin = get_origin(target_type)

    # Union[float, List[float]] support
    if origin is Union:
        args = get_args(target_type)
        list_type = next((t for t in args if get_origin(t) is list or t is list), None)
        if list_type:
            if "," in raw or "[" in raw:
                inner = float
                vals = raw.strip("[]")
                if not vals:
                    return []
                return [inner(x.strip()) for x in vals.split(",")]
            return float(raw)

    # List[T]
    if origin is list:
        inner = get_args(target_type)[0]
        vals = raw.strip("[]")
        if not vals:
            return []
        return [_parse_scalar(part.strip(), inner) for part in vals.split(",")]

    return _parse_scalar(raw, target_type)


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
        parsed = _parse_value(raw, field_type)
        kwargs[key] = parsed

    params = Params(**kwargs)
    opacity_path = Path(params.opacity_dir)
    if not opacity_path.is_absolute():
        params.opacity_dir = str((REPO_ROOT / opacity_path).resolve())
    return params


# Initialize once on import using only the defaults
params = read_params(None)
