from __future__ import annotations

from pathlib import Path
from typing import Union
import configparser


DEFAULT_PARAMS_FILE = Path(__file__).parent / "params.txt"


class ParamsHelper:
    """Wrapper around ConfigParser with convenience methods for common conversions."""
    
    def __init__(self, parser: configparser.ConfigParser):
        self._parser = parser
    
    def __getitem__(self, section: str) -> configparser.SectionProxy:
        """Access sections directly: params['simulation']['data_directory']"""
        return self._parser[section]
    
    def has_section(self, section: str) -> bool:
        return self._parser.has_section(section)
    
    def has_option(self, section: str, option: str) -> bool:
        return self._parser.has_option(section, option)
    
    def get(self, section: str, option: str, fallback=None):
        value = self._parser.get(section, option, fallback=fallback)
        # Strip inline comments for string values as well
        if isinstance(value, str):
            value = value.split('#')[0].strip()
        return value
    
    def getint(self, section: str, option: str, fallback=None) -> int:
        if fallback is not None and not self._parser.has_option(section, option):
            return fallback
        # Handle scientific notation like 1e7
        raw = self._parser.get(section, option)
        # Strip inline comments
        raw = raw.split('#')[0].strip()
        return int(float(raw))
    
    def getfloat(self, section: str, option: str, fallback=None) -> float:
        if fallback is not None and not self._parser.has_option(section, option):
            return fallback
        # Handle scientific notation like 1e7
        raw = self._parser.get(section, option)
        # Strip inline comments
        raw = raw.split('#')[0].strip()
        return float(raw)
    
    def getbool(self, section: str, option: str, fallback=None) -> bool:
        if fallback is not None and not self._parser.has_option(section, option):
            return fallback
        raw = self._parser.get(section, option)
        # Strip inline comments
        raw = raw.split('#')[0].strip()
        return raw.lower() in ("1", "true", "t", "yes", "y")
    
    # Convenience getters with unit conversions
    def get_dust_size_m(self, which: str) -> float:
        """Get amin or amax from [dust_sizes] in meters (file has microns)."""
        microns = self.getfloat('dust_sizes', which, fallback=0.0)
        return microns * 1.0e-6


def read_params(filename: Union[str, Path, None] = None) -> ParamsHelper:
    """Read params.txt file and automatically update the global diskbridge.params.
    
    Access parameters via section/key:
        params['simulation']['data_directory']
        params.getint('simulation', 'output_number')
        params.getfloat('wavelengths', 'lambda_min_micron')
    
    Args:
        filename: Path to params file. If None, uses package default.
    
    Returns:
        ParamsHelper wrapper around ConfigParser
    """
    global params
    
    if filename is None:
        filename = DEFAULT_PARAMS_FILE
    
    path = Path(filename)
    parser = configparser.ConfigParser()
    
    if path.exists():
        with open(path, "r") as f:
            parser.read_file(f)
    
    # If a global ParamsHelper already exists, update its underlying parser in-place
    # so that all imports of "params" see the new values.
    existing = globals().get("params", None)
    if isinstance(existing, ParamsHelper):
        base_parser = existing._parser
        for section in parser.sections():
            if not base_parser.has_section(section):
                base_parser.add_section(section)
            for key, value in parser.items(section):
                base_parser.set(section, key, value)
        params = existing
        return params
    
    # First-time initialization: create a new ParamsHelper
    new_params = ParamsHelper(parser)
    params = new_params
    return params


# Global parameter instance, created on import
params = read_params()
