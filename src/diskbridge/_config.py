# db-keywords: gow17, config, model, io, serialization, paths
# db-role: canonical
# db-scope: package
# db-purpose: Package module for gow17, config, model, io.

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping, Union, Tuple
import toml

CONFIG_FILE = Path(__file__).parent / "config.toml"

_config_cache: MutableMapping[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def read_config(filename: Union[str, Path, None] = None) -> MutableMapping[str, Any]:
    """Read config file (TOML). If None, use the package's default config.toml."""
    if filename is None:
        filename = CONFIG_FILE
    return toml.load(filename)


def get_config() -> MutableMapping[str, Any]:
    """Get the cached configuration. Loads default config on first call."""
    global _config_cache
    if _config_cache is None:
        _config_cache = read_config(CONFIG_FILE)
    return _config_cache


def load_config(user_config_path: Union[str, Path]) -> None:
    """Load a user config file that overrides package defaults.
    
    Must be called before importing other diskbridge modules that use constants.
    User config only needs to specify values that differ from defaults.
    
    Parameters
    ----------
    user_config_path : str or Path
        Path to user's config.toml file
        
    Raises
    ------
    RuntimeError
        If config has already been loaded by another module
        
    Examples
    --------
    >>> import diskbridge
    >>> diskbridge.load_config('/path/to/my_config.toml')
    >>> from diskbridge.chemistry import run_chemistry  # Now uses merged config
    """
    global _config_cache
    if _config_cache is not None:
        raise RuntimeError(
            "Configuration already loaded. Call load_config() before importing "
            "other diskbridge modules (e.g., before 'from diskbridge import ...')."
        )
    
    base_config = read_config(CONFIG_FILE)
    user_config = read_config(user_config_path)
    _config_cache = _deep_merge(base_config, user_config)


def write_config(filename: Union[str, Path]) -> None:
    """Write a copy of the package default config.toml to the given path."""
    config = read_config()
    with open(filename, mode="w") as f:
        toml.dump(config, f)


# db-keywords: config
# db-role: canonical
def resolve_model_config(
    config_path: Tuple[str, ...],
    overrides: Union[dict, None] = None,
    config_file: Union[str, Path, None] = None,
) -> dict:
    """Resolve model configuration from TOML file with optional overrides.
    
    Parameters
    ----------
    config_path : tuple of str
        Path to the configuration section (e.g., ('chemistry', 'gow17'))
    overrides : dict, optional
        User-provided overrides to merge with defaults
    config_file : str or Path, optional
        Path to config file (default: package config.toml)
        
    Returns
    -------
    dict
        Merged configuration dictionary (still contains string values)
        
    Examples
    --------
    >>> cfg = resolve_model_config(("chemistry", "gow17"), overrides={"b_kms": 0.5})
    """
    if config_file is None:
        config = get_config()
    else:
        config = read_config(config_file)
    
    section = config
    for key in config_path:
        section = section.get(key, {})
    
    if overrides is None:
        return dict(section)
    
    return _deep_merge(dict(section), overrides)
