from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping, Union
import toml

CONFIG_FILE = Path(__file__).parent / "config.toml"


def read_config(filename: Union[str, Path, None] = None) -> MutableMapping[str, Any]:
    """Read config file (TOML). If None, use the package's default config.toml."""
    if filename is None:
        filename = CONFIG_FILE
    return toml.load(filename)


def write_config(filename: Union[str, Path]) -> None:
    """Write a copy of the package default config.toml to the given path."""
    config = read_config()
    with open(filename, mode="w") as f:
        toml.dump(config, f)
