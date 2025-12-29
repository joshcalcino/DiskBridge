from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file contents.
    
    Parameters
    ----------
    path : Path
        File to hash
        
    Returns
    -------
    str
        Hex digest of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_network_npz(path: str | Path, network: dict, meta: dict) -> None:
    """Save compiled network to NPZ format with metadata.
    
    Parameters
    ----------
    path : str or Path
        Output .npz file path
    network : dict
        Compiled network arrays from compile_network()
    meta : dict
        Metadata dictionary containing:
        - umist_path: str
        - umist_sha256: str
        - species_init or species_list: list[str]
        - reduce_mode: str
        - allowed_elements: list[str] (optional)
        - max_species: int (optional)
        - diskbridge_version: str (optional)
        
    Raises
    ------
    ValueError
        If network is missing required keys
    """
    path = Path(path)
    
    required_keys = ['species', 'charge', 'ir1', 'ir2', 'ip1', 'ip2', 'ip3', 'ip4',
                     'rtype', 'alpha', 'beta', 'gamma', 'S']
    for key in required_keys:
        if key not in network:
            raise ValueError(f"Network missing required key: {key}")
    
    meta_json = json.dumps(meta, indent=2)
    
    save_dict = network.copy()
    save_dict['_metadata'] = np.array([meta_json], dtype='U')
    
    np.savez_compressed(path, **save_dict)


def load_network_npz(path: str | Path) -> tuple[dict, dict]:
    """Load compiled network from NPZ file.
    
    Parameters
    ----------
    path : str or Path
        Path to .npz file
        
    Returns
    -------
    tuple[dict, dict]
        (network_arrays, metadata)
        
    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If file is corrupt or missing required data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Network cache file not found: {path}")
    
    npz = np.load(path, allow_pickle=False)
    
    network = {}
    meta = {}
    
    for key in npz.files:
        if key == '_metadata':
            meta_json = str(npz[key][0])
            meta = json.loads(meta_json)
        else:
            network[key] = npz[key]
    
    required_keys = ['species', 'charge', 'ir1', 'ir2', 'ip1', 'ip2', 'ip3', 'ip4',
                     'rtype', 'alpha', 'beta', 'gamma', 'S']
    for key in required_keys:
        if key not in network:
            raise ValueError(f"Cached network missing required key: {key}")
    
    return network, meta
