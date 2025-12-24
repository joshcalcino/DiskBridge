"""Cache validation for RADMC-3D computations.

This module provides utilities for checking whether cached RADMC-3D outputs
are still valid based on parameter snapshots and signatures.
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional

from diskbridge._logging import logger
from diskbridge._params import Params
from .utils import _read_params_snapshot, _params_signature


def check_cache_validity(
    output_file: Path,
    current_params_path: Path,
    saved_params_path: Path,
    param_keys: Sequence[str],
    allow_overrides: bool = True,
) -> bool:
    """Check if cached output is valid based on parameter signatures.
    
    Parameters
    ----------
    output_file : Path
        Path to the cached output file to check
    current_params_path : Path
        Path to current params.txt file
    saved_params_path : Path
        Path to saved params.txt from when cache was created
    param_keys : Sequence[str]
        Parameter names to include in signature comparison
    allow_overrides : bool, optional
        If False, only use cache if no parameter overrides were used
        
    Returns
    -------
    bool
        True if cache is valid and can be used
        
    Notes
    -----
    A cache is considered valid if:
    1. The output file exists
    2. Both current and saved params files exist
    3. Parameter signatures match for all keys in param_keys
    """
    if not output_file.exists():
        return False
    
    if not current_params_path.exists() or not saved_params_path.exists():
        logger.debug("Cache check failed: params files missing")
        return False
    
    try:
        params_current = _read_params_snapshot(current_params_path)
        params_saved = _read_params_snapshot(saved_params_path)
        
        sig_current = _params_signature(params_current, tuple(param_keys))
        sig_saved = _params_signature(params_saved, tuple(param_keys))
        
        if sig_current == sig_saved:
            logger.debug(f"Cache valid: {output_file.name}")
            return True
        else:
            logger.debug(f"Cache invalid: parameter signatures differ")
            return False
            
    except Exception as e:
        logger.debug(f"Cache check failed: {e}")
        return False


def find_cached_output(
    output_dir: Path,
    candidate_files: Sequence[str | Path],
) -> Optional[Path]:
    """Find first existing output file from a list of candidates.
    
    Parameters
    ----------
    output_dir : Path
        Directory to search for cached outputs
    candidate_files : Sequence[str or Path]
        Filenames or relative paths to check (in priority order)
        
    Returns
    -------
    Path or None
        Path to first existing file, or None if none found
        
    Examples
    --------
    >>> output_dir = Path('radmc3d_outputs')
    >>> cached = find_cached_output(
    ...     output_dir,
    ...     ['dust_temperature.bdat', 'dust_temperature.dat']
    ... )
    """
    for candidate in candidate_files:
        filepath = output_dir / candidate
        if filepath.exists():
            return filepath
    return None


def should_use_cache(
    output_dir: Path,
    candidate_files: Sequence[str | Path],
    current_params_path: Path,
    param_keys: Sequence[str],
    force: bool = False,
    **override_flags,
) -> tuple[bool, Optional[Path]]:
    """Determine if cache should be used and return cached file if valid.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing cached outputs
    candidate_files : Sequence[str or Path]
        Output filenames to check (in priority order)
    current_params_path : Path
        Path to current params.txt
    param_keys : Sequence[str]
        Parameter names for signature comparison
    force : bool, optional
        If True, always recompute (ignore cache)
    **override_flags
        Named flags indicating parameter overrides (e.g., use_params_nphot=True)
        
    Returns
    -------
    use_cache : bool
        True if cache should be used
    cached_file : Path or None
        Path to valid cached file, or None if cache shouldn't be used
        
    Examples
    --------
    >>> use_cache, cached = should_use_cache(
    ...     output_dir=Path('radmc3d_outputs'),
    ...     candidate_files=['mean_intensity.bout', 'mean_intensity.out'],
    ...     current_params_path=Path('params.txt'),
    ...     param_keys=['nphot_mono', 'uv_min', 'uv_max'],
    ...     force=False,
    ...     use_params_nphot=True,
    ...     use_params_uv_min=True,
    ... )
    """
    if force:
        return False, None
    
    cached_file = find_cached_output(output_dir, candidate_files)
    if cached_file is None:
        return False, None
    
    # Check if any parameter overrides were used
    has_overrides = not all(override_flags.values())
    if has_overrides:
        logger.info(
            f"Existing output found at {cached_file} but parameter "
            "overrides were used; will recompute."
        )
        return False, cached_file
    
    saved_params_path = output_dir / 'params.txt'
    is_valid = check_cache_validity(
        output_file=cached_file,
        current_params_path=current_params_path,
        saved_params_path=saved_params_path,
        param_keys=param_keys,
    )
    
    if is_valid:
        logger.info(f"Using cached output: {cached_file}")
        return True, cached_file
    else:
        logger.info(
            f"Existing output found at {cached_file} but parameters "
            "have changed; will recompute."
        )
        return False, cached_file
