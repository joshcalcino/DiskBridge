"""Wavelength grid utilities for RADMC-3D.

This module provides utilities for creating, validating, and writing
wavelength grids for RADMC-3D computations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

from diskbridge._units import Quantity
from diskbridge._logging import logger


def validate_wavelength_array(
    wavelengths: np.ndarray,
    name: str = "wavelengths",
) -> None:
    """Validate wavelength array for RADMC-3D use.
    
    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array to validate
    name : str, optional
        Name for error messages
        
    Raises
    ------
    ValueError
        If wavelengths are invalid
    """
    if wavelengths.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if wavelengths.size < 2:
        raise ValueError(f"{name} must contain at least 2 wavelengths")
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all(wavelengths[1:] > wavelengths[:-1]):
        raise ValueError(f"{name} must be strictly increasing")


def check_wavelength_range(
    wavelengths_um: np.ndarray,
    global_min: Quantity,
    global_max: Quantity,
    grid_name: str = "wavelength grid",
) -> None:
    """Check that wavelengths are within global bounds.
    
    Parameters
    ----------
    wavelengths_um : ndarray
        Wavelengths in microns
    global_min : Quantity
        Minimum allowed wavelength
    global_max : Quantity
        Maximum allowed wavelength
    grid_name : str, optional
        Name for error messages
        
    Raises
    ------
    ValueError
        If wavelengths are outside global bounds
    """
    lam_min = wavelengths_um[0]
    lam_max = wavelengths_um[-1]
    
    global_min_um = global_min.to('micron').magnitude
    global_max_um = global_max.to('micron').magnitude
    
    if lam_min < global_min_um or lam_max > global_max_um:
        raise ValueError(
            f"{grid_name} is outside the global wavelength grid: "
            f"requested=[{lam_min:.6g},{lam_max:.6g}] micron, "
            f"global=[{global_min_um:.6g},{global_max_um:.6g}] micron"
        )


def build_wavelength_grid(
    wmin: Quantity,
    wmax: Quantity,
    n_wavelengths: int,
    spacing: str = 'linear',
) -> np.ndarray:
    """Build wavelength grid for RADMC-3D.
    
    Parameters
    ----------
    wmin : Quantity
        Minimum wavelength
    wmax : Quantity
        Maximum wavelength
    n_wavelengths : int
        Number of wavelengths
    spacing : str, optional
        'linear' or 'log' spacing (default: 'linear')
        
    Returns
    -------
    ndarray
        Wavelength array in microns
        
    Examples
    --------
    >>> from diskbridge._units import units
    >>> wl = build_wavelength_grid(
    ...     wmin=units('0.09 micron'),
    ...     wmax=units('0.20 micron'),
    ...     n_wavelengths=10,
    ... )
    """
    wmin_um = wmin.to('micron').magnitude
    wmax_um = wmax.to('micron').magnitude
    
    if spacing == 'linear':
        wavelengths = np.linspace(wmin_um, wmax_um, n_wavelengths)
    elif spacing == 'log':
        wavelengths = np.logspace(
            np.log10(wmin_um),
            np.log10(wmax_um),
            n_wavelengths,
        )
    else:
        raise ValueError(f"Unknown spacing: {spacing}")
    
    validate_wavelength_array(wavelengths, "wavelength grid")
    return wavelengths


def write_wavelength_file(
    filepath: Path,
    wavelengths_um: np.ndarray,
    file_format: str = 'mcmono',
) -> None:
    """Write wavelength file for RADMC-3D.
    
    Parameters
    ----------
    filepath : Path
        Output file path
    wavelengths_um : ndarray
        Wavelengths in microns
    file_format : str, optional
        Format: 'mcmono' or 'standard' (default: 'mcmono')
        
    Notes
    -----
    mcmono format: 
        n_wavelengths
        wavelength_1
        wavelength_2
        ...
        
    standard format (wavelength_micron.inp):
        n_wavelengths
        wavelength_1 wavelength_2 ...
    """
    validate_wavelength_array(wavelengths_um, "wavelengths")
    
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        if file_format == 'mcmono':
            f.write(f'{wavelengths_um.size}\n')
            for lam in wavelengths_um:
                f.write(f'{lam:.6f}\n')
        elif file_format == 'standard':
            f.write(f'{wavelengths_um.size}\n')
            for lam in wavelengths_um:
                f.write(f'{lam:.6e}\n')
        else:
            raise ValueError(f"Unknown file_format: {file_format}")
    
    logger.debug(f"Wrote {filepath} with {wavelengths_um.size} wavelengths")


def read_wavelength_file(filepath: Path) -> np.ndarray:
    """Read wavelength file from RADMC-3D.
    
    Parameters
    ----------
    filepath : Path
        Path to wavelength file
        
    Returns
    -------
    ndarray
        Wavelengths in microns
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Wavelength file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        n_wavelengths = int(f.readline().strip())
        wavelengths = []
        
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            wavelengths.extend(map(float, line.split()))
            if len(wavelengths) >= n_wavelengths:
                break
    
    wavelengths = np.array(wavelengths[:n_wavelengths])
    validate_wavelength_array(wavelengths, "wavelength file")
    
    logger.debug(f"Read {wavelengths.size} wavelengths from {filepath}")
    return wavelengths


def build_mcmono_wavelengths(
    wavelength_source: str,
    wavelength_file: Optional[Path],
    uv_min_um: float,
    uv_max_um: float,
    n_wavelengths: Optional[int],
    n_uv_enforce: int,
    spacing: str = 'log',
    provided_wavelengths: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build mcmono wavelength grid with optional UV enforcement.
    
    Parameters
    ----------
    wavelength_source : str
        Source: 'uv' (UV range only) or 'external' (from file)
    wavelength_file : Path or None
        Path to wavelength_micron.inp (required if source='external')
    uv_min_um : float
        UV range minimum in microns
    uv_max_um : float
        UV range maximum in microns
    n_wavelengths : int or None
        Number of wavelengths (None = use all from file)
    n_uv_enforce : int
        Number of UV wavelengths to enforce (0 = no enforcement)
    spacing : str, optional
        'linear' or 'log' (default: 'log')
    provided_wavelengths : ndarray or None
        Pre-provided wavelength array (overrides all other options)
        
    Returns
    -------
    ndarray
        Wavelength grid in microns
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if provided_wavelengths is not None:
        wavelengths = np.asarray(provided_wavelengths, dtype=float)
        validate_wavelength_array(wavelengths, "provided wavelengths")
        return wavelengths
    
    if wavelength_source == 'uv':
        if n_wavelengths is None:
            raise ValueError("n_wavelengths required for source='uv'")
        if n_wavelengths < 2:
            raise ValueError("n_wavelengths must be >= 2")
        
        wavelengths = build_wavelength_grid(
            wmin=Quantity(uv_min_um, 'micron'),
            wmax=Quantity(uv_max_um, 'micron'),
            n_wavelengths=n_wavelengths,
            spacing=spacing,
        )
        
    elif wavelength_source == 'external':
        if wavelength_file is None:
            raise ValueError("wavelength_file required for source='external'")
        if not wavelength_file.exists():
            raise FileNotFoundError(f"Wavelength file not found: {wavelength_file}")
        
        wav_global = read_wavelength_file(wavelength_file)
        
        if n_wavelengths is None:
            wavelengths = wav_global
        else:
            if n_wavelengths < 2:
                raise ValueError("n_wavelengths must be >= 2")
            lam_min = float(np.min(wav_global))
            lam_max = float(np.max(wav_global))
            
            wavelengths = build_wavelength_grid(
                wmin=Quantity(lam_min, 'micron'),
                wmax=Quantity(lam_max, 'micron'),
                n_wavelengths=n_wavelengths,
                spacing=spacing,
            )
    else:
        raise ValueError(f"wavelength_source must be 'uv' or 'external', got '{wavelength_source}'")
    
    # Enforce UV coverage if requested
    if n_uv_enforce > 0:
        if n_uv_enforce == 1:
            raise ValueError("n_uv_enforce must be >= 2 or 0 (to disable)")
        uv_grid = np.linspace(uv_min_um, uv_max_um, n_uv_enforce)
        wavelengths = np.unique(np.concatenate([wavelengths, uv_grid]))
    
    validate_wavelength_array(wavelengths, "mcmono wavelength grid")
    return wavelengths
