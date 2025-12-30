"""Coordinate transformation and grid helpers.

This module removes duplicate meshgrid/coordinate transformation boilerplate
and ensures consistent axis ordering across the codebase.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from diskbridge._units import Quantity


def spherical_grids(
    r: Quantity,
    phi: Quantity,
    theta: Quantity,
) -> Tuple[Quantity, Quantity, Quantity]:
    """Create 3D meshgrids for spherical coordinates.
    
    Parameters
    ----------
    r : Quantity
        Radial coordinate array [length]
    phi : Quantity
        Azimuthal coordinate array [angle]
    theta : Quantity
        Polar coordinate array [angle]
        
    Returns
    -------
    r_grid : Quantity
        3D radial grid with shape (nr, nphi, ntheta) [length]
    phi_grid : Quantity
        3D azimuthal grid with shape (nr, nphi, ntheta) [angle]
    theta_grid : Quantity
        3D polar grid with shape (nr, nphi, ntheta) [angle]
        
    Notes
    -----
    Uses indexing='ij' for consistent axis ordering.
    Numpy meshgrid works directly with Quantity objects.
    """
    r_grid, phi_grid, theta_grid = np.meshgrid(r, phi, theta, indexing="ij")
    
    return r_grid, phi_grid, theta_grid


def cylindrical_from_spherical(
    r_grid: Quantity,
    theta_grid: Quantity,
) -> Tuple[Quantity, Quantity]:
    """Convert spherical to cylindrical coordinates.
    
    Parameters
    ----------
    r_grid : Quantity
        Spherical radial coordinate [length]
    theta_grid : Quantity
        Polar angle [angle]
        
    Returns
    -------
    R_cyl : Quantity
        Cylindrical radius R = r * sin(theta) [length]
    z : Quantity
        Height above midplane z = r * cos(theta) [length]
        
    Notes
    -----
    Standard spherical to cylindrical transformation:
    - R_cyl = r * sin(theta)
    - z = r * cos(theta)
    - phi is the same in both systems
    """
    R_cyl = r_grid * np.sin(theta_grid)
    z = r_grid * np.cos(theta_grid)
    
    return R_cyl, z


def cartesian_from_spherical(
    r_grid: Quantity,
    phi_grid: Quantity,
    theta_grid: Quantity,
) -> Tuple[Quantity, Quantity, Quantity]:
    """Convert spherical to Cartesian coordinates.
    
    Parameters
    ----------
    r_grid : Quantity
        Spherical radial coordinate [length]
    phi_grid : Quantity
        Azimuthal angle [angle]
    theta_grid : Quantity
        Polar angle [angle]
        
    Returns
    -------
    x : Quantity
        Cartesian x coordinate [length]
    y : Quantity
        Cartesian y coordinate [length]
    z : Quantity
        Cartesian z coordinate [length]
    """
    x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z = r_grid * np.cos(theta_grid)
    
    return x, y, z
