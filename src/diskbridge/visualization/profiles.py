"""Profile and averaging utilities for disk analysis.

This module provides functions for computing azimuthally-averaged profiles
and other 1D/2D reductions commonly used in disk analysis:

- Azimuthal averages (radial profiles)
- Vertical profiles at fixed radius
- Midplane slices
- Column density integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from diskbridge.model.model import Model
    from diskbridge.radmc3d.model import RadModel
    import yt

from diskbridge._logging import logger


def azimuthal_average(
    data: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    """Compute azimuthal average of 3D data.
    
    Parameters
    ----------
    data : ndarray
        3D data array
    axis : int, optional
        Axis to average over (default: 1, the phi axis in DiskBridge order)
        
    Returns
    -------
    ndarray
        2D array with phi dimension averaged out
        
    Notes
    -----
    For DiskBridge data in (r, phi, theta) order, use axis=1.
    For yt/RADMC-3D data in (r, theta, phi) order, use axis=2.
    """
    return np.mean(data, axis=axis)


def compute_radial_profile(
    model: "Model",
    field_name: str,
    theta_idx: Optional[int] = None,
    weight_field: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute azimuthally-averaged radial profile of a field.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance
    field_name : str
        Name of the field to profile (e.g., 'density', 'temperature')
    theta_idx : int, optional
        If provided, extract at this theta index (e.g., midplane).
        If None, average over all theta.
    weight_field : str, optional
        Field to use as weight for averaging (e.g., 'density' for mass-weighted)
        
    Returns
    -------
    r : ndarray
        Radial coordinates in AU
    profile : ndarray
        Azimuthally-averaged profile values
        
    Examples
    --------
    >>> r, rho_profile = compute_radial_profile(model, 'density')
    >>> plt.loglog(r, rho_profile)
    """
    mesh = model.mesh
    r = mesh.centers('r').to('au').magnitude
    theta = mesh.centers('theta').magnitude
    
    # Get field data
    if field_name not in model.gas:
        raise KeyError(f"Field '{field_name}' not found in model.gas")
    
    field = model.gas[field_name]
    data = field.data
    if hasattr(data, 'magnitude'):
        data = data.magnitude
    
    # Data is in (r, phi, theta) order
    # Average over phi (axis 1)
    data_phi_avg = np.mean(data, axis=1)  # (nr, ntheta)
    
    if theta_idx is not None:
        # Extract at specific theta
        profile = data_phi_avg[:, theta_idx]
    else:
        # Average over theta as well
        if weight_field is not None and weight_field in model.gas:
            # Weighted average
            weight = model.gas[weight_field].data
            if hasattr(weight, 'magnitude'):
                weight = weight.magnitude
            weight_avg = np.mean(weight, axis=1)
            profile = np.sum(data_phi_avg * weight_avg, axis=1) / np.sum(weight_avg, axis=1)
        else:
            profile = np.mean(data_phi_avg, axis=1)
    
    return r, profile


def compute_midplane_profile(
    model: "Model",
    field_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute azimuthally-averaged midplane radial profile.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance
    field_name : str
        Name of the field to profile
        
    Returns
    -------
    r : ndarray
        Radial coordinates in AU
    profile : ndarray
        Midplane profile values
    """
    theta = model.mesh.centers('theta').magnitude
    midplane_idx = np.argmin(np.abs(theta - np.pi / 2))
    return compute_radial_profile(model, field_name, theta_idx=midplane_idx)


def compute_vertical_profile(
    model: "Model",
    field_name: str,
    r_target: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute azimuthally-averaged vertical profile at a given radius.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance
    field_name : str
        Name of the field to profile
    r_target : float
        Target radius in AU
        
    Returns
    -------
    z : ndarray
        Height above midplane in AU
    profile : ndarray
        Vertical profile values
    """
    mesh = model.mesh
    r = mesh.centers('r').to('au').magnitude
    theta = mesh.centers('theta').magnitude
    
    # Find nearest radial index
    r_idx = np.argmin(np.abs(r - r_target))
    actual_r = r[r_idx]
    
    # Get field data
    if field_name not in model.gas:
        raise KeyError(f"Field '{field_name}' not found in model.gas")
    
    field = model.gas[field_name]
    data = field.data
    if hasattr(data, 'magnitude'):
        data = data.magnitude
    
    # Data is in (r, phi, theta) order
    # Average over phi
    data_phi_avg = np.mean(data, axis=1)  # (nr, ntheta)
    
    # Extract at this radius
    profile = data_phi_avg[r_idx, :]
    
    # Compute z = r * cos(theta)
    z = actual_r * np.cos(theta)
    
    return z, profile


def compute_column_density(
    model: "Model",
    field_name: str = 'density',
    integrate_axis: str = 'theta',
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute column density by integrating along a line of sight.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance
    field_name : str, optional
        Density field to integrate (default: 'density')
    integrate_axis : str, optional
        Axis to integrate along: 'theta' for face-on, 'phi' for edge-on
        
    Returns
    -------
    coords : ndarray
        Coordinate array (r for theta integration, r for phi integration)
    sigma : ndarray
        Column density in g/cm^2
        
    Notes
    -----
    For face-on view (integrate along theta), returns Sigma(r) azimuthally averaged.
    """
    mesh = model.mesh
    r = mesh.centers('r').to('cm').magnitude
    r_edges = mesh.edges('r').to('cm').magnitude
    theta = mesh.centers('theta').magnitude
    theta_edges = mesh.edges('theta').magnitude
    phi = mesh.centers('phi').magnitude
    
    # Get density field
    if field_name not in model.gas:
        raise KeyError(f"Field '{field_name}' not found in model.gas")
    
    rho = model.gas[field_name].data
    if hasattr(rho, 'magnitude'):
        rho = rho.to('g/cm**3').magnitude
    
    # Data is in (r, phi, theta) order
    nr, nphi, ntheta = rho.shape
    
    if integrate_axis == 'theta':
        # Face-on: integrate along theta (z direction)
        # Column density: Sigma = integral(rho * r * dtheta) 
        # This gives mass per unit area when viewed face-on
        dtheta = np.diff(theta_edges)
        
        sigma = np.zeros((nr, nphi))
        for i_r in range(nr):
            for i_phi in range(nphi):
                # Integrate rho * r * dtheta
                integrand = rho[i_r, i_phi, :] * r[i_r] * dtheta
                sigma[i_r, i_phi] = np.sum(integrand)
        
        # Azimuthal average
        sigma_avg = np.mean(sigma, axis=1)
        r_au = mesh.centers('r').to('au').magnitude
        
        return r_au, sigma_avg
    
    else:
        raise NotImplementedError(f"Integration axis '{integrate_axis}' not yet supported")


def compute_surface_density_from_3d(
    model: "Model",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute surface density from 3D density by vertical integration.
    
    This is specifically for puffed-up 3D models where we want to recover
    the surface density Sigma(R).
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance with 3D spherical mesh
        
    Returns
    -------
    R : ndarray
        Cylindrical radius in AU
    Sigma : ndarray
        Surface density in g/cm^2
    """
    return compute_column_density(model, 'density', 'theta')


class ProfilePlotter:
    """Helper class for creating profile plots from DiskBridge data.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance
    radmodel : RadModel, optional
        RADMC-3D model for additional fields
    """
    
    def __init__(
        self,
        model: "Model",
        radmodel: Optional["RadModel"] = None,
    ):
        self.model = model
        self.radmodel = radmodel
        self.mesh = model.mesh
        
        # Cache coordinate arrays
        self.r = self.mesh.centers('r').to('au').magnitude
        self.theta = self.mesh.centers('theta').magnitude
        self.phi = self.mesh.centers('phi').magnitude
        
        # Compute cylindrical coordinates
        R_grid, phi_grid, theta_grid = np.meshgrid(
            self.r, self.phi, self.theta, indexing='ij'
        )
        self.R_cyl = R_grid * np.sin(theta_grid)
        self.z_cyl = R_grid * np.cos(theta_grid)
    
    def radial_profile(
        self,
        field_name: str,
        at_midplane: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get radial profile of a field.
        
        Parameters
        ----------
        field_name : str
            Field name
        at_midplane : bool, optional
            If True, extract at midplane. Otherwise, average over theta.
            
        Returns
        -------
        r, profile : tuple of ndarray
        """
        if at_midplane:
            return compute_midplane_profile(self.model, field_name)
        else:
            return compute_radial_profile(self.model, field_name)
    
    def vertical_profile(
        self,
        field_name: str,
        r_target: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get vertical profile at a radius.
        
        Parameters
        ----------
        field_name : str
            Field name
        r_target : float
            Radius in AU
            
        Returns
        -------
        z, profile : tuple of ndarray
        """
        return compute_vertical_profile(self.model, field_name, r_target)
    
    def temperature_profile(
        self,
        at_midplane: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get temperature radial profile from RadModel.
        
        Parameters
        ----------
        at_midplane : bool, optional
            If True, extract at midplane.
            
        Returns
        -------
        r, T : tuple of ndarray
        """
        if self.radmodel is None or self.radmodel.temperature is None:
            raise ValueError("No temperature data available (radmodel not set or temperature not loaded)")
        
        temp = self.radmodel.temperature
        if hasattr(temp, 'magnitude'):
            temp = temp.magnitude
        
        # RadModel data is in (r, theta, phi) order
        # Average over phi
        temp_avg = np.mean(temp, axis=2)  # (nr, ntheta)
        
        if at_midplane:
            midplane_idx = np.argmin(np.abs(self.theta - np.pi / 2))
            profile = temp_avg[:, midplane_idx]
        else:
            profile = np.mean(temp_avg, axis=1)
        
        return self.r, profile
