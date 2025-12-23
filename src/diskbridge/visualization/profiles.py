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
import matplotlib.tri as mtri

if TYPE_CHECKING:
    from diskbridge.model.model import Model
    from diskbridge.radmc3d.model import RadModel
    from diskbridge.model.field import Field
    import yt

from diskbridge._logging import logger
from diskbridge.model.field import Field
import matplotlib.pyplot as plt

from diskbridge.model.profiles import (
    compute_cell_volumes as _compute_cell_volumes,
    compute_volume_weighted_mean_radial_profile as _compute_volume_weighted_mean_radial_profile,
    find_r_split as _find_r_split,
)


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


def compute_cell_volumes(model: "Model") -> np.ndarray:
    """Compute cell volumes for a spherical mesh.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance with spherical mesh
        
    Returns
    -------
    volumes : ndarray
        Cell volumes in cm^3, shape (nr, nphi, ntheta) matching DiskBridge order
    """
    return _compute_cell_volumes(model)


def compute_volume_weighted_mean_radial_profile(
    model: "Model",
    field_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute volume-weighted mean radial profile of a field.
    
    For each radial bin, computes the weighted mean over all (phi, theta) cells,
    using cell volume as the weight.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance
    field_name : str
        Name of the field to profile (e.g., 'chi', 'temperature')
        
    Returns
    -------
    r : ndarray
        Radial coordinates in AU
    profile : ndarray
        Volume-weighted mean profile values
    """
    return _compute_volume_weighted_mean_radial_profile(model, field_name)


def find_r_split(
    r_au: np.ndarray,
    r_edges_au: np.ndarray,
    chi_profile: np.ndarray,
    T_profile: np.ndarray,
    tol_chi: float = 0.01,
    tol_T: float = 0.01,
    window_fraction: float = 0.1,
    r_clip_min_au: float = 1.0,
) -> Tuple[float, dict]:
    """Find the split radius where chi and T profiles reach asymptotic values.
    
    Scans inward from the outer boundary to find the innermost radius where
    both chi and T are within the specified tolerance of their asymptotic
    (outer-window mean) values.
    
    Parameters
    ----------
    r_au : ndarray
        Radial cell centers in AU
    r_edges_au : ndarray
        Radial cell edges in AU
    chi_profile : ndarray
        Volume-weighted mean chi profile
    T_profile : ndarray
        Volume-weighted mean temperature profile
    tol_chi : float
        Fractional tolerance for chi asymptote (default: 0.01 = 1%)
    tol_T : float
        Fractional tolerance for T asymptote (default: 0.01 = 1%)
    window_fraction : float
        Fraction of radial domain for asymptote window (default: 0.1 = 10%)
    r_clip_min_au : float
        Minimum allowed R_split in AU (raises error if R_split < this)
        
    Returns
    -------
    r_split_au : float
        Split radius in AU (snapped to cell edge)
    info : dict
        Diagnostic info with keys:
        - 'chi_asymptote': asymptotic chi value
        - 'T_asymptote': asymptotic T value
        - 'r_split_cell_idx': index of the split cell
        - 'window_r_min': inner edge of asymptote window
        
    Raises
    ------
    ValueError
        If R_split < r_clip_min or no valid split point found
    """
    return _find_r_split(
        r_au=r_au,
        r_edges_au=r_edges_au,
        chi_profile=chi_profile,
        T_profile=T_profile,
        tol_chi=tol_chi,
        tol_T=tol_T,
        window_fraction=window_fraction,
        r_clip_min_au=r_clip_min_au,
    )


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
        raise ValueError(f"Unsupported integrate_axis: {integrate_axis}")


def register_small_dust_density_field(
    model: "Model",
    amax_um: float = 1.0,
    field_name: str = "dust_density_small",
) -> Optional["Field"]:
    dust = getattr(model, "dust", None)
    if dust is None or not getattr(dust, "bins", None):
        logger.warning("No dust distribution available on model; cannot build small-dust field")
        return None

    small_total = None
    sample_field: Optional[Field] = None

    for _, dust_bin in dust.bins.items():
        size_um = dust_bin.size.to("um").magnitude
        if size_um > amax_um:
            continue
        global_idx = dust_bin.bin_index
        bin_field = dust._compute_bin_density(global_idx)
        if small_total is None:
            small_total = bin_field.data.copy()
            sample_field = bin_field
        else:
            small_total = small_total + bin_field.data

    if small_total is None or sample_field is None:
        logger.warning("No dust bins found with size < %.3g um", amax_um)
        return None

    field = Field(
        data=small_total,
        quantity=sample_field.quantity,
        axis_order=sample_field.axis_order,
    )
    model.gas_register(field_name, field)
    logger.info("Registered small-dust field '%s' with amax=%.3g um", field_name, amax_um)
    return field


def compute_small_dust_midplane_profile(
    model: "Model",
    amax_um: float = 1.0,
    field_name: str = "dust_density_small",
) -> Tuple[np.ndarray, np.ndarray]:
    if field_name not in model.gas:
        reg_field = register_small_dust_density_field(model, amax_um=amax_um, field_name=field_name)
        if reg_field is None:
            raise ValueError("Could not construct small-dust field; check dust configuration")
    return compute_midplane_profile(model, field_name)


def plot_small_dust_midplane_profile(
    model: "Model",
    amax_um: float = 1.0,
    output: Union[str, "Path"] = "small_dust_radial_profile.png",
    field_name: str = "dust_density_small",
):
    r_au, profile = compute_small_dust_midplane_profile(model, amax_um=amax_um, field_name=field_name)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_au, profile)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r [au]")
    ax.set_ylabel("rho_dust(<%.3g um) [g/cm^3]" % amax_um)
    ax.set_title("Midplane small-grain dust density")
    fig.tight_layout()
    fig.savefig(str(output), dpi=200)
    plt.close(fig)
    return fig


def plot_phi_avg_rz_slice(
    model: "Model",
    field: Union[str, "Field"],
    output: Union[str, "Path"] = "rz_slice.png",
    *,
    x_axis: str = "R",
    y_axis: str = "z",
    log10: bool = True,
    log10_dyn_range_dex: Optional[float] = 6.0,
    cmap: str = "viridis",
    xscale: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    vline_x: Optional[float] = None,
):
    mesh = model.mesh
    r = mesh.centers("r").to("au").magnitude
    theta = mesh.centers("theta").magnitude
    r_edges = mesh.edges("r").to("au").magnitude
    theta_edges = mesh.edges("theta").to("radian").magnitude

    if isinstance(field, str):
        f = model.gas[field]
        data = f.data
    else:
        data = field.data

    if hasattr(data, "to"):
        data_mag = data.to_base_units().magnitude
    elif hasattr(data, "magnitude"):
        data_mag = data.magnitude
    else:
        data_mag = np.asarray(data)

    data_phi_avg = np.mean(data_mag, axis=1)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing="ij")

    x_axis_norm = x_axis.strip().lower()
    y_axis_norm = y_axis.strip().lower()

    vmin = None
    vmax = None
    if log10:
        tiny = np.finfo(np.float64).tiny
        z_plot = np.log10(np.maximum(data_phi_avg, tiny))
        vmax = float(np.nanmax(z_plot))
        if (log10_dyn_range_dex is not None) and np.isfinite(vmax):
            vmin = vmax - float(log10_dyn_range_dex)
    else:
        z_plot = data_phi_avg

    fig, ax = plt.subplots(figsize=(6, 4))

    if x_axis_norm == "r" and y_axis_norm in ("z/r", "z_over_r", "z_over_r_sph"):
        x1 = r_edges
        y1 = np.cos(theta_edges)
        c = z_plot.T
        if y1[0] > y1[-1]:
            y1 = y1[::-1]
            c = c[::-1, :]
        pc = ax.pcolormesh(x1, y1, c, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("r [au]")
        ax.set_ylabel("z/r")
    else:
        if x_axis_norm == "r":
            x_pts = r_grid
            xlabel = "r [au]"
        else:
            x_pts = r_grid * np.sin(theta_grid)
            xlabel = "R [au]"

        if y_axis_norm in ("z/r", "z_over_r", "z_over_r_sph"):
            y_pts = np.cos(theta_grid)
            ylabel = "z/r"
        else:
            y_pts = r_grid * np.cos(theta_grid)
            ylabel = "z [au]"

        tri = mtri.Triangulation(x_pts.ravel(), y_pts.ravel())
        pc = ax.tripcolor(tri, z_plot.ravel(), shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if xscale is not None:
        ax.set_xscale(xscale)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    if vline_x is not None:
        ax.axvline(float(vline_x), color="k", linestyle="--", linewidth=1.0)

    fig.colorbar(pc, ax=ax)
    fig.tight_layout()
    fig.savefig(str(output), dpi=200)
    plt.close(fig)
    return fig


def plot_small_dust_rz_slice(
    model: "Model",
    amax_um: float = 1.0,
    output: Union[str, "Path"] = "small_dust_rz_slice.png",
    field_name: str = "dust_density_small",
):
    """Plot a phi-averaged R-z slice of small-grain dust density.

    The slice is constructed in cylindrical coordinates (R, z) by
    averaging the small-dust density over azimuth and mapping the
    spherical (r, theta) grid to (R = r sin(theta), z = r cos(theta)).
    """
    if field_name not in model.gas:
        reg_field = register_small_dust_density_field(model, amax_um=amax_um, field_name=field_name)
        if reg_field is None:
            raise ValueError("Could not construct small-dust field; check dust configuration")

    mesh = model.mesh
    r = mesh.centers("r").to("au").magnitude
    theta = mesh.centers("theta").magnitude

    field = model.gas[field_name]
    data = field.data
    if hasattr(data, "to"):
        data = data.to("g/cm**3").magnitude
    elif hasattr(data, "magnitude"):
        data = data.magnitude

    # data is (r, phi, theta); average over phi for an axisymmetric slice
    rho_phi_avg = np.mean(data, axis=1)  # (nr, ntheta)

    # Prepare spherical axes (1D): r has length nr, theta has length ntheta
    r_au = r  # already in au, shape (nr,)
    theta_rad = theta  # shape (ntheta,)

    z_val = np.log10(rho_phi_avg + 1e-99)  # (nr, ntheta)

    fig, ax = plt.subplots(figsize=(6, 4))
    # Use 1D axes: X length N (nr), Y length M (ntheta), C shape (M, N) -> transpose
    pc = ax.pcolormesh(r_au, theta_rad, z_val.T, shading="auto")
    ax.set_xscale("log")
    ax.set_xlabel("r [au]")
    ax.set_ylabel("theta [rad]")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label("log10 rho_dust(<%.3g um) [g/cm^3]" % amax_um)
    ax.set_title("Phi-averaged small-grain dust density (R-z slice)")
    fig.tight_layout()
    fig.savefig(str(output), dpi=200)
    plt.close(fig)
    return fig


def plot_small_dust_midplane_map(
    model: "Model",
    amax_um: float = 1.0,
    output: Union[str, "Path"] = "small_dust_midplane_map.png",
    field_name: str = "dust_density_small",
):
    if field_name not in model.gas:
        reg_field = register_small_dust_density_field(model, amax_um=amax_um, field_name=field_name)
        if reg_field is None:
            raise ValueError("Could not construct small-dust field; check dust configuration")

    mesh = model.mesh
    r = mesh.centers('r').to('au').magnitude
    phi = mesh.centers('phi').magnitude
    theta = mesh.centers('theta').magnitude

    theta_mid_idx = int(np.argmin(np.abs(theta - np.pi / 2.0)))

    field = model.gas[field_name]
    data = field.data
    if hasattr(data, 'to'):
        data = data.to('g/cm**3').magnitude
    elif hasattr(data, 'magnitude'):
        data = data.magnitude

    # data is (r, phi, theta); extract midplane slice -> (nr, nphi)
    slice_mid = data[:, :, theta_mid_idx]

    r_edges = mesh.edges('r').to('au').magnitude
    phi_edges = mesh.edges('phi').magnitude

    z = np.log10(slice_mid + 1e-99)

    fig, ax = plt.subplots(figsize=(6, 4))
    pc = ax.pcolormesh(r_edges, phi_edges, z.T, shading='auto')
    ax.set_xscale('log')
    ax.set_xlabel('r [au]')
    ax.set_ylabel('phi [rad]')
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label("log10 rho_dust(<%.3g um) [g/cm^3]" % amax_um)
    ax.set_title('Midplane small-grain dust density')
    fig.tight_layout()
    fig.savefig(str(output), dpi=200)
    plt.close(fig)
    return fig


def plot_midplane_xy_map(
    model: "Model",
    field: Union[str, "Field"],
    output: Union[str, "Path"] = "midplane_xy_map.png",
    *,
    log10: bool = True,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    mesh = model.mesh
    r_edges = mesh.edges("r").to("au").magnitude
    phi_edges = mesh.edges("phi").magnitude
    theta = mesh.centers("theta").magnitude

    theta_mid_idx = int(np.argmin(np.abs(theta - np.pi / 2.0)))

    if isinstance(field, str):
        f = model.gas[field]
        data = f.data
        title = field
    else:
        data = field.data
        title = field.quantity

    if hasattr(data, "to"):
        data_mag = data.to_base_units().magnitude
    elif hasattr(data, "magnitude"):
        data_mag = data.magnitude
    else:
        data_mag = np.asarray(data)

    slice_mid = data_mag[:, :, theta_mid_idx]

    rr, pp = np.meshgrid(r_edges, phi_edges, indexing="ij")
    x = rr * np.cos(pp)
    y = rr * np.sin(pp)

    tiny = np.finfo(np.float64).tiny
    c_plot = np.log10(np.maximum(slice_mid, tiny)) if log10 else slice_mid

    fig, ax = plt.subplots(figsize=(6, 6))
    pc = ax.pcolormesh(x, y, c_plot, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x [au]")
    ax.set_ylabel("y [au]")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    fig.colorbar(pc, ax=ax)
    fig.tight_layout()
    fig.savefig(str(output), dpi=200)
    plt.close(fig)
    return fig


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
    
    def chi_profile(
        self,
        at_midplane: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get chi (UV field) radial profile from RadModel.
        
        Parameters
        ----------
        at_midplane : bool, optional
            If True, extract at midplane.
            
        Returns
        -------
        r, chi : tuple of ndarray
        """
        if self.radmodel is None or self.radmodel.chi is None:
            raise ValueError("No chi data available (radmodel not set or chi not loaded)")
        
        chi = self.radmodel.chi
        if hasattr(chi, 'magnitude'):
            chi = chi.magnitude
        
        # RadModel data is in (r, theta, phi) order
        # Average over phi
        chi_avg = np.mean(chi, axis=2)  # (nr, ntheta)
        
        if at_midplane:
            midplane_idx = np.argmin(np.abs(self.theta - np.pi / 2))
            profile = chi_avg[:, midplane_idx]
        else:
            profile = np.mean(chi_avg, axis=1)
        
        return self.r, profile
    
    def get_rz_slice(
        self,
        field_source: str,
        field_name: str = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get azimuthally-averaged R-z slice of a field.
        
        Parameters
        ----------
        field_source : str
            'temperature', 'chi', or a field name from model.gas
        field_name : str, optional
            Field name if field_source is 'gas'
            
        Returns
        -------
        R : ndarray
            Cylindrical radius in AU, shape (nr, ntheta)
        z : ndarray
            Height above midplane in AU, shape (nr, ntheta)
        data : ndarray
            Azimuthally-averaged data, shape (nr, ntheta)
        """
        # Get data based on source
        if field_source == 'temperature':
            if self.radmodel is None or self.radmodel.temperature is None:
                raise ValueError("No temperature data available")
            data = self.radmodel.temperature
            if hasattr(data, 'magnitude'):
                data = data.magnitude
            # RadModel data is (r, theta, phi) - average over phi
            data_avg = np.mean(data, axis=2)
        elif field_source == 'chi':
            if self.radmodel is None or self.radmodel.chi is None:
                raise ValueError("No chi data available")
            data = self.radmodel.chi
            if hasattr(data, 'magnitude'):
                data = data.magnitude
            # RadModel data is (r, theta, phi) - average over phi
            data_avg = np.mean(data, axis=2)
        else:
            # Assume it's a gas field from model
            fname = field_name if field_name else field_source
            if fname not in self.model.gas:
                raise KeyError(f"Field '{fname}' not found in model.gas")
            data = self.model.gas[fname].data
            if hasattr(data, 'magnitude'):
                data = data.magnitude
            # Model data is (r, phi, theta) - average over phi, then transpose
            data_avg = np.mean(data, axis=1)  # (nr, ntheta)
        
        # Compute R and z grids
        r_grid, theta_grid = np.meshgrid(self.r, self.theta, indexing='ij')
        R = r_grid * np.sin(theta_grid)
        z = r_grid * np.cos(theta_grid)
        
        return R, z, data_avg
