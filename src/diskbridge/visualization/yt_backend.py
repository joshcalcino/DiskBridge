"""yt-based visualization backend for DiskBridge.

This module provides the Visualizer class that implements visualization
methods using yt-project. It supports:

- Slice plots (constant phi, theta, or r)
- Projection plots (face-on, edge-on views)
- Azimuthal slice plots (R-z plane, optionally half-disk only)
- Profile plots via yt's ProfilePlot

Key features:
- Automatic handling of spherical coordinates
- Support for half-disk (0 < R) azimuthal slices
- Integration with both Model and RadModel data sources
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union, List
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from diskbridge.model.core import Model
    from diskbridge.radmc3d.model import RadModel

from diskbridge._logging import logger

from .backend import VisualizationBackend
from .dataset import DiskBridgeDataset

# yt is a hard dependency; import directly
import yt


class Visualizer(VisualizationBackend):
    """Main visualization class for DiskBridge using yt-project.
    
    This class creates visualizations from DiskBridge Model and RadModel
    objects using yt's plotting infrastructure. It handles the complexity
    of spherical coordinates and data shape transformations.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance with mesh and gas data
    radmodel : RadModel, optional
        RADMC-3D model wrapper for temperature, UV field, etc.
    include_dust : bool, optional
        Include dust density fields (default: True)
        
    Attributes
    ----------
    ds : yt.Dataset
        The underlying yt dataset
    wrapper : DiskBridgeDataset
        The dataset wrapper object
        
    Examples
    --------
    >>> from diskbridge.visualization import Visualizer
    >>> 
    >>> viz = Visualizer(model, radmodel)
    >>> 
    >>> # Midplane slice
    >>> fig = viz.slice(('gas', 'density'), axis='phi')
    >>> 
    >>> # Face-on projection
    >>> fig = viz.projection(('gas', 'density'), axis='theta')
    >>> 
    >>> # Half-disk R-z slice
    >>> fig = viz.azimuthal_slice(('gas', 'temperature'), half_only=True)
    """
    
    def __init__(
        self,
        model: "Model",
        radmodel: Optional["RadModel"] = None,
        include_dust: bool = True,
    ):
        self.model = model
        self.radmodel = radmodel
        
        # Create the yt dataset wrapper
        self.wrapper = DiskBridgeDataset(
            model,
            radmodel,
            include_dust=include_dust,
        )
        self.ds = self.wrapper.ds
    
    def slice(
        self,
        field: Union[str, Tuple[str, str]],
        axis: str = "phi",
        coord: Optional[float] = None,
        output: Optional[Path] = None,
        width: Optional[Tuple[float, str]] = None,
        center: Optional[str] = "c",
        **kwargs,
    ) -> "Figure":
        """Create a slice plot through the data.
        
        Parameters
        ----------
        field : str or tuple
            Field to plot, e.g., 'density' or ('gas', 'density')
        axis : str, optional
            Axis normal to slice: 'r', 'theta', or 'phi' (default: 'phi')
        coord : float, optional
            Coordinate value for the slice. For phi, this is in radians.
            If None, uses center of domain.
        output : Path, optional
            If provided, save plot to this path
        width : tuple, optional
            Plot width as (value, unit), e.g., (200, 'au')
        center : str, optional
            Center specification (default: 'c' for domain center)
        **kwargs
            Additional arguments passed to SlicePlot
            
        Returns
        -------
        matplotlib.figure.Figure
            The plot figure
            
        Notes
        -----
        For spherical coordinates:
        - phi slice: Shows R-z plane (poloidal view)
        - theta slice: Shows x-y plane at constant colatitude
        - r slice: Shows theta-phi plane at constant radius
        """
        # Normalize field to tuple
        if isinstance(field, str):
            field = ('gas', field)
        
        # Create slice plot
        if coord is not None:
            # yt SlicePlot with coordinate
            slc = yt.SlicePlot(
                self.ds, 
                axis, 
                field,
                center=center,
                **kwargs
            )
        else:
            slc = yt.SlicePlot(
                self.ds,
                axis,
                field,
                center=center,
                **kwargs
            )
        
        # Apply width if specified
        if width is not None:
            slc.set_width(width)
        
        # Save if output specified
        if output is not None:
            slc.save(str(output))
            logger.info(f"Saved slice plot to {output}")
        
        # Get the matplotlib figure
        fig = slc.plots[field].figure
        
        return fig
    
    def projection(
        self,
        field: Union[str, Tuple[str, str]],
        axis: str = "theta",
        weight_field: Optional[Union[str, Tuple[str, str]]] = None,
        output: Optional[Path] = None,
        width: Optional[Tuple[float, str]] = None,
        **kwargs,
    ) -> "Figure":
        """Create a projection plot (integrated along line of sight).
        
        Parameters
        ----------
        field : str or tuple
            Field to project
        axis : str, optional
            Axis to project along: 'theta' for face-on (default)
        weight_field : str or tuple, optional
            Field to weight by (e.g., 'density' for mass-weighted)
        output : Path, optional
            If provided, save plot to this path
        width : tuple, optional
            Plot width as (value, unit)
        **kwargs
            Additional arguments passed to ProjectionPlot
            
        Returns
        -------
        matplotlib.figure.Figure
            The plot figure
        """
        # Normalize fields
        if isinstance(field, str):
            field = ('gas', field)
        if isinstance(weight_field, str):
            weight_field = ('gas', weight_field)
        
        prj = yt.ProjectionPlot(
            self.ds,
            axis,
            field,
            weight_field=weight_field,
            **kwargs
        )
        
        if width is not None:
            prj.set_width(width)
        
        if output is not None:
            prj.save(str(output))
            logger.info(f"Saved projection plot to {output}")
        
        fig = prj.plots[field].figure
        return fig
    
    def azimuthal_slice(
        self,
        field: Union[str, Tuple[str, str]],
        phi: float = 0.0,
        half_only: bool = True,
        output: Optional[Path] = None,
        r_max: Optional[Tuple[float, str]] = None,
        **kwargs,
    ) -> "Figure":
        """Create an azimuthal slice showing R-z plane.
        
        This creates a slice at constant phi, showing the poloidal (R-z)
        structure of the disk. By default, only shows R > 0 (half-disk).
        
        Parameters
        ----------
        field : str or tuple
            Field to plot
        phi : float, optional
            Azimuthal angle in radians (default: 0)
        half_only : bool, optional
            If True, only show R > 0 (default: True)
        output : Path, optional
            If provided, save plot to this path
        r_max : tuple, optional
            Maximum radius as (value, unit), e.g., (200, 'au')
        **kwargs
            Additional arguments passed to SlicePlot
            
        Returns
        -------
        matplotlib.figure.Figure
            The plot figure
        """
        # Normalize field
        if isinstance(field, str):
            field = ('gas', field)
        
        # Create phi slice
        slc = yt.SlicePlot(
            self.ds,
            'phi',
            field,
            **kwargs
        )
        
        # For half-disk view, we need to modify the matplotlib axis after rendering
        # First, render the plot to create the figure
        slc._setup_plots()
        
        if half_only:
            # Get domain extent in AU
            r_right = self.ds.domain_right_edge[0].to('AU').value
            if r_max is not None:
                r_val, r_unit = r_max
                from unyt import unyt_quantity
                r_limit = unyt_quantity(r_val, r_unit).to('AU').value
                r_right = min(r_right, r_limit)
            
            # Access the matplotlib axes and set xlim
            # The plot object contains the axes
            ax = slc.plots[field].axes
            # Get current limits to preserve y
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()
            
            # Set x to positive only (R > 0)
            # Need to convert from cm to AU for display
            au_in_cm = 1.496e13
            ax.set_xlim(0, r_right * au_in_cm)
        
        if output is not None:
            slc.save(str(output))
            logger.info(f"Saved azimuthal slice to {output}")
        
        fig = slc.plots[field].figure
        return fig
    
    def midplane_slice(
        self,
        field: Union[str, Tuple[str, str]],
        output: Optional[Path] = None,
        **kwargs,
    ) -> "Figure":
        """Create a midplane (theta = pi/2) slice.
        
        Shows the disk in the x-y plane at the midplane.
        
        Parameters
        ----------
        field : str or tuple
            Field to plot
        output : Path, optional
            If provided, save plot to this path
        **kwargs
            Additional arguments
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        # Normalize field
        if isinstance(field, str):
            field = ('gas', field)
        
        # theta slice at midplane
        slc = yt.SlicePlot(
            self.ds,
            'theta',
            field,
            center=[0, np.pi/2, 0],  # center at midplane
            **kwargs
        )
        
        if output is not None:
            slc.save(str(output))
        
        return slc.plots[field].figure
    
    def radial_profile_plot(
        self,
        field: Union[str, Tuple[str, str]],
        data_source: Optional[str] = None,
        weight_field: Optional[Union[str, Tuple[str, str]]] = None,
        n_bins: int = 64,
        output: Optional[Path] = None,
        **kwargs,
    ) -> "Figure":
        """Create a 1D radial profile plot using yt.
        
        Parameters
        ----------
        field : str or tuple
            Field to profile
        data_source : optional
            yt data object to use. If None, uses all_data()
        weight_field : str or tuple, optional
            Field to weight by
        n_bins : int, optional
            Number of radial bins (default: 64)
        output : Path, optional
            If provided, save plot to this path
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        # Normalize fields
        if isinstance(field, str):
            field = ('gas', field)
        if isinstance(weight_field, str):
            weight_field = ('gas', weight_field)
        
        # Get data source
        if data_source is None:
            src = self.ds.all_data()
        else:
            src = data_source
        
        # Create profile plot
        plot = yt.ProfilePlot(
            src,
            ('index', 'spherical_r'),
            field,
            weight_field=weight_field,
            n_bins=n_bins,
            **kwargs
        )
        
        # Set x-axis to AU
        plot.set_unit(('index', 'spherical_r'), 'AU')
        
        if output is not None:
            plot.save(str(output))
        
        return plot.plots[field].figure
    
    def set_colormap(
        self,
        plot,
        field: Union[str, Tuple[str, str]],
        cmap: str = 'viridis',
    ):
        """Set colormap for a plot.
        
        Parameters
        ----------
        plot : yt plot object
            The plot to modify
        field : str or tuple
            Field to set colormap for
        cmap : str
            Matplotlib colormap name
        """
        if isinstance(field, str):
            field = ('gas', field)
        plot.set_cmap(field, cmap)
    
    def set_zlim(
        self,
        plot,
        field: Union[str, Tuple[str, str]],
        zmin: float,
        zmax: float,
    ):
        """Set color scale limits for a plot.
        
        Parameters
        ----------
        plot : yt plot object
            The plot to modify
        field : str or tuple
            Field to set limits for
        zmin, zmax : float
            Min and max values for color scale
        """
        if isinstance(field, str):
            field = ('gas', field)
        plot.set_zlim(field, zmin, zmax)


# Convenience functions for quick plotting

def quick_slice(
    model: "Model",
    field: str = 'density',
    axis: str = 'phi',
    radmodel: Optional["RadModel"] = None,
    output: Optional[Path] = None,
    **kwargs,
) -> "Figure":
    """Quick slice plot from a Model.
    
    Parameters
    ----------
    model : Model
        DiskBridge model
    field : str
        Field name (default: 'density')
    axis : str
        Slice axis (default: 'phi' for poloidal view)
    radmodel : RadModel, optional
        RADMC-3D model for temperature etc.
    output : Path, optional
        Output file path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    viz = Visualizer(model, radmodel)
    return viz.slice(field, axis=axis, output=output, **kwargs)


def quick_projection(
    model: "Model",
    field: str = 'density',
    axis: str = 'theta',
    radmodel: Optional["RadModel"] = None,
    output: Optional[Path] = None,
    **kwargs,
) -> "Figure":
    """Quick projection plot from a Model.
    
    Parameters
    ----------
    model : Model
        DiskBridge model
    field : str
        Field name (default: 'density')
    axis : str
        Projection axis (default: 'theta' for face-on)
    radmodel : RadModel, optional
        RADMC-3D model
    output : Path, optional
        Output file path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    viz = Visualizer(model, radmodel)
    return viz.projection(field, axis=axis, output=output, **kwargs)
