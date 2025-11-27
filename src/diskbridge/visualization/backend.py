"""Abstract base class for visualization backends.

This module defines the interface that all visualization backends must implement.
Currently the main implementation is YTBackend which uses yt-project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class VisualizationBackend(ABC):
    """Abstract base class for visualization backends.
    
    All visualization backends must implement the slice() and projection()
    methods. Additional methods like azimuthal_slice() and radial_profile()
    are optional but recommended.
    """
    
    @abstractmethod
    def slice(
        self,
        field: Union[str, Tuple[str, str]],
        axis: str = "phi",
        coord: Optional[float] = None,
        output: Optional[Path] = None,
        **kwargs,
    ) -> "Figure":
        """Create a slice plot through the data.
        
        Parameters
        ----------
        field : str or tuple
            Field to plot
        axis : str
            Axis normal to the slice
        coord : float, optional
            Coordinate value for the slice
        output : Path, optional
            If provided, save plot to this path
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        ...

    @abstractmethod
    def projection(
        self,
        field: Union[str, Tuple[str, str]],
        axis: str = "theta",
        weight_field: Optional[Union[str, Tuple[str, str]]] = None,
        output: Optional[Path] = None,
        **kwargs,
    ) -> "Figure":
        """Create a projection plot (integrated along line of sight).
        
        Parameters
        ----------
        field : str or tuple
            Field to project
        axis : str
            Axis to project along
        weight_field : str or tuple, optional
            Field to weight by
        output : Path, optional
            If provided, save plot to this path
            
        Returns
        -------
        matplotlib.figure.Figure
        """
        ...
    
    def azimuthal_slice(
        self,
        field: Union[str, Tuple[str, str]],
        phi: float = 0.0,
        half_only: bool = True,
        output: Optional[Path] = None,
        **kwargs,
    ) -> "Figure":
        """Create an azimuthal slice showing R-z plane.
        
        Optional method - not all backends may implement this.
        """
        raise NotImplementedError("azimuthal_slice not implemented for this backend")
    
    def radial_profile_plot(
        self,
        field: Union[str, Tuple[str, str]],
        weight_field: Optional[Union[str, Tuple[str, str]]] = None,
        output: Optional[Path] = None,
        **kwargs,
    ) -> "Figure":
        """Create a 1D radial profile plot.
        
        Optional method - not all backends may implement this.
        """
        raise NotImplementedError("radial_profile_plot not implemented for this backend")
