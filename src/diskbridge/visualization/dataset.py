"""DiskBridge dataset wrapper for yt-project integration.

This module provides the DiskBridgeDataset class that creates a yt-compatible
dataset from DiskBridge Model and RadModel objects. It handles:

- Coordinate system mapping (spherical)
- Data shape transformations between DiskBridge and yt conventions
- Field registration from both Model and RadModel sources
- Unit conversions from Pint to unyt

Data Shape Conventions:
-----------------------
- DiskBridge Model: (nr, nphi, ntheta) with axis_order ('r', 'phi', 'theta')
- RADMC-3D/RadData: (nr, ntheta, nphi) - RADMC-3D native order  
- yt spherical: (nr, ntheta, nphi) - same as RADMC-3D

When loading data, Model fields are transposed to match yt/RADMC-3D order.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple
import numpy as np

if TYPE_CHECKING:
    from diskbridge.model.model import Model
    from diskbridge.radmc3d.model import RadModel

from diskbridge._logging import logger

# yt is a hard dependency; import directly
import yt
from yt.loaders import load_uniform_grid

from .units import pint_to_unyt_cgs, check_unyt_available


class DiskBridgeDataset:
    """Wrapper that creates a yt dataset from DiskBridge Model and RadModel.
    
    This class combines data from a DiskBridge Model (hydrodynamic simulation)
    and optionally a RadModel (RADMC-3D outputs) into a single yt dataset
    for visualization.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance with mesh and gas data
    radmodel : RadModel, optional
        RADMC-3D model wrapper with temperature, UV field, etc.
    include_dust : bool, optional
        Include dust density fields (default: True)
    length_unit : str, optional
        Length unit for the dataset (default: 'au')
        
    Attributes
    ----------
    ds : yt.Dataset
        The yt dataset object
    model : Model
        Reference to the DiskBridge model
    radmodel : RadModel or None
        Reference to the RadModel if provided
        
    Examples
    --------
    >>> import diskbridge
    >>> from diskbridge.visualization import DiskBridgeDataset
    >>> 
    >>> model = diskbridge.load_model('data/', file_n=10)
    >>> model.puff_up_model(n=64)
    >>> 
    >>> # Create dataset from model only
    >>> dbd = DiskBridgeDataset(model)
    >>> 
    >>> # Or with RadModel for temperature etc.
    >>> radmodel = diskbridge.radmc3d.RadModel(model)
    >>> radmodel.read_temperature()
    >>> dbd = DiskBridgeDataset(model, radmodel)
    >>> 
    >>> # Access the yt dataset
    >>> import yt
    >>> slc = yt.SlicePlot(dbd.ds, 'phi', ('gas', 'density'))
    """
    
    def __init__(
        self,
        model: "Model",
        radmodel: Optional["RadModel"] = None,
        include_dust: bool = True,
        length_unit: str = 'au',
    ):
        check_unyt_available()
        
        self.model = model
        self.radmodel = radmodel
        self.include_dust = include_dust
        self.length_unit = length_unit
        
        # Validate model has required data
        if model.mesh is None:
            raise ValueError("Model has no mesh defined")
        if model.mesh.coord_system != 'spherical':
            raise ValueError(
                f"Only spherical coordinates supported, got {model.mesh.coord_system}"
            )
        
        # Build the yt dataset
        self.ds = self._create_dataset()
    
    def _get_mesh_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
        """Extract mesh information in yt order (r, theta, phi).
        
        Returns
        -------
        r_edges : ndarray
            Radial cell edges in cm
        theta_edges : ndarray  
            Colatitude cell edges in radians
        phi_edges : ndarray
            Azimuthal cell edges in radians
        shape : tuple
            Grid shape (nr, ntheta, nphi)
        """
        mesh = self.model.mesh
        
        # Get edges and convert to CGS
        r_edges = mesh.edges('r').to('cm').magnitude
        theta_edges = mesh.edges('theta').magnitude  # radians
        phi_edges = mesh.edges('phi').magnitude  # radians
        
        # Get shape in yt order: (nr, ntheta, nphi)
        nr = len(mesh.centers('r'))
        ntheta = len(mesh.centers('theta'))
        nphi = len(mesh.centers('phi'))
        shape = (nr, ntheta, nphi)
        
        return r_edges, theta_edges, phi_edges, shape
    
    def _transpose_model_to_yt(self, data: np.ndarray) -> np.ndarray:
        """Transpose Model data from (r, phi, theta) to yt order (r, theta, phi).
        
        Parameters
        ----------
        data : ndarray
            Data array in DiskBridge order (nr, nphi, ntheta)
            
        Returns
        -------
        ndarray
            Data array in yt order (nr, ntheta, nphi)
        """
        # (nr, nphi, ntheta) -> (nr, ntheta, nphi)
        return np.transpose(data, (0, 2, 1))
    
    def _collect_fields(self) -> Dict[str, np.ndarray]:
        """Collect all field data from Model and RadModel.
        
        Returns
        -------
        dict
            Dictionary mapping field tuples to data arrays in yt order
        """
        fields = {}
        
        # Get gas fields from Model
        if hasattr(self.model, 'gas') and self.model.gas is not None:
            for field_name in self.model.gas.keys():
                field = self.model.gas[field_name]
                data = field.data
                
                # Convert Pint to numpy array in CGS
                if hasattr(data, 'magnitude'):
                    data_cgs = data.to_base_units().magnitude
                else:
                    data_cgs = np.asarray(data)
                
                # Transpose from (r, phi, theta) to (r, theta, phi)
                data_yt = self._transpose_model_to_yt(data_cgs)
                
                # Map field names to yt conventions
                yt_field_name = self._map_field_name(field_name)
                fields[yt_field_name] = data_yt
                
                logger.debug(
                    f"Added field {yt_field_name}: shape {data_yt.shape}, "
                    f"range [{np.min(data_yt):.3e}, {np.max(data_yt):.3e}]"
                )
        
        # Get RadModel fields (already in yt order)
        if self.radmodel is not None:
            # Temperature
            if self.radmodel.temperature is not None:
                temp = self.radmodel.temperature
                if hasattr(temp, 'magnitude'):
                    temp_data = temp.magnitude
                else:
                    temp_data = np.asarray(temp)
                fields[('gas', 'temperature')] = temp_data
                logger.debug(f"Added temperature: range [{np.min(temp_data):.1f}, {np.max(temp_data):.1f}] K")
            
            # UV field (chi)
            if self.radmodel.chi is not None:
                chi = self.radmodel.chi
                if hasattr(chi, 'magnitude'):
                    chi_data = chi.magnitude
                else:
                    chi_data = np.asarray(chi)
                fields[('radmc', 'uv_field')] = chi_data
                logger.debug(f"Added UV field: range [{np.min(chi_data):.2e}, {np.max(chi_data):.2e}]")
            
            # Number density (nH)
            if self.radmodel.nH is not None:
                nH = self.radmodel.nH
                if hasattr(nH, 'magnitude'):
                    nH_data = nH.to('cm**-3').magnitude
                else:
                    nH_data = np.asarray(nH)
                fields[('gas', 'H_nuclei_density')] = nH_data
        
        # Get dust fields if requested
        if self.include_dust and hasattr(self.model, 'dust') and self.model.dust is not None:
            dust = self.model.dust
            if hasattr(dust, 'nbin') and dust.nbin > 0:
                # Get total dust density
                try:
                    total_dust = None
                    for i in range(dust.nbin):
                        bin_field = dust._compute_bin_density(i)
                        bin_data = bin_field.data.to_base_units().magnitude
                        bin_yt = self._transpose_model_to_yt(bin_data)
                        
                        if total_dust is None:
                            total_dust = bin_yt.copy()
                        else:
                            total_dust += bin_yt
                        
                        # Optionally add individual bin fields
                        fields[('dust', f'density_bin{i:02d}')] = bin_yt
                    
                    if total_dust is not None:
                        fields[('dust', 'density')] = total_dust
                        logger.debug(f"Added total dust density: range [{np.min(total_dust):.2e}, {np.max(total_dust):.2e}]")
                except Exception as e:
                    logger.warning(f"Failed to add dust fields: {e}")
        
        return fields
    
    def _map_field_name(self, name: str) -> Tuple[str, str]:
        """Map DiskBridge field name to yt field tuple.
        
        Parameters
        ----------
        name : str
            DiskBridge field name like 'density', 'vr', 'vphi'
            
        Returns
        -------
        tuple
            yt field tuple like ('gas', 'density')
        """
        # Velocity components
        if name == 'vr':
            return ('gas', 'velocity_spherical_r')
        elif name == 'vphi':
            return ('gas', 'velocity_spherical_phi')
        elif name == 'vtheta':
            return ('gas', 'velocity_spherical_theta')
        elif name == 'density':
            return ('gas', 'density')
        elif name == 'surface_density':
            return ('gas', 'surface_density')
        elif name == 'temperature':
            return ('gas', 'temperature')
        else:
            # Generic mapping
            return ('gas', name)
    
    def _create_dataset(self) -> "yt.Dataset":
        """Create the yt dataset from collected fields.
        
        Returns
        -------
        yt.Dataset
            The yt dataset object
        """
        # Get mesh info
        r_edges, theta_edges, phi_edges, shape = self._get_mesh_info()
        
        # Collect field data
        fields = self._collect_fields()
        
        if not fields:
            raise ValueError("No fields available to create dataset")
        
        # Build bbox for yt
        # yt expects bbox as [[r_min, r_max], [theta_min, theta_max], [phi_min, phi_max]]
        bbox = np.array([
            [r_edges[0], r_edges[-1]],
            [theta_edges[0], theta_edges[-1]],
            [phi_edges[0], phi_edges[-1]],
        ])
        
        # Create data dict for yt
        data = {}
        for field_tuple, field_data in fields.items():
            data[field_tuple] = field_data
        
        # Load as uniform grid with spherical geometry
        ds = yt.load_uniform_grid(
            data,
            shape,
            length_unit='cm',  # We converted to CGS
            bbox=bbox,
            geometry='spherical',
            axis_order=('r', 'theta', 'phi'),
        )
        
        logger.info(
            f"Created yt dataset: shape={shape}, "
            f"r=[{r_edges[0]/1.496e13:.1f}, {r_edges[-1]/1.496e13:.1f}] au, "
            f"fields={list(fields.keys())}"
        )
        
        return ds
    
    def get_field(self, field: Tuple[str, str]) -> np.ndarray:
        """Get raw field data from the dataset.
        
        Parameters
        ----------
        field : tuple
            yt field tuple like ('gas', 'density')
            
        Returns
        -------
        ndarray
            Field data array
        """
        ad = self.ds.all_data()
        return ad[field]
    
    @property
    def r_au(self) -> np.ndarray:
        """Radial cell centers in AU."""
        return self.model.mesh.centers('r').to('au').magnitude
    
    @property
    def theta(self) -> np.ndarray:
        """Colatitude cell centers in radians."""
        return self.model.mesh.centers('theta').magnitude
    
    @property
    def phi(self) -> np.ndarray:
        """Azimuthal cell centers in radians."""
        return self.model.mesh.centers('phi').magnitude


def create_dataset(
    model: "Model",
    radmodel: Optional["RadModel"] = None,
    **kwargs
) -> "yt.Dataset":
    """Convenience function to create a yt dataset from DiskBridge objects.
    
    Parameters
    ----------
    model : Model
        DiskBridge Model instance
    radmodel : RadModel, optional
        RADMC-3D model wrapper
    **kwargs
        Additional arguments passed to DiskBridgeDataset
        
    Returns
    -------
    yt.Dataset
        The yt dataset object
        
    Examples
    --------
    >>> from diskbridge.visualization import create_dataset
    >>> ds = create_dataset(model, radmodel)
    >>> slc = yt.SlicePlot(ds, 'phi', ('gas', 'density'))
    """
    wrapper = DiskBridgeDataset(model, radmodel, **kwargs)
    return wrapper.ds
