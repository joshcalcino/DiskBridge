"""DiskBridge Visualization Submodule.

This submodule provides visualization tools for DiskBridge models using
yt-project as the primary backend. It supports:

- Slice plots (midplane, poloidal views)
- Projection plots (face-on, edge-on views)  
- Azimuthal slices with half-disk (0 < R) option
- Radial and vertical profile plots
- Integration of both Model and RadModel (RADMC-3D) data

Quick Start
-----------
>>> import diskbridge
>>> from diskbridge.visualization import Visualizer, create_dataset
>>> 
>>> # Load model
>>> model = diskbridge.load_model('data/', file_n=10)
>>> model.puff_up_model(n=64)
>>> 
>>> # Create visualizer
>>> viz = Visualizer(model)
>>> 
>>> # Create plots
>>> fig1 = viz.azimuthal_slice(('gas', 'density'), half_only=True)
>>> fig2 = viz.projection(('gas', 'density'), axis='theta')

With RADMC-3D Data
------------------
>>> from diskbridge.radmc3d import RadModel
>>> 
>>> radmodel = RadModel(model, model_dir='radmc_output/')
>>> radmodel.read_temperature()
>>> 
>>> viz = Visualizer(model, radmodel)
>>> fig = viz.azimuthal_slice(('gas', 'temperature'))

Using yt Directly
-----------------
>>> import yt
>>> from diskbridge.visualization import create_dataset
>>> 
>>> ds = create_dataset(model, radmodel)
>>> slc = yt.SlicePlot(ds, 'phi', ('gas', 'density'))
>>> slc.save('density_slice.png')
"""

from __future__ import annotations

# Core classes
from .backend import VisualizationBackend
from .dataset import DiskBridgeDataset, create_dataset
from .yt_backend import (
    Visualizer,
    quick_slice,
    quick_projection,
)

# Profile utilities
from .profiles import (
    azimuthal_average,
    compute_radial_profile,
    compute_midplane_profile,
    compute_vertical_profile,
    compute_column_density,
    compute_surface_density_from_3d,
    register_small_dust_density_field,
    compute_small_dust_midplane_profile,
    plot_small_dust_midplane_profile,
    plot_small_dust_midplane_map,
    plot_phi_avg_rz_slice,
    plot_small_dust_rz_slice,
    ProfilePlotter,
)
from .diagnostics import (
    make_chemistry_diagnostic_plots,
    make_dust_component_diagnostic_plots,
    make_segmented_rt_diagnostic_plots,
)

# Unit conversion utilities
from .units import (
    pint_to_unyt,
    pint_to_unyt_cgs,
    unyt_to_pint,
)


__all__ = [
    # Core
    'VisualizationBackend',
    'DiskBridgeDataset',
    'create_dataset',
    # Main visualizer
    'Visualizer',
    # Profiles
    'azimuthal_average',
    'compute_radial_profile',
    'compute_midplane_profile',
    'compute_vertical_profile',
    'compute_column_density',
    'compute_surface_density_from_3d',
    'register_small_dust_density_field',
    'compute_small_dust_midplane_profile',
    'plot_small_dust_midplane_profile',
    'plot_small_dust_midplane_map',
    'plot_phi_avg_rz_slice',
    'plot_small_dust_rz_slice',
    'ProfilePlotter',
    'make_chemistry_diagnostic_plots',
    'make_dust_component_diagnostic_plots',
    'make_segmented_rt_diagnostic_plots',
    # Quick functions
    'quick_slice',
    'quick_projection',
    # Unit conversion
    'pint_to_unyt',
    'pint_to_unyt_cgs',
    'unyt_to_pint'
]
