# DiskBridge

DiskBridge is a Python library for turning 2D or 3D disk hydrodynamics
simulations (for example FARGO-3D runs) into RADMC-3D models and
synthetic observations.

## What it does

- Reads hydro snapshots into a structured disk model with physical
  units on a common mesh
- Builds gas and dust structure, including vertical puffing and grain
  size bins
- Writes a complete set of RADMC-3D input files: grid, dust density,
  dust opacities, stellar source, control file, and optional gas
  temperature/velocity and molecular number densities
- Optionally runs RADMC-3D to compute dust temperatures and line
  emission
- Converts RADMC-3D outputs into FITS images and data cubes
- CO self-shielding using HEALpix, ray-tracing, and Visser et al. 2009 self-shielding tables, with selectable 'uniform' (fast) or 'sortedsearch' (reference) ray integration methods

## Typical workflow

In a typical use case you:

1. Prepare a plain-text parameter file `params.txt` that defines
   opacities, stellar properties, Monte Carlo settings, and chemistry
   switches.
2. Load a hydro snapshot or a saved model into `diskbridge.model.Model`
   using `diskbridge.load_model`.
3. Configure the dust distribution with `model.dust` helpers
   (e.g. proportional to gas density, grain size range, number of bins).
4. Use `diskbridge.radmc3d.RadWriter` to write RADMC-3D input files and
   (optionally) compute dust opacities for each dust bin. (this will be updated to be automatic)
5. Use `diskbridge.radmc3d.RadModel` to run RADMC-3D, compute dust
   temperatures, apply simple CO chemistry, and write number density
   files.
6. Use `diskbridge.radmc3d.RadImage` (and related helpers) to read
   RADMC-3D images or spectra and export them to FITS.

## Main components

- `diskbridge.load_model` and `diskbridge.model.Model`:
  load simulations, rescale units, store gas and dust fields on a mesh.
- `model.dust`:
  set dust size distribution and dust-to-gas coupling.
- `diskbridge.radmc3d.RadWriter`:
  write grid, dust density, opacity, and control files for RADMC-3D.
- `diskbridge.radmc3d.RadModel`:
  manage RADMC-3D runs, dust temperature, and simple CO chemistry
  following Pinte 2018-style switches.
- `diskbridge.radmc3d.RadImage`:
  read RADMC-3D images and write FITS files with WCS information.

## Installation

From this directory:

```bash
pip install -e .
```

This installs the `diskbridge` Python package.

## Requirements

- Python 3.9 or later
- Python dependencies listed in `pyproject.toml`, including at least:
  numpy, scipy, astropy, pint, numba
- A working `radmc3d` executable on your PATH to run radiative
  transfer
- Hydro simulation outputs in a supported format (for example
  FARGO-3D snapshots, others can be added easily)

## Examples

Ready-to-run examples and parameter files live in the `examples/`
directory and demonstrate the workflows described above.

## Acknowledgements 
DiskBridge was not written from scratch. It builds on and borrows ideas
from the following public code bases:
- `fargo2radmc3d`: https://github.com/charango/fargo2radmc3d
- `radmc3d-2.0` (including `radmc3dPy`): https://github.com/dullemond/radmc3d-2.0

## Implemented:

- Proper efficient ray tracing using algorithm from Lile Wang's Kratos code
- Visser+09 self-shielding per cell in the simulation. We compute the Visser+09 factor per ray, then average f per cell to update the CO numberdensity 


## TODO/broken:

- make units in parameter file more consistent (e.g. dust size, wavelengths, all in um) (should be implemented correctly)
- make radmc3d submodule use pint quantities (implemented, not fully tested)
- mu_h should be read from simulation data
- hydro temperature probably is broken when we rescale the disc, likely affects the dust settling 
