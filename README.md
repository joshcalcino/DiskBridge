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
- Computes molecular abundances with a chemistry step, from simple
  abundance "switch" models up to the Gong, Ostriker & Wolfire 2017
  reduced network, including CO self-shielding computed per cell from
  HEALPix ray tracing and the Visser et al. 2009 shielding tables
- Converts RADMC-3D outputs into FITS images and data cubes
- Writes a run-provenance manifest recording the full active parameter
  set, DiskBridge version, and git commit for a run

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
5. Use `diskbridge.radmc3d.RadModel` to run RADMC-3D and compute dust
   temperatures.
6. Use `diskbridge.chemistry.run_chemistry` to compute molecular
   abundances and write number-density files.
7. Use `diskbridge.radmc3d.RadImage` (and related helpers) to read
   RADMC-3D images or spectra and export them to FITS.

## Main components

- `diskbridge.load_model` and `diskbridge.model.Model`:
  load simulations, rescale units, store gas and dust fields on a mesh.
- `model.dust`:
  set dust size distribution and dust-to-gas coupling.
- `diskbridge.radmc3d.RadWriter`:
  write grid, dust density, opacity, and control files for RADMC-3D.
- `diskbridge.radmc3d.RadModel`:
  manage RADMC-3D runs and dust temperature.
- `diskbridge.chemistry.run_chemistry`:
  compute molecular abundances and number densities with the selected
  chemistry model (abundance switches or the GOW17 network with
  HEALPix shielding).
- `diskbridge.radmc3d.RadImage`:
  read RADMC-3D images and write FITS files with WCS information.
- `diskbridge.provenance.write_run_manifest`:
  record a reproducible, human-readable manifest of a run's parameters,
  version, and git commit.

## Installation

From this directory:

```bash
pip install -e .
```

This installs the `diskbridge` Python package.

## Requirements

- Python 3.9 or later
- Python dependencies listed in `pyproject.toml`, including at least:
  numpy, scipy, astropy, pint, numba, healpy
- A working `radmc3d` executable on your PATH to run radiative
  transfer
- Hydro simulation outputs in a supported format (for example
  FARGO-3D snapshots, others can be added easily)

## Examples

Ready-to-run examples and parameter files live in the `examples/`
directory and demonstrate the workflows described above.

## Tests and validation

- `tests/` contains fast, deterministic pytest checks covering the
  model, chemistry, and RADMC-3D subsystems. See `tests/README.md`.
- `validation/` contains larger, slower physics validation workflows
  (benchmark reproductions, code comparisons, diagnostic plots) that
  are not run as part of the normal pytest suite. Each validation
  subdirectory documents its own setup and outputs.

## Documentation

- `docs/chemistry_models.md`: overview of the available chemistry
  models.
- `docs/gow17_model_and_extensions.md`: the GOW17 network and
  DiskBridge's extensions to it.
- `CHANGELOG.md`: release-facing summary of notable changes.

## Acknowledgements 
DiskBridge was not written from scratch. It builds on and borrows ideas
from the following public code bases:
- `fargo2radmc3d`: https://github.com/charango/fargo2radmc3d
- `radmc3d-2.0` (including `radmc3dPy`): https://github.com/dullemond/radmc3d-2.0

## Implemented

- Proper efficient ray tracing using algorithm from Lile Wang's Kratos code
- The Gong, Ostriker & Wolfire 2017 reduced chemistry network, with
  Visser+09 CO self-shielding computed per ray from 3-D HEALPix ray
  tracing and averaged to a per-cell shielding factor

## Known limitations

- `mu_h` (mean molecular weight per hydrogen nucleus) is a hardcoded
  constant rather than being derived from simulation data
  (`model/dust.py`, `radmc3d/model.py`).
- The `pressure` field is not rescaled when a model is loaded with a
  non-default `length_scale`/`mass_scale`, unlike density and
  temperature; this only matters for the dust-settling fallback path
  that reads pressure when temperature is unavailable.
- Parameter-file units are not fully consistent: dust size and
  wavelength parameters are in microns, but `uv_min`/`uv_max` are
  still in nanometers.
- Pint quantities are used throughout the core of the `radmc3d`
  submodule (`model.py`, `segmented.py`, `data.py`, `uv_products.py`,
  `writer.py`, `wavelengths.py`), but `opacities.py`, `image.py`,
  `run.py`, `colliders.py`, `cache.py`, and `dustkappa_reader.py` still
  use plain floats/arrays.
