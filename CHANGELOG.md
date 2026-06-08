# Changelog

All notable DiskBridge changes should be recorded here.

This file is release-facing. Detailed implementation plans and progress logs live in `projects/`.

## Unreleased

### Added

- Sphinx documentation site (myst-parser, autodoc/autosummary, numpydoc, sphinxcontrib-bibtex, pydata-sphinx-theme) with a written documentation standard at `docs/contributing/documentation.md`, a literature bibliography at `docs/refs.bib` with ADS links, a shielding user guide at `docs/guides/shielding.md`, and a generated shielding API reference. Build with `sphinx-build -W -b html docs docs/_build/html`.
- Getting-started documentation under `docs/getting_started/` (overview, installation, quickstart, workflow), covering the FARGO snapshot -> model -> RADMC-3D -> chemistry -> imaging pipeline, verified against the `examples/3d_disk/` example.
- Chemistry, dust, RADMC-3D, and line-transfer user guides (`docs/guides/chemistry.md`, `docs/guides/dust.md`, `docs/guides/radmc3d.md`, `docs/guides/line_transfer.md`) and a chemistry API reference page (`run_chemistry`, `load_chemistry_outputs`, `ChemistryResult`). The guide deep dives document the governing equations verified against the implementation: the disk/ISM dust mask, grain settling and Mie opacities, the radiation sources (stellar + accretion + external field), the GOW17 network and its DiskBridge extensions (spatially varying gas-dust coupling, process-dependent UV bands, multi-phase CO ice), the 3-D HEALPix shielding with effective Doppler widths, and the non-LTE escape-probability line transfer.
- Expanded `docs/refs.bib` with 21 ADS-verified literature references covering the documented physics.

### Changed

- Brought the public `diskbridge.chemistry.shielding` docstrings (HEALPix and 1-D shielding, the Draine & Bertoldi H2 function, and the Visser table loader) to the NumPy-style + References standard.

### Fixed

- `examples/3d_disk/run_workflow.py` called the non-existent `chemistry.compute_abundance`; updated it to the real `run_chemistry` entry point.

### Removed

- Relocated historical plan/progress notes out of `docs/` into `projects/`, and retired `docs/healpix_shielding.md` in favor of the verified shielding guide.

### Validation

### Internal

- Integrated code-map tag guidance into the capability registry, source tags,
  agent workflows, generated indexes, and capability-map tests.
- Reorganized pytest files into subsystem folders, added a curated public API smoke test, removed validation-driver helper tests from pytest, and updated scoped test helper and repo-map paths.
- Removed stale pytest expectations that Cartesian stellar UV weighting is unsupported, obsolete CO shielding keyword spelling, and LTE line-mode gas-temperature policy.
