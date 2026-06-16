# Changelog

All notable DiskBridge changes should be recorded here.

This file is release-facing. Detailed implementation plans and progress logs live in `projects/`.

## Unreleased

### Added

- GOW17 now accepts `shielding_ray_average = "weighted" | "uniform"` so
  HEALPix shielding can explicitly use RADMC-derived directional UV weights or
  direct uniform ray averaging.
- Run-provenance API: `diskbridge.write_run_manifest()` writes a deterministic, human-readable JSON manifest recording the complete active parameter set (`Params.to_dict()`), the DiskBridge version, and a best-effort git commit and timestamp, so a run can be reproduced and audited. `Params.to_dict()` serializes every parameter field with Pint quantities rendered as `{"value", "unit"}` in canonical units.
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

- Added `validation/cartesian_wedge_uv_weighting`, a Cartesian sphere-plus-cone
  UV validation comparing no shielding, uniform HEALPix shielding, and
  weighted HEALPix shielding chemistry from one shared RADMC-3D transport run
  with an explicit `2 R_sun`, `10000 K` blackbody source.
- Updated the cube validation workflow for current GOW17/RADMC-3D UV products, mctherm-derived dust temperatures, explicit shielding-off comparisons, and current GOW17 diagnostic plots.
- Adjusted the high-resolution cube validation density setup to use a 100x lower base density with a localized 100x boost of the densest clumps.
- Added a cube validation `--prepare-only` mode that writes RADMC-3D inputs and setup summaries without launching transport or GOW17 chemistry.
- Corrected the cube validation dust setup to scalar `0.01-0.25 micron`,
  `p=3.5` grains and allowed UV product interpolation through exact-zero
  mean-intensity samples in fully shielded cells.

### Internal

- Integrated code-map tag guidance into the capability registry, source tags,
  agent workflows, generated indexes, and capability-map tests.
- Consolidated duplicated SHA256 file, array, and RADMC vector binary hashing
  helpers into `diskbridge.utils`.
- Removed the shadowed duplicate `RadData._findDataFile` and
  `RadData.read_mean_intensity_file` definitions (and their dead ASCII reader
  family) in `radmc3d/data.py`, and consolidated the strict `_as_f64` shielding
  coercion into `diskbridge.chemistry.shielding._array_utils`.
- Adopted `write_run_manifest` across the cube and GOW17 validation drivers
  (replacing per-script `_write_inputs`), and consolidated duplicated validation
  helpers: shared JSON I/O and matplotlib setup in
  `validation/gow17_infall_stream_1d/_common.py`, and shared synthetic-field
  generators in `validation/cube_test/_fields.py`.
- Reorganized pytest files into subsystem folders, added a curated public API smoke test, removed validation-driver helper tests from pytest, and updated scoped test helper and repo-map paths.
- Removed stale pytest expectations that Cartesian stellar UV weighting is unsupported, obsolete CO shielding keyword spelling, and LTE line-mode gas-temperature policy.
