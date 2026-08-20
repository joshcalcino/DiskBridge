# Changelog

All notable DiskBridge changes should be recorded here.

This file is release-facing. Detailed implementation plans and progress logs live in `projects/`.

## Unreleased

### Added

- A Bondi All-Stars snapshot analysis now pairs the two snapshots from each of
  the three `0.5`, `1.0`, and `2.0 Msun` hydro simulations sharing the fixed
  `ISMDens = 3e-21 g cm^-3` background. It color-codes strict
  `ism_weight > 0.9999` total-volume histograms across three radial shells
  spanning the full 50,000 au domain, with equivalent mass-density and
  `n_H = rho / (1.4 m_H)` figure sets and machine-readable distributions.
  Companion all-cell complementary cumulative figures combine all cells within
  5000 au in paired-snapshot panels, span `1e-22`--`1e-14 g cm^-3`, use a
  logarithmic aperture-volume axis, and internally stack every
  density-threshold bar by signed
  `2 * disk_weight - 1` with a diverging symmetric-log colour scale from
  ISM-like through mixed to disk-like material. Matched cumulative gas-mass
  fractions retain the same aperture, axes, and colour encoding to expose
  small dense structures that occupy little volume.
- Conservative finite-time radial dust surface-density transport helpers with
  pressure drift, turbulent diffusion, absorbing inner outflow, an open outer
  diffusion boundary with no imposed inflow, explicit mass-loss diagnostics,
  and a coherent smoothed radial gas background with bounded negative pressure
  slopes that prevent artificial outer dust traps.
- The standalone 1--500 au DiskBridge disc validation now generates
  chemistry-derived LTE/non-LTE [C I] 1-0 and HCO+ 4-3 channel figures from one
  physical RADMC-3D/GOW17 run, without abundance or collider scaling.
- External HEALPix non-LTE solves now expose spherical inner-boundary
  (`stop` or `vacuum_cavity`) and truncated-theta (`vacuum` or
  `boundary_cell_to_rmax`) policies in `HealpixSEConfig`. Ray-step exhaustion
  returns the truncated result with one aggregated warning.
- External HEALPix non-LTE solves can restrict the iterative solve with an
  optional emitting-species abundance floor relative to hydrogen nuclei.
  Positive-density cells outside the active mask retain local LTE populations
  in final and optional checkpoint level-population files, while manifests and
  checkpoint fingerprints record the selection inputs and cell counts.
- Stable HEALPix shielding validation report under
  `docs/testing/spherical_healpix_shielding.md`, covering the exact Cartesian
  reduction, boundary-screen test, absorption and scattering RADMC-3D
  controls, wavelength and photon convergence, closure sensitivity, spherical
  traversal invariants, and the resolved non-uniform sphere.
- Segmented RADMC-3D diagnostics now include native per-segment plots and a
  `segments_summary.json` manifest whenever segmented RT diagnostics are
  enabled.
- GOW17 now accepts `shielding_ray_average = "weighted" | "uniform"` so
  HEALPix shielding can explicitly use RADMC-derived directional UV weights or
  direct uniform ray averaging.
- Run-provenance API: `diskbridge.write_run_manifest()` writes a deterministic, human-readable JSON manifest recording the complete active parameter set (`Params.to_dict()`), the DiskBridge version, and a best-effort git commit and timestamp, so a run can be reproduced and audited. `Params.to_dict()` serializes every parameter field with Pint quantities rendered as `{"value", "unit"}` in canonical units.
- Sphinx documentation site (myst-parser, autodoc/autosummary, numpydoc, sphinxcontrib-bibtex, pydata-sphinx-theme) with a written documentation standard at `docs/contributing/documentation.md`, a literature bibliography at `docs/refs.bib` with ADS links, a shielding user guide at `docs/guides/shielding.md`, and a generated shielding API reference. Build with `sphinx-build -W -b html docs docs/_build/html`.
- Getting-started documentation under `docs/getting_started/` (overview, installation, quickstart, workflow), covering the FARGO snapshot -> model -> RADMC-3D -> chemistry -> imaging pipeline, verified against the `examples/3d_disk/` example.
- Chemistry, dust, RADMC-3D, and line-transfer user guides (`docs/guides/chemistry.md`, `docs/guides/dust.md`, `docs/guides/radmc3d.md`, `docs/guides/line_transfer.md`) and a chemistry API reference page (`run_chemistry`, `load_chemistry_outputs`, `ChemistryResult`). The guide deep dives document the governing equations verified against the implementation: the disk/ISM dust mask, grain settling and Mie opacities, the radiation sources (stellar + accretion + external field), the GOW17 network and its DiskBridge extensions (spatially varying gas-dust coupling, process-dependent UV bands, multi-phase CO ice), the 3-D HEALPix shielding with effective Doppler widths, and the non-LTE escape-probability line transfer.
- Expanded `docs/refs.bib` with 21 ADS-verified literature references covering the documented physics.

### Changed

- Bondi All-Stars line-only Slurm jobs now request 48 GB of memory, retaining
  substantial headroom over the approximately 18 GB observed peak instead of
  reserving 240 GB.
- Bondi All-Stars line imaging can now explicitly render physical GOW17 HCO+
  J=4-3 LTE cubes for both dust and gas temperature choices, with a directly
  viewable channel-map PNG written beside every selected-species FITS cube.
- Bondi All-Stars line imaging also supports the 492.160651 GHz `[C I]`
  3P1--3P0 ALMA Band 8 line from the physical GOW17 atomic-carbon abundance.
- Segmented UV transport can optionally recover a failed low-budget scout with
  one independently seeded final-budget monochromatic run. The runner combines
  all scout and retry packets with count weights, recomputes unequal-budget
  uncertainty, and retries boundary selection without wasting the initial
  scout or launching a redundant terminal UV calculation.
- Bondi All-Stars now maintains only the ten `_8` snapshots. The four 0.5-Msun
  `_9` snapshots, whose ambient densities are ten times lower, are preserved
  separately under `examples/bondi_low_density/` and are excluded from current
  staging and maintained diagnostics.
- Bondi All-Stars local production, cluster staging, and maintained dust
  diagnostics now consume one canonical `bondi_defaults.py` setup containing
  the accepted connected Joos mask, mass-scaled midplane-density support,
  settling, and radial-`a_max` settings. Per-run configuration names the
  angular-momentum estimator aperture `AXIS_SUPPORT_AU`; it is not a material
  disk radius. The committed local run selects `m20_8_dmid_f120_c10` with its
  matching 2-Msun stellar and `chi=1` radiation parameters.
- Bondi All-Stars dust construction now uses a locally normalized ten-bin
  radial maximum-grain-size profile instead of assigning a finite transport
  age. The profile has `a_max = 100 um` at 100 au, decreases as `R**-2`, and
  applies a smooth 0.25-dex complementary-error-function turnover across bin
  sizes. The total settled dust column remains `0.01` times the actual
  mask-weighted disc gas column; no smoothed gas background enters the size
  mix. All configured grain bins remain present, no hard bin cutoff is imposed,
  and disconnected soft-mask tails are not amplified. Maintained Bondi
  directories include `settled_dust_radial_amax.png`,
  `settled_dust_size_threshold_sideon.png`,
  `settled_dust_size_ceiling_sideon.png`,
  `settled_dust_size_threshold_profiles.png`, and
  `settled_dust_size_threshold_faceon.png`, exposing all ten bins and
  cumulative dust above 0.1, 1, 10, and 100 microns. Standalone
  `settled_dust_total_dust_to_gas_sideon.png` and
  `settled_dust_total_dust_to_gas_faceon.png` products show the cell-local
  total of every disc and ISM dust bin relative to gas in phi-averaged edge-on
  and azimuth-preserving top-down views.
- Bondi All-Stars settled-disc weights now include an azimuthal-median,
  disc-frame midplane-density support score. The mass-scaled density midpoint
  is `1.5e-17 (Mstar/Msun) g/cm^3`, with a 0.50-dex transition, so diffuse
  captured material fades into the ISM component without a geometric outer
  radius or a higher local atmosphere density floor.
- Joos disc weights now evaluate the full loaded mesh and enforce both vertical
  and outward radial connectivity. `r_max_for_axis` is solely the required
  angular-momentum support aperture; the physical `r_max`, implicit
  `fthres_vr_inner` ramp, and unused `weight_delta_bins` arguments were removed.
- The Joos disc classifier now uses one explicit poloidal-Mach limit instead of
  separate orbital-to-radial and orbital-to-vertical velocity thresholds.
  Density-weighted smoothed gas with transonic or slower poloidal motion,
  rotational support, and sufficient density seeds the connected disc. Bondi
  All-Stars uses the default `max_poloidal_mach = 1`, rejecting supersonic
  infalling streamers from the settled-disc dust component. The connected
  continuous weight is now the public default rather than an opt-in mode.
- Directional UV-weight construction now ray-integrates one opacity-weighted
  dust-extinction field instead of retaining per-bin ray columns. Parallel
  first-touch and fused source normalization remove redundant ray-sized
  temporaries while preserving the public weights and diagnostic products.
- Supported line species now use a pinned, validated LAMDA snapshot installed
  with DiskBridge, without runtime network access or a user-supplied molecule-
  file path. HCO+ uses the current separate para-H2 and ortho-H2 collision
  tables. Exact molecule-file hashes remain in line-transfer provenance, and
  collision coefficients always use the nearest tabulated temperature endpoint
  outside the table range while the physical gas temperature remains unchanged.
- GOW17 molecular shielding now recomputes the H2 and CO Doppler widths from
  the current gas temperature at each existing non-local update. Directional
  CO cooling consumes the same current CO width during the fused shielding
  traversal; the configured microturbulent field remains fixed.
- The Bondi All-Stars production chemistry now explicitly includes direct CO
  desorption by cosmic-ray whole-grain heating using the HH93 70 K duty-cycle
  approximation with the ProDiMo ionization-rate scaling; CR-induced UV
  photodesorption remains a separate enabled channel.
- Variable-linewidth Visser CO shielding, effective linewidths, H2/C shielding
  algebra, and directional reductions now use parallel array kernels. Fused
  H2*CO reductions and in-place linewidth storage remove ray-sized
  temporaries, GOW17 elemental-budget projection is parallel over cells,
  HEALPix DDA workers receive cyclic spatial work, and directional Omukai
  cooling uses guided OpenMP scheduling.
- Native GOW17 batch solves reuse one SUNDIALS/CVODE solver per OpenMP worker
  and report aggregate failures without emitting per-cell `t+h=t` warning
  floods. Coupled updates also pass their already-projected state directly to
  the solver instead of allocating an immediate duplicate projection.
- GOW17 AstroChem restart state is stored in the compact atomic
  `checkpoint/gow17_state.h5` format. Checkpoints contain chemistry state,
  shielding, status, histories, and mesh provenance rather than a complete
  model snapshot.
- The segmented-UV validation retains 128 radial cells while using a `24 x 24`
  angular grid; its tenfold full-domain reference multiplier now applies only
  to `mcmono`.
- Segmented RADMC-3D UV transport now uses paired seeded scouts, measured
  shell/P99 Monte Carlo uncertainty, conservative stellar screening,
  wavelength-resolved one-pass calibration of inherited UV boundaries that
  preserve the original IR/CMB field, warning-only parent-child join checks,
  and a direct full-budget terminal run without redundant paired scouts.
- Brought the public `diskbridge.chemistry.shielding` docstrings (HEALPix and 1-D shielding, the Draine & Bertoldi H2 function, and the Visser table loader) to the NumPy-style + References standard.

### Fixed

- Segmented RADMC-3D restarts now validate and load compatible terminal
  products before considering the lower-budget scout pass. Terminal photon
  counts, mesh and source identities, wavelength/product metadata, parameter
  snapshots, physical input fingerprints, and exact binary payload sizes are
  checked; `force=True` remains the explicit recomputation path.
- Directional GOW17 CO-cooling reduction now handles shallow non-monotonic
  interpolation cusps by bracketing sorted directional samples and choosing
  the inverse nearest their mean logarithmic column. The normal monotonic
  endpoint path is unchanged, and production diagnostics report how many
  cells required local bracketing.
- Radial dust diffusion no longer reflects from the last connected disc
  annulus. The outer boundary now uses zero exterior settled-dust concentration
  and records the one-way escaped flux, preventing fake outer rings while
  preserving the full mass budget.
- Model-level radial dust transport no longer treats native spherical shells
  as disc annuli or divides evolved columns by negligible unsupported columns.
  Detached outer soft-mask material therefore remains ISM dominated instead of
  being amplified into spurious settled-disc dust blobs.
- Radial maximum-grain-size redistribution now covers the full contiguous soft
  mask tail down to `weight_floor`, rather than stopping at the boolean 0.5
  contour and leaving an unmodified large-grain ring immediately outside it.
- Soft Joos disc weights no longer snap high scores to exactly one. Their
  complementary ISM fractions therefore remain smooth and positive, removing
  artificial sharp changes in settled-disc/ISM dust-density ratio maps.
- Joos disc-axis measurement now always has finite radial support through the
  required `r_max_for_axis`. Empty or degenerate axis selections raise instead
  of silently broadening the density population or choosing positive z.
- The high-level GOW17 equilibrium slab now translates physical incident
  Draine units to the native solver according to radiation geometry: beamed
  slabs retain `G0 = 2 chi`, while one-sided isotropic slabs use `G0 = chi`.
  Its returned actual dissociation and ionization fields use the same
  illuminated-face normalization.
- Native and Python incident-slab GOW17 chemistry now attenuate the external
  CO-photodesorption continuum as `exp(-1.8 A_V)` instead of reusing the much
  broader dust-heating ISRF attenuation. Native slab atomic-carbon shielding
  also excludes carbon locked in CO ice, and its returned diagnostics expose
  the actual photodesorption and C-shielding fields used by the solver.
- The native GOW17 equilibrium slab now receives its actual uniform dust
  temperature separately from the initial gas temperature, so CO freeze-out
  and gas--dust thermal coupling use the configured slab dust field.
- Cartesian RADMC-3D gas temperature, number density, microturbulence,
  velocity, and external level-population products now consistently serialize
  cells with x varying fastest, as required by the RADMC-3D grid contract.
- Spherical HEALPix traversal now rejects the opposite halves of constant-phi
  planes and theta cones, includes the physical inner radial boundary, and
  resolves tied/polar crossings from a stable beyond-boundary probe. Oblique
  spherical columns no longer terminate early or accumulate non-physical path
  lengths.
- HEALPix Cartesian, spherical, and starward DDA columns now include the
  geometric source-center-to-face segment exactly once. Grid-normal rays
  therefore use the same half-cell starting convention as the canonical 1-D
  shielding reducer.
- RADMC-3D thermal and monochromatic cache identity now includes the exact
  `external_source.inp` content hash. Segmented setup writes missing configured
  or inherited spectra, reuses exact matches, and rejects mismatches without
  overwriting them.
- RADMC-3D thermal, monochromatic, and resolved segmented photon counts are now
  rejected before execution when they exceed the signed 32-bit counter range.
- Weighted HEALPix shielding now uses exact cell-to-star H2/C/CO columns for
  the direct stellar contribution, avoiding point-source artifacts from pairing
  stellar UV weights with HEALPix pixel-center boundary columns.
- `examples/3d_disk/run_workflow.py` called the non-existent `chemistry.compute_abundance`; updated it to the real `run_chemistry` entry point.

### Removed

- Removed the selectable `HealpixSEConfig.collision_temperature_policy` and
  the legacy effective-H2 HCO+ LAMDA path from DiskBridge, along with
  external-population APIs that accepted local molecule files.
- Retired the exploratory GOW17 Cartesian slab, RADMC-3D scattering-closure,
  3D-PDR/beta, and resolved-sphere validation runners and their raw outputs
  after consolidating their conclusions in the stable shielding testing
  report. None of the empirical angular closures is a supported production
  implementation.
- Removed the unused shielding UV-boundary module and obsolete flat-profile
  `find_r_split` helpers; calibrated noise-aware boundary selection is the sole
  segmented-UV split implementation.
- Removed the GOW17 `checkpoint.include_dust` setting; restart checkpoints do
  not serialize model dust fields.
- Removed superseded segmented RT split controls and output discovery paths;
  `segmented_tol` is the sole scout-uncertainty and join-warning tolerance.
- Relocated historical plan/progress notes out of `docs/` into `projects/`, and retired `docs/healpix_shielding.md` in favor of the verified shielding guide.

### Validation

- Promoted the accepted Bondi streamer abundance figures into the manuscript.
  The paper candidates show CO, C, C+, CHx, HCO+, and neutral H from `1e-12`
  to `1e-1`, with all six species in a single-row legend on the established
  `8.5888 x 7.92`-inch paper canvas.
- Bondi All-Stars one-dimensional streamer histories now archive every ordered
  GOW17 state component for both time-dependent and equilibrium branches,
  alongside derived neutral H, C, and O. Combined abundance figures show CO,
  C, C+, CHx, HCO+, and H in a single-row species legend over the displayed
  range `1e-12` to `1e-1`. Neutral H is the sole plotted atomic-gas proxy,
  while CO-poor H2-fraction summaries test whether CO loss occurs while the
  infall remains predominantly molecular.
- The Bondi All-Stars streamer plotter now exports the combined abundance figures
  as both PNG and vector PDF, with inward ticks on all four panel edges,
  an explicit two-point inter-row gap, compact canvases, and a buffered header
  above the grids. Y-axis labels appear on alternate decades while all
  logarithmic tick marks remain visible.
- Made the Bondi All-Stars one-dimensional streamer comparison explicitly use
  a 0.1-Draine ambient field, added a matched one-Draine six-case grid, and
  recorded the resulting chemistry and temperature sensitivity in a direct
  comparison report.
- Regenerated the maintained interface and azimuthal diagnostics for the
  representative `m20_8_dmid_f120_c10` snapshot with the smooth radial
  maximum-grain-size prescription. The total settled column follows the actual
  mask-weighted gas, while the normalized bin fractions vary continuously with
  the prescribed local `a_max(R)`.
- A full-native 23,415,480-cell Bondi hydro/mask A/B reduced the contaminated
  `m10_8_dm_f250_c10` Joos tilt from 27.178 degrees with 50,000-au support to
  0.079 degrees with 387.1-au support and removed its pinwheel-like mask. The
  nearly aligned `m20_8_dmid_f120_c10` control changed from 2.678 to 0.013
  degrees and retained a 0.9815 mask Jaccard overlap.
- Added a gas-phase GOW17 Figure 23 reproduction using the paper's one-sided
  60-degree isotropic approximation, unit incident field, 1000-zone grid, and
  strict `reltol = 1e-4` convergence control at `nH = 100` and
  `1000 cm^-3`. It disables CO phase chemistry, overlays current DiskBridge
  profiles on vector curves extracted from the published figure, and records
  the numerical criterion in its report. The paired CO-only phase comparison
  uses the same strict control; both workflows retain their figures, numerical
  outputs, extracted paper curves, and reports in dedicated artifact folders
  with complete provenance.
- Added a deterministic complete directional UV-weight benchmark that records
  scientific samples, row closure, wall time, CPU occupancy, sampled RSS
  history, and known-array memory estimates for baseline/optimized comparison.
- Added `validation/gow17_fig23`, a two-panel equilibrium slab comparison at
  `nH = 100` and `1000 cm^-3` showing CO, C, C+, HCO+, OHx, CHx, O, and O+
  with CO phase chemistry disabled and enabled. The enabled branch inherits
  the regular fiducial DiskBridge adsorption and desorption settings without
  validation-specific physics overrides, while both branches solve gas
  temperature.
- Added an HCO+ inventory-weighted gas-temperature histogram and cumulative
  distribution to the standalone disc validation, written directly beside the
  channel-map figures.
- Added the standalone `validation/diskbridge_disc` line workflow. It builds a
  240 by 192 by 1 axisymmetric DiskBridge disc from scratch over 1--500 au,
  then compares LTE and non-LTE CO J=6-5 using identical physical H2, CO,
  temperature, velocity, and microturbulence fields. The two maintained
  figures live directly in the validation directory and the reproducibility
  products use one shallow `out/{lte,nonlte,levelpop}` tree. No density or
  collider transform, PRIZMO or GOW17 chemistry product, Tdust comparison,
  cross-chemistry branch, or CSV result table is included.
- Added `validation/nonlte_ray_boundaries`, which compares the two spherical
  theta and inner boundary policies on the saved PRIZMO/GOW17 disk and checks
  boundary-cell extrapolation against an explicitly widened polar grid.
- Added a Bondi All-Stars 1D streamer grid for `0.5`, `1.0`, and `2.0 Msun`
  sources at optimistic `nH = 1e5` and `1e6 cm^-3` densities, with the
  production accretion rates `3e-9`, `1e-8`, and `5e-8 Msun yr^-1`,
  respectively, spectrum-aware UV products, strict
  thermochemical-equilibrium tracking, evolved gas energy, fixed dust
  temperature, concurrent star-level execution, and complete ordered GOW17
  state archives for both chemistry branches. Each density has one combined
  CO, C, C+, CHx, HCO+, and H profile figure comparing all
  three masses at stream-front distances near `3000`, `1000`, and `500 au`,
  with a single-row species legend, a displayed upper limit of `1e-1`, and
  matching vector-PDF exports. Density-specific stream-front
  gas-temperature histories and full evolving/equilibrium temperature
  profiles are retained for both the fiducial and one-Draine grids.
- Calibrated segmented-UV joins reduced shell-mean `chi_broad` offsets from
  0.56-2.15% to 0.04-0.07%. Against the saved full-domain reference, its
  volume-weighted P99 residual improved from 2.89% to 1.35%; the remaining
  few-percent inner radial residual is consistent with the isotropic,
  angle-averaged boundary approximation.
- Simplified `validation/segmented_uv` to one canonical diagnostic layout:
  native per-segment and join diagnostics under `plots/segmented_rt/`, UV
  product comparisons under `plots/uv_products/`, and root summary metrics and
  radial comparison only. Plot regeneration no longer runs RADMC-3D.
- Added `validation/gow17_healpix_performance`, a provenance-rich sustained
  scaling and numerical-comparison workflow for the Bondi All-Stars `nside=4`
  interpolation and complete post-ray shielding stages, plus the native GOW17
  batch solver.
- Added `validation/cartesian_wedge_uv_weighting`, a Cartesian sphere-plus-cone
  UV validation comparing no shielding, uniform HEALPix shielding, and
  weighted HEALPix shielding chemistry from one shared RADMC-3D transport run
  with an explicit `2 R_sun`, `10000 K` blackbody source.
- Added a selected-cell point-source benchmark to
  `validation/cartesian_wedge_uv_weighting`. HEALPix figures now expose the
  production direct-star and final weights as per-pixel UV percentages on one
  logarithmic color scale, the linear integrated star/residual split,
  molecular columns, shielding factors, and weighted contributions without
  rerunning the retained RADMC-3D calculation.
- Updated the cube validation workflow for current GOW17/RADMC-3D UV products, mctherm-derived dust temperatures, explicit shielding-off comparisons, and current GOW17 diagnostic plots.
- Adjusted the high-resolution cube validation density setup to use a 100x lower base density with a localized 100x boost of the densest clumps.
- Added a cube validation `--prepare-only` mode that writes RADMC-3D inputs and setup summaries without launching transport or GOW17 chemistry.
- Corrected the cube validation dust setup to scalar `0.01-0.25 micron`,
  `p=3.5` grains and allowed UV product interpolation through exact-zero
  mean-intensity samples in fully shielded cells.
- Added a PPDwind 400 au spherical-model RADMC-3D/GOW17/CO validation with
  normalized `R/R0`, `z/R0` Joos-mask and dust overview plots, depleted
  wind-side small grains, stellar plus accretion UV with the accretion rate
  derived from the selected PPDwind solution, RADMC-3D dust temperature and UV
  products, PRIZMO-style gas-temperature diagnostics, GOW17 CO abundance fields
  using RADMC dust surface areas, and 30 degree LTE CO 3-2 channel maps.

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
