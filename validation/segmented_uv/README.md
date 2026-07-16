# Segmented UV Runner Validation

## Scientific Question

Does the segmented RADMC-3D UV workflow reproduce a full-domain calculation
closely enough for UV chemistry while reducing Monte Carlo sampling cost in the
inner domain?

The validation also measures how much error is introduced when the parent
shell's angle-averaged `J_nu` is used as the child's isotropic incoming boundary
spectrum.

## Expected Behavior

- Independent scout pairs should provide stable shell-mean and cell-level UV
  noise estimates.
- Split boundaries should remain outside the configured stellar-screening
  threshold in two adjacent radial shells.
- Parent-child join discrepancies should be recorded for spectra, `chi`, and
  active UV products without terminating an otherwise usable run.
- One wavelength-resolved child calibration should remove cumulative scalar
  boundary-normalization drift without iterating to force a match.
- The merged segmented fields should follow the full-domain reference through
  the radial range where that reference is adequately sampled.

## Inputs

`params.txt` contains the DiskBridge and RADMC-3D configuration. The segmented
calculation uses its configured thermal and monochromatic photon budgets. The
full-domain reference uses the same thermal budget and ten times the configured
monochromatic budget. Intermediate segments use paired UV scouts; each
disposable boundary calibration uses one scout-estimator budget; and the known
terminal segment runs directly at the final budgets. Every individual RADMC-3D
photon count must fit in a signed 32-bit integer.

The driver builds one UV wavelength grid containing the configured base samples
and every active UV-product boundary, then passes that exact grid to both
calculations. Both calculations derive `chi_broad`, `G_CO_diss`, `G_H2_diss`,
`G_C_ion`, and `G_CO_pdes` from their respective `mean_intensity.bout` files.

The model is a uniform-density `128 x 24 x 24` spherical grid with logarithmic
radial spacing. The full 128-cell radial resolution is retained to resolve the
radial `chi` profile; only the angular resolution is reduced. The driver
constructs identical models for the full-domain reference and segmented
calculation.

## Run

From the repository root:

```bash
python validation/segmented_uv/run.py
```

On a Slurm system, submit `validation/segmented_uv/run.slurm` from within the
validation directory after adapting its environment setup to the cluster.

To regenerate plots from saved calculations without running RADMC-3D:

```bash
python validation/segmented_uv/plot_diagnostics.py \
  --workdir validation/segmented_uv
```

## Outputs

Generated products remain ignored by Git:

- `baseline_run/`: unsegmented full-domain RADMC-3D reference.
- `segmented_run/`: segmented RADMC-3D calculations and merged fields.
- `plots/`: the single output root for validation diagnostics.
- `plots/metrics.json`: photon accounting and full-domain field differences.
- `plots/compare_radial_profiles.png`: full-domain versus merged segmented
  radial dust-temperature and `chi` profiles, with every split radius marked.
- `plots/uv_products/metrics.json`: product-by-product cell residual statistics.
- `plots/uv_products/<field>.png`: full-domain and segmented radial profiles
  with signed fractional residuals for each configured scalar UV product.
- `plots/segmented_rt/`: native DiskBridge merged-field and per-segment scout,
  field, and parent-child join diagnostics.
- Each child's `radmc3d_outputs/mcmono/boundary_calibration.json` records its
  wavelength-resolved boundary correction. Disposable RADMC-3D calibration
  directories are removed after a successful handoff.

The runner enables native segmented diagnostics and stores paired-estimator
uncertainty, stellar-screening, boundary-calibration, and warning metadata in
each segment's canonical output. Native diagnostics include
`uv_scout_quality.png` for scouted segments and, for child segments,
`uv_join_diagnostics.png`. The validation does not create a second per-segment
plot tree or rerun `mcmono` while plotting.

On restart, configured and inherited external-source files must exactly match
the spectra expected from the current inputs and parent segment. A mismatch
stops before RADMC-3D runs and identifies the segment directory that must be
removed; existing sources are never silently replaced.

The committed validation directory contains only the workflow files. Run
products appear in the ignored directories above after execution.

## What To Inspect

- Radial `chi` and UV-product profiles against the full-domain reference.
- Volume-weighted RMS, median absolute, P99 absolute, and maximum cell-level
  fractional differences for every configured UV product.
- Parent-child spectral residuals in each comparison shell.
- Shell-mean and volume-weighted P99 Monte Carlo uncertainty.
- Cell-level residual maps near each split and farther into the child domain.
- Whether discrepancies are random noise, scalar normalization offsets, or
  spatially structured errors consistent with angular-boundary information
  loss.
- Warning records for every join outside the selected scientific tolerance.

## Known Limitations

- `mean_intensity.bout` contains angle-averaged `J_nu`, not a directional
  inward intensity or inward-crossing flux.
- The saved full-domain reference is one Monte Carlo realization and cannot by
  itself measure its residual noise.
- Generated cache metadata records the preflight-validated photon count passed
  to each RADMC-3D calculation, the wavelength-grid identity, full mesh
  identity, and exact external-source file hash; `params.txt` records the
  configured budgets.
