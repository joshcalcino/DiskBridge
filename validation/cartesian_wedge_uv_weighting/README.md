# Cartesian Wedge UV Weighting Validation

This validation is being built to compare GOW17 CO chemistry when HEALPix
shielding factors are averaged directly over directions versus averaged with
directional UV weights.

## Scientific Question

How does UV-weighted HEALPix shielding change the final CO distribution in an
asymmetric Cartesian cloud where an external UV field illuminates the box and a
point-like UV source shines into a wedge-shaped cavity?

## Expected Behavior

The dense circular cloud should preserve more CO than the low-density ambient
box and wedge cavity. The weighted-average run is expected to respond more
strongly to the illuminated wedge and source direction than the direct uniform
HEALPix average.

## Inputs

The default geometry and physical parameters live in `config.toml`:

- 3-D Cartesian cloud whose central `z = 0` slice is a circle.
- Dense spherical cloud centered above the coordinate origin.
- Point-like `2 R_sun`, `10000 K` blackbody source below the circle, at the
  coordinate origin.
- No external isotropic UV background by default (`chi = 0`); the UV field is
  dominated by the origin-centered stellar source.
- Conical cavity with its narrow apex at the cloud center, widening downward
  toward the source; the cavity uses the same density as the ambient gas.
- Low-density ambient gas outside the circle.
- ISM/MRN dust with the same size range, power-law index, and bin count used by
  `validation/cube_test`.
- RADMC-3D photon counts, UV wavelength range, chemistry iteration count, and
  microturbulent width matched to the cube-test validation.

## Run

Full validation:

```bash
python validation/cartesian_wedge_uv_weighting/run.py --overwrite
```

This runs RADMC-3D transport once and then runs three GOW17 chemistry variants
from the same dust-temperature and UV-product fields:

- `no_shielding`: unity shielding factors, no shielding fixed-point iteration,
  normal chemistry equilibrium solve.
- `uniform_healpix_average`: HEALPix shielding with direct uniform ray
  averaging.
- `weighted_healpix_average`: HEALPix shielding with RADMC-derived directional
  UV weights.

Geometry-only preview:

```bash
python validation/cartesian_wedge_uv_weighting/run.py --density-only --overwrite
```

Central-cell HEALPix column maps:

```bash
python validation/cartesian_wedge_uv_weighting/healpix_center_map.py --overwrite
```

This diagnostic uses the current density setup only. It does not run RADMC-3D
or chemistry.

Point-source angular-weight benchmark:

```bash
PYTHONPATH=src python validation/cartesian_wedge_uv_weighting/weight_benchmark.py --overwrite
```

This post-processes the retained `outputs/default` calculation; it does not
launch RADMC-3D or rerun chemistry. That calculation used 200 million
monochromatic photons, one point star, no external UV field, and no scattering.
It is therefore an objective direct-source benchmark: the incident angular UV
weight should be concentrated in the HEALPix pixel containing the direction to
the star. The scalar RADMC-3D UV magnitude remains the normalization.

## Outputs

The full validation writes:

- `outputs/default/summary.json`
- `outputs/default/report.md`
- `outputs/default/density_fields.npz`
- `outputs/default/radmc3d_inputs/`
- `outputs/default/radmc3d_outputs/`
- `outputs/default/variants/<variant>/summary.json`
- `outputs/default/variants/<variant>/chemistry_fields.npz`
- `outputs/default/plots/*.png`
- `outputs/healpix_center_map/healpix_sphere-center_cell_nH_column_nsides.png`
- `outputs/healpix_center_map/healpix_sphere-center_cell_nH_column_summary.json`
- `outputs/healpix_center_map/healpix_sphere-center_cell_columns.npz`
- `outputs/weight_benchmark/summary.json`
- `outputs/weight_benchmark/report.md`
- `outputs/weight_benchmark/weight_benchmark_arrays.npz`
- `outputs/weight_benchmark/plots/selected_cells_geometry.png`
- `outputs/weight_benchmark/plots/healpix_weight_components.png`
- `outputs/weight_benchmark/plots/healpix_molecular_shielding.png`
- `outputs/weight_benchmark/plots/scalar_weight_benchmark.png`

The density preview writes the same density and cone plots under
`outputs/density_preview/`.

## What To Inspect

Inspect `density_structure.png`, `cone_geometry_3d.png`, the three-variant
`X_CO`, `n_CO`, `theta_CO`, and `chi_eff` panels, and the weighted/uniform
ratio maps. The central slice should show the circle and downward-opening
cavity, and the 3-D view should show a true cone with its apex at the cloud
center.

For the angular benchmark, start with `selected_cells_geometry.png` and
`scalar_weight_benchmark.png`, then inspect the HEALPix maps. The cyan star in
each map marks the true source direction. In `healpix_weight_components.png`,
the direct-star and final angular maps give the percentage of the total local
UV assigned to each HEALPix pixel. Both use the same logarithmic color scale,
labelled in percentages, from `0.001%` to `100%`; values below the display floor
are not changed in the saved arrays. The stacked bar at right remains linear
and gives the integrated split between the star and the residual; the residual
is then distributed uniformly over all 192 pixels. The molecular map shows the
directional H2 and CO columns, their shielding factors, and each ray's
contribution to the weighted CO photodissociation rate.

The retained result locates the source in the correct maximum-weight pixel,
but the current scalar closure does not recover a pure point-source angular
field at the dense cavity wall. At the selected illuminated-wall cell it
assigns about 27% of the weight to the source and distributes about 73%
isotropically. This is a failure of the direct-source angular benchmark, not
evidence for physical diffuse UV, because scattering and external illumination
are disabled.

## Known Limitations

The central plot is a 2-D diagnostic slice of a 3-D Cartesian density field.
The `z` extent and resolution affect the true cone, the spherical cloud volume,
and the eventual HEALPix columns.

The point-source benchmark tests direct stellar attribution only. It cannot
determine the correct angular closure when scattering, diffuse emission,
multiple sources, or an external radiation field are present. Normalized
weights in cells where the retained scalar UV field is effectively zero are
also Monte Carlo-noise diagnostics rather than meaningful source fractions.
