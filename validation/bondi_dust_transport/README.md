# Bondi soft edge and finite-time dust transport

## Overview

This validation exercises the emergent Joos disc edge and the low-level
finite-time radial dust-transport solver on Bondi All Stars snapshot
`m05_9_dm_f140_c10`. The snapshot was selected because the earlier 504.8 au
hard cutoff produced an especially conspicuous discontinuity in the disc/ISM
dust ratio.

The run compares the production radially connected Joos weight with a
diagnostic control formed by multiplying that same weight by the old hard
radial gate. The production mask uses a transonic poloidal-Mach limit. It then
computes 15 logarithmic MRN size bins from 0.01 to 1000 um and evolves each for
50 kyr on a frozen, axisymmetrized gas background. The overview retains the
original three representative sizes, while cumulative plots sum complete bins
above `a_thresh = 0.1, 1, 10, 100 um`.

## Usage

From the repository root:

```bash
python validation/bondi_dust_transport/run.py
```

The local Bondi snapshot must be present under
`examples/bondi_all_stars/7_b_0.5_300_9_d2_small_dm`. Existing outputs are
preserved unless `--overwrite` is passed.

All generated plots, arrays, and summaries are written to the established
snapshot directory:
`examples/bondi_all_stars/plots/completed_uv_fields/m05_9_dm_f140_c10/`.
The validation directory contains only the driver and this README.

## Deep dive

The transport equation uses upwind pressure drift and conservative turbulent
diffusion of `Sigma_d / Sigma_g`. Gas radial advection is excluded, the inner
boundary absorbs inward dust, and no outer dust inflow is imposed. The outer
diffusion boundary assumes zero exterior settled-dust concentration, allowing
outwardly diffusing settled dust to leave the connected disc support. The frozen
gas background smooths `ln(Sigma_g)`, `ln(T)`, and `ln(P)` over 21 native radial
bins. Its pressure slope is restricted to
`-5 <= d ln(P) / d ln(R) <= -0.25`, preventing both outward pressure traps and
unresolved disc/ISM cliffs. Midplane density is reconstructed from the
smoothed pressure and temperature using the ideal-gas relation.

The calculation is deliberately one-dimensional and uses snapshot profiles to
isolate the finite-volume solver. The production model-level helper separately
projects native cells into cylindrical annuli about the measured Joos disc axis,
limits evolution to connected annular support, and reconstructs gained mass
from the coherent settled-disc template. This validation preserves the initial
azimuthal pattern when constructing its `R-phi` solver diagnostic and therefore
does not claim to model spiral, streamer, or vortex-driven transport.
The cumulative threshold profiles compare the initial and transported
azimuthally averaged columns. The companion `(R, phi)` figure shows the final
surface density without azimuthal averaging. Its radial evolution is still the
same 1-D transport factor at every azimuth.

## Result

The connected Joos weight has no hard cutoff at the 504.8-au axis aperture.
The transonic poloidal criterion prevents rapidly infalling or vertically
moving material from being classified as settled disc solely because it is
dense, bound, and rotating.

The reconstructed pressure is strictly decreasing, so the trial contains no
outer pressure traps and all positive-Stokes drift is inward. The smallest
grains remain closely coupled while progressively larger grains move inward
more strongly over the 50 kyr exposure. The 0.068 um bin remains near 121 au,
the 3.16 um bin moves from 121 to 111 au, and the 147 um bin moves from 119 to
40 au. The 681 um bin moves from 113 to 16 au and loses 3.0% of its mass through
the absorbing inner boundary. The smoothed outer background intentionally lies
above the nearly vacant snapshot column after the disc/ISM cliff, providing the
monotone frozen tail used by the parameterized transport model.

For the cumulative populations, `a > 0.1 um` moves from a mass-weighted mean
radius of 117 to 43 au, while `a > 100 um` moves from 116 to 24 au. The four
curves are not independent grain populations: each higher threshold is a subset
of the lower-threshold sum. Because the adopted MRN mass distribution is
weighted toward its largest grains, all four cumulative columns are strongly
influenced by the rapidly drifting upper end of the 1000 um distribution.

The main threshold outputs are:

- `dust_surface_density_above_size_threshold_profiles.png`: initial and final
  azimuthally averaged radial columns.
- `dust_surface_density_above_size_threshold_rphi.png`: final azimuthally
  resolved columns on a common logarithmic scale.
- `radial_transport_profiles.npz`: every per-bin and cumulative threshold
  profile and `(R, phi)` array.
- `summary.json`: bin definitions, threshold masses, and mean radii.

## References

- [Takeuchi & Lin (2002), radial flow and dust migration](https://arxiv.org/abs/astro-ph/0208552)
- [Youdin & Lithwick (2007), particle diffusion](https://arxiv.org/abs/0707.2975)
- [Birnstiel, Dullemond & Brauer (2010), dust evolution](https://arxiv.org/abs/1002.0335)
