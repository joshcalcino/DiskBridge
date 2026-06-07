# HEALPix Shielding

This document covers DiskBridge's HEALPix-based UV shielding stack:
ray geometry, per-ray shielding factors (CO, H2, C), and the directional UV
weighting that turns per-direction shielding factors into a single effective
factor per cell.

The whole machinery is built around one expression. For each chemistry cell:

```text
theta_eff = sum_k W(k) * theta(k)
```

The two factors describe distinct physics:

```text
theta(k) — per-ray shielding factor, set by the molecular/dust columns
           accumulated through the 3D density field along HEALPix direction k.
W(k)     — per-ray UV weight, set by where we estimate UV photons arrive from.
```

The density anisotropy lives entirely inside `theta(k)`: rays through dense
media see large CO/H2 columns and small `theta`; rays into a cavity see small
columns and `theta` close to 1. `W(k)` describes something separate: how much
UV is incident from each direction. It is the only place the star geometry
enters.

## Key files

- [angular_uv_weights.py](../src/diskbridge/chemistry/shielding/angular_uv_weights.py)
  — builds `C_ext`, `C_star`, `C_iso`, normalizes `W_rays`, and deposits the
  attenuated stellar contribution into `k_star`.
- [healpix_columns.py](../src/diskbridge/chemistry/shielding/healpix_columns.py)
  — per-ray shielding factors `theta(k)` and the final
  `sum(W_rays * theta_rays)` reduction.
- [healpix_utils.py](../src/diskbridge/chemistry/shielding/healpix_utils.py)
  — Numba DDA kernels for ray marching, plus `integrate_starward_rays_multi`.
- [dust_uv_tau.py](../src/diskbridge/chemistry/shielding/dust_uv_tau.py)
  — converts per-bin dust mass columns into UV optical depth using
  band-averaged extinction opacities from `dustkappa_*.inp`.
- [visser_shielding.py](../src/diskbridge/chemistry/shielding/visser_shielding.py)
  — Visser+09 CO and H2 shielding tables.
- [w_rays_cache.py](../src/diskbridge/chemistry/shielding/w_rays_cache.py)
  — disk-backed cache of computed `W_rays` arrays.

## Ray geometry and integration

The sky is tessellated into `Npix = 12 * nside^2` equal-area HEALPix pixels
(Gorski et al. 2005). Each pixel center defines a unit direction `n_hat(k)`.
A typical disc-scale problem uses `nside = 4` (192 directions) or `nside = 8`
(768 directions).

Two ray tracers are selected automatically from the mesh coordinate system:

- **`SphericalHealpixRayTracer`** — spherical `(r, theta, phi)` meshes. Uses
  log-uniform radial indexing and uniform angular indexing for fast cell
  lookup.
- **`CartesianHealpixRayTracer`** — Cartesian `(x, y, z)` meshes. Uses uniform
  grid indexing.

Both use a DDA (Digital Differential Analyzer) stepping algorithm in
Numba-compiled kernels. The step size is set as a fraction of the minimum
cell size (`ds_fraction`, default 0.5).

From each cell center, rays march outward along each HEALPix direction until
they exit the domain boundary, accumulating column densities:

```text
N_X(n_hat(k)) = integral_0^s_boundary n_X(s) ds
```

The starting cell contributes with a fixed weight `HEALPIX_SELF_WEIGHT = 0.5`,
approximating a half-cell path from the cell center. 

Multiple fields are integrated simultaneously via
`compute_column_rays_healpix(...)`, which stacks fields and runs a single
ray-marching pass per cell-direction pair.

## Per-ray shielding factors theta(k)

Per-ray CO and H2 shielding factors are evaluated from the columns
accumulated along each HEALPix ray:

```text
theta_CO(k) = theta_CO( N_CO(k), N_H2(k), b )    (Visser+09)
f_sh_H2(k)  = f_sh_H2( N_H2(k), b )              (Draine & Bertoldi 1996)
```

`b` is the Doppler parameter (km/s). The Visser+09 tables account for both CO
self-shielding and mutual shielding by H2 line overlap.

The framework is species-agnostic: any per-ray factor `f(k)` (CO, H2, C, C+,
future terms) can be passed through the same weighted sum. Adding a new
shielding term requires only producing an `f(k)` array — no new averaging
logic.

References:

- Visser, van Dishoeck & Black 2009, A&A 503, 323 — CO shielding tables.
- Draine & Bertoldi 1996, ApJ 468, 269 — H2 self-shielding.
- Draine 1978, ApJS 36, 595 — ISRF normalization.

## What "isotropic UV" actually means

If the local UV field really is isotropic, the correct weights are uniform:

```text
W(k) = 1/Npix    =>    theta_eff = mean_k theta(k)
```

A worked example helps. Take a cell in a disk midplane with three
representative directions:

```text
direction      column                      theta(k)
midplane       thick CO/H2 column          ~1e-6     strong shielding
disk surface   moderate column             ~1e-2     weak shielding
polar cavity   tiny column                 ~1e-1     almost unshielded
```

With isotropic UV, every direction delivers the same incoming flux. The
fraction of that flux reaching the cell is just the average over directions:

```text
theta_eff = (1e-6 + 1e-2 + 1e-1 + ...) / Npix
```

The density anisotropy still shapes this number — the midplane direction
contributes ~1e-6, the cavity direction ~1e-1 — but each direction is
weighted equally because each direction delivers equal UV.

Now imagine using `W(k) ∝ exp(-tau_ext(k))` instead. That says the cavity
direction delivers *more* UV than the midplane direction (because the
dust-thin path lets external UV through). Applied to a field that's actually
isotropic — e.g., one dominated by RADMC-3D's scattered/diffuse residual —
the weighted average overweights the high-`theta` cavity direction and
returns a `theta_eff` that is too large (too little shielding). The
directional weighting is a statement about the UV field's anisotropy; if the
field isn't anisotropic, applying directional weights biases the answer.

When `W_rays=None` is passed to the shielding routines, the code falls back to
the uniform mean.

## Building W(k): three components

For each cell, DiskBridge constructs an unnormalized directional UV
contribution map from three pieces (see
[angular_uv_weights.py](../src/diskbridge/chemistry/shielding/angular_uv_weights.py)):

```text
C(k) = C_ext(k) + C_star(k) + C_iso
W(k) = C(k) / sum_j C(j)
```

If `sum_j C(j) == 0` (a fully dark cell, rare), the code falls back to
uniform weights.

### External direct field

```text
C_ext(k) = chi_ext0 * exp(-tau_ext(k))
```

`tau_ext(k)` is the dust UV optical depth from the cell to the domain
boundary along direction `k`. It is computed by integrating per-bin dust mass
columns and converting via dustkappa opacities:

```text
tau_ext(k) = sum_bin kappa_ext_UV[bin] * Sigma_dust_bin(k)
```

`chi_ext0` is the unattenuated boundary UV in Draine units (set to 0 if no
external field is enabled).

The scalar direct-external contribution at the cell is the pixel mean:

```text
chi_ext_dir = mean_k C_ext(k)   (scalar per cell, used in chi_iso below)
```

### Stellar direct field

The star is treated as a point source at the mesh origin. From a cell at
position `r_vec`, the incoming stellar direction is

```text
omega_star = -r_vec / |r_vec|
k_star     = hp.vec2pix(nside, omega_star)
```

The starward ray (next section) gives the dust UV optical depth `tau_star`
along that line. The unattenuated stellar Draine factor at radius `r` from
geometric dilution is

```text
F_uv = L_uv / (4 pi r^2)

chi_star_unatt = F_uv / (c * U_ref)   (energy weighting; default)
chi_star_unatt = F_uv / U_ref         (photon weighting)
```

with `U_ref` the Draine reference energy density (or photon flux) in the same
UV band. The factor of `c` in the energy branch converts flux to energy
density before the ratio. `L_uv` is computed from the same stellar and
accretion sources used to write `stars.inp` for RADMC-3D (photosphere plus
optional accretion hotspot), integrated over the same UV band as
`chi_radmc` for consistency.

After attenuation,

```text
chi_star_att = chi_star_unatt * exp(-tau_star)
```

This is then deposited into a single pixel `k_star`, with an `Npix` factor so
the discretization is independent of `nside`:

```text
C_star(k_star) = Npix * chi_star_att
C_star(other)  = 0

mean_k C_star(k) = (1/Npix) * Npix * chi_star_att = chi_star_att
```

Without that `Npix` factor, the pixel-mean stellar contribution would
understate the true direct stellar UV by a factor of `Npix` and results would
depend on the angular resolution.

### Isotropic residual

DiskBridge already has a scalar UV field from RADMC-3D, `chi_radmc`, derived
from the UV-band integral of the mean intensity. It represents the total
local UV including scattered and diffuse contributions. The residual that is
not explained by the directly attenuated external and stellar fields is
treated as isotropic:

```text
chi_iso = max(chi_radmc - chi_ext_dir - chi_star_att, 0)
C_iso   = chi_iso        (same value in every pixel)
```

`chi_ext_dir` here is a scalar — the equal-pixel-mean of the directionally
attenuated external field — used as a bookkeeping subtraction so the
isotropic piece only carries what the direct fields don't already account
for.

## What the starward ray actually does

The starward ray exists for one purpose: to compute `tau_star`, the dust UV
optical depth along the cell-to-star line, so the stellar direct beam can be
properly attenuated before being deposited into `k_star`.

```text
Sigma_dust(bin) = integral rho_dust(bin) ds   along cell -> origin
tau_star        = sum_bin kappa_ext_UV(bin) * Sigma_dust(bin)
```

On spherical meshes this is an exact radial inward sum to the inner boundary.
On Cartesian meshes, direct stellar attenuation is currently unsupported
(see the Cartesian limitation section below).

The starward ray feeds only into `W(k_star)`; all `theta(k)` values come
from the boundary rays.

## When directional weighting matters

The four limiting cases are the cleanest way to keep things straight:

```text
Case 1 — isotropic UV, isotropic density
  theta(k) ~ const, W(k) = 1/Npix.
  Reduces to a scalar shielding factor; directional machinery is moot.

Case 2 — isotropic UV, anisotropic density
  theta(k) varies, W(k) = 1/Npix.
  theta_eff is the equal-solid-angle mean of the per-ray shielding factors.
  The star direction gets no special treatment.

Case 3 — anisotropic UV, isotropic density
  W(k) varies, theta(k) ~ const.
  The weighting has little effect because all directions shield the same.

Case 4 — anisotropic UV, anisotropic density
  Both vary. This is where directional weighting genuinely matters.
  A bright UV direction through a thin column gives weak shielding;
  a bright UV direction through a thick column gives strong shielding.
```

The starward treatment is only load-bearing in cases 3 and 4 — and only to
the extent that the attenuated direct stellar beam contributes meaningfully
to the local UV budget. In a cell where `chi_iso` dominates `C(k)`, `W(k)`
collapses back to nearly uniform regardless of star geometry.

## Segmented outer-region override

The normal formula is `C(k) = C_ext(k) + C_star(k) + C_iso` and
`W(k) = C(k) / sum_j C(j)`. There is one exception in
`compute_uv_direction_weights_healpix(...)`: if an outer override radius is
set, cells with

```text
|cell_center| >= isotropic_outside_r_cm
```

have their `W_rays` rows replaced after the normal weights are computed. The
replacement mode is controlled by `outer_weight_mode`:

```text
outer_weight_mode = "uniform"
  W(k) = 1 / Npix

outer_weight_mode = "tau"
  W(k) proportional to exp(-tau_ext(k))
```

This is for segmented workflows where the outer/frozen region is deliberately
treated as an external-background problem instead of a direct-stellar
problem. It is not part of the ordinary starward UV model.

## Caching

HEALPix geometry (cell indices, direction vectors, cell centers) and
integrated column densities are cached to disk as `.npz` files when a
`cache_dir` is provided. Cache keys are computed from mesh geometry and field
hashes, so changing any of these triggers recomputation.

`W_rays` itself is cached separately via
[w_rays_cache.py](../src/diskbridge/chemistry/shielding/w_rays_cache.py).
Cached weights are reused across chemistry iterations as long as inputs
haven't changed; if you change UV parameters or dust opacities without
invalidating the cache, you will see stale weights.

## Usage

### Compute `W_rays`

```python
from diskbridge.chemistry.shielding.angular_uv_weights import (
    compute_uv_direction_weights_healpix,
)

# chi_radmc:     3D scalar UV field from RADMC (Draine units)
# dust_rho_bins: list of 3D arrays (g/cm^3), one per dust bin
# kext_uv:       (nbin,) UV extinction opacity (cm^2/g), band-averaged

W_rays, candidate_idx, dirs, cell_centers, debug = compute_uv_direction_weights_healpix(
    mesh=rad.model.mesh,
    chi_radmc=chi_radmc,
    nside=8,
    dust_rho_bins=dust_rho_bins,
    kext_uv=kext_uv,
    chi_ext0=params.chi_ext0,
    star_uv_luminosity_erg_s=L_uv,
    cache_dir=rad.cache_dir,
)

rad.W_rays = W_rays
```

### Apply `W_rays` to PDR shielding

```python
from diskbridge.chemistry.shielding.healpix_columns import compute_pdr_shielding_healpix

theta_h2, theta_co, theta_c, theta_pdr, chi_eff = compute_pdr_shielding_healpix(
    mesh=rad.model.mesh,
    nH=nH_cm3,
    chi=chi_radmc,
    visser=visser,
    nCO=nCO_cm3,
    nC=nC_cm3,
    nH2=nH2_cm3,
    nside=8,
    W_rays=rad.W_rays,
)
```

### CO only

```python
from diskbridge.chemistry.shielding.healpix_columns import compute_co_shielding_healpix
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding

visser = VisserShielding(b_kms=0.3)

theta_co, chi_eff = compute_co_shielding_healpix(
    mesh=rad.model.mesh,
    nH=nH_cm3,
    chi=chi_radmc,
    visser=visser,
    nCO=nCO_cm3,
    nH2=nH2_cm3,
    nside=4,
    b_kms=0.3,
    W_rays=rad.W_rays,    # optional; omit for uniform isotropic mean
)
```

`chi_eff = chi_radmc * theta_eff` is the effective photodissociating UV field.

## Diagnostics

When `keep_debug_arrays=True` is passed to
`compute_uv_direction_weights_healpix(...)`, the returned `debug` dict
contains per-cell arrays useful for validating the weight construction:

- `chi_ext_dir`, `chi_star_dir_att`, `chi_iso`, `chi_radmc_cand`
- `tau_ext_rays`, `tau_star`, `chi_star_unatt`, `k_star`
- `uv_ext_contrib`, `uv_star_contrib`, `uv_iso_contrib`, `uv_total_contrib`

Sanity checks worth running:

- `chi_iso >= 0` everywhere (by construction).
- `chi_ext_dir + chi_star_dir_att + chi_iso <= chi_radmc + eps`.
- Star-dominated cells: `W(k)` concentrated in `k_star`.
- External-dominated cells: `W(k)` tracks `exp(-tau_ext(k))`.
- Isotropic-dominated cells: `W(k)` approaches uniform.

## Cartesian direct-stellar UV is disabled

Cartesian meshes can still use HEALPix boundary rays, isotropic UV, and
direct external UV weighting. They cannot currently use the direct stellar
component in `W_rays`.

If `star_uv_luminosity_erg_s > 0` and the mesh is Cartesian,
`integrate_starward_rays_multi(...)` raises `NotImplementedError`. That is
intentional. A valid Cartesian implementation needs a true finite
cell-to-source segment with an explicit stop at the stellar surface or a
configured inner source radius, which the previous DDA path lacked (it
marched to grid exit).

Until that exists, use one of these options:

```text
Use a spherical star-centered mesh for direct stellar UV weights.
Set star_uv_luminosity_erg_s = 0.0 when building Cartesian W_rays.
Treat the remaining UV as isotropic/external according to the configured model.
```
