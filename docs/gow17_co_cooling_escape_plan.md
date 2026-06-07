# Plan: Geometry-Aware CO Cooling Escape in GOW17

## Motivation

The current full 3D workflow uses GOW17's local CO cooling escape approximation with scalar defaults:

```python
gradv = 1.0e-14      # s^-1
Leff_CO_max = 3e20   # cm
```

These values are broadcast to every cell unless the user explicitly selects special modes. That is not a good default for a disk + ISM grid because the CO cooling optical depth depends on the local geometry, velocity field, and microturbulence.

The HEALPix shielding calculation already traces rays through the 3D grid. We should reuse that traversal to derive the quantities needed by CO cooling instead of doing a separate ray pass.

## Current Behavior

For photochemistry, the HEALPix shielding path computes directional columns:

- `N_H2(ray)`
- `N_CO(ray)`
- `N_C(ray)`
- optionally `N_CO * b_CO^2(ray)` for microturbulence-weighted CO shielding

These are used for H2/CO/C shielding and UV-weighted directional averages.

For CO rotational cooling, the C++ GOW17 solver does not use those shielding columns. With `isCoolingCOThin=False`, it computes:

```cpp
vth = sqrt(2 k_B T / m_CO)
nCO = nH * xCO
grad_small = vth / Leff_CO_max
gradeff = max(gradv, grad_small)
NCOeff = nCO / gradeff
```

So the current full workflow does not use the 3D CO column map for CO cooling. It uses a local LVG-style approximation with scalar `gradv` and scalar `Leff_CO_max`.

## Desired Behavior

For 3D GOW17 runs, `gradv` and `Leff_CO_max` should be derived automatically from the model:

- `Leff_CO_max` should come from directional path lengths through the grid.
- `gradv` should come from the velocity field plus the microturbulence field.
- The expensive HEALPix geometry/ray traversal should be reused from the shielding calculation.
- The workflow should not expose `gradv`, `gradv_mode`, `Leff_CO_max`, or `Leff_CO_max_mode`.
- `isCoolingCOThin` should not be exposed in the workflow. The physical default should be the non-thin escape approximation, using derived per-cell arrays.

## Target User-Facing Model

The workflow config should not contain low-level CO cooling escape switches.

Acceptable high-level GOW17 config:

```python
GOW17_CONFIG = {
    "mode": "equilibrium",
    "temperature": {"mode": "computed"},
    "enable_co_phase": True,
    "ion_rate": "2e-16 1/s",
    "Zg": 1.0,
    "astrochem_t_end_yr": 1.0e8,
}
```

Internally, 3D GOW17 should behave as if:

```python
co_cooling_escape = "healpix"
```

but the user should not need to set this in the normal full workflow.

## Implementation Plan

### 1. Introduce a Small Internal Data Object

Add an internal result container for HEALPix-derived shielding and CO-cooling quantities.

Proposed location:

```text
src/diskbridge/chemistry/shielding/healpix_columns.py
```

Possible shape:

```python
@dataclass
class PDRRayProducts:
    candidate_idx: np.ndarray
    dirs: np.ndarray
    theta_h2: np.ndarray
    theta_co: np.ndarray
    theta_c: np.ndarray
    theta_pdr: np.ndarray
    chi_eff_pdr: np.ndarray
    Leff_CO_max: np.ndarray | None = None
    gradv: np.ndarray | None = None
```

This keeps the existing tuple-returning function from becoming unreadable. After this is in place, the old tuple return can be removed if we are not preserving backward compatibility.

### 2. Compute Path Lengths During the Shielding Ray Pass

The existing `integrate_rays_with_pathlength()` computes path lengths by integrating a field of ones. That works, but if called separately it repeats ray traversal.

Instead, when `compute_pdr_shielding_healpix()` builds the `fields` dict, include a path-length field:

```python
fields["path_length"] = np.ones_like(nH_cgs)
```

Then the same call to `compute_column_rays_healpix()` returns:

```python
S_rays = cols["path_length"]   # cm
```

No second HEALPix traversal is needed.

### 3. Reduce Directional Path Lengths to `Leff_CO_max`

For each candidate cell, reduce `S_rays[cell, direction]` to one escape length.

Initial reducer:

```python
Leff_CO_max = percentile_20(S_rays, axis=direction)
```

Reasoning:

- Cooling photons preferentially escape along shorter optical-depth paths.
- A mean path length is probably too large in flattened disk geometry.
- A minimum path length may be too noisy and grid-sensitive.
- A low percentile is a simple compromise.

Keep the reducer internal for now:

```python
CO_COOLING_ESCAPE_LENGTH_REDUCTION = "percentile_20"
```

Do not expose this in `run_workflow.py`.

Clamp only if physically necessary, and make the clamp explicit:

```python
L_min = local cell scale
L_max = model-scale path upper bound
```

Avoid silent fallback values. If path lengths are non-finite or non-positive, raise an error.

### 4. Compute a Ray-Based Velocity Decorrelation Scale

For each cell and HEALPix direction, compute the line-of-sight velocity difference:

```python
v_los_cell = dot(v_cell, ray_dir)
v_los_far = representative velocity along ray
dv_los = abs(v_los_far - v_los_cell)
```

The first implementation does not need full line profile integration. A practical version is:

1. During ray integration, also integrate:

   ```python
   N_H(ray) = integral n_H ds
   V_H(ray) = integral n_H * v_los ds
   ```

2. Compute the density-weighted mean line-of-sight velocity along each ray:

   ```python
   v_los_mean_ray = V_H(ray) / N_H(ray)
   dv_los_ray = abs(v_los_mean_ray - v_los_cell)
   ```

3. Combine with microturbulence:

   ```python
   b_turb_ray = density_or_CO_weighted_microturbulence(ray)
   dv_eff_ray = sqrt(dv_los_ray**2 + b_turb_ray**2)
   ```

4. Convert to a directional velocity gradient:

   ```python
   gradv_ray = dv_eff_ray / S_rays
   ```

This is not a full radiative transfer escape probability, but it is a real grid-derived velocity decorrelation estimate and is much better than a scalar constant.

### 5. Choose the Weighting for Velocity and Microturbulence

For CO cooling, the most relevant weighting is CO-weighted, not total-H weighted, because the cooling opacity comes from CO.

Use:

```python
N_CO(ray) = integral n_CO ds
V_CO(ray) = integral n_CO * v_los ds
B2_CO(ray) = integral n_CO * b_turb^2 ds
```

Then:

```python
v_los_mean_ray = V_CO(ray) / N_CO(ray)
b_turb_ray = sqrt(B2_CO(ray) / N_CO(ray))
```

This matches the earlier choice for CO shielding linewidth: use CO weighting and do not add branchy fallback behavior. If `N_CO` is tiny, the normalization still represents the CO-weighted ray value. If it becomes numerically invalid, raise an error rather than hiding it.

### 6. Reduce Directional `gradv_ray` to Per-Cell `gradv`

The directional values should be reduced to one per-cell value for the existing C++ GOW17 API.

Initial reducer:

```python
gradv = weighted_harmonic_mean(gradv_ray, weights=escape_weights)
```

Possible escape weights:

```python
escape_weights = 1 / max(N_CO(ray), floor)
```

or reuse the directional UV weights if we want consistency with the shielding angular weighting.

The first implementation should be conservative and simple:

```python
gradv = percentile_20(gradv_ray)
```

Then document that this is an escape-gradient proxy. Avoid pretending it is exact.

### 7. Wire the Derived Arrays into GOW17

In `src/diskbridge/chemistry/models/gow17.py`, replace the current scalar default setup:

```python
Leff_CO_max_arr = _broadcast_scalar_or_array(Leff_CO_max_scalar, ncells)
gradv_arr = _broadcast_scalar_or_array(gradv_scalar, ncells)
```

with:

```python
if 3D HEALPix shielding is active:
    ray_products = compute_pdr_shielding_healpix(..., compute_co_cooling_escape=True)
    gradv_arr = ray_products.gradv.reshape(ncells)
    Leff_CO_max_arr = ray_products.Leff_CO_max.reshape(ncells)
else:
    use the existing 1D/local path only
```

The fallback should not be silent. If a 3D model lacks velocity or microturbulence fields needed for geometry-aware escape, raise a clear error.

### 8. Remove User-Facing Legacy Knobs

Once geometry-aware escape is the default for 3D GOW17:

- Remove `isCoolingCOThin` from `examples/leon_full_test/run_workflow.py`.
- Remove `gradv`, `gradv_mode`, `gradv_q`, `gradv_N0`, `gradv_p`, `gradv_f_corr`, `gradv_gmin`, `gradv_gmax` from workflow-facing config.
- Remove `Leff_CO_max`, `Leff_CO_max_mode`, `L_geo_reduction`, `L_geo_min`, `L_geo_max` from workflow-facing config.
- Keep any internal constants in one place in the GOW17 implementation, not in the example workflow.

If optically thin CO cooling is still useful for tests, expose it only through a test/debug path, not the scientific workflow.

### 9. Update Metadata

The GOW17 output metadata should record what was actually used:

```json
{
  "co_cooling_escape": {
    "method": "healpix",
    "Leff_CO_max_reduction": "percentile_20",
    "gradv_reduction": "percentile_20",
    "velocity_weighting": "CO",
    "microturbulence_weighting": "CO",
    "min_gradv_s^-1": "...",
    "max_gradv_s^-1": "...",
    "min_Leff_CO_max_cm": "...",
    "max_Leff_CO_max_cm": "..."
  }
}
```

This gives us reproducibility without making the workflow carry low-level knobs.

### 10. Validation

Start with small controlled checks before running the full Leon model:

1. Uniform static slab:
   - `dv_los` should be zero.
   - `gradv` should be set by microturbulence/path length.
   - `Leff_CO_max` should match the expected directional path scale.

2. Uniform expanding/compressing box:
   - `gradv` should recover the imposed velocity gradient approximately.

3. Rotating disk toy model:
   - `gradv` should be larger where shear is larger.
   - `Leff_CO_max` should be shorter near disk surfaces than through the midplane.

4. Full Leon downsample:
   - Compare old scalar escape versus new geometry-aware escape.
   - Inspect maps of `gradv`, `Leff_CO_max`, `Tgas`, CO gas density, and CO ice density.

Do not add broad trivial unit tests. Add focused tests only around the ray-product calculations and reducer behavior.

## Expected Code Simplification

This change should remove conceptual bloat from the workflow:

- No workflow `isCoolingCOThin`.
- No workflow `gradv` or `Leff_CO_max`.
- No user-visible mode switches for CO cooling escape.

It may add some code inside `healpix_columns.py`, but that is the right place because the calculation belongs to the ray products. The net user-facing API should become simpler.

## Open Modeling Choices

These need explicit decisions before implementation:

1. Should `Leff_CO_max` use `percentile_20`, harmonic mean, or another escape-weighted reduction?
2. Should `gradv` use CO-weighted velocity, H-weighted velocity, or local finite-difference velocity gradients?
3. Should microturbulence enter as local-cell only or CO-column-weighted along each ray?
4. Should the velocity decorrelation use ray-mean velocity or a more local coherence-length estimate?

My recommendation for the first implementation:

```text
Leff_CO_max: percentile_20(path_length_rays)
gradv: percentile_20(sqrt(dv_los_CO_weighted^2 + b_turb_CO_weighted^2) / path_length_rays)
velocity weighting: CO
microturbulence weighting: CO
```

That is simple, uses the existing HEALPix data naturally, and avoids pretending the scalar GOW17 default knows the 3D geometry.
