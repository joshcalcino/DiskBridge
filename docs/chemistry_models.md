# Chemistry Models in DiskBridge

This document inventories the chemistry models in [src/diskbridge/chemistry/models/](../src/diskbridge/chemistry/models/), the public functions each one defines, and where each function is used.

The three user-facing models are wired into the chemistry registry at [registry.py](../src/diskbridge/chemistry/registry.py):

| Registry name              | Module                                                                                       | Entry point             |
| -------------------------- | -------------------------------------------------------------------------------------------- | ----------------------- |
| `pinte_switches`           | [models/pinte_switches.py](../src/diskbridge/chemistry/models/pinte_switches.py)             | `run_pinte_switches`    |
| `layered_column_switches`  | [models/abundance_switches.py](../src/diskbridge/chemistry/models/abundance_switches.py)     | `run_abundance_switches`|
| `gow17`                    | [models/gow17.py](../src/diskbridge/chemistry/models/gow17.py)                               | `run_gow17`             |

One additional internal module supports the registered models:
- [models/gow17_timestep.py](../src/diskbridge/chemistry/models/gow17_timestep.py) — single-`dt` time-stepper class for the GOW17 network, used by an out-of-tree validation driver.

> **Note:** The former `carbon_reduced` model and its supporting modules (`_carbon_reduced_math`, `chemistry/processes/`, `chemistry/tau_form.py`) were removed on 2026-04-29. `gow17` with `enable_co_phase=True` covers the same physics. The original code is preserved locally (gitignored) at [`legacy/carbon_reduced.py`](../legacy/carbon_reduced.py) along with the `evolve_co_time_dependent_infall_age` workflow it powered.

In every "Used" column below, a function marked **internal** is only called from within its own module. Cross-references are listed explicitly.

---

## 1. `pinte_switches` — [models/pinte_switches.py](../src/diskbridge/chemistry/models/pinte_switches.py)

Pinte-style "switch" abundance model. Starts with a constant abundance `X0` and applies multiplicative switches for freeze-out, photodesorption escape, and photodissociation based on temperature and `χ/n_H` thresholds (with optional smooth transitions).

| Function | Purpose | Used by |
| --- | --- | --- |
| [`run_pinte_switches`](../src/diskbridge/chemistry/models/pinte_switches.py#L19) | Registry entry point. Pulls `T`, `n_H`, `χ` from the `RadModel`, calls `compute_abundance_pinte`, packages the result. | Registry — `pinte_switches` callable. Looked up only by [registry.py:11](../src/diskbridge/chemistry/registry.py#L11). |
| [`compute_abundance_pinte`](../src/diskbridge/chemistry/models/pinte_switches.py#L71) | Applies freeze-out, photodesorption escape, and photodissociation switches in sequence, returning `X` and `n_mol`. | **Internal** (called by `run_pinte_switches`). |
| [`compute_freezeout_factor`](../src/diskbridge/chemistry/models/pinte_switches.py#L162) | Returns the freeze-out multiplicative factor and a freeze mask. Supports a smoothed transition over `smooth_Tfrz_K`. | Internal **and** [`abundance_switches.py:15`](../src/diskbridge/chemistry/models/abundance_switches.py#L15) — `compute_abundance_layered_column` reuses it. |
| [`apply_photodesorption_escape`](../src/diskbridge/chemistry/models/pinte_switches.py#L194) | Restores `X` toward `X0` in cells where `χ/n_H` exceeds the photodesorption threshold (with optional smoothing). | **Internal**. |
| [`apply_photodissociation`](../src/diskbridge/chemistry/models/pinte_switches.py#L240) | Zeroes (or smoothly damps) `X` in cells where `χ/n_H` exceeds the photodissociation threshold. | **Internal**. |
| [`_smoothstep01`](../src/diskbridge/chemistry/models/pinte_switches.py#L264) | Cubic smoothstep `3t² − 2t³` clamped to `[0, 1]`. | **Internal** (used by all three switch functions in this module). |

---

## 2. `layered_column_switches` — [models/abundance_switches.py](../src/diskbridge/chemistry/models/abundance_switches.py)

Variant on the Pinte switches that uses a vertical column density `N_H` (computed via a HEALPix `nside=1` ray to `+ẑ`) to gate photodesorption and photodissociation, instead of the local `χ/n_H` ratio.

| Function | Purpose | Used by |
| --- | --- | --- |
| [`run_abundance_switches`](../src/diskbridge/chemistry/models/abundance_switches.py#L23) | Registry entry point (`layered_column_switches`). Calls `compute_abundance_layered_column`. | Registry — looked up only by [registry.py:12](../src/diskbridge/chemistry/registry.py#L12). |
| [`compute_vertical_cd_cm2`](../src/diskbridge/chemistry/models/abundance_switches.py#L63) | Builds an `nside=1` HEALPix tracer and integrates `n_H` along `+ẑ` to give a per-cell vertical column. | **Internal** (called by `compute_abundance_layered_column`). |
| [`compute_abundance_layered_column`](../src/diskbridge/chemistry/models/abundance_switches.py#L104) | Applies freeze-out (via `compute_freezeout_factor`), then column-density-gated photodesorption (only inside frozen cells) and photodissociation. | **Internal**. |

This module imports [`compute_freezeout_factor`](../src/diskbridge/chemistry/models/pinte_switches.py#L162) from `pinte_switches` — the only cross-model dependency among the registered chemistries.

---

## 3. `gow17` — [models/gow17.py](../src/diskbridge/chemistry/models/gow17.py)

Full Gong, Ostriker & Wolfire (2017) reduced reaction network. Wraps the C/CVODE batch solver in [`diskbridge._gow17`](../src/diskbridge/_gow17.cpp) and iterates abundances against shielding factors. Supports two coupling modes (`fixed_point`, `astrochem`), an optional 1-D slab equilibrium fast path, and optional Aitken acceleration of the macro-iteration. With `enable_co_phase=True` it also evolves CO ice (freeze-out, thermal desorption, photodesorption).

| Function | Purpose | Used by |
| --- | --- | --- |
| [`run_gow17`](../src/diskbridge/chemistry/models/gow17.py#L363) | Registry entry point. Sets up arrays, runs the shielding/chemistry coupling loop, returns `ChemistryResult`. | Registry — looked up only by [registry.py:13](../src/diskbridge/chemistry/registry.py#L13). |
| [`_cv_cold`](../src/diskbridge/chemistry/models/gow17.py#L80) | Cold-gas heat capacity per H nucleus (erg/K/H). | Internal **and** [`gow17_timestep.py:44`](../src/diskbridge/chemistry/models/gow17_timestep.py#L44) **and** [`validation/reproduce_gow17_fig2_3dhealpix.py:26`](../validation/reproduce_gow17_fig2_3dhealpix.py). |
| [`_electron_abundance`](../src/diskbridge/chemistry/models/gow17.py#L85) | Sums all ion abundances → `xe`. | Internal **and** [`gow17_timestep.py:45`](../src/diskbridge/chemistry/models/gow17_timestep.py#L45). |
| [`_apply_astrochem_aitken_acceleration`](../src/diskbridge/chemistry/models/gow17.py#L100) | Guarded scalar Aitken acceleration on `H₂`, `CO`, and (if enabled) `CO_ice`. | **Internal** (called only inside `run_gow17`'s `astrochem` branch). |
| [`_compute_shielding_and_gph`](../src/diskbridge/chemistry/models/gow17.py#L181) | From the current abundance state, computes `θ_H₂`, `θ_CO`, `θ_C` and assembles per-cell radiation field arrays `Gph`, `GPE`, `GISRF`. Dispatches between 1-D, HEALPix, and slab paths. | Internal **and** [`gow17_timestep.py:49`](../src/diskbridge/chemistry/models/gow17_timestep.py#L49). |
| [`_as_cgs_f64`](../src/diskbridge/chemistry/models/gow17.py#L164) | `Quantity → contiguous float64 cgs ndarray`. | Internal **and** re-exported to [`gow17_timestep.py:46`](../src/diskbridge/chemistry/models/gow17_timestep.py#L46). |
| [`_broadcast_scalar_or_array`](../src/diskbridge/chemistry/models/gow17.py#L168) | Broadcasts scalar or 1-element-array config values to length `ncells`. | Internal **and** [`gow17_timestep.py:47`](../src/diskbridge/chemistry/models/gow17_timestep.py#L47). |
| [`_maybe_quantity_to_float`](../src/diskbridge/chemistry/models/gow17.py#L175) | Parses a string `Quantity` or scalar to a CGS float in the requested unit. | Internal **and** [`gow17_timestep.py:48`](../src/diskbridge/chemistry/models/gow17_timestep.py#L48). |

`gow17` also re-exports a long list of integer indices and constants (`N_Y`, `N_PH`, `IPH_C`, … `I_E`, `XC_STD`, `XHE`) from the C extension. These are imported by `gow17_timestep.py` but are not functions.

---

## 4. `gow17_timestep` — [models/gow17_timestep.py](../src/diskbridge/chemistry/models/gow17_timestep.py)

A class-based wrapper around the GOW17 batch solvers designed for **external time-dependent** problems where `χ`, `T_dust`, etc. change every step. Not in the registry — instantiated directly by callers.

| Function/class | Purpose | Used by |
| --- | --- | --- |
| [`_build_abstol`](../src/diskbridge/chemistry/models/gow17_timestep.py#L56) | Per-species absolute tolerance vector for CVODE. | **Internal**. |
| [`_default_y0_single`](../src/diskbridge/chemistry/models/gow17_timestep.py#L72) | Default single-cell initial abundance vector. | **Internal**. |
| [`Gow17TimeStepper`](../src/diskbridge/chemistry/models/gow17_timestep.py#L83) (class) | Holds persistent state (`y_state`, prior `θ` factors, broadcasted config arrays) and exposes `.step(dt_s)`, `.solve_equilibrium()`, `._prepare_environment`, `._commit_solution`, `._repair_failed_cells`. | **No `src/` callers.** Only used by [`validation/gow17_infall_stream_1d/driver.py`](../validation/gow17_infall_stream_1d/driver.py). |

---

## Summary of cross-module / external usage

The vast majority of model-internal helpers are exactly that — internal. The cross-module edges are:

1. **`pinte_switches.compute_freezeout_factor`** is reused by `abundance_switches.compute_abundance_layered_column`.
2. **`gow17` low-level helpers** (`_cv_cold`, `_electron_abundance`, `_compute_shielding_and_gph`, `_as_cgs_f64`, `_broadcast_scalar_or_array`, `_maybe_quantity_to_float`) are re-imported by `gow17_timestep.py` so the two modules share state-vector conventions.
3. **`gow17._cv_cold`** is also imported by [`validation/reproduce_gow17_fig2_3dhealpix.py`](../validation/reproduce_gow17_fig2_3dhealpix.py).
4. **`Gow17TimeStepper`** has no `src/` callers; only [`validation/gow17_infall_stream_1d/driver.py`](../validation/gow17_infall_stream_1d/driver.py) uses it.

## Broken-on-purpose after the carbon_reduced removal

The following files reference the deleted `carbon_reduced` model / `_carbon_reduced_math` module / `chemistry.processes` / `chemistry.tau_form` and will fail at import time. They have not been updated:

- [`validation/run_physics_checks.py`](../validation/run_physics_checks.py)
- [`validation/carbon_reduced_slab.py`](../validation/carbon_reduced_slab.py)
- [`validation/freezeout_slab.py`](../validation/freezeout_slab.py)
- [`validation/thermochem_slab.py`](../validation/thermochem_slab.py)
- [`validation/cube_test/run.py`](../validation/cube_test/run.py)
- [`examples/thermal_balance_example.py`](../examples/thermal_balance_example.py)
