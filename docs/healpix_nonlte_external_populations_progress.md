# HEALPix Non-LTE External Populations — Implementation Progress

Plan: [healpix_nonlte_external_populations_plan.md](healpix_nonlte_external_populations_plan.md)

## Status: complete (initial implementation)

All planned modules are implemented and the test suite passes (16 unit + 2
end-to-end smoke tests, plus the existing 13 non-LTE staging tests still
green).

## Modules

All new code lives under [src/diskbridge/radmc3d/line_transfer/](../src/diskbridge/radmc3d/line_transfer/):

- [molecular_rates.py](../src/diskbridge/radmc3d/line_transfer/molecular_rates.py)
  — LAMDA parser + `MoleculeData`, LTE initializer, strict temperature-range
  policy, helpers to pack collider tables into Numba-friendly arrays.
- [escape_healpix.py](../src/diskbridge/radmc3d/line_transfer/escape_healpix.py)
  — Numba `@njit(parallel=True)` HEALPix beta kernels (spherical + Cartesian)
  with velocity-coherent Doppler overlap, DDA from
  `chemistry/shielding/healpix_utils.py`. Returns direction-averaged `beta` per
  candidate cell per line; chunked over candidate cells.
- [se_solver.py](../src/diskbridge/radmc3d/line_transfer/se_solver.py)
  — Numba parallel `compute_line_center_opacity`,
  `compute_collisional_rates`, `build_rate_matrix_and_rhs`; batched
  `np.linalg.solve` driven by `solve_statistical_equilibrium` (chunked by SE
  matrix memory).
- [writer.py](../src/diskbridge/radmc3d/writer.py)
  — `RadWriter.flatten_scalar_to_radmc_order`,
  `RadWriter.write_levelpop_dat`, and `RadWriter.write_levelpop`.
- [external_populations.py](../src/diskbridge/radmc3d/line_transfer/external_populations.py)
  — `HealpixSEConfig`, `solve_and_write_healpix_levelpop` (iteration driver,
  Cartesian-velocity conversion for spherical meshes, manifest).
- [staging.py](../src/diskbridge/radmc3d/line_transfer/staging.py)
  — `prepare_external_population_line_run` reuses the common RADMC input
  staging helpers plus `write_line_radmc3d_inp` and `write_lines_inp`.
- [external_validation.py](../src/diskbridge/radmc3d/line_transfer/external_validation.py)
  — `validate_external_population_run`: validates required files, levelpop
  format/level list, non-negativity, cell-count match, gas-temperature /
  microturbulence / gas-velocity sanity, and sum-over-levels equals
  `numberdens_<species>`.

Existing scalar `.binp` writers (microturbulence, gas velocity) are reused
from [src/diskbridge/radmc3d/writer.py](../src/diskbridge/radmc3d/writer.py)
when the upstream `RadModel` writes its inputs.

## Numba parallelism

Every compute-heavy kernel uses `@numba.njit(parallel=True, cache=True)` with
`prange` over candidate cells:

- `compute_beta_cartesian`, `compute_beta_spherical` — outer `prange` over
  candidates; per-cell thread-local `tau` / `beta_sum` scratch arrays;
  Doppler overlap and DDA walks are fully inside the JIT.
- `compute_line_center_opacity`, `compute_collisional_rates`,
  `build_rate_matrix_and_rhs`, `_lte_populations_flat`,
  `_scatter_alpha0_to_full`, `_scatter_levelpop_to_full` — all parallel over
  candidate cells.

## Public API

```python
from diskbridge.radmc3d.line_transfer import (
    HealpixSEConfig,
    solve_and_write_healpix_levelpop,
    prepare_external_population_line_run,
    validate_external_population_run,
)
```

## Tests

- [tests/radmc3d/test_external_populations.py](../tests/radmc3d/test_external_populations.py)
  - LAMDA parser: CO levels/lines/colliders parsed correctly.
  - LTE: sum=1, Boltzmann ratio recovered.
  - `beta_of_tau` small- and large-tau limits.
  - Cartesian HEALPix kernel: optically thin -> beta=1, monotone decrease
    with alpha0.
  - SE: very high collider density -> LTE recovered; negligible collisions +
    CMB-only -> radiative equilibrium with Tbg.
  - `RadWriter.write_levelpop_dat` format + rejection of negatives.
  - Validation: well-formed dir accepted; lines_mode mismatch, missing
    microturbulence, negative populations, and sum mismatch all rejected.
  - Strict temperature-range policy raises out of range, ignores
    non-candidate cells.
- [tests/radmc3d/test_external_population_workflow.py](../tests/radmc3d/test_external_population_workflow.py)
  - End-to-end: small Cartesian 4^3 grid + real CO LAMDA file -> writes
    valid levelpop file (sum-over-levels matches species density).
  - Stage + validation an external-population run.

```
$ python -m pytest tests/radmc3d/test_external_populations.py \
                    tests/radmc3d/test_external_population_workflow.py \
                    tests/radmc3d/test_nonlte.py
38 passed
```

## Deliberate deviations from the plan

- The `np.linalg.solve` call uses the broadcast form `solve(M, b[..., None])[..., 0]`
  to remain compatible with NumPy >= 2.0 (which no longer treats
  `(N, m)` RHS as a stack of column vectors).
- The microturbulence writer in the plan is replaced by reusing the existing
  `RadWriter.write_microturbulence`. DiskBridge's `microturbulence` field now
  follows RADMC-3D's convention directly: it is the turbulent line-width
  parameter `a_turb` used in `sqrt(a_turb**2 + 2 k T / m_mol)`.
- Symlink-pattern edits in `radmc3d/model.py` / `radmc3d/image.py` are not
  needed for the staged-run-directory path: `prepare_external_population_line_run`
  stages files directly into `radmc3d_inputs/` including `levelpop_*.dat`
  and `microturbulence.binp`.

## Known follow-ups (not blocking)

- The driver fully recomputes `alpha0_full` and `beta` from scratch every
  iteration. Caching by candidate slab could speed convergence if the field
  becomes near-constant between iterations.
- For very large meshes the alpha0_full array (`nlin x ncells x 8 B`) and the
  velocity field (`3 x ncells x 8 B`) dominate memory. The kernel itself
  supports chunked candidate processing; persisting alpha0_full and
  velocity_xyz on disk for huge meshes is a future option.
- No production-scale RADMC-3D smoke run is yet wired into CI (only the
  external-population validation is exercised). A `lines_mode = 50` end-to-end
  check that calls RADMC-3D could be added next to
  [tests/radmc3d/test_nonlte.py](../tests/radmc3d/test_nonlte.py)
  when ready.
