# 3D HEALPix Non-LTE External Populations for RADMC-3D

## 1. Goal

Implement a single DiskBridge-native pathway where:

- GOW17 chemistry supplies abundances, gas temperature, and collider densities.
- DiskBridge's 3D HEALPix ray machinery supplies velocity-coherent escape probabilities.
- DiskBridge solves the statistical-equilibrium equations itself.
- RADMC-3D performs only the final Doppler-shifted ray tracing from
  `levelpop_<species>.dat` using `lines_mode = 50`.

Pathway:

```
GOW17 chemistry / DiskBridge fields
    -> 3D HEALPix escape-probability population solver
    -> levelpop_<species>.dat
    -> RADMC-3D ray tracing with lines_mode = 50
```

Done one species at a time: `co`, `catom`, `hco+` (use `catom` for neutral
atomic carbon).

The implementation must not compute `escprob_lengthscale.*` and must not ask
RADMC-3D to solve the populations.

## 2. Performance requirement (global)

All compute-heavy kernels must be written with Numba and parallelized across
cores:

- Use `@numba.njit(parallel=True, fastmath=False, cache=True)`.
- Use `numba.prange` over candidate cells (outer loop), never over directions.
- Operate on plain `float64` contiguous CGS arrays inside kernels (strip unit
  wrappers at the boundary).
- Avoid Python-level allocations inside hot loops; pre-allocate per-thread
  scratch buffers when needed.
- Keep arrays in DiskBridge native mesh order during ray tracing; reorder to
  RADMC order only when writing files.

Kernels that must be Numba-parallel:

- HEALPix optical-depth / `beta` accumulation.
- Per-cell line-center opacity stack assembly.
- Per-cell collisional-rate assembly across colliders.
- Stacked statistical-equilibrium matrix construction (the linear solve uses
  batched `np.linalg.solve`).
- Convergence-error reduction across cells.

## 3. RADMC-3D contract

For each staged species, the run directory must contain:

```
amr_grid.inp
wavelength_micron.inp
dust_density.binp
dustopac.inp
dust_temperature.bdat          # if dust continuum is included
gas_temperature.binp
gas_velocity.binp
microturbulence.binp
numberdens_<species>.binp
molecule_<species>.inp
lines.inp
levelpop_<species>.dat
radmc3d.inp
```

`radmc3d.inp` for the external-population render:

```
incl_lines = 1
lines_mode = 50
tgas_eq_tdust = 0
itempdecoup = 1
rto_style = 3
```

`lines.inp` declares zero RADMC collision partners (RADMC is not solving):

```
2
1
co    leiden    0    0    0
```

Analogously `catom`, `hco+`. The DiskBridge solver still uses collider
densities internally before writing `levelpop_<species>.dat`.

`levelpop_<species>.dat` holds absolute level populations in cm^-3.

## 4. Module layout

Add new code under `src/diskbridge/radmc3d/line_transfer/`:

```
external_populations.py    # public API + iteration driver
molecular_rates.py         # MoleculeData, LAMDA parsing, collision rates
escape_healpix.py          # Numba HEALPix beta kernel
```

The `levelpop_<species>.dat` writer lives with the other RADMC-3D file writers
on `RadWriter` in `src/diskbridge/radmc3d/writer.py`.

`lines.py` (in `other_codes/`) is used only as a formula/reference source for
LAMDA parsing, collision-rate interpolation, Boltzmann initialization, and SE
matrix construction. Do not import the 1D classes into the new modules. The
new code is 3D-native and operates on DiskBridge mesh-order arrays.

## 5. Configuration

```python
@dataclass(frozen=True)
class HealpixSEConfig:
    nside: int = 4
    maxiter: int = 60
    convcrit: float = 1.0e-5
    tbg_K: float = 2.7255
    memory_budget_gib: float = 8.0
    max_ray_steps: int = 200_000
```

Single pathway, single config object. No alternate solver modes.

## 6. Public API

```python
def solve_and_write_healpix_levelpop(
    *,
    rad,
    chemistry_result,
    species: str,
    molecule_file: Path,
    output_dir: Path,
    config: HealpixSEConfig,
) -> Path:
    """
    Solve non-LTE level populations for one species using DiskBridge's
    3D HEALPix escape-probability closure and write levelpop_<species>.dat.
    Returns the path to the written levelpop file.
    """
```

Steps:

1. Load molecule data (levels, transitions, colliders) via the validated
   molecule file.
2. Pull species density from `chemistry_result.number_densities`.
3. Pull collider densities from `chemistry_result.number_densities`.
4. Pull GOW17 gas temperature.
5. Pull gas velocity and convert to Cartesian components.
6. Pull or construct the microturbulence field.
7. Run the statistical-equilibrium iteration.
8. Write `levelpop_<species>.dat` in RADMC-3D cell order.
9. Write a manifest JSON with diagnostics.

Inputs are in-memory DiskBridge arrays. The solver does not read whatever
happens to be on disk. File staging happens after the population solve.

## 7. Molecule and collision-rate handling

`molecular_rates.py`:

```python
@dataclass(frozen=True)
class MoleculeData:
    name: str
    molweight: float              # proton-mass units
    nlev: int
    nlin: int
    energy_erg: np.ndarray        # shape (nlev,)
    weight: np.ndarray            # shape (nlev,)
    iup: np.ndarray               # shape (nlin,), 0-indexed
    ilow: np.ndarray              # shape (nlin,), 0-indexed
    aud: np.ndarray               # shape (nlin,), s^-1
    freq_hz: np.ndarray           # shape (nlin,)
    collider_names: list[str]
    collider_tgrid_K: list[np.ndarray]
    collider_down_rates: list[np.ndarray]  # (nT, nlev, nlev), downward
```

Use 0-indexing internally. Convert to 1-indexed only when writing
`levelpop_<species>.dat`.

Collision rates stored as `gamma[u, l]` (cm^3 s^-1, `u > l`). Per-cell
collisional rates:

```
C_ul = sum_p n_p * gamma_ul,p(T)
C_lu = C_ul * (g_u / g_l) * exp(-(E_u - E_l) / (k * T))
```

Required collider order (use `install_validated_molecule_file()` to enforce
this — do not infer from LAMDA comments):

- `co`     -> `p-h2`, `o-h2`
- `catom`  -> `h`, `p-h2`, `o-h2`, `e`
- `hco+`   -> `h2`

Strict temperature-bound policy: for any cell with `n_species > 0`, `Tgas`
must lie within the tabulated range of every required collision table.
Failure raises an error including species, collider, range, and number of
affected cells. No silent extrapolation or clipping.

## 8. Velocity and microturbulence conventions

The population solver uses Cartesian velocity components internally because
HEALPix directions are Cartesian unit vectors.

Spherical -> Cartesian at cell centers:

```
e_r     = ( sin t cos p,  sin t sin p,  cos t )
e_theta = ( cos t cos p,  cos t sin p, -sin t )
e_phi   = (-sin p,        cos p,        0     )

v_xyz = v_r e_r + v_theta e_theta + v_phi e_phi
```

Cartesian meshes pass through unchanged.

Microturbulence uses RADMC-3D's line-width convention. The
`microturbulence.*` file value is the turbulent line-width parameter
`a_turb` in cm/s, not a 1D RMS value that RADMC-3D converts internally:

```
a_line = sqrt(a_turb^2 + 2 k T_gas / m_mol)
```

Use this same `a_turb` field in the solver and when writing
`microturbulence.binp`:

```python
write_microturbulence(rad, a_turb, output_dir, binary=True)
```

Binary format matches scalar RADMC `.binp`:

```
int64: [1, 8, ncells]
float64 data in RADMC cell order, cm/s
```

The same `a_turb` field is used in the population solver and in the RADMC-3D
render — otherwise the populations and ray-traced opacity will be
inconsistent.

## 9. Statistical-equilibrium equations

Fractional populations per cell:

```
sum_i f_i = 1
n_i_level = n_species * f_i        # written to levelpop_<species>.dat in cm^-3
```

Escape-probability closure per radiative transition u -> l:

```
J_ul   = (1 - beta_ul) S_ul + beta_ul J_bg_ul
beta_ul = (1 - exp(-tau_ul)) / tau_ul
```

After substituting the line source function:

```
R_ul_rad = A_ul beta_ul + B_ul beta_ul J_bg_ul
R_lu_rad = B_lu beta_ul J_bg_ul
```

CMB background by default:

```
J_bg_ul = B_nu(T_bg) = (2 h nu^3 / c^2) / (exp(h nu / (k T_bg)) - 1)
```

Add collisions and assemble the rate matrix:

```
R_ij     = R_ij_rad + C_ij
M_ii     = -sum_{j!=i} R_ij
M_ij     = R_ji            (i != j)
```

Replace the last row with normalization (`M[N-1, :] = 1`, `b[N-1] = 1`,
others zero), then solve `M f = b`.

Post-solve checks: all finite, no significant negatives, `sum(f) == 1`
within tolerance. Values in `[-1e-14, 0]` may be set to zero and renormalized
(roundoff only). Anything below `-1e-14` raises a physics error.

## 10. HEALPix escape probability

No scalar escape length in this pathway. Compute `beta_ul` directly from
HEALPix rays.

Per species, per cell `x`, per transition u -> l, per HEALPix direction
`n_k`, compute a direction-dependent velocity-coherent optical depth:

```
tau_ul,k(x) = integral_{x -> boundary}
              alpha0_ul(x')
              * exp( - ((v(x') - v(x)) . n_k / a_line(x'))^2 )
              ds
```

with line-center opacity (using current iteration f_l, f_u):

```
alpha0_ul = (c^3 A_ul) / (8 pi^{3/2} nu_ul^3 a_line)
            * n_species
            * (g_u/g_l * f_l - f_u)
```

If `alpha0_ul` is significantly negative in any cell with non-negligible
species density, raise. Maser/inverted populations are not supported.

Per-direction escape probability with stable small-tau expansion:

```
beta_ul,k = (1 - exp(-tau_ul,k)) / tau_ul,k
beta(tau) ~ 1 - tau/2 + tau^2/6      for tau << 1
```

Direction average (uniform HEALPix weights):

```
beta_ul(x) = (1 / Npix) sum_k beta_ul,k(x)
```

UV angular weights are not used here — they belong to photochemistry, not
isotropic local line escape.

## 11. Numba HEALPix kernel

`escape_healpix.py` adds a new kernel — do not reuse
`compute_column_rays_healpix()` directly (it integrates scalar columns and
applies the PDR self-cell convention). Reuse the DDA logic from
`chemistry/shielding/healpix_utils.py`.

```python
@numba.njit(parallel=True, fastmath=False, cache=True)
def compute_escape_probabilities_healpix(
    mesh_geom,           # precomputed cell faces / edges
    dirs,                # (Npix, 3) Cartesian HEALPix unit vectors
    candidate_idx,       # (Ncand, 3) native-mesh index of each candidate cell
    cell_centers,        # (Ncand, 3) Cartesian
    alpha0_stack,        # (nlin, n0, n1, n2) cm^-1
    velocity_xyz,        # (n0, n1, n2, 3) cm/s
    a_line,              # (n0, n1, n2) cm/s
    max_ray_steps,
):
    """Return beta with shape (Ncand, nlin)."""
```

Per-direction hot loop:

```
for c in prange(Ncand):
    idx0 = candidate_idx[c]
    v0   = velocity_xyz[idx0]
    for k in range(Npix):
        tau[:] = 0.0
        # DDA walk from cell center to domain boundary
        while inside:
            idx = current cell
            ds  = path length through current cell
            dv  = dot(velocity_xyz[idx] - v0, dirs[k])
            ov  = exp(- (dv / a_line[idx])**2 )
            for m in range(nlin):
                tau[m] += alpha0_stack[m, idx] * ov * ds
            step DDA
        for m in range(nlin):
            beta_sum[c, m] += beta_of_tau(tau[m])
    for m in range(nlin):
        beta[c, m] = beta_sum[c, m] / Npix
```

Implementation details:

- Start rays at the cell center; use the actual first DDA segment length
  to the next face. Do not multiply the first segment by
  `HEALPIX_SELF_WEIGHT`.
- Equal HEALPix weights.
- Integrate to the simulation boundary.
- Do not include dust opacity in the population-solver escape probability.
- Return only direction-averaged `beta` — do not store per-ray optical
  depths.
- Process candidate cells in chunks (see Section 13) so memory is bounded.
- Per-thread `tau` scratch buffer of length `nlin` allocated once outside
  the cell loop.

## 12. Iteration loop

Single fixed-point iteration:

```python
f = lte_populations(molecule, Tgas)
for iteration in range(config.maxiter):
    alpha0 = compute_line_center_opacity(
        molecule=molecule,
        fracpop=f,
        n_species=n_species,
        Tgas=Tgas,
        a_turb=a_turb,
    )
    beta = compute_escape_probabilities_healpix(
        alpha0_stack=alpha0,
        velocity_xyz=velocity_xyz,
        a_line=a_line,
        ...,
    )
    f_new = solve_statistical_equilibrium_all_cells(
        molecule=molecule,
        Tgas=Tgas,
        collider_densities=collider_stack,
        beta=beta,
        tbg_K=config.tbg_K,
    )
    err = convergence_error(f_new, f)
    f = f_new
    if err < config.convcrit:
        break
else:
    raise NonLTEConvergenceError(...)
```

LTE initializer:

```
f_i_LTE = g_i exp(-E_i / kT) / sum_j g_j exp(-E_j / kT)
```

Candidate cells: `candidate_mask = n_species > 0`. Cells with exactly zero
species density skip both the HEALPix and SE solves and get all-zero level
populations.

Convergence metric (does not blow up for nearly empty high levels):

```
eps = max_i |f_new_i - f_old_i| / max(f_new_i, f_old_i, 1e-20)
```

Failure to converge raises and writes no production levelpop file.

`compute_line_center_opacity`, `solve_statistical_equilibrium_all_cells`,
and `convergence_error` are Numba-parallel over cells.

## 13. Chunking and memory

- `float64` contiguous arrays.
- Strip unit wrappers before entering Numba.
- DiskBridge native mesh order until the writer.
- `numba.njit(parallel=True)` + `prange` over candidate cells.
- Accumulate `beta`, not `tau_rays`.
- Cache HEALPix geometry, not `beta`.

Approximate chunk-size estimator (matrix solve dominates after we stopped
storing per-ray tau):

```
bytes_per_cell ~ 8 * nlev * nlev    # SE matrix
chunk_size     = floor(memory_budget_bytes / max(bytes_per_cell, 1))
```

Stacked SE solve:

```
M_chunk: (chunk, nlev, nlev)
b_chunk: (chunk, nlev)
f_chunk = np.linalg.solve(M_chunk, b_chunk)
```

Reduce chunk size under memory pressure. Do not switch physics pathways.

## 14. RADMC cell ordering

Solver arrays stay in native DiskBridge mesh order during ray tracing. The
writer converts to RADMC cell order using the same convention as
`write_number_density()`. For spherical grids that means transpose to
`(phi, theta, r)` then C-order flatten.

Shared helper:

```python
def flatten_scalar_to_radmc_order(mesh, arr3d) -> np.ndarray:
    if mesh.coord_system == "spherical":
        arr = transpose_to_axis_order(
            arr3d,
            from_order=mesh.axis_names(),
            to_order=("phi", "theta", "r"),
        )
        return np.asarray(arr).flatten(order="C")
    if mesh.coord_system == "cartesian":
        return np.asarray(arr3d).flatten(order="C")
    raise ValueError(...)
```

Each level is written in this same cell order.

## 15. levelpop writer

`RadWriter`:

```python
def write_levelpop_dat(
    path: Path,
    *,
    levelpop_cm3_radmc_order: np.ndarray,   # (n_cells, n_levels)
    level_numbers_1based: np.ndarray,        # (n_levels,)
) -> Path:
    ...
```

First implementation writes all levels: `np.arange(1, molecule.nlev + 1)`.

File format:

```
1
<n_cells>
<n_levels>
1 2 3 ... n_levels
<pop_cell_1_level_1> <pop_cell_1_level_2> ...
<pop_cell_2_level_1> <pop_cell_2_level_2> ...
...
```

Before writing, verify `sum_i n_i_level == n_species` in RADMC cell order
within tolerance.

Output locations:

```
radmc3d_inputs/levelpop_co.dat
radmc3d_inputs/levelpop_catom.dat
radmc3d_inputs/levelpop_hco+.dat
```

## 16. Staging

New staging function (do not extend the old non-LTE staging name):

```python
def prepare_external_population_line_run(
    *,
    source_inputs_dir: Path,
    chemistry_inputs_dir: Path,
    work_dir: Path,
    species: str,
    levelpop_file: Path,
    copy_mode: str = "symlink",
    exist_ok: bool = False,
) -> Path:
    ...
```

Stages:

Required:
- `amr_grid.inp`
- `wavelength_micron.inp`
- `radmc3d.inp` (write with `lines_mode = 50`)
- `lines.inp` (zero RADMC colliders)
- `molecule_<species>.inp`
- `levelpop_<species>.dat`
- `numberdens_<species>.binp`
- `gas_temperature.binp`
- `gas_velocity.binp`
- `microturbulence.binp`

Optional:
- `dust_density.binp`
- `dustopac.inp`
- `dustkappa_*.inp`
- `dust_temperature.bdat`
- `stars.inp`
- `external_source.inp`

Update symlink patterns in `src/diskbridge/radmc3d/model.py` and
`src/diskbridge/radmc3d/image.py` to include `microturbulence.*` and
`levelpop_*.dat`.

Update `RadImage._ensure_radmc3d_inp_configured()`: for `lines_mode = 50`
set `tgas_eq_tdust = 0` (not 1). The final ray tracing still needs gas
temperature for line broadening.

## 17. Validation

```python
def validate_external_population_run(work_dir: Path, species: str) -> dict:
    ...
```

Required-file checks:

- `radmc3d.inp` has `lines_mode = 50`
- `lines.inp` has one species and zero RADMC colliders
- `molecule_<species>.inp` exists
- `levelpop_<species>.dat` exists
- `numberdens_<species>.binp` exists
- `gas_temperature.binp` exists
- `gas_velocity.binp` exists
- `microturbulence.binp` exists
- `amr_grid.inp` exists

Content checks:

- `levelpop` n_cells == grid n_cells
- level list is `1..nlev`
- all levelpop values finite and `>= 0`
- `sum(levelpop over levels) == numberdens_species` within tolerance
- gas_temperature positive and finite
- microturbulence nonnegative and finite
- gas_velocity finite

Collider density files are not required in the staged RADMC render
directory.

## 18. Manifest and diagnostics

Write `external_levelpop_manifest.json` next to the staged run:

```json
{
  "species": "co",
  "line_mode": 50,
  "solver": "healpix_escape_probability_statistical_equilibrium",
  "nside": 4,
  "maxiter": 60,
  "convcrit": 1e-5,
  "iterations": 17,
  "final_convergence_error": 7.2e-6,
  "tbg_K": 2.7255,
  "molecule_file": ".../molecule_co.inp",
  "n_levels": 41,
  "n_lines": 40,
  "colliders": ["p-h2", "o-h2"],
  "temperature_range_K": [min, max],
  "microturbulence_range_cm_s": [min, max],
  "species_density_range_cm3": [min, max],
  "beta_range": [min, max],
  "levelpop_file": ".../levelpop_co.dat",
  "iteration_history": [
    {"iteration": 1, "err": 0.23, "beta_min": 0.001, "beta_max": 1.0}
  ]
}
```

## 19. User-facing workflow

```python
from pathlib import Path
from diskbridge.chemistry.api import run_chemistry
from diskbridge.radmc3d.line_transfer.external_populations import (
    HealpixSEConfig,
    solve_and_write_healpix_levelpop,
    prepare_external_population_line_run,
)

chem = run_chemistry(
    rad,
    model="gow17",
    config={
        "line_h2_opr": 3.0,
        "temperature": {"mode": "computed"},
        "enable_co_phase": True,
    },
    write=True,
)

cfg = HealpixSEConfig(
    nside=4,
    maxiter=60,
    convcrit=1.0e-5,
    tbg_K=2.7255,
    memory_budget_gib=8.0,
)

for species in ["co", "catom", "hco+"]:
    molecule_file = install_validated_molecule_file(
        species=species,
        moldata_dir=Path("data/moldata"),
        inputs_dir=Path("population_inputs"),
    )
    levelpop = solve_and_write_healpix_levelpop(
        rad=rad,
        chemistry_result=chem,
        species=species,
        molecule_file=molecule_file,
        output_dir=Path(f"nonlte_{species}/radmc3d_inputs"),
        config=cfg,
    )
    prepare_external_population_line_run(
        source_inputs_dir=Path("radmc3d_inputs"),
        chemistry_inputs_dir=Path("radmc3d_inputs"),
        work_dir=Path(f"nonlte_{species}"),
        species=species,
        levelpop_file=levelpop,
        copy_mode="symlink",
        exist_ok=True,
    )
```

Ordering is fixed: chemistry first, population solve second, RADMC staging
third, RADMC imaging/spectrum last.

## 20. Out of scope

Do not include:

- 1D line-transfer classes
- Lambda iteration from `lines.py`
- MALI approximate operators
- Scalar `escprob_lengthscale` construction
- UV-weighted line escape
- RADMC-computed populations
- Silent temperature extrapolation
- Silent population clipping beyond roundoff
- Automatic changes of molecular abundance in the ISM

If the ambient ISM is real, HEALPix rays must include it. If part of the
ambient medium is numerical padding, the chemistry/abundance model sets the
relevant molecular abundance before this solver runs. The population solver
does not silently reinterpret the domain.

## 21. Tests

Unit tests:

1. LAMDA parser reproduces level energies, weights, A coefficients, and
   collision tables.
2. Collision-rate calculation matches `lines.py` for one cell and one
   collider-density vector.
3. LTE initializer sums to one and approaches expected Boltzmann ratios.
4. `beta_of_tau` is stable at `tau << 1` and behaves like `1/tau` at
   `tau >> 1`.
5. SE matrix solve conserves `sum(f) = 1` and produces nonnegative
   populations.
6. Very high collider density -> populations approach Boltzmann at `Tgas`.
7. Negligible collisions + CMB background only -> low levels approach
   radiative equilibrium with `Tbg`.
8. `levelpop` writer produces a file whose summed levels match
   `numberdens_species`.
9. External-population validation rejects missing microturbulence, wrong
   cell count, negative populations, or wrong level list.
10. Tiny RADMC-3D smoke run with `lines_mode = 50` confirms RADMC reads
    `levelpop_<species>.dat`.

Primary regression test:

```
sum(levelpop_<species> over levels) == numberdens_<species>     (RADMC cell order)
```

Performance tests:

- Numba kernel scales with `OMP_NUM_THREADS` / `NUMBA_NUM_THREADS` on a
  representative grid.
- Kernel produces identical output to a single-threaded reference (within
  float64 roundoff).
