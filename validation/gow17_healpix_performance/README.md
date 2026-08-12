# GOW17 HEALPix performance comparison

## Overview

This validation measures HEALPix DDA ray marching, directional UV-weight
construction, the variable-linewidth Visser CO shielding stage, the complete
post-ray shielding algebra, the directional Omukai cooling reduction, and the
native GOW17 per-cell CVODE stage at scales relevant to the Bondi All-Stars
run. It is a performance and numerical-comparison workflow, not a pytest test.
The largest default shielding case has
`349,525` cells and `192` HEALPix directions (`nside=4`), for `67,108,800`
interpolation samples per call.

The benchmark keeps the shielding tables and deterministic input construction
fixed. It records wall time, process CPU time, throughput, peak resident
memory, source and table hashes, output statistics, and 8,192 output samples
per array size. The largest case repeats for at least ten measured seconds so
the comparison is not based on a sub-second microbenchmark.

## Usage

Run the driver from the repository root:

```bash
python validation/gow17_healpix_performance/run.py
NUMBA_NUM_THREADS=16 python validation/gow17_healpix_performance/run_w_rays.py
NUMBA_NUM_THREADS=16 python validation/gow17_healpix_performance/run_postprocess.py
OMP_NUM_THREADS=16 python validation/gow17_healpix_performance/run_solver.py
```

Each invocation writes a timestamped, source-hash-qualified directory under
`outputs/`, so repeated measurements cannot overwrite each other.

The solver driver uses deterministic heterogeneous densities, temperatures,
radiation fields, dust scalings, velocity gradients, and initial states. Its
largest default batch has 500,000 cells and repeats for at least ten measured
seconds. Set `OMP_NUM_THREADS` to the worker count being measured.

The post-processing driver begins with already-integrated ray columns and
measures effective H2/CO linewidths, H2 and C shielding, variable-linewidth CO
shielding, uniform or weighted directional reductions, and the combined H2*CO
PDR factor. Its default is the full `349,525 x 192` production chunk. Set
`NUMBA_NUM_THREADS` to the worker count being measured.

The W-ray driver measures the complete directional UV-weight builder on a
deterministic spherical `60 x 40 x 8` mesh with 192 directions and 20 dust
bins. It includes both external and stellar optical depths, uses a full-mesh
chunk by default, and records row normalization, sampled weights, process CPU
occupancy, sampled RSS history, and the builder's known-array memory estimate.
Its output includes `summary.json`, `output_samples.npz`, and
`w_rays_diagnostics.png`.

For a bounded production-geometry scaling comparison, retain the native mesh
and distribute a deterministic candidate sample through it:

```bash
NUMBA_NUM_THREADS=18 python validation/gow17_healpix_performance/run_w_rays.py \
  --shape 486,146,330 --candidate-count 250000 --chunk-size 250000 \
  --min-repeats 1 --sustained-seconds 0.01
```

The output records every native worker's cumulative CPU time at 0.1-second
cadence. This command is a scaling sample, not the default sustained validation
and not a substitute for the full cached-RADMC cluster comparison.

Use `--candidate-layout contiguous --candidate-start OFFSET` to exercise the
native C-order spatial bands used by production chunking. Plot a completed
old/current pair on common absolute-time and fractional-time axes with:

```bash
python validation/gow17_healpix_performance/plot_scheduler_ab.py \
  OLD_OUTPUT_ROOT CURRENT_OUTPUT_ROOT --output-directory COMPARISON_OUTPUT
```

The comparison also records wall time, the final below-half-allocation tail,
source hashes, and bitwise equality of the saved numerical samples.

### Bounded PDR ray field-count diagnostic

`run_pdr_ray_fields.py` isolates the canonical spherical
`integrate_rays_multi()` call from RADMC-3D, post-ray shielding algebra, and
chemistry. Its cluster workload uses the Bondi production shape
`(486, 146, 330)`, 192 directions, a contiguous 120,000-cell native-order band
beginning at flat offset 11,500,000, and either five or six deterministic input
fields. One invocation performs one Numba warm-up outside the timed interval
and one measured traversal, then writes `summary.json`, `output_samples.npz`,
and `ray_fields_diagnostics.png` below a timestamped directory.

Exercise the real driver locally at a small scale with:

```bash
NUMBA_NUM_THREADS=4 python \
  validation/gow17_healpix_performance/run_pdr_ray_fields.py \
  --shape 12,8,6 --candidate-count 96 --candidate-start 100 \
  --nside 1 --field-count 6 --source-revision local-check
```

The cluster launcher runs immutable source snapshots sequentially on one node:

- `419a58a`: pre-cyclic canonical traversal;
- `3501a25`: importable direct successor to cyclic-scheduling commit
  `95ffcfd`, with the identical `healpix_utils.py` ray kernel;
- `702c397`: deployed Bondi production source;
- `9b20be5`: accepted edge-cache and backend-aware scheduling source.

Five and six fields are compared for the first three revisions; the latest
revision runs the production six-field workload. Materialize each snapshot as
`source_<revision>/src/` below the Slurm submission directory, then submit the
same launcher once to each node family:

```bash
sbatch --partition=milan --nodelist='dave[1-147]' \
  run_cluster_pdr_ray_diagnostic.slurm
sbatch --partition=turin-c run_cluster_pdr_ray_diagnostic.slurm
```

The explicit Dave node list is required because OzSTAR's submission plugin may
expand the nominal `milan` partition to other compatible CPU partitions.

Each job requests 64 CPUs, 24 GiB, and a hard 30-minute limit. The launcher
records host, partition, source hashes, software versions, CPU topology,
threading backend, binding, and timestamps. It finishes by writing
`comparison.json` and `comparison.png` under `outputs/<job-id>/`. Do not report
field or scheduler ratios across different jobs when the same-node comparison
is available.

Inspect the six-over-five field ratio within a revision and the
`3501a25/419a58a` ratio at fixed field count. Bitwise-equal saved samples for
those two revisions confirm that the scheduling comparison preserves the
sampled ray arithmetic. The Turin job is a node-family control. This benchmark
uses deterministic production-shaped geometry rather than a loaded Bondi
snapshot and does not include W-ray construction, post-processing, or the
directional Omukai reducer. If it does not reproduce the Milan slowdown, a
separate bounded 32-versus-64-core NUMA comparison is required before changing
production placement or scheduling.

### Same-Dave NUMA diagnostic

`run_cluster_pdr_numa_diagnostic.slurm` distinguishes thread scaling from
memory placement on one Dave node. A single 64-CPU allocation runs the deployed
`702c397` six-field source twice in each of three cases: 32 CPUs with default
first-touch placement, 64 CPUs with default placement, and 64 CPUs with memory
interleaved across NUMA nodes. Every case uses the production mesh, 192
directions, and the same contiguous 20,000-cell band. The launcher records the
Slurm CPU masks, CPU/socket/NUMA topology, source hashes, software environment,
and the normal per-run JSON, NPZ, and PNG outputs.

Materialize `source_702c397/src/` beside the launcher and submit it only to a
Dave node:

```bash
sbatch --partition=milan --nodelist='dave[1-147]' \
  run_cluster_pdr_numa_diagnostic.slurm
```

Compare the two repetitions rather than a single timing. Healthy 32-to-64-core
scaling should materially reduce wall time. If default 64-core scaling is poor
but `numactl --interleave=all` improves it, first-touch NUMA placement is the
likely limiting factor. If both 64-core cases are similar, profile the corrected
spherical-boundary kernel before changing production placement. This bounded
diagnostic does not include chemistry or change production jobs.

The laptop tail-scheduling campaign uses the exact factor-two Bondi mesh
coarsening `243 x 73 x 165` (`2,926,935` cells), 192 directions, and one
uniformly distributed candidate chunk. Run separate processes with 4, 8, and
12 Numba threads; compare the 12-thread result against the unchanged source.
The summary reports both whole-call and final-decile sampled worker occupancy.
Calibrate candidate count first and keep the complete campaign below the
agreed 20-minute wall-time ceiling.

The canonical spherical ray kernel uses bounded dynamic packets only when the
initialized Numba threading layer is `tbb`. OpenMP and workqueue execute that
same kernel with static chunksize zero. Every summary records
`numba_threading_layer`; do not attribute dynamic load balancing to a run whose
summary reports a non-TBB backend.

The sustained local TBB acceptance case uses the native `(486, 146, 330)`
mesh, a contiguous 350,000-candidate band beginning at flat offset 11,500,000,
192 directions, and 18 threads. Run old and current sources in separate
processes with `NUMBA_THREADING_LAYER=tbb`, require each summary to report
`tbb`, and compare them with `plot_scheduler_ab.py`. The accepted three-repeat
result and shared CPU trace are under
`outputs/tbb_scheduler_local/long_18_tbb_*`.

To measure complete live Bondi jobs rather than an isolated kernel, run:

```bash
python validation/gow17_healpix_performance/run_cluster_cpu_timeline.py
```

The live monitor discovers running 64-core Bondi All-Stars jobs, reads their
cumulative Slurm CPU counters without modifying the jobs, and differences
successive samples to recover interval core usage. Its timestamped output
contains raw JSON, derived NPZ arrays, an ensemble heatmap, per-job time-series
panels, and a summary. Values plotted above the requested allocation are capped
only for display; the raw values retain Slurm counter/timestamp uncertainty.

Useful optional controls are the drivers' size lists, `--min-repeats`,
`--sustained-seconds`, and `--output-root`. Defaults represent the acceptance
workloads; reducing them changes the performance validation and should not be
reported as a sustained result.

Compare the matching keys in the two `summary.json` files and arrays in
`output_samples.npz`. Scientific acceptance requires matching input/table
provenance, output bounds and means, and sampled outputs within the numerical
tolerance established by the direct SciPy reference test in
`tests/chemistry/test_shielding.py`.

For the native solver, scientific acceptance requires identical status and
failure diagnostics plus sampled state values matching at floating-point
roundoff.

Each run writes `summary.json`, `output_samples.npz`, and a scaling plot in a
timestamped output directory. The paired interpretation is recorded in
`report.md`. Inspect the sampled differences, failure counts, CPU occupancy,
peak RSS, and whether throughput remains stable at the largest size.

## Deep dive

Each driver isolates reusable DiskBridge codebase behavior. None measures
Slurm binding, requested CPU count, the configured chunk-memory budget,
checkpoint cadence, or the complete coupled shielding/chemistry update. Those
are separate Bondi pipeline and end-to-end measurements. The live cluster CPU
monitor is the exception: it measures the complete running workflow but cannot
identify internal phases unless the corresponding job emits stage markers.

The implementation is warmed before measurement so Numba compilation time is
excluded; production jobs likewise compile once and reuse the kernels over
many shielding chunks and chemistry updates.

The DDA comparison integrates the six fields required by hybrid CO cooling on
the saved `60 x 40 x 8`, `nside=4` validation grid. Separate pytest coverage
requires bit-for-bit identical Cartesian and spherical columns with one and
multiple Numba threads. The directional Omukai benchmark uses deterministic
cells that mix full column inversion with inexpensive early-return cases so
that scheduling behavior is measured without changing the cooling problem.

The W-ray comparison exercises the production algebra that forms one scalar
extinction coefficient from all dust bins, integrates external and starward
optical depths, applies source weights and outer-boundary rules, and normalizes
every candidate row. It therefore measures the complete allocation lifetime
that was absent from the original isolated DDA benchmark.

The native-solver comparison tests whether one SUNDIALS/CVODE allocation per
OpenMP worker can replace one allocation per cell without changing any solver
input, equation, tolerance, status, or state output. It records sampled outputs
instead of all states to keep validation artifacts small.

This is an isolated kernel/batch validation. The local W-ray case is much
smaller than the 23,415,480-cell production mesh. Current RSS sampling is exact
on Linux; other platforms may provide only a process high-water mark. These
drivers do not establish complete Bondi wall time, filesystem throughput, or
scientific equivalence after all 30 coupled updates; those require the planned
same-input cluster A/B run.

## References

The interpolation tables follow Visser et al. (2009), *A&A*, 503, 323. See the
DiskBridge shielding documentation for the scientific model and table source.
