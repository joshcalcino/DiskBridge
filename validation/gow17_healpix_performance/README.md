# GOW17 HEALPix performance comparison

## Overview

This validation measures HEALPix DDA ray marching, the variable-linewidth
Visser CO shielding stage, the complete post-ray shielding algebra, the
directional Omukai cooling reduction, and the native GOW17 per-cell CVODE
stage at scales relevant to the Bondi All-Stars run. It is a performance and
numerical-comparison workflow, not a pytest test.
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
are separate Bondi pipeline and end-to-end measurements.

The implementation is warmed before measurement so Numba compilation time is
excluded; production jobs likewise compile once and reuse the kernels over
many shielding chunks and chemistry updates.

The DDA comparison integrates the six fields required by hybrid CO cooling on
the saved `60 x 40 x 8`, `nside=4` validation grid. Separate pytest coverage
requires bit-for-bit identical Cartesian and spherical columns with one and
multiple Numba threads. The directional Omukai benchmark uses deterministic
cells that mix full column inversion with inexpensive early-return cases so
that scheduling behavior is measured without changing the cooling problem.

The native-solver comparison tests whether one SUNDIALS/CVODE allocation per
OpenMP worker can replace one allocation per cell without changing any solver
input, equation, tolerance, status, or state output. It records sampled outputs
instead of all states to keep validation artifacts small.

This is an isolated kernel/batch validation. It does not establish complete
Bondi wall time, filesystem throughput, or scientific equivalence after all 30
coupled updates; those require the planned same-input cluster A/B run.

## References

The interpolation tables follow Visser et al. (2009), *A&A*, 503, 323. See the
DiskBridge shielding documentation for the scientific model and table source.
