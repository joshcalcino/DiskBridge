# GOW17 HEALPix performance report

## Codebase results

The original shielding and solver comparisons use commit
`f919e3f8f74072ef409cd7b7575bd78e66dc012c` through the detached baseline
worktree and the optimized source through `perf/gow17-healpix-efficiency`.
The later scheduling comparisons directly switch only the scheduling strategy
within the same source tree. Scientific inputs and thread counts are fixed
within each comparison.

| Workload | Unchanged | Optimized | Speedup | Numerical result |
|---|---:|---:|---:|---|
| Visser interpolation, 67,108,800 rays | 4.51764 s | 0.129119 s | 34.99x | sampled max abs `2.22e-16` |
| Weighted ray reduction, 67,108,800 rays | 0.023810 s | 0.004904 s | 4.86x | max abs `1.55e-15` |
| Complete post-ray shielding, 67,108,800 rays | 4.92482 s | 0.249548 s | 19.73x | sampled max abs `2.89e-15` |
| Native GOW17 solve, 500,000 cells, 16 threads | 24.00065 s | 23.78550 s | 1.009x | sampled states bit-for-bit identical |
| C/O budget projection, 23,415,480 cells | 1.072 s | 0.03339 s | 32.1x | analytic budget tests pass |
| Six-field spherical DDA, 19,200 cells x 192 rays | 3.302 s | 2.917 s | 1.13x | column checksum identical; exact toy regressions |
| Directional UV weights, 250,000 candidates, 8 CPUs | 474.43 s | 253.05 s | 1.87x | sampled max abs `6.94e-18` |
| Directional UV weights, 250,000 candidates, 18 CPUs | 273.99 s | 134.27 s | 2.04x | sampled max abs `6.94e-18` |
| Coarsened Bondi W rays, 150,000 candidates, 12 CPUs | 49.219 s | 39.386 s | 1.25x | sampled values bit-for-bit identical |
| Native-order W rays, 350,000 candidates, 18 TBB threads | 197.16 s | 172.64 s | 1.14x | sampled values bit-for-bit identical |
| Directional Omukai reduction, 1,200,000 cells | 1.048 s | 1.023 s | 1.02x | equivalent-column checksum identical |

The complete post-ray comparison includes effective H2 and CO linewidths, H2
and C formulas, Visser CO interpolation, weighted ray reductions, and the
combined H2*CO factor. Average CPU occupancy increased from 1.00 to 11.71
cores and peak RSS fell from 15.67 to 5.86 GiB. At 67 chunks per update, this
isolated stage projects to 330.0 s unchanged versus 16.7 s optimized, a saving
of about 5.22 minutes per shielding update. It does not include ray marching
or the chemistry solve.

The DDA result uses the completed `60 x 40 x 8`, `nside=4` hybrid-cooling
validation grid and all six fields required by the cooling calculation. A
no-copy cyclic cell assignment raised median occupancy to 17.0 of 18 cores.
Numba dynamic chunk sizes were also measured and were slower, at 3.60--9.27 s,
so they were rejected. For the directional Omukai reduction, a deterministic
mixture of full column inversions and early-return cells improved modestly
with guided OpenMP scheduling while preserving the checksum.

The production-geometry W-ray comparison uses the native `(486, 146, 330)`
mesh, 250,000 candidates distributed uniformly through all 23,415,480 cells,
192 directions, 20 deterministic dust bins, an external field, and a direct
stellar contribution. At 8 CPUs, sampled peak RSS fell from 19.72 GiB to
4.83 GiB (75.5%); at 18 CPUs it fell from 19.77 GiB to 4.83 GiB (75.6%).
Candidate indices and directions were identical, the maximum sampled relative
weight difference was `1.25e-15`, and row normalization remained at roundoff.

Native-thread CPU time was sampled every 0.1 s. The optimized build averaged
7.92 of 8 and 17.46 of 18 cores. Individual optimized-worker utilization was
98.2--99.6% at 8 CPUs and 96.7--97.3% at 18 CPUs, so there is no measurable
worker-tail imbalance to fix. Intervals below 80% of the allocation totaled
3.60 s at 8 CPUs and 1.10 s at 18 CPUs. They occur at final ray completion and
source-map assembly; the 18-CPU interval is only 0.8% of wall time. The
baseline spends additional low-CPU intervals at twenty-field allocation,
external/starward phase boundaries, and advanced-index normalization. Those
intervals total 5.92 s at 8 CPUs and 10.29 s at 18 CPUs and are removed or
shortened by the scalar-field and fused-fill implementation.

Scaling from 8 to 18 CPUs improved from 1.73x in the baseline to 1.88x in the
optimized build. Because all optimized workers remain active, the remaining
shortfall from ideal 2.25x scaling is memory-bandwidth and per-core throughput
contention, not idle workers. A same-hardware linear candidate-count projection
for one 1,864,135-cell production chunk is 59.0 versus 31.4 minutes at 8 CPUs
and 34.1 versus 16.7 minutes at 18 CPUs. This is not a cluster wall-time
prediction; the cached-RADMC 32/48/64-core comparison remains required.

Machine-readable summaries, raw per-thread samples, and plots are under
`outputs/w_rays_scaling/production_geometry_250k/`; the combined interpretation
is in `analysis/scaling_summary.json` and `analysis/scaling_comparison.png`.

### Laptop spherical-tail scheduling check

A later bounded-scheduling check used the exact factor-two coarsening of the
Bondi mesh, `(243, 73, 165)` or 2,926,935 cells. One chunk contained 150,000
candidates distributed uniformly through the mesh, with `nside=4`. The
32-packet dynamic scheduler took 196.31, 131.37, and 99.62 seconds at 4, 8,
and 12 cores. These correspond to 1.49x and 1.97x speedups over four cores,
or 74.7% and 65.7% parallel efficiency.

At 12 cores, the cyclic-static implementation took 85.51 seconds. Dynamic
schedules with 32, 64, and 128 packets per worker took 99.62, 88.71, and 86.90
seconds: 16.5%, 3.7%, and 1.6% slower, respectively. The 128-packet schedule
reached 96.3% median occupancy in the final time decile, but it still did not
provide a wall-time improvement. All sampled weights, candidate indices,
directions, centers, and row sums were bit-for-bit identical, and peak RSS did
not increase materially.

That unconditional bounded scheduler failed the laptop performance acceptance
gate and was reverted. Those runs used Numba `workqueue`, which cannot provide
TBB work stealing. Raw summaries and per-thread histories are under
`outputs/w_rays_tail_scaling/`.

The current scheduler policy therefore applies 128 bounded packets per worker
only when Numba reports the `tbb` threading layer. OpenMP and workqueue run the
same kernel with static chunksize zero. On the local workqueue backend, an
identical 30,000-candidate contiguous coarsened-Bondi workload took 8.477
seconds before and 8.467 seconds after adding the selector, with bit-for-bit
identical saved outputs. This establishes a neutral static fallback, not a TBB
speed result.

A longer fallback run used 18 threads, the complete `(486, 146, 330)` mesh,
the contiguous 350,000-candidate band beginning at flat offset 11,500,000, and
192 directions. Three workqueue/static repetitions took 175.48, 189.86, and
199.36 seconds, with a median 16.76 effective cores and 5.663 GiB sampled peak
RSS. Per-repeat median and final-decile occupancy were 94.3--94.7% and
94.0--94.7%; only 2.42--2.95 seconds per repetition fell below half of the
18-core allocation. These longer measurements show that the non-TBB fallback
does not stall on this native-order chunk. They remain inapplicable to TBB
dynamic-speed claims.

An isolated conda-forge environment provided Numba 0.65.0 and the native
Apple-silicon TBB backend for a direct local comparison. Both summaries report
`numba_threading_layer=tbb`. On the same full mesh, contiguous 350,000-cell
band, 192 directions, and 18 threads, the unchanged TBB runner took 190.92,
197.16, and 201.05 seconds. The bounded-packet runner took 166.94, 172.64, and
174.94 seconds, reducing the median wall time by 12.4% (1.14x).

Median active cores increased from 17.14 to 17.66, and final-decile occupancy
increased from 98.35% to 98.75%. Sampled peak RSS changed by 0.32%, from 5.260
to 5.277 GiB. Saved weights, row sums, candidate indices, centers, and
directions are bit-for-bit identical. Raw summaries and the shared CPU trace
are under `outputs/tbb_scheduler_local/long_18_tbb_*`. This accepts the local
TBB scheduler change; the 64-core cluster run remains necessary to quantify
its production-node benefit.

### Balanced-worker follow-up

A three-repeat follow-up retained cyclic-static assignment and optimized the
work performed by those already-balanced workers. Spherical theta and phi mesh
edges are fixed, but the DDA previously recalculated their trigonometric
factors at every boundary test. The accepted kernel prepares these small edge
tables once per integration call and reuses them without changing ray order or
column summation order.

On the same 150,000-candidate coarsened Bondi workload, median wall times were
134.56 versus 108.14 seconds at 4 threads, 71.01 versus 56.84 seconds at 8
threads, and 49.22 versus 39.39 seconds at 12 threads. The optimized kernel was
19.6--20.0% faster at every count. Median and final-decile occupancy remained
above 99%, sampled peak RSS remained 1.372--1.373 GiB, and saved weights,
normalization sums, candidate indices, centers, and directions were bit-for-bit
identical. Individual wall times and native-thread histories are under
`outputs/scheduler_phase2f/`.

Plain contiguous cell bands, cyclic blocks of 4 or 16 cells, and one-time
direction normalization did not improve the balanced cyclic baseline and were
removed. This result does not overturn the bounded-dynamic rejection: it shows
that lowering repeated per-ray work, rather than adding scheduler packets, was
the effective optimization.

The later three-repeat 12-thread baseline (49.22 seconds) is faster than the
earlier single 85.51-second scheduling observation because the campaigns ran
under different laptop load and thermal state. The accepted percentage uses
only the adjacent three-repeat old/new pairs above; 39.39 seconds must not be
compared directly with the earlier 85.51-second observation.

### Live 64-core Bondi CPU timeline

A read-only 21.8-minute Slurm trace sampled all thirteen running 64-core Bondi
jobs at approximately six-second cadence. Interval core usage is the change in
cumulative Slurm CPU seconds divided by measured wall time. Every job reached
the full allocation, but every job also fell below 4.2 active cores. Per-job
median usage ranged from 50.0 to 63.5 cores, and jobs spent 10.4--26.7% of
sampled intervals below 32 cores.

Individual traces show repeated plateaus near 64 cores, smooth multi-minute
declines to one or a few cores, and abrupt returns to 64. These events are
staggered across separate nodes, ruling out a shared affinity cap or
synchronized cluster outage. The morphology is consistent with static workers
finishing unequal chunk workloads, followed by all workers starting the next
chunk. Buffered job logs do not contain per-chunk stage markers, so the exact
function attribution remains an inference rather than a direct log match.

These jobs use commit `702c397` and spherical-kernel SHA-256 `5f729983f6df`;
they do not contain the later local edge-trigonometry optimization. The live
result shows that high occupancy in the uniformly sampled 12-thread laptop
case does not establish sustained occupancy for native 64-core jobs. Raw
samples, derived arrays, summary statistics, an ensemble heatmap, and per-job
panels are under `outputs/cluster_cpu_timeline/20260729T164950Z/`.

### Native-order old/current comparison

A local A/B used a contiguous 350,000-candidate band at flat offset 11,500,000
on the complete `(486, 146, 330)` mesh with 18 threads. The unchanged runner
took 205.01 seconds and the current edge-cached runner took 163.55 seconds, a
20.2% reduction. Saved weights, row sums, indices, centers, and directions were
bit-for-bit identical.

The current runner still exhibited the abrupt low-core tail. One-second-
smoothed time below half of the allocation was 5.81 seconds unchanged and 6.19
seconds current. The edge-cache change therefore improves the parallel work
but does not correct worker completion imbalance. The shared trace and summary
are under `outputs/scheduler_phase2h/local_old_vs_current_18/`.

The same-node 64-core A/B subsequently completed as Slurm job `14918324` on
`dave307`. The unchanged source took 108.01 seconds and the current source took
84.70 seconds, a 21.6% improvement. Sampled peak RSS was 4.63 versus 4.66 GiB,
and all saved numerical samples were bit-for-bit identical.

The utilization result is unambiguous: both traces fall from approximately 64
cores to approximately one core over the final decile. With a one-second
median suppressing the brief final rebound, the below-half-allocation interval
is 7.40 seconds unchanged and 5.89 seconds current. These are nearly identical
fractions of wall time, 6.85% and 6.96%. Final-decile median occupancy is 1.685%
for both, or about 1.08 active cores. The edge cache accelerates the parallel
work but does not fix the worker-completion collapse. Downloaded summaries,
raw traces, job log, and the shared plot are under
`outputs/scheduler_phase2h/cluster_job_14918324/`.

Machine-readable complete-postprocessing results are in:

- unchanged: `outputs/postprocess/20260714T121912Z_89b2cffb1a14`
- optimized: `outputs/postprocess/20260714T121847Z_f7944dd35d0a`

The native-solver comparison includes three repeats at 20,000, 100,000, and
500,000 cells. At the largest size, both builds report 5,941 CVODE failures and
identical negative-abundance correction diagnostics. Median CPU occupancy is
15.87 of 16 cores and peak RSS is about 413 MB in both builds.

The unchanged solver emitted 57,039 warning lines (13.4 MB) during the
sustained local run. The optimized solver suppresses per-cell `t+h=t` warnings
through the documented SUNDIALS setting and retains aggregate failure counts.
This should remove substantially more log I/O in the multi-day Bondi jobs.

Passing the already-projected state directly to each native solve removes one
redundant `(23,415,480, 15)` projection per coupling update. The eliminated
state copy alone is 2.62 GiB; a standalone full-size projection took 1.07 s and
peaked at 10.1 GB RSS including its input, output, budget arrays, and
temporaries.

The remaining canonical projection is now a parallel per-cell kernel. A
full-size, 16-thread run sustained a 0.03339 s median over 189 repetitions;
the same kernel took 0.20792 s with one thread, establishing 6.23x measured
thread scaling independently of the NumPy-to-kernel rewrite.

Machine-readable native-solver results and plots are in:

- unchanged: `outputs/solver/20260714T100127Z_46cd16d57f5c`
- optimized: `outputs/solver/20260714T095941Z_489d8ec07cd7`

### Bounded PDR ray field-count diagnostic

Two read-only production-kernel diagnostics were submitted on 2026-08-03 from
`/fred/oz015/jcalcino/validation/gow17_pdr_ray_diagnostic_20260803`:

- job `15038473`, constrained to Milan-generation `dave[1-147]` nodes;
- job `15038471` on the `turin-c` partition.

Each job requests 64 CPUs, 24 GiB, and at most 30 minutes. Within one node it
sequentially measures 120,000 contiguous native-order candidates at five and
six fields using immutable source snapshots `419a58a`, `3501a25`, and
`702c397`, plus the six-field current source `9b20be5`. The source snapshots'
`healpix_utils.py` SHA-256 prefixes are `c5902472e11b`, `4452d53112ce`,
`5f729983f6df`, and `a51e9d6e121b`, respectively. Results remain pending; no
production Bondi job was changed or interrupted. The original unstarted Milan
submission `15038470` was cancelled and replaced after OzSTAR expanded its
eligible partitions beyond Dave nodes; no benchmark work was lost.

## Bondi pipeline settings

These are pipeline choices, not reusable DiskBridge code changes:

- derive OpenMP and Numba thread counts from `SLURM_CPUS_PER_TASK` and bind the
  single task to cores;
- checkpoint every five astrochem updates;
- remove the unused pre-RT full-model snapshot;
- retain the 64-CPU, 240-GiB request and 8-GiB shielding chunk budget until a
  same-node scaling and memory trial supports changing them.

The parallel kernels, fused reductions, in-place linewidth storage, and
revised live-array accounting above are codebase changes. Thread binding,
checkpoint cadence, CPU/memory requests, and the selected 8-GiB budget remain
Bondi pipeline choices.

At `every=5`, a 30-update run writes six compact checkpoints, projected at
about 20.8 GB total. Based on the historical 2.37-hour update duration, the
maximum completed work exposed to an interruption is about 11.8 hours.

## Remaining validation

No optimized Bondi job has yet been submitted. The next isolated cluster trial
must record `shielding_s`, `chemistry_s`, `step_s`, checkpoint throughput, CPU
efficiency, and peak RSS, then compare the final shielding, chemistry state,
temperature, and status diagnostics against an unchanged run with identical
physics inputs. CPU-count and chunk-memory tuning remain pipeline experiments
and must not be attributed to codebase speedups.
