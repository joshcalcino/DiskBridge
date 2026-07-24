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
