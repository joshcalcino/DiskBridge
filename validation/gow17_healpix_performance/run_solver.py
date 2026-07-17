"""Benchmark native GOW17 batch-solver setup at sustained heterogeneous scale."""

# db-keywords: validation, gow17, chemistry, performance, gas-temperature
# db-role: validation
# db-scope: script
# db-purpose: Baseline/optimized native GOW17 solver throughput and equivalence.

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

import diskbridge._gow17 as _gow17
from diskbridge._constants import E_BIND_CO, N_LAY, N_SURF, NU0_CO, Y_CO


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _inputs(ncells: int) -> dict[str, np.ndarray | float | int | bool]:
    """Construct deterministic heterogeneous cells spanning Bondi-like regimes."""
    index = np.arange(ncells, dtype=np.float64)
    f1 = (index % 1009.0) / 1008.0
    f2 = f1
    f3 = f1

    y0 = np.zeros((ncells, _gow17.N_Y), dtype=np.float64)
    y0[:, _gow17.I_HEP] = 1.45e-8
    y0[:, _gow17.I_OHX] = 2.68e-7
    y0[:, _gow17.I_CHX] = 1.0e-4
    y0[:, _gow17.I_CO] = 1.0e-7 * np.power(100.0, f2)
    y0[:, _gow17.I_H2] = 0.01 + 0.48 * f1

    nH = np.power(10.0, 1.0 + 9.0 * f1)
    Tgas = np.power(10.0, np.log10(8.0) + 3.7 * f2)
    Tdust = np.power(10.0, np.log10(7.0) + 2.0 * f3)
    electron = (
        y0[:, _gow17.I_HEP]
        + y0[:, _gow17.I_CP]
        + y0[:, _gow17.I_HCOP]
        + y0[:, _gow17.I_H3P]
        + y0[:, _gow17.I_H2P]
        + y0[:, _gow17.I_HP]
        + y0[:, _gow17.I_SP]
        + y0[:, _gow17.I_SIP]
        + y0[:, _gow17.I_OP]
    )
    heat_capacity = (
        1.5
        * 1.380649e-16
        * ((1.0 - 2.0 * y0[:, _gow17.I_H2]) + y0[:, _gow17.I_H2] + _gow17.XHE + electron)
    )
    y0[:, _gow17.I_E] = heat_capacity * Tgas

    dust_scale = 0.3 + 1.7 * f3
    radiation = np.power(10.0, -5.0 + 7.0 * f2)
    gph = radiation[:, None] * np.array(
        [1.0, 0.7, 0.3, 0.5, 0.2, 0.8, 0.6], dtype=np.float64
    )

    return {
        "y0": y0,
        "nH": nH,
        "Tgas": Tgas,
        "Tdust": Tdust,
        "Zd": dust_scale,
        "Dpah": dust_scale,
        "Dh2gr": dust_scale,
        "Zgd": dust_scale,
        "Zg": np.ones(ncells, dtype=np.float64),
        "ion_rate": np.full(ncells, 2.0e-16, dtype=np.float64),
        "GPE": radiation,
        "F_CO_pdes_photon": radiation,
        "Gph": gph,
        "sigma_d_CO_per_H": 1.0e-21 * dust_scale,
        "reltol": 1.0e-3,
        "abstol": np.full(_gow17.N_Y, 1.0e-9, dtype=np.float64),
        "mxsteps": 10_000,
        "maxord": 5,
        "t_end": 2.65e13,
        "const_temp": False,
        "gradv": np.power(10.0, -20.0 + 12.0 * f3),
        "Leff_CO_max": np.full(ncells, 3.0e20, dtype=np.float64),
        "NCOeff_external": np.zeros(ncells, dtype=np.float64),
        "use_NCOeff_external": False,
        "isDust_cooling": True,
        "isCoolingCOThin": False,
        "fH2gr": 1.0,
        "fHplusgr": 0.6,
        "fCplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "fCplusCR": 1.0,
        "co_E_bind_co": float(E_BIND_CO),
        "co_nu0_co": float(NU0_CO),
        "co_Y_CO": float(Y_CO),
        "co_N_SURF": float(N_SURF),
        "co_N_LAY": int(N_LAY),
        "co_S_CO": 1.0,
        "co_F_CRUV_CO_pdes": np.zeros(ncells, dtype=np.float64),
        "co_k_crdes_CO": np.zeros(ncells, dtype=np.float64),
        "userJac": False,
        "verbose": False,
    }


def _measure(
    ncells: int,
    *,
    min_repeats: int,
    min_seconds: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    inputs = _inputs(ncells)
    wall_times: list[float] = []
    cpu_times: list[float] = []
    elapsed = 0.0
    result = None
    while len(wall_times) < min_repeats or elapsed < min_seconds:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = _gow17.solve_batch_time(**inputs)
        cpu_times.append(time.process_time() - cpu_start)
        wall_times.append(time.perf_counter() - wall_start)
        elapsed += wall_times[-1]

    assert result is not None
    y = np.asarray(result["y"], dtype=np.float64)
    status = np.asarray(result["status"], dtype=np.int32)
    sample_indices = np.linspace(0, ncells - 1, min(8192, ncells), dtype=np.int64)
    median_wall = statistics.median(wall_times)
    median_cpu = statistics.median(cpu_times)
    measurement = {
        "ncells": ncells,
        "repeats": len(wall_times),
        "wall_seconds": wall_times,
        "cpu_seconds": cpu_times,
        "median_wall_seconds": median_wall,
        "median_cpu_seconds": median_cpu,
        "median_cpu_cores": median_cpu / median_wall,
        "throughput_cells_per_second": ncells / median_wall,
        "status_failure_cells": int(np.count_nonzero(status)),
        "cvode_failure_cells": int(result["cvode_failure_cells"]),
        "exception_failure_cells": int(result["exception_failure_cells"]),
        "negative_abundance_cells": int(result["negative_abundance_cells"]),
        "negative_abundance_corrections": int(result["negative_abundance_corrections"]),
        "output_min_by_species": np.min(y, axis=0).tolist(),
        "output_max_by_species": np.max(y, axis=0).tolist(),
        "output_mean_by_species": np.mean(y, axis=0).tolist(),
        "sample_indices": sample_indices.tolist(),
    }
    return measurement, y[sample_indices], status[sample_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "solver",
    )
    parser.add_argument("--cell-counts", default="20000,100000,500000")
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--sustained-seconds", type=float, default=10.0)
    args = parser.parse_args()

    sizes = [int(value) for value in args.cell_counts.split(",")]
    if any(value <= 0 for value in sizes):
        raise ValueError("all cell counts must be positive")
    if args.min_repeats <= 0 or args.sustained_seconds <= 0.0:
        raise ValueError("min-repeats and sustained-seconds must be positive")

    binary_file = Path(inspect.getfile(_gow17)).resolve()
    repo_root = binary_file.parents[2]
    source_file = repo_root / "src" / "diskbridge" / "_gow17.cpp"
    source_hash = _sha256(source_file)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / f"{stamp}_{source_hash[:12]}"
    output_dir.mkdir(parents=True, exist_ok=False)

    _gow17.solve_batch_time(**_inputs(128))
    measurements = []
    samples: dict[str, np.ndarray] = {}
    largest = max(sizes)
    for ncells in sizes:
        measurement, y_sample, status_sample = _measure(
            ncells,
            min_repeats=args.min_repeats,
            min_seconds=args.sustained_seconds if ncells == largest else 0.0,
        )
        measurements.append(measurement)
        samples[f"y_{ncells}"] = y_sample
        samples[f"status_{ncells}"] = status_sample
        print(
            f"n={ncells:,}: {measurement['median_wall_seconds']:.6f} s, "
            f"{measurement['throughput_cells_per_second']:.1f} cells/s, "
            f"failures={measurement['status_failure_cells']}, "
            f"{measurement['repeats']} repeats",
            flush=True,
        )

    summary = {
        "benchmark": "gow17_native_solver_reuse",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "binary_file": str(binary_file),
        "binary_sha256": _sha256(binary_file),
        "source_file": str(source_file),
        "source_sha256": source_hash,
        "git_sha": _git_sha(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "max_rss_bytes": _max_rss_bytes(),
        "measurements": measurements,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    np.savez(output_dir / "output_samples.npz", **samples)

    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.plot(
        [item["ncells"] / 1.0e3 for item in measurements],
        [item["throughput_cells_per_second"] / 1.0e3 for item in measurements],
        marker="o",
    )
    axis.set_xlabel("Cells (thousands)")
    axis.set_ylabel("Throughput (thousand cells/s)")
    axis.set_title("Native GOW17 time-solver scaling")
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "solver_scaling.png", dpi=180)
    plt.close(figure)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
