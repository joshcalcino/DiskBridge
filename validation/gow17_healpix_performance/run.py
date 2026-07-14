"""Benchmark Visser shielding at Bondi All-Stars HEALPix chunk scale."""

# db-keywords: validation, shielding, co-shielding, healpix-columns, gow17, performance
# db-role: validation
# db-scope: script
# db-purpose: Sustained baseline/optimized Visser shielding scaling comparison.

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
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

import diskbridge.chemistry.shielding.visser_shielding as visser_module
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding


BONDI_CHUNK_CELLS = 349_525
NSIDE4_RAYS = 192
BONDI_RAY_SAMPLES = BONDI_CHUNK_CELLS * NSIDE4_RAYS


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


def _inputs(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(n_samples, dtype=np.float64)
    nco = np.power(10.0, 9.0 + 11.0 * ((index % 10007.0) / 10006.0))
    nh2 = np.power(10.0, 9.0 + 14.0 * ((index % 10009.0) / 10008.0))
    b_kms = 0.01 + 4.99 * ((index % 9973.0) / 9972.0)
    return nco, nh2, b_kms


def _measure(
    visser: VisserShielding,
    n_samples: int,
    *,
    min_repeats: int,
    min_seconds: float,
) -> tuple[dict, np.ndarray]:
    nco, nh2, b_kms = _inputs(n_samples)
    wall_times = []
    cpu_times = []
    elapsed = 0.0
    output = np.empty(0, dtype=np.float64)
    while len(wall_times) < min_repeats or elapsed < min_seconds:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        output = visser.theta_interpolated_b("co", nco, nh2, b_kms)
        cpu_times.append(time.process_time() - cpu_start)
        wall_times.append(time.perf_counter() - wall_start)
        elapsed += wall_times[-1]

    sample_indices = np.linspace(
        0,
        n_samples - 1,
        min(8192, n_samples),
        dtype=np.int64,
    )
    median_wall = statistics.median(wall_times)
    median_cpu = statistics.median(cpu_times)
    result = {
        "n_samples": n_samples,
        "repeats": len(wall_times),
        "wall_seconds": wall_times,
        "cpu_seconds": cpu_times,
        "median_wall_seconds": median_wall,
        "median_cpu_seconds": median_cpu,
        "median_cpu_cores": median_cpu / median_wall,
        "throughput_samples_per_second": n_samples / median_wall,
        "output_min": float(np.min(output)),
        "output_max": float(np.max(output)),
        "output_mean": float(np.mean(output)),
        "sample_indices": sample_indices.tolist(),
    }
    return result, output[sample_indices]


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "visser",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--scaling-samples",
        default=f"2000000,8000000,25000000,{BONDI_RAY_SAMPLES}",
    )
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--sustained-seconds", type=float, default=10.0)
    args = parser.parse_args()

    sizes = [int(value) for value in args.scaling_samples.split(",")]
    if any(value <= 0 for value in sizes):
        raise ValueError("all scaling sample counts must be positive")
    if args.min_repeats <= 0 or args.sustained_seconds <= 0.0:
        raise ValueError("min-repeats and sustained-seconds must be positive")

    source_file = Path(inspect.getfile(visser_module)).resolve()
    repo_root = source_file.parents[4]
    source_hash = _sha256(source_file)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / f"{stamp}_{source_hash[:12]}"
    output_dir.mkdir(parents=True, exist_ok=False)

    visser = VisserShielding(data_dir=args.data_dir, auto_download=False)
    visser.theta_interpolated_b(
        "co",
        np.full(100_000, 1.0e14),
        np.full(100_000, 1.0e18),
        np.full(100_000, 1.0),
    )

    measurements = []
    samples = {}
    largest = max(sizes)
    for n_samples in sizes:
        min_seconds = args.sustained_seconds if n_samples == largest else 0.0
        result, output_sample = _measure(
            visser,
            n_samples,
            min_repeats=args.min_repeats,
            min_seconds=min_seconds,
        )
        measurements.append(result)
        samples[f"n_{n_samples}"] = output_sample
        print(
            f"n={n_samples:,}: {result['median_wall_seconds']:.6f} s, "
            f"{result['throughput_samples_per_second'] / 1.0e6:.3f} M samples/s, "
            f"{result['repeats']} repeats"
        )

    table_files = [visser.data_dir / name for name in visser._available_b_family()[1]]
    summary = {
        "benchmark": "gow17_healpix_visser_scaling",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bondi_chunk_cells": BONDI_CHUNK_CELLS,
        "nside": 4,
        "rays_per_cell": NSIDE4_RAYS,
        "bondi_ray_samples": BONDI_RAY_SAMPLES,
        "source_file": str(source_file),
        "source_sha256": source_hash,
        "git_sha": _git_sha(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "table_sha256": {path.name: _sha256(path) for path in table_files},
        "max_rss_bytes": _max_rss_bytes(),
        "measurements": measurements,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    np.savez(output_dir / "output_samples.npz", **samples)

    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.plot(
        [item["n_samples"] / 1.0e6 for item in measurements],
        [item["throughput_samples_per_second"] / 1.0e6 for item in measurements],
        marker="o",
    )
    axis.set_xlabel("Ray samples (millions)")
    axis.set_ylabel("Throughput (million samples/s)")
    axis.set_title("Visser linewidth interpolation scaling")
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "scaling.png", dpi=180)
    plt.close(figure)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
