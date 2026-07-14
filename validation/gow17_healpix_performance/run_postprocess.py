"""Benchmark complete post-ray PDR shielding algebra at Bondi chunk scale."""

# db-keywords: validation, shielding, co-shielding, healpix-columns, gow17, performance
# db-role: validation
# db-scope: script
# db-purpose: Sustained canonical post-ray shielding performance validation.

from __future__ import annotations

import argparse
import gc
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

import diskbridge.chemistry.shielding.healpix_columns as columns_module
import diskbridge.chemistry.shielding.h2_db96 as h2_module
import diskbridge.chemistry.shielding.healpix_utils as utils_module
import diskbridge.chemistry.shielding.visser_shielding as visser_module
from diskbridge.chemistry.shielding.h2_db96 import h2_self_shielding_db96
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding


BONDI_CHUNK_CELLS = 349_525
NSIDE4_RAYS = 192


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


def _inputs(n_cells: int) -> dict[str, np.ndarray]:
    n_samples = n_cells * NSIDE4_RAYS
    index = np.arange(n_samples, dtype=np.float64)
    phase = (index % 10_007.0) / 10_006.0
    nco = np.power(10.0, 9.0 + 11.0 * phase).reshape(n_cells, NSIDE4_RAYS)
    nh2 = np.power(10.0, 9.0 + 14.0 * phase).reshape(n_cells, NSIDE4_RAYS)
    nc = np.power(10.0, 8.0 + 12.0 * phase).reshape(n_cells, NSIDE4_RAYS)
    bco = 0.03 + 4.97 * phase
    bh2 = 0.05 + 2.95 * phase
    b2co = (nco.reshape(-1) * bco * bco).reshape(n_cells, NSIDE4_RAYS)
    b2h2 = (nh2.reshape(-1) * bh2 * bh2).reshape(n_cells, NSIDE4_RAYS)
    weights = np.full(
        (n_cells, NSIDE4_RAYS),
        1.0 / NSIDE4_RAYS,
        dtype=np.float64,
    )
    return {
        "nco": nco,
        "nh2": nh2,
        "nc": nc,
        "b2co": b2co,
        "b2h2": b2h2,
        "weights": weights,
    }


def _evaluate(
    visser: VisserShielding,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    b2h2 = inputs["b2h2"].copy()
    b2co = inputs["b2co"].copy()
    b_h2 = columns_module._effective_b_from_columns(
        b2h2,
        inputs["nh2"],
        fallback_kms=0.2,
    )
    h2_rays = h2_self_shielding_db96(inputs["nh2"], b5=b_h2)
    c_rays = columns_module._c_shielding(inputs["nc"], inputs["nh2"])

    b_co = columns_module._effective_b_from_columns(
        b2co,
        inputs["nco"],
        fallback_kms=0.2,
    )
    co_rays = visser.theta_interpolated_b(
        "co",
        inputs["nco"],
        inputs["nh2"],
        b_co,
    )
    theta_h2 = columns_module._average_rays(h2_rays, inputs["weights"])
    theta_c = columns_module._average_rays(c_rays, inputs["weights"])
    theta_co = columns_module._average_rays(co_rays, inputs["weights"])
    theta_pdr = columns_module._average_product_rays(
        h2_rays,
        co_rays,
        inputs["weights"],
    )
    return {
        "theta_h2": theta_h2,
        "theta_c": theta_c,
        "theta_co": theta_co,
        "theta_pdr": theta_pdr,
    }


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
        default=Path(__file__).resolve().parent / "outputs" / "postprocess",
    )
    parser.add_argument("--cells", type=int, default=BONDI_CHUNK_CELLS)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--sustained-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.cells <= 0 or args.min_repeats <= 0 or args.sustained_seconds <= 0.0:
        raise ValueError("cells, min-repeats, and sustained-seconds must be positive")

    source_file = Path(inspect.getfile(columns_module)).resolve()
    repo_root = source_file.parents[4]
    source_hash = _sha256(source_file)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / f"{stamp}_{source_hash[:12]}"
    output_dir.mkdir(parents=True, exist_ok=False)

    visser = VisserShielding(data_dir=args.data_dir, auto_download=False)
    _, table_names = visser._available_b_family()
    table_hashes = {
        name: _sha256(Path(visser.data_dir) / name)
        for name in table_names
    }
    warm_inputs = _inputs(512)
    _evaluate(visser, warm_inputs)
    del warm_inputs
    gc.collect()

    inputs = _inputs(args.cells)
    wall_times: list[float] = []
    cpu_times: list[float] = []
    elapsed = 0.0
    output: dict[str, np.ndarray] = {}
    while len(wall_times) < args.min_repeats or elapsed < args.sustained_seconds:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        output = _evaluate(visser, inputs)
        cpu_times.append(time.process_time() - cpu_start)
        wall_times.append(time.perf_counter() - wall_start)
        elapsed += wall_times[-1]

    median_wall = statistics.median(wall_times)
    median_cpu = statistics.median(cpu_times)
    sample_indices = np.linspace(
        0,
        args.cells - 1,
        min(8192, args.cells),
        dtype=np.int64,
    )
    samples = {name: values[sample_indices] for name, values in output.items()}
    measurement = {
        "n_cells": args.cells,
        "n_rays": NSIDE4_RAYS,
        "n_samples": args.cells * NSIDE4_RAYS,
        "repeats": len(wall_times),
        "wall_seconds": wall_times,
        "cpu_seconds": cpu_times,
        "median_wall_seconds": median_wall,
        "median_cpu_seconds": median_cpu,
        "median_cpu_cores": median_cpu / median_wall,
        "throughput_samples_per_second": args.cells * NSIDE4_RAYS / median_wall,
        "implementation": "parallel_fused",
        "output_min": {name: float(np.min(value)) for name, value in output.items()},
        "output_max": {name: float(np.max(value)) for name, value in output.items()},
        "output_mean": {name: float(np.mean(value)) for name, value in output.items()},
        "sample_indices": sample_indices.tolist(),
    }
    summary = {
        "benchmark": "gow17_healpix_postprocess",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_file),
        "source_sha256": source_hash,
        "source_files": {
            str(Path(inspect.getfile(module)).resolve()): _sha256(
                Path(inspect.getfile(module)).resolve()
            )
            for module in (
                columns_module,
                h2_module,
                utils_module,
                visser_module,
            )
        },
        "visser_table_sha256": table_hashes,
        "git_sha": _git_sha(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
        "max_rss_bytes": _max_rss_bytes(),
        "measurement": measurement,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    np.savez(output_dir / "output_samples.npz", **samples)

    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    names = list(output)
    axis.bar(names, [measurement["output_mean"][name] for name in names])
    axis.set_yscale("log")
    axis.set_ylabel("Mean shielding factor")
    axis.set_title("Bondi-scale post-ray shielding outputs")
    figure.savefig(output_dir / "shielding_means.png", dpi=180)
    plt.close(figure)
    print(
        f"{measurement['implementation']}: {median_wall:.6f} s, "
        f"{measurement['throughput_samples_per_second'] / 1.0e6:.3f} M samples/s, "
        f"{measurement['median_cpu_cores']:.2f} CPU cores, "
        f"{measurement['repeats']} repeats"
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
