from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from validation.gow17_fig2 import run as run_gow17_fig2
    from validation.carbon_reduced_slab import run as run_carbon_reduced_slab
    from validation.thermochem_slab import run as run_thermochem_slab

    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / "validation_out"
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("gow17_fig2", run_gow17_fig2),
        ("carbon_reduced_slab", run_carbon_reduced_slab),
        ("thermochem_slab", run_thermochem_slab),
    ]

    failures: list[str] = []
    for name, fn in jobs:
        try:
            fn(out_root / name)
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")

    if failures:
        for msg in failures:
            print(msg, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
