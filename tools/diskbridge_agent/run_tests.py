"""Run fast DiskBridge pytest tests by scope."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPES = {
    "all": ["tests"],
    "public": ["tests/test_public_api.py"],
    "gow17": [
        "tests/chemistry/test_gow17_config.py",
        "tests/chemistry/test_gow17_batch.py",
        "tests/chemistry/test_gow17_fig2.py",
    ],
    "uv": [
        "tests/radmc3d/test_uv_products.py",
    ],
    "shielding": [
        "tests/chemistry/test_shielding.py",
    ],
    "radmc3d": [
        "tests/radmc3d/test_data_grid.py",
        "tests/radmc3d/test_dust_opacity.py",
        "tests/radmc3d/test_dust_outputs.py",
        "tests/radmc3d/test_external_radiation.py",
        "tests/radmc3d/test_uv_products.py",
    ],
    "nonlte": [
        "tests/radmc3d/test_external_populations.py",
        "tests/radmc3d/test_external_population_workflow.py",
        "tests/radmc3d/test_nonlte.py",
    ],
    "masks": [
        "tests/model/test_masks.py",
    ],
}


def run_tests(scope: str) -> int:
    """Run pytest for a named DiskBridge test scope.

    Parameters
    ----------
    scope : str
        Test scope name.

    Returns
    -------
    int
        Process return code.
    """
    if scope not in SCOPES:
        valid = ", ".join(sorted(SCOPES))
        raise ValueError(f"Unknown scope {scope!r}. Valid scopes: {valid}")

    cmd = ["pytest", "-q", *SCOPES[scope]]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def main() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scope",
        choices=sorted(SCOPES),
        help="Named test scope to run",
    )
    args = parser.parse_args()
    raise SystemExit(run_tests(args.scope))


if __name__ == "__main__":
    main()
