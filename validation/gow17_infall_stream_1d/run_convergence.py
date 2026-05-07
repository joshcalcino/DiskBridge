from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from driver import (
    InfallStream1DConfig,
    RADIATION_MODES,
    RADIATION_MODE_DRAINE_SCALAR,
    run_with_error_plot,
)
from run import DEFAULT_PRESET_TEFF_K, PMS_STELLAR_PRESETS


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run gow17_infall_stream_1d convergence sweeps over output cadence and n_cells."
    )
    parser.add_argument(
        "--nH-cm3",
        "--nH",
        dest="nH_cm3",
        type=float,
        required=True,
        help="Single hydrogen nuclei density to test, in cm^-3.",
    )
    parser.add_argument(
        "--n-steps-list",
        dest="n_steps_list",
        type=int,
        nargs="+",
        default=500,
        help="List of recorded output/history step counts to sweep.",
    )
    parser.add_argument(
        "--n-cells-list",
        dest="n_cells_list",
        type=int,
        nargs="+",
        default=(256, 512, 1024),
        help="List of 1D grid cell counts to sweep.",
    )
    parser.add_argument(
        "--max-dlnchi",
        dest="max_dlnchi",
        type=float,
        default=InfallStream1DConfig.max_dlnchi,
        help="Maximum allowed change in ln(chi) of the front cell per adaptive chemistry step.",
    )
    parser.add_argument(
        "--output-time-power",
        dest="output_time_power",
        type=float,
        default=InfallStream1DConfig.output_time_power,
        help="Power-law bias for recorded output times; values > 1 cluster outputs toward late infall.",
    )
    parser.add_argument(
        "--evolve-energy",
        "--evolve_energy",
        dest="evolve_energy",
        type=int,
        choices=(0, 1),
        required=True,
    )
    parser.add_argument(
        "--estimate-tdust",
        "--estimate_tdust",
        dest="estimate_tdust",
        type=int,
        choices=(0, 1),
        required=True,
    )
    parser.add_argument(
        "--track-infall-equilibrium",
        "--track_infall_equilibrium",
        dest="track_infall_equilibrium",
        type=int,
        choices=(0, 1),
        default=int(InfallStream1DConfig.track_infall_equilibrium),
        help="Also solve the chemistry to local equilibrium at each infall output snapshot.",
    )
    parser.add_argument(
        "--teff-k",
        "--teff_K",
        "--teff",
        dest="teff_K",
        type=int,
        choices=tuple(PMS_STELLAR_PRESETS),
        default=DEFAULT_PRESET_TEFF_K,
        help="Stellar effective temperature in K. Supported PMS presets: 4000, 6000, 6750, 7500, 10000.",
    )
    parser.add_argument(
        "--min-chi",
        dest="min_chi",
        type=float,
        default=InfallStream1DConfig.min_chi,
        help="Minimum UV field strength chi applied during equilibrium and evolution.",
    )
    parser.add_argument(
        "--radiation-mode",
        dest="radiation_mode",
        choices=RADIATION_MODES,
        default=RADIATION_MODE_DRAINE_SCALAR,
        help=(
            "Radiation treatment. draine_scalar reproduces the old broad-band "
            "Draine-shaped approximation; stellar_products uses spectrum-aware "
            "GOW17 UV products while retaining the 1D incident slab geometry."
        ),
    )
    parser.add_argument(
        "--make-movies",
        dest="make_movies",
        type=int,
        choices=(0, 1),
        default=int(InfallStream1DConfig.make_movies),
        help="Write mp4 abundance movies.",
    )
    parser.add_argument(
        "--stream-length-au",
        dest="stream_length_au",
        type=float,
        default=InfallStream1DConfig.stream_length_au,
        help="Physical length of the 1D stream segment in au.",
    )
    parser.add_argument(
        "--r-face-start-au",
        dest="r_face_start_au",
        type=float,
        default=InfallStream1DConfig.r_face_start_au,
        help="Initial star-to-front distance in au.",
    )
    parser.add_argument(
        "--r-face-stop-au",
        dest="r_face_stop_au",
        type=float,
        default=InfallStream1DConfig.r_face_stop_au,
        help="Final star-to-front distance in au.",
    )
    parser.add_argument(
        "--out-dir-name",
        dest="out_dir_name",
        default="out_convergence",
        help="Top-level output directory name inside gow17_infall_stream_1d.",
    )
    args = parser.parse_args()

    n_steps_list = tuple(sorted({int(n) for n in args.n_steps_list}))
    n_cells_list = tuple(sorted({int(n) for n in args.n_cells_list}))
    if any(n <= 0 for n in n_steps_list):
        raise ValueError("all n_steps values must be > 0")
    if any(n <= 0 for n in n_cells_list):
        raise ValueError("all n_cells values must be > 0")
    if not (args.nH_cm3 > 0.0):
        raise ValueError("nH_cm3 must be > 0")
    if not (args.max_dlnchi > 0.0):
        raise ValueError("max_dlnchi must be > 0")
    if not (args.output_time_power > 0.0):
        raise ValueError("output_time_power must be > 0")

    stellar = PMS_STELLAR_PRESETS[int(args.teff_K)]

    base_cfg = InfallStream1DConfig(
        nH_list_cm3=(float(args.nH_cm3),),
        mstar_msun=float(stellar["mstar_msun"]),
        rstar_rsun=float(stellar["rstar_rsun"]),
        teff_K=float(stellar["teff_K"]),
        mdot_msun_yr=float(stellar.get("mdot_msun_yr", 0.0)),
        min_chi=float(args.min_chi),
        stream_length_au=float(args.stream_length_au),
        r_face_start_au=float(args.r_face_start_au),
        r_face_stop_au=float(args.r_face_stop_au),
        max_dlnchi=float(args.max_dlnchi),
        output_time_power=float(args.output_time_power),
        radiation_mode=str(args.radiation_mode),
        make_movies=bool(args.make_movies),
        evolve_energy=bool(args.evolve_energy),
        estimate_tdust=bool(args.estimate_tdust),
        track_infall_equilibrium=bool(args.track_infall_equilibrium),
    )

    out_root = Path(__file__).resolve().parent / str(args.out_dir_name)
    teff_tag = f"{int(base_cfg.teff_K)}"
    nH_tag = f"{float(args.nH_cm3):.3e}".replace("+", "")
    sweep_dir = out_root / (
        f"nH_{nH_tag}_energy_{int(base_cfg.evolve_energy)}"
        f"_tdust_{int(base_cfg.estimate_tdust)}"
        f"_eqtrack_{int(base_cfg.track_infall_equilibrium)}_teff_{teff_tag}K"
        f"{'' if base_cfg.radiation_mode == RADIATION_MODE_DRAINE_SCALAR else f'_rad_{base_cfg.radiation_mode}'}"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        sweep_dir / "sweep_inputs.json",
        {
            "job": "gow17_infall_stream_1d_convergence",
            "time": datetime.now().isoformat(timespec="seconds"),
            "anchor_star": stellar.get("anchor_star"),
            "stellar_preset": stellar,
            "base_config": asdict(base_cfg),
            "sweep": {
                "n_steps_list": n_steps_list,
                "n_cells_list": n_cells_list,
            },
        },
    )

    rows: list[dict] = []
    overall_status = 0
    for n_cells in n_cells_list:
        for n_steps in n_steps_list:
            cfg = InfallStream1DConfig(
                **{
                    **asdict(base_cfg),
                    "n_cells": int(n_cells),
                    "n_steps": int(n_steps),
                }
            )
            case_dir = sweep_dir / f"cells_{int(n_cells):04d}_steps_{int(n_steps):04d}"
            status = int(run_with_error_plot(case_dir, cfg))
            rows.append(
                {
                    "case_dir": str(case_dir),
                    "n_cells": int(n_cells),
                    "n_steps": int(n_steps),
                    "max_dlnchi": float(cfg.max_dlnchi),
                    "output_time_power": float(cfg.output_time_power),
                    "status": int(status),
                }
            )
            overall_status = max(overall_status, status)

    _write_json(sweep_dir / "sweep_summary.json", {"rows": rows})
    return int(overall_status)


if __name__ == "__main__":
    raise SystemExit(main())
