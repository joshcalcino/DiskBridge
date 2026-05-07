from __future__ import annotations

from pathlib import Path

from driver import InfallStream1DConfig, RADIATION_MODES, RADIATION_MODE_DRAINE_SCALAR, run_with_error_plot


def _radius_rsun_from_lstar_teff(lstar_lsun: float, teff_K: float, *, teff_sun_K: float = 5772.0) -> float:
    return float((float(lstar_lsun) ** 0.5) * (float(teff_sun_K) / float(teff_K)) ** 2)


PMS_STELLAR_PRESETS = {
    4000: {
        "anchor_star": "TW Hya",
        "mstar_msun": 0.6,
        "rstar_rsun": 1.22,
        "teff_K": 4000.0,
        # Herczeg et al. (2023): 25-year average accretion rate = 2.51e-9 Msun/yr
        "mdot_msun_yr": 2.51e-9,
    },
    6000: {
        "anchor_star": "AK Sco",
        "mstar_msun": 1.2,
        "rstar_rsun": 1.3,
        "teff_K": 6000.0,
        # Wichittanakom et al. (2020), as quoted in Brittain et al. (2025): log10(Mdot) = -8.2
        # => 6.31e-9 Msun/yr. Fairlamb et al. (2015) reported an upper limit <= 1.26e-8 Msun/yr.
        "mdot_msun_yr": 6.31e-9,
    },
    6750: {
        "anchor_star": "CQ Tau",
        "mstar_msun": 1.5,
        "lstar_lsun": 6.7,
        "rstar_rsun": _radius_rsun_from_lstar_teff(6.7, 6750.0),
        "teff_K": 6750.0,
        # User-provided log10(Mdot / Msun yr^-1) = -7.145 => 7.16e-8 Msun/yr.
        "mdot_msun_yr": 7.161434102129027e-8,
    },
    7500: {
        "anchor_star": "HD 169142",
        "mstar_msun": 1.6,
        "rstar_rsun": 1.6,
        "teff_K": 7500.0,
        # Wagner et al. (2015): May 2013 Paβ/Brγ-based accretion rate = (1.5-2.7)e-9 Msun/yr
        # Using midpoint here. Note: literature spans to higher values (~4.5e-8 Msun/yr in some homogeneous HAeBe compilations).
        "mdot_msun_yr": 2.1e-9,
    },
    10000: {
        "anchor_star": "AB Aur",
        "mstar_msun": 2.4,
        "rstar_rsun": 2.3,
        "teff_K": 10000.0,
        # Zhou et al. (2025): average log10(Mdot) = -6.28 +/- 0.12
        # => 5.25e-7 Msun/yr. Literature spread is broad (~1.8e-8 to 7.4e-7 Msun/yr depending on method/epoch).
        "mdot_msun_yr": 5.25e-7,
    },
}
DEFAULT_PRESET_TEFF_K = next(iter(PMS_STELLAR_PRESETS))

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    out_root = Path(__file__).resolve().parent / "out"
    stellar = PMS_STELLAR_PRESETS[int(args.teff_K)]

    cfg = InfallStream1DConfig(
        evolve_energy=bool(args.evolve_energy),
        estimate_tdust=bool(args.estimate_tdust),
        track_infall_equilibrium=bool(args.track_infall_equilibrium),
        mstar_msun=float(stellar["mstar_msun"]),
        rstar_rsun=float(stellar["rstar_rsun"]),
        teff_K=float(stellar["teff_K"]),
        mdot_msun_yr=float(stellar.get("mdot_msun_yr", 0.0)),
        min_chi=float(args.min_chi),
        radiation_mode=str(args.radiation_mode),
        make_movies=bool(args.make_movies),
        max_dlnchi=float(args.max_dlnchi),
        output_time_power=float(args.output_time_power),
    )

    teff_tag = f"{int(cfg.teff_K)}"
    mode_tag = "" if cfg.radiation_mode == RADIATION_MODE_DRAINE_SCALAR else f"_rad_{cfg.radiation_mode}"
    case_dir = out_root / (
        f"energy_{int(cfg.evolve_energy)}_tdust_{int(cfg.estimate_tdust)}"
        f"_eqtrack_{int(cfg.track_infall_equilibrium)}_teff_{teff_tag}K"
        f"{mode_tag}"
    )
    return int(run_with_error_plot(case_dir, cfg))


if __name__ == "__main__":
    raise SystemExit(main())
