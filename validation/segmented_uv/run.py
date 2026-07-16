#!/usr/bin/env python3
"""Run the full-domain and segmented UV RADMC-3D validation."""

# db-keywords: uv-products, validation, radmc3d, model, mesh, field, plotting
# db-role: validation
# db-scope: validation
# db-purpose: Compare full-domain and segmented UV radiation fields on a shared spherical model.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import diskbridge
from diskbridge._config import get_config
from diskbridge._units import Quantity, units
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.model.core import Model, SubModel
from diskbridge.radmc3d import RadModel
from diskbridge.radmc3d.model import _validate_radmc_photon_count
from diskbridge.radmc3d.uv_products import (
    uv_product_edges_from_specs,
    uv_product_specs_from_config,
)
from diskbridge.radmc3d.wavelengths import build_mcmono_wavelengths

try:
    from .plot_diagnostics import write_radial_comparison, write_uv_product_comparison
except ImportError:
    from plot_diagnostics import write_radial_comparison, write_uv_product_comparison


BASELINE_MONO_MULTIPLIER = 10


def make_spherical_logr_model(
    *,
    nr: int,
    nphi: int,
    ntheta: int,
    r_min_au: float,
    r_max_au: float,
    rho_gas_cgs: float,
    dust_to_gas: float,
) -> Model:
    r_edges = np.logspace(np.log10(r_min_au), np.log10(r_max_au), nr + 1) * units("au")
    phi_edges = np.linspace(0.0, 2.0 * np.pi, nphi + 1) * units("radian")
    theta_edges = np.linspace(0.0, np.pi, ntheta + 1) * units("radian")

    mesh = Mesh.spherical(
        r=Axis(edges=r_edges),
        theta=Axis(edges=theta_edges),
        phi=Axis(edges=phi_edges),
    )

    model = Model()
    model.mesh = mesh
    model.coord_system = "spherical"
    model.gas = SubModel(model)

    shape = (nr, ntheta, nphi)
    rho = Quantity(np.full(shape, rho_gas_cgs, dtype=float), "g/cm^3")
    model.gas.register(
        "density",
        Field(quantity="density", data=rho, axis_order=("r", "theta", "phi")),
    )

    from diskbridge.model.dust import Dust

    model.dust = Dust(model)
    model.dust.set_distribution(mode="proportional", dust_to_gas_ratio=dust_to_gas)

    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", type=int, default=128)
    parser.add_argument("--nphi", type=int, default=24)
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--r-min-au", type=float, default=1.0)
    parser.add_argument("--r-max-au", type=float, default=1.0e5)
    parser.add_argument("--rho-gas-cgs", type=float, default=1.0e-22)
    parser.add_argument("--dust-to-gas", type=float, default=1.0e-2)
    parser.add_argument(
        "--mcmono-wavelength-spacing",
        type=str,
        default="log",
        choices=("log", "linear"),
    )
    parser.add_argument("--mcmono-n-wavelengths", type=int, default=None)
    parser.add_argument("--mcmono-uv-n-wavelengths", type=int, default=None)
    parser.add_argument("--mcmono-setthreads", type=int, default=None)
    parser.add_argument("--max-splits", type=int, default=3)
    parser.add_argument(
        "--segmented-final-nphot-multiplier",
        type=float,
        default=None,
        help="Multiplier applied only to the terminal segmented run.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Working directory for runs and outputs (default: script directory)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    workdir = Path(args.workdir).resolve() if args.workdir is not None else script_dir
    params_path = workdir / "params.txt"

    p = diskbridge.read_params(str(params_path))
    nphot_thermal = int(p.nphot_thermal)
    nphot_mono = int(p.nphot_mono)
    baseline_nphot_thermal = _validate_radmc_photon_count(
        nphot_thermal,
        name="baseline nphot_thermal",
    )
    baseline_nphot_mono = _validate_radmc_photon_count(
        BASELINE_MONO_MULTIPLIER * nphot_mono,
        name="baseline nphot_mono",
    )
    uv_min_um = p.uv_min.to("micron")
    if p.lambda_min.to("micron") > uv_min_um:
        p.lambda_min = uv_min_um

    diskbridge.params = p
    baseline_dir = workdir / "baseline_run"
    segmented_dir = workdir / "segmented_run"
    plots_dir = workdir / "plots"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    segmented_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    (baseline_dir / "params.txt").write_text(params_path.read_text())
    (segmented_dir / "params.txt").write_text(params_path.read_text())

    baseline_model = make_spherical_logr_model(
        nr=int(args.nr),
        nphi=int(args.nphi),
        ntheta=int(args.ntheta),
        r_min_au=float(args.r_min_au),
        r_max_au=float(args.r_max_au),
        rho_gas_cgs=float(args.rho_gas_cgs),
        dust_to_gas=float(args.dust_to_gas),
    )
    segmented_model = make_spherical_logr_model(
        nr=int(args.nr),
        nphi=int(args.nphi),
        ntheta=int(args.ntheta),
        r_min_au=float(args.r_min_au),
        r_max_au=float(args.r_max_au),
        rho_gas_cgs=float(args.rho_gas_cgs),
        dust_to_gas=float(args.dust_to_gas),
    )

    rad_base = RadModel(baseline_model, model_dir=baseline_dir)
    rad_base.writer.write_all_input_files(baseline_dir)
    rad_base.writer.ensure_external_source(baseline_dir)
    rad_base.writer.compute_and_write_dust_opacities(baseline_dir)

    uv_cfg = get_config().get("radmc3d", {}).get("uv_products", {})
    product_specs = uv_product_specs_from_config(uv_cfg)
    product_edges_um = uv_product_edges_from_specs(product_specs) * 1.0e-3
    broad_spec = next(spec for spec in product_specs if spec.field_name == "chi_broad")
    uv_min = Quantity(float(broad_spec.band.lam_min_nm), "nm")
    uv_max = Quantity(float(broad_spec.band.lam_max_nm), "nm")
    n_wavelengths = (
        int(args.mcmono_uv_n_wavelengths)
        if args.mcmono_uv_n_wavelengths is not None
        else int(args.mcmono_n_wavelengths)
        if args.mcmono_n_wavelengths is not None
        else int(p.uv_n_wavelengths)
    )
    mcmono_wav_um_use = build_mcmono_wavelengths(
        wavelength_source="uv",
        wavelength_file=None,
        uv_min_um=float(uv_min.to("micron").magnitude),
        uv_max_um=float(uv_max.to("micron").magnitude),
        n_wavelengths=n_wavelengths,
        n_uv_enforce=0,
        spacing=args.mcmono_wavelength_spacing,
        extra_enforced_wavelengths_um=product_edges_um,
    )

    rad_base.compute_temperature(
        nphot=int(baseline_nphot_thermal),
        output_dir=baseline_dir / "radmc3d_outputs" / "temperature",
        force=False,
    )
    rad_base.compute_mcmono(
        nphot=int(baseline_nphot_mono),
        output_dir=baseline_dir / "radmc3d_outputs" / "mcmono",
        force=False,
        wavelengths_um=mcmono_wav_um_use,
        uv_min=uv_min,
        uv_max=uv_max,
        setthreads=args.mcmono_setthreads,
        compute_uv_products=True,
    )

    rad_seg = RadModel(segmented_model, model_dir=segmented_dir)
    result_seg = rad_seg.compute_segmented_rt(
        nphot_therm=nphot_thermal,
        nphot_mono=nphot_mono,
        mcmono_n_wavelengths=args.mcmono_n_wavelengths,
        mcmono_uv_n_wavelengths=args.mcmono_uv_n_wavelengths,
        mcmono_wavelength_spacing=args.mcmono_wavelength_spacing,
        mcmono_wavelengths_um=mcmono_wav_um_use,
        segmented_final_nphot_multiplier=args.segmented_final_nphot_multiplier,
        force=False,
        max_splits=args.max_splits,
        diagnostic_plots=True,
        plots_dir=plots_dir / "segmented_rt",
    )

    segmented_mode = str(result_seg.get("mode", "unknown"))
    split_radii_au = list(result_seg.get("split_radii_au", []))
    scout_runs = list(result_seg.get("scout_runs", []))
    product_names = tuple(spec.field_name for spec in product_specs)

    for m, name in (
        (baseline_model, "baseline"),
        (segmented_model, "segmented"),
    ):
        if "dust_temperature" not in m.gas:
            raise RuntimeError(f"missing {name} dust_temperature field")
        if "chi" not in m.gas:
            raise RuntimeError(f"missing {name} chi field")
        missing_products = [field for field in product_names if field not in m.gas]
        if missing_products:
            raise RuntimeError(
                f"missing {name} UV products: {', '.join(missing_products)}"
            )

    uv_product_metrics = write_uv_product_comparison(
        baseline_model=baseline_model,
        segmented_model=segmented_model,
        output_dir=plots_dir / "uv_products",
    )

    from diskbridge.model.profiles import compute_cell_volumes

    write_radial_comparison(
        baseline_model=baseline_model,
        segmented_model=segmented_model,
        output_path=plots_dir / "compare_radial_profiles.png",
        split_radii_au=split_radii_au,
    )

    def _volume_weighted_rms_fracdiff(
        *,
        base: np.ndarray,
        other: np.ndarray,
        volumes: np.ndarray,
        valid_mask: np.ndarray,
    ) -> float:
        """Compute volume-weighted RMS fractional difference.

        Parameters
        ----------
        base
            Reference field array.
        other
            Comparison field array.
        volumes
            Cell volumes array matching the field shape.
        valid_mask
            Boolean mask selecting cells to include.

        Returns
        -------
        float
            sqrt(sum(V * ((other-base)/base)^2) / sum(V)) over valid cells.
        """
        if base.shape != other.shape:
            raise ValueError(f"shape mismatch: base={base.shape}, other={other.shape}")
        if volumes.shape != base.shape:
            raise ValueError(f"shape mismatch: volumes={volumes.shape}, base={base.shape}")

        mask = valid_mask & np.isfinite(base) & np.isfinite(other)
        if not np.any(mask):
            return float("nan")

        frac = np.zeros_like(base, dtype=float)
        frac[mask] = (other[mask] - base[mask]) / base[mask]
        w = np.asarray(volumes[mask], dtype=float)
        wt_sum = float(np.sum(w))
        if wt_sum <= 0.0:
            return float("nan")
        return float(np.sqrt(np.sum(w * frac[mask] ** 2) / wt_sum))

    segments_info = list(result_seg.get("segments", []))

    temp_b = baseline_model.gas["dust_temperature"].data
    temp_s = segmented_model.gas["dust_temperature"].data

    temp_b_mag = np.asarray(getattr(temp_b, "magnitude", temp_b))
    temp_s_mag = np.asarray(getattr(temp_s, "magnitude", temp_s))

    chi_b = baseline_model.gas["chi"].data
    chi_s = segmented_model.gas["chi"].data

    chi_b_mag = np.asarray(getattr(chi_b, "magnitude", chi_b))
    chi_s_mag = np.asarray(getattr(chi_s, "magnitude", chi_s))

    volumes = compute_cell_volumes(baseline_model)
    if volumes.shape != temp_b_mag.shape:
        raise ValueError(
            f"volume shape {volumes.shape} does not match field shape {temp_b_mag.shape}"
        )

    t_valid = np.isfinite(temp_b_mag) & (temp_b_mag > 0.0)
    chi_valid = np.isfinite(chi_b_mag) & (chi_b_mag > 0.0)

    t_rms = _volume_weighted_rms_fracdiff(
        base=temp_b_mag,
        other=temp_s_mag,
        volumes=volumes,
        valid_mask=t_valid,
    )
    chi_rms = _volume_weighted_rms_fracdiff(
        base=chi_b_mag,
        other=chi_s_mag,
        volumes=volumes,
        valid_mask=chi_valid,
    )

    n_wav_use = int(np.asarray(mcmono_wav_um_use, dtype=float).size)
    baseline_photons_total = float(baseline_nphot_thermal) + float(
        baseline_nphot_mono
    ) * float(n_wav_use)

    nphot_ratio = float(getattr(p, "segmented_nphot_ratio", 1.0))
    final_nphot_multiplier = float(
        result_seg.get(
            "segmented_final_nphot_multiplier",
            args.segmented_final_nphot_multiplier
            if args.segmented_final_nphot_multiplier is not None
            else getattr(p, "segmented_final_nphot_multiplier", 1.0),
        )
    )
    nphot_therm_nominal = nphot_thermal
    nphot_mono_nominal = nphot_mono
    nphot_therm_final = int(final_nphot_multiplier * nphot_therm_nominal)
    nphot_mono_final = int(final_nphot_multiplier * nphot_mono_nominal)
    nphot_therm_intermediate = int(nphot_ratio * nphot_therm_nominal)
    nphot_mono_intermediate = int(nphot_ratio * nphot_mono_nominal)
    if nphot_therm_intermediate < 1:
        raise ValueError(
            f"segmented_nphot_ratio={nphot_ratio} yields nphot_therm_intermediate={nphot_therm_intermediate}"
        )
    if nphot_mono_intermediate < 1:
        raise ValueError(
            f"segmented_nphot_ratio={nphot_ratio} yields nphot_mono_intermediate={nphot_mono_intermediate}"
        )

    photon_packages = {
        key: float(value)
        for key, value in dict(result_seg.get("photon_packages", {})).items()
    }
    required_package_fields = {
        "scout_thermal",
        "scout_mono",
        "boundary_calibration_mono",
        "final_thermal",
        "final_mono",
        "total",
    }
    missing_package_fields = sorted(required_package_fields - photon_packages.keys())
    if missing_package_fields:
        raise RuntimeError(
            "segmented runner did not report photon accounting: "
            + ", ".join(missing_package_fields)
        )
    segmented_photons_total = float(photon_packages["total"])

    metrics = {
        "n_wavelengths_mcmono": int(n_wav_use),
        "baseline": {
            "mcmono_multiplier": int(BASELINE_MONO_MULTIPLIER),
            "nphot_thermal": int(baseline_nphot_thermal),
            "nphot_mono": int(baseline_nphot_mono),
            "photon_packages_total": float(baseline_photons_total),
        },
        "segmented": {
            "mode": segmented_mode,
            "segmented_nphot_ratio": float(nphot_ratio),
            "segmented_final_nphot_multiplier": float(final_nphot_multiplier),
            "nphot_thermal_nominal": int(nphot_therm_nominal),
            "nphot_mono_nominal": int(nphot_mono_nominal),
            "nphot_thermal_final": int(nphot_therm_final),
            "nphot_mono_final": int(nphot_mono_final),
            "nphot_thermal_intermediate": int(nphot_therm_intermediate),
            "nphot_mono_intermediate": int(nphot_mono_intermediate),
            "n_scout_runs": int(len(scout_runs)),
            "n_segments": int(len(segments_info)),
            "photon_packages_by_role": photon_packages,
            "photon_packages_total": float(segmented_photons_total),
        },
        "errors": {
            "tdust_volume_weighted_rms_frac": float(t_rms),
            "chi_volume_weighted_rms_frac": float(chi_rms),
        },
        "uv_products": uv_product_metrics,
        "efficiency": {
            "baseline_over_segmented_photon_packages": float(
                baseline_photons_total / segmented_photons_total
            )
            if segmented_photons_total > 0.0
            else float("nan"),
        },
        "split_radii_au": [float(x) for x in split_radii_au],
    }
    (plots_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(f"workdir: {workdir}")
    print(f"baseline_dir: {baseline_dir}")
    print(f"segmented_dir: {segmented_dir}")
    print(f"split_radii_au: {split_radii_au}")
    print(f"wrote: {plots_dir / 'metrics.json'}")
    print(f"wrote: {plots_dir / 'compare_radial_profiles.png'}")
    print(f"wrote: {plots_dir / 'uv_products'}")
    print(f"wrote: {plots_dir / 'segmented_rt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
