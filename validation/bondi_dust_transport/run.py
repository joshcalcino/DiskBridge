#!/usr/bin/env python3
"""Validate a soft Joos edge and finite-time drift on one Bondi snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import diskbridge
from diskbridge.model.dust import stokes_number
from diskbridge.model.dust_transport import (
    build_smoothed_radial_gas_background,
    dust_diffusivity,
    evolve_radial_surface_density,
    pressure_drift_velocity,
    smoothed_log_pressure_gradient,
)


ROOT = Path(__file__).resolve().parents[2]
CASE_NAME = "m05_9_dm_f140_c10"
OUTPUT_DIR = (
    ROOT
    / "examples/bondi_low_density/plots/completed_uv_fields"
    / CASE_NAME
)
DATA_DIR = ROOT / "examples/bondi_low_density/data/7_b_0.5_300_9_d2_small_dm"
FRAME = 140
STELLAR_MASS = 0.5 * diskbridge.units("solar_mass")
AXIS_SUPPORT = 504.7830094260165 * diskbridge.units("au")
LOAD_R_MAX = 1_000.0 * diskbridge.units("au")
RHO_DISK_MIN = 1.0e-23 * diskbridge.units("g/cm^3")
RHO_CORE_MIN = 10.0 * RHO_DISK_MIN
JOOS_MAX_POLOIDAL_MACH = 1.0
JOOS_ROTATIONAL_SUPPORT_FACTOR = 2.0
SETTLING_ALPHA = 1.0e-2
TRANSPORT_TIME = 50_000.0 * diskbridge.units("yr")
DUST_NBIN = 15
DUST_SIZE_THRESHOLDS_UM = (0.1, 1.0, 10.0, 100.0)
REPRESENTATIVE_SIZE_TARGETS_UM = (0.068, 3.16, 147.0)
PRESSURE_SMOOTHING_BINS = 21
PRESSURE_LOG_GRADIENT_BOUNDS = (-5.0, -0.25)
SURFACE_DENSITY_LOG_GRADIENT_BOUNDS = (-5.0, 0.0)


def _parse_case_parameters(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        fields = raw_line.split()
        if len(fields) >= 2:
            values[fields[0].upper()] = fields[1]
    return values


def _variables_text(case_par: Path) -> str:
    values = _parse_case_parameters(case_par)
    required = ("SIGMA0", "ISMDENS", "ROUT")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Missing {missing} in {case_par}")
    return f"""ALPHA 0.001
ASPECTRATIO 0.03799
FLARINGINDEX 0.25
GAMMA 1.66666667
MU 2.35
NU 0
CS 1
SIGMA0 {values['SIGMA0']}
SIGMASLOPE 1.5
ISMDENS {values['ISMDENS']}
ROUT {values['ROUT']}
YMIN 7.06649505e13
YMAX {values['ROUT']}
WRITEENERGY 0
WRITEENERGYRAD 0
OMEGAFRAME 0
COORDINATES spherical
SETUP bondi
NX 330
NY 486
NZ 146
XMIN -3.14159265358979
XMAX 3.14159265358979
ZMIN 0.11715224
ZMAX 3.02444042
"""


def _load_model():
    case_par = DATA_DIR.parent / f"{DATA_DIR.name}.par"
    with tempfile.TemporaryDirectory(prefix="diskbridge-bondi-transport-") as tmp:
        stage = Path(tmp)
        for source in DATA_DIR.iterdir():
            if source.is_file():
                os.symlink(source.resolve(), stage / source.name)
        (stage / "variables.par").write_text(_variables_text(case_par))
        model = diskbridge.load_model(
            stage,
            reader="fargo",
            file_n=FRAME,
            file_units="cgs",
            length_scale=1.0,
            mass_scale=1.0,
            r_max=LOAD_R_MAX,
            downsample=None,
        )
    model.variables["mstar"] = STELLAR_MASS
    return model


def _surface_density_rphi(model, density: np.ndarray) -> np.ndarray:
    """Integrate spherical shells into approximate cylindrical columns."""
    r_edges = model.mesh.edges("r").to("cm").magnitude
    theta_edges = model.mesh.edges("theta").to("radian").magnitude
    dcos = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    shell_factor = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
    annulus_factor = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    columns = np.empty((density.shape[0], density.shape[2]), dtype=float)
    for radial_index in range(density.shape[0]):
        columns[radial_index] = (
            shell_factor[radial_index]
            * np.sum(density[radial_index] * dcos[:, None], axis=0)
            / annulus_factor[radial_index]
        )
    return columns


def _midplane_profiles(model, weight: np.ndarray) -> dict[str, diskbridge.Quantity]:
    density = model.gas["density"].data.to("g/cm^3").magnitude
    score = density * weight
    theta_indices = np.argmax(score, axis=1)[:, None, :]
    valid_radius = np.max(weight, axis=(1, 2)) > 0.0
    log_radius = np.log(model.mesh.centers("r").to("cm").magnitude)
    profiles: dict[str, diskbridge.Quantity] = {}
    for name, unit in (
        ("density", "g/cm^3"),
        ("temperature", "K"),
        ("pressure", "dyn/cm^2"),
    ):
        values = model.gas[name].data.to(unit).magnitude
        selected = np.take_along_axis(values, theta_indices, axis=1)[:, 0, :]
        profile = np.nanmedian(selected, axis=1)
        supported = valid_radius & np.isfinite(profile) & (profile > 0.0)
        if np.count_nonzero(supported) < 2:
            raise ValueError(f"Cannot construct a supported midplane {name} profile")
        profile = np.exp(
            np.interp(log_radius, log_radius[supported], np.log(profile[supported]))
        )
        profiles[name] = diskbridge.Quantity(profile, unit)
    return profiles


def _plot_mask_comparison(
    output: Path,
    model,
    weight: np.ndarray,
    hard_control: np.ndarray,
) -> None:
    radius = model.mesh.centers("r").to("au").magnitude
    phi = model.mesh.centers("phi").to("radian").magnitude
    new_mid = np.max(weight, axis=1)
    hard_mid = np.max(hard_control, axis=1)
    difference = new_mid - hard_mid

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = (
        (hard_mid, "Hard-edge control", "viridis", 0.0, 1.0),
        (new_mid, "Radially connected Joos", "viridis", 0.0, 1.0),
        (difference, "Connected minus hard control", "magma", 0.0, 1.0),
    )
    for ax, (values, title, cmap, vmin, vmax) in zip(axes.flat[:3], panels):
        image = ax.pcolormesh(
            radius,
            phi,
            values.T,
            shading="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.axvline(AXIS_SUPPORT.to("au").magnitude, color="white", linestyle=":")
        ax.set_xlim(0.0, LOAD_R_MAX.to("au").magnitude)
        ax.set_xlabel("spherical radius [au]")
        ax.set_ylabel("azimuth [rad]")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="disc weight")

    ax = axes[1, 1]
    ax.plot(radius, np.mean(hard_mid, axis=1), label="hard-edge control")
    ax.plot(radius, np.mean(new_mid, axis=1), label="radially connected")
    ax.axvline(
        AXIS_SUPPORT.to("au").magnitude,
        color="black",
        linestyle=":",
        label="axis support / old hard edge",
    )
    ax.set_xlim(0.0, LOAD_R_MAX.to("au").magnitude)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("spherical radius [au]")
    ax.set_ylabel("azimuth-mean column-max weight")
    ax.set_title("Radial profile")
    ax.legend(fontsize=8)
    fig.suptitle("m05_9_dm_f140_c10: imposed versus emergent disc edge")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_transport_profiles(
    output: Path,
    radius_au: np.ndarray,
    gas_sigma: np.ndarray,
    pressure_gradient_raw: np.ndarray,
    pressure_gradient_moving_average: np.ndarray,
    pressure_gradient_background: np.ndarray,
    results: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(radius_au, pressure_gradient_raw, color="0.65", label="raw")
    ax.plot(
        radius_au,
        pressure_gradient_moving_average,
        color="tab:orange",
        alpha=0.8,
        label="21-bin moving average",
    )
    ax.plot(
        radius_au,
        pressure_gradient_background,
        color="black",
        label="bounded gas background",
    )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_ylabel(r"$d\ln P/d\ln R$")
    ax.legend()

    ax = axes[0, 1]
    for result in results:
        ax.plot(radius_au, result["stokes"], label=result["label"])
    ax.set_yscale("log")
    ax.set_ylabel("Stokes number")
    ax.legend()

    ax = axes[1, 0]
    sigma_peak = max(
        float(np.max(result[key]))
        for result in results
        for key in ("sigma_initial", "sigma_final")
    )
    sigma_floor = sigma_peak * 1.0e-12
    for result in results:
        ax.plot(
            radius_au,
            np.maximum(result["sigma_initial"], sigma_floor),
            linestyle="--",
            alpha=0.75,
        )
        ax.plot(
            radius_au,
            np.maximum(result["sigma_final"], sigma_floor),
            label=result["label"],
        )
    ax.set_yscale("log")
    ax.set_ylabel(r"dust $\Sigma$ [g cm$^{-2}$]")
    ax.legend(title="solid: 50 kyr; dashed: initial", fontsize=8)

    ax = axes[1, 1]
    gas_safe = np.maximum(gas_sigma, np.max(gas_sigma) * 1.0e-15)
    for result in results:
        initial_ratio = np.maximum(result["sigma_initial"] / gas_safe, 1.0e-12)
        final_ratio = np.maximum(result["sigma_final"] / gas_safe, 1.0e-12)
        ax.plot(radius_au, initial_ratio, linestyle="--", alpha=0.75)
        ax.plot(radius_au, final_ratio, label=result["label"])
    ax.set_yscale("log")
    ax.set_ylabel("dust / gas surface-density ratio")

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlim(radius_au[0], radius_au[-1])
        ax.set_xlabel("radius [au]")
        ax.grid(alpha=0.2)
    fig.suptitle("m05_9_dm_f140_c10: finite-time radial dust transport")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_gas_background(
    output: Path,
    radius_au: np.ndarray,
    gas_sigma: np.ndarray,
    midplane: dict[str, diskbridge.Quantity],
    background,
    pressure_gradient_raw: np.ndarray,
    pressure_gradient_moving_average: np.ndarray,
) -> None:
    background_profiles = {
        "surface_density": background.surface_density.to("g/cm^2").magnitude,
        "density": background.midplane_density.to("g/cm^3").magnitude,
        "temperature": background.temperature.to("K").magnitude,
        "pressure": background.pressure.to("dyn/cm^2").magnitude,
    }
    raw_profiles = {
        "surface_density": gas_sigma,
        "density": midplane["density"].to("g/cm^3").magnitude,
        "temperature": midplane["temperature"].to("K").magnitude,
        "pressure": midplane["pressure"].to("dyn/cm^2").magnitude,
    }
    definitions = (
        ("surface_density", r"$\Sigma_g$ [g cm$^{-2}$]"),
        ("density", r"midplane $\rho_g$ [g cm$^{-3}$]"),
        ("temperature", "midplane temperature [K]"),
        ("pressure", r"midplane pressure [dyn cm$^{-2}$]"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat[:4], definitions):
        ax.plot(radius_au, raw_profiles[key], color="0.65", label="snapshot")
        ax.plot(radius_au, background_profiles[key], color="black", label="background")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(radius_au, pressure_gradient_raw, color="0.7", label="raw")
    ax.plot(
        radius_au,
        pressure_gradient_moving_average,
        color="tab:orange",
        label="moving average",
    )
    ax.plot(
        radius_au,
        background.pressure_log_gradient,
        color="black",
        label="bounded background",
    )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_ylabel(r"$d\ln P/d\ln R$")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    for key, label in (
        ("surface_density", r"$\Sigma_g$"),
        ("density", r"$\rho_{g,0}$"),
        ("temperature", "$T$"),
        ("pressure", "$P$"),
    ):
        ax.plot(
            radius_au,
            background_profiles[key] / raw_profiles[key],
            label=label,
        )
    ax.axhline(1.0, color="0.5", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("background / snapshot")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlim(radius_au[0], radius_au[-1])
        ax.set_xlabel("radius [au]")
        ax.grid(alpha=0.2)
    fig.suptitle("m05_9_dm_f140_c10: smoothed radial gas background")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_largest_bin_azimuth(
    output: Path,
    model,
    gas_sigma_rphi: np.ndarray,
    result: dict,
) -> None:
    radius = model.mesh.centers("r").to("au").magnitude
    phi = model.mesh.centers("phi").to("radian").magnitude
    initial = result["sigma_rphi_initial"]
    scale = np.divide(
        result["sigma_final"],
        result["sigma_initial"],
        out=np.zeros_like(result["sigma_final"]),
        where=result["sigma_initial"] > 0.0,
    )
    final = initial * scale[:, None]
    gas_safe = np.maximum(gas_sigma_rphi, np.nanmax(gas_sigma_rphi) * 1.0e-15)
    before = initial / gas_safe
    after = final / gas_safe
    positive = np.concatenate([before[before > 0.0], after[after > 0.0]])
    norm = LogNorm(
        vmin=max(float(np.percentile(positive, 1.0)), 1.0e-12),
        vmax=float(np.percentile(positive, 99.5)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, values, title in (
        (axes[0], before, "initial"),
        (axes[1], after, "after 50 kyr"),
    ):
        image = ax.pcolormesh(
            radius,
            phi,
            values.T,
            shading="nearest",
            norm=norm,
            cmap="magma",
            rasterized=True,
        )
        ax.axvline(AXIS_SUPPORT.to("au").magnitude, color="cyan", linestyle=":")
        ax.set_xlabel("radius [au]")
        ax.set_ylabel("azimuth [rad]")
        ax.set_title(title)
    fig.colorbar(image, ax=axes[:2], label="settled dust / gas column")
    visible = before > max(float(np.nanmax(before)) * 1.0e-10, 1.0e-12)
    ratio = np.divide(after, before, out=np.full_like(after, np.nan), where=visible)
    image = axes[2].pcolormesh(
        radius,
        phi,
        np.log10(ratio).T,
        shading="nearest",
        cmap="RdBu_r",
        vmin=-2.0,
        vmax=2.0,
        rasterized=True,
    )
    axes[2].axvline(AXIS_SUPPORT.to("au").magnitude, color="black", linestyle=":")
    axes[2].set_xlabel("radius [au]")
    axes[2].set_ylabel("azimuth [rad]")
    axes[2].set_title("log10(after / initial)")
    fig.colorbar(image, ax=axes[2], label="dex")
    for ax in axes:
        ax.set_xlim(0.0, LOAD_R_MAX.to("au").magnitude)
    fig.suptitle(f"Largest representative bin ({result['label']}): retained azimuthal structure")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _transported_rphi(result: dict) -> np.ndarray:
    """Reapply a transported radial column to its initial azimuthal pattern."""
    scale = np.divide(
        result["sigma_final"],
        result["sigma_initial"],
        out=np.zeros_like(result["sigma_final"]),
        where=result["sigma_initial"] > 0.0,
    )
    return result["sigma_rphi_initial"] * scale[:, None]


def _cumulative_threshold_results(results: list[dict]) -> list[dict]:
    """Sum complete size bins lying above each requested grain-size edge."""
    cumulative: list[dict] = []
    for threshold_um in DUST_SIZE_THRESHOLDS_UM:
        selected = [
            result
            for result in results
            if result["size_min_um"] >= threshold_um * (1.0 - 1.0e-10)
        ]
        if not selected:
            raise ValueError(f"No dust bins lie above a_thresh={threshold_um:g} um")
        cumulative.append(
            {
                "threshold_um": threshold_um,
                "n_bins": len(selected),
                "size_min_um": min(result["size_min_um"] for result in selected),
                "size_max_um": max(result["size_max_um"] for result in selected),
                "sigma_initial": np.sum(
                    [result["sigma_initial"] for result in selected], axis=0
                ),
                "sigma_final": np.sum(
                    [result["sigma_final"] for result in selected], axis=0
                ),
                "sigma_rphi_initial": np.sum(
                    [result["sigma_rphi_initial"] for result in selected], axis=0
                ),
                "sigma_rphi_final": np.sum(
                    [_transported_rphi(result) for result in selected], axis=0
                ),
            }
        )
    return cumulative


def _plot_cumulative_threshold_profiles(
    output: Path,
    radius_au: np.ndarray,
    threshold_results: list[dict],
) -> None:
    """Plot azimuthally averaged dust columns above each grain-size edge."""
    peak = max(
        float(np.max(result[key]))
        for result in threshold_results
        for key in ("sigma_initial", "sigma_final")
    )
    floor = peak * 1.0e-12
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, result in zip(axes.flat, threshold_results):
        ax.plot(
            radius_au,
            np.maximum(result["sigma_initial"], floor),
            color="0.55",
            linestyle="--",
            label="initial",
        )
        ax.plot(
            radius_au,
            np.maximum(result["sigma_final"], floor),
            color="tab:blue",
            label="after 50 kyr",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(radius_au[0], radius_au[-1])
        ax.set_ylim(floor, peak * 1.15)
        ax.set_xlabel("radius [au]")
        ax.set_ylabel(r"$\Sigma_{\rm dust}(a>a_{\rm thresh})$ [g cm$^{-2}$]")
        ax.set_title(
            rf"$a_{{\rm thresh}}={result['threshold_um']:g}\,\mu$m "
            f"({result['n_bins']} bins)"
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "m05_9_dm_f140_c10: cumulative dust surface density by grain size"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_cumulative_threshold_rphi(
    output: Path,
    model,
    threshold_results: list[dict],
) -> None:
    """Plot final azimuthally resolved dust columns above each size edge."""
    radius = model.mesh.centers("r").to("au").magnitude
    phi = model.mesh.centers("phi").to("radian").magnitude
    peak = max(
        float(np.nanmax(result["sigma_rphi_final"]))
        for result in threshold_results
    )
    positive_min = min(
        float(np.nanmin(values[values > 0.0]))
        for result in threshold_results
        for values in (result["sigma_rphi_final"],)
        if np.any(values > 0.0)
    )
    norm = LogNorm(vmin=max(positive_min, peak * 1.0e-8), vmax=peak)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, result in zip(axes.flat, threshold_results):
        image = ax.pcolormesh(
            radius,
            phi,
            result["sigma_rphi_final"].T,
            shading="nearest",
            cmap="magma",
            norm=norm,
            rasterized=True,
        )
        ax.axvline(AXIS_SUPPORT.to("au").magnitude, color="cyan", linestyle=":")
        ax.set_xlim(0.0, LOAD_R_MAX.to("au").magnitude)
        ax.set_xlabel("radius [au]")
        ax.set_ylabel("azimuth [rad]")
        ax.set_title(rf"$a>{result['threshold_um']:g}\,\mu$m")
    fig.colorbar(
        image,
        ax=axes,
        label=r"$\Sigma_{\rm dust}(a>a_{\rm thresh})$ [g cm$^{-2}$]",
    )
    fig.suptitle(
        "m05_9_dm_f140_c10: azimuthal dust surface density after 50 kyr"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(*, overwrite: bool) -> None:
    outputs = {
        "mask": OUTPUT_DIR / "mask_edge_comparison.png",
        "gas_background": OUTPUT_DIR / "smoothed_gas_background.png",
        "profiles": OUTPUT_DIR / "radial_transport_profiles.png",
        "azimuth": OUTPUT_DIR / "largest_bin_dust_to_gas_rphi.png",
        "threshold_profiles": OUTPUT_DIR / "dust_surface_density_above_size_threshold_profiles.png",
        "threshold_azimuth": OUTPUT_DIR / "dust_surface_density_above_size_threshold_rphi.png",
        "data": OUTPUT_DIR / "radial_transport_profiles.npz",
        "summary": OUTPUT_DIR / "summary.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; rerun with --overwrite: " + ", ".join(map(str, existing)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading m05_9_dm_f140_c10", flush=True)
    model = _load_model()
    print(f"Loaded shape {model.mesh.shape}; computing production Joos weight", flush=True)
    model.set_mask_from_joos_disk(
        rho_disk_min=RHO_DISK_MIN,
        rho_core_min=RHO_CORE_MIN,
        r_max_for_axis=AXIS_SUPPORT,
        max_poloidal_mach=JOOS_MAX_POLOIDAL_MACH,
        rotational_support_factor=JOOS_ROTATIONAL_SUPPORT_FACTOR,
        weight_mode="soft_connected",
        soft_delta={"mach": 0.20, "rot": 0.20, "rho": 0.30},
        weight_m0=0.50,
        weight_floor=1.0e-4,
    )
    weight = model.gas["disk_weight"].data.to("dimensionless").magnitude
    radius = model.mesh.centers("r").to("au").magnitude
    hard_control = weight * (radius[:, None, None] <= AXIS_SUPPORT.to("au").magnitude)
    _plot_mask_comparison(outputs["mask"], model, weight, hard_control)

    gas_density = model.gas["density"].data.to("g/cm^3").magnitude
    gas_sigma_rphi = _surface_density_rphi(model, gas_density * weight)
    gas_sigma = np.mean(gas_sigma_rphi, axis=1)
    gas_floor = np.max(gas_sigma) * 1.0e-15
    gas_sigma_safe = np.maximum(gas_sigma, gas_floor)
    midplane = _midplane_profiles(model, weight)
    radial_centres = model.mesh.centers("r")
    radial_edges = model.mesh.edges("r")

    pressure_gradient_raw = smoothed_log_pressure_gradient(
        radial_centres,
        midplane["pressure"],
        smoothing_bins=1,
    )
    pressure_gradient_moving_average = smoothed_log_pressure_gradient(
        radial_centres,
        midplane["pressure"],
        smoothing_bins=PRESSURE_SMOOTHING_BINS,
    )
    mu = 2.35
    gas_background = build_smoothed_radial_gas_background(
        radial_centres,
        diskbridge.Quantity(gas_sigma_safe, "g/cm^2"),
        midplane["temperature"],
        midplane["pressure"],
        mean_molecular_weight=mu,
        smoothing_bins=PRESSURE_SMOOTHING_BINS,
        pressure_log_gradient_bounds=PRESSURE_LOG_GRADIENT_BOUNDS,
        surface_density_log_gradient_bounds=SURFACE_DENSITY_LOG_GRADIENT_BOUNDS,
    )
    _plot_gas_background(
        outputs["gas_background"],
        radius,
        gas_sigma_safe,
        midplane,
        gas_background,
        pressure_gradient_raw,
        pressure_gradient_moving_average,
    )
    sound_speed = np.sqrt(
        diskbridge.units("k_B") * gas_background.temperature
        / (mu * diskbridge.units("m_H"))
    ).to("cm/s")
    omega = np.sqrt(
        diskbridge.units("G") * STELLAR_MASS / radial_centres.to("cm") ** 3
    ).to("1/s")
    keplerian_velocity = (omega * radial_centres.to("cm")).to("cm/s")
    scale_height = (sound_speed / omega).to("cm")

    model.dust.add_component_from_mask(
        mask="disk_weight",
        mode="settling",
        amin=0.01 * diskbridge.units("um"),
        amax=1_000.0 * diskbridge.units("um"),
        nbin=DUST_NBIN,
        dust_to_gas_ratio=1.0e-2,
        alpha=SETTLING_ALPHA,
        delta=SETTLING_ALPHA,
        mean_molecular_weight=mu,
    )

    results: list[dict] = []
    for bin_index in range(DUST_NBIN):
        dust_bin = model.dust[f"bin_{bin_index}"]
        print(f"Computing settled density for {dust_bin.size.to('um')}", flush=True)
        density = dust_bin["density"].data.to("g/cm^3").magnitude
        sigma_rphi = _surface_density_rphi(model, density)
        sigma_initial = np.mean(sigma_rphi, axis=1)
        stokes = stokes_number(
            dust_bin.size,
            gas_background.midplane_density,
            gas_background.temperature,
            omega,
            dust_bin.density_material,
            mu,
        ).to("dimensionless").magnitude
        drift = pressure_drift_velocity(
            stokes,
            sound_speed,
            keplerian_velocity,
            gas_background.pressure_log_gradient,
        )
        diffusion = dust_diffusivity(
            SETTLING_ALPHA,
            sound_speed,
            scale_height,
            stokes,
        )
        sigma_final, diagnostics = evolve_radial_surface_density(
            radial_edges,
            diskbridge.Quantity(sigma_initial, "g/cm^2"),
            gas_background.surface_density,
            drift,
            diffusion,
            TRANSPORT_TIME,
        )
        label = f"{float(dust_bin.size.to('um').magnitude):.3g} µm"
        results.append(
            {
                "label": label,
                "size_um": float(dust_bin.size.to("um").magnitude),
                "size_min_um": float(dust_bin.size_min.to("um").magnitude),
                "size_max_um": float(dust_bin.size_max.to("um").magnitude),
                "stokes": stokes,
                "drift_cm_s": drift.to("cm/s").magnitude,
                "sigma_initial": sigma_initial,
                "sigma_final": sigma_final.to("g/cm^2").magnitude,
                "sigma_rphi_initial": sigma_rphi,
                "diagnostics": diagnostics,
            }
        )
        del density, sigma_rphi

    representative_results = [
        min(results, key=lambda result: abs(np.log(result["size_um"] / target_um)))
        for target_um in REPRESENTATIVE_SIZE_TARGETS_UM
    ]
    threshold_results = _cumulative_threshold_results(results)

    _plot_transport_profiles(
        outputs["profiles"],
        radius,
        gas_background.surface_density.to("g/cm^2").magnitude,
        pressure_gradient_raw,
        pressure_gradient_moving_average,
        gas_background.pressure_log_gradient,
        representative_results,
    )
    _plot_largest_bin_azimuth(
        outputs["azimuth"], model, gas_sigma_rphi, representative_results[-1]
    )
    _plot_cumulative_threshold_profiles(
        outputs["threshold_profiles"], radius, threshold_results
    )
    _plot_cumulative_threshold_rphi(
        outputs["threshold_azimuth"], model, threshold_results
    )

    np.savez_compressed(
        outputs["data"],
        radius_au=radius,
        radial_edges_au=radial_edges.to("au").magnitude,
        gas_sigma_g_cm2=gas_sigma,
        gas_background_sigma_g_cm2=gas_background.surface_density.to("g/cm^2").magnitude,
        gas_background_midplane_density_g_cm3=gas_background.midplane_density.to("g/cm^3").magnitude,
        gas_background_temperature_K=gas_background.temperature.to("K").magnitude,
        gas_background_pressure_dyn_cm2=gas_background.pressure.to("dyn/cm^2").magnitude,
        pressure_gradient_raw=pressure_gradient_raw,
        pressure_gradient_moving_average=pressure_gradient_moving_average,
        pressure_gradient_background=gas_background.pressure_log_gradient,
        **{
            f"bin_{index}_{key}": result[key]
            for index, result in enumerate(results)
            for key in (
                "size_um",
                "size_min_um",
                "size_max_um",
                "stokes",
                "drift_cm_s",
                "sigma_initial",
                "sigma_final",
            )
        },
        **{
            f"threshold_{result['threshold_um']:g}_um_{key}": result[key]
            for result in threshold_results
            for key in (
                "sigma_initial",
                "sigma_final",
                "sigma_rphi_initial",
                "sigma_rphi_final",
            )
        },
    )
    annulus_area_au2 = np.pi * np.diff(radial_edges.to("au").magnitude ** 2)
    annulus_area_cm2 = np.pi * np.diff(radial_edges.to("cm").magnitude ** 2)
    mean_column_weight = np.mean(np.max(weight, axis=1), axis=1)

    def summarize_bin(result: dict) -> dict:
        initial_annulus_mass = result["sigma_initial"] * annulus_area_au2
        final_annulus_mass = result["sigma_final"] * annulus_area_au2
        cumulative = np.cumsum(initial_annulus_mass) / np.sum(initial_annulus_mass)
        mass_bearing = cumulative <= 0.999
        mass_bearing[int(np.argmax(cumulative >= 0.999))] = True
        changes: dict[str, float | None] = {}
        meaningful_initial = float(np.max(result["sigma_initial"])) * 1.0e-12
        for target in (350.0, float(AXIS_SUPPORT.to("au").magnitude), 600.0):
            index = int(np.argmin(np.abs(radius - target)))
            initial = float(result["sigma_initial"][index])
            changes[f"{target:g}_au"] = (
                float(result["sigma_final"][index] / initial)
                if initial > meaningful_initial
                else None
            )
        return {
            "size_um": result["size_um"],
            "stokes_range_full": [
                float(np.min(result["stokes"])),
                float(np.max(result["stokes"])),
            ],
            "stokes_range_inner_99p9_mass": [
                float(np.min(result["stokes"][mass_bearing])),
                float(np.max(result["stokes"][mass_bearing])),
            ],
            "drift_cm_s_range_full": [
                float(np.min(result["drift_cm_s"])),
                float(np.max(result["drift_cm_s"])),
            ],
            "drift_cm_s_range_inner_99p9_mass": [
                float(np.min(result["drift_cm_s"][mass_bearing])),
                float(np.max(result["drift_cm_s"][mass_bearing])),
            ],
            "initial_mass_g": float(result["diagnostics"].initial_mass.to("g").magnitude),
            "final_mass_g": float(result["diagnostics"].final_mass.to("g").magnitude),
            "inner_mass_lost_g": float(result["diagnostics"].inner_mass_lost.to("g").magnitude),
            "outer_mass_lost_g": float(result["diagnostics"].outer_mass_lost.to("g").magnitude),
            "n_steps": result["diagnostics"].n_steps,
            "mass_weighted_mean_radius_au_initial": float(
                np.sum(radius * initial_annulus_mass) / np.sum(initial_annulus_mass)
            ),
            "mass_weighted_mean_radius_au_final": float(
                np.sum(radius * final_annulus_mass) / np.sum(final_annulus_mass)
            ),
            "surface_density_final_over_initial": changes,
        }

    summary = {
        "case": "m05_9_dm_f140_c10",
        "frame": FRAME,
        "loaded_r_max_au": float(LOAD_R_MAX.to("au").magnitude),
        "axis_support_au": float(AXIS_SUPPORT.to("au").magnitude),
        "physical_mask_r_max": None,
        "joos": {
            "max_poloidal_mach": JOOS_MAX_POLOIDAL_MACH,
            "rotational_support_factor": JOOS_ROTATIONAL_SUPPORT_FACTOR,
        },
        "transport_time_yr": float(TRANSPORT_TIME.to("yr").magnitude),
        "dust_distribution": {
            "amin_um": 0.01,
            "amax_um": 1_000.0,
            "n_bins": DUST_NBIN,
            "power_index": 3.5,
        },
        "gas_background": {
            "smoothing_bins": PRESSURE_SMOOTHING_BINS,
            "pressure_log_gradient_bounds": list(PRESSURE_LOG_GRADIENT_BOUNDS),
            "surface_density_log_gradient_bounds": list(SURFACE_DENSITY_LOG_GRADIENT_BOUNDS),
            "pressure_gradient_range": [
                float(np.min(gas_background.pressure_log_gradient)),
                float(np.max(gas_background.pressure_log_gradient)),
            ],
            "ideal_gas_mean_molecular_weight": mu,
        },
        "pressure_gradient_range_raw": [float(np.min(pressure_gradient_raw)), float(np.max(pressure_gradient_raw))],
        "pressure_gradient_range_moving_average": [
            float(np.min(pressure_gradient_moving_average)),
            float(np.max(pressure_gradient_moving_average)),
        ],
        "emergent_edge_au": {
            f"mean_column_max_weight_below_{threshold:g}": (
                float(radius[np.flatnonzero(mean_column_weight < threshold)[0]])
                if np.any(mean_column_weight < threshold)
                else None
            )
            for threshold in (0.9, 0.5, 0.1)
        },
        "bins": [summarize_bin(result) for result in results],
        "grain_size_thresholds": [
            {
                "a_thresh_um": result["threshold_um"],
                "n_complete_bins_included": result["n_bins"],
                "included_size_range_um": [
                    result["size_min_um"],
                    result["size_max_um"],
                ],
                "initial_mass_g": float(
                    np.sum(result["sigma_initial"] * annulus_area_cm2)
                ),
                "final_mass_g": float(
                    np.sum(result["sigma_final"] * annulus_area_cm2)
                ),
                "mass_weighted_mean_radius_au_initial": float(
                    np.sum(radius * result["sigma_initial"] * annulus_area_cm2)
                    / np.sum(result["sigma_initial"] * annulus_area_cm2)
                ),
                "mass_weighted_mean_radius_au_final": float(
                    np.sum(radius * result["sigma_final"] * annulus_area_cm2)
                    / np.sum(result["sigma_final"] * annulus_area_cm2)
                ),
            }
            for result in threshold_results
        ],
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
