"""Build the deterministic 2D outer-disk model for CO phase validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import json
import tomllib

import numpy as np

from diskbridge._constants import G_GRAV, M_H, SOLAR_MASS
from diskbridge._units import Quantity
from diskbridge.model.core import Model, SubModel
from diskbridge.model.dust import Dust
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.model.utils import field_data_as_order


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.toml"
AU_CM = float(Quantity("1 au").to("cm").magnitude)


@dataclass(frozen=True)
class RunSettings:
    """Resolved settings for one validation run size."""

    name: str
    grid: dict[str, Any]
    radmc3d: dict[str, Any]
    chemistry: dict[str, Any]
    disk: dict[str, Any]
    diagnostics: dict[str, Any]


def load_validation_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    """Load the TOML validation configuration."""

    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_run_settings(run_name: str, config: dict[str, Any] | None = None) -> RunSettings:
    """Return the default validation run settings."""

    cfg = load_validation_config() if config is None else config
    return RunSettings(
        name=run_name,
        grid=copy.deepcopy(cfg["grid"]),
        radmc3d={},
        chemistry=copy.deepcopy(cfg["chemistry_run"]),
        disk=copy.deepcopy(cfg["disk"]),
        diagnostics=copy.deepcopy(cfg["chemistry"]),
    )


def variant_configs(base_chemistry: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return the controlled GOW17 variant override dictionaries."""

    base = {
        "shielding_max_iter": 10,
        "temperature": {"initial": "gas_temperature"},
        "dust": {"sigma_d_CO_per_H_source": "constant"},
        "co_cooling": {"method": "legacy_scalar"},
    }
    if base_chemistry:
        base.update(
            {
                "shielding_max_iter": int(base_chemistry["shielding_max_iter"]),
                "astrochem_n_updates": int(base_chemistry["astrochem_n_updates"]),
                "astrochem_t_end_yr": float(base_chemistry["astrochem_t_end_yr"]),
            }
        )

    variants = {
        "gas_only_no_co_phase": {
            **copy.deepcopy(base),
            "enable_co_phase": False,
        },
        "co_ice_no_photodesorption": {
            **copy.deepcopy(base),
            "enable_co_phase": True,
            "co_phase": {
                "Y_CO": 0.0,
                "enable_cruv_pdes": False,
                "enable_crdes_CO": False,
            },
        },
        "co_ice_full_photodesorption": {
            **copy.deepcopy(base),
            "enable_co_phase": True,
            "co_phase": {
                "Y_CO": 1.0e-3,
                "enable_cruv_pdes": True,
                "enable_crdes_CO": False,
            },
        },
    }
    return variants


def _edges_from_centers(centers: np.ndarray, *, low: float, high: float) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size < 1:
        raise ValueError("centers must contain at least one value")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    if centers.size == 1:
        width = min(0.1 * max(abs(centers[0]), 1.0), high - low)
        edges[:] = [max(low, centers[0] - 0.5 * width), min(high, centers[0] + 0.5 * width)]
        return edges
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = max(low, centers[0] - (edges[1] - centers[0]))
    edges[-1] = min(high, centers[-1] + (centers[-1] - edges[-2]))
    return edges


def _theta_centers(ntheta_upper: int, opening: float) -> np.ndarray:
    if ntheta_upper < 2:
        raise ValueError("ntheta_upper must be at least 2")
    upper_edges = np.linspace(np.pi / 2.0 - opening, np.pi / 2.0, ntheta_upper + 1)
    upper = 0.5 * (upper_edges[:-1] + upper_edges[1:])
    theta = np.concatenate([upper, np.pi - upper])
    return np.sort(theta)


def _register(model: Model, name: str, data: np.ndarray, unit: str) -> None:
    model.gas_register(
        name,
        Field(
            quantity=name,
            data=Quantity(np.asarray(data, dtype=np.float64), unit),
            axis_order=model.mesh.axis_names(),  # type: ignore[union-attr]
        ),
    )


def _geometry(model: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = model.mesh.shape  # type: ignore[union-attr]
    radius = model.mesh.centers_f64("r", "cm")[:, None, None]  # type: ignore[union-attr]
    theta = model.mesh.centers_f64("theta", "rad")[None, :, None]  # type: ignore[union-attr]
    r = np.broadcast_to(radius, shape)
    R = np.broadcast_to(radius * np.sin(theta), shape)
    z = np.broadcast_to(radius * np.cos(theta), shape)
    return r, R, z


def _outer_taper(R_au: np.ndarray, disk: dict[str, Any]) -> np.ndarray:
    start = float(disk["outer_taper_start_au"])
    scale = float(disk["outer_taper_scale_au"])
    power = float(disk["outer_taper_power"])
    if scale <= 0.0:
        raise ValueError("outer_taper_scale_au must be positive")
    if power <= 0.0:
        raise ValueError("outer_taper_power must be positive")
    x = np.maximum((R_au - start) / scale, 0.0)
    return np.exp(-(x**power))


def build_outer_disk_model(settings: RunSettings) -> tuple[Model, dict[str, Any]]:
    """Build the deterministic axisymmetric outer-disk validation model."""

    grid = settings.grid
    disk = settings.disk
    nr = int(grid["nr"])
    ntheta_upper = int(grid["ntheta_upper"])
    nphi = int(grid["nphi"])
    r_min = float(grid["r_min_au"])
    r_max = float(grid["r_max_au"])

    r_edges_au = np.geomspace(r_min, r_max, nr + 1)
    theta = _theta_centers(ntheta_upper, float(grid["theta_opening_rad"]))
    theta_edges = _edges_from_centers(theta, low=1.0e-8, high=np.pi - 1.0e-8)

    model = Model()
    model.coord_system = "spherical"
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(r_edges_au * AU_CM, "cm")),
        theta=Axis(centers=Quantity(theta, "radian"), edges=Quantity(theta_edges, "radian")),
        phi=Axis(edges=Quantity(np.linspace(0.0, 2.0 * np.pi, nphi + 1), "radian")),
    )
    model.gas = SubModel(model)

    _, R, z = _geometry(model)
    R_au = np.maximum(R / AU_CM, r_min)
    abs_z = np.abs(z)

    h_over_r = float(disk["aspect_ratio_100au"]) * (R_au / 100.0) ** float(disk["flaring_power"])
    H = np.maximum(h_over_r * R, 1.0e-12)
    outer_taper = _outer_taper(R_au, disk)
    sigma = (
        float(disk["sigma_100_g_cm2"])
        * (R_au / 100.0) ** (-float(disk["surface_density_power"]))
        * np.exp(-R_au / float(disk["characteristic_radius_au"]))
        * outer_taper
    )
    rho = sigma / (np.sqrt(2.0 * np.pi) * H) * np.exp(-0.5 * (abs_z / H) ** 2)
    rho = np.maximum(rho, float(disk["rho_floor_g_cm3"]))

    t_mid = np.maximum(
        float(disk["midplane_temperature_1au_K"]) * R_au ** -0.5,
        float(disk["temperature_floor_K"]),
    )
    surface_weight = 1.0 - np.exp(-0.5 * (abs_z / np.maximum(2.0 * H, 1.0e-12)) ** 2)
    t_surface = float(disk["surface_temperature_100au_K"]) * (R_au / 100.0) ** -0.30
    tdust = np.maximum(t_mid + surface_weight * t_surface, float(disk["temperature_floor_K"]))

    nH = rho / (1.4 * float(M_H))
    chi_initial = np.maximum(1.0e-30, np.exp(-np.minimum(nH / 1.0e7, 80.0)))
    microturbulence = np.full(model.mesh.shape, 0.2, dtype=np.float64)  # type: ignore[union-attr]
    stellar_mass = float(disk["stellar_mass_msun"]) * float(SOLAR_MASS)
    vphi = np.broadcast_to(
        np.sqrt(G_GRAV * stellar_mass / np.maximum(R, 1.0)),
        model.mesh.shape,  # type: ignore[union-attr]
    ).copy()
    vr = np.zeros(model.mesh.shape, dtype=np.float64)  # type: ignore[union-attr]
    vtheta = np.zeros(model.mesh.shape, dtype=np.float64)  # type: ignore[union-attr]

    _register(model, "density", rho, "g/cm^3")
    _register(model, "number_density_H", nH, "cm^-3")
    _register(model, "dust_temperature", tdust, "K")
    _register(model, "gas_temperature", tdust, "K")
    _register(model, "temperature", tdust, "K")
    _register(model, "chi", chi_initial, "dimensionless")
    _register(model, "microturbulence", microturbulence, "km/s")
    _register(model, "vr", vr, "cm/s")
    _register(model, "vtheta", vtheta, "cm/s")
    _register(model, "vphi", vphi, "cm/s")

    model.dust = Dust(model)
    model.dust.set_distribution(
        amin=Quantity(0.05, "micron"),
        amax=Quantity(0.25, "micron"),
        nbin=1,
        power_index=3.5,
        grain_density=Quantity(2.7, "g/cm^3"),
        dust_to_gas_ratio=float(disk["dust_to_gas_ratio"]),
        mode="proportional",
    )

    meta = {
        "grid": "deterministic mirrored spherical outer disk",
        "run": settings.name,
        "nr": int(model.mesh.ncell("r")),
        "ntheta": int(model.mesh.ncell("theta")),
        "nphi": int(model.mesh.ncell("phi")),
        "shape": list(model.mesh.shape),
        "r_min_au": r_min,
        "r_max_au": r_max,
        "stellar_mass_msun": float(disk["stellar_mass_msun"]),
        "velocity_field": "Keplerian azimuthal rotation with vr=vtheta=0",
        "outer_taper_start_au": float(disk["outer_taper_start_au"]),
        "outer_taper_scale_au": float(disk["outer_taper_scale_au"]),
        "outer_taper_power": float(disk["outer_taper_power"]),
        "outer_taper_at_r_max": float(
            np.exp(
                -(
                    max(r_max - float(disk["outer_taper_start_au"]), 0.0)
                    / float(disk["outer_taper_scale_au"])
                )
                ** float(disk["outer_taper_power"])
            )
        ),
        "nH_min_cm3": float(np.nanmin(nH)),
        "nH_max_cm3": float(np.nanmax(nH)),
        "Tdust_min_K": float(np.nanmin(tdust)),
        "Tdust_max_K": float(np.nanmax(tdust)),
        "dust": "single proportional 0.05-0.25 micron bin",
    }
    return model, meta


def field_array(model: Model, name: str, unit: str = "dimensionless") -> np.ndarray:
    """Return a gas field as a NumPy array in canonical mesh order."""

    if model.gas is None or name not in model.gas:
        raise KeyError(f"Missing gas field {name!r}")
    q = field_data_as_order(model.gas[name], model.mesh.axis_names()).to(unit)  # type: ignore[union-attr]
    return np.asarray(q.magnitude, dtype=np.float64)


def geometry_arrays(model: Model) -> dict[str, np.ndarray]:
    """Return cylindrical geometry arrays in au for diagnostics."""

    _, R, z = _geometry(model)
    return {"R_au": R / AU_CM, "z_au": z / AU_CM, "abs_z_over_R": np.abs(z) / np.maximum(R, 1.0)}


def region_masks(model: Model, settings: RunSettings, chi_field: str = "chi") -> dict[str, np.ndarray]:
    """Return objective diagnostic region masks for the validation grid."""

    geom = geometry_arrays(model)
    nH = field_array(model, "number_density_H", "cm^-3")
    tdust = field_array(model, "dust_temperature", "K")
    try:
        chi = field_array(model, chi_field, "dimensionless")
    except KeyError:
        chi = field_array(model, "chi", "dimensionless")

    dense_min = float(settings.diagnostics["dense_midplane_nH_min_cm3"])
    low_max = float(settings.diagnostics["low_density_nH_max_cm3"])
    outer_min = float(settings.diagnostics["outer_disk_min_au"])
    midplane = geom["abs_z_over_R"] < 0.12
    surface = geom["abs_z_over_R"] > 0.35
    outer = geom["R_au"] >= outer_min

    return {
        "dense_midplane": midplane & (nH >= dense_min) & (tdust <= 40.0),
        "molecular_layer": (geom["abs_z_over_R"] >= 0.12) & (geom["abs_z_over_R"] <= 0.35) & (nH >= 1.0e5),
        "outer_disk_edge": outer & (nH >= low_max),
        "uv_irradiated_surface": surface & (chi >= 1.0e-4),
        "low_density_outer_material": outer & (nH <= low_max),
    }


def summarize_regions(model: Model, settings: RunSettings, fields: dict[str, str]) -> dict[str, Any]:
    """Summarize selected fields over named diagnostic regions."""

    masks = region_masks(model, settings, chi_field=fields.get("chi", "chi"))
    out: dict[str, Any] = {}
    for region, mask in masks.items():
        region_summary: dict[str, Any] = {"cells": int(np.count_nonzero(mask))}
        for label, field_name in fields.items():
            if model.gas is None or field_name not in model.gas:
                region_summary[label] = None
                continue
            unit = "K" if "temperature" in field_name else "cm^-3" if field_name == "number_density_H" else "dimensionless"
            if "F_CO_pdes" in field_name:
                unit = "1/(cm^2 s)"
            arr = field_array(model, field_name, unit)
            vals = arr[mask]
            if vals.size:
                region_summary[label] = {
                    "median": float(np.nanmedian(vals)),
                    "min": float(np.nanmin(vals)),
                    "max": float(np.nanmax(vals)),
                }
            else:
                region_summary[label] = None
        out[region] = region_summary
    return out


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
