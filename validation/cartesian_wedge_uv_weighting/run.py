"""Run the Cartesian wedge UV-weighting validation."""

# db-keywords: shielding, healpix-columns, uv-products, validation, config, units, radmc3d, model, mesh, field
# db-role: validation
# db-scope: validation
# db-purpose: Validation geometry, RADMC-3D transport, and GOW17 UV-weighting diagnostics.

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import diskbridge
from diskbridge._params import DEFAULT_PARAMS_FILE
from diskbridge._units import Quantity
from diskbridge.chemistry.api import run_chemistry
from diskbridge.model.core import Model, SubModel
from diskbridge.model.dust import Dust
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel
from diskbridge.serialization import jsonable
from diskbridge.visualization.scales import log10_display_limits, log10_display_values


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.toml"
M_H_G = 1.6735575e-24


@dataclass(frozen=True)
class GridConfig:
    nx: int
    ny: int
    nz: int
    x_min_au: float
    x_max_au: float
    y_min_au: float
    y_max_au: float
    z_min_au: float
    z_max_au: float


@dataclass(frozen=True)
class GeometryConfig:
    circle_center_x_au: float
    circle_center_y_au: float
    circle_center_z_au: float
    circle_radius_au: float
    star_x_au: float
    star_y_au: float
    star_z_au: float
    wedge_half_angle_deg: float
    wedge_inner_radius_au: float
    wedge_outer_radius_au: float


@dataclass(frozen=True)
class DensityConfig:
    ambient_nH_cm3: float
    cloud_nH_cm3: float
    wedge_nH_cm3: float
    mean_particle_mass_per_H: float


@dataclass(frozen=True)
class DustConfig:
    dust_to_gas_ratio: float
    dust_nbins: int
    dust_amin_um: float
    dust_amax_um: float
    dust_pindex: float


@dataclass(frozen=True)
class RadiationConfig:
    external_chi: float


@dataclass(frozen=True)
class StarConfig:
    rstar_rsun: float
    teff_K: float
    mstar_msun: float
    mdot_msun_per_yr: float


@dataclass(frozen=True)
class RadmcConfig:
    nphot_thermal: int
    nphot_mono: int
    nphot_scat: int
    scat_mode: int
    nbcores: int
    lambda_min_um: float
    uv_min_nm: float
    uv_max_nm: float
    uv_n_wavelengths: int


@dataclass(frozen=True)
class ChemistryConfig:
    nside: int
    shielding_max_iter: int
    b_kms: float
    temperature_mode: str


@dataclass(frozen=True)
class ValidationConfig:
    grid: GridConfig
    geometry: GeometryConfig
    density: DensityConfig
    dust: DustConfig
    radiation: RadiationConfig
    star: StarConfig
    radmc3d: RadmcConfig
    chemistry: ChemistryConfig
    raw: dict


@contextmanager
def _using_diskbridge_params(params):
    previous = diskbridge.params
    try:
        diskbridge.params = params
        yield params
    finally:
        diskbridge.params = previous


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config.toml missing [{name}] section")
    return value


def _load_config(path: Path) -> ValidationConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return ValidationConfig(
        grid=GridConfig(**_section(raw, "grid")),
        geometry=GeometryConfig(**_section(raw, "geometry")),
        density=DensityConfig(**_section(raw, "density")),
        dust=DustConfig(**_section(raw, "dust")),
        radiation=RadiationConfig(**_section(raw, "radiation")),
        star=StarConfig(**_section(raw, "star")),
        radmc3d=RadmcConfig(**_section(raw, "radmc3d")),
        chemistry=ChemistryConfig(**_section(raw, "chemistry")),
        raw=raw,
    )


def _edges_au(lo: float, hi: float, n: int) -> np.ndarray:
    if int(n) <= 0:
        raise ValueError("grid cell counts must be positive")
    if not float(hi) > float(lo):
        raise ValueError("grid upper edge must exceed lower edge")
    return np.linspace(float(lo), float(hi), int(n) + 1, dtype=np.float64)


def build_mesh(cfg: GridConfig) -> Mesh:
    """Return the Cartesian validation mesh."""

    return Mesh.cartesian(
        x=Axis(edges=Quantity(_edges_au(cfg.x_min_au, cfg.x_max_au, cfg.nx), "au")),
        y=Axis(edges=Quantity(_edges_au(cfg.y_min_au, cfg.y_max_au, cfg.ny), "au")),
        z=Axis(edges=Quantity(_edges_au(cfg.z_min_au, cfg.z_max_au, cfg.nz), "au")),
    )


def _centers_from_edges(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(edges[:-1], dtype=np.float64) + np.asarray(edges[1:], dtype=np.float64))


def _axis_unit_vector(geom: GeometryConfig) -> tuple[float, float, float]:
    dx = float(geom.star_x_au) - float(geom.circle_center_x_au)
    dy = float(geom.star_y_au) - float(geom.circle_center_y_au)
    dz = float(geom.star_z_au) - float(geom.circle_center_z_au)
    norm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    if norm <= 0.0:
        raise ValueError("circle center must not coincide with the source")
    return dx / norm, dy / norm, dz / norm


def build_density_structure(cfg: ValidationConfig) -> dict[str, np.ndarray]:
    """Build density and mask arrays for the validation geometry."""

    mesh = build_mesh(cfg.grid)
    x_au = _centers_from_edges(mesh.edges_f64("x", "au"))
    y_au = _centers_from_edges(mesh.edges_f64("y", "au"))
    z_au = _centers_from_edges(mesh.edges_f64("z", "au"))
    x3 = x_au[:, None, None]
    y3 = y_au[None, :, None]
    z3 = z_au[None, None, :]

    geom = cfg.geometry
    dens = cfg.density
    dx_cloud = x3 - float(geom.circle_center_x_au)
    dy_cloud = y3 - float(geom.circle_center_y_au)
    dz_cloud = z3 - float(geom.circle_center_z_au)
    sphere = (
        dx_cloud * dx_cloud
        + dy_cloud * dy_cloud
        + dz_cloud * dz_cloud
    ) <= float(geom.circle_radius_au) ** 2

    axis_x, axis_y, axis_z = _axis_unit_vector(geom)
    axis_distance2 = dx_cloud * dx_cloud + dy_cloud * dy_cloud + dz_cloud * dz_cloud
    projection = dx_cloud * axis_x + dy_cloud * axis_y + dz_cloud * axis_z
    perpendicular2 = np.maximum(axis_distance2 - projection * projection, 0.0)
    angle_from_axis = np.arctan2(np.sqrt(perpendicular2), projection)
    cone = (
        sphere
        & (projection >= float(geom.wedge_inner_radius_au))
        & (projection <= float(geom.wedge_outer_radius_au))
        & (angle_from_axis <= np.deg2rad(float(geom.wedge_half_angle_deg)))
    )

    shape = (int(cfg.grid.nx), int(cfg.grid.ny), int(cfg.grid.nz))
    if tuple(sphere.shape) != shape:
        raise RuntimeError(f"internal geometry shape mismatch: {sphere.shape} vs {shape}")
    nH = np.full(shape, float(dens.ambient_nH_cm3), dtype=np.float64)
    nH[sphere] = float(dens.cloud_nH_cm3)
    nH[cone] = float(dens.wedge_nH_cm3)
    dense_cloud = sphere & ~cone
    ambient = ~sphere
    rho = nH * float(dens.mean_particle_mass_per_H) * M_H_G

    return {
        "x_au": x_au,
        "y_au": y_au,
        "z_au": z_au,
        "nH_cm3": nH,
        "rho_g_cm3": rho,
        "circle_mask": sphere,
        "dense_cloud_mask": dense_cloud,
        "wedge_mask": cone,
        "ambient_mask": ambient,
    }


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "p10": float(np.nanpercentile(arr, 10.0)),
        "p50": float(np.nanpercentile(arr, 50.0)),
        "p90": float(np.nanpercentile(arr, 90.0)),
    }


def _fraction(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    return float(np.count_nonzero(arr) / arr.size)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(
        np.asarray(num, dtype=np.float64),
        np.asarray(den, dtype=np.float64),
        out=np.full_like(np.asarray(num, dtype=np.float64), np.nan),
        where=np.asarray(den, dtype=np.float64) > 0.0,
    )


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - float(target))))


def _rotate2(vx: float, vy: float, angle_rad: float) -> tuple[float, float]:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return c * vx - s * vy, s * vx + c * vy


def _draw_xy_overlay(ax, cfg: ValidationConfig) -> None:
    from matplotlib.patches import Circle

    geom = cfg.geometry
    circle = Circle(
        (float(geom.circle_center_x_au), float(geom.circle_center_y_au)),
        float(geom.circle_radius_au),
        fill=False,
        edgecolor="cyan",
        linewidth=1.2,
    )
    ax.add_patch(circle)

    axis_x, axis_y, _axis_z = _axis_unit_vector(geom)
    half = np.deg2rad(float(geom.wedge_half_angle_deg))
    for sign in (-1.0, 1.0):
        rx, ry = _rotate2(axis_x, axis_y, sign * half)
        x0 = float(geom.circle_center_x_au)
        y0 = float(geom.circle_center_y_au)
        x1 = x0 + float(geom.wedge_outer_radius_au) * rx
        y1 = y0 + float(geom.wedge_outer_radius_au) * ry
        ax.plot([x0, x1], [y0, y1], color="white", linestyle="--", linewidth=1.0)

    ax.scatter(
        [float(geom.circle_center_x_au)],
        [float(geom.circle_center_y_au)],
        marker="o",
        s=28,
        color="cyan",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )
    ax.scatter(
        [float(geom.star_x_au)],
        [float(geom.star_y_au)],
        marker="*",
        s=85,
        color="yellow",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )


def _draw_yz_overlay(ax, cfg: ValidationConfig) -> None:
    from matplotlib.patches import Circle

    geom = cfg.geometry
    circle = Circle(
        (float(geom.circle_center_y_au), float(geom.circle_center_z_au)),
        float(geom.circle_radius_au),
        fill=False,
        edgecolor="cyan",
        linewidth=1.2,
    )
    ax.add_patch(circle)
    half = np.deg2rad(float(geom.wedge_half_angle_deg))
    for sign in (-1.0, 1.0):
        y0 = float(geom.circle_center_y_au)
        z0 = float(geom.circle_center_z_au)
        y1 = y0 - float(geom.wedge_outer_radius_au) * np.cos(half)
        z1 = z0 + sign * float(geom.wedge_outer_radius_au) * np.sin(half)
        ax.plot([y0, y1], [z0, z1], color="white", linestyle="--", linewidth=1.0)
    ax.scatter(
        [float(geom.circle_center_y_au)],
        [float(geom.circle_center_z_au)],
        marker="o",
        s=28,
        color="cyan",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )
    ax.scatter(
        [float(geom.star_y_au)],
        [float(geom.star_z_au)],
        marker="*",
        s=85,
        color="yellow",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_density_plot(out_dir: Path, cfg: ValidationConfig, fields: dict[str, np.ndarray]) -> dict[str, Path]:
    """Write density diagnostics for the 3-D conical cavity."""

    plt = _setup_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    x_au = fields["x_au"]
    y_au = fields["y_au"]
    z_au = fields["z_au"]
    nH = fields["nH_cm3"]
    geom = cfg.geometry
    iz = _nearest_index(z_au, float(geom.circle_center_z_au))
    ix = _nearest_index(x_au, float(geom.circle_center_x_au))
    mid_y = 0.5 * (float(geom.circle_center_y_au) + float(geom.star_y_au))
    iy_mid = _nearest_index(y_au, mid_y)

    log_nH = np.log10(np.maximum(nH, 1.0e-30))
    vmin = float(np.nanmin(log_nH))
    vmax = float(np.nanmax(log_nH))

    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)
    axs[0].imshow(
        log_nH[:, :, iz].T,
        origin="lower",
        extent=[cfg.grid.x_min_au, cfg.grid.x_max_au, cfg.grid.y_min_au, cfg.grid.y_max_au],
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    _draw_xy_overlay(axs[0], cfg)
    axs[0].set_title(f"x-y slice at z={z_au[iz]:.0f} au")
    axs[0].set_xlabel("x [au]")
    axs[0].set_ylabel("y [au]")

    axs[1].imshow(
        log_nH[ix, :, :].T,
        origin="lower",
        extent=[cfg.grid.y_min_au, cfg.grid.y_max_au, cfg.grid.z_min_au, cfg.grid.z_max_au],
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    _draw_yz_overlay(axs[1], cfg)
    axs[1].set_title(f"y-z slice at x={x_au[ix]:.0f} au")
    axs[1].set_xlabel("y [au]")
    axs[1].set_ylabel("z [au]")

    im = axs[2].imshow(
        log_nH[:, iy_mid, :].T,
        origin="lower",
        extent=[cfg.grid.x_min_au, cfg.grid.x_max_au, cfg.grid.z_min_au, cfg.grid.z_max_au],
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    axs[2].set_title(f"x-z slice at y={y_au[iy_mid]:.0f} au")
    axs[2].set_xlabel("x [au]")
    axs[2].set_ylabel("z [au]")

    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("log10(nH [cm^-3])")

    out_path = plot_dir / "density_structure.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7.0, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    r = float(geom.circle_radius_au)
    cx = float(geom.circle_center_x_au)
    cy = float(geom.circle_center_y_au)
    cz = float(geom.circle_center_z_au)
    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    phi = np.linspace(0.0, np.pi, 24)
    xs = cx + r * np.outer(np.cos(theta), np.sin(phi))
    ys = cy + r * np.outer(np.sin(theta), np.sin(phi))
    zs = cz + r * np.outer(np.ones_like(theta), np.cos(phi))
    ax.plot_wireframe(xs, ys, zs, color="cyan", linewidth=0.35, alpha=0.35)

    half = np.deg2rad(float(geom.wedge_half_angle_deg))
    length = float(geom.wedge_outer_radius_au)
    cone_r = length * np.tan(half)
    t = np.linspace(0.0, 1.0, 36)
    ang = np.linspace(0.0, 2.0 * np.pi, 64)
    tt, aa = np.meshgrid(t, ang, indexing="ij")
    cone_x = cx + (tt * cone_r) * np.cos(aa)
    cone_y = cy - tt * length
    cone_z = cz + (tt * cone_r) * np.sin(aa)
    ax.plot_surface(cone_x, cone_y, cone_z, color="purple", alpha=0.45, linewidth=0)

    ax.scatter([cx], [cy], [cz], color="cyan", s=35, edgecolors="black", label="cone apex")
    ax.scatter(
        [float(geom.star_x_au)],
        [float(geom.star_y_au)],
        [float(geom.star_z_au)],
        color="yellow",
        marker="*",
        s=110,
        edgecolors="black",
        label="UV source",
    )
    ax.set_xlabel("x [au]")
    ax.set_ylabel("y [au]")
    ax.set_zlabel("z [au]")
    ax.set_title("3D cloud and conical cavity geometry")
    ax.set_box_aspect((1.0, 1.0, 0.8))
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc="upper left", fontsize=8)
    geometry_path = plot_dir / "cone_geometry_3d.png"
    fig.savefig(geometry_path, dpi=180)
    plt.close(fig)
    return {"density_plot": out_path, "cone_geometry_3d": geometry_path}


def _base_summary(cfg: ValidationConfig, fields: dict[str, np.ndarray]) -> dict:
    geom = cfg.geometry
    bottom_offset_au = float(geom.circle_center_y_au) - float(geom.circle_radius_au) - float(geom.star_y_au)
    source_axis_offset_au = float(
        np.sqrt(
            (float(geom.star_x_au) - float(geom.circle_center_x_au)) ** 2
            + (float(geom.star_z_au) - float(geom.circle_center_z_au)) ** 2
        )
    )
    return {
        "validation": "cartesian_wedge_uv_weighting",
        "config": {
            "grid": asdict(cfg.grid),
            "geometry": asdict(cfg.geometry),
            "density": asdict(cfg.density),
            "dust": asdict(cfg.dust),
            "radiation": asdict(cfg.radiation),
            "star": asdict(cfg.star),
            "radmc3d": asdict(cfg.radmc3d),
            "chemistry": asdict(cfg.chemistry),
        },
        "shape": [int(v) for v in fields["nH_cm3"].shape],
        "nH_stats_cm3": _summary_stats(fields["nH_cm3"]),
        "rho_stats_g_cm3": _summary_stats(fields["rho_g_cm3"]),
        "volume_fractions": {
            "sphere": _fraction(fields["circle_mask"]),
            "dense_cloud": _fraction(fields["dense_cloud_mask"]),
            "cone_cavity": _fraction(fields["wedge_mask"]),
            "ambient": _fraction(fields["ambient_mask"]),
        },
        "geometry_checks": {
            "source_to_circle_bottom_offset_au": bottom_offset_au,
            "source_axis_offset_xz_au": source_axis_offset_au,
            "cone_apex_at_cloud_center": True,
            "cone_axis_points_from_cloud_center_to_source": True,
        },
    }


def write_density_outputs(out_dir: Path, cfg: ValidationConfig, fields: dict[str, np.ndarray]) -> dict:
    """Write density-preview arrays, plot, and summary."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = write_density_plot(out_dir, cfg, fields)
    np.savez_compressed(
        out_dir / "fields.npz",
        x_au=fields["x_au"],
        y_au=fields["y_au"],
        z_au=fields["z_au"],
        nH_cm3=fields["nH_cm3"],
        rho_g_cm3=fields["rho_g_cm3"],
        circle_mask=fields["circle_mask"],
        dense_cloud_mask=fields["dense_cloud_mask"],
        wedge_mask=fields["wedge_mask"],
        ambient_mask=fields["ambient_mask"],
    )

    summary = _base_summary(cfg, fields)
    summary.update(
        {
            "stage": "density_preview",
            "ran_radmc3d": False,
            "ran_chemistry": False,
            "outputs": {
                "fields_npz": str(out_dir / "fields.npz"),
                "density_plot": str(plot_paths["density_plot"]),
                "cone_geometry_3d": str(plot_paths["cone_geometry_3d"]),
            },
        }
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_model(cfg: ValidationConfig, fields: dict[str, np.ndarray]) -> Model:
    model = Model()
    model.coord_system = "cartesian"
    model.mesh = build_mesh(cfg.grid)
    model.gas = SubModel(model)
    model.gas_register(
        "density",
        Field(
            quantity="density",
            data=Quantity(np.asarray(fields["rho_g_cm3"], dtype=np.float64), "g/cm^3"),
            axis_order=model.mesh.axis_names(),
        ),
    )
    model.dust = Dust(model)
    model.dust.set_distribution(
        amin=Quantity(float(cfg.dust.dust_amin_um), "micrometer"),
        amax=Quantity(float(cfg.dust.dust_amax_um), "micrometer"),
        nbin=int(cfg.dust.dust_nbins),
        power_index=float(cfg.dust.dust_pindex),
        dust_to_gas_ratio=float(cfg.dust.dust_to_gas_ratio),
        mode="proportional",
    )
    model.validate_canonical_axis_orders(include_dust=False)
    return model


def _write_radmc3d_params_txt(*, model_dir: Path, cfg: ValidationConfig) -> Path:
    params_path = Path(model_dir) / "params.txt"
    text = Path(DEFAULT_PARAMS_FILE).read_text(encoding="utf-8").splitlines(True)
    updates = {
        "nbcores": str(int(cfg.radmc3d.nbcores)),
        "nphot_thermal": str(int(cfg.radmc3d.nphot_thermal)),
        "nphot_mono": str(int(cfg.radmc3d.nphot_mono)),
        "nphot_scat": str(int(cfg.radmc3d.nphot_scat)),
        "scat_mode": str(int(cfg.radmc3d.scat_mode)),
        "lambda_min": str(float(cfg.radmc3d.lambda_min_um)),
        "uv_min": str(float(cfg.radmc3d.uv_min_nm)),
        "uv_max": str(float(cfg.radmc3d.uv_max_nm)),
        "uv_n_wavelengths": str(int(cfg.radmc3d.uv_n_wavelengths)),
        "external_uv": "T",
        "external_uv_chi": str(float(cfg.radiation.external_chi)),
        "rstar": str(float(cfg.star.rstar_rsun)),
        "teff": str(float(cfg.star.teff_K)),
        "mstar": str(float(cfg.star.mstar_msun)),
        "mdot": str(float(cfg.star.mdot_msun_per_yr)),
        "nside": str(int(cfg.chemistry.nside)),
        "amin": str(float(cfg.dust.dust_amin_um)),
        "amax": str(float(cfg.dust.dust_amax_um)),
        "pindex": str(float(cfg.dust.dust_pindex)),
        "nbins": str(int(cfg.dust.dust_nbins)),
        "dust_to_gas_ratio": str(float(cfg.dust.dust_to_gas_ratio)),
        "microturbulence": str(float(cfg.chemistry.b_kms)),
    }

    out_lines: list[str] = []
    seen: set[str] = set()
    for line in text:
        if "=" not in line or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            out_lines.append(f"{key} = {updates[key]}\n")
            seen.add(key)
        else:
            out_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            out_lines.append(f"{key} = {value}\n")

    params_path.write_text("".join(out_lines), encoding="utf-8")
    return params_path


def _run_radmc3d_transport(*, cfg: ValidationConfig, rad: RadModel, model_dir: Path) -> object:
    model_dir = Path(model_dir)
    inputs_dir = model_dir / "radmc3d_inputs"
    outputs_dir = model_dir / "radmc3d_outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    params_path = _write_radmc3d_params_txt(model_dir=model_dir, cfg=cfg)
    new_params = diskbridge.read_params(params_path)
    previous_rad_params = rad.params
    previous_writer_params = rad.writer.params

    try:
        with _using_diskbridge_params(new_params):
            rad.params = new_params
            rad.writer.params = new_params
            rad.writer.write_amr_grid(model_dir)
            rad.writer.write_wavelength_grid(model_dir)
            rad.writer.write_stars(model_dir)
            rad.writer.write_radmc3d_inp(
                model_dir,
                scattering_mode_max=int(new_params.scat_mode),
                nphot=int(new_params.nphot_thermal),
                nphot_mono=int(new_params.nphot_mono),
                nphot_scat=int(new_params.nphot_scat),
                setthreads=int(new_params.nbcores),
            )
            rad.writer.write_dust_density(model_dir, binary=True)
            rad.writer.write_dustopac(model_dir, scattering_mode=int(new_params.scat_mode))
            rad.writer.compute_and_write_dust_opacities(
                model_dir,
                scattering_mode=int(new_params.scat_mode),
            )
            rad.writer.write_external_source(model_dir, chi=float(cfg.radiation.external_chi))
            rad.compute_temperature(
                nphot=int(new_params.nphot_thermal),
                output_dir=outputs_dir,
                force=True,
            )
            rad.compute_mcmono(
                nphot=int(new_params.nphot_mono),
                output_dir=outputs_dir,
                force=True,
                setthreads=int(new_params.nbcores),
                compute_uv_products=True,
            )
    finally:
        rad.params = previous_rad_params
        rad.writer.params = previous_writer_params

    return new_params


def _chemistry_config(cfg: ValidationConfig, variant: str) -> dict:
    if variant == "no_shielding":
        return {
            "skip_shielding": True,
            "shielding_max_iter": 1,
            "shielding_ray_average": "uniform",
            "temperature": {"mode": str(cfg.chemistry.temperature_mode)},
        }
    if variant == "uniform_healpix_average":
        return {
            "skip_shielding": False,
            "shielding_max_iter": int(cfg.chemistry.shielding_max_iter),
            "shielding_ray_average": "uniform",
            "temperature": {"mode": str(cfg.chemistry.temperature_mode)},
        }
    if variant == "weighted_healpix_average":
        return {
            "skip_shielding": False,
            "shielding_max_iter": int(cfg.chemistry.shielding_max_iter),
            "shielding_ray_average": "weighted",
            "temperature": {"mode": str(cfg.chemistry.temperature_mode)},
        }
    raise ValueError(f"unknown chemistry variant: {variant}")


def _reset_chemistry_state(rad: RadModel) -> None:
    for name in (
        "gow17_y",
        "gow17_convergence",
        "Tgas_gow17",
        "theta_co",
        "chi_eff",
        "k_diss_co",
        "tau_diss_co",
        "nco_gas",
        "nco_ice",
        "nH2",
        "nH_atom",
        "nCplus",
        "nC",
        "ne",
    ):
        if hasattr(rad, name):
            setattr(rad, name, None)


def _qarray(quantity, unit: str) -> np.ndarray:
    return np.asarray(quantity.to(unit).magnitude, dtype=np.float64)


def _variant_arrays(result) -> dict[str, np.ndarray]:
    return {
        "Xco": _qarray(result.abundances["co"], "dimensionless"),
        "nco_cm3": _qarray(result.number_densities["co"], "cm^-3"),
        "theta_co": _qarray(result.fields["theta_co"], "dimensionless"),
        "chi_eff": _qarray(result.fields["chi_eff"], "dimensionless"),
    }


def _write_variant_outputs(variant_dir: Path, variant: str, result, arrays: dict[str, np.ndarray]) -> dict:
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(variant_dir / "chemistry_fields.npz", **arrays)
    meta = jsonable(result.meta)
    (variant_dir / "gow17_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "variant": variant,
        "arrays_npz": str(variant_dir / "chemistry_fields.npz"),
        "Xco_stats": _summary_stats(arrays["Xco"]),
        "nco_cm3_stats": _summary_stats(arrays["nco_cm3"]),
        "theta_co_stats": _summary_stats(arrays["theta_co"]),
        "chi_eff_stats": _summary_stats(arrays["chi_eff"]),
        "meta": {
            "n_fail": int(result.meta.get("n_fail", -1)),
            "max_status": int(result.meta.get("max_status", 0)),
            "temperature_mode": result.meta.get("temperature_mode"),
            "shielding_max_iter": result.meta.get("shielding_max_iter"),
            "gow17_diagnostics": jsonable(result.meta.get("gow17_diagnostics", {})),
        },
    }
    (variant_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _central_slice(arr: np.ndarray, fields: dict[str, np.ndarray], cfg: ValidationConfig) -> np.ndarray:
    iz = _nearest_index(fields["z_au"], float(cfg.geometry.circle_center_z_au))
    return np.asarray(arr[:, :, iz], dtype=np.float64)


def _plot_field_triplet(
    *,
    out_path: Path,
    title: str,
    field_name: str,
    variants: dict[str, dict[str, np.ndarray]],
    fields: dict[str, np.ndarray],
    cfg: ValidationConfig,
    floor: float,
    cmap: str,
) -> None:
    plt = _setup_matplotlib()
    names = ["no_shielding", "uniform_healpix_average", "weighted_healpix_average"]
    data = [_central_slice(variants[name][field_name], fields, cfg) for name in names]
    vmin, vmax = log10_display_limits(*data, floor=floor, max_decades=8.0)

    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)
    im = None
    for ax, name, arr in zip(axs, names, data):
        im = ax.imshow(
            log10_display_values(arr, floor=floor).T,
            origin="lower",
            extent=[cfg.grid.x_min_au, cfg.grid.x_max_au, cfg.grid.y_min_au, cfg.grid.y_max_au],
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        _draw_xy_overlay(ax, cfg)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("x [au]")
        ax.set_ylabel("y [au]")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label(f"log10({title})")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ratio(
    *,
    out_path: Path,
    numerator: np.ndarray,
    denominator: np.ndarray,
    title: str,
    fields: dict[str, np.ndarray],
    cfg: ValidationConfig,
) -> None:
    plt = _setup_matplotlib()
    ratio = _central_slice(_safe_ratio(numerator, denominator), fields, cfg)
    log_ratio = np.log10(np.clip(ratio, 1.0e-6, 1.0e6))
    finite = log_ratio[np.isfinite(log_ratio)]
    lim = float(max(0.25, np.nanpercentile(np.abs(finite), 98.0))) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    im = ax.imshow(
        log_ratio.T,
        origin="lower",
        extent=[cfg.grid.x_min_au, cfg.grid.x_max_au, cfg.grid.y_min_au, cfg.grid.y_max_au],
        cmap="coolwarm",
        interpolation="nearest",
        aspect="equal",
        vmin=-lim,
        vmax=lim,
    )
    _draw_xy_overlay(ax, cfg)
    ax.set_title(title)
    ax.set_xlabel("x [au]")
    ax.set_ylabel("y [au]")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log10(ratio)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_comparison_plots(
    out_dir: Path,
    cfg: ValidationConfig,
    fields: dict[str, np.ndarray],
    variants: dict[str, dict[str, np.ndarray]],
) -> dict[str, str]:
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    field_specs = [
        ("Xco", "X_CO", 1.0e-30, "viridis"),
        ("nco_cm3", "n_CO [cm^-3]", 1.0e-30, "magma"),
        ("theta_co", "theta_CO", 1.0e-12, "cividis"),
        ("chi_eff", "chi_eff", 1.0e-12, "plasma"),
    ]
    for key, title, floor, cmap in field_specs:
        path = plot_dir / f"{key}_variants.png"
        _plot_field_triplet(
            out_path=path,
            title=title,
            field_name=key,
            variants=variants,
            fields=fields,
            cfg=cfg,
            floor=floor,
            cmap=cmap,
        )
        outputs[f"{key}_variants"] = str(path)

    weighted = variants["weighted_healpix_average"]
    uniform = variants["uniform_healpix_average"]
    for key, title in (
        ("Xco", "weighted / uniform X_CO"),
        ("nco_cm3", "weighted / uniform n_CO"),
        ("theta_co", "weighted / uniform theta_CO"),
    ):
        path = plot_dir / f"{key}_weighted_over_uniform.png"
        _plot_ratio(
            out_path=path,
            numerator=weighted[key],
            denominator=uniform[key],
            title=title,
            fields=fields,
            cfg=cfg,
        )
        outputs[f"{key}_weighted_over_uniform"] = str(path)

    return outputs


def _write_report(out_dir: Path, summary: dict) -> Path:
    path = Path(out_dir) / "report.md"
    lines = [
        "# Cartesian Wedge UV Weighting Validation",
        "",
        f"Run time: {summary['time']}",
        "",
        "## Variants",
        "",
    ]
    for name, item in summary["variants"].items():
        diag = item["meta"]["gow17_diagnostics"]
        lines.extend(
            [
                f"- `{name}`: skip_shielding={diag.get('skip_shielding')}, "
                f"shielding_ray_average={diag.get('shielding_ray_average')}, "
                f"directional_weights_used={diag.get('directional_weights_used')}, "
                f"n_fail={item['meta']['n_fail']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Human Inspection",
            "",
            "Inspect the density plot, the three variant CO maps, and the weighted/uniform ratio maps. "
            "The no-shielding branch solves chemistry with unity shielding factors, so it is a chemical "
            "equilibrium comparison without a shielding fixed-point iteration.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_chemistry_variants(
    *,
    cfg: ValidationConfig,
    rad: RadModel,
    params,
    out_dir: Path,
) -> tuple[dict[str, dict], dict[str, dict[str, np.ndarray]]]:
    variants = ["no_shielding", "uniform_healpix_average", "weighted_healpix_average"]
    summaries: dict[str, dict] = {}
    arrays_by_variant: dict[str, dict[str, np.ndarray]] = {}

    previous_rad_params = rad.params
    previous_writer_params = rad.writer.params
    try:
        with _using_diskbridge_params(params):
            rad.params = params
            rad.writer.params = params
            for variant in variants:
                print(f"[chemistry] Running {variant} ...", flush=True)
                _reset_chemistry_state(rad)
                result = run_chemistry(
                    rad,
                    model="gow17",
                    config=_chemistry_config(cfg, variant),
                    write=True,
                    output_dir=Path(out_dir) / "variants" / variant,
                    force=True,
                    use_cache=False,
                )
                arrays = _variant_arrays(result)
                arrays_by_variant[variant] = arrays
                summaries[variant] = _write_variant_outputs(
                    Path(out_dir) / "variants" / variant,
                    variant,
                    result,
                    arrays,
                )
    finally:
        rad.params = previous_rad_params
        rad.writer.params = previous_writer_params

    return summaries, arrays_by_variant


def run_density_preview(
    *,
    config_path: Path = CONFIG_FILE,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Run the density-preview stage without RADMC-3D or chemistry."""

    cfg = _load_config(Path(config_path))
    if out_dir is None:
        out_dir = ROOT / "outputs" / "density_preview"
    out_dir = Path(out_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {out_dir}. "
                "Pass --overwrite to replace this preview output."
            )
        shutil.rmtree(out_dir)

    fields = build_density_structure(cfg)
    return write_density_outputs(out_dir, cfg, fields)


def run_full_validation(
    *,
    config_path: Path = CONFIG_FILE,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Run RADMC-3D once and compare three GOW17 shielding variants."""

    cfg = _load_config(Path(config_path))
    if out_dir is None:
        outputs = _section(cfg.raw, "outputs")
        out_dir = ROOT / str(outputs.get("default_dir", "outputs/default"))
    out_dir = Path(out_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {out_dir}. "
                "Pass --overwrite to replace this validation output."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = build_density_structure(cfg)
    plot_paths = write_density_plot(out_dir, cfg, fields)
    np.savez_compressed(
        out_dir / "density_fields.npz",
        x_au=fields["x_au"],
        y_au=fields["y_au"],
        z_au=fields["z_au"],
        nH_cm3=fields["nH_cm3"],
        rho_g_cm3=fields["rho_g_cm3"],
        circle_mask=fields["circle_mask"],
        dense_cloud_mask=fields["dense_cloud_mask"],
        wedge_mask=fields["wedge_mask"],
        ambient_mask=fields["ambient_mask"],
    )

    model = _build_model(cfg, fields)
    rad = RadModel(model, model_dir=out_dir)
    print("[radmc3d] Running mctherm and mcmono ...", flush=True)
    params = _run_radmc3d_transport(cfg=cfg, rad=rad, model_dir=out_dir)
    chi = rad.ensure_uv_product("chi_broad", fallback_to_chi=True)
    dust_temperature = rad.ensure_dust_temperature()

    variant_summaries, arrays_by_variant = _run_chemistry_variants(
        cfg=cfg,
        rad=rad,
        params=params,
        out_dir=out_dir,
    )
    comparison_plots = _write_comparison_plots(out_dir, cfg, fields, arrays_by_variant)

    weighted = arrays_by_variant["weighted_healpix_average"]
    uniform = arrays_by_variant["uniform_healpix_average"]
    no_shielding = arrays_by_variant["no_shielding"]
    comparison = {
        "weighted_over_uniform_Xco_stats": _summary_stats(_safe_ratio(weighted["Xco"], uniform["Xco"])),
        "weighted_over_uniform_nco_stats": _summary_stats(_safe_ratio(weighted["nco_cm3"], uniform["nco_cm3"])),
        "uniform_over_no_shielding_Xco_stats": _summary_stats(_safe_ratio(uniform["Xco"], no_shielding["Xco"])),
        "weighted_over_no_shielding_Xco_stats": _summary_stats(_safe_ratio(weighted["Xco"], no_shielding["Xco"])),
    }

    summary = _base_summary(cfg, fields)
    summary.update(
        {
            "stage": "full_validation",
            "time": datetime.now().isoformat(timespec="seconds"),
            "ran_radmc3d": True,
            "ran_chemistry": True,
            "radmc3d": {
                "params_path": str(out_dir / "params.txt"),
                "inputs_dir": str(out_dir / "radmc3d_inputs"),
                "outputs_dir": str(out_dir / "radmc3d_outputs"),
                "nphot_thermal": int(cfg.radmc3d.nphot_thermal),
                "nphot_mono": int(cfg.radmc3d.nphot_mono),
                "nbcores": int(cfg.radmc3d.nbcores),
                "external_uv_chi": float(cfg.radiation.external_chi),
                "star": asdict(cfg.star),
                "star_position_au": [
                    float(cfg.geometry.star_x_au),
                    float(cfg.geometry.star_y_au),
                    float(cfg.geometry.star_z_au),
                ],
                "chi_broad_stats": _summary_stats(_qarray(chi, "dimensionless")),
                "dust_temperature_K_stats": _summary_stats(_qarray(dust_temperature, "K")),
            },
            "chemistry_variant_order": [
                "no_shielding",
                "uniform_healpix_average",
                "weighted_healpix_average",
            ],
            "variants": variant_summaries,
            "comparison": comparison,
            "outputs": {
                "density_fields_npz": str(out_dir / "density_fields.npz"),
                "density_plot": str(plot_paths["density_plot"]),
                "cone_geometry_3d": str(plot_paths["cone_geometry_3d"]),
                "comparison_plots": comparison_plots,
                "summary_json": str(out_dir / "summary.json"),
            },
        }
    )
    report_path = _write_report(out_dir, summary)
    summary["outputs"]["report_md"] = str(report_path)
    (out_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(prog="cartesian_wedge_uv_weighting")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--density-only",
        action="store_true",
        help="Write density/cone preview outputs without running RADMC-3D or chemistry.",
    )
    args = parser.parse_args()

    if args.density_only:
        summary = run_density_preview(
            config_path=args.config,
            out_dir=args.out_dir,
            overwrite=bool(args.overwrite),
        )
    else:
        summary = run_full_validation(
            config_path=args.config,
            out_dir=args.out_dir,
            overwrite=bool(args.overwrite),
        )
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
