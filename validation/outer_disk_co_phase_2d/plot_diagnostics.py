"""Plot diagnostics for the 2D outer-disk CO phase validation."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import numpy as np

from diskbridge.model.io_hdf5 import load_model_hdf5
from diskbridge.model.utils import field_data_as_order

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "default"
ABUNDANCE_FLOOR = 1.0e-20
ABUNDANCE_SPAN_DEX = 10.0
UV_FLOOR = 1.0e-20
UV_SPAN_DEX = 12.0
RATE_FLOOR = 1.0e-30
RATE_SPAN_DEX = 12.0
RATIO_LIMIT = 1.0e3
TGAS_TDUST_RATIO_LIMIT = 1.0e2
TEMPERATURE_MIN_K = 1.0
TEMPERATURE_MAX_K = 2000.0
TEMPERATURE_CMAP = LinearSegmentedColormap.from_list(
    "temperature_blue_cyan_green_yellow_red",
    ["#061a8a", "#00d5ff", "#18a84f", "#ffe600", "#d7191c"],
)

SCALE_CMAPS = {
    "density": "viridis",
    "uv": "viridis",
    "rate": "viridis",
    "abundance": "viridis",
    "fraction": "viridis",
    "ratio": "coolwarm",
    "id": "tab20",
    "status": "coolwarm",
}

FIELD_CMAPS = {
    "CO": "viridis",
    "CO_ice": "viridis",
    "Cplus": "viridis",
    "H2": "viridis",
    "chi_broad": "viridis",
    "chi_eff": "viridis",
    "G_C_ion_actual": "viridis",
    "G_CO_diss_actual": "viridis",
    "F_CO_pdes_total": "viridis",
}

CO_LOSS_ID_LABELS = {
    0: "0 none",
    1: "1 freeze-out",
    2: "2 photodiss.",
    3: "3 He+ destr.",
}
CO_GAIN_ID_LABELS = {
    0: "0 none",
    1: "1 photodes.",
    2: "2 thermal des.",
    3: "3 CR des.",
    4: "4 gas-form proxy",
}
CHEM_STATUS_LABELS = {
    -1: "-1 failed",
    0: "0 OK",
}

VARIANTS = (
    "gas_only_no_co_phase",
    "co_ice_no_photodesorption",
    "co_ice_full_photodesorption",
    "co_ice_full_no_external_uv",
)

PANELS = (
    ("number_density_H", "cm^-3", "nH", "density", "Hydrogen nuclei density", "nH [cm^-3]"),
    ("dust_temperature", "K", "Tdust", "temperature", "Dust temperature", "Tdust [K]"),
    ("gas_temperature", "K", "Tgas", "temperature", "Gas temperature", "Tgas [K]"),
    ("chi_broad", "dimensionless", "chi_broad", "uv", "Unshielded broad UV field", "chi_broad [Draine]"),
    ("chem_chi_eff", "dimensionless", "chi_eff", "uv", "Shielded effective UV field", "chi_eff [Draine]"),
    ("chem_G_C_ion_actual", "dimensionless", "G_C_ion_actual", "uv", "Shielded C ionization field", "G_C_ion [Draine]"),
    ("chem_G_CO_diss_actual", "dimensionless", "G_CO_diss_actual", "uv", "Shielded CO dissociation field", "G_CO_diss [Draine]"),
    ("chem_F_CO_pdes_photon_total", "1/(cm^2 s)", "F_CO_pdes_total", "rate", "CO photodesorption photon flux", "photons cm^-2 s^-1"),
    ("abundance_co", "dimensionless", "CO", "abundance", "Gas-phase CO abundance", "x(CO) per H nucleus"),
    ("abundance_co_ice", "dimensionless", "CO_ice", "abundance", "CO ice abundance", "x(CO ice) per H nucleus"),
    ("abundance_c+", "dimensionless", "Cplus", "abundance", "C+ abundance", "x(C+) per H nucleus"),
    ("abundance_h2", "dimensionless", "H2", "fraction", "H2 abundance", "x(H2) per H nucleus"),
    ("chem_CO_loss_dominant_id", "dimensionless", "CO_loss_id", "id", "Dominant CO loss process", "dominant loss process"),
    ("chem_CO_gain_dominant_id", "dimensionless", "CO_gain_id", "id", "Dominant CO gain process", "dominant gain process"),
    ("chem_status", "dimensionless", "chem_status", "status", "Final GOW17 solver status", "final solver status"),
)

CATEGORY_LABELS = {
    "CO_loss_id": CO_LOSS_ID_LABELS,
    "CO_gain_id": CO_GAIN_ID_LABELS,
    "chem_status": CHEM_STATUS_LABELS,
}


def _field(model, name: str, unit: str) -> np.ndarray | None:
    if model.gas is None or name not in model.gas:
        return None
    q = field_data_as_order(model.gas[name], model.mesh.axis_names()).to(unit)
    return np.asarray(q.magnitude, dtype=np.float64)[:, :, 0]


def _geometry(model) -> tuple[np.ndarray, np.ndarray]:
    au_cm = 1.495978707e13
    r = model.mesh.centers_f64("r", "cm")[:, None] / au_cm
    theta = model.mesh.centers_f64("theta", "rad")[None, :]
    return r * np.sin(theta), r * np.cos(theta)


def _cmap_for(label: str, scale: str):
    if scale == "temperature":
        return TEMPERATURE_CMAP
    return FIELD_CMAPS.get(label, SCALE_CMAPS.get(scale, "cividis"))


def _format_disk_axes(ax, R: np.ndarray, z: np.ndarray) -> None:
    finite_R = R[np.isfinite(R)]
    finite_z = z[np.isfinite(z)]
    ax.set_xlabel("R [au]")
    ax.set_ylabel("z [au]")
    if finite_R.size:
        ax.set_xlim(0.0, float(np.nanmax(finite_R)))
    if finite_z.size:
        zmax = float(np.nanmax(np.abs(finite_z)))
        ax.set_ylim(-zmax, zmax)
    ax.set_aspect("auto")
    ax.axhline(0.0, color="0.2", linewidth=0.5)


def _format_colorbar_tick(value: float) -> str:
    if value == 0.0:
        return "0"
    abs_value = abs(float(value))
    if 1.0e-2 <= abs_value < 1.0e4:
        return f"{value:g}"
    return f"{value:.0e}"


def _decade_colorbar_ticks(vmin: float, vmax: float) -> tuple[list[float], list[str]]:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0.0 or vmax <= vmin:
        return [], []
    emin = int(np.ceil(np.log10(vmin)))
    emax = int(np.floor(np.log10(vmax)))
    ticks = [10.0**exp for exp in range(emin, emax + 1)]
    if len(ticks) > 7:
        step = int(np.ceil(len(ticks) / 7))
        ticks = ticks[::step]
    return ticks, [_format_colorbar_tick(tick) for tick in ticks]


def _positive_values(values: list[np.ndarray]) -> np.ndarray:
    parts = []
    for value in values:
        arr = np.asarray(value, dtype=np.float64)
        parts.append(arr[np.isfinite(arr) & (arr > 0.0)])
    if not parts:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(parts)


def _positive_log_limits(
    values: list[np.ndarray],
    *,
    floor: float,
    span_dex: float,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[float, float]:
    positive = _positive_values(values)
    if not positive.size:
        return floor, 1.0
    vmax = float(np.nanpercentile(positive, upper_percentile))
    vmax = max(vmax, float(np.nanmax(positive)))
    vmin = max(
        float(np.nanpercentile(positive, lower_percentile)),
        floor,
        vmax / (10.0**span_dex),
    )
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0.0 or vmin >= vmax:
        return floor, max(1.0, floor * 10.0)
    return vmin, vmax


def _collect_global_limits(models: dict[str, object]) -> dict[str, tuple[float, float]]:
    groups = {
        "uv": [],
        "abundance": [],
        "temperature": [],
        "density": [],
        "rate": [],
        "fraction": [],
    }
    for model in models.values():
        for field, unit, _label, scale, _title, _cbar_label in PANELS:
            if scale not in groups:
                continue
            arr = _field(model, field, unit)
            if arr is not None:
                groups[scale].append(arr)

    limits: dict[str, tuple[float, float]] = {}
    limits["abundance"] = _positive_log_limits(
        groups["abundance"],
        floor=ABUNDANCE_FLOOR,
        span_dex=ABUNDANCE_SPAN_DEX,
    )
    limits["uv"] = _positive_log_limits(
        groups["uv"],
        floor=UV_FLOOR,
        span_dex=UV_SPAN_DEX,
    )
    limits["rate"] = _positive_log_limits(
        groups["rate"],
        floor=RATE_FLOOR,
        span_dex=RATE_SPAN_DEX,
    )
    limits["fraction"] = _positive_log_limits(
        groups["fraction"],
        floor=1.0e-8,
        span_dex=8.0,
    )
    limits["density"] = _positive_log_limits(
        groups["density"],
        floor=1.0e-4,
        span_dex=16.0,
    )
    if groups["temperature"]:
        positive = _positive_values(groups["temperature"])
        vmin = max(float(np.nanpercentile(positive, 1.0)), TEMPERATURE_MIN_K) if positive.size else TEMPERATURE_MIN_K
        vmax = min(max(float(np.nanpercentile(positive, 99.0)), vmin * 10.0), TEMPERATURE_MAX_K) if positive.size else TEMPERATURE_MAX_K
        limits["temperature"] = (vmin, vmax)
    else:
        limits["temperature"] = (TEMPERATURE_MIN_K, TEMPERATURE_MAX_K)
    return limits


def _triangulation(R: np.ndarray, z: np.ndarray) -> tuple[mtri.Triangulation, np.ndarray]:
    coords = np.column_stack([R.ravel(), z.ravel()])
    _, keep = np.unique(np.round(coords, decimals=9), axis=0, return_index=True)
    return mtri.Triangulation(coords[keep, 0], coords[keep, 1]), keep


def _plot_values(
    ax,
    R: np.ndarray,
    z: np.ndarray,
    arr: np.ndarray,
    *,
    scale: str,
    limits: tuple[float, float] | None,
    cmap: str = "viridis",
    category_labels: dict[int, str] | None = None,
):
    tri, keep = _triangulation(R, z)
    vals = np.asarray(arr, dtype=np.float64).ravel()[keep]

    if scale in {"density", "temperature", "uv", "rate", "abundance", "fraction"}:
        vmin, vmax = limits or _positive_log_limits([arr], floor=1.0e-30, span_dex=10.0)
        vals = np.where(np.isfinite(vals) & (vals > 0.0), vals, vmin)
        vals = np.maximum(vals, vmin)
        levels = np.geomspace(vmin, vmax, 96)
        image = ax.tricontourf(
            tri,
            vals,
            levels=levels,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extend="both",
        )
        ticks, labels = _decade_colorbar_ticks(vmin, vmax)
        return image, ticks, labels

    if scale in {"id", "status"}:
        finite = vals[np.isfinite(vals)]
        if finite.size:
            vmin = int(np.nanmin(finite))
            vmax = int(np.nanmax(finite))
        else:
            vmin, vmax = 0, 1
        levels = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
        vals = np.where(np.isfinite(vals), vals, vmin)
        image = ax.tricontourf(
            tri,
            vals,
            levels=levels,
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1),
            extend="neither",
        )
        if category_labels:
            ticks = [tick for tick in sorted(category_labels) if vmin <= tick <= vmax]
            if not ticks:
                ticks = list(range(vmin, vmax + 1))
            labels = [category_labels.get(tick, str(tick)) for tick in ticks]
        else:
            ticks = list(range(vmin, vmax + 1))
            labels = [str(tick) for tick in ticks]
        return image, ticks, labels

    if scale == "ratio":
        vmin, vmax = limits or (1.0 / RATIO_LIMIT, RATIO_LIMIT)
        vals = np.where(np.isfinite(vals) & (vals > 0.0), vals, 1.0)
        vals = np.clip(vals, vmin, vmax)
        levels = np.geomspace(vmin, vmax, 96)
        image = ax.tricontourf(
            tri,
            vals,
            levels=levels,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extend="both",
        )
        ticks = [tick for tick in [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1.0e3] if vmin <= tick <= vmax]
        return image, ticks, [_format_colorbar_tick(tick) for tick in ticks]

    finite = vals[np.isfinite(vals)]
    if finite.size and np.nanmin(finite) < 0.0:
        vmax = float(np.nanpercentile(np.abs(finite), 99.0))
        vmax = max(vmax, 1.0)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        image = ax.tricontourf(
            tri,
            vals,
            levels=np.linspace(-vmax, vmax, 97),
            cmap="coolwarm",
            norm=SymLogNorm(linthresh=max(vmax * 1.0e-4, 1.0e-12), vmin=-vmax, vmax=vmax),
            extend="both",
        )
        return image, [], []

    vmin = float(np.nanpercentile(finite, 1.0)) if finite.size else 0.0
    vmax = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    vals = np.where(np.isfinite(vals), vals, vmin)
    image = ax.tricontourf(
        tri,
        vals,
        levels=np.linspace(vmin, vmax, 96),
        cmap=cmap,
        extend="both",
    )
    return image, [], []


def _plot_panel(
    model,
    field: str,
    unit: str,
    filename_label: str,
    scale: str,
    title: str,
    cbar_label: str,
    out_path: Path,
    limits: dict[str, tuple[float, float]],
) -> bool:
    arr = _field(model, field, unit)
    if arr is None:
        return False
    R, z = _geometry(model)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    im, ticks, labels = _plot_values(
        ax,
        R,
        z,
        arr,
        scale=scale,
        limits=limits.get(scale),
        cmap=_cmap_for(filename_label, scale),
        category_labels=CATEGORY_LABELS.get(filename_label),
    )
    ax.set_title(title)
    _format_disk_axes(ax, R, z)
    cbar = fig.colorbar(im, ax=ax, label=cbar_label)
    if ticks:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_variant(snapshot: Path, plots_dir: Path, limits: dict[str, tuple[float, float]]) -> list[str]:
    model = load_model_hdf5(snapshot)
    written: list[str] = []
    for field, unit, filename_label, scale, title, cbar_label in PANELS:
        out_path = plots_dir / f"{filename_label}.png"
        if _plot_panel(model, field, unit, filename_label, scale, title, cbar_label, out_path, limits):
            written.append(str(out_path))
    tgas = _field(model, "gas_temperature", "K")
    tdust = _field(model, "dust_temperature", "K")
    if tgas is not None and tdust is not None:
        ratio = np.divide(tgas, tdust, out=np.full_like(tgas, np.nan), where=tdust > 0.0)
        R, z = _geometry(model)
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        im, ticks, labels = _plot_values(
            ax,
            R,
            z,
            ratio,
            scale="ratio",
            limits=(1.0 / TGAS_TDUST_RATIO_LIMIT, TGAS_TDUST_RATIO_LIMIT),
            cmap=SCALE_CMAPS["ratio"],
        )
        ax.set_title("Gas/dust temperature ratio")
        _format_disk_axes(ax, R, z)
        cbar = fig.colorbar(im, ax=ax, label="Tgas / Tdust")
        if ticks:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels)
        path = plots_dir / "Tgas_over_Tdust.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def _load_variant_models(out_dir: Path):
    models = {}
    for variant in VARIANTS:
        snapshot = out_dir / "variants" / variant / "snapshot.h5"
        if snapshot.exists():
            models[variant] = load_model_hdf5(snapshot)
    return models


def _ratio_plot(
    model_a,
    model_b,
    field: str,
    unit: str,
    title: str,
    out_path: Path,
    *,
    cbar_label: str = "ratio",
    floor: float = ABUNDANCE_FLOOR,
) -> bool:
    a = _field(model_a, field, unit)
    b = _field(model_b, field, unit)
    if a is None or b is None:
        return False
    a_safe = np.where(np.isfinite(a) & (a > 0.0), a, floor)
    b_safe = np.where(np.isfinite(b) & (b > 0.0), b, floor)
    ratio = a_safe / b_safe
    R, z = _geometry(model_a)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    im, ticks, labels = _plot_values(
        ax,
        R,
        z,
        ratio,
        scale="ratio",
        limits=(1.0 / RATIO_LIMIT, RATIO_LIMIT),
        cmap=SCALE_CMAPS["ratio"],
    )
    ax.set_title(title)
    _format_disk_axes(ax, R, z)
    cbar = fig.colorbar(im, ax=ax, label=cbar_label)
    if ticks:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def write_all_plots(out_dir: Path) -> dict[str, list[str]]:
    """Write all per-variant and cross-variant plots for an output directory."""

    out_dir = Path(out_dir)
    written: dict[str, list[str]] = {}
    models = _load_variant_models(out_dir)
    limits = _collect_global_limits(models)
    for variant in VARIANTS:
        snapshot = out_dir / "variants" / variant / "snapshot.h5"
        if snapshot.exists():
            written[variant] = _plot_variant(
                snapshot,
                out_dir / "variants" / variant / "plots",
                limits,
            )

    comparison_dir = out_dir / "comparison" / "plots"
    comparison_written: list[str] = []
    full = models.get("co_ice_full_photodesorption")
    no_external = models.get("co_ice_full_no_external_uv")
    no_pdes = models.get("co_ice_no_photodesorption")
    gas_only = models.get("gas_only_no_co_phase")
    if full is not None and no_pdes is not None:
        comparisons = (
            (
                "abundance_co_ice",
                "full_over_no_pdes_CO_ice",
                "CO ice: full photodesorption / no photodesorption",
            ),
            (
                "abundance_co",
                "full_over_no_pdes_CO",
                "Gas CO: full photodesorption / no photodesorption",
            ),
        )
        for field, label, title in comparisons:
            path = comparison_dir / f"{label}.png"
            if _ratio_plot(full, no_pdes, field, "dimensionless", title, path, cbar_label="abundance ratio"):
                comparison_written.append(str(path))
    if gas_only is not None and full is not None:
        path = comparison_dir / "gas_only_over_full_CO.png"
        title = "Gas CO: gas-only chemistry / full CO ice chemistry"
        if _ratio_plot(
            gas_only,
            full,
            "abundance_co",
            "dimensionless",
            title,
            path,
            cbar_label="abundance ratio",
            floor=ABUNDANCE_FLOOR,
        ):
            comparison_written.append(str(path))
    if full is not None and no_external is not None:
        comparisons = (
            (
                "abundance_co",
                "dimensionless",
                "external_over_no_external_CO",
                "Gas CO: external UV / no external UV",
                "abundance ratio",
                ABUNDANCE_FLOOR,
            ),
            (
                "abundance_co_ice",
                "dimensionless",
                "external_over_no_external_CO_ice",
                "CO ice: external UV / no external UV",
                "abundance ratio",
                ABUNDANCE_FLOOR,
            ),
            (
                "chem_chi_eff",
                "dimensionless",
                "external_over_no_external_chi_eff",
                "Shielded effective UV: external UV / no external UV",
                "UV-field ratio",
                UV_FLOOR,
            ),
            (
                "chem_F_CO_pdes_photon_total",
                "1/(cm^2 s)",
                "external_over_no_external_F_CO_pdes_total",
                "CO photodesorption photon flux: external UV / no external UV",
                "photon-flux ratio",
                RATE_FLOOR,
            ),
        )
        for field, unit, label, title, cbar_label, floor in comparisons:
            path = comparison_dir / f"{label}.png"
            if _ratio_plot(full, no_external, field, unit, title, path, cbar_label=cbar_label, floor=floor):
                comparison_written.append(str(path))
    written["comparison"] = comparison_written
    return written


def main() -> None:
    write_all_plots(OUTPUT_DIR)


if __name__ == "__main__":
    main()
