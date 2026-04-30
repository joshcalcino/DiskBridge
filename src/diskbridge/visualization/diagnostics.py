"""Diagnostic plotting helpers for DiskBridge workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from diskbridge._logging import logger
from diskbridge._units import Quantity
from diskbridge.model.field import Field
from diskbridge.model.profiles import compute_volume_weighted_mean_radial_profile
from diskbridge.model.utils import field_data_as_order
from diskbridge.serialization import jsonable
from diskbridge.visualization.profiles import plot_midplane_xy_map, plot_phi_avg_rz_slice

if TYPE_CHECKING:
    from diskbridge.chemistry.types import ChemistryResult
    from diskbridge.model.core import Model
    from diskbridge.radmc3d.model import RadModel


def _safe_name(name: str) -> str:
    """Convert a field name to a filesystem-safe plot stem.

    Parameters
    ----------
    name : str
        Field name.

    Returns
    -------
    str
        Safe filename stem.
    """
    return (
        name.replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _register_quantity_field(
    model: "Model",
    name: str,
    data: np.ndarray,
    *,
    quantity: str | None = None,
    unit: str = "dimensionless",
    attrs: dict[str, Any] | None = None,
) -> None:
    """Register a dimensionless or unit-bearing diagnostic field.

    Parameters
    ----------
    model : Model
        Model that receives the field.
    name : str
        Field name to register.
    data : ndarray
        Field values matching the model mesh shape.
    quantity : str, optional
        Semantic quantity name. Defaults to ``name``.
    unit : str, optional
        Pint-compatible unit string.
    attrs : dict, optional
        Metadata attached to the field.

    Returns
    -------
    None
        Registers the field in place.
    """
    model.gas_register(
        name,
        Field(
            quantity=quantity or name,
            data=Quantity(np.asarray(data, dtype=float), unit),
            axis_order=model.mesh.axis_names(),
            attrs=attrs or {},
        ),
    )


def _register_derived_abundance_fields(rad: "RadModel") -> list[str]:
    """Register abundances derived from chemistry number-density fields.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D wrapper containing chemistry number-density fields.

    Returns
    -------
    list[str]
        Names of derived abundance fields that were registered.
    """
    model = rad.model
    nH = np.asarray(
        field_data_as_order(
            Field(
                quantity="nH",
                data=rad.ensure_nH().to("cm^-3"),
                axis_order=model.mesh.axis_names(),
            ),
            model.mesh.axis_names(),
        ).magnitude,
        dtype=float,
    )

    registered: list[str] = []
    for source_name in sorted(model.gas.keys()):
        source_name = str(source_name)
        if not source_name.startswith("number_density_"):
            continue
        species = source_name.removeprefix("number_density_")
        output_name = f"abundance_{_safe_name(species)}"
        if output_name in model.gas:
            continue
        number_density = field_data_as_order(
            model.gas[source_name],
            model.mesh.axis_names(),
        ).to("cm^-3").magnitude
        abundance = np.divide(
            np.asarray(number_density, dtype=float),
            nH,
            out=np.zeros_like(nH, dtype=float),
            where=nH > 0.0,
        )
        _register_quantity_field(
            model,
            output_name,
            abundance,
            attrs={
                "source": "derived",
                "kind": "abundance",
                "from": source_name,
                "normalization": "nH",
            },
        )
        registered.append(output_name)
    return registered


def _register_uv_directional_diagnostics(rad: "RadModel") -> list[str]:
    """Register diagnostic fields derived from cached HEALPix UV weights.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D wrapper that may contain ``W_rays``.

    Returns
    -------
    list[str]
        Names of UV diagnostic fields that were registered.
    """
    model = rad.model
    W_rays = getattr(rad, "W_rays", None)
    if W_rays is None:
        return []

    W = np.asarray(W_rays, dtype=float)
    if W.ndim != 2 or W.shape[0] != int(np.prod(model.mesh.shape)):
        return []

    npix = int(W.shape[1])
    shape = model.mesh.shape
    uniform_weight = 1.0 / float(npix)
    wmax = np.max(W, axis=1).reshape(shape)
    direct_to_iso = wmax / uniform_weight

    _register_quantity_field(
        model,
        "chem_uv_direct_to_isotropic",
        direct_to_iso,
        quantity="chem_uv_direct_to_isotropic",
        attrs={
            "source": "W_rays",
            "description": "max directional UV weight divided by isotropic ray weight; 1 is isotropic",
            "nside": int(round((npix / 12.0) ** 0.5)),
            "npix": npix,
        },
    )
    _register_quantity_field(
        model,
        "chem_uv_max_ray_weight",
        wmax,
        quantity="chem_uv_max_ray_weight",
        attrs={
            "source": "W_rays",
            "description": "largest angular UV weight in each cell",
            "nside": int(round((npix / 12.0) ** 0.5)),
            "npix": npix,
        },
    )
    return ["chem_uv_direct_to_isotropic", "chem_uv_max_ray_weight"]


def _plot_convergence(result: "ChemistryResult", plots_dir: Path) -> Path | None:
    """Plot GOW17 convergence histories when available.

    Parameters
    ----------
    result : ChemistryResult
        Chemistry result with ``gow17_diagnostics`` metadata.
    plots_dir : pathlib.Path
        Output directory.

    Returns
    -------
    pathlib.Path or None
        Path to the plot, or ``None`` if no convergence data exist.
    """
    diag = result.meta.get("gow17_diagnostics", {})
    d_h2 = np.asarray(diag.get("d_h2_hist", []), dtype=float)
    d_co = np.asarray(diag.get("d_co_hist", []), dtype=float)
    d_tgas = np.asarray(diag.get("d_tgas_hist", []), dtype=float)
    d_h2_median = np.asarray(diag.get("d_h2_median_hist", []), dtype=float)
    d_co_median = np.asarray(diag.get("d_co_median_hist", []), dtype=float)
    d_tgas_median = np.asarray(diag.get("d_tgas_median_hist", []), dtype=float)
    well_fraction = np.asarray(
        diag.get("well_converged_fraction_hist", []),
        dtype=float,
    )
    bad_status = np.asarray(diag.get("bad_status_hist", []), dtype=float)
    if d_h2.size == 0 and d_co.size == 0:
        return None

    import matplotlib.pyplot as plt

    out = plots_dir / "convergence_gow17.png"
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax = axes[0]
    for values, marker, label in (
        (d_h2, "o", "H2 max"),
        (d_co, "s", "CO max"),
        (d_tgas, "^", "Tgas max"),
    ):
        if values.size:
            ax.semilogy(
                np.arange(1, values.size + 1),
                np.maximum(values, 1e-300),
                marker=marker,
                label=label,
            )
    for values, label in (
        (d_h2_median, "H2 median"),
        (d_co_median, "CO median"),
        (d_tgas_median, "Tgas median"),
    ):
        if values.size:
            ax.semilogy(
                np.arange(1, values.size + 1),
                np.maximum(values, 1e-300),
                linestyle="--",
                label=label,
            )
    reltol = diag.get("well_converged_reltol")
    if reltol is not None:
        ax.axhline(float(reltol), color="k", linewidth=0.8, alpha=0.5, label="reltol")
    ax.set_ylabel("relative change")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    ax_cells = axes[1]
    if well_fraction.size:
        ax_cells.plot(
            np.arange(1, well_fraction.size + 1),
            100.0 * well_fraction,
            marker="o",
            label="well-converged cells",
        )
        ax_cells.set_ylim(-2.0, 102.0)
        ax_cells.set_ylabel("cells below reltol [%]")
    if bad_status.size:
        ax_status = ax_cells.twinx()
        ax_status.plot(
            np.arange(1, bad_status.size + 1),
            bad_status,
            color="0.35",
            linestyle=":",
            marker=".",
            label="bad cells",
        )
        ax_status.set_ylabel("solver nonzero-status cells")
    ax_cells.set_xlabel("chemistry/shielding update")
    ax_cells.grid(True, alpha=0.25)
    ax_cells.legend(loc="lower right", fontsize=8)

    axes[0].set_title(
        f"n_fail={result.meta.get('n_fail')}, shielding_iter={diag.get('shielding_iter')}"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _plot_radial_profile_panel(
    model: "Model",
    fields: list[tuple[str, str]],
    out: Path,
    *,
    ylabel: str,
    logy: bool = True,
) -> Path | None:
    """Plot a panel of volume-weighted radial profiles.

    Parameters
    ----------
    model : Model
        Model containing fields to plot.
    fields : list[tuple[str, str]]
        Field names and display labels.
    out : pathlib.Path
        Output image path.
    ylabel : str
        Y-axis label.
    logy : bool, optional
        Whether to use logarithmic y-scaling.

    Returns
    -------
    pathlib.Path or None
        Path to the plot, or ``None`` if none of the fields exist.
    """
    available = [(name, label) for name, label in fields if name in model.gas]
    if not available:
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label in available:
        r_au, profile = compute_volume_weighted_mean_radial_profile(model, name)
        y = np.asarray(profile, dtype=float)
        if logy:
            y = np.maximum(y, np.finfo(np.float64).tiny)
        ax.plot(r_au, y, label=label)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("r [au]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _plot_field_maps(
    model: "Model",
    field_name: str,
    plots_dir: Path,
    *,
    log10: bool = True,
    cmap: str = "viridis",
    log10_dyn_range_dex: float | None = 8.0,
    vline_x: float | None = None,
) -> list[Path]:
    """Plot R-z and midplane maps for a model field.

    Parameters
    ----------
    model : Model
        Model containing the field.
    field_name : str
        Field to plot.
    plots_dir : pathlib.Path
        Output directory.
    log10 : bool, optional
        Whether to plot log10 values.
    cmap : str, optional
        Matplotlib colormap.
    log10_dyn_range_dex : float, optional
        Dynamic range for log-scaled plots.
    vline_x : float, optional
        Optional radial marker for the R-z map.

    Returns
    -------
    list[pathlib.Path]
        Paths to generated plots.
    """
    if field_name not in model.gas:
        return []

    safe = _safe_name(field_name)
    outputs = [
        plots_dir / f"rz_{safe}.png",
        plots_dir / f"xy_{safe}.png",
    ]
    plot_phi_avg_rz_slice(
        model,
        field_name,
        outputs[0],
        x_axis="r",
        y_axis="z/r",
        log10=log10,
        log10_dyn_range_dex=log10_dyn_range_dex,
        cmap=cmap,
        xscale="log",
        ylim=(-1.0, 1.0),
        vline_x=vline_x,
    )
    plot_midplane_xy_map(
        model,
        field_name,
        outputs[1],
        log10=log10,
        cmap=cmap,
    )
    return outputs


def _write_summary(
    *,
    rad: "RadModel",
    plots_dir: Path,
    result: "ChemistryResult | None" = None,
    segmented_result: dict[str, Any] | None = None,
) -> None:
    """Write a JSON summary for diagnostic plotting outputs.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D wrapper used for diagnostics.
    plots_dir : pathlib.Path
        Diagnostic output directory.
    result : ChemistryResult, optional
        Chemistry result to summarize.
    segmented_result : dict, optional
        Segmented RT metadata to summarize.

    Returns
    -------
    None
        Writes ``diagnostic_summary.json``.
    """
    model = rad.model
    summary: dict[str, Any] = {
        "plots_dir": str(plots_dir),
        "fields_available": sorted([str(k) for k in model.gas.keys()]),
    }
    if result is not None:
        summary["chemistry_meta"] = jsonable(result.meta)

    W_rays = getattr(rad, "W_rays", None)
    if W_rays is not None:
        W = np.asarray(W_rays)
        summary["W_rays"] = {
            "shape": list(W.shape),
            "max_weight_min": float(np.nanmin(np.max(W, axis=1))) if W.size else None,
            "max_weight_max": float(np.nanmax(np.max(W, axis=1))) if W.size else None,
        }
    if segmented_result is not None:
        summary["segmented_rt"] = {
            key: jsonable(value)
            for key, value in segmented_result.items()
            if key not in ("temperature", "chi")
        }

    (plots_dir / "diagnostic_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n"
    )


def make_chemistry_diagnostic_plots(
    rad: "RadModel",
    result: "ChemistryResult",
    plots_dir: str | Path,
) -> list[Path]:
    """Create diagnostic plots for chemistry outputs.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D wrapper whose model contains the chemistry fields.
    result : ChemistryResult
        Chemistry result returned by ``run_chemistry``.
    plots_dir : str or pathlib.Path
        Directory where diagnostic plots and summary JSON are written.

    Returns
    -------
    list[pathlib.Path]
        Paths of generated plot files.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model = rad.model

    _register_derived_abundance_fields(rad)
    _register_uv_directional_diagnostics(rad)

    made: list[Path] = []
    convergence = _plot_convergence(result, plots_dir)
    if convergence is not None:
        made.append(convergence)

    profile_specs = [
        (
            [("dust_temperature", "Tdust"), ("gas_temperature", "Tgas")],
            plots_dir / "radial_profiles_temperature.png",
            "volume-weighted temperature [K]",
            True,
        ),
        (
            [
                ("abundance_h2", "H2"),
                ("abundance_h", "H"),
                ("abundance_co", "CO"),
                ("abundance_co_ice", "CO ice"),
                ("abundance_cplus", "C+"),
                ("abundance_catom", "C"),
                ("abundance_e", "e-"),
            ],
            plots_dir / "radial_profiles_abundances.png",
            "volume-weighted abundance",
            True,
        ),
        (
            [
                ("abundance_co", "gas CO"),
                ("abundance_co_ice", "frozen CO"),
            ],
            plots_dir / "radial_profiles_co_freezeout.png",
            "volume-weighted CO abundance",
            True,
        ),
        (
            [
                ("chi", "chi"),
                ("chem_chi_eff", "chi_eff"),
                ("chem_theta_h2", "theta_H2"),
                ("chem_theta_co", "theta_CO"),
                ("chem_theta_c", "theta_C"),
                ("chem_uv_direct_to_isotropic", "direct/isotropic UV proxy"),
            ],
            plots_dir / "radial_profiles_shielding_uv.png",
            "volume-weighted shielding / UV factor",
            True,
        ),
    ]
    for fields, out, ylabel, logy in profile_specs:
        profile = _plot_radial_profile_panel(model, fields, out, ylabel=ylabel, logy=logy)
        if profile is not None:
            made.append(profile)

    map_specs = [
        ("density", True, "magma", 8.0),
        ("dust_temperature", True, "inferno", 3.0),
        ("gas_temperature", True, "inferno", 3.0),
        ("chi", True, "viridis", 8.0),
        ("chem_chi_eff", True, "viridis", 8.0),
        ("chem_theta_h2", True, "cividis", 8.0),
        ("chem_theta_co", True, "cividis", 8.0),
        ("chem_theta_c", True, "cividis", 8.0),
        ("chem_uv_direct_to_isotropic", True, "plasma", 4.0),
        ("abundance_h2", True, "viridis", 8.0),
        ("abundance_h", True, "viridis", 8.0),
        ("abundance_co", True, "viridis", 10.0),
        ("abundance_co_ice", True, "cividis", 10.0),
        ("abundance_cplus", True, "viridis", 8.0),
        ("abundance_catom", True, "viridis", 8.0),
        ("abundance_e", True, "viridis", 8.0),
    ]
    for field_name, log10, cmap, dyn_range in map_specs:
        try:
            made.extend(
                _plot_field_maps(
                    model,
                    field_name,
                    plots_dir,
                    log10=log10,
                    cmap=cmap,
                    log10_dyn_range_dex=dyn_range,
                )
            )
        except Exception as exc:
            logger.warning("Failed to plot chemistry field %s: %s", field_name, exc)

    _write_summary(rad=rad, result=result, plots_dir=plots_dir)
    return made


def make_segmented_rt_diagnostic_plots(
    rad: "RadModel",
    segmented_result: dict[str, Any],
    plots_dir: str | Path,
) -> list[Path]:
    """Create diagnostic plots for a segmented RADMC-3D run.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D wrapper after ``compute_segmented_rt`` has populated merged
        ``dust_temperature`` and ``chi`` fields.
    segmented_result : dict
        Result dictionary returned by ``compute_segmented_rt``.
    plots_dir : str or pathlib.Path
        Directory where diagnostic plots and summary JSON are written.

    Returns
    -------
    list[pathlib.Path]
        Paths of generated plot files.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model = rad.model

    split_radii = list(segmented_result.get("split_radii_au", []))
    vline_x = float(split_radii[-1]) if split_radii else None

    made: list[Path] = []
    profile = _plot_radial_profile_panel(
        model,
        [("dust_temperature", "Tdust"), ("chi", "chi")],
        plots_dir / "radial_profiles_segmented_rt.png",
        ylabel="volume-weighted RT field",
        logy=True,
    )
    if profile is not None:
        made.append(profile)

    for field_name, cmap, dyn_range in (
        ("density", "magma", 8.0),
        ("dust_temperature", "inferno", 3.0),
        ("chi", "viridis", 8.0),
    ):
        try:
            made.extend(
                _plot_field_maps(
                    model,
                    field_name,
                    plots_dir,
                    log10=True,
                    cmap=cmap,
                    log10_dyn_range_dex=dyn_range,
                    vline_x=vline_x,
                )
            )
        except Exception as exc:
            logger.warning("Failed to plot segmented RT field %s: %s", field_name, exc)

    _write_summary(rad=rad, segmented_result=segmented_result, plots_dir=plots_dir)
    return made
