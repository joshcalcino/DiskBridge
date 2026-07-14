"""Diagnostic plotting helpers for DiskBridge workflows."""

# db-keywords: uv-products, units, radmc3d, model, mesh, field
# db-role: entrypoint
# db-scope: package
# db-purpose: Diagnostic plotting helpers for DiskBridge workflows.

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


def _register_uv_closure_diagnostics(rad: "RadModel") -> list[str]:
    """Register scalar UV closure diagnostics from optional W_rays metadata.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D wrapper that may contain ``W_rays_closure_diagnostics``.

    Returns
    -------
    list[str]
        Names of diagnostic fields registered on the model.
    """
    diagnostics = getattr(rad, "W_rays_closure_diagnostics", None)
    if not isinstance(diagnostics, dict):
        return []
    fields = diagnostics.get("fields", {})
    if not isinstance(fields, dict) or not fields:
        return []

    model = rad.model
    shape = tuple(int(v) for v in diagnostics.get("shape", model.mesh.shape))
    if shape != tuple(model.mesh.shape):
        logger.warning(
            "Skipping W_rays closure diagnostics with shape %s; model shape is %s",
            shape,
            model.mesh.shape,
        )
        return []

    registered: list[str] = []
    descriptions = {
        "chi_radmc": "scalar UV product used to build directional weights",
        "chi_ext_dir": "angle-mean direct external UV contribution",
        "chi_star_dir_att": "attenuated direct stellar UV contribution",
        "chi_iso": "isotropic residual UV contribution",
        "chi_direct": "direct external plus direct stellar UV contribution",
        "chi_reconstructed": "direct plus isotropic reconstructed UV contribution",
        "closure_residual": "chi_radmc - chi_reconstructed",
        "direct_excess": "max(chi_direct - chi_radmc, 0)",
        "direct_to_radmc": "chi_direct / chi_radmc",
        "f_ext": "fraction of reconstructed UV from direct external component",
        "f_star": "fraction of reconstructed UV from direct stellar component",
        "f_iso": "fraction of reconstructed UV from isotropic residual component",
        "tau_star": "dust UV optical depth along the starward ray",
        "chi_star_unatt": "unattenuated direct stellar UV contribution",
        "outer_weight_overridden": "1 where segmented outer weighting replaced the normal weights",
    }

    for name, values in fields.items():
        arr = np.asarray(values, dtype=float)
        if arr.size != int(np.prod(shape)):
            continue
        field_name = f"chem_uv_closure_{name}"
        _register_quantity_field(
            model,
            field_name,
            arr.reshape(shape),
            quantity=field_name,
            attrs={
                "source": "W_rays_closure_diagnostics",
                "description": descriptions.get(str(name), str(name)),
                "nside": diagnostics.get("nside"),
                "npix": diagnostics.get("npix"),
                "outer_weight_mode": diagnostics.get("outer_weight_mode"),
                "isotropic_outside_r_cm": diagnostics.get("isotropic_outside_r_cm"),
            },
        )
        registered.append(field_name)
    return registered


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


def _positive_log_percentile_limits(
    data: np.ndarray,
    *,
    lower: float = 1.0,
    upper: float = 99.0,
    max_dyn_range_dex: float | None = 8.0,
) -> tuple[float, float] | tuple[None, None]:
    values = np.asarray(data, dtype=float)
    ok = np.isfinite(values) & (values > 0.0)
    if not np.any(ok):
        return None, None
    log_values = np.log10(values[ok])
    vmin = float(np.nanpercentile(log_values, lower))
    vmax = float(np.nanpercentile(log_values, upper))
    if max_dyn_range_dex is not None and np.isfinite(vmax):
        vmin = max(vmin, vmax - float(max_dyn_range_dex))
    return vmin, vmax


def _component_density_fields(model: "Model") -> dict[int, Field]:
    dust = getattr(model, "dust", None)
    if dust is None:
        return {}

    totals: dict[int, Quantity] = {}
    for bin_name, dust_bin in getattr(dust, "bins", {}).items():
        density = dust_bin["density"]
        if hasattr(dust, "_global_bins"):
            comp_idx, _ = dust._global_bins[bin_name]
        else:
            comp_idx = 0
        if comp_idx not in totals:
            totals[int(comp_idx)] = density.data.copy()
        else:
            totals[int(comp_idx)] = totals[int(comp_idx)] + density.data

    total_density = dust["density"] if getattr(dust, "bins", None) else None
    if total_density is None:
        return {}

    fields: dict[int, Field] = {}
    for comp_idx, data in totals.items():
        fields[comp_idx] = Field(
            data=data,
            quantity=total_density.quantity,
            axis_order=total_density.axis_order,
            attrs={"source": "dust_component_bins", "component_index": comp_idx},
        )
    return fields


def _dust_to_gas_field(
    model: "Model",
    dust_density: Field,
    *,
    quantity: str,
) -> Field:
    rho_g = field_data_as_order(
        model.gas["density"],
        model.mesh.axis_names(),
    ).to("g/cm^3").magnitude
    rho_d = field_data_as_order(
        dust_density,
        model.mesh.axis_names(),
    ).to("g/cm^3").magnitude
    ratio = np.divide(
        np.asarray(rho_d, dtype=float),
        np.asarray(rho_g, dtype=float),
        out=np.full_like(rho_g, np.nan, dtype=float),
        where=rho_g > 0.0,
    )
    return Field(
        data=Quantity(ratio, "dimensionless"),
        quantity=quantity,
        axis_order=model.mesh.axis_names(),
    )


def _plot_dust_component_masks(
    model: "Model",
    masks: list[tuple[str, Field]],
    output: Path,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path | None:
    if not masks:
        return None

    import matplotlib.pyplot as plt

    mesh = model.mesh
    r_edges_au = mesh.edges("r").to("au").magnitude
    theta_edges = mesh.edges("theta").to("radian").magnitude
    y_edges = np.cos(theta_edges)

    fig, axes = plt.subplots(
        1,
        len(masks),
        figsize=(4.5 * len(masks), 3.6),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, field) in zip(axes, masks):
        data = field_data_as_order(field, mesh.axis_names()).to("dimensionless").magnitude
        z_plot = np.nanmean(np.asarray(data, dtype=float), axis=2).T
        y_plot = y_edges
        if y_plot[0] > y_plot[-1]:
            y_plot = y_plot[::-1]
            z_plot = z_plot[::-1, :]
        pc = ax.pcolormesh(
            r_edges_au,
            y_plot,
            z_plot,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(pc, ax=ax)
        ax.set_xscale("log")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(f"{label} mask")
        ax.set_xlabel("r [au]")
        ax.set_ylabel("z/r")

    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def _plot_dust_component_comparison(
    model: "Model",
    panels: list[tuple[str, Field]],
    output: Path,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path | None:
    if not panels:
        return None

    import matplotlib.pyplot as plt

    mesh = model.mesh
    r_edges_au = mesh.edges("r").to("au").magnitude
    theta_edges = mesh.edges("theta").to("radian").magnitude
    y_edges = np.cos(theta_edges)

    ncols = 2 if len(panels) > 1 else 1
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.6 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, field) in zip(axes, panels):
        data = field_data_as_order(field, model.mesh.axis_names()).to("g/cm^3").magnitude
        vmin, vmax = _positive_log_percentile_limits(data, upper=99.5)
        z_plot = np.log10(np.maximum(np.nanmean(data, axis=2), np.finfo(np.float64).tiny)).T
        y_plot = y_edges
        if y_plot[0] > y_plot[-1]:
            y_plot = y_plot[::-1]
            z_plot = z_plot[::-1, :]
        pc = ax.pcolormesh(
            r_edges_au,
            y_plot,
            z_plot,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(pc, ax=ax)
        ax.set_xscale("log")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("r [au]")
        ax.set_ylabel("z/r")
        ax.set_title(label)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def _plot_midplane_dust_diagnostics(
    model: "Model",
    *,
    output: Path,
    component_density_fields: dict[int, Field],
    component_labels: dict[int, str],
    component_dtg_fields: dict[int, Field],
    mask_field: Field | None,
) -> Path | None:
    if "density" not in model.gas or not hasattr(model, "dust"):
        return None
    try:
        dust_density = model.dust["density"]
    except Exception:
        return None

    import matplotlib.pyplot as plt

    r_au = model.mesh.centers("r").to("au").magnitude
    theta = model.mesh.centers("theta").to("radian").magnitude
    mid_idx = int(np.argmin(np.abs(theta - 0.5 * np.pi)))

    def mid_phi_mean(field: Field) -> np.ndarray:
        data = field_data_as_order(field, model.mesh.axis_names()).magnitude
        return np.nanmean(np.asarray(data, dtype=float)[:, mid_idx, :], axis=1)

    total_dtg = _dust_to_gas_field(model, dust_density, quantity="dust_to_gas_total")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].loglog(r_au, mid_phi_mean(model.gas["density"]), label="gas", color="0.15")
    axes[0].loglog(r_au, mid_phi_mean(dust_density), label="total dust", color="tab:blue")
    for comp_idx, field in sorted(component_density_fields.items()):
        axes[0].loglog(
            r_au,
            mid_phi_mean(field),
            label=f"{component_labels.get(comp_idx, f'component {comp_idx}')} dust",
        )
    axes[0].set_ylabel("midplane density [g cm^-3]")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].semilogx(r_au, mid_phi_mean(total_dtg), label="total", color="tab:blue")
    for comp_idx, field in sorted(component_dtg_fields.items()):
        axes[1].semilogx(
            r_au,
            mid_phi_mean(field),
            label=component_labels.get(comp_idx, f"component {comp_idx}"),
        )
    axes[1].axhline(1e-2, color="0.3", linestyle="--", linewidth=1.0, label="1e-2")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("midplane dust/gas")
    axes[1].legend(loc="best", fontsize=8)

    if mask_field is not None:
        axes[2].semilogx(r_au, mid_phi_mean(mask_field), color="tab:purple")
        axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("midplane disc weight")
    axes[2].set_xlabel("r [au]")

    positive = r_au[np.isfinite(r_au) & (r_au > 0.0)]
    if positive.size:
        axes[2].set_xlim(float(np.nanmin(positive)), float(np.nanmax(positive)))
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def make_dust_component_diagnostic_plots(
    model: "Model",
    plots_dir: str | Path,
    *,
    diagnostics: bool = True,
    component_labels: dict[int, str] | None = None,
    inner_r_max_au: float | None = 80.0,
    inner_z_over_r: float = 0.35,
    include_bin_plots: bool = False,
) -> list[Path]:
    """Create dust distribution diagnostics for disc/ISM component workflows.

    The function is intentionally workflow-level: run files only pass
    ``diagnostics=True`` and receive saved plots plus a JSON manifest.
    """
    if not diagnostics:
        return []
    if not hasattr(model, "dust"):
        return []

    components = list(getattr(model.dust, "_components", []))
    if not components:
        return []

    labels = {0: "disc", 1: "ISM"}
    if component_labels is not None:
        labels.update(component_labels)

    plots_dir = Path(plots_dir)
    key_dir = plots_dir / "key"
    support_dir = plots_dir / "supporting"
    bin_dir = plots_dir / "dust_bins"
    key_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    if include_bin_plots:
        bin_dir.mkdir(parents=True, exist_ok=True)

    r_au = model.mesh.centers("r").to("au").magnitude
    r_positive = r_au[np.isfinite(r_au) & (r_au > 0.0)]
    if r_positive.size:
        r_min_au = float(np.nanmin(r_positive))
        r_max_au = float(np.nanmax(r_positive))
    else:
        r_min_au = 1.0
        r_max_au = 1.0
    xlim_full = (r_min_au, r_max_au)
    xlim_inner = (
        r_min_au,
        min(float(inner_r_max_au), r_max_au) if inner_r_max_au is not None else r_max_au,
    )
    ylim_inner = (-float(inner_z_over_r), float(inner_z_over_r))

    made: list[Path] = []
    summary: dict[str, Any] = {
        "plots_dir": str(plots_dir),
        "n_components": len(components),
        "components": [],
        "plots": [],
    }

    component_density_fields = _component_density_fields(model)
    component_dtg_fields: dict[int, Field] = {}
    try:
        total_dust_density = model.dust["density"]
    except Exception:
        total_dust_density = None
    masks: list[tuple[str, Field]] = []
    for idx, component in enumerate(components):
        label = labels.get(idx, f"component_{idx}")
        distribution = getattr(component, "distribution", None)
        meta = {
            "component_index": idx,
            "label": label,
            "mode": getattr(component, "mode", None),
            "dust_to_gas_ratio": getattr(component, "dust_to_gas_ratio", None),
            "nbin": getattr(distribution, "nbin", None),
        }
        if distribution is not None:
            meta["amin_um"] = float(distribution.amin.to("um").magnitude)
            meta["amax_um"] = float(distribution.amax.to("um").magnitude)
            meta["bin_centers_um"] = [
                float(value)
                for value in np.asarray(distribution.bin_centers.to("um").magnitude, dtype=float)
            ]
            meta["mass_fractions"] = [
                float(value)
                for value in np.asarray(distribution.mass_fractions, dtype=float)
            ]
        summary["components"].append(jsonable(meta))
        mask = getattr(component, "mask", None)
        if mask is not None:
            masks.append((label, mask))

    for comp_idx, field in sorted(component_density_fields.items()):
        label = labels.get(comp_idx, f"component_{comp_idx}")
        safe = _safe_name(label.lower())
        density_data = field_data_as_order(field, model.mesh.axis_names()).to("g/cm^3").magnitude
        vmin, vmax = _positive_log_percentile_limits(density_data, upper=99.5)
        out = support_dir / f"dust_density_{safe}_total_zoverr_vs_r.png"
        plot_phi_avg_rz_slice(
            model,
            field,
            output=out,
            x_axis="r",
            y_axis="z/r",
            log10=True,
            xscale="log",
            xlim=xlim_full,
            vmin=vmin,
            vmax=vmax,
        )
        made.append(out)

        dtg_field = _dust_to_gas_field(
            model,
            field,
            quantity=f"dust_to_gas_{safe}",
        )
        component_dtg_fields[comp_idx] = dtg_field
        dtg_data = field_data_as_order(dtg_field, model.mesh.axis_names()).magnitude
        vmin_dtg, vmax_dtg = _positive_log_percentile_limits(dtg_data)
        out = support_dir / f"dust_to_gas_{safe}_total_zoverr_vs_r.png"
        plot_phi_avg_rz_slice(
            model,
            dtg_field,
            output=out,
            x_axis="r",
            y_axis="z/r",
            log10=True,
            xscale="log",
            xlim=xlim_full,
            vmin=vmin_dtg,
            vmax=vmax_dtg,
        )
        made.append(out)

    if "density" in model.gas:
        out = support_dir / "gas_density_zoverr_vs_r.png"
        plot_phi_avg_rz_slice(
            model,
            "density",
            output=out,
            x_axis="r",
            y_axis="z/r",
            log10=True,
            xscale="log",
            xlim=xlim_full,
        )
        made.append(out)

    if total_dust_density is not None:
        dust_density = total_dust_density
        dust_data = field_data_as_order(dust_density, model.mesh.axis_names()).to("g/cm^3").magnitude
        vmin_dust, vmax_dust = _positive_log_percentile_limits(dust_data, upper=99.5)
        for out, xlim, ylim in (
            (support_dir / "dust_density_total_zoverr_vs_r.png", xlim_full, None),
            (key_dir / "dust_density_inner_zoverr_vs_r.png", xlim_inner, ylim_inner),
        ):
            plot_phi_avg_rz_slice(
                model,
                dust_density,
                output=out,
                x_axis="r",
                y_axis="z/r",
                log10=True,
                xscale="log",
                xlim=xlim,
                ylim=ylim,
                vmin=vmin_dust,
                vmax=vmax_dust,
            )
            made.append(out)

        total_dtg = _dust_to_gas_field(model, dust_density, quantity="dust_to_gas_total")
        dtg_data = field_data_as_order(total_dtg, model.mesh.axis_names()).magnitude
        vmin_dtg, vmax_dtg = _positive_log_percentile_limits(dtg_data)
        out = key_dir / "dust_to_gas_inner_zoverr_vs_r.png"
        plot_phi_avg_rz_slice(
            model,
            total_dtg,
            output=out,
            x_axis="r",
            y_axis="z/r",
            log10=True,
            xscale="log",
            xlim=xlim_inner,
            ylim=ylim_inner,
            vmin=vmin_dtg,
            vmax=vmax_dtg,
        )
        made.append(out)

    masks_plot = _plot_dust_component_masks(
        model,
        masks,
        key_dir / "dust_component_masks_inner_zoverr_vs_r.png",
        xlim=xlim_inner,
        ylim=ylim_inner,
    )
    if masks_plot is not None:
        made.append(masks_plot)
    for label, mask in masks:
        safe = _safe_name(label.lower())
        out = support_dir / f"dust_component_{safe}_mask_zoverr_vs_r.png"
        plot_phi_avg_rz_slice(
            model,
            mask,
            output=out,
            x_axis="r",
            y_axis="z/r",
            log10=False,
            xscale="log",
            xlim=xlim_full,
        )
        made.append(out)

    panels = [("gas density", model.gas["density"])] if "density" in model.gas else []
    if total_dust_density is not None:
        panels.append(("total dust", total_dust_density))
    for comp_idx, field in sorted(component_density_fields.items()):
        panels.append((f"{labels.get(comp_idx, f'component {comp_idx}')} dust", field))
    comparison = _plot_dust_component_comparison(
        model,
        panels,
        key_dir / "dust_components_inner_zoverr_vs_r.png",
        xlim=xlim_inner,
        ylim=ylim_inner,
    )
    if comparison is not None:
        made.append(comparison)

    mask_for_profile = masks[0][1] if masks else None
    profile = _plot_midplane_dust_diagnostics(
        model,
        output=key_dir / "midplane_radial_dust_diagnostics.png",
        component_density_fields=component_density_fields,
        component_labels=labels,
        component_dtg_fields=component_dtg_fields,
        mask_field=mask_for_profile,
    )
    if profile is not None:
        made.append(profile)

    if include_bin_plots:
        for bin_name, dust_bin in getattr(model.dust, "bins", {}).items():
            density = dust_bin["density"]
            comp_idx, local_idx = getattr(model.dust, "_global_bins", {}).get(bin_name, (0, bin_name))
            size_um = float(dust_bin.size.to("um").magnitude)
            label = _safe_name(labels.get(int(comp_idx), f"component_{comp_idx}").lower())
            out = bin_dir / (
                f"dust_bin_{label}_local{local_idx}_{bin_name}_{size_um:.6g}um_zoverr_vs_r.png"
            )
            plot_phi_avg_rz_slice(
                model,
                density,
                output=out,
                x_axis="r",
                y_axis="z/r",
                log10=True,
                xscale="log",
                xlim=xlim_full,
            )
            made.append(out)

    summary["plots"] = [str(path) for path in made]
    (plots_dir / "dust_component_diagnostics_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    return made


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
    closure = getattr(rad, "W_rays_closure_diagnostics", None)
    if isinstance(closure, dict):
        summary["W_rays_closure_diagnostics"] = {
            "shape": list(closure.get("shape", [])),
            "nside": closure.get("nside"),
            "npix": closure.get("npix"),
            "outer_weight_mode": closure.get("outer_weight_mode"),
            "isotropic_outside_r_cm": closure.get("isotropic_outside_r_cm"),
            "summary": jsonable(closure.get("summary", {})),
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
    _register_uv_closure_diagnostics(rad)

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
                ("chem_uv_closure_direct_to_radmc", "direct/RADMC"),
                ("chem_uv_closure_f_star", "stellar UV fraction"),
                ("chem_uv_closure_f_ext", "external UV fraction"),
                ("chem_uv_closure_f_iso", "isotropic residual fraction"),
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
        ("chem_uv_closure_direct_to_radmc", True, "magma", 4.0),
        ("chem_uv_closure_direct_excess", True, "magma", 8.0),
        ("chem_uv_closure_closure_residual", False, "coolwarm", None),
        ("chem_uv_closure_f_star", False, "plasma", None),
        ("chem_uv_closure_f_ext", False, "viridis", None),
        ("chem_uv_closure_f_iso", False, "cividis", None),
        ("chem_uv_closure_outer_weight_overridden", False, "gray", None),
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
        Merged plots are written directly in this directory. Per-segment plots
        generated by segmented RT runners are written under the sibling
        ``segments/<segment_name>`` tree.

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


def make_segmented_rt_segment_diagnostic_plots(
    rad: "RadModel",
    segment_metadata: dict[str, Any],
    plots_dir: str | Path,
) -> dict[str, Any]:
    """Create diagnostic plots for one saved segmented RADMC-3D segment.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D wrapper for the segment model after saved temperature and
        mean-intensity outputs have been loaded.
    segment_metadata : dict
        Segment metadata from the segmented RT runner.
    plots_dir : str or pathlib.Path
        Directory where segment plots and ``segment_summary.json`` are written.

    Returns
    -------
    dict
        JSON-serializable summary containing generated plots and skipped fields.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model = rad.model

    made: list[Path] = []
    skipped_fields: list[str] = []

    profile = _plot_radial_profile_panel(
        model,
        [("dust_temperature", "Tdust"), ("chi", "chi")],
        plots_dir / "radial_profiles_segment.png",
        ylabel="volume-weighted segment RT field",
        logy=True,
    )
    if profile is not None:
        made.append(profile)
    else:
        skipped_fields.extend(["dust_temperature", "chi"])

    metrics_path_value = segment_metadata.get("scout_metrics_file")
    if metrics_path_value:
        metrics_path = Path(str(metrics_path_value))
        if metrics_path.exists():
            try:
                import matplotlib.pyplot as plt

                metrics = np.load(metrics_path)
                r_au = np.asarray(metrics["r_au"], dtype=float)
                scout_metadata = segment_metadata.get("scout") or {}
                noise_tolerance = float(scout_metadata.get("noise_tolerance", 0.01))
                stellar_tolerance = float(
                    segment_metadata.get("stellar_fraction_threshold") or 0.01
                )
                product_names = sorted(
                    key.removesuffix("__shell_fractional")
                    for key in metrics.files
                    if key.endswith("__shell_fractional")
                )

                fig, axes = plt.subplots(4, 1, figsize=(8.5, 12.0), sharex=True)
                chi_mean = np.asarray(metrics["chi_broad__shell_mean"], dtype=float)
                chi_sigma = chi_mean * np.asarray(
                    metrics["chi_broad__shell_fractional"], dtype=float
                )
                axes[0].plot(r_au, chi_mean, color="black", linewidth=1.5)
                axes[0].fill_between(
                    r_au,
                    np.maximum(chi_mean - chi_sigma, np.finfo(float).tiny),
                    chi_mean + chi_sigma,
                    color="tab:blue",
                    alpha=0.3,
                )
                axes[0].set_yscale("log")
                axes[0].set_ylabel("chi broad")

                axes[1].plot(
                    r_au,
                    np.max(np.asarray(metrics["j_shell_fractional"]), axis=1),
                    color="black",
                    linewidth=1.8,
                    label="Jnu max wavelength",
                )
                for name in product_names:
                    line = axes[1].plot(
                        r_au,
                        metrics[f"{name}__shell_fractional"],
                        linewidth=1.1,
                        label=f"{name} shell mean",
                    )[0]
                    axes[1].plot(
                        r_au,
                        metrics[f"{name}__cell_fractional_p99"],
                        linewidth=0.9,
                        linestyle=":",
                        color=line.get_color(),
                        label=f"{name} cell P99",
                    )
                    axes[1].plot(
                        r_au,
                        metrics[f"{name}__cell_fractional_max"],
                        linewidth=0.7,
                        linestyle="--",
                        color=line.get_color(),
                        alpha=0.55,
                    )
                    axes[2].plot(
                        r_au,
                        metrics[f"{name}__failing_volume_fraction"],
                        linewidth=1.0,
                        label=name,
                    )
                axes[1].axhline(noise_tolerance, color="0.35", linestyle="--")
                axes[1].set_yscale("log")
                axes[1].set_ylabel("paired fractional uncertainty")
                axes[1].legend(fontsize=6.5, ncol=2)
                axes[2].set_ylabel("volume fraction above tolerance")
                axes[2].set_ylim(-0.02, 1.02)

                for name in product_names:
                    key = f"{name}__stellar_fraction"
                    if key in metrics:
                        line = axes[3].plot(r_au, metrics[key], linewidth=1.2, label=name)[0]
                        axes[3].plot(
                            r_au,
                            metrics[f"{name}__stellar_fraction_p99"],
                            color=line.get_color(),
                            linestyle=":",
                            linewidth=0.9,
                        )
                        axes[3].plot(
                            r_au,
                            metrics[f"{name}__stellar_fraction_max"],
                            color=line.get_color(),
                            linestyle="--",
                            linewidth=0.7,
                            alpha=0.55,
                        )
                axes[3].axhline(stellar_tolerance, color="0.35", linestyle="--")
                axes[3].set_xscale("log")
                axes[3].set_yscale("log")
                axes[3].set_xlabel("radius [au]")
                axes[3].set_ylabel("unattenuated stellar fraction")
                axes[3].legend(fontsize=7, ncol=2)

                split = segment_metadata.get("split") or {}
                for index_key, color in (
                    ("comparison_shell_idx", "tab:blue"),
                    ("source_shell_idx", "tab:orange"),
                ):
                    index = split.get(index_key)
                    if index is not None and 0 <= int(index) < r_au.size:
                        for axis in axes:
                            axis.axvline(r_au[int(index)], color=color, alpha=0.75)
                fig.tight_layout()
                quality_path = plots_dir / "uv_scout_quality.png"
                fig.savefig(quality_path, dpi=180)
                plt.close(fig)
                made.append(quality_path)

                join = segment_metadata.get("join") or {}
                if join:
                    labels = ["Jnu max"] + sorted((join.get("products") or {}).keys())
                    values = [float(join.get("j_fractional_max", np.nan))] + [
                        float(join["products"][name]["delta"]) for name in labels[1:]
                    ]
                    frequency_hz = np.asarray(join["frequency_hz"], dtype=float)
                    wavelength_nm = 1.0e7 * 2.99792458e10 / frequency_hz
                    order = np.argsort(wavelength_nm)
                    fig, join_axes = plt.subplots(3, 1, figsize=(8.0, 9.0))
                    join_axes[0].plot(
                        wavelength_nm[order], np.asarray(join["parent_j"])[order], label="parent"
                    )
                    join_axes[0].plot(
                        wavelength_nm[order], np.asarray(join["child_j"])[order], label="child"
                    )
                    join_axes[0].set_yscale("log")
                    join_axes[0].set_ylabel("shell mean Jnu")
                    join_axes[0].legend()
                    join_axes[1].plot(
                        wavelength_nm[order], np.asarray(join["j_fractional"])[order], label="join"
                    )
                    join_axes[1].plot(
                        wavelength_nm[order], np.asarray(join["j_sigma_rel"])[order], label="paired sigma"
                    )
                    join_axes[1].axhline(
                        float(join.get("tolerance", 0.01)), color="0.25", linestyle="--"
                    )
                    join_axes[1].set_yscale("log")
                    join_axes[1].set_xlabel("wavelength [nm]")
                    join_axes[1].set_ylabel("fractional difference")
                    join_axes[1].legend()
                    join_axes[2].bar(np.arange(len(labels)), values, color="tab:blue")
                    join_axes[2].axhline(
                        float(join.get("tolerance", 0.01)), color="0.25", linestyle="--"
                    )
                    join_axes[2].set_yscale("log")
                    join_axes[2].set_ylabel("product discrepancy")
                    join_axes[2].set_xticks(
                        np.arange(len(labels)), labels, rotation=35, ha="right"
                    )
                    fig.tight_layout()
                    join_path = plots_dir / "uv_join_diagnostics.png"
                    fig.savefig(join_path, dpi=180)
                    plt.close(fig)
                    made.append(join_path)
            except Exception as exc:
                logger.warning(
                    "Failed to plot segmented UV scout diagnostics for %s: %s",
                    segment_metadata.get("segment_name", segment_metadata.get("work_dir")),
                    exc,
                )

    for field_name, cmap, dyn_range in (
        ("density", "magma", 8.0),
        ("dust_temperature", "inferno", 3.0),
        ("chi", "viridis", 8.0),
    ):
        if field_name not in model.gas:
            skipped_fields.append(field_name)
            continue
        try:
            made.extend(
                _plot_field_maps(
                    model,
                    field_name,
                    plots_dir,
                    log10=True,
                    cmap=cmap,
                    log10_dyn_range_dex=dyn_range,
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to plot segmented RT segment field %s for %s: %s",
                field_name,
                segment_metadata.get("segment_name", segment_metadata.get("work_dir")),
                exc,
            )
            skipped_fields.append(field_name)

    summary = {
        "segment": jsonable(segment_metadata),
        "plots_dir": str(plots_dir),
        "plots": [str(path) for path in made],
        "skipped_fields": sorted(set(skipped_fields)),
        "fields_available": sorted([str(k) for k in model.gas.keys()]),
    }
    (plots_dir / "segment_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary
