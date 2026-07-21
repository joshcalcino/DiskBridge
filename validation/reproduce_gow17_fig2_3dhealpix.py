"""Reproduce Gong+17 Fig 2 with a 3-D HEALPix-shielded sphere.

Sections
--------
1. Imports + constants
2. Plotting utilities
3. Non-plotting utilities (reference loading, binning, postprocess helpers, model builders)
4. Core run functions (RADMC-3D chi; chemistry run wrapper)
5. Diagnostic and comparison helpers
6. Canonical measured validation entry point
"""
# db-keywords: healpix-columns, photodesorption, gow17, validation, units, radmc3d, model, mesh, field
# db-role: validation
# db-scope: validation
# db-purpose: Reproduce Gong+17 Fig 2 with a 3-D HEALPix-shielded sphere.

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import diskbridge
import diskbridge._gow17 as gow17_native
from diskbridge._constants import EPS_CHI, K_B
from diskbridge._units import Quantity
from diskbridge.chemistry.api import run_chemistry
from diskbridge.chemistry.models.gow17 import _KPH_AVFAC, _cv_cold
from diskbridge.model.core import Model, SubModel
from diskbridge.model.dust import Dust
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel
from diskbridge.radmc3d.utils import link_dustkappa_opacities
from diskbridge.serialization import jsonable

I_CHX = gow17_native.I_CHX
I_CO = gow17_native.I_CO
I_CO_ICE = gow17_native.I_CO_ICE
I_CP = gow17_native.I_CP
I_H2 = gow17_native.I_H2
I_H2P = gow17_native.I_H2P
I_H3P = gow17_native.I_H3P
I_HCOP = gow17_native.I_HCOP
I_HEP = gow17_native.I_HEP
I_HP = gow17_native.I_HP
I_OHX = gow17_native.I_OHX
I_OP = gow17_native.I_OP
I_SIP = gow17_native.I_SIP
I_SP = gow17_native.I_SP
I_E = gow17_native.I_E
IPH_C = gow17_native.IPH_C
IPH_CO = gow17_native.IPH_CO
IPH_H2 = gow17_native.IPH_H2
N_Y = gow17_native.N_Y
XHE = gow17_native.XHE

SPEC_LIST_REF = [
    "He+",
    "OHx",
    "CHx",
    "CO",
    "C+",
    "HCO+",
    "H2",
    "H+",
    "H3+",
    "H2+",
    "S+",
    "Si+",
    "O+",
    "E",
]

_SPEC_INDEX_REF = {name: i for i, name in enumerate(SPEC_LIST_REF)}

COLORS = {
    "CO": "#1f77b4",
    "C": "#d62728",
    "C+": "#ff7f0e",
    "H3+": "#2ca02c",
    "OHx": "#9467bd",
    "CHx": "#8c564b",
    "He+": "#e377c2",
}


# ============================================================================
# 2. Plotting utilities
# ============================================================================

def _safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log10(np.maximum(x, 1.0e-99))


def _plot_lines(
    *,
    x: np.ndarray,
    lines: List[Tuple[np.ndarray, str, str, str]],
    xlabel: str,
    ylabel: str,
    yscale: str,
    ylim: Tuple[float, float],
    output_path: Path,
    figsize: Tuple[float, float],
    dpi: int,
    ncol_legend: int,
) -> None:
    """Shared helper for line plots.

    Parameters
    ----------
    x : np.ndarray
        X-axis values.
    lines : list of (y, label, linestyle, color) tuples
        Each entry is plotted as a line.
    xlabel, ylabel : str
        Axis labels.
    yscale : str
        ``"linear"`` or ``"log"``.
    ylim : tuple of float
        Y-axis limits.
    output_path : Path
        Where to save the figure.
    figsize : tuple of float
        Figure size in inches.
    dpi : int
        Figure resolution.
    ncol_legend : int
        Number of columns in the legend.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    for y, label, ls, color in lines:
        ax.plot(x, y, lw=1.4, ls=ls, color=color, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_ylim(*ylim)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=ncol_legend)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_compare(
    *,
    outdir: Path,
    Av_ref: np.ndarray,
    Av_db: np.ndarray,
    ref: "RefSlab",
    nH_index: int,
    abd_db: Dict[str, np.ndarray],
    species: Iterable[str],
    suffix: str,
) -> None:
    """Compare DiskBridge species profiles against the 1-D reference."""
    lines: List[Tuple[np.ndarray, str, str, str]] = []
    for spec in species:
        y_ref = ref.abd[spec][:, nH_index]
        y_db = abd_db[spec].reshape(-1)
        color = COLORS.get(spec, "#444444")
        lines.append((_safe_log10(y_ref), f"{spec} ref", "-", color))
        lines.append((_safe_log10(y_db), f"{spec} diskbridge", "--", color))

    # Both ref and db share Av_ref as x (db was binned onto ref.Av)
    outdir.mkdir(parents=True, exist_ok=True)
    _plot_lines(
        x=Av_ref,
        lines=lines,
        xlabel="A_V",
        ylabel="log10 abundance per H",
        yscale="linear",
        ylim=(-14.0, 0.0),
        output_path=outdir / f"gow17_fig2_3dhealpix_{suffix}.png",
        figsize=(7.2, 4.6),
        dpi=220,
        ncol_legend=2,
    )


def _plot_convergence(
    *,
    outdir: Path,
    diag: dict,
    fname: str,
) -> Path:
    """Plot shielding iteration convergence history with budget summary."""
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d_h2 = np.asarray(diag.get("d_h2_hist", []), dtype=float)
    d_co = np.asarray(diag.get("d_co_hist", []), dtype=float)
    iters = np.arange(1, len(d_h2) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)
    if iters.size > 0:
        ax.semilogy(iters, d_h2, "o-", lw=1.2, ms=4, label="d_h2")
        ax.semilogy(iters, d_co, "s-", lw=1.2, ms=4, label="d_co")
    ax.set_xlabel("Shielding iteration")
    ax.set_ylabel("Max relative change")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    text_lines = [
        f"shielding_iter = {diag.get('shielding_iter', '?')}",
        f"h_xH_atom_min = {diag.get('h_xH_atom_min', '?'):.3e}",
        f"c_xC_neutral_min = {diag.get('c_xC_neutral_min', '?'):.3e}",
        f"h_budget_violation = {diag.get('h_budget_violation', '?'):.3e}",
        f"c_budget_violation = {diag.get('c_budget_violation', '?'):.3e}",
    ]
    ax.text(
        0.98, 0.98, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=7, verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
    )

    fig.tight_layout()
    p = outdir / fname
    fig.savefig(p)
    plt.close(fig)
    return p


def _plot_sphere_vs_slab(
    *,
    outdir: Path,
    sphere: Dict[str, np.ndarray],
    slab: Dict[str, np.ndarray],
    fname: str,
) -> Path:
    """Plot abundance and gas-temperature sphere-versus-slab profiles."""
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, (ax_abd, ax_temp) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        dpi=220,
        sharex=True,
        constrained_layout=True,
    )
    for key, label, color in [
        ("xH2", "H2", "#1f77b4"),
        ("xCO", "CO", "#d62728"),
        ("xCplus", "C+", "#ff7f0e"),
    ]:
        ax_abd.plot(
            sphere["Av"],
            _safe_log10(sphere[key]),
            color=color,
            label=f"{label} sphere",
        )
        ax_abd.plot(
            slab["Av"],
            _safe_log10(slab[key]),
            color=color,
            linestyle="--",
            label=f"{label} slab",
        )
    ax_abd.set_ylabel("log10 abundance per H")
    ax_abd.set_ylim(-14.0, 0.0)
    ax_abd.legend(ncol=2, fontsize=8)
    ax_abd.grid(alpha=0.25)

    ax_temp.plot(sphere["Av"], sphere["Tgas"], label="sphere")
    ax_temp.plot(slab["Av"], slab["Tgas"], linestyle="--", label="slab")
    ax_temp.set_xlabel(r"Physical perpendicular $A_V$")
    ax_temp.set_ylabel(r"$T_{\rm gas}$ [K]")
    ax_temp.set_yscale("log")
    ax_temp.legend(fontsize=8)
    ax_temp.grid(alpha=0.25)

    output_path = outdir / fname
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_sphere_vs_slab_radiation(
    *,
    outdir: Path,
    sphere: Dict[str, np.ndarray],
    slab: Dict[str, np.ndarray],
    fname: str,
) -> Path:
    """Plot the separate radiation and molecular-shielding terms."""
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 9.0),
        dpi=220,
        sharex=True,
        constrained_layout=True,
    )
    species = (
        ("H2", r"H$_2$", "#1f77b4"),
        ("CO", "CO", "#d62728"),
        ("C", "C", "#ff7f0e"),
    )

    for profile, geometry, linestyle, linewidth in (
        (sphere, "sphere", "-", 2.0),
        (slab, "slab", "--", 1.25),
    ):
        axes[0].plot(
            profile["Av"],
            np.maximum(profile["chi_input"], 1.0e-99),
            color="#333333",
            linestyle=linestyle,
            linewidth=linewidth,
            label=rf"$\chi$ {geometry}",
        )
        for key, label, color in species:
            axes[0].plot(
                profile["Av"],
                np.maximum(profile[f"G_{key}_dust"], 1.0e-99),
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"{label} dust-only {geometry}",
            )
            axes[1].plot(
                profile["Av"],
                np.maximum(profile[f"theta_{key}"], 1.0e-99),
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"{label} {geometry}",
            )
            axes[2].plot(
                profile["Av"],
                np.maximum(profile[f"G_{key}_actual"], 1.0e-99),
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"{label} {geometry}",
            )

    axes[0].set_ylabel("input / dust-only field")
    axes[1].set_ylabel(r"molecular shielding $\Theta$")
    axes[2].set_ylabel("field used by chemistry")
    axes[2].set_xlabel(r"Physical perpendicular $A_V$")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    output_path = outdir / fname
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_mode_comparison(
    *,
    outdir: Path,
    sphere1: Dict[str, np.ndarray],
    sphere2: Dict[str, np.ndarray],
    label1: str,
    label2: str,
    fname: str,
) -> Path:
    """Overlay shell-averaged profiles from two modes."""
    lines: List[Tuple[np.ndarray, str, str, str]] = []
    for key, label_sp, color in [
        ("xH2", "H2", "#1f77b4"),
        ("xCO", "CO", "#d62728"),
        ("xCplus", "C+", "#ff7f0e"),
    ]:
        lines.append((_safe_log10(sphere1[key]), f"{label_sp} {label1}", "-", color))
        lines.append((_safe_log10(sphere2[key]), f"{label_sp} {label2}", "--", color))

    outdir.mkdir(parents=True, exist_ok=True)
    _plot_lines(
        x=sphere1["Av"],
        lines=lines,
        xlabel="A_V",
        ylabel="log10 abundance per H",
        yscale="linear",
        ylim=(-14.0, 0.0),
        output_path=outdir / fname,
        figsize=(7.2, 4.6),
        dpi=220,
        ncol_legend=2,
    )
    return outdir / fname


def _plot_compare_fixed_vs_astrochem_species(
    *,
    outdir: Path,
    fixed: Dict[str, np.ndarray],
    astro: Dict[str, np.ndarray],
    fname: str,
) -> Path:
    """Overlay shell-averaged species profiles from fixed-point and astrochem modes."""
    lines: List[Tuple[np.ndarray, str, str, str]] = []
    for key, label, color in [
        ("xH2", "H2", "#1f77b4"),
        ("xCO", "CO", "#d62728"),
        ("xCplus", "C+", "#ff7f0e"),
    ]:
        lines.append((_safe_log10(fixed[key]), f"{label} fixed_point", "-", color))
        lines.append((_safe_log10(astro[key]), f"{label} astrochem", "--", color))

    outdir.mkdir(parents=True, exist_ok=True)
    _plot_lines(
        x=fixed["Av"],
        lines=lines,
        xlabel="A_V",
        ylabel="log10 abundance per H",
        yscale="linear",
        ylim=(-14.0, 0.0),
        output_path=outdir / fname,
        figsize=(7.2, 4.6),
        dpi=220,
        ncol_legend=2,
    )
    return outdir / fname


def _plot_compare_fixed_vs_astrochem_convergence(
    *,
    outdir: Path,
    fname: str,
    i_values: List[int],
    rel_err_H2: List[float],
    rel_err_CO: List[float],
    rel_err_Cplus: List[float],
    rel_err_max: List[float],
    rel_err_H2_vs_fp: List[float],
    rel_err_CO_vs_fp: List[float],
    rel_err_Cplus_vs_fp: List[float],
    rel_err_max_vs_fp: List[float],
) -> Path:
    """Plot astrochem convergence sweep: vs N_max and vs fixed-point."""
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), dpi=220)
    iv = np.asarray(i_values, dtype=float)

    # Panel 1: convergence within astrochem (vs N_max)
    ax = axes[0]
    ax.semilogy(iv, rel_err_H2, "o-", lw=1.2, ms=5, label="H2")
    ax.semilogy(iv, rel_err_CO, "s-", lw=1.2, ms=5, label="CO")
    ax.semilogy(iv, rel_err_Cplus, "^-", lw=1.2, ms=5, label="C+")
    ax.semilogy(iv, rel_err_max, "D--", lw=1.6, ms=6, color="k", label="max")
    ax.set_xlabel("astrochem_n_updates (N)")
    ax.set_ylabel("Relative error vs astrochem(N_max)")
    ax.set_title("Convergence within astrochem")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    # Panel 2: astrochem(N) vs fixed-point
    ax = axes[1]
    ax.semilogy(iv, rel_err_H2_vs_fp, "o-", lw=1.2, ms=5, label="H2")
    ax.semilogy(iv, rel_err_CO_vs_fp, "s-", lw=1.2, ms=5, label="CO")
    ax.semilogy(iv, rel_err_Cplus_vs_fp, "^-", lw=1.2, ms=5, label="C+")
    ax.semilogy(iv, rel_err_max_vs_fp, "D--", lw=1.6, ms=6, color="k", label="max")
    ax.set_xlabel("astrochem_n_updates (N)")
    ax.set_ylabel("Relative error vs fixed-point")
    ax.set_title("Astrochem(N) vs fixed-point")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    p = outdir / fname
    fig.savefig(p)
    plt.close(fig)
    return p


def _make_sweep_gifs_from_data(
    sweep_dir: Path,
    radm_list: List[RadModel],
    y_out_list: List[np.ndarray],
    i_values: List[int],
    duration_ms: int = 800,
) -> None:
    """Build per-figure-type GIF animations directly from sweep data.

    No intermediate PNG files are written.  Requires Pillow.
    """
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    fig_types = [
        "diag_fields_midplane",
        "diag_species_midplane",
        "diag_fields_meridional",
        "diag_radial_chi_av",
        "diag_radial_temperatures",
        "diag_radial_species",
    ]

    frames: Dict[str, List[Image.Image]] = {ft: [] for ft in fig_types}

    N = len(i_values)
    for idx, (radm, y_out, i) in enumerate(zip(radm_list, y_out_list, i_values)):
        figs = _plot_diagnostics(
            outdir=sweep_dir,
            suffix=f"astrochem_i_{i}",
            radm=radm,
            y_out=y_out,
            return_figs=True,
            frame_label=f"{idx + 1}/{N}",
        )
        assert figs is not None
        for ft in fig_types:
            buf = io.BytesIO()
            figs[ft].savefig(buf, format="png")
            plt.close(figs[ft])
            buf.seek(0)
            frames[ft].append(Image.open(buf).copy())

    for ft in fig_types:
        if len(frames[ft]) < 2:
            continue
        gif_path = sweep_dir / f"{ft}_sweep.gif"
        frames[ft][0].save(
            gif_path,
            save_all=True,
            append_images=frames[ft][1:],
            loop=0,
            duration=duration_ms,
        )
        print(f"Saved GIF: {gif_path.name}")


def _plot_diagnostics(
    *,
    outdir: Path,
    suffix: str,
    radm: RadModel,
    y_out: np.ndarray,
    return_figs: bool = False,
    frame_label: Optional[str] = None,
) -> Optional[Dict[str, "plt.Figure"]]:
    """Plot 2-D diagnostic slices and radial profiles.

    When *return_figs* is True the figures are returned in a dict keyed by
    figure-type name and are *not* saved to disk or closed.  When False
    (default) they are saved to *outdir* and closed.

    *frame_label* (e.g. ``"3/9"``) is stamped in the bottom-right corner of
    every figure when provided.
    """
    import matplotlib.pyplot as plt

    mesh = radm.model.mesh
    if mesh is None:
        raise ValueError("radm.model.mesh is None")

    if mesh.coord_system != "spherical":
        raise ValueError(f"Diagnostics currently only support spherical meshes, got {mesh.coord_system}")

    r_au = mesh.centers("r").to("au").magnitude
    r_edges_au = mesh.edges("r").to("au").magnitude
    depth_au = float(r_edges_au[-1]) - r_au
    depth_max_au = float(r_edges_au[-1] - r_edges_au[0])
    depth_order = np.argsort(depth_au)
    theta_deg = mesh.centers("theta").to("degree").magnitude
    phi_deg = mesh.centers("phi").to("degree").magnitude

    itheta_mid = int(np.argmin(np.abs(theta_deg - 90.0)))
    iphi0 = 0

    def log10_field(arr: np.ndarray) -> np.ndarray:
        return _safe_log10(np.asarray(arr, dtype=float))

    def _stamp_label(fig: "plt.Figure") -> None:
        if frame_label is not None:
            fig.text(0.98, 0.02, frame_label, transform=fig.transFigure,
                     ha="right", va="bottom", fontsize=10, color="0.4")

    chi = np.asarray(radm.chi.to("dimensionless").magnitude, dtype=float)
    av = np.asarray(radm.Av_perp.to("dimensionless").magnitude, dtype=float)
    tgas = np.asarray(radm.gas_temperature.to("K").magnitude, dtype=float)
    tdust = np.asarray(radm.dust_temperature.to("K").magnitude, dtype=float)

    species_fields: Dict[str, np.ndarray] = {
        "H2": y_out[..., I_H2],
        "CO": y_out[..., I_CO],
        "C+": y_out[..., I_CP],
        "He+": y_out[..., I_HEP],
        "HCO+": y_out[..., I_HCOP],
    }

    if not return_figs:
        outdir.mkdir(parents=True, exist_ok=True)

    figs: Dict[str, "plt.Figure"] = {}

    fields_main: Dict[str, Tuple[np.ndarray, str, str]] = {
        "log10_chi": (log10_field(chi), "r-phi @ midplane", "log10 chi"),
        "Av": (av, "r-phi @ midplane", "Av"),
        "Tgas": (tgas, "r-phi @ midplane", "Tgas [K]"),
        "Tdust": (tdust, "r-phi @ midplane", "Tdust [K]"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=220)
    axes = np.asarray(axes)

    for ax, (name, (field, _title, cbar_label)) in zip(axes.reshape(-1), fields_main.items()):
        sl = field[::-1, itheta_mid, :]
        vmin, vmax = np.nanpercentile(sl, [1.0, 99.0])
        im = ax.imshow(
            sl,
            origin="lower",
            aspect="auto",
            extent=[float(phi_deg.min()), float(phi_deg.max()), 0.0, depth_max_au],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(name)
        ax.set_xlabel("phi [deg]")
        ax.set_ylabel("Depth from surface [au]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    fig.tight_layout()
    _stamp_label(fig)
    if return_figs:
        figs["diag_fields_midplane"] = fig
    else:
        fig.savefig(outdir / f"diag_fields_midplane_{suffix}.png")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), dpi=220)
    axes = np.asarray(axes).reshape(-1)

    for ax, (name, field) in zip(axes, species_fields.items()):
        sl = log10_field(field[::-1, itheta_mid, :])
        vmin, vmax = np.nanpercentile(sl, [1.0, 99.0])
        im = ax.imshow(
            sl,
            origin="lower",
            aspect="auto",
            extent=[float(phi_deg.min()), float(phi_deg.max()), 0.0, depth_max_au],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(f"log10 {name}")
        ax.set_xlabel("phi [deg]")
        ax.set_ylabel("Depth from surface [au]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(species_fields):]:
        ax.axis("off")

    fig.tight_layout()
    _stamp_label(fig)
    if return_figs:
        figs["diag_species_midplane"] = fig
    else:
        fig.savefig(outdir / f"diag_species_midplane_{suffix}.png")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=220)
    axes = np.asarray(axes)

    for ax, (name, (field, _title, cbar_label)) in zip(axes.reshape(-1), fields_main.items()):
        sl = field[::-1, :, iphi0]
        vmin, vmax = np.nanpercentile(sl, [1.0, 99.0])
        im = ax.imshow(
            sl,
            origin="lower",
            aspect="auto",
            extent=[float(theta_deg.min()), float(theta_deg.max()), 0.0, depth_max_au],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(f"{name} (phi=0)")
        ax.set_xlabel("theta [deg]")
        ax.set_ylabel("Depth from surface [au]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    fig.tight_layout()
    _stamp_label(fig)
    if return_figs:
        figs["diag_fields_meridional"] = fig
    else:
        fig.savefig(outdir / f"diag_fields_meridional_{suffix}.png")
        plt.close(fig)

    r_edges_cm = mesh.edges("r").to("cm").magnitude
    theta_edges = mesh.edges("theta").to("radian").magnitude
    phi_edges = mesh.edges("phi").to("radian").magnitude

    vr = (r_edges_cm[1:] ** 3 - r_edges_cm[:-1] ** 3) / 3.0
    vth = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    vph = phi_edges[1:] - phi_edges[:-1]
    vol = vr[:, None, None] * vth[None, :, None] * vph[None, None, :]
    vol_r = np.sum(vol, axis=(1, 2))

    def radial_avg(field: np.ndarray) -> np.ndarray:
        f = np.asarray(field, dtype=float)
        return np.sum(f * vol, axis=(1, 2)) / vol_r

    prof = {
        "chi": radial_avg(chi),
        "Av": radial_avg(av),
        "Tgas": radial_avg(tgas),
        "Tdust": radial_avg(tdust),
        "H2": radial_avg(species_fields["H2"]),
        "CO": radial_avg(species_fields["CO"]),
        "C+": radial_avg(species_fields["C+"]),
    }

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)
    ax.plot(
        depth_au[depth_order],
        log10_field(prof["chi"])[depth_order],
        lw=1.2,
        label="log10 chi",
    )
    ax.plot(depth_au[depth_order], prof["Av"][depth_order], lw=1.2, label="Av")
    ax.set_xlabel("Depth from surface [au]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _stamp_label(fig)
    if return_figs:
        figs["diag_radial_chi_av"] = fig
    else:
        fig.savefig(outdir / f"diag_radial_chi_av_{suffix}.png")
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)
    ax.plot(depth_au[depth_order], prof["Tgas"][depth_order], lw=1.2, label="Tgas")
    ax.plot(depth_au[depth_order], prof["Tdust"][depth_order], lw=1.2, label="Tdust")
    ax.set_xlabel("Depth from surface [au]")
    ax.set_ylabel("T [K]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _stamp_label(fig)
    if return_figs:
        figs["diag_radial_temperatures"] = fig
    else:
        fig.savefig(outdir / f"diag_radial_temperatures_{suffix}.png")
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)
    for name in ["H2", "CO", "C+"]:
        ax.plot(
            depth_au[depth_order],
            log10_field(prof[name])[depth_order],
            lw=1.2,
            label=f"log10 {name}",
        )
    ax.set_xlabel("Depth from surface [au]")
    ax.set_ylabel("log10 abundance per H")
    ax.set_ylim(-14.0, 0.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _stamp_label(fig)
    if return_figs:
        figs["diag_radial_species"] = fig
        return figs
    else:
        fig.savefig(outdir / f"diag_radial_species_{suffix}.png")
        plt.close(fig)
    return None


# ============================================================================
# 3. Non-plotting utilities
# ============================================================================

@dataclass(frozen=True)
class RefSlab:
    NH: np.ndarray
    Av: np.ndarray
    nH_values: np.ndarray
    abd: Dict[str, np.ndarray]
    E: np.ndarray


def _load_reference(dir_out: Path) -> RefSlab:
    nH_values = np.loadtxt(dir_out / "nH_arr.dat")
    NH = np.loadtxt(dir_out / "colH_arr.dat")
    Av = NH / 1.87e21

    abd: Dict[str, np.ndarray] = {}
    E_list = []
    for islab, _nH in enumerate(np.atleast_1d(nH_values)):
        slab = np.loadtxt(dir_out / f"slab{islab:06d}.dat")
        for spec, j in _SPEC_INDEX_REF.items():
            abd.setdefault(spec, []).append(slab[:, j])
        E_list.append(slab[:, _SPEC_INDEX_REF["E"]])

    for spec in list(abd):
        abd[spec] = np.stack(abd[spec], axis=1)

    E = np.stack(E_list, axis=1)

    xC = 1.6e-4
    abd["C"] = xC - abd["CHx"] - abd["CO"] - abd["C+"] - abd["HCO+"]

    return RefSlab(NH=NH, Av=Av, nH_values=np.asarray(nH_values, dtype=float), abd=abd, E=E)


def _xe_from_abundances(abd: Dict[str, np.ndarray]) -> np.ndarray:
    xe = np.zeros_like(abd["H2"], dtype=float)
    for spec in ["He+", "C+", "HCO+", "H3+", "H2+", "H+", "Si+", "S+", "O+"]:
        if spec in abd:
            xe = xe + np.asarray(abd[spec], dtype=float)
    return xe


def _temperature_from_energy(
    *,
    E: np.ndarray,
    xH2: np.ndarray,
    xe: np.ndarray,
) -> np.ndarray:
    Cv_cold = _cv_cold(np.asarray(xH2, dtype=float), np.asarray(xe, dtype=float))
    return np.asarray(E, dtype=float) / Cv_cold


def _electron_abundance_from_state(y: np.ndarray) -> np.ndarray:
    """Return the GOW17 electron abundance implied by charged species."""
    y = np.asarray(y, dtype=float)
    return (
        y[..., I_HEP]
        + y[..., I_CP]
        + y[..., I_HCOP]
        + y[..., I_HP]
        + y[..., I_H3P]
        + y[..., I_H2P]
        + y[..., I_SP]
        + y[..., I_SIP]
        + y[..., I_OP]
    )


def _av_bin_edges(av: np.ndarray) -> np.ndarray:
    av = np.asarray(av, dtype=float)
    if av.ndim != 1 or av.size < 2:
        raise ValueError("av must be a 1D array with at least 2 points")
    edges = np.empty(av.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (av[:-1] + av[1:])
    edges[0] = av[0]
    edges[-1] = av[-1] + 0.5 * (av[-1] - av[-2])
    return edges


def _mean_profile_vs_av(
    *,
    av_centers: np.ndarray,
    av_samples: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    av_centers = np.asarray(av_centers, dtype=float)
    edges = _av_bin_edges(av_centers)

    av_samples = np.asarray(av_samples, dtype=float)
    values = np.asarray(values, dtype=float)
    if av_samples.shape != values.shape:
        raise ValueError("av_samples and values must have the same shape")

    valid = (
        np.isfinite(av_samples)
        & np.isfinite(values)
        & (av_samples >= edges[0])
        & (av_samples <= edges[-1])
    )
    idx = np.digitize(av_samples[valid], edges) - 1
    idx = np.clip(idx, 0, av_centers.size - 1)

    sums = np.bincount(idx, weights=values[valid], minlength=av_centers.size)
    counts = np.bincount(idx, minlength=av_centers.size)

    out = np.full(av_centers.size, np.nan, dtype=float)
    m = counts > 0
    out[m] = sums[m] / counts[m]

    if not np.any(m):
        raise ValueError("No samples fell into any Av bin")

    if not np.all(m):
        x = av_centers[m]
        y = out[m]
        out = np.interp(av_centers, x, y, left=float(y[0]), right=float(y[-1]))
    return out


def _chi_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite values in chi array")
    p1, p50, p99 = np.percentile(x, [1.0, 50.0, 99.0])
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p1": float(p1),
        "p50": float(p50),
        "p99": float(p99),
    }


def _print_chi_stats(label: str, chi: Quantity, axis_order: Tuple[str, ...]) -> None:
    chi_arr = np.asarray(chi.to("dimensionless").magnitude, dtype=float)
    stats_all = _chi_stats(chi_arr)
    r_ax = int(axis_order.index("r"))
    chi_inner = np.take(chi_arr, 0, axis=r_ax)
    chi_outer = np.take(chi_arr, chi_arr.shape[r_ax] - 1, axis=r_ax)
    stats_inner = _chi_stats(chi_inner)
    stats_outer = _chi_stats(chi_outer)

    def _fmt(s: Dict[str, float]) -> str:
        return (
            f"min={s['min']:.6g}, max={s['max']:.6g}, "
            f"p1={s['p1']:.6g}, p50={s['p50']:.6g}, p99={s['p99']:.6g}"
        )

    print(f"chi stats ({label}) all: {_fmt(stats_all)}")
    print(f"chi stats ({label}) r_inner: {_fmt(stats_inner)}")
    print(f"chi stats ({label}) r_outer: {_fmt(stats_outer)}")


def _build_spherical_cloud_model(
    *,
    nH_cm3: float,
    r_edges_cm: np.ndarray,
    ntheta: int,
    nphi: int,
    model_dir: str | Path,
) -> RadModel:
    units = diskbridge.units
    m_H = units("m_H")

    r_edges_cm = np.asarray(r_edges_cm, dtype=float)
    if r_edges_cm.ndim != 1 or r_edges_cm.size < 2:
        raise ValueError("r_edges_cm must be a one-dimensional array of edges")
    if not np.all(np.isfinite(r_edges_cm)) or not np.all(np.diff(r_edges_cm) > 0.0):
        raise ValueError("r_edges_cm must contain finite, strictly increasing edges")
    if not (r_edges_cm[0] > 0.0):
        raise ValueError("the innermost spherical radius must be positive")

    ntheta = int(ntheta)
    nphi = int(nphi)
    if ntheta <= 0:
        raise ValueError(f"ntheta must be >= 1, got {ntheta}")
    if nphi <= 0:
        raise ValueError(f"nphi must be >= 1, got {nphi}")

    th_edges = np.linspace(0.0, np.pi, ntheta + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, nphi + 1)

    r_axis = Axis(edges=Quantity(r_edges_cm, "cm"))
    theta_axis = Axis(edges=Quantity(th_edges, "rad"))
    phi_axis = Axis(edges=Quantity(ph_edges, "rad"))

    mesh = Mesh.spherical(r=r_axis, theta=theta_axis, phi=phi_axis)
    shape = mesh.shape
    axis_order = mesh.axis_names()

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    nH_field = np.full(shape, float(nH_cm3), dtype=float)
    rho = (Quantity(nH_field, "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register(
        "density",
        Field(quantity="density", data=rho, axis_order=axis_order),
    )

    sigma = Quantity(np.full(shape, 1.0e-21, dtype=float), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=axis_order),
    )
    model.gas_register(
        "microturbulence",
        Field(
            quantity="microturbulence",
            data=Quantity(np.full(shape, 0.3, dtype=float), "km/s"),
            axis_order=axis_order,
            attrs={
                "mode": "constant",
                "spatially_constant": True,
                "value": "0.3 km/s",
            },
        ),
    )
    for velocity_name in ("vr", "vtheta", "vphi"):
        model.gas_register(
            velocity_name,
            Field(
                quantity=velocity_name,
                data=Quantity(np.zeros(shape, dtype=float), "cm/s"),
                axis_order=axis_order,
            ),
        )

    return RadModel(model, model_dir=model_dir)


def _sphere_radial_edges(
    *,
    nH_cm3: float,
    comparison_NH_cm2: float,
    center_Av: float,
    comparison_cells: int,
    core_cells: int,
) -> np.ndarray:
    """Return radial edges for the comparison layer and optional opaque core."""
    nH_cm3 = float(nH_cm3)
    comparison_NH_cm2 = float(comparison_NH_cm2)
    center_Av = float(center_Av)
    comparison_cells = int(comparison_cells)
    core_cells = int(core_cells)

    if not (nH_cm3 > 0.0):
        raise ValueError("nH_cm3 must be positive")
    if not (comparison_NH_cm2 > 0.0):
        raise ValueError("comparison_NH_cm2 must be positive")
    if comparison_cells < 1 or core_cells < 0:
        raise ValueError("comparison_cells must be positive and core_cells non-negative")

    center_NH_cm2 = center_Av * 1.87e21
    if core_cells == 0:
        if not np.isclose(center_NH_cm2, comparison_NH_cm2, rtol=1.0e-12):
            raise ValueError(
                "center_Av must equal the comparison depth when core_cells is zero"
            )
        radius_cm = comparison_NH_cm2 / nH_cm3
        inner_radius_cm = 0.5 * radius_cm / comparison_cells
        edges = np.linspace(
            inner_radius_cm,
            radius_cm,
            comparison_cells + 1,
            dtype=float,
        )
        if not np.all(np.diff(edges) > 0.0):
            raise RuntimeError("constructed spherical radial edges are not increasing")
        return edges

    if not (center_NH_cm2 > comparison_NH_cm2):
        raise ValueError(
            "center_Av must place an opaque core beyond the comparison column"
        )

    radius_cm = center_NH_cm2 / nH_cm3
    comparison_depth_cm = comparison_NH_cm2 / nH_cm3
    core_outer_radius_cm = radius_cm - comparison_depth_cm

    # Keep the coordinate singularity outside the mesh, following the previous
    # spherical validation, while making the omitted central radius half of one
    # coarse core-cell width.
    core_dr_cm = core_outer_radius_cm / core_cells
    inner_radius_cm = 0.5 * core_dr_cm
    core_edges = np.linspace(
        inner_radius_cm,
        core_outer_radius_cm,
        core_cells + 1,
        dtype=float,
    )
    comparison_edges = np.linspace(
        core_outer_radius_cm,
        radius_cm,
        comparison_cells + 1,
        dtype=float,
    )
    edges = np.concatenate((core_edges, comparison_edges[1:]))
    if not np.all(np.diff(edges) > 0.0):
        raise RuntimeError("constructed spherical radial edges are not increasing")
    return edges


def _compute_chi_with_radmc3d(
    *,
    radm: RadModel,
    model_dir: Path,
    chi0: float,
    isrf_path: Path,
    dust_nbins: int,
    vacuum_dust: bool,
    nphot_mono: int,
    nphot_thermal: int,
    nphot_scat: int,
    scat_mode: int,
    nbcores: int,
    force: bool,
    uv_min: Quantity,
    uv_max: Quantity,
    uv_n_wavelengths: int,
    baseline_radmc3d_inputs_dir: Path,
) -> Tuple[Quantity, Quantity]:
    model_dir = Path(model_dir)
    inputs_dir = model_dir / "radmc3d_inputs"
    outputs_dir = model_dir / "radmc3d_outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    lambda_min_um = float(diskbridge.params.lambda_min.to("micron").magnitude)
    uv_min_um = float(uv_min.to("micron").magnitude)
    lambda_min_new_um = min(lambda_min_um, uv_min_um)

    params_path = model_dir / "params.txt"
    params_path.write_text(
        "\n".join(
            [
                "external_uv = T",
                f"external_uv_chi = {float(chi0)}",
                f"nbcores = {int(nbcores)}",
                f"nphot_thermal = {int(nphot_thermal)}",
                f"nphot_scat = {int(nphot_scat)}",
                f"nphot_mono = {int(nphot_mono)}",
                f"scat_mode = {int(scat_mode)}",
                f"lambda_min = {lambda_min_new_um}",
                f"uv_min = {float(uv_min.to('nm').magnitude)}",
                f"uv_max = {float(uv_max.to('nm').magnitude)}",
                f"uv_n_wavelengths = {int(uv_n_wavelengths)}",
            ]
        )
        + "\n"
    )

    params_local = diskbridge.read_params(params_path)
    radm.params = params_local
    radm.writer.params = params_local

    if radm.model.dust is None:
        radm.model.dust = Dust(radm.model)

    d2g = params_local.dust_to_gas_ratio
    if isinstance(d2g, list):
        raise ValueError("dust_to_gas_ratio must be scalar for this validation run")
    dust_to_gas_ratio = float(d2g)
    if vacuum_dust:
        dust_to_gas_ratio = 0.0

    radm.model.dust.set_distribution(
        nbin=int(dust_nbins),
        dust_to_gas_ratio=float(dust_to_gas_ratio),
        mode="proportional",
    )

    radm.writer.write_amr_grid(model_dir)
    radm.writer.write_wavelength_grid(model_dir)
    wavelength_lines = (inputs_dir / "wavelength_micron.inp").read_text().splitlines()
    n_wavelengths = int(wavelength_lines[0])
    stellar_lines = ["2", f"0 {n_wavelengths}", *wavelength_lines[1:]]
    (inputs_dir / "stars.inp").write_text("\n".join(stellar_lines) + "\n")
    radm.writer.write_radmc3d_inp(
        model_dir,
        scattering_mode_max=int(scat_mode),
        nphot=int(nphot_thermal),
        nphot_mono=int(nphot_mono),
        nphot_scat=int(nphot_scat),
        setthreads=int(nbcores),
    )
    radm.writer.write_dust_density(model_dir, binary=True)
    radm.writer.write_dustopac(model_dir, scattering_mode=int(scat_mode))

    link_dustkappa_opacities(
        src_inputs_dir=Path(baseline_radmc3d_inputs_dir),
        dest_inputs_dir=inputs_dir,
    )

    radm.writer.write_external_source(
        model_dir,
        chi=float(chi0),
        isrf_path=Path(isrf_path),
    )

    radm.compute_temperature(
        nphot=int(nphot_thermal),
        output_dir=outputs_dir / "temperature",
        force=bool(force),
    )
    _, dust_temperature = radm.compute_gas_dust_surface_area_coupling()

    chi = radm.ensure_chi(
        force=bool(force),
        uv_min=Quantity(float(uv_min.to("nm").magnitude), "nm"),
        uv_max=Quantity(float(uv_max.to("nm").magnitude), "nm"),
        n_wavelengths=int(uv_n_wavelengths),
    )
    return chi, dust_temperature


def _gow17_reference_solver_controls() -> dict:
    """Return solver controls used by the bundled external GOW17 benchmark."""
    return {
        "tmax": "6.32e16 s",
        "tmin": "3.16e12 s",
        "reltol": 1.0e-2,
        "abstol0": 1.0e-9,
        "mxsteps": 5_000_000,
        "maxord": 3,
        "tolfac": 10.0,
        "gradv": 9.0e-14,
        "NCOeff_global": True,
        "bCO_L": True,
        "Leff_CO_max": 3.0e20,
        "tolerances": {
            "abstol_default": 1.0e-9,
            "abstol_Heplus": 1.0e-15,
            "abstol_OHx": 1.0e-15,
            "abstol_CHx": 1.0e-15,
            "abstol_CO": 1.0e-15,
            "abstol_CO_ice": 1.0e-9,
            "abstol_Cplus": 1.0e-15,
            "abstol_HCOplus": 1.0e-30,
            "abstol_H2": 1.0e-8,
            "abstol_Hplus": 1.0e-15,
            "abstol_H3plus": 1.0e-15,
            "abstol_H2plus": 1.0e-15,
            "abstol_Splus": 1.0e-9,
            "abstol_Siplus": 1.0e-9,
            "abstol_Oplus": 1.0e-9,
        },
        "dust_cooling": {
            "mode": "gow17_original",
            "sigma_d_H_ref": "1.0e-21 cm^2",
            "gow17_original_Zd": 1.0,
            "gow17_original_Tdust": "10 K",
        },
    }


def _save_chemistry_healpix_state(
    *,
    outdir: str | Path,
    y_out: np.ndarray,
    Av_perp: np.ndarray,
    radm: RadModel,
    chem_result,
    gow17_cfg: dict,
) -> None:
    """Save the complete GOW17 state and retained shielding products."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "gow17_y": np.asarray(y_out),
        "Av_perp": np.asarray(Av_perp),
        "Av_attenuation": np.asarray(radm.Av.to("dimensionless").magnitude),
        "chi": np.asarray(radm.chi.to("dimensionless").magnitude),
        "r_edges_cm": np.asarray(radm.model.mesh.edges("r").to("cm").magnitude),
        "theta_edges_rad": np.asarray(
            radm.model.mesh.edges("theta").to("rad").magnitude
        ),
        "phi_edges_rad": np.asarray(
            radm.model.mesh.edges("phi").to("rad").magnitude
        ),
    }
    units = {
        "gow17_y": "dimensionless state except energy component (erg per H)",
        "Av_perp": "dimensionless",
        "Av_attenuation": "dimensionless",
        "chi": "dimensionless",
        "r_edges_cm": "cm",
        "theta_edges_rad": "rad",
        "phi_edges_rad": "rad",
    }

    retained_fields = (
        "Tgas",
        "status",
        "Tgas_status",
        "b_H2_kms",
        "b_CO_kms",
        "NCOeff_CO_cooling",
        "theta_h2",
        "theta_co",
        "theta_c",
        "chi_eff",
        "G_CO_diss_actual",
        "G_C_ion_actual",
        "G_H2_diss_actual",
    )
    for name in retained_fields:
        quantity = chem_result.fields.get(name)
        if quantity is None:
            continue
        arrays[name] = np.asarray(quantity.magnitude)
        units[name] = str(quantity.units)

    for name in ("nco_gas", "nH2", "nH_atom", "nCplus", "nC", "ne"):
        quantity = getattr(radm, name, None)
        if quantity is None:
            continue
        arrays[name] = np.asarray(quantity.magnitude)
        units[name] = str(quantity.units)

    np.savez(outdir / "chemistry_healpix_state.npz", **arrays)
    state_indices = {
        "He+": I_HEP,
        "OHx": I_OHX,
        "CHx": I_CHX,
        "CO": I_CO,
        "CO_ice": I_CO_ICE,
        "C+": I_CP,
        "HCO+": I_HCOP,
        "H2": I_H2,
        "H+": I_HP,
        "H3+": I_H3P,
        "H2+": I_H2P,
        "S+": I_SP,
        "Si+": I_SIP,
        "O+": I_OP,
        "E": I_E,
    }
    metadata = {
        "shape": [int(v) for v in y_out.shape[:-1]],
        "axis_order": list(radm.model.mesh.axis_names()),
        "gow17_state_indices": state_indices,
        "units": units,
        "configuration": gow17_cfg,
        "gow17_diagnostics": chem_result.meta.get("gow17_diagnostics", {}),
        "directional_ray_arrays_retained": False,
        "directional_ray_note": (
            "Per-direction HEALPix columns are streamed in chunks and are not "
            "retained by the production chemistry result. The saved arrays are "
            "the final direction-averaged shielding factors and equivalent CO "
            "cooling column used by the solve."
        ),
    }
    (outdir / "chemistry_healpix_state.json").write_text(
        json.dumps(jsonable(metadata), indent=2, sort_keys=True) + "\n"
    )


def run_gow17_internal_3d_healpix_sphere(
    *,
    nH_cm3: float,
    chi0: float,
    xi_cr: float,
    shielding_max_iter: int,
    const_temp: bool,
    nside: int,
    comparison_nr: int,
    core_nr: int,
    sphere_center_Av: float,
    ntheta: int,
    nphi: int,
    ref: RefSlab,
    nH_index: int,
    use_radmc3d_chi: bool,
    radmc3d_model_dir: str | Path,
    radmc3d_force: bool,
    radmc3d_nphot_mono: int,
    radmc3d_nphot_thermal: int,
    radmc3d_nphot_scat: int,
    radmc3d_scat_mode: int,
    radmc3d_dust_nbins: int,
    radmc3d_vacuum_dust: bool,
    radmc3d_isrf_path: Path,
    radmc3d_uv_min: Quantity,
    radmc3d_uv_max: Quantity,
    radmc3d_uv_n_wavelengths: int,
    baseline_radmc3d_inputs_dir: Path,
    diagnostic_outdir: str | Path,
    diagnostic_suffix: str,
    coupling_mode: str,
    astrochem_n_updates: int,
    astrochem_t_end_yr: float,
    skip_diagnostics: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict, np.ndarray, RadModel]:

    NH_ref = np.asarray(ref.NH, dtype=float)
    comparison_NH_max = float(np.max(NH_ref))
    comparison_Av_max = comparison_NH_max / 1.87e21
    r_edges_cm = _sphere_radial_edges(
        nH_cm3=nH_cm3,
        comparison_NH_cm2=comparison_NH_max,
        center_Av=sphere_center_Av,
        comparison_cells=comparison_nr,
        core_cells=core_nr,
    )
    R_cm = float(r_edges_cm[-1])

    radm = _build_spherical_cloud_model(
        nH_cm3=nH_cm3,
        r_edges_cm=r_edges_cm,
        ntheta=int(ntheta),
        nphi=int(nphi),
        model_dir=Path(radmc3d_model_dir) if use_radmc3d_chi else ".",
    )

    shape = radm.model.mesh.shape
    r_cent = radm.model.mesh.centers("r").to("cm").magnitude
    r3 = np.broadcast_to(r_cent[:, None, None], shape)

    chi0_incident = 2.0 * float(chi0)

    depth_cm = np.maximum(float(R_cm) - r3, 0.0)
    inside = r3 <= float(R_cm)
    NH_depth = float(nH_cm3) * depth_cm
    Av_3d = NH_depth / 1.87e21

    nH_index = int(nH_index)
    if nH_index < 0 or nH_index >= int(ref.nH_values.size):
        raise ValueError(f"nH_index out of range: {nH_index}")

    xe_ref = _xe_from_abundances(ref.abd)
    T_ref = _temperature_from_energy(E=ref.E, xH2=ref.abd["H2"], xe=xe_ref)

    Av_flat = Av_3d.reshape(-1)
    T_line = np.asarray(T_ref[:, nH_index], dtype=float)
    T_init_flat = np.interp(Av_flat, ref.Av, T_line, left=T_line[0], right=T_line[-1])
    T_init = T_init_flat.reshape(shape)

    radm.gas_temperature = Quantity(T_init, "K")
    radm.dust_temperature = Quantity(T_init, "K")

    # -- Av: perpendicular depth with Gong+17 Appendix 3D-slab factor (x2) --
    Av_perp = NH_depth / 1.87e21
    radm.Av_perp = Quantity(Av_perp, "dimensionless")
    radm.Av = Quantity(2.0 * Av_perp, "dimensionless")

    if use_radmc3d_chi:
        chi_rt, dust_temperature_rt = _compute_chi_with_radmc3d(
            radm=radm,
            model_dir=Path(radmc3d_model_dir),
            chi0=float(chi0_incident),
            isrf_path=Path(radmc3d_isrf_path),
            dust_nbins=int(radmc3d_dust_nbins),
            vacuum_dust=bool(radmc3d_vacuum_dust),
            nphot_mono=int(radmc3d_nphot_mono),
            nphot_thermal=int(radmc3d_nphot_thermal),
            nphot_scat=int(radmc3d_nphot_scat),
            scat_mode=int(radmc3d_scat_mode),
            nbcores=int(diskbridge.params.nbcores) if isinstance(diskbridge.params.nbcores, int) else int(diskbridge.params.nbcores),
            force=bool(radmc3d_force),
            uv_min=radmc3d_uv_min,
            uv_max=radmc3d_uv_max,
            uv_n_wavelengths=int(radmc3d_uv_n_wavelengths),
            baseline_radmc3d_inputs_dir=Path(baseline_radmc3d_inputs_dir),
        )
        radm.chi = chi_rt.to("dimensionless")
        radm.dust_temperature = dust_temperature_rt.to("K")
        radm.radiation_mode = "local_chi"
        # RADMC-3D needs a dust submodel to transport the UV field, whereas the
        # baseline GOW17 chemistry scales H2 formation by Zd. Drop only the
        # transport-only dust object after chi is cached so the analytic and
        # RADMC sphere calculations retain the same original-GOW17 chemistry.
        radm.model.dust = None
    else:
        # Incident (unattenuated) field everywhere; dust attenuation
        # is handled inside the chemistry from the model's incident slab mode.
        radm.set_incident_uv(
            chi=Quantity(np.full(shape, 0.5 * chi0_incident, dtype=float), "dimensionless"),
            Av=radm.Av,
        )

    chi_label = "radmc3d" if use_radmc3d_chi else "analytic"
    _print_chi_stats(chi_label, radm.chi, tuple(radm.model.mesh.axis_names()))

    y0 = np.zeros(N_Y, dtype=float)
    y0[I_HEP] = 1.450654e-08
    y0[I_H3P] = 2.681411e-07
    y0[I_CP] = 1.0e-4
    y0[I_CO] = 1.0e-7
    y0[I_H2] = 0.1
    y0[I_CO_ICE] = 0.0

    y_init = np.zeros(shape + (N_Y,), dtype=float)
    for j in range(N_Y):
        y_init[..., j] = y0[j]

    def _interp(spec: str) -> np.ndarray:
        prof = np.asarray(ref.abd[spec][:, nH_index], dtype=float)
        return np.interp(Av_flat, ref.Av, prof, left=prof[0], right=prof[-1]).reshape(shape)

    y_init[..., I_HEP] = np.where(inside, _interp("He+"), y_init[..., I_HEP])
    y_init[..., I_OHX] = np.where(inside, _interp("OHx"), y_init[..., I_OHX])
    y_init[..., I_CHX] = np.where(inside, _interp("CHx"), y_init[..., I_CHX])
    y_init[..., I_CO] = np.where(inside, _interp("CO"), y_init[..., I_CO])
    y_init[..., I_CP] = np.where(inside, _interp("C+"), y_init[..., I_CP])
    y_init[..., I_HCOP] = np.where(inside, _interp("HCO+"), y_init[..., I_HCOP])
    y_init[..., I_H2] = np.where(inside, _interp("H2"), y_init[..., I_H2])
    y_init[..., I_HP] = np.where(inside, _interp("H+"), y_init[..., I_HP])
    y_init[..., I_H3P] = np.where(inside, _interp("H3+"), y_init[..., I_H3P])
    y_init[..., I_H2P] = np.where(inside, _interp("H2+"), y_init[..., I_H2P])
    y_init[..., I_SP] = np.where(inside, _interp("S+"), y_init[..., I_SP])
    y_init[..., I_SIP] = np.where(inside, _interp("Si+"), y_init[..., I_SIP])
    y_init[..., I_OP] = np.where(inside, _interp("O+"), y_init[..., I_OP])
    y_init[..., I_CO_ICE] = 0.0

    # Energy species: E = Cv * T, matching _cv_cold in gow17.py
    xH2_init = y_init[..., I_H2]
    xe_init = (
        y_init[..., I_HEP]
        + y_init[..., I_CP]
        + y_init[..., I_HCOP]
        + y_init[..., I_HP]
        + y_init[..., I_H3P]
        + y_init[..., I_H2P]
        + y_init[..., I_SP]
        + y_init[..., I_SIP]
        + y_init[..., I_OP]
    )
    Cv_init = _cv_cold(np.asarray(xH2_init, dtype=float), np.asarray(xe_init, dtype=float))
    y_init[..., I_E] = Cv_init * np.asarray(T_init, dtype=float)

    radm.gow17_y = np.ascontiguousarray(y_init, dtype=np.float64)

    gow17_cfg = {
        "mode": "equilibrium",
        **_gow17_reference_solver_controls(),
        "temperature": {"mode": "dust" if bool(const_temp) else "computed"},
        "nside": int(nside),
        "chi0": float(chi0_incident),
        "ion_rate": f"{float(xi_cr)} 1/s",
        "b_kms": 0.3,
        "Zg": 1.0,
        "Zd": 1.0,
        "fH2gr": 1.0,
        "fHplusgr": 0.6,
        "fCplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "fCplusCR": 1.0,
        "shielding_outer_coupling": "pseudotime",
        "shielding_max_iter": int(shielding_max_iter),
        "shielding_reltol": 1.0e-3,
        "shielding_abstol": 1.0e-20,
        "enable_co_phase": False,
        "co_cooling": {"method": "healpix_hybrid_lvg"},
        "coupling_mode": str(coupling_mode),
        "astrochem_n_updates": int(astrochem_n_updates),
        "astrochem_t_end_yr": float(astrochem_t_end_yr),
    }

    chem_result = run_chemistry(radm, model="gow17", config=gow17_cfg)

    y_out = np.asarray(radm.gow17_y, dtype=float)
    Av_out = np.asarray(radm.Av.to("dimensionless").magnitude, dtype=float)
    _save_chemistry_healpix_state(
        outdir=diagnostic_outdir,
        y_out=y_out,
        Av_perp=Av_3d,
        radm=radm,
        chem_result=chem_result,
        gow17_cfg=gow17_cfg,
    )

    if not skip_diagnostics:
        outdir = Path(diagnostic_outdir)
        _plot_diagnostics(
            outdir=outdir,
            suffix=str(diagnostic_suffix),
            radm=radm,
            y_out=y_out,
        )

    # Compare profiles on the physical perpendicular depth coordinate. The
    # doubled GOW17 attenuation depth remains internal to ``radm.Av``.
    Av_samp = Av_3d.reshape(-1)
    abd: Dict[str, np.ndarray] = {}
    for name, idx in [
        ("He+", I_HEP),
        ("OHx", I_OHX),
        ("CHx", I_CHX),
        ("CO", I_CO),
        ("C+", I_CP),
        ("HCO+", I_HCOP),
        ("H2", I_H2),
        ("H+", I_HP),
        ("H3+", I_H3P),
        ("H2+", I_H2P),
        ("S+", I_SP),
        ("Si+", I_SIP),
        ("O+", I_OP),
    ]:
        abd[name] = _mean_profile_vs_av(
            av_centers=ref.Av,
            av_samples=Av_samp,
            values=y_out[..., idx].reshape(-1),
        )

    abd["E"] = (
        abd["He+"]
        + abd["C+"]
        + abd["HCO+"]
        + abd["H+"]
        + abd["H3+"]
        + abd["H2+"]
        + abd["S+"]
        + abd["Si+"]
        + abd["O+"]
    )

    xC = 1.6e-4
    abd["C"] = xC - abd["CHx"] - abd["CO"] - abd["C+"] - abd["HCO+"]

    gow17_diag = chem_result.meta.get("gow17_diagnostics", {})

    info = {
        "shielding_mode": "gow17_internal_3d_healpix_sphere",
        "nside": int(nside),
        "shape": tuple(int(s) for s in shape),
        "R_cm": float(R_cm),
        "sphere_center_Av": float(sphere_center_Av),
        "comparison_Av_max": float(comparison_Av_max),
        "comparison_NH_max_cm2": float(comparison_NH_max),
        "opaque_core_Av": float(sphere_center_Av - comparison_Av_max),
        "nH_index": int(nH_index),
        "nr": int(shape[0]),
        "comparison_nr": int(comparison_nr),
        "core_nr": int(core_nr),
        "ntheta": int(ntheta),
        "nphi": int(nphi),
        "co_phase_enabled": False,
        "use_radmc3d_chi": bool(use_radmc3d_chi),
        "gow17_diagnostics": gow17_diag,
    }

    return (
        np.asarray(y_out, dtype=float),
        abd,
        np.asarray(ref.Av, dtype=float),
        info,
        np.asarray(Av_3d, dtype=float),
        radm,
    )


def _save_convergence_json(
    outdir: Path,
    diag: dict,
    fname: str,
) -> Path:
    """Save convergence diagnostics dict to JSON, converting numpy arrays."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in diag.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    p = outdir / fname
    p.write_text(json.dumps(out, indent=2) + "\n")
    return p


def _shell_average_sphere(
    *,
    y_out: np.ndarray,
    Av_3d: np.ndarray,
    nbins: int,
    comparison_Av_max: float,
) -> Dict[str, np.ndarray]:
    """Shell-average abundances over the GOW17 comparison layer.

    Bins cells by their ``Av_3d`` value (which is a monotonic function
    of depth from the sphere surface), restricts the requested comparison
    interval, and computes the mean abundance in each bin.

    Parameters
    ----------
    y_out : np.ndarray
        GOW17 abundance array, shape ``(nr, ntheta, nphi, N_Y)``.
    Av_3d : np.ndarray
        Visual extinction array, shape ``(nr, ntheta, nphi)``.
    nbins : int
        Number of Av bins.
    comparison_Av_max : float
        Maximum perpendicular visual extinction included in the comparison.

    Returns
    -------
    dict
        Keys: ``Av``, ``xH2``, ``xCO``, ``xCplus``, and ``Tgas``, with 1-D
        arrays of length *nbins*.
    """
    av_flat = Av_3d.ravel()
    comparison_Av_max = float(comparison_Av_max)
    if not (comparison_Av_max > 0.0):
        raise ValueError("comparison_Av_max must be positive")
    comparison_mask = (
        np.isfinite(av_flat)
        & (av_flat >= 0.0)
        & (av_flat <= comparison_Av_max * (1.0 + 1.0e-12))
    )
    if not np.any(comparison_mask):
        raise ValueError("no sphere cells lie inside the requested comparison layer")

    av_compare = av_flat[comparison_mask]
    bin_edges = np.linspace(0.0, comparison_Av_max, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    idx = np.digitize(av_compare, bin_edges) - 1
    idx = np.clip(idx, 0, nbins - 1)

    def _bin_mean(field_flat: np.ndarray) -> np.ndarray:
        field_compare = np.asarray(field_flat, dtype=float)[comparison_mask]
        sums = np.bincount(idx, weights=field_compare, minlength=nbins).astype(float)
        counts = np.bincount(idx, minlength=nbins).astype(float)
        out = np.full(nbins, np.nan, dtype=float)
        m = counts > 0
        out[m] = sums[m] / counts[m]
        return out

    xH2_prof = _bin_mean(y_out[..., I_H2].ravel())
    xCO_prof = _bin_mean(y_out[..., I_CO].ravel())
    xCplus_prof = _bin_mean(y_out[..., I_CP].ravel())
    Tgas = _temperature_from_energy(
        E=y_out[..., I_E],
        xH2=y_out[..., I_H2],
        xe=_electron_abundance_from_state(y_out),
    )
    Tgas_prof = _bin_mean(Tgas.ravel())

    return {
        "Av": bin_centers,
        "xH2": xH2_prof,
        "xCO": xCO_prof,
        "xCplus": xCplus_prof,
        "Tgas": Tgas_prof,
    }


def _save_sphere_profile(
    outdir: Path,
    profile: Dict[str, np.ndarray],
    fname: str,
) -> Path:
    """Save shell-averaged profile to CSV."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / fname
    header = "Av,xH2,xCO,xCplus,Tgas_K"
    data = np.column_stack([
        profile["Av"],
        profile["xH2"],
        profile["xCO"],
        profile["xCplus"],
        profile["Tgas"],
    ])
    np.savetxt(p, data, delimiter=",", header=header, comments="")
    return p


def _run_slab_reference(
    *,
    Av_ref: np.ndarray,
    comparison_Av_max: float,
    nH_cm3: float,
    chi0: float,
    xi_cr: float,
    const_temp: bool,
    nside: int,
) -> Dict[str, np.ndarray]:
    """Run a 1-D slab reference with the GOW17 angular factors.

    Profiles and the native beamed slab chemistry both use the physical
    perpendicular ``Av`` coordinate. The sphere-side curvature approximation
    is not applied a second time to this published GOW17 slab reference.
    Shielding iteration settings are independent of the 3-D sphere run.

    Parameters
    ----------
    Av_ref : np.ndarray
        Physical perpendicular visual-extinction values.
    comparison_Av_max : float
        Full physical comparison depth, including the outer edge beyond the
        deepest sphere bin centre.
    nH_cm3 : float
        Hydrogen number density [cm^-3].
    chi0 : float
        Draine-field chi_0 parameter.
    xi_cr : float
        Cosmic-ray ionisation rate [1/s].

    Returns
    -------
    dict
        Keys: ``Av``, ``xH2``, ``xCO``, ``xCplus``.
    """
    shielding_max_iter = 200
    units = diskbridge.units
    m_H = units("m_H")
    chi0_incident = 2.0 * float(chi0)

    # Build a dense log-spaced Av grid for the internal calculation.
    # The sphere's 64 shell-average bins are too coarse and linearly spaced,
    # giving almost no resolution near the surface where PDR transitions occur.
    # After running, results are interpolated back to Av_ref for plotting.
    # Match the published GOW17 benchmark resolution. The native solver
    # accumulates molecular shielding columns sequentially, so a substantially
    # coarser grid changes the H/H2 transition rather than merely resampling it.
    N_slab = 2000
    Av_max = float(comparison_Av_max)
    Av_lo = 1.0e17 / 1.87e21
    Av_perp_dense = np.logspace(np.log10(Av_lo), np.log10(Av_max), N_slab)

    NH_flat = Av_perp_dense * 1.87e21

    dr = np.empty(N_slab, dtype=float)
    dr[0] = NH_flat[0] / float(nH_cm3) if nH_cm3 > 0 else 1.0
    dr[1:] = np.diff(NH_flat) / float(nH_cm3)
    dr = np.maximum(dr, 1.0)

    r_edges = np.zeros(N_slab + 1, dtype=float)
    r_edges[1:] = np.cumsum(dr)

    r_axis = Axis(edges=Quantity(r_edges, "cm"))
    theta_axis = Axis(edges=Quantity(np.array([0.0, np.pi]), "rad"))
    phi_axis = Axis(edges=Quantity(np.array([0.0, 2.0 * np.pi]), "rad"))

    mesh = Mesh.spherical(r=r_axis, theta=theta_axis, phi=phi_axis)
    shape = mesh.shape
    axis_order = mesh.axis_names()

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    nH_field = np.full(shape, float(nH_cm3), dtype=float)
    rho = (Quantity(nH_field, "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=axis_order))

    sigma = Quantity(np.full(shape, 1.0e-21, dtype=float), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=axis_order),
    )
    model.gas_register(
        "microturbulence",
        Field(
            quantity="microturbulence",
            data=Quantity(np.full(shape, 0.3, dtype=float), "km/s"),
            axis_order=axis_order,
            attrs={
                "mode": "constant",
                "spatially_constant": True,
                "value": "0.3 km/s",
            },
        ),
    )

    radm = RadModel(model, model_dir=".")

    radm.Av = Quantity(Av_perp_dense.reshape(shape), "dimensionless")
    radm.set_incident_uv(
        # The native slab solver internally constructs G0 = 2 * chi and then
        # applies its one-sided factor of 1/2. Supplying chi0 here therefore
        # matches the generic 3D GOW17 path's unattenuated field.
        chi=Quantity(
            np.full(shape, 0.5 * chi0_incident, dtype=float),
            "dimensionless",
        ),
        Av=radm.Av,
    )
    radm.gas_temperature = Quantity(np.full(shape, 100.0, dtype=float), "K")
    radm.dust_temperature = Quantity(np.full(shape, 100.0, dtype=float), "K")

    y0 = np.zeros(N_Y, dtype=float)
    y0[I_HEP] = 1.450654e-08
    y0[I_H3P] = 2.681411e-07
    y0[I_CP] = 1.0e-4
    y0[I_CO] = 1.0e-7
    y0[I_H2] = 0.1
    y0[I_CO_ICE] = 0.0

    y_init = np.zeros(shape + (N_Y,), dtype=float)
    for j in range(N_Y):
        y_init[..., j] = y0[j]
    radm.gow17_y = np.ascontiguousarray(y_init, dtype=np.float64)

    gow17_cfg = {
        "mode": "equilibrium",
        **_gow17_reference_solver_controls(),
        "temperature": {"mode": "dust" if bool(const_temp) else "computed"},
        "nside": int(nside),
        "chi0": float(chi0_incident),
        "NH_min": f"{float(np.min(NH_flat))} cm^-2",
        "NH_total": f"{float(np.max(NH_flat))} cm^-2",
        "logNH": True,
        "field_geo": 0,
        "ion_rate": f"{float(xi_cr)} 1/s",
        "b_kms": 0.3,
        "Zg": 1.0,
        "Zd": 1.0,
        "fH2gr": 1.0,
        "fHplusgr": 0.6,
        "fCplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "fCplusCR": 1.0,
        "shielding_max_iter": int(shielding_max_iter),
        "shielding_reltol": 1.0e-3,
        "shielding_abstol": 1.0e-20,
        "enable_co_phase": False,
    }

    run_chemistry(radm, model="gow17_slab_equilibrium", config=gow17_cfg)

    y_slab = np.asarray(radm.gow17_y, dtype=float)
    Av_out = np.asarray(Av_perp_dense, dtype=float)
    Tgas_slab = np.asarray(radm.gas_temperature.to("K").magnitude, dtype=float).ravel()

    # Interpolate dense-grid results back onto the sphere's Av bins for plotting.
    Av_ref_arr = np.asarray(Av_ref, dtype=float)
    return {
        "Av": Av_ref_arr,
        "xH2": np.interp(Av_ref_arr, Av_out, y_slab[..., I_H2].ravel()),
        "xCO": np.interp(Av_ref_arr, Av_out, y_slab[..., I_CO].ravel()),
        "xCplus": np.interp(Av_ref_arr, Av_out, y_slab[..., I_CP].ravel()),
        "Tgas": np.interp(Av_ref_arr, Av_out, Tgas_slab),
    }


def _postprocess_run(
    *,
    outdir: Path,
    y_out: np.ndarray,
    info: dict,
    Av_3d: np.ndarray,
    nH_cm3: float,
    chi0: float,
    xi_cr: float,
    const_temp: bool,
    nside: int,
    suffix: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Common post-processing: save convergence, shell-avg, slab ref, overlay.

    Returns the shell-averaged sphere and native 1D slab profile dictionaries.
    """
    diag = info.get("gow17_diagnostics", {})

    conv_json_name = f"gow17_convergence{suffix}.json"
    conv_plot_name = f"gow17_convergence{suffix}.png"
    sphere_csv_name = f"sphere_profile{suffix}.csv"
    overlay_name = f"3d_sphere{suffix}_vs_1d_slab_species_profiles.png"

    _save_convergence_json(outdir, diag, fname=conv_json_name)
    _plot_convergence(outdir=outdir, diag=diag, fname=conv_plot_name)

    sphere = _shell_average_sphere(
        y_out=y_out,
        Av_3d=Av_3d,
        nbins=64,
        comparison_Av_max=float(info["comparison_Av_max"]),
    )
    _save_sphere_profile(outdir, sphere, fname=sphere_csv_name)

    slab = _run_slab_reference(
        Av_ref=sphere["Av"],
        comparison_Av_max=float(info["comparison_Av_max"]),
        nH_cm3=nH_cm3,
        chi0=chi0,
        xi_cr=xi_cr,
        const_temp=const_temp,
        nside=nside,
    )
    _plot_sphere_vs_slab(
        outdir=outdir,
        sphere=sphere,
        slab=slab,
        fname=overlay_name,
    )

    print(f"  Saved: {conv_json_name}, {conv_plot_name}, {sphere_csv_name}, {overlay_name}")
    return sphere, slab


def _profile_comparison_stats(
    sphere: Dict[str, np.ndarray],
    slab: Dict[str, np.ndarray],
) -> dict:
    """Return descriptive sphere-versus-slab profile differences."""
    stats = {}
    for key in ("xH2", "xCO", "xCplus"):
        sphere_values = np.asarray(sphere[key], dtype=float)
        slab_values = np.asarray(slab[key], dtype=float)
        valid = np.isfinite(sphere_values) & np.isfinite(slab_values)
        delta = np.abs(
            _safe_log10(sphere_values[valid]) - _safe_log10(slab_values[valid])
        )
        stats[key] = {
            "n_bins": int(delta.size),
            "median_abs_log10_difference": float(np.median(delta)),
            "p90_abs_log10_difference": float(np.percentile(delta, 90.0)),
            "max_abs_log10_difference": float(np.max(delta)),
        }

    sphere_temperature = np.asarray(sphere["Tgas"], dtype=float)
    slab_temperature = np.asarray(slab["Tgas"], dtype=float)
    valid = (
        np.isfinite(sphere_temperature)
        & np.isfinite(slab_temperature)
        & (slab_temperature > 0.0)
    )
    fractional = np.abs(
        sphere_temperature[valid] - slab_temperature[valid]
    ) / slab_temperature[valid]
    stats["Tgas"] = {
        "n_bins": int(fractional.size),
        "median_abs_fractional_difference": float(np.median(fractional)),
        "p90_abs_fractional_difference": float(np.percentile(fractional, 90.0)),
        "max_abs_fractional_difference": float(np.max(fractional)),
    }
    return stats


def _radial_solid_angle_mean(radm: RadModel, values: np.ndarray) -> np.ndarray:
    """Return a solid-angle-weighted mean for every radial shell."""
    arr = np.asarray(values, dtype=float)
    if arr.shape != tuple(radm.model.mesh.shape):
        raise ValueError(
            f"field shape {arr.shape} does not match mesh {radm.model.mesh.shape}"
        )
    theta_edges = radm.model.mesh.edges("theta").to("rad").magnitude
    phi_edges = radm.model.mesh.edges("phi").to("rad").magnitude
    theta_weights = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    phi_weights = np.diff(phi_edges)
    weights = theta_weights[:, None] * phi_weights[None, :]
    return np.sum(arr * weights[None, :, :], axis=(1, 2)) / np.sum(weights)


def _field_on_mesh(values: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Return either a full mesh field or a broadcast radial profile."""
    arr = np.asarray(values, dtype=float)
    if arr.shape == tuple(shape):
        return arr.copy()
    if arr.shape == (int(shape[0]),):
        return np.broadcast_to(arr[:, None, None], shape).copy()
    raise ValueError(
        f"expected full field shape {shape} or radial shape {(shape[0],)}, got {arr.shape}"
    )


def _initial_gow17_state(
    *,
    ref: RefSlab,
    nH_index: int,
    Av_perp: np.ndarray,
    temperature_K: np.ndarray,
) -> np.ndarray:
    """Interpolate the bundled slab state onto a validation mesh."""
    shape = tuple(np.asarray(Av_perp).shape)
    av_flat = np.asarray(Av_perp, dtype=float).reshape(-1)
    y_init = np.zeros(shape + (N_Y,), dtype=float)

    defaults = {
        I_HEP: 1.450654e-08,
        I_H3P: 2.681411e-07,
        I_CP: 1.0e-4,
        I_CO: 1.0e-7,
        I_H2: 0.1,
        I_CO_ICE: 0.0,
    }
    for index, value in defaults.items():
        y_init[..., index] = value

    for name, index in (
        ("He+", I_HEP),
        ("OHx", I_OHX),
        ("CHx", I_CHX),
        ("CO", I_CO),
        ("C+", I_CP),
        ("HCO+", I_HCOP),
        ("H2", I_H2),
        ("H+", I_HP),
        ("H3+", I_H3P),
        ("H2+", I_H2P),
        ("S+", I_SP),
        ("Si+", I_SIP),
        ("O+", I_OP),
    ):
        line = np.asarray(ref.abd[name][:, nH_index], dtype=float)
        y_init[..., index] = np.interp(
            av_flat,
            ref.Av,
            line,
            left=float(line[0]),
            right=float(line[-1]),
        ).reshape(shape)
    y_init[..., I_CO_ICE] = 0.0

    xe = _electron_abundance_from_state(y_init)
    y_init[..., I_E] = _cv_cold(y_init[..., I_H2], xe) * np.asarray(
        temperature_K, dtype=float
    )
    return np.ascontiguousarray(y_init, dtype=np.float64)


def _radial_chemistry_profile(
    *, radm: RadModel, y_out: np.ndarray, Av_perp: np.ndarray
) -> Dict[str, np.ndarray]:
    """Return an outer-to-inner radial chemistry profile."""
    temperature = np.asarray(radm.gas_temperature.to("K").magnitude, dtype=float)
    profile = {
        "Av": _radial_solid_angle_mean(radm, Av_perp),
        "xH2": _radial_solid_angle_mean(radm, y_out[..., I_H2]),
        "xCO": _radial_solid_angle_mean(radm, y_out[..., I_CO]),
        "xCplus": _radial_solid_angle_mean(radm, y_out[..., I_CP]),
        "Tgas": _radial_solid_angle_mean(radm, temperature),
    }
    order = np.argsort(profile["Av"])
    return {name: np.asarray(values)[order] for name, values in profile.items()}


def _radial_radiation_profile(
    *,
    radm: RadModel,
    chem_result,
    Av_perp: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return shell profiles of each factor entering the photochemical field."""
    Av_arr = np.asarray(Av_perp, dtype=float)
    Av_profile = _radial_solid_angle_mean(radm, Av_arr)
    order = np.argsort(Av_profile)
    profile = {
        "chi_input": _radial_solid_angle_mean(
            radm,
            np.asarray(radm.chi.to("dimensionless").magnitude, dtype=float),
        )[order]
    }
    uses_slab_dust = str(getattr(radm, "radiation_mode", "")).startswith(
        "incident_slab"
    )
    field_specs = (
        ("H2", "G_H2_diss", "G_H2_diss_actual", "theta_h2", IPH_H2),
        ("CO", "G_CO_diss", "G_CO_diss_actual", "theta_co", IPH_CO),
        ("C", "G_C_ion", "G_C_ion_actual", "theta_c", IPH_C),
    )
    for key, base_name, actual_name, theta_name, photo_index in field_specs:
        base = np.asarray(chem_result.fields[base_name].magnitude, dtype=float)
        dust = base
        if uses_slab_dust:
            dust = base * np.exp(-float(_KPH_AVFAC[photo_index]) * Av_arr)
        profile[f"G_{key}_dust"] = _radial_solid_angle_mean(radm, dust)[order]
        profile[f"theta_{key}"] = _radial_solid_angle_mean(
            radm,
            np.asarray(chem_result.fields[theta_name].magnitude, dtype=float),
        )[order]
        profile[f"G_{key}_actual"] = _radial_solid_angle_mean(
            radm,
            np.asarray(chem_result.fields[actual_name].magnitude, dtype=float),
        )[order]
    return profile


def _run_controlled_geometry(
    *,
    label: str,
    outdir: Path,
    r_edges_cm: np.ndarray,
    ntheta: int,
    nphi: int,
    nH_cm3: float,
    chi0: float,
    xi_cr: float,
    nside: int,
    shielding_max_iter: int,
    temperature_values_K: np.ndarray,
    radiation_mode: str,
    chi_values: Optional[np.ndarray],
    ref: RefSlab,
    nH_index: int,
    save_healpix_state: bool,
) -> Tuple[Dict[str, np.ndarray], dict, RadModel]:
    """Run one fixed-temperature GOW17 geometry with supplied radial fields."""
    radm = _build_spherical_cloud_model(
        nH_cm3=nH_cm3,
        r_edges_cm=r_edges_cm,
        ntheta=ntheta,
        nphi=nphi,
        model_dir=outdir,
    )
    shape = tuple(radm.model.mesh.shape)
    radius_cm = float(r_edges_cm[-1])
    r_centers_cm = radm.model.mesh.centers("r").to("cm").magnitude
    Av_radial = nH_cm3 * (radius_cm - r_centers_cm) / 1.87e21
    Av_perp = _field_on_mesh(Av_radial, shape)
    temperature = _field_on_mesh(temperature_values_K, shape)

    radm.Av_perp = Quantity(Av_perp, "dimensionless")
    radm.Av = Quantity(Av_perp, "dimensionless")
    radm.dust_temperature = Quantity(temperature, "K")
    radm.gas_temperature = Quantity(temperature, "K")

    if radiation_mode == "incident_slab_chi":
        radm.set_incident_uv(
            chi=Quantity(np.full(shape, float(chi0)), "dimensionless"),
            Av=radm.Av,
        )
    elif radiation_mode == "local_chi":
        if chi_values is None:
            raise ValueError("local_chi requires chi_values")
        radm.chi = Quantity(_field_on_mesh(chi_values, shape), "dimensionless")
        radm.radiation_mode = "local_chi"
    else:
        raise ValueError(f"unsupported radiation_mode={radiation_mode!r}")

    radm.gow17_y = _initial_gow17_state(
        ref=ref,
        nH_index=nH_index,
        Av_perp=Av_perp,
        temperature_K=temperature,
    )
    gow17_cfg = {
        "mode": "equilibrium",
        **_gow17_reference_solver_controls(),
        "temperature": {"mode": "dust"},
        "nside": int(nside),
        "chi0": float(chi0),
        "ion_rate": f"{float(xi_cr)} 1/s",
        "b_kms": 0.3,
        "Zg": 1.0,
        "Zd": 1.0,
        "fH2gr": 1.0,
        "fHplusgr": 0.6,
        "fCplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "fCplusCR": 1.0,
        "shielding_outer_coupling": "pseudotime",
        "shielding_outer_1d": "max",
        "shielding_ray_average": "uniform",
        "shielding_max_iter": int(shielding_max_iter),
        "shielding_reltol": 1.0e-3,
        "shielding_abstol": 1.0e-20,
        "enable_co_phase": False,
        "co_cooling": {"method": "legacy_scalar"},
        "coupling_mode": "fixed_point",
        "astrochem_n_updates": 0,
        "astrochem_t_end_yr": 1.0e6,
    }
    result = run_chemistry(radm, model="gow17", config=gow17_cfg)
    y_out = np.asarray(radm.gow17_y, dtype=float)
    diagnostics = result.meta.get("gow17_diagnostics", {})

    outdir.mkdir(parents=True, exist_ok=True)
    _save_convergence_json(outdir, diagnostics, "gow17_convergence.json")
    _plot_convergence(outdir=outdir, diag=diagnostics, fname="gow17_convergence.png")
    if save_healpix_state:
        _save_chemistry_healpix_state(
            outdir=outdir,
            y_out=y_out,
            Av_perp=Av_perp,
            radm=radm,
            chem_result=result,
            gow17_cfg=gow17_cfg,
        )
        _plot_diagnostics(
            outdir=outdir,
            suffix=label,
            radm=radm,
            y_out=y_out,
        )
    else:
        state_arrays = {
            "gow17_y": y_out,
            "Av_perp": Av_perp,
            "chi": np.asarray(radm.chi.to("dimensionless").magnitude),
            "Tdust_K": temperature,
        }
        for name in (
            "G_H2_diss",
            "G_CO_diss",
            "G_C_ion",
            "theta_h2",
            "theta_co",
            "theta_c",
            "G_H2_diss_actual",
            "G_CO_diss_actual",
            "G_C_ion_actual",
        ):
            state_arrays[name] = np.asarray(chem_result.fields[name].magnitude)
        np.savez(outdir / "chemistry_state.npz", **state_arrays)

    profile = _radial_chemistry_profile(radm=radm, y_out=y_out, Av_perp=Av_perp)
    profile.update(
        _radial_radiation_profile(
            radm=radm,
            chem_result=result,
            Av_perp=Av_perp,
        )
    )
    _save_sphere_profile(outdir, profile, "profile.csv")
    return profile, diagnostics, radm


def _run_controlled_case(
    *,
    label: str,
    outdir: Path,
    r_edges_cm: np.ndarray,
    ntheta: int,
    nphi: int,
    nH_cm3: float,
    chi0: float,
    xi_cr: float,
    nside: int,
    shielding_max_iter: int,
    sphere_temperature_K: np.ndarray,
    slab_temperature_radial_K: np.ndarray,
    sphere_radiation_mode: str,
    sphere_chi: Optional[np.ndarray],
    ref: RefSlab,
    nH_index: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """Run the 3D sphere against a slab retaining its own UV treatment."""
    sphere, sphere_diag, _ = _run_controlled_geometry(
        label=label,
        outdir=outdir / "sphere_3d",
        r_edges_cm=r_edges_cm,
        ntheta=ntheta,
        nphi=nphi,
        nH_cm3=nH_cm3,
        chi0=chi0,
        xi_cr=xi_cr,
        nside=nside,
        shielding_max_iter=shielding_max_iter,
        temperature_values_K=sphere_temperature_K,
        radiation_mode=sphere_radiation_mode,
        chi_values=sphere_chi,
        ref=ref,
        nH_index=nH_index,
        save_healpix_state=True,
    )
    slab, slab_diag, _ = _run_controlled_geometry(
        label=label,
        outdir=outdir / "slab_1d",
        r_edges_cm=r_edges_cm,
        ntheta=1,
        nphi=1,
        nH_cm3=nH_cm3,
        chi0=chi0,
        xi_cr=xi_cr,
        nside=nside,
        shielding_max_iter=shielding_max_iter,
        temperature_values_K=slab_temperature_radial_K,
        radiation_mode="incident_slab_chi",
        chi_values=None,
        ref=ref,
        nH_index=nH_index,
        save_healpix_state=False,
    )
    _plot_sphere_vs_slab(
        outdir=outdir,
        sphere=sphere,
        slab=slab,
        fname="3d_sphere_vs_1d_slab_species_profiles.png",
    )
    _plot_sphere_vs_slab_radiation(
        outdir=outdir,
        sphere=sphere,
        slab=slab,
        fname="3d_sphere_vs_1d_slab_radiation_fields.png",
    )
    return sphere, slab, {"sphere_3d": sphere_diag, "slab_1d": slab_diag}


def _max_rss_bytes(who: int) -> int:
    """Return platform-normalized maximum resident memory for one usage class."""
    value = int(resource.getrusage(who).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _git_sha(repo_root: Path) -> Optional[str]:
    """Return the current Git revision when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _physical_memory_bytes() -> Optional[int]:
    """Return host physical memory on macOS when available."""
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _write_report(output_dir: Path, summary: dict) -> None:
    """Write the concise human-readable validation report."""
    cfg = summary["configuration"]
    timing = summary["resources"]
    comparisons = summary["profile_comparisons"]
    lines = [
        "# GOW17 3D sphere versus 1D slab validation",
        "",
        "This compares a 3D HEALPix sphere with a one-dimensional outward-column "
        "GOW17 calculation in two controlled fixed-temperature cases.",
        "",
        "## Configuration",
        "",
        f"- Physical comparison range: `0 <= A_V <= {cfg['comparison_Av_max']:.4g}`",
        f"- Sphere centre: `A_V = {cfg['sphere_center_Av']:.4g}`",
        f"- Grid: `{cfg['comparison_nr']}` radial by "
        f"`{cfg['ntheta']} x {cfg['nphi']}` angular cells",
        f"- HEALPix shielding: `nside = {cfg['nside']}`",
        f"- RADMC-3D: `{cfg['nphot_mono']}` photon packets, "
        f"`{cfg['uv_n_wavelengths']}` wavelengths",
        f"- Nominal packets per cell per wavelength: "
        f"`{cfg['nominal_photon_packets_per_cell_per_wavelength']:.1f}`",
        "- Temperature: `Tgas = Tdust` in both geometries",
        "- CO cooling is bypassed by the fixed-temperature chemistry solve",
        f"- CO phase chemistry: `{cfg['co_phase_enabled']}`",
        "",
        "## Resource use",
        "",
        f"- Python wall time: `{timing['python_wall_seconds']:.1f} s`",
        f"- Python peak RSS: `{timing['python_max_rss_bytes'] / 2**30:.3f} GiB`",
        f"- Maximum child-process RSS: `{timing['children_max_rss_bytes'] / 2**30:.3f} GiB`",
        "",
        "The separately retained `/usr/bin/time -l` log is the authoritative "
        "whole-command timing and memory record for the scheduled measured run.",
        "",
        "## Profile differences",
        "",
    ]
    for case_name, case_stats in comparisons.items():
        lines.extend([f"### {case_name}", ""])
        for key in ("xH2", "xCO", "xCplus"):
            item = case_stats[key]
            lines.append(
                f"- {key}: median/p90/max absolute log10 difference = "
                f"`{item['median_abs_log10_difference']:.4g}` / "
                f"`{item['p90_abs_log10_difference']:.4g}` / "
                f"`{item['max_abs_log10_difference']:.4g}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "In `prescribed_sphere`, the slab retains its standard one-dimensional "
            "radiation treatment and that treatment is prescribed on the sphere. "
            "In `radmc_sphere`, only the sphere uses local RADMC-3D UV; the slab "
            "retains its own UV treatment and receives the sphere's angularly "
            "averaged dust temperature. Inspect both profile plots, the sphere "
            "angular diagnostics, and convergence JSON before drawing a conclusion.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    """Run the canonical sphere-versus-slab validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        help="Unique directory name under the validation runs directory.",
    )
    args = parser.parse_args()

    wall_start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    ref = _load_reference(root / "other_codes" / "pdr" / "out_example_simple")
    validation_dir = Path(__file__).resolve().parent / "reproduce_gow17_fig2_3dhealpix"
    outdir = validation_dir / "runs" / str(args.run_id)
    outdir.mkdir(parents=True, exist_ok=False)

    nH_index = 0
    nH_val = float(ref.nH_values[nH_index])
    chi0 = 1.0
    xi_cr = 2.0e-16
    shielding_max_iter = 200
    comparison_nr = 128
    core_nr = 0
    sphere_center_Av = float(np.max(ref.Av))
    ntheta = 16
    nphi = 16
    nside = 4

    radmc3d_uv_n_wavelengths = 8
    radmc3d_nphot_mono = 10_000_000
    radmc3d_nphot_thermal = 10_000_000
    radmc3d_nphot_scat = 10_000_000
    radmc3d_dust_nbins = (
        int(diskbridge.params.nbins)
        if not isinstance(diskbridge.params.nbins, list)
        else int(diskbridge.params.nbins[0])
    )
    radmc3d_scat_mode = int(diskbridge.params.scat_mode)
    radmc3d_uv_min = Quantity(91.2, "nm")
    radmc3d_uv_max = Quantity(207.0, "nm")

    if int(diskbridge.params.nside) != nside:
        raise RuntimeError(
            f"validation requires nside={nside}, got global nside={diskbridge.params.nside}"
        )

    comparison_NH_max = float(np.max(ref.NH))
    r_edges_cm = _sphere_radial_edges(
        nH_cm3=nH_val,
        comparison_NH_cm2=comparison_NH_max,
        center_Av=sphere_center_Av,
        comparison_cells=comparison_nr,
        core_cells=core_nr,
    )
    radius_cm = float(r_edges_cm[-1])
    r_centers_cm = 0.5 * (r_edges_cm[:-1] + r_edges_cm[1:])
    Av_radial = nH_val * (radius_cm - r_centers_cm) / 1.87e21
    xe_ref = _xe_from_abundances(ref.abd)
    T_ref = _temperature_from_energy(E=ref.E, xH2=ref.abd["H2"], xe=xe_ref)
    reference_temperature = np.interp(
        Av_radial,
        ref.Av,
        np.asarray(T_ref[:, nH_index], dtype=float),
    )

    print("=" * 60)
    print("Prescribed slab UV on the 3D sphere")
    print("=" * 60)
    prescribed_sphere, prescribed_slab, prescribed_diag = _run_controlled_case(
        label="prescribed_sphere",
        outdir=outdir / "prescribed_sphere",
        r_edges_cm=r_edges_cm,
        ntheta=ntheta,
        nphi=nphi,
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        nside=nside,
        shielding_max_iter=shielding_max_iter,
        sphere_temperature_K=reference_temperature,
        slab_temperature_radial_K=reference_temperature,
        sphere_radiation_mode="incident_slab_chi",
        sphere_chi=None,
        ref=ref,
        nH_index=nH_index,
    )

    print("=" * 60)
    print("RADMC-3D UV and dust temperature on the 3D sphere")
    print("=" * 60)
    radmc_model_dir = outdir / "radmc3d_model"
    transport = _build_spherical_cloud_model(
        nH_cm3=nH_val,
        r_edges_cm=r_edges_cm,
        ntheta=ntheta,
        nphi=nphi,
        model_dir=radmc_model_dir,
    )
    chi_rt, dust_temperature_rt = _compute_chi_with_radmc3d(
        radm=transport,
        model_dir=radmc_model_dir,
        chi0=2.0 * chi0,
        isrf_path=root / "data" / "ISRF.dat",
        dust_nbins=radmc3d_dust_nbins,
        vacuum_dust=False,
        nphot_mono=radmc3d_nphot_mono,
        nphot_thermal=radmc3d_nphot_thermal,
        nphot_scat=radmc3d_nphot_scat,
        scat_mode=radmc3d_scat_mode,
        nbcores=int(diskbridge.params.nbcores),
        force=False,
        uv_min=radmc3d_uv_min,
        uv_max=radmc3d_uv_max,
        uv_n_wavelengths=radmc3d_uv_n_wavelengths,
        baseline_radmc3d_inputs_dir=(
            root
            / "validation"
            / "cube_test"
            / "rho0_2.000e-20"
            / "radmc3d_inputs"
        ),
    )
    chi_rt_values = np.asarray(chi_rt.to("dimensionless").magnitude, dtype=float)
    dust_temperature_values = np.asarray(
        dust_temperature_rt.to("K").magnitude, dtype=float
    )
    chi_radial = _radial_solid_angle_mean(transport, chi_rt_values)
    dust_temperature_radial = _radial_solid_angle_mean(
        transport, dust_temperature_values
    )
    np.savetxt(
        radmc_model_dir / "shell_averaged_profiles.csv",
        np.column_stack((Av_radial, chi_radial, dust_temperature_radial)),
        delimiter=",",
        header="Av,chi,Tdust_K",
        comments="",
    )
    radmc_sphere, radmc_slab, radmc_diag = _run_controlled_case(
        label="radmc_sphere",
        outdir=outdir / "radmc_sphere",
        r_edges_cm=r_edges_cm,
        ntheta=ntheta,
        nphi=nphi,
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        nside=nside,
        shielding_max_iter=shielding_max_iter,
        sphere_temperature_K=dust_temperature_values,
        slab_temperature_radial_K=dust_temperature_radial,
        sphere_radiation_mode="local_chi",
        sphere_chi=chi_rt_values,
        ref=ref,
        nH_index=nH_index,
    )

    wall_seconds = time.perf_counter() - wall_start
    n_cells = comparison_nr * ntheta * nphi
    summary = {
        "validation": "gow17_3d_healpix_sphere_vs_1d_slab",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(root),
        "python": sys.version,
        "platform": platform.platform(),
        "configuration": {
            "nH_cm3": nH_val,
            "chi0": chi0,
            "xi_cr_s-1": xi_cr,
            "temperature_mode": "dust",
            "gas_temperature_equals_dust_temperature": True,
            "sphere_center_Av": sphere_center_Av,
            "comparison_Av_max": float(np.max(ref.Av)),
            "opaque_core_Av": 0.0,
            "comparison_nr": comparison_nr,
            "core_nr": core_nr,
            "ntheta": ntheta,
            "nphi": nphi,
            "nside": nside,
            "n_cells": n_cells,
            "co_phase_enabled": False,
            "co_cooling_method": "not_exercised_fixed_temperature",
            "microturbulence_km_s": 0.3,
            "radmc3d_threads": int(diskbridge.params.nbcores),
            "radmc3d_external_source_only": True,
            "nphot_mono": radmc3d_nphot_mono,
            "nphot_thermal": radmc3d_nphot_thermal,
            "nominal_photon_packets_per_cell_per_wavelength": (
                radmc3d_nphot_mono / n_cells
            ),
            "uv_n_wavelengths": radmc3d_uv_n_wavelengths,
            "uv_min_nm": float(radmc3d_uv_min.to("nm").magnitude),
            "uv_max_nm": float(radmc3d_uv_max.to("nm").magnitude),
            "prescribed_case_sphere_radiation": "incident_slab_chi",
            "radmc_case_sphere_radiation": "local_radmc3d_chi",
            "slab_radiation_both_cases": "incident_slab_chi",
            "radmc_case_slab_temperature": "sphere_solid_angle_mean",
        },
        "resources": {
            "python_wall_seconds": wall_seconds,
            "python_max_rss_bytes": _max_rss_bytes(resource.RUSAGE_SELF),
            "children_max_rss_bytes": _max_rss_bytes(resource.RUSAGE_CHILDREN),
            "host_physical_memory_bytes": _physical_memory_bytes(),
        },
        "profile_comparisons": {
            "prescribed_sphere": _profile_comparison_stats(
                prescribed_sphere, prescribed_slab
            ),
            "radmc_sphere": _profile_comparison_stats(radmc_sphere, radmc_slab),
        },
        "gow17_diagnostics": {
            "prescribed_sphere": jsonable(prescribed_diag),
            "radmc_sphere": jsonable(radmc_diag),
        },
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_report(outdir, summary)
    print(f"\nWrote validation run: {outdir}")
    print(f"Python wall time: {wall_seconds:.1f} s")


if __name__ == "__main__":
    main()
