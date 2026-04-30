"""Reproduce Gong+17 Fig 2 with a 3-D HEALPix-shielded sphere.

Sections
--------
1. Imports + constants
2. Plotting utilities
3. Non-plotting utilities (reference loading, binning, postprocess helpers, model builders)
4. Core run functions (RADMC-3D chi; chemistry run wrapper)
5. Convergence sweep drivers (astrochem sweep; fixed-point vs astrochem comparison)
6. main()
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import diskbridge
import diskbridge._gow17 as gow17_native
from diskbridge._constants import EPS_CHI, K_B
from diskbridge._units import Quantity
from diskbridge.chemistry.api import run_chemistry
from diskbridge.chemistry.models.gow17 import _cv_cold
from diskbridge.chemistry.shielding.columns_1d import is_effectively_1d
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
    """Overlay plot comparing sphere shell-average to 1-D slab reference."""
    lines: List[Tuple[np.ndarray, str, str, str]] = []
    for key, label, color in [
        ("xH2", "H2", "#1f77b4"),
        ("xCO", "CO", "#d62728"),
        ("xCplus", "C+", "#ff7f0e"),
    ]:
        lines.append((_safe_log10(sphere[key]), f"{label} sphere", "-", color))
        lines.append((_safe_log10(slab[key]), f"{label} slab", "--", color))

    outdir.mkdir(parents=True, exist_ok=True)
    _plot_lines(
        x=sphere["Av"],
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
    av = np.asarray(radm.Av.to("dimensionless").magnitude, dtype=float)
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
        sl = field[:, itheta_mid, :]
        vmin, vmax = np.nanpercentile(sl, [1.0, 99.0])
        im = ax.imshow(
            sl,
            origin="lower",
            aspect="auto",
            extent=[float(phi_deg.min()), float(phi_deg.max()), float(r_au.min()), float(r_au.max())],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(name)
        ax.set_xlabel("phi [deg]")
        ax.set_ylabel("r [au]")
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
        sl = log10_field(field[:, itheta_mid, :])
        vmin, vmax = np.nanpercentile(sl, [1.0, 99.0])
        im = ax.imshow(
            sl,
            origin="lower",
            aspect="auto",
            extent=[float(phi_deg.min()), float(phi_deg.max()), float(r_au.min()), float(r_au.max())],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(f"log10 {name}")
        ax.set_xlabel("phi [deg]")
        ax.set_ylabel("r [au]")
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
        sl = field[:, :, iphi0]
        vmin, vmax = np.nanpercentile(sl, [1.0, 99.0])
        im = ax.imshow(
            sl,
            origin="lower",
            aspect="auto",
            extent=[float(theta_deg.min()), float(theta_deg.max()), float(r_au.min()), float(r_au.max())],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(f"{name} (phi=0)")
        ax.set_xlabel("theta [deg]")
        ax.set_ylabel("r [au]")
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
    ax.plot(r_au, log10_field(prof["chi"]), lw=1.2, label="log10 chi")
    ax.plot(r_au, prof["Av"], lw=1.2, label="Av")
    ax.set_xlabel("r [au]")
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
    ax.plot(r_au, prof["Tgas"], lw=1.2, label="Tgas")
    ax.plot(r_au, prof["Tdust"], lw=1.2, label="Tdust")
    ax.set_xlabel("r [au]")
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
        ax.plot(r_au, log10_field(prof[name]), lw=1.2, label=f"log10 {name}")
    ax.set_xlabel("r [au]")
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

    idx = np.digitize(av_samples, edges) - 1
    idx = np.clip(idx, 0, av_centers.size - 1)

    sums = np.bincount(idx, weights=values, minlength=av_centers.size)
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
    R_cm: float,
    nr: int,
    ntheta: int,
    nphi: int,
    model_dir: str | Path,
) -> RadModel:
    units = diskbridge.units
    m_H = units("m_H")

    R_cm = float(R_cm)
    if not (R_cm > 0.0):
        raise ValueError(f"R_cm must be > 0, got {R_cm}")

    nr = int(nr)
    ntheta = int(ntheta)
    nphi = int(nphi)
    if nr <= 0:
        raise ValueError(f"nr must be >= 1, got {nr}")
    if ntheta <= 0:
        raise ValueError(f"ntheta must be >= 1, got {ntheta}")
    if nphi <= 0:
        raise ValueError(f"nphi must be >= 1, got {nphi}")

    dr_cm = float(R_cm) / float(nr)
    r_edges_cm = (np.arange(nr + 1, dtype=float) * dr_cm) + 0.5 * dr_cm
    r_edges_cm[-1] = float(R_cm)
    th_edges = np.linspace(0.0, np.pi, ntheta + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, nphi + 1)

    r_axis = Axis(edges=Quantity(r_edges_cm, "cm"))
    theta_axis = Axis(edges=Quantity(th_edges, "rad"))
    phi_axis = Axis(edges=Quantity(ph_edges, "rad"))

    mesh = Mesh.spherical(r=r_axis, theta=theta_axis, phi=phi_axis)
    shape = mesh.shape
    axis_order = mesh.axis_names()

    if is_effectively_1d(mesh, shape):
        raise RuntimeError(f"Expected non-1D mesh but got shape={shape}")

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

    return RadModel(model, model_dir=model_dir)


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
) -> Quantity:
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
                f"external_uv_chi = {float(2.0 * float(chi0))}",
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
    radm.writer.write_stars(model_dir)
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
        chi=float(2.0 * float(chi0)),
        isrf_path=Path(isrf_path),
    )

    dust_temp = getattr(radm, "dust_temperature", None)
    if dust_temp is None:
        raise RuntimeError("dust_temperature must be set on radm before running RADMC-3D mcmono")
    radm.writer.write_dust_temperature(
        dust_temp,
        output_dir=outputs_dir / "temperature",
        nspec=int(radm.model.dust.nbin),
    )

    chi = radm.ensure_chi(
        force=bool(force),
        uv_min=Quantity(float(uv_min.to("nm").magnitude), "nm"),
        uv_max=Quantity(float(uv_max.to("nm").magnitude), "nm"),
        n_wavelengths=int(uv_n_wavelengths),
    )
    return chi


def run_gow17_internal_3d_healpix_sphere(
    *,
    nH_cm3: float,
    chi0: float,
    xi_cr: float,
    shielding_max_iter: int,
    const_temp: bool,
    nside: int,
    self_weight: float,
    nr: int,
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
    NH_max = float(np.max(NH_ref))
    R_cm = NH_max / float(nH_cm3)

    radm = _build_spherical_cloud_model(
        nH_cm3=nH_cm3,
        R_cm=R_cm,
        nr=int(nr),
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
    radm.Av = Quantity(2.0 * Av_perp, "dimensionless")

    if use_radmc3d_chi:
        chi_rt = _compute_chi_with_radmc3d(
            radm=radm,
            model_dir=Path(radmc3d_model_dir),
            chi0=float(chi0),
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
        radm.chi = chi_rt
    else:
        # Incident (unattenuated) field everywhere; dust attenuation
        # is handled inside the chemistry via chi_is_incident=True.
        radm.chi = Quantity(np.full(shape, chi0_incident, dtype=float), "dimensionless")

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
        "const_temp": bool(const_temp),
        "t_end": "2.0e9 yr",
        "nside": int(nside),
        "chi0": float(chi0_incident),
        "ion_rate": f"{float(xi_cr)} 1/s",
        "max_iter": 80,
        "reltol": 1.0e-2,
        "abstol0": 1.0e-9,
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
        "local_chi_factor": 0.5,
        "shielding_self_weight": float(self_weight),
        "enable_co_phase": False,
        "coupling_mode": str(coupling_mode),
        "astrochem_n_updates": int(astrochem_n_updates),
        "astrochem_t_end_yr": float(astrochem_t_end_yr),
    }
    if use_radmc3d_chi:
        gow17_cfg["chi_is_incident"] = False
    else:
        gow17_cfg["chi_is_incident"] = True

    chem_result = run_chemistry(radm, model="gow17", config=gow17_cfg)

    y_out = np.asarray(radm.gow17_y, dtype=float)
    Av_out = np.asarray(radm.Av.to("dimensionless").magnitude, dtype=float)

    if not skip_diagnostics:
        outdir = Path(diagnostic_outdir)
        _plot_diagnostics(
            outdir=outdir,
            suffix=str(diagnostic_suffix),
            radm=radm,
            y_out=y_out,
        )

    Av_samp = Av_out.reshape(-1)
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
        "self_weight": float(self_weight),
        "shape": tuple(int(s) for s in shape),
        "R_cm": float(R_cm),
        "nH_index": int(nH_index),
        "nr": int(nr),
        "ntheta": int(ntheta),
        "nphi": int(nphi),
        "use_radmc3d_chi": bool(use_radmc3d_chi),
        "gow17_diagnostics": gow17_diag,
    }

    return np.asarray(y_out, dtype=float), abd, np.asarray(ref.Av, dtype=float), info, Av_out, radm


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
) -> Dict[str, np.ndarray]:
    """Shell-average abundances by Av bin and return binned profiles.

    Bins cells by their ``Av_3d`` value (which is a monotonic function
    of depth from the sphere surface) and computes the mean abundance
    in each bin.

    Parameters
    ----------
    y_out : np.ndarray
        GOW17 abundance array, shape ``(nr, ntheta, nphi, N_Y)``.
    Av_3d : np.ndarray
        Visual extinction array, shape ``(nr, ntheta, nphi)``.
    nbins : int
        Number of Av bins.

    Returns
    -------
    dict
        Keys: ``Av``, ``xH2``, ``xCO``, ``xCplus``, with 1-D arrays of
        length *nbins*.
    """
    av_flat = Av_3d.ravel()
    av_min, av_max = float(av_flat.min()), float(av_flat.max())
    bin_edges = np.linspace(av_min, av_max, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    idx = np.digitize(av_flat, bin_edges) - 1
    idx = np.clip(idx, 0, nbins - 1)

    def _bin_mean(field_flat: np.ndarray) -> np.ndarray:
        sums = np.bincount(idx, weights=field_flat, minlength=nbins).astype(float)
        counts = np.bincount(idx, minlength=nbins).astype(float)
        out = np.zeros(nbins, dtype=float)
        m = counts > 0
        out[m] = sums[m] / counts[m]
        return out

    xH2_prof = _bin_mean(y_out[..., I_H2].ravel())
    xCO_prof = _bin_mean(y_out[..., I_CO].ravel())
    xCplus_prof = _bin_mean(y_out[..., I_CP].ravel())

    return {
        "Av": bin_centers,
        "xH2": xH2_prof,
        "xCO": xCO_prof,
        "xCplus": xCplus_prof,
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
    header = "Av,xH2,xCO,xCplus"
    data = np.column_stack([
        profile["Av"], profile["xH2"], profile["xCO"], profile["xCplus"],
    ])
    np.savetxt(p, data, delimiter=",", header=header, comments="")
    return p


def _run_slab_reference(
    *,
    Av_ref: np.ndarray,
    nH_cm3: float,
    chi0: float,
    xi_cr: float,
    const_temp: bool,
    nside: int,
) -> Dict[str, np.ndarray]:
    """Run 1-D slab reference with Gong+17 Appendix factors.

    Uses ``chi_is_incident=True`` and ``Av = 2 * Av_perp`` (the doubling
    is already baked into *Av_ref* from the sphere run).  Shielding
    iteration settings are chosen for 1-D convergence and are independent
    of the 3-D sphere run parameters.

    Parameters
    ----------
    Av_ref : np.ndarray
        1-D array of Av values (already doubled).
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
    N_slab = 500
    Av_max = float(np.max(Av_ref))
    Av_lo = max(float(np.min(Av_ref)), Av_max / N_slab)
    Av_dense = np.logspace(np.log10(Av_lo), np.log10(Av_max), N_slab)

    NH_flat = Av_dense * 1.87e21

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

    radm = RadModel(model, model_dir=".")

    radm.chi = Quantity(np.full(shape, chi0_incident, dtype=float), "dimensionless")
    radm.Av = Quantity(Av_dense.reshape(shape), "dimensionless")
    radm.gas_temperature = Quantity(np.full(shape, 50.0, dtype=float), "K")
    radm.dust_temperature = Quantity(np.full(shape, 50.0, dtype=float), "K")

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
        "const_temp": bool(const_temp),
        "t_end": "2.0e9 yr",
        "nside": int(nside),
        "chi0": float(chi0_incident),
        "ion_rate": f"{float(xi_cr)} 1/s",
        "max_iter": 80,
        "reltol": 1.0e-2,
        "abstol0": 1.0e-9,
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
        "local_chi_factor": 0.5,
        "enable_co_phase": False,
        "chi_is_incident": True,
    }

    run_chemistry(radm, model="gow17", config=gow17_cfg)

    y_slab = np.asarray(radm.gow17_y, dtype=float)
    Av_out = np.asarray(radm.Av.to("dimensionless").magnitude, dtype=float).ravel()

    # Interpolate dense-grid results back onto the sphere's Av bins for plotting.
    Av_ref_arr = np.asarray(Av_ref, dtype=float)
    return {
        "Av": Av_ref_arr,
        "xH2": np.interp(Av_ref_arr, Av_out, y_slab[..., I_H2].ravel()),
        "xCO": np.interp(Av_ref_arr, Av_out, y_slab[..., I_CO].ravel()),
        "xCplus": np.interp(Av_ref_arr, Av_out, y_slab[..., I_CP].ravel()),
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
) -> Dict[str, np.ndarray]:
    """Common post-processing: save convergence, shell-avg, slab ref, overlay.

    Returns the shell-averaged sphere profile dict.
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
    )
    _save_sphere_profile(outdir, sphere, fname=sphere_csv_name)

    slab = _run_slab_reference(
        Av_ref=sphere["Av"],
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
    return sphere


def main() -> None:

    root = Path(__file__).resolve().parents[1]
    ref_dir = root / "other_codes" / "pdr" / "out_example_simple"
    ref = _load_reference(ref_dir)

    outdir = Path(__file__).resolve().parent / "reproduce_gow17_fig2_3dhealpix"
    outdir.mkdir(parents=True, exist_ok=True)

    species = ["CO", "C", "C+", "H3+", "OHx", "CHx", "He+"]

    nH_index = 0
    nH_val = float(ref.nH_values[int(nH_index)])
    chi0 = 1.0
    xi_cr = 2.0e-16
    shielding_max_iter = 20
    const_temp = True
    nr = 256
    ntheta = 16
    nphi = 16
    nside = 4

    baseline_radmc3d_inputs_dir = root / "examples" / "testbed" / "baseline_run" / "radmc3d_inputs"
    radmc3d_isrf_path = root / "data" / "ISRF.dat"
    radmc3d_uv_min = Quantity(91.2, "nm")
    radmc3d_uv_max = Quantity(200.0, "nm")
    radmc3d_uv_n_wavelengths = 30

    radmc3d_dust_nbins = int(diskbridge.params.nbins) if not isinstance(diskbridge.params.nbins, list) else int(diskbridge.params.nbins[0])
    radmc3d_scat_mode = int(diskbridge.params.scat_mode)
    radmc3d_nphot_thermal = 2e8
    radmc3d_nphot_scat = 2e8
    radmc3d_nphot_mono = 1e8
    radmc3d_force = False
    radmc3d_vacuum_dust = False

    diag_suffix_base = f"sphere_nH_{int(nH_val)}_nr_{nr}_nt_{ntheta}_np_{nphi}_nside_{nside}"

    # Common kwargs for the sphere runner (analytic chi).
    _common_kw = dict(
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        shielding_max_iter=shielding_max_iter,
        const_temp=const_temp,
        nside=nside,
        self_weight=1.0,
        nr=nr,
        ntheta=ntheta,
        nphi=nphi,
        ref=ref,
        nH_index=int(nH_index),
        use_radmc3d_chi=False,
        radmc3d_model_dir=str(outdir / "radmc3d_unused"),
        radmc3d_force=bool(radmc3d_force),
        radmc3d_nphot_mono=int(radmc3d_nphot_mono),
        radmc3d_nphot_thermal=int(radmc3d_nphot_thermal),
        radmc3d_nphot_scat=int(radmc3d_nphot_scat),
        radmc3d_scat_mode=int(radmc3d_scat_mode),
        radmc3d_dust_nbins=int(radmc3d_dust_nbins),
        radmc3d_vacuum_dust=bool(radmc3d_vacuum_dust),
        radmc3d_isrf_path=Path(radmc3d_isrf_path),
        radmc3d_uv_min=radmc3d_uv_min,
        radmc3d_uv_max=radmc3d_uv_max,
        radmc3d_uv_n_wavelengths=int(radmc3d_uv_n_wavelengths),
        baseline_radmc3d_inputs_dir=Path(baseline_radmc3d_inputs_dir),
        diagnostic_outdir=str(outdir / "out_unused"),
        diagnostic_suffix=str(diag_suffix_base),
        coupling_mode="fixed_point",
        astrochem_n_updates=0,
        astrochem_t_end_yr=1.0e6,
    )

    # ----------------------------------------------------------------
    # Run 1: fixed-point coupling (baseline)
    # ----------------------------------------------------------------
    out_fixed = outdir / "out_fixed_point"
    out_fixed.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Run 1: fixed-point coupling (chi_is_incident=True)")
    print("=" * 60)

    y_out_fp, abd_fp, Av_db_fp, info_fp, Av3d_fp, _ = run_gow17_internal_3d_healpix_sphere(
        **{**_common_kw, "diagnostic_outdir": str(out_fixed), "diagnostic_suffix": diag_suffix_base, "coupling_mode": "fixed_point"},
    )

    suffix_fp = (
        f"sphere_nH_{int(nH_val)}_nr_{nr}_nt_{ntheta}_np_{nphi}_nside_{nside}"
    )
    _plot_compare(
        outdir=out_fixed,
        Av_ref=ref.Av,
        Av_db=Av_db_fp,
        ref=ref,
        nH_index=int(nH_index),
        abd_db=abd_fp,
        species=species,
        suffix=suffix_fp,
    )

    sphere_fp = _postprocess_run(
        outdir=out_fixed,
        y_out=y_out_fp,
        info=info_fp,
        Av_3d=Av3d_fp,
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        const_temp=const_temp,
        nside=nside,
        suffix="",
    )

    diag_fp = info_fp.get("gow17_diagnostics", {})
    print(
        f"nH={nH_val:.3e}: shape={info_fp['shape']}, R_cm={info_fp['R_cm']:.3e}"
    )

    # ----------------------------------------------------------------
    # Run 2: astrochem coupling
    # ----------------------------------------------------------------
    out_astro = outdir / "out_astrochem"
    out_astro.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Run 2: AstroChem-style coupling (chi_is_incident=True)")
    print("=" * 60)

    y_out_ac, abd_ac, Av_db_ac, info_ac, Av3d_ac, _ = run_gow17_internal_3d_healpix_sphere(
        **{**_common_kw, "diagnostic_outdir": str(out_astro), "diagnostic_suffix": diag_suffix_base, "coupling_mode": "astrochem", "astrochem_n_updates": 5, "astrochem_t_end_yr": 1.0e6},
    )

    _plot_compare(
        outdir=out_astro,
        Av_ref=ref.Av,
        Av_db=Av_db_ac,
        ref=ref,
        nH_index=int(nH_index),
        abd_db=abd_ac,
        species=species,
        suffix=suffix_fp,
    )

    sphere_ac = _postprocess_run(
        outdir=out_astro,
        y_out=y_out_ac,
        info=info_ac,
        Av_3d=Av3d_ac,
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        const_temp=const_temp,
        nside=nside,
        suffix="",
    )

    # ----------------------------------------------------------------
    # Comparison plots: fixed_point vs astrochem
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Comparison: fixed_point vs astrochem")
    print("=" * 60)

    _plot_compare_fixed_vs_astrochem_species(
        outdir=outdir,
        fixed=sphere_fp,
        astro=sphere_ac,
        fname="compare_fixed_vs_astrochem_species.png",
    )
    print("Saved: compare_fixed_vs_astrochem_species.png")

    # ----------------------------------------------------------------
    # Run 3: RADMC-3D chi (fixed-point only)
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Run 3: RADMC-3D chi (chi_is_incident=False)")
    print("=" * 60)

    radmc3d_model_dir = outdir / (
        f"radmc3d_sphere_nH_{int(float(nH_val))}_nr_{int(nr)}_nt_{int(ntheta)}_np_{int(nphi)}"
    )
    diag_suffix_rt = f"radmc3d_{diag_suffix_base}"
    y_out_m2, abd_m2, Av_db_m2, info_m2, Av3d_m2, _ = run_gow17_internal_3d_healpix_sphere(
        **{
            **_common_kw,
            "use_radmc3d_chi": True,
            "radmc3d_model_dir": str(radmc3d_model_dir),
            "radmc3d_force": bool(radmc3d_force),
            "radmc3d_nphot_mono": int(radmc3d_nphot_mono),
            "radmc3d_nphot_thermal": int(radmc3d_nphot_thermal),
            "radmc3d_nphot_scat": int(radmc3d_nphot_scat),
            "radmc3d_scat_mode": int(radmc3d_scat_mode),
            "radmc3d_dust_nbins": int(radmc3d_dust_nbins),
            "radmc3d_vacuum_dust": bool(radmc3d_vacuum_dust),
            "diagnostic_outdir": str(outdir),
            "diagnostic_suffix": str(diag_suffix_rt),
            "coupling_mode": "fixed_point",
        }
    )

    run_vacuum = False
    if run_vacuum:
        print("\n" + "=" * 60)
        print("Run 4: RADMC-3D chi vacuum dust")
        print("=" * 60)
        _ = run_gow17_internal_3d_healpix_sphere(
            **{
                **_common_kw,
                "use_radmc3d_chi": True,
                "radmc3d_model_dir": str(outdir / "radmc3d_vacuum_dust"),
                "radmc3d_force": True,
                "radmc3d_vacuum_dust": True,
                "diagnostic_outdir": str(outdir),
                "diagnostic_suffix": f"radmc3d_vacuum_{diag_suffix_base}",
                "coupling_mode": "fixed_point",
            }
        )

    suffix_m2 = (
        f"radmc3d_sphere_nH_{int(nH_val)}_nr_{nr}_nt_{ntheta}_np_{nphi}_nside_{nside}"
    )
    _plot_compare(
        outdir=outdir,
        Av_ref=ref.Av,
        Av_db=Av_db_m2,
        ref=ref,
        nH_index=int(nH_index),
        abd_db=abd_m2,
        species=species,
        suffix=suffix_m2,
    )

    sphere_m2 = _postprocess_run(
        outdir=outdir,
        y_out=y_out_m2,
        info=info_m2,
        Av_3d=Av3d_m2,
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        const_temp=const_temp,
        nside=nside,
        suffix="_radmc3d",
    )

    # ----------------------------------------------------------------
    # Astrochem convergence sweep: i = 0..N
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Astrochem convergence sweep (i=0..N)")
    print("=" * 60)

    sweep_dir = outdir / "out_astrochem_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    N_max = 8
    i_values: List[int] = list(range(N_max + 1))

    # Reference (i=N_max) — diagnostics suppressed; radm collected for GIF
    y_refN, _abdN, _AvdbN, info_refN, Av3d_refN, _ = run_gow17_internal_3d_healpix_sphere(
        **{**_common_kw, "diagnostic_outdir": str(sweep_dir), "diagnostic_suffix": f"astrochem_i_{N_max}", "coupling_mode": "astrochem", "astrochem_n_updates": int(N_max), "astrochem_t_end_yr": 1.0e6, "skip_diagnostics": True},
    )
    prof_refN = _shell_average_sphere(y_out=y_refN, Av_3d=Av3d_refN, nbins=64)

    rel_err_H2: List[float] = []
    rel_err_CO: List[float] = []
    rel_err_Cplus: List[float] = []
    rel_err_max: List[float] = []
    rel_err_H2_vs_fp: List[float] = []
    rel_err_CO_vs_fp: List[float] = []
    rel_err_Cplus_vs_fp: List[float] = []
    rel_err_max_vs_fp: List[float] = []

    sweep_radms: List[RadModel] = []
    sweep_y_outs: List[np.ndarray] = []

    eps = float(EPS_CHI)

    def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = np.maximum(np.abs(b), eps)
        return float(np.max(np.abs(a - b) / denom))

    for i in i_values:
        y_i, _abdi, _Avdbi, info_i, Av3d_i, radm_i = run_gow17_internal_3d_healpix_sphere(
            **{**_common_kw, "diagnostic_outdir": str(sweep_dir), "diagnostic_suffix": f"astrochem_i_{i}", "coupling_mode": "astrochem", "astrochem_n_updates": int(i), "astrochem_t_end_yr": 1.0e6, "skip_diagnostics": True},
        )
        sweep_radms.append(radm_i)
        sweep_y_outs.append(y_i)
        prof_i = _shell_average_sphere(y_out=y_i, Av_3d=Av3d_i, nbins=64)

        eH2 = _rel_err(prof_i["xH2"], prof_refN["xH2"])
        eCO = _rel_err(prof_i["xCO"], prof_refN["xCO"])
        eCp = _rel_err(prof_i["xCplus"], prof_refN["xCplus"])
        rel_err_H2.append(eH2)
        rel_err_CO.append(eCO)
        rel_err_Cplus.append(eCp)
        rel_err_max.append(max(eH2, eCO, eCp))

        eH2_fp = _rel_err(prof_i["xH2"], sphere_fp["xH2"])
        eCO_fp = _rel_err(prof_i["xCO"], sphere_fp["xCO"])
        eCp_fp = _rel_err(prof_i["xCplus"], sphere_fp["xCplus"])
        rel_err_H2_vs_fp.append(eH2_fp)
        rel_err_CO_vs_fp.append(eCO_fp)
        rel_err_Cplus_vs_fp.append(eCp_fp)
        rel_err_max_vs_fp.append(max(eH2_fp, eCO_fp, eCp_fp))

    sweep_json = {
        "N_max": int(N_max),
        "i_values": [int(x) for x in i_values],
        "rel_err_H2": rel_err_H2,
        "rel_err_CO": rel_err_CO,
        "rel_err_Cplus": rel_err_Cplus,
        "rel_err_max": rel_err_max,
        "rel_err_H2_vs_fp": rel_err_H2_vs_fp,
        "rel_err_CO_vs_fp": rel_err_CO_vs_fp,
        "rel_err_Cplus_vs_fp": rel_err_Cplus_vs_fp,
        "rel_err_max_vs_fp": rel_err_max_vs_fp,
        "reference": jsonable(info_refN.get("gow17_diagnostics", {})),
    }
    (sweep_dir / "astrochem_sweep.json").write_text(json.dumps(sweep_json, indent=2) + "\n")
    _make_sweep_gifs_from_data(sweep_dir, sweep_radms, sweep_y_outs, i_values)
    _plot_compare_fixed_vs_astrochem_convergence(
        outdir=sweep_dir,
        fname="compare_fixed_vs_astrochem_convergence.png",
        i_values=i_values,
        rel_err_H2=rel_err_H2,
        rel_err_CO=rel_err_CO,
        rel_err_Cplus=rel_err_Cplus,
        rel_err_max=rel_err_max,
        rel_err_H2_vs_fp=rel_err_H2_vs_fp,
        rel_err_CO_vs_fp=rel_err_CO_vs_fp,
        rel_err_Cplus_vs_fp=rel_err_Cplus_vs_fp,
        rel_err_max_vs_fp=rel_err_max_vs_fp,
    )

    # ----------------------------------------------------------------
    # Final comparison: analytic (fixed_point) vs RADMC-3D
    # ----------------------------------------------------------------
    _plot_mode_comparison(
        outdir=outdir,
        sphere1=sphere_fp,
        sphere2=sphere_m2,
        label1="analytic",
        label2="RADMC-3D",
        fname="analytic_vs_radmc3d_species_profiles.png",
    )
    print(f"\nSaved: analytic_vs_radmc3d_species_profiles.png")
    print("Done.")


if __name__ == "__main__":
    main()
