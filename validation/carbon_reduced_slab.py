from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from diskbridge._constants import TAU_CO_FORM, X_C_TOT
from diskbridge._units import Quantity, units
from diskbridge.chemistry.api import run_chemistry
from diskbridge.chemistry.models._carbon_reduced_math import compute_carbon_reduced_rates
from diskbridge.chemistry.shielding.h2_db96 import h2_self_shielding_db96
from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel


@dataclass(frozen=True)
class CarbonReducedSlabConfig:
    n_cells: int = 256
    Av_max: float = 10.0
    nH_cm3: float = 1.0e4
    chi0: float = 1.0
    Tdust_K: float = 20.0

    shielding_iter: int = 2


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _annotate(ax, lines: list[str]) -> None:
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
    )


def _write_inputs(out_dir: Path, cfg: CarbonReducedSlabConfig) -> None:
    inputs = {
        "job": "carbon_reduced_slab",
        "time": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.__dict__,
    }
    (out_dir / "inputs.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True), encoding="utf-8"
    )


def _build_slab_on_Av_grid(cfg: CarbonReducedSlabConfig) -> tuple[RadModel, np.ndarray]:
    m_H = units("m_H")

    gamma = 3.02
    NH_per_Av = 1.87e21

    Av_edges = np.linspace(0.0, float(cfg.Av_max), int(cfg.n_cells) + 1, dtype=float)
    NH_edges = Av_edges * 1.87e21

    dNH = np.diff(NH_edges)
    dx_cm = dNH / float(cfg.nH_cm3)

    x_edges_cm = np.empty(int(cfg.n_cells) + 1, dtype=float)
    x_edges_cm[0] = 0.0
    x_edges_cm[1:] = np.cumsum(dx_cm)

    mesh = Mesh.cartesian(
        x=Axis(edges=Quantity(x_edges_cm, "cm")),
        y=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
        z=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
    )

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    rho = (
        Quantity(np.full(mesh.shape, float(cfg.nH_cm3)), "cm^-3") * (1.4 * m_H)
    ).to("g/cm^3")
    model.gas_register(
        "density", Field(quantity="density", data=rho, axis_order=("x", "y", "z"))
    )

    sigma_d_per_H_cm2 = 1.0 / (1.086 * float(NH_per_Av))
    sigma = Quantity(np.full(mesh.shape, sigma_d_per_H_cm2), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")),
    )

    rad = RadModel(model)
    rad.nH = Quantity(np.full(mesh.shape, float(cfg.nH_cm3)), "cm^-3")
    rad.dust_temperature = Quantity(np.full(mesh.shape, float(cfg.Tdust_K)), "K")

    Av_centers = 0.5 * (Av_edges[:-1] + Av_edges[1:])
    rad.Av = Quantity(Av_centers.reshape(mesh.shape), "dimensionless")
    chi_local = float(cfg.chi0) * np.exp(-float(gamma) * Av_centers)
    rad.chi = Quantity(chi_local.reshape(mesh.shape), "dimensionless")

    return rad, Av_centers


def _as_1d(q: Quantity, unit: str) -> np.ndarray:
    return np.asarray(q.to(unit).magnitude, dtype=float).reshape(-1)


def _placeholder_plot(out_dir: Path, title: str, text: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.01, 0.99, text, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "error.png", dpi=150)
    plt.close(fig)


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = CarbonReducedSlabConfig()
    _write_inputs(out_dir, cfg)

    rad, Av = _build_slab_on_Av_grid(cfg)
    tau_form = Quantity(np.full(rad.model.mesh.shape, float(TAU_CO_FORM)), "s")

    error: str | None = None
    summary: dict = {}

    try:
        # --- Run 1: baseline (no C self-shielding) ---
        chem = run_chemistry(
            rad,
            model="carbon_reduced",
            config={
                "skip_shielding": False,
                "shielding_iter": int(cfg.shielding_iter),
                "tau_form": tau_form,
                "c_self_shielding": False,
            },
            write=False,
        )

        nH_cm3 = _as_1d(rad.ensure_nH(), "cm^-3")
        nH2_cm3 = _as_1d(rad.nH2, "cm^-3")
        nHI_cm3 = _as_1d(rad.nH_atom, "cm^-3")
        nco_gas_cm3 = _as_1d(rad.nco_gas, "cm^-3")
        nco_ice_cm3 = _as_1d(rad.nco_ice, "cm^-3")
        nCplus_cm3 = _as_1d(rad.nCplus, "cm^-3")
        nC_cm3 = _as_1d(rad.nC, "cm^-3")

        theta_co = _as_1d(rad.theta_co, "dimensionless")

        # Save baseline C+/C for comparison
        nCplus_baseline = nCplus_cm3.copy()
        nC_baseline = nC_cm3.copy()
        nco_gas_baseline = nco_gas_cm3.copy()

        NH_centers = Av * 1.87e21
        dNH = np.empty_like(NH_centers)
        dNH[:-1] = NH_centers[1:] - NH_centers[:-1]
        dNH[-1] = dNH[-2]

        xH2 = np.where(nH_cm3 > 0.0, nH2_cm3 / nH_cm3, 0.0)
        N_H2 = np.zeros_like(NH_centers)
        if NH_centers.size >= 2:
            N_H2[1:] = np.cumsum(xH2[:-1] * dNH[:-1])
        theta_h2 = h2_self_shielding_db96(N_H2, b5=2.0)

        rates = compute_carbon_reduced_rates(
            nH=rad.ensure_nH(),
            T=rad.dust_temperature,
            chi=rad.chi,
            theta_co=rad.theta_co,
            sigma_d_per_H=rad.ensure_sigma_d_per_H(),
            min_rate=float(chem.meta.get("min_rate", 0.0)),
        )

        tau_form_s = _as_1d(tau_form, "s")
        nco_total_cm3 = nco_gas_cm3 + nco_ice_cm3

        Xco_tot = float(chem.meta.get("Xco_tot", 0.0))
        nco_max = Xco_tot * nH_cm3

        R_form = np.where(
            tau_form_s > 0.0, (nco_max - nco_total_cm3) / tau_form_s, np.nan
        )
        R_pdiss = _as_1d(rad.k_diss_co, "1/s") * nco_gas_cm3
        R_freeze = _as_1d(rates.k_fo, "1/s") * nco_gas_cm3
        R_desorb = _as_1d(rates.k_td, "1/s") * nco_ice_cm3

        carbon_budget = np.where(
            (nH_cm3 > 0.0) & (float(X_C_TOT) > 0.0),
            (nCplus_cm3 + nC_cm3 + nco_total_cm3) / (float(X_C_TOT) * nH_cm3),
            np.nan,
        )

        # --- Run 2: with C self-shielding ---
        rad2, _ = _build_slab_on_Av_grid(cfg)
        tau_form2 = Quantity(np.full(rad2.model.mesh.shape, float(TAU_CO_FORM)), "s")
        chem2 = run_chemistry(
            rad2,
            model="carbon_reduced",
            config={
                "skip_shielding": False,
                "shielding_iter": int(cfg.shielding_iter),
                "tau_form": tau_form2,
                "c_self_shielding": True,
                "c_shielding_iter": 1,
            },
            write=False,
        )

        nCplus_shielded = _as_1d(rad2.nCplus, "cm^-3")
        nC_shielded = _as_1d(rad2.nC, "cm^-3")
        nco_gas_shielded = _as_1d(rad2.nco_gas, "cm^-3")
        theta_c = _as_1d(rad2.theta_c, "dimensionless")

        # Verify CO is essentially unchanged (only carbon closure changed)
        co_reldiff = np.where(
            nco_gas_baseline > 1e-30,
            np.abs(nco_gas_shielded - nco_gas_baseline) / nco_gas_baseline,
            0.0,
        )
        co_max_reldiff = float(np.max(co_reldiff))

        plt = _setup_matplotlib()

        params_box = [
            f"nH={cfg.nH_cm3:.3e} cm^-3",
            f"chi0={cfg.chi0:g}",
            f"Tdust={cfg.Tdust_K:g} K",
            f"Av_max={cfg.Av_max:g}",
            f"n_cells={cfg.n_cells}",
            f"shielding_iter={cfg.shielding_iter}",
        ]

        def _logx(n):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log10(np.where(nH_cm3 > 0.0, n / nH_cm3, np.nan))

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(Av, _logx(nH2_cm3), label="H2")
        ax.plot(Av, _logx(nHI_cm3), label="H")
        ax.plot(Av, _logx(nCplus_cm3), label="C+")
        ax.plot(Av, _logx(nC_cm3), label="C")
        ax.plot(Av, _logx(nco_gas_cm3), label="CO(gas)")
        ax.plot(Av, _logx(nco_ice_cm3), label="CO(ice)")
        ax.set_xlabel("Av")
        ax.set_ylabel("log10(x)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        _annotate(ax, params_box)
        fig.tight_layout()
        fig.savefig(out_dir / "abundances_vs_Av.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(Av, _logx(nco_gas_cm3), label="CO(gas)")
        ax.plot(Av, _logx(nco_ice_cm3), label="CO(ice)")
        ax.plot(Av, _logx(nco_total_cm3), label="CO(total)")
        ax.set_xlabel("Av")
        ax.set_ylabel("log10(x)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        _annotate(ax, params_box)
        fig.tight_layout()
        fig.savefig(out_dir / "co_partition_vs_Av.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(Av, np.maximum(theta_h2, 1.0e-300), label="theta_H2")
        ax.plot(Av, np.maximum(theta_co, 1.0e-300), label="theta_CO")
        ax.plot(Av, np.maximum(theta_c, 1.0e-300), label="theta_C")
        ax.set_xlabel("Av")
        ax.set_ylabel("shielding factor")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        _annotate(ax, params_box)
        fig.tight_layout()
        fig.savefig(out_dir / "shielding_vs_Av.png", dpi=150)
        plt.close(fig)

        # C self-shielding comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
        ax = axes[0]
        ax.plot(Av, _logx(nCplus_baseline), label="C+ (no C shield)", ls="--")
        ax.plot(Av, _logx(nCplus_shielded), label="C+ (C shielded)")
        ax.plot(Av, _logx(nC_baseline), label="C (no C shield)", ls="--")
        ax.plot(Av, _logx(nC_shielded), label="C (C shielded)")
        ax.set_xlabel("Av")
        ax.set_ylabel("log10(x)")
        ax.set_title("C self-shielding effect on C/C+")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
        _annotate(ax, params_box)

        ax = axes[1]
        ax.plot(Av, theta_c, label="theta_C", color="tab:green")
        ax.set_xlabel("Av")
        ax.set_ylabel("theta_C")
        ax.set_title(f"C shielding factor (CO max reldiff={co_max_reldiff:.2e})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / "c_self_shielding_comparison.png", dpi=150)
        plt.close(fig)

        def _log10_rate(r):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log10(np.where(r > 0.0, r, np.nan))

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(Av, _log10_rate(R_form), label="CO formation")
        ax.plot(Av, _log10_rate(R_pdiss), label="CO photodiss")
        ax.plot(Av, _log10_rate(R_freeze), label="freezeout")
        ax.plot(Av, _log10_rate(R_desorb), label="thermal desorb")
        ax.set_xlabel("Av")
        ax.set_ylabel("log10(rate) [cm^-3 s^-1]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        _annotate(ax, params_box)
        fig.tight_layout()
        fig.savefig(out_dir / "rates_vs_Av.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(Av, carbon_budget, color="k")
        ax.axhline(1.0, color="0.5", linestyle="--")
        ax.set_xlabel("Av")
        ax.set_ylabel("(C+ + C + CO) / (X_C_tot * nH)")
        ax.grid(True, alpha=0.3)
        _annotate(ax, params_box)
        fig.tight_layout()
        fig.savefig(out_dir / "carbon_budget_vs_Av.png", dpi=150)
        plt.close(fig)

        summary = {
            "job": "carbon_reduced_slab",
            "time": datetime.now().isoformat(timespec="seconds"),
            "config": cfg.__dict__,
            "pass": True,
            "Xco_tot": float(Xco_tot),
            "carbon_budget": {
                "min": float(np.nanmin(carbon_budget)),
                "max": float(np.nanmax(carbon_budget)),
                "p90": float(np.nanpercentile(carbon_budget, 90.0)),
            },
            "c_self_shielding": {
                "co_max_reldiff": co_max_reldiff,
                "theta_c_min": float(np.min(theta_c)),
                "theta_c_max": float(np.max(theta_c)),
            },
        }

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        _placeholder_plot(out_dir, "carbon_reduced_slab failed", error)
        summary = {
            "job": "carbon_reduced_slab",
            "time": datetime.now().isoformat(timespec="seconds"),
            "config": cfg.__dict__,
            "pass": False,
            "error": error,
        }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    if error is not None:
        raise RuntimeError(error)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / "validation_out" / "carbon_reduced_slab"
    try:
        run(out_root)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
