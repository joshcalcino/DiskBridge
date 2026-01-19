from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from diskbridge._constants import TAU_CO_FORM
from diskbridge._units import Quantity, units
from diskbridge.chemistry.api import run_chemistry
from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel


@dataclass(frozen=True)
class FreezeoutSlabConfig:
    n_cells: int = 256
    Av_max: float = 15.0
    nH_cm3: float = 1.0e6
    chi0: float = 1.0
    Tdust_K: float = 12.0

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


def _write_inputs(out_dir: Path, cfg: FreezeoutSlabConfig) -> None:
    inputs = {
        "job": "freezeout_slab",
        "time": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.__dict__,
    }
    (out_dir / "inputs.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True), encoding="utf-8"
    )


def _build_slab_on_Av_grid(cfg: FreezeoutSlabConfig) -> tuple[RadModel, np.ndarray]:
    m_H = units("m_H")

    gamma = 3.02
    NH_per_Av = 1.87e21

    Av_edges = np.linspace(0.0, float(cfg.Av_max), int(cfg.n_cells) + 1, dtype=float)
    NH_edges = Av_edges * NH_per_Av

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

    rho = (Quantity(np.full(mesh.shape, float(cfg.nH_cm3)), "cm^-3") * (1.4 * m_H)).to(
        "g/cm^3"
    )
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


def _placeholder_error_plot(out_dir: Path, title: str, text: str) -> None:
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
    cfg = FreezeoutSlabConfig()
    _write_inputs(out_dir, cfg)

    rad, Av = _build_slab_on_Av_grid(cfg)

    tau_form = Quantity(np.full(rad.model.mesh.shape, float(TAU_CO_FORM)), "s")

    error: str | None = None
    summary: dict = {}

    try:
        run_chemistry(
            rad,
            model="carbon_reduced",
            config={
                "skip_shielding": False,
                "shielding_iter": int(cfg.shielding_iter),
                "tau_form": tau_form,
            },
            write=False,
        )

        nH_cm3 = _as_1d(rad.ensure_nH(), "cm^-3")
        nco_gas_cm3 = _as_1d(rad.nco_gas, "cm^-3")
        nco_ice_cm3 = _as_1d(rad.nco_ice, "cm^-3")

        nco_total_cm3 = nco_gas_cm3 + nco_ice_cm3
        denom = np.where(nco_total_cm3 > 0.0, nco_total_cm3, np.nan)
        ice_frac = nco_ice_cm3 / denom

        Av_max = float(cfg.Av_max)
        idx_max = int(np.argmax(Av))
        ice_frac_at_max = float(ice_frac[idx_max])
        gas_frac_at_max = float(1.0 - ice_frac_at_max)

        failures: list[str] = []
        if not np.isfinite(ice_frac_at_max):
            failures.append("ice_frac_at_Av_max is not finite")
        else:
            if not (ice_frac_at_max > 0.5):
                failures.append("expected ice_frac_at_Av_max > 0.5")

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
        ax.plot(Av, ice_frac, color="k")
        ax.axhline(0.5, color="0.5", linestyle="--")
        ax.set_xlabel("Av")
        ax.set_ylabel("CO ice fraction")
        ax.grid(True, alpha=0.3)
        _annotate(ax, params_box)
        fig.tight_layout()
        fig.savefig(out_dir / "co_ice_fraction_vs_Av.png", dpi=150)
        plt.close(fig)

        summary = {
            "job": "freezeout_slab",
            "time": datetime.now().isoformat(timespec="seconds"),
            "config": cfg.__dict__,
            "Av_max": Av_max,
            "ice_frac_at_Av_max": ice_frac_at_max,
            "gas_frac_at_Av_max": gas_frac_at_max,
        }

        if failures:
            error = "freezeout_slab failed: " + "; ".join(failures)
            summary["pass"] = False
            summary["error"] = error
        else:
            summary["pass"] = True

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        _placeholder_error_plot(out_dir, "freezeout_slab failed", error)
        summary = {
            "job": "freezeout_slab",
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
    out_root = repo_root / "validation_out" / "freezeout_slab"
    try:
        run(out_root)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
