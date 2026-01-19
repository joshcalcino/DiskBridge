from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np


def _get_git_commit(repo_root: Path) -> str | None:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return None
    txt = head.read_text(encoding="utf-8").strip()
    if txt.startswith("ref:"):
        ref = txt.split(":", 1)[1].strip()
        ref_path = repo_root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        return None
    return txt


def _as_1d(q, unit: str) -> np.ndarray:
    return np.asarray(q.to(unit).magnitude).reshape(-1)


def _column_to_outer_boundary(mesh, field_cm3: np.ndarray) -> np.ndarray:
    from diskbridge.chemistry.shielding.columns_1d import (
        column_to_outer_boundary_1d,
        effective_1d_axis,
    )

    axis_name, axis_index = effective_1d_axis(mesh, tuple(field_cm3.shape))
    return column_to_outer_boundary_1d(
        mesh,
        field_cm3,
        axis_name=axis_name,
        axis_index=axis_index,
        outer="min",
    )


def _compute_Av(mesh, nH_cm3: np.ndarray, sigma_d_per_H_cm2: np.ndarray) -> np.ndarray:
    mag_per_tau = 1.086
    N_H = _column_to_outer_boundary(mesh, nH_cm3)
    tau = N_H * sigma_d_per_H_cm2
    return mag_per_tau * tau


_NH_PER_AV_CM2 = 1.87e21
_GAMMA_DUST_ATTEN = 3.02


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _annotate_params(ax, lines: list[str]):
    text = "\n".join(lines)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
    )


def _plot_abundances_vs_Av(
    *,
    out_png: Path,
    Av: np.ndarray,
    nH_cm3: np.ndarray,
    nH2_cm3: np.ndarray,
    nco_gas_cm3: np.ndarray,
    nco_ice_cm3: np.ndarray,
    nCplus_cm3: np.ndarray,
    nC_cm3: np.ndarray,
    ne_cm3: np.ndarray,
    params_box: list[str],
):
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    def _x(n):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where(nH_cm3 > 0.0, n / nH_cm3, np.nan))

    nco_total = nco_gas_cm3 + nco_ice_cm3

    ax.plot(Av, _x(nH2_cm3), label="H2")
    ax.plot(Av, _x(nco_gas_cm3), label="CO(gas)")
    ax.plot(Av, _x(nco_ice_cm3), label="CO(ice)")
    ax.plot(Av, _x(nco_total), label="CO(total)")
    ax.plot(Av, _x(nCplus_cm3), label="C+")
    ax.plot(Av, _x(nC_cm3), label="C")
    ax.plot(Av, _x(ne_cm3), label="e")

    ax.set_xlabel("Av")
    ax.set_ylabel("log10(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _annotate_params(ax, params_box)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_shielding_fields_vs_Av(
    *,
    out_png: Path,
    Av: np.ndarray,
    theta_co: np.ndarray,
    theta_h2: np.ndarray,
    params_box: list[str],
):
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(Av, theta_co, label="theta_CO")
    ax.plot(Av, theta_h2, label="theta_H2")
    ax.set_xlabel("Av")
    ax.set_ylabel("shielding factor")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _annotate_params(ax, params_box)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_rates_vs_Av(
    *,
    out_png: Path,
    Av: np.ndarray,
    R_form: np.ndarray,
    R_pdiss: np.ndarray,
    R_freeze: np.ndarray,
    R_desorb: np.ndarray,
    R_pdes: np.ndarray,
    params_box: list[str],
):
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    def _log10_rate(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where(r > 0.0, r, np.nan))

    ax.plot(Av, _log10_rate(R_form), label="CO formation")
    ax.plot(Av, _log10_rate(R_pdiss), label="CO photodiss")
    ax.plot(Av, _log10_rate(R_freeze), label="freezeout")
    ax.plot(Av, _log10_rate(R_desorb + R_pdes), label="desorp (th + photo)")

    ax.set_xlabel("Av")
    ax.set_ylabel("log10(rate) [cm^-3 s^-1]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _annotate_params(ax, params_box)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_carbon_budget_vs_Av(
    *,
    out_png: Path,
    Av: np.ndarray,
    nH_cm3: np.ndarray,
    nCplus_cm3: np.ndarray,
    nC_cm3: np.ndarray,
    nco_total_cm3: np.ndarray,
    params_box: list[str],
):
    from diskbridge._constants import X_C_TOT

    denom = X_C_TOT * nH_cm3
    with np.errstate(divide="ignore", invalid="ignore"):
        budget = np.where(denom > 0.0, (nCplus_cm3 + nC_cm3 + nco_total_cm3) / denom, np.nan)

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(Av, budget, color="k")
    ax.axhline(1.0, color="0.5", linestyle="--")
    ax.set_xlabel("Av")
    ax.set_ylabel("(nCplus + nC + nCO_total)/(X_C_tot*nH)")
    ax.grid(True, alpha=0.3)
    _annotate_params(ax, params_box)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _build_radmodel_1d(*, n_cells: int, nH_cm3: float, chi: float, Tdust_K: float, Av_max: float):
    from diskbridge._units import Quantity, units
    from diskbridge.model.core import Model, SubModel
    from diskbridge.model.field import Field
    from diskbridge.model.mesh import Axis, Mesh
    from diskbridge.radmc3d.model import RadModel

    m_H = units("m_H")

    Av_edges = np.linspace(0.0, float(Av_max), int(n_cells) + 1, dtype=float)
    NH_edges = Av_edges * _NH_PER_AV_CM2

    dNH = np.diff(NH_edges)
    dx_cm = dNH / float(nH_cm3)

    x_edges_cm = np.empty(int(n_cells) + 1, dtype=float)
    x_edges_cm[0] = 0.0
    x_edges_cm[1:] = np.cumsum(dx_cm)

    Av_centers = 0.5 * (Av_edges[:-1] + Av_edges[1:])

    mesh = Mesh.cartesian(
        x=Axis(edges=Quantity(x_edges_cm, "cm")),
        y=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
        z=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
    )

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    rho = (Quantity(np.full(mesh.shape, float(nH_cm3)), "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=("x", "y", "z")))

    sigma_d_per_H_cm2 = 1.0 / (1.086 * float(_NH_PER_AV_CM2))
    sigma = Quantity(np.full(mesh.shape, sigma_d_per_H_cm2), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")),
    )

    rad = RadModel(model)
    rad.dust_temperature = Quantity(np.full(mesh.shape, float(Tdust_K)), "K")
    chi_local = float(chi) * np.exp(-float(_GAMMA_DUST_ATTEN) * Av_centers)
    rad.chi = Quantity(chi_local.reshape(mesh.shape), "dimensionless")
    rad.Av = Quantity(Av_centers.reshape(mesh.shape), "dimensionless")

    return rad


def _case_A_slab_carbon_reduced(out_dir: Path, *, n_cells: int = 128, Av_max: float = 10.0):
    from diskbridge._units import Quantity
    from diskbridge._constants import TAU_CO_FORM
    from diskbridge.chemistry.api import run_chemistry
    from diskbridge.chemistry.models._carbon_reduced_math import compute_carbon_reduced_rates
    from diskbridge.chemistry.shielding.h2_db96 import h2_self_shielding_db96

    nH_cm3 = 1.0e4
    chi0 = 1.0
    Tdust_K = 20.0

    rad = _build_radmodel_1d(
        n_cells=n_cells,
        nH_cm3=nH_cm3,
        chi=chi0,
        Tdust_K=Tdust_K,
        Av_max=float(Av_max),
    )

    tau_form = Quantity(np.full(rad.model.mesh.shape, float(TAU_CO_FORM)), "s")

    chem = run_chemistry(
        rad,
        model="carbon_reduced",
        config={
            "skip_shielding": False,
            "shielding_iter": 1,
            "tau_form": tau_form,
        },
        write=False,
    )

    nH = rad.ensure_nH()
    sigma = rad.ensure_sigma_d_per_H()

    nH_cm3_arr = _as_1d(nH, "cm^-3")
    sigma_cm2_arr = _as_1d(sigma, "cm^2")

    Av = _compute_Av(rad.model.mesh, nH_cm3_arr.reshape(rad.model.mesh.shape), sigma_cm2_arr.reshape(rad.model.mesh.shape)).reshape(-1)

    nH2_cm3 = _as_1d(rad.nH2, "cm^-3")
    nHI_cm3 = _as_1d(rad.nH_atom, "cm^-3")

    nco_gas_cm3 = _as_1d(rad.nco_gas, "cm^-3")
    nco_ice_cm3 = _as_1d(rad.nco_ice, "cm^-3")

    nCplus_cm3 = _as_1d(rad.nCplus, "cm^-3")
    nC_cm3 = _as_1d(rad.nC, "cm^-3")
    ne_cm3 = _as_1d(rad.ne, "cm^-3")

    theta_co = _as_1d(rad.theta_co, "dimensionless")

    N_H2 = _column_to_outer_boundary(rad.model.mesh, nH2_cm3.reshape(rad.model.mesh.shape))
    theta_h2 = h2_self_shielding_db96(N_H2, b5=2.0).reshape(-1)

    rates = compute_carbon_reduced_rates(
        nH=nH,
        T=rad.dust_temperature,
        chi=rad.chi,
        theta_co=rad.theta_co,
        sigma_d_per_H=sigma,
        min_rate=float(chem.meta.get("min_rate", 0.0)),
    )

    tau_form_s = _as_1d(tau_form, "s")
    nco_total_cm3 = nco_gas_cm3 + nco_ice_cm3
    nco_max = float(chem.meta.get("Xco_tot", 0.0)) * nH_cm3_arr

    R_form = np.where(tau_form_s > 0.0, (nco_max - nco_total_cm3) / tau_form_s, np.nan)
    R_pdiss = _as_1d(rad.k_diss_co, "1/s") * nco_gas_cm3

    R_freeze = _as_1d(rates.k_fo, "1/s") * nco_gas_cm3
    R_desorb = _as_1d(rates.k_td, "1/s") * nco_ice_cm3
    R_pdes = _as_1d(chem.fields["R_pd"], "cm^-3/s")

    now = datetime.now().isoformat(timespec="seconds")
    commit = _get_git_commit(Path(__file__).resolve().parents[1])

    sigma_d_per_H_cm2 = 1.0 / (1.086 * float(_NH_PER_AV_CM2))
    params_box = [
        f"case=slab_carbon_reduced",
        f"nH={nH_cm3:.3e} cm^-3",
        f"chi0={chi0}",
        f"Tdust={Tdust_K} K",
        f"sigma_d/H={sigma_d_per_H_cm2:.3e} cm^2",
        f"time={now}",
        f"commit={commit}",
    ]

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "case": "slab_carbon_reduced",
                "time": now,
                "commit": commit,
                "n_cells": int(n_cells),
                "nH_cm3": float(nH_cm3),
                "chi0": float(chi0),
                "Tdust_K": float(Tdust_K),
                "chem_meta": chem.meta,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    _plot_abundances_vs_Av(
        out_png=out_dir / "abundances_vs_Av.png",
        Av=Av,
        nH_cm3=nH_cm3_arr,
        nH2_cm3=nH2_cm3,
        nco_gas_cm3=nco_gas_cm3,
        nco_ice_cm3=nco_ice_cm3,
        nCplus_cm3=nCplus_cm3,
        nC_cm3=nC_cm3,
        ne_cm3=ne_cm3,
        params_box=params_box,
    )

    _plot_shielding_fields_vs_Av(
        out_png=out_dir / "shielding_fields_vs_Av.png",
        Av=Av,
        theta_co=theta_co,
        theta_h2=theta_h2,
        params_box=params_box,
    )

    _plot_rates_vs_Av(
        out_png=out_dir / "rates_vs_Av.png",
        Av=Av,
        R_form=R_form,
        R_pdiss=R_pdiss,
        R_freeze=R_freeze,
        R_desorb=R_desorb,
        R_pdes=R_pdes,
        params_box=params_box,
    )

    _plot_carbon_budget_vs_Av(
        out_png=out_dir / "carbon_budget_vs_Av.png",
        Av=Av,
        nH_cm3=nH_cm3_arr,
        nCplus_cm3=nCplus_cm3,
        nC_cm3=nC_cm3,
        nco_total_cm3=nco_total_cm3,
        params_box=params_box,
    )

    return {
        "out_dir": str(out_dir),
        "pngs": [
            "abundances_vs_Av.png",
            "shielding_fields_vs_Av.png",
            "rates_vs_Av.png",
            "carbon_budget_vs_Av.png",
        ],
    }


def _plot_convergence_metric(out_png: Path, metrics: list[float], *, params_box: list[str]):
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    if len(metrics) == 0:
        ax.text(0.5, 0.5, "No metrics recorded", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        raise RuntimeError("No convergence metrics recorded")

    iters = np.arange(1, len(metrics) + 1, dtype=int)
    ax.plot(iters, np.asarray(metrics, dtype=float), marker="o", linestyle="-", color="k")
    ax.set_xticks(iters)
    ax.set_xlabel("outer iteration transition")
    ax.set_ylabel("max rel dT")
    ax.grid(True, alpha=0.3)

    table_lines = [f"{i}: {m:.3e}" for i, m in zip(iters.tolist(), metrics)]
    _annotate_params(ax, params_box + ["metrics:"] + table_lines)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_Tgas_vs_Av_by_iteration(out_png: Path, Av: np.ndarray, Tgas_by_iter: list[np.ndarray], *, params_box: list[str]):
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, Tg in enumerate(Tgas_by_iter):
        lw = 1.0
        alpha = 0.6
        if i == len(Tgas_by_iter) - 1:
            lw = 2.2
            alpha = 1.0
        ax.plot(Av, Tg, linewidth=lw, alpha=alpha, label=f"iter {i}")

    ax.set_xlabel("Av")
    ax.set_ylabel("Tgas [K]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _annotate_params(ax, params_box)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_abundances_vs_Av_final(out_png: Path, Av: np.ndarray, nH_cm3: np.ndarray, fields: dict[str, np.ndarray], *, params_box: list[str]):
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    def _x(n):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where(nH_cm3 > 0.0, n / nH_cm3, np.nan))

    ax.plot(Av, _x(fields["nH2"]) , label="H2")
    ax.plot(Av, _x(fields["nco_gas"]) , label="CO(gas)")
    ax.plot(Av, _x(fields["nco_ice"]) , label="CO(ice)")
    ax.plot(Av, _x(fields["nco_total"]) , label="CO(total)")
    ax.plot(Av, _x(fields["nCplus"]) , label="C+")
    ax.plot(Av, _x(fields["nC"]) , label="C")
    ax.plot(Av, _x(fields["ne"]) , label="e")

    ax.set_xlabel("Av")
    ax.set_ylabel("log10(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _annotate_params(ax, params_box)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_heating_cooling_budget(out_png: Path, Av: np.ndarray, therm_fields: dict, *, params_box: list[str]):
    plt = _setup_matplotlib()

    def _as_rate(name: str) -> np.ndarray:
        if name not in therm_fields:
            raise KeyError(f"Missing thermal field {name!r}; set thermal_config={'store_terms': True}")
        return _as_1d(therm_fields[name], "erg/(cm^3*s)")

    rate_cr = _as_rate("rate_cosmic_ray")
    rate_pe = _as_rate("rate_photoelectric")
    rate_cii = _as_rate("rate_cii")
    rate_ci = _as_rate("rate_ci")
    rate_oi = _as_rate("rate_oi")
    rate_co = _as_rate("rate_co")
    rate_gd = _as_rate("rate_gas_dust")

    heating = rate_cr + rate_pe
    cooling_mag = (-rate_cii) + (-rate_ci) + (-rate_oi) + (-rate_co)

    net = rate_cr + rate_pe + rate_cii + rate_ci + rate_oi + rate_co + rate_gd

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    def _log_mag(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where(x > 0.0, x, np.nan))

    ax0.plot(Av, _log_mag(rate_cr), label="CR heating")
    ax0.plot(Av, _log_mag(rate_pe), label="PE heating")
    ax0.plot(Av, _log_mag(-rate_cii), label="CII cooling")
    ax0.plot(Av, _log_mag(-rate_ci), label="CI cooling")
    ax0.plot(Av, _log_mag(-rate_oi), label="OI cooling")
    ax0.plot(Av, _log_mag(-rate_co), label="CO cooling")

    ax0.set_ylabel("log10(|rate|) [erg cm^-3 s^-1]")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=8)

    ax1.plot(Av, net, color="k", label="net")
    ax1.axhline(0.0, color="0.5", linestyle="--")
    ax1.set_xlabel("Av")
    ax1.set_ylabel("net [erg cm^-3 s^-1]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    _annotate_params(ax0, params_box)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _case_B_thermochemistry_convergence(
    out_dir: Path, *, n_cells: int = 64, n_iter: int = 5, Av_max: float = 10.0
):
    from diskbridge.chemistry.api import run_chemistry
    from diskbridge.chemistry.thermal import run_thermal

    nH_cm3 = 1.0e5
    chi0 = 1.0
    Tdust_K = 20.0

    rad = _build_radmodel_1d(
        n_cells=n_cells,
        nH_cm3=nH_cm3,
        chi=chi0,
        Tdust_K=Tdust_K,
        Av_max=float(Av_max),
    )

    metrics: list[float] = []
    Tgas_by_iter: list[np.ndarray] = []

    prev_T = None

    therm_last = None

    for it in range(int(n_iter)):
        run_chemistry(rad, model="carbon_reduced", config={"skip_shielding": True}, write=False)
        therm = run_thermal(rad, model="thermal_balance", config={"store_terms": True}, write=False)
        therm_last = therm

        T = _as_1d(therm.tgas, "K")
        Tgas_by_iter.append(T)

        if prev_T is not None:
            dt = np.abs(T - prev_T)
            denom = np.where(prev_T != 0.0, np.abs(prev_T), np.inf)
            metrics.append(float(np.max(dt / denom)))

        prev_T = T

    run_chemistry(rad, model="carbon_reduced", config={"skip_shielding": True}, write=False)

    nH = rad.ensure_nH()
    sigma = rad.ensure_sigma_d_per_H()

    nH_cm3_arr = _as_1d(nH, "cm^-3")
    sigma_cm2_arr = _as_1d(sigma, "cm^2")

    Av = _compute_Av(rad.model.mesh, nH_cm3_arr.reshape(rad.model.mesh.shape), sigma_cm2_arr.reshape(rad.model.mesh.shape)).reshape(-1)

    fields = {
        "nH2": _as_1d(rad.nH2, "cm^-3"),
        "nco_gas": _as_1d(rad.nco_gas, "cm^-3"),
        "nco_ice": _as_1d(rad.nco_ice, "cm^-3"),
        "nco_total": _as_1d(rad.nco_gas, "cm^-3") + _as_1d(rad.nco_ice, "cm^-3"),
        "nCplus": _as_1d(rad.nCplus, "cm^-3"),
        "nC": _as_1d(rad.nC, "cm^-3"),
        "ne": _as_1d(rad.ne, "cm^-3"),
    }

    now = datetime.now().isoformat(timespec="seconds")
    commit = _get_git_commit(Path(__file__).resolve().parents[1])

    params_box = [
        f"case=thermochemistry_convergence",
        f"nH={nH_cm3:.3e} cm^-3",
        f"chi0={chi0}",
        f"Tdust={Tdust_K} K",
        f"n_iter={int(n_iter)}",
        f"time={now}",
        f"commit={commit}",
    ]

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "case": "thermochemistry_convergence",
                "time": now,
                "commit": commit,
                "n_cells": int(n_cells),
                "n_iter": int(n_iter),
                "nH_cm3": float(nH_cm3),
                "chi0": float(chi0),
                "Tdust_K": float(Tdust_K),
                "metrics": list(metrics),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    _plot_convergence_metric(out_dir / "convergence_metric.png", metrics, params_box=params_box)
    _plot_Tgas_vs_Av_by_iteration(out_dir / "Tgas_vs_Av_by_iteration.png", Av, Tgas_by_iter, params_box=params_box)
    _plot_abundances_vs_Av_final(
        out_dir / "abundances_vs_Av_final.png",
        Av,
        nH_cm3_arr,
        fields,
        params_box=params_box,
    )

    if therm_last is None:
        raise RuntimeError("No thermal result produced")

    _plot_heating_cooling_budget(
        out_dir / "heating_cooling_budget_vs_Av_final.png",
        Av,
        therm_last.fields,
        params_box=params_box,
    )

    return {
        "out_dir": str(out_dir),
        "pngs": [
            "convergence_metric.png",
            "Tgas_vs_Av_by_iteration.png",
            "abundances_vs_Av_final.png",
            "heating_cooling_budget_vs_Av_final.png",
        ],
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / "validation_out"
    out_root.mkdir(parents=True, exist_ok=True)

    results = []

    caseA_dir = out_root / "case_A_slab_carbon_reduced"
    caseA_dir.mkdir(parents=True, exist_ok=True)
    results.append(_case_A_slab_carbon_reduced(caseA_dir))

    caseB_dir = out_root / "case_B_thermochemistry_convergence"
    caseB_dir.mkdir(parents=True, exist_ok=True)
    results.append(_case_B_thermochemistry_convergence(caseB_dir))

    print(str(out_root))
    for r in results:
        print(r["out_dir"])
        for p in r["pngs"]:
            print(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
