from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import traceback

import numpy as np

import diskbridge._gow17 as _gow17


@dataclass(frozen=True)
class Gow17Fig2Config:
    nH_cm3: float = 100.0
    chi0: float = 1.0
    xi_CR_s: float = 2.0e-16
    ngrid: int = 2000
    Zdg: float = 1.0
    NH_min: float = 1.0e17
    NH_total: float = 1.0e22
    logNH: bool = True
    field_geo: int = 0
    isdust: bool = True
    isfsH2: bool = True
    isfsCO: bool = True
    isfsC: bool = True
    gradv: float = 9.0e-14
    NCOeff_global: bool = True
    bCO_L: bool = True
    reltol: float = 1.0e-2
    abstol0: float = 1.0e-9
    mxsteps: int = 5000000
    maxord: int = 3
    tolfac: float = 10.0
    tmin_s: float = 0.1 * 1.0e6 * 3.16e7
    tmax_s: float = 2000.0 * 1.0e6 * 3.16e7
    const_temp: bool = False
    Tgas_K: float = 100.0

    fH2gr: float = 1.0
    fHplusgr: float = 0.6
    fCplusgr: float = 0.6
    fHeplusgr: float = 0.6
    fSplusgr: float = 0.6
    fSiplusgr: float = 0.6
    fCplusCR: float = 1.0

    userJac: bool = False


_SPEC_LIST_REF = [
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


def _ref_dir(repo_root: Path) -> Path:
    return repo_root / "other_codes" / "pdr" / "out_example_simple"


def _load_reference(repo_root: Path, slab_id: int) -> tuple[np.ndarray, np.ndarray]:
    ref = _ref_dir(repo_root)
    NH = np.loadtxt(ref / "colH_arr.dat")
    slab = np.loadtxt(ref / f"slab{slab_id:06d}.dat")
    return NH, slab


def _initial_conditions(temp_K: float) -> tuple[np.ndarray, np.ndarray]:
    y0 = np.zeros(_gow17.N_Y, dtype=np.float64)
    y0[_gow17.I_HEP] = 1.450654e-08
    y0[_gow17.I_H3P] = 2.681411e-07
    y0[_gow17.I_CP] = 1.0e-4
    y0[_gow17.I_CO] = 1.0e-7
    y0[_gow17.I_H2] = 0.1

    Cv = 1.5 * 1.380649e-16 * ((1.0 - 2.0 * 0.1) + 0.1 + _gow17.XHE + 0.0)
    y0[_gow17.I_E] = float(temp_K) * float(Cv)

    abstol = np.full(_gow17.N_Y, 1.0e-9, dtype=np.float64)
    abstol[_gow17.I_HEP] = 1.0e-15
    abstol[_gow17.I_OHX] = 1.0e-15
    abstol[_gow17.I_CHX] = 1.0e-15
    abstol[_gow17.I_CO] = 1.0e-15
    abstol[_gow17.I_CP] = 1.0e-15
    abstol[_gow17.I_HCOP] = max(1.0e-9, 1.0e-20)
    abstol[_gow17.I_H2] = 1.0e-8
    abstol[_gow17.I_HP] = 1.0e-15
    abstol[_gow17.I_H3P] = 1.0e-15
    abstol[_gow17.I_H2P] = 1.0e-15
    abstol[_gow17.I_E] = float(Cv) * 1.0

    return y0, abstol


def _write_inputs(out_dir: Path, cfg: Gow17Fig2Config, *, slab_id: int, rtol_fail: float) -> None:
    inputs = {
        "job": "gow17_fig2",
        "time": datetime.now().isoformat(timespec="seconds"),
        "slab_id": int(slab_id),
        "rtol_fail": float(rtol_fail),
        "config": cfg.__dict__,
    }
    (out_dir / "inputs.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True), encoding="utf-8"
    )


def _placeholder_error_plot(out_dir: Path, title: str, text: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.01, 0.99, text, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "error.png", dpi=150)
    plt.close(fig)


def run(out_dir: Path, *, slab_id: int = 0, rtol_fail: float = 1.0e-5) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Gow17Fig2Config()

    _write_inputs(out_dir, cfg, slab_id=slab_id, rtol_fail=rtol_fail)

    error: str | None = None
    max_rel_all: float | None = None

    try:
        NH_ref, slab_ref = _load_reference(repo_root, slab_id)
        Av = NH_ref / 1.87e21

        y0, abstol = _initial_conditions(cfg.Tgas_K)

        slab = _gow17.solve_slab_1d_equilibrium(
            nH=float(cfg.nH_cm3),
            G0=float(cfg.chi0) * 2.0,
            ngrid=int(cfg.ngrid),
            NH_total=float(cfg.NH_total / cfg.Zdg),
            logNH=bool(cfg.logNH),
            NH_min=float(cfg.NH_min / cfg.Zdg),
            field_geo=int(cfg.field_geo),
            isdust=bool(cfg.isdust),
            isfsH2=bool(cfg.isfsH2),
            isfsCO=bool(cfg.isfsCO),
            isfsC=bool(cfg.isfsC),
            Zg=float(cfg.Zdg),
            Zd=float(cfg.Zdg),
            ion_rate=float(cfg.xi_CR_s),
            reltol=float(cfg.reltol),
            abstol=abstol,
            mxsteps=int(cfg.mxsteps),
            maxord=int(cfg.maxord),
            tolfac=float(cfg.tolfac),
            tmin=float(cfg.tmin_s),
            tmax=float(cfg.tmax_s),
            verbose=False,
            y0=y0,
            const_temp=bool(cfg.const_temp),
            Tgas=float(cfg.Tgas_K),
            gradv=float(cfg.gradv),
            NCOeff_global=bool(cfg.NCOeff_global),
            bCO_L=bool(cfg.bCO_L),
            fH2gr=float(cfg.fH2gr),
            fHplusgr=float(cfg.fHplusgr),
            fCplusgr=float(cfg.fCplusgr),
            fHeplusgr=float(cfg.fHeplusgr),
            fSplusgr=float(cfg.fSplusgr),
            fSiplusgr=float(cfg.fSiplusgr),
            fCplusCR=float(cfg.fCplusCR),
            co_sigma_d_per_H_ref=0.0,
            co_E_bind_co=0.0,
            co_nu0_co=0.0,
            co_F_DRAINE=0.0,
            co_Y_CO=0.0,
            co_N_SURF=0.0,
            co_N_LAY=0,
            userJac=bool(cfg.userJac),
        )

        y = np.asarray(slab["y"], dtype=np.float64)
        if y.shape[0] != slab_ref.shape[0]:
            raise RuntimeError(
                f"slab length mismatch: got {y.shape[0]} vs ref {slab_ref.shape[0]}"
            )

        idx_db = {
            "He+": _gow17.I_HEP,
            "OHx": _gow17.I_OHX,
            "CHx": _gow17.I_CHX,
            "CO": _gow17.I_CO,
            "C+": _gow17.I_CP,
            "HCO+": _gow17.I_HCOP,
            "H2": _gow17.I_H2,
            "H+": _gow17.I_HP,
            "H3+": _gow17.I_H3P,
            "H2+": _gow17.I_H2P,
            "S+": _gow17.I_SP,
            "Si+": _gow17.I_SIP,
            "O+": _gow17.I_OP,
            "E": _gow17.I_E,
        }

        max_rel_by_species: dict[str, float] = {}
        med_rel_by_species: dict[str, float] = {}
        p90_rel_by_species: dict[str, float] = {}

        for j, name in enumerate(_SPEC_LIST_REF):
            ref = np.asarray(slab_ref[:, j], dtype=np.float64)
            db = np.asarray(y[:, idx_db[name]], dtype=np.float64)

            nonzero = ref != 0.0
            if not np.any(nonzero):
                max_rel = 0.0
                med_rel = 0.0
                p90_rel = 0.0
            else:
                rel = np.abs(db[nonzero] - ref[nonzero]) / np.abs(ref[nonzero])
                max_rel = float(np.max(rel))
                med_rel = float(np.median(rel))
                p90_rel = float(np.percentile(rel, 90.0))

            max_rel_by_species[name] = max_rel
            med_rel_by_species[name] = med_rel
            p90_rel_by_species[name] = p90_rel

        max_rel_all = float(max(max_rel_by_species.values()))

        plt = _setup_matplotlib()

        params_box = [
            f"nH={cfg.nH_cm3:.3e} cm^-3",
            f"G0={cfg.chi0:g}",
            f"xi_CR={cfg.xi_CR_s:.2e} 1/s",
            f"NH=[{cfg.NH_min:.1e},{cfg.NH_total:.1e}] cm^-2",
            f"grid={cfg.ngrid}",
            f"reltol={cfg.reltol:.1e}",
            f"rtol_fail={rtol_fail:.1e}",
        ]

        for name in ["CO", "C+", "H2"]:
            j = _SPEC_LIST_REF.index(name)
            ref = np.asarray(slab_ref[:, j], dtype=np.float64)
            db = np.asarray(y[:, idx_db[name]], dtype=np.float64)
            nonzero = ref != 0.0

            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            with np.errstate(divide="ignore", invalid="ignore"):
                y_ref = np.log10(np.where(ref > 0.0, ref, np.nan))
                y_db = np.log10(np.where(db > 0.0, db, np.nan))
            ax.plot(Av, y_ref, label="ref", color="k")
            ax.plot(Av, y_db, label="db", color="tab:blue")
            ax.set_xlabel("Av")
            ax.set_ylabel(f"log10(x_{name})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            _annotate(ax, params_box)
            fig.tight_layout()
            fig.savefig(out_dir / f"profile_{name}.png", dpi=150)
            plt.close(fig)

            if np.any(nonzero):
                rel = np.abs(db[nonzero] - ref[nonzero]) / np.abs(ref[nonzero])
                fig, ax = plt.subplots(figsize=(7.0, 4.5))
                ax.semilogy(
                    Av[nonzero],
                    rel,
                    color="k",
                    marker=".",
                    linestyle="none",
                    markersize=2,
                )
                ax.set_xlabel("Av")
                ax.set_ylabel(f"abs rel error {name}")
                ax.grid(True, alpha=0.3)
                _annotate(ax, params_box)
                fig.tight_layout()
                fig.savefig(out_dir / f"residual_{name}.png", dpi=150)
                plt.close(fig)

        summary = {
            "job": "gow17_fig2",
            "time": datetime.now().isoformat(timespec="seconds"),
            "slab_id": int(slab_id),
            "rtol_fail": float(rtol_fail),
            "config": cfg.__dict__,
            "pass": True,
            "max_rel_by_species": max_rel_by_species,
            "p90_rel_by_species": p90_rel_by_species,
            "median_rel_by_species": med_rel_by_species,
            "max_rel_all": float(max_rel_all),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

        if float(max_rel_all) > float(rtol_fail):
            error = f"gow17_fig2 failed: max_rel_all={max_rel_all:.3e} > {rtol_fail:.3e}"
            summary["pass"] = False
            summary["error"] = error
            (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )

    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        error = f"{type(e).__name__}: {e}"
        _placeholder_error_plot(out_dir, "gow17_fig2 failed", error + "\n\n" + tb)
        summary = {
            "job": "gow17_fig2",
            "time": datetime.now().isoformat(timespec="seconds"),
            "slab_id": int(slab_id),
            "rtol_fail": float(rtol_fail),
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
    out_root = repo_root / "validation_out" / "gow17_fig2"
    try:
        run(out_root)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
