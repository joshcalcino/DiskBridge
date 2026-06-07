"""Plot GOW17 diagnostic fields from the PRIZMO disk validation snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from diskbridge._constants import M_H
from diskbridge._units import Quantity


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
SNAPSHOT = OUT_DIR / "prizmo_diskbridge_full_disk.h5"
METADATA = OUT_DIR / "metadata.json"
DIAGNOSTIC_DIR = OUT_DIR / "gow17_diagnostics"
ABUNDANCE_FLOOR = 1.0e-30


def _read_field(h5: h5py.File, name: str) -> np.ndarray:
    path = f"gas/fields/{name}/data"
    if path not in h5:
        raise KeyError(f"Snapshot is missing field {name!r}")
    return np.asarray(h5[path], dtype=np.float64)[:, :, 0]


def _read_grid(h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(h5["mesh/axes/r/centers"], dtype=np.float64)
    theta = np.asarray(h5["mesh/axes/theta/centers"], dtype=np.float64)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    au = float(Quantity(1.0, "au").to("cm").magnitude)
    return rr * np.sin(tt) / au, rr * np.cos(tt) / au


def _positive_log_limits(values: list[np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate(
        [
            np.asarray(value, dtype=float)[
                np.isfinite(value) & (np.asarray(value, dtype=float) > 0.0)
            ]
            for value in values
        ]
    )
    if flat.size == 0:
        return ABUNDANCE_FLOOR, 1.0
    vmax = float(np.nanpercentile(flat, 99.0))
    vmax = max(vmax, float(np.nanmax(flat)))
    vmin = max(float(np.nanpercentile(flat, 1.0)), ABUNDANCE_FLOOR, vmax * 1.0e-10)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return ABUNDANCE_FLOOR, 1.0
    return vmin, vmax


def _plot_panel(
    R: np.ndarray,
    z: np.ndarray,
    panels: list[dict],
    out: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import LogNorm, SymLogNorm

    coords = np.column_stack([R.ravel(), z.ravel()])
    _, keep = np.unique(np.round(coords, decimals=9), axis=0, return_index=True)
    tri = mtri.Triangulation(coords[keep, 0], coords[keep, 1])

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    for ax, panel in zip(axes.ravel(), panels):
        values = np.asarray(panel["values"], dtype=float)
        vals = values.ravel()[keep]
        mode = panel.get("mode", "linear")
        if mode == "log":
            vmin, vmax = panel.get("limits") or _positive_log_limits([values])
            vals = np.where(np.isfinite(vals) & (vals > 0.0), vals, vmin)
            norm = LogNorm(vmin=vmin, vmax=vmax)
            levels = np.geomspace(vmin, vmax, 96)
        elif mode == "symlog":
            finite = vals[np.isfinite(vals)]
            vmax = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
            vmax = max(vmax, 1.0e-40)
            linthresh = max(vmax * 1.0e-4, 1.0e-40)
            norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
            levels = np.linspace(-vmax, vmax, 97)
            vals = np.where(np.isfinite(vals), vals, 0.0)
        else:
            finite = vals[np.isfinite(vals)]
            vmin = float(np.nanpercentile(finite, 1.0)) if finite.size else 0.0
            vmax = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0
            norm = None
            levels = np.linspace(vmin, vmax, 96)
            vals = np.where(np.isfinite(vals), vals, vmin)

        image = ax.tricontourf(
            tri,
            vals,
            levels=levels,
            cmap=panel.get("cmap", "viridis"),
            norm=norm,
            extend="both",
        )
        ax.set_title(panel["title"])
        ax.set_xlabel("R [au]")
        ax.set_ylabel("z [au]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(left=0.0)
        ax.axhline(0.0, color="0.2", linewidth=0.5)
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.suptitle(title)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _status_counts(values: np.ndarray) -> dict[str, int]:
    codes, counts = np.unique(values.astype(np.int64), return_counts=True)
    return {str(int(code)): int(count) for code, count in zip(codes, counts)}


def main() -> None:
    """Create PRIZMO-disk GOW17 diagnostic plots and a compact summary."""
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(SNAPSHOT, "r") as h5:
        R, z = _read_grid(h5)
        density = _read_field(h5, "density")
        nH = density / (1.4 * float(M_H))
        chi = _read_field(h5, "chi")
        chi_broad = _read_field(h5, "chem_chi_broad")
        chi_eff = _read_field(h5, "chem_chi_eff")
        g_c_ion = _read_field(h5, "chem_G_C_ion_actual")
        g_co_diss = _read_field(h5, "chem_G_CO_diss_actual")
        cplus = _read_field(h5, "abundance_c+")
        co = _read_field(h5, "abundance_co")
        status = _read_field(h5, "chem_status")
        tgas = _read_field(h5, "chem_Tgas")
        tdust = _read_field(h5, "dust_temperature")
        tgd = _read_field(h5, "chem_Tdust_gd_surface_weighted")
        exchange = _read_field(h5, "chem_gas_dust_exchange_signed")
        rhs = _read_field(h5, "chem_gow17_rhs_residual_max")

    uv_limits = _positive_log_limits([chi, chi_broad, chi_eff, g_c_ion, g_co_diss])
    abundance_limits = _positive_log_limits([cplus, co])
    ratio = np.divide(tgas, tdust, out=np.full_like(tgas, np.nan), where=tdust > 0.0)
    dense = nH >= float(np.nanpercentile(nH[np.isfinite(nH)], 90.0))

    _plot_panel(
        R,
        z,
        [
            {"title": "nH [cm^-3]", "values": nH, "mode": "log"},
            {"title": "input chi", "values": chi, "mode": "log", "limits": uv_limits},
            {"title": "chi broad used", "values": chi_broad, "mode": "log", "limits": uv_limits},
            {"title": "chi_eff / G_CO_diss_actual", "values": chi_eff, "mode": "log", "limits": uv_limits},
            {"title": "G_C_ion_actual", "values": g_c_ion, "mode": "log", "limits": uv_limits},
            {"title": "G_CO_diss_actual", "values": g_co_diss, "mode": "log", "limits": uv_limits},
        ],
        DIAGNOSTIC_DIR / "uv_shielding_diagnostics.png",
        title="PRIZMO Disk GOW17 UV And Shielding Diagnostics",
    )

    _plot_panel(
        R,
        z,
        [
            {"title": "C+ abundance", "values": cplus, "mode": "log", "limits": abundance_limits},
            {"title": "CO abundance", "values": co, "mode": "log", "limits": abundance_limits},
            {"title": "final chem_status", "values": status, "mode": "linear", "cmap": "coolwarm"},
            {"title": "Tgas / Tdust", "values": ratio, "mode": "log"},
            {"title": "Tgas - gas-dust target [K]", "values": tgas - tgd, "mode": "symlog", "cmap": "coolwarm"},
            {"title": "gas-dust exchange [erg s^-1]", "values": exchange, "mode": "symlog", "cmap": "coolwarm"},
        ],
        DIAGNOSTIC_DIR / "cplus_temperature_status_diagnostics.png",
        title="PRIZMO Disk GOW17 C+, Temperature, And Status Diagnostics",
    )

    _plot_panel(
        R,
        z,
        [
            {"title": "RHS residual max", "values": rhs, "mode": "log"},
            {"title": "C+ abundance", "values": cplus, "mode": "log", "limits": abundance_limits},
            {"title": "G_C_ion_actual", "values": g_c_ion, "mode": "log", "limits": uv_limits},
            {"title": "chi_eff", "values": chi_eff, "mode": "log", "limits": uv_limits},
            {"title": "Tgas / Tdust", "values": ratio, "mode": "log"},
            {"title": "final chem_status", "values": status, "mode": "linear", "cmap": "coolwarm"},
        ],
        DIAGNOSTIC_DIR / "residual_cplus_context.png",
        title="PRIZMO Disk GOW17 Residual And C+ Context",
    )

    metadata = {}
    if METADATA.exists():
        metadata = json.loads(METADATA.read_text())
    gow17_diag = (
        metadata.get("diskbridge", {})
        .get("result_meta", {})
        .get("gow17_diagnostics", {})
    )

    summary = {
        "snapshot": str(SNAPSHOT),
        "plots": [
            str(DIAGNOSTIC_DIR / "uv_shielding_diagnostics.png"),
            str(DIAGNOSTIC_DIR / "cplus_temperature_status_diagnostics.png"),
            str(DIAGNOSTIC_DIR / "residual_cplus_context.png"),
        ],
        "final_status_counts": _status_counts(status),
        "metadata_accumulated_status_hist": gow17_diag.get("accumulated_status_hist"),
        "metadata_tevol_max_cells_total": gow17_diag.get("tevol_max_cells_total"),
        "metadata_n_fail": metadata.get("diskbridge", {}).get("result_meta", {}).get("n_fail"),
        "dense_mask": {
            "definition": "top 10 percent by nH",
            "n_cells": int(np.sum(dense)),
            "nH_min_cm3": float(np.nanmin(nH[dense])),
            "cplus_median": float(np.nanmedian(cplus[dense])),
            "co_median": float(np.nanmedian(co[dense])),
            "chi_eff_median": float(np.nanmedian(chi_eff[dense])),
            "G_C_ion_actual_median": float(np.nanmedian(g_c_ion[dense])),
            "Tgas_over_Tdust_median": float(np.nanmedian(ratio[dense])),
            "rhs_residual_max_median": float(np.nanmedian(rhs[dense])),
            "final_status_counts": _status_counts(status[dense]),
        },
        "all_cells": {
            "cplus_median": float(np.nanmedian(cplus)),
            "chi_eff_median": float(np.nanmedian(chi_eff)),
            "Tgas_over_Tdust_median": float(np.nanmedian(ratio)),
            "rhs_residual_max": float(np.nanmax(rhs)),
        },
    }
    (DIAGNOSTIC_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
