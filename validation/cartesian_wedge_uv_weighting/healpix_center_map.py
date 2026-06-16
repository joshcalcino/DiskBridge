"""Plot HEALPix column maps seen by one cell in the wedge validation."""

# db-keywords: shielding, healpix-columns, validation, plotting, model, mesh
# db-role: validation
# db-scope: validation
# db-purpose: HEALPix central-cell column diagnostics for the Cartesian wedge validation.

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import healpy as hp  # type: ignore
import numpy as np

import diskbridge
from diskbridge.chemistry.shielding.healpix_columns import compute_column_rays_healpix

from run import (
    CONFIG_FILE,
    ROOT,
    M_H_G,
    _load_config,
    _nearest_index,
    _safe_ratio,
    _setup_matplotlib,
    build_density_structure,
    build_mesh,
)


MASS_PER_H_NUCLEUS_G = 1.4 * M_H_G


def _parse_nsides(s: str) -> list[int]:
    nsides: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            nsides.append(int(part))
    if not nsides:
        raise ValueError("--nsides must contain at least one integer")
    for nside in nsides:
        if not hp.isnsideok(int(nside)):
            raise ValueError(f"Invalid HEALPix nside={nside}")
    return nsides


def _safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _select_cell(
    *,
    mode: str,
    fields: dict[str, np.ndarray],
    cfg,
    ix: int | None,
    iy: int | None,
    iz: int | None,
) -> tuple[int, int, int, str]:
    explicit = (ix, iy, iz)
    if any(value is not None for value in explicit):
        if any(value is None for value in explicit):
            raise ValueError("Provide all of --ix, --iy, and --iz for an explicit cell")
        cell = (int(ix), int(iy), int(iz))
        shape = fields["nH_cm3"].shape
        for value, size, name in zip(cell, shape, ("ix", "iy", "iz"), strict=True):
            if value < 0 or value >= int(size):
                raise ValueError(f"{name}={value} is outside density shape {shape}")
        return cell[0], cell[1], cell[2], "explicit"

    if mode == "sphere-center":
        geom = cfg.geometry
        return (
            _nearest_index(fields["x_au"], float(geom.circle_center_x_au)),
            _nearest_index(fields["y_au"], float(geom.circle_center_y_au)),
            _nearest_index(fields["z_au"], float(geom.circle_center_z_au)),
            "sphere-center",
        )
    if mode == "box-center":
        shape = fields["nH_cm3"].shape
        return int(shape[0] // 2), int(shape[1] // 2), int(shape[2] // 2), "box-center"
    if mode == "max-density":
        cell = np.unravel_index(int(np.nanargmax(fields["nH_cm3"])), fields["nH_cm3"].shape)
        return int(cell[0]), int(cell[1]), int(cell[2]), "max-density"
    raise ValueError(f"Unknown cell mode {mode!r}")


def _plot_density_slices(
    *,
    out_dir: Path,
    cfg,
    fields: dict[str, np.ndarray],
    ix: int,
    iy: int,
    iz: int,
) -> Path:
    plt = _setup_matplotlib()
    nH = np.asarray(fields["nH_cm3"], dtype=np.float64)
    log_nH = np.log10(np.maximum(nH, 1.0e-30))
    vmin = float(np.nanpercentile(log_nH, 1.0))
    vmax = float(np.nanpercentile(log_nH, 99.0))

    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    im0 = axs[0].imshow(
        log_nH[:, :, iz].T,
        origin="lower",
        extent=[cfg.grid.x_min_au, cfg.grid.x_max_au, cfg.grid.y_min_au, cfg.grid.y_max_au],
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    axs[0].plot([fields["x_au"][ix]], [fields["y_au"][iy]], marker="x", color="cyan", mew=1.6)
    axs[0].set_title("x-y")
    axs[0].set_xlabel("x [au]")
    axs[0].set_ylabel("y [au]")

    axs[1].imshow(
        log_nH[:, iy, :].T,
        origin="lower",
        extent=[cfg.grid.x_min_au, cfg.grid.x_max_au, cfg.grid.z_min_au, cfg.grid.z_max_au],
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    axs[1].plot([fields["x_au"][ix]], [fields["z_au"][iz]], marker="x", color="cyan", mew=1.6)
    axs[1].set_title("x-z")
    axs[1].set_xlabel("x [au]")
    axs[1].set_ylabel("z [au]")

    axs[2].imshow(
        log_nH[ix, :, :].T,
        origin="lower",
        extent=[cfg.grid.y_min_au, cfg.grid.y_max_au, cfg.grid.z_min_au, cfg.grid.z_max_au],
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    axs[2].plot([fields["y_au"][iy]], [fields["z_au"][iz]], marker="x", color="cyan", mew=1.6)
    axs[2].set_title("y-z")
    axs[2].set_xlabel("y [au]")
    axs[2].set_ylabel("z [au]")

    cb = fig.colorbar(im0, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("log10(nH [cm^-3])")
    path = out_dir / "density_slices_selected_cell.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_healpix_grid(
    *,
    out_dir: Path,
    label: str,
    nsides: list[int],
    maps: list[np.ndarray],
    title_prefix: str,
    quantity_label: str,
) -> Path:
    log_maps = [np.log10(np.maximum(np.asarray(m, dtype=np.float64), 1.0e-30)) for m in maps]
    all_vals = np.concatenate([m.reshape(-1) for m in log_maps])
    vmin = float(np.nanpercentile(all_vals, 1.0))
    vmax = float(np.nanpercentile(all_vals, 99.0))

    plt = _setup_matplotlib()
    n = len(nsides)
    ncols = int(np.ceil(np.sqrt(float(n))))
    nrows = int(np.ceil(float(n) / float(ncols)))
    fig = plt.figure(figsize=(4.4 * ncols, 3.4 * nrows))

    for i, nside in enumerate(nsides):
        hp.mollview(
            log_maps[i],
            fig=fig.number,
            sub=(nrows, ncols, i + 1),
            title=f"{title_prefix} nside={int(nside)}",
            min=vmin,
            max=vmax,
            cbar=False,
        )
    fig.text(0.5, 0.03, f"log10({quantity_label})", ha="center", va="center")
    path = out_dir / f"healpix_{label}_cell_nH_column_nsides.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def run_healpix_center_map(
    *,
    config_path: Path = CONFIG_FILE,
    out_dir: Path | None = None,
    nsides: list[int] | None = None,
    cell_mode: str = "sphere-center",
    ix: int | None = None,
    iy: int | None = None,
    iz: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Write HEALPix column maps for one selected validation cell."""

    cfg = _load_config(Path(config_path))
    if nsides is None:
        nsides = [1, 2, 4, 8, 16, 32]
    if out_dir is None:
        out_dir = ROOT / "outputs" / "healpix_center_map"
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is non-empty: {out_dir}. "
                "Pass --overwrite to replace these HEALPix diagnostics."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = build_mesh(cfg.grid)
    fields = build_density_structure(cfg)
    nH_cm3 = np.asarray(fields["nH_cm3"], dtype=np.float64)
    rho_g_cm3 = np.asarray(fields["rho_g_cm3"], dtype=np.float64)

    ix, iy, iz, cell_label = _select_cell(
        mode=str(cell_mode),
        fields=fields,
        cfg=cfg,
        ix=ix,
        iy=iy,
        iz=iz,
    )

    candidate_mask = np.zeros(nH_cm3.shape, dtype=bool)
    candidate_mask[ix, iy, iz] = True

    density_plot = _plot_density_slices(out_dir=out_dir, cfg=cfg, fields=fields, ix=ix, iy=iy, iz=iz)

    maps_nH: list[np.ndarray] = []
    maps_rho: list[np.ndarray] = []
    for nside in nsides:
        candidate_idx, _dirs, cols = compute_column_rays_healpix(
            mesh,
            fields={
                "nH": nH_cm3,
                "rho": rho_g_cm3,
            },
            nside=int(nside),
            candidate_mask=candidate_mask,
            progress_chunks=None,
            cache_dir=None,
        )
        if int(candidate_idx.shape[0]) != 1:
            raise RuntimeError(f"Expected one candidate cell, got {int(candidate_idx.shape[0])}")
        maps_nH.append(np.asarray(cols["nH"][0, :], dtype=np.float64))
        maps_rho.append(np.asarray(cols["rho"][0, :], dtype=np.float64))

    safe_cell_label = _safe_label(cell_label)
    map_plot = _plot_healpix_grid(
        out_dir=out_dir,
        label=safe_cell_label,
        nsides=nsides,
        maps=maps_nH,
        title_prefix="N_H",
        quantity_label="N_H [cm^-2]",
    )

    npz_path = out_dir / f"healpix_{safe_cell_label}_cell_columns.npz"
    np.savez_compressed(
        npz_path,
        nsides=np.asarray(nsides, dtype=np.int64),
        cell_index=np.asarray([ix, iy, iz], dtype=np.int64),
        cell_center_au=np.asarray(
            [fields["x_au"][ix], fields["y_au"][iy], fields["z_au"][iz]],
            dtype=np.float64,
        ),
        **{f"nH_nside_{int(nside)}": maps_nH[i] for i, nside in enumerate(nsides)},
        **{f"rho_nside_{int(nside)}": maps_rho[i] for i, nside in enumerate(nsides)},
    )

    summary = {
        "validation": "cartesian_wedge_uv_weighting",
        "diagnostic": "central_cell_healpix_column_map",
        "cell_mode": cell_label,
        "cell_index": [int(ix), int(iy), int(iz)],
        "cell_center_au": [
            float(fields["x_au"][ix]),
            float(fields["y_au"][iy]),
            float(fields["z_au"][iz]),
        ],
        "shape": [int(v) for v in nH_cm3.shape],
        "nH_cell_cm3": float(nH_cm3[ix, iy, iz]),
        "rho_cell_g_cm3": float(rho_g_cm3[ix, iy, iz]),
        "ambient_nH_cm3": float(cfg.density.ambient_nH_cm3),
        "cloud_nH_cm3": float(cfg.density.cloud_nH_cm3),
        "wedge_nH_cm3": float(cfg.density.wedge_nH_cm3),
        "wedge_to_ambient_density_ratio": float(
            _safe_ratio(
                np.asarray([cfg.density.wedge_nH_cm3], dtype=np.float64),
                np.asarray([cfg.density.ambient_nH_cm3], dtype=np.float64),
            )[0]
        ),
        "nsides": [int(nside) for nside in nsides],
        "column_stats": {
            str(int(nside)): {
                "N_H_cm2_min": float(np.nanmin(maps_nH[i])),
                "N_H_cm2_median": float(np.nanmedian(maps_nH[i])),
                "N_H_cm2_max": float(np.nanmax(maps_nH[i])),
                "rho_column_g_cm2_min": float(np.nanmin(maps_rho[i])),
                "rho_column_g_cm2_median": float(np.nanmedian(maps_rho[i])),
                "rho_column_g_cm2_max": float(np.nanmax(maps_rho[i])),
            }
            for i, nside in enumerate(nsides)
        },
        "outputs": {
            "density_slices": str(density_plot),
            "healpix_map": str(map_plot),
            "columns_npz": str(npz_path),
            "summary_json": str(out_dir / f"healpix_{safe_cell_label}_cell_nH_column_summary.json"),
        },
    }
    summary_path = out_dir / f"healpix_{safe_cell_label}_cell_nH_column_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diskbridge._logging.logger.info(f"Wrote {map_plot}")
    diskbridge._logging.logger.info(f"Wrote {summary_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(prog="cartesian_wedge_healpix_center_map")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--nsides", type=str, default="1,2,4,8,16,32")
    parser.add_argument(
        "--cell",
        choices=("sphere-center", "box-center", "max-density"),
        default="sphere-center",
    )
    parser.add_argument("--ix", type=int, default=None)
    parser.add_argument("--iy", type=int, default=None)
    parser.add_argument("--iz", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = run_healpix_center_map(
        config_path=args.config,
        out_dir=args.out_dir,
        nsides=_parse_nsides(args.nsides),
        cell_mode=str(args.cell),
        ix=args.ix,
        iy=args.iy,
        iz=args.iz,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
