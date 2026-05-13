#!/usr/bin/env python3
"""

This script:
- loads the FARGO 3D disk snapshot,
- writes RADMC-3D input files with DiskBridge,
- computes dust temperature,
- computes CO abundance with optional photochemistry switches
  read from params.txt.
"""

import diskbridge
from diskbridge.visualization.profiles import plot_phi_avg_rz_slice
from diskbridge.visualization.diagnostics import make_dust_component_diagnostic_plots
import numpy as np
from diskbridge.model.field import Field
import matplotlib.pyplot as plt
from pathlib import Path
import os


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

WEIGHT_MODES_TO_TEST = ("cell",)


def write_weight_mode_index():
    plot_root = Path("plots")
    plot_root.mkdir(parents=True, exist_ok=True)
    index = """# Leon sim6 weight-mode comparison

This run uses the cell-wise Joos mask criterion.

The `cell/` directory contains `key/`, `supporting/`, and `dust_bins/`
diagnostic plots.
    """
    (plot_root / "README.md").write_text(index)


for WEIGHT_MODE in WEIGHT_MODES_TO_TEST:
    print(
        "Running leon_sim6 dust diagnostic "
        f"weight_mode={WEIGHT_MODE}"
    )

    # Load parameters for this example and update the global params
    p = diskbridge.read_params("params.txt")
    diskbridge.params = p
    
    # Load the FARGO model from the 3D disk data 
    model = diskbridge.load_model(
        "data/",
        reader="fargo",
        file_n=220,
        file_units="cgs"
    )
    
    rho_disk_min = 1e-23 * diskbridge.units('g/cm^3')
    r_max = 200 * diskbridge.units('au')
    
    mesh = model.mesh
    
    PLOT_DIR = Path("plots") / WEIGHT_MODE
    KEY_PLOT_DIR = PLOT_DIR / "key"
    BIN_PLOT_DIR = PLOT_DIR / "dust_bins"
    SUPPORT_PLOT_DIR = PLOT_DIR / "supporting"
    for _plot_dir in (KEY_PLOT_DIR, BIN_PLOT_DIR, SUPPORT_PLOT_DIR):
        _plot_dir.mkdir(parents=True, exist_ok=True)
    
    
    def key_plot(filename):
        return KEY_PLOT_DIR / filename
    
    
    def bin_plot(filename):
        return BIN_PLOT_DIR / filename
    
    
    def supporting_plot(filename):
        return SUPPORT_PLOT_DIR / filename
    
    
    def write_plot_index():
        index = f"""# Leon sim6 plot guide: weight_mode={WEIGHT_MODE}
    
    Start with these:
    
    1. `key/midplane_radial_dust_diagnostics_{WEIGHT_MODE}.png`
       Shows whether the bright inner feature is gas, disc dust, ISM dust, or dust/gas.
    2. `key/dust_components_inner_zoverr_vs_r_{WEIGHT_MODE}_percentile.png`
       Side-by-side inner gas, total dust, disc dust, and ISM dust maps.
    3. `key/dust_density_inner_zoverr_vs_r_{WEIGHT_MODE}_percentile.png`
       Cleaner zoom of the bright inner dust structure.
    4. `key/dust_to_gas_inner_zoverr_vs_r_{WEIGHT_MODE}.png`
       Shows where the dust-to-gas ratio is being boosted or suppressed.
    5. `key/dust_weight_inner_theta_zoverr_vs_r_{WEIGHT_MODE}.png`
       Shows the effective dust disc component mask.
    6. `key/dust_component_masks_inner_zoverr_vs_r_{WEIGHT_MODE}.png`
       Side-by-side view of the actual disk and ISM component masks used for dust.
    
    Supporting plots:
    
    - `supporting/` contains full-domain gas, dust, mask, component-mask, and component-total maps.
    - `dust_bins/` contains one plot per dust-size bin. Use these only when you need
      to inspect which grain sizes are responsible.
    """
        (PLOT_DIR / "README.md").write_text(index)
    
    
    def log_percentile_limits(data, lower=1.0, upper=99.0, max_dyn_range_dex=8.0):
        values = np.asarray(data, dtype=float)
        ok = np.isfinite(values) & (values > 0.0)
        if not np.any(ok):
            raise ValueError("No positive finite values for log percentile limits")
        logv = np.log10(values[ok])
        vmin = float(np.nanpercentile(logv, lower))
        vmax = float(np.nanpercentile(logv, upper))
        if max_dyn_range_dex is not None and np.isfinite(vmax):
            vmin = max(vmin, vmax - float(max_dyn_range_dex))
        return vmin, vmax
    
    
    def phi_avg_log_panel(
        ax,
        model,
        field,
        *,
        title,
        vmin=None,
        vmax=None,
        xlim=None,
        ylim=None,
        cmap="viridis",
    ):
        r_edges_au = model.mesh.edges("r").to("au").magnitude
        theta_edges = model.mesh.edges("theta").to("radian").magnitude
        data = field.data.to("g/cm^3").magnitude
        data_phi = np.mean(data, axis=2)
        z_plot = np.log10(np.maximum(data_phi, np.finfo(np.float64).tiny)).T
        y_edges = np.cos(theta_edges)
        if y_edges[0] > y_edges[-1]:
            y_edges = y_edges[::-1]
            z_plot = z_plot[::-1, :]
        pc = ax.pcolormesh(
            r_edges_au,
            y_edges,
            z_plot,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xscale("log")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("r [au]")
        ax.set_ylabel("z/r")
        return pc
    
    
    def plot_inner_component_comparison(
        *,
        gas_field,
        total_dust_field,
        disk_dust_field,
        ism_dust_field,
        output,
        xlim,
        ylim,
    ):
        panels = [
            ("gas density", gas_field),
            ("total dust", total_dust_field),
            ("disc dust", disk_dust_field),
            ("ISM dust", ism_dust_field),
        ]
    
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        for ax, (title, field) in zip(axes.ravel(), panels):
            data = field.data.to("g/cm^3").magnitude
            vmin, vmax = log_percentile_limits(data, 1.0, 99.5)
            pc = phi_avg_log_panel(
                ax,
                model,
                field,
                title=title,
                vmin=vmin,
                vmax=vmax,
                xlim=xlim,
                ylim=ylim,
            )
            cbar = fig.colorbar(pc, ax=ax)
            cbar.set_label("log10 density [g cm^-3]")
    
        fig.tight_layout()
        fig.savefig(output, dpi=220)
        plt.close(fig)

    def plot_inner_component_masks(*, disk_mask_field, ism_mask_field, output, xlim, ylim):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
        panels = [
            ("disk component mask", disk_mask_field),
            ("ISM component mask", ism_mask_field),
        ]
        for ax, (title, field) in zip(axes, panels):
            r_edges_au = mesh.edges("r").to("au").magnitude
            theta_edges = mesh.edges("theta").to("radian").magnitude
            data = field.data.to("dimensionless").magnitude
            data_phi = np.mean(data, axis=2).T
            y_edges = np.cos(theta_edges)
            if y_edges[0] > y_edges[-1]:
                y_edges = y_edges[::-1]
                data_phi = data_phi[::-1, :]
            pc = ax.pcolormesh(
                r_edges_au,
                y_edges,
                data_phi,
                shading="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            fig.colorbar(pc, ax=ax)
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(title)
            ax.set_xlabel("r [au]")
            ax.set_ylabel("z/r")
        fig.tight_layout()
        fig.savefig(output, dpi=220)
        plt.close(fig)
    
    
    def plot_midplane_radial_diagnostics(
        *,
        gas_density,
        total_dust_density,
        disk_dust_density,
        ism_dust_density,
        dtg_total,
        dtg_disk,
        dtg_ism,
        disk_weight,
        output,
    ):
        r_au = model.mesh.centers("r").to("au").magnitude
        theta = model.mesh.centers("theta").to("radian").magnitude
        mid_idx = int(np.argmin(np.abs(theta - 0.5 * np.pi)))
    
        def mid_phi_mean(arr):
            arr = np.asarray(arr, dtype=float)
            return np.nanmean(arr[:, mid_idx, :], axis=1)
    
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    
        axes[0].loglog(r_au, mid_phi_mean(gas_density), label="gas", color="0.15")
        axes[0].loglog(r_au, mid_phi_mean(total_dust_density), label="total dust", color="tab:blue")
        axes[0].loglog(r_au, mid_phi_mean(disk_dust_density), label="disc dust", color="tab:orange")
        axes[0].loglog(r_au, mid_phi_mean(ism_dust_density), label="ISM dust", color="tab:green")
        axes[0].set_ylabel("midplane density [g cm^-3]")
        axes[0].legend(loc="best", fontsize=8)
    
        axes[1].semilogx(r_au, mid_phi_mean(dtg_total), label="total", color="tab:blue")
        axes[1].semilogx(r_au, mid_phi_mean(dtg_disk), label="disc component", color="tab:orange")
        axes[1].semilogx(r_au, mid_phi_mean(dtg_ism), label="ISM component", color="tab:green")
        axes[1].axhline(1e-2, color="0.3", linestyle="--", linewidth=1.0, label="1e-2")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("midplane dust/gas")
        axes[1].legend(loc="best", fontsize=8)
    
        axes[2].semilogx(r_au, mid_phi_mean(disk_weight), color="tab:purple")
        axes[2].set_ylabel("midplane dust disc weight")
        axes[2].set_xlabel("r [au]")
        axes[2].set_ylim(-0.05, 1.05)
    
        for ax in axes:
            ax.set_xlim(float(np.nanmin(r_au[r_au > 0.0])), 300.0)
            ax.grid(True, which="both", alpha=0.25)
    
        fig.tight_layout()
        fig.savefig(output, dpi=220)
        plt.close(fig)
    
    fthres = 2.0
    
    r_max_au = float(r_max.to("au").magnitude)
    fthres_vr_inner = 1.0
    
    r_plot_au = mesh.centers("r").to("au").magnitude
    r_positive = r_plot_au[np.isfinite(r_plot_au) & (r_plot_au > 0.0)]
    r_min_au = float(np.min(r_positive))
    
    plot_phi_avg_rz_slice(
        model,
        "density",
        output=supporting_plot("gas_density_zoverr_vs_r_premask.png"),
        x_axis="r",
        y_axis="z/r",
        log10=True,
        xscale="log",
        xlim=(r_min_au, 300.0),
    )
    
    def fthres_vr_inner_relaxed(R_au: np.ndarray) -> np.ndarray:
        R_au = np.asarray(R_au, dtype=float)
        R_au = np.clip(R_au, r_min_au, r_max_au)
        x = np.log(R_au / r_min_au) / np.log(r_max_au / r_min_au)
        return fthres_vr_inner + (fthres - fthres_vr_inner) * x
    
    disk = model.set_mask_from_joos_disk(
        rho_disk_min=rho_disk_min,
        r_max=r_max,
        fthres=fthres,
        fthres_vr=fthres_vr_inner_relaxed,
        weight_mode=WEIGHT_MODE,
        weight_name="disk_weight",
    )
    
    model.gas_register("disk_mask", disk.mask)
    
    mask = disk.mask.data.magnitude.astype(bool)
    print(f"mask fraction (theta) = {mask.mean():.6f}")
    
    model.dust.add_component_from_mask(mask="disk_mask", mode="settling")
    model.dust.add_component_from_mask(
        mask="disk_mask",
        complement=True,
        mode="proportional",
    )

    make_dust_component_diagnostic_plots(
        model,
        PLOT_DIR / "dust_components",
        diagnostics=True,
    )

    disk_component_mask_field = model.dust._components[0].mask
    ism_component_mask_field = model.dust._components[1].mask
    
    dust_density = model.dust["density"]
    
    rho_g = model.gas["density"].data.to("g/cm^3").magnitude
    rho_d = dust_density.data.to("g/cm^3").magnitude
    
    r_e = mesh.edges("r")
    theta_e = mesh.edges("theta")
    phi_e = mesh.edges("phi")
    if r_e is None or theta_e is None or phi_e is None:
        raise ValueError("Mesh is missing one or more spherical edge axes")
    
    r3 = (r_e[1:] ** 3 - r_e[:-1] ** 3) / 3.0
    theta_e_rad = theta_e.to("radian").magnitude
    dcos = np.cos(theta_e_rad[:-1]) - np.cos(theta_e_rad[1:])
    phi_e_rad = phi_e.to("radian").magnitude
    dphi = phi_e_rad[1:] - phi_e_rad[:-1]
    dV = r3[:, None, None] * dcos[None, :, None] * dphi[None, None, :]
    dV_mag = dV.to("cm^3").magnitude
    
    gas_mass_disk = float(np.sum(rho_g[mask] * dV_mag[mask]))
    dust_mass_disk = float(np.sum(rho_d[mask] * dV_mag[mask]))
    if gas_mass_disk <= 0.0:
        raise ValueError("Disk gas mass is non-positive; cannot compute dust-to-gas ratio")
    dtg_disk_mass = dust_mass_disk / gas_mass_disk
    print(f"disk dust_to_gas_mass_ratio = {dtg_disk_mass:.6e}")
    
    local_dtg = np.full_like(rho_g, np.nan, dtype=float)
    sel = mask & (rho_g > 0.0)
    local_dtg[sel] = rho_d[sel] / rho_g[sel]
    print(
        "disk local_dust_to_gas_ratio (rho_d/rho_g): "
        f"min={np.nanmin(local_dtg):.6e} "
        f"median={np.nanmedian(local_dtg):.6e} "
        f"max={np.nanmax(local_dtg):.6e}"
    )
    
    r_plot = mesh.centers("r").to("au").magnitude
    r_positive = r_plot[np.isfinite(r_plot) & (r_plot > 0.0)]
    r_min = float(np.min(r_positive))
    
    plot_phi_avg_rz_slice(
        model,
        "disk_mask",
        output=supporting_plot("disk_mask_theta_zoverr_vs_r.png"),
        x_axis="r",
        y_axis="z/r",
        log10=False,
        xscale="log",
        xlim=(r_min, 300.0),
    )

    plot_phi_avg_rz_slice(
        model,
        disk_component_mask_field,
        output=supporting_plot(f"dust_component_disk_mask_zoverr_vs_r_{WEIGHT_MODE}.png"),
        x_axis="r",
        y_axis="z/r",
        log10=False,
        xscale="log",
        xlim=(r_min, 300.0),
    )

    plot_phi_avg_rz_slice(
        model,
        ism_component_mask_field,
        output=supporting_plot(f"dust_component_ism_mask_zoverr_vs_r_{WEIGHT_MODE}.png"),
        x_axis="r",
        y_axis="z/r",
        log10=False,
        xscale="log",
        xlim=(r_min, 300.0),
    )

    plot_inner_component_masks(
        disk_mask_field=disk_component_mask_field,
        ism_mask_field=ism_component_mask_field,
        output=key_plot(f"dust_component_masks_inner_zoverr_vs_r_{WEIGHT_MODE}.png"),
        xlim=(r_min, 80.0),
        ylim=(-0.35, 0.35),
    )
    
    component_totals = {}
    dust_density_disk_total = None
    dust_density_ism_total = None
    dtg_disk = None
    dtg_ism = None
    
    for bin_name, dust_bin in model.dust.bins.items():
        bin_density = dust_bin["density"]
    
        if hasattr(model.dust, "_global_bins"):
            comp_idx, local_idx = model.dust._global_bins[bin_name]
            prefix = f"dust_bin_comp{comp_idx}_local{local_idx}_"
        else:
            prefix = "dust_bin_"
            comp_idx = -1
            local_idx = -1
    
        if comp_idx not in component_totals:
            component_totals[comp_idx] = bin_density.data.copy()
        else:
            component_totals[comp_idx] = component_totals[comp_idx] + bin_density.data
    
        size_um = float(dust_bin.size.to("um").magnitude)
        output = bin_plot(f"{prefix}{bin_name}_{size_um:.6g}um_zoverr_vs_r.png")
    
        plot_phi_avg_rz_slice(
            model,
            bin_density,
            output=output,
            x_axis="r",
            y_axis="z/r",
            log10=True,
            xscale="log",
            xlim=(r_min, 300.0),
        )
    
    if 0 in component_totals:
        dust_density_disk_total = Field(
            data=component_totals[0],
            quantity=dust_density.quantity,
            axis_order=dust_density.axis_order,
        )
        plot_phi_avg_rz_slice(
            model,
            dust_density_disk_total,
            output=supporting_plot("dust_density_disk_total_zoverr_vs_r.png"),
            x_axis="r",
            y_axis="z/r",
            log10=True,
            xscale="log",
            xlim=(r_min, 300.0),
        )
    
        rho_d_disk = dust_density_disk_total.data.to("g/cm^3").magnitude
        rho_g_disk = model.gas["density"].data.to("g/cm^3").magnitude
        dtg_disk = np.full_like(rho_g_disk, np.nan, dtype=float)
        sel_dtg_disk = rho_g_disk > 0.0
        dtg_disk[sel_dtg_disk] = rho_d_disk[sel_dtg_disk] / rho_g_disk[sel_dtg_disk]
        ok_dtg_disk = np.isfinite(dtg_disk) & (dtg_disk > 0.0)
        if not np.any(ok_dtg_disk):
            raise ValueError("Disk-component dust-to-gas ratio has no positive finite values")
        dtg_disk_log = np.log10(dtg_disk[ok_dtg_disk])
        vmin_dtg_disk = float(np.nanpercentile(dtg_disk_log, 1.0))
        vmax_dtg_disk = float(np.nanpercentile(dtg_disk_log, 99.0))
        dtg_disk_field = Field(
            data=diskbridge.Quantity(dtg_disk, "dimensionless"),
            quantity="mask",
            axis_order=dust_density.axis_order,
        )
        plot_phi_avg_rz_slice(
            model,
            dtg_disk_field,
            output=supporting_plot("dust_to_gas_disk_total_zoverr_vs_r.png"),
            x_axis="r",
            y_axis="z/r",
            log10=True,
            xscale="log",
            xlim=(r_min, 300.0),
            vmin=vmin_dtg_disk,
            vmax=vmax_dtg_disk,
        )
    
    if 1 in component_totals:
        dust_density_ism_total = Field(
            data=component_totals[1],
            quantity=dust_density.quantity,
            axis_order=dust_density.axis_order,
        )
        plot_phi_avg_rz_slice(
            model,
            dust_density_ism_total,
            output=supporting_plot("dust_density_ism_total_zoverr_vs_r.png"),
            x_axis="r",
            y_axis="z/r",
            log10=True,
            xscale="log",
            xlim=(r_min, 300.0),
        )
    
        rho_d_ism = dust_density_ism_total.data.to("g/cm^3").magnitude
        rho_g_ism = model.gas["density"].data.to("g/cm^3").magnitude
        dtg_ism = np.full_like(rho_g_ism, np.nan, dtype=float)
        sel_dtg_ism = rho_g_ism > 0.0
        dtg_ism[sel_dtg_ism] = rho_d_ism[sel_dtg_ism] / rho_g_ism[sel_dtg_ism]
        ok_dtg_ism = np.isfinite(dtg_ism) & (dtg_ism > 0.0)
        if not np.any(ok_dtg_ism):
            raise ValueError("ISM-component dust-to-gas ratio has no positive finite values")
        dtg_ism_log = np.log10(dtg_ism[ok_dtg_ism])
        vmin_dtg_ism = float(np.nanpercentile(dtg_ism_log, 1.0))
        vmax_dtg_ism = float(np.nanpercentile(dtg_ism_log, 99.0))
        dtg_ism_field = Field(
            data=diskbridge.Quantity(dtg_ism, "dimensionless"),
            quantity="mask",
            axis_order=dust_density.axis_order,
        )
        plot_phi_avg_rz_slice(
            model,
            dtg_ism_field,
            output=supporting_plot("dust_to_gas_ism_total_zoverr_vs_r.png"),
            x_axis="r",
            y_axis="z/r",
            log10=True,
            xscale="log",
            xlim=(r_min, 300.0),
            vmin=vmin_dtg_ism,
            vmax=vmax_dtg_ism,
        )
    
    plot_phi_avg_rz_slice(
        model,
        "density",
        output=supporting_plot("gas_density_zoverr_vs_r.png"),
        x_axis="r",
        y_axis="z/r",
        log10=True,
        xscale="log",
        xlim=(r_min, 300.0),
    )
    # Plot global total dust density with full colorbar range
    dust_density_data = dust_density.data.to("g/cm^3").magnitude
    positive = dust_density_data > 0.0
    if not np.any(positive):
        raise ValueError("Global dust density has no positive values; cannot compute log10 color limits")
    dust_density_log = np.log10(dust_density_data[positive])
    vmin_global = float(np.nanmax(dust_density_log)) - 16
    vmax_global = float(np.nanmax(dust_density_log)) - 3
    
    plot_phi_avg_rz_slice(
        model,
        dust_density,
        output=supporting_plot(f"dust_density_zoverr_vs_r_{WEIGHT_MODE}.png"),
        x_axis="r",
        y_axis="z/r",
        log10=True,
        xscale="log",
        xlim=(r_min, 300.0),
        vmin=vmin_global,
        vmax=vmax_global,
    )
    
    vmin_dust_p, vmax_dust_p = log_percentile_limits(dust_density_data, 1.0, 99.5)
    plot_phi_avg_rz_slice(
        model,
        dust_density,
        output=supporting_plot(f"dust_density_zoverr_vs_r_{WEIGHT_MODE}_percentile.png"),
        x_axis="r",
        y_axis="z/r",
        log10=True,
        xscale="log",
        xlim=(r_min, 300.0),
        vmin=vmin_dust_p,
        vmax=vmax_dust_p,
    )
    
    plot_phi_avg_rz_slice(
        model,
        dust_density,
        output=key_plot(f"dust_density_inner_zoverr_vs_r_{WEIGHT_MODE}_percentile.png"),
        x_axis="r",
        y_axis="z/r",
        log10=True,
        xscale="log",
        xlim=(r_min, 80.0),
        ylim=(-0.35, 0.35),
        vmin=vmin_dust_p,
        vmax=vmax_dust_p,
    )
    
    dtg_total = np.full_like(rho_g, np.nan, dtype=float)
    sel_dtg_total = rho_g > 0.0
    dtg_total[sel_dtg_total] = rho_d[sel_dtg_total] / rho_g[sel_dtg_total]
    ok_dtg_total = np.isfinite(dtg_total) & (dtg_total > 0.0)
    if not np.any(ok_dtg_total):
        raise ValueError("Global dust-to-gas ratio has no positive finite values")
    dtg_total_log = np.log10(dtg_total[ok_dtg_total])
    vmin_dtg_total = float(np.nanpercentile(dtg_total_log, 1.0))
    vmax_dtg_total = float(np.nanpercentile(dtg_total_log, 99.0))
    dtg_total_field = Field(
        data=diskbridge.Quantity(dtg_total, "dimensionless"),
        quantity="mask",
        axis_order=dust_density.axis_order,
    )
    
    plot_phi_avg_rz_slice(
        model,
        dtg_total_field,
        output=supporting_plot(f"dust_to_gas_zoverr_vs_r_{WEIGHT_MODE}.png"),
        x_axis="r",
        y_axis="z/r",
        log10=True,
        xscale="log",
        xlim=(r_min, 300.0),
        vmin=vmin_dtg_total,
        vmax=vmax_dtg_total,
    )
    
    plot_phi_avg_rz_slice(
        model,
        dtg_total_field,
        output=key_plot(f"dust_to_gas_inner_zoverr_vs_r_{WEIGHT_MODE}.png"),
        x_axis="r",
        y_axis="z/r",
        log10=True,
        xscale="log",
        xlim=(r_min, 80.0),
        ylim=(-0.35, 0.35),
        vmin=vmin_dtg_total,
        vmax=vmax_dtg_total,
    )
    
    plot_phi_avg_rz_slice(
        model,
        model.gas["disk_weight"],
        output=supporting_plot(f"disk_weight_theta_zoverr_vs_r_{WEIGHT_MODE}.png"),
        x_axis="r",
        y_axis="z/r",
        log10=False,
        xscale="log",
        xlim=(r_min, 300.0),
    )
    
    plot_phi_avg_rz_slice(
        model,
        disk_component_mask_field,
        output=key_plot(f"dust_weight_inner_theta_zoverr_vs_r_{WEIGHT_MODE}.png"),
        x_axis="r",
        y_axis="z/r",
        log10=False,
        xscale="log",
        xlim=(r_min, 80.0),
        ylim=(-0.35, 0.35),
    )
    
    if dust_density_disk_total is not None and dust_density_ism_total is not None:
        plot_inner_component_comparison(
            gas_field=model.gas["density"],
            total_dust_field=dust_density,
            disk_dust_field=dust_density_disk_total,
            ism_dust_field=dust_density_ism_total,
            output=key_plot(f"dust_components_inner_zoverr_vs_r_{WEIGHT_MODE}_percentile.png"),
            xlim=(r_min, 80.0),
            ylim=(-0.35, 0.35),
        )
    
    if dtg_disk is not None and dtg_ism is not None:
        plot_midplane_radial_diagnostics(
            gas_density=rho_g,
            total_dust_density=rho_d,
            disk_dust_density=dust_density_disk_total.data.to("g/cm^3").magnitude,
            ism_dust_density=dust_density_ism_total.data.to("g/cm^3").magnitude,
            dtg_total=dtg_total,
            dtg_disk=dtg_disk,
            dtg_ism=dtg_ism,
            disk_weight=disk_component_mask_field.data.to("dimensionless").magnitude,
            output=key_plot(f"midplane_radial_dust_diagnostics_{WEIGHT_MODE}.png"),
        )
    
    rho_d = model.dust["density"].data.to("g/cm^3").magnitude
    w = disk_component_mask_field.data.to("dimensionless").magnitude
    
    rho_d_phi = np.mean(rho_d, axis=2)
    w_phi = np.mean(w, axis=2)
    
    tiny = np.finfo(np.float64).tiny
    logrho = np.log10(np.maximum(rho_d_phi, tiny))
    
    grad = np.abs(np.diff(logrho, axis=1))
    w_mid = 0.5 * (w_phi[:, :-1] + w_phi[:, 1:])
    sel = (w_mid > 0.1) & (w_mid < 0.9)
    
    edge = np.nanpercentile(grad[sel], 90) if np.any(sel) else np.nan
    print(f"[{WEIGHT_MODE}] edge_harshness_p90_dex_per_theta_cell = {edge}")
    write_plot_index()

write_weight_mode_index()
