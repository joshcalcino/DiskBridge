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
import numpy as np
from diskbridge.model.field import Field


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

rho_disk_min = 1e-21 * diskbridge.units('g/cm^3')
r_max = 200 * diskbridge.units('au')

WEIGHT_MODE = "margins"

mesh = model.mesh

fthres = 2.0

r_max_au = float(r_max.to("au").magnitude)
fthres_vr_inner = 1.0

r_plot_au = mesh.centers("r").to("au").magnitude
r_positive = r_plot_au[np.isfinite(r_plot_au) & (r_plot_au > 0.0)]
r_min_au = float(np.min(r_positive))

plot_phi_avg_rz_slice(
    model,
    "density",
    output="gas_density_zoverr_vs_r_premask.png",
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
    weight_delta_bins=3.0,
    weight_m0=0.25,
)

model.gas_register("disk_mask", disk.mask)

mask = disk.mask.data.magnitude.astype(bool)
print(f"mask fraction (theta) = {mask.mean():.6f}")

w = model.gas["disk_weight"].data.to("dimensionless").magnitude
w_ism = 1.0 - w

disk_weight_field = Field(
    data=diskbridge.Quantity(w, "dimensionless"),
    quantity="mask",
    axis_order=mesh.axis_names(),
)
ism_weight_field = Field(
    data=diskbridge.Quantity(w_ism, "dimensionless"),
    quantity="mask",
    axis_order=mesh.axis_names(),
)

model.dust.add_component_from_mask(mask=disk_weight_field, mode="settling")
model.dust.add_component_from_mask(mask=ism_weight_field, mode="proportional")

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
    output="disk_mask_theta_zoverr_vs_r.png",
    x_axis="r",
    y_axis="z/r",
    log10=False,
    xscale="log",
    xlim=(r_min, 300.0),
)

component_totals = {}

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
    output = f"{prefix}{bin_name}_{size_um:.6g}um_zoverr_vs_r.png"

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
        output="dust_density_disk_total_zoverr_vs_r.png",
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
        output="dust_to_gas_disk_total_zoverr_vs_r.png",
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
        output="dust_density_ism_total_zoverr_vs_r.png",
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
        output="dust_to_gas_ism_total_zoverr_vs_r.png",
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
    output="gas_density_zoverr_vs_r.png",
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
    output=f"dust_density_zoverr_vs_r_{WEIGHT_MODE}.png",
    x_axis="r",
    y_axis="z/r",
    log10=True,
    xscale="log",
    xlim=(r_min, 300.0),
    vmin=vmin_global,
    vmax=vmax_global,
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
    output=f"dust_to_gas_zoverr_vs_r_{WEIGHT_MODE}.png",
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
    model.gas["disk_weight"],
    output=f"disk_weight_theta_zoverr_vs_r_{WEIGHT_MODE}.png",
    x_axis="r",
    y_axis="z/r",
    log10=False,
    xscale="log",
    xlim=(r_min, 300.0),
)

rho_d = model.dust["density"].data.to("g/cm^3").magnitude
w = model.gas["disk_weight"].data.to("dimensionless").magnitude

rho_d_phi = np.mean(rho_d, axis=2)
w_phi = np.mean(w, axis=2)

tiny = np.finfo(np.float64).tiny
logrho = np.log10(np.maximum(rho_d_phi, tiny))

grad = np.abs(np.diff(logrho, axis=1))
w_mid = 0.5 * (w_phi[:, :-1] + w_phi[:, 1:])
sel = (w_mid > 0.1) & (w_mid < 0.9)

edge = np.nanpercentile(grad[sel], 90) if np.any(sel) else np.nan
print(f"[{WEIGHT_MODE}] edge_harshness_p90_dex_per_theta_cell = {edge}")

