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
)

model.gas_register("disk_mask", disk.mask)

mask = disk.mask.data.magnitude.astype(bool)
print(f"mask fraction (theta) = {mask.mean():.6f}")

# Configure dust size distribution inside the masked disk region
disk.dust.set_distribution(mode="settling")

ism_mask = ~disk.mask.data.magnitude.astype(bool)
ism = model.set_mask_from_array(ism_mask, is_a_disk=False)
ism.dust.set_distribution(mode="proportional")

dust_density = model.dust["density"]

r_plot = mesh.centers("r").to("au").magnitude
r_positive = r_plot[np.isfinite(r_plot) & (r_plot > 0.0)]
r_min = float(np.min(r_positive))

# plot_phi_avg_rz_slice(
#     model,
#     "disk_mask",
#     output="disk_mask_theta_zoverr_vs_r.png",
#     x_axis="r",
#     y_axis="z/r",
#     log10=False,
#     xscale="log",
#     xlim=(r_min, 300.0),
# )

# component_totals = {}

# for bin_name, dust_bin in model.dust.bins.items():
#     bin_density = dust_bin["density"]

#     if hasattr(model.dust, "_global_bins"):
#         comp_idx, local_idx = model.dust._global_bins[bin_name]
#         prefix = f"dust_bin_comp{comp_idx}_local{local_idx}_"
#     else:
#         prefix = "dust_bin_"
#         comp_idx = -1
#         local_idx = -1

#     if comp_idx not in component_totals:
#         component_totals[comp_idx] = bin_density.data.copy()
#     else:
#         component_totals[comp_idx] = component_totals[comp_idx] + bin_density.data

#     size_um = float(dust_bin.size.to("um").magnitude)
#     output = f"{prefix}{bin_name}_{size_um:.6g}um_zoverr_vs_r.png"

#     plot_phi_avg_rz_slice(
#         model,
#         bin_density,
#         output=output,
#         x_axis="r",
#         y_axis="z/r",
#         log10=True,
#         xscale="log",
#         xlim=(r_min, 300.0),
#     )

# if 0 in component_totals:
#     dust_density_disk_total = Field(
#         data=component_totals[0],
#         quantity=dust_density.quantity,
#         axis_order=dust_density.axis_order,
#     )
#     plot_phi_avg_rz_slice(
#         model,
#         dust_density_disk_total,
#         output="dust_density_disk_total_zoverr_vs_r.png",
#         x_axis="r",
#         y_axis="z/r",
#         log10=True,
#         xscale="log",
#         xlim=(r_min, 300.0),
#     )

# if 1 in component_totals:
#     dust_density_ism_total = Field(
#         data=component_totals[1],
#         quantity=dust_density.quantity,
#         axis_order=dust_density.axis_order,
#     )
#     plot_phi_avg_rz_slice(
#         model,
#         dust_density_ism_total,
#         output="dust_density_ism_total_zoverr_vs_r.png",
#         x_axis="r",
#         y_axis="z/r",
#         log10=True,
#         xscale="log",
#         xlim=(r_min, 300.0),
#     )

# plot_phi_avg_rz_slice(
#     model,
#     "density",
#     output="gas_density_zoverr_vs_r.png",
#     x_axis="r",
#     y_axis="z/r",
#     log10=True,
#     xscale="log",
#     xlim=(r_min, 300.0),
# )
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
    output="dust_density_zoverr_vs_r.png",
    x_axis="r",
    y_axis="z/r",
    log10=True,
    xscale="log",
    xlim=(r_min, 300.0),
    vmin=vmin_global,
    vmax=vmax_global,
)

