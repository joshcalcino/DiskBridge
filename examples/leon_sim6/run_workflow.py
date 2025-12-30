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
from diskbridge.visualization import plot_phi_avg_rz_slice
import numpy as np


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

dust_density = model.dust["density"]

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
plot_phi_avg_rz_slice(
    model,
    dust_density,
    output="dust_density_zoverr_vs_r.png",
    x_axis="r",
    y_axis="z/r",
    log10=True,
    xscale="log",
    xlim=(r_min, 300.0),
)



# Write RADMC-3D input files and compute dust opacities
# writer = RadWriter(model)
# writer.write_all_input_files()
# writer.compute_and_write_dust_opacities()
#
# # Initialize RADMC-3D model wrapper
# rad = RadModel(model)
#
# # Compute dust temperature with Monte Carlo
# rad.compute_temperature()
#
# # Compute CO abundance using Pinte+2018 switches from params
# X_co, n_co = chemistry.compute_abundance(
#     rad,
#     molecule="co",
#     X0=p.abundance,
#     write_output=True,
# )
#
# mean_X = float(X_co.magnitude.mean())
# print(f"CO mean abundance X = {mean_X:.3e}")
#
# img = RadImage(model_dir=".", model=model)
# img.make_line_image(
#     molecule="co",
#     transition=p.iline,
# )

