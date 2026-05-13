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
import diskbridge.chemistry as chemistry
from diskbridge.radmc3d import RadWriter, RadModel, RadImage
from diskbridge.visualization.diagnostics import make_dust_component_diagnostic_plots

import os
from pathlib import Path


# Load parameters for this example and update the global params
p = diskbridge.read_params("params.txt")
diskbridge.params = p

# Load the FARGO model from the 3D disk data (snapshot 5, code units)
model = diskbridge.load_model(
    "data/",
    reader="fargo",
    file_n=5,
    file_units="code",
    length_scale=100.0,
)

# Set dust size distribution from gas, proportional to gas density
model.dust.set_distribution(mode="proportional")

DIAGNOSTIC_PLOTS = os.environ.get("DISKBRIDGE_DIAGNOSTIC_PLOTS", "1").lower() not in {
    "0",
    "false",
    "f",
    "no",
    "n",
    "off",
}
make_dust_component_diagnostic_plots(
    model,
    Path("plots") / "dust_components",
    diagnostics=DIAGNOSTIC_PLOTS,
)

# Write RADMC-3D input files and compute dust opacities
writer = RadWriter(model)
writer.write_all_input_files()
writer.compute_and_write_dust_opacities()

# Initialize RADMC-3D model wrapper
rad = RadModel(model)

# Compute dust temperature with Monte Carlo
rad.compute_temperature(force=True)

# Compute CO abundance using Pinte+2018 switches from params
X_co, n_co = chemistry.compute_abundance(
    rad,
    molecule="co",
    X0=p.abundance,
    write_output=True,
)

mean_X = float(X_co.magnitude.mean())
print(f"CO mean abundance X = {mean_X:.3e}")

img = RadImage(model_dir=".", model=model)
img.make_line_image(
    molecule="co",
    transition=p.iline,
)
