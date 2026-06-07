# 2D Outer Disk CO Phase Validation

This validation runs a DiskBridge-only, axisymmetric 2D disk to 500 au and
compares three GOW17 chemistry variants on the same RADMC-3D UV products:

- `gas_only_no_co_phase`
- `co_ice_no_photodesorption`
- `co_ice_full_photodesorption`

It also runs the full CO ice model a second time with RADMC external UV disabled:

- `co_ice_full_no_external_uv`

The goal is to inspect where CO remains in the gas, freezes onto grains, or is
returned by photodesorption in the lower-density outer disk and irradiated
surface. There is no PRIZMO reference in this pathway.

The disk surface density includes an additional exponential outer taper so that
the physical disk is nearly depleted by 500 au while the grid still covers the
low-density outer material needed for the ISM-field comparison.

## Run

```bash
python validation/outer_disk_co_phase_2d/run.py
```

The script has no command-line modes. The validation defaults live in
`config.toml` and `params.txt`; `params.txt` sets `nphot_thermal = 1e7` and
`nphot_mono = 1e7`. The RADMC-3D scattering mode is set explicitly in
`params.txt`; this validation uses `scat_mode = 1` because 2D spherical
`mcmono` does not support scattering mode 2.

## Scattering Limitation

This validation currently uses isotropic dust scattering. That is a conscious
limitation, not a hidden fallback. RADMC-3D stops 2D spherical `mcmono` runs
with ordinary anisotropic Henyey-Greenstein scattering (`scat_mode = 2`). Proper
2D anisotropic scattering requires RADMC-3D's special `dust_2daniso` machinery,
full scattering-matrix opacity support, and compatible angular bookkeeping. The
follow-up project is tracked in
`projects/active/2026-06-07-radmc3d-2d-anisotropic-scattering.md`.

## Temporary UV Floor

The validation runner temporarily floors exact zero RADMC-3D UV mean-intensity
samples to `1e-300 erg/(s*cm^2*Hz*sr)` before UV-product interpolation. This is
local to this validation script. Production zero-intensity handling is tracked
in `projects/active/2026-06-07-radmc3d-uv-zero-intensity-handling.md`.

Outputs are written under `validation/outer_disk_co_phase_2d/outputs/` and are
ignored by git. The important files are the source scripts, `config.toml`,
`params.txt`, this README, and the active project plan.

## Expected Outputs

Each run writes:

- `grid_summary.json`
- `radmc3d_uv_summary.json`
- `radmc3d_uv_no_external_summary.json`
- `uv_products.npz`
- `uv_products_no_external.npz`
- `variants/<variant>/summary.json`
- `variants/<variant>/gow17_meta.json`
- `variants/<variant>/snapshot.h5`
- per-variant plots
- `comparison/summary.json`
- cross-variant comparison plots

This is the validation run. It should be inspected through the summaries and
plots, not treated as a pytest-style pass/fail check.

Known current issue: the default run completes, but the GOW17 solver reports
large accumulated convergence/CVODE diagnostics in the chemistry variants.
