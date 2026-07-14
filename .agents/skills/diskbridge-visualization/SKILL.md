---
name: diskbridge-visualization
description: Create, refactor, or review DiskBridge plotting and visualization code. Use when adding plots, improving diagnostic figures, choosing scales/colormaps/colorbars, moving reusable plotting helpers into src/diskbridge/visualization, or deciding what should remain validation-specific.
---

# DiskBridge Visualization Skill

Use this skill when the user asks for plots, figure-quality diagnostics,
colormap/scale fixes, plot layout cleanup, or reusable visualization helpers.

DiskBridge is single-user development code. Plotters and diagnostic loaders
support the canonical current outputs only. Replace superseded names and
formats directly rather than retaining aliases or compatibility discovery,
unless the user explicitly requests it.

## Main Boundary

Reusable plotting code belongs in `src/diskbridge/visualization/`.
Validation-specific orchestration, file discovery, and interpretation belong in
`validation/<case>/`.

Do not bury broadly useful plotting policy inside one validation script. If a
scale rule, colormap, geometry helper, field label, ratio helper, categorical
legend, or shared diagnostic panel will be reused, add a small helper to the
visualization module and call it from the validation.

## Before Editing

1. Inspect existing plotting code in `src/diskbridge/visualization/`.
2. Inspect the relevant validation script and, for disk chemistry plots, compare
   against `validation/prizmo_disk/` plotting conventions.
3. Identify which logic is reusable library behavior and which is run-specific
   validation glue.
4. If the change is non-trivial, update the active project plan before editing.

## Scientific Plot Rules

- Use log scales for densities, abundances, rates, UV fields, columns, optical
  depths, and other fields that span orders of magnitude.
- Do not plot abundance or rate tails down to meaningless numerical junk. Use a
  physically useful floor and cap the displayed dynamic range when appropriate.
- Never let log-scale colorbar limits follow numerical zeros or machine-float
  floors down to values such as `1e-300`. Use a plotting floor and a reasonable
  capped dynamic range, normally via `diskbridge.visualization.scales`
  (`log10_display_values` and `log10_display_limits`), so the visible scale
  emphasizes physically interpretable variation.
- Use percentile-aware limits, but never let percentiles hide true extrema that
  are scientifically important for the diagnostic.
- Use ratio maps for comparisons and make the denominator/floor policy explicit.
- Use symmetric or symlog scales for signed residuals, deltas, and exchange
  terms.
- Use discrete levels and labelled ticks for categorical IDs and status codes.
- Colorbar labels must describe physical meaning and units. Avoid internal
  names such as `chem status`, `co gain id`, or raw field keys unless the plot is
  explicitly a developer field inventory.

## Colormaps

- Do not default every panel to `viridis`.
- Temperature maps should use the established DiskBridge/PRIZMO-style
  temperature colormap when available, or a perceptually ordered
  blue/cyan/green/yellow/red map.
- Ratios, signed differences, residuals, and status-like maps should use
  diverging or discrete colormaps as appropriate.
- Categorical process IDs should use discrete colormaps with labelled tick
  meanings.
- Abundances, UV fields, density, and rates may use sequential colormaps, but
  choose a single shared scale per field family when comparing variants.

## Geometry And Layout

- For 2D disk plots, use physical cylindrical axes (`R [au]`, `z [au]`) and
  avoid excess whitespace. Prefer stable figure sizes and `constrained_layout`.
- Preserve the useful disk geometry; do not force equal aspect if it makes a
  wide disk unreadable, and do not crop away the outer disk when that is the
  diagnostic target.
- Close Matplotlib figures after saving and use a non-interactive backend for
  scripted runs.
- Keep filenames stable and descriptive.

## Visualization Module Expectations

When adding helpers under `src/diskbridge/visualization/`:

- Keep public helpers small and composable.
- Use NumPy arrays and explicit units/labels at the plotting boundary.
- Avoid importing heavy optional plotting backends at module import time when
  possible; import Matplotlib inside plotting functions if that matches local
  style.
- Add NumPy-style docstrings for public helpers.
- Export new public helpers from `src/diskbridge/visualization/__init__.py`
  only when they are intended as user-facing API.

## Tests And Validation

- Fast deterministic helper behavior can be tested with pytest, such as scale
  limit selection, label lookup, safe filenames, and categorical tick handling.
- Do not add pytest tests for full validation plotting workflows. Those remain
  validation outputs under `validation/`.
- For validation plots, run the relevant plotting script or full validation and
  report output paths, plot count, and any remaining scientific caveats.

## Final Report

State which code stayed validation-specific, which helpers moved into
`src/diskbridge/visualization/`, what plots were generated or checked, and what
scientific interpretation remains unresolved.
