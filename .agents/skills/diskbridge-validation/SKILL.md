---
name: diskbridge-validation
description: Design, create, run, or review larger DiskBridge physics validation workflows. Use for full pipelines, code comparisons, scientific diagnostics, plots, validation reports, or human-interpreted checks. Do not use for fast pytest tests.
---

# DiskBridge Validation Skill

Use this skill when the user asks to validate physics with a larger workflow, compare against another code, reproduce a paper result, run a full pipeline, generate diagnostic plots, or decide whether new physics behaves plausibly.

## Definition Of A Validation

A DiskBridge validation is a physics-oriented workflow that is too slow, broad, data-heavy, interpretive, or workflow-oriented to run on every ordinary code change. It may be multi-step, subjective, or require human interpretation.

It should answer questions like:

- Does the new physics behave as expected?
- Does DiskBridge reproduce a known benchmark?
- Does a pipeline remain scientifically plausible end-to-end?
- Does a change alter CO, H2, C, C+, gas temperature, UV products, shielding, or line-transfer outputs in an expected way?
- Does DiskBridge agree qualitatively or quantitatively with another code?

## Tests Versus Validations

Tests go in `tests/` and use pytest for production code behavior and small objective physics invariants. Do not add pytest files for validation-specific drivers, configs, plotting helpers, or workflow glue. Do not create `tests/validation/`; that name is reserved for the repo-level `validation/` workflow tree.

Validations go in the repository-level `validation/` directory and must not use pytest as their main driver. Use validations for slower pipelines, comparisons to papers or other codes, generated diagnostic plots, archived outputs, and human-interpreted scientific judgment.

Validations should produce human-readable outputs such as plots, tables, JSON summaries, markdown reports, saved comparison arrays, or logs with exact commands and config choices.

## Validation Directory Structure

Each validation should normally live in its own subdirectory:

```text
validation/<validation_name>/
  README.md
  run.py
  config.toml
  outputs/
  plots/
  report.md
```

Follow existing validation layout when updating an established validation.
Never place validation workflows under `tests/` or create a `tests/validation/`
directory. Do not create pytest coverage for validation-specific helper logic;
the validation's own run summaries, metadata, plots, and reports are the review
surface.

## Required Validation README

Every validation should explain:

1. Scientific question.
2. Expected behavior.
3. Inputs.
4. Exact command to run.
5. Outputs produced.
6. What a human should inspect.
7. Known limitations.

## Required Outputs

A validation should usually write:

- `outputs/summary.json`
- `plots/*.png`
- `report.md`

Existing validations may use established `out/` directories, but new validations should prefer explicit `outputs/` and `plots/` folders.

## Validation Rules

- Do not hide subjective interpretation inside pass/fail pytest assertions.
- Do not move a simple objective physics invariant out of pytest only because it is physics-based.
- Include plots.
- Save enough metadata to reproduce the run.
- Clearly state whether the validation is quantitative, qualitative, or exploratory.
- Do not overwrite previous validation outputs unless the script has an explicit `--overwrite` option.
- If the validation depends on unavailable external data, provide a small synthetic fallback only if it still answers the scientific question.
- Prefer a single readable validation entry point that runs with no required
  CLI flags. Put ordinary defaults in committed config or parameter files.
  Avoid adding many optional arguments or "smoke" modes to validation scripts
  unless the user explicitly requests them.
- Do not present toy "smoke" runs as validation evidence. For Monte Carlo
  RADMC-3D validations, configure enough photon packets per relevant cell to
  meet the stated noise target; a 1% target implies roughly `1e4` packets per
  cell for the sampled product, before allowing for optical-depth and geometry
  inefficiencies. Very short runs may be useful only for wiring/debug checks and
  must be labeled as such.

## Final Report

Report what validation was created or run, what physics it tests, where outputs are saved, which plots a human should inspect, and what result would indicate success or failure.
