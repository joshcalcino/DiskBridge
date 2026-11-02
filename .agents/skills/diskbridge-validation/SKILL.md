---
name: diskbridge-validation
description: Design, create, run, or review larger DiskBridge physics validations that test a stated scientific or code-behavior claim against an explicit expectation, reference, benchmark, invariant, or controlled comparison. Do not use for descriptive analyses, routine plots of existing outputs, inventories, atlases, exploratory measurements without an expected result, or fast pytest tests.
---

# DiskBridge Validation Skill

Use this skill when the user asks to validate physics with a larger workflow,
compare against another code, reproduce a paper result, test an end-to-end
pipeline expectation, generate diagnostic plots as evidence for a stated
validation question, or decide whether new physics behaves plausibly against
an explicit expectation.

DiskBridge is single-user development code. Validation workflows use the
canonical current API, config, and output format. Update a maintained workflow
in place and remove superseded modes; do not add migration or compatibility
paths unless the user explicitly requests them.

## Validation Gate

Classify work by the question it answers, not by runtime, dataset size, plot
count, report generation, or need for human interpretation. Before creating or
using `validation/<case>/`, state all three of the following:

1. The scientific claim or code behavior being tested.
2. The reference, benchmark, invariant, controlled comparison, or explicit
   expected behavior that supplies evidence.
3. What result would support or contradict the claim.

If any item is missing, do not call the work a validation and do not place it
under `validation/`.

Descriptive measurements of existing runs—including distributions,
inventories, snapshot atlases, and exploratory plots—belong beside the owning
example or project, such as `examples/<workflow>/plots/<analysis>/`. A plot may
be evidence within a validation, but producing a scientific plot does not make
the workflow a validation.

## Definition Of A Validation

A DiskBridge validation is a broader physics or workflow check that passes the
validation gate and is too slow, data-heavy, multi-step, or interpretive for
ordinary pytest coverage.

It should answer questions like:

- Does the new physics behave as expected?
- Does DiskBridge reproduce a known benchmark?
- Does a pipeline remain scientifically plausible end-to-end?
- Does a change alter CO, H2, C, C+, gas temperature, UV products, shielding, or line-transfer outputs in an expected way?
- Does DiskBridge agree qualitatively or quantitatively with another code?

## Tests Versus Validations

Tests go in tests/ for reusable production code, core tooling and small objective physics invariants.

NEVER add or retain pytest tests of example, problem, project, campaign or
validation pipelines. This includes their drivers, configuration/defaults,
plotting scripts, cluster scripts, CLI wrappers, source-text/AST assertions and
workflow glue, even with mocks, tiny inputs or fast execution. Delete prohibited
tests; do not rename, relocate, hide from collection, replace or recreate them.
Never create tests/validation/. Test reusable production-package behavior, core
repository tooling and objective physics/reference cases directly. Do not move
problem-only helpers into src merely to evade this ban.

Validations go in the repository-level `validation/` directory and must not use
pytest. Comparisons to papers or other codes, controlled
physics comparisons, and end-to-end expectation checks are typical examples.
Slow pipelines, archived outputs, diagnostic plots, and human judgment are not
sufficient unless the validation gate is also satisfied.

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
directory. Never create pytest coverage for validation drivers, configurations,
plotting scripts or helper glue. The validation's own run summaries, metadata,
plots and reports are its review surface; example/problem pipeline tests are
also prohibited.

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
- Clearly state whether the evidence and acceptance interpretation are
  quantitative or qualitative.
- Do not overwrite previous validation outputs unless the script has an explicit `--overwrite` option.
- If the validation depends on unavailable external data, provide a small synthetic fallback only if it still answers the scientific question.
- Prefer a single readable validation entry point that runs with no required
  CLI flags. Put ordinary defaults in committed config or parameter files.
  Avoid adding many optional arguments or "smoke" modes to validation scripts
  unless the user explicitly requests them.
- For validations that consume or produce RADMC-3D files, use the supported
  binary-only RADMC-3D data representation. Do not add validation-local
  ASCII/text fallback paths or alternate data structures that bypass production
  RADMC-3D I/O policy.
- Do not present toy "smoke" runs as validation evidence. For Monte Carlo
  RADMC-3D validations, configure enough photon packets per relevant cell to
  meet the stated noise target; a 1% target implies roughly `1e4` packets per
  cell for the sampled product, before allowing for optical-depth and geometry
  inefficiencies. Very short runs may be useful only for wiring/debug checks and
  must be labeled as such.

## Final Report

Report what validation was created or run, what physics it tests, where outputs are saved, which plots a human should inspect, and what result would indicate success or failure.
