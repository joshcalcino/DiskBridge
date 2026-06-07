---
name: diskbridge-tests
description: Add, fix, organize, run, or review fast pytest tests for DiskBridge code behavior and simple objective physics invariants. Use for public API smoke tests, regression tests, shape tests, unit tests, config tests, small synthetic physics checks, and error-condition tests. Do not use for long or interpretive validation workflows.
---

# DiskBridge Tests Skill

Use this skill when the task is to add or update fast tests under `tests/`.

## Definition Of A Test

A DiskBridge test is a fast, deterministic pytest check that can run on ordinary development changes. It should verify code behavior or a simple objective physics invariant. It should answer questions like:

- Does this function run?
- Are array shapes correct?
- Are units converted correctly?
- Does a config option resolve correctly?
- Does a known small input produce the expected output?
- Does invalid input raise the expected error?
- Did a previous bug stay fixed?
- Does a simple conservation, monotonicity, range, or analytic toy-case invariant hold?

Tests can include simple physics checks when they are cheap, deterministic, objective, and based on synthetic or analytic toy inputs. Tests are not a substitute for larger validation workflows.

## Where Tests Go

Put persistent tests in `tests/` and use pytest. Do not put long physical workflows in `tests/`.

Do not create `tests/validation/`. In DiskBridge, validation workflows live in
the repository-level `validation/` directory. Do not add pytest files for
validation-specific drivers, configs, plotting helpers, or workflow glue. If
reusable production code is extracted from a validation into `src/diskbridge/`,
test that production code in the normal subsystem path.

Use one curated public API smoke test, normally `tests/test_public_api.py`, for import and public-surface checks. Do not spread many import-only tests across the suite unless they exercise meaningfully different optional dependency behavior.

Keep `tests/README.md` current with the local testing standard. Use `docs/testing/` for human-readable explanations of analytic cases, benchmark assumptions, literature references, and recurring testing conventions.

## Good Tests

Good tests are:

- small
- deterministic
- quick to run
- independent of large external data
- focused on stable behavior
- useful for regression protection
- objective enough to pass or fail without human interpretation

Prefer synthetic mini-models over full simulation outputs.

## Naming And Organization

Test paths should describe the contract area clearly and concisely. Prefer subsystem folders when the suite grows, for example:

```text
tests/
  test_public_api.py
  chemistry/test_gow17_config.py
  chemistry/test_shielding.py
  model/test_masks.py
  radmc3d/test_dust_outputs.py
```

Avoid long filenames that encode the full historical bug or implementation detail. Avoid repeating the subsystem name inside every filename when the test already lives in a subsystem folder.

Good file names:

- `test_public_api.py`
- `test_uv_products.py`
- `chemistry/test_shielding.py`
- `radmc3d/test_external_radiation.py`
- `model/test_masks.py`

Poor file names:

- `test_radmc3d_external_i_nu_legacy_dust_normalization_path.py`
- `test_gow17_this_specific_previous_bug_with_config_mode.py`

## Test Documentation

Every test file should start with a module docstring that explains:

- the behavior or contract protected by the file;
- whether the tests use synthetic, analytic, or reference data;
- where larger scientific validation belongs, if relevant;
- any essential paper or benchmark reference used throughout the file.

Non-obvious test functions should have short docstrings or comments explaining the invariant, analytic limit, regression, or physical expectation being checked. Do not add boilerplate docstrings to obvious assertions.

Use this placement rule:

- One assertion or one test needs context: put a short comment or docstring in the test.
- A whole test file uses the same reference or analytic assumption: put it in the module docstring.
- The explanation is reusable, literature-based, or longer than a short paragraph: put it in `docs/testing/` and link or name it from the test docstring.
- Machine-readable data consumed by pytest belongs in `tests/reference/`.

Do not create one markdown reference file per test. That will become clutter.

## Import And API Smoke Tests

Simple import tests are useful only as a small public API smoke check. They should catch broken package imports, missing required dependencies, circular imports, unexpected heavy top-level side effects, and missing public objects.

Prefer one curated test file that checks:

- `import diskbridge`
- public subpackage imports
- key public classes/functions are reachable
- tiny public object construction or API-shape checks where practical

Do not dynamically import every private module unless the project explicitly wants to enforce that every internal module imports cleanly without optional runtime state.

## Simple Physics Tests

Physics-based pytest checks are encouraged when they are fast and objective. Good examples include:

- conservation on a tiny synthetic grid
- optical depth increasing with column density
- shielding factors decreasing monotonically with column density
- mask weights staying in expected ranges
- UV luminosity or energy budgets being conserved across bins
- unit conversions preserving expected magnitudes
- analytic toy cases with clear expected values
- invalid physical inputs raising clear errors

Move the work to `validation/` when the check is slow, depends on large or external data, requires plots for interpretation, compares against papers or other codes, runs a full pipeline, or asks a human to judge scientific plausibility.
Do not add pytest coverage for code that exists only to run or plot such a
validation workflow.

## What Not To Keep

Do not keep temporary one-off tests that only inspect a private helper unless they protect a meaningful regression. It is acceptable to create temporary local tests during development, but remove them once the change is working unless they serve as useful regression tests.

## Useful Test Categories

- config resolution tests
- unit conversion tests
- shape/order tests
- file writer/reader smoke tests
- RADMC-3D binary-only reader/writer tests, including rejection of unsupported
  ASCII/text files and confirmation that one internal file-backed data
  representation preserves units, shapes, and axis order
- small-grid shielding tests
- small-grid UV product tests
- budget conservation tests
- monotonicity/range tests for simple physical invariants
- public API smoke tests
- error-condition tests
- regression tests for fixed bugs

## Test Cleanup Review

When cleaning or reviewing tests, classify each test as:

- Keep: fast, deterministic, and protects a useful behavior or invariant.
- Merge: duplicates another test contract.
- Rename: useful test with unclear or overly long naming.
- Document: useful test whose intent, reference, or analytic expectation is not clear.
- Move to validation: slow, data-heavy, plot-based, full-pipeline, or human-interpreted.
- Delete: brittle private-helper-only test or import-only noise with no distinct value.

## After Adding Tests

Run the smallest relevant pytest command, for example:

```bash
pytest -q tests/test_uv_products.py
pytest -q tests/test_gow17_config.py
pytest -q tests/test_soft_joos_weight.py
```

If unsure, run:

```bash
pytest -q tests
```

## Final Report

Report which tests were added or changed, what behavior they protect, which pytest command was run, and whether any validation is still needed.
