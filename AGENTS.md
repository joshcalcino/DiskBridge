# AGENTS.md

## DiskBridge Context

DiskBridge is a scientific astrophysics codebase for post-processing hydrodynamic simulations, RADMC-3D workflows, chemistry, shielding, dust, gas temperature, and line-transfer preparation. It is research code in active development.

## Repository Layout

- `src/diskbridge/model/`: model objects, mesh, fields, dust, masks, and simulation readers.
- `src/diskbridge/chemistry/`: chemistry API, GOW17, abundance models, and shielding.
- `src/diskbridge/chemistry/shielding/`: shielding and HEALPix-related logic.
- `src/diskbridge/radmc3d/`: RADMC-3D writer, runner, UV products, segmented RT, and line-transfer setup.
- `src/diskbridge/visualization/`: plotting and diagnostics helpers.
- `tests/`: fast pytest tests that check code behavior and prevent regressions.
- `validation/`: larger physics validation workflows. These are not pytest tests.
- `projects/`: active project notes and implementation plans.
- `docs/`: stable explanatory documentation.

## General Rules

- Preserve physical meaning before making code prettier.
- Keep public APIs concise.
- Avoid duplicate code paths and fallback branches unless explicitly requested.
- Do not add unnecessary abstraction.
- Keep Pint quantities at public/user-facing boundaries.
- Keep performance-critical kernels unitless and array-oriented.
- Prefer NumPy/Numba-compatible implementations for heavy per-cell operations.
- Use NumPy-style docstrings for new or modified public functions.
- Documentation should describe current behavior only, not previous corrections or change history.
- Ask for clarification when critical scientific or workflow information is missing.

## Project Notes

Projects in progress are tracked in `projects/`. When working on an active project, update the relevant project markdown file before editing code. Keep project notes concise and current, and remove stale information rather than accumulating history.

## Planning, Review, And Documentation Workflow

For any non-trivial change, create or update a project plan before editing code. A non-trivial change includes any change that:

- touches multiple files;
- changes public APIs;
- changes configuration;
- changes physics behavior;
- changes validation behavior;
- changes file outputs;
- affects performance-sensitive code;
- modifies chemistry, shielding, RADMC-3D, dust, gas temperature, masks, or line-transfer workflows.

Project plans live in `projects/active/` while work is ongoing. When complete, move them to `projects/completed/`. If replaced, move them to `projects/superseded/`. If discarded, move them to `projects/abandoned/`.

Stable user-facing documentation belongs in `docs/`. Documentation must describe current behavior only, not the history of corrections or previous implementations. Release-facing summaries belong in `CHANGELOG.md`.

The documentation is a Sphinx site (myst-parser, autodoc/autosummary, numpydoc, sphinxcontrib-bibtex, pydata-sphinx-theme). The authoring standard is `docs/contributing/documentation.md`: Markdown guide pages with an Overview / Usage / Deep dive / References structure, NumPy-style docstrings, and literature cited inline as author-year text with per-page ADS links resolved from `docs/refs.bib`. Follow it for any documentation change.

## Skill Usage

When using an AI coding agent that supports skills, use the DiskBridge skills for non-trivial work. Before editing code, state which skill or skills are being used.

Default mapping:

- Planning a change: `diskbridge-planning`
- Reviewing a plan or implementation: `diskbridge-review`
- Adding new code: `diskbridge-implementation`
- Refactoring existing code: `diskbridge-refactor`
- Adding or running fast pytest checks: `diskbridge-tests`
- Creating or running larger physics/workflow checks: `diskbridge-validation`
- Creating, refactoring, or reviewing plotting helpers and diagnostic figures:
  `diskbridge-visualization`
- Updating docs, docstrings, validation READMEs, or release notes: `diskbridge-documenter`

For non-trivial changes, do not skip planning unless the user explicitly says to skip planning.

## Tests Versus Validations

Tests belong in `tests/` and should be fast, deterministic, and runnable with pytest on ordinary development changes. Tests should protect meaningful behavior such as public workflows, regression cases, shape/order guarantees, units, config resolution, file I/O smoke paths, expected errors, and simple objective physics invariants.

Keep import/API smoke coverage curated. Prefer one public API smoke test over many repeated import-only tests. Use it to check that the public package surface imports cleanly and that key public objects are reachable or can be constructed with tiny inputs.

Simple physics checks are appropriate in pytest when they are cheap, deterministic, objective, and based on synthetic or analytic toy inputs. Examples include conservation checks, monotonic shielding trends, valid ranges for mask weights, unit conversion magnitudes, and small-grid budget checks.

Test files should have clear, concise names that describe the contract area, not the full historical bug. Each test file should start with a module docstring explaining what behavior it protects. Non-obvious test functions should have short docstrings or comments explaining the invariant, reference case, or regression.

Use `tests/reference/` for machine-readable reference data consumed by pytest. Use `docs/testing/` for human-readable explanations of analytic cases, benchmark assumptions, literature references, and recurring testing conventions. Keep one-off assertion explanations close to the test instead of creating a separate reference page.

Avoid keeping tiny tests that only inspect one or two private helpers unless they protect a meaningful regression. Temporary local tests are acceptable during development, but remove them once they have served their purpose.

Validations belong in `validation/`. They are for checks that are too slow, broad, data-heavy, interpretive, or workflow-oriented to run on every normal code change. They may run longer pipelines, compare against papers or other codes, generate plots, write output data, and require human interpretation. Validations must not be written as pytest tests.

## Before Editing Code

- Identify which physics or workflow area is affected.
- Check relevant docs and project notes.
- If working on an active project, update the relevant project file before code edits.
- Prefer the smallest coherent change.

## After Editing Code

- Run the smallest relevant pytest tests.
- Report what was tested.
- Clearly separate structural/code cleanup from scientific behavior changes.
