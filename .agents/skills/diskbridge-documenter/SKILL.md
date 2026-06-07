---
name: diskbridge-documenter
description: Update DiskBridge documentation, NumPy-style docstrings, README files, validation READMEs, or CHANGELOG release notes. Use when user-facing behavior, APIs, config, outputs, physics assumptions, or validation workflows need documentation.
---

# DiskBridge Documenter Skill

Use this skill when documentation should be created or updated.

## Main Rule

Documentation should describe the current behavior of DiskBridge. Do not write documentation as a history of previous mistakes, corrections, or implementation changes.

Development history belongs in `projects/`. Release-facing summaries belong in `CHANGELOG.md`. Stable user-facing explanations belong in `docs/`.

## Documentation Standard

The `docs/` tree is a Sphinx site (myst-parser, autodoc/autosummary, numpydoc, sphinxcontrib-bibtex, pydata-sphinx-theme). The authoring standard is `docs/contributing/documentation.md`. Read it before writing documentation and follow it:

- Guide pages are Markdown, one per physics/workflow area, structured as Overview / Usage / Deep dive / References.
- The API reference is generated from NumPy-style docstrings; do not hand-write API pages.
- Literature is cited inline as author-year text (sphinxcontrib-bibtex roles) with per-page ADS links resolved from `docs/refs.bib`, the single citation source of truth.
- Write from the code and verify against current behavior; older `docs/` notes are reference material only.
- Builds must be warning-free: `sphinx-build -W -b html docs docs/_build/html`.

## Documentation Locations

Use:

- `docs/` for stable explanations of current behavior;
- `docs/guides/<area>.md` for per-area user-guide pages;
- `docs/reference/` for the generated API reference;
- `docs/contributing/documentation.md` for the documentation standard itself;
- `docs/refs.bib` as the single source of truth for literature citations;
- `docs/testing/` for analytic test cases, benchmark assumptions, literature references, and recurring pytest conventions;
- `README.md` for high-level project usage;
- `tests/README.md` for local pytest organization and naming standards;
- NumPy-style docstrings for public functions/classes;
- `validation/<case>/README.md` for validation workflows in the repo-level
  `validation/` directory;
- `CHANGELOG.md` for release-facing summaries;
- `projects/active/` or `projects/completed/` for implementation plans and progress logs.

Do not document or create a `tests/validation/` convention. Validation
workflows are not pytest suites; they live under `validation/<case>/`. Do not
document validation-specific helper pytest files as an expected practice.

## What To Document

Document public APIs, important config options, expected units, file outputs, validation workflows, physics assumptions, current limitations, required user decisions, and reusable test references or analytic assumptions.

For RADMC-3D I/O documentation, state the current binary-only policy and the
single supported internal file-backed RADMC-3D data representation when that
surface is relevant. Do not document ASCII/text fallback paths or parallel data
structures as supported behavior.

Do not document temporary implementation details, old behavior unless needed for migration, previous failed attempts, or "we fixed X by doing Y" language in stable docs.

## Docstring Rules

For new or modified public functions, use NumPy-style docstrings. Include a concise summary, parameters, returns, units if relevant, shape expectations if relevant, and raised errors if important.

Avoid long scientific essays in function docstrings. Put extended explanations in `docs/`.

## Release Notes

Update `CHANGELOG.md` when a change affects public APIs, config keys, file outputs, physics behavior, validation workflows, performance-relevant behavior, or user-facing documentation.

Use categories:

- Added
- Changed
- Fixed
- Removed
- Validation
- Internal

## Final Report

When done, report documentation files updated, docstrings updated, release notes updated or not needed, and any documentation still missing.
