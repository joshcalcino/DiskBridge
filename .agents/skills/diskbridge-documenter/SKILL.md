---
name: diskbridge-documenter
description: Update DiskBridge documentation, NumPy-style docstrings, README files, validation READMEs, or CHANGELOG release notes. Use when user-facing behavior, APIs, config, outputs, physics assumptions, or validation workflows need documentation.
---

# DiskBridge Documenter Skill

Use this skill when documentation should be created or updated.

## Main Rule

Documentation should describe the current behavior of DiskBridge. Do not write documentation as a history of previous mistakes, corrections, or implementation changes.

Development history belongs in `projects/`. Release-facing summaries belong in `CHANGELOG.md`. Stable user-facing explanations belong in `docs/`.

## Documentation Locations

Use:

- `docs/` for stable explanations of current behavior;
- `docs/testing/` for analytic test cases, benchmark assumptions, literature references, and recurring pytest conventions;
- `README.md` for high-level project usage;
- `tests/README.md` for local pytest organization and naming standards;
- NumPy-style docstrings for public functions/classes;
- `validation/<case>/README.md` for validation workflows;
- `CHANGELOG.md` for release-facing summaries;
- `projects/active/` or `projects/completed/` for implementation plans and progress logs.

## What To Document

Document public APIs, important config options, expected units, file outputs, validation workflows, physics assumptions, current limitations, required user decisions, and reusable test references or analytic assumptions.

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
