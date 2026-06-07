---
name: diskbridge-refactor
description: Refactor DiskBridge code while preserving scientific behavior. Use when simplifying, reorganizing, de-bloating, moving, merging, deleting duplicate code, improving module boundaries, or cleaning up existing implementation paths.
---

# DiskBridge Refactor Skill

Use this skill when the user asks to refactor, simplify, clean up bloat, remove duplicate code paths, improve module boundaries, or make the codebase easier to maintain.

## Goal

Improve structure without changing scientific behavior unless the user explicitly requests a physics change.

## Before Editing

1. Identify the affected subsystem:
   - model/grid/fields
   - chemistry/GOW17
   - shielding/HEALPix
   - RADMC-3D writing/running
   - UV products
   - line transfer
   - visualization
   - validation/test infrastructure
2. Find public entry points and internal helpers.
3. Identify what must remain stable:
   - public API signatures
   - output file names
   - array shapes/order
   - units
   - config keys
   - validation expectations
   - existing tests
4. Do not delete code only because it has no obvious static caller. Some code may be used by examples, validation drivers, or external workflows.

## Refactor Rules

- Prefer one clear public entry point over many parallel pathways.
- Remove duplicated logic only after identifying the canonical implementation.
- Do not create fallback branches unless the user explicitly requests them.
- Do not mix physics changes with structural cleanup.
- Keep Pint quantities at public boundaries.
- Keep Numba/performance kernels unitless and array-oriented.
- Preserve existing file formats unless the user explicitly requests a format change.
- Keep module boundaries clear:
  - chemistry logic belongs in `src/diskbridge/chemistry/`
  - shielding belongs in `src/diskbridge/chemistry/shielding/`
  - RADMC-3D I/O belongs in `src/diskbridge/radmc3d/`
  - plotting belongs in `src/diskbridge/visualization/`

## After Editing

1. Run the smallest relevant pytest tests.
2. Report what was moved, what was deleted, what public behavior changed, which tests were run, and which validations may need human review.

## Final Report

Separate the final report into:

- Code structure changes
- Scientific behavior changes
- Tests run
- Remaining risks
