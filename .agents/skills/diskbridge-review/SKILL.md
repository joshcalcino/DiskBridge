---
name: diskbridge-review
description: Review a DiskBridge plan, implementation, test, validation, or documentation change before accepting it. Use when the user asks to check, review, critique, verify, compare implementation against plan, or decide whether work is complete.
---

# DiskBridge Review Skill

Use this skill to review plans or completed implementations.

## Main Rule

Review first. Do not edit code unless explicitly asked.

## Plan Review

Check whether the plan:

- has a clear goal;
- separates refactoring from physics changes;
- identifies affected modules;
- identifies relevant capability keywords, tagged code inspected, and the
  symbol-index searches plus reuse/extend/merge/add decision;
- identifies relevant utility-consolidation guide entries and helper-module
  impact when the change touches reusable helpers;
- identifies public APIs and config keys;
- identifies file outputs and array-shape assumptions;
- identifies whether RADMC-3D I/O remains binary-only with one supported
  internal file-backed data representation;
- includes appropriate fast pytest tests;
- includes simple objective physics tests when they are cheap and deterministic;
- includes validation work when physical behavior changes need slower, broader, or human-interpreted checks;
- avoids fallback branches and duplicate pathways;
- respects DiskBridge module boundaries;
- respects Pint/public API and raw-array/internal-kernel separation;
- has clear completion criteria.

## Implementation Review

Check whether the implementation:

- follows the plan;
- checked existing capability tags and generated symbol entries before adding
  new code or private helpers;
- checked the utility-consolidation guide before adding reusable helpers;
- changed more files than necessary;
- changed scientific behavior intentionally or accidentally;
- introduced duplicate logic;
- silently changed defaults;
- changed units, shapes, or file outputs;
- added ASCII/text RADMC-3D fallbacks, broad extension discovery, or parallel
  data structures for the same RADMC-3D quantity;
- added useful tests;
- added or updated sparse capability tags when new discoverable canonical code
  was added or moved;
- added or proposed required validations;
- updated documentation;
- updated the project plan;
- added release notes if needed;
- left temporary/debug code behind.

## Test Review

Check whether tests are fast, deterministic, pytest-based, focused on behavior or simple objective physics invariants, broad enough to protect useful behavior, and not overly specific private-helper tests that should be discarded.

Import/API smoke coverage should normally be one curated public API test, not many repeated import-only tests.

Test files should have concise names, a module docstring explaining the protected behavior, and short function docstrings or comments for non-obvious invariants, regressions, analytic limits, or literature-based expectations. Check that machine-readable reference data lives in `tests/reference/` and human-readable reference explanations live in `docs/testing/`. Flag any `tests/validation/` path as wrong. Also flag pytest files that exist only to test validation-specific drivers, configs, plotting helpers, or workflow glue.

## Validation Review

Check whether validations live under the repo-level `validation/` directory, have a scientific question, define expected behavior, produce plots, summaries, and reports, require human interpretation when appropriate, and are not simple objective checks that should run in pytest instead. Reject validation workflows placed under `tests/`, and reject pytest coverage added only for validation-specific helpers.

## Documentation Review

Check whether documentation describes current behavior only, avoids historical correction language, matches the implemented code, is concise, and belongs in `docs/` rather than `projects/`.

## Output Format

Use this structure:

```text
Decision: accept / revise / reject
Main issues:
- ...
Required changes:
- ...
Missing tests:
- ...
Missing validations:
- ...
Documentation/release-note issues:
- ...
Minimal path to acceptance:
- ...
```

If reviewing against a plan, also state whether the plan file should be moved to `projects/completed/`, `projects/superseded/`, or remain in `projects/active/`.
