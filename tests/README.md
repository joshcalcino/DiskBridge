# DiskBridge Tests

This directory contains fast pytest tests that should be runnable during ordinary development.

Tests protect code contracts, regression cases, shape/order guarantees, units, config resolution, file I/O smoke paths, expected errors, and simple objective physics invariants. Larger, slower, data-heavy, plot-based, or human-interpreted scientific checks belong in `validation/`.

## Current Layout

```text
tests/
  README.md
  test_public_api.py
  chemistry/
  model/
  radmc3d/
  reference/
```

- `chemistry/`: GOW17, shielding, and chemistry behavior tests.
- `model/`: model object, field, mesh, and mask behavior tests.
- `radmc3d/`: RADMC-3D writer/reader, UV product, dust, radiation, and line-transfer tests.
- `reference/`: machine-readable data consumed by pytest.

Do not keep pytest tests only to test validation-driver helper code. If validation helper logic becomes important enough for commit-time tests, first move the reusable logic into `src/` and then test it under the relevant subsystem.

## Import And API Smoke Tests

Use one curated public API smoke test, normally `tests/test_public_api.py`, for import and public-surface checks. Avoid many repeated import-only tests unless they exercise meaningfully different optional dependency behavior.

## Naming

Use concise names that describe the contract area, not the whole historical bug.

Good examples:

- `test_public_api.py`
- `test_uv_products.py`
- `chemistry/test_shielding.py`
- `model/test_masks.py`
- `radmc3d/test_external_radiation.py`

Avoid long names that encode implementation details or previous fixes. If the suite grows, prefer subsystem folders over repeated prefixes in every filename.

## Documentation Expectations

Every test file should start with a module docstring explaining what behavior it protects. Mention whether the file uses synthetic, analytic, or reference data.

Add short docstrings or comments for non-obvious test functions, especially when a test encodes:

- an analytic limit;
- a conservation or monotonicity invariant;
- a regression for a past bug;
- a literature or benchmark expectation;
- a subtle unit, shape, ordering, or file-format contract.

Do not add boilerplate comments to obvious assertions.

## References

Use `tests/reference/` for machine-readable data consumed by pytest.

Use `docs/testing/` for human-readable explanations of analytic cases, benchmark assumptions, literature references, and recurring testing conventions.

Use inline comments or test docstrings when the explanation is short and local to one assertion. Do not create one markdown reference file per test.

## Cleanup Categories

When reviewing the suite, classify tests as:

- Keep: fast, deterministic, and protects a useful behavior or invariant.
- Merge: duplicates another test contract.
- Rename: useful test with unclear or overly long naming.
- Document: useful test whose intent, reference, or analytic expectation is unclear.
- Move to validation: slow, data-heavy, plot-based, full-pipeline, or human-interpreted.
- Delete: brittle private-helper-only test or import-only noise with no distinct value.
