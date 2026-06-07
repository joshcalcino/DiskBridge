# DiskBridge Testing Notes

This directory documents the reasoning behind DiskBridge's pytest tests.

Use it for human-readable explanations of analytic toy cases, benchmark assumptions, literature references, and recurring testing conventions. Keep executable tests in `tests/`, machine-readable pytest data in `tests/reference/`, and larger scientific validation workflows in `validation/`.

## What Belongs Here

- Analytic limits used by pytest tests.
- Literature references behind objective test expectations.
- Benchmark assumptions that appear in more than one test.
- Shared conventions for units, array ordering, file formats, or tolerances.
- Explanations that are too long for a test docstring but too small to be full validation documentation.

## What Does Not Belong Here

- Full validation reports or plots. Put those in `validation/<case>/`.
- Machine-readable test fixtures. Put those in `tests/reference/`.
- Development history or implementation plans. Put those in `projects/`.
- One-off explanations that fit naturally in a test comment or docstring.

## Writing Guidance

Keep pages short and current. Describe the present testing assumption or analytic result, not the history of previous implementations.

When a test depends on a page here, mention the page in the test module docstring so the connection is easy to find.
