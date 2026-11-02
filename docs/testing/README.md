# DiskBridge Testing Notes

Pytest coverage of example, problem, project, campaign or validation pipelines is
prohibited, including their drivers, configurations/defaults, plotting and
cluster scripts, source/AST assertions and workflow glue. Mocks and tiny inputs
do not create an exception. Delete these tests rather than renaming, relocating
or recreating them. Test reusable package behavior or core tooling directly;
standalone scientific validations remain outside pytest.

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
