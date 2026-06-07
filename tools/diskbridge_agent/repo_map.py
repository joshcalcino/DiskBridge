"""Print a compact map of DiskBridge workflow files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = {
    "skills": ROOT / ".agents" / "skills",
    "active_projects": ROOT / "projects" / "active",
    "completed_projects": ROOT / "projects" / "completed",
    "superseded_projects": ROOT / "projects" / "superseded",
    "abandoned_projects": ROOT / "projects" / "abandoned",
    "project_templates": ROOT / "projects" / "templates",
    "tests": ROOT / "tests",
    "validations": ROOT / "validation",
}
IGNORED_NAMES = {".DS_Store", "__pycache__"}
ALLOWED_PROJECT_ROOT_FILES = {"README.md", "index.md"}


def iter_paths(path: Path, pattern: str = "*") -> list[Path]:
    """Return sorted paths under a directory, relative to the repo root."""
    if not path.exists():
        return []
    return sorted(
        item.relative_to(ROOT)
        for item in path.glob(pattern)
        if item.name not in IGNORED_NAMES and not item.name.startswith("._")
    )


def unclassified_project_records() -> list[Path]:
    """Return top-level project records outside the status directories."""
    projects = ROOT / "projects"
    if not projects.exists():
        return []

    return sorted(
        item.relative_to(ROOT)
        for item in projects.iterdir()
        if item.is_file()
        and item.name not in IGNORED_NAMES
        and not item.name.startswith("._")
        and item.name not in ALLOWED_PROJECT_ROOT_FILES
    )


def print_repo_map() -> None:
    """Print a concise repository workflow map."""
    print(f"repo: {ROOT}")
    for label, path in SECTIONS.items():
        print(f"\n[{label}]")
        if label == "skills":
            paths = iter_paths(path, "*/SKILL.md")
        elif label in {
            "active_projects",
            "completed_projects",
            "superseded_projects",
            "abandoned_projects",
        }:
            paths = iter_paths(path, "*")
        elif label == "tests":
            paths = iter_paths(path, "**/test_*.py")
        elif label == "validations":
            paths = iter_paths(path, "*")
        else:
            paths = iter_paths(path, "*.md")

        if not paths:
            print("(none)")
            continue

        for item in paths:
            print(item)

    print("\n[unclassified_project_records]")
    paths = unclassified_project_records()
    if not paths:
        print("(none)")
    else:
        for item in paths:
            print(item)


def main() -> None:
    """Run the command-line interface."""
    print_repo_map()


if __name__ == "__main__":
    main()
