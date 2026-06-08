"""Create a new DiskBridge project plan from the standard template."""

# db-keywords: io, paths, workflow, cli
# db-role: entrypoint
# db-scope: module
# db-purpose: Create a new DiskBridge project plan from the standard template.

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "projects" / "templates" / "implementation_plan_template.md"
ACTIVE = ROOT / "projects" / "active"


def slugify(text: str) -> str:
    """Return a simple filename-safe slug."""
    cleaned: list[str] = []
    for char in text.lower().strip():
        if char.isalnum():
            cleaned.append(char)
        elif char in {" ", "_", "-"}:
            cleaned.append("-")

    slug = "".join(cleaned)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def create_plan(title: str) -> Path:
    """Create a new active project plan.

    Parameters
    ----------
    title : str
        Short human-readable plan title.

    Returns
    -------
    pathlib.Path
        Path to the created plan file.
    """
    ACTIVE.mkdir(parents=True, exist_ok=True)

    template = TEMPLATE.read_text(encoding="utf-8")
    today = date.today().isoformat()
    path = ACTIVE / f"{today}-{slugify(title)}.md"
    if path.exists():
        raise FileExistsError(f"Plan already exists: {path}")

    content = template.replace("<Short descriptive plan title>", title)
    content = content.replace("### YYYY-MM-DD", f"### {today}")
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="Short title for the new plan")
    args = parser.parse_args()
    path = create_plan(args.title)
    print(path)


if __name__ == "__main__":
    main()
