from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from diskbridge._logging import logger


def _extract_radmc_errors(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except Exception:
        return ""
    errors = [line.strip() for line in lines if "ERROR" in line.upper()]
    if not errors:
        return ""
    return "\n".join(errors[-10:])


def create_symlinks_for_file_map(
    model_dir: Path,
    file_map: Dict[Path, List[str]],
    active_symlinks: List[Path],
) -> None:
    import os

    for source_dir, file_patterns in file_map.items():
        if not source_dir.exists():
            continue

        for pattern in file_patterns:
            if "*" in pattern:
                source_files = list(source_dir.glob(pattern))
            else:
                source_files = [source_dir / pattern]

            for source in source_files:
                if not source.exists():
                    continue

                target = model_dir / source.name

                if target.is_symlink() and target.resolve() == source.resolve():
                    active_symlinks.append(target)
                    continue

                if target.exists() or target.is_symlink():
                    target.unlink()

                os.symlink(source, target)
                active_symlinks.append(target)
                logger.debug(f"Created symlink: {target} -> {source}")

    logger.info(f"Created {len(active_symlinks)} symlinks to input files")


def cleanup_symlinks(active_symlinks: List[Path]) -> None:
    removed_count = 0
    for symlink in active_symlinks:
        if symlink.is_symlink():
            symlink.unlink()
            logger.debug(f"Removed symlink: {symlink}")
            removed_count += 1

    active_symlinks.clear()
    logger.info(f"Cleaned up {removed_count} symlinks")

