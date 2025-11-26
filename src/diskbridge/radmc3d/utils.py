from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import subprocess

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


def run_radmc3d_command(cmd: list[str], model_dir: Path) -> tuple[int, str, str]:
    process = subprocess.Popen(
        cmd,
        cwd=str(model_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    try:
        if process.stdout is not None:
            for line in process.stdout:
                logger.info(line.rstrip())
                stdout_lines.append(line)
        process.wait()
        stderr_text = ""
        if process.stderr is not None:
            stderr_text = process.stderr.read()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    return process.returncode, "".join(stdout_lines), stderr_text


def cleanup_symlinks(active_symlinks: List[Path]) -> None:
    removed_count = 0
    for symlink in active_symlinks:
        if symlink.is_symlink():
            symlink.unlink()
            logger.debug(f"Removed symlink: {symlink}")
            removed_count += 1

    active_symlinks.clear()
    logger.info(f"Cleaned up {removed_count} symlinks")

