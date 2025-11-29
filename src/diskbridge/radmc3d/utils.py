from __future__ import annotations

from pathlib import Path
from typing import Dict, List, get_type_hints
from dataclasses import fields
import subprocess

from diskbridge._logging import logger
from diskbridge._units import Quantity
from diskbridge._params import (
    Params,
    DEFAULT_PARAMS_FILE,
    _parse_param_file,
    _parse_value,
)


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


def _read_params_snapshot(params_path: Path) -> Params:
    values = _parse_param_file(DEFAULT_PARAMS_FILE)
    user_vals = _parse_param_file(params_path)
    values.update(user_vals)

    type_hints = get_type_hints(Params)
    kwargs = {}
    for field in fields(Params):
        key = field.name
        if key not in values:
            raise KeyError(
                f"Missing required parameter '{key}' in params.txt (no default provided)."
            )
        raw = values[key]
        field_type = type_hints[key]
        parsed = _parse_value(raw, field_type, param_name=key)
        kwargs[key] = parsed

    return Params(**kwargs)


def _normalize_param_value(val):
    if isinstance(val, Quantity):
        base = val.to_base_units()
        return float(base.magnitude), str(base.units)
    if isinstance(val, list):
        return tuple(_normalize_param_value(v) for v in val)
    return val


def _params_signature(params_obj: Params, names: tuple[str, ...]) -> tuple:
    items = []
    for name in names:
        value = getattr(params_obj, name)
        items.append((name, _normalize_param_value(value)))
    return tuple(items)
