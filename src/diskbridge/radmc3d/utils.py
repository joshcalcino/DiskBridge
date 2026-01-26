from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, get_type_hints
from dataclasses import fields
import os
import shlex
import shutil
import subprocess
import numpy as np

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

                source_resolved = source.resolve()

                target = model_dir / source.name

                if target.is_symlink() and target.resolve() == source.resolve():
                    active_symlinks.append(target)
                    continue

                if target.exists() or target.is_symlink():
                    target.unlink()

                os.symlink(str(source_resolved), str(target))
                active_symlinks.append(target)
                logger.debug(f"Created symlink: {target} -> {source}")

    logger.info(f"Created {len(active_symlinks)} symlinks to input files")


def create_radmc3d_symlinks(
    model_dir: Path,
    inputs_dir: Path,
    input_files: List[str],
    active_symlinks: List[Path],
    outputs_dir: Path | None = None,
    output_files: List[str] | None = None,
) -> None:
    file_map: Dict[Path, List[str]] = {inputs_dir: list(input_files)}
    if outputs_dir is not None and output_files is not None:
        file_map[outputs_dir] = list(output_files)

    if inputs_dir.exists():
        opacity_files = list(inputs_dir.glob('dustkappa_*.inp'))
        for opac_file in opacity_files:
            file_map[inputs_dir].append(opac_file.name)

    create_symlinks_for_file_map(model_dir, file_map, active_symlinks)


def link_dustkappa_opacities(src_inputs_dir: Path, dest_inputs_dir: Path) -> None:
    dest_inputs_dir.mkdir(parents=True, exist_ok=True)
    file_map: Dict[Path, List[str]] = {src_inputs_dir: ['dustkappa_*.inp']}
    create_symlinks_for_file_map(dest_inputs_dir, file_map, [])


def run_radmc3d_command(cmd: list[str], model_dir: Path) -> tuple[int, str, str]:
    if not cmd:
        raise ValueError("RADMC-3D command list is empty")

    use_shell = False
    if cmd[0] == 'radmc3d':
        exe_override = os.environ.get('RADMC3D_EXECUTABLE')
        if exe_override:
            cmd = [exe_override] + cmd[1:]
        else:
            resolved = shutil.which('radmc3d')
            if resolved is not None:
                cmd = [resolved] + cmd[1:]
            else:
                use_shell = True

    popen_cmd: list[str] | str
    if use_shell:
        popen_cmd = ['bash', '-lc', shlex.join(cmd)]
    else:
        popen_cmd = cmd

    process = subprocess.Popen(
        popen_cmd,
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


def run_radmc3d_and_log(
    cmd: list[str],
    model_dir: Path,
    *,
    section: str,
    log_path: Optional[Path] = None,
    command_str: Optional[str] = None,
    preserve_existing: bool = False,
) -> tuple[int, str, str, str, str]:
    model_dir = Path(model_dir)
    if log_path is None:
        log_path = model_dir / 'radmc3d.out'

    previous_log = ""
    if preserve_existing and log_path.exists():
        try:
            with open(log_path, 'r') as f:
                previous_log = f.read()
        except Exception:
            previous_log = ""

    returncode, stdout, stderr = run_radmc3d_command(cmd, model_dir)

    section_text = f"\n--- {section} ---\n"
    if command_str is not None:
        section_text += f"command = {command_str}\n"
    if stdout:
        section_text += stdout
    if stderr:
        section_text += "\n[stderr]\n"
        section_text += stderr

    combined_log = ""
    try:
        if preserve_existing:
            combined_log = previous_log + section_text
            with open(log_path, 'w') as f:
                f.write(combined_log)
        else:
            with open(log_path, 'a') as f:
                f.write(section_text)
    except Exception:
        combined_log = previous_log + section_text

    return returncode, stdout, stderr, combined_log, section_text


def cleanup_symlink_paths(active_symlinks: List[Path]) -> None:
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
