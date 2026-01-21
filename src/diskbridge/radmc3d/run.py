"""RADMC-3D execution context and runner utilities.

This module provides utilities for managing the symlink context needed for
RADMC-3D runs and executing RADMC-3D commands with proper logging.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING
import shutil
import datetime

if TYPE_CHECKING:
    from diskbridge.model.core import Model

from diskbridge._logging import logger
from .utils import (
    create_radmc3d_symlinks,
    cleanup_symlink_paths,
    run_radmc3d_and_log,
    _extract_radmc_errors,
)


class SymlinkContext:
    """Context manager for RADMC-3D symlink lifecycle.
    
    Creates symlinks on entry and cleans them up on exit, ensuring
    the model directory stays clean even if execution fails.
    
    Parameters
    ----------
    model_dir : Path
        Root directory where RADMC-3D will run
    inputs_dir : Path
        Directory containing input files to symlink
    input_files : Sequence[str]
        File patterns to symlink from inputs_dir
    outputs_dir : Path, optional
        Directory containing output files to symlink
    output_files : Sequence[str], optional
        File patterns to symlink from outputs_dir
        
    Examples
    --------
    >>> with SymlinkContext(
    ...     model_dir=Path('.'),
    ...     inputs_dir=Path('radmc3d_inputs'),
    ...     input_files=['amr_grid.inp', 'dust_density.binp'],
    ... ) as ctx:
    ...     # RADMC-3D runs here with symlinks active
    ...     pass
    >>> # Symlinks automatically cleaned up
    """
    
    def __init__(
        self,
        model_dir: Path,
        inputs_dir: Path,
        input_files: Sequence[str],
        outputs_dir: Optional[Path] = None,
        output_files: Optional[Sequence[str]] = None,
    ):
        self.model_dir = Path(model_dir)
        self.inputs_dir = Path(inputs_dir)
        self.input_files = list(input_files)
        self.outputs_dir = Path(outputs_dir) if outputs_dir else None
        self.output_files = list(output_files) if output_files else None
        self._active_symlinks: list[Path] = []
    
    def __enter__(self):
        """Create symlinks."""
        create_radmc3d_symlinks(
            model_dir=self.model_dir,
            inputs_dir=self.inputs_dir,
            input_files=self.input_files,
            active_symlinks=self._active_symlinks,
            outputs_dir=self.outputs_dir,
            output_files=self.output_files,
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up symlinks."""
        cleanup_symlink_paths(self._active_symlinks)
        return False
    
    def add_symlink(self, source: Path, target: Optional[Path] = None) -> None:
        """Add an additional symlink during the context.
        
        Parameters
        ----------
        source : Path
            Source file to link to
        target : Path, optional
            Target symlink path (default: model_dir / source.name)
        """
        import os
        
        if target is None:
            target = self.model_dir / source.name
        
        if not source.exists():
            logger.warning(f"Source file does not exist: {source}")
            return
        
        if target.is_symlink() and target.resolve() == source.resolve():
            self._active_symlinks.append(target)
            return
        
        if target.exists() or target.is_symlink():
            target.unlink()
        
        os.symlink(str(source.resolve()), str(target))
        self._active_symlinks.append(target)
        logger.debug(f"Created symlink: {target} -> {source}")


def run_radmc3d(
    command: str | Sequence[str],
    model_dir: Path,
    log_section: str,
    log_path: Optional[Path] = None,
    preserve_log: bool = False,
) -> tuple[int, str, str]:
    """Run RADMC-3D command with logging.
    
    Parameters
    ----------
    command : str or Sequence[str]
        RADMC-3D command to run (e.g., 'mctherm' or ['mcmono', 'setthreads', '8'])
    model_dir : Path
        Directory where RADMC-3D will run
    log_section : str
        Section name for log file (e.g., 'mctherm', 'mcmono')
    log_path : Path, optional
        Path to log file (default: model_dir/radmc3d.out)
    preserve_log : bool, optional
        If True, append to existing log (default: False)
        
    Returns
    -------
    returncode : int
        Command return code (0 = success)
    stdout : str
        Standard output
    stderr : str
        Standard error
        
    Raises
    ------
    RuntimeError
        If RADMC-3D command fails
    """
    if isinstance(command, str):
        cmd_list = ['radmc3d', command]
    else:
        cmd_list = ['radmc3d'] + list(command)
    
    if log_path is None:
        log_path = model_dir / 'radmc3d.out'
    
    returncode, stdout, stderr, combined_log, section_text = run_radmc3d_and_log(
        cmd_list,
        model_dir,
        section=log_section,
        log_path=log_path,
        preserve_existing=preserve_log,
    )
    
    if returncode != 0:
        logger.error(f"RADMC-3D {log_section} failed (see {log_path})")
        errors = _extract_radmc_errors(log_path)
        if errors:
            logger.error(f"RADMC-3D errors:\n{errors}")
            raise RuntimeError(f"{log_section} failed with RADMC-3D errors:\n{errors}")
        raise RuntimeError(f"{log_section} failed")
    
    logger.info(f"{log_section} completed (log written to {log_path})")
    return returncode, stdout, stderr


def organize_outputs(
    output_dir: Path,
    model_dir: Path,
    output_files: Sequence[str],
    description: str,
) -> None:
    """Move RADMC-3D output files to organized directory.
    
    Parameters
    ----------
    output_dir : Path
        Target directory for organized outputs
    model_dir : Path
        Directory where RADMC-3D ran
    output_files : Sequence[str]
        Output filenames to move
    description : str
        Description for timestamped README
        
    Notes
    -----
    Creates output_dir if needed and writes a timestamped README
    documenting when outputs were created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for filename in output_files:
        src = model_dir / filename
        if not src.exists():
            continue
        
        dst = output_dir / filename
        if dst.exists():
            dst.unlink()
        
        shutil.move(str(src), str(dst))
        logger.debug(f"Moved {filename} to {output_dir}")
        moved_count += 1
    
    if moved_count > 0:
        readme = output_dir / 'README.txt'
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(readme, 'w') as f:
            f.write(f"Created: {timestamp}\n")
            f.write(f"Command: {description}\n")
        
        logger.info(f"Moved {moved_count} output files to {output_dir}")


def ensure_temperature_symlink(
    model_dir: Path,
    outputs_dir: Path,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Ensure dust_temperature file is symlinked for mcmono.
    
    RADMC-3D mcmono needs temperature file but we keep it organized
    in subdirectories. This creates a symlink if needed.
    
    Parameters
    ----------
    model_dir : Path
        Directory where RADMC-3D will run
    outputs_dir : Path
        Root outputs directory
    output_dir : Path, optional
        Specific output directory to check
        
    Returns
    -------
    Path or None
        Path to created symlink, or None if temperature not found
    """
    import os
    
    # Check if temperature already exists in model_dir
    for suffix in ['.bdat', '.dat']:
        dst_temp = model_dir / f'dust_temperature{suffix}'
        if dst_temp.exists():
            return dst_temp
    
    # Search for temperature file in candidate locations
    search_paths = [
        outputs_dir / 'temperature',
        outputs_dir,
    ]
    if output_dir:
        search_paths.insert(0, output_dir.parent / 'temperature')
        search_paths.insert(0, output_dir)
    
    for suffix in ['.bdat', '.dat']:
        for search_dir in search_paths:
            src_temp = search_dir / f'dust_temperature{suffix}'
            if not src_temp.exists():
                continue
            
            dst_temp = model_dir / f'dust_temperature{suffix}'
            os.symlink(src_temp, dst_temp)
            logger.debug(f"Created temperature symlink: {dst_temp} -> {src_temp}")
            return dst_temp
    
    logger.warning("No dust_temperature file found for symlinking")
    return None
