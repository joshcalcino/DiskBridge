"""Run-provenance manifests for reproducible DiskBridge runs."""

# db-keywords: config, serialization, io, paths
# db-role: canonical
# db-scope: package
# db-purpose: Write reproducible, human-readable run-provenance manifests.

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .serialization import jsonable


def _git_commit() -> Optional[str]:
    """Return the current git commit hash, or ``None`` if unavailable.

    Best-effort and non-fatal: any failure (not a git repo, git missing,
    timeout) degrades to ``None`` rather than raising.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def build_run_manifest(
    *,
    config: Optional[Any] = None,
    extra: Optional[Any] = None,
    include_git: bool = True,
    include_time: bool = True,
    params: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a run-provenance manifest dict without writing it.

    Parameters
    ----------
    config, extra : Any, optional
        Caller-supplied run configuration and any additional provenance. Both
        are coerced to JSON-compatible values via
        :func:`diskbridge.serialization.jsonable`.
    include_git : bool, optional
        Capture the current git commit hash (best-effort). Default ``True``.
    include_time : bool, optional
        Record an ISO-8601 UTC timestamp. Disable for deterministic output.
        Default ``True``.
    params : Params, optional
        Parameter set to snapshot. Defaults to the active global
        ``diskbridge.params``.

    Returns
    -------
    dict
        Manifest with ``diskbridge_version`` and a complete ``params`` snapshot,
        plus ``created``/``git_commit``/``config``/``extra`` when applicable.
    """

    import diskbridge  # lazy: avoids an import cycle; initialized at call time

    if params is None:
        params = diskbridge.params

    manifest: Dict[str, Any] = {
        "diskbridge_version": diskbridge.__version__,
        "params": params.to_dict(),
    }
    if include_time:
        manifest["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if include_git:
        manifest["git_commit"] = _git_commit()
    if config is not None:
        manifest["config"] = jsonable(config)
    if extra is not None:
        manifest["extra"] = jsonable(extra)
    return manifest


def write_run_manifest(
    path: "str | Path",
    *,
    config: Optional[Any] = None,
    extra: Optional[Any] = None,
    include_git: bool = True,
    include_time: bool = True,
    params: Optional[Any] = None,
) -> Dict[str, Any]:
    """Write a deterministic, human-readable JSON run-provenance manifest.

    The manifest records the complete active parameter set (via
    :meth:`Params.to_dict`), the DiskBridge version, and—best-effort—a git
    commit and timestamp, so a run can be reproduced and audited later. JSON is
    written with sorted keys and two-space indent for stable diffs.

    Parameters
    ----------
    path : str or Path
        Output ``.json`` path. Parent directories are created if needed.
    config, extra, include_git, include_time, params
        See :func:`build_run_manifest`.

    Returns
    -------
    dict
        The manifest that was written.
    """

    manifest = build_run_manifest(
        config=config,
        extra=extra,
        include_git=include_git,
        include_time=include_time,
        params=params,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
