"""Shared helper functions for DiskBridge internals."""

# db-keywords: hashing, io, arrays, radmc3d
# db-role: canonical
# db-scope: package
# db-purpose: Shared helper functions for file, array, and RADMC binary fingerprints.

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file's raw bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    """Return the SHA256 digest for an array fingerprint.

    The digest includes the contiguous array dtype string, shape encoded as
    int64, and raw contiguous bytes. This preserves the non-LTE checkpoint
    fingerprint semantics.
    """

    arr = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def sha256_radmc3d_vector_binp(vectors: np.ndarray) -> str:
    """Hash the exact RADMC-3D vector ``.binp`` payload bytes."""

    arr = np.ascontiguousarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"RADMC-3D vector binary hash requires shape (n_cells, 3), got {arr.shape}"
        )
    digest = hashlib.sha256()
    digest.update(np.asarray([1, 8, arr.shape[0]], dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()
