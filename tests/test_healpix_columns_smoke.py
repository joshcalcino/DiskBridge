from __future__ import annotations

import numpy as np
import pytest

from diskbridge.model.mesh import Axis, Mesh
from diskbridge._units import Quantity
from diskbridge.chemistry.shielding.healpix_columns import compute_column_rays_healpix


def test_healpix_columns_smoke() -> None:
    pytest.importorskip("healpy")

    mesh = Mesh.cartesian(
        x=Axis(edges=Quantity(np.array([0.0, 1.0, 2.0]), "cm")),
        y=Axis(edges=Quantity(np.array([0.0, 1.0, 2.0]), "cm")),
        z=Axis(edges=Quantity(np.array([0.0, 1.0, 2.0]), "cm")),
    )

    shape = mesh.shape
    n_field = np.full(shape, 1.0, dtype=np.float64)
    candidate_mask = np.ones(shape, dtype=bool)

    candidate_idx, dirs, cols = compute_column_rays_healpix(
        mesh,
        fields={"n": n_field},
        nside=1,
        candidate_mask=candidate_mask,
        progress_chunks=None,
        cache_dir=None,
        self_weight=1.0,
    )

    assert candidate_idx.ndim == 2
    assert candidate_idx.shape[1] == 3

    assert dirs.ndim == 2
    assert dirs.shape[1] == 3

    assert "n" in cols
    N = cols["n"]
    assert N.shape[0] == candidate_idx.shape[0]
    assert N.shape[1] == dirs.shape[0]
    assert np.all(np.isfinite(N))
    assert np.min(N) >= 0.0
