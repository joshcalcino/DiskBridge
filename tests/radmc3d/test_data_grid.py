"""Tests for RADMC-3D data/grid consistency checks.

These tests write tiny synthetic RADMC-3D-like files and assert that readers
reject outputs whose cell counts do not match the model mesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diskbridge._units import Quantity
from diskbridge.model.core import Model
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.data import RadData


def _model_with_cartesian_shape(nx: int, ny: int, nz: int) -> Model:
    model = Model()
    model.mesh = Mesh.cartesian(
        x=Axis(edges=Quantity(np.arange(nx + 1, dtype=float), "cm")),
        y=Axis(edges=Quantity(np.arange(ny + 1, dtype=float), "cm")),
        z=Axis(edges=Quantity(np.arange(nz + 1, dtype=float), "cm")),
    )
    return model


def _write_amr_grid(path: Path, nx: int, ny: int, nz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "1",
                "0",
                "1",
                "0",
                "1 1 1",
                f"{nx} {ny} {nz}",
                "0.0 1.0",
                "0.0 1.0",
                "0.0 1.0",
            ]
        )
        + "\n"
    )


def _write_dust_temperature(path: Path, ncells: int) -> None:
    with path.open("wb") as f:
        np.asarray([1, 8, ncells, 1], dtype=np.int64).tofile(f)
        np.ones(ncells, dtype=np.float64).tofile(f)


def _write_mean_intensity(path: Path, ncells: int, nwav: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.asarray([2, 8, ncells, nwav], dtype=np.int64).tofile(f)
        np.asarray([1.0e15, 2.0e15], dtype=np.float64).tofile(f)
        np.ones(ncells * nwav, dtype=np.float64).tofile(f)


def test_dust_temperature_cell_count_must_match_model_mesh(tmp_path: Path) -> None:
    model = _model_with_cartesian_shape(2, 2, 1)
    data = RadData(model, tmp_path)
    _write_amr_grid(tmp_path / "radmc3d_inputs" / "amr_grid.inp", 2, 2, 1)
    _write_dust_temperature(tmp_path / "dust_temperature.bdat", ncells=2)

    with pytest.raises(ValueError, match="RADMC-3D grid mismatch.*file has 2 cells"):
        data.readDustTemp(fname=tmp_path / "dust_temperature.bdat")


def test_mean_intensity_cell_count_must_match_amr_grid(tmp_path: Path) -> None:
    model = _model_with_cartesian_shape(2, 1, 1)
    data = RadData(model, tmp_path)
    _write_amr_grid(tmp_path / "radmc3d_inputs" / "amr_grid.inp", 2, 2, 1)
    _write_mean_intensity(tmp_path / "radmc3d_outputs" / "mean_intensity.bout", ncells=2)

    with pytest.raises(ValueError, match="amr_grid\\.inp.*expects 4"):
        data.read_mean_intensity_file(tmp_path / "radmc3d_outputs" / "mean_intensity.bout")
