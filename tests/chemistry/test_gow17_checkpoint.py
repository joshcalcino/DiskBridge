"""Tests for compact, restart-complete GOW17 checkpoints.

These checks protect exact state round trips, mesh/config validation, and the
canonical checkpoint layout without running the full chemistry solver.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from diskbridge._units import Quantity
from diskbridge.chemistry.models.gow17 import (
    I_CO_ICE,
    N_Y,
    _gow17_checkpoint_config,
    _load_gow17_checkpoint,
    _write_gow17_checkpoint,
)
from diskbridge.model.mesh import Axis, Mesh


def _rad_model(tmp_path, *, offset: float = 0.0):
    x_edges = Quantity(np.linspace(offset, offset + 3.0, 4), "au")
    y_edges = Quantity(np.linspace(-2.0, 2.0, 3), "au")
    z_edges = Quantity(np.linspace(-1.0, 1.0, 2), "au")
    mesh = Mesh(
        coord_system="cartesian",
        axes={
            "x": Axis(edges=x_edges),
            "y": Axis(edges=y_edges),
            "z": Axis(edges=z_edges),
        },
    )
    return SimpleNamespace(
        model=SimpleNamespace(mesh=mesh),
        model_dir=tmp_path,
    )


def _checkpoint_config(tmp_path, *, resume: bool = True):
    return {
        "enabled": True,
        "resume": resume,
        "path": tmp_path / "checkpoint",
        "every": 1,
        "config_fingerprint": "test-scientific-config",
    }


def test_compact_checkpoint_round_trip_preserves_restart_state(tmp_path, caplog):
    """Every array consumed by resume must round-trip without conversion loss."""
    rad = _rad_model(tmp_path)
    config = _checkpoint_config(tmp_path)
    shape = rad.model.mesh.shape
    ncells = int(np.prod(shape))
    rng = np.random.default_rng(20260714)
    y = rng.normal(size=(ncells, N_Y))
    theta_h2 = rng.random(ncells)
    theta_co = rng.random(ncells)
    theta_c = rng.random(ncells)
    status = rng.integers(-3, 5, size=ncells, dtype=np.int32)
    histories = {
        "d_h2_hist": [0.5, 0.1],
        "equilibrium_solver_diag_hist": {"calls": [2, 4]},
    }

    _write_gow17_checkpoint(
        rad,
        config,
        y=y,
        theta_h2=theta_h2,
        theta_co=theta_co,
        theta_c=theta_c,
        status=status,
        completed_updates=2,
        total_updates=5,
        t_target_yr=100.0,
        histories=histories,
    )

    checkpoint_file = config["path"] / "gow17_state.h5"
    assert sorted(path.name for path in config["path"].iterdir()) == [
        "gow17_state.h5"
    ]
    with h5py.File(checkpoint_file, "r") as handle:
        assert set(handle) == {
            "histories_json",
            "status",
            "theta_c",
            "theta_co",
            "theta_h2",
            "y",
        }
        assert handle["y"].dtype == np.dtype("float64")
        assert handle["status"].dtype == np.dtype("int32")

    restored = _load_gow17_checkpoint(
        rad,
        config,
        shape=shape,
        ncells=ncells,
        total_updates=5,
        enable_co_phase=True,
    )

    assert restored is not None
    np.testing.assert_array_equal(restored.y, y)
    np.testing.assert_array_equal(restored.theta_h2, theta_h2)
    np.testing.assert_array_equal(restored.theta_co, theta_co)
    np.testing.assert_array_equal(restored.theta_c, theta_c)
    np.testing.assert_array_equal(restored.status, status)
    assert restored.completed_updates == 2
    assert restored.histories == histories
    assert "GiB in" in caplog.text
    assert "GiB/s" in caplog.text


def test_checkpoint_rejects_same_shape_with_different_coordinates(tmp_path):
    """Shape equality alone must not permit resume on a different mesh."""
    rad = _rad_model(tmp_path)
    config = _checkpoint_config(tmp_path)
    shape = rad.model.mesh.shape
    ncells = int(np.prod(shape))
    _write_gow17_checkpoint(
        rad,
        config,
        y=np.zeros((ncells, N_Y)),
        theta_h2=np.ones(ncells),
        theta_co=np.ones(ncells),
        theta_c=np.ones(ncells),
        status=np.zeros(ncells, dtype=np.int32),
        completed_updates=1,
        total_updates=3,
        t_target_yr=None,
        histories={},
    )

    shifted_rad = _rad_model(tmp_path, offset=0.25)
    with pytest.raises(ValueError, match="mesh coordinates"):
        _load_gow17_checkpoint(
            shifted_rad,
            config,
            shape=shape,
            ncells=ncells,
            total_updates=3,
            enable_co_phase=True,
        )


def test_checkpoint_rejects_different_scientific_config(tmp_path):
    """Resume must not combine saved state with changed chemistry settings."""
    rad = _rad_model(tmp_path)
    config = _checkpoint_config(tmp_path)
    shape = rad.model.mesh.shape
    ncells = int(np.prod(shape))
    _write_gow17_checkpoint(
        rad,
        config,
        y=np.zeros((ncells, N_Y)),
        theta_h2=np.ones(ncells),
        theta_co=np.ones(ncells),
        theta_c=np.ones(ncells),
        status=np.zeros(ncells, dtype=np.int32),
        completed_updates=1,
        total_updates=3,
        t_target_yr=None,
        histories={},
    )

    changed_config = dict(config, config_fingerprint="changed-scientific-config")
    with pytest.raises(ValueError, match="scientific configuration"):
        _load_gow17_checkpoint(
            rad,
            changed_config,
            shape=shape,
            ncells=ncells,
            total_updates=3,
            enable_co_phase=True,
        )


def test_checkpoint_disables_co_ice_when_active_config_does(tmp_path):
    """Resume still projects the saved state onto the active CO-phase choice."""
    rad = _rad_model(tmp_path)
    config = _checkpoint_config(tmp_path)
    shape = rad.model.mesh.shape
    ncells = int(np.prod(shape))
    y = np.zeros((ncells, N_Y))
    y[:, I_CO_ICE] = 0.25
    _write_gow17_checkpoint(
        rad,
        config,
        y=y,
        theta_h2=np.ones(ncells),
        theta_co=np.ones(ncells),
        theta_c=np.ones(ncells),
        status=np.zeros(ncells, dtype=np.int32),
        completed_updates=1,
        total_updates=3,
        t_target_yr=None,
        histories={},
    )

    restored = _load_gow17_checkpoint(
        rad,
        config,
        shape=shape,
        ncells=ncells,
        total_updates=3,
        enable_co_phase=False,
    )

    assert restored is not None
    np.testing.assert_array_equal(restored.y[:, I_CO_ICE], 0.0)


def test_checkpoint_rejects_incomplete_state_file(tmp_path):
    """A partially written state file must fail with the missing dataset name."""
    rad = _rad_model(tmp_path)
    config = _checkpoint_config(tmp_path)
    shape = rad.model.mesh.shape
    ncells = int(np.prod(shape))
    _write_gow17_checkpoint(
        rad,
        config,
        y=np.zeros((ncells, N_Y)),
        theta_h2=np.ones(ncells),
        theta_co=np.ones(ncells),
        theta_c=np.ones(ncells),
        status=np.zeros(ncells, dtype=np.int32),
        completed_updates=1,
        total_updates=3,
        t_target_yr=None,
        histories={},
    )
    with h5py.File(config["path"] / "gow17_state.h5", "a") as handle:
        del handle["theta_co"]

    with pytest.raises(KeyError, match="incomplete.*theta_co"):
        _load_gow17_checkpoint(
            rad,
            config,
            shape=shape,
            ncells=ncells,
            total_updates=3,
            enable_co_phase=True,
        )


def test_checkpoint_config_rejects_obsolete_full_model_option(tmp_path):
    """The compact format has no dust/full-model serialization switch."""
    rad = _rad_model(tmp_path)
    config = {
        "checkpoint": {"enabled": True, "include_dust": True},
        "_output_dir": str(tmp_path),
    }

    with pytest.raises(ValueError, match="include_dust is no longer supported"):
        _gow17_checkpoint_config(rad, config)
