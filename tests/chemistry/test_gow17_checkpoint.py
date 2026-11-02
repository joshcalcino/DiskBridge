"""Tests for compact, restart-complete GOW17 checkpoints.

These synthetic checks protect exact state round trips, scientific-input
invalidation, strict mesh/file validation and the canonical checkpoint layout
without running the full chemistry solver.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from diskbridge._units import Quantity
from diskbridge.model.dust import DustBin
from diskbridge.model.dust_opacity import DustOpacity
from diskbridge.chemistry.models.gow17.model import (
    I_CO_ICE,
    N_Y,
    _gow17_checkpoint_config,
    _load_gow17_checkpoint,
    _write_gow17_checkpoint,
)
from diskbridge.chemistry.shielding.angular_uv_weights import SHIELDING_ANGULAR_WEIGHTING
from diskbridge.chemistry.shielding.healpix_utils import SHIELDING_RAY_GEOMETRY
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.model.field import Field
from diskbridge.radmc3d.cache import stable_json_hash
from diskbridge.radmc3d.model import DUST_TEMPERATURE_WEIGHTING


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


def _add_dust(rad):
    class DustFields(dict):
        nbin = 2

        def _compute_bin_density(self, index):
            return self[f"density_bin_{index}"]

    dust = DustFields()
    dust.bins = {}
    for index, temperature in enumerate((20.0, 40.0)):
        dust[f"temperature_bin_{index}"] = Field(
            "dust_temperature", Quantity(np.full(rad.model.mesh.shape, temperature), "K"),
            rad.model.mesh.axis_names(),
        )
        dust[f"density_bin_{index}"] = Field(
            "density", Quantity(np.full(rad.model.mesh.shape, 1.e-18), "g/cm^3"),
            rad.model.mesh.axis_names(),
        )
        dust.bins[f"bin_{index}"] = DustBin(
            parent_dust=dust, bin_index=index,
            size=Quantity(index + 1.0, "um"),
            size_min=Quantity(index + 0.5, "um"),
            size_max=Quantity(index + 1.5, "um"),
            mass_fraction=0.5,
            density_material=Quantity(2.7, "g/cm^3"),
            opacity=DustOpacity("mix_2species_60silicates_40carbons"),
        )
    rad.model.dust = dust
    return dust


def test_compact_checkpoint_round_trip_preserves_restart_state(tmp_path, caplog):
    """Every array consumed by resume must round-trip without conversion loss."""
    rad = _rad_model(tmp_path)
    _add_dust(rad)
    config = _gow17_checkpoint_config(rad, {"checkpoint": {"enabled": True}})
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


def test_checkpoint_starts_fresh_for_different_scientific_config(tmp_path, caplog):
    """Changed scientific settings start a new solve without manual cleanup."""
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
    restored = _load_gow17_checkpoint(
        rad,
        changed_config,
        shape=shape,
        ncells=ncells,
        total_updates=3,
        enable_co_phase=True,
    )
    assert restored is None
    assert "scientific inputs changed; starting a fresh solve" in caplog.text


@pytest.mark.parametrize(
    "missing_identity", [
        "shielding_ray_geometry", "shielding_angular_weighting",
        "dust_temperature_weighting",
    ],
)
def test_checkpoint_requires_current_physics_prescription(tmp_path, missing_identity):
    """Resume cannot restore state from an obsolete scientific prescription."""
    rad = _rad_model(tmp_path)
    current = _gow17_checkpoint_config(
        rad, {"checkpoint": {"enabled": True}}, nside=4,
    )
    scientific_context = {
        "config": {},
        "nside": 4,
        "shielding_ray_geometry": SHIELDING_RAY_GEOMETRY,
        "shielding_angular_weighting": SHIELDING_ANGULAR_WEIGHTING,
        "dust_temperature_weighting": DUST_TEMPERATURE_WEIGHTING,
        "dust_temperatures": {},
        "dust_bins": {},
    }
    assert current["config_fingerprint"] == stable_json_hash(scientific_context)
    scientific_context.pop(missing_identity)
    previous = dict(current, config_fingerprint=stable_json_hash(scientific_context))
    shape = rad.model.mesh.shape
    ncells = int(np.prod(shape))
    _write_gow17_checkpoint(
        rad, previous,
        y=np.zeros((ncells, N_Y)),
        theta_h2=np.ones(ncells), theta_co=np.ones(ncells), theta_c=np.ones(ncells),
        status=np.zeros(ncells, dtype=np.int32),
        completed_updates=1, total_updates=3, t_target_yr=None, histories={},
    )

    assert _load_gow17_checkpoint(
        rad, current, shape=shape, ncells=ncells,
        total_updates=3, enable_co_phase=True,
    ) is None


def test_checkpoint_identity_tracks_individual_dust_temperatures(tmp_path):
    """The representative mean cannot identify different bin temperatures."""
    rad = _rad_model(tmp_path)
    dust = _add_dust(rad)
    current = _gow17_checkpoint_config(rad, {}, nside=4)
    dust["temperature_bin_1"].data.magnitude[...] = 41.0
    changed = _gow17_checkpoint_config(rad, {}, nside=4)

    assert current["config_fingerprint"] != changed["config_fingerprint"]


@pytest.mark.parametrize("changed_input", [
    "material", "porosity", "size", "size_min", "size_max",
    "density_material", "mass_fraction", "density",
])
def test_checkpoint_starts_fresh_for_changed_dust_at_fixed_temperature(
    tmp_path, changed_input, caplog,
):
    """Dust changes cannot reuse a chemical state even at fixed temperature."""
    rad = _rad_model(tmp_path)
    dust = _add_dust(rad)
    settings = {"checkpoint": {"enabled": True}}
    current = _gow17_checkpoint_config(rad, settings, nside=4)
    shape = rad.model.mesh.shape
    ncells = int(np.prod(shape))
    _write_gow17_checkpoint(
        rad, current,
        y=np.zeros((ncells, N_Y)),
        theta_h2=np.ones(ncells), theta_co=np.ones(ncells), theta_c=np.ones(ncells),
        status=np.zeros(ncells, dtype=np.int32),
        completed_updates=1, total_updates=3, t_target_yr=None, histories={},
    )

    grain = dust.bins["bin_0"]
    if changed_input == "material":
        grain.opacity = DustOpacity("mix_2species_ice70")
    elif changed_input == "porosity":
        grain.opacity = DustOpacity(grain.opacity.material, porosity=0.25)
    elif changed_input == "density":
        dust["density_bin_0"].data.magnitude[...] *= 2.0
    else:
        setattr(grain, changed_input, getattr(grain, changed_input) * 1.1)
    changed = _gow17_checkpoint_config(rad, settings, nside=4)

    assert current["config_fingerprint"] != changed["config_fingerprint"]
    assert _load_gow17_checkpoint(
        rad, changed, shape=shape, ncells=ncells,
        total_updates=3, enable_co_phase=True,
    ) is None
    assert "scientific inputs changed; starting a fresh solve" in caplog.text


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


@pytest.mark.parametrize("corruption", ["missing_dataset", "wrong_shape"])
def test_checkpoint_rejects_corrupted_state_even_when_inputs_change(tmp_path, corruption):
    """Scientific invalidation must not hide an incomplete or malformed file."""
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
        if corruption == "wrong_shape":
            handle.create_dataset("theta_co", data=np.ones(ncells + 1))

    error = KeyError if corruption == "missing_dataset" else ValueError
    with pytest.raises(error, match="theta_co"):
        _load_gow17_checkpoint(
            rad,
            dict(config, config_fingerprint="changed-scientific-config"),
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
