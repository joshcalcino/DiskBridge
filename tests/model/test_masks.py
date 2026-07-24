"""Tests for DiskBridge disk-mask weighting contracts.

These tests use tiny synthetic models to check objective mask invariants such as
soft-weight ranges, threshold behavior, monotonicity, and binary cell mode.
"""

# db-keywords: disk-mask, validation, units, model, mesh, field
# db-role: validation
# db-scope: test
# db-purpose: Tests for DiskBridge disk-mask weighting contracts.

import numpy as np

from diskbridge._units import Quantity
from diskbridge.model.coords import spherical_grids
from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.masking import (
    DiskFrameData,
    JoosCriterionData,
    _build_disk_weight,
    _compute_disk_orientation,
    _soft_cut_from_ratio,
)
from diskbridge.model.mesh import Axis, Mesh


def _criteria_fixture() -> JoosCriterionData:
    valid = np.ones((1, 1, 2), dtype=bool)
    q = np.ones_like(valid, dtype=float)
    return JoosCriterionData(
        valid=valid,
        q_vr=q,
        q_vz=q,
        q_rot=q,
        q_rho=q,
        hard_pass=valid.copy(),
    )


def _disk_frame_fixture() -> DiskFrameData:
    shape = (1, 1, 2)
    zeros = np.zeros(shape, dtype=float)
    velocity = Quantity(zeros, "cm/s")
    return DiskFrameData(
        k_hat=np.array([0.0, 0.0, 1.0], dtype=float),
        R_d=zeros,
        z_d=zeros,
        vR_d=velocity,
        vphi_d=velocity,
        vz_d=velocity,
        theta_from_midplane=zeros,
    )


def test_soft_cut_threshold_is_half():
    q = np.array([1.0])
    s = _soft_cut_from_ratio(q, delta=0.20, floor=0.0)
    assert np.allclose(s, 0.5)


def test_soft_cut_is_monotonic():
    q = np.array([0.5, 1.0, 2.0])
    s = _soft_cut_from_ratio(q, delta=0.20, floor=0.0)
    assert s[0] < s[1] < s[2]


def test_soft_cut_preserves_smooth_high_score_complement():
    """High disc scores must not create a hard turn-on in their complement."""
    logits = np.array([8.0, 9.0, 10.0])
    q = np.exp(0.20 * logits)
    score = _soft_cut_from_ratio(q, delta=0.20, floor=1.0e-4)
    complement = 1.0 - score

    expected_complement = 1.0 / (1.0 + np.exp(logits))
    assert np.all((score > 0.0) & (score < 1.0))
    assert np.all(np.diff(score) > 0.0)
    assert np.all(complement > 0.0)
    assert np.all(np.diff(complement) < 0.0)
    assert np.allclose(complement, expected_complement, rtol=1.0e-11)


def test_cell_mode_stays_binary():
    w = _build_disk_weight(
        hard_mask=np.array([[[False, True]]]),
        criteria=_criteria_fixture(),
        disk_frame=_disk_frame_fixture(),
        weight_mode="cell",
        soft_delta=0.20,
        weight_m0=0.50,
        weight_floor=1e-4,
    )
    assert set(np.unique(w)) <= {0.0, 1.0}


def test_soft_connected_mode_produces_fractional_weight_on_tiny_model():
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.array([1.0, 2.0]), "cm")),
        theta=Axis(
            edges=Quantity(
                np.array(
                    [
                        np.pi / 2.0 - 0.20,
                        np.pi / 2.0 - 0.10,
                        np.pi / 2.0,
                        np.pi / 2.0 + 0.10,
                        np.pi / 2.0 + 0.20,
                    ]
                ),
                "radian",
            )
        ),
        phi=Axis(edges=Quantity(np.array([0.0, 1.0]), "radian")),
    )
    model.coord_system = model.mesh.coord_system
    model.gas = SubModel(model)

    axis_order = model.mesh.axis_names()
    shape = model.mesh.shape
    rho = np.array([0.12, 0.20, 0.20, 0.12], dtype=float).reshape(shape)
    model.gas_register(
        "density",
        Field("density", Quantity(rho, "g/cm^3"), axis_order),
    )
    model.gas_register(
        "pressure",
        Field("pressure", Quantity(np.full(shape, 0.01), "dyn/cm^2"), axis_order),
    )
    model.gas_register(
        "vr",
        Field("vr", Quantity(np.full(shape, 0.1), "cm/s"), axis_order),
    )
    model.gas_register(
        "vphi",
        Field("vphi", Quantity(np.full(shape, 3.0), "cm/s"), axis_order),
    )
    model.gas_register(
        "vtheta",
        Field("vtheta", Quantity(np.zeros(shape), "cm/s"), axis_order),
    )

    model.set_mask_from_joos_disk(
        rho_disk_min=Quantity(0.1, "g/cm^3"),
        rho_core_min=Quantity(0.1, "g/cm^3"),
        r_max=Quantity(2.0, "cm"),
        weight_mode="soft_connected",
        soft_delta={"vr": 0.20, "vz": 0.20, "rot": 0.20, "rho": 0.30},
        weight_m0=0.50,
        weight_floor=1e-4,
    )

    w = model.gas["disk_weight"].data.to("dimensionless").magnitude
    assert np.nanmin(w) >= 0.0
    assert np.nanmax(w) <= 1.0
    assert np.any((w > 0.0) & (w < 1.0))


def test_axis_radial_support_excludes_massive_outer_stream():
    """A distant transverse flow must not rotate an inner-disc axis."""
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.array([1.0, 2.0, 10.0]), "cm")),
        theta=Axis(
            edges=Quantity(
                np.array([np.pi / 2.0 - 0.05, np.pi / 2.0 + 0.05]),
                "radian",
            )
        ),
        phi=Axis(edges=Quantity(np.array([-0.05, 0.05]), "radian")),
    )
    r_grid, theta_grid, phi_grid = spherical_grids(
        model.mesh.centers("r"),
        model.mesh.centers("theta"),
        model.mesh.centers("phi"),
    )
    shape = model.mesh.shape
    rho = Quantity(np.array([1.0, 100.0]).reshape(shape), "g/cm^3")
    dV = Quantity(np.ones(shape), "cm^3")
    vr = Quantity(np.zeros(shape), "cm/s")
    vphi = Quantity(np.array([1.0, 0.0]).reshape(shape), "cm/s")
    vtheta = Quantity(np.array([0.0, -10.0]).reshape(shape), "cm/s")

    inner_axis = _compute_disk_orientation(
        model,
        rho,
        dV,
        r_grid,
        theta_grid,
        phi_grid,
        vr,
        vphi,
        vtheta,
        Quantity(0.5, "g/cm^3"),
        Quantity(2.0, "cm"),
    )
    full_axis = _compute_disk_orientation(
        model,
        rho,
        dV,
        r_grid,
        theta_grid,
        phi_grid,
        vr,
        vphi,
        vtheta,
        Quantity(0.5, "g/cm^3"),
        Quantity(10.0, "cm"),
    )

    assert np.allclose(inner_axis, [0.0, 0.0, 1.0], atol=1e-12)
    assert abs(full_axis[1]) > 0.99

    with np.testing.assert_raises_regex(ValueError, "No cells satisfy"):
        _compute_disk_orientation(
            model,
            rho,
            dV,
            r_grid,
            theta_grid,
            phi_grid,
            vr,
            vphi,
            vtheta,
            Quantity(1.0e6, "g/cm^3"),
            Quantity(2.0, "cm"),
        )
    with np.testing.assert_raises_regex(ValueError, "angular momentum is zero"):
        _compute_disk_orientation(
            model,
            rho,
            dV,
            r_grid,
            theta_grid,
            phi_grid,
            vr,
            Quantity(np.zeros(shape), "cm/s"),
            Quantity(np.zeros(shape), "cm/s"),
            Quantity(0.5, "g/cm^3"),
            Quantity(2.0, "cm"),
        )


def test_joos_axis_support_is_required():
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.array([1.0, 2.0]), "cm")),
        theta=Axis(
            edges=Quantity(
                np.array([np.pi / 2.0 - 0.1, np.pi / 2.0 + 0.1]),
                "radian",
            )
        ),
        phi=Axis(edges=Quantity(np.array([-0.1, 0.1]), "radian")),
    )
    model.coord_system = model.mesh.coord_system
    model.gas = SubModel(model)
    shape = model.mesh.shape
    axis_order = model.mesh.axis_names()
    for name, data, unit in (
        ("density", np.ones(shape), "g/cm^3"),
        ("pressure", np.ones(shape), "dyn/cm^2"),
        ("vr", np.zeros(shape), "cm/s"),
        ("vphi", np.ones(shape), "cm/s"),
        ("vtheta", np.zeros(shape), "cm/s"),
    ):
        model.gas_register(name, Field(name, Quantity(data, unit), axis_order))

    with np.testing.assert_raises_regex(ValueError, "requires r_max_for_axis or r_max"):
        model.set_mask_from_joos_disk(
            rho_disk_min=Quantity(0.1, "g/cm^3"),
        )
