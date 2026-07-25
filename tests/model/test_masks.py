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
    _apply_connected_midplane_coherence,
    _apply_vertical_connectivity,
    _build_disk_weight,
    _compute_disk_orientation,
    _compute_joos_criterion_data,
    _apply_radial_midplane_connectivity,
    _cylindrical_midplane_support,
    _soft_cut_from_ratio,
)
from diskbridge.model.mesh import Axis, Mesh


def _criteria_fixture() -> JoosCriterionData:
    valid = np.ones((1, 1, 2), dtype=bool)
    q = np.ones_like(valid, dtype=float)
    return JoosCriterionData(
        valid=valid,
        q_pol=q,
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


def test_poloidal_mach_uses_combined_radial_and_vertical_speed():
    """Two subsonic components may still form a supersonic poloidal flow."""
    shape = (1, 1, 1)
    velocity = Quantity(np.full(shape, 0.08), "cm/s")
    frame = DiskFrameData(
        k_hat=np.array([0.0, 0.0, 1.0]),
        R_d=np.ones(shape),
        z_d=np.zeros(shape),
        vR_d=velocity,
        vphi_d=Quantity(np.ones(shape), "cm/s"),
        vz_d=velocity,
        theta_from_midplane=np.zeros(shape),
    )
    criteria = _compute_joos_criterion_data(
        Quantity(np.ones(shape), "g/cm^3"),
        frame,
        Quantity(np.full(shape, 0.01), "dyn/cm^2"),
        np.ones(shape, dtype=bool),
        Quantity(0.1, "g/cm^3"),
        1.0,
        2.0,
    )

    assert np.isclose(criteria.q_pol.item(), 0.1 / np.hypot(0.08, 0.08))
    assert not criteria.hard_pass.item()


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
        r_max_for_axis=Quantity(2.0, "cm"),
        soft_delta={"mach": 0.20, "rot": 0.20, "rho": 0.30},
        weight_m0=0.50,
        weight_floor=1e-4,
    )

    w = model.gas["disk_weight"].data.to("dimensionless").magnitude
    assert np.nanmin(w) >= 0.0
    assert np.nanmax(w) <= 1.0
    assert np.any((w > 0.0) & (w < 1.0))
    assert np.allclose(
        model.gas["disk_weight"].attrs["disk_axis_cartesian"],
        model.gas["disk_mask"].attrs["disk_axis_cartesian"],
    )


def test_transonic_poloidal_limit_rejects_fast_infall():
    """Supersonic poloidal flow must not seed the settled-disc weight."""
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
    for name, values, unit in (
        ("density", np.full(shape, 1.0), "g/cm^3"),
        ("pressure", np.full(shape, 0.01), "dyn/cm^2"),
        ("vr", np.full(shape, 0.6), "cm/s"),
        ("vphi", np.full(shape, 1.0), "cm/s"),
        ("vtheta", np.zeros(shape), "cm/s"),
    ):
        model.gas_register(name, Field(name, Quantity(values, unit), axis_order))

    common = {
        "rho_disk_min": Quantity(0.1, "g/cm^3"),
        "rho_core_min": Quantity(0.1, "g/cm^3"),
        "r_max_for_axis": Quantity(2.0, "cm"),
        "weight_mode": "soft_connected",
        "weight_m0": 0.50,
        "weight_floor": 1.0e-4,
    }
    model.set_mask_from_joos_disk(max_poloidal_mach=10.0, **common)
    loose = model.gas["disk_weight"].data.to("dimensionless").magnitude.copy()
    model.set_mask_from_joos_disk(max_poloidal_mach=1.0, **common)
    strict = model.gas["disk_weight"].data.to("dimensionless").magnitude

    assert np.any(loose > 0.0)
    assert np.all(strict == 0.0)
    assert np.all(strict <= loose)


def test_axis_support_radius_does_not_truncate_disk_weight():
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.array([1.0, 2.0, 3.0]), "cm")),
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
    fields = {
        "density": (np.full(shape, 0.2), "g/cm^3"),
        "pressure": (np.full(shape, 0.01), "dyn/cm^2"),
        "vr": (np.full(shape, 0.1), "cm/s"),
        "vphi": (np.full(shape, 3.0), "cm/s"),
        "vtheta": (np.zeros(shape), "cm/s"),
    }
    for name, (values, unit) in fields.items():
        model.gas_register(name, Field(name, Quantity(values, unit), axis_order))

    model.set_mask_from_joos_disk(
        rho_disk_min=Quantity(0.1, "g/cm^3"),
        rho_core_min=Quantity(0.1, "g/cm^3"),
        r_max_for_axis=Quantity(2.0, "cm"),
        weight_mode="soft_connected",
        weight_m0=0.5,
    )

    weight = model.gas["disk_weight"].data.magnitude
    assert np.any(weight[1] > 0.0)


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


def test_radial_connectivity_caps_outward_weight_and_rejects_reentry():
    values = np.array([0.8, 0.6, 0.7, 0.9], dtype=float).reshape(4, 1, 1)
    valid = np.ones_like(values, dtype=bool)
    frame = DiskFrameData(
        k_hat=np.array([0.0, 0.0, 1.0]),
        R_d=np.arange(4, dtype=float).reshape(4, 1, 1),
        z_d=np.zeros_like(values),
        vR_d=Quantity(np.zeros_like(values), "cm/s"),
        vphi_d=Quantity(np.zeros_like(values), "cm/s"),
        vz_d=Quantity(np.zeros_like(values), "cm/s"),
        theta_from_midplane=np.zeros_like(values),
    )

    connected = _apply_radial_midplane_connectivity(
        values,
        frame,
        valid,
        seed_min=0.5,
        floor=0.0,
    )
    assert np.allclose(connected[:, 0, 0], [0.8, 0.6, 0.6, 0.6])

    values[2, 0, 0] = 0.0
    connected = _apply_radial_midplane_connectivity(
        values,
        frame,
        valid,
        seed_min=0.5,
        floor=0.0,
    )
    assert np.allclose(connected[:, 0, 0], [0.8, 0.6, 0.0, 0.0])


def test_cylindrical_support_uses_the_cells_midplane_footpoint():
    """An elevated outer-shell cell may connect to an inner cylindrical footpoint."""
    shape = (2, 3, 2)
    local = np.zeros(shape, dtype=float)
    local[0, 1, :] = 0.9
    local[1, 1, :] = 0.1
    local[1, 2, :] = 0.8
    valid = np.ones(shape, dtype=bool)
    theta = np.broadcast_to(
        np.array([-0.2, 0.0, 0.2])[None, :, None],
        shape,
    )
    radius = np.empty(shape, dtype=float)
    radius[0] = 0.5
    radius[1] = 1.5
    radius[1, 2, :] = 0.5
    phi = np.broadcast_to(
        np.array([-0.5 * np.pi, 0.5 * np.pi])[None, None, :],
        shape,
    )
    zeros = np.zeros(shape, dtype=float)
    frame = DiskFrameData(
        k_hat=np.array([0.0, 0.0, 1.0]),
        R_d=radius,
        z_d=zeros,
        vR_d=Quantity(zeros, "cm/s"),
        vphi_d=Quantity(zeros, "cm/s"),
        vz_d=Quantity(zeros, "cm/s"),
        theta_from_midplane=theta,
        phi_d=phi,
    )

    connected, support = _cylindrical_midplane_support(
        local,
        frame,
        valid,
        Quantity(np.array([0.0, 1.0, 2.0]), "cm"),
        seed_min=0.5,
    )

    assert np.allclose(support[0], 0.9)
    assert np.allclose(connected[1, 2], 0.8)
    assert np.all(connected[1, 1] == 0.0)


def test_connected_coherence_rejects_narrow_streamer_and_detached_island():
    """Only radially connected, annularly coherent midplane gas remains disc."""
    nr, nphi = 16, 10
    shape = (nr, 1, nphi)
    values = np.full(shape, 0.9, dtype=float)
    support = np.zeros((nr, nphi), dtype=float)
    support[:6] = 0.9
    support[6:, 0] = 0.9
    support[8:11, 1] = 0.9
    radius = np.broadcast_to(
        (np.arange(nr, dtype=float) + 0.5)[:, None, None],
        shape,
    )
    zeros = np.zeros(shape, dtype=float)
    frame = DiskFrameData(
        k_hat=np.array([0.0, 0.0, 1.0]),
        R_d=radius,
        z_d=zeros,
        vR_d=Quantity(zeros, "cm/s"),
        vphi_d=Quantity(zeros, "cm/s"),
        vz_d=Quantity(zeros, "cm/s"),
        theta_from_midplane=zeros,
    )

    coherent, radial, radial_support, coverage, coherence = (
        _apply_connected_midplane_coherence(
            values,
            support,
            frame,
            np.ones(shape, dtype=bool),
            Quantity(np.arange(nr + 1, dtype=float), "cm"),
            seed_min=0.5,
            floor=1.0e-4,
            soft_delta=0.2,
        )
    )

    assert np.allclose(radial_support[:6], 0.9)
    assert np.allclose(radial_support[6:, 0], 0.9)
    assert np.all(radial_support[8:11, 1] == 0.0)
    assert np.all(radial[8:11, 0, 1] == 0.0)
    assert np.allclose(coverage[-1], 0.09)
    assert coherence[-1] < 0.5
    assert np.all(coherent[-1, 0] < 0.5)


def test_connected_coherence_retains_a_full_coherent_annulus():
    nr, nphi = 8, 6
    shape = (nr, 1, nphi)
    values = np.full(shape, 0.8, dtype=float)
    support = np.full((nr, nphi), 0.8, dtype=float)
    radius = np.broadcast_to(
        (np.arange(nr, dtype=float) + 0.5)[:, None, None],
        shape,
    )
    zeros = np.zeros(shape, dtype=float)
    frame = DiskFrameData(
        k_hat=np.array([0.0, 0.0, 1.0]),
        R_d=radius,
        z_d=zeros,
        vR_d=Quantity(zeros, "cm/s"),
        vphi_d=Quantity(zeros, "cm/s"),
        vz_d=Quantity(zeros, "cm/s"),
        theta_from_midplane=zeros,
    )

    coherent, _radial, _radial_support, coverage, coherence = (
        _apply_connected_midplane_coherence(
            values,
            support,
            frame,
            np.ones(shape, dtype=bool),
            Quantity(np.arange(nr + 1, dtype=float), "cm"),
            seed_min=0.5,
            floor=1.0e-4,
            soft_delta=0.2,
        )
    )

    assert np.allclose(coverage, 0.8)
    assert np.all(coherence >= 0.8)
    assert np.allclose(coherent, 0.8)


def test_vertical_connectivity_rejects_reentry_on_each_side():
    """A failed layer blocks re-entry without erasing the opposite side."""
    shape = (1, 6, 4)
    values = np.zeros(shape, dtype=float)
    values[:, 2:4, :] = 0.9
    values[:, :2, :] = 0.8
    values[:, 4:, 0] = 0.9
    values[:, 5, 1] = 0.9
    theta_centers = np.array([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25])
    theta = np.broadcast_to(theta_centers[None, :, None], shape)
    radius = np.full(shape, 0.5, dtype=float)
    zeros = np.zeros(shape, dtype=float)
    frame = DiskFrameData(
        k_hat=np.array([0.0, 0.0, 1.0]),
        R_d=radius,
        z_d=zeros,
        vR_d=Quantity(zeros, "cm/s"),
        vphi_d=Quantity(zeros, "cm/s"),
        vz_d=Quantity(zeros, "cm/s"),
        theta_from_midplane=theta,
    )

    vertical, support = _apply_vertical_connectivity(
        values,
        frame,
        np.ones(shape, dtype=bool),
        Quantity(np.array([0.0, 1.0]), "cm"),
        np.linspace(-1.5, 1.5, 7),
        floor=1.0e-4,
    )

    assert support[0, 5, 1] == 0.0
    assert vertical[0, 5, 1] == 0.0
    assert np.isclose(vertical[0, 5, 0], 0.9)
    assert np.allclose(vertical[0, 0], 0.8)


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

    with np.testing.assert_raises_regex(TypeError, "r_max_for_axis"):
        model.set_mask_from_joos_disk(
            rho_disk_min=Quantity(0.1, "g/cm^3"),
        )
