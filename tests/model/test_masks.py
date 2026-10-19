"""Tests for DiskBridge disk-mask weighting contracts.

These tests use tiny synthetic models to check objective mask invariants such as
soft-weight ranges, threshold behavior, monotonicity, and connectivity.
"""

# db-keywords: disk-mask, validation, units, model, mesh, field
# db-role: validation
# db-scope: test
# db-purpose: Tests for DiskBridge disk-mask weighting contracts.

import numpy as np
import pytest

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
    _connectivity_gate,
    _cylindrical_midplane_support,
    _soft_cut_from_ratio,
)
from diskbridge.model.mesh import Axis, Mesh


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


def test_connectivity_gate_saturates_smoothly_without_rounding_local_scores():
    """Saturation has no jump or slope discontinuity at the existing seed."""
    scores = np.linspace(0.0, 1.0, 1001)
    gate = _connectivity_gate(scores, 0.5)
    assert np.all(np.diff(gate) >= 0)
    assert np.all((gate >= 0) & (gate <= 1))
    np.testing.assert_array_equal(gate[scores >= 0.5], 1.0)
    assert gate[0] == 0.0
    offsets = np.array([1e-3, 5e-4, 2.5e-4])
    deficit = 1 - _connectivity_gate(0.5-offsets, 0.5)
    # Quadratic approach gives a vanishing left derivative; right is flat.
    np.testing.assert_allclose(deficit[:-1]/deficit[1:], 4.0, rtol=0.002)


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
        1.0,
        2.0,
    )

    assert np.isclose(criteria.q_pol.item(), 0.1 / np.hypot(0.08, 0.08))
    assert criteria.q_pol.item() < 1.0


@pytest.fixture
def rotating_model():
    """Diffuse, subsonic rotating gas with a well-defined angular-momentum axis."""
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.array([1.0, 2.0]), "cm")),
        theta=Axis(edges=Quantity(
            np.pi / 2 + np.array([-0.2, -0.1, 0.0, 0.1, 0.2]), "radian"
        )),
        phi=Axis(edges=Quantity(np.array([0.0, 1.0]), "radian")),
    )
    model.coord_system = model.mesh.coord_system
    model.gas = SubModel(model)
    for name, value, unit in (
        ("density", 1e-24, "g/cm^3"),
        ("pressure", 1e-26, "dyn/cm^2"),
        ("vr", 0.01, "cm/s"),
        ("vphi", 1.0, "cm/s"),
        ("vtheta", 0.0, "cm/s"),
    ):
        model.gas_register(name, Field(
            name, Quantity(np.full(model.mesh.shape, value), unit),
            model.mesh.axis_names(),
        ))
    return model


def test_axis_density_is_explicit(rotating_model):
    with pytest.raises(TypeError, match="rho_core_min"):
        rotating_model.set_mask_from_joos_disk(r_max_for_axis=Quantity(2.0, "cm"))


def test_connected_mask_produces_fractional_weight_on_tiny_model():
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
        rho_core_min=Quantity(0.1, "g/cm^3"),
        r_max_for_axis=Quantity(2.0, "cm"),
        soft_delta={"mach": 0.20, "rot": 0.20},
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
        "rho_core_min": Quantity(0.1, "g/cm^3"),
        "r_max_for_axis": Quantity(2.0, "cm"),
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
        rho_core_min=Quantity(0.1, "g/cm^3"),
        r_max_for_axis=Quantity(2.0, "cm"),
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

    support = _cylindrical_midplane_support(
        local,
        frame,
        valid,
        Quantity(np.array([0.0, 1.0, 2.0]), "cm"),
    )

    assert np.allclose(support[0], 0.9)
    assert np.allclose(support[1], 0.1)
    _, radial, _, _, _ = _apply_connected_midplane_coherence(
        _connectivity_gate(local, 0.5), support, frame, valid,
        Quantity(np.array([0.0, 1.0, 2.0]), "cm"), seed_min=0.5, floor=1e-4,
    )
    assert np.allclose(np.minimum(local[1, 2], radial[1, 2]), 0.8)
    assert np.all(radial[1, 1] < 0.5)


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
        )
    )

    assert np.allclose(radial_support[:6], 1.0)
    assert np.allclose(radial_support[6:, 0], 1.0)
    assert np.all(radial_support[8:11, 1] == 0.0)
    assert np.all(radial[8:11, 0, 1] == 0.0)
    assert np.allclose(coverage[-1], 0.1)
    assert np.all((coherence >= 0.0) & (coherence <= 1.0))
    assert np.all(np.diff(coherence) <= 0.0)
    assert coherence[-1] < 0.5
    assert np.all(coherent[-1, 0] < 0.5)


@pytest.mark.parametrize("support_value", [0.0, 0.5, 0.8, 1.0])
def test_connected_coherence_retains_a_full_coherent_annulus(support_value):
    """Full support must admit pure disc without rounding up partial weights."""
    nr, nphi = 8, 6
    shape = (nr, 1, nphi)
    values = np.full(shape, support_value, dtype=float)
    support = np.full((nr, nphi), support_value, dtype=float)
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
        )
    )

    expected_gate = 0.0 if support_value == 0.0 else 1.0
    assert np.allclose(coverage, expected_gate)
    assert np.all((coherence >= support_value) & (coherence <= 1))
    assert np.allclose(coherent, support_value)
    np.testing.assert_allclose(coherence, expected_gate, rtol=0, atol=2*np.finfo(float).eps)


def test_connected_mask_allows_nearly_pure_disc(rotating_model):
    """Strong local support must not acquire a fixed ISM complement."""
    rotating_model.set_mask_from_joos_disk(
        rho_core_min=Quantity(5e-25, "g/cm^3"),
        r_max_for_axis=Quantity(2.0, "cm"),
    )
    weight = rotating_model.gas["disk_weight"].data.magnitude
    assert weight.max() > 0.999
    assert np.all((weight >= 0) & (weight <= 1))


@pytest.mark.parametrize("direction", ["radial", "vertical"])
def test_connected_gates_preserve_passing_scores_and_taper_weak_links(direction):
    """Passing links leave local fractions intact; failing links attenuate smoothly."""
    shape = (12, 6, 8)
    radius = np.broadcast_to((np.arange(12)+0.5)[:, None, None], shape)
    angles = np.broadcast_to(np.linspace(-0.5, 0.5, 6)[None, :, None], shape)
    zeros = np.zeros(shape)
    frame = DiskFrameData(
        k_hat=np.array([0.0, 0.0, 1.0]), R_d=radius, z_d=zeros,
        vR_d=Quantity(zeros, "cm/s"), vphi_d=Quantity(zeros, "cm/s"),
        vz_d=Quantity(zeros, "cm/s"), theta_from_midplane=angles,
    )
    target = (11, 3, 0) if direction == "radial" else (11, 5, 0)
    sampled = []
    for link_score in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
        local = np.full(shape, 0.95)
        local[0] = 0.6  # This passing inner link must not cap the outer disc.
        if direction == "radial":
            local[5] = link_score
        else:
            local[:, 4] = link_score
        q = (local/(1-local))**0.2
        criteria = JoosCriterionData(
            valid=np.ones(shape, dtype=bool), q_pol=q,
            q_rot=np.full(shape, 1e6),
        )
        weight = _build_disk_weight(
            criteria, frame,
            soft_delta=0.2, weight_m0=0.5, weight_floor=1e-4,
            r_edges=Quantity(np.arange(13), "cm"),
            theta_edges=np.linspace(-0.6, 0.6, 7),
        )
        assert np.all(weight <= local+1e-12)
        sampled.append(weight[target])
        if link_score >= 0.5:
            np.testing.assert_allclose(weight, local, rtol=1e-7, atol=1e-7)
    assert sampled[0] == 0.0
    assert np.all(np.diff(sampled) >= -1e-7)
    assert 0.0 < sampled[1] < sampled[2] < sampled[3] < sampled[4]
    assert sampled[-1] == pytest.approx(0.95)


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
            rho_core_min=Quantity(0.1, "g/cm^3"),
        )
