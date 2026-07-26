"""Tests for conservative and parameterized radial dust distributions."""

# db-keywords: dust, model, arrays, coordinates, units
# db-role: validation
# db-scope: test
# db-purpose: Protect radial drift signs, diffusion, and mass conservation.

import numpy as np
import pytest
from scipy.special import erfc

from diskbridge._units import Quantity, units
from diskbridge.model.core import Model, SubModel
from diskbridge.model.dust import Dust
from diskbridge.model.dust_transport import (
    _smooth_radial_amax_distribution,
    apply_radial_amax,
    apply_radial_dust_transport,
    build_smoothed_radial_gas_background,
    dust_diffusivity,
    evolve_radial_surface_density,
    pressure_drift_velocity,
    smoothed_log_pressure_gradient,
)
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh


def test_smoothed_gas_background_preserves_supported_power_law_and_ideal_gas_closure():
    radius = Quantity(np.geomspace(1.0, 100.0, 41), "au")
    r_au = radius.to("au").magnitude
    sigma = Quantity(100.0 * r_au**-1.5, "g/cm^2")
    temperature = Quantity(200.0 * r_au**-0.5, "K")
    pressure = Quantity(1.0e-4 * r_au**-2.5, "dyn/cm^2")

    background = build_smoothed_radial_gas_background(
        radius,
        sigma,
        temperature,
        pressure,
        mean_molecular_weight=2.35,
        smoothing_bins=1,
    )

    assert np.allclose(background.surface_density.magnitude, sigma.magnitude)
    assert np.allclose(background.temperature.magnitude, temperature.magnitude)
    assert np.allclose(background.pressure.magnitude, pressure.magnitude)
    assert np.allclose(background.pressure_log_gradient, -2.5)
    pressure_from_state = (
        background.midplane_density
        * units("k_B")
        * background.temperature
        / (2.35 * units("m_H"))
    ).to("dyn/cm^2")
    assert np.allclose(pressure_from_state.magnitude, background.pressure.magnitude)


def test_smoothed_gas_background_removes_pressure_traps_and_bounds_cliffs():
    radius = Quantity(np.geomspace(10.0, 1_000.0, 61), "au")
    r_au = radius.to("au").magnitude
    sigma_values = 30.0 * (r_au / 10.0) ** -1.5
    pressure_values = 1.0e-5 * (r_au / 10.0) ** -2.5
    sigma_values[35:39] *= 5.0
    pressure_values[35:39] *= 20.0
    pressure_values[50:] *= 1.0e-8

    background = build_smoothed_radial_gas_background(
        radius,
        Quantity(sigma_values, "g/cm^2"),
        Quantity(100.0 * (r_au / 10.0) ** -0.4, "K"),
        Quantity(pressure_values, "dyn/cm^2"),
        mean_molecular_weight=2.35,
        smoothing_bins=7,
        pressure_log_gradient_bounds=(-4.0, -0.2),
    )

    assert np.all(np.diff(background.pressure.magnitude) < 0.0)
    assert np.all(np.diff(background.surface_density.magnitude) <= 0.0)
    assert np.min(background.pressure_log_gradient) >= -4.0
    assert np.max(background.pressure_log_gradient) <= -0.2
    drift = pressure_drift_velocity(
        np.full(r_au.shape, 0.1),
        Quantity(np.full(r_au.shape, 1.0e5), "cm/s"),
        Quantity(np.full(r_au.shape, 1.0e6), "cm/s"),
        background.pressure_log_gradient,
    )
    assert np.all(drift.magnitude < 0.0)


def _transport_inputs(n: int = 8):
    edges = Quantity(np.linspace(1.0, 9.0, n + 1), "au")
    sigma_d = Quantity(np.linspace(1.0, 2.0, n), "g/cm^2")
    sigma_g = Quantity(np.full(n, 100.0), "g/cm^2")
    return edges, sigma_d, sigma_g


def test_smoothed_pressure_gradient_recovers_power_law():
    radius = Quantity(np.geomspace(1.0, 100.0, 41), "au")
    pressure = Quantity(radius.to("au").magnitude ** -2.5, "dyn/cm^2")

    gradient = smoothed_log_pressure_gradient(
        radius,
        pressure,
        smoothing_bins=1,
    )

    assert np.allclose(gradient, -2.5, atol=1.0e-12)


def test_pressure_drift_is_inward_and_peaks_at_stokes_one():
    stokes = np.array([0.1, 1.0, 10.0])
    velocity = pressure_drift_velocity(
        stokes,
        Quantity(np.full(3, 1.0e5), "cm/s"),
        Quantity(np.full(3, 1.0e6), "cm/s"),
        np.full(3, -2.0),
    ).to("cm/s").magnitude

    assert np.all(velocity < 0.0)
    assert abs(velocity[1]) > abs(velocity[0])
    assert np.isclose(velocity[0], velocity[2])


def test_dust_diffusivity_decreases_with_stokes_number():
    diffusion = dust_diffusivity(
        1.0e-3,
        Quantity(np.ones(3), "cm/s"),
        Quantity(np.ones(3), "cm"),
        np.array([0.0, 1.0, 10.0]),
    ).to("cm^2/s").magnitude

    assert np.all(np.diff(diffusion) < 0.0)


def test_smooth_radial_amax_is_bounded_and_locally_normalized():
    """The smooth cutoff follows analytic a_max and depletes large grains."""
    radius = np.array([0.0, 100.0, 250.0, 500.0])
    sizes = np.array([10.0, 100.0, 1_000.0])
    base_fractions = np.array([0.2, 0.3, 0.5])
    radial_amax, retention, fractions = (
        _smooth_radial_amax_distribution(
            radius,
            sizes,
            base_fractions,
            minimum_grain_size=1.0,
            maximum_grain_size=1_000.0,
            reference_amax=100.0,
            reference_radius=250.0,
            radial_exponent=2.0,
            transition_width_dex=0.25,
        )
    )

    expected_amax = np.array([1_000.0, 625.0, 100.0, 25.0])
    assert np.allclose(radial_amax, expected_amax)
    expected_retention = 0.5 * erfc(
        (np.log10(sizes[:, None]) - np.log10(expected_amax[None, :]))
        / (np.sqrt(2.0) * 0.25)
    )
    assert np.allclose(
        retention,
        expected_retention,
    )
    assert retention[1, 2] == pytest.approx(0.5)
    assert np.all(np.diff(radial_amax) < 0.0)
    assert np.all(np.diff(retention, axis=1) <= 0.0)
    assert np.all(fractions >= 0.0)
    assert np.allclose(np.sum(fractions, axis=0), 1.0)
    assert fractions[-1, -1] < fractions[-1, 0]

    for keyword in (
        "minimum_grain_size",
        "maximum_grain_size",
        "reference_amax",
        "reference_radius",
        "radial_exponent",
        "transition_width_dex",
    ):
        parameters = {
            "minimum_grain_size": 1.0,
            "maximum_grain_size": 1_000.0,
            "reference_amax": 100.0,
            "reference_radius": 250.0,
            "radial_exponent": 2.0,
            "transition_width_dex": 0.25,
        }
        parameters[keyword] = 0.0
        with pytest.raises(ValueError, match=keyword):
            _smooth_radial_amax_distribution(
                radius,
                sizes,
                base_fractions,
                **parameters,
            )

    with pytest.raises(ValueError, match="reference_amax"):
        _smooth_radial_amax_distribution(
            radius,
            sizes,
            base_fractions,
            minimum_grain_size=1.0,
            maximum_grain_size=1_000.0,
            reference_amax=2_000.0,
            reference_radius=250.0,
            radial_exponent=2.0,
            transition_width_dex=0.25,
        )


def test_zero_duration_returns_input_without_steps():
    edges, sigma_d, sigma_g = _transport_inputs()
    evolved, diagnostics = evolve_radial_surface_density(
        edges,
        sigma_d,
        sigma_g,
        Quantity(np.zeros(8), "cm/s"),
        Quantity(np.zeros(8), "cm^2/s"),
        Quantity(0.0, "yr"),
    )

    assert np.array_equal(evolved.magnitude, sigma_d.magnitude)
    assert diagnostics.n_steps == 0
    assert diagnostics.final_mass == diagnostics.initial_mass


def test_diffusion_leaves_through_open_outer_boundary():
    edges, sigma_d, sigma_g = _transport_inputs()
    evolved, diagnostics = evolve_radial_surface_density(
        edges,
        sigma_d,
        sigma_g,
        Quantity(np.zeros(8), "cm/s"),
        Quantity(np.full(8, 1.0e14), "cm^2/s"),
        Quantity(100.0, "yr"),
    )

    assert np.all(evolved.magnitude >= 0.0)
    assert evolved[-1] < sigma_d[-1]
    initial = diagnostics.initial_mass.to("g").magnitude
    final = diagnostics.final_mass.to("g").magnitude
    outer = diagnostics.outer_mass_lost.to("g").magnitude
    assert final < initial
    assert diagnostics.inner_mass_lost.to("g").magnitude == 0.0
    assert outer > 0.0
    assert np.isclose(initial - final, outer, rtol=1.0e-12)


def test_inward_advection_only_loses_mass_through_inner_boundary():
    edges, sigma_d, sigma_g = _transport_inputs()
    evolved, diagnostics = evolve_radial_surface_density(
        edges,
        sigma_d,
        sigma_g,
        Quantity(np.full(8, -1.0e3), "cm/s"),
        Quantity(np.zeros(8), "cm^2/s"),
        Quantity(1.0, "yr"),
    )

    assert np.all(evolved.magnitude >= 0.0)
    initial = diagnostics.initial_mass.to("g").magnitude
    final = diagnostics.final_mass.to("g").magnitude
    inner = diagnostics.inner_mass_lost.to("g").magnitude
    assert final < initial
    assert np.isclose(initial - final, inner, rtol=1.0e-10)
    assert diagnostics.outer_mass_lost.to("g").magnitude == 0.0


def test_apply_radial_transport_registers_cylindrical_settled_density():
    """The model helper must realize its evolved column in disc annuli."""
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), "au")),
        theta=Axis(
            edges=Quantity(
                np.pi / 2.0 + np.array([-0.20, -0.05, 0.05, 0.20]),
                "radian",
            )
        ),
        phi=Axis(
            edges=Quantity(
                np.array([-0.5 * np.pi, 0.5 * np.pi, 1.5 * np.pi]),
                "radian",
            )
        ),
    )
    model.coord_system = model.mesh.coord_system
    model.gas = SubModel(model)
    model.dust = Dust(model)
    model.variables["mstar"] = Quantity(1.0, "solar_mass")

    radius = model.mesh.centers("r").to("au").magnitude
    shape = model.mesh.shape
    angular_pattern = np.array([0.8, 1.2])[None, None, :]
    density = (
        1.0e-14
        * (radius / radius[0])[:, None, None] ** -2.0
        * np.ones((1, shape[1], 1))
        * angular_pattern
    )
    temperature = (
        120.0
        * (radius / radius[0])[:, None, None] ** -0.5
        * np.ones((1, shape[1], shape[2]))
    )
    pressure = (
        Quantity(density, "g/cm^3")
        * units("k_B")
        * Quantity(temperature, "K")
        / (2.35 * units("m_H"))
    ).to("dyn/cm^2")
    axis_order = model.mesh.axis_names()
    for name, data in (
        ("density", Quantity(density, "g/cm^3")),
        ("temperature", Quantity(temperature, "K")),
        ("pressure", pressure),
    ):
        model.gas_register(name, Field(name, data, axis_order))
    model.gas_register(
        "disk_weight",
        Field(
            "disk_weight",
            Quantity(np.ones(shape), "dimensionless"),
            axis_order,
            attrs={
                "disk_axis_cartesian": (np.sin(0.5), 0.0, np.cos(0.5)),
                "weight_m0": 0.5,
                "weight_floor": 1.0e-4,
            },
        ),
    )

    model.dust.add_component_from_mask(
        "disk_weight",
        mode="settling",
        amin=Quantity(90.0, "um"),
        amax=Quantity(110.0, "um"),
        nbin=1,
        dust_to_gas_ratio=0.01,
        alpha=0.01,
        delta=0.01,
        mean_molecular_weight=2.35,
    )
    initial = model.dust["bin_0"]["density"].data.to("g/cm^3").magnitude.copy()

    result = apply_radial_dust_transport(
        model,
        Quantity(50_000.0, "yr"),
        smoothing_bins=1,
    )
    transported_field = model.dust["bin_0"]["density"]
    transported = transported_field.data.to("g/cm^3").magnitude

    assert len(result.bins) == 1
    assert transported_field.attrs["radial_transport"] is True
    assert transported_field.attrs["transport_time_yr"] == 50_000.0
    assert transported_field.attrs["transport_coordinate"] == "disc_cylindrical_radius"
    assert (
        result.bins[0].final_mean_radius.to("au").magnitude
        < result.bins[0].initial_mean_radius.to("au").magnitude
    )
    assert np.all(np.isfinite(transported))
    assert np.all(transported >= 0.0)
    assert np.max(np.abs(transported - initial)) > 1.0e-6 * np.max(initial)

    radial_edges = model.mesh.edges("r").to("cm").magnitude
    radius_grid, theta_grid, phi_grid = np.meshgrid(
        model.mesh.centers("r").to("cm").magnitude,
        model.mesh.centers("theta").to("radian").magnitude,
        model.mesh.centers("phi").to("radian").magnitude,
        indexing="ij",
    )
    x_grid = radius_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y_grid = radius_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = radius_grid * np.cos(theta_grid)
    axis = np.array([np.sin(0.5), 0.0, np.cos(0.5)])
    height = x_grid * axis[0] + y_grid * axis[1] + z_grid * axis[2]
    cylindrical_radius = np.sqrt(np.maximum(radius_grid**2 - height**2, 0.0))
    radial_bin = np.digitize(cylindrical_radius, radial_edges) - 1
    dr3 = np.diff(radial_edges**3) / 3.0
    dcos = np.cos(model.mesh.edges("theta").to("radian").magnitude[:-1]) - np.cos(
        model.mesh.edges("theta").to("radian").magnitude[1:]
    )
    dphi = np.diff(model.mesh.edges("phi").to("radian").magnitude)
    volume = dr3[:, None, None] * dcos[None, :, None] * dphi[None, None, :]
    valid = (radial_bin >= 0) & (radial_bin < radial_edges.size - 1)
    mass = np.bincount(
        radial_bin[valid],
        weights=transported[valid] * volume[valid],
        minlength=radial_edges.size - 1,
    )
    realized = mass / (np.pi * np.diff(radial_edges**2))
    assert np.allclose(
        realized,
        result.bins[0].final_surface_density.to("g/cm^2").magnitude,
    )
    diagnostics = result.bins[0].diagnostics
    accounted = (
        diagnostics.final_mass
        + diagnostics.inner_mass_lost
        + diagnostics.outer_mass_lost
    ).to("g").magnitude
    assert np.isclose(accounted, diagnostics.initial_mass.to("g").magnitude)


def test_transport_does_not_amplify_unsupported_soft_weight_blob():
    """An isolated outer mask tail is not part of the 1-D disc reservoir."""
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.arange(10.0, 70.0, 10.0), "au")),
        theta=Axis(
            edges=Quantity(
                np.pi / 2.0 + np.array([-0.25, -0.08, 0.08, 0.25]),
                "radian",
            )
        ),
        phi=Axis(edges=Quantity(np.array([0.0, np.pi, 2.0 * np.pi]), "radian")),
    )
    model.coord_system = model.mesh.coord_system
    model.gas = SubModel(model)
    model.dust = Dust(model)
    model.variables["mstar"] = Quantity(1.0, "solar_mass")
    shape = model.mesh.shape
    radius = model.mesh.centers("r").to("au").magnitude
    density = 1.0e-14 * (radius / radius[0])[:, None, None] ** -2.0 * np.ones(
        (1, shape[1], shape[2])
    )
    temperature = 120.0 * (radius / radius[0])[:, None, None] ** -0.5 * np.ones(
        (1, shape[1], shape[2])
    )
    pressure = (
        Quantity(density, "g/cm^3")
        * units("k_B")
        * Quantity(temperature, "K")
        / (2.35 * units("m_H"))
    ).to("dyn/cm^2")
    axis_order = model.mesh.axis_names()
    for name, data in (
        ("density", Quantity(density, "g/cm^3")),
        ("temperature", Quantity(temperature, "K")),
        ("pressure", pressure),
    ):
        model.gas_register(name, Field(name, data, axis_order))

    weight = np.zeros(shape, dtype=float)
    weight[:3] = 1.0
    weight[-1, 0, 0] = 2.0e-4
    model.gas_register(
        "disk_weight",
        Field(
            "disk_weight",
            Quantity(weight, "dimensionless"),
            axis_order,
            attrs={
                "disk_axis_cartesian": (0.0, 0.0, 1.0),
                "weight_m0": 0.5,
                "weight_floor": 1.0e-4,
            },
        ),
    )
    model.dust.add_component_from_mask(
        "disk_weight",
        mode="settling",
        amin=Quantity(0.09, "um"),
        amax=Quantity(0.11, "um"),
        nbin=1,
        dust_to_gas_ratio=0.01,
        alpha=0.01,
        delta=0.01,
        mean_molecular_weight=2.35,
    )
    initial = model.dust["bin_0"]["density"].data.to("g/cm^3").magnitude.copy()

    result = apply_radial_dust_transport(
        model,
        Quantity(50_000.0, "yr"),
        smoothing_bins=1,
    )
    transported = model.dust["bin_0"]["density"].data.to("g/cm^3").magnitude

    assert np.array_equal(result.active_annuli, np.array([True, True, True, False, False]))
    assert initial[-1, 0, 0] > 0.0
    assert transported[-1, 0, 0] == initial[-1, 0, 0]
    assert np.max(np.abs(transported[:3] - initial[:3])) > 0.0


def test_radial_amax_sets_total_column_through_connected_soft_tail():
    """The radial a_max must not stop at the mask's boolean 0.5 boundary."""
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.arange(10.0, 70.0, 10.0), "au")),
        theta=Axis(
            edges=Quantity(
                np.pi / 2.0 + np.array([-0.25, -0.08, 0.08, 0.25]),
                "radian",
            )
        ),
        phi=Axis(edges=Quantity(np.array([0.0, np.pi, 2.0 * np.pi]), "radian")),
    )
    model.coord_system = model.mesh.coord_system
    model.gas = SubModel(model)
    model.dust = Dust(model)
    shape = model.mesh.shape
    radius = model.mesh.centers("r").to("au").magnitude
    density = 2.0e-13 * (radius / radius[0])[:, None, None] ** -2.0 * np.ones(
        (1, shape[1], shape[2])
    )
    density[2] *= 1.0e-4
    temperature = 100.0 * (radius / radius[0])[:, None, None] ** -0.5 * np.ones(
        (1, shape[1], shape[2])
    )
    pressure = (
        Quantity(density, "g/cm^3")
        * units("k_B")
        * Quantity(temperature, "K")
        / (2.35 * units("m_H"))
    ).to("dyn/cm^2")
    axis_order = model.mesh.axis_names()
    for name, data in (
        ("density", Quantity(density, "g/cm^3")),
        ("temperature", Quantity(temperature, "K")),
        ("pressure", pressure),
    ):
        model.gas_register(name, Field(name, data, axis_order))

    weight = np.zeros(shape, dtype=float)
    weight[:3] = 1.0
    weight[3] = 0.1
    weight[4] = 0.01
    model.gas_register(
        "disk_weight",
        Field(
            "disk_weight",
            Quantity(weight, "dimensionless"),
            axis_order,
            attrs={
                "disk_axis_cartesian": (0.0, 0.0, 1.0),
                "weight_m0": 0.5,
                "weight_floor": 1.0e-4,
            },
        ),
    )
    model.dust.add_component_from_mask(
        "disk_weight",
        mode="settling",
        amin=Quantity(0.1, "um"),
        amax=Quantity(1_000.0, "um"),
        nbin=2,
        power_index=3.0,
        dust_to_gas_ratio=0.01,
        alpha=0.01,
        delta=0.01,
        mean_molecular_weight=2.35,
    )
    result = apply_radial_amax(
        model,
        reference_amax=Quantity(100.0, "um"),
        reference_radius=Quantity(25.0, "au"),
        radial_exponent=2.0,
        transition_width_dex=0.25,
    )

    assert np.all(result.active_annuli)
    fractions = np.asarray([bin_result.mass_fraction for bin_result in result.bins])
    assert np.allclose(np.sum(fractions, axis=0), 1.0)
    assert fractions[-1, 2] < fractions[-1, 0]
    final_total = np.sum(
        [
            bin_result.final_surface_density.to("g/cm^2").magnitude
            for bin_result in result.bins
        ],
        axis=0,
    )
    expected_total = (
        result.dust_to_gas_ratio
        * result.disc_gas_surface_density.to("g/cm^2").magnitude
    )
    assert np.allclose(
        final_total,
        expected_total,
    )
    amax = result.maximum_grain_size.to("um").magnitude
    assert np.all(np.diff(amax) < 0.0)
    for index in range(2):
        field = model.dust[f"bin_{index}"]["density"]
        assert field.attrs["radial_amax"] is True
        assert field.attrs["reference_amax_um"] == 100.0
        assert field.attrs["reference_radius_au"] == 25.0
        assert field.attrs["radial_exponent"] == 2.0
        assert field.attrs["transition_width_dex"] == 0.25


def test_diffusion_remains_positive_across_sharp_gas_edge():
    edges = Quantity(np.linspace(1.0, 7.0, 7), "au")
    sigma_g = Quantity(np.array([100.0, 30.0, 1.0, 1.0e-8, 1.0e-8, 1.0e-8]), "g/cm^2")
    sigma_d = Quantity(np.array([1.0, 0.3, 0.01, 0.0, 0.0, 0.0]), "g/cm^2")

    evolved, diagnostics = evolve_radial_surface_density(
        edges,
        sigma_d,
        sigma_g,
        Quantity(np.zeros(6), "cm/s"),
        Quantity(np.full(6, 1.0e14), "cm^2/s"),
        Quantity(10.0, "yr"),
    )

    assert np.all(evolved.magnitude >= 0.0)
    assert np.isclose(
        diagnostics.final_mass.to("g").magnitude,
        diagnostics.initial_mass.to("g").magnitude,
        rtol=1.0e-12,
    )
