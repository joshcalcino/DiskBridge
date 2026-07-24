"""Tests for conservative finite-time radial dust transport."""

# db-keywords: dust, model, arrays, coordinates, units
# db-role: validation
# db-scope: test
# db-purpose: Protect radial drift signs, diffusion, and mass conservation.

import numpy as np

from diskbridge._units import Quantity, units
from diskbridge.model.dust_transport import (
    build_smoothed_radial_gas_background,
    dust_diffusivity,
    evolve_radial_surface_density,
    pressure_drift_velocity,
    smoothed_log_pressure_gradient,
)


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


def test_diffusion_conserves_mass_with_closed_boundaries():
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
    assert np.isclose(
        diagnostics.final_mass.to("g").magnitude,
        diagnostics.initial_mass.to("g").magnitude,
        rtol=1.0e-12,
    )
    assert diagnostics.inner_mass_lost.to("g").magnitude == 0.0
    assert diagnostics.outer_mass_lost.to("g").magnitude == 0.0


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
