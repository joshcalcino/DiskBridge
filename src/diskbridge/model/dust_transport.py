"""Finite-time radial transport for axisymmetric dust surface densities."""

# db-keywords: dust, model, arrays, coordinates, units
# db-role: canonical
# db-scope: package
# db-purpose: Conservative finite-time radial dust drift and diffusion.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diskbridge._units import Quantity, units


@dataclass(frozen=True)
class RadialGasBackground:
    """Smoothed axisymmetric gas profiles used by radial dust transport.

    Attributes
    ----------
    radius : Quantity
        Radial cell centres [length], shape ``(n,)``.
    surface_density : Quantity
        Smoothed gas surface density [mass / area], shape ``(n,)``.
    midplane_density : Quantity
        Ideal-gas midplane density reconstructed from ``pressure`` and
        ``temperature`` [mass / volume], shape ``(n,)``.
    temperature : Quantity
        Smoothed midplane gas temperature [temperature], shape ``(n,)``.
    pressure : Quantity
        Smoothed, slope-bounded midplane pressure, shape ``(n,)``.
    pressure_log_gradient : ndarray
        Bounded ``d ln(P) / d ln(R)``, shape ``(n,)``.
    smoothing_bins : int
        Requested odd moving-average width.
    pressure_log_gradient_bounds : tuple[float, float]
        Minimum and maximum permitted logarithmic pressure slopes.
    """

    radius: Quantity
    surface_density: Quantity
    midplane_density: Quantity
    temperature: Quantity
    pressure: Quantity
    pressure_log_gradient: np.ndarray
    smoothing_bins: int
    pressure_log_gradient_bounds: tuple[float, float]


@dataclass(frozen=True)
class RadialTransportDiagnostics:
    """Conservation diagnostics for one radial transport integration.

    Attributes
    ----------
    initial_mass, final_mass : Quantity
        Axisymmetric dust mass before and after transport.
    inner_mass_lost, outer_mass_lost : Quantity
        Integrated outward-oriented boundary losses. Both are reported as
        non-negative masses.
    n_steps : int
        Number of explicit finite-volume substeps.
    """

    initial_mass: Quantity
    final_mass: Quantity
    inner_mass_lost: Quantity
    outer_mass_lost: Quantity
    n_steps: int


def _validated_profile_arrays(
    radius: Quantity,
    profile: Quantity,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(radius.to("cm").magnitude, dtype=float)
    values = np.asarray(profile.to_base_units().magnitude, dtype=float)
    if r.ndim != 1 or values.shape != r.shape:
        raise ValueError("radius and profiles must be matching one-dimensional arrays")
    if r.size < 2 or np.any(~np.isfinite(r)) or np.any(np.diff(r) <= 0.0):
        raise ValueError("radius must contain at least two finite increasing values")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("radial profiles must be finite and positive")
    return r, values


def _smoothing_width(smoothing_bins: int, size: int) -> int:
    if int(smoothing_bins) != smoothing_bins or smoothing_bins < 1:
        raise ValueError("smoothing_bins must be a positive odd integer")
    if smoothing_bins % 2 == 0:
        raise ValueError("smoothing_bins must be odd")
    return min(int(smoothing_bins), size if size % 2 else size - 1)


def _smooth_log_values(values: np.ndarray, width: int) -> np.ndarray:
    log_values = np.log(values)
    if width == 1:
        return log_values
    pad = width // 2
    padded = np.pad(log_values, pad, mode="edge")
    return np.convolve(
        padded,
        np.full(width, 1.0 / width),
        mode="valid",
    )


def _bounded_smoothed_log_profile(
    radius_cm: np.ndarray,
    values: np.ndarray,
    *,
    smoothing_bins: int,
    gradient_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = (float(bound) for bound in gradient_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("log-gradient bounds must be finite and ordered")
    width = _smoothing_width(smoothing_bins, values.size)
    log_radius = np.log(radius_cm)
    log_smoothed = _smooth_log_values(values, width)
    interface_gradient = np.diff(log_smoothed) / np.diff(log_radius)
    interface_gradient = np.clip(interface_gradient, lower, upper)

    log_bounded = np.empty_like(log_smoothed)
    log_bounded[0] = log_smoothed[0]
    log_bounded[1:] = log_bounded[0] + np.cumsum(
        interface_gradient * np.diff(log_radius)
    )
    log_bounded += float(np.mean(log_smoothed - log_bounded))

    gradient = np.empty_like(log_bounded)
    gradient[0] = interface_gradient[0]
    gradient[-1] = interface_gradient[-1]
    log_widths = np.diff(log_radius)
    gradient[1:-1] = (
        interface_gradient[:-1] * log_widths[:-1]
        + interface_gradient[1:] * log_widths[1:]
    ) / (
        log_widths[:-1] + log_widths[1:]
    )
    return np.exp(log_bounded), gradient


def build_smoothed_radial_gas_background(
    radius: Quantity,
    gas_surface_density: Quantity,
    temperature: Quantity,
    pressure: Quantity,
    *,
    mean_molecular_weight: float,
    smoothing_bins: int = 21,
    pressure_log_gradient_bounds: tuple[float, float] = (-5.0, -0.25),
    surface_density_log_gradient_bounds: tuple[float, float] = (-5.0, 0.0),
) -> RadialGasBackground:
    """Build a smooth, trap-free radial gas background from snapshot profiles.

    The input gas column, midplane temperature, and midplane pressure are
    smoothed in logarithmic space. Pressure slopes are bounded to remain
    negative, preventing local pressure maxima from trapping inward-drifting
    dust, while the lower bound softens unresolved radial cliffs. Midplane
    density is reconstructed from the smoothed pressure and temperature using
    the ideal-gas relation.

    Parameters
    ----------
    radius : Quantity
        Increasing radial cell centres [length], shape ``(n,)``.
    gas_surface_density : Quantity
        Positive snapshot gas surface density [mass / area], shape ``(n,)``.
    temperature : Quantity
        Positive snapshot midplane gas temperature, shape ``(n,)``.
    pressure : Quantity
        Positive snapshot midplane gas pressure, shape ``(n,)``.
    mean_molecular_weight : float
        Mean gas-particle mass in hydrogen-mass units.
    smoothing_bins : int, optional
        Odd moving-average width on the native radial grid.
    pressure_log_gradient_bounds : tuple[float, float], optional
        Minimum and maximum allowed ``d ln(P) / d ln(R)``. The maximum must be
        negative to guarantee inward pressure drift.
    surface_density_log_gradient_bounds : tuple[float, float], optional
        Minimum and maximum allowed ``d ln(Sigma_g) / d ln(R)``.

    Returns
    -------
    RadialGasBackground
        Self-consistent smoothed profiles on the input radial centres.

    Raises
    ------
    ValueError
        If profiles or radii are invalid, smoothing width is not a positive odd
        integer, mean molecular weight is not positive, or gradient bounds are
        invalid.
    """
    radius_cm, sigma_base = _validated_profile_arrays(radius, gas_surface_density)
    temperature_radius, temperature_base = _validated_profile_arrays(radius, temperature)
    pressure_radius, pressure_base = _validated_profile_arrays(radius, pressure)
    if not (
        np.array_equal(radius_cm, temperature_radius)
        and np.array_equal(radius_cm, pressure_radius)
    ):
        raise ValueError("all gas-background profiles must use identical radii")
    if not np.isfinite(mean_molecular_weight) or mean_molecular_weight <= 0.0:
        raise ValueError("mean_molecular_weight must be finite and positive")
    if float(pressure_log_gradient_bounds[1]) >= 0.0:
        raise ValueError("maximum pressure log-gradient must be negative")

    sigma_values, _ = _bounded_smoothed_log_profile(
        radius_cm,
        sigma_base,
        smoothing_bins=smoothing_bins,
        gradient_bounds=surface_density_log_gradient_bounds,
    )
    width = _smoothing_width(smoothing_bins, temperature_base.size)
    temperature_values = np.exp(_smooth_log_values(temperature_base, width))
    pressure_values, pressure_gradient = _bounded_smoothed_log_profile(
        radius_cm,
        pressure_base,
        smoothing_bins=smoothing_bins,
        gradient_bounds=pressure_log_gradient_bounds,
    )

    temperature_model = Quantity(temperature_values, temperature.to_base_units().units)
    pressure_model = Quantity(pressure_values, pressure.to_base_units().units)
    midplane_density = (
        pressure_model
        * float(mean_molecular_weight)
        * units("m_H")
        / (units("k_B") * temperature_model)
    ).to("g/cm^3")
    return RadialGasBackground(
        radius=Quantity(radius_cm, "cm"),
        surface_density=Quantity(
            sigma_values,
            gas_surface_density.to_base_units().units,
        ).to("g/cm^2"),
        midplane_density=midplane_density,
        temperature=temperature_model.to("K"),
        pressure=pressure_model.to("dyn/cm^2"),
        pressure_log_gradient=pressure_gradient,
        smoothing_bins=int(smoothing_bins),
        pressure_log_gradient_bounds=(
            float(pressure_log_gradient_bounds[0]),
            float(pressure_log_gradient_bounds[1]),
        ),
    )


def smoothed_log_pressure_gradient(
    radius: Quantity,
    pressure: Quantity,
    *,
    smoothing_bins: int = 9,
) -> np.ndarray:
    """Return a smoothed dimensionless pressure slope.

    Parameters
    ----------
    radius : Quantity
        Positive radial cell centres.
    pressure : Quantity
        Positive pressure samples at the same centres.
    smoothing_bins : int, optional
        Odd moving-average width applied to ``ln(P)`` before differentiating.

    Returns
    -------
    ndarray
        ``d ln(P) / d ln(R)`` at the input radii.
    """
    r, p = _validated_profile_arrays(radius, pressure)
    width = _smoothing_width(smoothing_bins, r.size)
    log_pressure = _smooth_log_values(p, width)
    edge_order = 2 if r.size >= 3 else 1
    return np.gradient(log_pressure, np.log(r), edge_order=edge_order)


def pressure_drift_velocity(
    stokes: np.ndarray,
    sound_speed: Quantity,
    keplerian_velocity: Quantity,
    pressure_log_gradient: np.ndarray,
) -> Quantity:
    """Compute dust drift relative to gas from a pressure slope.

    Parameters
    ----------
    stokes : ndarray
        Dimensionless Stokes number at radial cell centres.
    sound_speed, keplerian_velocity : Quantity
        Sound and Keplerian speeds at the same centres.
    pressure_log_gradient : ndarray
        ``d ln(P) / d ln(R)`` at the same centres.

    Returns
    -------
    Quantity
        Radial drift velocity. Negative values point inward.

    Notes
    -----
    Gas radial advection is intentionally excluded. The returned velocity is
    ``St / (1 + St**2) * c_s**2 / v_K * dlnP/dlnR``.
    """
    st = np.asarray(stokes, dtype=float)
    grad = np.asarray(pressure_log_gradient, dtype=float)
    cs = np.asarray(sound_speed.to("cm/s").magnitude, dtype=float)
    vk = np.asarray(keplerian_velocity.to("cm/s").magnitude, dtype=float)
    if not (st.shape == grad.shape == cs.shape == vk.shape):
        raise ValueError("all drift inputs must have matching shapes")
    if np.any(st < 0.0) or np.any(~np.isfinite(st)):
        raise ValueError("stokes must be finite and non-negative")
    if np.any(vk <= 0.0) or np.any(~np.isfinite(vk)):
        raise ValueError("keplerian_velocity must be finite and positive")
    velocity = (st / (1.0 + st * st)) * (cs * cs / vk) * grad
    return Quantity(velocity, "cm/s")


def dust_diffusivity(
    delta: float,
    sound_speed: Quantity,
    scale_height: Quantity,
    stokes: np.ndarray,
) -> Quantity:
    """Return ``D_d = delta c_s H / (1 + St**2)``.

    Parameters
    ----------
    delta : float
        Dimensionless turbulent diffusion strength.
    sound_speed, scale_height : Quantity
        Radial sound-speed and gas-scale-height profiles.
    stokes : ndarray
        Dimensionless Stokes-number profile.

    Returns
    -------
    Quantity
        Dust diffusivity in area per time.
    """
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be finite and non-negative")
    st = np.asarray(stokes, dtype=float)
    cs = np.asarray(sound_speed.to("cm/s").magnitude, dtype=float)
    height = np.asarray(scale_height.to("cm").magnitude, dtype=float)
    if not (st.shape == cs.shape == height.shape):
        raise ValueError("all diffusivity inputs must have matching shapes")
    if np.any(st < 0.0) or np.any(height < 0.0):
        raise ValueError("stokes and scale_height must be non-negative")
    return Quantity(delta * cs * height / (1.0 + st * st), "cm^2/s")


def evolve_radial_surface_density(
    radial_edges: Quantity,
    dust_surface_density: Quantity,
    gas_surface_density: Quantity,
    drift_velocity: Quantity,
    diffusivity: Quantity,
    transport_time: Quantity,
    *,
    cfl: float = 0.4,
    max_steps: int = 1_000_000,
) -> tuple[Quantity, RadialTransportDiagnostics]:
    """Evolve dust surface density with cylindrical advection and diffusion.

    Parameters
    ----------
    radial_edges : Quantity
        Increasing cylindrical radial cell edges with length ``n + 1``.
    dust_surface_density, gas_surface_density : Quantity
        Initial dust and frozen gas columns with length ``n``.
    drift_velocity : Quantity
        Dust velocity relative to gas at cell centres. Gas advection is not
        included.
    diffusivity : Quantity
        Dust diffusivity at cell centres.
    transport_time : Quantity
        Finite transport time. Zero returns the input profile unchanged.
    cfl : float, optional
        Explicit advection-diffusion stability factor.
    max_steps : int, optional
        Safety limit on explicit substeps.

    Returns
    -------
    evolved : Quantity
        Evolved dust surface density.
    diagnostics : RadialTransportDiagnostics
        Mass conservation and boundary-loss diagnostics.

    Notes
    -----
    The inner boundary absorbs inward-moving dust. The outer boundary permits
    outward loss but supplies no incoming dust. Diffusive boundary fluxes are
    zero. Advection uses first-order upwinding and diffusion acts on
    ``Sigma_d / Sigma_g``.
    """
    edges = np.asarray(radial_edges.to("cm").magnitude, dtype=float)
    sigma = np.asarray(dust_surface_density.to("g/cm^2").magnitude, dtype=float).copy()
    sigma_g = np.asarray(gas_surface_density.to("g/cm^2").magnitude, dtype=float)
    velocity = np.asarray(drift_velocity.to("cm/s").magnitude, dtype=float)
    diffusion = np.asarray(diffusivity.to("cm^2/s").magnitude, dtype=float)
    total_time = float(transport_time.to("s").magnitude)

    if edges.ndim != 1 or edges.size != sigma.size + 1:
        raise ValueError("radial_edges must be one-dimensional with length n + 1")
    if not (sigma.shape == sigma_g.shape == velocity.shape == diffusion.shape):
        raise ValueError("all cell-centred transport profiles must have matching shapes")
    if np.any(~np.isfinite(edges)) or np.any(edges <= 0.0) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("radial_edges must be finite, positive, and increasing")
    if np.any(~np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError("dust_surface_density must be finite and non-negative")
    if np.any(~np.isfinite(sigma_g)) or np.any(sigma_g <= 0.0):
        raise ValueError("gas_surface_density must be finite and positive")
    if np.any(~np.isfinite(velocity)):
        raise ValueError("drift_velocity must be finite")
    if np.any(~np.isfinite(diffusion)) or np.any(diffusion < 0.0):
        raise ValueError("diffusivity must be finite and non-negative")
    if not np.isfinite(total_time) or total_time < 0.0:
        raise ValueError("transport_time must be finite and non-negative")
    if not np.isfinite(cfl) or not 0.0 < cfl <= 0.5:
        raise ValueError("cfl must satisfy 0 < cfl <= 0.5")
    if int(max_steps) != max_steps or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    annulus_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    mass = sigma * annulus_area
    initial_mass = float(np.sum(mass))
    if total_time == 0.0:
        diagnostics = RadialTransportDiagnostics(
            initial_mass=Quantity(initial_mass, "g"),
            final_mass=Quantity(initial_mass, "g"),
            inner_mass_lost=Quantity(0.0, "g"),
            outer_mass_lost=Quantity(0.0, "g"),
            n_steps=0,
        )
        return Quantity(sigma, "g/cm^2"), diagnostics

    velocity_face = np.empty(sigma.size + 1, dtype=float)
    velocity_face[0] = velocity[0]
    velocity_face[-1] = velocity[-1]
    velocity_face[1:-1] = 0.5 * (velocity[:-1] + velocity[1:])
    diffusion_face = 0.5 * (diffusion[:-1] + diffusion[1:])
    gas_face = 0.5 * (sigma_g[:-1] + sigma_g[1:])
    centre_spacing = np.diff(centres)

    outgoing_rate = np.zeros_like(sigma)
    outward_right = velocity_face[1:] > 0.0
    outgoing_rate[outward_right] += (
        2.0
        * np.pi
        * edges[1:][outward_right]
        * velocity_face[1:][outward_right]
        / annulus_area[outward_right]
    )
    inward_left = velocity_face[:-1] < 0.0
    outgoing_rate[inward_left] += (
        -2.0
        * np.pi
        * edges[:-1][inward_left]
        * velocity_face[:-1][inward_left]
        / annulus_area[inward_left]
    )
    diffusion_coefficient = (
        2.0
        * np.pi
        * edges[1:-1]
        * diffusion_face
        * gas_face
        / centre_spacing
    )
    outgoing_rate[:-1] += diffusion_coefficient / (
        sigma_g[:-1] * annulus_area[:-1]
    )
    outgoing_rate[1:] += diffusion_coefficient / (
        sigma_g[1:] * annulus_area[1:]
    )
    active_rate = outgoing_rate > 0.0
    stable_dt = (
        cfl / float(np.max(outgoing_rate[active_rate]))
        if np.any(active_rate)
        else total_time
    )

    elapsed = 0.0
    n_steps = 0
    inner_lost = 0.0
    outer_lost = 0.0
    while elapsed < total_time:
        if n_steps >= max_steps:
            raise RuntimeError(
                f"radial transport exceeded max_steps={max_steps} at "
                f"{elapsed / total_time:.3%} of the requested duration"
            )
        dt = min(stable_dt, total_time - elapsed)
        sigma = mass / annulus_area
        flux = np.zeros(sigma.size + 1, dtype=float)

        inward = velocity_face[1:-1] < 0.0
        upwind_sigma = np.where(inward, sigma[1:], sigma[:-1])
        flux[1:-1] = (
            2.0 * np.pi * edges[1:-1] * velocity_face[1:-1] * upwind_sigma
        )
        concentration = sigma / sigma_g
        flux[1:-1] += -(
            2.0
            * np.pi
            * edges[1:-1]
            * diffusion_face
            * gas_face
            * np.diff(concentration)
            / centre_spacing
        )

        if velocity_face[0] < 0.0:
            flux[0] = 2.0 * np.pi * edges[0] * velocity_face[0] * sigma[0]
        if velocity_face[-1] > 0.0:
            flux[-1] = 2.0 * np.pi * edges[-1] * velocity_face[-1] * sigma[-1]

        mass -= dt * np.diff(flux)
        tolerance = 1.0e-12 * max(initial_mass, 1.0)
        if np.min(mass) < -tolerance:
            raise RuntimeError("radial transport produced negative dust mass; reduce cfl")
        mass = np.maximum(mass, 0.0)
        inner_lost += max(0.0, -dt * flux[0])
        outer_lost += max(0.0, dt * flux[-1])
        elapsed += dt
        n_steps += 1

    final_mass = float(np.sum(mass))
    diagnostics = RadialTransportDiagnostics(
        initial_mass=Quantity(initial_mass, "g"),
        final_mass=Quantity(final_mass, "g"),
        inner_mass_lost=Quantity(inner_lost, "g"),
        outer_mass_lost=Quantity(outer_lost, "g"),
        n_steps=n_steps,
    )
    return Quantity(mass / annulus_area, "g/cm^2"), diagnostics


__all__ = [
    "RadialGasBackground",
    "RadialTransportDiagnostics",
    "build_smoothed_radial_gas_background",
    "dust_diffusivity",
    "evolve_radial_surface_density",
    "pressure_drift_velocity",
    "smoothed_log_pressure_gradient",
]
