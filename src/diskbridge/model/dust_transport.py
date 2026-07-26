"""Axisymmetric radial prescriptions for dust surface densities."""

# db-keywords: dust, model, arrays, coordinates, units
# db-role: canonical
# db-scope: package
# db-purpose: Conservative radial drift/diffusion and parameterized radial grain-size distributions.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erfc

from diskbridge._units import Quantity, units

if TYPE_CHECKING:
    from diskbridge.model.core import Model


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


@dataclass(frozen=True)
class RadialDustBinTransport:
    """Result of applying radial transport to one model dust bin.

    Attributes
    ----------
    bin_index : int
        Global dust-bin index in the model.
    grain_size : Quantity
        Representative grain radius.
    initial_surface_density, final_surface_density : Quantity
        Axisymmetric dust columns before and after transport [mass / area],
        shape ``(nr,)``.
    initial_mean_radius, final_mean_radius : Quantity
        Dust-mass-weighted mean radii before and after transport.
    stokes : ndarray
        Midplane Stokes number, shape ``(nr,)``.
    drift_velocity : Quantity
        Radial pressure-drift velocity, shape ``(nr,)``.
    diffusivity : Quantity
        Radial turbulent diffusivity, shape ``(nr,)``.
    diagnostics : RadialTransportDiagnostics
        Boundary losses, mass accounting, and integration step count.
    """

    bin_index: int
    grain_size: Quantity
    initial_surface_density: Quantity
    final_surface_density: Quantity
    initial_mean_radius: Quantity
    final_mean_radius: Quantity
    stokes: np.ndarray
    drift_velocity: Quantity
    diffusivity: Quantity
    diagnostics: RadialTransportDiagnostics


@dataclass(frozen=True)
class RadialDustTransportResult:
    """Result of applying radial transport to a model dust component.

    Attributes
    ----------
    transport_time : Quantity
        Requested finite transport exposure.
    gas_background : RadialGasBackground
        Smoothed frozen gas background used by every grain bin.
    bins : tuple of RadialDustBinTransport
        Per-bin transport results in global bin order.
    disc_axis : ndarray
        Unit angular-momentum axis used to define cylindrical disc coordinates,
        shape ``(3,)`` in native Cartesian coordinates.
    active_annuli : ndarray
        Boolean radial support of the connected disc reservoir, shape ``(nr,)``.
    """

    transport_time: Quantity
    gas_background: RadialGasBackground
    bins: tuple[RadialDustBinTransport, ...]
    disc_axis: np.ndarray
    active_annuli: np.ndarray


@dataclass(frozen=True)
class RadialAmaxDustBin:
    """Radial result for one bin in a smooth maximum-grain-size profile.

    Attributes
    ----------
    bin_index : int
        Global dust-bin index in the model.
    grain_size : Quantity
        Representative grain radius.
    initial_surface_density, final_surface_density : Quantity
        Settled-disc columns before and after redistribution [mass / area], shape
        ``(nr,)``.
    retention : ndarray
        Smooth retention weight relative to the local maximum grain size,
        shape ``(nr,)``.
    mass_fraction : ndarray
        Locally normalized fraction of the settled dust column assigned to
        this bin, shape ``(nr,)``.
    """

    bin_index: int
    grain_size: Quantity
    initial_surface_density: Quantity
    final_surface_density: Quantity
    retention: np.ndarray
    mass_fraction: np.ndarray


@dataclass(frozen=True)
class RadialAmaxResult:
    """Result of applying a smooth radial maximum-grain-size profile.

    Attributes
    ----------
    reference_amax : Quantity
        Maximum-grain-size profile value at ``reference_radius``.
    reference_radius : Quantity
        Cylindrical radius at which ``a_max = reference_amax``.
    radial_exponent : float
        Positive exponent in ``a_max proportional to R**(-radial_exponent)``.
    transition_width_dex : float
        Standard-deviation width of the smooth size cutoff in dex.
    dust_to_gas_ratio : float
        Total settled dust-to-gas surface-density ratio imposed on active
        annuli.
    disc_gas_surface_density : Quantity
        Actual mask-weighted gas column that sets the total dust budget
        [mass / area], shape ``(nr,)``.
    radius : Quantity
        Cylindrical annulus centres [length], shape ``(nr,)``.
    maximum_grain_size : Quantity
        Bounded radial maximum-grain-size profile, shape ``(nr,)``.
    bins : tuple of RadialAmaxDustBin
        Per-bin radial distributions in global bin order.
    disc_axis : ndarray
        Unit angular-momentum axis defining cylindrical disc coordinates,
        shape ``(3,)``.
    active_annuli : ndarray
        Connected radial support on which tapering is applied, shape
        ``(nr,)``.
    """

    reference_amax: Quantity
    reference_radius: Quantity
    radial_exponent: float
    transition_width_dex: float
    dust_to_gas_ratio: float
    disc_gas_surface_density: Quantity
    radius: Quantity
    maximum_grain_size: Quantity
    bins: tuple[RadialAmaxDustBin, ...]
    disc_axis: np.ndarray
    active_annuli: np.ndarray


@dataclass(frozen=True)
class _DiscCylindricalGeometry:
    """Native-cell mapping into measured disc-frame cylindrical annuli."""

    axis: np.ndarray
    radial_bin: np.ndarray
    phi_bin: np.ndarray
    nphi: int
    valid: np.ndarray
    volume_cm3: np.ndarray
    annulus_area_cm2: np.ndarray


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
    outward advective loss and uses zero exterior settled-dust concentration
    for diffusion, so diffusing dust can leave but cannot enter the domain.
    Advection uses first-order upwinding and diffusion acts on
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
    outer_spacing = edges[-1] - centres[-1]
    outer_diffusion_coefficient = (
        2.0
        * np.pi
        * edges[-1]
        * diffusion[-1]
        * sigma_g[-1]
        / outer_spacing
    )

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
    outgoing_rate[-1] += outer_diffusion_coefficient / (
        sigma_g[-1] * annulus_area[-1]
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
        flux[-1] = outer_diffusion_coefficient * concentration[-1]
        if velocity_face[-1] > 0.0:
            flux[-1] += 2.0 * np.pi * edges[-1] * velocity_face[-1] * sigma[-1]

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


def _disc_axis_from_mask(mask) -> np.ndarray:
    """Return the normalized Cartesian disc axis recorded by the mask."""
    raw_axis = mask.attrs.get("disk_axis_cartesian")
    if raw_axis is None:
        raise ValueError(
            "radial dust transport requires a mask with disk_axis_cartesian metadata"
        )
    axis = np.asarray(raw_axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or np.any(~np.isfinite(axis)) or norm <= 0.0:
        raise ValueError("disk_axis_cartesian must contain three finite nonzero values")
    return axis / norm


def _disc_cylindrical_geometry(
    model: "Model",
    axis: np.ndarray,
) -> _DiscCylindricalGeometry:
    """Map spherical cell centres into cylindrical coordinates about ``axis``."""
    from diskbridge.model.coords import cartesian_from_spherical, spherical_grids
    from diskbridge.model.profiles import compute_cell_volumes

    phi_edges = model.mesh.edges("phi").to("radian").magnitude
    if not np.isclose(phi_edges[-1] - phi_edges[0], 2.0 * np.pi):
        raise ValueError("disc-frame axisymmetric transport requires full 2-pi azimuth coverage")

    r_grid, theta_grid, phi_grid = spherical_grids(
        model.mesh.centers("r").to("cm"),
        model.mesh.centers("theta").to("radian"),
        model.mesh.centers("phi").to("radian"),
    )
    x, y, z = cartesian_from_spherical(r_grid, phi_grid, theta_grid)
    x_cm = np.asarray(x.to("cm").magnitude, dtype=float)
    y_cm = np.asarray(y.to("cm").magnitude, dtype=float)
    z_cm = np.asarray(z.to("cm").magnitude, dtype=float)
    height = x_cm * axis[0] + y_cm * axis[1] + z_cm * axis[2]
    radius_squared = np.maximum(
        x_cm * x_cm + y_cm * y_cm + z_cm * z_cm - height * height,
        0.0,
    )
    radius = np.sqrt(radius_squared)

    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(axis[2])) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    e1 = np.cross(axis, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    x_disc = x_cm * e1[0] + y_cm * e1[1] + z_cm * e1[2]
    y_disc = x_cm * e2[0] + y_cm * e2[1] + z_cm * e2[2]
    phi_disc = np.arctan2(y_disc, x_disc)

    radial_edges = model.mesh.edges("r").to("cm").magnitude
    radial_bin = np.digitize(radius, radial_edges).astype(np.int32) - 1
    nr = radial_edges.size - 1
    nphi = model.mesh.ncell("phi")
    phi_fraction = np.mod(phi_disc + np.pi, 2.0 * np.pi) / (2.0 * np.pi)
    phi_bin = np.minimum((phi_fraction * nphi).astype(np.int32), nphi - 1)
    valid = (radial_bin >= 0) & (radial_bin < nr)
    return _DiscCylindricalGeometry(
        axis=axis,
        radial_bin=radial_bin,
        phi_bin=phi_bin,
        nphi=nphi,
        valid=valid,
        volume_cm3=np.asarray(compute_cell_volumes(model), dtype=float),
        annulus_area_cm2=np.pi * np.diff(radial_edges**2),
    )


def _cylindrical_surface_density(
    model: "Model",
    density: Quantity,
    geometry: _DiscCylindricalGeometry,
) -> Quantity:
    """Conservatively project native cell masses into disc-frame annuli."""
    values = np.asarray(density.to("g/cm^3").magnitude, dtype=float)
    if values.shape != model.mesh.shape:
        raise ValueError(
            f"density shape {values.shape} does not match mesh shape {model.mesh.shape}"
        )
    valid = geometry.valid & np.isfinite(values) & (values >= 0.0)
    mass = np.bincount(
        geometry.radial_bin[valid],
        weights=values[valid] * geometry.volume_cm3[valid],
        minlength=geometry.annulus_area_cm2.size,
    )
    return Quantity(mass / geometry.annulus_area_cm2, "g/cm^2")


def _connected_annular_support(
    weight: np.ndarray,
    geometry: _DiscCylindricalGeometry,
    *,
    seed_min: float,
) -> np.ndarray:
    """Return the contiguous annuli supported around most disc azimuths."""
    nr = geometry.annulus_area_cm2.size
    nphi = geometry.nphi
    support = np.zeros(nr * nphi, dtype=float)
    valid = geometry.valid & np.isfinite(weight)
    combined = geometry.radial_bin[valid] * nphi + geometry.phi_bin[valid]
    np.maximum.at(support, combined, weight[valid])
    score = np.mean(support.reshape(nr, nphi), axis=1)
    candidates = np.flatnonzero(score >= seed_min)
    if candidates.size < 2:
        raise ValueError("fewer than two cylindrical annuli have connected disc support")
    start = int(candidates[0])
    stop = start
    while stop + 1 < nr and score[stop + 1] >= seed_min:
        stop += 1
    if stop == start:
        raise ValueError("connected disc support spans fewer than two adjacent annuli")
    active = np.zeros(nr, dtype=bool)
    active[start : stop + 1] = True
    return active


def _disc_midplane_profiles(
    model: "Model",
    weight: np.ndarray,
    geometry: _DiscCylindricalGeometry,
) -> tuple[Quantity, Quantity]:
    """Extract disc-frame azimuth-median midplane temperature and pressure."""
    from diskbridge.model.utils import field_data_as_order

    required = ("density", "temperature", "pressure")
    missing = [name for name in required if name not in model.gas]
    if missing:
        raise KeyError(f"radial dust transport requires gas fields {missing}")

    target = model.mesh.axis_names()
    density = np.asarray(
        field_data_as_order(model.gas["density"], target)
        .to("g/cm^3")
        .magnitude,
        dtype=float,
    )
    nr = geometry.annulus_area_cm2.size
    nphi = model.mesh.ncell("phi")
    score = density * weight
    valid = geometry.valid & np.isfinite(score) & (score > 0.0)
    combined = geometry.radial_bin[valid] * nphi + geometry.phi_bin[valid]
    best = np.full(nr * nphi, -np.inf, dtype=float)
    np.maximum.at(best, combined, score[valid])
    all_combined = geometry.radial_bin * nphi + geometry.phi_bin
    best_cells = valid & np.isclose(
        score,
        best[np.clip(all_combined, 0, best.size - 1)],
        rtol=1.0e-12,
        atol=0.0,
    )
    log_radius = np.log(model.mesh.centers("r").to("cm").magnitude)
    profiles: list[Quantity] = []
    for name, unit in (("temperature", "K"), ("pressure", "dyn/cm^2")):
        values = np.asarray(
            field_data_as_order(model.gas[name], target).to(unit).magnitude,
            dtype=float,
        )
        usable = best_cells & np.isfinite(values) & (values > 0.0)
        selected_combined = all_combined[usable]
        log_sum = np.bincount(
            selected_combined,
            weights=np.log(values[usable]),
            minlength=nr * nphi,
        )
        count = np.bincount(selected_combined, minlength=nr * nphi)
        selected = np.full(nr * nphi, np.nan, dtype=float)
        populated = count > 0
        selected[populated] = np.exp(log_sum[populated] / count[populated])
        selected = selected.reshape(nr, nphi)
        profile = np.full(nr, np.nan, dtype=float)
        populated_rows = np.any(np.isfinite(selected), axis=1)
        profile[populated_rows] = np.nanmedian(selected[populated_rows], axis=1)
        supported = np.isfinite(profile) & (profile > 0.0)
        if np.count_nonzero(supported) < 2:
            raise ValueError(f"cannot construct a supported midplane {name} profile")
        profile = np.exp(
            np.interp(log_radius, log_radius[supported], np.log(profile[supported]))
        )
        profiles.append(Quantity(profile, unit))
    return profiles[0], profiles[1]


def _reconstruct_cylindrical_density(
    initial_density: Quantity,
    initial_surface_density: Quantity,
    final_surface_density: Quantity,
    weight: np.ndarray,
    geometry: _DiscCylindricalGeometry,
    active_annuli: np.ndarray,
    *,
    seed_min: float,
    weight_floor: float,
) -> Quantity:
    """Realize target annular masses without amplifying soft-weight tails."""
    initial = np.asarray(initial_density.to("g/cm^3").magnitude, dtype=float)
    initial_sigma = np.asarray(
        initial_surface_density.to("g/cm^2").magnitude,
        dtype=float,
    )
    final_sigma = np.asarray(
        final_surface_density.to("g/cm^2").magnitude,
        dtype=float,
    )
    target_mass = final_sigma * geometry.annulus_area_cm2
    initial_mass = initial_sigma * geometry.annulus_area_cm2
    transported = initial.copy()
    if not 0.0 <= weight_floor < seed_min <= 1.0:
        raise ValueError("mask metadata must satisfy 0 <= weight_floor < weight_m0 <= 1")
    coherent = np.clip((weight - weight_floor) / (seed_min - weight_floor), 0.0, 1.0)

    for radial_index in np.flatnonzero(active_annuli):
        cells = geometry.valid & (geometry.radial_bin == radial_index)
        mass0 = float(initial_mass[radial_index])
        mass1 = float(target_mass[radial_index])
        if mass0 <= 0.0:
            raise ValueError(f"active cylindrical annulus {radial_index} has no dust mass")
        if mass1 <= mass0:
            transported[cells] *= mass1 / mass0
            continue

        template_mass = initial[cells] * coherent[cells] * geometry.volume_cm3[cells]
        template_total = float(np.sum(template_mass))
        if template_total <= 0.0:
            raise ValueError(
                f"active cylindrical annulus {radial_index} has no coherent reconstruction template"
            )
        transported[cells] += (
            (mass1 - mass0)
            * template_mass
            / template_total
            / geometry.volume_cm3[cells]
        )
    return Quantity(transported, "g/cm^3")


def _smooth_radial_amax_distribution(
    radius: np.ndarray,
    grain_sizes: np.ndarray,
    base_mass_fractions: np.ndarray,
    minimum_grain_size: float,
    maximum_grain_size: float,
    reference_amax: float,
    reference_radius: float,
    radial_exponent: float,
    transition_width_dex: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radial ``a_max``, smooth retention, and normalized fractions."""
    radial_centres = np.asarray(radius, dtype=float)
    sizes = np.asarray(grain_sizes, dtype=float)
    fractions = np.asarray(base_mass_fractions, dtype=float)
    if radial_centres.ndim != 1 or radial_centres.size < 1:
        raise ValueError("radius must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(radial_centres)) or np.any(radial_centres < 0.0):
        raise ValueError("radius must be finite and non-negative")
    if sizes.ndim != 1 or sizes.size < 1:
        raise ValueError("grain_sizes must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(sizes)) or np.any(sizes <= 0.0):
        raise ValueError("grain_sizes must be finite and positive")
    if fractions.shape != sizes.shape:
        raise ValueError("base_mass_fractions must have shape (nbin,)")
    if np.any(~np.isfinite(fractions)) or np.any(fractions <= 0.0):
        raise ValueError("base_mass_fractions must be finite and positive")
    for name, value in (
        ("minimum_grain_size", minimum_grain_size),
        ("maximum_grain_size", maximum_grain_size),
        ("reference_amax", reference_amax),
        ("reference_radius", reference_radius),
        ("radial_exponent", radial_exponent),
        ("transition_width_dex", transition_width_dex),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if minimum_grain_size >= maximum_grain_size:
        raise ValueError("minimum_grain_size must be smaller than maximum_grain_size")
    if not minimum_grain_size <= reference_amax <= maximum_grain_size:
        raise ValueError(
            "reference_amax must lie within the configured grain-size interval"
        )

    fractions = fractions / np.sum(fractions)
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        radial_amax = float(reference_amax) * np.power(
            radial_centres / float(reference_radius),
            -float(radial_exponent),
        )
    radial_amax[radial_centres == 0.0] = float(maximum_grain_size)
    radial_amax = np.clip(
        radial_amax,
        float(minimum_grain_size),
        float(maximum_grain_size),
    )
    log_size_offset = (
        np.log10(sizes[:, None]) - np.log10(radial_amax[None, :])
    ) / (np.sqrt(2.0) * float(transition_width_dex))
    retention = np.clip(0.5 * erfc(log_size_offset), 0.0, 1.0)
    weighted = fractions[:, None] * retention
    normalization = np.sum(weighted, axis=0)
    if np.any(~np.isfinite(normalization)) or np.any(normalization <= 0.0):
        raise ValueError("radial a_max has no finite local normalization")
    return radial_amax, retention, weighted / normalization[None, :]


def apply_radial_amax(
    model: "Model",
    *,
    reference_amax: Quantity = Quantity(100.0, "um"),
    reference_radius: Quantity = Quantity(250.0, "au"),
    radial_exponent: float = 2.0,
    transition_width_dex: float = 0.25,
    component_index: int = 0,
) -> RadialAmaxResult:
    """Apply a smooth radial maximum-grain-size profile.

    The local profile is ``a_max = reference_amax *
    (R / reference_radius)**(-radial_exponent)``, bounded by the configured
    component size interval. Each representative bin is retained through a
    complementary-error-function turnover in log grain size and the resulting
    fractions are normalized at every radius. The actual mask-weighted disc
    gas column sets the total settled dust column. Existing three-dimensional
    settling templates are rescaled in cylindrical annuli about the measured
    disc axis; other components and annuli outside connected disc support are
    unchanged.

    Parameters
    ----------
    model : Model
        Spherical model containing gas fields and an already configured dust
        component whose mask records ``disk_axis_cartesian`` metadata.
    reference_amax : Quantity, optional
        Maximum-grain-size profile value at ``reference_radius``.
    reference_radius : Quantity, optional
        Positive cylindrical radius at which ``a_max = reference_amax``.
    radial_exponent : float, optional
        Positive exponent controlling how rapidly ``a_max`` decreases outward.
    transition_width_dex : float, optional
        Positive standard-deviation width of the smooth bin turnover in dex.
    component_index : int, optional
        Settled dust component to redistribute.

    Returns
    -------
    RadialAmaxResult
        Per-bin radial distributions and the actual gas-column budget.

    Raises
    ------
    ValueError
        If the mesh, mask metadata, component, profiles, or prescription
        parameters are invalid.
    KeyError
        If required gas or dust fields are unavailable.

    Notes
    -----
    This is a parameterized radial size distribution, not time evolution or a
    dust-growth calculation. See the dust guide for the governing equations.
    """
    from diskbridge.model.field import Field
    from diskbridge.model.utils import field_data_as_order

    if model.mesh is None or model.mesh.coord_system != "spherical":
        raise ValueError("radial a_max requires a spherical model mesh")
    if model.gas is None or model.dust is None:
        raise ValueError("radial a_max requires gas and dust submodels")
    if int(component_index) != component_index:
        raise ValueError("component_index must be an integer")
    component_index = int(component_index)
    if component_index < 0 or component_index >= len(model.dust._components):
        raise ValueError(f"dust component {component_index} does not exist")
    reference_amax_um = float(reference_amax.to("um").magnitude)
    reference_radius_au = float(reference_radius.to("au").magnitude)
    for name, value in (
        ("reference_amax", reference_amax_um),
        ("reference_radius", reference_radius_au),
        ("radial_exponent", radial_exponent),
        ("transition_width_dex", transition_width_dex),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    radial_exponent = float(radial_exponent)
    transition_width_dex = float(transition_width_dex)

    component = model.dust._components[component_index]
    if component.mask is None:
        raise ValueError("radial a_max requires a component mask or weight")
    minimum_size_um = float(component.distribution.amin.to("um").magnitude)
    maximum_size_um = float(component.distribution.amax.to("um").magnitude)
    if not minimum_size_um <= reference_amax_um <= maximum_size_um:
        raise ValueError(
            "reference_amax must lie within the component grain-size interval"
        )
    disc_axis = _disc_axis_from_mask(component.mask)
    geometry = _disc_cylindrical_geometry(model, disc_axis)
    target_order = model.mesh.axis_names()
    weight = np.clip(
        np.asarray(
            field_data_as_order(component.mask, target_order)
            .to("dimensionless")
            .magnitude,
            dtype=float,
        ),
        0.0,
        1.0,
    )
    seed_min = float(component.mask.attrs.get("weight_m0", np.nan))
    weight_floor = float(component.mask.attrs.get("weight_floor", np.nan))
    if not np.isfinite(seed_min) or not np.isfinite(weight_floor):
        raise ValueError(
            "radial a_max requires weight_m0 and weight_floor mask metadata"
        )
    support_min = max(weight_floor, np.nextafter(0.0, 1.0))
    active_annuli = _connected_annular_support(
        weight,
        geometry,
        seed_min=support_min,
    )

    gas_density = field_data_as_order(model.gas["density"], target_order).to(
        "g/cm^3"
    )
    disc_gas_surface_density = _cylindrical_surface_density(
        model,
        gas_density * weight,
        geometry,
    ).to("g/cm^2")
    gas_sigma_values = np.asarray(
        disc_gas_surface_density.magnitude,
        dtype=float,
    )
    if not np.any(gas_sigma_values > 0.0):
        raise ValueError("component-weighted gas surface density is empty")
    if np.any(~np.isfinite(gas_sigma_values)) or np.any(gas_sigma_values < 0.0):
        raise ValueError("component-weighted gas surface density is invalid")
    global_bin_indices = sorted(
        int(name.split("_")[-1])
        for name, (mapped_component, _) in model.dust._global_bins.items()
        if mapped_component == component_index
    )
    if not global_bin_indices:
        raise ValueError(f"dust component {component_index} contains no grain bins")

    sizes: list[Quantity] = []
    initial_densities: list[Quantity] = []
    initial_columns: list[Quantity] = []
    base_fractions: list[float] = []
    for bin_index in global_bin_indices:
        dust_bin = model.dust[f"bin_{bin_index}"]
        _, local_bin_index = model.dust._global_bins[f"bin_{bin_index}"]
        sizes.append(dust_bin.size)
        initial_density = field_data_as_order(
            dust_bin["density"], target_order
        ).to("g/cm^3")
        initial_densities.append(initial_density)
        initial_columns.append(
            _cylindrical_surface_density(
                model,
                initial_density,
                geometry,
            ).to("g/cm^2")
        )
        base_fractions.append(
            float(component.distribution.mass_fractions[local_bin_index])
        )

    radius = model.mesh.centers("r").to("au")
    maximum_grain_size_um, retention, local_fractions = (
        _smooth_radial_amax_distribution(
            np.asarray(radius.magnitude, dtype=float),
            np.asarray(
                [size.to("um").magnitude for size in sizes],
                dtype=float,
            ),
            np.asarray(base_fractions, dtype=float),
            minimum_size_um,
            maximum_size_um,
            reference_amax_um,
            reference_radius_au,
            radial_exponent,
            transition_width_dex,
        )
    )
    target_total = (
        float(component.dust_to_gas_ratio) * disc_gas_surface_density
    ).to("g/cm^2")

    bin_results: list[RadialAmaxDustBin] = []
    for position, bin_index in enumerate(global_bin_indices):
        initial_column = initial_columns[position]
        final_values = np.asarray(
            initial_column.to("g/cm^2").magnitude,
            dtype=float,
        ).copy()
        desired = (
            target_total * local_fractions[position]
        ).to("g/cm^2").magnitude
        final_values[active_annuli] = desired[active_annuli]
        final_column = Quantity(final_values, "g/cm^2")
        redistributed_density = _reconstruct_cylindrical_density(
            initial_densities[position],
            initial_column,
            final_column,
            weight,
            geometry,
            active_annuli,
            seed_min=seed_min,
            weight_floor=weight_floor,
        )
        model.dust.register(
            f"density_bin_{bin_index}",
            Field(
                quantity="density",
                data=redistributed_density,
                axis_order=target_order,
                attrs={
                    "radial_amax": True,
                    "reference_amax_um": reference_amax_um,
                    "reference_radius_au": reference_radius_au,
                    "radial_exponent": radial_exponent,
                    "transition_width_dex": transition_width_dex,
                    "component_index": component_index,
                    "bin_index": bin_index,
                    "amax_coordinate": "disc_cylindrical_radius",
                    "total_dust_column_source": "actual_mask_weighted_disc_gas",
                    "disk_axis_cartesian": tuple(
                        float(value) for value in disc_axis
                    ),
                },
            ),
        )
        bin_results.append(
            RadialAmaxDustBin(
                bin_index=bin_index,
                grain_size=sizes[position],
                initial_surface_density=initial_column,
                final_surface_density=final_column,
                retention=np.asarray(retention[position], dtype=float),
                mass_fraction=np.asarray(local_fractions[position], dtype=float),
            )
        )

    return RadialAmaxResult(
        reference_amax=reference_amax.to("um"),
        reference_radius=reference_radius.to("au"),
        radial_exponent=radial_exponent,
        transition_width_dex=transition_width_dex,
        dust_to_gas_ratio=float(component.dust_to_gas_ratio),
        disc_gas_surface_density=disc_gas_surface_density,
        radius=radius,
        maximum_grain_size=Quantity(maximum_grain_size_um, "um"),
        bins=tuple(bin_results),
        disc_axis=np.asarray(disc_axis, dtype=float),
        active_annuli=active_annuli,
    )


def apply_radial_dust_transport(
    model: "Model",
    transport_time: Quantity,
    *,
    component_index: int = 0,
    stellar_mass: Quantity | None = None,
    mean_molecular_weight: float | None = None,
    diffusion_delta: float | None = None,
    smoothing_bins: int = 21,
    pressure_log_gradient_bounds: tuple[float, float] = (-5.0, -0.25),
    surface_density_log_gradient_bounds: tuple[float, float] = (-5.0, 0.0),
    cfl: float = 0.4,
    max_steps: int = 1_000_000,
) -> RadialDustTransportResult:
    """Apply finite-time radial transport to one model dust component.

    The function constructs a frozen, smoothed gas background from the
    component-weighted snapshot in the measured disc frame, evolves every grain
    bin independently in cylindrical radius, and registers transported
    three-dimensional bin densities on ``model.dust``. Dust removed from an
    annulus is removed proportionally. Dust added to an annulus follows the
    coherent settled-disc template rather than amplifying low-weight material.
    Other dust components are not modified.

    Parameters
    ----------
    model : Model
        Three-dimensional spherical model containing gas density, temperature,
        pressure, and an already configured dust component whose mask records
        ``disk_axis_cartesian`` metadata.
    transport_time : Quantity
        Finite radial transport exposure.
    component_index : int, optional
        Dust component to evolve. The Bondi workflow uses component zero for
        settled disc dust.
    stellar_mass : Quantity, optional
        Central mass. Defaults to ``model.variables['mstar']`` or one solar
        mass when the model does not define it.
    mean_molecular_weight : float, optional
        Gas-particle mass in hydrogen-mass units. Defaults to the component
        value.
    diffusion_delta : float, optional
        Turbulent radial diffusion strength. Defaults to the component's
        settling diffusion parameter.
    smoothing_bins : int, optional
        Odd log-profile smoothing width on the native radial grid.
    pressure_log_gradient_bounds, surface_density_log_gradient_bounds : tuple, optional
        Bounds applied to the smoothed gas profiles.
    cfl : float, optional
        Explicit solver stability factor.
    max_steps : int, optional
        Maximum substeps allowed for each grain bin.

    Returns
    -------
    RadialDustTransportResult
        Frozen gas background and per-bin transport diagnostics.

    Raises
    ------
    ValueError
        If the mesh, component, fields, transport parameters, or profiles are
        invalid.
    KeyError
        If required gas or dust fields are unavailable.

    Notes
    -----
    This is a frozen, axisymmetric exposure model. It does not evolve gas. The
    one-dimensional solver acts only on the contiguous annuli supported by the
    connected disc mask; this emergent support replaces any fixed outer radius.
    """
    from diskbridge.model.dust import stokes_number
    from diskbridge.model.field import Field
    from diskbridge.model.utils import field_data_as_order

    if model.mesh is None or model.mesh.coord_system != "spherical":
        raise ValueError("radial dust transport requires a spherical model mesh")
    if model.gas is None or model.dust is None:
        raise ValueError("radial dust transport requires gas and dust submodels")
    if int(component_index) != component_index:
        raise ValueError("component_index must be an integer")
    component_index = int(component_index)
    if component_index < 0 or component_index >= len(model.dust._components):
        raise ValueError(f"dust component {component_index} does not exist")

    component = model.dust._components[component_index]
    if component.mask is None:
        raise ValueError("radial dust transport requires a component mask or weight")
    disc_axis = _disc_axis_from_mask(component.mask)
    geometry = _disc_cylindrical_geometry(model, disc_axis)
    target_order = model.mesh.axis_names()
    weight = np.clip(
        np.asarray(
            field_data_as_order(component.mask, target_order)
            .to("dimensionless")
            .magnitude,
            dtype=float,
        ),
        0.0,
        1.0,
    )
    seed_min = float(component.mask.attrs.get("weight_m0", np.nan))
    weight_floor = float(component.mask.attrs.get("weight_floor", np.nan))
    if not np.isfinite(seed_min) or not np.isfinite(weight_floor):
        raise ValueError(
            "radial dust transport requires weight_m0 and weight_floor mask metadata"
        )
    active_annuli = _connected_annular_support(
        weight,
        geometry,
        seed_min=seed_min,
    )
    active_indices = np.flatnonzero(active_annuli)
    active_slice = slice(int(active_indices[0]), int(active_indices[-1]) + 1)
    gas_density = field_data_as_order(model.gas["density"], target_order).to(
        "g/cm^3"
    )
    gas_surface_density = _cylindrical_surface_density(
        model,
        gas_density * weight,
        geometry,
    ).to("g/cm^2")
    gas_sigma_values = np.asarray(gas_surface_density.magnitude, dtype=float)
    positive_gas = gas_sigma_values[np.isfinite(gas_sigma_values) & (gas_sigma_values > 0.0)]
    if positive_gas.size == 0:
        raise ValueError("component-weighted gas surface density is empty")
    gas_surface_density = Quantity(
        np.maximum(gas_sigma_values, float(np.max(positive_gas)) * 1.0e-15),
        "g/cm^2",
    )

    temperature, pressure = _disc_midplane_profiles(model, weight, geometry)
    mu = (
        float(component.mean_molecular_weight)
        if mean_molecular_weight is None
        else float(mean_molecular_weight)
    )
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mean_molecular_weight must be finite and positive")
    delta = component.delta if diffusion_delta is None else diffusion_delta
    if delta is None or not np.isfinite(delta) or float(delta) < 0.0:
        raise ValueError("diffusion_delta must be finite and non-negative")
    delta = float(delta)

    radial_centres = model.mesh.centers("r")
    radial_edges = model.mesh.edges("r")
    background = build_smoothed_radial_gas_background(
        radial_centres,
        gas_surface_density,
        temperature,
        pressure,
        mean_molecular_weight=mu,
        smoothing_bins=smoothing_bins,
        pressure_log_gradient_bounds=pressure_log_gradient_bounds,
        surface_density_log_gradient_bounds=surface_density_log_gradient_bounds,
    )
    sound_speed = np.sqrt(
        units("k_B") * background.temperature / (mu * units("m_H"))
    ).to("cm/s")
    if stellar_mass is None:
        stellar_mass = model.variables.get("mstar", units("solar_mass"))
    if not hasattr(stellar_mass, "to"):
        stellar_mass = float(stellar_mass) * units("solar_mass")
    stellar_mass = stellar_mass.to("g")
    if not np.isfinite(stellar_mass.magnitude) or stellar_mass.magnitude <= 0.0:
        raise ValueError("stellar_mass must be finite and positive")
    omega = np.sqrt(
        units("G") * stellar_mass / radial_centres.to("cm") ** 3
    ).to("1/s")
    keplerian_velocity = (omega * radial_centres.to("cm")).to("cm/s")
    scale_height = (sound_speed / omega).to("cm")

    global_bin_indices = sorted(
        int(name.split("_")[-1])
        for name, (mapped_component, _) in model.dust._global_bins.items()
        if mapped_component == component_index
    )
    if not global_bin_indices:
        raise ValueError(f"dust component {component_index} contains no grain bins")

    bin_results: list[RadialDustBinTransport] = []
    for bin_index in global_bin_indices:
        dust_bin = model.dust[f"bin_{bin_index}"]
        initial_field = dust_bin["density"]
        initial_density = field_data_as_order(initial_field, target_order).to("g/cm^3")
        initial_columns = _cylindrical_surface_density(
            model,
            initial_density,
            geometry,
        ).to("g/cm^2")
        stokes = stokes_number(
            dust_bin.size,
            background.midplane_density,
            background.temperature,
            omega,
            dust_bin.density_material,
            mu,
        ).to("dimensionless").magnitude
        drift = pressure_drift_velocity(
            stokes,
            sound_speed,
            keplerian_velocity,
            background.pressure_log_gradient,
        )
        diffusion = dust_diffusivity(delta, sound_speed, scale_height, stokes)
        evolved_active, diagnostics = evolve_radial_surface_density(
            radial_edges[active_indices[0] : active_indices[-1] + 2],
            initial_columns[active_slice],
            background.surface_density[active_slice],
            drift[active_slice],
            diffusion[active_slice],
            transport_time,
            cfl=cfl,
            max_steps=max_steps,
        )
        final_values = np.asarray(initial_columns.to("g/cm^2").magnitude, dtype=float).copy()
        final_values[active_slice] = evolved_active.to("g/cm^2").magnitude
        final_columns = Quantity(final_values, "g/cm^2")

        initial_values = np.asarray(initial_columns.to("g/cm^2").magnitude, dtype=float)
        radial_edges_cm = radial_edges.to("cm").magnitude
        annulus_area = np.pi * np.diff(radial_edges_cm**2)
        radial_centres_cm = radial_centres.to("cm").magnitude
        initial_annulus_mass = initial_values * annulus_area
        final_annulus_mass = final_values * annulus_area
        initial_mass_sum = float(np.sum(initial_annulus_mass))
        final_mass_sum = float(np.sum(final_annulus_mass))
        if initial_mass_sum <= 0.0:
            raise ValueError(f"dust bin {bin_index} has no transportable mass")
        initial_mean_radius = Quantity(
            np.sum(radial_centres_cm * initial_annulus_mass)
            / initial_mass_sum,
            "cm",
        )
        final_mean_radius = Quantity(
            (
                np.sum(radial_centres_cm * final_annulus_mass) / final_mass_sum
                if final_mass_sum > 0.0
                else np.nan
            ),
            "cm",
        )
        transported_density = _reconstruct_cylindrical_density(
            initial_density,
            initial_columns,
            final_columns,
            weight,
            geometry,
            active_annuli,
            seed_min=seed_min,
            weight_floor=weight_floor,
        )
        model.dust.register(
            f"density_bin_{bin_index}",
            Field(
                quantity="density",
                data=transported_density,
                axis_order=target_order,
                attrs={
                    "radial_transport": True,
                    "transport_time_yr": float(transport_time.to("yr").magnitude),
                    "component_index": component_index,
                    "bin_index": bin_index,
                    "transport_coordinate": "disc_cylindrical_radius",
                    "disk_axis_cartesian": tuple(float(value) for value in disc_axis),
                    "active_radius_min_au": float(
                        radial_edges[active_indices[0]].to("au").magnitude
                    ),
                    "active_radius_max_au": float(
                        radial_edges[active_indices[-1] + 1].to("au").magnitude
                    ),
                    "gas_background_smoothing_bins": int(smoothing_bins),
                    "pressure_log_gradient_bounds": tuple(
                        float(value) for value in pressure_log_gradient_bounds
                    ),
                    "surface_density_log_gradient_bounds": tuple(
                        float(value) for value in surface_density_log_gradient_bounds
                    ),
                },
            ),
        )
        bin_results.append(
            RadialDustBinTransport(
                bin_index=bin_index,
                grain_size=dust_bin.size,
                initial_surface_density=initial_columns,
                final_surface_density=final_columns,
                initial_mean_radius=initial_mean_radius,
                final_mean_radius=final_mean_radius,
                stokes=np.asarray(stokes, dtype=float),
                drift_velocity=drift,
                diffusivity=diffusion,
                diagnostics=diagnostics,
            )
        )

    return RadialDustTransportResult(
        transport_time=transport_time,
        gas_background=background,
        bins=tuple(bin_results),
        disc_axis=np.asarray(disc_axis, dtype=float),
        active_annuli=active_annuli,
    )


__all__ = [
    "RadialDustBinTransport",
    "RadialDustTransportResult",
    "RadialGasBackground",
    "RadialTransportDiagnostics",
    "RadialAmaxDustBin",
    "RadialAmaxResult",
    "apply_radial_amax",
    "apply_radial_dust_transport",
    "build_smoothed_radial_gas_background",
    "dust_diffusivity",
    "evolve_radial_surface_density",
    "pressure_drift_velocity",
    "smoothed_log_pressure_gradient",
]
