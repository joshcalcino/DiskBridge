"""Dust grain submodel for DiskBridge.

This module provides dust grain distribution modeling with power-law size distributions,
memory-efficient storage for gas-proportional dust, and integration with radmc3d.
"""

# db-keywords: dust, gas-temperature, config, units, radmc3d, model, field, coordinates, mesh
# db-role: helper
# db-scope: package
# db-purpose: Dust grain submodel for DiskBridge.

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Callable, List, Literal, Union
import numpy as np
from scipy.special import erfcx

from diskbridge._units import Quantity, units
from diskbridge._logging import logger
from diskbridge.model.field import Field
from diskbridge.model.disk import scale_height, keplerian_frequency
from diskbridge._params import canonicalize_dust_params
from diskbridge.model.dust_opacity import DustOpacity

if TYPE_CHECKING:
    from .core import Model

# Import SubModel for inheritance
from .core import SubModel

k_B = units('k_B')  # Boltzmann constant
m_H = units('m_H')  # Hydrogen mass
G = units('G')      # Gravitational constant

PAH_N_C_DEFAULT = 54
PAH_N_H_DEFAULT = 18
PAH_X_ISM_DEFAULT = 3.0e-7 * (50.0 / PAH_N_C_DEFAULT)
AMU_G = 1.66053906660e-24
PAH_MASS_G_DEFAULT = (
    PAH_N_C_DEFAULT * 12.011 + PAH_N_H_DEFAULT * 1.008
) * AMU_G


_EPS = np.finfo(np.float64).eps
_LOG_TWO = np.log(2.0)
_SQRT_PI = np.sqrt(np.pi)
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(8)
_GAUSS_NODES = (_GAUSS_NODES + 1.0) / 2.0
_GAUSS_WEIGHTS = _GAUSS_WEIGHTS / 2.0
_WIDE_GAUSS_NODES, _WIDE_GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(16)
_WIDE_GAUSS_NODES = (_WIDE_GAUSS_NODES + 1.0) / 2.0
_WIDE_GAUSS_WEIGHTS = _WIDE_GAUSS_WEIGHTS / 2.0


def _local_wedge_log_average(
    q: float, xa: np.ndarray, dx: np.ndarray, *, wider: bool = False,
) -> np.ndarray:
    """Integrate a smooth, small-exponent-spread wedge without cap differences."""
    base = q * xa
    integral = np.zeros_like(xa)
    nodes, weights = (
        (_WIDE_GAUSS_NODES, _WIDE_GAUSS_WEIGHTS)
        if wider else (_GAUSS_NODES, _GAUSS_WEIGHTS)
    )
    for radial_node, radial_weight in zip(nodes, weights):
        radial_offset = (1.0 - q) * radial_node
        radial = q + radial_offset
        for polar_node, polar_weight in zip(nodes, weights):
            polar_offset = dx * polar_node
            x = xa + polar_offset
            difference = radial_offset * x + q * polar_offset
            exponent = difference * (2.0 * base + difference)
            integral += radial_weight * polar_weight * radial * radial * np.exp(-exponent)
    return -(base * base) + np.log(integral / ((1.0 + q + q * q) / 3.0))


def _local_cap_correction(q: float, x: np.ndarray) -> np.ndarray:
    """Integrate a radially thin cap before its two boundary values cancel."""
    integral = np.zeros_like(x)
    for node, weight in zip(_GAUSS_NODES, _GAUSS_WEIGHTS):
        offset = (1.0 - q) * node
        radial = q + offset
        exponent = x * x * offset * (2.0 * q + offset)
        integral += weight * radial * erfcx(x * radial) * np.exp(-exponent)
    return np.log(integral / ((1.0 + q) / 2.0))


def _one_minus_power(q: float, power: int) -> float:
    if q == 0.0:
        return 1.0
    return -np.expm1(power * np.log(q))


def _log_j_scaled(x: np.ndarray) -> np.ndarray:
    """Return log(exp(x**2) j(x)) for the Gaussian cap primitive.

    j(x) = (1/4 - x**2/2) erfc(x) + x exp(-x**2)/(2 sqrt(pi)).
    A scaled asymptotic series avoids subtraction of large terms in the tail.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    ordinary = x < 12.0
    xo = x[ordinary]
    scaled = (0.25 - 0.5 * xo * xo) * erfcx(xo) + xo / (2.0 * _SQRT_PI)
    out[ordinary] = np.log(scaled)
    xt = x[~ordinary]
    if xt.size:
        inv_square = (1.0 / xt) ** 2
        term = np.ones_like(xt)
        total = term.copy()
        for k in range(1, 64):
            term *= -(k + 1.0) * (2.0 * k - 1.0) / (2.0 * k) * inv_square
            total += term
            if np.all(np.abs(term) <= _EPS * np.abs(total)):
                break
        out[~ordinary] = -np.log(2.0 * _SQRT_PI) - np.log(xt) + np.log(total)
    return out


def _small_cap_log_ratio(x: np.ndarray, q: float) -> np.ndarray:
    """Return log(T(mu)/T(0)) using its convergent erfc series."""
    dq2 = _one_minus_power(q, 2)
    power = x.copy()
    correction = np.zeros_like(x)
    for n in range(32):
        term = (
            4.0 / _SQRT_PI * power
            * _one_minus_power(q, 2 * n + 3)
            / (dq2 * (2 * n + 1) * (2 * n + 3))
        )
        correction += term
        if np.all((term == 0.0) | (np.abs(term) <= _EPS * np.abs(correction))):
            break
        power *= -(x * x) / (n + 1.0)
    return np.log1p(-correction)


def _cap_parts(x: np.ndarray, log_x: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Represent log(T(mu)/T(0)) as -(q*x)**2 + a small correction."""
    base = np.zeros_like(x)
    correction = np.zeros_like(x)
    small = x < 0.5
    xs = x[small]
    base[small] = -(q * xs) ** 2
    correction[small] = _small_cap_log_ratio(xs, q) - base[small]
    large = ~small
    xl = x[large]
    if xl.size:
        xlo = q * xl if q != 0.0 else np.zeros_like(xl)
        log_scaled_lo = _log_j_scaled(xlo)
        log_scaled_hi = _log_j_scaled(xl)
        dq2 = _one_minus_power(q, 2)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            delta = -(xl * xl) * dq2 + log_scaled_hi - log_scaled_lo
            log_difference = np.log(-np.expm1(delta))
            base[large] = -(xlo * xlo)
            correction[large] = (
                log_scaled_lo + log_difference - 2.0 * log_x[large]
                - np.log(dq2 / 2.0)
            )
    local = large & (1.0 - q < 1.0e-3) & (x * x * _one_minus_power(q, 2) <= 1.0)
    if np.any(local):
        base[local] = -(q * x[local]) ** 2
        correction[local] = _local_cap_correction(q, x[local])
    return base, correction


def _central_log_average(xhi: np.ndarray, mu_ratio: np.ndarray, q: float) -> np.ndarray:
    """Convergent exp moment series, avoiding subtraction of central caps."""
    angular_gap = 1.0 - mu_ratio
    with np.errstate(divide="ignore"):
        log_ratio = np.log(mu_ratio)
    power = np.ones_like(xhi)
    total = np.ones_like(xhi)
    dq3 = _one_minus_power(q, 3)
    for n in range(1, 32):
        power *= -(xhi * xhi) / n
        radial_moment = 3.0 * _one_minus_power(q, 2 * n + 3) / ((2 * n + 3) * dq3)
        angular_moment = -np.expm1((2 * n + 1) * log_ratio) / (
            (2 * n + 1) * angular_gap
        )
        term = power * radial_moment * angular_moment
        total += term
        if np.all(np.abs(term) <= _EPS * np.abs(total)):
            break
    return np.log(total)


def _positive_wedge_log_integral(
    q: float, log_h: np.ndarray, mu_lo: np.ndarray, mu_hi: np.ndarray,
) -> np.ndarray:
    """Integrate a positive-z wedge in units of r_hi**3 per unit azimuth.

    With mu = cos(theta), q = r_lo/r_hi and h = H/r_hi, subtract
    Gaussian caps T(mu) = integral_q^1 s**2 integral_mu^infinity
    exp[-s**2 u**2/(2 h**2)] du ds. Caps use the exact radial primitive
    j; convergent series or bounded Gauss integrals regularize cancellation.
    """
    log_h, mu_lo, mu_hi = np.broadcast_arrays(log_h, mu_lo, mu_hi)
    shape = log_h.shape
    log_h = log_h.ravel()
    mu_lo = mu_lo.ravel()
    mu_hi = mu_hi.ravel()
    out = np.full(log_h.size, -np.inf)
    valid = (mu_hi > mu_lo) & (mu_lo >= 0.0)
    if not np.any(valid):
        return out.reshape(shape)
    h = log_h[valid]
    a, b = mu_lo[valid], mu_hi[valid]
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        log_xa = np.log(a) - 0.5 * _LOG_TWO - h
        log_xb = np.log(b) - 0.5 * _LOG_TWO - h
        xa, xb = np.exp(log_xa), np.exp(log_xb)
    values = np.full(h.size, -np.inf)
    central = xb < 0.5
    log_volume = np.log(_one_minus_power(q, 3) / 3.0) + np.log(b - a)
    if np.any(central):
        values[central] = log_volume[central] + _central_log_average(
            xb[central], a[central] / b[central], q,
        )
    # Two successive cap subtractions lose digits for a thin wedge. Integrate
    # the bounded, factored integrand directly with 8 or 16 Gauss nodes when
    # its exponent spread is small enough for that quadrature.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dx = np.exp(np.log(b - a) - 0.5 * _LOG_TWO - h)
        product_difference = (1.0 - q) * xb + q * dx
        spread = product_difference * (xb + q * xa)
    local = (~central) & (spread <= 16.0) & ((1.0 - q) * (b - a) / b < 1.0e-4)
    if np.any(local):
        values[local] = log_volume[local] + _local_wedge_log_average(
            q, xa[local], dx[local], wider=bool(np.any(spread[local] > 1.0)),
        )
    # Infinite xa means the log tail itself lies beyond float64's range.
    remaining = ~central & ~local & np.isfinite(xa)
    if np.any(remaining):
        aa, bb = xa[remaining], xb[remaining]
        base_a, corr_a = _cap_parts(aa, log_xa[remaining], q)
        base_b, corr_b = _cap_parts(bb, log_xb[remaining], q)
        if q == 0.0:
            base_delta = np.zeros_like(aa)
        else:
            # Form the exponent difference before subtracting large logs.
            with np.errstate(over="ignore", invalid="ignore"):
                delta_x = np.exp(np.log(b[remaining] - a[remaining]) - 0.5 * _LOG_TWO - h[remaining])
                base_delta = -(q * delta_x) * (q * (bb + aa))
        delta = base_delta + corr_b - corr_a
        if q == 0.0:
            regular = aa >= 0.5
            if np.any(regular):
                ja = np.exp(_log_j_scaled(aa[regular]) - aa[regular] ** 2)
                jb = np.exp(_log_j_scaled(bb[regular]) - bb[regular] ** 2)
                # The dominant mu**-2 term must use the actual edge gap,
                # rather than subtraction of two independently rounded logs.
                delta[regular] = (
                    -2.0 * np.log1p((b[remaining][regular] - a[remaining][regular]) / a[remaining][regular])
                    + np.log1p(-4.0 * jb) - np.log1p(-4.0 * ja)
                )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_difference = np.log(-np.expm1(delta))
        log_t0 = h[remaining] + 0.5 * np.log(np.pi / 2.0) + np.log(_one_minus_power(q, 2) / 2.0)
        values[remaining] = log_t0 + base_a + corr_a + log_difference
    out[valid] = values
    return out.reshape(shape)


def _spherical_gaussian_log_average(
    radial_edges_cm: np.ndarray,
    theta_edges_rad: np.ndarray,
    scale_height_cm: np.ndarray,
) -> np.ndarray:
    """Return log spherical-cell volume averages of exp(-z**2/(2 H**2)).

    H is held fixed within each native radial/azimuthal column cell. The
    polar and radial integrals use the actual spherical volume element;
    azimuthal width cancels from the average. Nonpositive/nonfinite H gives
    log zero. The returned shape is (nr, ntheta, nphi).
    """
    radial = np.asarray(radial_edges_cm, dtype=float)
    theta = np.asarray(theta_edges_rad, dtype=float)
    heights = np.asarray(scale_height_cm, dtype=float)
    if (
        radial.ndim != 1 or radial.size < 2
        or np.any(~np.isfinite(radial)) or np.any(radial < 0.0)
        or np.any(np.diff(radial) <= 0.0)
    ):
        raise ValueError("radial edges must be finite, increasing and nonnegative")
    if (
        theta.ndim != 1 or theta.size < 2
        or np.any(~np.isfinite(theta)) or np.any(np.diff(theta) <= 0.0)
        or theta[0] < 0.0 or theta[-1] > np.pi
    ):
        raise ValueError("theta edges must increase within [0, pi]")
    if heights.ndim != 2 or heights.shape[0] != radial.size - 1:
        raise ValueError("scale heights must have shape (nr, nphi)")
    # Algebraically cos(theta), with an exactly zero global-midplane face.
    mu = -np.sin(theta - np.pi / 2.0)
    low, high = mu[1:], mu[:-1]
    positive_lo = np.maximum(low, 0.0)
    positive_hi = np.maximum(high, 0.0)
    negative_lo = np.maximum(-high, 0.0)
    negative_hi = np.maximum(-low, 0.0)
    result = np.full((heights.shape[0], theta.size - 1, heights.shape[1]), -np.inf)
    for index, (rlo, rhi) in enumerate(zip(radial[:-1], radial[1:])):
        valid = np.isfinite(heights[index]) & (heights[index] > 0.0)
        if not np.any(valid):
            continue
        log_h = np.log(heights[index, valid]) - np.log(rhi)
        q = rlo / rhi
        log_positive = _positive_wedge_log_integral(
            q, log_h[None, :], positive_lo[:, None], positive_hi[:, None],
        )
        log_negative = _positive_wedge_log_integral(
            q, log_h[None, :], negative_lo[:, None], negative_hi[:, None],
        )
        log_volume = np.log(_one_minus_power(q, 3) / 3.0) + np.log(high - low)
        averaged = np.logaddexp(log_positive, log_negative) - log_volume[:, None]
        result[index][:, valid] = np.minimum(averaged, 0.0)
    return result


def _current_params():
    import diskbridge

    return diskbridge.params


def _select_component_value(value, index: int = 0):
    return value[index] if isinstance(value, list) else value


def stokes_number(
    grain_size: Quantity,
    gas_density: Quantity,
    gas_temperature: Quantity,
    keplerian_freq: Quantity,
    grain_density: Quantity,
    mean_molecular_weight: float,
) -> Quantity:
    """Compute Stokes number for Epstein drag regime.
    
    St = Omega_K * t_s where t_s = (rho_s * a) / (rho_g * c_s)
    
    Parameters
    ----------
    grain_size : Quantity
        Grain radius [length]
    gas_density : Quantity
        Gas density [mass/volume]
    gas_temperature : Quantity
        Gas temperature [temperature]
    keplerian_freq : Quantity
        Keplerian frequency Omega_K [1/time]
    grain_density : Quantity
        Material density of dust grains [mass/volume]
    mean_molecular_weight : float
        Mean molecular weight of gas (dimensionless)
        
    Returns
    -------
    Quantity
        Stokes number (dimensionless)
        
    Notes
    -----
    Valid in the Epstein drag regime where grain size << gas mean free path.
    """
    c_s = np.sqrt(k_B * gas_temperature / (mean_molecular_weight * m_H))
    t_s = (grain_density * grain_size) / (gas_density * c_s)
    St = keplerian_freq * t_s
    
    return St

class DustBin:
    """Represents a single dust size bin with properties and density field.
    
    This class provides access to an individual dust bin's properties and 
    can compute its density field on-the-fly if dust is proportional to gas.
    
    Attributes:
        name: Bin identifier (e.g., 'bin_0')
        size: Grain size (geometric mean for this bin)
        density_material: Bulk grain density, including material porosity
        mass_fraction: Fraction of total dust mass in this bin
        size_min: Minimum size of this bin
        size_max: Maximum size of this bin
    """
    
    def __init__(
        self,
        parent_dust: 'Dust',
        bin_index: int,
        size: Quantity,
        size_min: Quantity,
        size_max: Quantity,
        mass_fraction: float,
        density_material: Quantity,
        opacity: DustOpacity,
    ):
        """Initialize a grain bin with resolved material properties.

        Parameters
        ----------
        parent_dust : Dust
            Owning dust submodel.
        bin_index : int
            Global bin index across all components.
        size, size_min, size_max : Quantity
            Representative, minimum and maximum grain radii.
        mass_fraction : float
            Bin mass fraction within its component.
        density_material : Quantity
            Resolved bulk grain density, including porosity.
        opacity : DustOpacity
            Material, shape and porosity specification shared by the component.
        """
        self.parent_dust = parent_dust
        self.bin_index = bin_index
        self.name = f"bin_{bin_index}"
        self.size = size
        self.size_min = size_min
        self.size_max = size_max
        self.mass_fraction = mass_fraction
        self.density_material = density_material
        self.species_grain_density = density_material
        self.opacity = opacity
        self.opacity_species = f"{opacity.species_name}{bin_index}"
        self._stored_fields: Dict[str, Field] = {}
        
    def __getitem__(self, key: str) -> Field:
        """Access fields for this bin (e.g., bin['density']).
        
        Args:
            key: Field name (e.g., 'density')
            
        Returns:
            Field object for this bin
        """
        if key == 'density':
            return self._get_density()
        elif key in self._stored_fields:
            return self._stored_fields[key]
        else:
            raise KeyError(f"Field '{key}' not available for dust bin {self.name}")
            
    def _get_density(self) -> Field:
        """Get dust density field for this bin."""
        return self.parent_dust._compute_bin_density(self.bin_index)
        
    def __repr__(self) -> str:
        return (
            f"DustBin(name='{self.name}', size={self.size}, "
            f"mass_fraction={self.mass_fraction:.4f})"
        )
        
    def __str__(self) -> str:
        return self.__repr__()


class DustDistribution:
    """Manages dust grain size distribution with power-law.
    
    This class handles the grain size bins and computes mass fractions
    following the same power-law formalism as fargo2radmc3d:
    
    frac[ibin] = (a[i+1]^(4-p) - a[i]^(4-p)) / (amax^(4-p) - amin^(4-p))
    
    where p is the power-law index (typically 3.5 for MRN distribution).
    """
    
    def __init__(
        self,
        amin: Quantity,
        amax: Quantity,
        nbin: int,
        grain_density: Quantity,
        power_index: float = 3.5,
    ):
        """Initialize dust grain size distribution.
        
        Args:
            amin: Minimum grain size
            amax: Maximum grain size
            nbin: Number of size bins
            power_index: Power-law exponent (3.5 for MRN)
            grain_density: Material density of grains
        """
        self.amin = amin
        self.amax = amax
        self.nbin = nbin
        self.power_index = power_index
        self.grain_density = grain_density
        
        # Create logarithmically spaced bin edges
        amin_val = amin.to('m').magnitude
        amax_val = amax.to('m').magnitude
        bin_edges = np.logspace(
            np.log10(amin_val),
            np.log10(amax_val),
            nbin + 1
        ) * units('m')
        
        self.bin_edges = bin_edges
        self.bin_centers = self._compute_bin_centers()
        self.mass_fractions = self._compute_mass_fractions()
        
    def _compute_bin_centers(self) -> Quantity:
        """Compute geometric mean grain size for each bin."""
        edges = self.bin_edges.magnitude
        centers = 10 ** (0.5 * (np.log10(edges[:-1]) + np.log10(edges[1:])))
        return centers * self.bin_edges.units
        
    def _compute_mass_fractions(self) -> np.ndarray:
        """Compute mass fraction in each size bin using power-law.
        
        Following fargo2radmc3d formalism:
        frac[i] = (a[i+1]^(4-p) - a[i]^(4-p)) / (amax^(4-p) - amin^(4-p))
        """
        edges = self.bin_edges.magnitude
        p = self.power_index
        
        # Compute normalization factor
        norm = edges[-1]**(4.0 - p) - edges[0]**(4.0 - p)
        
        # Compute fraction for each bin
        fractions = np.zeros(self.nbin)
        for i in range(self.nbin):
            fractions[i] = (edges[i+1]**(4.0 - p) - edges[i]**(4.0 - p)) / norm
            
        # Verify normalization
        assert np.abs(np.sum(fractions) - 1.0) < 1e-10, "Mass fractions don't sum to 1"
        
        return fractions
        
class DustComponent:
    """Represents a single dust component with its own distribution and spatial region.
    
    Attributes:
        distribution: DustDistribution for this component
        dust_to_gas_ratio: Dust-to-gas mass ratio for this component
        mode: 'proportional' or 'settling'
        mask: Optional Field defining where this component exists
        alpha: Turbulent viscosity parameter (for settling mode)
        delta: Turbulent diffusion parameter (for settling mode)
        mean_molecular_weight: Mean molecular weight for gas
        species_base: Base name for dust species (for opacity files)
        opacity: Material, grain-shape and porosity specification
        component_index: Index of this component in the parent Dust
    """
    
    def __init__(
        self,
        distribution: DustDistribution,
        dust_to_gas_ratio: float,
        opacity: DustOpacity,
        mode: Literal['proportional', 'settling'] = 'proportional',
        mask: Optional[Field] = None,
        alpha: Optional[float] = None,
        delta: Optional[float] = None,
        mean_molecular_weight: float = 2.3,
        component_index: int = 0,
    ):
        """Initialize a component from a resolved grain distribution.

        Parameters
        ----------
        distribution : DustDistribution
            Size distribution with its resolved bulk grain density.
        dust_to_gas_ratio : float
            Component dust-to-gas mass ratio.
        opacity : DustOpacity
            Material, shape and porosity specification.
        mode : {"proportional", "settling"}
            Spatial distribution relative to the gas.
        mask : Field, optional
            Dimensionless spatial weight for this component.
        alpha, delta : float, optional
            Turbulent viscosity and diffusion parameters.
        mean_molecular_weight : float, optional
            Gas molecular weight in hydrogen-mass units.
        component_index : int, optional
            Index in the owning dust submodel.

        Notes
        -----
        The supplied distribution already contains the resolved density.
        Constructing a component does not query material files or executables.
        """
        self.distribution = distribution
        self.dust_to_gas_ratio = dust_to_gas_ratio
        self.mode = mode
        self.mask = mask
        self.alpha = alpha
        self.delta = delta if delta is not None else alpha
        self.mean_molecular_weight = mean_molecular_weight
        self.opacity = opacity
        self.species_base = opacity.species_name
        self.grain_density = distribution.grain_density
        self.species_grain_density = distribution.grain_density
        self.component_index = component_index
        
    def __repr__(self) -> str:
        return (
            f"DustComponent(index={self.component_index}, "
            f"nbin={self.distribution.nbin}, "
            f"size_range={self.distribution.amin} to {self.distribution.amax}, "
            f"dust/gas={self.dust_to_gas_ratio:.3e}, mode='{self.mode}', "
            f"species='{self.species_base}')"
        )


class Dust(SubModel):
    """SubModel for dust grains in a disk model.
    
    This class provides memory-efficient storage of dust distributions.
    Supports two modes:
    1. 'proportional': Dust density proportional to gas (simple scaling)
    2. 'settling': Vertical settling-diffusion equilibrium (Youdin & Lithwick 2007)
    
    Access patterns:
        model.dust['density']  # Total dust density
        model.dust['bin_0']    # Access individual bin (returns DustBin object)
        model.dust['bin_0']['density']  # Density of specific bin
        model.dust['bin_0'].size  # Grain size of bin
    """
    
    def __init__(self, parent_model: Model):
        """Initialize dust submodel.
        
        Args:
            parent_model: Parent Model instance
        """
        # Call SubModel.__init__
        super().__init__(parent_model)
        
        # Dust-specific fields
        self._dust_fields: Dict[str, Field] = {}
        self._lazy_builders: Dict[str, Callable[[], Field]] = {}
        self._bins: Dict[str, DustBin] = {}
        
        # Multi-component support
        self._components: List[DustComponent] = []
        self._global_bins: Dict[str, tuple[int, int]] = {}  # bin_name -> (comp_idx, local_bin_idx)
        self._has_region_components: bool = False
        
        # Legacy single-component attributes (for backward compatibility)
        # These now proxy to the first component when available
        self.distribution: Optional[DustDistribution] = None
        self.dust_to_gas_ratio: Optional[float] = None
        self.mode: Literal['proportional', 'settling'] = 'proportional'
        
        # Settling-diffusion parameters (legacy)
        self.alpha: Optional[float] = None  # Turbulent viscosity parameter
        self.delta: Optional[float] = None  # Turbulent diffusion parameter (default: = alpha)
        self.mean_molecular_weight: float = 2.3  # Mean molecular weight (for H2)
        
        
    def set_distribution(
        self,
        amin: Optional[Quantity] = None,
        amax: Optional[Quantity] = None,
        nbin: Optional[int] = None,
        power_index: Optional[float] = None,
        grain_density: Optional[Quantity] = None,
        dust_to_gas_ratio: Optional[float] = None,
        mode: Literal['proportional', 'settling'] = 'proportional',
        alpha: Optional[float] = None,
        delta: Optional[float] = None,
        mean_molecular_weight: float = 2.3,
        opacity: DustOpacity | None = None,
    ) -> None:
        """Set the global grain distribution and its material properties.

        Parameters
        ----------
        amin, amax : Quantity, optional
            Minimum and maximum grain radii. Defaults come from the first
            component of ``diskbridge.params``.
        nbin : int, optional
            Number of logarithmically spaced size bins.
        power_index : float, optional
            Number-distribution exponent in ``dn/da proportional to a**(-p)``.
        grain_density : Quantity, optional
            Compact material density when ``opacity`` is omitted. With an
            explicit specification, set its ``density`` instead.
        dust_to_gas_ratio : float, optional
            Dust-to-gas mass ratio.
        mode : {"proportional", "settling"}
            Spatial distribution of dust relative to gas.
        alpha, delta : float, optional
            Turbulent viscosity and diffusion parameters for settling.
            ``delta`` defaults to ``alpha``.
        mean_molecular_weight : float, optional
            Gas molecular weight in hydrogen-mass units.
        opacity : DustOpacity, optional
            Material, shape and porosity settings. If omitted, construct a Mie
            specification from the first component's species and grain density.
            The resolved bulk density is used by settling and surface areas.

        Raises
        ------
        ValueError
            If region dust is already configured, settling lacks turbulence
            parameters, or both ``opacity`` and ``grain_density`` are supplied.
        """
        if getattr(self, "_has_region_components", False) and self._components:
            raise ValueError(
                "Cannot use global dust.set_distribution after region-based dust components "
                "have been configured. Use region.dust.set_distribution() instead."
            )

        current_params = _current_params()

        # Pull defaults from params if not provided
        if amin is None:
            # params.amin is already a Quantity in microns
            amin = _select_component_value(current_params.amin)
        if amax is None:
            # params.amax is already a Quantity in microns
            amax = _select_component_value(current_params.amax)
        if nbin is None:
            nbin = int(_select_component_value(current_params.nbins))
        if power_index is None:
            power_index = float(_select_component_value(current_params.pindex))
        if dust_to_gas_ratio is None:
            dust_to_gas_ratio = float(_select_component_value(current_params.dust_to_gas_ratio))

        if opacity is None:
            opacity = DustOpacity(
                material=_select_component_value(current_params.species),
                density=(grain_density if grain_density is not None else
                         _select_component_value(current_params.grain_density)),
            )
        elif grain_density is not None:
            raise ValueError("Set density on opacity instead of supplying grain_density with opacity")
        _, grain_density = opacity.resolve_material(opacity_dir=current_params.opacity_dir)
        
        distribution = DustDistribution(
            amin=amin,
            amax=amax,
            nbin=nbin,
            power_index=power_index,
            grain_density=grain_density,
        )
        
        # Set turbulence parameters for settling mode
        if mode == 'settling':
            if alpha is None:
                if self.parent.disk is not None:
                    alpha_q = self.parent.disk.parameters.get("alphavisocity")
                else:
                    alpha_q = None
                if alpha_q is None:
                    raise ValueError("alpha parameter required for settling mode")
                alpha = float(getattr(alpha_q, "magnitude", alpha_q))
            delta = delta if delta is not None else alpha  # Assume Sc ~ 1
            logger.info(f"Settling mode: alpha={alpha}, delta={delta}")
        
        # Create a single component (for backward compatibility)
        component = DustComponent(
            distribution=distribution,
            dust_to_gas_ratio=dust_to_gas_ratio,
            opacity=opacity,
            mode=mode,
            mask=self.mask,  # Use dust-level mask if set
            alpha=alpha,
            delta=delta,
            mean_molecular_weight=mean_molecular_weight,
            component_index=0,
        )
        
        # Set as the only component
        self._components = [component]
        
        # Rebuild global bins mapping
        self._global_bins = {}
        self._bins = {}
        global_bin_idx = 0
        for i in range(nbin):
            # Map global bin to (component_index, local_bin_index)
            self._global_bins[f"bin_{global_bin_idx}"] = (0, i)
            
            # Create DustBin for backward compatibility
            dust_bin = DustBin(
                parent_dust=self,
                bin_index=global_bin_idx,
                size=distribution.bin_centers[i],
                size_min=distribution.bin_edges[i],
                size_max=distribution.bin_edges[i + 1],
                mass_fraction=distribution.mass_fractions[i],
                density_material=component.grain_density,
                opacity=component.opacity,
            )
            self._bins[f"bin_{global_bin_idx}"] = dust_bin
            global_bin_idx += 1
        
        # Update legacy attributes to proxy to first component
        self.distribution = distribution
        self.dust_to_gas_ratio = dust_to_gas_ratio
        self.mode = mode
        self.alpha = alpha
        self.delta = delta if delta is not None else alpha
        self.mean_molecular_weight = mean_molecular_weight
        
        logger.info(
            f"Dust distribution set: {nbin} bins from {amin.to('um')} to {amax.to('um')}, "
            f"dust/gas={dust_to_gas_ratio:.3e}, power_index={power_index}, mode={mode}"
        )
    
    def _resolve_mask_field(
        self,
        mask: Union[Field, str],
        *,
        complement: bool = False,
    ) -> Field:
        if isinstance(mask, str):
            if self.parent.gas is None or mask not in self.parent.gas:
                raise KeyError(f"Gas mask/weight field {mask!r} not found")
            source = self.parent.gas[mask]
            source_name = mask
        else:
            source = mask
            source_name = None

        from .utils import validate_field_against_mesh, field_data_as_order

        validate_field_against_mesh(source, self.parent.mesh)  # type: ignore[arg-type]
        target = self.parent.mesh.axis_names()  # type: ignore[union-attr]
        data = field_data_as_order(source, target).to("dimensionless")
        values = np.clip(np.asarray(data.magnitude, dtype=float), 0.0, 1.0)
        attrs = dict(source.attrs)

        if complement:
            values = 1.0 - values
            if source_name is not None:
                attrs["complement_of"] = source_name

        return Field(
            data=Quantity(values, "dimensionless"),
            quantity=source.quantity,
            axis_order=target,
            attrs=attrs,
        )

    def add_component_from_mask(
        self,
        mask: Union[Field, str],
        mode: Literal['proportional', 'settling'] = 'proportional',
        complement: bool = False,
        amin: Optional[Quantity] = None,
        amax: Optional[Quantity] = None,
        nbin: Optional[int] = None,
        power_index: Optional[float] = None,
        grain_density: Optional[Quantity] = None,
        dust_to_gas_ratio: Optional[float] = None,
        alpha: Optional[float] = None,
        delta: Optional[float] = None,
        mean_molecular_weight: float = 2.3,
        opacity: DustOpacity | None = None,
    ) -> None:
        """Add a masked grain distribution with its own material properties.

        Parameters
        ----------
        mask : Field or str
            Dimensionless gas weight field, clipped to the range [0, 1].
        mode : {"proportional", "settling"}
            Spatial distribution of dust relative to gas.
        complement : bool, optional
            Use one minus the clipped weight.
        amin, amax : Quantity, optional
            Minimum and maximum grain radii.
        nbin : int, optional
            Number of logarithmically spaced size bins.
        power_index : float, optional
            Number-distribution exponent in ``dn/da proportional to a**(-p)``.
        grain_density : Quantity, optional
            Compact material density when ``opacity`` is omitted. With an
            explicit specification, set its ``density`` instead.
        dust_to_gas_ratio : float, optional
            Dust-to-gas mass ratio for this component.
        alpha, delta : float, optional
            Turbulent viscosity and diffusion parameters for settling.
            ``delta`` defaults to ``alpha``.
        mean_molecular_weight : float, optional
            Gas molecular weight in hydrogen-mass units.
        opacity : DustOpacity, optional
            Material, shape and porosity settings. Defaults use this component's
            species and density from ``diskbridge.params``. Resolve bulk density
            before constructing the distribution or computing settling.

        Raises
        ------
        ValueError
            If component defaults or settling parameters are missing, or both
            ``opacity`` and ``grain_density`` are supplied.
        """
        mask = self._resolve_mask_field(mask, complement=complement)

        # Determine this component's index
        self._has_region_components = True
        component_index = len(self._components)

        # Pull canonical dust parameters from global params
        canon = canonicalize_dust_params(_current_params())
        ncomp = canon['ncomp']
        if component_index >= ncomp:
            if ncomp == 1:
                param_index = 0
            else:
                raise ValueError(
                    f"Requested dust component index {component_index}, "
                    f"but canonicalized params define only {ncomp} components."
                )
        else:
            param_index = component_index

        # Use canonical per-component values if not explicitly provided
        if amin is None:
            amin = canon['amin'][param_index]
        if amax is None:
            amax = canon['amax'][param_index]
        if nbin is None:
            nbin = int(canon['nbins'][param_index])
        if power_index is None:
            power_index = float(canon['pindex'][param_index])
        if dust_to_gas_ratio is None:
            dust_to_gas_ratio = float(canon['dust_to_gas_ratio'][param_index])

        # Validate settling mode
        if mode == 'settling':
            if alpha is None:
                if self.parent.disk is not None:
                    alpha_q = self.parent.disk.parameters.get("alphavisocity")
                else:
                    alpha_q = None
                if alpha_q is None:
                    raise ValueError("alpha parameter required for settling mode")
                alpha = float(getattr(alpha_q, "magnitude", alpha_q))
            delta = delta if delta is not None else alpha
            logger.info(f"Settling mode component: alpha={alpha}, delta={delta}")
        
        if opacity is None:
            opacity = DustOpacity(
                material=canon['species'][param_index],
                density=(grain_density if grain_density is not None else
                         canon['grain_density'][param_index]),
            )
        elif grain_density is not None:
            raise ValueError("Set density on opacity instead of supplying grain_density with opacity")
        _, grain_density = opacity.resolve_material(opacity_dir=_current_params().opacity_dir)

        # Create distribution for this component
        distribution = DustDistribution(
            amin=amin,
            amax=amax,
            nbin=nbin,
            power_index=power_index,
            grain_density=grain_density,
        )
        
        # Create component
        component = DustComponent(
            distribution=distribution,
            dust_to_gas_ratio=dust_to_gas_ratio,
            opacity=opacity,
            mode=mode,
            mask=mask,
            alpha=alpha,
            delta=delta,
            mean_molecular_weight=mean_molecular_weight,
            component_index=component_index,
        )
        
        # Add to components list
        self._components.append(component)
        
        # Update global bins mapping
        global_bin_start = len(self._global_bins)
        for local_idx in range(nbin):
            global_bin_idx = global_bin_start + local_idx
            bin_name = f"bin_{global_bin_idx}"
            self._global_bins[bin_name] = (component_index, local_idx)
            
            # Create DustBin for backward compatibility
            dust_bin = DustBin(
                parent_dust=self,
                bin_index=global_bin_idx,
                size=distribution.bin_centers[local_idx],
                size_min=distribution.bin_edges[local_idx],
                size_max=distribution.bin_edges[local_idx + 1],
                mass_fraction=distribution.mass_fractions[local_idx],
                density_material=component.grain_density,
                opacity=component.opacity,
            )
            self._bins[bin_name] = dust_bin
        
        logger.info(
            f"Added dust component {component_index}: {nbin} bins, "
            f"{amin.to('um')} to {amax.to('um')}, "
            f"dust/gas={dust_to_gas_ratio:.3e}, mode={mode}"
        )

    def add_to_disk(
        self,
        mode: Optional[Literal['proportional', 'settling']] = None,
        *,
        include_ism: bool = True,
        alpha: Optional[float] = None,
        mask_field: Optional[str] = None,
        opacity: DustOpacity | None = None,
        ism_opacity: DustOpacity | None = None,
    ) -> None:
        """Add disk dust and optionally an independent ISM dust component.

        Parameters
        ----------
        mode : {"proportional", "settling"}, optional
            Disk distribution mode; defaults to ``params.disk_dust_mode``.
        include_ism : bool, optional
            Add proportional dust on the complementary gas weight.
        alpha : float, optional
            Turbulent viscosity parameter for disk settling.
        mask_field : str, optional
            Gas weight field. Prefer ``disk_weight`` when available.
        opacity : DustOpacity, optional
            Disk material, shape and porosity settings.
        ism_opacity : DustOpacity, optional
            ISM material, shape and porosity settings. Requires ``include_ism``.

        Notes
        -----
        Omitted specifications use the corresponding component's parameter
        defaults; each component retains its own resolved bulk grain density.
        """
        if ism_opacity is not None and not include_ism:
            raise ValueError("ism_opacity requires include_ism=True")
        if self.parent.gas is None:
            raise KeyError("Model has no gas submodel")
        if mask_field is None:
            mask_field = "disk_weight" if "disk_weight" in self.parent.gas else "disk_mask"
        if mask_field not in self.parent.gas:
            raise KeyError("Disk mask/weight not found; call set_mask_from_joos_disk first")

        current_params = _current_params()
        disk_mode = mode if mode is not None else current_params.disk_dust_mode
        if disk_mode not in ("proportional", "settling"):
            raise ValueError("disk_dust_mode must be 'proportional' or 'settling'")

        disk_alpha = alpha
        if disk_mode == "settling" and disk_alpha is None:
            disk_alpha = current_params.disk_dust_settling_alpha

        self.add_component_from_mask(
            mask=mask_field,
            mode=disk_mode,
            alpha=disk_alpha,
            opacity=opacity,
        )

        if include_ism:
            self.add_component_from_mask(
                mask=mask_field,
                complement=True,
                mode="proportional",
                opacity=ism_opacity,
            )

    def add_pah_to_disk(
        self,
        *,
        f_pah_disk: float = 0.01,
        include_ism: bool = True,
        disk_weight_field: str = "disk_weight",
        disk_mask_field: str = "disk_mask",
    ) -> None:
        """Construct PAH abundance fields from the disk/ISM material split.

        PAHs are registered as gas-side chemistry/material fields, not as
        ordinary dust bins, so they do not contribute to dust surface-area
        moments used for gas-dust coupling or CO freeze-out.
        """
        if self.parent.gas is None:
            raise KeyError("Model has no gas submodel")
        if "density" not in self.parent.gas:
            raise KeyError("Need gas['density'] to construct PAH abundance")
        if f_pah_disk < 0.0:
            raise ValueError("f_pah_disk must be non-negative")

        if disk_weight_field in self.parent.gas:
            w_field = self.parent.gas[disk_weight_field]
        elif disk_mask_field in self.parent.gas:
            w_field = self.parent.gas[disk_mask_field]
        else:
            raise KeyError("Need disk_weight or disk_mask to construct disk PAH field")

        target = self.parent.mesh.axis_names()  # type: ignore[union-attr]
        from .utils import field_data_as_order

        rho_gas = field_data_as_order(
            self.parent.gas["density"], target
        ).to("g/cm^3")
        w_disk_q = field_data_as_order(w_field, target).to("dimensionless")
        w_disk = np.clip(np.asarray(w_disk_q.magnitude, dtype=float), 0.0, 1.0)
        w_ism = 1.0 - w_disk

        # Match the nH convention used by the RADMC/GOW17 bridge.
        nH = rho_gas / (1.4 * m_H)
        if include_ism:
            D_pah = w_ism + f_pah_disk * w_disk
        else:
            D_pah = f_pah_disk * w_disk
        D_pah = np.where(np.isfinite(D_pah) & (D_pah >= 0.0), D_pah, 0.0)
        rho_pah = (
            nH
            * Quantity(PAH_X_ISM_DEFAULT, "dimensionless")
            * Quantity(PAH_MASS_G_DEFAULT, "g")
            * Quantity(D_pah, "dimensionless")
        ).to("g/cm^3")
        attrs = {
            "role": "pah",
            "pah_model": "C54H18",
            "N_C": PAH_N_C_DEFAULT,
            "N_H": PAH_N_H_DEFAULT,
            "x_PAH_ISM": PAH_X_ISM_DEFAULT,
            "m_PAH_g": PAH_MASS_G_DEFAULT,
            "f_PAH_disk": float(f_pah_disk),
            "include_ism": bool(include_ism),
            "source": "disk_mask_mixture",
        }
        self.parent.gas_register(
            "pah_abundance_rel_ism",
            Field(
                quantity="pah_abundance_rel_ism",
                data=Quantity(D_pah, "dimensionless"),
                axis_order=target,
                attrs=attrs,
            ),
        )
        self.parent.gas_register(
            "pah_density",
            Field(
                quantity="pah_density",
                data=rho_pah,
                axis_order=target,
                attrs=attrs,
            ),
        )

    def h2_formation_reference_area_per_H(self) -> Quantity:
        """ISM-reference projected ordinary-grain area per H."""
        if not self._components:
            raise RuntimeError("No dust components available")

        candidates = []
        for comp in self._components:
            attrs = getattr(comp.mask, "attrs", {}) if comp.mask is not None else {}
            if attrs.get("complement_of") in {"disk_mask", "disk_weight"}:
                candidates.append(comp)

        if len(candidates) == 1:
            comp = candidates[0]
        elif len(candidates) == 0 and len(self._components) == 1:
            comp = self._components[0]
        else:
            raise RuntimeError(
                "Cannot identify a unique ISM dust reference component for H2 formation"
            )

        mu_H = 1.4
        m_H_g = float(m_H.to("g").magnitude)
        sigma = 0.0
        for i in range(comp.distribution.nbin):
            a_i = float(comp.distribution.bin_centers[i].to("cm").magnitude)
            rho_s = float(comp.grain_density.to("g/cm^3").magnitude)
            f_i = float(comp.distribution.mass_fractions[i])
            sigma += (
                3.0
                * float(comp.dust_to_gas_ratio)
                * f_i
                * mu_H
                * m_H_g
                / (4.0 * a_i * rho_s)
            )
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise RuntimeError("Invalid ISM H2 formation surface reference")
        return Quantity(sigma, "cm^2")
        
    def _compute_dust_scale_height(
        self,
        gas_scale_height: Quantity,
        stokes_number: Quantity,
    ) -> Quantity:
        """Compute dust scale height from settling-diffusion equilibrium.
        
        H_d / H_g = sqrt(delta / (St + delta))
        
        Following Youdin & Lithwick (2007), with turbulent diffusion parameter delta.
        For Schmidt number Sc ~ 1, delta ~ alpha.
        
        Args:
            gas_scale_height: Gas scale height H_g
            stokes_number: Stokes number
            
        Returns:
            Dust scale height H_d
        """
        if self.delta is None:
            raise ValueError("delta parameter not set (required for settling mode)")
            
        delta = self.delta
        
        # Handle Quantity arrays
        if isinstance(stokes_number, Quantity):
            St_mag = stokes_number.magnitude
        else:
            St_mag = stokes_number
            
        # Compute scale height ratio
        ratio = np.sqrt(delta / (St_mag + delta))
        
        H_d = gas_scale_height * ratio
        
        return H_d
        
    def _compute_gas_scale_height(self) -> Quantity:
        # FLAG THIS FOR FUTURE CONSIDERATION
        """Compute gas scale height H = c_s / Omega_K.
        
        Returns:
            Gas scale height as function of radius
        """
        mesh = self.parent.mesh
        
        # Get radial coordinates
        r = mesh.centers('r')
        
        # Option 1: Use disk parameters if available
        if self.parent.disk is not None:
            h0 = self.parent.disk.parameters.get('aspectratio')
            fl = self.parent.disk.parameters.get('flaringindex', 0.0)
            r0 = self.parent.disk.parameters.get('r0')
            
            if h0 is not None and r0 is not None:
                H = scale_height(r, h0, r0, fl)
                return H
                
        # Option 2: Compute from temperature and stellar mass
        if 'temperature' in self.parent.gas:
            temp = self.parent.gas['temperature'].data
            
            # Get stellar mass
            M_star = units('solar_mass').to('g')
            if hasattr(self.parent, 'variables'):
                if 'mstar' in self.parent.variables:
                    M_star = self.parent.variables['mstar']
                    
            # Keplerian frequency (1D array)
            Omega_K_1d = keplerian_frequency(r, M_star)
            
            # Sound speed (3D array)
            mu = self.mean_molecular_weight
            c_s = np.sqrt(k_B * temp / (mu * m_H))
            
            # Broadcast Omega_K to match c_s shape 
            phi = mesh.centers('phi')
            theta = mesh.centers('theta')
            Omega_K = np.meshgrid(Omega_K_1d, theta, phi, indexing='ij')[0]
            
            # Scale height (3D array)
            H = c_s / Omega_K
            return H
            
        raise ValueError("Cannot compute scale height: no disk parameters or temperature")
    
    def _compute_bin_density(self, bin_index: int) -> Field:
        """Compute dust density field for a specific size bin.
        
        Component-aware: delegates to the appropriate component's logic.
        
        Args:
            bin_index: Global index of the size bin (0 to total_nbin-1)
            
        Returns:
            Field containing dust density for this bin
        """
        # Check for stored field first
        key = f"density_bin_{bin_index}"
        if key in self._dust_fields:
            return self._dust_fields[key]
        
        # Look up which component this bin belongs to
        bin_name = f"bin_{bin_index}"
        if bin_name not in self._global_bins:
            raise KeyError(f"Bin {bin_name} not found in global bins mapping")
        
        comp_idx, local_bin_idx = self._global_bins[bin_name]
        component = self._components[comp_idx]
        
        # Get gas density
        if 'density' in self.parent.gas:
            gas_density = self.parent.gas['density']
        elif 'surface_density' in self.parent.gas:
            # For 2D models, use surface density (proportional mode only)
            if component.mode == 'settling':
                raise ValueError("Settling mode requires 3D model with density field")
            gas_density = self.parent.gas['surface_density']
        else:
            raise KeyError("No gas density or surface_density field found")
        
        # Delegate to component-specific computation
        return self._compute_component_bin_density(
            component, local_bin_idx, gas_density
        )
    
    def _compute_component_bin_density(
        self, component: DustComponent, local_bin_idx: int, gas_density: Field
    ) -> Field:
        """Compute dust density for a specific bin within a component.
        
        Args:
            component: DustComponent instance
            local_bin_idx: Index within the component's bins
            gas_density: Gas density field
            
        Returns:
            Field containing dust density for this bin
        """
        # Mode 1: Proportional (simple scaling)
        if component.mode == 'proportional':
            dust_density_data = (
                component.dust_to_gas_ratio * 
                component.distribution.mass_fractions[local_bin_idx] *
                gas_density.data
            )
            
            # Apply component mask if set
            if component.mask is not None:
                w = np.asarray(component.mask.data.to("dimensionless").magnitude, dtype=float)
                w = np.clip(w, 0.0, 1.0)
                dust_density_data = dust_density_data * w
                
            dust_field = Field(
                data=dust_density_data,
                quantity=gas_density.quantity,
                axis_order=gas_density.axis_order,
            )
            
            return dust_field
            
        # Mode 2: Settling-diffusion equilibrium
        elif component.mode == 'settling':
            return self._compute_settling_density(component, local_bin_idx, gas_density)
            
        else:
            raise ValueError(f"Unknown mode: {component.mode}")
    
    
    def _compute_settling_density(self, component: DustComponent, local_bin_idx: int, gas_density: Field) -> Field:
        """Compute masked, volume-averaged settling density on spherical cells.

        Parameters
        ----------
        component : DustComponent
            Grain distribution, settling parameters and optional spatial mask.
        local_bin_idx : int
            Grain index within the component.
        gas_density : Field
            Native spherical gas density.

        Returns
        -------
        Field
            Masked cell-average density without column renormalization.
        """
        prefactor, log_average = self._settling_cell_profile(
            component, local_bin_idx, gas_density,
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            dust_density_data = Quantity(
                np.exp(np.log(prefactor.to("g/cm^3").magnitude) + log_average),
                "g/cm^3",
            )

        if component.mask is not None:
            w = np.asarray(component.mask.data.to("dimensionless").magnitude, dtype=float)
            w = np.clip(w, 0.0, 1.0)
            dust_density_data = dust_density_data * w

        dust_density_data = np.where(np.isfinite(dust_density_data), dust_density_data, 0.0)
        dust_density_data = np.where(dust_density_data >= 0.0, dust_density_data, 0.0)
        return Field(
            data=dust_density_data,
            quantity=gas_density.quantity,
            axis_order=gas_density.axis_order,
        )

    def _settling_cell_profile(
        self, component: DustComponent, local_bin_idx: int, gas_density: Field,
    ) -> tuple[Quantity, np.ndarray]:
        """Return a grain's amplitude and log cell-averaged Gaussian.

        The Gaussian is integrated over spherical cell volumes, with amplitude
        and scale height fixed within each native radial/azimuthal column cell.
        Separate factors retain tiny integrated tails for annular normalization.
        The component mask is cell-wise and is applied by the caller.

        Parameters
        ----------
        component : DustComponent
            Grain distribution and settling parameters.
        local_bin_idx : int
            Grain index within the component.
        gas_density : Field
            Native spherical gas density supplying midplane properties.

        Returns
        -------
        prefactor : Quantity
            Unmasked midplane density [mass / volume], broadcast to native cells.
        log_average : ndarray
            Logarithm of the dimensionless Gaussian volume average, on native
            spherical cells in ``(r, theta, phi)`` order.
        """
        prefactor, dust_height = self._settling_column_properties(
            component, local_bin_idx, gas_density,
        )
        log_average = _spherical_gaussian_log_average(
            self.parent.mesh.edges("r").to("cm").magnitude,
            self.parent.mesh.edges("theta").to("radian").magnitude,
            dust_height.to("cm").magnitude,
        )
        return Quantity(
            np.broadcast_to(prefactor.to("g/cm^3").magnitude[:, None, :], log_average.shape),
            "g/cm^3",
        ), log_average

    def _settling_column_properties(
        self, component: DustComponent, local_bin_idx: int, gas_density: Field,
    ) -> tuple[Quantity, Quantity]:
        """Return grain midplane density and scale height on native columns.

        Both arrays have shape ``(nr, nphi)``. The global midplane, gas
        reference and Epstein settling prescription define the continuous
        profile before its integration over cells.
        """
        mesh = self.parent.mesh
        grain_size = component.distribution.bin_centers[local_bin_idx]
        mass_fraction = component.distribution.mass_fractions[local_bin_idx]
        
        has_temperature = 'temperature' in self.parent.gas
        has_pressure = 'pressure' in self.parent.gas
        if not has_temperature and not has_pressure:
            raise ValueError("Settling mode requires temperature or pressure field")
        gas_temp = self.parent.gas['temperature'].data if has_temperature else None
        gas_pressure = self.parent.gas['pressure'].data if has_pressure else None
        
        logger.debug(
            f"Computing dust settling directly in spherical coordinates for "
            f"component {component.component_index} bin {local_bin_idx} (mass-conserving)"
        )
        
        # Midplane properties are constant through each native theta column.
        theta = mesh.centers('theta')
        
        # Find midplane index (theta closest to pi/2)
        theta_mid_idx = np.argmin(np.abs(theta.to("radian") - np.pi/2))
        
        # Get midplane properties
        rho_g0 = gas_density.data[:, theta_mid_idx, :]
        T0 = gas_temp[:, theta_mid_idx, :] if gas_temp is not None else None
        P0 = gas_pressure[:, theta_mid_idx, :] if gas_pressure is not None else None

        mu = component.mean_molecular_weight
        if T0 is None:
            T0 = (mu * m_H / k_B) * (P0 / rho_g0)
        
        # Get stellar mass
        M_star = units('solar_mass')
        if hasattr(self.parent, 'variables') and 'mstar' in self.parent.variables:
            M_star = self.parent.variables['mstar']
        
        # Compute Keplerian frequency at r
        r = self.parent.mesh.centers('r')
        Omega_K_1d = keplerian_frequency(r, M_star)
        Omega_K = Omega_K_1d[:, None]
        
        # Sound speed at midplane
        c_s0 = np.sqrt(k_B * T0 / (mu * m_H))
        
        # Gas scale height at midplane: H_g = c_s / Omega_K
        # Convert to same units as r for consistent length units
        H_g0 = (c_s0 / Omega_K).to(r.units)
        
        # Stokes number at midplane
        St0 = stokes_number(
            grain_size=grain_size,
            gas_density=rho_g0,
            gas_temperature=T0,
            keplerian_freq=Omega_K,
            grain_density=component.grain_density,
            mean_molecular_weight=mu,
        )
        
        # Dust scale height from midplane Stokes number
        delta = component.delta
        H_d0 = H_g0 * np.sqrt(delta / (St0 + delta))
        
        dust_to_gas_local = component.dust_to_gas_ratio * mass_fraction
        with np.errstate(divide="ignore", invalid="ignore"):
            prefactor = rho_g0 * dust_to_gas_local * (H_g0 / H_d0)
        return prefactor, H_d0
        
    def _get_total_dust_density(self) -> Field:
        """Get total dust density summed over all size bins.
        
        Returns:
            Field containing total dust density
        """
        # Sum over all bins (works for both proportional and settling modes)
        total_data = None
        first_field = None
        for i in range(self.nbin):
            bin_density = self._compute_bin_density(i)
            if total_data is None:
                total_data = bin_density.data.copy()
                first_field = bin_density
            else:
                total_data = total_data + bin_density.data
        
        return Field(
            data=total_data,
            quantity=first_field.quantity,
            axis_order=first_field.axis_order,
        )
            
    def register(self, name: str, field: Field) -> None:
        """Register a dust field (for explicitly stored dust fields).
        
        Args:
            name: Field name
            field: Field object
        """
        from .utils import validate_field_against_mesh, field_data_as_order

        validate_field_against_mesh(field, self.mesh)  # type: ignore[arg-type]
        target = self.mesh.axis_names()  # type: ignore[union-attr]
        if field.axis_order != target:
            logger.warning(
                "dust.register canonicalizing field '%s' axis_order %s -> %s",
                name,
                field.axis_order,
                target,
            )
        data = field_data_as_order(field, target)
        self._dust_fields[name] = Field(
            quantity=field.quantity,
            data=data,
            axis_order=target,
            attrs=field.attrs,
        )
        
    def __getitem__(self, key: str):
        """Access dust fields or bins by name.
        
        Args:
            key: Field name ('density' for total) or bin name ('bin_0', 'bin_1', etc.)
            
        Returns:
            Field object for 'density', or DustBin object for 'bin_X'
        """
        # Special case: total density
        if key == 'density':
            return self._get_total_dust_density()
            
        # Check for bin access
        if key.startswith("bin_") or key.startswith("bin"):
            # Normalize key (handle both 'bin_0' and 'bin0')
            if "_" not in key:
                key = f"bin_{key[3:]}"
            if key in self._bins:
                return self._bins[key]
                
        # Check explicit fields
        if key in self._dust_fields:
            return self._dust_fields[key]
            
        # Check lazy builders
        if key in self._lazy_builders:
            field = self._lazy_builders[key]()
            self._dust_fields[key] = field
            return field
                
        raise KeyError(f"Dust field or bin '{key}' not found")
        
    def __contains__(self, key: str) -> bool:
        """Check if a dust field or bin exists."""
        if key == 'density' and self.distribution is not None:
            return True
        if key.startswith("bin"):
            # Normalize key
            if "_" not in key:
                key = f"bin_{key[3:]}"
            return key in self._bins
        return (
            key in self._dust_fields or
            key in self._lazy_builders
        )
        
    def keys(self):
        """Return available dust field and bin names."""
        keys = list(self._dust_fields.keys()) + list(self._lazy_builders.keys())
        if self.distribution is not None:
            keys.append('density')  # Total density always available
            keys.extend(self._bins.keys())
        return keys
        
    @property
    def bins(self) -> Dict[str, DustBin]:
        """Get all dust bins as a dictionary."""
        return self._bins
        
    @property
    def nbin(self) -> int:
        """Total number of dust bins across all components."""
        return len(self._global_bins)
        
    def __repr__(self) -> str:
        if self.distribution is None:
            return "Dust(no distribution set)"
        repr_str = (
            f"Dust(nbin={self.nbin}, "
            f"size_range={self.distribution.amin} to {self.distribution.amax}, "
            f"dust/gas={self.dust_to_gas_ratio:.3e}, mode='{self.mode}'"
        )
        if self.mode == 'settling' and self.alpha is not None:
            repr_str += f", alpha={self.alpha}"
        repr_str += ")"
        return repr_str
        
    def __str__(self) -> str:
        return self.__repr__()
