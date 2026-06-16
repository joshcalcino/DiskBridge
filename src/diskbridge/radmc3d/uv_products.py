"""Partitioned UV radiation products for RADMC-3D mean intensities."""

# db-keywords: uv-products, disk-mask, config, units, radmc3d, field, serialization, paths
# db-role: canonical
# db-scope: package
# db-purpose: Partitioned UV radiation products for RADMC-3D mean intensities.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np

from diskbridge._units import Quantity, units
from diskbridge.utils import sha256_file as file_sha256


H_CGS = units("h").to("erg*s").magnitude
C_CGS = units("c").to("cm/s").magnitude

DRAINE_BROAD_NM = (91.2, 206.7)
CO_PDES_NM = (91.2, 205.0)
UV_BAND_INTEGRATION_NQUAD = 16


@dataclass(frozen=True)
class UVBand:
    """Wavelength interval and weighting type for a UV product.

    Parameters
    ----------
    name : str
        Band identifier.
    lam_min_nm : float
        Lower wavelength edge in nm.
    lam_max_nm : float
        Upper wavelength edge in nm.
    weight : {"energy", "photon"}
        Integration weighting for the band.
    """

    name: str
    lam_min_nm: float
    lam_max_nm: float
    weight: str


@dataclass(frozen=True)
class UVPartition:
    """Non-overlapping wavelength partition used for UV products.

    Parameters
    ----------
    name : str
        Partition identifier.
    lam_min_nm : float
        Lower wavelength edge in nm.
    lam_max_nm : float
        Upper wavelength edge in nm.
    """

    name: str
    lam_min_nm: float
    lam_max_nm: float


@dataclass(frozen=True)
class UVProductSpec:
    """Definition of a stored UV product.

    Parameters
    ----------
    field_name : str
        Output field name.
    band : UVBand
        Wavelength band and weighting definition.
    normalize_to : str, optional
        Reference normalization. The supported value is
        ``"draine_same_band"``.
    """

    field_name: str
    band: UVBand
    normalize_to: str = "draine_same_band"


DEFAULT_UV_PRODUCT_SPECS = (
    UVProductSpec("chi_broad", UVBand("broad", 91.2, 206.7, "energy")),
    UVProductSpec("G_CO_diss", UVBand("hard", 91.2, 111.8, "photon")),
    UVProductSpec("G_H2_diss", UVBand("hard", 91.2, 111.8, "photon")),
    UVProductSpec("G_C_ion", UVBand("c_ion", 91.2, 110.1, "photon")),
    UVProductSpec("G_CO_pdes", UVBand("co_pdes", 91.2, 205.0, "photon")),
)

UV_PRODUCT_FIELD_NAMES = tuple(spec.field_name for spec in DEFAULT_UV_PRODUCT_SPECS)
UV_PRODUCT_MERGED_FIELD_NAMES = UV_PRODUCT_FIELD_NAMES + (
    "F_CO_pdes_photon",
    "F_CO_pdes_photon_bands",
)


@dataclass(frozen=True)
class UVRuntimeMode:
    """Validated UV product runtime settings.

    Parameters
    ----------
    products_enabled : bool
        Whether process-specific UV products are enabled.
    mode : str
        UV product computation mode.
    outer_product_policy : str
        Explicit policy for product fields outside product-measured regions.
    register_measured_mask : bool
        Whether segmented merging registers a measured-product mask.
    stellar_fraction_threshold : float
        Threshold for stellar/accretion UV dominance in segmented logic.
    """

    products_enabled: bool
    mode: str
    outer_product_policy: str
    register_measured_mask: bool
    stellar_fraction_threshold: float


def uv_product_specs_from_config(cfg: dict | None) -> tuple[UVProductSpec, ...]:
    """Build UV product specifications from configuration.

    Parameters
    ----------
    cfg : dict or None
        ``[radmc3d.uv_products]`` configuration dictionary. Missing values use
        package defaults.

    Returns
    -------
    tuple of UVProductSpec
        Active UV product specifications.
    """
    cfg = {} if cfg is None else dict(cfg)
    band_cfg = cfg.get("bands", {}) if isinstance(cfg.get("bands", {}), dict) else {}
    weight_cfg = cfg.get("weights", {}) if isinstance(cfg.get("weights", {}), dict) else {}

    defaults = {spec.field_name: spec for spec in DEFAULT_UV_PRODUCT_SPECS}
    band_keys = {
        "chi_broad": "broad",
        "G_CO_diss": "hard",
        "G_H2_diss": "hard",
        "G_C_ion": "c_ion",
        "G_CO_pdes": "co_pdes",
    }

    specs: list[UVProductSpec] = []
    for field_name in UV_PRODUCT_FIELD_NAMES:
        default = defaults[field_name]
        band_key = band_keys[field_name]
        edge_val = band_cfg.get(f"{band_key}_nm", None)
        if edge_val is None:
            lam_min = float(default.band.lam_min_nm)
            lam_max = float(default.band.lam_max_nm)
        else:
            if len(edge_val) != 2:
                raise ValueError(f"UV product band {band_key}_nm must have two edges")
            lam_min = float(edge_val[0])
            lam_max = float(edge_val[1])
        weight = str(weight_cfg.get(band_key, default.band.weight)).lower()
        specs.append(
            UVProductSpec(
                field_name,
                UVBand(default.band.name, lam_min, lam_max, weight),
            )
        )
    return tuple(specs)


def uv_product_edges_from_specs(specs=DEFAULT_UV_PRODUCT_SPECS) -> np.ndarray:
    """Return sorted wavelength edges from UV product specifications.

    Parameters
    ----------
    specs : sequence of UVProductSpec, optional
        Product specifications.

    Returns
    -------
    ndarray
        Sorted unique wavelength edges in nm.
    """
    edges = sorted(
        {float(spec.band.lam_min_nm) for spec in specs}
        | {float(spec.band.lam_max_nm) for spec in specs}
    )
    return np.asarray(edges, dtype=np.float64)


UV_PRODUCT_EDGES_NM = uv_product_edges_from_specs(DEFAULT_UV_PRODUCT_SPECS)


def default_isrf_path() -> Path:
    """Return the packaged Draine reference spectrum path.

    Returns
    -------
    pathlib.Path
        Path to ``data/ISRF.dat``.
    """
    return Path(__file__).resolve().parents[3] / "data" / "ISRF.dat"


def load_draine_reference(isrf_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load the Draine reference photon spectrum.

    Parameters
    ----------
    isrf_path : str or pathlib.Path or None, optional
        Path to the ISRF table. If omitted, the packaged ``ISRF.dat`` is used.

    Returns
    -------
    lam_nm, photon_flux_nm : ndarray
        Wavelengths in nm and photon flux density in
        photons cm^-2 s^-1 nm^-1.
    """
    path = default_isrf_path() if isrf_path is None else Path(isrf_path)
    data = np.loadtxt(str(path), comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid Draine reference table: {path}")
    lam_nm = np.asarray(data[:, 0], dtype=np.float64)
    photon_flux_nm = np.asarray(data[:, 1], dtype=np.float64)
    order = np.argsort(lam_nm)
    return lam_nm[order], photon_flux_nm[order]


def make_uv_partitions(product_specs=DEFAULT_UV_PRODUCT_SPECS) -> list[UVPartition]:
    """Return sorted non-overlapping intervals from product band edges.

    Parameters
    ----------
    product_specs : sequence of UVProductSpec, optional
        Product specifications whose band edges define the partition.

    Returns
    -------
    list of UVPartition
        Non-overlapping wavelength intervals sorted by wavelength.
    """
    edges = uv_product_edges_from_specs(product_specs).tolist()
    return [
        UVPartition(f"P{i}", edges[i], edges[i + 1])
        for i in range(len(edges) - 1)
    ]


def _trapezoid_coefficients(x: np.ndarray) -> np.ndarray:
    coeff = np.zeros_like(x, dtype=np.float64)
    if x.size < 2:
        return coeff
    dx = np.diff(x)
    coeff[0] = 0.5 * dx[0]
    coeff[-1] = 0.5 * dx[-1]
    if x.size > 2:
        coeff[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return coeff


def build_partition_weights(
    freq_hz,
    lam_nm: np.ndarray,
    partitions: list[UVPartition],
    *,
    weighting: str,
) -> np.ndarray:
    """Build Jnu quadrature weights for UV partitions.

    Parameters
    ----------
    freq_hz : Quantity or ndarray
        Frequency grid in Hz.
    lam_nm : ndarray
        Wavelength grid in nm matching ``freq_hz``.
    partitions : list of UVPartition
        Non-overlapping wavelength intervals.
    weighting : {"energy", "photon"}
        ``"energy"`` returns weights for energy density in erg cm^-3.
        ``"photon"`` returns weights for photon flux in cm^-2 s^-1.

    Returns
    -------
    ndarray
        Weight matrix with shape ``(npartitions, nwav)``.
    """
    if hasattr(freq_hz, "to"):
        freq = np.asarray(freq_hz.to("Hz").magnitude, dtype=np.float64)
    else:
        freq = np.asarray(freq_hz, dtype=np.float64)
    lam = np.asarray(lam_nm, dtype=np.float64)
    if freq.ndim != 1 or lam.ndim != 1 or freq.size != lam.size:
        raise ValueError("freq_hz and lam_nm must be matching 1-D arrays")

    weighting = str(weighting).lower()
    if weighting not in {"energy", "photon"}:
        raise ValueError("weighting must be 'energy' or 'photon'")

    weights = np.zeros((len(partitions), freq.size), dtype=np.float64)
    for ipart, part in enumerate(partitions):
        mask = (
            (lam >= np.nextafter(float(part.lam_min_nm), -np.inf))
            & (lam <= np.nextafter(float(part.lam_max_nm), np.inf))
        )
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            raise ValueError(
                "Partition %.6g-%.6g nm has %d wavelength points; "
                "enforce UV product edges in the mcmono grid"
                % (float(part.lam_min_nm), float(part.lam_max_nm), int(idx.size))
            )
        order = idx[np.argsort(freq[idx])]
        coeff = _trapezoid_coefficients(freq[order])
        if weighting == "energy":
            local_weights = (4.0 * np.pi / C_CGS) * coeff
        else:
            local_weights = 4.0 * np.pi * coeff / (H_CGS * freq[order])
        weights[ipart, order] = local_weights
    return weights


def integrate_partitioned(
    Jnu_flat,
    weights: np.ndarray,
    *,
    chunk_ncells: int | None = None,
) -> np.ndarray:
    """Integrate partitioned UV products for all cells.

    Parameters
    ----------
    Jnu_flat : Quantity or ndarray
        Mean intensity with shape ``(ncells, nwav)``.
    weights : ndarray
        Weight matrix with shape ``(npartitions, nwav)``.
    chunk_ncells : int or None, optional
        Number of cells per chunk. If omitted, all cells are integrated at
        once.

    Returns
    -------
    ndarray
        Partition integrals with shape ``(npartitions, ncells)``.
    """
    if hasattr(Jnu_flat, "to"):
        J = np.asarray(Jnu_flat.to("erg/(s*cm^2*Hz*sr)").magnitude, dtype=np.float64)
    else:
        J = np.asarray(Jnu_flat, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)
    if J.ndim != 2 or W.ndim != 2 or J.shape[1] != W.shape[1]:
        raise ValueError("Jnu_flat must have shape (ncells, nwav) matching weights")

    ncells = int(J.shape[0])
    result = np.empty((W.shape[0], ncells), dtype=np.float64)
    if chunk_ncells is None or int(chunk_ncells) <= 0 or int(chunk_ncells) >= ncells:
        result[:, :] = W @ J.T
        return result

    chunk = int(chunk_ncells)
    for start in range(0, ncells, chunk):
        stop = min(start + chunk, ncells)
        result[:, start:stop] = W @ J[start:stop, :].T
    return result


def _loglinear_interp_strict(lam_sample, J_sample, lam_quad):
    """
    Interpolate J_nu onto lam_quad.

    Positive intervals use linear interpolation in log(J_nu)-log(lambda)
    space. Intervals touching exact zero use linear interpolation in J_nu so
    optically thick cells with no sampled UV photons remain valid instead of
    forcing an artificial floor.

    lam_sample: shape (nwave,)
    J_sample:   shape (ncell, nwave)
    lam_quad:   shape (nquad,)
    """
    if np.any(J_sample < 0.0) or np.any(~np.isfinite(J_sample)):
        raise ValueError(
            "UV mean intensity contains negative or non-finite values. "
            "Increase mcmono photons or check RADMC-3D output."
        )
    x = np.log(lam_sample)
    xq = np.log(lam_quad)
    idx = np.searchsorted(x, xq) - 1
    idx = np.clip(idx, 0, len(x) - 2)
    w = (xq - x[idx]) / (x[idx + 1] - x[idx])

    left = J_sample[:, idx]
    right = J_sample[:, idx + 1]
    linear = (1.0 - w[None, :]) * left + w[None, :] * right
    out = linear

    positive = (left > 0.0) & (right > 0.0)
    logJq = (1.0 - w[None, :]) * np.log(left, where=positive, out=np.zeros_like(left)) + (
        w[None, :] * np.log(right, where=positive, out=np.zeros_like(right))
    )
    out[positive] = np.exp(logJq[positive])
    return out


def integrate_partitions_loglinear(
    freq_hz,
    lam_nm: np.ndarray,
    Jnu_flat,
    partitions: list[UVPartition],
    *,
    weighting: str,
    nquad: int = UV_BAND_INTEGRATION_NQUAD,
    chunk_ncells: int | None = None,
) -> np.ndarray:
    """Integrate UV partitions using log-linear reconstructed J_nu."""
    if hasattr(freq_hz, "to"):
        freq = np.asarray(freq_hz.to("Hz").magnitude, dtype=np.float64)
    else:
        freq = np.asarray(freq_hz, dtype=np.float64)
    lam = np.asarray(lam_nm, dtype=np.float64)
    if hasattr(Jnu_flat, "to"):
        J = np.asarray(Jnu_flat.to("erg/(s*cm^2*Hz*sr)").magnitude, dtype=np.float64)
    else:
        J = np.asarray(Jnu_flat, dtype=np.float64)
    if freq.ndim != 1 or lam.ndim != 1 or freq.size != lam.size:
        raise ValueError("freq_hz and lam_nm must be matching 1-D arrays")
    if J.ndim != 2 or J.shape[1] != lam.size:
        raise ValueError("Jnu_flat must have shape (ncells, nwav)")

    weighting = str(weighting).lower()
    if weighting not in {"energy", "photon"}:
        raise ValueError("weighting must be 'energy' or 'photon'")
    nquad = int(nquad)
    if nquad < 2:
        raise ValueError("nquad must be at least 2")

    order_lam = np.argsort(lam)
    lam_sample = lam[order_lam]
    J_sample = J[:, order_lam]
    ncells = int(J.shape[0])
    result = np.empty((len(partitions), ncells), dtype=np.float64)

    if chunk_ncells is None or int(chunk_ncells) <= 0 or int(chunk_ncells) >= ncells:
        chunks = [(0, ncells)]
    else:
        chunk = int(chunk_ncells)
        chunks = [(start, min(start + chunk, ncells)) for start in range(0, ncells, chunk)]

    for ipart, part in enumerate(partitions):
        lam_quad = np.linspace(
            float(part.lam_min_nm),
            float(part.lam_max_nm),
            nquad,
            dtype=np.float64,
        )
        nu_quad = C_CGS / (lam_quad * 1.0e-7)
        order_nu = np.argsort(nu_quad)
        nu_sorted = nu_quad[order_nu]
        for start, stop in chunks:
            J_quad = _loglinear_interp_strict(
                lam_sample,
                J_sample[start:stop, :],
                lam_quad,
            )
            J_sorted = J_quad[:, order_nu]
            if weighting == "energy":
                integrand = (4.0 * np.pi / C_CGS) * J_sorted
            else:
                integrand = 4.0 * np.pi * J_sorted / (H_CGS * nu_sorted[None, :])
            result[ipart, start:stop] = np.trapezoid(
                integrand,
                nu_sorted,
                axis=1,
            )
    return result


def _interp_with_edges(
    lam_nm: np.ndarray,
    values: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (lam_nm > lo) & (lam_nm < hi)
    lam = np.concatenate(([lo], lam_nm[mask], [hi])).astype(np.float64)
    val = np.interp(lam, lam_nm, values).astype(np.float64)
    return lam, val


def integrate_draine_partitions(
    draine_spectrum: tuple[np.ndarray, np.ndarray],
    partitions: list[UVPartition],
    *,
    weighting: str,
) -> np.ndarray:
    """Compute Draine reference integrals for UV partitions.

    Parameters
    ----------
    draine_spectrum : tuple of ndarray
        ``(lam_nm, photon_flux_nm)`` reference spectrum.
    partitions : list of UVPartition
        UV partition intervals.
    weighting : {"energy", "photon"}
        Reference weighting to compute.

    Returns
    -------
    ndarray
        Reference integral per partition. Energy-weighted values are energy
        densities in erg cm^-3; photon-weighted values are photon fluxes in
        cm^-2 s^-1.
    """
    lam_nm, photon_flux_nm = draine_spectrum
    weighting = str(weighting).lower()
    refs = np.empty(len(partitions), dtype=np.float64)
    for ipart, part in enumerate(partitions):
        lam, phot = _interp_with_edges(
            np.asarray(lam_nm, dtype=np.float64),
            np.asarray(photon_flux_nm, dtype=np.float64),
            float(part.lam_min_nm),
            float(part.lam_max_nm),
        )
        if weighting == "photon":
            refs[ipart] = float(np.trapezoid(phot, lam))
        elif weighting == "energy":
            energy_flux_nm = phot * H_CGS * C_CGS / (lam * 1.0e-7)
            refs[ipart] = float(np.trapezoid(energy_flux_nm, lam) / C_CGS)
        else:
            raise ValueError("weighting must be 'energy' or 'photon'")
    return refs


def _partition_indices_for_band(
    partitions: list[UVPartition],
    lam_min_nm: float,
    lam_max_nm: float,
) -> list[int]:
    indices = []
    for i, part in enumerate(partitions):
        if (
            part.lam_min_nm >= np.nextafter(float(lam_min_nm), -np.inf)
            and part.lam_max_nm <= np.nextafter(float(lam_max_nm), np.inf)
        ):
            indices.append(i)
    if not indices:
        raise ValueError(f"No UV partitions cover {lam_min_nm}-{lam_max_nm} nm")
    return indices


def _spec_by_name(name: str, specs=DEFAULT_UV_PRODUCT_SPECS) -> UVProductSpec:
    for spec in specs:
        if spec.field_name == name:
            return spec
    raise KeyError(f"Unknown UV product {name!r}")


def _reference_product_name(name: str) -> str:
    """Map stored physical fields onto their normalized UV product spec."""
    if name in {"F_CO_pdes_photon", "F_CO_pdes_photon_bands"}:
        return "G_CO_pdes"
    return name


def integrate_draine_reference(
    spec: UVProductSpec,
    *,
    isrf_path: str | Path | None = None,
) -> Quantity:
    """Integrate the Draine reference for one UV product.

    Parameters
    ----------
    spec : UVProductSpec
        Product specification to integrate.
    isrf_path : str or pathlib.Path or None, optional
        Draine reference spectrum path.

    Returns
    -------
    Quantity
        Energy density for energy-weighted products, or photon flux for
        photon-weighted products.
    """
    draine = load_draine_reference(isrf_path)
    partition = [UVPartition(spec.band.name, spec.band.lam_min_nm, spec.band.lam_max_nm)]
    refs = integrate_draine_partitions(draine, partition, weighting=spec.band.weight)
    unit = "erg/cm^3" if spec.band.weight == "energy" else "1/(cm^2 s)"
    return Quantity(float(refs[0]), unit)


def draine_reference_for_product(
    name: str,
    specs=DEFAULT_UV_PRODUCT_SPECS,
    *,
    isrf_path: str | Path | None = None,
) -> Quantity:
    """Return the Draine reference for a UV product.

    Parameters
    ----------
    name : str
        Product name. ``"F_CO_pdes_photon"`` returns the physical Draine
        photon flux over the CO photodesorption band.
    specs : sequence of UVProductSpec, optional
        Product specifications.
    isrf_path : str or pathlib.Path or None, optional
        Draine reference spectrum path.

    Returns
    -------
    Quantity
        Draine reference quantity.
    """
    lookup_name = _reference_product_name(name)
    return integrate_draine_reference(_spec_by_name(lookup_name, specs), isrf_path=isrf_path)


def partitions_for_product(
    name: str,
    specs=DEFAULT_UV_PRODUCT_SPECS,
) -> list[UVPartition]:
    """Return UV partitions covered by a product's wavelength band."""
    spec = _spec_by_name(_reference_product_name(name), specs)
    partitions = make_uv_partitions(specs)
    idx = _partition_indices_for_band(
        partitions,
        spec.band.lam_min_nm,
        spec.band.lam_max_nm,
    )
    return [partitions[i] for i in idx]


def draine_references_for_product_partitions(
    name: str,
    specs=DEFAULT_UV_PRODUCT_SPECS,
    *,
    isrf_path: str | Path | None = None,
) -> Quantity:
    """Return Draine reference integrals for each partition in a product band."""
    spec = _spec_by_name(_reference_product_name(name), specs)
    partitions = make_uv_partitions(specs)
    idx = _partition_indices_for_band(
        partitions,
        spec.band.lam_min_nm,
        spec.band.lam_max_nm,
    )
    draine = load_draine_reference(isrf_path)
    refs = integrate_draine_partitions(draine, partitions, weighting=spec.band.weight)
    unit = "erg/cm^3" if spec.band.weight == "energy" else "1/(cm^2 s)"
    return Quantity(np.asarray(refs[idx], dtype=np.float64), unit)


def _validate_frequency_coverage(freq_hz, lam_min_nm: float, lam_max_nm: float) -> None:
    if hasattr(freq_hz, "to"):
        freq = np.asarray(freq_hz.to("Hz").magnitude, dtype=np.float64)
    else:
        freq = np.asarray(freq_hz, dtype=np.float64)
    lam = C_CGS / freq * 1.0e7
    tol = 1.0e-8 * max(1.0, abs(float(lam_min_nm)), abs(float(lam_max_nm)))
    if np.nanmin(lam) > float(lam_min_nm) + tol or np.nanmax(lam) < float(lam_max_nm) - tol:
        raise ValueError(
            "Mean-intensity wavelength grid does not cover canonical UV band "
            f"{lam_min_nm:.3g}-{lam_max_nm:.3g} nm; got "
            f"{float(np.nanmin(lam)):.3g}-{float(np.nanmax(lam)):.3g} nm"
        )


def compute_chi_broad(
    freq_hz,
    Jnu_flat,
    mesh_shape: tuple[int, int, int],
    specs=DEFAULT_UV_PRODUCT_SPECS,
    isrf_path: str | Path | None = None,
    *,
    chunk_ncells: int | None = None,
) -> Quantity:
    """Compute canonical broad Draine-normalized UV field.

    Parameters
    ----------
    freq_hz : Quantity or ndarray
        Frequency grid in Hz with shape ``(nwav,)``.
    Jnu_flat : Quantity or ndarray
        Mean intensity with shape ``(ncells, nwav)``.
    mesh_shape : tuple of int
        Output mesh shape.
    specs : sequence of UVProductSpec, optional
        Product specifications containing ``chi_broad``.
    isrf_path : str or pathlib.Path or None, optional
        Draine reference spectrum path.
    chunk_ncells : int or None, optional
        Optional cell chunk size.

    Returns
    -------
    Quantity
        Canonical broad UV field in Draine units.
    """
    spec = _spec_by_name("chi_broad", specs)
    _validate_frequency_coverage(freq_hz, spec.band.lam_min_nm, spec.band.lam_max_nm)
    return compute_uv_products(
        freq_hz,
        Jnu_flat,
        mesh_shape,
        specs=(spec,),
        isrf_path=isrf_path,
        chunk_ncells=chunk_ncells,
    )["chi_broad"]


# db-keywords: uv-products, photodesorption, radmc3d
# db-role: canonical
def compute_uv_products(
    freq_hz,
    Jnu_flat,
    mesh_shape: tuple[int, int, int],
    specs=DEFAULT_UV_PRODUCT_SPECS,
    isrf_path: str | Path | None = None,
    *,
    chunk_ncells: int | None = None,
) -> dict[str, Quantity]:
    """Compute Draine-normalized UV products from RADMC mean intensity.

    Parameters
    ----------
    freq_hz : Quantity or ndarray
        Frequency grid in Hz with shape ``(nwav,)``.
    Jnu_flat : Quantity or ndarray
        Mean intensity with shape ``(ncells, nwav)``.
    mesh_shape : tuple of int
        Output mesh shape used to reshape product fields.
    specs : sequence of UVProductSpec, optional
        Product specifications to compute.
    isrf_path : str or pathlib.Path or None, optional
        Draine reference spectrum path.
    chunk_ncells : int or None, optional
        Optional cell chunk size for matrix multiplication.

    Returns
    -------
    dict
        Product fields. Dimensionless products are returned as
        ``dimensionless`` quantities and ``F_CO_pdes_photon`` as
        ``1/(cm^2 s)``.
    """
    if hasattr(freq_hz, "to"):
        freq = np.asarray(freq_hz.to("Hz").magnitude, dtype=np.float64)
    else:
        freq = np.asarray(freq_hz, dtype=np.float64)
    lam_nm = C_CGS / freq * 1.0e7
    for spec in specs:
        _validate_frequency_coverage(freq, spec.band.lam_min_nm, spec.band.lam_max_nm)

    partitions = make_uv_partitions(specs)
    energy_parts = integrate_partitions_loglinear(
        freq,
        lam_nm,
        Jnu_flat,
        partitions,
        weighting="energy",
        chunk_ncells=chunk_ncells,
    )
    photon_parts = integrate_partitions_loglinear(
        freq,
        lam_nm,
        Jnu_flat,
        partitions,
        weighting="photon",
        chunk_ncells=chunk_ncells,
    )

    draine = load_draine_reference(isrf_path)
    draine_energy = integrate_draine_partitions(draine, partitions, weighting="energy")
    draine_photon = integrate_draine_partitions(draine, partitions, weighting="photon")

    products: dict[str, Quantity] = {}
    for spec in specs:
        idx = _partition_indices_for_band(
            partitions,
            spec.band.lam_min_nm,
            spec.band.lam_max_nm,
        )
        if spec.band.weight == "energy":
            numerator = np.sum(energy_parts[idx, :], axis=0)
            reference = float(np.sum(draine_energy[idx]))
        elif spec.band.weight == "photon":
            numerator = np.sum(photon_parts[idx, :], axis=0)
            reference = float(np.sum(draine_photon[idx]))
        else:
            raise ValueError(f"Unsupported UV weighting: {spec.band.weight}")
        if reference <= 0.0:
            raise ValueError(f"Non-positive Draine reference for {spec.field_name}")
        products[spec.field_name] = Quantity(
            (numerator / reference).reshape(mesh_shape, order="F"),
            "dimensionless",
        )

    if any(spec.field_name == "G_CO_pdes" for spec in specs):
        pdes_spec = _spec_by_name("G_CO_pdes", specs)
        pdes_idx = _partition_indices_for_band(
            partitions,
            pdes_spec.band.lam_min_nm,
            pdes_spec.band.lam_max_nm,
        )
        pdes_band_fields = np.stack(
            [
                photon_parts[ipart, :].reshape(mesh_shape, order="F")
                for ipart in pdes_idx
            ],
            axis=0,
        )
        products["F_CO_pdes_photon"] = Quantity(
            np.sum(photon_parts[pdes_idx, :], axis=0).reshape(mesh_shape, order="F"),
            "1/(cm^2 s)",
        )
        products["F_CO_pdes_photon_bands"] = Quantity(
            pdes_band_fields,
            "1/(cm^2 s)",
        )
    return products


def uv_product_schema(specs=DEFAULT_UV_PRODUCT_SPECS) -> dict:
    """Return the active UV product schema.

    Returns
    -------
    dict
        JSON-serializable schema metadata for cache keys.
    """
    partitions = make_uv_partitions(specs)
    return {
        "products": [
            {
                "field_name": spec.field_name,
                "band": spec.band.name,
                "lam_min_nm": float(spec.band.lam_min_nm),
                "lam_max_nm": float(spec.band.lam_max_nm),
                "weight": spec.band.weight,
            }
            for spec in specs
        ],
        "partition_edges_nm": [float(partitions[0].lam_min_nm)]
        + [float(part.lam_max_nm) for part in partitions],
    }


def uv_product_schema_hash(specs=DEFAULT_UV_PRODUCT_SPECS) -> str:
    """Return a stable hash of the active UV product schema.

    Returns
    -------
    str
        SHA256 hash string.
    """
    import json

    payload = json.dumps(uv_product_schema(specs), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_uv_chemistry_config(
    cfg: dict,
    *,
    segmented: bool = False,
) -> UVRuntimeMode:
    """Validate UV-product and GOW17 chemistry configuration.

    Parameters
    ----------
    cfg : dict
        Full DiskBridge configuration dictionary.
    segmented : bool, optional
        Whether validation is for segmented RADMC-3D mode.

    Returns
    -------
    UVRuntimeMode
        Validated runtime UV settings.

    Raises
    ------
    ValueError
        If UV-product and GOW17 flags are inconsistent.
    """
    radmc_cfg = cfg.get("radmc3d", {}) if isinstance(cfg.get("radmc3d", {}), dict) else {}
    uv_cfg = radmc_cfg.get("uv_products", {}) if isinstance(radmc_cfg.get("uv_products", {}), dict) else {}
    products_enabled = bool(uv_cfg.get("enabled", True))
    mode = str(uv_cfg.get("mode", "disc_segment_only"))
    outer_policy = str(uv_cfg.get("outer_product_policy", "draine_equivalent"))
    register_mask = bool(uv_cfg.get("register_measured_mask", True))
    threshold = float(uv_cfg.get("stellar_fraction_threshold", 0.01))

    if mode == "disc_segment_only" and outer_policy != "draine_equivalent":
        raise ValueError(
            "disc_segment_only UV products require "
            "outer_product_policy='draine_equivalent'"
        )
    if threshold <= 0.0:
        raise ValueError("radmc3d.uv_products.stellar_fraction_threshold must be positive")

    return UVRuntimeMode(
        products_enabled=products_enabled,
        mode=mode,
        outer_product_policy=outer_policy,
        register_measured_mask=register_mask,
        stellar_fraction_threshold=threshold,
    )
