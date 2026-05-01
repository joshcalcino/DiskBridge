"""Partitioned UV radiation products for RADMC-3D mean intensities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np

from diskbridge._units import Quantity, units


H_CGS = units("h").to("erg*s").magnitude
C_CGS = units("c").to("cm/s").magnitude

DRAINE_BROAD_NM = (91.2, 206.7)
CO_PDES_NM = (91.2, 205.0)


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
UV_PRODUCT_MERGED_FIELD_NAMES = UV_PRODUCT_FIELD_NAMES + ("F_CO_pdes_photon",)


@dataclass(frozen=True)
class UVRuntimeMode:
    """Validated UV product runtime settings.

    Parameters
    ----------
    products_enabled : bool
        Whether process-specific UV products are enabled.
    mode : str
        UV product computation mode.
    fallback : str
        Fallback field outside product-measured regions.
    register_measured_mask : bool
        Whether segmented merging registers a measured-product mask.
    use_hard_directional_weights : bool
        Whether GOW17 directional shielding should use hard UV weights.
    use_physical_pdes_flux : bool
        Whether CO photodesorption should use physical photon flux.
    stellar_fraction_threshold : float
        Threshold for stellar/accretion UV dominance in segmented logic.
    """

    products_enabled: bool
    mode: str
    fallback: str
    register_measured_mask: bool
    use_hard_directional_weights: bool
    use_physical_pdes_flux: bool
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
    lookup_name = "G_CO_pdes" if name == "F_CO_pdes_photon" else name
    return integrate_draine_reference(_spec_by_name(lookup_name, specs), isrf_path=isrf_path)


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
    energy_weights = build_partition_weights(freq, lam_nm, partitions, weighting="energy")
    photon_weights = build_partition_weights(freq, lam_nm, partitions, weighting="photon")

    energy_parts = integrate_partitioned(
        Jnu_flat,
        energy_weights,
        chunk_ncells=chunk_ncells,
    )
    photon_parts = integrate_partitioned(
        Jnu_flat,
        photon_weights,
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
        products["F_CO_pdes_photon"] = Quantity(
            np.sum(photon_parts[pdes_idx, :], axis=0).reshape(mesh_shape, order="F"),
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


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 hash for a file.

    Parameters
    ----------
    path : str or pathlib.Path
        File path to hash.

    Returns
    -------
    str
        SHA256 hash string.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    chem_cfg = cfg.get("chemistry", {}) if isinstance(cfg.get("chemistry", {}), dict) else {}
    gow_cfg = chem_cfg.get("gow17", {}) if isinstance(chem_cfg.get("gow17", {}), dict) else {}

    products_enabled = bool(uv_cfg.get("enabled", True))
    mode = str(uv_cfg.get("mode", "disc_segment_only"))
    fallback = str(uv_cfg.get("fallback_outside_disc_segment", "chi_broad"))
    register_mask = bool(uv_cfg.get("register_measured_mask", True))
    threshold = float(uv_cfg.get("stellar_fraction_threshold", 0.01))

    use_uv_products = bool(gow_cfg.get("use_uv_products", products_enabled))
    hard_weights = bool(gow_cfg.get("hard_uv_for_directional_shielding", products_enabled))
    use_physical_flux = bool(gow_cfg.get("use_physical_co_pdes_photon_flux", products_enabled))
    chi_is_incident = bool(gow_cfg.get("chi_is_incident", False))

    if products_enabled:
        if not use_uv_products:
            raise ValueError("radmc3d.uv_products.enabled requires chemistry.gow17.use_uv_products=true")
        if not hard_weights:
            raise ValueError(
                "radmc3d.uv_products.enabled requires "
                "chemistry.gow17.hard_uv_for_directional_shielding=true"
            )
        if not use_physical_flux:
            raise ValueError(
                "radmc3d.uv_products.enabled requires "
                "chemistry.gow17.use_physical_co_pdes_photon_flux=true"
            )
    if use_uv_products and not products_enabled:
        raise ValueError("chemistry.gow17.use_uv_products requires radmc3d.uv_products.enabled=true")
    if chi_is_incident and products_enabled:
        raise ValueError("chemistry.gow17.chi_is_incident=true is incompatible with UV products")
    if chi_is_incident and segmented:
        raise ValueError("chemistry.gow17.chi_is_incident=true is incompatible with segmented RT")
    if mode == "disc_segment_only" and fallback != "chi_broad":
        raise ValueError("disc_segment_only UV products require fallback_outside_disc_segment='chi_broad'")
    if threshold <= 0.0:
        raise ValueError("radmc3d.uv_products.stellar_fraction_threshold must be positive")

    return UVRuntimeMode(
        products_enabled=products_enabled,
        mode=mode,
        fallback=fallback,
        register_measured_mask=register_mask,
        use_hard_directional_weights=hard_weights,
        use_physical_pdes_flux=use_physical_flux,
        stellar_fraction_threshold=threshold,
    )
