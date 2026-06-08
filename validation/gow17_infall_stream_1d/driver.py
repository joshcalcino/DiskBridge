# db-keywords: shielding, uv-products, photodesorption, gow17, gas-temperature, validation, units, radmc3d, model, mesh, field
# db-role: validation
# db-scope: validation
# db-purpose: Validation module for shielding, uv-products, photodesorption, gow17.

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from diskbridge._units import Quantity, units
from diskbridge._constants import (
    AU,
    G_GRAV,
    SOLAR_MASS,
    SOLAR_RADIUS,
    C_LIGHT,
    U_DRAINE,
    T_CMB,
)
from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel
from diskbridge.chemistry.shielding.angular_uv_weights import compute_star_uv_luminosity
from diskbridge.chemistry.shielding.angular_uv_weights import compute_star_uv_source_strength
from diskbridge.chemistry.models.gow17_timestep import Gow17TimeStepper
from diskbridge.radmc3d.uv_products import (
    DEFAULT_UV_PRODUCT_SPECS,
    draine_references_for_product_partitions,
    draine_reference_for_product,
    partitions_for_product,
)
import diskbridge._gow17 as _gow17

I_CO = _gow17.I_CO
I_CO_ICE = _gow17.I_CO_ICE
I_CHX = _gow17.I_CHX
I_HCOP = _gow17.I_HCOP
I_CP = _gow17.I_CP
I_H2 = _gow17.I_H2
XC_STD = _gow17.XC_STD
N_Y = _gow17.N_Y

RADIATION_MODE_DRAINE_SCALAR = "draine_scalar"
RADIATION_MODE_STELLAR_PRODUCTS = "stellar_products"
RADIATION_MODES = (RADIATION_MODE_DRAINE_SCALAR, RADIATION_MODE_STELLAR_PRODUCTS)


@dataclass(frozen=True)
class InfallStream1DConfig:
    n_cells: int = 512
    stream_length_au: float = 1000.0

    nH_list_cm3: tuple[float, ...] = (1e4, 1e5, 1e6, 1e7)

    mstar_msun: float = 1.0
    rstar_rsun: float = 2.0
    teff_K: float = 8000.0
    mdot_msun_yr: float = 0.0
    accretion_fill_factor: float = 0.01

    uv_lam_min_cm: float = 9.12e-6
    uv_lam_max_cm: float = 2.067e-5
    min_chi: float = 0.1
    radiation_mode: str = RADIATION_MODE_DRAINE_SCALAR

    r_face_start_au: float = 2.0e4
    r_face_stop_au: float = 100.0

    n_steps: int = 500
    movie_fps: int = 12
    movie_max_duration_s: float = 20.0
    make_movies: bool = True
    output_time_power: float = 3.0
    max_dlnchi: float = 0.05
    max_inner_steps: int = 100000

    init_relax: bool = True
    init_relax_n_steps: int = 16000
    init_relax_min_steps: int = 500
    init_relax_dt_frac: float = 1.0
    init_relax_rtol: float = 1e-8
    track_infall_equilibrium: bool = False
    equilibrium_astrochem_n_updates: int = 500
    equilibrium_shielding_max_iter: int = 1000
    equilibrium_shielding_reltol: float = 1e-6
    equilibrium_shielding_abstol: float = 1e-15
    equilibrium_astrochem_t_end_yr: float = 1.0e6

    evolve_energy: bool = False
    estimate_tdust: bool = False

    Tdust_const_K: float = 20.0
    Tgas_init_K: float = 50.0

    b_kms: float = 0.3
    ion_rate: str = "2e-16 s^-1"
    Zg: float = 1.0
    enable_co_phase: bool = False
    reltol: float = 1e-6
    abstol0: float = 1e-15
    mxsteps: int = 100000
    maxord: int = 5
    tolfac: float = 10.0
    userJac: bool = False
    verbose: bool = False

    shielding_outer_1d: str = "min"


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _build_stream_model(cfg: InfallStream1DConfig, *, nH_cm3: float) -> tuple[RadModel, np.ndarray]:
    m_H = units("m_H")

    L_cm = float(cfg.stream_length_au) * float(AU)
    x_edges_cm = np.linspace(0.0, L_cm, int(cfg.n_cells) + 1, dtype=float)
    x_cent_cm = 0.5 * (x_edges_cm[:-1] + x_edges_cm[1:])

    mesh = Mesh.cartesian(
        x=Axis(edges=Quantity(x_edges_cm, "cm")),
        y=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
        z=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
    )
    shape = mesh.shape

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    rho = (Quantity(np.full(shape, float(nH_cm3)), "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register(
        "density",
        Field(quantity="density", data=rho, axis_order=("x", "y", "z")),
    )

    sigma_d_per_H = 1.0 / (1.086 * 1.87e21)
    sigma = Quantity(np.full(shape, sigma_d_per_H), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")),
    )

    rad = RadModel(model)
    rad.nH = Quantity(np.full(shape, float(nH_cm3)), "cm^-3")

    NH_cent = float(nH_cm3) * x_cent_cm
    Av_cent = NH_cent / 1.87e21
    rad.Av = Quantity(Av_cent.reshape(shape), "dimensionless")

    rad.dust_temperature = Quantity(np.full(shape, float(cfg.Tdust_const_K)), "K")
    rad.gas_temperature = Quantity(np.full(shape, float(cfg.Tgas_init_K)), "K")

    rad.chi = Quantity(np.full(shape, 0.0), "dimensionless")

    return rad, x_cent_cm


def _star_uv_luminosity(cfg: InfallStream1DConfig) -> float:
    params = SimpleNamespace(
        rstar=Quantity(float(cfg.rstar_rsun) * float(SOLAR_RADIUS), "cm"),
        teff=Quantity(float(cfg.teff_K), "K"),
        mdot=float(cfg.mdot_msun_yr),
        mstar=Quantity(float(cfg.mstar_msun) * float(SOLAR_MASS), "g"),
        accretion_fill_factor=float(cfg.accretion_fill_factor),
    )
    return float(
        compute_star_uv_luminosity(
            params,
            float(cfg.uv_lam_min_cm),
            float(cfg.uv_lam_max_cm),
        )
    )


def _validate_radiation_mode(mode: str) -> str:
    mode_str = str(mode)
    if mode_str not in RADIATION_MODES:
        raise ValueError(f"radiation_mode must be one of {RADIATION_MODES}, got {mode_str!r}")
    return mode_str


def _star_uv_product_sources(cfg: InfallStream1DConfig) -> dict[str, float]:
    """Return stellar UV source strengths before geometric dilution.

    Dimensionless product entries are stored as luminosity divided by the
    relevant Draine reference.  Dividing by ``4*pi*r^2`` later gives the
    local Draine-normalized product.  ``F_CO_pdes_photon`` is stored as the
    physical photon luminosity and is diluted directly into photon flux.
    """
    params = SimpleNamespace(
        rstar=Quantity(float(cfg.rstar_rsun) * float(SOLAR_RADIUS), "cm"),
        teff=Quantity(float(cfg.teff_K), "K"),
        mdot=float(cfg.mdot_msun_yr),
        mstar=Quantity(float(cfg.mstar_msun) * float(SOLAR_MASS), "g"),
        accretion_fill_factor=float(cfg.accretion_fill_factor),
    )

    out: dict[str, float] = {
        "chi_broad": float(
            compute_star_uv_source_strength(
                params,
                float(cfg.uv_lam_min_cm),
                float(cfg.uv_lam_max_cm),
                weighting="energy",
            )
        )
        / (float(C_LIGHT) * float(U_DRAINE)),
    }

    spec_by_name = {spec.field_name: spec for spec in DEFAULT_UV_PRODUCT_SPECS}
    for name in ("G_CO_diss", "G_H2_diss", "G_C_ion", "G_CO_pdes"):
        spec = spec_by_name[name]
        photon_lum = float(
            compute_star_uv_source_strength(
                params,
                float(spec.band.lam_min_nm) * 1.0e-7,
                float(spec.band.lam_max_nm) * 1.0e-7,
                weighting=spec.band.weight,
            )
        )
        unit = "1/(cm^2 s)" if spec.band.weight == "photon" else "erg/cm^3"
        ref = float(draine_reference_for_product(name).to(unit).magnitude)
        if spec.band.weight == "photon":
            out[name] = photon_lum / ref
        else:
            out[name] = photon_lum / (float(C_LIGHT) * ref)

    pdes_spec = spec_by_name["G_CO_pdes"]
    out["F_CO_pdes_photon"] = float(
        compute_star_uv_source_strength(
            params,
            float(pdes_spec.band.lam_min_nm) * 1.0e-7,
            float(pdes_spec.band.lam_max_nm) * 1.0e-7,
            weighting="photon",
        )
    )
    out["F_CO_pdes_draine"] = float(
        draine_reference_for_product("F_CO_pdes_photon").to("1/(cm^2 s)").magnitude
    )
    pdes_partitions = partitions_for_product("F_CO_pdes_photon", DEFAULT_UV_PRODUCT_SPECS)
    out["F_CO_pdes_photon_bands"] = np.array(
        [
            float(
                compute_star_uv_source_strength(
                    params,
                    float(part.lam_min_nm) * 1.0e-7,
                    float(part.lam_max_nm) * 1.0e-7,
                    weighting="photon",
                )
            )
            for part in pdes_partitions
        ],
        dtype=np.float64,
    )
    out["F_CO_pdes_draine_bands"] = draine_references_for_product_partitions(
        "F_CO_pdes_photon"
    ).to("1/(cm^2 s)").magnitude
    return out


def _compute_uv_product_cells(
    sources: dict[str, float],
    r_cell_cm: np.ndarray,
    *,
    min_chi: float,
) -> dict[str, np.ndarray]:
    r = np.maximum(np.asarray(r_cell_cm, dtype=float), 1.0e-30)
    dilution = 1.0 / (4.0 * np.pi * r**2)

    chi_raw = float(sources["chi_broad"]) * dilution
    ambient = float(min_chi)

    products = {
        "chi_broad": chi_raw + ambient,
        "G_CO_diss": float(sources["G_CO_diss"]) * dilution + ambient,
        "G_H2_diss": float(sources["G_H2_diss"]) * dilution + ambient,
        "G_C_ion": float(sources["G_C_ion"]) * dilution + ambient,
        "G_CO_pdes": float(sources["G_CO_pdes"]) * dilution + ambient,
        "F_CO_pdes_photon": (
            float(sources["F_CO_pdes_photon"]) * dilution
            + ambient * float(sources["F_CO_pdes_draine"])
        ),
    }
    for name in ("G_S_ion", "G_Si_ion", "G_CH_diss", "G_OH_diss"):
        if name in sources:
            products[name] = float(sources[name]) * dilution + ambient
    if "F_CO_pdes_photon_bands" in sources:
        pdes_band_lum = np.asarray(sources["F_CO_pdes_photon_bands"], dtype=float)
        pdes_band_draine = np.asarray(sources["F_CO_pdes_draine_bands"], dtype=float)
        products["F_CO_pdes_photon_bands"] = (
            pdes_band_lum[:, None] * dilution[None, :]
            + ambient * pdes_band_draine[:, None]
        )
    else:
        products["F_CO_pdes_photon_bands"] = products["F_CO_pdes_photon"][None, :]
    return {name: np.asarray(value, dtype=float) for name, value in products.items()}


def _compute_chi_cells(Luv_erg_s: float, r_cell_cm: np.ndarray) -> np.ndarray:
    r = np.maximum(np.asarray(r_cell_cm, dtype=float), 1e-30)
    F_uv = float(Luv_erg_s) / (4.0 * np.pi * r**2)
    chi = F_uv / (float(C_LIGHT) * float(U_DRAINE))
    return np.asarray(chi, dtype=float)


def _add_ambient_chi(chi_cells: np.ndarray, min_chi: float) -> np.ndarray:
    return np.asarray(chi_cells, dtype=float) + float(min_chi)


def _estimate_tdust_grey(cfg: InfallStream1DConfig, *, r_cell_cm: np.ndarray, Av_cent: np.ndarray) -> np.ndarray:
    R_cm = float(cfg.rstar_rsun) * float(SOLAR_RADIUS)
    T_star = float(cfg.teff_K)

    r = np.maximum(np.asarray(r_cell_cm, dtype=float), 1e-30)
    T_thin = T_star * np.sqrt(R_cm / (2.0 * r))

    tau = np.asarray(Av_cent, dtype=float) / 1.086
    T_att = T_thin * np.exp(-0.25 * tau)

    return np.maximum(T_att, float(T_CMB))


def _evaluate_environment(
    cfg: InfallStream1DConfig,
    *,
    Luv_erg_s: float,
    x_cent_cm: np.ndarray,
    Av_cent: np.ndarray,
    r_face_start_cm: float,
    r_stop_cm: float,
    M_g: float,
    t_s: float,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    r_face_cm = max(_freefall_radius_from_time(r_face_start_cm, t_s, M_g), r_stop_cm)
    r_cell = r_face_cm + np.asarray(x_cent_cm, dtype=float)
    chi_cells = _add_ambient_chi(_compute_chi_cells(Luv_erg_s, r_cell), cfg.min_chi)
    td_cells = None
    if cfg.estimate_tdust:
        td_cells = _estimate_tdust_grey(cfg, r_cell_cm=r_cell, Av_cent=Av_cent)
    return float(r_face_cm), np.asarray(chi_cells, dtype=float), td_cells


def _evaluate_product_environment(
    cfg: InfallStream1DConfig,
    *,
    uv_sources: dict[str, float],
    x_cent_cm: np.ndarray,
    Av_cent: np.ndarray,
    r_face_start_cm: float,
    r_stop_cm: float,
    M_g: float,
    t_s: float,
) -> tuple[float, dict[str, np.ndarray], np.ndarray | None]:
    r_face_cm = max(_freefall_radius_from_time(r_face_start_cm, t_s, M_g), r_stop_cm)
    r_cell = r_face_cm + np.asarray(x_cent_cm, dtype=float)
    uv_products = _compute_uv_product_cells(uv_sources, r_cell, min_chi=cfg.min_chi)
    td_cells = None
    if cfg.estimate_tdust:
        td_cells = _estimate_tdust_grey(cfg, r_cell_cm=r_cell, Av_cent=Av_cent)
    return float(r_face_cm), uv_products, td_cells


def _assign_environment(
    rad: RadModel,
    shape: tuple[int, ...],
    *,
    chi_cells: np.ndarray,
    td_cells: np.ndarray | None,
) -> None:
    rad.set_incident_uv(
        chi=Quantity((2.0 * np.asarray(chi_cells, dtype=float)).reshape(shape), "dimensionless"),
        Av=rad.Av,
    )
    if td_cells is not None:
        rad.dust_temperature = Quantity(np.asarray(td_cells, dtype=float).reshape(shape), "K")


def _assign_product_environment(
    rad: RadModel,
    shape: tuple[int, ...],
    *,
    uv_products: dict[str, np.ndarray],
    td_cells: np.ndarray | None,
) -> None:
    products = {
        "chi_broad": Quantity(np.asarray(uv_products["chi_broad"], dtype=float).reshape(shape), "dimensionless"),
        "G_CO_diss": Quantity(np.asarray(uv_products["G_CO_diss"], dtype=float).reshape(shape), "dimensionless"),
        "G_H2_diss": Quantity(np.asarray(uv_products["G_H2_diss"], dtype=float).reshape(shape), "dimensionless"),
        "G_C_ion": Quantity(np.asarray(uv_products["G_C_ion"], dtype=float).reshape(shape), "dimensionless"),
        "G_CO_pdes": Quantity(np.asarray(uv_products["G_CO_pdes"], dtype=float).reshape(shape), "dimensionless"),
        "F_CO_pdes_photon": Quantity(
            np.asarray(uv_products["F_CO_pdes_photon"], dtype=float).reshape(shape),
            "1/(cm^2 s)",
        ),
        "F_CO_pdes_photon_bands": Quantity(
            np.asarray(uv_products["F_CO_pdes_photon_bands"], dtype=float).reshape(
                (np.asarray(uv_products["F_CO_pdes_photon_bands"]).shape[0],) + tuple(shape)
            ),
            "1/(cm^2 s)",
        ),
    }
    rad.set_incident_uv_products(products=products, Av=rad.Av)
    if td_cells is not None:
        rad.dust_temperature = Quantity(np.asarray(td_cells, dtype=float).reshape(shape), "K")


def _build_output_times(t_end_s: float, n_steps: int, power: float) -> np.ndarray:
    n = int(n_steps)
    if n <= 0:
        raise ValueError("n_steps must be > 0")
    p = float(power)
    if not (p > 0.0):
        raise ValueError("output_time_power must be > 0")
    u = np.linspace(0.0, 1.0, n + 1, dtype=float)
    return float(t_end_s) * (1.0 - np.power(1.0 - u, p))


def _suggest_infall_dt(
    cfg: InfallStream1DConfig,
    *,
    Luv_erg_s: float,
    t_now_s: float,
    t_limit_s: float,
    r_face_now_cm: float,
    r_face_start_cm: float,
    r_stop_cm: float,
    x_front_cm: float,
    M_g: float,
) -> float:
    t_now = float(t_now_s)
    t_limit = float(t_limit_s)
    if t_limit <= t_now:
        return 0.0

    max_dlnchi = float(cfg.max_dlnchi)
    if not (max_dlnchi > 0.0):
        raise ValueError("max_dlnchi must be > 0")

    x_front = float(x_front_cm)
    r_face_now = float(r_face_now_cm)
    r_front_cell_now = max(r_face_now + x_front, 1.0e-30)
    chi_front_raw = float(_compute_chi_cells(Luv_erg_s, np.asarray([r_front_cell_now], dtype=float))[0])
    chi_front_total = chi_front_raw + float(cfg.min_chi)
    chi_front_target = chi_front_total * np.exp(max_dlnchi)
    chi_raw_target = chi_front_target - float(cfg.min_chi)
    if not (chi_raw_target > chi_front_raw > 0.0) or not np.isfinite(chi_raw_target):
        return t_limit - t_now

    r_front_cell_target = np.sqrt(
        float(Luv_erg_s) / (4.0 * np.pi * float(C_LIGHT) * float(U_DRAINE) * chi_raw_target)
    )
    r_face_target = max(float(r_front_cell_target) - x_front, float(r_stop_cm))
    if not (r_face_target < r_face_now):
        return t_limit - t_now

    t_target = _freefall_time_to_radius(r_face_start_cm, r_face_target, M_g)
    dt_target = max(float(t_target) - t_now, 0.0)
    if not (dt_target > 0.0):
        return t_limit - t_now
    return min(dt_target, t_limit - t_now)


def _freefall_time_to_radius(r0_cm: float, r_cm: float, M_g: float) -> float:
    r0 = float(r0_cm)
    r = min(max(float(r_cm), 0.0), r0)
    if r0 <= 0.0:
        raise ValueError("r0_cm must be > 0")
    prefac = 2.0 / (3.0 * np.sqrt(2.0 * float(G_GRAV) * float(M_g)))
    return float(prefac * (r0**1.5 - r**1.5))


def _freefall_radius_from_time(r0_cm: float, t_s: float, M_g: float) -> float:
    r0 = float(r0_cm)
    if r0 <= 0.0:
        raise ValueError("r0_cm must be > 0")
    term = r0**1.5 - 1.5 * np.sqrt(2.0 * float(G_GRAV) * float(M_g)) * max(float(t_s), 0.0)
    if term <= 0.0:
        return 0.0
    return float(term ** (2.0 / 3.0))


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _neutral_c_abundance(y: np.ndarray, Zg: float) -> np.ndarray:
    xCtot = float(Zg) * float(XC_STD)
    xC = xCtot - (
        y[..., I_HCOP]
        + y[..., I_CHX]
        + y[..., I_CO]
        + y[..., I_CO_ICE]
        + y[..., I_CP]
    )
    return np.maximum(xC, 0.0)


def _extract_abundance_profiles(y: np.ndarray, Zg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(y[:, I_CO], dtype=float).copy(),
        _neutral_c_abundance(y, Zg).copy(),
        np.asarray(y[:, I_CP], dtype=float).copy(),
        np.asarray(y[:, I_CHX], dtype=float).copy(),
        np.asarray(y[:, I_HCOP], dtype=float).copy(),
    )


def _max_movie_frames(fps: int, max_duration_s: float) -> int:
    fps_eff = max(int(fps), 1)
    return max(int(np.floor(float(max_duration_s) * fps_eff)), 1)


def _build_init_relax_record_steps(max_steps: int, max_frames: int) -> np.ndarray:
    n_steps = int(max_steps)
    n_frames = int(max_frames)
    if n_steps <= 0:
        return np.array([0], dtype=int)
    if n_frames <= 1:
        return np.array([0], dtype=int)
    if n_steps + 1 <= n_frames:
        return np.arange(n_steps + 1, dtype=int)

    # Bias snapshots toward early relaxation, where abundances usually change fastest.
    u = np.linspace(0.0, 1.0, n_frames, dtype=float)
    idx = np.rint(np.expm1(u * np.log1p(float(n_steps)))).astype(int)
    idx[0] = 0
    idx[-1] = n_steps
    idx = np.clip(idx, 0, n_steps)
    return np.unique(idx)


def _movie_frame_indices(n_frames: int, fps: int, max_duration_s: float | None) -> np.ndarray:
    n = int(n_frames)
    if n <= 0:
        raise ValueError("n_frames must be > 0")
    if max_duration_s is None:
        return np.arange(n, dtype=int)

    max_frames = _max_movie_frames(fps, max_duration_s)
    if n <= max_frames:
        return np.arange(n, dtype=int)

    idx = np.rint(np.linspace(0.0, float(n - 1), max_frames, dtype=float)).astype(int)
    idx[0] = 0
    idx[-1] = n - 1
    idx = np.clip(idx, 0, n - 1)
    return np.unique(idx)


def _fmt_compact(value: float, precision: int = 3) -> str:
    s = f"{float(value):.{int(precision)}g}"
    return s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")


def _density_tag(nH_cm3: float) -> str:
    return f"nH_{_fmt_compact(nH_cm3)}"


def _render_abundance_movie(
    out_path: Path,
    *,
    nH_cm3: float,
    x_au: np.ndarray,
    t_hist: np.ndarray,
    r_face_hist: np.ndarray,
    chi_face_hist: np.ndarray,
    teff_K: float,
    xco_hist: np.ndarray,
    xc_hist: np.ndarray,
    xcp_hist: np.ndarray,
    xchx_hist: np.ndarray,
    xhcop_hist: np.ndarray,
    xco_eq_hist: np.ndarray | None = None,
    xc_eq_hist: np.ndarray | None = None,
    xcp_eq_hist: np.ndarray | None = None,
    xchx_eq_hist: np.ndarray | None = None,
    xhcop_eq_hist: np.ndarray | None = None,
    fps: int,
    max_duration_s: float | None = None,
) -> None:
    plt = _setup_matplotlib()
    frame_idx = _movie_frame_indices(len(t_hist), fps, max_duration_s)
    t_movie = np.asarray(t_hist, dtype=float)[frame_idx]
    r_face_movie = np.asarray(r_face_hist, dtype=float)[frame_idx]
    chi_face_movie = np.asarray(chi_face_hist, dtype=float)[frame_idx]
    xco_movie = np.asarray(xco_hist, dtype=float)[frame_idx]
    xc_movie = np.asarray(xc_hist, dtype=float)[frame_idx]
    xcp_movie = np.asarray(xcp_hist, dtype=float)[frame_idx]
    xchx_movie = np.asarray(xchx_hist, dtype=float)[frame_idx]
    xhcop_movie = np.asarray(xhcop_hist, dtype=float)[frame_idx]
    xco_eq_movie = None if xco_eq_hist is None else np.asarray(xco_eq_hist, dtype=float)[frame_idx]
    xc_eq_movie = None if xc_eq_hist is None else np.asarray(xc_eq_hist, dtype=float)[frame_idx]
    xcp_eq_movie = None if xcp_eq_hist is None else np.asarray(xcp_eq_hist, dtype=float)[frame_idx]
    xchx_eq_movie = None if xchx_eq_hist is None else np.asarray(xchx_eq_hist, dtype=float)[frame_idx]
    xhcop_eq_movie = None if xhcop_eq_hist is None else np.asarray(xhcop_eq_hist, dtype=float)[frame_idx]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    line_co, = ax.plot([], [], label="CO", lw=2.0)
    line_c, = ax.plot([], [], label="C", lw=2.0)
    line_cp, = ax.plot([], [], label="C+", lw=2.0)
    line_chx, = ax.plot([], [], label="CHx", lw=2.0)
    line_hcop, = ax.plot([], [], label="HCO+", lw=2.0)
    line_co_eq = None
    line_c_eq = None
    line_cp_eq = None
    line_chx_eq = None
    line_hcop_eq = None
    if xco_eq_movie is not None:
        line_co_eq, = ax.plot([], [], "--", color=line_co.get_color(), label="_nolegend_", lw=1.8)
        line_c_eq, = ax.plot([], [], "--", color=line_c.get_color(), label="_nolegend_", lw=1.8)
        line_cp_eq, = ax.plot([], [], "--", color=line_cp.get_color(), label="_nolegend_", lw=1.8)
        line_chx_eq, = ax.plot([], [], "--", color=line_chx.get_color(), label="_nolegend_", lw=1.8)
        line_hcop_eq, = ax.plot([], [], "--", color=line_hcop.get_color(), label="_nolegend_", lw=1.8)

    ax.set_xlim(float(np.min(x_au)), float(np.max(x_au)))
    ax.set_ylim(1.0e-13, 2.0e-3)
    ax.set_yscale("log")
    ax.set_xlabel("x along stream [au] (front=0)")
    ax.set_ylabel("abundance x")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    def _update(i: int):
        line_co.set_data(x_au, np.maximum(xco_movie[i], 1.0e-300))
        line_c.set_data(x_au, np.maximum(xc_movie[i], 1.0e-300))
        line_cp.set_data(x_au, np.maximum(xcp_movie[i], 1.0e-300))
        line_chx.set_data(x_au, np.maximum(xchx_movie[i], 1.0e-300))
        line_hcop.set_data(x_au, np.maximum(xhcop_movie[i], 1.0e-300))
        artists = [line_co, line_c, line_cp, line_chx, line_hcop]
        if xco_eq_movie is not None:
            line_co_eq.set_data(x_au, np.maximum(xco_eq_movie[i], 1.0e-300))
            line_c_eq.set_data(x_au, np.maximum(xc_eq_movie[i], 1.0e-300))
            line_cp_eq.set_data(x_au, np.maximum(xcp_eq_movie[i], 1.0e-300))
            line_chx_eq.set_data(x_au, np.maximum(xchx_eq_movie[i], 1.0e-300))
            line_hcop_eq.set_data(x_au, np.maximum(xhcop_eq_movie[i], 1.0e-300))
            artists.extend([line_co_eq, line_c_eq, line_cp_eq, line_chx_eq, line_hcop_eq])
        title = (
            f"nH={_fmt_compact(nH_cm3)} cm^-3 | "
            f"r={_fmt_compact(float(r_face_movie[i]) / float(AU), 4)} au | "
            f"chi={_fmt_compact(float(chi_face_movie[i]), 4)} | "
            f"Tbb={_fmt_compact(teff_K, 4)} K | "
            f"t={_fmt_compact(float(t_movie[i]) / (365.25 * 24.0 * 3600.0), 4)} yr"
        )
        ax.set_title(title)
        return tuple(artists)

    fps_eff = max(int(fps), 1)
    anim = FuncAnimation(fig, _update, frames=int(len(t_movie)), interval=max(int(1000 / fps_eff), 1), blit=False)
    anim.save(out_path, writer=FFMpegWriter(fps=fps_eff))
    plt.close(fig)


def _placeholder_error_plot(out_dir: Path, title: str, text: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.01, 0.99, text, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out_dir / "error.png", dpi=150)
    plt.close(fig)


def _run_one_density(cfg: InfallStream1DConfig, *, nH_cm3: float, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    radiation_mode = _validate_radiation_mode(cfg.radiation_mode)

    rad, x_cent_cm = _build_stream_model(cfg, nH_cm3=nH_cm3)
    shape = rad.model.mesh.shape
    ncells = int(np.prod(shape))
    Av_cent_1d = rad.Av.to("dimensionless").magnitude.reshape(-1)

    M_g = float(cfg.mstar_msun) * float(SOLAR_MASS)
    r_face_start_cm = float(cfg.r_face_start_au) * float(AU)
    r_stop_cm = float(cfg.r_face_stop_au) * float(AU)
    n_output_steps = int(cfg.n_steps)
    if n_output_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if not (float(cfg.output_time_power) > 0.0):
        raise ValueError("output_time_power must be > 0")
    if not (float(cfg.max_dlnchi) > 0.0):
        raise ValueError("max_dlnchi must be > 0")
    if int(cfg.max_inner_steps) <= 0:
        raise ValueError("max_inner_steps must be > 0")
    if not (float(cfg.movie_max_duration_s) > 0.0):
        raise ValueError("movie_max_duration_s must be > 0")
    if int(cfg.equilibrium_astrochem_n_updates) < 1:
        raise ValueError("equilibrium_astrochem_n_updates must be >= 1")
    if int(cfg.equilibrium_shielding_max_iter) < int(cfg.equilibrium_astrochem_n_updates):
        raise ValueError("equilibrium_shielding_max_iter must be >= equilibrium_astrochem_n_updates")
    if not (float(cfg.equilibrium_shielding_reltol) > 0.0):
        raise ValueError("equilibrium_shielding_reltol must be > 0")
    if not (float(cfg.equilibrium_shielding_abstol) > 0.0):
        raise ValueError("equilibrium_shielding_abstol must be > 0")
    if not (float(cfg.equilibrium_astrochem_t_end_yr) > 0.0):
        raise ValueError("equilibrium_astrochem_t_end_yr must be > 0")
    if not (0.0 < r_stop_cm < r_face_start_cm):
        raise ValueError("r_face_stop_au must satisfy 0 < r_face_stop_au < r_face_start_au")

    uv_sources = None
    if radiation_mode == RADIATION_MODE_STELLAR_PRODUCTS:
        uv_sources = _star_uv_product_sources(cfg)
        Luv = float(uv_sources["chi_broad"]) * float(C_LIGHT) * float(U_DRAINE)
    else:
        Luv = _star_uv_luminosity(cfg)

    t_end = _freefall_time_to_radius(r_face_start_cm, r_stop_cm, M_g)
    t_output = _build_output_times(t_end, n_output_steps, cfg.output_time_power)
    dt_output_ref = float(t_end) / float(n_output_steps)

    gow_cfg = {
        "shielding_outer_1d": str(cfg.shielding_outer_1d),
        "b_kms": float(cfg.b_kms),
        "ion_rate": str(cfg.ion_rate),
        "Zg": float(cfg.Zg),
        "enable_co_phase": bool(cfg.enable_co_phase),
        "temperature": {"mode": "computed" if bool(cfg.evolve_energy) else "dust"},
        "reltol": float(cfg.reltol),
        "abstol0": float(cfg.abstol0),
        "mxsteps": int(cfg.mxsteps),
        "maxord": int(cfg.maxord),
        "tolfac": float(cfg.tolfac),
        "userJac": bool(cfg.userJac),
        "verbose": bool(cfg.verbose),
        "isDust_cooling": True,
        "isCoolingCOThin": False,
        "gradv": 1.0e-14,
        "Leff_CO_max": 3.0e20,
    }
    gow_cfg_eq = {
        **gow_cfg,
        "astrochem_n_updates": int(cfg.equilibrium_astrochem_n_updates),
        "astrochem_t_end_yr": float(cfg.equilibrium_astrochem_t_end_yr),
        "shielding_max_iter": int(cfg.equilibrium_shielding_max_iter),
        "shielding_reltol": float(cfg.equilibrium_shielding_reltol),
        "shielding_abstol": float(cfg.equilibrium_shielding_abstol),
    }

    if radiation_mode == RADIATION_MODE_STELLAR_PRODUCTS:
        r_face_cm, uv_products, td_cells = _evaluate_product_environment(
            cfg,
            uv_sources=uv_sources,
            x_cent_cm=x_cent_cm,
            Av_cent=Av_cent_1d,
            r_face_start_cm=r_face_start_cm,
            r_stop_cm=r_stop_cm,
            M_g=M_g,
            t_s=0.0,
        )
        _assign_product_environment(rad, shape, uv_products=uv_products, td_cells=td_cells)
        chi_cells = uv_products["chi_broad"]
    else:
        r_face_cm, chi_cells, td_cells = _evaluate_environment(
            cfg,
            Luv_erg_s=Luv,
            x_cent_cm=x_cent_cm,
            Av_cent=Av_cent_1d,
            r_face_start_cm=r_face_start_cm,
            r_stop_cm=r_stop_cm,
            M_g=M_g,
            t_s=0.0,
        )
        _assign_environment(rad, shape, chi_cells=chi_cells, td_cells=td_cells)

    stepper = Gow17TimeStepper(rad, gow_cfg)
    rad.gow17_y = np.asarray(stepper.y_state, dtype=float).reshape(shape + (N_Y,))
    density_tag = _density_tag(float(nH_cm3))

    if cfg.init_relax:
        dt_init = float(dt_output_ref) * float(cfg.init_relax_dt_frac)
        if not (dt_init > 0.0):
            raise ValueError("init_relax_dt_frac must give dt_init > 0")

        max_init_movie_frames = _max_movie_frames(int(cfg.movie_fps), float(cfg.movie_max_duration_s))
        relax_record_steps = _build_init_relax_record_steps(int(cfg.init_relax_n_steps), max_init_movie_frames)
        next_relax_record = 1

        key_idx = np.asarray([I_H2, I_CO, I_CHX, I_CP, I_HCOP], dtype=np.int64)
        y_relax0 = np.asarray(stepper.y_state, dtype=float).reshape(ncells, N_Y)
        xco0, xc0, xcp0, xchx0, xhcop0 = _extract_abundance_profiles(y_relax0, cfg.Zg)
        relax_t_hist = [0.0]
        relax_r_face_hist = [float(r_face_cm)]
        relax_chi_face_hist = [float(chi_cells[0])]
        relax_xco_hist = [xco0]
        relax_xc_hist = [xc0]
        relax_xcp_hist = [xcp0]
        relax_xchx_hist = [xchx0]
        relax_xhcop_hist = [xhcop0]
        t_relax = 0.0
        converged = False
        for i_relax in range(int(cfg.init_relax_n_steps)):
            y_prev = stepper.y_state[:, key_idx].copy()
            stepper.step(dt_init)
            t_relax += dt_init
            y_new = stepper.y_state[:, key_idx]
            denom = np.maximum(np.abs(y_new), float(cfg.abstol0))
            rel = float(np.max(np.abs(y_new - y_prev) / denom))
            y_relax = np.asarray(stepper.y_state, dtype=float).reshape(ncells, N_Y)
            xco_relax, xc_relax, xcp_relax, xchx_relax, xhcop_relax = _extract_abundance_profiles(y_relax, cfg.Zg)
            step_num = i_relax + 1
            should_record = False
            if next_relax_record < len(relax_record_steps) and step_num >= int(relax_record_steps[next_relax_record]):
                should_record = True
                while next_relax_record < len(relax_record_steps) and step_num >= int(relax_record_steps[next_relax_record]):
                    next_relax_record += 1
            if i_relax + 1 >= int(cfg.init_relax_min_steps) and rel <= float(cfg.init_relax_rtol):
                converged = True
                should_record = True
            if should_record:
                relax_t_hist.append(float(t_relax))
                relax_r_face_hist.append(float(r_face_cm))
                relax_chi_face_hist.append(float(chi_cells[0]))
                relax_xco_hist.append(xco_relax)
                relax_xc_hist.append(xc_relax)
                relax_xcp_hist.append(xcp_relax)
                relax_xchx_hist.append(xchx_relax)
                relax_xhcop_hist.append(xhcop_relax)
            if converged:
                break
        if cfg.make_movies:
            _render_abundance_movie(
                out_dir / f"abundances_init_relax_{density_tag}.mp4",
                nH_cm3=float(nH_cm3),
                x_au=x_cent_cm / float(AU),
                t_hist=np.asarray(relax_t_hist, dtype=float),
                r_face_hist=np.asarray(relax_r_face_hist, dtype=float),
                chi_face_hist=np.asarray(relax_chi_face_hist, dtype=float),
                teff_K=float(cfg.teff_K),
                xco_hist=np.asarray(relax_xco_hist, dtype=float),
                xc_hist=np.asarray(relax_xc_hist, dtype=float),
                xcp_hist=np.asarray(relax_xcp_hist, dtype=float),
                xchx_hist=np.asarray(relax_xchx_hist, dtype=float),
                xhcop_hist=np.asarray(relax_xhcop_hist, dtype=float),
                fps=int(cfg.movie_fps),
                max_duration_s=float(cfg.movie_max_duration_s),
            )
        if not converged:
            print(
                "initial steady-state relaxation did not converge within "
                f"{int(cfg.init_relax_n_steps)} steps"
            )

    t_hist = np.zeros(n_output_steps + 1, dtype=float)
    r_face_hist = np.zeros_like(t_hist)
    chi_face_hist = np.zeros_like(t_hist)

    i_front = 0
    i_back = int(ncells - 1)

    xco_hist = np.zeros((n_output_steps + 1, ncells), dtype=float)
    xc_hist = np.zeros_like(xco_hist)
    xcp_hist = np.zeros_like(xco_hist)
    xchx_hist = np.zeros_like(xco_hist)
    xhcop_hist = np.zeros_like(xco_hist)

    Tgas_front = np.zeros_like(t_hist)
    Tdust_front = np.zeros_like(t_hist)
    xco_eq_hist = None
    xc_eq_hist = None
    xcp_eq_hist = None
    xchx_eq_hist = None
    xhcop_eq_hist = None
    Tgas_front_eq = None
    Tdust_front_eq = None

    rad_eq = None
    stepper_eq = None
    if cfg.track_infall_equilibrium:
        rad_eq, _ = _build_stream_model(cfg, nH_cm3=nH_cm3)
        if radiation_mode == RADIATION_MODE_STELLAR_PRODUCTS:
            _assign_product_environment(rad_eq, shape, uv_products=uv_products, td_cells=td_cells)
        else:
            _assign_environment(rad_eq, shape, chi_cells=chi_cells, td_cells=td_cells)
        if cfg.evolve_energy and getattr(rad, "gas_temperature", None) is not None:
            rad_eq.gas_temperature = Quantity(
                np.asarray(rad.gas_temperature.to("K").magnitude, dtype=float).copy(),
                "K",
            )
        stepper_eq = Gow17TimeStepper(rad_eq, gow_cfg_eq)
        stepper_eq.y_state[:, :] = np.asarray(stepper.y_state, dtype=float)
        rad_eq.gow17_y = np.asarray(stepper_eq.y_state, dtype=float).reshape(shape + (N_Y,))
        stepper_eq.solve_equilibrium()

        xco_eq_hist = np.zeros_like(xco_hist)
        xc_eq_hist = np.zeros_like(xco_hist)
        xcp_eq_hist = np.zeros_like(xco_hist)
        xchx_eq_hist = np.zeros_like(xco_hist)
        xhcop_eq_hist = np.zeros_like(xco_hist)
        Tgas_front_eq = np.zeros_like(t_hist)
        Tdust_front_eq = np.zeros_like(t_hist)

    def _record_branch(
        rad_local: RadModel,
        *,
        xco_store: np.ndarray,
        xc_store: np.ndarray,
        xcp_store: np.ndarray,
        xchx_store: np.ndarray,
        xhcop_store: np.ndarray,
        Tgas_store: np.ndarray,
        Tdust_store: np.ndarray,
        k: int,
    ) -> None:
        y = np.asarray(rad_local.gow17_y, dtype=float).reshape(ncells, N_Y)
        xco_store[k, :] = y[:, I_CO]
        xcp_store[k, :] = y[:, I_CP]
        xchx_store[k, :] = y[:, I_CHX]
        xhcop_store[k, :] = y[:, I_HCOP]
        xc_store[k, :] = _neutral_c_abundance(y, cfg.Zg)

        Tdust_store[k] = float(rad_local.dust_temperature.to("K").magnitude.reshape(-1)[i_front])
        if cfg.evolve_energy and getattr(rad_local, "Tgas_gow17", None) is not None:
            Tgas_store[k] = float(rad_local.Tgas_gow17.to("K").magnitude.reshape(-1)[i_front])
        else:
            Tgas_store[k] = float(rad_local.gas_temperature.to("K").magnitude.reshape(-1)[i_front])

    def _record(k: int):
        _record_branch(
            rad,
            xco_store=xco_hist,
            xc_store=xc_hist,
            xcp_store=xcp_hist,
            xchx_store=xchx_hist,
            xhcop_store=xhcop_hist,
            Tgas_store=Tgas_front,
            Tdust_store=Tdust_front,
            k=k,
        )

        if cfg.track_infall_equilibrium:
            _record_branch(
                rad_eq,
                xco_store=xco_eq_hist,
                xc_store=xc_eq_hist,
                xcp_store=xcp_eq_hist,
                xchx_store=xchx_eq_hist,
                xhcop_store=xhcop_eq_hist,
                Tgas_store=Tgas_front_eq,
                Tdust_store=Tdust_front_eq,
                k=k,
            )

        r_face_hist[k] = r_face_cm
        chi_face_hist[k] = float(chi_cells[i_front])

    _record(0)

    t_evolve = 0.0
    x_front_cm = float(x_cent_cm[i_front])
    time_eps = max(np.finfo(float).eps * max(float(t_end), 1.0) * 32.0, 1.0e-30)
    inner_t_hist = [0.0]
    inner_dt_hist: list[float] = []
    inner_r_face_hist = [float(r_face_cm)]
    inner_chi_face_hist = [float(chi_cells[i_front])]
    n_inner_steps = 0

    for k in range(1, n_output_steps + 1):
        t_next = float(t_output[k])
        while t_evolve + time_eps < t_next:
            dt_limit = t_next - t_evolve
            dt_step = _suggest_infall_dt(
                cfg,
                Luv_erg_s=Luv,
                t_now_s=t_evolve,
                t_limit_s=t_next,
                r_face_now_cm=r_face_cm,
                r_face_start_cm=r_face_start_cm,
                r_stop_cm=r_stop_cm,
                x_front_cm=x_front_cm,
                M_g=M_g,
            )
            dt_step = min(dt_limit, dt_step)
            if dt_step <= time_eps:
                dt_step = dt_limit

            if radiation_mode == RADIATION_MODE_STELLAR_PRODUCTS:
                _, uv_mid, td_mid = _evaluate_product_environment(
                    cfg,
                    uv_sources=uv_sources,
                    x_cent_cm=x_cent_cm,
                    Av_cent=Av_cent_1d,
                    r_face_start_cm=r_face_start_cm,
                    r_stop_cm=r_stop_cm,
                    M_g=M_g,
                    t_s=t_evolve + 0.5 * dt_step,
                )
                _assign_product_environment(rad, shape, uv_products=uv_mid, td_cells=td_mid)
            else:
                _, chi_mid, td_mid = _evaluate_environment(
                    cfg,
                    Luv_erg_s=Luv,
                    x_cent_cm=x_cent_cm,
                    Av_cent=Av_cent_1d,
                    r_face_start_cm=r_face_start_cm,
                    r_stop_cm=r_stop_cm,
                    M_g=M_g,
                    t_s=t_evolve + 0.5 * dt_step,
                )
                _assign_environment(rad, shape, chi_cells=chi_mid, td_cells=td_mid)
            stepper.step(dt_step)

            t_evolve += dt_step
            if radiation_mode == RADIATION_MODE_STELLAR_PRODUCTS:
                r_face_cm, uv_products, td_cells = _evaluate_product_environment(
                    cfg,
                    uv_sources=uv_sources,
                    x_cent_cm=x_cent_cm,
                    Av_cent=Av_cent_1d,
                    r_face_start_cm=r_face_start_cm,
                    r_stop_cm=r_stop_cm,
                    M_g=M_g,
                    t_s=t_evolve,
                )
                _assign_product_environment(rad, shape, uv_products=uv_products, td_cells=td_cells)
                chi_cells = uv_products["chi_broad"]
            else:
                r_face_cm, chi_cells, td_cells = _evaluate_environment(
                    cfg,
                    Luv_erg_s=Luv,
                    x_cent_cm=x_cent_cm,
                    Av_cent=Av_cent_1d,
                    r_face_start_cm=r_face_start_cm,
                    r_stop_cm=r_stop_cm,
                    M_g=M_g,
                    t_s=t_evolve,
                )
                _assign_environment(rad, shape, chi_cells=chi_cells, td_cells=td_cells)

            inner_dt_hist.append(float(dt_step))
            inner_t_hist.append(float(t_evolve))
            inner_r_face_hist.append(float(r_face_cm))
            inner_chi_face_hist.append(float(chi_cells[i_front]))
            n_inner_steps += 1
            if n_inner_steps > int(cfg.max_inner_steps):
                raise RuntimeError("adaptive infall stepping exceeded max_inner_steps")

        t_evolve = t_next
        if radiation_mode == RADIATION_MODE_STELLAR_PRODUCTS:
            r_face_cm, uv_products, td_cells = _evaluate_product_environment(
                cfg,
                uv_sources=uv_sources,
                x_cent_cm=x_cent_cm,
                Av_cent=Av_cent_1d,
                r_face_start_cm=r_face_start_cm,
                r_stop_cm=r_stop_cm,
                M_g=M_g,
                t_s=t_evolve,
            )
            _assign_product_environment(rad, shape, uv_products=uv_products, td_cells=td_cells)
            chi_cells = uv_products["chi_broad"]
        else:
            r_face_cm, chi_cells, td_cells = _evaluate_environment(
                cfg,
                Luv_erg_s=Luv,
                x_cent_cm=x_cent_cm,
                Av_cent=Av_cent_1d,
                r_face_start_cm=r_face_start_cm,
                r_stop_cm=r_stop_cm,
                M_g=M_g,
                t_s=t_evolve,
            )
            _assign_environment(rad, shape, chi_cells=chi_cells, td_cells=td_cells)
        if cfg.track_infall_equilibrium:
            if radiation_mode == RADIATION_MODE_STELLAR_PRODUCTS:
                _assign_product_environment(rad_eq, shape, uv_products=uv_products, td_cells=td_cells)
            else:
                _assign_environment(rad_eq, shape, chi_cells=chi_cells, td_cells=td_cells)
            stepper_eq.solve_equilibrium()
        t_hist[k] = t_next
        _record(k)

        if t_hist[k] >= t_end or r_face_cm <= r_stop_cm:
            t_hist = t_hist[: k + 1]
            r_face_hist = r_face_hist[: k + 1]
            chi_face_hist = chi_face_hist[: k + 1]
            xco_hist = xco_hist[: k + 1, :]
            xc_hist = xc_hist[: k + 1, :]
            xcp_hist = xcp_hist[: k + 1, :]
            xchx_hist = xchx_hist[: k + 1, :]
            xhcop_hist = xhcop_hist[: k + 1, :]
            Tgas_front = Tgas_front[: k + 1]
            Tdust_front = Tdust_front[: k + 1]
            if cfg.track_infall_equilibrium:
                xco_eq_hist = xco_eq_hist[: k + 1, :]
                xc_eq_hist = xc_eq_hist[: k + 1, :]
                xcp_eq_hist = xcp_eq_hist[: k + 1, :]
                xchx_eq_hist = xchx_eq_hist[: k + 1, :]
                xhcop_eq_hist = xhcop_eq_hist[: k + 1, :]
                Tgas_front_eq = Tgas_front_eq[: k + 1]
                Tdust_front_eq = Tdust_front_eq[: k + 1]
            break

    history_data = {
        "t_s": t_hist,
        "r_face_cm": r_face_hist,
        "chi_face": chi_face_hist,
        "t_internal_s": np.asarray(inner_t_hist, dtype=float),
        "dt_internal_s": np.asarray(inner_dt_hist, dtype=float),
        "r_face_internal_cm": np.asarray(inner_r_face_hist, dtype=float),
        "chi_face_internal": np.asarray(inner_chi_face_hist, dtype=float),
        "x_cent_cm": x_cent_cm,
        "Av_cent": Av_cent_1d,
        "xco_hist": xco_hist,
        "xc_hist": xc_hist,
        "xcp_hist": xcp_hist,
        "xchx_hist": xchx_hist,
        "xhcop_hist": xhcop_hist,
        "Tgas_front": Tgas_front,
        "Tdust_front": Tdust_front,
    }
    if cfg.track_infall_equilibrium:
        history_data.update(
            {
                "xco_eq_hist": xco_eq_hist,
                "xc_eq_hist": xc_eq_hist,
                "xcp_eq_hist": xcp_eq_hist,
                "xchx_eq_hist": xchx_eq_hist,
                "xhcop_eq_hist": xhcop_eq_hist,
                "Tgas_front_eq": Tgas_front_eq,
                "Tdust_front_eq": Tdust_front_eq,
            }
        )
    np.savez(out_dir / "history.npz", **history_data)

    x_au = x_cent_cm / float(AU)

    if cfg.make_movies:
        _render_abundance_movie(
            out_dir / f"abundances_vs_time_{density_tag}.mp4",
            nH_cm3=float(nH_cm3),
            x_au=x_au,
            t_hist=t_hist,
            r_face_hist=r_face_hist,
            chi_face_hist=chi_face_hist,
            teff_K=float(cfg.teff_K),
            xco_hist=xco_hist,
            xc_hist=xc_hist,
            xcp_hist=xcp_hist,
            xchx_hist=xchx_hist,
            xhcop_hist=xhcop_hist,
            xco_eq_hist=xco_eq_hist,
            xc_eq_hist=xc_eq_hist,
            xcp_eq_hist=xcp_eq_hist,
            xchx_eq_hist=xchx_eq_hist,
            xhcop_eq_hist=xhcop_eq_hist,
            fps=int(cfg.movie_fps),
            max_duration_s=float(cfg.movie_max_duration_s),
        )

    xco_back_change = float(np.abs(xco_hist[-1, i_back] - xco_hist[0, i_back]) / max(xco_hist[0, i_back], 1e-30))
    xco_front_change = float(np.abs(xco_hist[-1, i_front] - xco_hist[0, i_front]) / max(xco_hist[0, i_front], 1e-30))

    out = {
        "nH_cm3": float(nH_cm3),
        "radiation_mode": radiation_mode,
        "n_steps": int(len(t_hist) - 1),
        "n_internal_steps": int(len(inner_dt_hist)),
        "max_dlnchi": float(cfg.max_dlnchi),
        "dt_internal_min_yr": (
            float(np.min(np.asarray(inner_dt_hist, dtype=float)) / (365.25 * 24 * 3600))
            if inner_dt_hist
            else 0.0
        ),
        "dt_internal_max_yr": (
            float(np.max(np.asarray(inner_dt_hist, dtype=float)) / (365.25 * 24 * 3600))
            if inner_dt_hist
            else 0.0
        ),
        "t_end_yr": float(t_hist[-1] / (365.25 * 24 * 3600)),
        "chi_front_start": float(chi_face_hist[0]),
        "chi_front_end": float(chi_face_hist[-1]),
        "xco_front_start": float(xco_hist[0, i_front]),
        "xco_front_end": float(xco_hist[-1, i_front]),
        "xco_back_start": float(xco_hist[0, i_back]),
        "xco_back_end": float(xco_hist[-1, i_back]),
        "xco_front_rel_change": float(xco_front_change),
        "xco_back_rel_change": float(xco_back_change),
    }
    if cfg.track_infall_equilibrium:
        out.update(
            {
                "xco_front_eq_end": float(xco_eq_hist[-1, i_front]),
                "xco_back_eq_end": float(xco_eq_hist[-1, i_back]),
                "xco_front_vs_eq_end_rel_diff": float(
                    np.abs(xco_hist[-1, i_front] - xco_eq_hist[-1, i_front])
                    / max(abs(xco_eq_hist[-1, i_front]), 1e-30)
                ),
                "xco_back_vs_eq_end_rel_diff": float(
                    np.abs(xco_hist[-1, i_back] - xco_eq_hist[-1, i_back])
                    / max(abs(xco_eq_hist[-1, i_back]), 1e-30)
                ),
            }
        )
    return out


def run(out_root: Path, cfg: InfallStream1DConfig) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        out_root / "inputs.json",
        {
            "job": "gow17_infall_stream_1d",
            "time": datetime.now().isoformat(timespec="seconds"),
            "config": cfg.__dict__,
        },
    )

    scan_rows = []
    for nH in cfg.nH_list_cm3:
        tag = f"nH_{nH:.3e}".replace("+", "")
        case_dir = out_root / tag
        row = _run_one_density(cfg, nH_cm3=float(nH), out_dir=case_dir)
        scan_rows.append(row)

    _write_json(out_root / "scan_summary.json", {"rows": scan_rows})

    plt = _setup_matplotlib()
    nH = np.array([r["nH_cm3"] for r in scan_rows], dtype=float)
    front = np.array([r["xco_front_rel_change"] for r in scan_rows], dtype=float)
    back = np.array([r["xco_back_rel_change"] for r in scan_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(nH, front, marker="o", label="front CO rel change")
    ax.plot(nH, back, marker="o", label="back CO rel change")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nH [cm^-3]")
    ax.set_ylabel("relative change in x_CO")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "density_scan_co_change.png", dpi=150)
    plt.close(fig)


def run_with_error_plot(out_root: Path, cfg: InfallStream1DConfig) -> int:
    try:
        run(out_root, cfg)
    except Exception as e:
        out_root.mkdir(parents=True, exist_ok=True)
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        _placeholder_error_plot(out_root, "gow17_infall_stream_1d failed", str(e) + "\n\n" + tb)
        return 1
    return 0
