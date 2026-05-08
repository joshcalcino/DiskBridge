from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np

import diskbridge
from diskbridge._units import Quantity
from diskbridge._config import get_config, resolve_model_config
from diskbridge._logging import logger
from diskbridge._constants import (
    E_BIND_CO,
    NU0_CO,
    F_DRAINE,
    N_LAY,
    N_SURF,
    Y_CO,
    SIGMA_D_ISM_REF,
    F_CRUV_CO_PDES_REF,
    ZETA_CRUV_REF,
    K_CRDES_CO,
    M_H,
)
from diskbridge.model.profiles import compute_cell_volumes
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.shielding.columns_1d import (
    compute_pdr_shielding_1d,
    is_effectively_1d,
)
from diskbridge.chemistry.shielding.healpix_columns import (
    _prepare_healpix_geometry,
    compute_column_rays_healpix,
    compute_gradv_nh_weighted,
    compute_L_geo_from_pathlengths,
)
from diskbridge.chemistry.shielding.healpix_utils import (
    integrate_rays_with_pathlength,
)
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
from diskbridge.chemistry.shielding.w_rays_cache import maybe_ensure_W_rays
from diskbridge.chemistry.validation import (
    validate_chemistry_state,
    gow17_budget_diagnostics,
    project_gow17_state_to_budgets,
)
from diskbridge.radmc3d.uv_products import (
    draine_reference_for_product,
    uv_product_specs_from_config,
    validate_uv_chemistry_config,
)

import diskbridge._gow17 as _gow17

N_Y = _gow17.N_Y
N_PH = _gow17.N_PH
IPH_C = _gow17.IPH_C
IPH_CH = _gow17.IPH_CH
IPH_CO = _gow17.IPH_CO
IPH_OH = _gow17.IPH_OH
IPH_H2 = _gow17.IPH_H2
IPH_S = _gow17.IPH_S
IPH_SI = _gow17.IPH_SI

XC_STD = _gow17.XC_STD
XO_STD = _gow17.XO_STD
XHE = _gow17.XHE

I_HEP = _gow17.I_HEP
I_OHX = _gow17.I_OHX
I_CHX = _gow17.I_CHX
I_CO = _gow17.I_CO
I_CO_ICE = _gow17.I_CO_ICE
I_CP = _gow17.I_CP
I_HCOP = _gow17.I_HCOP
I_H2 = _gow17.I_H2
I_HP = _gow17.I_HP
I_H3P = _gow17.I_H3P
I_H2P = _gow17.I_H2P
I_SP = _gow17.I_SP
I_SIP = _gow17.I_SIP
I_OP = _gow17.I_OP
I_E = _gow17.I_E

GOW17_STATE_SPECIES = {
    "he+": I_HEP,
    "ohx": I_OHX,
    "chx": I_CHX,
    "co": I_CO,
    "co_ice": I_CO_ICE,
    "c+": I_CP,
    "hco+": I_HCOP,
    "h2": I_H2,
    "h+": I_HP,
    "h3+": I_H3P,
    "h2+": I_H2P,
    "s+": I_SP,
    "si+": I_SIP,
    "o+": I_OP,
}

KB_CGS = 1.380649e-16
_YR_TO_S = float(Quantity("1 yr").to("s").magnitude)

_KPH_AVFAC = np.asarray([3.76, 2.12, 3.88, 2.66, 4.18, 3.10, 2.61], dtype=np.float64)
_KPH_BASE = np.asarray(
    [3.5e-10, 9.1e-10, 2.4e-10, 3.8e-10, 5.7e-11, 6.0e-10, 4.5e-9],
    dtype=np.float64,
)
KPH_C_BASE = float(_KPH_BASE[IPH_C])
KPH_CO_BASE = float(_KPH_BASE[IPH_CO])
KPH_H2_BASE = float(_KPH_BASE[IPH_H2])
_SIGMA_PE_CGS = 1.0e-21
_SIGMA_ISRF_CGS = 3.0e-22


def _resolve_local_chi_factor(cfg: dict, *, chi_is_incident: bool) -> float:
    """Return the UV normalization factor for the selected radiation mode."""
    if "local_chi_factor" in cfg:
        return float(cfg["local_chi_factor"])
    return 0.5 if bool(chi_is_incident) else 1.0


def _accumulate_solver_status(status_acc: np.ndarray, status_step: np.ndarray) -> np.ndarray:
    """Accumulate native solver statuses without erasing failures.

    Status priority is explicit: negative failure codes override everything;
    positive warning codes are kept only while a cell has no failure; zero means
    success and never clears a prior warning or failure.
    """
    acc = np.asarray(status_acc, dtype=np.int32)
    step = np.asarray(status_step, dtype=np.int32)
    if acc.shape != step.shape:
        raise ValueError(
            f"status shape mismatch: accumulator {acc.shape}, step {step.shape}"
        )
    failures = step < 0
    acc[failures] = step[failures]
    warnings = (acc == 0) & (step > 0)
    acc[warnings] = step[warnings]
    return acc


def _append_equilibrium_solver_diagnostics(hist: dict[str, list], result: dict) -> None:
    """Append native equilibrium solver diagnostic counters to histories."""
    hist["tevol_max_cells_hist"].append(int(result.get("tevol_max_cells", 0)))
    hist["tevol_max_residual_max_hist"].append(
        float(result.get("tevol_max_residual_max", 0.0))
    )
    hist["negative_abundance_cells_hist"].append(
        int(result.get("negative_abundance_cells", 0))
    )
    hist["negative_abundance_corrections_hist"].append(
        int(result.get("negative_abundance_corrections", 0))
    )
    hist["cvode_failure_cells_hist"].append(int(result.get("cvode_failure_cells", 0)))
    hist["exception_failure_cells_hist"].append(
        int(result.get("exception_failure_cells", 0))
    )


def _append_time_solver_diagnostics(hist: dict[str, list], result: dict) -> None:
    """Append native fixed-time solver diagnostic counters to histories."""
    hist["time_negative_abundance_cells_hist"].append(
        int(result.get("negative_abundance_cells", 0))
    )
    hist["time_negative_abundance_corrections_hist"].append(
        int(result.get("negative_abundance_corrections", 0))
    )
    hist["time_cvode_failure_cells_hist"].append(int(result.get("cvode_failure_cells", 0)))
    hist["time_exception_failure_cells_hist"].append(
        int(result.get("exception_failure_cells", 0))
    )


def _summarize_time_solver_diagnostics(hist: dict[str, list]) -> dict:
    """Return arrays plus totals/maxima for native fixed-time diagnostics."""
    out = {}
    for key, vals in hist.items():
        arr = np.asarray(vals, dtype=np.int64)
        out[key] = arr
        base = key.removesuffix("_hist")
        out[f"{base}_total"] = int(np.sum(arr)) if arr.size else 0
        out[f"{base}_max"] = int(np.max(arr)) if arr.size else 0
    return out


def _summarize_equilibrium_solver_diagnostics(hist: dict[str, list]) -> dict:
    """Return arrays plus totals/maxima for native equilibrium diagnostics."""
    out = {}
    for key, vals in hist.items():
        dtype = np.float64 if key.endswith("_residual_max_hist") else np.int64
        arr = np.asarray(vals, dtype=dtype)
        out[key] = arr

    for key in (
        "tevol_max_cells_hist",
        "negative_abundance_cells_hist",
        "negative_abundance_corrections_hist",
        "cvode_failure_cells_hist",
        "exception_failure_cells_hist",
    ):
        vals = np.asarray(hist.get(key, []), dtype=np.int64)
        base = key.removesuffix("_hist")
        out[f"{base}_total"] = int(np.sum(vals)) if vals.size else 0
        out[f"{base}_max"] = int(np.max(vals)) if vals.size else 0

    residual = np.asarray(hist.get("tevol_max_residual_max_hist", []), dtype=np.float64)
    out["tevol_max_residual_max"] = float(np.max(residual)) if residual.size else 0.0
    return out


def _warn_if_co_phase_settings_ignored(
    cfg: dict,
    *,
    enable_co_phase: bool,
    context: str,
) -> None:
    """Warn when CO phase settings are present but explicitly disabled."""
    if (not bool(enable_co_phase)) and ("co_phase" in cfg):
        logger.warning(
            "%s: [co_phase] settings are present but enable_co_phase=False; "
            "CO freeze-out/desorption is disabled.",
            context,
        )


def _resolve_temperature_config(cfg: dict) -> dict:
    """Resolve GOW17 gas-temperature mode and numerical controls."""
    temp_cfg = _nested_cfg(cfg, "temperature")
    if "mode" in temp_cfg:
        mode = str(temp_cfg["mode"]).lower()
    elif "const_temp" in cfg:
        mode = "constant_debug" if bool(cfg["const_temp"]) else "computed"
    else:
        mode = "computed"

    if mode not in ("computed", "dust", "constant_debug"):
        raise ValueError(
            "gow17 temperature mode must be 'computed', 'dust', or "
            f"'constant_debug', got {mode!r}"
        )

    return {
        "mode": mode,
        "const_temp": mode in ("dust", "constant_debug"),
        "constant_debug_Tgas": _maybe_quantity_to_float(
            temp_cfg.get("constant_debug_Tgas", "20 K"),
            "K",
        ),
        "Tgas_floor": _maybe_quantity_to_float(
            temp_cfg.get("Tgas_floor", "2.7 K"),
            "K",
        ),
        "Tgas_ceiling": _maybe_quantity_to_float(
            temp_cfg.get("Tgas_ceiling", "1.0e5 K"),
            "K",
        ),
        "max_thermal_iterations": int(temp_cfg.get("max_thermal_iterations", 50)),
        "thermal_rtol": float(temp_cfg.get("thermal_rtol", 1.0e-3)),
        "thermal_atol": _maybe_quantity_to_float(
            temp_cfg.get("thermal_atol", "1.0e-3 K"),
            "K",
        ),
    }


def _build_gow17_abstol(cfg: dict, abstol0: float) -> np.ndarray:
    """Build species-specific absolute tolerances for the native GOW17 solver."""
    tol_cfg = _nested_cfg(cfg, "tolerances")
    default = float(tol_cfg.get("abstol_default", abstol0))
    abstol = np.full(N_Y, default, dtype=np.float64)
    ion_tol = float(tol_cfg.get("abstol_e", default))
    for idx in (I_HEP, I_CP, I_HCOP, I_H3P, I_H2P, I_HP, I_SP, I_SIP, I_OP):
        abstol[idx] = ion_tol
    abstol[I_CO] = float(tol_cfg.get("abstol_CO", default))
    abstol[I_CO_ICE] = float(tol_cfg.get("abstol_CO_ice", default))
    abstol[I_CP] = float(tol_cfg.get("abstol_Cplus", default))
    abstol[I_HCOP] = float(tol_cfg.get("abstol_HCOplus", default))
    abstol[I_H2] = float(tol_cfg.get("abstol_H2", default))
    return abstol


def _cv_cold(xH2: np.ndarray, xe: np.ndarray) -> np.ndarray:
    """Heat capacity per H nucleus for cold gas (cgs: erg/K per H)."""
    return 1.5 * KB_CGS * ((1.0 - 2.0 * xH2) + xH2 + XHE + xe)


def _electron_abundance(y: np.ndarray) -> np.ndarray:
    """Sum of all ion abundances to get electron abundance."""
    return (
        y[..., I_HEP]
        + y[..., I_CP]
        + y[..., I_HCOP]
        + y[..., I_H3P]
        + y[..., I_H2P]
        + y[..., I_HP]
        + y[..., I_SP]
        + y[..., I_SIP]
        + y[..., I_OP]
    )


def _gow17_species_outputs(
    y_out: np.ndarray,
    nH_cm3: np.ndarray,
    *,
    x_h: np.ndarray,
    x_catom: np.ndarray,
    x_e: np.ndarray,
    line_h2_opr: float = 3.0,
) -> tuple[dict[str, Quantity], dict[str, Quantity]]:
    """Return abundance and number-density outputs for all GOW17 species."""

    opr = float(line_h2_opr)
    if not np.isfinite(opr) or opr <= 0.0:
        raise ValueError("line_h2_opr must be a positive finite number")

    abundances = {
        name: Quantity(y_out[..., idx], "dimensionless")
        for name, idx in GOW17_STATE_SPECIES.items()
    }
    abundances["h"] = Quantity(np.maximum(x_h, 0.0), "dimensionless")
    abundances["catom"] = Quantity(np.maximum(x_catom, 0.0), "dimensionless")
    abundances["e"] = Quantity(np.maximum(x_e, 0.0), "dimensionless")

    number_densities = {
        name: Quantity(q.magnitude * nH_cm3, "cm^-3")
        for name, q in abundances.items()
    }
    n_h2 = number_densities["h2"].to("cm^-3").magnitude
    number_densities["p-h2"] = Quantity(n_h2 / (1.0 + opr), "cm^-3")
    number_densities["o-h2"] = Quantity(n_h2 * opr / (1.0 + opr), "cm^-3")
    number_densities["he"] = Quantity(XHE * nH_cm3, "cm^-3")
    return abundances, number_densities


def _apply_astrochem_aitken_acceleration(
    *,
    y_state: np.ndarray,
    prev2: dict[int, np.ndarray],
    prev1: dict[int, np.ndarray],
    xCtot_flat: np.ndarray,
    enable_co_phase: bool,
    max_jump: float,
    floor: float,
) -> tuple[int, int]:
    """Apply guarded scalar Aitken acceleration to shielding-driving species."""
    species = [I_H2, I_CO]
    if enable_co_phase:
        species.append(I_CO_ICE)

    any_applied = np.zeros(y_state.shape[0], dtype=bool)
    applied_entries = 0
    min_den = max(float(floor), np.finfo(np.float64).tiny)
    max_jump = max(float(max_jump), 0.0)

    for species_idx in species:
        x0 = prev2[species_idx]
        x1 = prev1[species_idx]
        x2 = y_state[:, species_idx]

        denom = x2 - 2.0 * x1 + x0
        delta = x2 - x1
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            x_acc = x2 - (delta * delta) / denom

        upper = 0.5 if species_idx == I_H2 else xCtot_flat
        step_scale = np.maximum(np.abs(delta), min_den)
        valid = (
            np.isfinite(x_acc)
            & (np.abs(denom) > min_den)
            & (x_acc >= 0.0)
            & (x_acc <= upper)
            & (np.abs(x_acc - x2) <= max_jump * step_scale)
        )
        if not np.any(valid):
            continue

        x2[valid] = x_acc[valid]
        applied_entries += int(np.sum(valid))
        any_applied |= valid

    if enable_co_phase:
        co_total = y_state[:, I_CO] + y_state[:, I_CO_ICE]
        over = co_total > xCtot_flat
        if np.any(over):
            scale = np.divide(
                xCtot_flat[over],
                co_total[over],
                out=np.ones_like(co_total[over]),
                where=co_total[over] > 0.0,
            )
            y_state[over, I_CO] *= scale
            y_state[over, I_CO_ICE] *= scale
    else:
        y_state[:, I_CO_ICE] = 0.0

    return int(np.sum(any_applied)), int(applied_entries)


def _as_cgs_f64(q: Quantity, unit: str) -> np.ndarray:
    return np.ascontiguousarray(q.to(unit).magnitude, dtype=np.float64)


def _broadcast_scalar_or_array(val, ncells: int) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
    if arr.size == 1:
        return np.full(ncells, arr[0], dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(ncells), dtype=np.float64)


def _maybe_quantity_to_float(val, unit: str) -> float:
    if isinstance(val, str):
        return float(Quantity(val).to(unit).magnitude)
    return float(val)


def _nested_cfg(cfg: dict, key: str) -> dict:
    """Return a nested configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Model configuration.
    key : str
        Nested key to extract.

    Returns
    -------
    dict
        Nested dictionary, or an empty dictionary when absent.
    """
    val = cfg.get(key, {})
    return val if isinstance(val, dict) else {}


def _full_config_with_gow17_cfg(gow17_cfg: dict) -> dict:
    """Return current global config with the resolved GOW17 section injected."""
    full_cfg = dict(get_config())
    chemistry_cfg = dict(full_cfg.get("chemistry", {}))
    chemistry_cfg["gow17"] = dict(gow17_cfg)
    full_cfg["chemistry"] = chemistry_cfg
    return full_cfg


def _co_pdes_draine_flux() -> float:
    """Return configured Draine CO photodesorption photon flux."""
    uv_cfg = get_config().get("radmc3d", {}).get("uv_products", {})
    specs = uv_product_specs_from_config(uv_cfg)
    return float(
        draine_reference_for_product("F_CO_pdes_photon", specs)
        .to("1/(cm^2 s)")
        .magnitude
    )


def _actual_solver_uv_fields(
    *,
    Gph: np.ndarray,
    F_CO_pdes_external: np.ndarray,
    shape: tuple,
) -> dict[str, np.ndarray]:
    """Return UV fields exactly as passed to the native solver.

    ``Gph`` already includes local_chi_factor and molecular/atomic shielding.
    ``F_CO_pdes_external`` is the external CO photodesorption photon flux passed
    to the solver before CRUV/direct CR terms are added in CO phase diagnostics.
    """
    Gph_arr = np.asarray(Gph, dtype=np.float64)
    F_arr = np.asarray(F_CO_pdes_external, dtype=np.float64)
    return {
        "G_CO_diss_actual": Gph_arr[:, IPH_CO].reshape(shape),
        "G_C_ion_actual": Gph_arr[:, IPH_C].reshape(shape),
        "G_H2_diss_actual": Gph_arr[:, IPH_H2].reshape(shape),
        "F_CO_pdes_external_actual": F_arr.reshape(shape),
    }


def _resolve_co_dust_scalings(
    *,
    cfg: dict,
    sigma_d_cm2: np.ndarray,
    ncells: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resolve CO surface area and GOW17 grain abundance scalings.

    Parameters
    ----------
    cfg : dict
        GOW17 configuration dictionary.
    sigma_d_cm2 : ndarray
        Actual projected dust area per H in cm^2.
    ncells : int
        Number of cells.

    Returns
    -------
    sigma_d_CO_per_H, Zd_gow17_grain, sigma_d_ISM_ref : ndarray, ndarray, float
        Per-cell CO freeze-out surface area, per-cell GOW17 dust scaling, and
        reference ISM surface area.
    """
    dust_cfg = _nested_cfg(cfg, "dust")
    sigma_ref = _maybe_quantity_to_float(
        dust_cfg.get("sigma_d_ISM_ref", f"{SIGMA_D_ISM_REF:.16e} cm^2"),
        "cm^2",
    )
    if sigma_ref <= 0.0:
        raise ValueError("gow17 dust sigma_d_ISM_ref must be positive")

    sigma_source = str(dust_cfg.get("sigma_d_CO_per_H_source", "radmc")).lower()
    if sigma_source == "radmc":
        sigma_d_CO = np.ascontiguousarray(sigma_d_cm2, dtype=np.float64)
    elif sigma_source == "constant":
        sigma_const = _maybe_quantity_to_float(
            dust_cfg.get("sigma_d_CO_per_H_constant", f"{SIGMA_D_ISM_REF:.16e} cm^2"),
            "cm^2",
        )
        sigma_d_CO = np.full(ncells, float(sigma_const), dtype=np.float64)
    else:
        raise ValueError(
            "sigma_d_CO_per_H_source must be 'radmc' or 'constant', "
            f"got {sigma_source!r}"
        )

    if not np.all(np.isfinite(sigma_d_CO)):
        raise ValueError("sigma_d_CO_per_H contains non-finite values")
    sigma_d_CO = np.maximum(sigma_d_CO, 0.0)

    zd_source = str(dust_cfg.get("Zd_gow17_grain_source", "gas_dust_ratio")).lower()
    if zd_source == "constant":
        zd = np.full(
            ncells,
            float(dust_cfg.get("Zd_gow17_grain_constant", 1.0)),
            dtype=np.float64,
        )
    elif zd_source == "surface_area_relative":
        zd = np.divide(
            sigma_d_CO,
            sigma_ref,
            out=np.ones(ncells, dtype=np.float64),
            where=(sigma_ref > 0.0),
        )
    elif zd_source == "gas_dust_ratio":
        zd = np.full(
            ncells,
            float(dust_cfg.get("Zd_gow17_grain_constant", 1.0)),
            dtype=np.float64,
        )
    else:
        raise ValueError(
            "Zd_gow17_grain_source must be 'gas_dust_ratio', 'constant', "
            f"or 'surface_area_relative', got {zd_source!r}"
        )

    if not np.all(np.isfinite(zd)):
        raise ValueError("Zd_gow17_grain contains non-finite values")
    return sigma_d_CO, np.maximum(zd, 0.0), float(sigma_ref)


def _resolve_co_phase_controls(
    *,
    cfg: dict,
    ion_rate_arr: np.ndarray,
    ncells: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Resolve CO sticking, CRUV photodesorption, and CR desorption controls.

    Parameters
    ----------
    cfg : dict
        GOW17 configuration dictionary.
    ion_rate_arr : ndarray
        Cosmic-ray ionization rate per cell in s^-1.
    ncells : int
        Number of cells.

    Returns
    -------
    S_CO, F_CRUV_CO_pdes, k_crdes_CO : float, ndarray, ndarray
        Sticking coefficient, CRUV photon flux, and direct CR desorption rate.
    """
    co_cfg = _nested_cfg(cfg, "co_phase")
    s_mode = str(co_cfg.get("S_CO_mode", "constant")).lower()
    if s_mode != "constant":
        raise ValueError(f"Unsupported S_CO_mode={s_mode!r}")
    S_CO = float(co_cfg.get("S_CO", 1.0))
    if S_CO < 0.0:
        raise ValueError("S_CO must be non-negative")

    enable_cruv = bool(co_cfg.get("enable_cruv_pdes", True))
    if enable_cruv:
        F_ref = _maybe_quantity_to_float(
            co_cfg.get("F_CRUV_CO_pdes_ref", f"{F_CRUV_CO_PDES_REF:.16e} 1/(cm^2 s)"),
            "1/(cm^2 s)",
        )
        if bool(co_cfg.get("cruv_scales_with_zeta", True)):
            zeta_ref = _maybe_quantity_to_float(
                co_cfg.get("zeta_ref", f"{ZETA_CRUV_REF:.16e} 1/s"),
                "1/s",
            )
            scale = np.divide(
                ion_rate_arr,
                zeta_ref,
                out=np.zeros(ncells, dtype=np.float64),
                where=(zeta_ref > 0.0),
            )
            F_cruv = F_ref * scale
        else:
            F_cruv = np.full(ncells, F_ref, dtype=np.float64)
    else:
        F_cruv = np.zeros(ncells, dtype=np.float64)

    if bool(co_cfg.get("enable_crdes_CO", False)):
        k_crdes = _maybe_quantity_to_float(
            co_cfg.get("k_crdes_CO", f"{K_CRDES_CO:.16e} 1/s"),
            "1/s",
        )
        k_crdes_arr = np.full(ncells, k_crdes, dtype=np.float64)
    else:
        k_crdes_arr = np.zeros(ncells, dtype=np.float64)

    return float(S_CO), np.maximum(F_cruv, 0.0), np.maximum(k_crdes_arr, 0.0)




def _resolve_co_phase_runtime_params(cfg: dict) -> dict[str, float | int]:
    """Resolve CO phase physical parameters from runtime config.

    Import-time constants remain fallbacks, but sensitivity studies can now pass
    overrides through ``run_gow17(..., config=...)`` or ``Gow17TimeStepper``.
    """
    chemistry_cfg = get_config().get("chemistry", {})
    common_cfg = chemistry_cfg.get("common", {}) if isinstance(chemistry_cfg, dict) else {}
    co_cfg = _nested_cfg(cfg, "co_phase")

    def _lookup(names: tuple[str, ...], common_name: str, fallback):
        for name in names:
            if name in co_cfg:
                return co_cfg[name]
        if isinstance(common_cfg, dict) and common_name in common_cfg:
            return common_cfg[common_name]
        return fallback

    E_bind_CO = _maybe_quantity_to_float(
        _lookup(("E_bind_CO", "E_bind_co", "E_bind_co_K"), "E_bind_co", f"{float(E_BIND_CO):.16e} K"),
        "K",
    )
    nu0_CO = _maybe_quantity_to_float(
        _lookup(("nu0_CO", "nu0_co"), "nu0_co", f"{float(NU0_CO):.16e} 1/s"),
        "1/s",
    )
    Y_CO_local = float(_lookup(("Y_CO", "y_CO"), "Y_CO", float(Y_CO)))
    N_LAY_local = int(_lookup(("N_LAY", "n_lay"), "N_LAY", int(N_LAY)))
    N_SURF_local = _maybe_quantity_to_float(
        _lookup(("N_SURF", "n_surf", "N_surf"), "n_surf", f"{float(N_SURF):.16e} 1/cm^2"),
        "1/cm^2",
    )

    if E_bind_CO < 0.0 or nu0_CO < 0.0 or Y_CO_local < 0.0 or N_SURF_local < 0.0:
        raise ValueError("gow17 CO phase parameters must be non-negative")
    if N_LAY_local < 0:
        raise ValueError("gow17 CO phase N_LAY must be non-negative")

    return {
        "E_bind_CO": float(E_bind_CO),
        "nu0_CO": float(nu0_CO),
        "Y_CO": float(Y_CO_local),
        "N_LAY": int(N_LAY_local),
        "N_SURF": float(N_SURF_local),
    }


def _resolve_shielding_linewidth(cfg: dict, Tgas_K: np.ndarray) -> tuple[float, np.ndarray, dict]:
    """Resolve scalar Visser linewidth plus per-cell diagnostic linewidths."""
    shield_cfg = _nested_cfg(cfg, "shielding")
    mode = str(shield_cfg.get("b_CO_mode", "constant")).lower()
    b_const = float(shield_cfg.get("b_CO_constant_kms", cfg.get("b_kms", 0.3)))
    v_turb = float(shield_cfg.get("v_turb_kms", 0.03))
    b_min = float(shield_cfg.get("b_CO_min_kms", 0.03))
    b_max = float(shield_cfg.get("b_CO_max_kms", 3.0))
    if b_min <= 0.0 or b_max <= 0.0 or b_max < b_min:
        raise ValueError("gow17 shielding linewidth requires 0 < b_CO_min_kms <= b_CO_max_kms")

    T_arr = np.asarray(Tgas_K, dtype=np.float64)
    if mode == "constant":
        b_arr = np.full(T_arr.shape, b_const, dtype=np.float64)
    elif mode == "thermal+turbulent":
        mCO = 28.0 * float(M_H)
        with np.errstate(invalid="ignore", divide="ignore"):
            b_thermal = np.sqrt(2.0 * KB_CGS * np.maximum(T_arr, 0.0) / mCO) / 1.0e5
        b_arr = np.sqrt(np.maximum(b_thermal, 0.0) ** 2 + max(v_turb, 0.0) ** 2)
    else:
        raise ValueError(
            "gow17 shielding b_CO_mode must be 'constant' or 'thermal+turbulent', "
            f"got {mode!r}"
        )

    b_arr = np.clip(np.where(np.isfinite(b_arr), b_arr, b_const), b_min, b_max)
    finite = np.isfinite(b_arr)
    b_scalar = float(np.nanmedian(b_arr[finite])) if np.any(finite) else float(np.clip(b_const, b_min, b_max))
    b_scalar = float(np.clip(b_scalar, b_min, b_max))
    meta = {
        "b_CO_mode": mode,
        "b_CO_constant_kms": float(b_const),
        "v_turb_kms": float(v_turb),
        "b_CO_min_kms": float(b_min),
        "b_CO_max_kms": float(b_max),
        "b_CO_scalar_kms": float(b_scalar),
        "b_CO_scalar_approximation": bool(mode != "constant"),
        "b_CO_bins_used": [float(b_scalar)],
    }
    return b_scalar, b_arr, meta


def _resolve_visser_table_linewidth(
    b_kms: float,
    shielding_linewidth_meta: dict,
) -> tuple[float, VisserShielding, dict]:
    """Load the nearest Visser table and align the scalar shielding b value."""
    b_requested = float(b_kms)
    visser = VisserShielding(b_kms=b_requested)
    b_table = float(visser.b_kms)
    meta = dict(shielding_linewidth_meta)
    meta["b_CO_requested_scalar_kms"] = b_requested
    meta["b_CO_table_kms"] = b_table

    if abs(b_table - b_requested) > 1.0e-6:
        logger.warning(
            "gow17: requested CO shielding b_kms=%.6g km/s, but the nearest "
            "available Visser table is %.6g km/s; using the table value for "
            "CO/H2 shielding. Per-cell b_CO_kms is retained as a diagnostic.",
            b_requested,
            b_table,
        )
        meta["b_CO_scalar_approximation"] = True

    meta["b_CO_scalar_kms"] = b_table
    meta["b_CO_bins_used"] = [b_table]
    return b_table, visser, meta


def _dominant_process_id(rates: list[np.ndarray]) -> np.ndarray:
    """Return 1-based dominant process IDs, with 0 for cells with no positive rates."""
    if not rates:
        return np.zeros(0, dtype=np.int16)
    stack = np.stack([np.asarray(r, dtype=np.float64).reshape(-1) for r in rates], axis=0)
    finite = np.where(np.isfinite(stack), stack, -np.inf)
    ids = np.argmax(finite, axis=0).astype(np.int16) + 1
    ids[np.nanmax(np.where(np.isfinite(stack), stack, 0.0), axis=0) <= 0.0] = 0
    return ids


_THERMO_HEATING_NAMES = (
    "cosmic_ray",
    "photoelectric",
    "h2_grain",
    "h2_pump",
    "h2_diss",
)
_THERMO_COOLING_NAMES = (
    "cplus",
    "c",
    "o",
    "lya",
    "co_rot",
    "h2",
    "dust",
    "recombination",
    "h2_diss",
    "hi_ion",
)


def _empty_rhs_residual_diagnostics(shape: tuple, *, status: int, message: str) -> tuple[dict, dict]:
    nan = np.full(shape, np.nan, dtype=np.float64)
    fields = {
        "gow17_rhs_residual_max": Quantity(nan, "dimensionless"),
        "gow17_rhs_residual_CO": Quantity(nan, "dimensionless"),
        "gow17_rhs_residual_CO_ice": Quantity(nan, "dimensionless"),
        "gow17_rhs_residual_Cplus": Quantity(nan, "dimensionless"),
        "thermal_balance_residual": Quantity(nan, "erg/s"),
    }
    diag = {
        "gow17_rhs_status": int(status),
        "gow17_rhs_message": str(message),
        "gow17_rhs_residual_max_global": float("nan"),
    }
    return fields, diag


def _compute_rhs_residual_diagnostics(
    *,
    y_out: np.ndarray,
    nH_flat: np.ndarray,
    Tgas_flat: np.ndarray,
    Tdust_flat: np.ndarray,
    Zd_arr: np.ndarray,
    Zg_arr: np.ndarray,
    ion_rate_arr: np.ndarray,
    GPE: np.ndarray,
    GISRF: np.ndarray,
    Gph: np.ndarray,
    sigma_d_CO_per_H: np.ndarray,
    const_temp: bool,
    gradv_arr: np.ndarray,
    Leff_CO_max_arr: np.ndarray,
    isDust_cooling: bool,
    isCoolingCOThin: bool,
    fH2gr: float,
    fHplusgr: float,
    fCplusgr: float,
    fHeplusgr: float,
    fSplusgr: float,
    fSiplusgr: float,
    fCplusCR: float,
    co_phase_kw: dict,
    abstol: np.ndarray,
    reltol: float,
    t_resid: float,
    shape: tuple,
) -> tuple[dict, dict]:
    """Evaluate native RHS at the returned state and scale by solver tolerances."""
    if not hasattr(_gow17, "eval_rhs_batch"):
        return _empty_rhs_residual_diagnostics(
            shape,
            status=-2,
            message="native _gow17.eval_rhs_batch is unavailable; rebuild the extension",
        )

    y_flat = np.ascontiguousarray(np.asarray(y_out, dtype=np.float64).reshape(-1, N_Y))
    try:
        result = _gow17.eval_rhs_batch(
            y=y_flat,
            nH=np.ascontiguousarray(nH_flat, dtype=np.float64),
            Tgas=np.ascontiguousarray(Tgas_flat, dtype=np.float64),
            Tdust=np.ascontiguousarray(Tdust_flat, dtype=np.float64),
            Zd=np.ascontiguousarray(Zd_arr, dtype=np.float64),
            Zg=np.ascontiguousarray(Zg_arr, dtype=np.float64),
            ion_rate=np.ascontiguousarray(ion_rate_arr, dtype=np.float64),
            GPE=np.ascontiguousarray(GPE, dtype=np.float64),
            F_CO_pdes_photon=np.ascontiguousarray(GISRF, dtype=np.float64),
            Gph=np.ascontiguousarray(Gph, dtype=np.float64),
            sigma_d_CO_per_H=np.ascontiguousarray(sigma_d_CO_per_H, dtype=np.float64),
            const_temp=bool(const_temp),
            gradv=np.ascontiguousarray(gradv_arr, dtype=np.float64),
            Leff_CO_max=np.ascontiguousarray(Leff_CO_max_arr, dtype=np.float64),
            isDust_cooling=bool(isDust_cooling),
            isCoolingCOThin=bool(isCoolingCOThin),
            fH2gr=float(fH2gr),
            fHplusgr=float(fHplusgr),
            fCplusgr=float(fCplusgr),
            fHeplusgr=float(fHeplusgr),
            fSplusgr=float(fSplusgr),
            fSiplusgr=float(fSiplusgr),
            fCplusCR=float(fCplusCR),
            **co_phase_kw,
        )
    except Exception as exc:  # pragma: no cover - protects production runs from diagnostic failures
        return _empty_rhs_residual_diagnostics(shape, status=-1, message=str(exc))

    rhs = np.asarray(result.get("rhs"), dtype=np.float64).reshape(-1, N_Y)
    status = np.asarray(result.get("status", np.zeros(rhs.shape[0], dtype=np.int32)), dtype=np.int32)
    abstol_vec = np.asarray(abstol, dtype=np.float64).reshape(1, N_Y)
    scale = abstol_vec + float(reltol) * np.maximum(np.abs(y_flat), abstol_vec)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rhs_resid = np.abs(rhs) * max(float(t_resid), 0.0) / scale
    rhs_resid[~np.isfinite(rhs_resid)] = np.nan
    resid_max = np.nanmax(rhs_resid, axis=1)

    fields = {
        "gow17_rhs_residual_max": Quantity(resid_max.reshape(shape), "dimensionless"),
        "gow17_rhs_residual_CO": Quantity(rhs_resid[:, I_CO].reshape(shape), "dimensionless"),
        "gow17_rhs_residual_CO_ice": Quantity(rhs_resid[:, I_CO_ICE].reshape(shape), "dimensionless"),
        "gow17_rhs_residual_Cplus": Quantity(rhs_resid[:, I_CP].reshape(shape), "dimensionless"),
    }

    thermo = np.asarray(result.get("thermo_rates", np.zeros((rhs.shape[0], 15))), dtype=np.float64)
    thermo = thermo.reshape(rhs.shape[0], 15)
    heating = np.maximum(thermo[:, : len(_THERMO_HEATING_NAMES)], 0.0)
    cooling = np.maximum(thermo[:, len(_THERMO_HEATING_NAMES) :], 0.0)
    heating_total = np.sum(heating, axis=1)
    cooling_total = np.sum(cooling, axis=1)
    thermal_residual = rhs[:, I_E]
    fields["thermal_balance_residual"] = Quantity(thermal_residual.reshape(shape), "erg/s")
    fields["thermal_heating_total"] = Quantity(heating_total.reshape(shape), "erg/s")
    fields["thermal_cooling_total"] = Quantity(cooling_total.reshape(shape), "erg/s")

    for i, name in enumerate(_THERMO_HEATING_NAMES):
        fields[f"thermal_heating_{name}"] = Quantity(heating[:, i].reshape(shape), "erg/s")
    for i, name in enumerate(_THERMO_COOLING_NAMES):
        fields[f"thermal_cooling_{name}"] = Quantity(cooling[:, i].reshape(shape), "erg/s")

    heat_id = _dominant_process_id([heating[:, i] for i in range(heating.shape[1])])
    cool_id = _dominant_process_id([cooling[:, i] for i in range(cooling.shape[1])])
    fields["thermal_heating_dominant_id"] = Quantity(heat_id.reshape(shape), "dimensionless")
    fields["thermal_cooling_dominant_id"] = Quantity(cool_id.reshape(shape), "dimensionless")

    diag = {
        "gow17_rhs_status": int(0 if np.all(status == 0) else -1),
        "gow17_rhs_failure_cells": int(np.sum(status != 0)),
        "gow17_rhs_residual_max_global": float(np.nanmax(resid_max)) if resid_max.size else float("nan"),
        "gow17_rhs_residual_CO_max": float(np.nanmax(rhs_resid[:, I_CO])) if rhs_resid.size else float("nan"),
        "gow17_rhs_residual_CO_ice_max": float(np.nanmax(rhs_resid[:, I_CO_ICE])) if rhs_resid.size else float("nan"),
        "gow17_rhs_residual_Cplus_max": float(np.nanmax(rhs_resid[:, I_CP])) if rhs_resid.size else float("nan"),
        "rhs_residual_time_s": float(t_resid),
        "thermal_balance_residual_max_abs": float(np.nanmax(np.abs(thermal_residual))) if thermal_residual.size else float("nan"),
        "thermal_heating_process_id_map": {i + 1: name for i, name in enumerate(_THERMO_HEATING_NAMES)},
        "thermal_cooling_process_id_map": {i + 1: name for i, name in enumerate(_THERMO_COOLING_NAMES)},
    }
    return fields, diag

def _compute_co_phase_diagnostics(
    *,
    y_out: np.ndarray,
    nH_cm3: np.ndarray,
    Tgas_K: np.ndarray,
    Tdust_K: np.ndarray,
    sigma_d_CO_per_H: np.ndarray,
    S_CO: float,
    F_CO_pdes_photon: np.ndarray,
    F_CRUV_CO_pdes: np.ndarray,
    k_crdes_CO: np.ndarray,
    G_CO_diss_actual: np.ndarray,
    G_C_ion_actual: np.ndarray | None = None,
    G_H2_diss_actual: np.ndarray | None = None,
    enable_co_phase: bool,
    co_phase_params: dict[str, float | int] | None = None,
) -> dict:
    """Compute solver-consistent CO gas/ice phase-rate diagnostics."""
    shape = y_out.shape[:-1]
    ncells = int(np.prod(shape))

    params = co_phase_params or {}
    E_bind_CO_local = float(params.get("E_bind_CO", E_BIND_CO))
    nu0_CO_local = float(params.get("nu0_CO", NU0_CO))
    Y_CO_local = float(params.get("Y_CO", Y_CO))
    N_LAY_local = int(params.get("N_LAY", N_LAY))
    N_SURF_local = float(params.get("N_SURF", N_SURF))

    xCO = np.asarray(y_out[..., I_CO], dtype=np.float64).reshape(ncells)
    xCO_ice = np.asarray(y_out[..., I_CO_ICE], dtype=np.float64).reshape(ncells)
    xHeplus = np.asarray(y_out[..., I_HEP], dtype=np.float64).reshape(ncells)
    nH_flat = np.asarray(nH_cm3, dtype=np.float64).reshape(ncells)
    Tgas_flat = np.asarray(Tgas_K, dtype=np.float64).reshape(ncells)
    Tdust_flat = np.asarray(Tdust_K, dtype=np.float64).reshape(ncells)
    sigma_flat = np.asarray(sigma_d_CO_per_H, dtype=np.float64).reshape(ncells)
    F_star = np.asarray(F_CO_pdes_photon, dtype=np.float64).reshape(ncells)
    F_cruv = np.asarray(F_CRUV_CO_pdes, dtype=np.float64).reshape(ncells)
    k_crdes = np.asarray(k_crdes_CO, dtype=np.float64).reshape(ncells)
    Gco_actual = np.asarray(G_CO_diss_actual, dtype=np.float64).reshape(ncells)
    Gc_actual = (
        np.zeros(ncells, dtype=np.float64)
        if G_C_ion_actual is None
        else np.asarray(G_C_ion_actual, dtype=np.float64).reshape(ncells)
    )
    Gh2_actual = (
        np.zeros(ncells, dtype=np.float64)
        if G_H2_diss_actual is None
        else np.asarray(G_H2_diss_actual, dtype=np.float64).reshape(ncells)
    )

    if not enable_co_phase:
        sigma_flat = np.zeros(ncells, dtype=np.float64)
        F_cruv = np.zeros(ncells, dtype=np.float64)
        k_crdes = np.zeros(ncells, dtype=np.float64)
        E_bind_CO_local = 0.0
        nu0_CO_local = 0.0
        Y_CO_local = 0.0
        N_LAY_local = 0
        N_SURF_local = 0.0

    mCO = 28.0 * float(M_H)
    vth = np.sqrt(8.0 * KB_CGS * np.maximum(Tgas_flat, 0.0) / (np.pi * mCO))
    k_freeze = np.maximum(S_CO, 0.0) * sigma_flat * nH_flat * vth

    with np.errstate(over="ignore", invalid="ignore"):
        k_thermal = float(nu0_CO_local) * np.exp(
            -float(E_bind_CO_local) / np.maximum(Tdust_flat, 1.0e-30)
        )

    n_ice = xCO_ice * nH_flat
    n_ice_act_max = 4.0 * sigma_flat * nH_flat * float(N_SURF_local) * int(N_LAY_local)
    n_ice_act = np.minimum(np.maximum(n_ice, 0.0), np.maximum(n_ice_act_max, 0.0))
    f_active = np.divide(
        n_ice_act,
        n_ice,
        out=np.zeros(ncells, dtype=np.float64),
        where=(n_ice > 0.0),
    )
    F_total = np.maximum(F_star, 0.0) + np.maximum(F_cruv, 0.0)
    if int(N_LAY_local) > 0 and float(N_SURF_local) > 0.0:
        k_pd_surface = F_total * float(Y_CO_local) / (
            4.0 * float(N_SURF_local) * int(N_LAY_local)
        )
    else:
        k_pd_surface = np.zeros(ncells, dtype=np.float64)

    R_freeze_per_H = k_freeze * xCO
    R_pd_per_H = np.divide(
        k_pd_surface * n_ice_act,
        nH_flat,
        out=np.zeros(ncells, dtype=np.float64),
        where=(nH_flat > 0.0),
    )
    R_thermal_per_H = k_thermal * xCO_ice
    R_crdes_per_H = k_crdes * xCO_ice
    k_CO_photodiss = KPH_CO_BASE * Gco_actual
    k_C_photoion = KPH_C_BASE * Gc_actual
    k_H2_photodiss = KPH_H2_BASE * Gh2_actual
    R_photodiss_per_H = k_CO_photodiss * xCO
    R_heplus_per_H = 1.6e-9 * nH_flat * xHeplus * xCO
    R_gas_form_proxy_per_H = np.zeros(ncells, dtype=np.float64)
    total_co = xCO + xCO_ice
    co_gas_fraction = np.divide(
        xCO,
        total_co,
        out=np.zeros(ncells, dtype=np.float64),
        where=(total_co > 0.0),
    )
    co_ice_fraction = np.divide(
        xCO_ice,
        total_co,
        out=np.zeros(ncells, dtype=np.float64),
        where=(total_co > 0.0),
    )

    loss_id = _dominant_process_id(
        [R_freeze_per_H, R_photodiss_per_H, R_heplus_per_H]
    )
    gain_id = _dominant_process_id(
        [R_pd_per_H, R_thermal_per_H, R_crdes_per_H, R_gas_form_proxy_per_H]
    )
    c_reliability_flag = (
        (Tdust_flat < 25.0)
        & (co_ice_fraction > 0.9)
        & (G_C_ion_actual is not None)
        & (Gc_actual < 1.0e-8)
    ).astype(np.int16)

    return {
        "sigma_d_CO_per_H": Quantity(sigma_flat.reshape(shape), "cm^2"),
        "CO_sticking": Quantity(np.full(shape, float(S_CO), dtype=np.float64), "dimensionless"),
        "F_CRUV_CO_pdes": Quantity(F_cruv.reshape(shape), "1/(cm^2 s)"),
        "F_CO_pdes_external_actual": Quantity(F_star.reshape(shape), "1/(cm^2 s)"),
        "F_CO_pdes_photon_total": Quantity(F_total.reshape(shape), "1/(cm^2 s)"),
        "F_CO_pdes_total_actual": Quantity(F_total.reshape(shape), "1/(cm^2 s)"),
        "G_CO_diss_actual": Quantity(Gco_actual.reshape(shape), "dimensionless"),
        "G_C_ion_actual": Quantity(Gc_actual.reshape(shape), "dimensionless"),
        "G_H2_diss_actual": Quantity(Gh2_actual.reshape(shape), "dimensionless"),
        "k_CO_photodiss": Quantity(k_CO_photodiss.reshape(shape), "1/s"),
        "k_C_photoion": Quantity(k_C_photoion.reshape(shape), "1/s"),
        "k_H2_photodiss": Quantity(k_H2_photodiss.reshape(shape), "1/s"),
        "k_CO_freezeout": Quantity(k_freeze.reshape(shape), "1/s"),
        "k_CO_thermal_desorption": Quantity(k_thermal.reshape(shape), "1/s"),
        "k_CO_pdes_surface": Quantity(k_pd_surface.reshape(shape), "1/s"),
        "k_CO_crdes": Quantity(k_crdes.reshape(shape), "1/s"),
        "n_CO_ice_active_max": Quantity(n_ice_act_max.reshape(shape), "cm^-3"),
        "n_CO_ice_active": Quantity(n_ice_act.reshape(shape), "cm^-3"),
        "CO_ice_active_fraction": Quantity(f_active.reshape(shape), "dimensionless"),
        "CO_loss_freezeout_per_H": Quantity(R_freeze_per_H.reshape(shape), "1/s"),
        "CO_loss_photodiss_per_H": Quantity(R_photodiss_per_H.reshape(shape), "1/s"),
        "CO_loss_Heplus_per_H": Quantity(R_heplus_per_H.reshape(shape), "1/s"),
        "CO_gain_photodesorption_per_H": Quantity(R_pd_per_H.reshape(shape), "1/s"),
        "CO_gain_thermal_desorption_per_H": Quantity(R_thermal_per_H.reshape(shape), "1/s"),
        "CO_gain_crdes_per_H": Quantity(R_crdes_per_H.reshape(shape), "1/s"),
        "CO_gain_gas_phase_proxy_per_H": Quantity(R_gas_form_proxy_per_H.reshape(shape), "1/s"),
        "CO_loss_dominant_id": Quantity(loss_id.reshape(shape), "dimensionless"),
        "CO_gain_dominant_id": Quantity(gain_id.reshape(shape), "dimensionless"),
        "CO_gas_fraction": Quantity(co_gas_fraction.reshape(shape), "dimensionless"),
        "CO_ice_fraction": Quantity(co_ice_fraction.reshape(shape), "dimensionless"),
        "C_abundance_interpretation_flag": Quantity(c_reliability_flag.reshape(shape), "dimensionless"),
    }


def _compute_shielding_and_gph(
    *,
    y_flat: np.ndarray,
    nH_flat: np.ndarray,
    chi_dust_flat: np.ndarray,
    G_CO_diss_flat: np.ndarray,
    G_H2_diss_flat: np.ndarray,
    G_C_ion_flat: np.ndarray,
    G_CO_pdes_flat: np.ndarray,
    F_CO_pdes_photon_flat: np.ndarray | None,
    xCtot_flat: np.ndarray,
    Zd_arr: np.ndarray,
    Av_flat,
    shape: tuple,
    ncells: int,
    rad: "RadModel",
    nH_cm3: np.ndarray,
    chi_dust_arr: np.ndarray,
    visser,
    b_kms: float,
    nside: int,
    chi_is_incident: bool,
    local_chi_factor: float,
    shielding_outer_1d: str,
) -> tuple:
    """Compute shielding factors and radiation field arrays from current abundances.

    Returns
    -------
    theta_h2_flat, theta_co_flat, theta_c_flat : np.ndarray
        Shielding factors, shape ``(ncells,)``.
    Gph, GPE, GISRF : np.ndarray
        Radiation field arrays ready for the batch solver.
    """
    xCO = y_flat[:, I_CO]
    xCO_ice = y_flat[:, I_CO_ICE]
    xH2 = y_flat[:, I_H2]

    xC_neutral = xCtot_flat - (
        y_flat[:, I_HCOP]
        + y_flat[:, I_CHX]
        + xCO
        + xCO_ice
        + y_flat[:, I_CP]
    )
    xC_neutral = np.maximum(xC_neutral, 0.0)

    nCO_cm3 = np.ascontiguousarray(
        (xCO * nH_flat).reshape(shape), dtype=np.float64
    )
    nH2_cm3 = np.ascontiguousarray(
        (xH2 * nH_flat).reshape(shape), dtype=np.float64
    )
    nC_cm3 = np.ascontiguousarray(
        (xC_neutral * nH_flat).reshape(shape), dtype=np.float64
    )

    if is_effectively_1d(rad.model.mesh, nH_cm3.shape):
        if chi_is_incident:
            if Av_flat is None:
                raise RuntimeError("gow17: internal error (Av_flat missing)")

            NH_over_Zd = Av_flat * (1.87e21)
            NH_flat_loc = np.divide(
                NH_over_Zd,
                Zd_arr,
                out=np.zeros_like(NH_over_Zd, dtype=np.float64),
                where=(Zd_arr > 0.0),
            )

            dNH = np.empty_like(NH_flat_loc, dtype=np.float64)
            dNH[:-1] = NH_flat_loc[1:] - NH_flat_loc[:-1]
            dNH[-1] = 0.0

            xH2_use = np.ascontiguousarray(xH2, dtype=np.float64)
            xCO_use = np.ascontiguousarray(xCO, dtype=np.float64)
            xC_use = np.ascontiguousarray(xC_neutral, dtype=np.float64)

            N_H2 = np.zeros_like(NH_flat_loc, dtype=np.float64)
            N_CO = np.zeros_like(NH_flat_loc, dtype=np.float64)
            N_C = np.zeros_like(NH_flat_loc, dtype=np.float64)
            if NH_flat_loc.size >= 2:
                N_H2[1:] = np.cumsum(xH2_use[:-1] * dNH[:-1])
                N_CO[1:] = np.cumsum(xCO_use[:-1] * dNH[:-1])
                N_C[1:] = np.cumsum(xC_use[:-1] * dNH[:-1])

            from diskbridge.chemistry.shielding.h2_db96 import h2_self_shielding_db96

            theta_h2_flat = h2_self_shielding_db96(N_H2, b5=float(b_kms))
            theta_co_flat = visser.theta("co", N_CO, N_H2, b_kms=float(b_kms))

            AH2 = 1.17e-8
            tau_H2 = 1.2e-14 * 2.0 * N_H2
            y_shield = AH2 * tau_H2
            ry = np.exp(-y_shield) / (1.0 + y_shield)
            rc = np.exp(-1.6e-17 * N_C)
            theta_c_flat = rc * ry

            theta_h2_arr = theta_h2_flat.reshape(shape)
            theta_co_arr = theta_co_flat.reshape(shape)
            theta_c_arr = theta_c_flat.reshape(shape)
        else:
            theta_h2_arr, theta_co_arr, theta_c_arr, _, _ = compute_pdr_shielding_1d(
                mesh=rad.model.mesh,
                nH=nH_cm3,
                chi=chi_dust_arr,
                visser=visser,
                nCO=nCO_cm3,
                nC=nC_cm3,
                nH2=nH2_cm3,
                b_kms=b_kms,
                outer=shielding_outer_1d,
            )
    else:
        from diskbridge.chemistry.shielding.healpix_columns import (
            compute_pdr_shielding_healpix,
        )

        W_rays = getattr(rad, "W_rays", None)
        theta_h2_arr, theta_co_arr, theta_c_arr, _, _ = compute_pdr_shielding_healpix(
            mesh=rad.model.mesh,
            nH=nH_cm3,
            chi=chi_dust_arr,
            visser=visser,
            nCO=nCO_cm3,
            nC=nC_cm3,
            nH2=nH2_cm3,
            nside=nside,
            b_kms=b_kms,
            W_rays=W_rays,
        )

    theta_h2_flat = theta_h2_arr.reshape(ncells)
    theta_co_flat = theta_co_arr.reshape(ncells)
    theta_c_flat = theta_c_arr.reshape(ncells)

    # -- Assemble radiation field arrays --
    Gph = np.empty((ncells, N_PH), dtype=np.float64)
    if chi_is_incident:
        if Av_flat is None:
            raise RuntimeError("gow17: internal error (Av_flat missing)")
        G0_local = float(local_chi_factor) * chi_dust_flat
        Gph[:, :] = G0_local[:, None] * np.exp(-_KPH_AVFAC[None, :] * Av_flat[:, None])
        Gph[:, IPH_CO] = (
            float(local_chi_factor)
            * G_CO_diss_flat
            * np.exp(-float(_KPH_AVFAC[IPH_CO]) * Av_flat)
        )
        Gph[:, IPH_H2] = (
            float(local_chi_factor)
            * G_H2_diss_flat
            * np.exp(-float(_KPH_AVFAC[IPH_H2]) * Av_flat)
        )
        Gph[:, IPH_C] = (
            float(local_chi_factor)
            * G_C_ion_flat
            * np.exp(-float(_KPH_AVFAC[IPH_C]) * Av_flat)
        )
        NH_over_Zd = Av_flat * (1.87e21)
        NH_flat_loc = np.divide(
            NH_over_Zd,
            Zd_arr,
            out=np.zeros_like(NH_over_Zd, dtype=np.float64),
            where=(Zd_arr > 0.0),
        )
        GPE = np.ascontiguousarray(
            G0_local * np.exp(-NH_flat_loc * _SIGMA_PE_CGS * Zd_arr),
            dtype=np.float64,
        )
        if F_CO_pdes_photon_flat is None:
            F_CO_pdes_photon_flat = G_CO_pdes_flat * float(F_DRAINE)
        GISRF = np.ascontiguousarray(
            float(local_chi_factor)
            * F_CO_pdes_photon_flat
            * np.exp(-NH_flat_loc * _SIGMA_ISRF_CGS * Zd_arr),
            dtype=np.float64,
        )
    else:
        G_broad = local_chi_factor * chi_dust_flat
        Gph[:, :] = G_broad[:, None]
        Gph[:, IPH_CO] = local_chi_factor * G_CO_diss_flat
        Gph[:, IPH_H2] = local_chi_factor * G_H2_diss_flat
        Gph[:, IPH_C] = local_chi_factor * G_C_ion_flat
        GPE = np.ascontiguousarray(G_broad.copy(), dtype=np.float64)
        if F_CO_pdes_photon_flat is None:
            f_co_pdes_ref = _co_pdes_draine_flux()
            GISRF = np.ascontiguousarray(
                local_chi_factor * G_CO_pdes_flat * f_co_pdes_ref,
                dtype=np.float64,
            )
        else:
            GISRF = np.ascontiguousarray(
                local_chi_factor * F_CO_pdes_photon_flat,
                dtype=np.float64,
            )

    Gph[:, IPH_C] *= theta_c_flat
    Gph[:, IPH_CO] *= theta_co_flat
    Gph[:, IPH_H2] *= theta_h2_flat

    return theta_h2_flat, theta_co_flat, theta_c_flat, Gph, GPE, GISRF


def run_gow17(rad: "RadModel", config: dict) -> ChemistryResult:
    cfg = resolve_model_config(("chemistry", "gow17"), overrides=config)

    nside = int(diskbridge.params.nside)
    b_kms = float(cfg.get("b_kms", 0.3))

    ion_rate_s = Quantity(cfg.get("ion_rate", "2e-16 s^-1")).to("1/s").magnitude

    Zg = float(cfg.get("Zg", 1.0))
    Zd_mode = str(cfg.get("Zd_mode", "scalar"))

    enable_co_phase = bool(cfg.get("enable_co_phase", True))
    _warn_if_co_phase_settings_ignored(
        cfg,
        enable_co_phase=enable_co_phase,
        context="gow17",
    )

    fH2gr = float(cfg.get("fH2gr", 1.0))
    fHplusgr = float(cfg.get("fHplusgr", 1.0))
    fCplusgr = float(cfg.get("fCplusgr", 1.0))
    fHeplusgr = float(cfg.get("fHeplusgr", 1.0))
    fSplusgr = float(cfg.get("fSplusgr", 1.0))
    fSiplusgr = float(cfg.get("fSiplusgr", 1.0))
    fCplusCR = float(cfg.get("fCplusCR", 1.0))

    gradv_scalar = float(cfg.get("gradv", 1.0e-14))
    gradv_mode = str(cfg.get("gradv_mode", "scalar"))
    gradv_q = float(cfg.get("gradv_q", 1.5))
    gradv_N0 = float(cfg.get("gradv_N0", 1e21))
    gradv_p = float(cfg.get("gradv_p", 1.0))
    gradv_f_corr = float(cfg.get("gradv_f_corr", 4.0))
    gradv_gmin = float(cfg.get("gradv_gmin", 1e-20))
    gradv_gmax = float(cfg.get("gradv_gmax", 1e-8))

    Leff_CO_max_mode = str(cfg.get("Leff_CO_max_mode", "scalar"))
    L_geo_reduction = str(cfg.get("L_geo_reduction", "percentile_20"))
    L_geo_min = float(cfg.get("L_geo_min", 1e10))
    L_geo_max = float(cfg.get("L_geo_max", 1e20))
    Leff_CO_max_scalar = float(cfg.get("Leff_CO_max", 3.0e20))
    isDust_cooling = bool(cfg.get("isDust_cooling", True))
    isCoolingCOThin = bool(cfg.get("isCoolingCOThin", False))
    temperature_cfg = _resolve_temperature_config(cfg)
    temperature_mode = str(temperature_cfg["mode"])
    const_temp = bool(temperature_cfg["const_temp"])
    if temperature_mode == "constant_debug":
        logger.warning(
            "gow17: using constant_debug Tgas; this is for debugging only "
            "and is not a physically valid disc/PDR temperature model."
        )

    reltol = float(cfg.get("reltol", 1e-4))
    abstol0 = float(cfg.get("abstol0", 1e-15))
    tolfac = float(cfg.get("tolfac", 10.0))
    tmin = _maybe_quantity_to_float(cfg.get("tmin", 3.16e10), "s")

    tmax_val = cfg.get("tmax", None)
    if tmax_val is None:
        tmax_val = cfg.get("t_end", 3.16e14)
    tmax = _maybe_quantity_to_float(tmax_val, "s")
    mxsteps = int(cfg.get("mxsteps", 10000))
    maxord = int(cfg.get("maxord", 5))
    userJac = bool(cfg.get("userJac", False))
    verbose = bool(cfg.get("verbose", False))

    shielding_max_iter_cfg = cfg.get("shielding_max_iter", None)
    if shielding_max_iter_cfg is None:
        shielding_max_iter_cfg = cfg.get("max_iter", 10)
    shielding_max_iter = int(shielding_max_iter_cfg)
    shielding_reltol = float(cfg.get("shielding_reltol", 1e-3))
    shielding_abstol = float(cfg.get("shielding_abstol", 1e-15))
    shielding_outer_1d = str(cfg.get("shielding_outer_1d", "min"))
    tgas_convergence_reltol = float(cfg.get("tgas_convergence_reltol", shielding_reltol))

    # -- AstroChem-style coupling config --
    coupling_mode = str(cfg.get("coupling_mode", "fixed_point"))
    astrochem_n_updates = int(cfg.get("astrochem_n_updates", 5))
    astrochem_t_end_yr = float(cfg.get("astrochem_t_end_yr", 1.0e6))
    astrochem_acceleration = str(cfg.get("astrochem_acceleration", "none")).lower()
    astrochem_acceleration_start = int(cfg.get("astrochem_acceleration_start", 3))
    astrochem_acceleration_max_jump = float(cfg.get("astrochem_acceleration_max_jump", 2.0))
    if astrochem_acceleration not in ("none", "aitken"):
        raise ValueError(
            "astrochem_acceleration must be 'none' or 'aitken', "
            f"got {astrochem_acceleration!r}"
        )

    chi_is_incident = bool(cfg.get("chi_is_incident", False))
    incident_uv_products = bool(cfg.get("incident_uv_products", False))
    local_chi_factor = _resolve_local_chi_factor(
        cfg,
        chi_is_incident=chi_is_incident,
    )
    slab_1d_equilibrium = bool(cfg.get("slab_1d_equilibrium", False))
    runtime_mode = None
    if not chi_is_incident:
        runtime_mode = validate_uv_chemistry_config(_full_config_with_gow17_cfg(cfg))
    elif rad.has_uv_product("G_CO_diss") and not incident_uv_products:
        raise ValueError(
            "gow17: chi_is_incident=True is incompatible with registered UV "
            "product fields unless incident_uv_products=True"
        )

    Tdust = rad.ensure_dust_temperature()
    nH = rad.ensure_nH()
    if temperature_mode == "constant_debug":
        Tgas = Quantity(
            np.full(
                _as_cgs_f64(Tdust, "K").shape,
                float(temperature_cfg["constant_debug_Tgas"]),
                dtype=np.float64,
            ),
            "K",
        )
    elif temperature_mode == "dust":
        Tgas = Tdust
    else:
        Tgas = rad.ensure_gas_temperature()
        if Tgas is None:
            Tgas = Tdust

    f_co_pdes_ref = _co_pdes_draine_flux()
    if chi_is_incident:
        if incident_uv_products:
            chi = rad.ensure_uv_product("chi_broad", fallback_to_chi=False)
            G_CO_diss = rad.ensure_uv_product("G_CO_diss", fallback_to_chi=False)
            G_H2_diss = rad.ensure_uv_product("G_H2_diss", fallback_to_chi=False)
            G_C_ion = rad.ensure_uv_product("G_C_ion", fallback_to_chi=False)
            G_CO_pdes = rad.ensure_uv_product("G_CO_pdes", fallback_to_chi=False)
            F_CO_pdes_photon = rad.ensure_uv_product(
                "F_CO_pdes_photon",
                fallback_to_chi=False,
            )
        else:
            chi = rad.ensure_chi()
            G_CO_diss = chi
            G_H2_diss = chi
            G_C_ion = chi
            G_CO_pdes = chi
            F_CO_pdes_photon = Quantity(
                chi.to("dimensionless").magnitude * float(F_DRAINE),
                "1/(cm^2 s)",
            )
    elif runtime_mode is not None and runtime_mode.products_enabled:
        chi = rad.ensure_uv_product("chi_broad", fallback_to_chi=False)
        G_CO_diss = rad.ensure_uv_product("G_CO_diss", fallback_to_chi=False)
        G_H2_diss = rad.ensure_uv_product("G_H2_diss", fallback_to_chi=False)
        G_C_ion = rad.ensure_uv_product("G_C_ion", fallback_to_chi=False)
        G_CO_pdes = rad.ensure_uv_product("G_CO_pdes", fallback_to_chi=False)
        F_CO_pdes_photon = rad.ensure_uv_product(
            "F_CO_pdes_photon",
            fallback_to_chi=False,
        )
    else:
        chi = rad.ensure_chi()
        G_CO_diss = chi
        G_H2_diss = chi
        G_C_ion = chi
        G_CO_pdes = chi
        F_CO_pdes_photon = Quantity(
            G_CO_pdes.to("dimensionless").magnitude * f_co_pdes_ref,
            "1/(cm^2 s)",
        )

    # Pre-compute directional UV weights (W_rays) once for reuse across
    # shielding iterations.  Returns None for 1-D meshes or when dustkappa
    # files are unavailable (shielding then falls back to isotropic averaging).
    if chi_is_incident:
        directional_uv_product = "chi"
    else:
        directional_uv_product = (
            "G_CO_diss"
            if runtime_mode is None or runtime_mode.use_hard_directional_weights
            else "chi_broad"
        )
    maybe_ensure_W_rays(rad, nside=nside, uv_product=directional_uv_product)

    nH_cm3 = _as_cgs_f64(nH, "cm^-3")
    T_K = np.clip(
        _as_cgs_f64(Tgas, "K"),
        float(temperature_cfg["Tgas_floor"]),
        float(temperature_cfg["Tgas_ceiling"]),
    )
    chi_dust_arr = _as_cgs_f64(chi, "dimensionless")
    G_CO_diss_arr = _as_cgs_f64(G_CO_diss, "dimensionless")
    G_H2_diss_arr = _as_cgs_f64(G_H2_diss, "dimensionless")
    G_C_ion_arr = _as_cgs_f64(G_C_ion, "dimensionless")
    G_CO_pdes_arr = _as_cgs_f64(G_CO_pdes, "dimensionless")
    F_CO_pdes_photon_arr = _as_cgs_f64(F_CO_pdes_photon, "1/(cm^2 s)")

    shape = nH_cm3.shape
    ncells = nH_cm3.size

    Tdust_K = _as_cgs_f64(Tdust, "K")
    b_kms, b_CO_kms_arr, shielding_linewidth_meta = _resolve_shielding_linewidth(cfg, T_K)

    nH_flat = nH_cm3.reshape(ncells)
    T_flat = T_K.reshape(ncells)
    Tdust_flat = Tdust_K.reshape(ncells)
    chi_dust_flat = chi_dust_arr.reshape(ncells)
    G_CO_diss_flat = G_CO_diss_arr.reshape(ncells)
    G_H2_diss_flat = G_H2_diss_arr.reshape(ncells)
    G_C_ion_flat = G_C_ion_arr.reshape(ncells)
    G_CO_pdes_flat = G_CO_pdes_arr.reshape(ncells)
    F_CO_pdes_photon_flat = F_CO_pdes_photon_arr.reshape(ncells)

    Av_flat = None
    if chi_is_incident:
        if not is_effectively_1d(rad.model.mesh, nH_cm3.shape):
            raise ValueError("gow17: chi_is_incident=True requires an effectively-1D mesh")
        Av_q = getattr(rad, "Av", None)
        if Av_q is None:
            raise ValueError("gow17: chi_is_incident=True requires rad.Av to be set")
        Av_arr = _as_cgs_f64(Av_q, "dimensionless")
        if Av_arr.shape != shape:
            raise ValueError(f"gow17: rad.Av shape {Av_arr.shape} does not match nH shape {shape}")
        Av_flat = Av_arr.reshape(ncells)

    if slab_1d_equilibrium:
        if not chi_is_incident:
            raise ValueError("gow17: slab_1d_equilibrium requires chi_is_incident=True")
        if not is_effectively_1d(rad.model.mesh, nH_cm3.shape):
            raise ValueError("gow17: slab_1d_equilibrium requires an effectively-1D mesh")
        if Av_flat is None:
            raise RuntimeError("gow17: internal error (Av_flat missing)")

        NH_total_cfg = cfg.get("NH_total", None)
        NH_min_cfg = cfg.get("NH_min", None)
        if NH_total_cfg is None or NH_min_cfg is None:
            raise ValueError("gow17: slab_1d_equilibrium requires NH_total and NH_min in config")

        NH_total = _maybe_quantity_to_float(NH_total_cfg, "cm^-2")
        NH_min = _maybe_quantity_to_float(NH_min_cfg, "cm^-2")
        logNH = bool(cfg.get("logNH", True))
        field_geo = int(cfg.get("field_geo", 0))
        isdust = bool(cfg.get("isdust", True))
        isfsH2 = bool(cfg.get("isfsH2", True))
        isfsCO = bool(cfg.get("isfsCO", True))
        isfsC = bool(cfg.get("isfsC", True))

        nH0 = float(nH_flat[0])
        if not np.allclose(nH_flat, nH0, rtol=0.0, atol=0.0):
            raise ValueError("gow17: slab_1d_equilibrium requires uniform nH")

        chi0 = float(chi_dust_flat[0])
        if not np.allclose(chi_dust_flat, chi0, rtol=0.0, atol=0.0):
            raise ValueError("gow17: slab_1d_equilibrium requires uniform chi")

        sigma_d = rad.ensure_sigma_d_per_H()
        sigma_d_cm2 = _as_cgs_f64(sigma_d, "cm^2").reshape(ncells)
        valid = np.isfinite(sigma_d_cm2) & (sigma_d_cm2 > 0.0)
        if not np.any(valid):
            raise ValueError("gow17: slab_1d_equilibrium requires positive finite sigma_d_per_H")
        sigma_d_per_H_ref = float(np.nanmedian(sigma_d_cm2[valid]))
        Zd0 = float(np.nanmedian(sigma_d_cm2[valid]) / sigma_d_per_H_ref)

        Zg0 = float(Zg)
        ion0 = float(ion_rate_s)
        gradv0 = float(gradv_scalar)

        NCOeff_global = bool(cfg.get("NCOeff_global", True))
        bCO_L = bool(cfg.get("bCO_L", True))

        y0 = np.zeros(N_Y, dtype=np.float64)
        y0[I_HEP] = 1.450654e-08
        y0[I_H3P] = 2.681411e-07
        y0[I_CP] = 1.0e-4
        y0[I_CO] = 1.0e-7
        y0[I_H2] = 0.1
        y0[I_CO_ICE] = 0.0
        if not const_temp:
            Cv0 = _cv_cold(
                np.asarray([y0[I_H2]], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
            )[0]
            y0[I_E] = Cv0 * float(T_flat[0])

        abstol = _build_gow17_abstol(cfg, abstol0)
        abstol[I_E] = float(
            _cv_cold(
                np.asarray([0.1], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
            )[0]
        )

        slab = _gow17.solve_slab_1d_equilibrium(
            nH=nH0,
            G0=(2.0 * chi0),
            ngrid=int(ncells),
            NH_total=float(NH_total),
            logNH=bool(logNH),
            NH_min=float(NH_min),
            field_geo=int(field_geo),
            isdust=bool(isdust),
            isfsH2=bool(isfsH2),
            isfsCO=bool(isfsCO),
            isfsC=bool(isfsC),
            Zg=float(Zg0),
            Zd=float(Zd0),
            ion_rate=float(ion0),
            reltol=float(reltol),
            abstol=abstol,
            mxsteps=int(mxsteps),
            maxord=int(maxord),
            tolfac=float(tolfac),
            tmin=float(tmin),
            tmax=float(tmax),
            verbose=bool(verbose),
            y0=y0,
            const_temp=bool(const_temp),
            Tgas=float(T_flat[0]),
            gradv=float(gradv0),
            NCOeff_global=bool(NCOeff_global),
            bCO_L=bool(bCO_L),
            fH2gr=float(fH2gr),
            fHplusgr=float(fHplusgr),
            fCplusgr=float(fCplusgr),
            fHeplusgr=float(fHeplusgr),
            fSplusgr=float(fSplusgr),
            fSiplusgr=float(fSiplusgr),
            fCplusCR=float(fCplusCR),
            co_sigma_d_per_H_ref=0.0,
            co_E_bind_co=0.0,
            co_nu0_co=0.0,
            co_F_DRAINE=0.0,
            co_Y_CO=0.0,
            co_N_SURF=0.0,
            co_N_LAY=0,
            userJac=bool(userJac),
        )

        y_guess = np.asarray(slab["y"], dtype=np.float64).reshape(ncells, N_Y)
        theta_h2_arr = np.asarray(slab["fShieldH2"], dtype=np.float64).reshape(shape)
        theta_co_arr = np.asarray(slab["fShieldCO"], dtype=np.float64).reshape(shape)
        theta_c_arr = np.ones_like(theta_co_arr, dtype=np.float64)

        y_out = y_guess.reshape(shape + (N_Y,))
        status = np.zeros(shape, dtype=np.int32)

        xCO = y_out[..., I_CO]
        xCO_ice = y_out[..., I_CO_ICE]
        xH2 = y_out[..., I_H2]

        nco_gas = Quantity(xCO * nH_cm3, "cm^-3")
        nco_ice = Quantity(xCO_ice * nH_cm3, "cm^-3")
        nH2_out = Quantity(xH2 * nH_cm3, "cm^-3")

        xe = _electron_abundance(y_out)
        if not const_temp:
            Cv_slab = _cv_cold(y_out[..., I_H2], xe)
            with np.errstate(divide="ignore", invalid="ignore"):
                T_out_slab = y_out[..., I_E] / Cv_slab
            T_out_slab = np.clip(
                T_out_slab,
                float(temperature_cfg["Tgas_floor"]),
                float(temperature_cfg["Tgas_ceiling"]),
            )
        else:
            T_out_slab = T_flat.reshape(shape)
        T_status_slab = np.zeros(shape, dtype=np.int32)

        xH_atom = 1.0 - (
            y_out[..., I_OHX]
            + y_out[..., I_CHX]
            + y_out[..., I_HCOP]
            + 3.0 * y_out[..., I_H3P]
            + 2.0 * y_out[..., I_H2P]
            + y_out[..., I_HP]
            + 2.0 * y_out[..., I_H2]
        )

        xCtot = np.full(shape, float(Zg0) * float(XC_STD), dtype=np.float64)
        xC_neutral = xCtot - (
            y_out[..., I_HCOP]
            + y_out[..., I_CHX]
            + y_out[..., I_CO]
            + y_out[..., I_CO_ICE]
            + y_out[..., I_CP]
        )

        nCplus = Quantity(y_out[..., I_CP] * nH_cm3, "cm^-3")
        nC = Quantity(np.maximum(xC_neutral, 0.0) * nH_cm3, "cm^-3")
        ne = Quantity(xe * nH_cm3, "cm^-3")
        nH_atom = Quantity(np.maximum(xH_atom, 0.0) * nH_cm3, "cm^-3")

        abundances, number_densities = _gow17_species_outputs(
            y_out,
            nH_cm3,
            x_h=xH_atom,
            x_catom=xC_neutral,
            x_e=xe,
            line_h2_opr=cfg.get("line_h2_opr", 3.0),
        )

        fields = {
            "co_ice": nco_ice,
            "Tgas": Quantity(T_out_slab, "K"),
            "Tgas_minus_Tdust": Quantity(T_out_slab - Tdust_flat.reshape(shape), "K"),
            "Tgas_status": Quantity(T_status_slab, "dimensionless"),
            "b_CO_kms": Quantity(b_CO_kms_arr, "km/s"),
            "chi_broad": Quantity(chi_dust_arr, "dimensionless"),
            "G_CO_diss": Quantity(G_CO_diss_arr, "dimensionless"),
            "G_H2_diss": Quantity(G_H2_diss_arr, "dimensionless"),
            "G_C_ion": Quantity(G_C_ion_arr, "dimensionless"),
            "G_CO_pdes": Quantity(G_CO_pdes_arr, "dimensionless"),
            "F_CO_pdes_photon": Quantity(F_CO_pdes_photon_arr, "1/(cm^2 s)"),
            "theta_co": Quantity(theta_co_arr, "dimensionless"),
            "theta_h2": Quantity(theta_h2_arr, "dimensionless"),
            "theta_c": Quantity(theta_c_arr, "dimensionless"),
            "chi_eff": Quantity(G_CO_diss_arr * theta_co_arr, "dimensionless"),
            "G_CO_diss_actual": Quantity(G_CO_diss_arr * theta_co_arr, "dimensionless"),
            "G_C_ion_actual": Quantity(G_C_ion_arr * theta_c_arr, "dimensionless"),
            "G_H2_diss_actual": Quantity(G_H2_diss_arr * theta_h2_arr, "dimensionless"),
            "F_CO_pdes_external_actual": Quantity(F_CO_pdes_photon_arr, "1/(cm^2 s)"),
        }

        rad.gow17_y = y_out
        rad.nco_gas = nco_gas
        rad.nco_ice = nco_ice
        rad.theta_co = fields["theta_co"]
        rad.chi_eff = fields["chi_eff"]
        rad.nH2 = nH2_out
        rad.nH_atom = nH_atom
        rad.nCplus = nCplus
        rad.nC = nC
        rad.ne = ne
        rad.Tgas_gow17 = fields["Tgas"]
        rad.gas_temperature = rad.Tgas_gow17

        n_fail = 0
        max_status = 0

        meta = {
            "model": "gow17",
            "enable_co_phase": bool(enable_co_phase),
            "nside": int(nside),
            "b_kms": float(b_kms),
            "b_CO_kms_scalar": float(b_kms),
            "b_CO_mode": str(shielding_linewidth_meta["b_CO_mode"]),
            "ion_rate_s": float(ion_rate_s),
            "Zg": float(Zg0),
            "Zd": float(Zd0),
            "reltol": float(reltol),
            "abstol0": float(abstol0),
            "shielding_max_iter": int(shielding_max_iter),
            "shielding_reltol": float(shielding_reltol),
            "n_fail": int(n_fail),
            "max_status": int(max_status),
            "fail_idx_head": [],
            "status_hist": {0: int(ncells), -1: 0},
        }

        return ChemistryResult(
            abundances=abundances,
            number_densities=number_densities,
            fields=fields,
            meta=meta,
        )

    Zg_arr = _broadcast_scalar_or_array(Zg, ncells)
    ion_rate_arr = _broadcast_scalar_or_array(ion_rate_s, ncells)


    sigma_d = rad.ensure_sigma_d_per_H()
    sigma_d_cm2 = _as_cgs_f64(sigma_d, "cm^2").reshape(ncells)
    sigma_d_CO_per_H, Zd_arr, sigma_d_ISM_ref = _resolve_co_dust_scalings(
        cfg=cfg,
        sigma_d_cm2=sigma_d_cm2,
        ncells=ncells,
    )
    S_CO, F_CRUV_CO_pdes_arr, k_crdes_CO_arr = _resolve_co_phase_controls(
        cfg=cfg,
        ion_rate_arr=ion_rate_arr,
        ncells=ncells,
    )
    co_phase_params = _resolve_co_phase_runtime_params(cfg)


    Leff_CO_max_arr = _broadcast_scalar_or_array(Leff_CO_max_scalar, ncells)
    gradv_arr = _broadcast_scalar_or_array(gradv_scalar, ncells)

    candidate_mask_arr = None
    tracer = None
    dirs = None
    candidate_idx = None
    candidate_flat_idx = None
    cell_centers = None

    use_healpix_geometry = (
        (gradv_mode == "nh_weighted_shear+turb")
        or (Leff_CO_max_mode == "geo")
    )
    if use_healpix_geometry and (not is_effectively_1d(rad.model.mesh, nH_cm3.shape)):
        candidate_mask_arr = np.ones(shape, dtype=bool)
        tracer, dirs, candidate_idx, cell_centers = _prepare_healpix_geometry(
            rad.model.mesh,
            nside=int(nside),
            candidate_mask=candidate_mask_arr,
            cache_dir=None,
        )
        if candidate_idx.size > 0:
            candidate_flat_idx = np.ravel_multi_index(candidate_idx.T, dims=shape)

    if gradv_mode == "nh_weighted_shear+turb":
        if candidate_idx is None or candidate_idx.size == 0:
            logger.info("gradv_mode=nh_weighted_shear+turb: no candidate cells; using scalar gradv")
        else:
            if "vphi" not in rad.model.gas:
                raise ValueError("gradv_mode=nh_weighted_shear+turb requires gas field 'vphi'")
            vphi_cgs = _as_cgs_f64(rad.model.gas["vphi"].data, "cm/s")

            if rad.model.mesh.coord_system == "spherical":
                r_cent = rad.model.mesh.centers("r").to("cm").magnitude
                theta_cent = rad.model.mesh.centers("theta").to("radian").magnitude
                phi_cent = rad.model.mesh.centers("phi").to("radian").magnitude

                ir = candidate_idx[:, 0]
                itheta = candidate_idx[:, 1]
                iphi = candidate_idx[:, 2]

                r_cand = r_cent[ir]
                theta_cand = theta_cent[itheta]
                phi_cand = phi_cent[iphi]

                R_cyl = r_cand * np.sin(theta_cand)
                vphi_cand = np.abs(vphi_cgs[ir, itheta, iphi])
                Omega = np.where(R_cyl > 0.0, vphi_cand / R_cyl, 0.0)

                eR = np.zeros((candidate_idx.shape[0], 3), dtype=np.float64)
                eR[:, 0] = np.cos(phi_cand)
                eR[:, 1] = np.sin(phi_cand)

                volumes = compute_cell_volumes(rad.model)
                L_cell = np.cbrt(volumes)
                L_cell_cand = L_cell[ir, itheta, iphi]
            else:
                x_edges = rad.model.mesh.edges("x").to("cm").magnitude
                y_edges = rad.model.mesh.edges("y").to("cm").magnitude
                z_edges = rad.model.mesh.edges("z").to("cm").magnitude
                x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])
                z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

                dx = np.diff(x_edges)
                dy = np.diff(y_edges)
                dz = np.diff(z_edges)
                volumes = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
                L_cell = np.cbrt(volumes)

                ix = candidate_idx[:, 0]
                iy = candidate_idx[:, 1]
                iz = candidate_idx[:, 2]

                x_cand = x_cent[ix]
                y_cand = y_cent[iy]
                R_cyl = np.sqrt(x_cand**2 + y_cand**2)
                vphi_cand = np.abs(vphi_cgs[ix, iy, iz])
                Omega = np.where(R_cyl > 0.0, vphi_cand / R_cyl, 0.0)

                eR = np.zeros((candidate_idx.shape[0], 3), dtype=np.float64)
                mask_R = R_cyl > 0.0
                eR[mask_R, 0] = x_cand[mask_R] / R_cyl[mask_R]
                eR[mask_R, 1] = y_cand[mask_R] / R_cyl[mask_R]
                eR[~mask_R, 0] = 1.0

                L_cell_cand = L_cell[ix, iy, iz]

            _, _, cols = compute_column_rays_healpix(
                rad.model.mesh,
                fields={"nh": nH_cm3},
                nside=int(nside),
                candidate_mask=None,
                tracer=tracer,
                dirs=dirs,
                candidate_idx=candidate_idx,
                cell_centers=cell_centers,
            )
            NH_rays = cols["nh"]
            gradv_cand = compute_gradv_nh_weighted(
                NH_rays,
                dirs,
                Omega,
                eR,
                L_cell_cand,
                b_kms=float(b_kms),
                q=float(gradv_q),
                N0=float(gradv_N0),
                p=float(gradv_p),
                f_corr=float(gradv_f_corr),
                gmin=float(gradv_gmin),
                gmax=float(gradv_gmax),
            )
            gradv_arr[candidate_flat_idx] = gradv_cand
            rad.gradv_gow17 = gradv_arr.reshape(shape)

    if Leff_CO_max_mode == "geo":
        if candidate_idx is None or candidate_idx.size == 0:
            logger.info("Leff_CO_max_mode=geo: no candidate cells; using scalar Leff_CO_max")
        else:
            _, S_all = integrate_rays_with_pathlength(tracer, cell_centers, dirs, nH_cm3)
            L_geo_cand = compute_L_geo_from_pathlengths(
                S_all,
                reduction=L_geo_reduction,
                L_min=L_geo_min,
                L_max=L_geo_max,
            )
            Leff_CO_max_arr[candidate_flat_idx] = L_geo_cand
            rad.Lgeo_gow17 = Leff_CO_max_arr.reshape(shape)

    b_kms, visser, shielding_linewidth_meta = _resolve_visser_table_linewidth(
        b_kms,
        shielding_linewidth_meta,
    )

    y0_single = np.zeros(N_Y, dtype=np.float64)
    y0_single[I_HEP] = 1.45e-08
    y0_single[I_H3P] = 2.68e-07
    y0_single[I_CP] = 1.0e-4
    y0_single[I_CO] = 1.0e-7
    y0_single[I_H2] = 0.1
    y0_single[I_CO_ICE] = 0.0

    abstol = _build_gow17_abstol(cfg, abstol0)

    y_guess = np.zeros((ncells, N_Y), dtype=np.float64)
    y_prev = getattr(rad, "gow17_y", None)
    warm_start = y_prev is not None and np.shape(y_prev) == tuple(shape) + (N_Y,)
    if warm_start:
        y_guess[:, :] = np.ascontiguousarray(
            np.asarray(y_prev, dtype=np.float64).reshape(ncells, N_Y)
        )
    else:
        for j in range(N_Y):
            y_guess[:, j] = y0_single[j]

    if not enable_co_phase:
        y_guess[:, I_CO_ICE] = 0.0

    if not const_temp and not warm_start:
        xe0 = _electron_abundance(y_guess)
        Cv0 = _cv_cold(y_guess[:, I_H2], xe0)
        y_guess[:, I_E] = Cv0 * T_flat

    theta_h2_arr = np.ones(ncells, dtype=np.float64)
    theta_co_arr = np.ones(ncells, dtype=np.float64)
    theta_c_arr = np.ones(ncells, dtype=np.float64)

    xCtot_flat = Zg_arr * float(XC_STD)
    xOtot_flat = Zg_arr * float(XO_STD)

    status_acc = np.zeros(ncells, dtype=np.int32)

    d_h2_hist = []
    d_co_hist = []
    d_tgas_hist = []
    d_h2_median_hist = []
    d_co_median_hist = []
    d_tgas_median_hist = []
    well_converged_cells_hist = []
    well_converged_fraction_hist = []
    bad_status_hist = []
    astrochem_acceleration_cells_hist = []
    astrochem_acceleration_entries_hist = []
    projection_corrections_hist = []
    equilibrium_solver_diag_hist = {
        "tevol_max_cells_hist": [],
        "tevol_max_residual_max_hist": [],
        "negative_abundance_cells_hist": [],
        "negative_abundance_corrections_hist": [],
        "cvode_failure_cells_hist": [],
        "exception_failure_cells_hist": [],
    }
    time_solver_diag_hist = {
        "time_negative_abundance_cells_hist": [],
        "time_negative_abundance_corrections_hist": [],
        "time_cvode_failure_cells_hist": [],
        "time_exception_failure_cells_hist": [],
    }

    def _state_temperature(y: np.ndarray) -> np.ndarray:
        xe = _electron_abundance(y)
        Cv = _cv_cold(y[:, I_H2], xe)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            T_state = y[:, I_E] / Cv
        return np.clip(T_state, float(temperature_cfg["Tgas_floor"]), float(temperature_cfg["Tgas_ceiling"]))

    def _project(y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        y_proj = project_gow17_state_to_budgets(
            y_arr,
            xCtot=xCtot_flat,
            xOtot=xOtot_flat,
        )
        if not enable_co_phase:
            y_proj[:, I_CO_ICE] = 0.0
        projection_corrections_hist.append(int(np.sum(y_proj != y_arr)))
        return np.ascontiguousarray(y_proj, dtype=np.float64)

    y_guess[:, :] = _project(y_guess)

    # Common keyword dict for _compute_shielding_and_gph calls.
    _shielding_kw = dict(
        nH_flat=nH_flat,
        chi_dust_flat=chi_dust_flat,
        G_CO_diss_flat=G_CO_diss_flat,
        G_H2_diss_flat=G_H2_diss_flat,
        G_C_ion_flat=G_C_ion_flat,
        G_CO_pdes_flat=G_CO_pdes_flat,
        F_CO_pdes_photon_flat=F_CO_pdes_photon_flat,
        xCtot_flat=xCtot_flat,
        Zd_arr=Zd_arr,
        Av_flat=Av_flat,
        shape=shape,
        ncells=ncells,
        rad=rad,
        nH_cm3=nH_cm3,
        chi_dust_arr=chi_dust_arr,
        visser=visser,
        b_kms=b_kms,
        nside=nside,
        chi_is_incident=chi_is_incident,
        local_chi_factor=local_chi_factor,
        shielding_outer_1d=shielding_outer_1d,
    )

    # Common keyword dict for CO phase parameters passed to batch solvers.
    _zero_cell = np.zeros(ncells, dtype=np.float64)
    _co_phase_kw = dict(
        co_E_bind_co=(float(co_phase_params["E_bind_CO"]) if enable_co_phase else 0.0),
        co_nu0_co=(float(co_phase_params["nu0_CO"]) if enable_co_phase else 0.0),
        co_Y_CO=(float(co_phase_params["Y_CO"]) if enable_co_phase else 0.0),
        co_N_SURF=(float(co_phase_params["N_SURF"]) if enable_co_phase else 0.0),
        co_N_LAY=(int(co_phase_params["N_LAY"]) if enable_co_phase else 0),
        co_S_CO=(float(S_CO) if enable_co_phase else 0.0),
        co_F_CRUV_CO_pdes=(
            np.ascontiguousarray(F_CRUV_CO_pdes_arr, dtype=np.float64)
            if enable_co_phase
            else _zero_cell
        ),
        co_k_crdes_CO=(
            np.ascontiguousarray(k_crdes_CO_arr, dtype=np.float64)
            if enable_co_phase
            else _zero_cell
        ),
    )

    if coupling_mode == "fixed_point":
        # ====================================================================
        # Fixed-point equilibrium coupling (baseline)
        # ====================================================================
        for it in range(shielding_max_iter):
            y_guess[:, :] = _project(y_guess)
            xCO_old = np.ascontiguousarray(y_guess[:, I_CO].copy(), dtype=np.float64)
            xH2_old = np.ascontiguousarray(y_guess[:, I_H2].copy(), dtype=np.float64)
            Tgas_old = _state_temperature(y_guess) if not const_temp else None

            theta_h2_flat, theta_co_flat, theta_c_flat, Gph, GPE, GISRF = (
                _compute_shielding_and_gph(
                    y_flat=y_guess,
                    **_shielding_kw,
                )
            )

            result = _gow17.solve_batch_equilibrium(
                y0=_project(y_guess),
                nH=nH_flat,
                Tgas=T_flat,
                Tdust=Tdust_flat,
                Zd=Zd_arr,
                Zg=Zg_arr,
                ion_rate=ion_rate_arr,
                GPE=GPE,
                F_CO_pdes_photon=GISRF,
                Gph=Gph,
                sigma_d_CO_per_H=(
                    sigma_d_CO_per_H if enable_co_phase else _zero_cell
                ),
                reltol=reltol,
                abstol=abstol,
                mxsteps=mxsteps,
                maxord=maxord,
                tolfac=tolfac,
                tmin=tmin,
                tmax=tmax,
                const_temp=const_temp,
                gradv=gradv_arr,
                Leff_CO_max=Leff_CO_max_arr,
                isDust_cooling=isDust_cooling,
                isCoolingCOThin=isCoolingCOThin,
                fH2gr=fH2gr,
                fHplusgr=fHplusgr,
                fCplusgr=fCplusgr,
                fHeplusgr=fHeplusgr,
                fSplusgr=fSplusgr,
                fSiplusgr=fSiplusgr,
                fCplusCR=fCplusCR,
                **_co_phase_kw,
                userJac=userJac,
                verbose=verbose,
            )

            _append_equilibrium_solver_diagnostics(equilibrium_solver_diag_hist, result)
            y_new = _project(result["y"])
            status_step = result["status"]

            status_acc = _accumulate_solver_status(status_acc, status_step)
            y_guess[:, :] = y_new
            y_guess[:, :] = _project(y_guess)

            xCO_new = y_guess[:, I_CO]
            xH2_new = y_guess[:, I_H2]

            denom_h2 = np.maximum(np.abs(xH2_new), shielding_abstol)
            denom_co = np.maximum(np.abs(xCO_new), shielding_abstol)
            rel_h2 = np.abs(xH2_new - xH2_old) / denom_h2
            rel_co = np.abs(xCO_new - xCO_old) / denom_co
            rel_tgas = None
            if Tgas_old is not None:
                Tgas_new = _state_temperature(y_guess)
                denom_tgas = np.maximum(np.abs(Tgas_new), 1.0)
                rel_tgas = np.abs(Tgas_new - Tgas_old) / denom_tgas
            d_h2 = np.max(rel_h2)
            d_co = np.max(rel_co)
            d_h2_hist.append(float(d_h2))
            d_co_hist.append(float(d_co))
            d_h2_median_hist.append(float(np.median(rel_h2)))
            d_co_median_hist.append(float(np.median(rel_co)))
            if rel_tgas is not None:
                d_tgas_hist.append(float(np.max(rel_tgas)))
                d_tgas_median_hist.append(float(np.median(rel_tgas)))
            well_converged = np.maximum(rel_h2, rel_co) <= shielding_reltol
            if rel_tgas is not None:
                well_converged &= rel_tgas <= tgas_convergence_reltol
            well_converged_cells_hist.append(int(np.sum(well_converged)))
            well_converged_fraction_hist.append(float(np.mean(well_converged)))
            converged = max(d_h2, d_co) <= shielding_reltol
            if rel_tgas is not None:
                converged = converged and d_tgas_hist[-1] <= tgas_convergence_reltol
            if converged:
                break

    elif coupling_mode == "astrochem":
        # ====================================================================
        # AstroChem-style pseudo-time coupling
        #
        # Evolve abundances in pseudo-time while updating shielding a small
        # number of times, then finish with one equilibrium solve.
        #
        # Reference: inspired by AstroChemistry.jl coupling strategy.
        # ====================================================================
        N = astrochem_n_updates
        t_end_s = astrochem_t_end_yr * _YR_TO_S

        if N <= 0:
            # N=0: compute shielding once from y_guess, then go straight
            # to the final equilibrium solve (no pseudo-time integration).
            y_guess[:, :] = _project(y_guess)
            theta_h2_flat, theta_co_flat, theta_c_flat, Gph, GPE, GISRF = (
                _compute_shielding_and_gph(
                    y_flat=y_guess,
                    **_shielding_kw,
                )
            )

            result = _gow17.solve_batch_equilibrium(
                y0=_project(y_guess),
                nH=nH_flat,
                Tgas=T_flat,
                Tdust=Tdust_flat,
                Zd=Zd_arr,
                Zg=Zg_arr,
                ion_rate=ion_rate_arr,
                GPE=GPE,
                F_CO_pdes_photon=GISRF,
                Gph=Gph,
                sigma_d_CO_per_H=(
                    sigma_d_CO_per_H if enable_co_phase else _zero_cell
                ),
                reltol=reltol,
                abstol=abstol,
                mxsteps=mxsteps,
                maxord=maxord,
                tolfac=tolfac,
                tmin=tmin,
                tmax=tmax,
                const_temp=const_temp,
                gradv=gradv_arr,
                Leff_CO_max=Leff_CO_max_arr,
                isDust_cooling=isDust_cooling,
                isCoolingCOThin=isCoolingCOThin,
                fH2gr=fH2gr,
                fHplusgr=fHplusgr,
                fCplusgr=fCplusgr,
                fHeplusgr=fHeplusgr,
                fSplusgr=fSplusgr,
                fSiplusgr=fSiplusgr,
                fCplusCR=fCplusCR,
                **_co_phase_kw,
                userJac=userJac,
                verbose=verbose,
            )

            _append_equilibrium_solver_diagnostics(equilibrium_solver_diag_hist, result)
            y_guess[:, :] = _project(result["y"])
            status_acc = _accumulate_solver_status(status_acc, result["status"])

        else:
            # N >= 1: pseudo-time integration with N macro-updates.

            # Geometric schedule: small early steps, larger late steps.
            # t_targets[i] = t_end * (r^(i+1) - 1) / (r^N - 1) for i=0..N-1
            # so t_targets[0] > 0 and t_targets[N-1] = t_end.
            if N <= 1:
                t_targets = np.array([t_end_s], dtype=np.float64)
            else:
                r = 10.0 ** (1.0 / (N - 1))
                k = np.arange(1, N + 1, dtype=np.float64)
                t_targets = t_end_s * (r**k - 1.0) / (r**N - 1.0)
                # Ensure last target is exactly t_end_s.
                t_targets[-1] = t_end_s

            # Per-step durations.
            dt = np.empty(N, dtype=np.float64)
            dt[0] = t_targets[0]
            dt[1:] = np.diff(t_targets)

            y_state = y_guess.copy()
            y_state[:, :] = _project(y_state)
            theta_h2_flat = np.ones(ncells, dtype=np.float64)
            theta_co_flat = np.ones(ncells, dtype=np.float64)
            theta_c_flat = np.ones(ncells, dtype=np.float64)
            accel_species = [I_H2, I_CO] + ([I_CO_ICE] if enable_co_phase else [])
            accel_prev2: dict[int, np.ndarray] | None = None
            accel_prev1: dict[int, np.ndarray] | None = (
                {idx: y_state[:, idx].copy() for idx in accel_species}
                if astrochem_acceleration == "aitken"
                else None
            )

            for k_step in range(N):
                y_state[:, :] = _project(y_state)
                xCO_old = np.ascontiguousarray(y_state[:, I_CO].copy(), dtype=np.float64)
                xH2_old = np.ascontiguousarray(y_state[:, I_H2].copy(), dtype=np.float64)
                Tgas_old = _state_temperature(y_state) if not const_temp else None

                # Step 1+2: compute columns and shielding from current y_state.
                theta_h2_flat, theta_co_flat, theta_c_flat, Gph, GPE, GISRF = (
                    _compute_shielding_and_gph(
                        y_flat=y_state,
                        **_shielding_kw,
                    )
                )

                # Step 3: integrate chemistry forward by dt[k_step].
                result = _gow17.solve_batch_time(
                    y0=_project(y_state),
                    nH=nH_flat,
                    Tgas=T_flat,
                    Tdust=Tdust_flat,
                    Zd=Zd_arr,
                    Zg=Zg_arr,
                    ion_rate=ion_rate_arr,
                    GPE=GPE,
                    F_CO_pdes_photon=GISRF,
                    Gph=Gph,
                    sigma_d_CO_per_H=(
                        sigma_d_CO_per_H if enable_co_phase else _zero_cell
                    ),
                    reltol=reltol,
                    abstol=abstol,
                    mxsteps=mxsteps,
                    maxord=maxord,
                    t_end=float(dt[k_step]),
                    const_temp=const_temp,
                    gradv=gradv_arr,
                    Leff_CO_max=Leff_CO_max_arr,
                    isDust_cooling=isDust_cooling,
                    isCoolingCOThin=isCoolingCOThin,
                    fH2gr=fH2gr,
                    fHplusgr=fHplusgr,
                    fCplusgr=fCplusgr,
                    fHeplusgr=fHeplusgr,
                    fSplusgr=fSplusgr,
                    fSiplusgr=fSiplusgr,
                    fCplusCR=fCplusCR,
                    **_co_phase_kw,
                    userJac=userJac,
                    verbose=verbose,
                )
                _append_time_solver_diagnostics(time_solver_diag_hist, result)

                y_state[:, :] = _project(result["y"])
                acceleration_cells = 0
                acceleration_entries = 0
                if (
                    astrochem_acceleration == "aitken"
                    and accel_prev2 is not None
                    and accel_prev1 is not None
                    and (k_step + 1) >= astrochem_acceleration_start
                ):
                    acceleration_cells, acceleration_entries = (
                        _apply_astrochem_aitken_acceleration(
                            y_state=y_state,
                            prev2=accel_prev2,
                            prev1=accel_prev1,
                            xCtot_flat=xCtot_flat,
                            enable_co_phase=enable_co_phase,
                            max_jump=astrochem_acceleration_max_jump,
                            floor=shielding_abstol,
                        )
                    )
                if astrochem_acceleration == "aitken" and accel_prev1 is not None:
                    accel_prev2 = accel_prev1
                    accel_prev1 = {idx: y_state[:, idx].copy() for idx in accel_species}
                status_step = np.asarray(result["status"], dtype=np.int32)
                status_acc = _accumulate_solver_status(status_acc, status_step)
                y_state[:, :] = _project(y_state)

                # Step 4: convergence diagnostics.
                xCO_new = y_state[:, I_CO]
                xH2_new = y_state[:, I_H2]

                denom_h2 = np.maximum(np.abs(xH2_new), shielding_abstol)
                denom_co = np.maximum(np.abs(xCO_new), shielding_abstol)
                rel_h2 = np.abs(xH2_new - xH2_old) / denom_h2
                rel_co = np.abs(xCO_new - xCO_old) / denom_co
                rel_tgas = None
                if Tgas_old is not None:
                    Tgas_new = _state_temperature(y_state)
                    denom_tgas = np.maximum(np.abs(Tgas_new), 1.0)
                    rel_tgas = np.abs(Tgas_new - Tgas_old) / denom_tgas
                d_h2 = float(np.max(rel_h2))
                d_co = float(np.max(rel_co))
                d_h2_hist.append(d_h2)
                d_co_hist.append(d_co)
                d_h2_median_hist.append(float(np.median(rel_h2)))
                d_co_median_hist.append(float(np.median(rel_co)))
                if rel_tgas is not None:
                    d_tgas_hist.append(float(np.max(rel_tgas)))
                    d_tgas_median_hist.append(float(np.median(rel_tgas)))
                well_converged = np.maximum(rel_h2, rel_co) <= shielding_reltol
                if rel_tgas is not None:
                    well_converged &= rel_tgas <= tgas_convergence_reltol
                well_converged_cells_hist.append(int(np.sum(well_converged)))
                well_converged_fraction_hist.append(float(np.mean(well_converged)))
                bad_status_hist.append(int(np.sum(status_step != 0)))
                astrochem_acceleration_cells_hist.append(int(acceleration_cells))
                astrochem_acceleration_entries_hist.append(int(acceleration_entries))

                if rel_tgas is not None:
                    logger.info(
                        "astrochem step %d/%d: t_target=%.3e yr, dt=%.3e yr, "
                        "d_h2=%.3e, d_co=%.3e, d_tgas=%.3e, bad=%d, accel_cells=%d",
                        k_step + 1, N,
                        float(t_targets[k_step] / _YR_TO_S),
                        float(dt[k_step] / _YR_TO_S),
                        d_h2, d_co, d_tgas_hist[-1],
                        bad_status_hist[-1],
                        acceleration_cells,
                    )
                else:
                    logger.info(
                        "astrochem step %d/%d: t_target=%.3e yr, dt=%.3e yr, "
                        "d_h2=%.3e, d_co=%.3e, bad=%d, accel_cells=%d",
                        k_step + 1, N,
                        float(t_targets[k_step] / _YR_TO_S),
                        float(dt[k_step] / _YR_TO_S),
                        d_h2, d_co,
                        bad_status_hist[-1],
                        acceleration_cells,
                    )

            # After N macro-updates: recompute columns + shielding from final y_state.
            y_state[:, :] = _project(y_state)
            theta_h2_flat, theta_co_flat, theta_c_flat, Gph, GPE, GISRF = (
                _compute_shielding_and_gph(
                    y_flat=y_state,
                    **_shielding_kw,
                )
            )

            # Final equilibrium solve with shielding held fixed.
            result = _gow17.solve_batch_equilibrium(
                y0=_project(y_state),
                nH=nH_flat,
                Tgas=T_flat,
                Tdust=Tdust_flat,
                Zd=Zd_arr,
                Zg=Zg_arr,
                ion_rate=ion_rate_arr,
                GPE=GPE,
                F_CO_pdes_photon=GISRF,
                Gph=Gph,
                sigma_d_CO_per_H=(
                    sigma_d_CO_per_H if enable_co_phase else _zero_cell
                ),
                reltol=reltol,
                abstol=abstol,
                mxsteps=mxsteps,
                maxord=maxord,
                tolfac=tolfac,
                tmin=tmin,
                tmax=tmax,
                const_temp=const_temp,
                gradv=gradv_arr,
                Leff_CO_max=Leff_CO_max_arr,
                isDust_cooling=isDust_cooling,
                isCoolingCOThin=isCoolingCOThin,
                fH2gr=fH2gr,
                fHplusgr=fHplusgr,
                fCplusgr=fCplusgr,
                fHeplusgr=fHeplusgr,
                fSplusgr=fSplusgr,
                fSiplusgr=fSiplusgr,
                fCplusCR=fCplusCR,
                **_co_phase_kw,
                userJac=userJac,
                verbose=verbose,
            )

            _append_equilibrium_solver_diagnostics(equilibrium_solver_diag_hist, result)
            y_guess[:, :] = _project(result["y"])
            status_acc = _accumulate_solver_status(status_acc, result["status"])

    else:
        raise ValueError(
            f"gow17: unknown coupling_mode={coupling_mode!r}; "
            "expected 'fixed_point' or 'astrochem'"
        )

    n_shielding_iter = len(d_h2_hist)

    y_guess[:, :] = _project(y_guess)
    y_out = y_guess.reshape(shape + (N_Y,))
    status = status_acc.reshape(shape)

    y_out = project_gow17_state_to_budgets(
        y_out,
        xCtot=xCtot_flat.reshape(shape),
        xOtot=xOtot_flat.reshape(shape),
    )
    if not enable_co_phase:
        y_out[..., I_CO_ICE] = 0.0

    xCO = y_out[..., I_CO]
    xCO_ice = y_out[..., I_CO_ICE]
    xH2 = y_out[..., I_H2]

    nco_gas = Quantity(xCO * nH_cm3, "cm^-3")
    nco_ice = Quantity(xCO_ice * nH_cm3, "cm^-3")
    nH2_out = Quantity(xH2 * nH_cm3, "cm^-3")

    xe = (
        y_out[..., I_HEP]
        + y_out[..., I_CP]
        + y_out[..., I_HCOP]
        + y_out[..., I_H3P]
        + y_out[..., I_H2P]
        + y_out[..., I_HP]
        + y_out[..., I_SP]
        + y_out[..., I_SIP]
        + y_out[..., I_OP]
    )

    xH_atom = 1.0 - (
        y_out[..., I_OHX]
        + y_out[..., I_CHX]
        + y_out[..., I_HCOP]
        + 3.0 * y_out[..., I_H3P]
        + 2.0 * y_out[..., I_H2P]
        + y_out[..., I_HP]
        + 2.0 * y_out[..., I_H2]
    )

    xCtot = xCtot_flat.reshape(shape)
    xC_neutral = xCtot - (
        y_out[..., I_HCOP]
        + y_out[..., I_CHX]
        + y_out[..., I_CO]
        + y_out[..., I_CO_ICE]
        + y_out[..., I_CP]
    )

    nCplus = Quantity(y_out[..., I_CP] * nH_cm3, "cm^-3")
    nC = Quantity(np.maximum(xC_neutral, 0.0) * nH_cm3, "cm^-3")
    ne = Quantity(xe * nH_cm3, "cm^-3")
    nH_atom = Quantity(np.maximum(xH_atom, 0.0) * nH_cm3, "cm^-3")

    budget_diag = gow17_budget_diagnostics(
        y_out,
        xCtot=xCtot,
        xOtot=xOtot_flat.reshape(shape),
        rtol=float(reltol),
        logger=logger,
    )

    if np.any(status != 0):
        bad = int(np.sum(status != 0))
        logger.warning(f"gow17: {bad} cells did not converge")
    else:
        validate_chemistry_state(
            nH=nH,
            nH2=nH2_out,
            nC=nC,
            nCplus=nCplus,
            nco_total=Quantity(nco_gas.magnitude + nco_ice.magnitude, "cm^-3"),
            check_pd=False,
            rtol=float(reltol),
        )

    abundances, number_densities = _gow17_species_outputs(
        y_out,
        nH_cm3,
        x_h=xH_atom,
        x_catom=xC_neutral,
        x_e=xe,
        line_h2_opr=cfg.get("line_h2_opr", 3.0),
    )

    gow17_diag = {
        "coupling_mode": coupling_mode,
        "enable_co_phase": bool(enable_co_phase),
        "shielding_iter": n_shielding_iter,
        "d_h2_hist": np.asarray(d_h2_hist, dtype=np.float64),
        "d_co_hist": np.asarray(d_co_hist, dtype=np.float64),
        "d_tgas_hist": np.asarray(d_tgas_hist, dtype=np.float64),
        "d_h2_median_hist": np.asarray(d_h2_median_hist, dtype=np.float64),
        "d_co_median_hist": np.asarray(d_co_median_hist, dtype=np.float64),
        "d_tgas_median_hist": np.asarray(d_tgas_median_hist, dtype=np.float64),
        "well_converged_cells_hist": np.asarray(well_converged_cells_hist, dtype=np.int64),
        "well_converged_fraction_hist": np.asarray(well_converged_fraction_hist, dtype=np.float64),
        "well_converged_reltol": float(shielding_reltol),
        "tgas_convergence_reltol": float(tgas_convergence_reltol),
        "temperature_mode": temperature_mode,
        "Tgas_floor": float(temperature_cfg["Tgas_floor"]),
        "Tgas_ceiling": float(temperature_cfg["Tgas_ceiling"]),
        "max_thermal_iterations": int(temperature_cfg["max_thermal_iterations"]),
        "thermal_rtol": float(temperature_cfg["thermal_rtol"]),
        "thermal_atol": float(temperature_cfg["thermal_atol"]),
        "well_converged_includes_tgas": bool(not const_temp),
        "well_converged_ncells": int(ncells),
        "astrochem_acceleration": astrochem_acceleration,
        "astrochem_acceleration_start": int(astrochem_acceleration_start),
        "astrochem_acceleration_max_jump": float(astrochem_acceleration_max_jump),
        "astrochem_acceleration_cells_hist": np.asarray(
            astrochem_acceleration_cells_hist,
            dtype=np.int64,
        ),
        "astrochem_acceleration_entries_hist": np.asarray(
            astrochem_acceleration_entries_hist,
            dtype=np.int64,
        ),
        "projection_corrections_hist": np.asarray(
            projection_corrections_hist,
            dtype=np.int64,
        ),
        "projection_corrections_total": int(np.sum(projection_corrections_hist)),
        "projection_corrections_max": (
            int(np.max(projection_corrections_hist))
            if projection_corrections_hist
            else 0
        ),
        "abstol": np.asarray(abstol, dtype=np.float64),
        "co_phase_E_bind_CO": float(co_phase_params["E_bind_CO"]),
        "co_phase_nu0_CO": float(co_phase_params["nu0_CO"]),
        "co_phase_Y_CO": float(co_phase_params["Y_CO"]),
        "co_phase_N_SURF": float(co_phase_params["N_SURF"]),
        "co_phase_N_LAY": int(co_phase_params["N_LAY"]),
        "directional_weight_product": str(directional_uv_product),
        "directional_weight_approximation": True,
        "theta_c_uses_CO_direction_weights": bool(directional_uv_product == "G_CO_diss"),
        "theta_h2_uses_CO_direction_weights": bool(directional_uv_product == "G_CO_diss"),
        **shielding_linewidth_meta,
    }
    if coupling_mode == "astrochem":
        if astrochem_n_updates > 0:
            gow17_diag["astrochem_t_targets"] = (t_targets / _YR_TO_S).tolist()
        else:
            gow17_diag["astrochem_t_targets"] = []
        gow17_diag["bad_status_hist"] = bad_status_hist
    equilibrium_solver_diag = _summarize_equilibrium_solver_diagnostics(
        equilibrium_solver_diag_hist
    )
    gow17_diag.update(equilibrium_solver_diag)
    time_solver_diag = _summarize_time_solver_diagnostics(time_solver_diag_hist)
    gow17_diag.update(time_solver_diag)
    if equilibrium_solver_diag["tevol_max_cells_total"] > 0:
        logger.warning(
            "gow17: some cells reached tevol_max before SolveEq convergence; "
            "do not treat these cells as equilibrium solutions."
        )
    if equilibrium_solver_diag["cvode_failure_cells_total"] > 0:
        logger.warning(
            "gow17: CVODE failures occurred in %d accumulated equilibrium-solve cells.",
            equilibrium_solver_diag["cvode_failure_cells_total"],
        )
    if equilibrium_solver_diag["exception_failure_cells_total"] > 0:
        logger.warning(
            "gow17: non-CVODE exceptions occurred in %d accumulated equilibrium-solve cells.",
            equilibrium_solver_diag["exception_failure_cells_total"],
        )
    if equilibrium_solver_diag["negative_abundance_corrections_total"] > 0:
        logger.warning(
            "gow17: native solver applied %d negative-abundance corrections.",
            equilibrium_solver_diag["negative_abundance_corrections_total"],
        )
    if time_solver_diag.get("time_cvode_failure_cells_total", 0) > 0:
        logger.warning(
            "gow17: CVODE failures occurred in %d accumulated fixed-time cells.",
            time_solver_diag["time_cvode_failure_cells_total"],
        )
    if time_solver_diag.get("time_exception_failure_cells_total", 0) > 0:
        logger.warning(
            "gow17: non-CVODE exceptions occurred in %d accumulated fixed-time cells.",
            time_solver_diag["time_exception_failure_cells_total"],
        )
    if time_solver_diag.get("time_negative_abundance_corrections_total", 0) > 0:
        logger.warning(
            "gow17: native fixed-time solver applied %d negative-abundance corrections.",
            time_solver_diag["time_negative_abundance_corrections_total"],
        )
    gow17_diag.update(budget_diag)

    if not const_temp:
        T_out = _state_temperature(y_out.reshape(ncells, N_Y)).reshape(shape)
    else:
        T_out = T_flat.reshape(shape)
    T_status = np.zeros(shape, dtype=np.int32)
    bad_T = (
        (~np.isfinite(T_out))
        | (T_out < float(temperature_cfg["Tgas_floor"]))
        | (T_out > float(temperature_cfg["Tgas_ceiling"]))
    )
    T_status[bad_T] = -1

    rad.gow17_convergence = gow17_diag

    actual_uv = _actual_solver_uv_fields(
        Gph=Gph,
        F_CO_pdes_external=GISRF,
        shape=shape,
    )

    fields = {
        "co_ice": nco_ice,
        "Tgas": Quantity(T_out, "K"),
        "Tgas_minus_Tdust": Quantity(T_out - Tdust_flat.reshape(shape), "K"),
        "Tgas_status": Quantity(T_status, "dimensionless"),
        "b_CO_kms": Quantity(b_CO_kms_arr, "km/s"),
        "chi_broad": Quantity(chi_dust_arr, "dimensionless"),
        "G_CO_diss": Quantity(G_CO_diss_arr, "dimensionless"),
        "G_H2_diss": Quantity(G_H2_diss_arr, "dimensionless"),
        "G_C_ion": Quantity(G_C_ion_arr, "dimensionless"),
        "G_CO_pdes": Quantity(G_CO_pdes_arr, "dimensionless"),
        "F_CO_pdes_photon": Quantity(F_CO_pdes_photon_arr, "1/(cm^2 s)"),
        "theta_co": Quantity(theta_co_flat.reshape(shape), "dimensionless"),
        "theta_h2": Quantity(theta_h2_flat.reshape(shape), "dimensionless"),
        "theta_c": Quantity(theta_c_flat.reshape(shape), "dimensionless"),
        "chi_eff": Quantity(actual_uv["G_CO_diss_actual"], "dimensionless"),
        "G_CO_diss_actual": Quantity(actual_uv["G_CO_diss_actual"], "dimensionless"),
        "G_C_ion_actual": Quantity(actual_uv["G_C_ion_actual"], "dimensionless"),
        "G_H2_diss_actual": Quantity(actual_uv["G_H2_diss_actual"], "dimensionless"),
        "F_CO_pdes_external_actual": Quantity(
            actual_uv["F_CO_pdes_external_actual"],
            "1/(cm^2 s)",
        ),
    }
    fields.update(
        _compute_co_phase_diagnostics(
            y_out=y_out,
            nH_cm3=nH_cm3,
            Tgas_K=T_out,
            Tdust_K=Tdust_flat.reshape(shape),
            sigma_d_CO_per_H=(
                sigma_d_CO_per_H.reshape(shape)
                if enable_co_phase
                else np.zeros(shape, dtype=np.float64)
            ),
            S_CO=(float(S_CO) if enable_co_phase else 0.0),
            F_CO_pdes_photon=GISRF.reshape(shape),
            F_CRUV_CO_pdes=(
                F_CRUV_CO_pdes_arr.reshape(shape)
                if enable_co_phase
                else np.zeros(shape, dtype=np.float64)
            ),
            k_crdes_CO=(
                k_crdes_CO_arr.reshape(shape)
                if enable_co_phase
                else np.zeros(shape, dtype=np.float64)
            ),
            G_CO_diss_actual=actual_uv["G_CO_diss_actual"],
            G_C_ion_actual=actual_uv["G_C_ion_actual"],
            G_H2_diss_actual=actual_uv["G_H2_diss_actual"],
            enable_co_phase=enable_co_phase,
            co_phase_params=co_phase_params,
        )
    )
    gow17_diag["CO_loss_dominant_id_map"] = {
        0: "none",
        1: "freezeout",
        2: "photodissociation",
        3: "He+ destruction",
    }
    gow17_diag["CO_gain_dominant_id_map"] = {
        0: "none",
        1: "photodesorption",
        2: "thermal_desorption",
        3: "cosmic_ray_desorption",
        4: "gas_phase_formation_proxy",
    }
    gow17_diag["C_abundance_interpretation_flag_map"] = {
        0: "nominal",
        1: "cold_shielded_CO_freezeout_dominated_neutral_C_upper_limit",
    }

    rhs_t_cfg = cfg.get("rhs_residual_t", cfg.get("rhs_residual_time", tmax))
    rhs_t_resid = _maybe_quantity_to_float(rhs_t_cfg, "s")
    rhs_fields, rhs_diag = _compute_rhs_residual_diagnostics(
        y_out=y_out,
        nH_flat=nH_flat,
        Tgas_flat=T_out.reshape(ncells),
        Tdust_flat=Tdust_flat,
        Zd_arr=Zd_arr,
        Zg_arr=Zg_arr,
        ion_rate_arr=ion_rate_arr,
        GPE=GPE,
        GISRF=GISRF,
        Gph=Gph,
        sigma_d_CO_per_H=(sigma_d_CO_per_H if enable_co_phase else _zero_cell),
        const_temp=const_temp,
        gradv_arr=gradv_arr,
        Leff_CO_max_arr=Leff_CO_max_arr,
        isDust_cooling=isDust_cooling,
        isCoolingCOThin=isCoolingCOThin,
        fH2gr=fH2gr,
        fHplusgr=fHplusgr,
        fCplusgr=fCplusgr,
        fHeplusgr=fHeplusgr,
        fSplusgr=fSplusgr,
        fSiplusgr=fSiplusgr,
        fCplusCR=fCplusCR,
        co_phase_kw=_co_phase_kw,
        abstol=abstol,
        reltol=reltol,
        t_resid=rhs_t_resid,
        shape=shape,
    )
    fields.update(rhs_fields)
    gow17_diag.update(rhs_diag)
    rhs_warn = float(cfg.get("rhs_residual_warn", 1.0e6))
    rhs_max = float(gow17_diag.get("gow17_rhs_residual_max_global", np.nan))
    if np.isfinite(rhs_max) and rhs_max > rhs_warn:
        logger.warning(
            "gow17: native RHS residual diagnostic is large (max scaled residual=%0.3e).",
            rhs_max,
        )
    rad.gow17_convergence = gow17_diag

    rad.gow17_y = y_out
    rad.nco_gas = nco_gas
    rad.nco_ice = nco_ice
    rad.theta_co = fields["theta_co"]
    rad.chi_eff = fields["chi_eff"]
    rad.nH2 = nH2_out
    rad.nH_atom = nH_atom
    rad.nCplus = nCplus
    rad.nC = nC
    rad.ne = ne

    rad.Tgas_gow17 = Quantity(T_out, "K")
    rad.gas_temperature = rad.Tgas_gow17

    n_fail = int(np.sum(status != 0))
    max_status = int(np.max(status)) if status.size else 0

    fail_idx = np.flatnonzero(status != 0)
    fail_idx_head = fail_idx[:16].astype(np.int64)
    status_hist = {}
    if status.size:
        for code, count in zip(*np.unique(status, return_counts=True)):
            status_hist[int(code)] = int(count)

    Zd = float(np.mean(Zd_arr))

    meta = {
        "model": "gow17",
        "enable_co_phase": bool(enable_co_phase),
        "nside": int(nside),
        "b_kms": float(b_kms),
        "b_CO_kms_scalar": float(b_kms),
        "b_CO_mode": str(shielding_linewidth_meta["b_CO_mode"]),
        "ion_rate_s": float(ion_rate_s),
        "Zg": float(Zg),
        "Zd": float(Zd),
        "sigma_d_ISM_ref": float(sigma_d_ISM_ref),
        "S_CO": float(S_CO),
        "co_phase_E_bind_CO": float(co_phase_params["E_bind_CO"]),
        "co_phase_nu0_CO": float(co_phase_params["nu0_CO"]),
        "co_phase_Y_CO": float(co_phase_params["Y_CO"]),
        "co_phase_N_SURF": float(co_phase_params["N_SURF"]),
        "co_phase_N_LAY": int(co_phase_params["N_LAY"]),
        "co_phase_cruv_enabled": bool(_nested_cfg(cfg, "co_phase").get("enable_cruv_pdes", True)),
        "co_phase_crdes_enabled": bool(_nested_cfg(cfg, "co_phase").get("enable_crdes_CO", False)),
        "temperature_mode": temperature_mode,
        "Tgas_floor": float(temperature_cfg["Tgas_floor"]),
        "Tgas_ceiling": float(temperature_cfg["Tgas_ceiling"]),
        "reltol": float(reltol),
        "abstol0": float(abstol0),
        "shielding_max_iter": int(shielding_max_iter),
        "shielding_reltol": float(shielding_reltol),
        "n_fail": int(n_fail),
        "max_status": int(max_status),
        "fail_idx_head": fail_idx_head.tolist(),
        "status_hist": status_hist,
        "gow17_diagnostics": gow17_diag,
    }

    return ChemistryResult(
        abundances=abundances,
        number_densities=number_densities,
        fields=fields,
        meta=meta,
    )
