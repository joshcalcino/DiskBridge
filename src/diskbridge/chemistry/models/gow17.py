from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import numpy as np

import diskbridge
from diskbridge._units import Quantity
from diskbridge._config import get_config, resolve_model_config
from diskbridge._logging import logger
from diskbridge._constants import M_H
from diskbridge.model.profiles import compute_cell_volumes
from diskbridge.model.microturbulence import (
    ensure_microturbulence_field,
)
from diskbridge.chemistry.io import attach_chemistry_result_to_model
from diskbridge.chemistry.types import ChemistryResult
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
from diskbridge.model.utils import field_data_as_order
from diskbridge.radmc3d.uv_products import (
    UV_PRODUCT_MERGED_FIELD_NAMES,
    draine_reference_for_product,
    uv_product_specs_from_config,
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
ALPHA_GD_CGS = 3.2e-34
_YR_TO_S = float(Quantity("1 yr").to("s").magnitude)
PAH_N_C_DEFAULT = 54
PAH_N_H_DEFAULT = 18
PAH_X_ISM_DEFAULT = 3.0e-7 * (50.0 / PAH_N_C_DEFAULT)
PAH_MASS_G_DEFAULT = (
    PAH_N_C_DEFAULT * 12.011 + PAH_N_H_DEFAULT * 1.008
) * 1.66053906660e-24

_KPH_BASE = np.asarray(
    [3.5e-10, 9.1e-10, 2.4e-10, 3.8e-10, 5.7e-11, 6.0e-10, 4.5e-9],
    dtype=np.float64,
)
KPH_C_BASE = float(_KPH_BASE[IPH_C])
KPH_CO_BASE = float(_KPH_BASE[IPH_CO])
KPH_H2_BASE = float(_KPH_BASE[IPH_H2])


@dataclass(frozen=True)
class _Gow17RadiationMode:
    name: str
    uses_uv_products: bool
    directional_uv_product: str


@dataclass(frozen=True)
class _Gow17CheckpointState:
    y: np.ndarray
    theta_h2: np.ndarray
    theta_co: np.ndarray
    theta_c: np.ndarray
    status: np.ndarray
    completed_updates: int
    histories: dict


def _infer_gow17_radiation_mode(rad: "RadModel") -> _Gow17RadiationMode:
    """Infer the local-radiation mode for the 3D GOW17 wrapper."""
    mode = getattr(rad, "radiation_mode", None)
    has_products = all(rad.has_uv_product(name) for name in UV_PRODUCT_MERGED_FIELD_NAMES)

    if mode is None:
        mode = "local_uv_products" if has_products else "local_chi"

    valid = {"local_uv_products", "local_chi"}
    if mode not in valid:
        raise ValueError(
            f"gow17 is the 3D wrapper and only accepts local radiation modes "
            f"{sorted(valid)}; got {mode!r}. Use model='gow17_slab' for slab chemistry."
        )

    if mode.endswith("uv_products") and not has_products:
        missing = [name for name in UV_PRODUCT_MERGED_FIELD_NAMES if not rad.has_uv_product(name)]
        raise KeyError(f"gow17: {mode} requires missing UV product(s): {missing}")

    if mode == "local_uv_products":
        return _Gow17RadiationMode(mode, True, "G_CO_diss")
    return _Gow17RadiationMode(mode, False, "chi")


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
    mode = str(temp_cfg["mode"]).lower()
    initial = str(temp_cfg["initial"]).lower()

    if mode not in ("computed", "dust"):
        raise ValueError(
            "gow17 temperature mode must be 'computed' or 'dust', "
            f"got {mode!r}"
        )
    if initial not in ("dust", "gas_temperature"):
        raise ValueError(
            "gow17 temperature initial value must be 'dust' or "
            f"'gas_temperature', got {initial!r}"
        )

    return {
        "mode": mode,
        "initial": initial,
        "const_temp": mode == "dust",
        "Tgas_floor": _maybe_quantity_to_float(
            temp_cfg["Tgas_floor"],
            "K",
        ),
        "Tgas_ceiling": _maybe_quantity_to_float(
            temp_cfg["Tgas_ceiling"],
            "K",
        ),
        "max_thermal_iterations": int(temp_cfg["max_thermal_iterations"]),
        "thermal_rtol": float(temp_cfg["thermal_rtol"]),
        "thermal_atol": _maybe_quantity_to_float(
            temp_cfg["thermal_atol"],
            "K",
        ),
    }


def _initial_gas_temperature(rad, Tdust, temperature_cfg: dict):
    initial = str(temperature_cfg["initial"])
    if initial == "dust":
        return Tdust
    if initial == "gas_temperature":
        Tgas = rad.ensure_gas_temperature()
        if Tgas is None:
            raise ValueError(
                "gow17 temperature.initial='gas_temperature' requires a gas "
                "temperature field"
            )
        return Tgas
    raise ValueError(f"Unknown gow17 temperature.initial={initial!r}")


def _build_gow17_abstol(cfg: dict, abstol0: float) -> np.ndarray:
    """Build species-specific absolute tolerances for the native GOW17 solver."""
    tol_cfg = _nested_cfg(cfg, "tolerances")
    default = float(tol_cfg["abstol_default"])
    abstol = np.full(N_Y, default, dtype=np.float64)
    abstol[I_HEP] = float(tol_cfg["abstol_Heplus"])
    abstol[I_OHX] = float(tol_cfg["abstol_OHx"])
    abstol[I_CHX] = float(tol_cfg["abstol_CHx"])
    abstol[I_CO] = float(tol_cfg["abstol_CO"])
    abstol[I_CO_ICE] = float(tol_cfg["abstol_CO_ice"])
    abstol[I_CP] = float(tol_cfg["abstol_Cplus"])
    abstol[I_HCOP] = float(tol_cfg["abstol_HCOplus"])
    abstol[I_H2] = float(tol_cfg["abstol_H2"])
    abstol[I_HP] = float(tol_cfg["abstol_Hplus"])
    abstol[I_H3P] = float(tol_cfg["abstol_H3plus"])
    abstol[I_H2P] = float(tol_cfg["abstol_H2plus"])
    abstol[I_SP] = float(tol_cfg["abstol_Splus"])
    abstol[I_SIP] = float(tol_cfg["abstol_Siplus"])
    abstol[I_OP] = float(tol_cfg["abstol_Oplus"])
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


def _gow17_checkpoint_config(rad: "RadModel", cfg: dict) -> dict:
    cp = _nested_cfg(cfg, "checkpoint")
    mode = str(cp.get("mode", "latest")).lower()
    if mode != "latest":
        raise ValueError("gow17 checkpoint.mode currently supports only 'latest'")

    path_raw = cp.get("path", cp.get("directory", "checkpoint"))
    path = Path(str(path_raw))
    if not path.is_absolute():
        path = Path(str(cfg.get("_output_dir", rad.model_dir))) / path

    every = int(cp.get("every", 1))
    if every <= 0:
        raise ValueError("gow17 checkpoint.every must be positive")

    return {
        "enabled": bool(cp.get("enabled", False)),
        "resume": bool(cp.get("resume", True)),
        "path": path,
        "every": every,
        "include_dust": bool(cp.get("include_dust", True)),
    }


def _gow17_checkpoint_histories(
    *,
    d_h2_hist,
    d_co_hist,
    d_coice_hist,
    d_tgas_hist,
    d_h2_median_hist,
    d_co_median_hist,
    d_coice_median_hist,
    d_tgas_median_hist,
    well_converged_cells_hist,
    well_converged_fraction_hist,
    bad_status_hist,
    astrochem_acceleration_cells_hist,
    astrochem_acceleration_entries_hist,
    projection_corrections_hist,
    equilibrium_solver_diag_hist,
    time_solver_diag_hist,
) -> dict:
    return {
        "d_h2_hist": list(d_h2_hist),
        "d_co_hist": list(d_co_hist),
        "d_coice_hist": list(d_coice_hist),
        "d_tgas_hist": list(d_tgas_hist),
        "d_h2_median_hist": list(d_h2_median_hist),
        "d_co_median_hist": list(d_co_median_hist),
        "d_coice_median_hist": list(d_coice_median_hist),
        "d_tgas_median_hist": list(d_tgas_median_hist),
        "well_converged_cells_hist": list(well_converged_cells_hist),
        "well_converged_fraction_hist": list(well_converged_fraction_hist),
        "bad_status_hist": list(bad_status_hist),
        "astrochem_acceleration_cells_hist": list(astrochem_acceleration_cells_hist),
        "astrochem_acceleration_entries_hist": list(astrochem_acceleration_entries_hist),
        "projection_corrections_hist": list(projection_corrections_hist),
        "equilibrium_solver_diag_hist": {
            key: list(value) for key, value in equilibrium_solver_diag_hist.items()
        },
        "time_solver_diag_hist": {
            key: list(value) for key, value in time_solver_diag_hist.items()
        },
    }


def _restore_gow17_checkpoint_histories(histories: dict, locals_by_name: dict) -> None:
    for key in (
        "d_h2_hist",
        "d_co_hist",
        "d_coice_hist",
        "d_tgas_hist",
        "d_h2_median_hist",
        "d_co_median_hist",
        "d_coice_median_hist",
        "d_tgas_median_hist",
        "well_converged_cells_hist",
        "well_converged_fraction_hist",
        "bad_status_hist",
        "astrochem_acceleration_cells_hist",
        "astrochem_acceleration_entries_hist",
        "projection_corrections_hist",
    ):
        locals_by_name[key][:] = list(histories.get(key, []))

    for parent in ("equilibrium_solver_diag_hist", "time_solver_diag_hist"):
        saved = histories.get(parent, {})
        current = locals_by_name[parent]
        for key in current:
            current[key][:] = list(saved.get(key, []))


def _gow17_derived_species(
    y_grid: np.ndarray,
    nH_cm3: np.ndarray,
    xCtot: np.ndarray,
    *,
    line_h2_opr: float,
) -> tuple[dict[str, Quantity], dict[str, Quantity]]:
    xe = _electron_abundance(y_grid)
    x_h = 1.0 - (
        y_grid[..., I_OHX]
        + y_grid[..., I_CHX]
        + y_grid[..., I_HCOP]
        + 3.0 * y_grid[..., I_H3P]
        + 2.0 * y_grid[..., I_H2P]
        + y_grid[..., I_HP]
        + 2.0 * y_grid[..., I_H2]
    )
    x_catom = xCtot - (
        y_grid[..., I_HCOP]
        + y_grid[..., I_CHX]
        + y_grid[..., I_CO]
        + y_grid[..., I_CO_ICE]
        + y_grid[..., I_CP]
    )
    return _gow17_species_outputs(
        y_grid,
        nH_cm3,
        x_h=x_h,
        x_catom=x_catom,
        x_e=xe,
        line_h2_opr=line_h2_opr,
    )


def _gow17_checkpoint_result(
    *,
    y_flat: np.ndarray,
    nH_cm3: np.ndarray,
    xCtot: np.ndarray,
    shape: tuple[int, ...],
    T_out: np.ndarray,
    theta_h2_flat: np.ndarray,
    theta_co_flat: np.ndarray,
    theta_c_flat: np.ndarray,
    theta_CO_pdes_flat: np.ndarray,
    Gph: np.ndarray,
    GISRF: np.ndarray,
    status_acc: np.ndarray,
    line_h2_opr: float,
) -> ChemistryResult:
    y_grid = y_flat.reshape(shape + (N_Y,))
    abundances, number_densities = _gow17_derived_species(
        y_grid,
        nH_cm3,
        xCtot,
        line_h2_opr=line_h2_opr,
    )
    actual_uv = _actual_solver_uv_fields(Gph=Gph, F_CO_pdes_external=GISRF, shape=shape)
    fields = {
        "Tgas": Quantity(T_out.reshape(shape), "K"),
        "status": Quantity(status_acc.reshape(shape).astype(np.float64), "dimensionless"),
        "theta_h2": Quantity(theta_h2_flat.reshape(shape), "dimensionless"),
        "theta_co": Quantity(theta_co_flat.reshape(shape), "dimensionless"),
        "theta_c": Quantity(theta_c_flat.reshape(shape), "dimensionless"),
        "theta_CO_pdes": Quantity(
            theta_CO_pdes_flat.reshape(shape),
            "dimensionless",
        ),
        "chi_eff": Quantity(actual_uv["G_CO_diss_actual"], "dimensionless"),
        "G_CO_diss_actual": Quantity(actual_uv["G_CO_diss_actual"], "dimensionless"),
        "G_C_ion_actual": Quantity(actual_uv["G_C_ion_actual"], "dimensionless"),
        "G_H2_diss_actual": Quantity(actual_uv["G_H2_diss_actual"], "dimensionless"),
        "F_CO_pdes_external_actual": Quantity(
            actual_uv["F_CO_pdes_external_actual"],
            "1/(cm^2 s)",
        ),
    }
    return ChemistryResult(
        abundances=abundances,
        number_densities=number_densities,
        fields=fields,
    )


def _write_gow17_checkpoint(
    rad: "RadModel",
    cp_cfg: dict,
    result: ChemistryResult,
    *,
    y_grid: np.ndarray,
    T_out: np.ndarray,
    completed_updates: int,
    total_updates: int,
    t_target_yr: float | None,
    histories: dict,
) -> None:
    if not cp_cfg["enabled"]:
        return

    checkpoint_dir = Path(cp_cfg["path"])
    tmp_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.tmp")
    old_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.old")

    rad.gow17_y = y_grid
    rad.Tgas_gow17 = Quantity(T_out, "K")
    rad.gas_temperature = rad.Tgas_gow17
    attach_chemistry_result_to_model(rad, result)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rad.model.save_hdf5(
        tmp_dir / "model_snapshot.h5",
        include_dust=bool(cp_cfg["include_dust"]),
        overwrite=True,
    )
    metadata = {
        "checkpoint_type": "gow17_latest",
        "completed_updates": int(completed_updates),
        "total_updates": int(total_updates),
        "t_target_yr": None if t_target_yr is None else float(t_target_yr),
        "histories": histories,
    }
    (tmp_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    if old_dir.exists():
        shutil.rmtree(old_dir)
    if checkpoint_dir.exists():
        checkpoint_dir.rename(old_dir)
    tmp_dir.rename(checkpoint_dir)
    if old_dir.exists():
        shutil.rmtree(old_dir)
    logger.info("gow17 checkpoint: wrote %s", checkpoint_dir / "model_snapshot.h5")


def _load_gow17_checkpoint(
    rad: "RadModel",
    cp_cfg: dict,
    *,
    shape: tuple[int, ...],
    ncells: int,
    total_updates: int,
    const_temp: bool,
    enable_co_phase: bool,
) -> _Gow17CheckpointState | None:
    if not (cp_cfg["enabled"] and cp_cfg["resume"]):
        return None

    checkpoint_dir = Path(cp_cfg["path"])
    snapshot = checkpoint_dir / "model_snapshot.h5"
    metadata_path = checkpoint_dir / "metadata.json"
    if not (snapshot.exists() and metadata_path.exists()):
        return None

    from diskbridge.model.io_hdf5 import load_model_hdf5

    checkpoint_model = load_model_hdf5(snapshot)
    if tuple(checkpoint_model.mesh.shape) != tuple(shape):
        raise ValueError(
            "gow17 checkpoint mesh shape does not match active model: "
            f"{tuple(checkpoint_model.mesh.shape)} != {tuple(shape)}"
        )

    fields = checkpoint_model.gas
    y = np.zeros((ncells, N_Y), dtype=np.float64)
    for species, idx in GOW17_STATE_SPECIES.items():
        field_name = f"abundance_{species}"
        if field_name not in fields:
            raise KeyError(f"gow17 checkpoint is missing gas field {field_name!r}")
        y[:, idx] = np.asarray(
            fields[field_name].data.to("dimensionless").magnitude,
            dtype=np.float64,
        ).reshape(ncells)

    if not enable_co_phase:
        y[:, I_CO_ICE] = 0.0
    if not const_temp:
        if "gas_temperature" not in fields:
            raise KeyError("gow17 checkpoint is missing gas field 'gas_temperature'")
        T_restore = np.asarray(
            fields["gas_temperature"].data.to("K").magnitude,
            dtype=np.float64,
        ).reshape(ncells)
        y[:, I_E] = _cv_cold(y[:, I_H2], _electron_abundance(y)) * T_restore

    def _field_or_ones(name: str) -> np.ndarray:
        field_name = f"chem_{name}"
        if field_name not in fields:
            return np.ones(ncells, dtype=np.float64)
        return np.asarray(
            fields[field_name].data.to("dimensionless").magnitude,
            dtype=np.float64,
        ).reshape(ncells)

    status = np.zeros(ncells, dtype=np.int32)
    if "chem_status" in fields:
        status = np.asarray(
            fields["chem_status"].data.to("dimensionless").magnitude,
            dtype=np.int32,
        ).reshape(ncells)

    metadata = json.loads(metadata_path.read_text())
    saved_total = int(metadata.get("total_updates", total_updates))
    if saved_total != int(total_updates):
        raise ValueError(
            "gow17 checkpoint total_updates does not match active config: "
            f"{saved_total} != {int(total_updates)}. Delete the checkpoint or use matching config."
        )
    completed_updates = int(metadata.get("completed_updates", 0))
    logger.info(
        "gow17 checkpoint: resuming from %s after %d completed update(s)",
        snapshot,
        completed_updates,
    )
    return _Gow17CheckpointState(
        y=np.ascontiguousarray(y, dtype=np.float64),
        theta_h2=_field_or_ones("theta_h2"),
        theta_co=_field_or_ones("theta_co"),
        theta_c=_field_or_ones("theta_c"),
        status=status,
        completed_updates=completed_updates,
        histories=dict(metadata.get("histories", {})),
    )

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

    ``Gph`` already includes the radiation-mode normalization and molecular/atomic shielding.
    ``F_CO_pdes_external`` is the external CO photodesorption photon flux passed
    to the solver before CRUV/direct CR terms are added in CO phase diagnostics.
    """
    Gph_arr = np.asarray(Gph, dtype=np.float64)
    F_arr = np.asarray(F_CO_pdes_external, dtype=np.float64)
    return {
        "G_C_ion_actual": Gph_arr[:, IPH_C].reshape(shape),
        "G_CH_diss_actual": Gph_arr[:, IPH_CH].reshape(shape),
        "G_CO_diss_actual": Gph_arr[:, IPH_CO].reshape(shape),
        "G_OH_diss_actual": Gph_arr[:, IPH_OH].reshape(shape),
        "G_H2_diss_actual": Gph_arr[:, IPH_H2].reshape(shape),
        "G_S_ion_actual": Gph_arr[:, IPH_S].reshape(shape),
        "G_Si_ion_actual": Gph_arr[:, IPH_SI].reshape(shape),
        "F_CO_pdes_external_actual": F_arr.reshape(shape),
    }


def _shield_external_band_flux(
    F_external_unshielded: np.ndarray,
    theta_external: np.ndarray,
    *,
    ncells: int,
    name: str = "external band flux",
) -> tuple[np.ndarray, np.ndarray]:
    """Return effective shielding and shielded flux for scalar or banded fields."""
    F_raw = np.asarray(F_external_unshielded, dtype=np.float64)
    theta = np.asarray(theta_external, dtype=np.float64)

    if F_raw.ndim == 1:
        F_flat = F_raw.reshape(ncells)
        theta_flat = theta.reshape(ncells)
        return (
            np.ascontiguousarray(theta_flat, dtype=np.float64),
            np.ascontiguousarray(F_flat * theta_flat, dtype=np.float64),
        )

    if F_raw.ndim != 2:
        raise ValueError(
            f"{name} must be 1D or 2D "
            f"after flattening, got ndim={F_raw.ndim}"
        )

    F_bands = F_raw
    if F_bands.shape[-1] != ncells and F_bands.shape[0] == ncells:
        F_bands = F_bands.T
    if F_bands.shape[-1] != ncells:
        raise ValueError(
            f"{name} array must have one axis of length "
            f"ncells={ncells}, got shape={F_raw.shape}"
        )

    if theta.ndim == 1:
        theta_bands = np.broadcast_to(theta.reshape(1, ncells), F_bands.shape)
    else:
        theta_bands = theta
        if theta_bands.shape[-1] != ncells and theta_bands.shape[0] == ncells:
            theta_bands = theta_bands.T
        if theta_bands.shape != F_bands.shape:
            raise ValueError(
                f"{name} shielding array must match flux bands; "
                f"got theta={theta.shape}, flux={F_raw.shape}"
            )

    F_shielded = np.sum(F_bands * theta_bands, axis=0)
    F_sum = np.sum(F_bands, axis=0)
    theta_eff = np.divide(
        F_shielded,
        F_sum,
        out=np.ones(ncells, dtype=np.float64),
        where=(F_sum != 0.0),
    )
    return (
        np.ascontiguousarray(theta_eff, dtype=np.float64),
        np.ascontiguousarray(F_shielded, dtype=np.float64),
    )


def _flatten_band_field(
    arr: np.ndarray,
    *,
    shape: tuple,
    ncells: int,
    name: str = "band field",
) -> np.ndarray:
    """Flatten a scalar or band-resolved grid field to ``(nbands, ncells)``."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape == shape:
        return np.ascontiguousarray(arr.reshape(1, ncells), dtype=np.float64)
    if arr.ndim == len(shape) + 1 and arr.shape[1:] == shape:
        return np.ascontiguousarray(arr.reshape(arr.shape[0], ncells), dtype=np.float64)
    if arr.ndim == 2 and arr.shape[-1] == ncells:
        return np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == ncells:
        return np.ascontiguousarray(arr.T, dtype=np.float64)
    raise ValueError(
        f"{name} must have shape "
        f"shape, (nbands, *shape), (nbands, ncells), or (ncells, nbands); got {arr.shape}"
    )


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
        dust_cfg["sigma_d_ISM_ref"],
        "cm^2",
    )
    if sigma_ref <= 0.0:
        raise ValueError("gow17 dust sigma_d_ISM_ref must be positive")

    sigma_source = str(dust_cfg["sigma_d_CO_per_H_source"]).lower()
    if sigma_source == "radmc":
        sigma_d_CO = np.ascontiguousarray(sigma_d_cm2, dtype=np.float64)
    elif sigma_source == "constant":
        sigma_const = _maybe_quantity_to_float(
            dust_cfg["sigma_d_CO_per_H_constant"],
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

    for key in ("Zd_gow17_grain_source", "Zd_gow17_grain_constant", "allow_median_sigma_ref"):
        if key in dust_cfg:
            raise ValueError(
                f"gow17 dust option {key!r} is no longer supported; use top-level Zd "
                "for GOW17 grain chemistry/heating and [chemistry.gow17.dust_cooling] "
                "for gas-dust thermal coupling"
            )

    zd = _broadcast_scalar_or_array(float(cfg["Zd"]), ncells)

    if not np.all(np.isfinite(zd)):
        raise ValueError("Zd_gow17_grain contains non-finite values")
    return sigma_d_CO, np.maximum(zd, 0.0), float(sigma_ref)


def _maybe_model_field(rad: "RadModel", name: str):
    model = rad.model
    if model.gas is not None and name in model.gas:
        return model.gas[name], f"gas.{name}"
    if model.dust is not None and name in model.dust:
        return model.dust[name], f"dust.{name}"
    return None, None


def _resolve_pah_scaling(
    *,
    rad: "RadModel",
    nH_flat: np.ndarray,
    Zd_arr: np.ndarray,
    ncells: int,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Return D_PAH per cell, falling back to original GOW17 Zd behavior."""
    target = rad._chem_axis_order()
    abundance_field, abundance_source = _maybe_model_field(
        rad, "pah_abundance_rel_ism"
    )
    density_field, density_source = _maybe_model_field(rad, "pah_density")

    D_from_abundance = None
    D_from_density = None
    rho_pah_flat = None
    if abundance_field is not None:
        D_from_abundance = np.ascontiguousarray(
            field_data_as_order(abundance_field, target)
            .to("dimensionless")
            .magnitude,
            dtype=np.float64,
        ).reshape(ncells)
    if density_field is not None:
        rho_pah_flat = np.ascontiguousarray(
            field_data_as_order(density_field, target)
            .to("g/cm^3")
            .magnitude,
            dtype=np.float64,
        ).reshape(ncells)
        denom = nH_flat * PAH_X_ISM_DEFAULT * PAH_MASS_G_DEFAULT
        D_from_density = np.divide(
            rho_pah_flat,
            denom,
            out=np.zeros(ncells, dtype=np.float64),
            where=(denom > 0.0),
        )

    if D_from_abundance is not None and D_from_density is not None:
        ok = np.isfinite(D_from_abundance) & np.isfinite(D_from_density)
        if np.any(ok):
            rel_err = np.max(
                np.abs(D_from_abundance[ok] - D_from_density[ok])
                / np.maximum(1.0, np.abs(D_from_abundance[ok]))
            )
            if rel_err > 1.0e-6:
                raise ValueError(
                    "pah_abundance_rel_ism and pah_density are inconsistent "
                    f"(max relative mismatch {rel_err:.3e})"
                )
        D_pah = D_from_abundance
        source = f"{abundance_source}+{density_source}"
    elif D_from_abundance is not None:
        D_pah = D_from_abundance
        source = abundance_source
    elif D_from_density is not None:
        D_pah = D_from_density
        source = density_source
    else:
        D_pah = np.ascontiguousarray(Zd_arr, dtype=np.float64)
        source = "gow17_original_Zd"

    if not np.all(np.isfinite(D_pah)):
        raise ValueError("D_PAH contains non-finite values")
    D_pah = np.maximum(np.ascontiguousarray(D_pah, dtype=np.float64), 0.0)
    meta = {
        "pah_source": source,
        "pah_uses_explicit_component": source != "gow17_original_Zd",
        "pah_N_C": PAH_N_C_DEFAULT,
        "pah_N_H": PAH_N_H_DEFAULT,
        "pah_x_ISM": PAH_X_ISM_DEFAULT,
        "pah_mass_g": PAH_MASS_G_DEFAULT,
    }
    return D_pah, rho_pah_flat, meta


def _resolve_h2_grain_scaling(
    *,
    rad: "RadModel",
    Zd_arr: np.ndarray,
    ncells: int,
) -> tuple[np.ndarray, np.ndarray | None, float | None, dict]:
    """Resolve D_H2gr from ordinary-grain projected area, or fallback to Zd."""
    model = rad.model
    if model.dust is None or int(model.dust.nbin) == 0:
        D = np.ascontiguousarray(Zd_arr, dtype=np.float64)
        return D, None, None, {
            "h2gr_source": "gow17_original_Zd",
            "h2gr_uses_ordinary_dust": False,
            "sigma_H2gr_ref_cm2": None,
        }

    sigma_q = rad.compute_h2_formation_surface_area()
    sigma_arr = np.ascontiguousarray(
        sigma_q.to("cm^2").magnitude,
        dtype=np.float64,
    ).reshape(ncells)
    sigma_ref_q = model.dust.h2_formation_reference_area_per_H()
    sigma_ref = float(sigma_ref_q.to("cm^2").magnitude)
    if not np.isfinite(sigma_ref) or sigma_ref <= 0.0:
        raise ValueError("Invalid sigma_H2gr_ref")

    D = np.divide(
        sigma_arr,
        sigma_ref,
        out=np.zeros(ncells, dtype=np.float64),
        where=(sigma_ref > 0.0),
    )
    D = np.where(np.isfinite(D) & (D >= 0.0), D, 0.0)
    return np.ascontiguousarray(D, dtype=np.float64), sigma_arr, sigma_ref, {
        "h2gr_source": "ordinary_dust_surface_area",
        "h2gr_uses_ordinary_dust": True,
        "sigma_H2gr_ref_cm2": sigma_ref,
    }


def _resolve_dust_cooling_controls(
    *,
    cfg: dict,
    rad: "RadModel",
    ncells: int,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float]:
    """Resolve the only allowed gas-dust thermal coupling modes."""
    dust_cooling_cfg = _nested_cfg(cfg, "dust_cooling")
    mode = str(dust_cooling_cfg["mode"]).lower()
    if mode not in ("surface_area", "gow17_original"):
        raise ValueError(
            "gow17 dust_cooling.mode must be 'surface_area' or 'gow17_original', "
            f"got {mode!r}"
        )
    for legacy in ("gas_dust_ratio", "surface_area_relative", "allow_median_sigma_ref"):
        if legacy in dust_cooling_cfg:
            raise ValueError(
                f"gow17 dust_cooling option {legacy!r} is no longer supported; "
                "allowed modes are 'surface_area' and 'gow17_original'"
            )

    sigma_ref = _maybe_quantity_to_float(
        dust_cooling_cfg["sigma_d_H_ref"],
        "cm^2",
    )
    if sigma_ref <= 0.0:
        raise ValueError("gow17 dust_cooling sigma_d_H_ref must be positive")

    if mode == "surface_area":
        sigma_q, Tdust_q = rad.compute_gas_dust_surface_area_coupling()
        sigma_total = np.ascontiguousarray(sigma_q.to("cm^2").magnitude, dtype=np.float64).reshape(ncells)
        Tdust_gd = np.ascontiguousarray(Tdust_q.to("K").magnitude, dtype=np.float64).reshape(ncells)
        if not np.all(np.isfinite(sigma_total)):
            raise ValueError("sigma_d_per_H_total contains non-finite values")
        if not np.all(np.isfinite(Tdust_gd)):
            raise ValueError("Tdust_gd_surface_weighted contains non-finite values")
        sigma_total = np.maximum(sigma_total, 0.0)
        Zgd = np.divide(
            sigma_total,
            sigma_ref,
            out=np.zeros(ncells, dtype=np.float64),
            where=(sigma_ref > 0.0),
        )
        Tdust_gd = np.where(sigma_total > 0.0, Tdust_gd, 10.0)
    else:
        zgd_const = float(dust_cooling_cfg["gow17_original_Zd"])
        if zgd_const < 0.0:
            raise ValueError("gow17_original_Zd must be non-negative")
        sigma_total = np.zeros(ncells, dtype=np.float64)
        Zgd = np.full(
            ncells,
            zgd_const,
            dtype=np.float64,
        )
        Tdust_gd = np.full(
            ncells,
            _maybe_quantity_to_float(dust_cooling_cfg["gow17_original_Tdust"], "K"),
            dtype=np.float64,
        )

    if not np.all(np.isfinite(Zgd)):
        raise ValueError("Zgd contains non-finite values")
    if not np.all(np.isfinite(Tdust_gd)):
        raise ValueError("gas-dust coupling Tdust contains non-finite values")
    return mode, np.maximum(Zgd, 0.0), Tdust_gd, sigma_total, float(sigma_ref)


def _gas_dust_exchange(
    *,
    Zgd: np.ndarray,
    nH_cm3: np.ndarray,
    Tgas_K: np.ndarray,
    Tdust_K: np.ndarray,
) -> np.ndarray:
    Zgd_arr = np.asarray(Zgd, dtype=np.float64)
    nH_arr = np.asarray(nH_cm3, dtype=np.float64)
    Tgas_arr = np.asarray(Tgas_K, dtype=np.float64)
    Tdust_arr = np.asarray(Tdust_K, dtype=np.float64)
    out = np.zeros_like(Tgas_arr, dtype=np.float64)
    valid = (Zgd_arr > 0.0) & (nH_arr > 0.0) & (Tgas_arr > 0.0)
    out[valid] = (
        ALPHA_GD_CGS
        * Zgd_arr[valid]
        * nH_arr[valid]
        * np.sqrt(Tgas_arr[valid])
        * (Tgas_arr[valid] - Tdust_arr[valid])
    )
    return out


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
    s_mode = str(co_cfg["S_CO_mode"]).lower()
    if s_mode != "constant":
        raise ValueError(f"Unsupported S_CO_mode={s_mode!r}")
    S_CO = float(co_cfg["S_CO"])
    if S_CO < 0.0:
        raise ValueError("S_CO must be non-negative")

    enable_cruv = bool(co_cfg["enable_cruv_pdes"])
    if enable_cruv:
        F_ref = _maybe_quantity_to_float(
            co_cfg["F_CRUV_CO_pdes_ref"],
            "1/(cm^2 s)",
        )
        if bool(co_cfg["cruv_scales_with_zeta"]):
            zeta_ref = _maybe_quantity_to_float(
                co_cfg["zeta_ref"],
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

    if bool(co_cfg["enable_crdes_CO"]):
        k_crdes = _maybe_quantity_to_float(
            co_cfg["k_crdes_CO"],
            "1/s",
        )
        k_crdes_arr = np.full(ncells, k_crdes, dtype=np.float64)
    else:
        k_crdes_arr = np.zeros(ncells, dtype=np.float64)

    return float(S_CO), np.maximum(F_cruv, 0.0), np.maximum(k_crdes_arr, 0.0)




def _resolve_co_phase_runtime_params(cfg: dict) -> dict[str, float | int]:
    """Resolve CO phase physical parameters from resolved config."""
    chemistry_cfg = get_config().get("chemistry", {})
    common_cfg = chemistry_cfg.get("common", {}) if isinstance(chemistry_cfg, dict) else {}
    co_cfg = _nested_cfg(cfg, "co_phase")

    E_bind_CO = _maybe_quantity_to_float(
        co_cfg.get("E_bind_co", common_cfg["E_bind_co"]),
        "K",
    )
    nu0_CO = _maybe_quantity_to_float(
        co_cfg.get("nu0_co", common_cfg["nu0_co"]),
        "1/s",
    )
    Y_CO_local = float(co_cfg.get("Y_CO", common_cfg["Y_CO"]))
    N_LAY_local = int(co_cfg.get("N_LAY", common_cfg["N_LAY"]))
    N_SURF_local = _maybe_quantity_to_float(
        co_cfg.get("n_surf", common_cfg["n_surf"]),
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


def _resolve_shielding_linewidth(
    Tgas_K: np.ndarray,
    v_turb_grid_kms: np.ndarray,
) -> tuple[float, np.ndarray, float, np.ndarray, dict]:
    """Resolve H2 and CO shielding linewidths from Tgas and microturbulence."""
    T_arr = np.asarray(Tgas_K, dtype=np.float64)

    if not np.all(np.isfinite(T_arr)):
        bad = np.argwhere(~np.isfinite(T_arr))
        raise ValueError(
            "Tgas contains non-finite values while resolving CO shielding linewidth; "
            f"first bad index={tuple(int(i) for i in bad[0])}, "
            f"n_bad={bad.shape[0]}"
        )

    v_grid = np.asarray(v_turb_grid_kms, dtype=np.float64)
    if v_grid.shape != T_arr.shape:
        raise ValueError(
            f"microturbulence grid shape {v_grid.shape} does not match Tgas shape {T_arr.shape}"
        )
    if not np.all(np.isfinite(v_grid)):
        bad = np.argwhere(~np.isfinite(v_grid))
        raise ValueError(
            "microturbulence grid contains non-finite values; "
            f"first bad index={tuple(int(i) for i in bad[0])}, "
            f"n_bad={bad.shape[0]}"
        )
    if np.any(v_grid < 0.0):
        bad = np.argwhere(v_grid < 0.0)
        raise ValueError(
            "microturbulence grid contains negative values; "
            f"first bad index={tuple(int(i) for i in bad[0])}, "
            f"n_bad={bad.shape[0]}"
        )

    def _species_b_kms(mass_g: float) -> np.ndarray:
        b_thermal = np.sqrt(2.0 * KB_CGS * np.maximum(T_arr, 0.0) / mass_g) / 1.0e5
        return np.sqrt(b_thermal * b_thermal + v_grid * v_grid)

    b_H2_arr = _species_b_kms(2.0 * float(M_H))
    b_CO_arr = _species_b_kms(28.0 * float(M_H))

    for label, b_arr in (("H2", b_H2_arr), ("CO", b_CO_arr)):
        if not np.all(np.isfinite(b_arr)):
            bad = np.argwhere(~np.isfinite(b_arr))
            raise ValueError(
                f"{label} shielding linewidth b_{label} contains non-finite values; "
                f"first bad index={tuple(int(i) for i in bad[0])}, "
                f"n_bad={bad.shape[0]}"
            )
        if np.any(b_arr <= 0.0):
            bad = np.argwhere(b_arr <= 0.0)
            raise ValueError(
                f"{label} shielding linewidth b_{label} contains non-positive values; "
                f"first bad index={tuple(int(i) for i in bad[0])}, "
                f"n_bad={bad.shape[0]}"
            )

    b_H2_scalar = float(np.median(b_H2_arr))
    b_CO_scalar = float(np.median(b_CO_arr))
    meta = {
        "b_H2_scalar_kms": float(b_H2_scalar),
        "b_CO_scalar_kms": float(b_CO_scalar),
        "b_CO_scalar_approximation": True,
    }
    return b_H2_scalar, b_H2_arr, b_CO_scalar, b_CO_arr, meta


def _model_microturbulence_grid_kms(rad: "RadModel", shape: tuple[int, ...]) -> np.ndarray:
    gas = getattr(rad.model, "gas", None)
    if gas is None or "microturbulence" not in gas:
        raise ValueError("GOW17 shielding requires params.microturbulence")
    field = gas["microturbulence"]
    arr = np.asarray(field.data.to("km/s").magnitude, dtype=np.float64)
    if arr.shape != tuple(shape):
        raise ValueError(
            "model.gas['microturbulence'] shape "
            f"{arr.shape} does not match chemistry field shape {tuple(shape)}"
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _shielding_b_grid_or_none(
    b_kms_arr: np.ndarray,
) -> np.ndarray | None:
    """Return a b-grid only when it actually varies across the domain."""
    arr = np.asarray(b_kms_arr, dtype=np.float64)
    if arr.size == 0:
        return None
    first = float(arr.reshape(-1)[0])
    if np.allclose(arr, first, rtol=0.0, atol=0.0):
        return None
    return np.ascontiguousarray(arr, dtype=np.float64)


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
    try:
        b_available, _ = visser._available_b_family()
        meta["b_CO_table_interpolation"] = True
        meta["b_CO_bins_available"] = [float(v) for v in b_available]
    except Exception:
        meta["b_CO_table_interpolation"] = True

    if abs(b_table - b_requested) > 1.0e-6:
        logger.info(
            "gow17: requested representative CO shielding b_kms=%.6g km/s; "
            "loaded nearest Visser family at %.6g km/s for table interpolation. "
            "Ray-wise CO b values are interpolated across the available Visser tables.",
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
    D_pah_arr: np.ndarray,
    Dh2gr_arr: np.ndarray,
    Zgd_arr: np.ndarray,
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
            Dpah=np.ascontiguousarray(D_pah_arr, dtype=np.float64),
            Dh2gr=np.ascontiguousarray(Dh2gr_arr, dtype=np.float64),
            Zgd=np.ascontiguousarray(Zgd_arr, dtype=np.float64),
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
    co_phase_params: dict[str, float | int],
) -> dict:
    """Compute solver-consistent CO gas/ice phase-rate diagnostics."""
    shape = y_out.shape[:-1]
    ncells = int(np.prod(shape))

    E_bind_CO_local = float(co_phase_params["E_bind_CO"])
    nu0_CO_local = float(co_phase_params["nu0_CO"])
    Y_CO_local = float(co_phase_params["Y_CO"])
    N_LAY_local = int(co_phase_params["N_LAY"])
    N_SURF_local = float(co_phase_params["N_SURF"])

    xCO = np.asarray(y_out[..., I_CO], dtype=np.float64).reshape(ncells)
    xCO_ice = np.asarray(y_out[..., I_CO_ICE], dtype=np.float64).reshape(ncells)
    xHeplus = np.asarray(y_out[..., I_HEP], dtype=np.float64).reshape(ncells)
    nH_flat = np.asarray(nH_cm3, dtype=np.float64).reshape(ncells)
    Tgas_flat = np.asarray(Tgas_K, dtype=np.float64).reshape(ncells)
    Tdust_flat = np.asarray(Tdust_K, dtype=np.float64).reshape(ncells)
    sigma_flat = np.asarray(sigma_d_CO_per_H, dtype=np.float64).reshape(ncells)
    F_ext_shielded = np.asarray(F_CO_pdes_photon, dtype=np.float64).reshape(ncells)
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
    F_total = np.maximum(F_ext_shielded, 0.0) + np.maximum(F_cruv, 0.0)
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
        "F_CO_pdes_external_actual": Quantity(F_ext_shielded.reshape(shape), "1/(cm^2 s)"),
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
    F_CO_pdes_photon_bands_flat: np.ndarray | None,
    F_CO_pdes_photon_flat: np.ndarray | None,
    xCtot_flat: np.ndarray,
    Zd_arr: np.ndarray,
    shape: tuple,
    ncells: int,
    rad: "RadModel",
    nH_cm3: np.ndarray,
    chi_dust_arr: np.ndarray,
    visser,
    b_H2_kms: float,
    b_CO_kms: float,
    b_H2_kms_grid: np.ndarray | None,
    b_CO_kms_grid: np.ndarray | None,
    nside: int,
) -> tuple:
    """Compute shielding factors and radiation field arrays from current abundances.

    Returns
    -------
    theta_h2_flat, theta_co_flat, theta_c_flat, theta_CO_pdes_flat : np.ndarray
        Shielding factors, shape ``(ncells,)``.
    Gph, GPE, F_CO_pdes_ext_shielded : np.ndarray
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

    from diskbridge.chemistry.shielding.healpix_columns import (
        compute_pdr_shielding_healpix,
    )

    W_rays = getattr(rad, "W_rays", None)
    n_pdes_bands = (
        1
        if F_CO_pdes_photon_bands_flat is None
        else int(np.asarray(F_CO_pdes_photon_bands_flat).shape[0])
    )
    (
        theta_h2_arr,
        theta_co_arr,
        theta_c_arr,
        _,
        _,
        theta_CO_pdes_bands_arr,
    ) = compute_pdr_shielding_healpix(
        mesh=rad.model.mesh,
        nH=nH_cm3,
        chi=chi_dust_arr,
        visser=visser,
        nCO=nCO_cm3,
        nC=nC_cm3,
        nH2=nH2_cm3,
        nside=nside,
        b_H2_kms=b_H2_kms,
        b_CO_kms=b_CO_kms,
        b_H2_kms_grid=b_H2_kms_grid,
        b_CO_kms_grid=b_CO_kms_grid,
        W_rays=W_rays,
        chunk_size=getattr(diskbridge.params, "pdr_shielding_chunk_size", None),
        memory_budget_gib=getattr(
            diskbridge.params,
            "pdr_shielding_memory_budget_gib",
            None,
        ),
        co_shielding_nbands=n_pdes_bands,
    )

    theta_h2_flat = theta_h2_arr.reshape(ncells)
    theta_co_flat = theta_co_arr.reshape(ncells)
    theta_c_flat = theta_c_arr.reshape(ncells)

    # -- Assemble radiation field arrays --
    Gph = np.empty((ncells, N_PH), dtype=np.float64)
    G_broad = chi_dust_flat
    Gph[:, :] = G_broad[:, None]
    Gph[:, IPH_CO] = G_CO_diss_flat
    Gph[:, IPH_H2] = G_H2_diss_flat
    Gph[:, IPH_C] = G_C_ion_flat
    GPE = np.ascontiguousarray(G_broad.copy(), dtype=np.float64)
    if F_CO_pdes_photon_bands_flat is None and F_CO_pdes_photon_flat is None:
        f_co_pdes_ref = _co_pdes_draine_flux()
        F_CO_pdes_ext_unshielded_bands = np.ascontiguousarray(
            (G_CO_pdes_flat * f_co_pdes_ref).reshape(1, ncells),
            dtype=np.float64,
        )
    elif F_CO_pdes_photon_bands_flat is None:
        F_CO_pdes_ext_unshielded_bands = np.ascontiguousarray(
            np.asarray(F_CO_pdes_photon_flat, dtype=np.float64).reshape(1, ncells),
            dtype=np.float64,
        )
    else:
        F_CO_pdes_ext_unshielded_bands = np.ascontiguousarray(
            F_CO_pdes_photon_bands_flat,
            dtype=np.float64,
        )

    theta_CO_pdes_bands_flat = theta_CO_pdes_bands_arr.reshape(n_pdes_bands, ncells)
    theta_CO_pdes_flat, F_CO_pdes_ext_shielded = _shield_external_band_flux(
        F_CO_pdes_ext_unshielded_bands,
        theta_CO_pdes_bands_flat,
        ncells=ncells,
        name="CO photodesorption external flux",
    )

    Gph[:, IPH_C] *= theta_c_flat
    Gph[:, IPH_CO] *= theta_co_flat
    Gph[:, IPH_H2] *= theta_h2_flat

    return (
        theta_h2_flat,
        theta_co_flat,
        theta_c_flat,
        theta_CO_pdes_flat,
        Gph,
        GPE,
        F_CO_pdes_ext_shielded,
    )


def run_gow17(rad: "RadModel", config: dict) -> ChemistryResult:
    cfg = resolve_model_config(("chemistry", "gow17"), overrides=config)
    checkpoint_cfg = _gow17_checkpoint_config(rad, cfg)

    nside = int(diskbridge.params.nside)
    ion_rate_s = Quantity(cfg["ion_rate"]).to("1/s").magnitude

    Zg = float(cfg["Zg"])

    enable_co_phase = bool(cfg["enable_co_phase"])
    _warn_if_co_phase_settings_ignored(
        cfg,
        enable_co_phase=enable_co_phase,
        context="gow17",
    )

    fH2gr = float(cfg["fH2gr"])
    fHplusgr = float(cfg["fHplusgr"])
    fCplusgr = float(cfg["fCplusgr"])
    fHeplusgr = float(cfg["fHeplusgr"])
    fSplusgr = float(cfg["fSplusgr"])
    fSiplusgr = float(cfg["fSiplusgr"])
    fCplusCR = float(cfg["fCplusCR"])

    gradv_scalar = float(cfg["gradv"])
    gradv_mode = str(cfg["gradv_mode"])
    gradv_q = float(cfg["gradv_q"])
    gradv_N0 = float(cfg["gradv_N0"])
    gradv_p = float(cfg["gradv_p"])
    gradv_f_corr = float(cfg["gradv_f_corr"])
    gradv_gmin = float(cfg["gradv_gmin"])
    gradv_gmax = float(cfg["gradv_gmax"])

    Leff_CO_max_mode = str(cfg["Leff_CO_max_mode"])
    L_geo_reduction = str(cfg["L_geo_reduction"])
    L_geo_min = float(cfg["L_geo_min"])
    L_geo_max = float(cfg["L_geo_max"])
    Leff_CO_max_scalar = float(cfg["Leff_CO_max"])
    if "isDust_cooling" in cfg:
        raise ValueError("gow17 isDust_cooling is no longer supported; use dust_cooling.mode")
    isDust_cooling = True
    isCoolingCOThin = bool(cfg["isCoolingCOThin"])
    temperature_cfg = _resolve_temperature_config(cfg)
    temperature_mode = str(temperature_cfg["mode"])
    const_temp = bool(temperature_cfg["const_temp"])

    reltol = float(cfg["reltol"])
    abstol0 = float(cfg["abstol0"])
    tolfac = float(cfg["tolfac"])
    tmin = _maybe_quantity_to_float(cfg["tmin"], "s")
    tmax = _maybe_quantity_to_float(cfg["tmax"], "s")
    mxsteps = int(cfg["mxsteps"])
    maxord = int(cfg["maxord"])
    userJac = bool(cfg["userJac"])
    verbose = bool(cfg["verbose"])

    shielding_max_iter = int(cfg["shielding_max_iter"])
    shielding_reltol = float(cfg["shielding_reltol"])
    shielding_abstol = float(cfg["shielding_abstol"])
    tgas_convergence_reltol = float(cfg["tgas_convergence_reltol"])

    # -- AstroChem-style coupling config --
    coupling_mode = str(cfg["coupling_mode"])
    astrochem_n_updates = int(cfg["astrochem_n_updates"])
    astrochem_t_end_yr = float(cfg["astrochem_t_end_yr"])
    astrochem_acceleration = str(cfg["astrochem_acceleration"]).lower()
    astrochem_acceleration_start = int(cfg["astrochem_acceleration_start"])
    astrochem_acceleration_max_jump = float(cfg["astrochem_acceleration_max_jump"])
    if astrochem_acceleration not in ("none", "aitken"):
        raise ValueError(
            "astrochem_acceleration must be 'none' or 'aitken', "
            f"got {astrochem_acceleration!r}"
        )

    Tdust = rad.ensure_dust_temperature()
    nH = rad.ensure_nH()
    radiation_mode = _infer_gow17_radiation_mode(rad)
    if temperature_mode == "dust":
        Tgas = Tdust
    else:
        Tgas = _initial_gas_temperature(rad, Tdust, temperature_cfg)

    f_co_pdes_ref = _co_pdes_draine_flux()
    if radiation_mode.uses_uv_products:
        chi = rad.ensure_uv_product("chi_broad", fallback_to_chi=False)
        G_CO_diss = rad.ensure_uv_product("G_CO_diss", fallback_to_chi=False)
        G_H2_diss = rad.ensure_uv_product("G_H2_diss", fallback_to_chi=False)
        G_C_ion = rad.ensure_uv_product("G_C_ion", fallback_to_chi=False)
        G_CO_pdes = rad.ensure_uv_product("G_CO_pdes", fallback_to_chi=False)
        F_CO_pdes_photon = rad.ensure_uv_product(
            "F_CO_pdes_photon",
            fallback_to_chi=False,
        )
        F_CO_pdes_photon_bands = rad.ensure_uv_product(
            "F_CO_pdes_photon_bands",
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
        F_CO_pdes_photon_bands = F_CO_pdes_photon

    # Pre-compute directional UV weights (W_rays) once for reuse across
    # shielding iterations.  Returns None for 1-D meshes or when dustkappa
    # files are unavailable (shielding then falls back to isotropic averaging).
    directional_uv_product = radiation_mode.directional_uv_product
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
    F_CO_pdes_photon_bands_arr = _as_cgs_f64(
        F_CO_pdes_photon_bands,
        "1/(cm^2 s)",
    )

    shape = nH_cm3.shape
    ncells = nH_cm3.size

    ensure_microturbulence_field(rad.model, diskbridge.params)
    v_turb_grid_kms = _model_microturbulence_grid_kms(rad, shape)
    (
        b_H2_kms,
        b_H2_kms_arr,
        b_CO_kms,
        b_CO_kms_arr,
        shielding_linewidth_meta,
    ) = _resolve_shielding_linewidth(
        T_K,
        v_turb_grid_kms,
    )
    b_H2_kms_grid = _shielding_b_grid_or_none(b_H2_kms_arr)
    b_CO_kms_grid = _shielding_b_grid_or_none(b_CO_kms_arr)
    shielding_linewidth_meta["b_H2_ray_grid"] = b_H2_kms_grid is not None
    shielding_linewidth_meta["b_CO_ray_grid"] = b_CO_kms_grid is not None

    nH_flat = nH_cm3.reshape(ncells)
    T_flat = T_K.reshape(ncells)
    chi_dust_flat = chi_dust_arr.reshape(ncells)
    G_CO_diss_flat = G_CO_diss_arr.reshape(ncells)
    G_H2_diss_flat = G_H2_diss_arr.reshape(ncells)
    G_C_ion_flat = G_C_ion_arr.reshape(ncells)
    G_CO_pdes_flat = G_CO_pdes_arr.reshape(ncells)
    F_CO_pdes_photon_flat = F_CO_pdes_photon_arr.reshape(ncells)
    F_CO_pdes_photon_bands_flat = _flatten_band_field(
        F_CO_pdes_photon_bands_arr,
        shape=shape,
        ncells=ncells,
        name="CO photodesorption band field",
    )

    Zg_arr = _broadcast_scalar_or_array(Zg, ncells)
    ion_rate_arr = _broadcast_scalar_or_array(ion_rate_s, ncells)

    (
        dust_cooling_mode,
        Zgd_arr,
        Tdust_flat,
        sigma_d_per_H_total,
        sigma_d_H_ref,
    ) = _resolve_dust_cooling_controls(
        cfg=cfg,
        rad=rad,
        ncells=ncells,
    )

    sigma_d = rad.ensure_sigma_d_per_H()
    sigma_d_cm2 = _as_cgs_f64(sigma_d, "cm^2").reshape(ncells)
    sigma_d_CO_per_H, Zd_arr, sigma_d_ISM_ref = _resolve_co_dust_scalings(
        cfg=cfg,
        sigma_d_cm2=sigma_d_cm2,
        ncells=ncells,
    )
    D_pah_arr, rho_pah_flat, pah_meta = _resolve_pah_scaling(
        rad=rad,
        nH_flat=nH_flat,
        Zd_arr=Zd_arr,
        ncells=ncells,
    )
    Dh2gr_arr, sigma_H2gr_per_H, sigma_H2gr_ref, h2gr_meta = (
        _resolve_h2_grain_scaling(
            rad=rad,
            Zd_arr=Zd_arr,
            ncells=ncells,
        )
    )
    if h2gr_meta["h2gr_uses_ordinary_dust"] and abs(float(fH2gr) - 1.0) > 1.0e-12:
        raise ValueError(
            "fH2gr must remain 1.0 when H2 grain formation is derived from "
            "ordinary dust surface area"
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
    if use_healpix_geometry:
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
                b_kms=float(b_CO_kms),
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

    b_CO_kms, visser, shielding_linewidth_meta = _resolve_visser_table_linewidth(
        b_CO_kms,
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
    theta_CO_pdes_arr = np.ones(ncells, dtype=np.float64)

    xCtot_flat = Zg_arr * float(XC_STD)
    xOtot_flat = Zg_arr * float(XO_STD)

    status_acc = np.zeros(ncells, dtype=np.int32)
    status_final = np.zeros(ncells, dtype=np.int32)
    recovery_source = np.zeros(ncells, dtype=np.int32)
    recovery_diag = {
        "initial_failed": 0,
        "subset_retry_success": 0,
        "molecular_retry_candidates": 0,
        "molecular_retry_success": 0,
        "neighbor_retry_candidates": 0,
        "neighbor_retry_success": 0,
        "final_failed": 0,
    }

    d_h2_hist = []
    d_co_hist = []
    d_coice_hist = []
    d_tgas_hist = []
    d_h2_median_hist = []
    d_co_median_hist = []
    d_coice_median_hist = []
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

    def _project(y: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        if indices is None:
            xC_proj = xCtot_flat
            xO_proj = xOtot_flat
        else:
            idx = np.asarray(indices, dtype=np.int64)
            xC_proj = xCtot_flat[idx]
            xO_proj = xOtot_flat[idx]
        y_proj = project_gow17_state_to_budgets(
            y_arr,
            xCtot=xC_proj,
            xOtot=xO_proj,
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
        F_CO_pdes_photon_bands_flat=F_CO_pdes_photon_bands_flat,
        F_CO_pdes_photon_flat=F_CO_pdes_photon_flat,
        xCtot_flat=xCtot_flat,
        Zd_arr=Zd_arr,
        shape=shape,
        ncells=ncells,
        rad=rad,
        nH_cm3=nH_cm3,
        chi_dust_arr=chi_dust_arr,
        visser=visser,
        b_H2_kms=b_H2_kms,
        b_CO_kms=b_CO_kms,
        b_H2_kms_grid=b_H2_kms_grid,
        b_CO_kms_grid=b_CO_kms_grid,
        nside=nside,
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

    def _co_phase_subset(indices: np.ndarray) -> dict:
        out = dict(_co_phase_kw)
        out["co_F_CRUV_CO_pdes"] = np.ascontiguousarray(
            np.asarray(_co_phase_kw["co_F_CRUV_CO_pdes"], dtype=np.float64)[indices],
            dtype=np.float64,
        )
        out["co_k_crdes_CO"] = np.ascontiguousarray(
            np.asarray(_co_phase_kw["co_k_crdes_CO"], dtype=np.float64)[indices],
            dtype=np.float64,
        )
        return out

    def _solve_equilibrium_subset(indices: np.ndarray, y0_subset: np.ndarray) -> dict:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("GOW17 retry indices must be one-dimensional")
        y0_arr = np.ascontiguousarray(np.asarray(y0_subset, dtype=np.float64))
        if y0_arr.shape != (idx.size, N_Y):
            raise ValueError(
                f"GOW17 retry y0 must have shape {(idx.size, N_Y)}, got {y0_arr.shape}"
            )
        sigma_solver = sigma_d_CO_per_H if enable_co_phase else _zero_cell
        return _gow17.solve_batch_equilibrium(
            y0=y0_arr,
            nH=np.ascontiguousarray(nH_flat[idx], dtype=np.float64),
            Tgas=np.ascontiguousarray(T_flat[idx], dtype=np.float64),
            Tdust=np.ascontiguousarray(Tdust_flat[idx], dtype=np.float64),
            Zd=np.ascontiguousarray(Zd_arr[idx], dtype=np.float64),
            Dpah=np.ascontiguousarray(D_pah_arr[idx], dtype=np.float64),
            Dh2gr=np.ascontiguousarray(Dh2gr_arr[idx], dtype=np.float64),
            Zgd=np.ascontiguousarray(Zgd_arr[idx], dtype=np.float64),
            Zg=np.ascontiguousarray(Zg_arr[idx], dtype=np.float64),
            ion_rate=np.ascontiguousarray(ion_rate_arr[idx], dtype=np.float64),
            GPE=np.ascontiguousarray(GPE[idx], dtype=np.float64),
            F_CO_pdes_photon=np.ascontiguousarray(GISRF[idx], dtype=np.float64),
            Gph=np.ascontiguousarray(Gph[idx, :], dtype=np.float64),
            sigma_d_CO_per_H=np.ascontiguousarray(sigma_solver[idx], dtype=np.float64),
            reltol=reltol,
            abstol=abstol,
            mxsteps=mxsteps,
            maxord=maxord,
            tolfac=tolfac,
            tmin=tmin,
            tmax=tmax,
            const_temp=const_temp,
            gradv=np.ascontiguousarray(gradv_arr[idx], dtype=np.float64),
            Leff_CO_max=np.ascontiguousarray(Leff_CO_max_arr[idx], dtype=np.float64),
            isDust_cooling=isDust_cooling,
            isCoolingCOThin=isCoolingCOThin,
            fH2gr=fH2gr,
            fHplusgr=fHplusgr,
            fCplusgr=fCplusgr,
            fHeplusgr=fHeplusgr,
            fSplusgr=fSplusgr,
            fSiplusgr=fSiplusgr,
            fCplusCR=fCplusCR,
            **_co_phase_subset(idx),
            userJac=userJac,
            verbose=verbose,
        )

    def _cold_molecular_guess(indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        y = np.zeros((idx.size, N_Y), dtype=np.float64)
        xC_local = np.maximum(xCtot_flat[idx], 0.0)
        y[:, I_H2] = 0.5 - 1.0e-12
        if enable_co_phase:
            y[:, I_CO] = np.minimum(1.0e-20, xC_local * 1.0e-8)
            y[:, I_CO_ICE] = np.maximum(xC_local - y[:, I_CO], 0.0)
        else:
            y[:, I_CO] = 0.999 * xC_local
            y[:, I_CO_ICE] = 0.0
        y[:, I_CP] = np.minimum(1.0e-20, xC_local * 1.0e-8)
        y[:, I_H3P] = 1.0e-16
        T_seed = Tdust_flat[idx] if not const_temp else T_flat[idx]
        xe = _electron_abundance(y)
        y[:, I_E] = _cv_cold(y[:, I_H2], xe) * T_seed
        return _project(y, indices=idx)

    def _apply_successful_retry(
        indices: np.ndarray,
        result: dict,
        *,
        source_id: int,
    ) -> int:
        idx = np.asarray(indices, dtype=np.int64)
        retry_status = np.asarray(result["status"], dtype=np.int32)
        retry_y = _project(result["y"], indices=idx)
        ok = retry_status == 0
        if np.any(ok):
            ok_idx = idx[ok]
            y_guess[ok_idx, :] = retry_y[ok, :]
            status_final[ok_idx] = 0
            recovery_source[ok_idx] = int(source_id)
        if np.any(~ok):
            status_final[idx[~ok]] = retry_status[~ok]
        return int(np.sum(ok))

    def _neighbor_seed_guess(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64)
        status_grid = status_final.reshape(shape)
        y_grid = y_guess.reshape(shape + (N_Y,))
        seeds = []
        seed_idx = []
        offsets = np.array(
            np.meshgrid(*[[-1, 0, 1] for _ in shape], indexing="ij"),
            dtype=np.int64,
        ).reshape(len(shape), -1).T
        offsets = offsets[np.any(offsets != 0, axis=1)]

        for flat in idx:
            coord = np.array(np.unravel_index(int(flat), shape), dtype=np.int64)
            vals = []
            for off in offsets:
                nb = coord + off
                if np.any(nb < 0) or np.any(nb >= np.asarray(shape, dtype=np.int64)):
                    continue
                nb_tuple = tuple(int(v) for v in nb)
                if int(status_grid[nb_tuple]) == 0:
                    vals.append(y_grid[nb_tuple])
            if vals:
                seed_idx.append(int(flat))
                seeds.append(np.median(np.asarray(vals, dtype=np.float64), axis=0))

        if not seeds:
            return np.empty(0, dtype=np.int64), np.empty((0, N_Y), dtype=np.float64)
        out_idx = np.asarray(seed_idx, dtype=np.int64)
        out_y = _project(np.asarray(seeds, dtype=np.float64), indices=out_idx)
        return out_idx, out_y

    def _recover_final_failed_cells() -> None:
        failed = np.flatnonzero(status_final != 0)
        recovery_diag["initial_failed"] = int(failed.size)
        if failed.size == 0:
            return

        retry = _solve_equilibrium_subset(
            failed,
            _project(y_guess[failed, :], indices=failed),
        )
        _append_equilibrium_solver_diagnostics(equilibrium_solver_diag_hist, retry)
        status_acc[failed] = _accumulate_solver_status(status_acc[failed], retry["status"])
        recovery_diag["subset_retry_success"] = _apply_successful_retry(
            failed,
            retry,
            source_id=1,
        )

        remaining = np.flatnonzero(status_final != 0)
        if remaining.size == 0:
            return

        T_current = T_flat if const_temp else _state_temperature(y_guess)
        molecular_temperature = (
            np.minimum(T_current[remaining], Tdust_flat[remaining]) <= 150.0
        )
        dense = nH_flat[remaining] >= 1.0e6
        shielded = (
            np.maximum.reduce(
                [
                    np.abs(Gph[remaining, IPH_CO]),
                    np.abs(Gph[remaining, IPH_H2]),
                    np.abs(Gph[remaining, IPH_C]),
                ]
            )
            <= 1.0e-6
        )
        molecular_idx = remaining[molecular_temperature & dense & shielded]
        recovery_diag["molecular_retry_candidates"] = int(molecular_idx.size)
        if molecular_idx.size == 0:
            recovery_diag["final_failed"] = int(np.sum(status_final != 0))
            return

        cold_retry = _solve_equilibrium_subset(
            molecular_idx,
            _cold_molecular_guess(molecular_idx),
        )
        _append_equilibrium_solver_diagnostics(equilibrium_solver_diag_hist, cold_retry)
        status_acc[molecular_idx] = _accumulate_solver_status(
            status_acc[molecular_idx],
            cold_retry["status"],
        )
        recovery_diag["molecular_retry_success"] = _apply_successful_retry(
            molecular_idx,
            cold_retry,
            source_id=2,
        )
        recovery_diag["final_failed"] = int(np.sum(status_final != 0))

        remaining = np.flatnonzero(status_final != 0)
        if remaining.size == 0:
            return

        neighbor_idx, neighbor_y0 = _neighbor_seed_guess(remaining)
        recovery_diag["neighbor_retry_candidates"] = int(neighbor_idx.size)
        if neighbor_idx.size == 0:
            recovery_diag["final_failed"] = int(np.sum(status_final != 0))
            return

        neighbor_retry = _solve_equilibrium_subset(neighbor_idx, neighbor_y0)
        _append_equilibrium_solver_diagnostics(
            equilibrium_solver_diag_hist,
            neighbor_retry,
        )
        status_acc[neighbor_idx] = _accumulate_solver_status(
            status_acc[neighbor_idx],
            neighbor_retry["status"],
        )
        recovery_diag["neighbor_retry_success"] = _apply_successful_retry(
            neighbor_idx,
            neighbor_retry,
            source_id=3,
        )
        recovery_diag["final_failed"] = int(np.sum(status_final != 0))

    if coupling_mode == "fixed_point":
        # ====================================================================
        # Fixed-point equilibrium coupling (baseline)
        # ====================================================================
        for it in range(shielding_max_iter):
            y_guess[:, :] = _project(y_guess)
            xCO_old = np.ascontiguousarray(y_guess[:, I_CO].copy(), dtype=np.float64)
            xCO_ice_old = np.ascontiguousarray(y_guess[:, I_CO_ICE].copy(), dtype=np.float64)
            xH2_old = np.ascontiguousarray(y_guess[:, I_H2].copy(), dtype=np.float64)
            Tgas_old = _state_temperature(y_guess) if not const_temp else None

            theta_h2_flat, theta_co_flat, theta_c_flat, theta_CO_pdes_flat, Gph, GPE, GISRF = (
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
                Dpah=D_pah_arr,
                Dh2gr=Dh2gr_arr,
                Zgd=Zgd_arr,
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
            status_final[:] = np.asarray(status_step, dtype=np.int32)
            y_guess[:, :] = y_new
            y_guess[:, :] = _project(y_guess)

            xCO_new = y_guess[:, I_CO]
            xCO_ice_new = y_guess[:, I_CO_ICE]
            xH2_new = y_guess[:, I_H2]

            denom_h2 = np.maximum(np.abs(xH2_new), shielding_abstol)
            denom_co = np.maximum(np.abs(xCO_new), shielding_abstol)
            denom_coice = np.maximum(np.abs(xCO_ice_new), shielding_abstol)
            rel_h2 = np.abs(xH2_new - xH2_old) / denom_h2
            rel_co = np.abs(xCO_new - xCO_old) / denom_co
            rel_coice = np.abs(xCO_ice_new - xCO_ice_old) / denom_coice
            rel_tgas = None
            if Tgas_old is not None:
                Tgas_new = _state_temperature(y_guess)
                denom_tgas = np.maximum(np.abs(Tgas_new), 1.0)
                rel_tgas = np.abs(Tgas_new - Tgas_old) / denom_tgas
            d_h2 = np.max(rel_h2)
            d_co = np.max(rel_co)
            d_coice = np.max(rel_coice) if enable_co_phase else 0.0
            d_h2_hist.append(float(d_h2))
            d_co_hist.append(float(d_co))
            d_coice_hist.append(float(d_coice))
            d_h2_median_hist.append(float(np.median(rel_h2)))
            d_co_median_hist.append(float(np.median(rel_co)))
            d_coice_median_hist.append(float(np.median(rel_coice)) if enable_co_phase else 0.0)
            if rel_tgas is not None:
                d_tgas_hist.append(float(np.max(rel_tgas)))
                d_tgas_median_hist.append(float(np.median(rel_tgas)))
            rel_shield = np.maximum(rel_h2, rel_co)
            if enable_co_phase:
                rel_shield = np.maximum(rel_shield, rel_coice)
            well_converged = rel_shield <= shielding_reltol
            if rel_tgas is not None:
                well_converged &= rel_tgas <= tgas_convergence_reltol
            well_converged_cells_hist.append(int(np.sum(well_converged)))
            well_converged_fraction_hist.append(float(np.mean(well_converged)))
            converged = max(d_h2, d_co, d_coice) <= shielding_reltol
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
            theta_h2_flat, theta_co_flat, theta_c_flat, theta_CO_pdes_flat, Gph, GPE, GISRF = (
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
                Dpah=D_pah_arr,
                Dh2gr=Dh2gr_arr,
                Zgd=Zgd_arr,
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
            status_step = np.asarray(result["status"], dtype=np.int32)
            status_acc = _accumulate_solver_status(status_acc, status_step)
            status_final[:] = status_step

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
            theta_CO_pdes_flat = np.ones(ncells, dtype=np.float64)
            accel_species = [I_H2, I_CO] + ([I_CO_ICE] if enable_co_phase else [])
            accel_prev2: dict[int, np.ndarray] | None = None
            accel_prev1: dict[int, np.ndarray] | None = (
                {idx: y_state[:, idx].copy() for idx in accel_species}
                if astrochem_acceleration == "aitken"
                else None
            )
            checkpoint_start = 0
            loaded_checkpoint = _load_gow17_checkpoint(
                rad,
                checkpoint_cfg,
                shape=shape,
                ncells=ncells,
                total_updates=N,
                const_temp=const_temp,
                enable_co_phase=enable_co_phase,
            )
            if loaded_checkpoint is not None:
                y_state[:, :] = _project(loaded_checkpoint.y)
                y_guess[:, :] = y_state
                theta_h2_flat = loaded_checkpoint.theta_h2
                theta_co_flat = loaded_checkpoint.theta_co
                theta_c_flat = loaded_checkpoint.theta_c
                status_acc = loaded_checkpoint.status
                checkpoint_start = min(max(int(loaded_checkpoint.completed_updates), 0), N)
                _restore_gow17_checkpoint_histories(
                    loaded_checkpoint.histories,
                    {
                        "d_h2_hist": d_h2_hist,
                        "d_co_hist": d_co_hist,
                        "d_coice_hist": d_coice_hist,
                        "d_tgas_hist": d_tgas_hist,
                        "d_h2_median_hist": d_h2_median_hist,
                        "d_co_median_hist": d_co_median_hist,
                        "d_coice_median_hist": d_coice_median_hist,
                        "d_tgas_median_hist": d_tgas_median_hist,
                        "well_converged_cells_hist": well_converged_cells_hist,
                        "well_converged_fraction_hist": well_converged_fraction_hist,
                        "bad_status_hist": bad_status_hist,
                        "astrochem_acceleration_cells_hist": astrochem_acceleration_cells_hist,
                        "astrochem_acceleration_entries_hist": astrochem_acceleration_entries_hist,
                        "projection_corrections_hist": projection_corrections_hist,
                        "equilibrium_solver_diag_hist": equilibrium_solver_diag_hist,
                        "time_solver_diag_hist": time_solver_diag_hist,
                    },
                )
                if astrochem_acceleration == "aitken":
                    accel_prev2 = None
                    accel_prev1 = {idx: y_state[:, idx].copy() for idx in accel_species}

            for k_step in range(checkpoint_start, N):
                y_state[:, :] = _project(y_state)
                xCO_old = np.ascontiguousarray(y_state[:, I_CO].copy(), dtype=np.float64)
                xCO_ice_old = np.ascontiguousarray(y_state[:, I_CO_ICE].copy(), dtype=np.float64)
                xH2_old = np.ascontiguousarray(y_state[:, I_H2].copy(), dtype=np.float64)
                Tgas_old = _state_temperature(y_state) if not const_temp else None

                # Step 1+2: compute columns and shielding from current y_state.
                theta_h2_flat, theta_co_flat, theta_c_flat, theta_CO_pdes_flat, Gph, GPE, GISRF = (
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
                    Dpah=D_pah_arr,
                    Dh2gr=Dh2gr_arr,
                    Zgd=Zgd_arr,
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
                xCO_ice_new = y_state[:, I_CO_ICE]
                xH2_new = y_state[:, I_H2]

                denom_h2 = np.maximum(np.abs(xH2_new), shielding_abstol)
                denom_co = np.maximum(np.abs(xCO_new), shielding_abstol)
                denom_coice = np.maximum(np.abs(xCO_ice_new), shielding_abstol)
                rel_h2 = np.abs(xH2_new - xH2_old) / denom_h2
                rel_co = np.abs(xCO_new - xCO_old) / denom_co
                rel_coice = np.abs(xCO_ice_new - xCO_ice_old) / denom_coice
                rel_tgas = None
                if Tgas_old is not None:
                    Tgas_new = _state_temperature(y_state)
                    denom_tgas = np.maximum(np.abs(Tgas_new), 1.0)
                    rel_tgas = np.abs(Tgas_new - Tgas_old) / denom_tgas
                d_h2 = float(np.max(rel_h2))
                d_co = float(np.max(rel_co))
                d_coice = float(np.max(rel_coice)) if enable_co_phase else 0.0
                d_h2_hist.append(d_h2)
                d_co_hist.append(d_co)
                d_coice_hist.append(d_coice)
                d_h2_median_hist.append(float(np.median(rel_h2)))
                d_co_median_hist.append(float(np.median(rel_co)))
                d_coice_median_hist.append(float(np.median(rel_coice)) if enable_co_phase else 0.0)
                if rel_tgas is not None:
                    d_tgas_hist.append(float(np.max(rel_tgas)))
                    d_tgas_median_hist.append(float(np.median(rel_tgas)))
                rel_shield = np.maximum(rel_h2, rel_co)
                if enable_co_phase:
                    rel_shield = np.maximum(rel_shield, rel_coice)
                well_converged = rel_shield <= shielding_reltol
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

                if (
                    checkpoint_cfg["enabled"]
                    and (((k_step + 1) % int(checkpoint_cfg["every"]) == 0) or (k_step + 1 == N))
                ):
                    T_checkpoint = (
                        _state_temperature(y_state).reshape(shape)
                        if not const_temp
                        else T_flat.reshape(shape)
                    )
                    checkpoint_result = _gow17_checkpoint_result(
                        y_flat=y_state,
                        nH_cm3=nH_cm3,
                        xCtot=xCtot_flat.reshape(shape),
                        shape=shape,
                        T_out=T_checkpoint,
                        theta_h2_flat=theta_h2_flat,
                        theta_co_flat=theta_co_flat,
                        theta_c_flat=theta_c_flat,
                        theta_CO_pdes_flat=theta_CO_pdes_flat,
                        Gph=Gph,
                        GISRF=GISRF,
                        status_acc=status_acc,
                        line_h2_opr=cfg["line_h2_opr"],
                    )
                    _write_gow17_checkpoint(
                        rad,
                        checkpoint_cfg,
                        checkpoint_result,
                        y_grid=y_state.reshape(shape + (N_Y,)),
                        T_out=T_checkpoint,
                        completed_updates=k_step + 1,
                        total_updates=N,
                        t_target_yr=float(t_targets[k_step] / _YR_TO_S),
                        histories=_gow17_checkpoint_histories(
                            d_h2_hist=d_h2_hist,
                            d_co_hist=d_co_hist,
                            d_coice_hist=d_coice_hist,
                            d_tgas_hist=d_tgas_hist,
                            d_h2_median_hist=d_h2_median_hist,
                            d_co_median_hist=d_co_median_hist,
                            d_coice_median_hist=d_coice_median_hist,
                            d_tgas_median_hist=d_tgas_median_hist,
                            well_converged_cells_hist=well_converged_cells_hist,
                            well_converged_fraction_hist=well_converged_fraction_hist,
                            bad_status_hist=bad_status_hist,
                            astrochem_acceleration_cells_hist=astrochem_acceleration_cells_hist,
                            astrochem_acceleration_entries_hist=astrochem_acceleration_entries_hist,
                            projection_corrections_hist=projection_corrections_hist,
                            equilibrium_solver_diag_hist=equilibrium_solver_diag_hist,
                            time_solver_diag_hist=time_solver_diag_hist,
                        ),
                    )

            # After N macro-updates: recompute columns + shielding from final y_state.
            y_state[:, :] = _project(y_state)
            theta_h2_flat, theta_co_flat, theta_c_flat, theta_CO_pdes_flat, Gph, GPE, GISRF = (
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
                Dpah=D_pah_arr,
                Dh2gr=Dh2gr_arr,
                Zgd=Zgd_arr,
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
            status_step = np.asarray(result["status"], dtype=np.int32)
            status_acc = _accumulate_solver_status(status_acc, status_step)
            status_final[:] = status_step

    else:
        raise ValueError(
            f"gow17: unknown coupling_mode={coupling_mode!r}; "
            "expected 'fixed_point' or 'astrochem'"
        )

    n_shielding_iter = len(d_h2_hist)

    y_guess[:, :] = _project(y_guess)
    _recover_final_failed_cells()
    y_guess[:, :] = _project(y_guess)
    y_out = y_guess.reshape(shape + (N_Y,))
    status = status_final.reshape(shape)

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
    accumulated_status_hist = {}
    if status_acc.size:
        for code, count in zip(*np.unique(status_acc, return_counts=True)):
            accumulated_status_hist[int(code)] = int(count)

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
        line_h2_opr=cfg["line_h2_opr"],
    )

    gow17_diag = {
        "coupling_mode": coupling_mode,
        "enable_co_phase": bool(enable_co_phase),
        "dust_cooling_mode": dust_cooling_mode,
        "dust_cooling_mode_id_map": {1: "surface_area", 2: "gow17_original"},
        "sigma_d_H_ref": float(sigma_d_H_ref),
        "shielding_iter": n_shielding_iter,
        "d_h2_hist": np.asarray(d_h2_hist, dtype=np.float64),
        "d_co_hist": np.asarray(d_co_hist, dtype=np.float64),
        "d_coice_hist": np.asarray(d_coice_hist, dtype=np.float64),
        "d_tgas_hist": np.asarray(d_tgas_hist, dtype=np.float64),
        "d_h2_median_hist": np.asarray(d_h2_median_hist, dtype=np.float64),
        "d_co_median_hist": np.asarray(d_co_median_hist, dtype=np.float64),
        "d_coice_median_hist": np.asarray(d_coice_median_hist, dtype=np.float64),
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
        "well_converged_includes_coice": bool(enable_co_phase),
        "well_converged_ncells": int(ncells),
        "final_recovery": {
            "initial_failed": int(recovery_diag["initial_failed"]),
            "subset_retry_success": int(recovery_diag["subset_retry_success"]),
            "molecular_retry_candidates": int(
                recovery_diag["molecular_retry_candidates"]
            ),
            "molecular_retry_success": int(recovery_diag["molecular_retry_success"]),
            "neighbor_retry_candidates": int(
                recovery_diag["neighbor_retry_candidates"]
            ),
            "neighbor_retry_success": int(recovery_diag["neighbor_retry_success"]),
            "final_failed": int(recovery_diag["final_failed"]),
            "source_counts": {
                int(code): int(count)
                for code, count in zip(*np.unique(recovery_source, return_counts=True))
            },
            "source_id_map": {
                0: "normal_final_solve",
                1: "failed_cell_subset_retry",
                2: "shielded_molecular_retry",
                3: "neighbor_seeded_retry",
            },
        },
        "accumulated_status_hist": accumulated_status_hist,
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
    gow17_diag["pah_source_id_map"] = {
        0: "gow17_original_Zd",
        1: "explicit_pah_component",
    }
    gow17_diag.update(pah_meta)
    gow17_diag["h2gr_source_id_map"] = {
        0: "gow17_original_Zd",
        1: "ordinary_dust_surface_area",
    }
    gow17_diag.update(h2gr_meta)

    if not const_temp:
        T_out = _state_temperature(y_out.reshape(ncells, N_Y)).reshape(shape)
    else:
        T_out = T_flat.reshape(shape)
    gas_dust_exchange = _gas_dust_exchange(
        Zgd=Zgd_arr.reshape(shape),
        nH_cm3=nH_cm3,
        Tgas_K=T_out,
        Tdust_K=Tdust_flat.reshape(shape),
    )
    dust_cooling_mode_id = 1 if dust_cooling_mode == "surface_area" else 2
    pah_source_id = 1 if pah_meta["pah_uses_explicit_component"] else 0
    h2gr_source_id = (
        1 if h2gr_meta["h2gr_source"] == "ordinary_dust_surface_area" else 0
    )
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
        "status": Quantity(status.astype(np.float64), "dimensionless"),
        "Tgas_minus_Tdust": Quantity(T_out - Tdust_flat.reshape(shape), "K"),
        "Tgas_status": Quantity(T_status, "dimensionless"),
        "sigma_d_per_H_total": Quantity(sigma_d_per_H_total.reshape(shape), "cm^2"),
        "Zgd_surface": Quantity(Zgd_arr.reshape(shape), "dimensionless"),
        "pah_abundance_rel_ism": Quantity(
            D_pah_arr.reshape(shape),
            "dimensionless",
        ),
        "pah_source_id": Quantity(
            np.full(shape, pah_source_id, dtype=np.float64),
            "dimensionless",
        ),
        "h2gr_surface_rel_ism": Quantity(
            Dh2gr_arr.reshape(shape),
            "dimensionless",
        ),
        "h2gr_source_id": Quantity(
            np.full(shape, h2gr_source_id, dtype=np.float64),
            "dimensionless",
        ),
        "Tdust_gd_surface_weighted": Quantity(Tdust_flat.reshape(shape), "K"),
        "gas_dust_exchange_signed": Quantity(gas_dust_exchange, "erg/s"),
        "gas_to_dust_cooling": Quantity(np.maximum(gas_dust_exchange, 0.0), "erg/s"),
        "dust_to_gas_heating": Quantity(np.maximum(-gas_dust_exchange, 0.0), "erg/s"),
        "dust_cooling_mode": Quantity(
            np.full(shape, dust_cooling_mode_id, dtype=np.float64),
            "dimensionless",
        ),
        "b_H2_kms": Quantity(b_H2_kms_arr, "km/s"),
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
        "theta_CO_pdes": Quantity(theta_CO_pdes_flat.reshape(shape), "dimensionless"),
        "chi_eff": Quantity(actual_uv["G_CO_diss_actual"], "dimensionless"),
        "G_CO_diss_actual": Quantity(actual_uv["G_CO_diss_actual"], "dimensionless"),
        "G_C_ion_actual": Quantity(actual_uv["G_C_ion_actual"], "dimensionless"),
        "G_H2_diss_actual": Quantity(actual_uv["G_H2_diss_actual"], "dimensionless"),
        "F_CO_pdes_external_unshielded": Quantity(
            F_CO_pdes_photon_arr,
            "1/(cm^2 s)",
        ),
        "F_CO_pdes_external_actual": Quantity(
            actual_uv["F_CO_pdes_external_actual"],
            "1/(cm^2 s)",
        ),
    }
    if rho_pah_flat is not None:
        fields["pah_density"] = Quantity(
            rho_pah_flat.reshape(shape),
            "g/cm^3",
        )
    if sigma_H2gr_per_H is not None:
        fields["sigma_H2gr_per_H"] = Quantity(
            sigma_H2gr_per_H.reshape(shape),
            "cm^2",
        )
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
        D_pah_arr=D_pah_arr,
        Dh2gr_arr=Dh2gr_arr,
        Zgd_arr=Zgd_arr,
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
    rhs_warn = float(cfg["rhs_residual_warn"])
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
        "b_H2_kms_scalar": float(b_H2_kms),
        "b_CO_kms_scalar": float(b_CO_kms),
        "ion_rate_s": float(ion_rate_s),
        "Zg": float(Zg),
        "Zd": float(Zd),
        "sigma_d_ISM_ref": float(sigma_d_ISM_ref),
        "dust_cooling_mode": dust_cooling_mode,
        "sigma_d_H_ref": float(sigma_d_H_ref),
        "S_CO": float(S_CO),
        "co_phase_E_bind_CO": float(co_phase_params["E_bind_CO"]),
        "co_phase_nu0_CO": float(co_phase_params["nu0_CO"]),
        "co_phase_Y_CO": float(co_phase_params["Y_CO"]),
        "co_phase_N_SURF": float(co_phase_params["N_SURF"]),
        "co_phase_N_LAY": int(co_phase_params["N_LAY"]),
        "co_phase_cruv_enabled": bool(_nested_cfg(cfg, "co_phase")["enable_cruv_pdes"]),
        "co_phase_crdes_enabled": bool(_nested_cfg(cfg, "co_phase")["enable_crdes_CO"]),
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
