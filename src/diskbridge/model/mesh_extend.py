from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

from diskbridge._units import Quantity, units

from .field import Field
from .mesh import Axis, Mesh

if TYPE_CHECKING:
    from .core import Model


@dataclass
class ExtendPlan:
    old_mesh: Mesh
    r_min: Quantity
    spacing: Optional[str]
    density_match: Optional[str]
    n_add: int
    r_edges_new: Quantity
    new_mesh: Mesh
    density_units: Optional[Quantity]


def _infer_spacing(r_edges_old: Quantity) -> str:
    r_edges_old_mag = np.asarray(r_edges_old.magnitude, dtype=float)
    if not np.all(np.isfinite(r_edges_old_mag)):
        raise ValueError("Radial edges contain non-finite values")
    
    diffs = np.diff(r_edges_old_mag[: min(6, r_edges_old_mag.size)])
    ratios = (
        r_edges_old_mag[1 : min(6, r_edges_old_mag.size)]
        / r_edges_old_mag[: min(5, r_edges_old_mag.size - 1)]
    )
    eps = np.finfo(float).eps
    dr_spread = float(np.max(np.abs(diffs - diffs[0])))
    ratio_spread = float(np.max(np.abs(ratios - ratios[0])))
    if dr_spread <= 32.0 * eps * max(1.0, abs(diffs[0])):
        return "lin"
    elif ratio_spread <= 32.0 * eps * max(1.0, abs(ratios[0])):
        return "log"
    else:
        raise ValueError(
            "Could not infer radial spacing pattern from existing edges; provide spacing='log' or spacing='lin'"
        )


def _build_new_edges(
    r_edges_old: Quantity, r_min: Quantity, spacing_mode: str
) -> Quantity:

    r_edges_old_mag = np.asarray(r_edges_old.magnitude, dtype=float)
    r_min_use = r_min.to(r_edges_old.units)

    if spacing_mode.startswith("log"):
        ratio = float(r_edges_old_mag[1] / r_edges_old_mag[0])
        if not np.isfinite(ratio) or ratio <= 1.0:
            raise ValueError(
                "Cannot extend logarithmic grid inward: failed to infer ratio from first two radial edges"
            )
        inner_edges_desc: list[float] = []
        r_next = float(r_edges_old_mag[0])
        while True:
            if r_next <= float(r_min_use.magnitude):
                break
            r_next = r_next / ratio
            if not np.isfinite(r_next) or r_next <= 0.0:
                raise ValueError("Failed to extend inward: encountered non-positive radial edge")
            inner_edges_desc.append(r_next)
        if not inner_edges_desc:
            return r_edges_old
        inner_edges_mag = np.array(list(reversed(inner_edges_desc)), dtype=float)
    elif spacing_mode.startswith("lin"):
        dr = float(r_edges_old_mag[1] - r_edges_old_mag[0])
        if not np.isfinite(dr) or dr <= 0.0:
            raise ValueError(
                "Cannot extend linear grid inward: failed to infer positive dr from first two radial edges"
            )
        inner_edges_desc: list[float] = []
        r_next = float(r_edges_old_mag[0])
        while True:
            if r_next <= float(r_min_use.magnitude):
                break
            r_next = r_next - dr
            if not np.isfinite(r_next) or r_next <= 0.0:
                raise ValueError("Failed to extend inward: encountered non-positive radial edge")
            inner_edges_desc.append(r_next)
        if not inner_edges_desc:
            return r_edges_old
        inner_edges_mag = np.array(list(reversed(inner_edges_desc)), dtype=float)
    else:
        raise ValueError(
            f"Unsupported spacing mode '{spacing_mode}'. Use 'log' or 'lin'."
        )

    return np.concatenate([inner_edges_mag, r_edges_old_mag]) * r_edges_old.units


def _compute_initial_density_for_inner_region(
    model: "Model", plan: ExtendPlan
) -> Optional[Quantity]:
    if model.disk is None:
        return None
    sigma0 = model.disk.parameters.get("sigma0")
    sigmaslope = model.disk.parameters.get("sigmaslope")
    h0 = model.disk.parameters.get("aspectratio")
    fl = model.disk.parameters.get("flaringindex")
    r0 = model.disk.parameters.get("r0")
    if sigma0 is None or sigmaslope is None or h0 is None or fl is None or r0 is None:
        return None

    h0_f = float(getattr(h0, "magnitude", h0))
    fl_f = float(getattr(fl, "magnitude", fl))
    sigmaslope_f = float(getattr(sigmaslope, "magnitude", sigmaslope))

    r_centers_new = plan.new_mesh.centers("r")
    theta_centers = plan.new_mesh.centers("theta")
    phi_centers = plan.new_mesh.centers("phi")
    if r_centers_new is None or theta_centers is None or phi_centers is None:
        raise ValueError("Failed to compute spherical centers for new mesh")

    r_inner = r_centers_new[:plan.n_add]
    r_mag = np.asarray(r_inner.magnitude, dtype=float)
    theta_mag = np.asarray(theta_centers.to("radian").magnitude, dtype=float)
    phi_mag = np.asarray(phi_centers.to("radian").magnitude, dtype=float)

    r_grid, phi_grid, theta_grid = np.meshgrid(r_mag, phi_mag, theta_mag, indexing="ij")
    r_grid = r_grid * r_inner.units

    sin_t = np.sin(theta_grid)
    cos_t = np.cos(theta_grid)
    R_cyl = r_grid * sin_t
    z = r_grid * cos_t

    H = (h0_f * (r_grid / r0) ** fl_f) * r_grid
    Sigma = sigma0 * (r_grid / r0) ** (-sigmaslope_f)
    rho_mid = Sigma / (np.sqrt(2.0 * np.pi) * H)

    expo = (-(z**2) / (2.0 * H**2)).to("dimensionless").magnitude
    rho = rho_mid * np.exp(expo)

    rhofloor = model.variables.get("RHOFLOORGAS", None)
    if rhofloor is not None:
        rho_units = plan.density_units if plan.density_units is not None else rho.units
        rho_floor_q = float(rhofloor) * rho_units
        rho = Quantity(
            np.maximum(
                np.asarray(rho.to(rho_units).magnitude, dtype=float),
                float(rho_floor_q.to(rho_units).magnitude),
            ),
            rho_units,
        )

    return rho


def _scale_density_to_match_snapshot(
    model: "Model", plan: ExtendPlan, rho: Quantity, units_f
) -> Quantity:
    mode = (plan.density_match or "").strip().lower()
    if not mode or mode == "none":
        return rho
    if mode not in ("join_midplane",):
        raise ValueError(
            "density_match must be one of: None, 'none', 'join_midplane'"
        )
    if model.disk is None:
        raise ValueError("density_match requested but model has no disk")

    sigma0 = model.disk.parameters.get("sigma0")
    sigmaslope = model.disk.parameters.get("sigmaslope")
    h0 = model.disk.parameters.get("aspectratio")
    fl = model.disk.parameters.get("flaringindex")
    r0 = model.disk.parameters.get("r0")
    if sigma0 is None or sigmaslope is None or h0 is None or fl is None or r0 is None:
        raise ValueError("density_match requested but disk parameters are incomplete")

    h0_f = float(getattr(h0, "magnitude", h0))
    fl_f = float(getattr(fl, "magnitude", fl))
    sigmaslope_f = float(getattr(sigmaslope, "magnitude", sigmaslope))

    if "density" not in model.gas:
        raise ValueError("density_match requested but gas field 'density' is missing")

    f_density = model.gas["density"]
    if not {"r", "phi", "theta"}.issubset(set(f_density.axis_order)):
        raise ValueError(
            "density_match currently requires density axis_order to include ('r','phi','theta')"
        )

    r_axis = f_density.axis_order.index("r")
    phi_axis = f_density.axis_order.index("phi")
    theta_axis = f_density.axis_order.index("theta")

    old_mag = np.asarray(f_density.data.to(units_f).magnitude, dtype=float)
    old_mag_rpt = np.moveaxis(old_mag, [r_axis, phi_axis, theta_axis], [0, 1, 2])
    if old_mag_rpt.shape[0] <= 0:
        raise ValueError("density_match failed: density has no radial cells")

    r_centers_new = plan.new_mesh.centers("r")
    theta_centers = plan.new_mesh.centers("theta")
    if r_centers_new is None or theta_centers is None:
        raise ValueError("Failed to compute spherical centers for new mesh")

    theta_vals = np.asarray(theta_centers.to("radian").magnitude, dtype=float)
    theta_mid_idx = int(np.argmin(np.abs(theta_vals - 0.5 * np.pi)))
    snap_ref = float(np.mean(old_mag_rpt[0, :, theta_mid_idx]))
    if not np.isfinite(snap_ref) or snap_ref <= 0.0:
        raise ValueError("density_match failed: snapshot reference density is non-positive")

    r_ref = r_centers_new[plan.n_add]
    theta_mid = float(theta_vals[theta_mid_idx])
    z_ref = r_ref * np.cos(theta_mid)
    H_ref = (h0_f * (r_ref / r0) ** fl_f) * r_ref
    Sigma_ref = sigma0 * (r_ref / r0) ** (-sigmaslope_f)
    rho_mid_ref = Sigma_ref / (np.sqrt(2.0 * np.pi) * H_ref)
    expo_ref = (-(z_ref**2) / (2.0 * H_ref**2)).to("dimensionless").magnitude
    rho_ref = (rho_mid_ref * np.exp(expo_ref)).to(units_f)
    analytic_ref = float(getattr(rho_ref, "magnitude", rho_ref))
    if not np.isfinite(analytic_ref) or analytic_ref <= 0.0:
        raise ValueError("density_match failed: analytic reference density is non-positive")

    scale = snap_ref / analytic_ref
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("density_match failed: computed scale is non-positive")
    rho_scaled = rho * scale

    rhofloor = model.variables.get("RHOFLOORGAS", None)
    if rhofloor is not None:
        rho_floor_q = float(rhofloor) * units_f
        rho_scaled = Quantity(
            np.maximum(
                np.asarray(rho_scaled.to(units_f).magnitude, dtype=float),
                float(rho_floor_q.to(units_f).magnitude),
            ),
            units_f,
        )

    return rho_scaled


def _compute_initial_vphi_for_inner_region(
    plan: ExtendPlan, v_unit
) -> Quantity:
    import diskbridge

    mstar = getattr(diskbridge.params, "mstar", None)
    if mstar is None:
        raise ValueError("diskbridge.params.mstar is required to compute initial vphi")

    r_centers_new = plan.new_mesh.centers("r")
    theta_centers = plan.new_mesh.centers("theta")
    phi_centers = plan.new_mesh.centers("phi")
    if r_centers_new is None or theta_centers is None or phi_centers is None:
        raise ValueError("Failed to compute spherical centers for new mesh")

    r_inner = r_centers_new[:plan.n_add]
    r_mag = np.asarray(r_inner.to_base_units().magnitude, dtype=float)
    theta_mag = np.asarray(theta_centers.to("radian").magnitude, dtype=float)
    phi_mag = np.asarray(phi_centers.to("radian").magnitude, dtype=float)

    r_grid, phi_grid, theta_grid = np.meshgrid(r_mag, phi_mag, theta_mag, indexing="ij")
    r_grid = r_grid * r_inner.to_base_units().units

    sin_t = np.sin(theta_grid)
    R_cyl = r_grid * sin_t

    if np.any(np.asarray(R_cyl.magnitude) <= 0.0):
        raise ValueError("Cannot compute Keplerian vphi where cylindrical radius is non-positive")

    G = units("G")
    vphi = np.sqrt((G * mstar.to("g")) / R_cyl)
    return vphi.to(v_unit)


def _regrid_fields(
    model: "Model", plan: ExtendPlan, rho_inner: Optional[Quantity]
) -> Dict[str, Field]:
    new_fields: Dict[str, Field] = {}
    for name, f in list(model.gas.items()):
        if "r" not in f.axis_order:
            new_fields[name] = f
            continue

        r_axis = f.axis_order.index("r")
        old_mag = np.asarray(f.data.magnitude)
        units_f = f.data.units

        new_shape = list(old_mag.shape)
        new_shape[r_axis] = new_shape[r_axis] + plan.n_add
        new_mag = np.empty(new_shape, dtype=old_mag.dtype)

        dst = [slice(None)] * old_mag.ndim
        dst[r_axis] = slice(plan.n_add, None)
        new_mag[tuple(dst)] = old_mag

        src0 = np.take(old_mag, indices=0, axis=r_axis)
        pad = np.repeat(np.expand_dims(src0, axis=r_axis), repeats=plan.n_add, axis=r_axis)

        if name == "density" and rho_inner is not None:
            pad = np.asarray(rho_inner.to(units_f).magnitude, dtype=new_mag.dtype)
        elif name == "vphi":
            vphi_inner = _compute_initial_vphi_for_inner_region(plan, units_f)
            pad = np.asarray(vphi_inner.to(units_f).magnitude, dtype=new_mag.dtype)
        elif name in ("vr", "vtheta"):
            pad = np.zeros_like(pad)

        dst_inner = [slice(None)] * old_mag.ndim
        dst_inner[r_axis] = slice(0, plan.n_add)
        new_mag[tuple(dst_inner)] = pad

        new_data = Quantity(new_mag, units_f)
        new_fields[name] = Field(
            data=new_data,
            quantity=f.quantity,
            axis_order=f.axis_order,
            attrs=f.attrs,
        )
    return new_fields


def _extend_mask(model: "Model", plan: ExtendPlan) -> None:
    if model.disk is None or model.disk.mask is None:
        return
    
    mask_field = model.disk.mask
    if "r" not in mask_field.axis_order:
        return
    
    r_axis = mask_field.axis_order.index("r")
    mask_mag = np.asarray(mask_field.data.magnitude, dtype=bool)
    new_shape = list(mask_mag.shape)
    new_shape[r_axis] = new_shape[r_axis] + plan.n_add
    new_mask = np.zeros(new_shape, dtype=bool)

    dst = [slice(None)] * mask_mag.ndim
    dst[r_axis] = slice(plan.n_add, None)
    new_mask[tuple(dst)] = mask_mag

    src0 = np.take(mask_mag, indices=0, axis=r_axis)
    pad = np.repeat(np.expand_dims(src0, axis=r_axis), repeats=plan.n_add, axis=r_axis)

    dst_inner = [slice(None)] * mask_mag.ndim
    dst_inner[r_axis] = slice(0, plan.n_add)
    new_mask[tuple(dst_inner)] = pad

    model.disk.mask = Field(
        data=Quantity(new_mask, "dimensionless"),
        quantity=mask_field.quantity,
        axis_order=mask_field.axis_order,
        attrs=mask_field.attrs,
    )


def extend_spherical_grid_inwards(
    model: "Model",
    r_min: Quantity,
    *,
    spacing: Optional[str] = None,
    density_match: Optional[str] = None,
) -> "Model":
    mesh = model.mesh
    if mesh is None:
        raise ValueError("Model has no mesh")
    if mesh.coord_system != "spherical":
        raise ValueError(
            "extend_spherical_grid_inwards only supports spherical coordinates, "
            f"got {mesh.coord_system}"
        )

    r_edges_old = mesh.edges("r")
    theta_edges = mesh.edges("theta")
    phi_edges = mesh.edges("phi")
    if r_edges_old is None or theta_edges is None or phi_edges is None:
        raise ValueError("Mesh is missing one or more spherical edge axes")
    if r_edges_old.size < 2:
        raise ValueError("Radial edges must have at least 2 entries")

    r_min_use = r_min.to(r_edges_old.units)
    r0_old = r_edges_old[0]
    if float(r_min_use.magnitude) >= float(r0_old.magnitude):
        raise ValueError(
            f"Requested r_min={r_min_use} is not smaller than existing inner edge {r0_old}"
        )

    spacing_mode = (spacing or "").strip().lower()
    if not spacing_mode:
        spacing_mode = _infer_spacing(r_edges_old)

    r_edges_new = _build_new_edges(r_edges_old, r_min_use, spacing_mode)
    if r_edges_new.size <= r_edges_old.size:
        return model

    new_mesh = Mesh.spherical(
        r=Axis(edges=r_edges_new),
        theta=Axis(edges=theta_edges),
        phi=Axis(edges=phi_edges),
    )

    old_nr = int(r_edges_old.size - 1)
    new_nr = int(r_edges_new.size - 1)
    n_add = new_nr - old_nr

    if getattr(model.gas, "_lazy", None):
        for name in list(model.gas._lazy.keys()):
            _ = model.gas[name]

    density_units = None
    try:
        if "density" in model.gas:
            density_units = model.gas["density"].data.units
    except Exception:
        density_units = None

    plan = ExtendPlan(
        old_mesh=mesh,
        r_min=r_min_use,
        spacing=spacing_mode,
        density_match=density_match,
        n_add=n_add,
        r_edges_new=r_edges_new,
        new_mesh=new_mesh,
        density_units=density_units,
    )

    rho_inner = _compute_initial_density_for_inner_region(model, plan)
    if rho_inner is not None:
        units_density = density_units if density_units is not None else rho_inner.units
        rho_inner = _scale_density_to_match_snapshot(model, plan, rho_inner, units_density)

    new_fields = _regrid_fields(model, plan, rho_inner)

    model.mesh = new_mesh
    model.coord_system = "spherical"
    model.gas.mesh = new_mesh
    model.gas.coord_system = "spherical"
    if model.disk is not None:
        model.disk.mesh = new_mesh
        model.disk.coord_system = "spherical"
    if getattr(model, "dust", None) is not None:
        model.dust.mesh = new_mesh
        model.dust.coord_system = "spherical"
        try:
            model.dust._dust_fields.clear()
        except Exception:
            pass

    model.gas.clear()
    for name, field in new_fields.items():
        model.gas_register(name, field)

    _extend_mask(model, plan)

    return model
