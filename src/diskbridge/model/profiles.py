# db-keywords: disk-mask, uv-products, model, mesh, field, coordinates, serialization
# db-role: canonical
# db-scope: package
# db-purpose: Package module for disk-mask, model, mesh, field.

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from diskbridge.model.core import Model


def compute_cell_volumes(model: "Model") -> np.ndarray:
    mesh = model.mesh
    if mesh.coord_system != 'spherical':
        raise ValueError(f"Only spherical meshes supported, got {mesh.coord_system}")

    r_edges = mesh.edges('r').to('cm')
    theta_edges = mesh.edges('theta').to('radian')
    phi_edges = mesh.edges('phi').to('radian')

    dr3 = (r_edges[1:] ** 3 - r_edges[:-1] ** 3) / 3.0
    dcos_theta = np.cos(theta_edges[:-1].magnitude) - np.cos(theta_edges[1:].magnitude)
    dphi = np.diff(phi_edges.magnitude)

    volumes = (dr3[:, None, None] * dcos_theta[None, :, None] * dphi[None, None, :]).magnitude
    return volumes


def compute_volume_weighted_mean_radial_profile(
    model: "Model",
    field_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mesh = model.mesh
    r = mesh.centers('r').to('au').magnitude

    if field_name not in model.gas:
        raise KeyError(f"Field '{field_name}' not found in model.gas")

    field = model.gas[field_name]
    data = field.data
    if hasattr(data, 'magnitude'):
        data = data.magnitude

    if getattr(mesh, 'coord_system', None) != 'spherical':
        raise ValueError(
            "compute_volume_weighted_mean_radial_profile supports spherical meshes only"
        )

    axis_order = getattr(field, 'axis_order', None)
    if axis_order is None:
        raise ValueError(f"Field '{field_name}' is missing axis_order")
    if axis_order != mesh.axis_names():
        raise ValueError(
            f"Field '{field_name}' has axis_order={axis_order}; expected canonical {mesh.axis_names()}"
        )

    nr, _, _ = data.shape
    volumes_use = compute_cell_volumes(model)
    if volumes_use.shape != data.shape:
        raise ValueError(
            f"Field '{field_name}' data shape {data.shape} does not match volume shape {volumes_use.shape}"
        )

    profile = np.zeros(nr)
    for i_r in range(nr):
        vals = data[i_r, :, :].ravel()
        wts = volumes_use[i_r, :, :].ravel()
        wt_sum = float(np.sum(wts))
        if wt_sum <= 0.0:
            raise ValueError(f"Non-positive volume sum at radial index {i_r}")
        profile[i_r] = float(np.sum(vals * wts) / wt_sum)

    return r, profile


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return a weighted quantile of non-NaN values.

    Parameters
    ----------
    values, weights : ndarray
        Matching one-dimensional samples and non-negative weights.
    quantile : float
        Requested quantile in ``[0, 1]``.

    Returns
    -------
    float
        Weighted quantile.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if values.shape != weights.shape:
        raise ValueError("values and weights must have matching shapes")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    valid = ~np.isnan(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(quantile) * float(cumulative[-1])
    return float(values[min(np.searchsorted(cumulative, target, side="left"), values.size - 1)])


def paired_radial_uncertainty_metrics(
    first: np.ndarray,
    second: np.ndarray,
    volumes: np.ndarray,
    *,
    first_weight: float = 0.5,
    tolerance: float = 0.01,
) -> dict[str, np.ndarray]:
    """Measure paired-estimator uncertainty in each spherical radial shell.

    Parameters
    ----------
    first, second : ndarray
        Independent scalar estimators with shape ``(nr, ntheta, nphi)``.
    volumes : ndarray
        Cell volumes with the same shape.
    first_weight : float, optional
        Packet-count weight assigned to ``first``.
    tolerance : float, optional
        Fractional uncertainty used for the reported failing volume fraction.

    Returns
    -------
    dict
        Weighted mean, shell-mean uncertainty, P99 and maximum cell
        uncertainty, and failing volume fraction versus radius.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    if first.shape != second.shape or first.shape != volumes.shape or first.ndim != 3:
        raise ValueError("paired fields and volumes must have matching 3-D shapes")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("paired fields must contain only finite values")
    weight1 = float(first_weight)
    if not 0.0 <= weight1 <= 1.0:
        raise ValueError("first_weight must be in [0, 1]")
    weight2 = 1.0 - weight1
    mean = weight1 * first + weight2 * second
    sigma = 0.5 * np.abs(first - second)
    denominator = np.abs(mean)
    cell_fractional = np.divide(
        sigma,
        denominator,
        out=np.where(sigma == 0.0, 0.0, np.inf),
        where=denominator > 0.0,
    )

    nr = first.shape[0]
    shell_mean = np.empty(nr, dtype=np.float64)
    shell_fractional = np.empty(nr, dtype=np.float64)
    p99 = np.empty(nr, dtype=np.float64)
    maximum = np.empty(nr, dtype=np.float64)
    maximum_flat_index = np.empty(nr, dtype=np.int64)
    failing_fraction = np.empty(nr, dtype=np.float64)
    for ir in range(nr):
        shell_weights = volumes[ir].ravel()
        total_weight = float(np.sum(shell_weights))
        if total_weight <= 0.0:
            raise ValueError(f"Non-positive volume sum at radial index {ir}")
        first_mean = float(np.sum(first[ir].ravel() * shell_weights) / total_weight)
        second_mean = float(np.sum(second[ir].ravel() * shell_weights) / total_weight)
        shell_mean[ir] = weight1 * first_mean + weight2 * second_mean
        shell_sigma = 0.5 * abs(first_mean - second_mean)
        shell_fractional[ir] = (
            shell_sigma / abs(shell_mean[ir])
            if shell_mean[ir] != 0.0
            else (0.0 if shell_sigma == 0.0 else np.inf)
        )
        shell_cell_fractional = cell_fractional[ir].ravel()
        p99[ir] = weighted_quantile(shell_cell_fractional, shell_weights, 0.99)
        maximum[ir] = float(np.nanmax(shell_cell_fractional))
        maximum_flat_index[ir] = int(np.nanargmax(shell_cell_fractional))
        failing_fraction[ir] = float(
            np.sum(shell_weights[shell_cell_fractional > float(tolerance)]) / total_weight
        )

    return {
        "mean": mean,
        "sigma": sigma,
        "shell_mean": shell_mean,
        "shell_fractional": shell_fractional,
        "cell_fractional_p99": p99,
        "cell_fractional_max": maximum,
        "cell_fractional_max_flat_index": maximum_flat_index,
        "failing_volume_fraction": failing_fraction,
    }


def find_noise_aware_split(
    r_edges_au: np.ndarray,
    reliable_shells: np.ndarray,
    stellar_fraction_profiles: np.ndarray,
    *,
    stellar_fraction_threshold: float,
    r_clip_min_au: float,
) -> tuple[float, dict]:
    """Select a one-shell-overlap boundary from UV quality diagnostics.

    Parameters
    ----------
    r_edges_au : ndarray
        Radial cell edges with length ``nr + 1``.
    reliable_shells : ndarray
        Boolean radial mask satisfying all paired-estimator requirements.
    stellar_fraction_profiles : ndarray
        Shell-mean stellar screening fractions with shape ``(nproduct, nr)``.
    stellar_fraction_threshold : float
        Maximum allowed stellar fraction in source and comparison shells.
    r_clip_min_au : float
        Minimum permitted child outer radius.

    Returns
    -------
    r_split_au, info : tuple
        Child outer edge and explicit source/comparison shell indices.

    Raises
    ------
    ValueError
        If no inner shell needs refinement or no valid two-shell boundary exists.
    """
    edges = np.asarray(r_edges_au, dtype=np.float64)
    reliable = np.asarray(reliable_shells, dtype=bool)
    stellar = np.asarray(stellar_fraction_profiles, dtype=np.float64)
    nr = reliable.size
    if edges.shape != (nr + 1,):
        raise ValueError("r_edges_au must have one more entry than reliable_shells")
    if stellar.ndim == 1:
        stellar = stellar[None, :]
    if stellar.ndim != 2 or stellar.shape[1] != nr:
        raise ValueError("stellar_fraction_profiles must have shape (nproduct, nr)")
    stellar_ok = np.all(np.isfinite(stellar) & (stellar <= float(stellar_fraction_threshold)), axis=0)
    needs_refinement = ~reliable | ~stellar_ok
    bad = np.flatnonzero(needs_refinement)
    if bad.size == 0:
        raise ValueError("No radial shell needs UV refinement")

    comparison_idx = int(bad[-1] + 1)
    while comparison_idx + 1 < nr:
        source_idx = comparison_idx + 1
        if reliable[comparison_idx] and reliable[source_idx] and stellar_ok[comparison_idx] and stellar_ok[source_idx]:
            r_split_au = float(edges[comparison_idx + 1])
            if r_split_au < float(r_clip_min_au):
                raise ValueError(
                    f"R_split={r_split_au:.6g} AU < r_clip_min={float(r_clip_min_au):.6g} AU"
                )
            return r_split_au, {
                "comparison_shell_idx": comparison_idx,
                "source_shell_idx": source_idx,
                "child_first_shell_idx": 0,
                "child_last_shell_idx": comparison_idx,
                "outermost_refinement_shell_idx": int(bad[-1]),
                "reliable_shells": reliable,
                "stellar_ok_shells": stellar_ok,
                "needs_refinement": needs_refinement,
            }
        comparison_idx += 1
    raise ValueError("No valid source/comparison shell pair exists outside the refinement region")


def find_r_split(
    r_au: np.ndarray,
    r_edges_au: np.ndarray,
    chi_profile: np.ndarray,
    T_profile: np.ndarray,
    tol_chi: float = 0.01,
    tol_T: float = 0.01,
    window_fraction: float = 0.1,
    r_clip_min_au: float = 1.0,
) -> Tuple[float, dict]:
    nr = int(len(r_au))
    if nr == 0:
        raise ValueError("r_au is empty")

    if chi_profile.shape != (nr,) or T_profile.shape != (nr,):
        raise ValueError("chi_profile and T_profile must match r_au shape")

    r_max = float(r_au[-1])
    r_min = float(r_au[0])
    window_r_min = r_max - float(window_fraction) * (r_max - r_min)
    window_mask = r_au >= window_r_min

    if not np.any(window_mask):
        raise ValueError(f"Asymptote window is empty (window_r_min={window_r_min:.2f} AU)")

    chi_asymptote = float(np.mean(chi_profile[window_mask]))
    T_asymptote = float(np.mean(T_profile[window_mask]))

    if chi_asymptote <= 0.0:
        raise ValueError(f"Invalid chi_asymptote={chi_asymptote}")
    if T_asymptote <= 0.0:
        raise ValueError(f"Invalid T_asymptote={T_asymptote}")

    chi_dev = np.abs(chi_profile - chi_asymptote) / chi_asymptote
    T_dev = np.abs(T_profile - T_asymptote) / T_asymptote

    within_tol = (chi_dev <= float(tol_chi)) & (T_dev <= float(tol_T))

    r_split_idx = None
    for i in range(nr - 2, -1, -1):
        if within_tol[i]:
            r_split_idx = i
        else:
            break

    if r_split_idx is None:
        for i in range(nr - 1, -1, -1):
            if not within_tol[i]:
                r_split_idx = i + 1
                break
        if r_split_idx is None or r_split_idx >= nr:
            raise ValueError("Could not find valid R_split: profiles never reach asymptote")

    r_split_au = float(r_edges_au[int(r_split_idx)])

    if r_split_au < float(r_clip_min_au):
        raise ValueError(
            f"R_split={r_split_au:.2f} AU < r_clip_min={float(r_clip_min_au):.2f} AU. "
            "The stellar radiation dominates too far out for segmented RT."
        )

    info = {
        "chi_asymptote": float(chi_asymptote),
        "T_asymptote": float(T_asymptote),
        "r_split_cell_idx": int(r_split_idx),
        "window_r_min": float(window_r_min),
        "chi_dev_at_split": float(chi_dev[int(r_split_idx)]) if int(r_split_idx) < nr else np.nan,
        "T_dev_at_split": float(T_dev[int(r_split_idx)]) if int(r_split_idx) < nr else np.nan,
    }

    return r_split_au, info
