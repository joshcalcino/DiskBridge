from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import shutil

import diskbridge
import diskbridge._params as _params_module
import diskbridge._gow17 as gow17_native
from diskbridge._constants import K_B
from diskbridge._units import Quantity
from diskbridge.chemistry.api import run_chemistry
I_CHX = gow17_native.I_CHX
I_CO = gow17_native.I_CO
I_CO_ICE = gow17_native.I_CO_ICE
I_CP = gow17_native.I_CP
I_H2 = gow17_native.I_H2
I_H2P = gow17_native.I_H2P
I_H3P = gow17_native.I_H3P
I_HCOP = gow17_native.I_HCOP
I_HEP = gow17_native.I_HEP
I_HP = gow17_native.I_HP
I_OHX = gow17_native.I_OHX
I_OP = gow17_native.I_OP
I_SIP = gow17_native.I_SIP
I_SP = gow17_native.I_SP
I_E = gow17_native.I_E
N_Y = gow17_native.N_Y
XHE = gow17_native.XHE
from diskbridge.chemistry.shielding.columns_1d import is_effectively_1d
from diskbridge.model.core import Model, SubModel
from diskbridge.model.dust import Dust
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.model.utils import transpose_to_axis_order
from diskbridge.radmc3d.model import RadModel


SPEC_LIST_REF = [
    "He+",
    "OHx",
    "CHx",
    "CO",
    "C+",
    "HCO+",
    "H2",
    "H+",
    "H3+",
    "H2+",
    "S+",
    "Si+",
    "O+",
    "E",
]

_SPEC_INDEX_REF = {name: i for i, name in enumerate(SPEC_LIST_REF)}

COLORS = {
    "CO": "#1f77b4",
    "C": "#d62728",
    "C+": "#ff7f0e",
    "H3+": "#2ca02c",
    "OHx": "#9467bd",
    "CHx": "#8c564b",
    "He+": "#e377c2",
}


@dataclass(frozen=True)
class RefSlab:
    NH: np.ndarray
    Av: np.ndarray
    nH_values: np.ndarray
    abd: Dict[str, np.ndarray]
    E: np.ndarray


def _load_reference(dir_out: Path) -> RefSlab:
    nH_values = np.loadtxt(dir_out / "nH_arr.dat")
    NH = np.loadtxt(dir_out / "colH_arr.dat")
    Av = NH / 1.87e21

    abd: Dict[str, np.ndarray] = {}
    E_list = []
    for islab, _nH in enumerate(np.atleast_1d(nH_values)):
        slab = np.loadtxt(dir_out / f"slab{islab:06d}.dat")
        for spec, j in _SPEC_INDEX_REF.items():
            abd.setdefault(spec, []).append(slab[:, j])
        E_list.append(slab[:, _SPEC_INDEX_REF["E"]])

    for spec in list(abd):
        abd[spec] = np.stack(abd[spec], axis=1)

    E = np.stack(E_list, axis=1)

    xC = 1.6e-4
    abd["C"] = xC - abd["CHx"] - abd["CO"] - abd["C+"] - abd["HCO+"]

    return RefSlab(NH=NH, Av=Av, nH_values=np.asarray(nH_values, dtype=float), abd=abd, E=E)


def _xe_from_abundances(abd: Dict[str, np.ndarray]) -> np.ndarray:
    xe = np.zeros_like(abd["H2"], dtype=float)
    for spec in ["He+", "C+", "HCO+", "H3+", "H2+", "H+", "Si+", "S+", "O+"]:
        if spec in abd:
            xe = xe + np.asarray(abd[spec], dtype=float)
    return xe


def _temperature_from_energy(
    *,
    E: np.ndarray,
    xH2: np.ndarray,
    xe: np.ndarray,
    xHe: float = 0.1,
) -> np.ndarray:
    Cv_cold = 1.5 * float(K_B) * (1.0 - xH2 + float(xHe) + xe)
    return E / Cv_cold


def _build_spherical_cloud_model(
    *,
    nH_cm3: float,
    R_cm: float,
    nr: int,
    ntheta: int,
    nphi: int,
    model_dir: str | Path = ".",
) -> RadModel:
    units = diskbridge.units
    m_H = units("m_H")

    R_cm = float(R_cm)
    if not (R_cm > 0.0):
        raise ValueError(f"R_cm must be > 0, got {R_cm}")

    nr = int(nr)
    ntheta = int(ntheta)
    nphi = int(nphi)
    if nr <= 0:
        raise ValueError(f"nr must be >= 1, got {nr}")
    if ntheta <= 0:
        raise ValueError(f"ntheta must be >= 1, got {ntheta}")
    if nphi <= 0:
        raise ValueError(f"nphi must be >= 1, got {nphi}")

    dr_cm = float(R_cm) / float(nr)
    r_edges_cm = (np.arange(nr + 1, dtype=float) * dr_cm) + 0.5 * dr_cm
    r_edges_cm[-1] = float(R_cm)
    th_edges = np.linspace(0.0, np.pi, ntheta + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, nphi + 1)

    r_axis = Axis(edges=Quantity(r_edges_cm, "cm"))
    theta_axis = Axis(edges=Quantity(th_edges, "rad"))
    phi_axis = Axis(edges=Quantity(ph_edges, "rad"))

    mesh = Mesh.spherical(r=r_axis, theta=theta_axis, phi=phi_axis)
    shape = mesh.shape
    axis_order = mesh.axis_names()

    if is_effectively_1d(mesh, shape):
        raise RuntimeError(f"Expected non-1D mesh but got shape={shape}")

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    nH_field = np.full(shape, float(nH_cm3), dtype=float)
    rho = (Quantity(nH_field, "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=axis_order))

    sigma = Quantity(np.full(shape, 1.0e-21, dtype=float), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=axis_order),
    )

    return RadModel(model, model_dir=model_dir)


def _write_radmc3d_params(
    *,
    model_dir: Path,
    chi0: float,
    nphot_mono: Optional[int],
    nphot_thermal: Optional[int],
    scat_mode: Optional[int],
) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    params_path = model_dir / "params.txt"

    uv_min_um = float(diskbridge.params.uv_min.to("micron").magnitude)
    lambda_min_um = min(float(diskbridge.params.lambda_min.to("micron").magnitude), uv_min_um)

    lines = [
        "external_uv = T",
        f"external_uv_chi = {float(chi0)}",
        f"lambda_min = {lambda_min_um}",
    ]
    if nphot_mono is not None:
        lines.append(f"nphot_mono = {int(nphot_mono)}")
    if nphot_thermal is not None:
        lines.append(f"nphot_thermal = {int(nphot_thermal)}")
    if scat_mode is not None:
        lines.append(f"scat_mode = {int(scat_mode)}")

    params_path.write_text("\n".join(lines) + "\n")
    return params_path


def _ensure_dustkappa_files(
    *,
    inputs_dir: Path,
    species: str,
    nbin: int,
    source_dir: Path,
) -> None:
    inputs_dir = Path(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir)

    for i in range(int(nbin)):
        fname = f"dustkappa_{species}{int(i)}.inp"
        dst = inputs_dir / fname
        if dst.exists():
            continue
        src = source_dir / fname
        if not src.exists():
            raise FileNotFoundError(
                f"Missing required opacity file {src}. "
                "Point source_dir at a directory containing precomputed dustkappa files."
            )
        shutil.copy2(src, dst)


def _write_dust_temperature_bdat(
    *,
    model_dir: Path,
    mesh: Mesh,
    dust_temperature: Quantity,
    nspec: int,
) -> Path:
    """Write dust_temperature.bdat in RADMC-3D binary format.

    Format: int64 header [format=1, precis=8, ncells, nspec],
    then nspec blocks of ncells float64 values.
    """
    model_dir = Path(model_dir)
    out_dir = model_dir / "radmc3d_outputs" / "temperature"
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / "dust_temperature.bdat"

    temp = np.asarray(dust_temperature.to("K").magnitude, dtype=float)
    if temp.shape != mesh.shape:
        raise ValueError(f"dust_temperature shape {temp.shape} != mesh.shape {mesh.shape}")

    if mesh.coord_system == "spherical":
        temp_out = transpose_to_axis_order(
            temp,
            from_order=mesh.axis_names(),
            to_order=("phi", "theta", "r"),
        )
        temp_flat = np.ravel(temp_out)
    else:
        temp_flat = np.ravel(temp, order="F")

    nspec = int(nspec)
    if nspec <= 0:
        raise ValueError(f"nspec must be >= 1, got {nspec}")

    ncells = int(temp_flat.size)

    with open(fpath, "wb") as f:
        header = np.array([1, 8, ncells, nspec], dtype=np.int64)
        header.tofile(f)
        data = np.asarray(temp_flat, dtype=np.float64)
        for _ in range(nspec):
            data.tofile(f)

    return fpath


def _compute_chi_with_radmc3d(
    *,
    radm: RadModel,
    model_dir: Path,
    chi0: float,
    nphot_mono: Optional[int],
    nphot_thermal: Optional[int],
    scat_mode: Optional[int],
    dust_nbins: int,
    force: bool,
) -> Quantity:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "radmc3d_inputs").mkdir(parents=True, exist_ok=True)
    (model_dir / "radmc3d_outputs").mkdir(parents=True, exist_ok=True)

    params_path = _write_radmc3d_params(
        model_dir=model_dir,
        chi0=chi0,
        nphot_mono=nphot_mono,
        nphot_thermal=nphot_thermal,
        scat_mode=scat_mode,
    )

    prev_params = diskbridge.params
    new_params = diskbridge.read_params(params_path)
    diskbridge.params = new_params
    _params_module.params = new_params
    radm.params = new_params
    radm.writer.params = new_params

    if radm.model.dust is None:
        radm.model.dust = Dust(radm.model)

    species = new_params.species[0] if isinstance(new_params.species, list) else str(new_params.species)
    dust_to_gas_ratio = (
        float(new_params.dust_to_gas_ratio[0])
        if isinstance(new_params.dust_to_gas_ratio, list)
        else float(new_params.dust_to_gas_ratio)
    )

    radm.model.dust.set_distribution(
        nbin=int(dust_nbins),
        dust_to_gas_ratio=float(dust_to_gas_ratio),
        mode="proportional",
    )

    radm.writer.write_amr_grid(model_dir)
    radm.writer.write_wavelength_grid(model_dir)
    radm.writer.write_stars(model_dir)
    radm.writer.write_radmc3d_inp(
        model_dir,
        scattering_mode_max=int(new_params.scat_mode),
        nphot=int(new_params.nphot_thermal),
        nphot_mono=int(new_params.nphot_mono),
        nphot_scat=int(new_params.nphot_scat),
        setthreads=int(new_params.nbcores),
    )
    radm.writer.write_dust_density(model_dir, binary=True)
    radm.writer.write_dustopac(model_dir, scattering_mode=int(new_params.scat_mode))

    root = Path(__file__).resolve().parents[1]
    opac_src_dir = root / "examples" / "testbed" / "baseline_run" / "radmc3d_inputs"
    _ensure_dustkappa_files(
        inputs_dir=model_dir / "radmc3d_inputs",
        species=species,
        nbin=int(radm.model.dust.nbin),
        source_dir=opac_src_dir,
    )

    isrf_path = (root / "examples" / "testbed" / "ISRF.dat").resolve()
    radm.writer.write_external_source(model_dir, chi=float(chi0), isrf_path=isrf_path)

    dust_temp = getattr(radm, "dust_temperature", None)
    if dust_temp is None:
        raise RuntimeError("dust_temperature must be set on radm before running RADMC-3D mcmono")
    _write_dust_temperature_bdat(
        model_dir=model_dir,
        mesh=radm.model.mesh,
        dust_temperature=dust_temp,
        nspec=int(radm.model.dust.nbin),
    )
    # The default UV band (91.2-111.8 nm) is too narrow and captures only
    # ~16% of the FUV energy density, causing chi to be underestimated by ~6x
    # relative to U_DRAINE = 9e-14 erg/cm^3.  Use the full FUV range.
    chi = radm.ensure_chi(
        force=bool(force),
        uv_min=Quantity(91.2, "nm"),
        uv_max=Quantity(200.0, "nm"),
        n_wavelengths=30,
    )

    diskbridge.params = prev_params
    _params_module.params = prev_params

    return chi


def _av_bin_edges(av: np.ndarray) -> np.ndarray:
    av = np.asarray(av, dtype=float)
    if av.ndim != 1 or av.size < 2:
        raise ValueError("av must be a 1D array with at least 2 points")
    edges = np.empty(av.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (av[:-1] + av[1:])
    edges[0] = av[0]
    edges[-1] = av[-1] + 0.5 * (av[-1] - av[-2])
    return edges


def _mean_profile_vs_av(
    *,
    av_centers: np.ndarray,
    av_samples: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    av_centers = np.asarray(av_centers, dtype=float)
    edges = _av_bin_edges(av_centers)

    av_samples = np.asarray(av_samples, dtype=float)
    values = np.asarray(values, dtype=float)
    if av_samples.shape != values.shape:
        raise ValueError("av_samples and values must have the same shape")

    idx = np.digitize(av_samples, edges) - 1
    idx = np.clip(idx, 0, av_centers.size - 1)

    sums = np.bincount(idx, weights=values, minlength=av_centers.size)
    counts = np.bincount(idx, minlength=av_centers.size)

    out = np.full(av_centers.size, np.nan, dtype=float)
    m = counts > 0
    out[m] = sums[m] / counts[m]

    if not np.any(m):
        raise ValueError("No samples fell into any Av bin")

    if not np.all(m):
        x = av_centers[m]
        y = out[m]
        out = np.interp(av_centers, x, y, left=float(y[0]), right=float(y[-1]))
    return out


def run_gow17_internal_3d_healpix_sphere(
    *,
    nH_cm3: float,
    chi0: float = 1.0,
    xi_cr: float = 2.0e-16,
    shielding_max_iter: int = 20,
    shielding_mix: float = 0.5,
    const_temp: bool = True,
    nside: int = 4,
    self_weight: float = 1.0,
    nr: int = 128,
    ntheta: int = 8,
    nphi: int = 16,
    ref: RefSlab | None = None,
    nH_index: int = 0,
    use_radmc3d_chi: bool = False,
    radmc3d_model_dir: Optional[str | Path] = None,
    radmc3d_force: bool = False,
    radmc3d_nphot_mono: Optional[int] = None,
    radmc3d_nphot_thermal: Optional[int] = None,
    radmc3d_scat_mode: Optional[int] = None,
    radmc3d_dust_nbins: Optional[int] = None,
    diagnostic_outdir: Optional[str | Path] = None,
    diagnostic_suffix: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict]:
    if ref is None:
        raise ValueError("ref must be provided")

    NH_ref = np.asarray(ref.NH, dtype=float)
    NH_max = float(np.max(NH_ref))
    R_cm = NH_max / float(nH_cm3)

    if radmc3d_model_dir is None:
        radmc3d_model_dir = Path(__file__).resolve().parent / "reproduce_gow17_fig2_3dhealpix" / (
            f"radmc3d_sphere_nH_{int(float(nH_cm3))}_nr_{int(nr)}_nt_{int(ntheta)}_np_{int(nphi)}"
        )

    radm = _build_spherical_cloud_model(
        nH_cm3=nH_cm3,
        R_cm=R_cm,
        nr=int(nr),
        ntheta=int(ntheta),
        nphi=int(nphi),
        model_dir=Path(radmc3d_model_dir) if use_radmc3d_chi else ".",
    )

    shape = radm.model.mesh.shape
    r_cent = radm.model.mesh.centers("r").to("cm").magnitude
    r3 = np.broadcast_to(r_cent[:, None, None], shape)

    chi0_incident = 2.0 * float(chi0)

    depth_cm = np.maximum(float(R_cm) - r3, 0.0)
    inside = r3 <= float(R_cm)
    NH_depth = float(nH_cm3) * depth_cm
    Av_3d = NH_depth / 1.87e21

    nH_index = int(nH_index)
    if nH_index < 0 or nH_index >= int(ref.nH_values.size):
        raise ValueError(f"nH_index out of range: {nH_index}")

    xe_ref = _xe_from_abundances(ref.abd)
    T_ref = _temperature_from_energy(E=ref.E, xH2=ref.abd["H2"], xe=xe_ref)

    Av_flat = Av_3d.reshape(-1)
    T_line = np.asarray(T_ref[:, nH_index], dtype=float)
    T_init_flat = np.interp(Av_flat, ref.Av, T_line, left=T_line[0], right=T_line[-1])
    T_init = T_init_flat.reshape(shape)

    radm.gas_temperature = Quantity(T_init, "K")
    radm.dust_temperature = Quantity(T_init, "K")

    # -- Av: perpendicular depth with Gong+17 Appendix 3D-slab factor (x2) --
    Av_perp = NH_depth / 1.87e21
    radm.Av = Quantity(2.0 * Av_perp, "dimensionless")

    if use_radmc3d_chi:
        chi_rt = _compute_chi_with_radmc3d(
            radm=radm,
            model_dir=Path(radmc3d_model_dir),
            chi0=float(chi0),
            nphot_mono=radmc3d_nphot_mono,
            nphot_thermal=radmc3d_nphot_thermal,
            scat_mode=radmc3d_scat_mode,
            dust_nbins=int(radmc3d_dust_nbins) if radmc3d_dust_nbins is not None else int(diskbridge.params.nbins),
            force=bool(radmc3d_force),
        )
        radm.chi = chi_rt
    else:
        # Incident (unattenuated) field everywhere; dust attenuation
        # is handled inside the chemistry via chi_is_incident=True.
        radm.chi = Quantity(np.full(shape, chi0_incident, dtype=float), "dimensionless")

    y0 = np.zeros(N_Y, dtype=float)
    y0[I_HEP] = 1.450654e-08
    y0[I_H3P] = 2.681411e-07
    y0[I_CP] = 1.0e-4
    y0[I_CO] = 1.0e-7
    y0[I_H2] = 0.1
    y0[I_CO_ICE] = 0.0

    y_init = np.zeros(shape + (N_Y,), dtype=float)
    for j in range(N_Y):
        y_init[..., j] = y0[j]

    def _interp(spec: str) -> np.ndarray:
        prof = np.asarray(ref.abd[spec][:, nH_index], dtype=float)
        return np.interp(Av_flat, ref.Av, prof, left=prof[0], right=prof[-1]).reshape(shape)

    y_init[..., I_HEP] = np.where(inside, _interp("He+"), y_init[..., I_HEP])
    y_init[..., I_OHX] = np.where(inside, _interp("OHx"), y_init[..., I_OHX])
    y_init[..., I_CHX] = np.where(inside, _interp("CHx"), y_init[..., I_CHX])
    y_init[..., I_CO] = np.where(inside, _interp("CO"), y_init[..., I_CO])
    y_init[..., I_CP] = np.where(inside, _interp("C+"), y_init[..., I_CP])
    y_init[..., I_HCOP] = np.where(inside, _interp("HCO+"), y_init[..., I_HCOP])
    y_init[..., I_H2] = np.where(inside, _interp("H2"), y_init[..., I_H2])
    y_init[..., I_HP] = np.where(inside, _interp("H+"), y_init[..., I_HP])
    y_init[..., I_H3P] = np.where(inside, _interp("H3+"), y_init[..., I_H3P])
    y_init[..., I_H2P] = np.where(inside, _interp("H2+"), y_init[..., I_H2P])
    y_init[..., I_SP] = np.where(inside, _interp("S+"), y_init[..., I_SP])
    y_init[..., I_SIP] = np.where(inside, _interp("Si+"), y_init[..., I_SIP])
    y_init[..., I_OP] = np.where(inside, _interp("O+"), y_init[..., I_OP])
    y_init[..., I_CO_ICE] = 0.0

    # Energy species: E = Cv * T, matching _cv_cold in gow17.py
    xH2_init = y_init[..., I_H2]
    xe_init = (
        y_init[..., I_HEP]
        + y_init[..., I_CP]
        + y_init[..., I_HCOP]
        + y_init[..., I_HP]
        + y_init[..., I_H3P]
        + y_init[..., I_H2P]
        + y_init[..., I_SP]
        + y_init[..., I_SIP]
        + y_init[..., I_OP]
    )
    Cv_init = 1.5 * K_B * ((1.0 - 2.0 * xH2_init) + xH2_init + XHE + xe_init)
    y_init[..., I_E] = Cv_init * T_init

    radm.gow17_y = np.ascontiguousarray(y_init, dtype=np.float64)

    gow17_cfg = {
        "mode": "equilibrium",
        "const_temp": bool(const_temp),
        "t_end": "2.0e9 yr",
        "nside": int(nside),
        "chi0": float(chi0_incident),
        "ion_rate": f"{float(xi_cr)} 1/s",
        "max_iter": 80,
        "reltol": 1.0e-2,
        "abstol0": 1.0e-9,
        "b_kms": 0.3,
        "Zg": 1.0,
        "Zd": 1.0,
        "fH2gr": 1.0,
        "fHplusgr": 0.6,
        "fCplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "fCplusCR": 1.0,
        "shielding_outer_coupling": "pseudotime",
        "shielding_max_iter": int(shielding_max_iter),
        "shielding_reltol": 1.0e-6,
        "shielding_abstol": 1.0e-20,
        "shielding_mix": float(shielding_mix),
        "shielding_self_weight": float(self_weight),
        "enable_co_phase": False,
    }
    if use_radmc3d_chi:
        gow17_cfg["chi_is_incident"] = False
    else:
        gow17_cfg["chi_is_incident"] = True

    chem_result = run_chemistry(radm, model="gow17", config=gow17_cfg)

    y_out = np.asarray(radm.gow17_y, dtype=float)
    Av_out = np.asarray(radm.Av.to("dimensionless").magnitude, dtype=float)

    if diagnostic_outdir is not None:
        outdir = Path(diagnostic_outdir)
        if diagnostic_suffix is None:
            diagnostic_suffix = (
                f"sphere_nH_{int(float(nH_cm3))}_nr_{int(nr)}_nt_{int(ntheta)}_np_{int(nphi)}_nside_{int(nside)}"
            )
        _plot_diagnostics(
            outdir=outdir,
            suffix=str(diagnostic_suffix),
            radm=radm,
            y_out=y_out,
        )

    Av_samp = Av_out.reshape(-1)
    abd: Dict[str, np.ndarray] = {}
    for name, idx in [
        ("He+", I_HEP),
        ("OHx", I_OHX),
        ("CHx", I_CHX),
        ("CO", I_CO),
        ("C+", I_CP),
        ("HCO+", I_HCOP),
        ("H2", I_H2),
        ("H+", I_HP),
        ("H3+", I_H3P),
        ("H2+", I_H2P),
        ("S+", I_SP),
        ("Si+", I_SIP),
        ("O+", I_OP),
    ]:
        abd[name] = _mean_profile_vs_av(
            av_centers=ref.Av,
            av_samples=Av_samp,
            values=y_out[..., idx].reshape(-1),
        )

    abd["E"] = (
        abd["He+"]
        + abd["C+"]
        + abd["HCO+"]
        + abd["H+"]
        + abd["H3+"]
        + abd["H2+"]
        + abd["S+"]
        + abd["Si+"]
        + abd["O+"]
    )

    xC = 1.6e-4
    abd["C"] = xC - abd["CHx"] - abd["CO"] - abd["C+"] - abd["HCO+"]

    gow17_diag = chem_result.meta.get("gow17_diagnostics", {})

    info = {
        "shielding_mode": "gow17_internal_3d_healpix_sphere",
        "nside": int(nside),
        "self_weight": float(self_weight),
        "shape": tuple(int(s) for s in shape),
        "R_cm": float(R_cm),
        "nH_index": int(nH_index),
        "nr": int(nr),
        "ntheta": int(ntheta),
        "nphi": int(nphi),
        "use_radmc3d_chi": bool(use_radmc3d_chi),
        "gow17_diagnostics": gow17_diag,
    }

    return np.asarray(y_out, dtype=float), abd, np.asarray(ref.Av, dtype=float), info


def _safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log10(np.maximum(x, 1.0e-99))


def _plot_diagnostics(*, outdir: Path, suffix: str, radm: RadModel, y_out: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    mesh = radm.model.mesh
    if mesh is None:
        raise ValueError("radm.model.mesh is None")

    if mesh.coord_system != "spherical":
        raise ValueError(f"Diagnostics currently only support spherical meshes, got {mesh.coord_system}")

    r_au = mesh.centers("r").to("au").magnitude
    theta_deg = mesh.centers("theta").to("degree").magnitude
    phi_deg = mesh.centers("phi").to("degree").magnitude

    itheta_mid = int(np.argmin(np.abs(theta_deg - 90.0)))
    iphi0 = 0

    def log10_field(arr: np.ndarray) -> np.ndarray:
        return _safe_log10(np.asarray(arr, dtype=float))

    chi = np.asarray(radm.chi.to("dimensionless").magnitude, dtype=float)
    av = np.asarray(radm.Av.to("dimensionless").magnitude, dtype=float)
    tgas = np.asarray(radm.gas_temperature.to("K").magnitude, dtype=float)
    tdust = np.asarray(radm.dust_temperature.to("K").magnitude, dtype=float)

    species_fields: Dict[str, np.ndarray] = {
        "H2": y_out[..., I_H2],
        "CO": y_out[..., I_CO],
        "C+": y_out[..., I_CP],
        "He+": y_out[..., I_HEP],
        "HCO+": y_out[..., I_HCOP],
    }

    outdir.mkdir(parents=True, exist_ok=True)

    fields_main: Dict[str, Tuple[np.ndarray, str, str]] = {
        "log10_chi": (log10_field(chi), "r-phi @ midplane", "log10 chi"),
        "Av": (av, "r-phi @ midplane", "Av"),
        "Tgas": (tgas, "r-phi @ midplane", "Tgas [K]"),
        "Tdust": (tdust, "r-phi @ midplane", "Tdust [K]"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=220)
    axes = np.asarray(axes)

    for ax, (name, (field, _title, cbar_label)) in zip(axes.reshape(-1), fields_main.items()):
        sl = field[:, itheta_mid, :]
        vmin, vmax = np.nanpercentile(sl, [1.0, 99.0])
        im = ax.imshow(
            sl,
            origin="lower",
            aspect="auto",
            extent=[float(phi_deg.min()), float(phi_deg.max()), float(r_au.min()), float(r_au.max())],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(name)
        ax.set_xlabel("phi [deg]")
        ax.set_ylabel("r [au]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    fig.tight_layout()
    fig.savefig(outdir / f"diag_fields_midplane_{suffix}.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), dpi=220)
    axes = np.asarray(axes).reshape(-1)

    for ax, (name, field) in zip(axes, species_fields.items()):
        sl = log10_field(field[:, itheta_mid, :])
        vmin, vmax = np.nanpercentile(sl, [1.0, 99.0])
        im = ax.imshow(
            sl,
            origin="lower",
            aspect="auto",
            extent=[float(phi_deg.min()), float(phi_deg.max()), float(r_au.min()), float(r_au.max())],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(f"log10 {name}")
        ax.set_xlabel("phi [deg]")
        ax.set_ylabel("r [au]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(species_fields) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(outdir / f"diag_species_midplane_{suffix}.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=220)
    axes = np.asarray(axes)

    for ax, (name, (field, _title, cbar_label)) in zip(axes.reshape(-1), fields_main.items()):
        sl = field[:, :, iphi0]
        vmin, vmax = np.nanpercentile(sl, [1.0, 99.0])
        im = ax.imshow(
            sl,
            origin="lower",
            aspect="auto",
            extent=[float(theta_deg.min()), float(theta_deg.max()), float(r_au.min()), float(r_au.max())],
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.set_title(f"{name} (phi=0)")
        ax.set_xlabel("theta [deg]")
        ax.set_ylabel("r [au]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    fig.tight_layout()
    fig.savefig(outdir / f"diag_fields_meridional_{suffix}.png")
    plt.close(fig)

    r_edges_cm = mesh.edges("r").to("cm").magnitude
    theta_edges = mesh.edges("theta").to("radian").magnitude
    phi_edges = mesh.edges("phi").to("radian").magnitude

    vr = (r_edges_cm[1:] ** 3 - r_edges_cm[:-1] ** 3) / 3.0
    vth = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    vph = phi_edges[1:] - phi_edges[:-1]
    vol = vr[:, None, None] * vth[None, :, None] * vph[None, None, :]
    vol_r = np.sum(vol, axis=(1, 2))

    def radial_avg(field: np.ndarray) -> np.ndarray:
        f = np.asarray(field, dtype=float)
        return np.sum(f * vol, axis=(1, 2)) / vol_r

    prof = {
        "chi": radial_avg(chi),
        "Av": radial_avg(av),
        "Tgas": radial_avg(tgas),
        "Tdust": radial_avg(tdust),
        "H2": radial_avg(species_fields["H2"]),
        "CO": radial_avg(species_fields["CO"]),
        "C+": radial_avg(species_fields["C+"]),
    }

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)
    ax.plot(r_au, log10_field(prof["chi"]), lw=1.2, label="log10 chi")
    ax.plot(r_au, prof["Av"], lw=1.2, label="Av")
    ax.set_xlabel("r [au]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"diag_radial_chi_av_{suffix}.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)
    ax.plot(r_au, prof["Tgas"], lw=1.2, label="Tgas")
    ax.plot(r_au, prof["Tdust"], lw=1.2, label="Tdust")
    ax.set_xlabel("r [au]")
    ax.set_ylabel("T [K]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"diag_radial_temperatures_{suffix}.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)
    for name in ["H2", "CO", "C+"]:
        ax.plot(r_au, log10_field(prof[name]), lw=1.2, label=f"log10 {name}")
    ax.set_xlabel("r [au]")
    ax.set_ylabel("log10 abundance per H")
    ax.set_ylim(-14.0, 0.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"diag_radial_species_{suffix}.png")
    plt.close(fig)


def _plot_compare(
    *,
    outdir: Path,
    Av_ref: np.ndarray,
    Av_db: np.ndarray,
    ref: RefSlab,
    nH_index: int,
    abd_db: Dict[str, np.ndarray],
    species: Iterable[str],
    suffix: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)

    for spec in species:
        y_ref = ref.abd[spec][:, nH_index]
        y_db = abd_db[spec].reshape(-1)

        color = COLORS.get(spec, "#444444")

        ax.plot(Av_ref, _safe_log10(y_ref), lw=1.2, label=f"{spec} ref", color=color)
        ax.plot(Av_db, _safe_log10(y_db), lw=1.2, ls="--", label=f"{spec} diskbridge", color=color)

    ax.set_xlabel("A_V")
    ax.set_ylabel("log10 abundance per H")
    ax.set_ylim(-14.0, 0.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"gow17_fig2_3dhealpix_{suffix}.png")
    plt.close(fig)


def _save_convergence_json(
    outdir: Path,
    diag: dict,
    fname: str = "gow17_convergence.json",
) -> Path:
    """Save convergence diagnostics dict to JSON, converting numpy arrays."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in diag.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    p = outdir / fname
    p.write_text(json.dumps(out, indent=2) + "\n")
    return p


def _plot_convergence(
    outdir: Path,
    diag: dict,
    fname: str = "gow17_convergence.png",
) -> Path:
    """Plot shielding iteration convergence history with budget summary."""
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d_h2 = np.asarray(diag.get("d_h2_hist", []), dtype=float)
    d_co = np.asarray(diag.get("d_co_hist", []), dtype=float)
    iters = np.arange(1, len(d_h2) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)
    if iters.size > 0:
        ax.semilogy(iters, d_h2, "o-", lw=1.2, ms=4, label="d_h2")
        ax.semilogy(iters, d_co, "s-", lw=1.2, ms=4, label="d_co")
    ax.set_xlabel("Shielding iteration")
    ax.set_ylabel("Max relative change")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    text_lines = [
        f"shielding_iter = {diag.get('shielding_iter', '?')}",
        f"h_xH_atom_min = {diag.get('h_xH_atom_min', '?'):.3e}",
        f"c_xC_neutral_min = {diag.get('c_xC_neutral_min', '?'):.3e}",
        f"h_budget_violation = {diag.get('h_budget_violation', '?'):.3e}",
        f"c_budget_violation = {diag.get('c_budget_violation', '?'):.3e}",
    ]
    ax.text(
        0.98, 0.98, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=7, verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
    )

    fig.tight_layout()
    p = outdir / fname
    fig.savefig(p)
    plt.close(fig)
    return p


def _shell_average_sphere(
    *,
    y_out: np.ndarray,
    Av_3d: np.ndarray,
    nbins: int = 64,
) -> Dict[str, np.ndarray]:
    """Shell-average abundances by Av bin and return binned profiles.

    Bins cells by their ``Av_3d`` value (which is a monotonic function
    of depth from the sphere surface) and computes the mean abundance
    in each bin.

    Parameters
    ----------
    y_out : np.ndarray
        GOW17 abundance array, shape ``(nr, ntheta, nphi, N_Y)``.
    Av_3d : np.ndarray
        Visual extinction array, shape ``(nr, ntheta, nphi)``.
    nbins : int
        Number of Av bins.

    Returns
    -------
    dict
        Keys: ``Av``, ``xH2``, ``xCO``, ``xCplus``, with 1-D arrays of
        length *nbins*.
    """
    av_flat = Av_3d.ravel()
    av_min, av_max = float(av_flat.min()), float(av_flat.max())
    bin_edges = np.linspace(av_min, av_max, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    idx = np.digitize(av_flat, bin_edges) - 1
    idx = np.clip(idx, 0, nbins - 1)

    def _bin_mean(field_flat: np.ndarray) -> np.ndarray:
        sums = np.bincount(idx, weights=field_flat, minlength=nbins).astype(float)
        counts = np.bincount(idx, minlength=nbins).astype(float)
        out = np.zeros(nbins, dtype=float)
        m = counts > 0
        out[m] = sums[m] / counts[m]
        return out

    xH2_prof = _bin_mean(y_out[..., I_H2].ravel())
    xCO_prof = _bin_mean(y_out[..., I_CO].ravel())
    xCplus_prof = _bin_mean(y_out[..., I_CP].ravel())

    return {
        "Av": bin_centers,
        "xH2": xH2_prof,
        "xCO": xCO_prof,
        "xCplus": xCplus_prof,
    }


def _save_sphere_profile(
    outdir: Path,
    profile: Dict[str, np.ndarray],
    fname: str = "sphere_profile.csv",
) -> Path:
    """Save shell-averaged profile to CSV."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / fname
    header = "Av,xH2,xCO,xCplus"
    data = np.column_stack([
        profile["Av"], profile["xH2"], profile["xCO"], profile["xCplus"],
    ])
    np.savetxt(p, data, delimiter=",", header=header, comments="")
    return p


def _run_slab_reference(
    *,
    Av_ref: np.ndarray,
    nH_cm3: float,
    chi0: float,
    xi_cr: float,
    shielding_max_iter: int = 20,
    shielding_mix: float = 0.5,
    const_temp: bool = True,
    nside: int = 4,
) -> Dict[str, np.ndarray]:
    """Run 1-D slab reference with Gong+17 Appendix factors.

    Uses the same parameters as the sphere run: ``chi_is_incident=True``
    and ``Av = 2 * Av_perp`` (the doubling is already baked into
    *Av_ref* from the sphere run).

    Parameters
    ----------
    Av_ref : np.ndarray
        1-D array of Av values (already doubled).
    nH_cm3 : float
        Hydrogen number density [cm^-3].
    chi0 : float
        Draine-field chi_0 parameter.
    xi_cr : float
        Cosmic-ray ionisation rate [1/s].
    shielding_max_iter : int
        Max shielding iterations.

    Returns
    -------
    dict
        Keys: ``Av``, ``xH2``, ``xCO``, ``xCplus``.
    """
    units = diskbridge.units
    m_H = units("m_H")
    chi0_incident = 2.0 * float(chi0)

    ncells = int(Av_ref.size)
    NH_flat = Av_ref * 1.87e21

    dr = np.empty(ncells, dtype=float)
    dr[0] = NH_flat[0] / float(nH_cm3) if nH_cm3 > 0 else 1.0
    dr[1:] = np.diff(NH_flat) / float(nH_cm3)
    dr = np.maximum(dr, 1.0)

    r_edges = np.zeros(ncells + 1, dtype=float)
    r_edges[1:] = np.cumsum(dr)

    r_axis = Axis(edges=Quantity(r_edges, "cm"))
    theta_axis = Axis(edges=Quantity(np.array([0.0, np.pi]), "rad"))
    phi_axis = Axis(edges=Quantity(np.array([0.0, 2.0 * np.pi]), "rad"))

    mesh = Mesh.spherical(r=r_axis, theta=theta_axis, phi=phi_axis)
    shape = mesh.shape
    axis_order = mesh.axis_names()

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    nH_field = np.full(shape, float(nH_cm3), dtype=float)
    rho = (Quantity(nH_field, "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=axis_order))

    sigma = Quantity(np.full(shape, 1.0e-21, dtype=float), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=axis_order),
    )

    radm = RadModel(model, model_dir=".")

    radm.chi = Quantity(np.full(shape, chi0_incident, dtype=float), "dimensionless")
    radm.Av = Quantity(Av_ref.reshape(shape), "dimensionless")
    radm.gas_temperature = Quantity(np.full(shape, 50.0, dtype=float), "K")
    radm.dust_temperature = Quantity(np.full(shape, 50.0, dtype=float), "K")

    y0 = np.zeros(N_Y, dtype=float)
    y0[I_HEP] = 1.450654e-08
    y0[I_H3P] = 2.681411e-07
    y0[I_CP] = 1.0e-4
    y0[I_CO] = 1.0e-7
    y0[I_H2] = 0.1
    y0[I_CO_ICE] = 0.0

    y_init = np.zeros(shape + (N_Y,), dtype=float)
    for j in range(N_Y):
        y_init[..., j] = y0[j]
    radm.gow17_y = np.ascontiguousarray(y_init, dtype=np.float64)

    gow17_cfg = {
        "mode": "equilibrium",
        "const_temp": bool(const_temp),
        "t_end": "2.0e9 yr",
        "nside": int(nside),
        "chi0": float(chi0_incident),
        "ion_rate": f"{float(xi_cr)} 1/s",
        "max_iter": 80,
        "reltol": 1.0e-2,
        "abstol0": 1.0e-9,
        "b_kms": 0.3,
        "Zg": 1.0,
        "Zd": 1.0,
        "fH2gr": 1.0,
        "fHplusgr": 0.6,
        "fCplusgr": 0.6,
        "fHeplusgr": 0.6,
        "fSplusgr": 0.6,
        "fSiplusgr": 0.6,
        "fCplusCR": 1.0,
        "shielding_max_iter": int(shielding_max_iter),
        "shielding_reltol": 1.0e-6,
        "shielding_abstol": 1.0e-20,
        "shielding_mix": float(shielding_mix),
        "enable_co_phase": False,
        "chi_is_incident": True,
    }

    run_chemistry(radm, model="gow17", config=gow17_cfg)

    y_slab = np.asarray(radm.gow17_y, dtype=float)
    Av_out = np.asarray(radm.Av.to("dimensionless").magnitude, dtype=float).ravel()

    return {
        "Av": Av_out,
        "xH2": y_slab[..., I_H2].ravel(),
        "xCO": y_slab[..., I_CO].ravel(),
        "xCplus": y_slab[..., I_CP].ravel(),
    }


def _plot_sphere_vs_slab(
    outdir: Path,
    sphere: Dict[str, np.ndarray],
    slab: Dict[str, np.ndarray],
    fname: str = "sphere_vs_slab.png",
) -> Path:
    """Overlay plot comparing sphere shell-average to 1-D slab reference."""
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)

    for key, label, color in [
        ("xH2", "H2", "#1f77b4"),
        ("xCO", "CO", "#d62728"),
        ("xCplus", "C+", "#ff7f0e"),
    ]:
        ax.plot(
            sphere["Av"], _safe_log10(sphere[key]),
            lw=1.4, color=color, label=f"{label} sphere",
        )
        ax.plot(
            slab["Av"], _safe_log10(slab[key]),
            lw=1.4, ls="--", color=color, label=f"{label} slab",
        )

    ax.set_xlabel("A_V")
    ax.set_ylabel("log10 abundance per H")
    ax.set_ylim(-14.0, 0.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)

    fig.tight_layout()
    p = outdir / fname
    fig.savefig(p)
    plt.close(fig)
    return p


def _plot_mode_comparison(
    outdir: Path,
    sphere1: Dict[str, np.ndarray],
    sphere2: Dict[str, np.ndarray],
    label1: str = "mode1 (analytic)",
    label2: str = "RADMC-3D",
    fname: str = "mode1_vs_radmc3d.png",
) -> Path:
    """Overlay shell-averaged profiles from two modes."""
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=220)

    for key, label_sp, color in [
        ("xH2", "H2", "#1f77b4"),
        ("xCO", "CO", "#d62728"),
        ("xCplus", "C+", "#ff7f0e"),
    ]:
        ax.plot(
            sphere1["Av"], _safe_log10(sphere1[key]),
            lw=1.4, color=color, label=f"{label_sp} {label1}",
        )
        ax.plot(
            sphere2["Av"], _safe_log10(sphere2[key]),
            lw=1.4, ls="--", color=color, label=f"{label_sp} {label2}",
        )

    ax.set_xlabel("A_V")
    ax.set_ylabel("log10 abundance per H")
    ax.set_ylim(-14.0, 0.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)

    fig.tight_layout()
    p = outdir / fname
    fig.savefig(p)
    plt.close(fig)
    return p


def _postprocess_run(
    *,
    outdir: Path,
    y_out: np.ndarray,
    info: dict,
    Av_3d: np.ndarray,
    nH_cm3: float,
    chi0: float,
    xi_cr: float,
    shielding_max_iter: int,
    shielding_mix: float = 0.5,
    const_temp: bool = True,
    nside: int = 4,
    suffix: str = "",
) -> Dict[str, np.ndarray]:
    """Common post-processing: save convergence, shell-avg, slab ref, overlay.

    Returns the shell-averaged sphere profile dict.
    """
    diag = info.get("gow17_diagnostics", {})

    conv_json_name = f"gow17_convergence{suffix}.json"
    conv_plot_name = f"gow17_convergence{suffix}.png"
    sphere_csv_name = f"sphere_profile{suffix}.csv"
    overlay_name = f"sphere_vs_slab{suffix}.png"

    _save_convergence_json(outdir, diag, fname=conv_json_name)
    _plot_convergence(outdir, diag, fname=conv_plot_name)

    sphere = _shell_average_sphere(
        y_out=y_out, Av_3d=Av_3d, nbins=64,
    )
    _save_sphere_profile(outdir, sphere, fname=sphere_csv_name)

    slab = _run_slab_reference(
        Av_ref=sphere["Av"],
        nH_cm3=nH_cm3,
        chi0=chi0,
        xi_cr=xi_cr,
        shielding_max_iter=shielding_max_iter,
        shielding_mix=shielding_mix,
        const_temp=const_temp,
        nside=nside,
    )
    _plot_sphere_vs_slab(outdir, sphere, slab, fname=overlay_name)

    print(f"  Saved: {conv_json_name}, {conv_plot_name}, {sphere_csv_name}, {overlay_name}")
    return sphere


def main() -> None:

    root = Path(__file__).resolve().parents[1]
    ref_dir = root / "other_codes" / "pdr" / "out_example_simple"
    ref = _load_reference(ref_dir)

    outdir = Path(__file__).resolve().parent / "reproduce_gow17_fig2_3dhealpix"
    outdir.mkdir(parents=True, exist_ok=True)

    species = ["CO", "C", "C+", "H3+", "OHx", "CHx", "He+"]

    nH_index = 0
    nH_val = float(ref.nH_values[int(nH_index)])
    chi0 = 1.0
    xi_cr = 2.0e-16
    shielding_max_iter = 20
    shielding_mix = 0.5
    const_temp = True
    nr = 128
    ntheta = 8
    nphi = 16
    nside = 4

    # ----------------------------------------------------------------
    # Run 1: internal analytic field (no RADMC-3D)
    # ----------------------------------------------------------------
    print("=" * 60)
    print("Run 1: internal analytic (chi_is_incident=True)")
    print("=" * 60)

    y_out_m1, abd_m1, Av_db_m1, info_m1 = run_gow17_internal_3d_healpix_sphere(
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        shielding_max_iter=shielding_max_iter,
        shielding_mix=shielding_mix,
        const_temp=const_temp,
        nside=nside,
        self_weight=1.0,
        nr=nr,
        ntheta=ntheta,
        nphi=nphi,
        ref=ref,
        nH_index=int(nH_index),
        use_radmc3d_chi=False,
        diagnostic_outdir=outdir,
    )

    suffix_m1 = (
        f"sphere_nH_{int(nH_val)}_nr_{nr}_nt_{ntheta}_np_{nphi}_nside_{nside}"
    )
    _plot_compare(
        outdir=outdir,
        Av_ref=ref.Av,
        Av_db=Av_db_m1,
        ref=ref,
        nH_index=int(nH_index),
        abd_db=abd_m1,
        species=species,
        suffix=suffix_m1,
    )

    # Reconstruct Av_3d for shell averaging
    R_cm = float(info_m1["R_cm"])
    mesh_shape = tuple(info_m1["shape"])
    r_cent_arr = np.linspace(0.5 * R_cm / nr, R_cm - 0.5 * R_cm / nr, nr)
    r3 = np.broadcast_to(r_cent_arr[:, None, None], mesh_shape)
    depth_cm = np.maximum(R_cm - r3, 0.0)
    NH_depth = nH_val * depth_cm
    Av_3d_m1 = 2.0 * NH_depth / 1.87e21

    sphere_m1 = _postprocess_run(
        outdir=outdir,
        y_out=y_out_m1,
        info=info_m1,
        Av_3d=Av_3d_m1,
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        shielding_max_iter=shielding_max_iter,
        shielding_mix=shielding_mix,
        const_temp=const_temp,
        nside=nside,
        suffix="",
    )

    print(
        f"nH={nH_val:.3e}: shape={info_m1['shape']}, R_cm={info_m1['R_cm']:.3e}"
    )

    # ----------------------------------------------------------------
    # Run 2: RADMC-3D chi
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Run 2: RADMC-3D chi (chi_is_incident=False)")
    print("=" * 60)

    y_out_m2, abd_m2, Av_db_m2, info_m2 = run_gow17_internal_3d_healpix_sphere(
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        shielding_max_iter=shielding_max_iter,
        shielding_mix=shielding_mix,
        const_temp=const_temp,
        nside=nside,
        self_weight=1.0,
        nr=nr,
        ntheta=ntheta,
        nphi=nphi,
        ref=ref,
        nH_index=int(nH_index),
        use_radmc3d_chi=True,
        radmc3d_nphot_thermal=200000000,
        radmc3d_nphot_mono=200000000,
        diagnostic_outdir=outdir,
        diagnostic_suffix=f"radmc3d_sphere_nH_{int(nH_val)}_nr_{nr}_nt_{ntheta}_np_{nphi}_nside_{nside}",
    )

    suffix_m2 = (
        f"radmc3d_sphere_nH_{int(nH_val)}_nr_{nr}_nt_{ntheta}_np_{nphi}_nside_{nside}"
    )
    _plot_compare(
        outdir=outdir,
        Av_ref=ref.Av,
        Av_db=Av_db_m2,
        ref=ref,
        nH_index=int(nH_index),
        abd_db=abd_m2,
        species=species,
        suffix=suffix_m2,
    )

    Av_3d_m2 = 2.0 * nH_val * np.maximum(R_cm - r3, 0.0) / 1.87e21

    sphere_m2 = _postprocess_run(
        outdir=outdir,
        y_out=y_out_m2,
        info=info_m2,
        Av_3d=Av_3d_m2,
        nH_cm3=nH_val,
        chi0=chi0,
        xi_cr=xi_cr,
        shielding_max_iter=shielding_max_iter,
        shielding_mix=shielding_mix,
        const_temp=const_temp,
        nside=nside,
        suffix="_radmc3d",
    )

    # ----------------------------------------------------------------
    # Final comparison: mode1 vs RADMC-3D
    # ----------------------------------------------------------------
    _plot_mode_comparison(
        outdir, sphere_m1, sphere_m2,
        label1="analytic", label2="RADMC-3D",
        fname="mode1_vs_radmc3d.png",
    )
    print(f"\nSaved: mode1_vs_radmc3d.png")
    print("Done.")


if __name__ == "__main__":
    main()
