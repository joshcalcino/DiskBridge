from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

import diskbridge
from diskbridge._units import Quantity, units
from diskbridge._params import DEFAULT_PARAMS_FILE
from diskbridge.chemistry.api import run_chemistry
from diskbridge.chemistry.shielding.angular_uv_weights import compute_uv_direction_weights_healpix
from diskbridge.chemistry.shielding.dust_uv_tau import prepare_dust_density_fields
from diskbridge.radmc3d.dustkappa_reader import load_kext_uv_for_bins
from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.model.dust import Dust, DustDistribution
from diskbridge.radmc3d.model import RadModel
from diskbridge.radmc3d.writer import RadWriter
from diskbridge.radmc3d.cache import find_cached_output


@dataclass(frozen=True)
class CubeTestConfig:
    nx: int = 64
    ny: int = 64
    nz: int = 64

    L_au: float = 100000.0

    # Turbulent lognormal density: rho = rho0 * exp(s), where s is correlated Gaussian
    rho0_g_cm3_list: tuple[float, ...] = (1.0e-21,)
    sigma_s: float = 1.0
    peak_boost: float = 5.0
    p: float = 11.0 / 3.0
    k_min: float = 1.0
    k_max: float = 4.0
    k_multiplier: float = 2.0
    seed: int = 1

    # Dust/chemistry
    dust_nbins: int = 10
    dust_to_gas_ratio: float = 1.0e-2
    Tdust_K: float = 20.0

    chi0: float = 1.0

    # Shielding
    shielding_iter: int = 5
    nside: int = 4
    b_kms: float = 0.3

    # RADMC
    nphot_thermal: int = 20000000
    nphot_mono: int = 20000000
    scat_mode: int = 0
    nbcores: int = 12


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _annotate(ax, lines: list[str]) -> None:
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.6"},
    )


def _write_inputs(out_dir: Path, cfg: CubeTestConfig) -> None:
    inputs = {
        "job": "cube_test",
        "time": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.__dict__,
        "diskbridge_params": {
            "lambda_min_micron": float(diskbridge.params.lambda_min.to("micron").magnitude),
            "lambda_max_micron": float(diskbridge.params.lambda_max.to("micron").magnitude),
            "n_lambda": int(diskbridge.params.n_lambda),
            "species": diskbridge.params.species,
            "opacity_dir": str(diskbridge.params.opacity_dir),
        },
    }
    (out_dir / "inputs.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True), encoding="utf-8"
    )


def _build_cube_mesh(cfg: CubeTestConfig) -> Mesh:
    L_cm = Quantity(float(cfg.L_au), "au").to("cm").magnitude

    def edges(n: int) -> np.ndarray:
        return np.linspace(-0.5 * L_cm, 0.5 * L_cm, int(n) + 1, dtype=float)

    mesh = Mesh.cartesian(
        x=Axis(edges=Quantity(edges(cfg.nx), "cm")),
        y=Axis(edges=Quantity(edges(cfg.ny), "cm")),
        z=Axis(edges=Quantity(edges(cfg.nz), "cm")),
    )
    return mesh


def _make_correlated_gaussian_field(
    *,
    shape: tuple[int, int, int],
    p: float,
    k_min: float,
    k_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))

    # Use integer Fourier mode indices so k_min/k_max map to visually intuitive
    # structure sizes. For nx=32, k=1 is box-scale, k=2 is half-box, ...
    kx = np.fft.fftfreq(nx) * float(nx)
    ky = np.fft.fftfreq(ny) * float(ny)
    kz = np.fft.fftfreq(nz) * float(nz)
    kkx, kky, kkz = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kkx * kkx + kky * kky + kkz * kkz)

    kmin = float(k_min)
    kmax = float(k_max)

    amp = np.zeros_like(k, dtype=float)
    m = (k >= kmin) & (k <= kmax) & (k > 0.0)
    amp[m] = k[m] ** (-0.5 * float(p))

    g = rng.normal(size=shape)
    h = rng.normal(size=shape)

    fk = (g + 1j * h) * amp
    fk[0, 0, 0] = 0.0

    s = np.fft.ifftn(fk).real
    s = (s - float(np.mean(s))) / float(np.std(s))
    return s


def _make_lognormal_density(
    *,
    rho0_g_cm3: float,
    sigma_s: float,
    s: np.ndarray,
) -> np.ndarray:
    sigma_s = float(sigma_s)
    s_scaled = sigma_s * np.asarray(s, dtype=float)

    # Ensure mean density is rho0 by subtracting 0.5*sigma^2 (lognormal property)
    mu = -0.5 * sigma_s * sigma_s
    rho = float(rho0_g_cm3) * np.exp(mu + s_scaled)
    return np.asarray(rho, dtype=float)


def _sigma_for_peak_boost(*, s: np.ndarray, sigma0: float, boost: float) -> float:
    sigma0 = float(sigma0)
    boost = float(boost)
    if sigma0 <= 0.0:
        raise ValueError("sigma_s must be > 0")
    if boost <= 0.0:
        raise ValueError("peak_boost must be > 0")
    if boost == 1.0:
        return sigma0

    smax = float(np.nanmax(np.asarray(s, dtype=float)))
    if not np.isfinite(smax) or smax <= 0.0:
        raise ValueError("Cannot determine peak boost: non-finite or non-positive s_max")

    # Baseline peak ratio relative to rho0 for lognormal with mean fixed to rho0.
    # rho_max / rho0 = exp(-0.5*sigma^2 + sigma*smax)
    log_r0 = (-0.5 * sigma0 * sigma0) + (sigma0 * smax)
    log_r1 = log_r0 + np.log(boost)

    disc = (smax * smax) - (2.0 * log_r1)
    if disc <= 0.0:
        raise ValueError(
            "Requested peak_boost is too large for this realization; "
            "increase grid size or reduce peak_boost"
        )

    root = float(np.sqrt(disc))
    sigma1 = smax - root
    if sigma1 <= 0.0:
        sigma1 = smax + root
    if sigma1 <= 0.0:
        raise ValueError("Failed to compute a positive sigma for peak boost")
    return float(sigma1)


def _build_model_with_density(mesh: Mesh, rho_g_cm3: np.ndarray) -> Model:
    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    rho = Quantity(np.asarray(rho_g_cm3, dtype=float), "g/cm^3")
    model.gas_register(
        "density",
        Field(quantity="density", data=rho, axis_order=("x", "y", "z")),
    )

    return model


def _ensure_dust_setup(cfg: CubeTestConfig, model: Model) -> None:
    if model.dust is None:
        model.dust = Dust(model)

    model.dust.set_distribution(
        nbin=int(cfg.dust_nbins),
        dust_to_gas_ratio=float(cfg.dust_to_gas_ratio),
        mode="proportional",
    )

    if model.dust.nbin <= 0:
        raise RuntimeError("Dust distribution created zero bins")


def _ensure_dustkappa_files(
    *,
    inputs_dir: Path,
    species: str,
    nbin: int,
    source_dir: Path,
) -> list[str]:
    copied: list[str] = []
    for i in range(int(nbin)):
        fname = f"dustkappa_{species}{i}.inp"
        dst = inputs_dir / fname
        if dst.exists():
            continue
        src = source_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing opacity file {src}")
        dst.write_bytes(src.read_bytes())
        copied.append(fname)
    return copied


def _write_radmc3d_params_txt(
    *,
    model_dir: Path,
    cfg: CubeTestConfig,
) -> Path:
    params_path = model_dir / "params.txt"

    # Start from the repo's default params and overlay minimal keys.
    text = Path(DEFAULT_PARAMS_FILE).read_text(encoding="utf-8").splitlines(True)

    updates: dict[str, str] = {
        "nbcores": str(int(cfg.nbcores)),
        "nphot_thermal": str(int(cfg.nphot_thermal)),
        "nphot_mono": str(int(cfg.nphot_mono)),
        "nphot_scat": str(0),
        "scat_mode": str(int(cfg.scat_mode)),
        "lambda_min": str(0.05),
        "uv_min": str(91.2),
        "uv_max": str(111.8),
        "uv_n_wavelengths": str(10),
        "external_uv": "T",
        "external_uv_chi": str(float(cfg.chi0)),
        "nside": str(int(cfg.nside)),
        "nbins": str(int(cfg.dust_nbins)),
        "dust_to_gas_ratio": str(float(cfg.dust_to_gas_ratio)),
    }

    out_lines: list[str] = []
    seen: set[str] = set()
    for line in text:
        if "=" not in line or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            out_lines.append(f"{key} = {updates[key]}\n")
            seen.add(key)
        else:
            out_lines.append(line)

    for k, v in updates.items():
        if k not in seen:
            out_lines.append(f"{k} = {v}\n")

    params_path.write_text("".join(out_lines), encoding="utf-8")
    return params_path


def _write_stars_none(*, inputs_dir: Path) -> None:
    wav_path = inputs_dir / "wavelength_micron.inp"
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing wavelength grid file: {wav_path}")

    with open(wav_path, "r") as f:
        n = int(f.readline().strip())
        wav_um = [float(f.readline().strip()) for _ in range(n)]

    stars_path = inputs_dir / "stars.inp"
    with open(stars_path, "w") as f:
        f.write("2\n")
        f.write(f"0 {n}\n")
        for w in wav_um:
            f.write(f"{w:13.6e}\n")


def _compute_and_attach_uv_weights(
    *,
    rad: RadModel,
    cfg: CubeTestConfig,
    chi: Quantity,
) -> None:
    """Compute directional UV weights and attach to rad.W_rays.

    Uses dustkappa-derived UV extinction opacities and the external UV field
    strength to build per-direction weights via
    :func:`compute_uv_direction_weights_healpix`.  The weights are stored on
    ``rad.W_rays`` so that GOW17 chemistry picks them up automatically.

    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper (must have dust set up).
    cfg : CubeTestConfig
        Cube test configuration.
    chi : Quantity
        RADMC-3D chi field (Draine units), 3-D grid.
    """
    inputs_dir = Path(rad.model_dir) / "radmc3d_inputs"
    species = (
        diskbridge.params.species[0]
        if isinstance(diskbridge.params.species, list)
        else str(diskbridge.params.species)
    )
    kext_uv = load_kext_uv_for_bins(
        radmc_inputs_dir=inputs_dir,
        species_base=species,
        nbin=int(rad.model.dust.nbin),
        uv_min_um=float(diskbridge.params.uv_min.to("micrometer").magnitude),
        uv_max_um=float(diskbridge.params.uv_max.to("micrometer").magnitude),
    )
    if kext_uv is None:
        print("  [WARNING] Could not load dustkappa UV opacities; skipping UV weights")
        rad.W_rays = None
        return

    dust_rho_bins = prepare_dust_density_fields(rad.model)

    chi_arr = np.asarray(chi.to("dimensionless").magnitude, dtype=np.float64)

    W_rays, _, _, _, debug = compute_uv_direction_weights_healpix(
        rad.model.mesh,
        chi_radmc=chi_arr,
        nside=int(cfg.nside),
        dust_rho_bins=dust_rho_bins,
        kext_uv=kext_uv,
        chi_ext0=float(cfg.chi0),
        star_uv_luminosity_erg_s=0.0,
    )
    rad.W_rays = W_rays
    print(f"  UV weights computed: shape={W_rays.shape}, "
          f"median(chi_iso)={float(np.median(debug.get('chi_iso', [0.0]))):.3e}")


def _run_radmc3d_external_uv(
    *,
    cfg: CubeTestConfig,
    rad: RadModel,
    model_dir: Path,
) -> None:
    model_dir = Path(model_dir)
    inputs_dir = model_dir / "radmc3d_inputs"
    outputs_dir = model_dir / "radmc3d_outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Override global params for this run and sync writer/model.
    params_path = _write_radmc3d_params_txt(model_dir=model_dir, cfg=cfg)
    prev_params = diskbridge.params
    prev_rad_params = rad.params
    prev_writer_params = rad.writer.params
    new_params = diskbridge.read_params(params_path)
    diskbridge.params = new_params
    diskbridge._params_module.params = new_params
    rad.params = new_params
    rad.writer.params = new_params

    try:
        rad.writer.write_amr_grid(model_dir)
        rad.writer.write_wavelength_grid(model_dir)
        _write_stars_none(inputs_dir=inputs_dir)
        rad.writer.write_radmc3d_inp(
            model_dir,
            scattering_mode_max=int(new_params.scat_mode),
            nphot=int(new_params.nphot_thermal),
            nphot_mono=int(new_params.nphot_mono),
            nphot_scat=int(new_params.nphot_scat),
            setthreads=int(new_params.nbcores),
        )
        rad.writer.write_dust_density(model_dir, binary=True)
        rad.writer.write_dustopac(model_dir, scattering_mode=int(new_params.scat_mode))

        # Ensure dustkappa files exist (reuse baseline precomputed opacities).
        root = Path(__file__).resolve().parents[2]
        opac_src_dir = root / "examples" / "testbed" / "baseline_run" / "radmc3d_inputs"
        species = (
            new_params.species[0]
            if isinstance(new_params.species, list)
            else str(new_params.species)
        )
        _ensure_dustkappa_files(
            inputs_dir=inputs_dir,
            species=species,
            nbin=int(rad.model.dust.nbin),
            source_dir=opac_src_dir,
        )

        rad.writer.write_external_source(model_dir, chi=float(cfg.chi0))

        # Provide a fixed dust temperature, written in the location mcmono expects.
        temp = Quantity(
            np.full(rad.model.mesh.shape, float(cfg.Tdust_K), dtype=float),
            "K",
        )
        rad.dust_temperature = temp

        writer = RadWriter(rad.model, organize_files=True)
        writer.write_dust_temperature(
            temperature=temp,
            output_dir=outputs_dir,
            nspec=int(rad.model.dust.nbin),
        )

        rad.ensure_chi(force=True)

    finally:
        diskbridge.params = prev_params
        diskbridge._params_module.params = prev_params
        rad.params = prev_rad_params
        rad.writer.params = prev_writer_params


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0.0))


def _midplane_slice(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 3:
        raise ValueError("expected 3D array")
    k = a.shape[2] // 2
    return np.asarray(a[:, :, k], dtype=float)


def _write_cube_plots(
    *,
    out_dir: Path,
    cfg: CubeTestConfig,
    sigma_eff: float,
    rho_g_cm3: np.ndarray,
    chi: Quantity,
    chi_eff_off: Quantity,
    chi_eff_on: Quantity,
    theta_co_off: Quantity,
    theta_co_on: Quantity,
    Xco_off: Quantity,
    Xco_on: Quantity,
    nco_off: Quantity,
    nco_on: Quantity,
    network: str = "gow17",
) -> None:
    plt = _setup_matplotlib()

    tiny = float(np.finfo(np.float64).tiny)

    rho = np.asarray(rho_g_cm3, dtype=float)
    chi_arr = np.asarray(chi.to("dimensionless").magnitude, dtype=float)
    chi_eff_off_arr = np.asarray(chi_eff_off.to("dimensionless").magnitude, dtype=float)
    chi_eff_on_arr = np.asarray(chi_eff_on.to("dimensionless").magnitude, dtype=float)
    theta_co_off_arr = np.asarray(theta_co_off.to("dimensionless").magnitude, dtype=float) if hasattr(theta_co_off, 'to') else np.asarray(theta_co_off, dtype=float)
    theta_co_on_arr = np.asarray(theta_co_on.to("dimensionless").magnitude, dtype=float) if hasattr(theta_co_on, 'to') else np.asarray(theta_co_on, dtype=float)
    Xco_off_arr = np.asarray(Xco_off.to("dimensionless").magnitude, dtype=float)
    Xco_on_arr = np.asarray(Xco_on.to("dimensionless").magnitude, dtype=float)
    nco_off_arr = np.asarray(nco_off.to("cm^-3").magnitude, dtype=float)
    nco_on_arr = np.asarray(nco_on.to("cm^-3").magnitude, dtype=float)

    ratio = np.divide(
        Xco_on_arr,
        Xco_off_arr,
        out=np.full_like(Xco_on_arr, np.nan, dtype=float),
        where=(Xco_off_arr > 0.0),
    )

    nco_ratio = np.divide(
        nco_on_arr,
        nco_off_arr,
        out=np.full_like(nco_on_arr, np.nan, dtype=float),
        where=(nco_off_arr > 0.0),
    )

    params_box = [
        f"n={cfg.nx}x{cfg.ny}x{cfg.nz}",
        f"L={cfg.L_au:g} au",
        f"rho0={cfg.rho0_g_cm3_list}",
        f"sigma0={cfg.sigma_s:g} sigma_eff={float(sigma_eff):g} (peak_boost={cfg.peak_boost:g})",
        f"p={cfg.p:g}",
        f"k=[{cfg.k_min:g},{cfg.k_max:g}] * {cfg.k_multiplier:g} (integer modes)",
        f"seed={cfg.seed}",
        f"nside={cfg.nside}",
        f"shielding_iter={cfg.shielding_iter}",
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(np.log10(rho.reshape(-1)), bins=60, color="k", alpha=0.8)
    ax.set_xlabel("log10(rho[g/cm^3])")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    _annotate(ax, params_box)
    fig.tight_layout()
    fig.savefig(out_dir / f"{network}_hist_log10_rho.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(
        np.log10(np.maximum(Xco_off_arr.reshape(-1), tiny)),
        bins=60,
        alpha=0.7,
        label="Xco (shielding off)",
    )
    ax.hist(
        np.log10(np.maximum(Xco_on_arr.reshape(-1), tiny)),
        bins=60,
        alpha=0.7,
        label="Xco (shielding on)",
    )
    ax.set_xlabel("log10(Xco_gas)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _annotate(ax, params_box)
    fig.tight_layout()
    fig.savefig(out_dir / f"{network}_hist_log10_Xco.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(
        np.log10(np.maximum(nco_off_arr.reshape(-1), tiny)),
        bins=60,
        alpha=0.7,
        label="nCO_gas [cm^-3] (shielding off)",
    )
    ax.hist(
        np.log10(np.maximum(nco_on_arr.reshape(-1), tiny)),
        bins=60,
        alpha=0.7,
        label="nCO_gas [cm^-3] (shielding on)",
    )
    ax.set_xlabel("log10(nCO_gas [cm^-3])")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _annotate(ax, params_box)
    fig.tight_layout()
    fig.savefig(out_dir / f"{network}_hist_log10_nCO_gas.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(np.log10(np.maximum(ratio.reshape(-1), tiny)), bins=60, color="tab:blue", alpha=0.8)
    ax.set_xlabel("log10(Xco_on / Xco_off)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    _annotate(ax, params_box)
    fig.tight_layout()
    fig.savefig(out_dir / f"{network}_hist_log10_Xco_ratio.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(np.log10(np.maximum(nco_ratio.reshape(-1), tiny)), bins=60, color="tab:blue", alpha=0.8)
    ax.set_xlabel("log10(nCO_on / nCO_off)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    _annotate(ax, params_box)
    fig.tight_layout()
    fig.savefig(out_dir / f"{network}_hist_log10_nCO_ratio.png", dpi=150)
    plt.close(fig)

    def _imshow(ax, img, title, cbar_label, vmin=None, vmax=None, log=False):
        arr = np.asarray(img, dtype=float)
        if log:
            arr = np.log10(np.maximum(arr, tiny))
        im = ax.imshow(arr.T, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label)

    def _log10_img(a: np.ndarray) -> np.ndarray:
        return np.log10(np.maximum(np.asarray(a, dtype=float), tiny))

    rho_img = _log10_img(_midplane_slice(rho))
    xco_off_img = _log10_img(_midplane_slice(Xco_off_arr))
    xco_on_img = _log10_img(_midplane_slice(Xco_on_arr))
    nco_off_img = _log10_img(_midplane_slice(nco_off_arr))
    nco_on_img = _log10_img(_midplane_slice(nco_on_arr))
    chi_img = _log10_img(_midplane_slice(chi_arr))
    chi_eff_off_img = _log10_img(_midplane_slice(chi_eff_off_arr))
    chi_eff_on_img = _log10_img(_midplane_slice(chi_eff_on_arr))
    theta_co_off_img = np.asarray(_midplane_slice(theta_co_off_arr), dtype=float)
    theta_co_on_img = np.asarray(_midplane_slice(theta_co_on_arr), dtype=float)

    xco_vmin = float(np.nanmin([np.nanmin(xco_off_img), np.nanmin(xco_on_img)]))
    xco_vmax = float(np.nanmax([np.nanmax(xco_off_img), np.nanmax(xco_on_img)]))

    chi_eff_vmin = float(
        np.nanmin([np.nanmin(chi_eff_off_img), np.nanmin(chi_eff_on_img)])
    )
    chi_eff_vmax = float(
        np.nanmax([np.nanmax(chi_eff_off_img), np.nanmax(chi_eff_on_img)])
    )

    nco_vmin = float(np.nanmin([np.nanmin(nco_off_img), np.nanmin(nco_on_img)]))
    nco_vmax = float(np.nanmax([np.nanmax(nco_off_img), np.nanmax(nco_on_img)]))

    theta_vmin = float(np.nanmin([np.nanmin(theta_co_off_img), np.nanmin(theta_co_on_img)]))
    theta_vmax = float(np.nanmax([np.nanmax(theta_co_off_img), np.nanmax(theta_co_on_img)]))

    fig, axs = plt.subplots(2, 3, figsize=(12.0, 7.0))
    _imshow(axs[0, 0], rho_img, "rho midplane", "log10(g/cm^3)")
    _imshow(axs[0, 1], xco_off_img, "Xco (shielding off)", "log10", vmin=xco_vmin, vmax=xco_vmax)
    _imshow(axs[0, 2], xco_on_img, "Xco (shielding on)", "log10", vmin=xco_vmin, vmax=xco_vmax)
    _imshow(axs[1, 0], chi_img, "chi midplane", "log10")
    _imshow(
        axs[1, 1],
        chi_eff_off_img,
        "chi_eff (shielding off)",
        "log10",
        vmin=chi_eff_vmin,
        vmax=chi_eff_vmax,
    )
    _imshow(
        axs[1, 2],
        chi_eff_on_img,
        "chi_eff (shielding on)",
        "log10",
        vmin=chi_eff_vmin,
        vmax=chi_eff_vmax,
    )
    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("cube_test slices")
    fig.tight_layout()
    fig.savefig(out_dir / f"{network}_slices_midplane.png", dpi=150)
    plt.close(fig)

    def _log10_ratio_img(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        r = np.divide(
            np.asarray(num, dtype=float),
            np.asarray(den, dtype=float),
            out=np.full_like(np.asarray(num, dtype=float), np.nan, dtype=float),
            where=(np.asarray(den, dtype=float) > 0.0),
        )
        return np.log10(np.maximum(r, tiny))

    nco_ratio_img = _log10_ratio_img(_midplane_slice(nco_on_arr), _midplane_slice(nco_off_arr))
    xco_ratio_img = _log10_ratio_img(_midplane_slice(Xco_on_arr), _midplane_slice(Xco_off_arr))

    fig, axs = plt.subplots(2, 3, figsize=(12.0, 7.0))
    _imshow(axs[0, 0], nco_off_img, "nCO_gas (shielding off)", "log10(cm^-3)", vmin=nco_vmin, vmax=nco_vmax)
    _imshow(axs[0, 1], nco_on_img, "nCO_gas (shielding on)", "log10(cm^-3)", vmin=nco_vmin, vmax=nco_vmax)
    _imshow(axs[0, 2], nco_ratio_img, "nCO ratio (on/off)", "log10")
    _imshow(axs[1, 0], xco_off_img, "Xco (shielding off)", "log10", vmin=xco_vmin, vmax=xco_vmax)
    _imshow(axs[1, 1], xco_on_img, "Xco (shielding on)", "log10", vmin=xco_vmin, vmax=xco_vmax)
    _imshow(axs[1, 2], xco_ratio_img, "Xco ratio (on/off)", "log10")
    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("cube_test CO midplane (shared color scales)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{network}_co_midplane_compare.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.0))
    _imshow(axs[0], theta_co_off_img, "theta_co (shielding off)", "dimensionless", vmin=theta_vmin, vmax=theta_vmax)
    _imshow(axs[1], theta_co_on_img, "theta_co (shielding on)", "dimensionless", vmin=theta_vmin, vmax=theta_vmax)
    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("cube_test CO shielding factor")
    fig.tight_layout()
    fig.savefig(out_dir / f"{network}_theta_co_midplane.png", dpi=150)
    plt.close(fig)


def _compare_midplane(
    *,
    out_path: Path,
    title: str,
    off: np.ndarray,
    on: np.ndarray,
    unit_label: str,
    log10: bool,
    cmap: str,
    ratio_cmap: str,
    label_a: str = "shielding off",
    label_b: str = "shielding on",
) -> None:
    plt = _setup_matplotlib()
    tiny = float(np.finfo(np.float64).tiny)

    off = np.asarray(off, dtype=float)
    on = np.asarray(on, dtype=float)

    if log10:
        off_img = np.log10(np.maximum(off, tiny))
        on_img = np.log10(np.maximum(on, tiny))
    else:
        off_img = off
        on_img = on

    vmin = float(np.nanmin([np.nanmin(off_img), np.nanmin(on_img)]))
    vmax = float(np.nanmax([np.nanmax(off_img), np.nanmax(on_img)]))

    ratio = np.divide(
        on, off,
        out=np.full_like(on, np.nan, dtype=float),
        where=(off > 0.0),
    )
    ratio_img = np.log10(np.maximum(ratio, tiny))
    maxabs = float(np.nanmax(np.abs(ratio_img[np.isfinite(ratio_img)])))
    if not np.isfinite(maxabs) or maxabs <= 0.0:
        maxabs = 1.0

    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.0))
    im0 = axs[0].imshow(off_img.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(label_a)
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04).set_label(unit_label)

    im1 = axs[1].imshow(on_img.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title(label_b)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04).set_label(unit_label)

    im2 = axs[2].imshow(ratio_img.T, origin="lower", cmap=ratio_cmap, vmin=-maxabs, vmax=maxabs)
    axs[2].set_title(f"log10({label_b}/{label_a})")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04).set_label("log10")

    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_chemistry_compare_plots(
    *,
    out_dir: Path,
    rad: RadModel,
    chem_off,
    chem_on,
    network: str = "gow17",
) -> None:
    """Write midplane comparison plots for shielding-off vs shielding-on.

    Fields that are only available in a given GOW17 configuration are plotted
    conditionally.

    Parameters
    ----------
    out_dir : Path
        Output directory for plots.
    rad : RadModel
        RADMC-3D model wrapper.
    chem_off, chem_on : ChemistryResult
        Chemistry results with shielding off / on.
    network : str
        Chemistry network name (used in plot filenames).
    """
    out_dir = Path(out_dir)
    nH = rad.ensure_nH().to("cm^-3").magnitude

    def _get_field(chem, key, unit):
        """Safely extract a field/number_density/abundance as float64 array, or None."""
        for src in (chem.fields, chem.number_densities, chem.abundances):
            if key in src:
                from pint.errors import DimensionalityError as _DE
                try:
                    return np.asarray(src[key].to(unit).magnitude, dtype=float)
                except _DE:
                    continue
        return None

    def mid(a):
        return _midplane_slice(np.asarray(a, dtype=float))

    nco_off = _get_field(chem_off, "co", "cm^-3")
    nco_on = _get_field(chem_on, "co", "cm^-3")
    nco_ice_off = _get_field(chem_off, "co_ice", "cm^-3")
    nco_ice_on = _get_field(chem_on, "co_ice", "cm^-3")

    Xco_off = _get_field(chem_off, "co", "dimensionless")
    Xco_on = _get_field(chem_on, "co", "dimensionless")

    chi_eff_off = _get_field(chem_off, "chi_eff", "dimensionless")
    chi_eff_on = _get_field(chem_on, "chi_eff", "dimensionless")
    theta_off = _get_field(chem_off, "theta_co", "dimensionless")
    theta_on = _get_field(chem_on, "theta_co", "dimensionless")

    nCplus_off = _get_field(chem_off, "c+", "cm^-3")
    nCplus_on = _get_field(chem_on, "c+", "cm^-3")
    nC_off = _get_field(chem_off, "catom", "cm^-3")
    nC_on = _get_field(chem_on, "catom", "cm^-3")
    ne_off = _get_field(chem_off, "e", "cm^-3")
    ne_on = _get_field(chem_on, "e", "cm^-3")
    nH2_off = _get_field(chem_off, "h2", "cm^-3")
    nH2_on = _get_field(chem_on, "h2", "cm^-3")
    nHI_off = _get_field(chem_off, "h", "cm^-3")
    nHI_on = _get_field(chem_on, "h", "cm^-3")

    with np.errstate(divide="ignore", invalid="ignore"):
        X_Cplus_off = np.where(nH > 0.0, nCplus_off / nH, 0.0) if nCplus_off is not None else None
        X_Cplus_on = np.where(nH > 0.0, nCplus_on / nH, 0.0) if nCplus_on is not None else None
        X_C_off = np.where(nH > 0.0, nC_off / nH, 0.0) if nC_off is not None else None
        X_C_on = np.where(nH > 0.0, nC_on / nH, 0.0) if nC_on is not None else None
        X_e_off = np.where(nH > 0.0, ne_off / nH, 0.0) if ne_off is not None else None
        X_e_on = np.where(nH > 0.0, ne_on / nH, 0.0) if ne_on is not None else None

    # Build plot list: (key, title, off_arr, on_arr, unit_label, log10, cmap)
    # Only include plots where both off and on arrays are available.
    plots: list[tuple] = []

    def _add(key, title, off_arr, on_arr, unit_label, log10, cmap="inferno"):
        if off_arr is not None and on_arr is not None:
            plots.append((key, title, mid(off_arr), mid(on_arr), unit_label, log10, cmap))

    _add("chi_eff", "chi_eff (effective UV)", chi_eff_off, chi_eff_on, "log10", True)
    _add("theta_co", "theta_co (CO shielding)", theta_off, theta_on, "dimensionless", False, "viridis")
    _add("Xco", "Xco (CO gas abundance)", Xco_off, Xco_on, "log10", True)
    _add("nco_gas", "nCO_gas", nco_off, nco_on, "log10(cm^-3)", True)
    _add("nco_ice", "nCO_ice", nco_ice_off, nco_ice_on, "log10(cm^-3)", True)
    _add("nCplus", "nC+", nCplus_off, nCplus_on, "log10(cm^-3)", True)
    _add("nC", "nC", nC_off, nC_on, "log10(cm^-3)", True)
    _add("ne", "ne", ne_off, ne_on, "log10(cm^-3)", True)
    _add("X_Cplus", "X(C+)", X_Cplus_off, X_Cplus_on, "log10", True)
    _add("X_C", "X(C)", X_C_off, X_C_on, "log10", True)
    _add("X_e", "X(e)", X_e_off, X_e_on, "log10", True)
    _add("nH2", "nH2", nH2_off, nH2_on, "log10(cm^-3)", True)
    _add("nHI", "nHI", nHI_off, nHI_on, "log10(cm^-3)", True)

    theta_h2_off = _get_field(chem_off, "theta_h2", "dimensionless")
    theta_h2_on = _get_field(chem_on, "theta_h2", "dimensionless")
    theta_c_off = _get_field(chem_off, "theta_c", "dimensionless")
    theta_c_on = _get_field(chem_on, "theta_c", "dimensionless")
    _add("theta_h2", "theta_h2 (H2 shielding)", theta_h2_off, theta_h2_on, "dimensionless", False, "viridis")
    _add("theta_c", "theta_c (C shielding)", theta_c_off, theta_c_on, "dimensionless", False, "viridis")

    k_diss_off = _get_field(chem_off, "k_diss_co", "1/s")
    k_diss_on = _get_field(chem_on, "k_diss_co", "1/s")
    tau_diss_off = _get_field(chem_off, "tau_diss_co", "s")
    tau_diss_on = _get_field(chem_on, "tau_diss_co", "s")
    _add("k_diss", "k_diss_co", k_diss_off, k_diss_on, "log10(1/s)", True)
    _add("tau_diss", "tau_diss_co", tau_diss_off, tau_diss_on, "log10(s)", True)

    if k_diss_off is not None and k_diss_on is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            t_diss_off = np.where(k_diss_off > 0.0, 1.0 / k_diss_off, np.nan)
            t_diss_on = np.where(k_diss_on > 0.0, 1.0 / k_diss_on, np.nan)
        _add("t_diss_from_k", "1/k_diss_co", t_diss_off, t_diss_on, "log10(s)", True)

        R_pd_off = k_diss_off * nco_off
        R_pd_on = k_diss_on * nco_on
        _add("R_pd_diss", "CO photodissociation rate: k_diss*nCO", R_pd_off, R_pd_on, "log10(cm^-3 s^-1)", True)

    k_pd_surf_off = _get_field(chem_off, "k_pd_surf", "1/s")
    k_pd_surf_on = _get_field(chem_on, "k_pd_surf", "1/s")
    _add("k_pd_surf", "k_pd_surf", k_pd_surf_off, k_pd_surf_on, "log10(1/s)", True)

    R_pd_s_off = _get_field(chem_off, "R_pd", "cm^-3/s")
    R_pd_s_on = _get_field(chem_on, "R_pd", "cm^-3/s")
    _add("R_pd_surf", "CO photodesorption R_pd", R_pd_s_off, R_pd_s_on, "log10(cm^-3 s^-1)", True)

    n_ice_max_off = _get_field(chem_off, "n_ice_act_max", "cm^-3")
    n_ice_max_on = _get_field(chem_on, "n_ice_act_max", "cm^-3")
    _add("n_ice_act_max", "n_ice_act_max", n_ice_max_off, n_ice_max_on, "log10(cm^-3)", True)

    n_ice_off = _get_field(chem_off, "n_ice_act", "cm^-3")
    n_ice_on = _get_field(chem_on, "n_ice_act", "cm^-3")
    _add("n_ice_act", "n_ice_act", n_ice_off, n_ice_on, "log10(cm^-3)", True)

    for key, title, off2d, on2d, unit_label, log10, cmap in plots:
        _compare_midplane(
            out_path=out_dir / f"{network}_compare_{key}.png",
            title=f"{network}: {title}",
            off=off2d,
            on=on2d,
            unit_label=unit_label,
            log10=bool(log10),
            cmap=str(cmap),
            ratio_cmap="RdBu_r",
        )


def _summary_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "p10": float(np.nanpercentile(x, 10.0)),
        "p50": float(np.nanpercentile(x, 50.0)),
        "p90": float(np.nanpercentile(x, 90.0)),
    }


def _load_cfg_from_inputs(out_dir: Path) -> CubeTestConfig:
    inputs_path = Path(out_dir) / "inputs.json"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Missing inputs.json at {inputs_path}")
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    cfg_in = inputs.get("config")
    if not isinstance(cfg_in, dict):
        raise ValueError("inputs.json missing config dict")

    valid = {f.name for f in fields(CubeTestConfig)}
    kwargs = {k: v for k, v in cfg_in.items() if k in valid}
    return CubeTestConfig(**kwargs)


def _read_dust_density_bin0(
    *,
    dust_density_binp: Path,
    shape: tuple[int, int, int],
) -> np.ndarray:
    dust_density_binp = Path(dust_density_binp)
    if not dust_density_binp.exists():
        raise FileNotFoundError(f"Missing dust density file: {dust_density_binp}")

    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    ncells = nx * ny * nz

    with open(dust_density_binp, "rb") as f:
        header = np.fromfile(f, dtype=np.int64, count=4)
        if int(header.size) != 4:
            raise ValueError(f"Invalid dust density header in {dust_density_binp}")
        iformat, precision, ncells_in, nbin = (int(header[0]), int(header[1]), int(header[2]), int(header[3]))
        if iformat != 1:
            raise ValueError(f"Unexpected dust density format number {iformat}")
        if precision != 8:
            raise ValueError(f"Unexpected dust density precision {precision} (expected 8)")
        if int(ncells_in) != int(ncells):
            raise ValueError(f"Dust density ncells mismatch: file={int(ncells_in)} expected={int(ncells)}")
        if int(nbin) < 1:
            raise ValueError(f"Dust density nbin must be >= 1, got {int(nbin)}")

        rho0_flat = np.fromfile(f, dtype=np.float64, count=int(ncells))
        if int(rho0_flat.size) != int(ncells):
            raise ValueError("Dust density bin0 data truncated")

    rho0 = rho0_flat.reshape((nx, ny, nz), order="F")
    return np.asarray(rho0, dtype=float)


def _rho_from_dust_bin0(
    *,
    rho_dust_bin0_g_cm3: np.ndarray,
    params,
) -> np.ndarray:
    distribution = DustDistribution(
        amin=params.amin,
        amax=params.amax,
        nbin=int(params.nbins),
        power_index=float(params.pindex),
        grain_density=params.grain_density,
    )
    f0 = float(distribution.mass_fractions[0])
    if f0 <= 0.0:
        raise ValueError("Dust bin0 mass fraction is non-positive")
    d2g = float(params.dust_to_gas_ratio)
    if d2g <= 0.0:
        raise ValueError("dust_to_gas_ratio must be > 0")
    return np.asarray(rho_dust_bin0_g_cm3, dtype=float) / (d2g * f0)


def _load_chi_from_outputs(*, rad: RadModel, case_dir: Path, params) -> Quantity:
    outputs_dir = Path(case_dir) / "radmc3d_outputs"
    mean_path = find_cached_output(outputs_dir, ["mean_intensity.bout"])
    if mean_path is None:
        raise FileNotFoundError(
            f"Missing mean_intensity.bout in {outputs_dir}; cannot plot without rerunning RADMC-3D"
        )
    return rad._postprocess_chi(mean_path, params.uv_min, params.uv_max)


def _load_case(mesh: Mesh, case_dir: Path, cfg: CubeTestConfig):
    """Load a saved case: read density, build model+dust, create RadModel, load chi.

    Returns (rad, rho, chi, new_params).  Caller must wrap in _with_params().
    """
    new_params = diskbridge.read_params(
        _write_radmc3d_params_txt(model_dir=case_dir, cfg=cfg)
    )
    diskbridge.params = new_params
    diskbridge._params_module.params = new_params

    rho_d0 = _read_dust_density_bin0(
        dust_density_binp=case_dir / "radmc3d_inputs" / "dust_density.binp",
        shape=mesh.shape,
    )
    rho = _rho_from_dust_bin0(rho_dust_bin0_g_cm3=rho_d0, params=new_params)

    model = _build_model_with_density(mesh, rho)
    if model.dust is None:
        model.dust = Dust(model)
    model.dust.set_distribution(
        nbin=int(new_params.nbins),
        dust_to_gas_ratio=float(new_params.dust_to_gas_ratio),
        mode="proportional",
    )

    rad = RadModel(model, model_dir=case_dir)
    rad.params = new_params
    rad.writer.params = new_params

    chi = _load_chi_from_outputs(rad=rad, case_dir=case_dir, params=new_params)
    return rad, rho, chi, new_params


def _iter_cases(out_dir: Path, cfg: CubeTestConfig):
    """Yield (case_dir, rad, rho, chi) for each rho0_* subdirectory, restoring params on exit."""
    mesh = _build_cube_mesh(cfg)
    case_dirs = sorted([p for p in out_dir.glob("rho0_*") if p.is_dir()])
    if not case_dirs:
        raise FileNotFoundError(f"No rho0_* case directories found under {out_dir}")

    for case_dir in case_dirs:
        prev_params = diskbridge.params
        prev_module_params = diskbridge._params_module.params
        try:
            rad, rho, chi, _ = _load_case(mesh, case_dir, cfg)
            yield case_dir, rad, rho, chi
        finally:
            diskbridge.params = prev_params
            diskbridge._params_module.params = prev_module_params


def _build_chemistry_config(
    cfg: CubeTestConfig,
    *,
    skip_shielding: bool,
) -> dict:
    """Build the GOW17 chemistry config dict."""
    base = {
        "skip_shielding": bool(skip_shielding),
        "nside": int(cfg.nside),
        "b_kms": float(cfg.b_kms),
    }
    if skip_shielding:
        base["shielding_iter"] = 0
    else:
        base["shielding_iter"] = int(cfg.shielding_iter)

    base["shielding_max_iter"] = int(cfg.shielding_iter) if not skip_shielding else 1
    return base


def run_plots_only(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    err_path = out_dir / "error.png"
    if err_path.exists():
        err_path.unlink()
    cfg = _load_cfg_from_inputs(out_dir)
    network = "gow17"

    for case_dir, rad, rho, chi in _iter_cases(out_dir, cfg):
        print(f"[{case_dir.name}] Computing UV weights ...")
        _compute_and_attach_uv_weights(rad=rad, cfg=cfg, chi=chi)

        print(f"[{case_dir.name}] Running {network} shielding off ...")
        chem_off = run_chemistry(
            rad, model=network,
            config=_build_chemistry_config(cfg, skip_shielding=True),
            write=False,
        )
        print(f"[{case_dir.name}] Running {network} shielding on ...")
        chem_on = run_chemistry(
            rad,
            model=network,
            config=_build_chemistry_config(cfg, skip_shielding=False),
            write=False,
        )

        _write_chemistry_compare_plots(
            out_dir=case_dir, rad=rad,
            chem_off=chem_off, chem_on=chem_on, network=network,
        )

        chi_arr = np.asarray(chi.to("dimensionless").magnitude, dtype=float)
        chi_eff_on_arr = np.asarray(
            chem_on.fields["chi_eff"].to("dimensionless").magnitude, dtype=float
        )
        theta_co_on = _safe_ratio(chi_eff_on_arr, chi_arr)

        _write_cube_plots(
            out_dir=case_dir, cfg=cfg, sigma_eff=float("nan"),
            rho_g_cm3=rho, chi=chi,
            chi_eff_off=chem_off.fields["chi_eff"],
            chi_eff_on=chem_on.fields["chi_eff"],
            theta_co_off=chem_off.fields["theta_co"],
            theta_co_on=theta_co_on,
            Xco_off=chem_off.abundances["co"],
            Xco_on=chem_on.abundances["co"],
            nco_off=chem_off.number_densities["co"],
            nco_on=chem_on.number_densities["co"],
            network=network,
        )


def run(out_dir: Path) -> None:
    """Run the full cube test: build density, run RADMC-3D, run chemistry, plot.

    Parameters
    ----------
    out_dir : Path
        Root output directory.
    """
    network = "gow17"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    err_path = out_dir / "error.png"
    if err_path.exists():
        err_path.unlink()

    cfg = CubeTestConfig()
    _write_inputs(out_dir, cfg)

    error: str | None = None
    summary: dict = {}

    try:
        mesh = _build_cube_mesh(cfg)
        rng = np.random.default_rng(int(cfg.seed))
        s = _make_correlated_gaussian_field(
            shape=mesh.shape,
            p=float(cfg.p),
            k_min=float(cfg.k_min) * float(cfg.k_multiplier),
            k_max=float(cfg.k_max) * float(cfg.k_multiplier),
            rng=rng,
        )

        sigma_eff = _sigma_for_peak_boost(
            s=s,
            sigma0=float(cfg.sigma_s),
            boost=float(cfg.peak_boost),
        )

        cases: list[dict] = []

        for rho0 in cfg.rho0_g_cm3_list:
            case_dir = out_dir / ("rho0_%0.3e" % float(rho0))
            case_dir.mkdir(parents=True, exist_ok=True)

            rho = _make_lognormal_density(rho0_g_cm3=float(rho0), sigma_s=float(sigma_eff), s=s)
            peak_ratio = float(np.nanmax(rho) / float(rho0))

            model = _build_model_with_density(mesh, rho)
            _ensure_dust_setup(cfg, model)
            model.validate_canonical_axis_orders(include_dust=False)

            rad = RadModel(model, model_dir=case_dir)

            _run_radmc3d_external_uv(cfg=cfg, rad=rad, model_dir=case_dir)

            chi = rad.ensure_chi(force=False)

            # Compute and attach directional UV weights.
            print(f"[{case_dir.name}] Computing UV weights ...")
            _compute_and_attach_uv_weights(rad=rad, cfg=cfg, chi=chi)

            print(f"[{case_dir.name}] Running {network} shielding off ...")
            chem_off = run_chemistry(
                rad,
                model=network,
                config=_build_chemistry_config(cfg, skip_shielding=True),
                write=False,
            )
            Xco_off = chem_off.abundances["co"]
            chi_eff_off = chem_off.fields["chi_eff"]
            theta_co_off = chem_off.fields["theta_co"]
            nco_off = chem_off.number_densities["co"]

            print(f"[{case_dir.name}] Running {network} shielding on ...")
            chem_on = run_chemistry(
                rad,
                model=network,
                config=_build_chemistry_config(cfg, skip_shielding=False),
                write=False,
            )
            Xco_on = chem_on.abundances["co"]
            chi_eff_on = chem_on.fields["chi_eff"]
            theta_co_on = chem_on.fields["theta_co"]
            nco_on = chem_on.number_densities["co"]

            _write_chemistry_compare_plots(
                out_dir=case_dir,
                rad=rad,
                chem_off=chem_off,
                chem_on=chem_on,
                network=network,
            )

            # Diagnostics
            Xco_off_arr = np.asarray(Xco_off.to("dimensionless").magnitude, dtype=float)
            Xco_on_arr = np.asarray(Xco_on.to("dimensionless").magnitude, dtype=float)
            nco_off_arr = np.asarray(nco_off.to("cm^-3").magnitude, dtype=float)
            nco_on_arr = np.asarray(nco_on.to("cm^-3").magnitude, dtype=float)
            ratio = np.divide(
                Xco_on_arr,
                Xco_off_arr,
                out=np.full_like(Xco_on_arr, np.nan, dtype=float),
                where=(Xco_off_arr > 0.0),
            )

            nco_ratio = np.divide(
                nco_on_arr,
                nco_off_arr,
                out=np.full_like(nco_on_arr, np.nan, dtype=float),
                where=(nco_off_arr > 0.0),
            )

            frac_gt_1p1 = float(np.mean(ratio.reshape(-1) > 1.1))
            frac_gt_2 = float(np.mean(ratio.reshape(-1) > 2.0))

            chi_arr = np.asarray(chi.to("dimensionless").magnitude, dtype=float)
            chi_eff_on_arr = np.asarray(chi_eff_on.to("dimensionless").magnitude, dtype=float)
            theta_co_on_derived = np.divide(
                chi_eff_on_arr,
                chi_arr,
                out=np.full_like(chi_arr, np.nan, dtype=float),
                where=(chi_arr > 0.0),
            )

            _write_cube_plots(
                out_dir=case_dir,
                cfg=cfg,
                sigma_eff=float(sigma_eff),
                rho_g_cm3=rho,
                chi=chi,
                chi_eff_off=chi_eff_off,
                chi_eff_on=chi_eff_on,
                theta_co_off=theta_co_off,
                theta_co_on=theta_co_on_derived,
                Xco_off=Xco_off,
                Xco_on=Xco_on,
                nco_off=nco_off,
                nco_on=nco_on,
                network=network,
            )

            case_summary = {
                "rho0_g_cm3": float(rho0),
                "network": str(network),
                "sigma_eff": float(sigma_eff),
                "peak_ratio_rho_over_rho0": peak_ratio,
                "rho_stats": _summary_stats(rho.reshape(-1)),
                "chi_stats": _summary_stats(chi_arr.reshape(-1)),
                "Xco_off_stats": _summary_stats(Xco_off_arr.reshape(-1)),
                "Xco_on_stats": _summary_stats(Xco_on_arr.reshape(-1)),
                "Xco_ratio_stats": _summary_stats(ratio.reshape(-1)),
                "nCO_off_stats": _summary_stats(nco_off_arr.reshape(-1)),
                "nCO_on_stats": _summary_stats(nco_on_arr.reshape(-1)),
                "nCO_ratio_stats": _summary_stats(nco_ratio.reshape(-1)),
                "theta_co_on_stats": _summary_stats(theta_co_on_derived.reshape(-1)),
                "volume_fractions": {
                    "Xco_ratio_gt_1p1": frac_gt_1p1,
                    "Xco_ratio_gt_2": frac_gt_2,
                },
            }

            (case_dir / f"{network}_summary.json").write_text(
                json.dumps(case_summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            cases.append(case_summary)

        summary = {
            "job": "cube_test",
            "network": str(network),
            "time": datetime.now().isoformat(timespec="seconds"),
            "config": cfg.__dict__,
            "pass": True,
            "cases": cases,
        }

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        plt = _setup_matplotlib()
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.axis("off")
        ax.set_title(f"cube_test ({network}) failed")
        ax.text(0.01, 0.99, error, ha="left", va="top")
        fig.tight_layout()
        fig.savefig(out_dir / "error.png", dpi=150)
        plt.close(fig)
        summary = {
            "job": "cube_test",
            "network": str(network),
            "time": datetime.now().isoformat(timespec="seconds"),
            "config": cfg.__dict__,
            "pass": False,
            "error": error,
        }

    (out_dir / f"{network}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    if error is not None:
        raise RuntimeError(error)


def main() -> int:
    parser = argparse.ArgumentParser(prog="cube_test")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    try:
        if bool(args.plots_only):
            run_plots_only(out_dir)
        else:
            run(out_dir)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
