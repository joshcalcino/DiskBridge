"""Tests for the DiskBridge 3D HEALPix non-LTE external-population pipeline."""

# db-keywords: healpix-columns, nonlte, line-transfer, validation, radmc3d, field, coordinates, ray-tracing
# db-role: validation
# db-scope: test
# db-purpose: Tests for the DiskBridge 3D HEALPix non-LTE external-population pipeline.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diskbridge._constants import H_PLANCK, K_B
from diskbridge.radmc3d.colliders import INSTALLED_LAMDA_DIR
from diskbridge.radmc3d.line_transfer.escape_healpix import (
    _beta_of_tau,
    compute_beta_cartesian,
)
from diskbridge.radmc3d.line_transfer.external_validation import (
    validate_external_population_run,
)
from diskbridge.radmc3d.writer import RadWriter
from diskbridge.utils import sha256_file
from diskbridge.radmc3d.line_transfer.molecular_rates import (
    lte_populations,
    parse_lamda_molecule_file,
    stack_collider_tables,
)
from diskbridge.radmc3d.line_transfer.se_solver import (
    build_rate_matrix_and_rhs,
    compute_collisional_rates,
    compute_line_center_opacity,
    solve_statistical_equilibrium,
)


_CO_LAMDA = INSTALLED_LAMDA_DIR / "co.dat"


# ---------------------------------------------------------------------------
# LAMDA parser
# ---------------------------------------------------------------------------


def test_lamda_parser_co_levels_and_lines():
    mol = parse_lamda_molecule_file(_CO_LAMDA)
    assert mol.name.lower().startswith("co")
    assert mol.nlev > 10
    assert mol.nlin == mol.nlev - 1 or mol.nlin > 0
    # Lowest level energy is zero.
    assert mol.energy_erg[0] == pytest.approx(0.0, abs=1e-30)
    # Weights for CO rotational levels are 2J+1 -> integer odd.
    assert np.all(mol.weight > 0)
    # iup/ilow are 0-indexed and within bounds.
    assert int(mol.iup.max()) < mol.nlev
    assert int(mol.ilow.min()) >= 0
    # Aud positive, freq positive.
    assert np.all(mol.aud > 0)
    assert np.all(mol.freq_hz > 0)
    # Colliders parsed.
    assert len(mol.collider_names) >= 1
    assert all(t.size > 0 for t in mol.collider_tgrid_K)
    for tab in mol.collider_down_rates:
        # Upper triangle (u > l): nonzero somewhere.
        assert tab.shape[1] == mol.nlev
        assert np.any(tab > 0)


# ---------------------------------------------------------------------------
# LTE / Boltzmann
# ---------------------------------------------------------------------------


def test_lte_populations_sum_to_one_and_boltzmann_ratio():
    mol = parse_lamda_molecule_file(_CO_LAMDA)
    T = np.array([5.0, 30.0, 200.0])
    f = lte_populations(mol, T)
    sums = f.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-12)
    # Excited / ground ratio matches Boltzmann formula.
    for i, t in enumerate(T):
        ratio_expect = (
            mol.weight[1] / mol.weight[0]
            * np.exp(-(mol.energy_erg[1] - mol.energy_erg[0]) / (K_B * t))
        )
        assert f[i, 1] / f[i, 0] == pytest.approx(ratio_expect, rel=1e-10)


# ---------------------------------------------------------------------------
# beta(tau)
# ---------------------------------------------------------------------------


def test_beta_of_tau_limits():
    # Very small tau: beta ~ 1.
    assert _beta_of_tau(1.0e-8) == pytest.approx(1.0, rel=1e-7)
    # Large tau: beta ~ 1/tau.
    tau = 100.0
    assert _beta_of_tau(tau) == pytest.approx(1.0 / tau, rel=1e-5)
    # Series matches direct formula at the transition.
    tau = 1.0e-5
    direct = (1.0 - np.exp(-tau)) / tau
    assert _beta_of_tau(tau) == pytest.approx(direct, rel=1e-8)


# ---------------------------------------------------------------------------
# Cartesian HEALPix kernel: optically thin -> beta == 1
# ---------------------------------------------------------------------------


def _healpix_dirs_nside(nside: int) -> np.ndarray:
    import healpy as hp

    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    return np.ascontiguousarray(np.vstack([x, y, z]).T, dtype=np.float64)


def test_cartesian_beta_optically_thin_returns_one():
    nx = ny = nz = 6
    x_edges = np.linspace(-1.0, 1.0, nx + 1)
    y_edges = np.linspace(-1.0, 1.0, ny + 1)
    z_edges = np.linspace(-1.0, 1.0, nz + 1)
    nlin = 2
    alpha0 = np.zeros((nlin, nx, ny, nz), dtype=np.float64)  # optically thin
    velocity = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    a_line = np.full((nx, ny, nz), 1.0e5, dtype=np.float64)
    dirs = _healpix_dirs_nside(1)  # 12 pixels
    # One candidate cell at center.
    ix = nx // 2
    iy = ny // 2
    iz = nz // 2
    candidate_idx = np.array([[ix, iy, iz]], dtype=np.int64)
    cx = 0.5 * (x_edges[ix] + x_edges[ix + 1])
    cy = 0.5 * (y_edges[iy] + y_edges[iy + 1])
    cz = 0.5 * (z_edges[iz] + z_edges[iz + 1])
    cell_centers = np.array([[cx, cy, cz]], dtype=np.float64)
    beta = compute_beta_cartesian(
        candidate_idx,
        cell_centers,
        dirs,
        alpha0,
        velocity,
        a_line,
        x_edges,
        y_edges,
        z_edges,
        max_ray_steps=200,
    )
    np.testing.assert_allclose(beta, 1.0, rtol=1e-12)


def test_cartesian_beta_thick_uniform_decreases_with_alpha0():
    nx = ny = nz = 8
    x_edges = np.linspace(-1.0, 1.0, nx + 1)
    y_edges = np.linspace(-1.0, 1.0, ny + 1)
    z_edges = np.linspace(-1.0, 1.0, nz + 1)
    nlin = 1
    velocity = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    a_line = np.full((nx, ny, nz), 1.0e5, dtype=np.float64)
    dirs = _healpix_dirs_nside(1)
    ix, iy, iz = nx // 2, ny // 2, nz // 2
    candidate_idx = np.array([[ix, iy, iz]], dtype=np.int64)
    cx = 0.5 * (x_edges[ix] + x_edges[ix + 1])
    cy = 0.5 * (y_edges[iy] + y_edges[iy + 1])
    cz = 0.5 * (z_edges[iz] + z_edges[iz + 1])
    cell_centers = np.array([[cx, cy, cz]], dtype=np.float64)

    betas = []
    for alpha in [0.0, 0.1, 1.0, 10.0]:
        alpha0 = np.full((nlin, nx, ny, nz), alpha, dtype=np.float64)
        b = compute_beta_cartesian(
            candidate_idx, cell_centers, dirs, alpha0, velocity, a_line,
            x_edges, y_edges, z_edges, max_ray_steps=400,
        )
        betas.append(float(b[0, 0]))
    # Strictly decreasing with alpha0.
    assert betas[0] == pytest.approx(1.0, rel=1e-12)
    assert betas[0] > betas[1] > betas[2] > betas[3]
    assert betas[3] < 0.5  # very optically thick


# ---------------------------------------------------------------------------
# Statistical equilibrium: LTE limit at very high collider density
# ---------------------------------------------------------------------------


def _toy_two_level_molecule(nlev=2, A_ul=1.0e-5, gu=3.0, gl=1.0, dE_K=10.0):
    """Build a minimal MoleculeData for a 2-level system."""
    from diskbridge.radmc3d.line_transfer.molecular_rates import MoleculeData

    energy = np.array([0.0, dE_K * K_B], dtype=np.float64)
    weight = np.array([gl, gu], dtype=np.float64)
    iup = np.array([1], dtype=np.int64)
    ilow = np.array([0], dtype=np.int64)
    aud = np.array([A_ul], dtype=np.float64)
    freq = np.array([(energy[1] - energy[0]) / H_PLANCK], dtype=np.float64)
    tgrid = np.array([5.0, 10.0, 50.0, 200.0], dtype=np.float64)
    nt = tgrid.size
    table = np.zeros((nt, nlev, nlev), dtype=np.float64)
    table[:, 1, 0] = 1.0e-10  # downward collision rate (cm^3 s^-1)
    return MoleculeData(
        name="toy",
        molweight=28.0,
        nlev=nlev,
        nlin=1,
        energy_erg=energy,
        weight=weight,
        iup=iup,
        ilow=ilow,
        aud=aud,
        freq_hz=freq,
        collider_names=("p-h2",),
        collider_tgrid_K=(tgrid,),
        collider_down_rates=(table,),
    )


def test_se_high_collider_density_recovers_lte():
    mol = _toy_two_level_molecule()
    T = np.array([30.0], dtype=np.float64)
    a_line = np.array([1.0e5], dtype=np.float64)
    n_sp = np.array([1.0e10], dtype=np.float64)
    n_coll = np.array([[1.0e14]], dtype=np.float64)  # very dense collider
    beta = np.array([[1.0]], dtype=np.float64)  # optically thin
    tgrids, ntemps, tables = stack_collider_tables(mol)
    f = solve_statistical_equilibrium(
        molecule=mol,
        Tgas_cand=T,
        n_species_cand=n_sp,
        a_line_cand=a_line,
        collider_dens_cand=n_coll,
        beta=beta,
        tbg_K=2.7255,
        tgrids=tgrids,
        ntemps=ntemps,
        tables=tables,
    )
    lte = lte_populations(mol, T)
    np.testing.assert_allclose(f, lte, rtol=1e-3)


def test_se_negligible_collisions_cmb_only_limit():
    """With negligible collisions and only CMB background, populations
    approach radiative equilibrium with Tbg."""
    Tbg = 30.0
    mol = _toy_two_level_molecule(A_ul=1.0e-8, dE_K=5.0)
    T = np.array([10.0], dtype=np.float64)
    a_line = np.array([1.0e5], dtype=np.float64)
    n_sp = np.array([1.0e-3], dtype=np.float64)
    n_coll = np.array([[1.0e-30]], dtype=np.float64)
    beta = np.array([[1.0]], dtype=np.float64)
    tgrids, ntemps, tables = stack_collider_tables(mol)
    f = solve_statistical_equilibrium(
        molecule=mol,
        Tgas_cand=T,
        n_species_cand=n_sp,
        a_line_cand=a_line,
        collider_dens_cand=n_coll,
        beta=beta,
        tbg_K=Tbg,
        tgrids=tgrids,
        ntemps=ntemps,
        tables=tables,
    )
    rad_ratio_expect = (
        mol.weight[1] / mol.weight[0]
        * np.exp(-(mol.energy_erg[1] - mol.energy_erg[0]) / (K_B * Tbg))
    )
    ratio = f[0, 1] / f[0, 0]
    assert ratio == pytest.approx(rad_ratio_expect, rel=1e-3)


# ---------------------------------------------------------------------------
# level-population writer
# ---------------------------------------------------------------------------


def test_write_levelpop_dat_format(tmp_path: Path):
    n_cells, n_levels = 4, 3
    pop = np.array(
        [
            [1.0e3, 2.0e2, 3.0e1],
            [4.0e3, 5.0e2, 6.0e1],
            [7.0e3, 8.0e2, 9.0e1],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    path = RadWriter.write_levelpop_dat(
        tmp_path / "levelpop_co.dat",
        levelpop_cm3_radmc_order=pop,
        level_numbers_1based=np.array([1, 2, 3]),
    )
    text = path.read_text().splitlines()
    assert text[0] == "1"
    assert text[1] == "4"
    assert text[2] == "3"
    assert text[3].split() == ["1", "2", "3"]
    # 4 data rows.
    assert len(text) >= 3 + 1 + n_cells
    row0 = [float(x) for x in text[4].split()]
    assert row0 == pytest.approx([1.0e3, 2.0e2, 3.0e1])


def test_write_levelpop_dat_rejects_negative(tmp_path: Path):
    pop = np.array([[1.0, -1.0e-10]])
    with pytest.raises(ValueError, match="negative"):
        RadWriter.write_levelpop_dat(
            tmp_path / "x.dat",
            levelpop_cm3_radmc_order=pop,
            level_numbers_1based=np.array([1, 2]),
        )


# ---------------------------------------------------------------------------
# Validation rejects malformed staged dirs
# ---------------------------------------------------------------------------


def _write_amr_grid(path: Path, nr=2, ntheta=1, nphi=1):
    path.write_text(
        "\n".join(
            [
                "1",
                "0",
                "101",
                "0",
                "1 1 1",
                f"{nr} {ntheta} {nphi}",
                "1.0 2.0 3.0",
                "0.0 1.0",
                "0.0 6.283185307179586",
            ]
        )
        + "\n"
    )


def _write_scalar_binp(path: Path, values):
    with path.open("wb") as f:
        np.asarray([1, 8, len(values)], dtype=np.int64).tofile(f)
        np.asarray(values, dtype=np.float64).tofile(f)


def _write_gas_velocity_binp(path: Path, n: int):
    with path.open("wb") as f:
        np.asarray([1, 8, n], dtype=np.int64).tofile(f)
        np.zeros(3 * n, dtype=np.float64).tofile(f)


def _make_min_staged_dir(tmp_path: Path, species="co", n=2, pop=None) -> Path:
    work = tmp_path / "run"
    inp = work / "radmc3d_inputs"
    inp.mkdir(parents=True)
    _write_amr_grid(inp / "amr_grid.inp", nr=n)
    (inp / "wavelength_micron.inp").write_text("1\n1000.0\n")
    (inp / "radmc3d.inp").write_text(
        "incl_lines = 1\nlines_mode = 50\ntgas_eq_tdust = 0\n"
    )
    (inp / "lines.inp").write_text(f"2\n1\n{species}    leiden    0    0    0\n")
    molecule = inp / f"molecule_{species}.inp"
    molecule.write_text("!stub\n")
    (inp / f"external_levelpop_manifest_{species}.json").write_text(
        json.dumps(
            {
                "species": species,
                "converged": True,
                "allow_unconverged": False,
                "molecule_sha256": sha256_file(molecule),
            }
        )
        + "\n"
    )
    _write_scalar_binp(inp / f"numberdens_{species}.binp", [1.0e5, 2.0e5])
    _write_scalar_binp(inp / "gas_temperature.binp", [20.0, 30.0])
    _write_scalar_binp(inp / "microturbulence.binp", [1.0e4, 1.0e4])
    _write_gas_velocity_binp(inp / "gas_velocity.binp", n)
    if pop is None:
        pop = np.array(
            [
                [1.0e5, 0.0, 0.0],
                [2.0e5, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    RadWriter.write_levelpop_dat(
        inp / f"levelpop_{species}.dat",
        levelpop_cm3_radmc_order=pop,
        level_numbers_1based=np.array([1, 2, 3]),
    )
    return work


def test_validation_accepts_well_formed_dir(tmp_path: Path):
    work = _make_min_staged_dir(tmp_path)
    info = validate_external_population_run(work, species="co")
    assert info["line_mode"] == 50
    assert info["species"] == "co"
    assert info["n_levels"] == 3


def test_validation_rejects_lines_mode_mismatch(tmp_path: Path):
    work = _make_min_staged_dir(tmp_path)
    (work / "radmc3d_inputs" / "radmc3d.inp").write_text(
        "incl_lines = 1\nlines_mode = 3\ntgas_eq_tdust = 0\n"
    )
    with pytest.raises(ValueError, match="lines_mode"):
        validate_external_population_run(work, species="co")


def test_validation_rejects_population_sum_mismatch(tmp_path: Path):
    bad_pop = np.array(
        [
            [1.0e5, 0.0, 0.0],
            [9.0e5, 0.0, 0.0],   # n_species says 2e5
        ],
        dtype=np.float64,
    )
    work = _make_min_staged_dir(tmp_path, pop=bad_pop)
    with pytest.raises(ValueError, match="levelpop sum"):
        validate_external_population_run(work, species="co")


def test_validation_rejects_missing_microturbulence(tmp_path: Path):
    work = _make_min_staged_dir(tmp_path)
    (work / "radmc3d_inputs" / "microturbulence.binp").unlink()
    with pytest.raises(FileNotFoundError, match="microturbulence"):
        validate_external_population_run(work, species="co")


def test_validation_rejects_nonfinite_gas_velocity(tmp_path: Path):
    work = _make_min_staged_dir(tmp_path)
    with (work / "radmc3d_inputs" / "gas_velocity.binp").open("wb") as f:
        np.asarray([1, 8, 2], dtype=np.int64).tofile(f)
        np.asarray([0.0, 0.0, np.nan, 0.0, 0.0, 0.0], dtype=np.float64).tofile(f)
    with pytest.raises(ValueError, match="gas_velocity.*non-finite"):
        validate_external_population_run(work, species="co")


def test_validation_rejects_negative_population(tmp_path: Path):
    work = _make_min_staged_dir(tmp_path)
    # Manually write a bad file.
    (work / "radmc3d_inputs" / "levelpop_co.dat").write_text(
        "1\n2\n3\n1 2 3\n1.0e5 -1.0 0.0\n2.0e5 0.0 0.0\n"
    )
    with pytest.raises(ValueError, match="negative"):
        validate_external_population_run(work, species="co")


# ---------------------------------------------------------------------------
# Collision-table temperature endpoints
# ---------------------------------------------------------------------------


def test_collisional_rates_hold_nearest_temperature_endpoint():
    """Temperatures outside a table use its first or last downward rate."""

    temperatures = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    collider_densities = np.full((temperatures.size, 1), 2.0)
    tgrids = np.array([[10.0, 20.0]])
    ntemps = np.array([2], dtype=np.int64)
    tables = np.zeros((1, 2, 2, 2))
    tables[0, :, 1, 0] = [1.0, 3.0]

    rates = compute_collisional_rates(
        temperatures,
        collider_densities,
        tgrids,
        ntemps,
        tables,
        np.array([1.0, 3.0]),
        np.array([0.0, 1.0e-15]),
    )

    np.testing.assert_allclose(rates[:, 1, 0], [2.0, 2.0, 4.0, 6.0, 6.0])
