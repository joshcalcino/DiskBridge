"""End-to-end smoke test for the HEALPix non-LTE external-population driver."""

# db-keywords: healpix-columns, nonlte, line-transfer, gas-temperature, validation, config, units, radmc3d, model, mesh, field
# db-role: validation
# db-scope: test
# db-purpose: End-to-end smoke test for the HEALPix non-LTE external-population driver.

from __future__ import annotations

from dataclasses import replace
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from diskbridge._units import Quantity, units
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.line_transfer import (
    HealpixSEConfig,
    prepare_external_population_line_run,
    validate_external_population_run,
    solve_and_write_healpix_levelpop,
)
from diskbridge.radmc3d.line_transfer.molecular_rates import (
    lte_populations,
    parse_lamda_molecule_file,
)
from diskbridge.radmc3d.writer import RadWriter


_REPO = Path(__file__).resolve().parents[2]
_CO_LAMDA = _REPO / "src" / "diskbridge" / "data" / "moldata" / "co.dat"


@pytest.fixture(autouse=True)
def canonical_installed_lamda_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, Path]]:
    """Record use of the canonical installed molecule-data path."""

    calls: list[tuple[str, Path]] = []

    def install(species: str, inputs_dir: str | Path) -> Path:
        species = str(species).lower().strip()
        assert species == "co"
        destination = Path(inputs_dir) / f"molecule_{species}.inp"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_CO_LAMDA, destination)
        calls.append((species, destination))
        return destination

    monkeypatch.setattr(
        "diskbridge.radmc3d.line_transfer.external_populations."
        "install_validated_molecule_file",
        install,
    )
    monkeypatch.setattr(
        "diskbridge.radmc3d.line_transfer.staging.install_validated_molecule_file",
        install,
    )
    return calls


class _FakeGas:
    def __init__(self):
        self._fields: dict[str, Field] = {}

    def register(self, name, field):
        self._fields[name] = field

    def __getitem__(self, name):
        return self._fields[name]

    def __contains__(self, name):
        return name in self._fields


class _FakeModel:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.gas = _FakeGas()


def _add_field(model: _FakeModel, name: str, arr: np.ndarray, unit: str) -> None:
    quantity = Quantity(np.asarray(arr, dtype=np.float64), unit)
    model.gas.register(
        name,
        Field(
            quantity=name,
            data=quantity,
            axis_order=model.mesh.axis_names(),
        ),
    )


def _make_cartesian_rad(tmp_path: Path, n: int = 4):
    L = 1.0e16  # cm (~ 0.003 pc)
    edges = np.linspace(-L, L, n + 1)
    edges_q = edges * units("cm")
    mesh = Mesh.cartesian(
        x=Axis(edges=edges_q),
        y=Axis(edges=edges_q),
        z=Axis(edges=edges_q),
    )
    model = _FakeModel(mesh)
    shape = (n, n, n)
    Tgas = np.full(shape, 30.0, dtype=np.float64)
    _add_field(model, "gas_temperature", Tgas, "K")
    _add_field(model, "microturbulence", np.full(shape, 1.0e4), "cm/s")
    _add_field(model, "vx", np.zeros(shape), "cm/s")
    _add_field(model, "vy", np.zeros(shape), "cm/s")
    _add_field(model, "vz", np.zeros(shape), "cm/s")
    rad = SimpleNamespace(model=model)
    return rad, shape


def _make_chemistry_result(shape, n_co: float = 1.0e3, n_h2: float = 1.0e8):
    nco = Quantity(np.full(shape, n_co, dtype=np.float64), "cm^-3")
    n_ph2 = Quantity(np.full(shape, 0.25 * n_h2, dtype=np.float64), "cm^-3")
    n_oh2 = Quantity(np.full(shape, 0.75 * n_h2, dtype=np.float64), "cm^-3")
    return SimpleNamespace(
        number_densities={"co": nco, "p-h2": n_ph2, "o-h2": n_oh2},
    )


def test_solve_and_write_healpix_levelpop_smoke(
    tmp_path: Path,
    canonical_installed_lamda_installer: list[tuple[str, Path]],
):
    rad, shape = _make_cartesian_rad(tmp_path, n=4)
    chem = _make_chemistry_result(shape, n_co=10.0)

    out_dir = tmp_path / "out"
    config = HealpixSEConfig(
        nside=1,           # 12 directions; fast
        maxiter=20,
        convcrit=1.0e-3,
        memory_budget_gib=1.0,
        max_ray_steps=200,
    )
    path = solve_and_write_healpix_levelpop(
        rad=rad,
        chemistry_result=chem,
        species="co",
        output_dir=out_dir,
        config=config,
    )
    assert path.exists()
    assert path.name == "levelpop_co.dat"
    assert canonical_installed_lamda_installer == [
        ("co", out_dir / "molecule_co.inp")
    ]

    # Parse and validate the file.
    tokens = path.read_text().split()
    assert tokens[0] == "1"
    n_cells = int(tokens[1])
    n_levels = int(tokens[2])
    assert n_cells == int(np.prod(shape))
    levels = [int(x) for x in tokens[3 : 3 + n_levels]]
    assert levels == list(range(1, n_levels + 1))
    data = np.asarray(tokens[3 + n_levels : 3 + n_levels + n_cells * n_levels], dtype=float)
    pop = data.reshape(n_cells, n_levels)
    # All non-negative, finite.
    assert np.all(np.isfinite(pop))
    assert np.all(pop >= 0.0)
    # Sum across levels matches the species density everywhere.
    np.testing.assert_allclose(pop.sum(axis=1), 10.0, rtol=1e-6)

    # Manifest written.
    manifest = out_dir / "external_levelpop_manifest_co.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["microturbulence_range_cm_s"] == pytest.approx([1.0e4, 1.0e4])
    assert payload["spherical_inner_boundary"] == "vacuum_cavity"
    assert payload["spherical_theta_boundary"] == "boundary_cell_to_rmax"
    assert payload["collision_temperature_policy"] == "nearest_table_boundary"
    checkpoint = json.loads(
        (out_dir / "checkpoints" / "checkpoint_latest_co.json").read_text()
    )
    assert checkpoint["fingerprint"]["spherical_inner_boundary"] == "vacuum_cavity"
    assert (
        checkpoint["fingerprint"]["spherical_theta_boundary"]
        == "boundary_cell_to_rmax"
    )
    assert (
        checkpoint["fingerprint"]["collision_temperature_policy"]
        == "nearest_table_boundary"
    )
    with pytest.raises(ValueError, match="does not match the current inputs"):
        solve_and_write_healpix_levelpop(
            rad=rad,
            chemistry_result=chem,
            species="co",
            output_dir=out_dir,
            config=replace(
                config,
                maxiter=config.maxiter + 1,
                resume_from_checkpoint=True,
                spherical_theta_boundary="vacuum",
            ),
        )


@pytest.mark.parametrize("floor", [-1.0, np.nan, np.inf])
def test_healpix_se_config_rejects_invalid_abundance_floor(floor: float):
    with pytest.raises(ValueError, match="species_abundance_floor"):
        HealpixSEConfig(species_abundance_floor=floor)


def test_abundance_floor_requires_species_abundance(tmp_path: Path):
    rad, shape = _make_cartesian_rad(tmp_path, n=2)
    chem = _make_chemistry_result(shape, n_co=10.0)

    with pytest.raises(KeyError, match=r"abundances\['co'\]"):
        solve_and_write_healpix_levelpop(
            rad=rad,
            chemistry_result=chem,
            species="co",
            output_dir=tmp_path / "missing_abundance",
            config=HealpixSEConfig(species_abundance_floor=1.0e-8),
        )


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("shape", "abundance shape"),
        ("nonfinite", "non-finite"),
        ("negative", "negative"),
    ],
)
def test_abundance_floor_rejects_invalid_species_abundance(
    tmp_path: Path,
    invalid_kind: str,
    message: str,
):
    rad, shape = _make_cartesian_rad(tmp_path, n=2)
    chem = _make_chemistry_result(shape, n_co=10.0)
    if invalid_kind == "shape":
        abundance = np.full((1, 1, 1), 1.0e-6)
    else:
        abundance = np.full(shape, 1.0e-6)
        abundance.flat[0] = np.nan if invalid_kind == "nonfinite" else -1.0
    chem.abundances = {"co": Quantity(abundance, "dimensionless")}

    with pytest.raises(ValueError, match=message):
        solve_and_write_healpix_levelpop(
            rad=rad,
            chemistry_result=chem,
            species="co",
            output_dir=tmp_path / invalid_kind,
            config=HealpixSEConfig(species_abundance_floor=1.0e-8),
        )


def test_abundance_floor_uses_lte_for_excluded_positive_cells(tmp_path: Path):
    """Excluded emitters retain normalized LTE populations and line opacity."""

    rad, shape = _make_cartesian_rad(tmp_path, n=2)
    n_co = 10.0
    chem = _make_chemistry_result(shape, n_co=n_co)
    abundance = np.full(shape, 1.0e-10)
    chem.abundances = {"co": Quantity(abundance, "dimensionless")}

    output_dir = tmp_path / "abundance_floor"
    path = solve_and_write_healpix_levelpop(
        rad=rad,
        chemistry_result=chem,
        species="co",
        output_dir=output_dir,
        config=HealpixSEConfig(species_abundance_floor=1.0e-8),
    )

    molecule = parse_lamda_molecule_file(_CO_LAMDA)
    expected = n_co * lte_populations(molecule, np.array([30.0]))[0]
    tokens = path.read_text().split()
    n_cells = int(tokens[1])
    n_levels = int(tokens[2])
    populations = np.asarray(
        tokens[3 + n_levels : 3 + n_levels + n_cells * n_levels],
        dtype=float,
    ).reshape(n_cells, n_levels)
    np.testing.assert_allclose(
        populations,
        np.broadcast_to(expected, populations.shape),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(populations.sum(axis=1), n_co, rtol=1.0e-12)
    assert np.all(populations[:, 1:] > 0.0)

    manifest = json.loads(
        (output_dir / "external_levelpop_manifest_co.json").read_text()
    )
    assert manifest["species_abundance_floor"] == pytest.approx(1.0e-8)
    assert manifest["species_abundance_range_per_h_nucleus"] == pytest.approx(
        [1.0e-10, 1.0e-10]
    )
    assert manifest["cell_counts"] == {
        "lte_fallback_species_cells": int(np.prod(shape)),
        "solved_nonzero_species_cells": 0,
        "species_below_abundance_floor_cells": int(np.prod(shape)),
        "species_below_density_floor_cells": 0,
        "species_candidate_cells": 0,
        "species_zero_cells": 0,
        "total_grid_cells": int(np.prod(shape)),
        "zero_or_below_floor_species_cells": int(np.prod(shape)),
    }


def test_abundance_floor_selects_only_abundant_cells(tmp_path: Path):
    """The abundance mask limits SE cells and is checkpoint-fingerprinted."""

    rad, shape = _make_cartesian_rad(tmp_path, n=2)
    n_co = 10.0
    chem = _make_chemistry_result(shape, n_co=n_co)
    abundance = np.full(shape, 1.0e-6)
    abundance[0, :, :] = 1.0e-10
    chem.abundances = {"co": Quantity(abundance, "dimensionless")}

    output_dir = tmp_path / "partial_abundance_floor"
    path = solve_and_write_healpix_levelpop(
        rad=rad,
        chemistry_result=chem,
        species="co",
        output_dir=output_dir,
        config=HealpixSEConfig(
            nside=1,
            maxiter=20,
            convcrit=1.0e-3,
            max_ray_steps=200,
            species_abundance_floor=1.0e-8,
        ),
    )

    manifest = json.loads(
        (output_dir / "external_levelpop_manifest_co.json").read_text()
    )
    assert manifest["cell_counts"]["species_candidate_cells"] == 4
    assert manifest["cell_counts"]["lte_fallback_species_cells"] == 4
    checkpoint = json.loads(
        (output_dir / "checkpoints" / "checkpoint_latest_co.json").read_text()
    )
    assert checkpoint["fingerprint"]["version"] == 3
    assert checkpoint["fingerprint"]["species_abundance_floor"] == pytest.approx(
        1.0e-8
    )
    assert checkpoint["fingerprint"]["species_abundance_sha256"] is not None

    molecule = parse_lamda_molecule_file(_CO_LAMDA)
    expected_lte = n_co * lte_populations(molecule, np.array([30.0]))[0]
    tokens = path.read_text().split()
    n_cells = int(tokens[1])
    n_levels = int(tokens[2])
    populations = np.asarray(
        tokens[3 + n_levels : 3 + n_levels + n_cells * n_levels],
        dtype=float,
    ).reshape(n_cells, n_levels)
    excluded = RadWriter.flatten_scalar_to_radmc_order(
        rad.model.mesh,
        abundance < 1.0e-8,
    ).astype(bool)
    np.testing.assert_allclose(
        populations[excluded],
        np.broadcast_to(expected_lte, populations[excluded].shape),
        rtol=1.0e-12,
    )


def test_collision_temperature_endpoint_holding_is_audited(tmp_path: Path):
    """Boundary-held rates permit a solve without changing physical Tgas."""

    rad, shape = _make_cartesian_rad(tmp_path, n=2)
    hot_temperature = 5000.0
    _add_field(
        rad.model,
        "gas_temperature",
        np.full(shape, hot_temperature),
        "K",
    )
    chem = _make_chemistry_result(shape, n_co=10.0)

    output_dir = tmp_path / "out"
    path = solve_and_write_healpix_levelpop(
        rad=rad,
        chemistry_result=chem,
        species="co",
        output_dir=output_dir,
        config=HealpixSEConfig(
            nside=1,
            maxiter=20,
            convcrit=1.0e-3,
            max_ray_steps=200,
        ),
    )

    assert path.exists()
    payload = json.loads(
        (output_dir / "external_levelpop_manifest_co.json").read_text()
    )
    assert payload["collision_temperature_policy"] == "nearest_table_boundary"
    assert payload["temperature_range_K"] == [hot_temperature, hot_temperature]
    diagnostics = payload["collision_temperature_diagnostics"]
    assert diagnostics["p-h2"]["selected_cells_above"] == int(np.prod(shape))
    assert diagnostics["o-h2"]["selected_cells_above"] == int(np.prod(shape))


def test_prepare_external_population_run_and_validation(
    tmp_path: Path,
    canonical_installed_lamda_installer: list[tuple[str, Path]],
):
    rad, shape = _make_cartesian_rad(tmp_path, n=3)
    chem = _make_chemistry_result(shape, n_co=5.0)

    levelpop_dir = tmp_path / "pops"
    config = HealpixSEConfig(nside=1, maxiter=15, convcrit=1.0e-3, max_ray_steps=200)
    levelpop_path = solve_and_write_healpix_levelpop(
        rad=rad,
        chemistry_result=chem,
        species="co",
        output_dir=levelpop_dir,
        config=config,
    )

    # Construct a minimal source_inputs_dir.
    src_inputs = tmp_path / "src_run" / "radmc3d_inputs"
    src_inputs.mkdir(parents=True)
    (src_inputs / "amr_grid.inp").write_text(
        "\n".join(
            [
                "1", "0", "0", "0", "0 0 0",
                f"{shape[0]} {shape[1]} {shape[2]}",
                # Dummy edges; validation only reads the cell counts row.
            ]
        )
        + "\n"
    )
    (src_inputs / "wavelength_micron.inp").write_text("1\n1000.0\n")
    # Synthesize per-cell binary RADMC files (cells x correct count).
    ncells = int(np.prod(shape))
    def _scalar(name: str, value: float):
        with (src_inputs / name).open("wb") as f:
            np.asarray([1, 8, ncells], dtype=np.int64).tofile(f)
            np.full(ncells, value, dtype=np.float64).tofile(f)
    _scalar("numberdens_co.binp", 5.0)
    _scalar("gas_temperature.binp", 30.0)
    _scalar("microturbulence.binp", 1.0e4)
    with (src_inputs / "gas_velocity.binp").open("wb") as f:
        np.asarray([1, 8, ncells], dtype=np.int64).tofile(f)
        np.zeros(3 * ncells, dtype=np.float64).tofile(f)

    work = tmp_path / "run"
    prepare_external_population_line_run(
        source_inputs_dir=src_inputs,
        work_dir=work,
        species="co",
        levelpop_file=levelpop_path,
        copy_mode="copy",
    )
    assert canonical_installed_lamda_installer == [
        ("co", levelpop_dir / "molecule_co.inp"),
        ("co", work / "radmc3d_inputs" / "molecule_co.inp"),
    ]
    info = validate_external_population_run(work, species="co")
    assert info["line_mode"] == 50
    assert Path(info["levelpop_file"]).exists()

    staged_manifest = work / "radmc3d_inputs" / "external_levelpop_manifest_co.json"
    manifest_payload = json.loads(staged_manifest.read_text())
    manifest_payload.pop("molecule_sha256")
    staged_manifest.write_text(json.dumps(manifest_payload))
    with pytest.raises(ValueError, match="does not record molecule_sha256"):
        validate_external_population_run(work, species="co")


def test_external_population_apis_do_not_accept_local_molecule_files() -> None:
    assert "molecule_file" not in inspect.signature(
        solve_and_write_healpix_levelpop
    ).parameters
    assert "molecule_file" not in inspect.signature(
        prepare_external_population_line_run
    ).parameters
