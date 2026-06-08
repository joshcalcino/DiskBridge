"""End-to-end smoke test for the HEALPix non-LTE external-population driver."""

# db-keywords: healpix-columns, nonlte, line-transfer, gas-temperature, validation, config, units, radmc3d, model, mesh, field
# db-role: validation
# db-scope: test
# db-purpose: End-to-end smoke test for the HEALPix non-LTE external-population driver.

from __future__ import annotations

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


_REPO = Path(__file__).resolve().parents[2]
_CO_LAMDA = _REPO / "data" / "moldata" / "co.dat"


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


def test_solve_and_write_healpix_levelpop_smoke(tmp_path: Path):
    rad, shape = _make_cartesian_rad(tmp_path, n=4)
    chem = _make_chemistry_result(shape, n_co=10.0)
    # Use the real CO LAMDA file (already in p-H2, o-H2 order).
    molecule_file = tmp_path / "molecule_co.inp"
    shutil.copyfile(_CO_LAMDA, molecule_file)

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
        molecule_file=molecule_file,
        output_dir=out_dir,
        config=config,
    )
    assert path.exists()
    assert path.name == "levelpop_co.dat"

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


def test_prepare_external_population_run_and_validation(tmp_path: Path):
    rad, shape = _make_cartesian_rad(tmp_path, n=3)
    chem = _make_chemistry_result(shape, n_co=5.0)
    molecule_file = tmp_path / "molecule_co.inp"
    shutil.copyfile(_CO_LAMDA, molecule_file)

    levelpop_dir = tmp_path / "pops"
    config = HealpixSEConfig(nside=1, maxiter=15, convcrit=1.0e-3, max_ray_steps=200)
    levelpop_path = solve_and_write_healpix_levelpop(
        rad=rad,
        chemistry_result=chem,
        species="co",
        molecule_file=molecule_file,
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
        molecule_file=molecule_file,
        copy_mode="copy",
    )
    info = validate_external_population_run(work, species="co")
    assert info["line_mode"] == 50
    assert Path(info["levelpop_file"]).exists()
