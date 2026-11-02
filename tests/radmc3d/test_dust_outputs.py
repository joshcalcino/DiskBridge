"""Tests for RADMC-3D dust output and dust-species density contracts.

These tests use tiny synthetic models to protect deterministic writer behavior,
dust bin counts, and intrinsic grain-density policy.
"""

# db-keywords: validation, config, units, radmc3d, model, mesh, field
# db-role: validation
# db-scope: test
# db-purpose: Tests for RADMC-3D dust output and dust-species density contracts.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from diskbridge._units import Quantity
from diskbridge.model.dust_species import resolve_species_grain_density
from diskbridge.model.dust_opacity import DustOpacity
from diskbridge.model.core import Model
from diskbridge.model.dust import DustComponent, DustDistribution
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.writer import RadWriter


class _DummyDustBin:
    def __init__(self, size_um: float, density: Field, index: int):
        self.size = Quantity(size_um, "micron")
        self.opacity = DustOpacity(material="mix_2species_60silicates_40carbons")
        self.opacity_species = f"{self.opacity.species_name}{index}"
        self.size_min = Quantity(size_um * 0.5, "micron")
        self.density_material = Quantity(2.7, "g/cm^3")
        self.species_grain_density = Quantity(2.7, "g/cm^3")
        self._density = density

    def __getitem__(self, key: str) -> Field:
        if key != "density":
            raise KeyError(key)
        return self._density


class _DummyDust:
    def __init__(self, nbin: int, density: Field):
        self.nbin = nbin
        self.bins = {
            f"bin_{i}": _DummyDustBin(float(i + 1), density, i)
            for i in range(nbin)
        }


def _fake_optool_writer(calls):
    def generate(opacity, grain_size, wavelengths, output_dir, **kwargs):
        calls.append((opacity, grain_size, wavelengths, kwargs))
        prefix = "dustkapscatmat" if kwargs["scattering_mode"] >= 3 else "dustkappa"
        path = Path(output_dir) / f"{prefix}_{kwargs['species_name']}.inp"
        path.write_text("native executable output\n")
        return path
    return generate


def _cartesian_one_cell_model(nbin: int) -> Model:
    model = Model()
    model.mesh = Mesh.cartesian(
        x=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
        y=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
        z=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
    )
    density = Field(
        quantity="density",
        data=Quantity(np.ones(model.mesh.shape), "g/cm^3"),
        axis_order=model.mesh.axis_names(),
    )
    model.dust = _DummyDust(nbin=nbin, density=density)
    return model


def test_default_two_component_dust_outputs_have_matching_bin_count(tmp_path: Path, monkeypatch) -> None:
    nbin = 20
    species = "mix_2species_60silicates_40carbons"
    model = _cartesian_one_cell_model(nbin)
    writer = RadWriter(model)
    writer.params = SimpleNamespace(
        species=species,
        opacity_dir="data/opac",
        lambda_min=Quantity(0.1, "micron"),
        lambda_max=Quantity(10.0, "micron"),
        n_lambda=3,
        scat_mode=0,
        nbcores=2,
    )
    calls = []
    monkeypatch.setattr("diskbridge.radmc3d.writer.write_dust_opacity", _fake_optool_writer(calls))

    writer.write_dust_density(tmp_path)
    writer.write_dustopac(tmp_path)
    writer.compute_and_write_dust_opacities(tmp_path)

    inputs = tmp_path / "radmc3d_inputs"
    with (inputs / "dust_density.binp").open("rb") as f:
        header = np.fromfile(f, dtype=np.int64, count=4)
    np.testing.assert_array_equal(header, np.array([1, 8, 1, nbin], dtype=np.int64))

    dustopac_lines = (inputs / "dustopac.inp").read_text().splitlines()
    assert int(dustopac_lines[1]) == nbin
    expected_species = {grain.opacity_species for grain in model.dust.bins.values()}
    assert expected_species.issubset(set(dustopac_lines))

    expected_kappa = {f"dustkappa_{name}.inp" for name in expected_species}
    actual_kappa = {path.name for path in inputs.glob("dustkappa_*.inp")}
    assert actual_kappa == expected_kappa
    assert len(calls) == nbin
    assert calls[0][1] == Quantity(1.0, "micron")
    np.testing.assert_allclose(calls[0][2].to("micron").magnitude, [0.1, 1., 10.])
    assert all(call[3]["threads"] == 2 for call in calls)


def test_matrix_opacity_files_are_tracked_under_emitted_names(tmp_path: Path, monkeypatch) -> None:
    writer = RadWriter(_cartesian_one_cell_model(nbin=2))
    species = "mix_2species_60silicates_40carbons"
    writer.params = SimpleNamespace(
        species=species,
        opacity_dir="data/opac",
        lambda_min=Quantity(0.1, "micron"),
        lambda_max=Quantity(10.0, "micron"),
        n_lambda=3,
        scat_mode=5,
        nbcores=2,
    )

    calls = []
    monkeypatch.setattr("diskbridge.radmc3d.writer.write_dust_opacity", _fake_optool_writer(calls))
    writer.compute_and_write_dust_opacities(tmp_path, executable="/chosen/optool", threads=1)
    assert all(call[3]["executable"] == "/chosen/optool" for call in calls)
    assert all(call[3]["threads"] == 1 for call in calls)

    expected = {f"dustkapscatmat_{grain.opacity_species}.inp"
                for grain in writer.model.dust.bins.values()}
    assert set(writer.written_files) == expected
    assert {path.name for path in (tmp_path / "radmc3d_inputs").iterdir()} == expected
    assert all(path.is_file() for path in writer.written_files.values())


def test_known_species_density_is_authoritative_with_warning() -> None:
    with pytest.warns(UserWarning, match="fixed intrinsic density"):
        rho_s = resolve_species_grain_density(
            "mix_2species_60silicates_40carbons",
            Quantity(3.5, "g/cm^3"),
        )

    assert rho_s == Quantity(2.7, "g/cm^3")


def test_unknown_species_requires_explicit_grain_density() -> None:
    with pytest.raises(ValueError, match="requires grain_density"):
        resolve_species_grain_density("my_custom_species")

    rho_s = resolve_species_grain_density("my_custom_species", Quantity(3.1, "g/cm^3"))
    assert rho_s == Quantity(3.1, "g/cm^3")


def test_dust_component_preserves_resolved_porous_density() -> None:
    opacity = DustOpacity(material="mix_2species_60silicates_40carbons", porosity=0.5)
    dist = DustDistribution(
        amin=Quantity(0.01, "micron"), amax=Quantity(1., "micron"), nbin=2,
        power_index=3.5, grain_density=Quantity(1.35, "g/cm^3"),
    )
    component = DustComponent(dist, 0.01, opacity=opacity)
    assert component.grain_density == Quantity(1.35, "g/cm^3")
    assert component.opacity == opacity


def test_complete_writer_generates_once_with_matching_scattering_mode(tmp_path, monkeypatch):
    """Control-file choices and Optool choices agree in the complete workflow."""
    writer = RadWriter(_cartesian_one_cell_model(1))
    for method in ("write_all_input_files", "write_gas_velocity"):
        monkeypatch.setattr(writer, method, lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(writer, "compute_and_write_dust_opacities",
                        lambda *args, **kwargs: calls.append(kwargs))
    writer.write_model_inputs(tmp_path, scattering_mode=5, setthreads=3,
                              executable="/chosen/optool", include_external_source=False)
    assert calls == [{"scattering_mode": 5, "threads": 3, "executable": "/chosen/optool"}]


def test_dustopac_uses_component_specific_species_order(tmp_path):
    writer = RadWriter(_cartesian_one_cell_model(2))
    bins = list(writer.model.dust.bins.values())
    bins[0].opacity_species, bins[1].opacity_species = "silicate0", "carbon1"
    writer.write_dustopac(tmp_path, scattering_mode=5)
    rows = (tmp_path / "radmc3d_inputs/dustopac.inp").read_text().splitlines()
    assert rows[5] == "silicate0"
    assert rows[9] == "carbon1"
    assert rows[3] == rows[7] == "10"
