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
from diskbridge.dust_species import resolve_species_grain_density
from diskbridge.model.core import Model
from diskbridge.model.dust import DustComponent, DustDistribution
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.writer import RadWriter


class _DummyDustBin:
    def __init__(self, size_um: float, density: Field):
        self.size = Quantity(size_um, "micron")
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
            f"bin_{i}": _DummyDustBin(float(i + 1), density)
            for i in range(nbin)
        }


class _DummyOpacityCalculator:
    def __init__(self) -> None:
        self.grain_sizes: list[float] = []
        self.grain_densities: list[float] = []

    def compute_opacity(self, **kwargs):
        wavelengths = np.asarray(kwargs["wavelengths"], dtype=float)
        self.grain_sizes.append(float(kwargs["grain_size"]))
        self.grain_densities.append(float(kwargs["grain_density"]))
        return {
            "wav": wavelengths,
            "kabs": np.ones_like(wavelengths),
            "kscat": np.zeros_like(wavelengths),
            "gscat": np.zeros_like(wavelengths),
        }

    def write_radmc3d_opacity_file(
        self,
        opacity_data,
        output_path: str | Path,
        scattering_matrix: bool = False,
    ) -> None:
        output_path = Path(output_path)
        path = output_path.parent / f"dustkappa_{output_path.name}.inp"
        with path.open("w") as f:
            f.write("3\n")
            f.write(f"{len(opacity_data['wav'])}\n")
            for lam, kabs, kscat, gscat in zip(
                opacity_data["wav"],
                opacity_data["kabs"],
                opacity_data["kscat"],
                opacity_data["gscat"],
            ):
                f.write(f"{lam * 1.0e4:.8e} {kabs:.8e} {kscat:.8e} {gscat:.8e}\n")


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


def test_default_two_component_dust_outputs_have_matching_bin_count(tmp_path: Path) -> None:
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
    )
    opacity_calculator = _DummyOpacityCalculator()
    writer.opacity_calculator = opacity_calculator

    writer.write_dust_density(tmp_path)
    writer.write_dustopac(tmp_path)
    writer.compute_and_write_dust_opacities(tmp_path)

    inputs = tmp_path / "radmc3d_inputs"
    with (inputs / "dust_density.binp").open("rb") as f:
        header = np.fromfile(f, dtype=np.int64, count=4)
    np.testing.assert_array_equal(header, np.array([1, 8, 1, nbin], dtype=np.int64))

    dustopac_lines = (inputs / "dustopac.inp").read_text().splitlines()
    assert int(dustopac_lines[1]) == nbin
    expected_species = {f"{species}{i}" for i in range(nbin)}
    assert expected_species.issubset(set(dustopac_lines))

    expected_kappa = {f"dustkappa_{name}.inp" for name in expected_species}
    actual_kappa = {path.name for path in inputs.glob("dustkappa_*.inp")}
    assert actual_kappa == expected_kappa
    assert len(opacity_calculator.grain_sizes) == nbin
    assert opacity_calculator.grain_sizes[0] == Quantity(1.0, "micron").to_base_units().magnitude
    assert opacity_calculator.grain_densities == [2.7] * nbin


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


def test_dust_component_stores_registry_density_for_known_species() -> None:
    dist = DustDistribution(
        amin=Quantity(0.01, "micron"),
        amax=Quantity(1.0, "micron"),
        nbin=2,
        power_index=3.5,
        grain_density=Quantity(3.5, "g/cm^3"),
    )

    with pytest.warns(UserWarning, match="fixed intrinsic density"):
        component = DustComponent(
            distribution=dist,
            dust_to_gas_ratio=0.01,
            species_base="mix_2species_60silicates_40carbons",
        )

    assert component.grain_density == Quantity(2.7, "g/cm^3")
    assert component.species_grain_density == Quantity(2.7, "g/cm^3")
    assert component.distribution.grain_density == Quantity(2.7, "g/cm^3")
