"""Tests for RADMC-3D non-LTE line-transfer setup contracts.

These tests use small synthetic files and configs to protect LAMDA collider
policy, `lines.inp` staging, gas-temperature requirements, and run validation.
"""

# db-keywords: photodesorption, gow17, nonlte, line-transfer, gas-temperature, validation, config, radmc3d, model, mesh, field
# db-role: validation
# db-scope: test
# db-purpose: Tests for RADMC-3D non-LTE line-transfer setup contracts.

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from diskbridge._params import default_line_colliders_for_species
from diskbridge.radmc3d.image import RadImage, _line_mode_uses_gas_temperature
from diskbridge.radmc3d.colliders import (
    LAMDA_URLS,
    assert_lamda_collision_order,
    ensure_official_lamda_file,
    gow17_lamda_colliders,
    install_validated_molecule_file,
    read_lamda_collision_order,
)
from diskbridge.radmc3d.line_transfer import (
    NonLTELineTransferConfig,
    SpeciesLineConfig,
    prepare_nonlte_line_run,
    validate_line_run,
    write_lines_inp,
)


def _write_amr_grid(path: Path, nr: int = 2, ntheta: int = 1, nphi: int = 1) -> None:
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


def _write_scalar_binp(path: Path, values: list[float]) -> None:
    with path.open("wb") as f:
        np.asarray([1, 8, len(values)], dtype=np.int64).tofile(f)
        np.asarray(values, dtype=np.float64).tofile(f)


def _write_vector_binp(path: Path, values: list[tuple[float, float, float]]) -> None:
    with path.open("wb") as f:
        np.asarray([1, 8, len(values)], dtype=np.int64).tofile(f)
        np.asarray(values, dtype=np.float64).reshape(-1).tofile(f)


def _write_molecule(path: Path, colliders: list[str]) -> None:
    blocks = []
    for idx, collider in enumerate(colliders, start=2):
        blocks.extend(
            [
                "!COLLISIONS BETWEEN",
                f"{idx} CO + {collider}",
                "!NUMBER OF COLL TRANS",
                "0",
                "!NUMBER OF COLL TEMPS",
                "1",
                "!COLL TEMPS",
                "10.0",
                "!TRANS + UP + LOW + COLLRATES(cm^3 s^-1)",
            ]
        )
    path.write_text(
        "\n".join(
            [
                "!MOLECULE",
                "CO",
                "!NUMBER OF COLL PARTNERS",
                str(len(colliders)),
                *blocks,
            ]
        )
        + "\n"
    )


def _line_params(*, line_colliders="gow17_lamda", line_mode=3) -> SimpleNamespace:
    return SimpleNamespace(
        gasspecies=["co", "catom", "hco+"],
        iline=[3, 1, 4],
        abundance=[1.0e-4, 1.0e-4, 1.0e-9],
        width=5.0,
        nline=41,
        turbvel=0.0,
        line_mode=line_mode,
        line_colliders=line_colliders,
        photodissociation=True,
        freezeout=True,
        photodesorption=True,
    )


def _radimage_stub(
    tmp_path: Path,
    *,
    line_colliders="gow17_lamda",
    line_mode=3,
) -> RadImage:
    image = object.__new__(RadImage)
    image.params = _line_params(line_colliders=line_colliders, line_mode=line_mode)
    image.inputs_dir = tmp_path
    image.model_dir = tmp_path
    image.model = None
    return image


def _touch_density_files(inputs: Path, colliders: list[str]) -> None:
    inputs.mkdir(parents=True, exist_ok=True)
    for collider in colliders:
        (inputs / f"numberdens_{collider}.binp").write_bytes(b"stub")


def _write_base_inputs(base: Path, *, include_gas_temperature: bool = True) -> Path:
    inputs = base / "radmc3d_inputs"
    outputs = base / "radmc3d_outputs"
    inputs.mkdir(parents=True)
    outputs.mkdir(parents=True)
    _write_amr_grid(inputs / "amr_grid.inp")
    (inputs / "wavelength_micron.inp").write_text("1\n1000.0\n")
    (inputs / "radmc3d.inp").write_text("incl_lines = 1\nlines_mode = 3\ntgas_eq_tdust = 0\n")
    _write_molecule(inputs / "molecule_co.inp", ["p-H2", "o-H2"])
    _write_vector_binp(
        inputs / "gas_velocity.binp",
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    )
    _write_scalar_binp(inputs / "numberdens_co.binp", [1.0, 2.0])
    _write_scalar_binp(inputs / "numberdens_p-h2.binp", [2.5, 5.0])
    _write_scalar_binp(inputs / "numberdens_o-h2.binp", [7.5, 15.0])
    if include_gas_temperature:
        _write_scalar_binp(inputs / "gas_temperature.binp", [30.0, 40.0])
    (outputs / "dust_temperature.bdat").write_bytes(b"not used by unit validation")
    return inputs


def test_write_lines_inp_co_h2(tmp_path: Path) -> None:
    cfg = SpeciesLineConfig(species="co", line_mode=3, colliders=["p-h2", "o-h2"])
    path = write_lines_inp(tmp_path / "lines.inp", cfg)
    assert path.read_text() == "2\n1\nco    leiden    0    0    2\np-h2\no-h2\n"


def test_line_colliders_are_configured_only_for_known_nonlte_species() -> None:
    assert default_line_colliders_for_species("co") == ["p-h2", "o-h2"]
    assert default_line_colliders_for_species("catom") == [
        "h",
        "p-h2",
        "o-h2",
        "e",
    ]
    assert default_line_colliders_for_species("hco+") == ["p-h2", "o-h2"]
    with pytest.raises(ValueError, match="No strict GOW17/LAMDA collider policy"):
        default_line_colliders_for_species("unknown")


def test_strict_lamda_collision_policy_matches_official_order() -> None:
    moldata = Path(__file__).resolve().parents[2] / "data" / "moldata"
    assert gow17_lamda_colliders("co") == ["p-h2", "o-h2"]
    assert read_lamda_collision_order(moldata / "co.dat") == ["p-h2", "o-h2"]
    assert_lamda_collision_order(moldata / "co.dat", gow17_lamda_colliders("co"))
    assert gow17_lamda_colliders("catom") == [
        "h",
        "p-h2",
        "o-h2",
        "e",
    ]
    assert read_lamda_collision_order(moldata / "catom.dat") == [
        "h",
        "e",
        "h+",
        "he",
        "p-h2",
        "o-h2",
    ]
    assert gow17_lamda_colliders("hco+") == ["p-h2", "o-h2"]
    assert read_lamda_collision_order(moldata / "hco+.dat") == [
        "p-h2",
        "o-h2",
    ]
    assert_lamda_collision_order(
        moldata / "hco+.dat",
        gow17_lamda_colliders("hco+"),
    )


def test_supported_lamda_urls_have_no_legacy_xpol_path() -> None:
    assert LAMDA_URLS["hco+"] == (
        "https://home.strw.leidenuniv.nl/~moldata/datafiles/hco%2B.dat"
    )
    assert all("xpol" not in url.lower() for url in LAMDA_URLS.values())


def test_catom_molecule_file_is_refreshed_and_reordered_to_active_colliders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moldata = Path(__file__).resolve().parents[2] / "data" / "moldata"
    source = moldata / "catom.dat"
    local_moldata = tmp_path / "moldata"
    local_moldata.mkdir()
    local_file = local_moldata / "catom.dat"
    local_file.write_text("stale local data\n")
    downloads: list[str] = []

    def fake_download(command: list[str], *, check: bool):
        assert check is True
        downloads.append(command[-1])
        shutil.copyfile(source, command[-2])

    monkeypatch.setattr(
        "diskbridge.radmc3d.colliders.subprocess.run",
        fake_download,
    )
    staged = install_validated_molecule_file("catom", local_moldata, tmp_path)

    assert len(downloads) == 1
    assert local_file.read_bytes() == source.read_bytes()
    assert read_lamda_collision_order(staged) == ["h", "p-h2", "o-h2", "e"]
    assert_lamda_collision_order(staged, gow17_lamda_colliders("catom"))


def test_lamda_download_failure_does_not_fall_back_to_stale_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_file = tmp_path / "co.dat"
    local_file.write_text("stale local data\n")

    def fail_download(command: list[str], *, check: bool):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(
        "diskbridge.radmc3d.colliders.subprocess.run",
        fail_download,
    )

    with pytest.raises(RuntimeError, match="network unavailable"):
        ensure_official_lamda_file("co", tmp_path)

    assert local_file.read_text() == "stale local data\n"
    assert list(tmp_path.glob(".co.*.download")) == []


def test_malformed_lamda_download_does_not_replace_local_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_file = tmp_path / "hco+.dat"
    local_file.write_text("previous valid local data\n")

    def malformed_download(command: list[str], *, check: bool):
        assert check is True
        Path(command[-2]).write_text("not a LAMDA molecule file\n")

    monkeypatch.setattr(
        "diskbridge.radmc3d.colliders.subprocess.run",
        malformed_download,
    )

    with pytest.raises(ValueError, match="missing required collision block"):
        ensure_official_lamda_file("hco+", tmp_path)

    assert local_file.read_text() == "previous valid local data\n"
    assert list(tmp_path.glob(".hco+.*.download")) == []


def test_radimage_strict_gow17_lamda_lines_inp_for_catom(tmp_path: Path) -> None:
    _touch_density_files(tmp_path, ["h", "p-h2", "o-h2", "e"])
    image = _radimage_stub(tmp_path)

    image._ensure_lines_inp("catom")

    assert (tmp_path / "lines.inp").read_text() == (
        "2\n"
        "1\n"
        "catom    leiden    0    0    4\n"
        "h\n"
        "p-h2\n"
        "o-h2\n"
        "e\n"
    )


def test_radimage_refreshes_online_molecule_for_external_populations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _radimage_stub(tmp_path, line_mode=50)
    (tmp_path / "levelpop_co.dat").write_text("staged populations\n")
    (tmp_path / "external_levelpop_manifest_co.json").write_text("{}\n")
    calls: list[tuple[str, Path, Path]] = []

    def install(species: str, moldata_dir: str | Path, inputs_dir: str | Path) -> Path:
        destination = Path(inputs_dir) / f"molecule_{species}.inp"
        destination.write_text("current online molecule\n")
        calls.append((species, Path(moldata_dir), Path(inputs_dir)))
        return destination

    monkeypatch.setattr(
        "diskbridge.radmc3d.image.install_validated_molecule_file",
        install,
    )

    image._ensure_molecule_file("co")

    assert len(calls) == 1
    assert calls[0][0] == "co"
    assert calls[0][2] == tmp_path
    assert (tmp_path / "molecule_co.inp").read_text() == (
        "current online molecule\n"
    )


def test_radimage_strict_gow17_lamda_lines_inp_for_hcop(tmp_path: Path) -> None:
    _touch_density_files(tmp_path, ["p-h2", "o-h2"])
    image = _radimage_stub(tmp_path)

    image._ensure_lines_inp("hco+")

    assert (tmp_path / "lines.inp").read_text() == (
        "2\n"
        "1\n"
        "hco+    leiden    0    0    2\n"
        "p-h2\n"
        "o-h2\n"
    )


def test_radimage_strict_gow17_lamda_rejects_legacy_colliders(tmp_path: Path) -> None:
    _touch_density_files(tmp_path, ["p-h2", "o-h2"])
    image = _radimage_stub(tmp_path, line_colliders="h2")

    with pytest.raises(ValueError, match="line_colliders = gow17_lamda"):
        image._ensure_lines_inp("co")


def test_species_line_config_rejects_non_strict_nonlte_colliders() -> None:
    with pytest.raises(ValueError, match="strict GOW17/LAMDA collider order"):
        SpeciesLineConfig(species="catom", line_mode=3, colliders=["h", "p-h2"])


def test_radimage_strict_gow17_lamda_requires_collider_density_files(
    tmp_path: Path,
) -> None:
    _touch_density_files(tmp_path, ["h", "p-h2", "o-h2"])
    image = _radimage_stub(tmp_path)

    with pytest.raises(RuntimeError, match="numberdens_e.binp"):
        image._ensure_lines_inp("catom")


@pytest.mark.parametrize(
    ("line_mode", "expected"),
    [(2, False), (3, True), (4, True), (50, True), (-50, True)],
)
def test_line_mode_gas_temperature_policy(line_mode: int, expected: bool) -> None:
    assert _line_mode_uses_gas_temperature(line_mode) is expected


def test_validation_requires_gow17_gas_temperature(tmp_path: Path) -> None:
    _write_base_inputs(tmp_path, include_gas_temperature=False)
    write_lines_inp(
        tmp_path / "radmc3d_inputs" / "lines.inp",
        SpeciesLineConfig(species="co", line_mode=3, colliders=["p-h2", "o-h2"]),
    )

    with pytest.raises(FileNotFoundError, match="gas_temperature"):
        validate_line_run(
            tmp_path,
            NonLTELineTransferConfig(use_gow17_tgas=True),
            SpeciesLineConfig(species="co", line_mode=3, colliders=["p-h2", "o-h2"]),
        )


def test_validation_rejects_tgas_eq_tdust_true(tmp_path: Path) -> None:
    _write_base_inputs(tmp_path, include_gas_temperature=True)
    (tmp_path / "radmc3d_inputs" / "radmc3d.inp").write_text(
        "incl_lines = 1\nlines_mode = 3\ntgas_eq_tdust = 1\n"
    )
    write_lines_inp(
        tmp_path / "radmc3d_inputs" / "lines.inp",
        SpeciesLineConfig(species="co", line_mode=3, colliders=["p-h2", "o-h2"]),
    )

    with pytest.raises(ValueError, match="tgas_eq_tdust=1"):
        validate_line_run(
            tmp_path,
            NonLTELineTransferConfig(use_gow17_tgas=True),
            SpeciesLineConfig(species="co", line_mode=3, colliders=["p-h2", "o-h2"]),
        )


def test_prepare_nonlte_line_run_stages_co_h2_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_inputs = _write_base_inputs(tmp_path / "source", include_gas_temperature=True)
    work = tmp_path / "line_rt" / "co_J3_h2_lvg"
    source_molecule = Path(__file__).resolve().parents[2] / "data" / "moldata" / "co.dat"

    def fake_download(command: list[str], *, check: bool):
        assert check is True
        shutil.copyfile(source_molecule, command[-2])

    monkeypatch.setattr(
        "diskbridge.radmc3d.colliders.subprocess.run",
        fake_download,
    )
    species = SpeciesLineConfig(
        species="co",
        line_mode=3,
        colliders=["p-h2", "o-h2"],
        transition=3,
    )

    prepare_nonlte_line_run(
        source_inputs_dir=source_inputs,
        work_dir=work,
        config=NonLTELineTransferConfig(use_gow17_tgas=True),
        species_config=species,
        copy_mode="copy",
    )

    inputs = work / "radmc3d_inputs"
    assert (inputs / "radmc3d.inp").read_text().find("tgas_eq_tdust = 0") >= 0
    assert (inputs / "lines.inp").read_text() == "2\n1\nco    leiden    0    0    2\np-h2\no-h2\n"
    manifest = json.loads((work / "line_rt_manifest.json").read_text())
    assert manifest["species"] == "co"
    assert manifest["colliders"] == ["p-h2", "o-h2"]
    assert manifest["use_gow17_tgas"] is True


def test_validation_accepts_radmc3d_binp_scalar_headers(tmp_path: Path) -> None:
    inputs = tmp_path / "radmc3d_inputs"
    inputs.mkdir(parents=True)
    _write_amr_grid(inputs / "amr_grid.inp")
    (inputs / "radmc3d.inp").write_text("incl_lines = 1\nlines_mode = 3\ntgas_eq_tdust = 0\n")
    _write_molecule(inputs / "molecule_co.inp", ["p-H2", "o-H2"])
    _write_vector_binp(inputs / "gas_velocity.binp", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    _write_scalar_binp(inputs / "gas_temperature.binp", [30.0, 40.0])
    _write_scalar_binp(inputs / "numberdens_co.binp", [1.0, 2.0])
    _write_scalar_binp(inputs / "numberdens_p-h2.binp", [2.5, 5.0])
    _write_scalar_binp(inputs / "numberdens_o-h2.binp", [7.5, 15.0])
    write_lines_inp(
        inputs / "lines.inp",
        SpeciesLineConfig(species="co", line_mode=3, colliders=["p-h2", "o-h2"]),
    )

    summary = validate_line_run(
        tmp_path,
        NonLTELineTransferConfig(use_gow17_tgas=True),
        SpeciesLineConfig(species="co", line_mode=3, colliders=["p-h2", "o-h2"]),
    )

    assert summary["ncells"] == 2
    assert summary["gas_temperature_file"].endswith("gas_temperature.binp")
