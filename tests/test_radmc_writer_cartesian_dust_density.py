import numpy as np

from diskbridge._units import Quantity
from diskbridge.model.core import Model, SubModel
from diskbridge.model.dust import Dust
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.writer import RadWriter


def _make_cartesian_model_with_single_dust_bin(
    *, nx: int, ny: int, nz: int, rho_zyx: np.ndarray
) -> Model:
    model = Model()
    model.coord_system = "cartesian"
    model.mesh = Mesh.cartesian(
        x=Axis(edges=Quantity(np.linspace(0.0, 1.0, nx + 1), "cm")),
        y=Axis(edges=Quantity(np.linspace(0.0, 1.0, ny + 1), "cm")),
        z=Axis(edges=Quantity(np.linspace(0.0, 1.0, nz + 1), "cm")),
    )

    model.gas = SubModel(model)
    model.dust = Dust(model)
    model.dust.set_distribution(nbin=1, mode="proportional")

    model.dust._dust_fields["density_bin_0"] = Field(
        quantity="dust_density",
        data=Quantity(rho_zyx, "g/cm^3"),
        axis_order=("z", "y", "x"),
    )

    return model


def _expected_cartesian_flatten_xyz_fortran(*, nx: int, ny: int, nz: int) -> np.ndarray:
    out = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                out.append(ix + 10.0 * iy + 100.0 * iz)
    return np.asarray(out, dtype=np.float64)


def test_write_dust_density_cartesian_ascii_ordering(tmp_path) -> None:
    nx, ny, nz = 2, 3, 4

    rho_zyx = np.empty((nz, ny, nx), dtype=np.float64)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                rho_zyx[iz, iy, ix] = ix + 10.0 * iy + 100.0 * iz

    model = _make_cartesian_model_with_single_dust_bin(nx=nx, ny=ny, nz=nz, rho_zyx=rho_zyx)

    writer = RadWriter(model)
    writer.write_dust_density(output_dir=tmp_path, binary=False)

    path = tmp_path / writer.inputs_dir / "dust_density.inp"
    lines = path.read_text().strip().splitlines()

    assert int(lines[0]) == 1
    ncells = int(lines[1])
    nbin = int(lines[2])

    assert ncells == nx * ny * nz
    assert nbin == 1

    vals = np.array([float(x) for x in lines[3:]], dtype=np.float64)
    assert vals.size == ncells

    expected = _expected_cartesian_flatten_xyz_fortran(nx=nx, ny=ny, nz=nz)
    assert np.allclose(vals, expected, rtol=0.0, atol=0.0)


def test_write_dust_density_cartesian_binary_ordering(tmp_path) -> None:
    nx, ny, nz = 2, 3, 4

    rho_zyx = np.empty((nz, ny, nx), dtype=np.float64)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                rho_zyx[iz, iy, ix] = ix + 10.0 * iy + 100.0 * iz

    model = _make_cartesian_model_with_single_dust_bin(nx=nx, ny=ny, nz=nz, rho_zyx=rho_zyx)

    writer = RadWriter(model)
    writer.write_dust_density(output_dir=tmp_path, binary=True)

    path = tmp_path / writer.inputs_dir / "dust_density.binp"
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int64, count=4)
        data = np.fromfile(f, dtype=np.float64)

    assert header[0] == 1
    assert header[1] == 8
    assert int(header[2]) == nx * ny * nz
    assert int(header[3]) == 1

    expected = _expected_cartesian_flatten_xyz_fortran(nx=nx, ny=ny, nz=nz)
    assert data.size == expected.size
    assert np.allclose(data, expected, rtol=0.0, atol=0.0)
