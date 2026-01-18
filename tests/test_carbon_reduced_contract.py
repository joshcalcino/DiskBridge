from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel
from diskbridge._units import Quantity, units

from diskbridge.chemistry.api import run_chemistry, run_thermochemistry
from diskbridge.chemistry.thermal import run_thermal
 

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def _build_radmodel_1d(n_cells: int = 32, *, nH_cm3: float = 1.0e3, chi: float = 10.0, Tdust_K: float = 20.0) -> RadModel:
    m_H = units("m_H")

    x_centers = np.linspace(0.5, float(n_cells) - 0.5, int(n_cells))

    mesh = Mesh.cartesian(
        x=Axis(centers=Quantity(x_centers, "cm")),
        y=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
        z=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
    )

    shape = mesh.shape

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    rho = (Quantity(np.full(shape, float(nH_cm3)), "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=("x", "y", "z")))

    sigma = Quantity(np.full(shape, 1e-21), "cm^2")
    model.gas_register("sigma_d_per_H", Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")))

    rad = RadModel(model)
    rad.dust_temperature = Quantity(np.full(shape, float(Tdust_K)), "K")
    rad.chi = Quantity(np.full(shape, float(chi)), "dimensionless")

    return rad


def _as_flat_f64(q: Quantity, unit: str) -> np.ndarray:
    return np.ascontiguousarray(q.to(unit).magnitude.reshape(-1), dtype=np.float64)


def test_carbon_reduced_populates_required_fields() -> None:
    rad = _build_radmodel_1d(n_cells=16, nH_cm3=1.0e4, chi=3.0, Tdust_K=30.0)

    run_chemistry(rad, model="carbon_reduced", config={"skip_shielding": True}, write=False)

    for name in ("nH2", "nH_atom", "nCplus", "nC", "ne", "nco_gas"):
        assert getattr(rad, name, None) is not None

    assert rad.nco_ice is not None

    for q, unit in (
        (rad.nH2, "cm^-3"),
        (rad.nH_atom, "cm^-3"),
        (rad.nCplus, "cm^-3"),
        (rad.nC, "cm^-3"),
        (rad.ne, "cm^-3"),
        (rad.nco_gas, "cm^-3"),
        (rad.nco_ice, "cm^-3"),
    ):
        arr = _as_flat_f64(q, unit)
        assert np.all(np.isfinite(arr))
        assert np.min(arr) >= 0.0


def test_outer_loop_wiring_carbon_thermal_carbon_changes_closure() -> None:
    rad = _build_radmodel_1d(n_cells=32, nH_cm3=3.0e3, chi=30.0, Tdust_K=15.0)

    run_chemistry(rad, model="carbon_reduced", config={"skip_shielding": True}, write=False)

    nCplus_0 = _as_flat_f64(rad.nCplus, "cm^-3")
    nC_0 = _as_flat_f64(rad.nC, "cm^-3")
    ne_0 = _as_flat_f64(rad.ne, "cm^-3")

    assert rad.gas_temperature is None

    therm = run_thermal(rad, model="thermal_balance", config={}, write=False)
    assert therm is not None
    assert rad.gas_temperature is not None

    run_chemistry(rad, model="carbon_reduced", config={"skip_shielding": True}, write=False)

    nCplus_1 = _as_flat_f64(rad.nCplus, "cm^-3")
    nC_1 = _as_flat_f64(rad.nC, "cm^-3")
    ne_1 = _as_flat_f64(rad.ne, "cm^-3")

    changed = (
        (not np.allclose(nCplus_0, nCplus_1, rtol=0.0, atol=0.0))
        or (not np.allclose(nC_0, nC_1, rtol=0.0, atol=0.0))
        or (not np.allclose(ne_0, ne_1, rtol=0.0, atol=0.0))
    )
    assert changed


def test_run_thermochemistry_convergence_metrics_reasonable() -> None:
    rad = _build_radmodel_1d(n_cells=32, nH_cm3=1.0e5, chi=1.0, Tdust_K=20.0)

    _chem, therm = run_thermochemistry(
        rad,
        chemistry_model="carbon_reduced",
        chemistry_config={"skip_shielding": True},
        thermal_model="thermal_balance",
        thermal_config={},
        n_iter=5,
        convergence={"tgas_rtol": 1e-3},
        write=False,
    )

    assert therm is not None
    assert int(therm.meta.get("thermochemistry_n_iter")) <= 5

    metrics = therm.meta.get("tgas_rel_metrics", None)
    assert metrics is not None
    assert len(metrics) >= 1
    assert np.all(np.isfinite(np.asarray(metrics, dtype=float)))

    if len(metrics) >= 2:
        assert float(metrics[-1]) <= float(metrics[0])
