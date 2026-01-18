from __future__ import annotations

import numpy as np
import pytest

from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel
from diskbridge._units import Quantity, units
from diskbridge.chemistry.api import run_chemistry, run_thermochemistry
from diskbridge.chemistry.thermal import run_thermal


def _build_minimal_radmodel() -> RadModel:
    m_H = units("m_H")

    shape = (2, 1, 1)

    mesh = Mesh.cartesian(
        x=Axis(centers=Quantity(np.array([0.25, 0.75]), "cm")),
        y=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
        z=Axis(edges=Quantity(np.array([0.0, 1.0]), "cm")),
    )

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    nH_cm3 = 1.0e4
    rho = (Quantity(np.full(shape, nH_cm3), "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=("x", "y", "z")))

    sigma = Quantity(np.full(shape, 1e-21), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")),
    )

    rad = RadModel(model)

    rad.dust_temperature = Quantity(np.full(shape, 30.0), "K")
    rad.chi = Quantity(np.full(shape, 1.0), "dimensionless")

    return rad


def test_thermal_requires_chemistry_outputs() -> None:
    rad = _build_minimal_radmodel()

    with pytest.raises(RuntimeError, match=r"thermal_balance requires chemistry outputs"):
        run_thermal(rad, model="thermal_balance", write=False)


def test_run_thermochemistry_provides_closure_and_updates_temperature() -> None:
    rad = _build_minimal_radmodel()

    chem = run_chemistry(rad, model="carbon_reduced", config={"skip_shielding": True}, write=False)

    assert getattr(rad, "nCplus", None) is not None
    assert getattr(rad, "nC", None) is not None
    assert getattr(rad, "ne", None) is not None
    assert getattr(rad, "nH2", None) is not None
    assert getattr(rad, "nH_atom", None) is not None

    t0 = float(rad.dust_temperature.to("K").magnitude.flat[0])

    _, therm = run_thermochemistry(
        rad,
        chemistry_model="carbon_reduced",
        chemistry_config={"skip_shielding": True},
        thermal_model="thermal_balance",
        thermal_config={},
        n_iter=2,
        convergence=None,
        write=False,
    )

    assert therm is not None
    assert rad.gas_temperature is not None

    t1 = float(rad.gas_temperature.to("K").magnitude.flat[0])
    assert t1 != t0
