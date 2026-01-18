from __future__ import annotations

import numpy as np
import pytest

from diskbridge._units import Quantity
from diskbridge.chemistry.thermal import run_thermal


class _StubRad:
    def __init__(self):
        self.chi = Quantity(np.array([1.0]), "dimensionless")

    def ensure_nH(self) -> Quantity:
        return Quantity(np.array([1.0e4]), "cm^-3")

    def ensure_dust_temperature(self) -> Quantity:
        return Quantity(np.array([30.0]), "K")

    def ensure_chi(self) -> Quantity:
        return self.chi


def test_thermal_requires_chemistry_outputs_stub_rad() -> None:
    rad = _StubRad()

    with pytest.raises(RuntimeError, match=r"thermal_balance requires chemistry outputs"):
        run_thermal(rad, model="thermal_balance", write=False)
