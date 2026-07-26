"""Smoke tests for DiskBridge's public import surface.

These tests intentionally check only curated public imports and tiny public
object construction. They are not a private-module import census.
"""

# db-keywords: disk-mask, validation, config, units, radmc3d, model, mesh, field
# db-role: validation
# db-scope: test
# db-purpose: Smoke tests for DiskBridge's public import surface.

from __future__ import annotations

import diskbridge
from diskbridge import Model, Quantity, read_params, units
from diskbridge.chemistry import ChemistryResult, load_chemistry_outputs, run_chemistry
from diskbridge.model import (
    Field,
    Mesh,
    RadialGasBackground,
    apply_radial_amax,
    apply_radial_dust_transport,
    build_smoothed_radial_gas_background,
    set_mask_from_joos_disk,
)
from diskbridge.radmc3d import RadData, RadImage, RadModel, RadMolecule, RadWriter


def test_public_package_exports_are_available():
    """The curated top-level public API should import without side effects."""
    assert diskbridge.__version__
    assert Model is diskbridge.Model
    assert callable(read_params)
    assert units("cm") is not None
    assert Quantity(1.0, "cm").to("cm").magnitude == 1.0


def test_public_subpackage_exports_are_available():
    """Core public exports from main subpackages should remain reachable."""
    assert Field is not None
    assert Mesh is not None
    assert set_mask_from_joos_disk is not None
    assert RadialGasBackground is not None
    assert callable(apply_radial_amax)
    assert callable(apply_radial_dust_transport)
    assert callable(build_smoothed_radial_gas_background)
    assert ChemistryResult is not None
    assert callable(load_chemistry_outputs)
    assert callable(run_chemistry)
    assert RadWriter is not None
    assert RadData is not None
    assert RadModel is not None
    assert RadMolecule is not None
    assert RadImage is not None
