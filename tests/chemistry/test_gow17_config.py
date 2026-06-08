"""Tests for GOW17 configuration, scaling, diagnostics, and budget contracts.

These tests use synthetic model fields and small solver inputs to protect
deterministic chemistry API behavior and simple objective physical invariants.
"""

# db-keywords: shielding, uv-products, photodesorption, gow17, gas-temperature, dust-gas-coupling, validation, config, units, radmc3d, chemistry, model, mesh
# db-role: validation
# db-scope: test
# db-purpose: Tests for GOW17 configuration, scaling, diagnostics, and budget contracts.

import numpy as np
import pytest
from types import SimpleNamespace

from diskbridge._config import resolve_model_config
from diskbridge._constants import M_H
from diskbridge._units import Quantity
from diskbridge.chemistry.validation import (
    gow17_budget_diagnostics,
    project_gow17_state_to_budgets,
)
from diskbridge.chemistry.models.gow17 import (
    I_CHX,
    I_CO,
    I_CO_ICE,
    I_CP,
    I_H2P,
    I_H3P,
    I_HCOP,
    I_HEP,
    I_HP,
    I_H2,
    I_OHX,
    I_OP,
    I_SIP,
    I_SP,
    GOW17_STATE_SPECIES,
    KPH_CO_BASE,
    N_PH,
    N_Y,
    PAH_MASS_G_DEFAULT,
    PAH_X_ISM_DEFAULT,
    XHE,
    _accumulate_solver_status,
    _append_equilibrium_solver_diagnostics,
    _actual_solver_uv_fields,
    _build_gow17_abstol,
    _compute_co_phase_diagnostics,
    _compute_shielding_and_gph,
    _gas_dust_exchange,
    _gow17_species_outputs,
    _resolve_h2_grain_scaling,
    _resolve_pah_scaling,
    _resolve_dust_cooling_controls,
    _resolve_temperature_config,
    _summarize_equilibrium_solver_diagnostics,
)
from diskbridge.model.core import Model, SubModel
from diskbridge.model.dust import Dust
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel


def test_default_gow17_config_enables_co_phase():
    cfg = resolve_model_config(("chemistry", "gow17"), overrides={})

    assert cfg["enable_co_phase"] is True


def test_default_gow17_temperature_mode_is_computed():
    cfg = resolve_model_config(("chemistry", "gow17"), overrides={})
    temp = _resolve_temperature_config(cfg)

    assert temp["mode"] == "computed"
    assert temp["initial"] == "dust"
    assert temp["const_temp"] is False


def test_default_gow17_uses_surface_area_dust_cooling():
    cfg = resolve_model_config(("chemistry", "gow17"), overrides={})

    assert cfg["dust_cooling"]["mode"] == "surface_area"
    assert cfg["dust_cooling"]["sigma_d_H_ref"] == "1.0e-21 cm^2"


def test_gas_dust_exchange_sign_and_zero_surface_area():
    nH = np.array([1.0e8, 1.0e8, 1.0e8])
    Zgd = np.array([1.0, 1.0, 0.0])
    Tg = np.array([20.0, 5.0, 20.0])
    Td = np.array([10.0, 10.0, 10.0])

    gdust = _gas_dust_exchange(Zgd=Zgd, nH_cm3=nH, Tgas_K=Tg, Tdust_K=Td)

    assert gdust[0] > 0.0
    assert gdust[1] < 0.0
    assert gdust[2] == 0.0


def test_surface_area_mode_reference_matches_gow17_original_scaling():
    class FakeRad:
        def compute_gas_dust_surface_area_coupling(self):
            from diskbridge._units import Quantity

            return (
                Quantity(np.array([1.0e-21, 0.0]), "cm^2"),
                Quantity(np.array([10.0, 99.0]), "K"),
            )

    cfg = {
        "dust_cooling": {
            "mode": "surface_area",
            "sigma_d_H_ref": "1.0e-21 cm^2",
            "gow17_original_Zd": 1.0,
            "gow17_original_Tdust": "10 K",
        }
    }

    mode, Zgd, Tdust, sigma, sigma_ref = _resolve_dust_cooling_controls(
        cfg=cfg,
        rad=FakeRad(),
        ncells=2,
    )

    assert mode == "surface_area"
    assert sigma_ref == pytest.approx(1.0e-21)
    np.testing.assert_allclose(sigma, [1.0e-21, 0.0])
    np.testing.assert_allclose(Zgd, [1.0, 0.0])
    np.testing.assert_allclose(Tdust, [10.0, 10.0])


def test_gow17_original_dust_cooling_uses_constants():
    cfg = {
        "dust_cooling": {
            "mode": "gow17_original",
            "sigma_d_H_ref": "1.0e-21 cm^2",
            "gow17_original_Zd": 2.5,
            "gow17_original_Tdust": "11 K",
        }
    }

    mode, Zgd, Tdust, sigma, _ = _resolve_dust_cooling_controls(
        cfg=cfg,
        rad=object(),
        ncells=3,
    )

    assert mode == "gow17_original"
    np.testing.assert_allclose(Zgd, 2.5)
    np.testing.assert_allclose(Tdust, 11.0)
    np.testing.assert_allclose(sigma, 0.0)


class _FakeFieldModel:
    def __init__(self, gas=None, dust=None):
        self.gas = gas
        self.dust = dust


class _FakeRad:
    def __init__(self, gas=None, dust=None, axis_order=("cell",)):
        self.model = _FakeFieldModel(gas=gas, dust=dust)
        self._axis_order = axis_order

    def _chem_axis_order(self):
        return self._axis_order


def test_pah_scaling_falls_back_to_zd_without_explicit_fields():
    Zd = np.array([1.0, 0.2, 0.0], dtype=np.float64)

    D_pah, rho_pah, meta = _resolve_pah_scaling(
        rad=_FakeRad(gas={}, dust=None),
        nH_flat=np.ones(3, dtype=np.float64),
        Zd_arr=Zd,
        ncells=3,
    )

    np.testing.assert_allclose(D_pah, Zd)
    assert rho_pah is None
    assert meta["pah_source"] == "gow17_original_Zd"
    assert meta["pah_uses_explicit_component"] is False


def test_pah_scaling_from_density_field():
    nH = np.array([1.0e4, 2.0e4], dtype=np.float64)
    expected = np.array([0.01, 0.5], dtype=np.float64)
    rho_pah = nH * PAH_X_ISM_DEFAULT * PAH_MASS_G_DEFAULT * expected
    gas = {
        "pah_density": Field(
            quantity="pah_density",
            data=Quantity(rho_pah, "g/cm^3"),
            axis_order=("cell",),
        )
    }

    D_pah, rho_pah_out, meta = _resolve_pah_scaling(
        rad=_FakeRad(gas=gas, dust=None),
        nH_flat=nH,
        Zd_arr=np.ones(2, dtype=np.float64),
        ncells=2,
    )

    np.testing.assert_allclose(D_pah, expected)
    np.testing.assert_allclose(rho_pah_out, rho_pah)
    assert meta["pah_source"] == "gas.pah_density"
    assert meta["pah_uses_explicit_component"] is True


def test_pah_scaling_rejects_inconsistent_abundance_and_density():
    nH = np.array([1.0e4], dtype=np.float64)
    rho_pah = nH * PAH_X_ISM_DEFAULT * PAH_MASS_G_DEFAULT * 0.5
    gas = {
        "pah_abundance_rel_ism": Field(
            quantity="pah_abundance_rel_ism",
            data=Quantity(np.array([0.01]), "dimensionless"),
            axis_order=("cell",),
        ),
        "pah_density": Field(
            quantity="pah_density",
            data=Quantity(rho_pah, "g/cm^3"),
            axis_order=("cell",),
        ),
    }

    with pytest.raises(ValueError, match="inconsistent"):
        _resolve_pah_scaling(
            rad=_FakeRad(gas=gas, dust=None),
            nH_flat=nH,
            Zd_arr=np.ones(1, dtype=np.float64),
            ncells=1,
        )


def test_add_pah_to_disk_uses_smooth_disk_weight_and_registers_gas_fields():
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.arange(4.0), "cm")),
        theta=Axis(edges=Quantity(np.arange(2.0), "radian")),
        phi=Axis(edges=Quantity(np.arange(2.0), "radian")),
    )
    model.coord_system = model.mesh.coord_system
    model.gas = SubModel(model)
    model.dust = Dust(model)

    shape = model.mesh.shape
    nH = np.full(shape, 1.0e4, dtype=np.float64)
    rho = nH * 1.4 * M_H
    disk_weight = np.array([0.0, 1.0, 0.5], dtype=np.float64).reshape(shape)
    model.gas_register(
        "density",
        Field(
            quantity="density",
            data=Quantity(rho, "g/cm^3"),
            axis_order=model.mesh.axis_names(),
        ),
    )
    model.gas_register(
        "disk_weight",
        Field(
            quantity="disk_weight",
            data=Quantity(disk_weight, "dimensionless"),
            axis_order=model.mesh.axis_names(),
        ),
    )

    model.dust.add_pah_to_disk(f_pah_disk=0.01)

    D_pah = model.gas["pah_abundance_rel_ism"].data.to("dimensionless").magnitude
    np.testing.assert_allclose(D_pah.reshape(-1), [1.0, 0.01, 0.505])
    assert "pah_density" in model.gas
    assert "pah_density" not in model.dust


def _h2gr_surface_test_model(mask_name: str = "disk_mask") -> Model:
    model = Model()
    model.mesh = Mesh.spherical(
        r=Axis(edges=Quantity(np.arange(4.0), "cm")),
        theta=Axis(edges=Quantity(np.arange(2.0), "radian")),
        phi=Axis(edges=Quantity(np.arange(2.0), "radian")),
    )
    model.coord_system = model.mesh.coord_system
    model.gas = SubModel(model)
    model.dust = Dust(model)

    shape = model.mesh.shape
    nH = np.full(shape, 1.0e4, dtype=np.float64)
    rho = nH * 1.4 * M_H
    disk_mask = np.array([0.0, 1.0, 0.5], dtype=np.float64).reshape(shape)
    axis_order = model.mesh.axis_names()
    model.gas_register(
        "density",
        Field(
            quantity="density",
            data=Quantity(rho, "g/cm^3"),
            axis_order=axis_order,
        ),
    )
    model.gas_register(
        mask_name,
        Field(
            quantity=mask_name,
            data=Quantity(disk_mask, "dimensionless"),
            axis_order=axis_order,
        ),
    )
    model.dust.add_component_from_mask(
        mask_name,
        amin=Quantity(0.9, "um"),
        amax=Quantity(1.1, "um"),
        nbin=1,
        dust_to_gas_ratio=0.01,
    )
    model.dust.add_component_from_mask(
        mask_name,
        complement=True,
        amin=Quantity(0.09, "um"),
        amax=Quantity(0.11, "um"),
        nbin=1,
        dust_to_gas_ratio=0.01,
    )
    return model


def test_h2gr_surface_area_normalizes_to_ism_complement():
    model = _h2gr_surface_test_model()
    rad = RadModel(model)

    Dh2gr, sigma, sigma_ref, meta = _resolve_h2_grain_scaling(
        rad=rad,
        Zd_arr=np.full(3, 9.0, dtype=np.float64),
        ncells=3,
    )

    assert sigma is not None
    assert sigma_ref is not None
    assert meta["h2gr_source"] == "ordinary_dust_surface_area"
    np.testing.assert_allclose(Dh2gr, [1.0, 0.1, 0.55], rtol=1.0e-12)


def test_h2gr_surface_area_accepts_disk_weight_complement():
    model = _h2gr_surface_test_model(mask_name="disk_weight")
    rad = RadModel(model)

    Dh2gr, sigma, sigma_ref, meta = _resolve_h2_grain_scaling(
        rad=rad,
        Zd_arr=np.full(3, 9.0, dtype=np.float64),
        ncells=3,
    )

    assert sigma is not None
    assert sigma_ref is not None
    assert meta["h2gr_source"] == "ordinary_dust_surface_area"
    np.testing.assert_allclose(Dh2gr, [1.0, 0.1, 0.55], rtol=1.0e-12)


def test_h2gr_surface_area_ignores_pah_fields():
    model = _h2gr_surface_test_model()
    rad = RadModel(model)
    sigma_before = rad.compute_h2_formation_surface_area().to("cm^2").magnitude

    model.dust.add_pah_to_disk(f_pah_disk=0.01)
    sigma_after = rad.compute_h2_formation_surface_area().to("cm^2").magnitude

    np.testing.assert_allclose(sigma_after, sigma_before)


def test_h2gr_scaling_falls_back_to_zd_without_dust():
    Zd = np.array([1.0, 0.2], dtype=np.float64)

    D, sigma, sigma_ref, meta = _resolve_h2_grain_scaling(
        rad=_FakeRad(gas={}, dust=None),
        Zd_arr=Zd,
        ncells=2,
    )

    np.testing.assert_allclose(D, Zd)
    assert sigma is None
    assert sigma_ref is None
    assert meta["h2gr_source"] == "gow17_original_Zd"


def test_gow17_species_outputs_include_every_state_species_and_derived_fields():
    y = np.zeros((2, N_Y), dtype=np.float64)
    for i, idx in enumerate(GOW17_STATE_SPECIES.values(), start=1):
        y[:, idx] = i * 1.0e-8
    nH = np.array([1.0e4, 2.0e4], dtype=np.float64)

    abundances, number_densities = _gow17_species_outputs(
        y,
        nH,
        x_h=np.array([0.9, 0.8]),
        x_catom=np.array([1.0e-5, 2.0e-5]),
        x_e=np.array([1.0e-7, 2.0e-7]),
    )

    expected = set(GOW17_STATE_SPECIES) | {"h", "catom", "e"}
    expected_number_densities = expected | {"p-h2", "o-h2", "he"}
    assert set(abundances) == expected
    assert set(number_densities) == expected_number_densities
    np.testing.assert_allclose(
        number_densities["hco+"].magnitude,
        abundances["hco+"].magnitude * nH,
    )
    np.testing.assert_allclose(
        number_densities["p-h2"].magnitude,
        number_densities["h2"].magnitude / 4.0,
    )
    np.testing.assert_allclose(
        number_densities["o-h2"].magnitude,
        3.0 * number_densities["h2"].magnitude / 4.0,
    )
    np.testing.assert_allclose(number_densities["he"].magnitude, XHE * nH)


def test_temperature_dust_is_explicit_constant_mode():
    cfg = resolve_model_config(
        ("chemistry", "gow17"),
        overrides={"temperature": {"mode": "dust"}},
    )
    temp = _resolve_temperature_config(cfg)

    assert temp["mode"] == "dust"
    assert temp["initial"] == "dust"
    assert temp["const_temp"] is True


def test_species_specific_tolerance_overrides_build_solver_vector():
    abstol = _build_gow17_abstol(
        {
            "tolerances": {
                "abstol_default": 1.0e-18,
                "abstol_Heplus": 6.0e-22,
                "abstol_OHx": 7.0e-22,
                "abstol_CHx": 8.0e-22,
                "abstol_CO": 2.0e-20,
                "abstol_CO_ice": 3.0e-20,
                "abstol_Cplus": 4.0e-20,
                "abstol_HCOplus": 5.0e-22,
                "abstol_H2": 7.0e-10,
                "abstol_Hplus": 9.0e-22,
                "abstol_H3plus": 1.0e-21,
                "abstol_H2plus": 1.1e-21,
                "abstol_Splus": 1.2e-21,
                "abstol_Siplus": 1.3e-21,
                "abstol_Oplus": 1.4e-21,
            }
        },
        abstol0=1.0e-15,
    )

    assert abstol[I_CO] == pytest.approx(2.0e-20)
    assert abstol[I_CO_ICE] == pytest.approx(3.0e-20)
    assert abstol[I_CP] == pytest.approx(4.0e-20)
    assert abstol[I_HCOP] == pytest.approx(5.0e-22)
    assert abstol[I_HEP] == pytest.approx(6.0e-22)
    assert abstol[I_OHX] == pytest.approx(7.0e-22)
    assert abstol[I_CHX] == pytest.approx(8.0e-22)
    assert abstol[I_H2] == pytest.approx(7.0e-10)
    assert abstol[I_HP] == pytest.approx(9.0e-22)
    assert abstol[I_H3P] == pytest.approx(1.0e-21)
    assert abstol[I_H2P] == pytest.approx(1.1e-21)
    assert abstol[I_SP] == pytest.approx(1.2e-21)
    assert abstol[I_SIP] == pytest.approx(1.3e-21)
    assert abstol[I_OP] == pytest.approx(1.4e-21)


def test_explicit_disable_co_phase_remains_disabled_in_config():
    cfg = resolve_model_config(
        ("chemistry", "gow17"),
        overrides={"enable_co_phase": False},
    )

    assert cfg["enable_co_phase"] is False


def test_actual_solver_uv_fields_include_local_factor_and_shielding():
    shape = (2,)
    Gph = np.ones((2, N_PH), dtype=np.float64)
    Gph[:, 2] = [3.0, 7.0]
    Gph[:, 0] = [5.0, 11.0]
    Gph[:, 4] = [13.0, 17.0]
    Gph[:, 5] = [29.0, 31.0]
    Gph[:, 6] = [37.0, 41.0]
    Gph[:, 1] = [43.0, 47.0]
    Gph[:, 3] = [53.0, 59.0]
    F_external = np.array([19.0, 23.0], dtype=np.float64)

    actual = _actual_solver_uv_fields(
        Gph=Gph,
        F_CO_pdes_photon=F_external,
        shape=shape,
    )

    np.testing.assert_allclose(actual["G_CO_diss_actual"], [3.0, 7.0])
    np.testing.assert_allclose(actual["G_C_ion_actual"], [5.0, 11.0])
    np.testing.assert_allclose(actual["G_H2_diss_actual"], [13.0, 17.0])
    np.testing.assert_allclose(actual["G_S_ion_actual"], [29.0, 31.0])
    np.testing.assert_allclose(actual["G_Si_ion_actual"], [37.0, 41.0])
    np.testing.assert_allclose(actual["G_CH_diss_actual"], [43.0, 47.0])
    np.testing.assert_allclose(actual["G_OH_diss_actual"], [53.0, 59.0])
    np.testing.assert_allclose(actual["F_CO_pdes_photon"], [19.0, 23.0])


def test_co_phase_disabled_zeroes_phase_rates_and_keeps_photodissociation_diagnostic():
    from diskbridge.chemistry.models.gow17 import _resolve_co_phase_runtime_params

    cfg = resolve_model_config(("chemistry", "gow17"), overrides={})
    y = np.zeros((1, N_Y), dtype=np.float64)
    y[0, I_CO] = 2.0e-6
    y[0, I_CO_ICE] = 1.0e-5
    y[0, I_HEP] = 1.0e-8

    diag = _compute_co_phase_diagnostics(
        y_out=y,
        nH_cm3=np.array([1.0e6]),
        Tgas_K=np.array([20.0]),
        Tdust_K=np.array([10.0]),
        sigma_d_CO_per_H=np.array([1.0e-21]),
        S_CO=1.0,
        F_CO_pdes_photon=np.array([1.0e8]),
        F_CRUV_CO_pdes=np.array([1.0e4]),
        k_crdes_CO=np.array([1.0e-12]),
        G_CO_diss_actual=np.array([0.25]),
        G_C_ion_actual=np.array([0.5]),
        G_H2_diss_actual=np.array([0.75]),
        enable_co_phase=False,
        co_phase_params=_resolve_co_phase_runtime_params(cfg),
    )

    assert diag["sigma_d_CO_per_H"].magnitude[0] == 0.0
    assert diag["F_CRUV_CO_pdes"].magnitude[0] == 0.0
    assert diag["k_CO_crdes"].magnitude[0] == 0.0
    assert diag["k_CO_freezeout"].magnitude[0] == 0.0
    assert diag["F_CO_pdes_photon_total"].magnitude[0] == pytest.approx(1.0e8)
    assert diag["k_CO_photodiss"].magnitude[0] == pytest.approx(KPH_CO_BASE * 0.25)
    assert diag["CO_loss_photodiss_per_H"].magnitude[0] == pytest.approx(
        KPH_CO_BASE * 0.25 * 2.0e-6
    )


def test_radiation_product_normalization_has_no_hidden_half_factor():
    shape = (1,)
    theta_co = 0.2
    G_CO_diss = 6.0
    Gph = np.ones((1, N_PH), dtype=np.float64)
    Gph[0, 2] = G_CO_diss * theta_co

    actual = _actual_solver_uv_fields(
        Gph=Gph,
        F_CO_pdes_photon=np.array([1.0e8]),
        shape=shape,
    )

    assert actual["G_CO_diss_actual"][0] == pytest.approx(1.2)
    assert actual["G_CO_diss_actual"][0] != pytest.approx(0.6)


def test_co_pdes_radiation_assembly_keeps_photodesorption_flux_unshielded(monkeypatch):
    shape = (2,)
    ncells = 2
    theta_co = np.array([0.25, 1.0], dtype=np.float64)

    def fake_compute_pdr_shielding_healpix(**kwargs):
        return (
            np.ones(shape, dtype=np.float64),
            theta_co.reshape(shape),
            np.ones(shape, dtype=np.float64),
            None,
            None,
        )

    monkeypatch.setattr(
        "diskbridge.chemistry.shielding.healpix_columns.compute_pdr_shielding_healpix",
        fake_compute_pdr_shielding_healpix,
    )

    y = np.zeros((ncells, N_Y), dtype=np.float64)
    y[:, I_CO] = 1.0e-6
    y[:, I_H2] = 0.1
    F_raw = np.array([100.0, 200.0], dtype=np.float64)
    G_CO_diss = np.array([3.0, 4.0], dtype=np.float64)

    (
        theta_h2,
        theta_co_out,
        theta_c,
        Gph,
        GPE,
        F_pdes,
    ) = (
        _compute_shielding_and_gph(
            y_flat=y,
            nH_flat=np.full(ncells, 1.0e4, dtype=np.float64),
            chi_dust_flat=np.ones(ncells, dtype=np.float64),
            G_CO_diss_flat=G_CO_diss,
            G_H2_diss_flat=np.ones(ncells, dtype=np.float64),
            G_C_ion_flat=np.ones(ncells, dtype=np.float64),
            G_CO_pdes_flat=np.ones(ncells, dtype=np.float64),
            F_CO_pdes_photon_flat=F_raw,
            xCtot_flat=np.full(ncells, 1.0e-4, dtype=np.float64),
            Zd_arr=np.ones(ncells, dtype=np.float64),
            shape=shape,
            ncells=ncells,
            rad=SimpleNamespace(model=SimpleNamespace(mesh=object())),
            nH_cm3=np.full(shape, 1.0e4, dtype=np.float64),
            chi_dust_arr=np.ones(shape, dtype=np.float64),
            visser=None,
            b_H2_kms=1.0,
            b_CO_kms=1.0,
            b_H2_kms_grid=None,
            b_CO_kms_grid=None,
            nside=1,
        )
    )

    np.testing.assert_allclose(theta_co_out, theta_co)
    np.testing.assert_allclose(F_pdes, F_raw)
    np.testing.assert_allclose(Gph[:, 2], G_CO_diss * theta_co)
    np.testing.assert_allclose(GPE, np.ones(ncells))
    np.testing.assert_allclose(theta_h2, np.ones(ncells))
    np.testing.assert_allclose(theta_c, np.ones(ncells))


def test_co_pdes_diagnostics_use_photon_flux_plus_cruv():
    from diskbridge.chemistry.models.gow17 import _resolve_co_phase_runtime_params

    cfg = resolve_model_config(("chemistry", "gow17"), overrides={})
    params = _resolve_co_phase_runtime_params(cfg)
    y = np.zeros((1, N_Y), dtype=np.float64)
    y[0, I_CO] = 1.0e-8
    y[0, I_CO_ICE] = 1.0e-5

    def gain_for(F_ext, F_cruv):
        diag = _compute_co_phase_diagnostics(
            y_out=y,
            nH_cm3=np.array([1.0e6]),
            Tgas_K=np.array([20.0]),
            Tdust_K=np.array([10.0]),
            sigma_d_CO_per_H=np.array([1.0e-21]),
            S_CO=1.0,
            F_CO_pdes_photon=np.array([F_ext], dtype=np.float64),
            F_CRUV_CO_pdes=np.array([F_cruv], dtype=np.float64),
            k_crdes_CO=np.array([0.0]),
            G_CO_diss_actual=np.array([0.0]),
            G_C_ion_actual=np.array([0.0]),
            G_H2_diss_actual=np.array([0.0]),
            enable_co_phase=True,
            co_phase_params=params,
        )
        return diag

    raw = 100.0
    unshielded_diag = gain_for(raw, 0.0)

    assert unshielded_diag["F_CO_pdes_photon"].magnitude[0] == pytest.approx(raw)
    assert unshielded_diag["F_CO_pdes_photon_total"].magnitude[0] == pytest.approx(raw)

    cruv_diag = gain_for(raw, 40.0)
    assert cruv_diag["F_CO_pdes_photon_total"].magnitude[0] == pytest.approx(
        raw + 40.0
    )

    theta_zero_diag = gain_for(0.0, 40.0)
    assert theta_zero_diag["F_CO_pdes_photon"].magnitude[0] == pytest.approx(0.0)
    assert theta_zero_diag["F_CO_pdes_photon_total"].magnitude[0] == pytest.approx(40.0)


def test_status_accumulation_preserves_later_failure():
    status_acc = np.array([0, 1, 0], dtype=np.int32)
    status_step = np.array([-1, 0, 0], dtype=np.int32)

    out = _accumulate_solver_status(status_acc, status_step)

    np.testing.assert_array_equal(out, [-1, 1, 0])


def test_status_accumulation_preserves_warning_without_failure():
    status_acc = np.array([0, 2, 0], dtype=np.int32)
    status_step = np.array([3, 0, 0], dtype=np.int32)

    out = _accumulate_solver_status(status_acc, status_step)

    np.testing.assert_array_equal(out, [3, 2, 0])


def test_equilibrium_solver_diagnostics_are_preserved_and_summarized():
    hist = {
        "tevol_max_cells_hist": [],
        "tevol_max_residual_max_hist": [],
        "negative_abundance_cells_hist": [],
        "negative_abundance_corrections_hist": [],
        "cvode_failure_cells_hist": [],
        "exception_failure_cells_hist": [],
    }

    _append_equilibrium_solver_diagnostics(
        hist,
        {
            "tevol_max_cells": 2,
            "tevol_max_residual_max": 4.5,
            "negative_abundance_cells": 3,
            "negative_abundance_corrections": 7,
            "cvode_failure_cells": 1,
            "exception_failure_cells": 0,
        },
    )
    _append_equilibrium_solver_diagnostics(
        hist,
        {
            "tevol_max_cells": 0,
            "tevol_max_residual_max": 1.5,
            "negative_abundance_cells": 1,
            "negative_abundance_corrections": 2,
            "cvode_failure_cells": 0,
            "exception_failure_cells": 4,
        },
    )

    diag = _summarize_equilibrium_solver_diagnostics(hist)

    np.testing.assert_array_equal(diag["tevol_max_cells_hist"], [2, 0])
    np.testing.assert_allclose(diag["tevol_max_residual_max_hist"], [4.5, 1.5])
    assert diag["tevol_max_cells_total"] == 2
    assert diag["tevol_max_cells_max"] == 2
    assert diag["tevol_max_residual_max"] == pytest.approx(4.5)
    assert diag["negative_abundance_cells_total"] == 4
    assert diag["negative_abundance_corrections_total"] == 9
    assert diag["cvode_failure_cells_total"] == 1
    assert diag["exception_failure_cells_total"] == 4


def test_budget_projection_clips_negative_and_overbudget_species():
    y = np.zeros((2, N_Y), dtype=np.float64)
    y[0, I_CO] = -1.0e-5
    y[0, I_CP] = 3.0e-4
    y[0, I_CO_ICE] = 3.0e-4
    y[0, I_HCOP] = 3.0e-4
    y[0, I_CHX] = 3.0e-4
    y[0, I_OHX] = 5.0e-4
    y[0, I_OP] = 5.0e-4
    y[1, I_CO] = 1.0e-5

    xCtot = np.array([1.0e-4, 1.0e-4])
    xOtot = np.array([2.0e-4, 2.0e-4])

    y_proj = project_gow17_state_to_budgets(y, xCtot=xCtot, xOtot=xOtot)

    assert np.all(y_proj >= 0.0)
    c_sum = (
        y_proj[:, I_HCOP]
        + y_proj[:, I_CHX]
        + y_proj[:, I_CO]
        + y_proj[:, I_CO_ICE]
        + y_proj[:, I_CP]
    )
    o_sum = (
        y_proj[:, I_OHX]
        + y_proj[:, I_HCOP]
        + y_proj[:, I_CO]
        + y_proj[:, I_CO_ICE]
        + y_proj[:, I_OP]
    )
    assert np.all(c_sum <= xCtot * (1.0 + 1e-14))
    assert np.all(o_sum <= xOtot * (1.0 + 1e-14))


def test_oxygen_budget_diagnostics_report_valid_and_violating_states():
    y = np.zeros((2, N_Y), dtype=np.float64)
    y[0, I_CO] = 1.0e-5
    y[0, I_OHX] = 1.0e-5
    y[1, I_CO] = 3.0e-4
    y[1, I_CO_ICE] = 3.0e-4
    y[1, I_OP] = 3.0e-4

    diag = gow17_budget_diagnostics(
        y,
        xCtot=np.array([1.0e-3, 1.0e-3]),
        xOtot=np.array([1.0e-3, 5.0e-4]),
        rtol=1.0e-12,
    )

    assert diag["o_xO_neutral_min"] < 0.0
    assert diag["o_xO_accounted_max"] == pytest.approx(9.0e-4)
    assert diag["o_budget_violation"] == pytest.approx(4.0e-4)
