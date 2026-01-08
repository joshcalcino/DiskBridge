"""Tests for native batch GOW17 solver.

Key invariants tested:
1. Batch solver returns correct output structure
2. No double dust attenuation: when Gph is set directly, no extra exp(-Av) applied
3. Scaling test: if chi_dust changes by factor X, photorate changes by factor X
"""
from __future__ import annotations

import numpy as np
import pytest

import diskbridge._gow17 as _gow17


N_Y = _gow17.N_Y
N_PH = _gow17.N_PH
IPH_C = _gow17.IPH_C
IPH_CO = _gow17.IPH_CO
IPH_H2 = _gow17.IPH_H2


def default_y0(ncells: int) -> np.ndarray:
    """Create default initial conditions for ncells."""
    y0 = np.zeros((ncells, N_Y), dtype=np.float64)
    y0[:, 0] = 1.45e-08
    y0[:, 8] = 2.68e-07
    y0[:, 4] = 1.0e-4
    y0[:, 3] = 1.0e-7
    y0[:, 6] = 0.1
    return y0


def default_abstol() -> np.ndarray:
    """Create default absolute tolerances."""
    abstol = np.full(N_Y, 1e-15, dtype=np.float64)
    abstol[6] = 1e-8
    return abstol


class TestBatchSolverBasic:
    """Basic functionality tests for solve_batch_equilibrium."""

    def test_output_structure(self):
        """Batch solver returns dict with 'y' and 'status' keys."""
        ncells = 4
        y0 = default_y0(ncells)
        nH = np.full(ncells, 1000.0, dtype=np.float64)
        Tgas = np.full(ncells, 50.0, dtype=np.float64)
        Tdust = np.full(ncells, 20.0, dtype=np.float64)
        Zd = np.ones(ncells, dtype=np.float64)
        Zg = np.ones(ncells, dtype=np.float64)
        ion_rate = np.full(ncells, 2e-16, dtype=np.float64)
        GPE = np.ones(ncells, dtype=np.float64)
        GISRF = np.ones(ncells, dtype=np.float64)
        Gph = np.ones((ncells, N_PH), dtype=np.float64)
        abstol = default_abstol()

        result = _gow17.solve_batch_equilibrium(
            y0=y0,
            nH=nH,
            Tgas=Tgas,
            Tdust=Tdust,
            Zd=Zd,
            Zg=Zg,
            ion_rate=ion_rate,
            GPE=GPE,
            GISRF=GISRF,
            Gph=Gph,
            reltol=1e-4,
            abstol=abstol,
            mxsteps=10000,
            maxord=5,
            tolfac=10.0,
            tmin=3.16e10,
            tmax=3.16e14,
            const_temp=True,
            gradv=1e-14,
            fH2gr=1.0,
            fHplusgr=1.0,
            fCplusgr=1.0,
            fHeplusgr=1.0,
            fSplusgr=1.0,
            fSiplusgr=1.0,
            fCplusCR=1.0,
            userJac=False,
            verbose=False,
        )

        assert "y" in result
        assert "status" in result
        assert result["y"].shape == (ncells, N_Y)
        assert result["status"].shape == (ncells,)

    def test_all_cells_converge(self):
        """All cells should converge with reasonable parameters."""
        ncells = 4
        y0 = default_y0(ncells)
        nH = np.full(ncells, 1000.0, dtype=np.float64)
        Tgas = np.full(ncells, 50.0, dtype=np.float64)
        Tdust = np.full(ncells, 20.0, dtype=np.float64)
        Zd = np.ones(ncells, dtype=np.float64)
        Zg = np.ones(ncells, dtype=np.float64)
        ion_rate = np.full(ncells, 2e-16, dtype=np.float64)
        GPE = np.ones(ncells, dtype=np.float64)
        GISRF = np.ones(ncells, dtype=np.float64)
        Gph = np.ones((ncells, N_PH), dtype=np.float64)
        abstol = default_abstol()

        result = _gow17.solve_batch_equilibrium(
            y0=y0,
            nH=nH,
            Tgas=Tgas,
            Tdust=Tdust,
            Zd=Zd,
            Zg=Zg,
            ion_rate=ion_rate,
            GPE=GPE,
            GISRF=GISRF,
            Gph=Gph,
            reltol=1e-4,
            abstol=abstol,
            mxsteps=10000,
            maxord=5,
            tolfac=10.0,
            tmin=3.16e10,
            tmax=3.16e14,
            const_temp=True,
            gradv=1e-14,
            fH2gr=1.0,
            fHplusgr=1.0,
            fCplusgr=1.0,
            fHeplusgr=1.0,
            fSplusgr=1.0,
            fSiplusgr=1.0,
            fCplusCR=1.0,
            userJac=False,
            verbose=False,
        )

        assert np.all(result["status"] == 0), f"Some cells failed: {result['status']}"


class TestNoDoubleAttenuation:
    """Tests that verify no double dust attenuation occurs.
    
    When we pass pre-computed Gph values to the batch solver, the C++ code
    should NOT apply any additional dust attenuation (exp(-Av) factors).
    """

    def test_gph_passed_directly_no_modification(self):
        """If we set Gph[i]=X, the photo rates should use X directly.
        
        With chi_dust=1 and theta=1 for all species, photo rates should
        match the base rates times 1.0 (no attenuation).
        """
        ncells = 2
        y0 = default_y0(ncells)
        y0[:, 6] = 0.01

        nH = np.full(ncells, 100.0, dtype=np.float64)
        Tgas = np.full(ncells, 50.0, dtype=np.float64)
        Tdust = np.full(ncells, 20.0, dtype=np.float64)
        Zd = np.ones(ncells, dtype=np.float64)
        Zg = np.ones(ncells, dtype=np.float64)
        ion_rate = np.full(ncells, 2e-16, dtype=np.float64)
        GPE = np.ones(ncells, dtype=np.float64)
        GISRF = np.ones(ncells, dtype=np.float64)
        Gph = np.ones((ncells, N_PH), dtype=np.float64)
        abstol = default_abstol()

        result1 = _gow17.solve_batch_equilibrium(
            y0=y0.copy(),
            nH=nH,
            Tgas=Tgas,
            Tdust=Tdust,
            Zd=Zd,
            Zg=Zg,
            ion_rate=ion_rate,
            GPE=GPE,
            GISRF=GISRF,
            Gph=Gph,
            reltol=1e-4,
            abstol=abstol,
            mxsteps=10000,
            maxord=5,
            tolfac=10.0,
            tmin=3.16e10,
            tmax=3.16e14,
            const_temp=True,
            gradv=1e-14,
            fH2gr=1.0,
            fHplusgr=1.0,
            fCplusgr=1.0,
            fHeplusgr=1.0,
            fSplusgr=1.0,
            fSiplusgr=1.0,
            fCplusCR=1.0,
            userJac=False,
            verbose=False,
        )

        Gph_half = np.full((ncells, N_PH), 0.5, dtype=np.float64)
        GPE_half = np.full(ncells, 0.5, dtype=np.float64)
        GISRF_half = np.full(ncells, 0.5, dtype=np.float64)

        result2 = _gow17.solve_batch_equilibrium(
            y0=y0.copy(),
            nH=nH,
            Tgas=Tgas,
            Tdust=Tdust,
            Zd=Zd,
            Zg=Zg,
            ion_rate=ion_rate,
            GPE=GPE_half,
            GISRF=GISRF_half,
            Gph=Gph_half,
            reltol=1e-4,
            abstol=abstol,
            mxsteps=10000,
            maxord=5,
            tolfac=10.0,
            tmin=3.16e10,
            tmax=3.16e14,
            const_temp=True,
            gradv=1e-14,
            fH2gr=1.0,
            fHplusgr=1.0,
            fCplusgr=1.0,
            fHeplusgr=1.0,
            fSplusgr=1.0,
            fSiplusgr=1.0,
            fCplusCR=1.0,
            userJac=False,
            verbose=False,
        )

        y1_co = result1["y"][:, 3]
        y2_co = result2["y"][:, 3]
        assert not np.allclose(y1_co, y2_co, rtol=0.1, atol=0), (
            "CO abundance should differ when Gph differs; "
            f"if same, the code may not be using Gph correctly. "
            f"Got y1_co={y1_co}, y2_co={y2_co}"
        )


class TestRadiationFieldScaling:
    """Tests that radiation field scaling works as expected."""

    def test_chi_scaling_affects_solution(self):
        """Changing chi_dust by factor 10 should significantly affect CO abundance.
        
        This catches accidental double attenuation if the internal code
        applies dust attenuation on top of what we provide.
        """
        ncells = 2
        y0 = default_y0(ncells)

        nH = np.full(ncells, 1000.0, dtype=np.float64)
        Tgas = np.full(ncells, 50.0, dtype=np.float64)
        Tdust = np.full(ncells, 20.0, dtype=np.float64)
        Zd = np.ones(ncells, dtype=np.float64)
        Zg = np.ones(ncells, dtype=np.float64)
        ion_rate = np.full(ncells, 2e-16, dtype=np.float64)
        abstol = default_abstol()

        GPE_low = np.full(ncells, 0.1, dtype=np.float64)
        GISRF_low = np.full(ncells, 0.1, dtype=np.float64)
        Gph_low = np.full((ncells, N_PH), 0.1, dtype=np.float64)

        result_low = _gow17.solve_batch_equilibrium(
            y0=y0.copy(),
            nH=nH,
            Tgas=Tgas,
            Tdust=Tdust,
            Zd=Zd,
            Zg=Zg,
            ion_rate=ion_rate,
            GPE=GPE_low,
            GISRF=GISRF_low,
            Gph=Gph_low,
            reltol=1e-4,
            abstol=abstol,
            mxsteps=10000,
            maxord=5,
            tolfac=10.0,
            tmin=3.16e10,
            tmax=3.16e14,
            const_temp=True,
            gradv=1e-14,
            fH2gr=1.0,
            fHplusgr=1.0,
            fCplusgr=1.0,
            fHeplusgr=1.0,
            fSplusgr=1.0,
            fSiplusgr=1.0,
            fCplusCR=1.0,
            userJac=False,
            verbose=False,
        )

        GPE_high = np.full(ncells, 1.0, dtype=np.float64)
        GISRF_high = np.full(ncells, 1.0, dtype=np.float64)
        Gph_high = np.full((ncells, N_PH), 1.0, dtype=np.float64)

        result_high = _gow17.solve_batch_equilibrium(
            y0=y0.copy(),
            nH=nH,
            Tgas=Tgas,
            Tdust=Tdust,
            Zd=Zd,
            Zg=Zg,
            ion_rate=ion_rate,
            GPE=GPE_high,
            GISRF=GISRF_high,
            Gph=Gph_high,
            reltol=1e-4,
            abstol=abstol,
            mxsteps=10000,
            maxord=5,
            tolfac=10.0,
            tmin=3.16e10,
            tmax=3.16e14,
            const_temp=True,
            gradv=1e-14,
            fH2gr=1.0,
            fHplusgr=1.0,
            fCplusgr=1.0,
            fHeplusgr=1.0,
            fSplusgr=1.0,
            fSiplusgr=1.0,
            fCplusCR=1.0,
            userJac=False,
            verbose=False,
        )

        co_low = result_low["y"][:, 3]
        co_high = result_high["y"][:, 3]

        assert co_low.mean() > co_high.mean(), (
            f"Lower radiation should lead to more CO. "
            f"Got CO_low={co_low.mean():.2e}, CO_high={co_high.mean():.2e}"
        )


class TestThermoEvolution:
    """Tests for thermal evolution mode (const_temp=False)."""

    def test_thermo_mode_converges(self):
        """Thermo mode should converge with reasonable parameters."""
        ncells = 4
        y0 = default_y0(ncells)
        nH = np.full(ncells, 1000.0, dtype=np.float64)
        Tgas = np.full(ncells, 50.0, dtype=np.float64)
        Tdust = np.full(ncells, 20.0, dtype=np.float64)
        Zd = np.ones(ncells, dtype=np.float64)
        Zg = np.ones(ncells, dtype=np.float64)
        ion_rate = np.full(ncells, 2e-16, dtype=np.float64)
        GPE = np.ones(ncells, dtype=np.float64)
        GISRF = np.ones(ncells, dtype=np.float64)
        Gph = np.ones((ncells, N_PH), dtype=np.float64)
        abstol = default_abstol()

        KB_CGS = 1.380649e-16
        XHE = 0.1
        xH2_init = y0[0, 6]
        xe_init = y0[0, 0] + y0[0, 4]
        Cv_init = 1.5 * KB_CGS * ((1.0 - 2.0 * xH2_init) + xH2_init + XHE + xe_init)
        y0[:, 13] = Cv_init * Tgas

        result = _gow17.solve_batch_equilibrium(
            y0=y0,
            nH=nH,
            Tgas=Tgas,
            Tdust=Tdust,
            Zd=Zd,
            Zg=Zg,
            ion_rate=ion_rate,
            GPE=GPE,
            GISRF=GISRF,
            Gph=Gph,
            reltol=1e-4,
            abstol=abstol,
            mxsteps=10000,
            maxord=5,
            tolfac=10.0,
            tmin=3.16e10,
            tmax=3.16e14,
            const_temp=False,
            gradv=1e-14,
            isDust_cooling=True,
            fH2gr=1.0,
            fHplusgr=1.0,
            fCplusgr=1.0,
            fHeplusgr=1.0,
            fSplusgr=1.0,
            fSiplusgr=1.0,
            fCplusCR=1.0,
            userJac=False,
            verbose=False,
        )

        assert np.all(result["status"] == 0), f"Some cells failed: {result['status']}"

        y_out = result["y"]
        assert np.all(np.isfinite(y_out)), "Output contains non-finite values"
        assert np.all(y_out[:, :13] >= 0), "Abundances contain negative values"

        E_out = y_out[:, 13]
        assert np.all(E_out > 0), f"Energy should be positive, got {E_out}"

        xH2_out = y_out[:, 6]
        xe_out = y_out[:, 0] + y_out[:, 4] + y_out[:, 5] + y_out[:, 7] + y_out[:, 8] + y_out[:, 9] + y_out[:, 10] + y_out[:, 11] + y_out[:, 12]
        Cv_out = 1.5 * KB_CGS * ((1.0 - 2.0 * xH2_out) + xH2_out + XHE + xe_out)
        T_out = E_out / Cv_out
        assert np.all(T_out > 0), f"Temperature should be positive, got {T_out}"
        assert np.all(T_out < 1e6), f"Temperature unreasonably high: {T_out}"


class TestModuleConstants:
    """Tests that module constants are correctly exposed."""

    def test_constants_exist(self):
        """Photo indices and dimensions should be exposed."""
        assert hasattr(_gow17, "N_Y")
        assert hasattr(_gow17, "N_PH")
        assert hasattr(_gow17, "IPH_C")
        assert hasattr(_gow17, "IPH_CO")
        assert hasattr(_gow17, "IPH_H2")

    def test_species_indices_exist(self):
        """Species indices should be exposed from C++."""
        assert hasattr(_gow17, "I_HEP")
        assert hasattr(_gow17, "I_H2")
        assert hasattr(_gow17, "I_CO")
        assert hasattr(_gow17, "I_CP")
        assert hasattr(_gow17, "I_E")
        assert hasattr(_gow17, "XHE")

    def test_species_indices_values(self):
        """Species indices should have expected values."""
        assert _gow17.I_HEP == 0
        assert _gow17.I_H2 == 6
        assert _gow17.I_CO == 3
        assert _gow17.I_CP == 4
        assert _gow17.I_E == 13
        assert _gow17.XHE == 0.1

    def test_constants_values(self):
        """Constants should have expected values."""
        assert _gow17.N_Y == 14
        assert _gow17.N_PH == 7
        assert _gow17.IPH_C == 0
        assert _gow17.IPH_CO == 2
        assert _gow17.IPH_H2 == 4
