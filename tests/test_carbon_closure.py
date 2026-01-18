from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diskbridge._constants import X_C_TOT, GAMMA_C0, ALPHA_REC_C0, T_REC_EXP
from diskbridge.chemistry.processes.carbon_closure import (
    carbon_closure_cell_cgs,
    carbon_closure_cell_param_cgs,
)


def test_carbon_closure_single_cell_sanity_and_limits() -> None:
    nH = 1.0e4
    Tg = 100.0
    nco_total = 0.2 * float(X_C_TOT) * nH

    nCplus, nC, ne = carbon_closure_cell_param_cgs(
        nH_cm3=float(nH),
        chi=1.0,
        Tg_K=float(Tg),
        nco_total_cm3=float(nco_total),
        X_C_tot=float(X_C_TOT),
        Gamma_C0=float(GAMMA_C0),
        alpha_rec_c0=float(ALPHA_REC_C0),
        T_rec_exp=float(T_REC_EXP),
    )

    nC_tot = float(X_C_TOT) * nH
    assert 0.0 <= nC <= nC_tot
    assert 0.0 <= nCplus <= nC_tot
    assert ne == pytest.approx(nCplus, rel=0.0, abs=0.0)
    assert (nC + nCplus + nco_total) == pytest.approx(nC_tot, rel=0.0, abs=1e-12 * nC_tot)

    nCplus0, nC0, ne0 = carbon_closure_cell_cgs(
        nH_cm3=float(nH),
        chi=0.0,
        Tg_K=float(Tg),
        nco_total_cm3=float(nco_total),
    )
    assert nCplus0 == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert ne0 == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert (nC0 + nco_total) == pytest.approx(nC_tot, rel=0.0, abs=1e-12 * nC_tot)

    nCplus_hi, nC_hi, ne_hi = carbon_closure_cell_cgs(
        nH_cm3=float(nH),
        chi=1.0e12,
        Tg_K=float(Tg),
        nco_total_cm3=float(nco_total),
    )
    assert nCplus_hi >= nCplus
    assert ne_hi == pytest.approx(nCplus_hi, rel=0.0, abs=0.0)
    assert (nC_hi + nCplus_hi + nco_total) == pytest.approx(nC_tot, rel=0.0, abs=1e-12 * nC_tot)


def test_carbon_closure_monotonic_vs_chi() -> None:
    nH = 1.0e4
    Tg = 100.0
    nco_total = 0.2 * float(X_C_TOT) * nH

    chi_grid = np.logspace(-6.0, 6.0, 200)
    nCplus = np.empty_like(chi_grid)
    for i, chi in enumerate(chi_grid):
        nCplus[i] = carbon_closure_cell_cgs(
            nH_cm3=float(nH),
            chi=float(chi),
            Tg_K=float(Tg),
            nco_total_cm3=float(nco_total),
        )[0]

    diffs = np.diff(nCplus)
    assert np.min(diffs) >= -1e-12 * float(X_C_TOT) * nH

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
