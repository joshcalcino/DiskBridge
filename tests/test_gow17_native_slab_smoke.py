from __future__ import annotations

import numpy as np


def test_gow17_native_slab_smoke() -> None:
    import diskbridge._gow17 as gow17_native
    from diskbridge.chemistry.models._gow17_network import (
        I_CHX,
        I_CO,
        I_CP,
        I_H2,
        I_H2P,
        I_H3P,
        I_HCOP,
        I_HEP,
        I_HP,
        I_OHX,
        I_OP,
        I_SIP,
        I_SP,
        I_CO_ICE,
        N_Y,
    )

    y0 = np.zeros(N_Y, dtype=float)
    y0[I_HEP] = 1.450654e-08
    y0[I_H3P] = 2.681411e-07
    y0[I_CP] = 1.0e-4
    y0[I_CO] = 1.0e-7
    y0[I_H2] = 0.1
    y0[I_CO_ICE] = 0.0

    abstol = np.full(N_Y, 1.0e-9, dtype=float)
    abstol[I_HEP] = 1.0e-15
    abstol[I_OHX] = 1.0e-15
    abstol[I_CHX] = 1.0e-15
    abstol[I_CO] = 1.0e-15
    abstol[I_CP] = 1.0e-15
    abstol[I_HCOP] = 1.0e-30
    abstol[I_H2] = 1.0e-8
    abstol[I_HP] = 1.0e-15
    abstol[I_H3P] = 1.0e-15
    abstol[I_H2P] = 1.0e-15
    abstol[I_SP] = 1.0e-15
    abstol[I_SIP] = 1.0e-15
    abstol[I_OP] = 1.0e-15

    ngrid = 32
    NH_min = 1.0e17
    NH_total = 1.0e22

    out = gow17_native.solve_slab_1d_equilibrium(
        nH=100.0,
        G0=2.0,
        ngrid=int(ngrid),
        NH_total=float(NH_total),
        logNH=True,
        NH_min=float(NH_min),
        field_geo=0,
        isdust=True,
        isfsH2=True,
        isfsCO=True,
        isfsC=True,
        Zg=1.0,
        Zd=1.0,
        ion_rate=2.0e-16,
        reltol=1.0e-2,
        abstol=abstol,
        mxsteps=200000,
        maxord=3,
        tolfac=10.0,
        tmin=3.16e10,
        tmax=3.16e13,
        verbose=False,
        y0=y0,
        const_temp=True,
        Tgas=100.0,
        gradv=3.0e-14,
        NCOeff_global=True,
        bCO_L=True,
        fH2gr=1.0,
        fHplusgr=0.6,
        fCplusgr=0.6,
        fHeplusgr=0.6,
        fSplusgr=0.6,
        fSiplusgr=0.6,
        fCplusCR=1.0,
        userJac=False,
    )

    assert set(out.keys()) == {"NH", "y", "fShieldH2", "fShieldCO", "GPE", "dimen"}

    NH = np.asarray(out["NH"], dtype=float)
    y = np.asarray(out["y"], dtype=float)

    assert NH.shape == (ngrid,)
    assert y.shape == (ngrid, N_Y)

    assert np.isfinite(NH).all()
    assert np.isfinite(y).all()

    assert float(NH[0]) == float(NH_min)
    assert np.isclose(float(NH[-1]), float(NH_total), rtol=0.0, atol=1.0)

    assert float(np.min(y)) >= 0.0

    # Mass-fraction sanity: H2 is per-H abundance (<= 0.5)
    assert float(np.max(y[:, I_H2])) <= 0.5 + 1.0e-12
