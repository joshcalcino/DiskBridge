from __future__ import annotations

from pathlib import Path

import numpy as np

from diskbridge.chemistry.processes.co_phase import (
    co_freezeout_rate_cgs,
    co_thermal_desorption_rate_cgs,
    co_photodesorption_surface_rate_cgs,
)
 
import pytest


def test_co_phase_rates_qualitative_behavior() -> None:
    nH = 1.0e6
    sigma = 1.0e-21

    k_frz_10 = co_freezeout_rate_cgs(nH_cm3=float(nH), Tgas_K=10.0, sigma_d_per_H_cm2=float(sigma))
    k_frz_100 = co_freezeout_rate_cgs(nH_cm3=float(nH), Tgas_K=100.0, sigma_d_per_H_cm2=float(sigma))

    assert k_frz_10 > 0.0
    assert k_frz_100 > 0.0
    assert k_frz_100 > k_frz_10

    k_des_10 = co_thermal_desorption_rate_cgs(Tdust_K=10.0)
    k_des_50 = co_thermal_desorption_rate_cgs(Tdust_K=50.0)
    k_des_100 = co_thermal_desorption_rate_cgs(Tdust_K=100.0)

    assert k_des_10 >= 0.0
    assert k_des_50 > k_des_10
    assert k_des_100 > k_des_50

    k_pd_1 = co_photodesorption_surface_rate_cgs(chi=1.0)
    k_pd_10 = co_photodesorption_surface_rate_cgs(chi=10.0)

    assert k_pd_1 > 0.0
    assert k_pd_10 > k_pd_1

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_dir = Path(__file__).parent.parent / "visualization_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    Td = np.linspace(5.0, 120.0, 400)
    k_des = np.array([co_thermal_desorption_rate_cgs(float(t)) for t in Td], dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogy(Td, np.maximum(k_des, 1e-300), color="k")
    ax.set_xlabel("Tdust [K]")
    ax.set_ylabel("k_des [1/s]")
    fig.tight_layout()
    fig.savefig(out_dir / "co_desorption_rate_vs_T.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
