from __future__ import annotations

import numpy as np

import diskbridge
from diskbridge._units import Quantity
from diskbridge.chemistry.api import run_chemistry
from diskbridge.chemistry.models._gow17_network import N_Y, I_CO, I_CO_ICE, I_H2
from diskbridge.chemistry.models._gow17_numba import _fd_jacobian, gow17_rhs_cgs, newton_solve_fd
from diskbridge.chemistry.shielding.columns_1d import compute_pdr_shielding_1d
from diskbridge.chemistry.shielding.columns_1d import column_to_outer_boundary_1d, effective_1d_axis
from diskbridge.chemistry.shielding.visser_shielding import VisserShielding
from diskbridge.model.core import Model, SubModel
from diskbridge.model.field import Field
from diskbridge.model.mesh import Axis, Mesh
from diskbridge.radmc3d.model import RadModel


def main() -> None:
    au = diskbridge.units.au
    m_H = diskbridge.units("m_H")

    nx = 64
    nH0_cm3 = 1.0e4
    sigma_d_per_H_cm2 = 1.0e-21
    tau_d_max = 12.0
    N_H_max = tau_d_max / sigma_d_per_H_cm2
    x_max_cm = N_H_max / nH0_cm3
    x_edges = np.linspace(0.0, x_max_cm, nx + 1) * diskbridge.units.cm
    y_edges = np.array([0.0, 1.0]) * au
    z_edges = np.array([0.0, 1.0]) * au

    mesh = Mesh.cartesian(x=Axis(edges=x_edges), y=Axis(edges=y_edges), z=Axis(edges=z_edges))
    shape = mesh.shape

    model = Model()
    model.coord_system = mesh.coord_system
    model.mesh = mesh
    model.gas = SubModel(model)

    nH_cm3 = (np.full(nx, nH0_cm3, dtype=float)).reshape(shape)

    rho = (Quantity(nH_cm3, "cm^-3") * (1.4 * m_H)).to("g/cm^3")
    model.gas_register("density", Field(quantity="density", data=rho, axis_order=("x", "y", "z")))

    sigma = Quantity(np.full(shape, 1e-21, dtype=float), "cm^2")
    model.gas_register(
        "sigma_d_per_H",
        Field(quantity="sigma_d_per_H", data=sigma, axis_order=("x", "y", "z")),
    )

    radm = RadModel(model)

    radm.dust_temperature = Quantity(np.full(shape, 50.0, dtype=float), "K")

    axis_name, axis_index = effective_1d_axis(mesh, shape)
    nH_field = nH_cm3
    N_H = column_to_outer_boundary_1d(
        mesh,
        nH_field,
        axis_name=axis_name,
        axis_index=axis_index,
        outer="max",
    )
    tau_d = sigma_d_per_H_cm2 * N_H
    chi0 = 1.0e3
    chi_dust = chi0 * np.exp(-tau_d)
    radm.chi = Quantity(chi_dust, "dimensionless")

    print("=== gow17_pdr equilibrium ===")
    radm.nco_gas = Quantity(1e-12 * nH_cm3, "cm^-3")

    res_eq = None
    for _ in range(4):
        res_eq = run_chemistry(
            radm,
            model="gow17_pdr",
            config={
                "mode": "equilibrium",
                "nside": 1,
                "max_iter": 80,
                "b_kms": 0.3,
            },
        )
        radm.nco_gas = res_eq.number_densities["co"]

    print("equilibrium n_fail=", res_eq.meta.get("n_fail"), "max_status=", res_eq.meta.get("max_status"))

    if int(res_eq.meta.get("n_fail", 0)) != 0:
        # Diagnostics: identify which radial cells fail (theta=phi=1 so index is (ir,0,0))
        nH = radm.ensure_nH().to("cm^-3").magnitude
        Tdust = radm.ensure_dust_temperature().to("K").magnitude
        chi = radm.ensure_chi().to("dimensionless").magnitude
        sigma = radm.ensure_sigma_d_per_H().to("cm^2").magnitude

        # Build a consistent initial guess matching gow17_pdr y0
        y0 = np.zeros(N_Y, dtype=np.float64)
        y0[I_H2] = 0.1
        y0[I_CO] = 1.0e-7
        y0[I_CO_ICE] = 0.0
        y0[4] = 1.0e-4
        y0[0] = 1.450654e-08
        y0[8] = 2.681411e-07

        # Shielding inputs for equilibrium branch
        visser = VisserShielding(b_kms=0.3)
        nCO = (y0[I_CO] * nH).astype(np.float64)
        nC = (1.0e-4 * nH).astype(np.float64)
        nH2 = (y0[I_H2] * nH).astype(np.float64)

        th_h2, th_co, th_c, chi_eff = compute_pdr_shielding_1d(
            mesh=radm.model.mesh,
            nH=nH,
            chi=chi,
            visser=visser,
            nCO=nCO,
            nC=nC,
            nH2=nH2,
            b_kms=0.3,
            outer="max",
            return_quantity=False,
        )

        # Run per-cell Newton and report failures
        fail_cells = []
        for ix in range(nx):
            y_init = y0.copy()
            y_sol = np.empty(N_Y, dtype=np.float64)
            st = newton_solve_fd(
                y_init,
                0,
                y_prev=y_init,
                dt_s=0.0,
                nH_cm3=float(nH[ix, 0, 0]),
                T_K=float(Tdust[ix, 0, 0]),
                chi=float(chi[ix, 0, 0]),
                theta_h2=float(th_h2[ix, 0, 0]),
                theta_co=float(th_co[ix, 0, 0]),
                theta_c=float(th_c[ix, 0, 0]),
                ion_rate_s=1.0e-17,
                sigma_d_per_H_cm2=float(sigma[ix, 0, 0]),
                Tdust_K=float(Tdust[ix, 0, 0]),
                chi_eff_pdr=float(chi_eff[ix, 0, 0]),
                Zg=1.0,
                Zd=1.0,
                fH2gr=1.0,
                fHplusgr=1.0,
                fCplusgr=1.0,
                fHeplusgr=1.0,
                fSplusgr=1.0,
                fSiplusgr=1.0,
                fCplusCR=1.0,
                max_iter=60,
                reltol=1.0e-2,
                abstol=np.full(N_Y, 1.0e-9, dtype=np.float64),
                out_y=y_sol,
            )
            if int(st) != 0:
                fail_cells.append((ix, int(st)))

        print("equilibrium failing cells (ir, status):", fail_cells)

        # If we have a singular Jacobian failure, compute a diagnostic Jacobian for the first one.
        for ix, st in fail_cells:
            if st != 2:
                continue
            y = y0.copy()
            f = np.empty(N_Y, dtype=np.float64)
            gow17_rhs_cgs(
                y,
                nH_cm3=float(nH[ix, 0, 0]),
                T_K=float(Tdust[ix, 0, 0]),
                chi=float(chi[ix, 0, 0]),
                theta_h2=float(th_h2[ix, 0, 0]),
                theta_co=float(th_co[ix, 0, 0]),
                theta_c=float(th_c[ix, 0, 0]),
                ion_rate_s=1.0e-17,
                sigma_d_per_H_cm2=float(sigma[ix, 0, 0]),
                Tdust_K=float(Tdust[ix, 0, 0]),
                chi_eff_pdr=float(chi_eff[ix, 0, 0]),
                Zg=1.0,
                Zd=1.0,
                fH2gr=1.0,
                fHplusgr=1.0,
                fCplusgr=1.0,
                fHeplusgr=1.0,
                fSplusgr=1.0,
                fSiplusgr=1.0,
                fCplusCR=1.0,
                out_rhs=f,
            )
            J = np.empty((N_Y, N_Y), dtype=np.float64)
            _fd_jacobian(
                y,
                f,
                0,
                y_prev=y,
                dt_s=0.0,
                nH_cm3=float(nH[ix, 0, 0]),
                T_K=float(Tdust[ix, 0, 0]),
                chi=float(chi[ix, 0, 0]),
                theta_h2=float(th_h2[ix, 0, 0]),
                theta_co=float(th_co[ix, 0, 0]),
                theta_c=float(th_c[ix, 0, 0]),
                ion_rate_s=1.0e-17,
                sigma_d_per_H_cm2=float(sigma[ix, 0, 0]),
                Tdust_K=float(Tdust[ix, 0, 0]),
                chi_eff_pdr=float(chi_eff[ix, 0, 0]),
                Zg=1.0,
                Zd=1.0,
                fH2gr=1.0,
                fHplusgr=1.0,
                fCplusgr=1.0,
                fHeplusgr=1.0,
                fSplusgr=1.0,
                fSiplusgr=1.0,
                fCplusCR=1.0,
                reltol=1.0e-2,
                abstol=np.full(N_Y, 1.0e-9, dtype=np.float64),
                J_out=J,
            )
            row_norms = np.sum(np.abs(J), axis=1)
            print("diag: ir=", ix, "nH=", float(nH[ix, 0, 0]), "row_norms=", row_norms.tolist())
            break

    print("=== gow17_pdr time_dependent ===")
    res_td = run_chemistry(
        radm,
        model="gow17_pdr",
        config={
            "mode": "time_dependent",
            "nside": 1,
            "max_iter": 80,
            "b_kms": 0.3,
            "t_end": "3e4 yr",
            "dt": "3e2 yr",
            "shielding_update_every": 1,
        },
    )
    print("time_dependent n_fail=", res_td.meta.get("n_fail"), "max_status=", res_td.meta.get("max_status"))

    if int(res_eq.meta.get("n_fail", 999)) != 0:
        raise RuntimeError("equilibrium did not converge")

    for name in ("co", "c+", "catom", "e", "h2", "h"):
        arr = res_eq.number_densities[name].to("cm^-3").magnitude
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name}: non-finite")
        if not np.all(arr >= 0.0):
            raise RuntimeError(f"{name}: negative")

    try:
        import matplotlib.pyplot as plt
        from pathlib import Path

        nH_td = radm.ensure_nH().to("cm^-3").magnitude
        axis_name, axis_index = effective_1d_axis(mesh, shape)
        N_H = column_to_outer_boundary_1d(
            mesh,
            nH_td,
            axis_name=axis_name,
            axis_index=axis_index,
            outer="max",
        )

        tau_d = (sigma_d_per_H_cm2 * N_H).reshape(shape)

        xcoord_au = mesh.axes["x"].centers.to("au").magnitude
        depth_au = float(mesh.edges("x").to("au").magnitude[-1]) - xcoord_au

        nco = res_eq.number_densities["co"].to("cm^-3").magnitude.reshape(shape)
        ncplus = res_eq.number_densities["c+"].to("cm^-3").magnitude.reshape(shape)
        ncatom = res_eq.number_densities["catom"].to("cm^-3").magnitude.reshape(shape)
        nh2 = res_eq.number_densities["h2"].to("cm^-3").magnitude.reshape(shape)
        nh = res_eq.number_densities["h"].to("cm^-3").magnitude.reshape(shape)

        x_co = (nco / nH_td).reshape(shape)
        x_cplus = (ncplus / nH_td).reshape(shape)
        x_c = (ncatom / nH_td).reshape(shape)
        x_h2 = (2.0 * nh2 / nH_td).reshape(shape)
        x_h = (nh / nH_td).reshape(shape)

        fig, ax = plt.subplots(1, 1, figsize=(7, 4.2), dpi=140)
        ax.semilogy(tau_d[:, 0, 0], x_h2[:, 0, 0], label="H2")
        ax.semilogy(tau_d[:, 0, 0], x_h[:, 0, 0], label="H")
        ax.semilogy(tau_d[:, 0, 0], x_cplus[:, 0, 0], label="C+")
        ax.semilogy(tau_d[:, 0, 0], x_c[:, 0, 0], label="C")
        ax.semilogy(tau_d[:, 0, 0], x_co[:, 0, 0], label="CO")
        ax.set_xlabel("tau_dust to +x boundary")
        ax.set_ylabel("abundance per H nucleus")
        ax.set_ylim(1e-12, 2.0)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        out = Path("visualization_tests") / "gow17_pdr_1d_cartesian.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4.2), dpi=140)
        ax2.semilogy(N_H[:, 0, 0], x_h2[:, 0, 0], label="H2")
        ax2.semilogy(N_H[:, 0, 0], x_h[:, 0, 0], label="H")
        ax2.semilogy(N_H[:, 0, 0], x_cplus[:, 0, 0], label="C+")
        ax2.semilogy(N_H[:, 0, 0], x_c[:, 0, 0], label="C")
        ax2.semilogy(N_H[:, 0, 0], x_co[:, 0, 0], label="CO")
        ax2.set_xlabel("N_H to +x boundary [cm^-2]")
        ax2.set_ylabel("abundance per H nucleus")
        ax2.set_ylim(1e-12, 2.0)
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc="best", fontsize=9)
        out2 = Path("visualization_tests") / "gow17_pdr_1d_cartesian_vs_NH.png"
        fig2.tight_layout()
        fig2.savefig(out2)
        plt.close(fig2)

    except ModuleNotFoundError:
        pass

    print("PASS")


if __name__ == "__main__":
    main()
