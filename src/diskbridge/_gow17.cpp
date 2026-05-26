#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sundials/sundials_context.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cvodeDense.h"
#include "co_phase.h"
#include "gow17.h"
#include "slab.h"

namespace py = pybind11;

static constexpr int N_Y = 15;
static constexpr int N_PH = 7;
static constexpr int IPH_C = 0;
static constexpr int IPH_CH = 1;
static constexpr int IPH_CO = 2;
static constexpr int IPH_OH = 3;
static constexpr int IPH_H2 = 4;
static constexpr int IPH_S = 5;
static constexpr int IPH_SI = 6;

static bool is_cvode_integration_failure(const std::exception &e) {
    return std::string(e.what()).find("SUNDIALS:CVode failed with flag") != std::string::npos;
}

static py::dict solve_slab_1d_equilibrium(
    const double nH,
    const double G0,
    const long int ngrid,
    const double NH_total,
    const bool logNH,
    const double NH_min,
    const int field_geo,
    const bool isdust,
    const bool isfsH2,
    const bool isfsCO,
    const bool isfsC,
    const double Zg,
    const double Zd,
    const double ion_rate,
    const double reltol,
    const py::array_t<double, py::array::c_style | py::array::forcecast> abstol,
    const int mxsteps,
    const int maxord,
    const double tolfac,
    const double tmin,
    const double tmax,
    const bool verbose,
    const py::array_t<double, py::array::c_style | py::array::forcecast> y0,
    const bool const_temp,
    const double Tgas,
    const double gradv,
    const bool NCOeff_global,
    const bool bCO_L,
    const double fH2gr,
    const double fHplusgr,
    const double fCplusgr,
    const double fHeplusgr,
    const double fSplusgr,
    const double fSiplusgr,
    const double fCplusCR,

    const double co_sigma_d_per_H_ref,
    const double co_E_bind_co,
    const double co_nu0_co,
    const double co_F_DRAINE,
    const double co_Y_CO,
    const double co_N_SURF,
    const int co_N_LAY,
    const bool userJac) {
    gow17 ode;
    const int dim = ode.Dimen();

    if (abstol.ndim() != 1 || static_cast<int>(abstol.size()) != dim) {
        throw std::invalid_argument("abstol must be 1D with length == Dimen()");
    }

    if (y0.ndim() != 1 || static_cast<int>(y0.size()) != dim) {
        throw std::invalid_argument("y0 must be 1D with length == Dimen()");
    }

    CvodeDense solver(ode, reltol, abstol.data(), userJac);

    ode.SetInit(0.0, y0.data());
    ode.SetnH(nH);
    ode.SetIonRate(ion_rate);
    ode.SetZg(Zg);
    ode.SetZd(Zd);
    ode.SetTdust(Tgas);

    ode.SetfH2gr(fH2gr);
    ode.SetfHplusgr(fHplusgr);
    ode.SetfCplusgr(fCplusgr);
    ode.SetfHeplusgr(fHeplusgr);
    ode.SetfSplusgr(fSplusgr);
    ode.SetfSiplusgr(fSiplusgr);
    ode.SetfCplusCR(fCplusCR);

    ode.SetCOPhaseParams(
        co_sigma_d_per_H_ref,
        co_E_bind_co,
        co_nu0_co,
        co_Y_CO,
        co_N_SURF,
        co_N_LAY,
        1.0,
        0.0,
        0.0);

    ode.SetGradv(gradv);
    ode.SetNCOeffGlobal(NCOeff_global);
    ode.SetbCOL(bCO_L);

    if (const_temp) {
        ode.SetConstTemp(Tgas);
    }

    solver.ReInit();
    solver.SetMxsteps(mxsteps);
    solver.SetMaxOrd(maxord);

    Slab slab(ode, solver, ngrid, NH_total, G0, Zd, logNH, NH_min);
    slab.SetFieldGeo(field_geo);
    slab.IsH2MolSheilding(isfsH2);
    slab.IsCOMolSheilding(isfsCO);
    slab.IsCselfSheilding(isfsC);
    slab.IsDustSheilding(isdust);

    slab.SolveEq(tolfac, tmin, tmax, verbose, NULL);

    py::array_t<double> y(py::array::ShapeContainer{static_cast<py::ssize_t>(ngrid),
                                                    static_cast<py::ssize_t>(dim)});
    slab.CopyAbd(static_cast<double *>(y.mutable_data()));

    py::array_t<double> NH_arr(py::array::ShapeContainer{static_cast<py::ssize_t>(ngrid)});
    slab.CopyNH(static_cast<double *>(NH_arr.mutable_data()));

    py::array_t<double> fShieldH2(py::array::ShapeContainer{static_cast<py::ssize_t>(ngrid)});
    slab.CopyfShieldH2mol(static_cast<double *>(fShieldH2.mutable_data()));

    py::array_t<double> fShieldCO(py::array::ShapeContainer{static_cast<py::ssize_t>(ngrid)});
    slab.CopyfShieldCOmol(static_cast<double *>(fShieldCO.mutable_data()));

    py::array_t<double> GPE(py::array::ShapeContainer{static_cast<py::ssize_t>(ngrid)});
    slab.CopyGPE(static_cast<double *>(GPE.mutable_data()));

    py::dict out;
    out["NH"] = NH_arr;
    out["y"] = y;
    out["fShieldH2"] = fShieldH2;
    out["fShieldCO"] = fShieldCO;
    out["GPE"] = GPE;
    out["dimen"] = dim;
    return out;
}

static py::dict solve_batch_equilibrium(
    const py::array_t<double, py::array::c_style | py::array::forcecast> y0,
    const py::array_t<double, py::array::c_style | py::array::forcecast> nH,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Tgas,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Tdust,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zd,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zgd,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zg,
    const py::array_t<double, py::array::c_style | py::array::forcecast> ion_rate,
    const py::array_t<double, py::array::c_style | py::array::forcecast> GPE,
    const py::array_t<double, py::array::c_style | py::array::forcecast> F_CO_pdes_photon,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Gph,
    const py::array_t<double, py::array::c_style | py::array::forcecast> sigma_d_CO_per_H,
    const double reltol,
    const py::array_t<double, py::array::c_style | py::array::forcecast> abstol,
    const int mxsteps,
    const int maxord,
    const double tolfac,
    const double tmin,
    const double tmax,
    const bool const_temp,
    const py::array_t<double, py::array::c_style | py::array::forcecast> gradv,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Leff_CO_max,
    const bool isDust_cooling,
    const bool isCoolingCOThin,
    const double fH2gr,
    const double fHplusgr,
    const double fCplusgr,
    const double fHeplusgr,
    const double fSplusgr,
    const double fSiplusgr,
    const double fCplusCR,

    const double co_E_bind_co,
    const double co_nu0_co,
    const double co_Y_CO,
    const double co_N_SURF,
    const int co_N_LAY,
    const double co_S_CO,
    const py::array_t<double, py::array::c_style | py::array::forcecast> co_F_CRUV_CO_pdes,
    const py::array_t<double, py::array::c_style | py::array::forcecast> co_k_crdes_CO,
    const bool userJac,
    const bool verbose) {

    if (y0.ndim() != 2 || y0.shape(1) != N_Y) {
        throw std::invalid_argument("y0 must be 2D with shape (Ncells, 15)");
    }
    const py::ssize_t Ncells = y0.shape(0);

    if (nH.ndim() != 1 || nH.size() != Ncells) {
        throw std::invalid_argument("nH must be 1D with length Ncells");
    }
    if (Tgas.ndim() != 1 || Tgas.size() != Ncells) {
        throw std::invalid_argument("Tgas must be 1D with length Ncells");
    }
    if (Tdust.ndim() != 1 || Tdust.size() != Ncells) {
        throw std::invalid_argument("Tdust must be 1D with length Ncells");
    }
    if (Zd.ndim() != 1 || Zd.size() != Ncells) {
        throw std::invalid_argument("Zd must be 1D with length Ncells");
    }
    if (Zgd.ndim() != 1 || Zgd.size() != Ncells) {
        throw std::invalid_argument("Zgd must be 1D with length Ncells");
    }
    if (Zg.ndim() != 1 || Zg.size() != Ncells) {
        throw std::invalid_argument("Zg must be 1D with length Ncells");
    }
    if (ion_rate.ndim() != 1 || ion_rate.size() != Ncells) {
        throw std::invalid_argument("ion_rate must be 1D with length Ncells");
    }
    if (GPE.ndim() != 1 || GPE.size() != Ncells) {
        throw std::invalid_argument("GPE must be 1D with length Ncells");
    }
    if (F_CO_pdes_photon.ndim() != 1 || F_CO_pdes_photon.size() != Ncells) {
        throw std::invalid_argument("F_CO_pdes_photon must be 1D with length Ncells");
    }
    if (Gph.ndim() != 2 || Gph.shape(0) != Ncells || Gph.shape(1) != N_PH) {
        throw std::invalid_argument("Gph must be 2D with shape (Ncells, 7)");
    }
    if (abstol.ndim() != 1 || abstol.size() != N_Y) {
        throw std::invalid_argument("abstol must be 1D with length 15");
    }
    if (Leff_CO_max.ndim() != 1 || Leff_CO_max.size() != Ncells) {
        throw std::invalid_argument("Leff_CO_max must be 1D with length Ncells");
    }
    if (gradv.ndim() != 1 || gradv.size() != Ncells) {
        throw std::invalid_argument("gradv must be 1D with length Ncells");
    }
    if (sigma_d_CO_per_H.ndim() != 1 || sigma_d_CO_per_H.size() != Ncells) {
        throw std::invalid_argument("sigma_d_CO_per_H must be 1D with length Ncells");
    }
    if (co_F_CRUV_CO_pdes.ndim() != 1 || co_F_CRUV_CO_pdes.size() != Ncells) {
        throw std::invalid_argument("co_F_CRUV_CO_pdes must be 1D with length Ncells");
    }
    if (co_k_crdes_CO.ndim() != 1 || co_k_crdes_CO.size() != Ncells) {
        throw std::invalid_argument("co_k_crdes_CO must be 1D with length Ncells");
    }

    const double *y0_ptr = y0.data();
    const double *nH_ptr = nH.data();
    const double *Tgas_ptr = Tgas.data();
    const double *Tdust_ptr = Tdust.data();
    const double *Zd_ptr = Zd.data();
    const double *Zgd_ptr = Zgd.data();
    const double *Zg_ptr = Zg.data();
    const double *ion_rate_ptr = ion_rate.data();
    const double *GPE_ptr = GPE.data();
    const double *F_CO_pdes_photon_ptr = F_CO_pdes_photon.data();
    const double *Gph_ptr = Gph.data();
    const double *abstol_ptr = abstol.data();
    const double *Leff_CO_max_ptr = Leff_CO_max.data();
    const double *gradv_ptr = gradv.data();
    const double *sigma_d_CO_per_H_ptr = sigma_d_CO_per_H.data();
    const double *co_F_CRUV_CO_pdes_ptr = co_F_CRUV_CO_pdes.data();
    const double *co_k_crdes_CO_ptr = co_k_crdes_CO.data();

    py::array_t<double> y_out(py::array::ShapeContainer{Ncells, static_cast<py::ssize_t>(N_Y)});
    py::array_t<int> status_out(py::array::ShapeContainer{Ncells});
    double *y_out_ptr = static_cast<double *>(y_out.mutable_data());
    int *status_out_ptr = static_cast<int *>(status_out.mutable_data());

    std::atomic<int> n_fail{0};
    std::atomic<long long> n_cvode_fail{0};
    std::atomic<long long> n_other_exception_fail{0};
    std::atomic<long long> n_negative_abundance_cells{0};
    std::atomic<long long> n_negative_abundance_corrections{0};
    std::atomic<long long> n_tevol_max_cells{0};
    double tevol_max_residual_max = 0.0;

    /* Release the GIL so threads can run in parallel. */
    py::gil_scoped_release release;

    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t i = 0; i < Ncells; ++i) {
        gow17 ode;
        CvodeDense solver(ode, reltol, abstol_ptr, userJac);

        ode.SetInit(0.0, y0_ptr + i * N_Y);
        ode.SetnH(nH_ptr[i]);
        ode.SetIonRate(ion_rate_ptr[i]);
        ode.SetZg(Zg_ptr[i]);
        ode.SetZd(Zd_ptr[i]);
        ode.SetZgd(Zgd_ptr[i]);

        ode.SetfH2gr(fH2gr);
        ode.SetfHplusgr(fHplusgr);
        ode.SetfCplusgr(fCplusgr);
        ode.SetfHeplusgr(fHeplusgr);
        ode.SetfSplusgr(fSplusgr);
        ode.SetfSiplusgr(fSiplusgr);
        ode.SetfCplusCR(fCplusCR);
        ode.SetGradv(gradv_ptr[i]);
        ode.SetTdust(Tdust_ptr[i]);

        ode.SetCOPhaseParams(
            sigma_d_CO_per_H_ptr[i],
            co_E_bind_co,
            co_nu0_co,
            co_Y_CO,
            co_N_SURF,
            co_N_LAY,
            co_S_CO,
            co_F_CRUV_CO_pdes_ptr[i],
            co_k_crdes_CO_ptr[i]);

        ode.Leff_CO_max(Leff_CO_max_ptr[i]);
        ode.IsDustCooling(isDust_cooling);
        ode.SetCoolingCOThin(isCoolingCOThin);

        if (const_temp) {
            ode.SetConstTemp(Tgas_ptr[i]);
        }

        double GPE_cell = GPE_ptr[i];
        double F_CO_pdes_photon_cell = F_CO_pdes_photon_ptr[i];
        double Gph_cell[N_PH];
        for (int j = 0; j < N_PH; ++j) {
            Gph_cell[j] = Gph_ptr[i * N_PH + j];
        }
        ode.SetRadField(&GPE_cell, Gph_cell, &F_CO_pdes_photon_cell);

        solver.ReInit();
        solver.SetMxsteps(mxsteps);
        solver.SetMaxOrd(maxord);

        try {
            solver.SolveEq(tolfac, tmax, verbose, tmin);
            const long long n_negative = solver.GetNegativeCorrectionCount();
            if (n_negative > 0) {
                n_negative_abundance_cells.fetch_add(1, std::memory_order_relaxed);
                n_negative_abundance_corrections.fetch_add(
                    n_negative, std::memory_order_relaxed);
            }
            if (solver.ReachedTevolMax()) {
                n_tevol_max_cells.fetch_add(1, std::memory_order_relaxed);
                const double residual = solver.GetTevolMaxResidual();
                #pragma omp critical(gow17_tevol_max_residual)
                {
                    if (residual > tevol_max_residual_max) {
                        tevol_max_residual_max = residual;
                    }
                }
            }
            ode.CopyAbd(y_out_ptr + i * N_Y);
            status_out_ptr[i] = 0;
        } catch (const std::exception &e) {
            for (int j = 0; j < N_Y; ++j) {
                y_out_ptr[i * N_Y + j] = y0_ptr[i * N_Y + j];
            }
            status_out_ptr[i] = -1;
            n_fail.fetch_add(1, std::memory_order_relaxed);
            if (is_cvode_integration_failure(e)) {
                n_cvode_fail.fetch_add(1, std::memory_order_relaxed);
            } else {
                n_other_exception_fail.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    /* Re-acquire the GIL before touching Python objects. */
    py::gil_scoped_acquire acquire;

    if (n_cvode_fail.load() > 0) {
        printf("solve_batch_equilibrium: CVODE failed in %lld / %lld cells "
               "(per-cell SUNDIALS messages suppressed).\n",
               n_cvode_fail.load(), static_cast<long long>(Ncells));
    }
    if (n_other_exception_fail.load() > 0) {
        printf("solve_batch_equilibrium: %lld / %lld cells failed with non-CVODE exceptions.\n",
               n_other_exception_fail.load(), static_cast<long long>(Ncells));
    }
    if (n_negative_abundance_cells.load() > 0) {
        printf("solve_batch_equilibrium: corrected %lld / %lld cells with negative abundances "
               "(%lld species corrections total).\n",
               n_negative_abundance_cells.load(),
               static_cast<long long>(Ncells),
               n_negative_abundance_corrections.load());
    }
    if (n_tevol_max_cells.load() > 0) {
        printf("solve_batch_equilibrium: %lld / %lld cells did not reach equilibrium "
               "before tevol_max (max residual=%0.4e).\n",
               n_tevol_max_cells.load(),
               static_cast<long long>(Ncells),
               tevol_max_residual_max);
    }

    py::dict out;
    out["y"] = y_out;
    out["status"] = status_out;
    out["cvode_failure_cells"] = n_cvode_fail.load();
    out["exception_failure_cells"] = n_other_exception_fail.load();
    out["negative_abundance_cells"] = n_negative_abundance_cells.load();
    out["negative_abundance_corrections"] = n_negative_abundance_corrections.load();
    out["tevol_max_cells"] = n_tevol_max_cells.load();
    out["tevol_max_residual_max"] = tevol_max_residual_max;
    return out;
}

/* ---------------------------------------------------------------------------
 * solve_batch_time -- integrate each cell forward by t_end seconds.
 *
 * Same cell setup as solve_batch_equilibrium, but instead of calling
 * SolveEq() we call Solve(t_end) which integrates the CVODE system to an
 * absolute time t_end (starting from t=0).
 *
 * Returns dict with:
 *   "y"      : final abundances (Ncells, 15)
 *   "status" : per-cell status code (0 = ok, -1 = failure)
 * --------------------------------------------------------------------------- */
static py::dict solve_batch_time(
    const py::array_t<double, py::array::c_style | py::array::forcecast> y0,
    const py::array_t<double, py::array::c_style | py::array::forcecast> nH,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Tgas,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Tdust,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zd,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zgd,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zg,
    const py::array_t<double, py::array::c_style | py::array::forcecast> ion_rate,
    const py::array_t<double, py::array::c_style | py::array::forcecast> GPE,
    const py::array_t<double, py::array::c_style | py::array::forcecast> F_CO_pdes_photon,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Gph,
    const py::array_t<double, py::array::c_style | py::array::forcecast> sigma_d_CO_per_H,
    const double reltol,
    const py::array_t<double, py::array::c_style | py::array::forcecast> abstol,
    const int mxsteps,
    const int maxord,
    const double t_end,
    const bool const_temp,
    const py::array_t<double, py::array::c_style | py::array::forcecast> gradv,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Leff_CO_max,
    const bool isDust_cooling,
    const bool isCoolingCOThin,
    const double fH2gr,
    const double fHplusgr,
    const double fCplusgr,
    const double fHeplusgr,
    const double fSplusgr,
    const double fSiplusgr,
    const double fCplusCR,

    const double co_E_bind_co,
    const double co_nu0_co,
    const double co_Y_CO,
    const double co_N_SURF,
    const int co_N_LAY,
    const double co_S_CO,
    const py::array_t<double, py::array::c_style | py::array::forcecast> co_F_CRUV_CO_pdes,
    const py::array_t<double, py::array::c_style | py::array::forcecast> co_k_crdes_CO,
    const bool userJac,
    const bool verbose) {

    if (y0.ndim() != 2 || y0.shape(1) != N_Y) {
        throw std::invalid_argument("y0 must be 2D with shape (Ncells, 15)");
    }
    const py::ssize_t Ncells = y0.shape(0);

    if (nH.ndim() != 1 || nH.size() != Ncells) {
        throw std::invalid_argument("nH must be 1D with length Ncells");
    }
    if (Tgas.ndim() != 1 || Tgas.size() != Ncells) {
        throw std::invalid_argument("Tgas must be 1D with length Ncells");
    }
    if (Tdust.ndim() != 1 || Tdust.size() != Ncells) {
        throw std::invalid_argument("Tdust must be 1D with length Ncells");
    }
    if (Zd.ndim() != 1 || Zd.size() != Ncells) {
        throw std::invalid_argument("Zd must be 1D with length Ncells");
    }
    if (Zgd.ndim() != 1 || Zgd.size() != Ncells) {
        throw std::invalid_argument("Zgd must be 1D with length Ncells");
    }
    if (Zg.ndim() != 1 || Zg.size() != Ncells) {
        throw std::invalid_argument("Zg must be 1D with length Ncells");
    }
    if (ion_rate.ndim() != 1 || ion_rate.size() != Ncells) {
        throw std::invalid_argument("ion_rate must be 1D with length Ncells");
    }
    if (GPE.ndim() != 1 || GPE.size() != Ncells) {
        throw std::invalid_argument("GPE must be 1D with length Ncells");
    }
    if (F_CO_pdes_photon.ndim() != 1 || F_CO_pdes_photon.size() != Ncells) {
        throw std::invalid_argument("F_CO_pdes_photon must be 1D with length Ncells");
    }
    if (Gph.ndim() != 2 || Gph.shape(0) != Ncells || Gph.shape(1) != N_PH) {
        throw std::invalid_argument("Gph must be 2D with shape (Ncells, 7)");
    }
    if (abstol.ndim() != 1 || abstol.size() != N_Y) {
        throw std::invalid_argument("abstol must be 1D with length 15");
    }
    if (Leff_CO_max.ndim() != 1 || Leff_CO_max.size() != Ncells) {
        throw std::invalid_argument("Leff_CO_max must be 1D with length Ncells");
    }
    if (gradv.ndim() != 1 || gradv.size() != Ncells) {
        throw std::invalid_argument("gradv must be 1D with length Ncells");
    }
    if (sigma_d_CO_per_H.ndim() != 1 || sigma_d_CO_per_H.size() != Ncells) {
        throw std::invalid_argument("sigma_d_CO_per_H must be 1D with length Ncells");
    }
    if (co_F_CRUV_CO_pdes.ndim() != 1 || co_F_CRUV_CO_pdes.size() != Ncells) {
        throw std::invalid_argument("co_F_CRUV_CO_pdes must be 1D with length Ncells");
    }
    if (co_k_crdes_CO.ndim() != 1 || co_k_crdes_CO.size() != Ncells) {
        throw std::invalid_argument("co_k_crdes_CO must be 1D with length Ncells");
    }
    if (t_end <= 0.0) {
        throw std::invalid_argument("t_end must be > 0");
    }

    const double *y0_ptr = y0.data();
    const double *nH_ptr = nH.data();
    const double *Tgas_ptr = Tgas.data();
    const double *Tdust_ptr = Tdust.data();
    const double *Zd_ptr = Zd.data();
    const double *Zgd_ptr = Zgd.data();
    const double *Zg_ptr = Zg.data();
    const double *ion_rate_ptr = ion_rate.data();
    const double *GPE_ptr = GPE.data();
    const double *F_CO_pdes_photon_ptr = F_CO_pdes_photon.data();
    const double *Gph_ptr = Gph.data();
    const double *abstol_ptr = abstol.data();
    const double *Leff_CO_max_ptr = Leff_CO_max.data();
    const double *gradv_ptr = gradv.data();
    const double *sigma_d_CO_per_H_ptr = sigma_d_CO_per_H.data();
    const double *co_F_CRUV_CO_pdes_ptr = co_F_CRUV_CO_pdes.data();
    const double *co_k_crdes_CO_ptr = co_k_crdes_CO.data();

    py::array_t<double> y_out(py::array::ShapeContainer{Ncells, static_cast<py::ssize_t>(N_Y)});
    py::array_t<int> status_out(py::array::ShapeContainer{Ncells});
    double *y_out_ptr = static_cast<double *>(y_out.mutable_data());
    int *status_out_ptr = static_cast<int *>(status_out.mutable_data());

    std::atomic<int> n_fail{0};
    std::atomic<long long> n_cvode_fail{0};
    std::atomic<long long> n_other_exception_fail{0};
    std::atomic<long long> n_negative_abundance_cells{0};
    std::atomic<long long> n_negative_abundance_corrections{0};

    /* Release the GIL so threads can run in parallel. */
    py::gil_scoped_release release;

    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t i = 0; i < Ncells; ++i) {
        gow17 ode;
        CvodeDense solver(ode, reltol, abstol_ptr, userJac);

        ode.SetInit(0.0, y0_ptr + i * N_Y);
        ode.SetnH(nH_ptr[i]);
        ode.SetIonRate(ion_rate_ptr[i]);
        ode.SetZg(Zg_ptr[i]);
        ode.SetZd(Zd_ptr[i]);
        ode.SetZgd(Zgd_ptr[i]);

        ode.SetfH2gr(fH2gr);
        ode.SetfHplusgr(fHplusgr);
        ode.SetfCplusgr(fCplusgr);
        ode.SetfHeplusgr(fHeplusgr);
        ode.SetfSplusgr(fSplusgr);
        ode.SetfSiplusgr(fSiplusgr);
        ode.SetfCplusCR(fCplusCR);
        ode.SetGradv(gradv_ptr[i]);
        ode.SetTdust(Tdust_ptr[i]);

        ode.SetCOPhaseParams(
            sigma_d_CO_per_H_ptr[i],
            co_E_bind_co,
            co_nu0_co,
            co_Y_CO,
            co_N_SURF,
            co_N_LAY,
            co_S_CO,
            co_F_CRUV_CO_pdes_ptr[i],
            co_k_crdes_CO_ptr[i]);

        ode.Leff_CO_max(Leff_CO_max_ptr[i]);
        ode.IsDustCooling(isDust_cooling);
        ode.SetCoolingCOThin(isCoolingCOThin);

        if (const_temp) {
            ode.SetConstTemp(Tgas_ptr[i]);
        }

        double GPE_cell = GPE_ptr[i];
        double F_CO_pdes_photon_cell = F_CO_pdes_photon_ptr[i];
        double Gph_cell[N_PH];
        for (int j = 0; j < N_PH; ++j) {
            Gph_cell[j] = Gph_ptr[i * N_PH + j];
        }
        ode.SetRadField(&GPE_cell, Gph_cell, &F_CO_pdes_photon_cell);

        solver.ReInit();
        solver.SetMxsteps(mxsteps);
        solver.SetMaxOrd(maxord);

        try {
            solver.Solve(t_end);
            const long long n_negative = solver.GetNegativeCorrectionCount();
            if (n_negative > 0) {
                n_negative_abundance_cells.fetch_add(1, std::memory_order_relaxed);
                n_negative_abundance_corrections.fetch_add(
                    n_negative, std::memory_order_relaxed);
            }
            ode.CopyAbd(y_out_ptr + i * N_Y);
            status_out_ptr[i] = 0;
        } catch (const std::exception &e) {
            for (int j = 0; j < N_Y; ++j) {
                y_out_ptr[i * N_Y + j] = y0_ptr[i * N_Y + j];
            }
            status_out_ptr[i] = -1;
            n_fail.fetch_add(1, std::memory_order_relaxed);
            if (is_cvode_integration_failure(e)) {
                n_cvode_fail.fetch_add(1, std::memory_order_relaxed);
            } else {
                n_other_exception_fail.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    /* Re-acquire the GIL before touching Python objects. */
    py::gil_scoped_acquire acquire;

    if (n_cvode_fail.load() > 0) {
        printf("solve_batch_time: CVODE failed in %lld / %lld cells "
               "(per-cell SUNDIALS messages suppressed).\n",
               n_cvode_fail.load(), static_cast<long long>(Ncells));
    }
    if (n_other_exception_fail.load() > 0) {
        printf("solve_batch_time: %lld / %lld cells failed with non-CVODE exceptions.\n",
               n_other_exception_fail.load(), static_cast<long long>(Ncells));
    }
    if (n_negative_abundance_cells.load() > 0) {
        printf("solve_batch_time: corrected %lld / %lld cells with negative abundances "
               "(%lld species corrections total).\n",
               n_negative_abundance_cells.load(),
               static_cast<long long>(Ncells),
               n_negative_abundance_corrections.load());
    }

    py::dict out;
    out["y"] = y_out;
    out["status"] = status_out;
    out["cvode_failure_cells"] = n_cvode_fail.load();
    out["exception_failure_cells"] = n_other_exception_fail.load();
    out["negative_abundance_cells"] = n_negative_abundance_cells.load();
    out["negative_abundance_corrections"] = n_negative_abundance_corrections.load();
    return out;
}


static py::dict eval_rhs_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast> y,
    const py::array_t<double, py::array::c_style | py::array::forcecast> nH,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Tgas,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Tdust,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zd,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zgd,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zg,
    const py::array_t<double, py::array::c_style | py::array::forcecast> ion_rate,
    const py::array_t<double, py::array::c_style | py::array::forcecast> GPE,
    const py::array_t<double, py::array::c_style | py::array::forcecast> F_CO_pdes_photon,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Gph,
    const py::array_t<double, py::array::c_style | py::array::forcecast> sigma_d_CO_per_H,
    const bool const_temp,
    const py::array_t<double, py::array::c_style | py::array::forcecast> gradv,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Leff_CO_max,
    const bool isDust_cooling,
    const bool isCoolingCOThin,
    const double fH2gr,
    const double fHplusgr,
    const double fCplusgr,
    const double fHeplusgr,
    const double fSplusgr,
    const double fSiplusgr,
    const double fCplusCR,
    const double co_E_bind_co,
    const double co_nu0_co,
    const double co_Y_CO,
    const double co_N_SURF,
    const int co_N_LAY,
    const double co_S_CO,
    const py::array_t<double, py::array::c_style | py::array::forcecast> co_F_CRUV_CO_pdes,
    const py::array_t<double, py::array::c_style | py::array::forcecast> co_k_crdes_CO) {

    if (y.ndim() != 2 || y.shape(1) != N_Y) {
        throw std::invalid_argument("y must be 2D with shape (Ncells, 15)");
    }
    const py::ssize_t Ncells = y.shape(0);
    if (nH.ndim() != 1 || nH.size() != Ncells) {
        throw std::invalid_argument("nH must be 1D with length Ncells");
    }
    if (Tgas.ndim() != 1 || Tgas.size() != Ncells) {
        throw std::invalid_argument("Tgas must be 1D with length Ncells");
    }
    if (Tdust.ndim() != 1 || Tdust.size() != Ncells) {
        throw std::invalid_argument("Tdust must be 1D with length Ncells");
    }
    if (Zd.ndim() != 1 || Zd.size() != Ncells) {
        throw std::invalid_argument("Zd must be 1D with length Ncells");
    }
    if (Zgd.ndim() != 1 || Zgd.size() != Ncells) {
        throw std::invalid_argument("Zgd must be 1D with length Ncells");
    }
    if (Zg.ndim() != 1 || Zg.size() != Ncells) {
        throw std::invalid_argument("Zg must be 1D with length Ncells");
    }
    if (ion_rate.ndim() != 1 || ion_rate.size() != Ncells) {
        throw std::invalid_argument("ion_rate must be 1D with length Ncells");
    }
    if (GPE.ndim() != 1 || GPE.size() != Ncells) {
        throw std::invalid_argument("GPE must be 1D with length Ncells");
    }
    if (F_CO_pdes_photon.ndim() != 1 || F_CO_pdes_photon.size() != Ncells) {
        throw std::invalid_argument("F_CO_pdes_photon must be 1D with length Ncells");
    }
    if (Gph.ndim() != 2 || Gph.shape(0) != Ncells || Gph.shape(1) != N_PH) {
        throw std::invalid_argument("Gph must be 2D with shape (Ncells, 7)");
    }
    if (sigma_d_CO_per_H.ndim() != 1 || sigma_d_CO_per_H.size() != Ncells) {
        throw std::invalid_argument("sigma_d_CO_per_H must be 1D with length Ncells");
    }
    if (gradv.ndim() != 1 || gradv.size() != Ncells) {
        throw std::invalid_argument("gradv must be 1D with length Ncells");
    }
    if (Leff_CO_max.ndim() != 1 || Leff_CO_max.size() != Ncells) {
        throw std::invalid_argument("Leff_CO_max must be 1D with length Ncells");
    }
    if (co_F_CRUV_CO_pdes.ndim() != 1 || co_F_CRUV_CO_pdes.size() != Ncells) {
        throw std::invalid_argument("co_F_CRUV_CO_pdes must be 1D with length Ncells");
    }
    if (co_k_crdes_CO.ndim() != 1 || co_k_crdes_CO.size() != Ncells) {
        throw std::invalid_argument("co_k_crdes_CO must be 1D with length Ncells");
    }

    const double *y_ptr = y.data();
    const double *nH_ptr = nH.data();
    const double *Tgas_ptr = Tgas.data();
    const double *Tdust_ptr = Tdust.data();
    const double *Zd_ptr = Zd.data();
    const double *Zgd_ptr = Zgd.data();
    const double *Zg_ptr = Zg.data();
    const double *ion_rate_ptr = ion_rate.data();
    const double *GPE_ptr = GPE.data();
    const double *F_CO_pdes_photon_ptr = F_CO_pdes_photon.data();
    const double *Gph_ptr = Gph.data();
    const double *sigma_d_CO_per_H_ptr = sigma_d_CO_per_H.data();
    const double *gradv_ptr = gradv.data();
    const double *Leff_CO_max_ptr = Leff_CO_max.data();
    const double *co_F_CRUV_CO_pdes_ptr = co_F_CRUV_CO_pdes.data();
    const double *co_k_crdes_CO_ptr = co_k_crdes_CO.data();

    py::array_t<double> rhs_out(py::array::ShapeContainer{Ncells, static_cast<py::ssize_t>(N_Y)});
    py::array_t<double> thermo_out(py::array::ShapeContainer{Ncells, static_cast<py::ssize_t>(15)});
    py::array_t<int> status_out(py::array::ShapeContainer{Ncells});
    double *rhs_ptr = static_cast<double *>(rhs_out.mutable_data());
    double *thermo_ptr = static_cast<double *>(thermo_out.mutable_data());
    int *status_ptr = static_cast<int *>(status_out.mutable_data());

    std::atomic<long long> n_failure{0};

    py::gil_scoped_release release;

    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t i = 0; i < Ncells; ++i) {
        SUNContext sunctx = NULL;
        N_Vector y_vec = NULL;
        N_Vector ydot_vec = NULL;
        try {
            int flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
            if (flag != 0 || sunctx == NULL) {
                throw std::runtime_error("eval_rhs_batch: SUNContext_Create failed");
            }
            y_vec = N_VNew_Serial(N_Y, sunctx);
            ydot_vec = N_VNew_Serial(N_Y, sunctx);
            if (y_vec == NULL || ydot_vec == NULL) {
                throw std::runtime_error("eval_rhs_batch: N_VNew_Serial failed");
            }
            for (int j = 0; j < N_Y; ++j) {
                NV_Ith_S(y_vec, j) = y_ptr[i * N_Y + j];
                NV_Ith_S(ydot_vec, j) = 0.0;
                rhs_ptr[i * N_Y + j] = 0.0;
            }

            gow17 ode;
            ode.SetnH(nH_ptr[i]);
            ode.SetIonRate(ion_rate_ptr[i]);
            ode.SetZg(Zg_ptr[i]);
            ode.SetZd(Zd_ptr[i]);
            ode.SetZgd(Zgd_ptr[i]);
            ode.SetfH2gr(fH2gr);
            ode.SetfHplusgr(fHplusgr);
            ode.SetfCplusgr(fCplusgr);
            ode.SetfHeplusgr(fHeplusgr);
            ode.SetfSplusgr(fSplusgr);
            ode.SetfSiplusgr(fSiplusgr);
            ode.SetfCplusCR(fCplusCR);
            ode.SetGradv(gradv_ptr[i]);
            ode.SetTdust(Tdust_ptr[i]);
            ode.SetCOPhaseParams(
                sigma_d_CO_per_H_ptr[i],
                co_E_bind_co,
                co_nu0_co,
                co_Y_CO,
                co_N_SURF,
                co_N_LAY,
                co_S_CO,
                co_F_CRUV_CO_pdes_ptr[i],
                co_k_crdes_CO_ptr[i]);
            ode.Leff_CO_max(Leff_CO_max_ptr[i]);
            ode.IsDustCooling(isDust_cooling);
            ode.SetCoolingCOThin(isCoolingCOThin);
            if (const_temp) {
                ode.SetConstTemp(Tgas_ptr[i]);
            }

            double GPE_cell = GPE_ptr[i];
            double F_CO_pdes_photon_cell = F_CO_pdes_photon_ptr[i];
            double Gph_cell[N_PH];
            for (int j = 0; j < N_PH; ++j) {
                Gph_cell[j] = Gph_ptr[i * N_PH + j];
            }
            ode.SetRadField(&GPE_cell, Gph_cell, &F_CO_pdes_photon_cell);

            const int rhs_flag = ode.RHS(0.0, y_vec, ydot_vec);
            if (rhs_flag != 0) {
                throw std::runtime_error("eval_rhs_batch: gow17 RHS returned nonzero status");
            }
            for (int j = 0; j < N_Y; ++j) {
                rhs_ptr[i * N_Y + j] = NV_Ith_S(ydot_vec, j);
            }
            ode.CopyThermoRates(thermo_ptr + i * 15);
            status_ptr[i] = 0;
        } catch (const std::exception &) {
            for (int j = 0; j < N_Y; ++j) {
                rhs_ptr[i * N_Y + j] = 0.0;
            }
            for (int j = 0; j < 15; ++j) {
                thermo_ptr[i * 15 + j] = 0.0;
            }
            status_ptr[i] = -1;
            n_failure.fetch_add(1, std::memory_order_relaxed);
        }
        if (ydot_vec != NULL) {
            N_VDestroy(ydot_vec);
        }
        if (y_vec != NULL) {
            N_VDestroy(y_vec);
        }
        if (sunctx != NULL) {
            SUNContext_Free(&sunctx);
        }
    }

    py::gil_scoped_acquire acquire;

    py::dict out;
    out["rhs"] = rhs_out;
    out["thermo_rates"] = thermo_out;
    out["status"] = status_out;
    out["failure_cells"] = n_failure.load();
    return out;
}


PYBIND11_MODULE(_gow17, m) {
    m.attr("N_Y") = N_Y;
    m.attr("N_PH") = N_PH;
    m.attr("IPH_C") = IPH_C;
    m.attr("IPH_CH") = IPH_CH;
    m.attr("IPH_CO") = IPH_CO;
    m.attr("IPH_OH") = IPH_OH;
    m.attr("IPH_H2") = IPH_H2;
    m.attr("IPH_S") = IPH_S;
    m.attr("IPH_SI") = IPH_SI;

    m.attr("XC_STD") = gow17::xC_std();
    m.attr("XO_STD") = gow17::xO_std();
    m.attr("XHE") = 0.1;

    m.attr("I_HEP") = gow17::id("He+");
    m.attr("I_OHX") = gow17::id("OHx");
    m.attr("I_CHX") = gow17::id("CHx");
    m.attr("I_CO") = gow17::id("CO");
    m.attr("I_CO_ICE") = gow17::id("CO_ice");
    m.attr("I_CP") = gow17::id("C+");
    m.attr("I_HCOP") = gow17::id("HCO+");
    m.attr("I_H2") = gow17::id("H2");
    m.attr("I_HP") = gow17::id("H+");
    m.attr("I_H3P") = gow17::id("H3+");
    m.attr("I_H2P") = gow17::id("H2+");
    m.attr("I_SP") = gow17::id("S+");
    m.attr("I_SIP") = gow17::id("Si+");
    m.attr("I_OP") = gow17::id("O+");
    m.attr("I_E") = gow17::id("E");

    m.def(
        "solve_slab_1d_equilibrium",
        &solve_slab_1d_equilibrium,
        py::arg("nH"),
        py::arg("G0"),
        py::arg("ngrid"),
        py::arg("NH_total"),
        py::arg("logNH"),
        py::arg("NH_min"),
        py::arg("field_geo"),
        py::arg("isdust"),
        py::arg("isfsH2"),
        py::arg("isfsCO"),
        py::arg("isfsC"),
        py::arg("Zg"),
        py::arg("Zd"),
        py::arg("ion_rate"),
        py::arg("reltol"),
        py::arg("abstol"),
        py::arg("mxsteps"),
        py::arg("maxord"),
        py::arg("tolfac"),
        py::arg("tmin"),
        py::arg("tmax"),
        py::arg("verbose"),
        py::arg("y0"),
        py::arg("const_temp"),
        py::arg("Tgas"),
        py::arg("gradv"),
        py::arg("NCOeff_global"),
        py::arg("bCO_L"),
        py::arg("fH2gr"),
        py::arg("fHplusgr"),
        py::arg("fCplusgr"),
        py::arg("fHeplusgr"),
        py::arg("fSplusgr"),
        py::arg("fSiplusgr"),
        py::arg("fCplusCR"),
        py::arg("co_sigma_d_per_H_ref"),
        py::arg("co_E_bind_co"),
        py::arg("co_nu0_co"),
        py::arg("co_F_DRAINE"),
        py::arg("co_Y_CO"),
        py::arg("co_N_SURF"),
        py::arg("co_N_LAY"),
        py::arg("userJac"));

    m.def(
        "solve_batch_equilibrium",
        &solve_batch_equilibrium,
        py::arg("y0"),
        py::arg("nH"),
        py::arg("Tgas"),
        py::arg("Tdust"),
        py::arg("Zd"),
        py::arg("Zgd"),
        py::arg("Zg"),
        py::arg("ion_rate"),
        py::arg("GPE"),
        py::arg("F_CO_pdes_photon"),
        py::arg("Gph"),
        py::arg("sigma_d_CO_per_H"),
        py::arg("reltol"),
        py::arg("abstol"),
        py::arg("mxsteps"),
        py::arg("maxord"),
        py::arg("tolfac"),
        py::arg("tmin"),
        py::arg("tmax"),
        py::arg("const_temp"),
        py::arg("gradv"),
        py::arg("Leff_CO_max"),
        py::arg("isDust_cooling") = false,
        py::arg("isCoolingCOThin") = false,
        py::arg("fH2gr"),
        py::arg("fHplusgr"),
        py::arg("fCplusgr"),
        py::arg("fHeplusgr"),
        py::arg("fSplusgr"),
        py::arg("fSiplusgr"),
        py::arg("fCplusCR"),
        py::arg("co_E_bind_co"),
        py::arg("co_nu0_co"),
        py::arg("co_Y_CO"),
        py::arg("co_N_SURF"),
        py::arg("co_N_LAY"),
        py::arg("co_S_CO"),
        py::arg("co_F_CRUV_CO_pdes"),
        py::arg("co_k_crdes_CO"),
        py::arg("userJac"),
        py::arg("verbose") = false);

    m.def(
        "solve_batch_time",
        &solve_batch_time,
        py::arg("y0"),
        py::arg("nH"),
        py::arg("Tgas"),
        py::arg("Tdust"),
        py::arg("Zd"),
        py::arg("Zgd"),
        py::arg("Zg"),
        py::arg("ion_rate"),
        py::arg("GPE"),
        py::arg("F_CO_pdes_photon"),
        py::arg("Gph"),
        py::arg("sigma_d_CO_per_H"),
        py::arg("reltol"),
        py::arg("abstol"),
        py::arg("mxsteps"),
        py::arg("maxord"),
        py::arg("t_end"),
        py::arg("const_temp"),
        py::arg("gradv"),
        py::arg("Leff_CO_max"),
        py::arg("isDust_cooling") = false,
        py::arg("isCoolingCOThin") = false,
        py::arg("fH2gr"),
        py::arg("fHplusgr"),
        py::arg("fCplusgr"),
        py::arg("fHeplusgr"),
        py::arg("fSplusgr"),
        py::arg("fSiplusgr"),
        py::arg("fCplusCR"),
        py::arg("co_E_bind_co"),
        py::arg("co_nu0_co"),
        py::arg("co_Y_CO"),
        py::arg("co_N_SURF"),
        py::arg("co_N_LAY"),
        py::arg("co_S_CO"),
        py::arg("co_F_CRUV_CO_pdes"),
        py::arg("co_k_crdes_CO"),
        py::arg("userJac"),
        py::arg("verbose") = false);


    m.def(
        "eval_rhs_batch",
        &eval_rhs_batch,
        py::arg("y"),
        py::arg("nH"),
        py::arg("Tgas"),
        py::arg("Tdust"),
        py::arg("Zd"),
        py::arg("Zgd"),
        py::arg("Zg"),
        py::arg("ion_rate"),
        py::arg("GPE"),
        py::arg("F_CO_pdes_photon"),
        py::arg("Gph"),
        py::arg("sigma_d_CO_per_H"),
        py::arg("const_temp"),
        py::arg("gradv"),
        py::arg("Leff_CO_max"),
        py::arg("isDust_cooling") = false,
        py::arg("isCoolingCOThin") = false,
        py::arg("fH2gr"),
        py::arg("fHplusgr"),
        py::arg("fCplusgr"),
        py::arg("fHeplusgr"),
        py::arg("fSplusgr"),
        py::arg("fSiplusgr"),
        py::arg("fCplusCR"),
        py::arg("co_E_bind_co"),
        py::arg("co_nu0_co"),
        py::arg("co_Y_CO"),
        py::arg("co_N_SURF"),
        py::arg("co_N_LAY"),
        py::arg("co_S_CO"),
        py::arg("co_F_CRUV_CO_pdes"),
        py::arg("co_k_crdes_CO"));

    m.def(
        "co_freezeout_rate_cgs",
        [](const double nH_cm3, const double Tgas_K, const double sigma_d_per_H_cm2,
           const double sticking, const double kB, const double mCO) {
            return co_phase::co_freezeout_rate(
                nH_cm3, Tgas_K, sigma_d_per_H_cm2, sticking, kB, mCO);
        },
        py::arg("nH_cm3"),
        py::arg("Tgas_K"),
        py::arg("sigma_d_per_H_cm2"),
        py::arg("sticking"),
        py::arg("kB"),
        py::arg("mCO"));

    m.def(
        "co_thermal_desorption_rate_cgs",
        [](const double Tdust_K, const double nu0_co, const double E_bind_co) {
            return co_phase::co_thermal_desorption_rate(Tdust_K, nu0_co, E_bind_co);
        },
        py::arg("Tdust_K"),
        py::arg("nu0_co"),
        py::arg("E_bind_co"));

    m.def(
        "co_photodesorption_surface_rate_cgs",
        [](const double chi, const double F_DRAINE, const double Y_CO, const double N_SURF,
           const int N_LAY) {
            return co_phase::co_photodesorption_surface_rate(
                chi, F_DRAINE, Y_CO, N_SURF, N_LAY);
        },
        py::arg("chi"),
        py::arg("F_DRAINE"),
        py::arg("Y_CO"),
        py::arg("N_SURF"),
        py::arg("N_LAY"));

    m.def(
        "co_photodesorption_surface_rate_from_flux_cgs",
        [](const double F_photon_cm2_s, const double Y_CO, const double N_SURF,
           const int N_LAY) {
            return co_phase::co_photodesorption_surface_rate_from_flux(
                F_photon_cm2_s, Y_CO, N_SURF, N_LAY);
        },
        py::arg("F_photon_cm2_s"),
        py::arg("Y_CO"),
        py::arg("N_SURF"),
        py::arg("N_LAY"));

    m.def(
        "co_active_ice_cgs",
        [](const double nH_cm3, const double sigma_d_per_H_cm2, const double nco_ice_cm3,
           const double N_SURF, const int N_LAY) {
            double n_act_max = 0.0;
            double n_act = 0.0;
            co_phase::co_active_ice(
                nH_cm3, sigma_d_per_H_cm2, nco_ice_cm3, N_SURF, N_LAY, n_act_max, n_act);
            return std::make_pair(n_act_max, n_act);
        },
        py::arg("nH_cm3"),
        py::arg("sigma_d_per_H_cm2"),
        py::arg("nco_ice_cm3"),
        py::arg("N_SURF"),
        py::arg("N_LAY"));

    m.def(
        "co_photodesorption_R_cgs",
        [](const double k_pd_surf_s, const double n_ice_act_cm3) {
            return co_phase::co_photodesorption_R(k_pd_surf_s, n_ice_act_cm3);
        },
        py::arg("k_pd_surf_s"),
        py::arg("n_ice_act_cm3"));
}
