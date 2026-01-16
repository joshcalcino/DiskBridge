#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
        co_F_DRAINE,
        co_Y_CO,
        co_N_SURF,
        co_N_LAY);

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
    const py::array_t<double, py::array::c_style | py::array::forcecast> Zg,
    const py::array_t<double, py::array::c_style | py::array::forcecast> ion_rate,
    const py::array_t<double, py::array::c_style | py::array::forcecast> GPE,
    const py::array_t<double, py::array::c_style | py::array::forcecast> GISRF,
    const py::array_t<double, py::array::c_style | py::array::forcecast> Gph,
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

    const double co_sigma_d_per_H_ref,
    const double co_E_bind_co,
    const double co_nu0_co,
    const double co_F_DRAINE,
    const double co_Y_CO,
    const double co_N_SURF,
    const int co_N_LAY,
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
    if (Zg.ndim() != 1 || Zg.size() != Ncells) {
        throw std::invalid_argument("Zg must be 1D with length Ncells");
    }
    if (ion_rate.ndim() != 1 || ion_rate.size() != Ncells) {
        throw std::invalid_argument("ion_rate must be 1D with length Ncells");
    }
    if (GPE.ndim() != 1 || GPE.size() != Ncells) {
        throw std::invalid_argument("GPE must be 1D with length Ncells");
    }
    if (GISRF.ndim() != 1 || GISRF.size() != Ncells) {
        throw std::invalid_argument("GISRF must be 1D with length Ncells");
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

    const double *y0_ptr = y0.data();
    const double *nH_ptr = nH.data();
    const double *Tgas_ptr = Tgas.data();
    const double *Tdust_ptr = Tdust.data();
    const double *Zd_ptr = Zd.data();
    const double *Zg_ptr = Zg.data();
    const double *ion_rate_ptr = ion_rate.data();
    const double *GPE_ptr = GPE.data();
    const double *GISRF_ptr = GISRF.data();
    const double *Gph_ptr = Gph.data();
    const double *abstol_ptr = abstol.data();
    const double *Leff_CO_max_ptr = Leff_CO_max.data();
    const double *gradv_ptr = gradv.data();

    py::array_t<double> y_out(py::array::ShapeContainer{Ncells, static_cast<py::ssize_t>(N_Y)});
    py::array_t<int> status_out(py::array::ShapeContainer{Ncells});
    double *y_out_ptr = static_cast<double *>(y_out.mutable_data());
    int *status_out_ptr = static_cast<int *>(status_out.mutable_data());

    for (py::ssize_t i = 0; i < Ncells; ++i) {
        gow17 ode;
        CvodeDense solver(ode, reltol, abstol_ptr, userJac);

        ode.SetInit(0.0, y0_ptr + i * N_Y);
        ode.SetnH(nH_ptr[i]);
        ode.SetIonRate(ion_rate_ptr[i]);
        ode.SetZg(Zg_ptr[i]);
        ode.SetZd(Zd_ptr[i]);

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
            co_sigma_d_per_H_ref,
            co_E_bind_co,
            co_nu0_co,
            co_F_DRAINE,
            co_Y_CO,
            co_N_SURF,
            co_N_LAY);

        ode.Leff_CO_max(Leff_CO_max_ptr[i]);
        ode.IsDustCooling(isDust_cooling);
        ode.SetCoolingCOThin(isCoolingCOThin);

        if (const_temp) {
            ode.SetConstTemp(Tgas_ptr[i]);
        }

        double GPE_cell = GPE_ptr[i];
        double GISRF_cell = GISRF_ptr[i];
        double Gph_cell[N_PH];
        for (int j = 0; j < N_PH; ++j) {
            Gph_cell[j] = Gph_ptr[i * N_PH + j];
        }
        ode.SetRadField(&GPE_cell, Gph_cell, &GISRF_cell);

        solver.ReInit();
        solver.SetMxsteps(mxsteps);
        solver.SetMaxOrd(maxord);

        try {
            solver.SolveEq(tolfac, tmax, verbose, tmin);
            ode.CopyAbd(y_out_ptr + i * N_Y);
            status_out_ptr[i] = 0;
        } catch (const std::exception &e) {
            for (int j = 0; j < N_Y; ++j) {
                y_out_ptr[i * N_Y + j] = y0_ptr[i * N_Y + j];
            }
            status_out_ptr[i] = -1;
        }
    }

    py::dict out;
    out["y"] = y_out;
    out["status"] = status_out;
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
        py::arg("Zg"),
        py::arg("ion_rate"),
        py::arg("GPE"),
        py::arg("GISRF"),
        py::arg("Gph"),
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
        py::arg("co_sigma_d_per_H_ref"),
        py::arg("co_E_bind_co"),
        py::arg("co_nu0_co"),
        py::arg("co_F_DRAINE"),
        py::arg("co_Y_CO"),
        py::arg("co_N_SURF"),
        py::arg("co_N_LAY"),
        py::arg("userJac"),
        py::arg("verbose") = false);

    m.def(
        "co_freezeout_rate_cgs",
        [](const double nH_cm3, const double Tgas_K, const double sigma_d_per_H_cm2,
           const double kB, const double mCO) {
            return co_phase::co_freezeout_rate(
                nH_cm3, Tgas_K, sigma_d_per_H_cm2, kB, mCO);
        },
        py::arg("nH_cm3"),
        py::arg("Tgas_K"),
        py::arg("sigma_d_per_H_cm2"),
        py::arg("kB"),
        py::arg("mCO"));

    m.def(
        "co_freezeout_rate_field_cgs",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> nH_cm3,
           py::array_t<double, py::array::c_style | py::array::forcecast> Tgas_K,
           py::array_t<double, py::array::c_style | py::array::forcecast> sigma_d_per_H_cm2,
           const double kB, const double mCO) {
            const auto b_nH = nH_cm3.request();
            const auto b_T = Tgas_K.request();
            const auto b_sig = sigma_d_per_H_cm2.request();
            if (b_T.size != b_nH.size || b_sig.size != b_nH.size) {
                throw std::invalid_argument("co_freezeout_rate_field_cgs: input arrays must have same size");
            }
            if (b_T.ndim != b_nH.ndim || b_sig.ndim != b_nH.ndim) {
                throw std::invalid_argument("co_freezeout_rate_field_cgs: input arrays must have same ndim");
            }
            for (py::ssize_t d = 0; d < b_nH.ndim; ++d) {
                if (b_T.shape[d] != b_nH.shape[d] || b_sig.shape[d] != b_nH.shape[d]) {
                    throw std::invalid_argument("co_freezeout_rate_field_cgs: input arrays must have same shape");
                }
            }

            std::vector<py::ssize_t> out_shape(static_cast<size_t>(b_nH.ndim));
            for (py::ssize_t d = 0; d < b_nH.ndim; ++d) {
                out_shape[static_cast<size_t>(d)] = b_nH.shape[d];
            }
            py::array_t<double> out(out_shape);
            auto b_out = out.request();
            const double *nH_ptr = static_cast<const double *>(b_nH.ptr);
            const double *T_ptr = static_cast<const double *>(b_T.ptr);
            const double *sig_ptr = static_cast<const double *>(b_sig.ptr);
            double *out_ptr = static_cast<double *>(b_out.ptr);
            for (py::ssize_t i = 0; i < b_nH.size; ++i) {
                out_ptr[i] = co_phase::co_freezeout_rate(nH_ptr[i], T_ptr[i], sig_ptr[i], kB, mCO);
            }
            return out;
        },
        py::arg("nH_cm3"),
        py::arg("Tgas_K"),
        py::arg("sigma_d_per_H_cm2"),
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
        "co_thermal_desorption_rate_field_cgs",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> Tdust_K,
           const double nu0_co, const double E_bind_co) {
            const auto b_Td = Tdust_K.request();
            std::vector<py::ssize_t> out_shape(static_cast<size_t>(b_Td.ndim));
            for (py::ssize_t d = 0; d < b_Td.ndim; ++d) {
                out_shape[static_cast<size_t>(d)] = b_Td.shape[d];
            }
            py::array_t<double> out(out_shape);
            auto b_out = out.request();
            const double *Td_ptr = static_cast<const double *>(b_Td.ptr);
            double *out_ptr = static_cast<double *>(b_out.ptr);
            for (py::ssize_t i = 0; i < b_Td.size; ++i) {
                out_ptr[i] = co_phase::co_thermal_desorption_rate(Td_ptr[i], nu0_co, E_bind_co);
            }
            return out;
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
        "co_photodesorption_surface_rate_field_cgs",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> chi,
           const double F_DRAINE, const double Y_CO, const double N_SURF, const int N_LAY) {
            const auto b_chi = chi.request();
            std::vector<py::ssize_t> out_shape(static_cast<size_t>(b_chi.ndim));
            for (py::ssize_t d = 0; d < b_chi.ndim; ++d) {
                out_shape[static_cast<size_t>(d)] = b_chi.shape[d];
            }
            py::array_t<double> out(out_shape);
            auto b_out = out.request();
            const double *chi_ptr = static_cast<const double *>(b_chi.ptr);
            double *out_ptr = static_cast<double *>(b_out.ptr);
            for (py::ssize_t i = 0; i < b_chi.size; ++i) {
                out_ptr[i] = co_phase::co_photodesorption_surface_rate(
                    chi_ptr[i], F_DRAINE, Y_CO, N_SURF, N_LAY);
            }
            return out;
        },
        py::arg("chi"),
        py::arg("F_DRAINE"),
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
        "co_active_ice_field_cgs",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> nH_cm3,
           py::array_t<double, py::array::c_style | py::array::forcecast> sigma_d_per_H_cm2,
           py::array_t<double, py::array::c_style | py::array::forcecast> nco_ice_cm3,
           const double N_SURF, const int N_LAY) {
            const auto b_nH = nH_cm3.request();
            const auto b_sig = sigma_d_per_H_cm2.request();
            const auto b_nco = nco_ice_cm3.request();
            if (b_sig.size != b_nH.size || b_nco.size != b_nH.size) {
                throw std::invalid_argument("co_active_ice_field_cgs: input arrays must have same size");
            }
            if (b_sig.ndim != b_nH.ndim || b_nco.ndim != b_nH.ndim) {
                throw std::invalid_argument("co_active_ice_field_cgs: input arrays must have same ndim");
            }
            for (py::ssize_t d = 0; d < b_nH.ndim; ++d) {
                if (b_sig.shape[d] != b_nH.shape[d] || b_nco.shape[d] != b_nH.shape[d]) {
                    throw std::invalid_argument("co_active_ice_field_cgs: input arrays must have same shape");
                }
            }

            std::vector<py::ssize_t> out_shape(static_cast<size_t>(b_nH.ndim));
            for (py::ssize_t d = 0; d < b_nH.ndim; ++d) {
                out_shape[static_cast<size_t>(d)] = b_nH.shape[d];
            }
            py::array_t<double> out_max(out_shape);
            py::array_t<double> out_act(out_shape);
            auto b_out_max = out_max.request();
            auto b_out_act = out_act.request();

            const double *nH_ptr = static_cast<const double *>(b_nH.ptr);
            const double *sig_ptr = static_cast<const double *>(b_sig.ptr);
            const double *nco_ptr = static_cast<const double *>(b_nco.ptr);
            double *out_max_ptr = static_cast<double *>(b_out_max.ptr);
            double *out_act_ptr = static_cast<double *>(b_out_act.ptr);

            for (py::ssize_t i = 0; i < b_nH.size; ++i) {
                double n_act_max = 0.0;
                double n_act = 0.0;
                co_phase::co_active_ice(
                    nH_ptr[i], sig_ptr[i], nco_ptr[i], N_SURF, N_LAY, n_act_max, n_act);
                out_max_ptr[i] = n_act_max;
                out_act_ptr[i] = n_act;
            }

            return std::make_pair(out_max, out_act);
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

    m.def(
        "co_photodesorption_R_field_cgs",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> k_pd_surf_s,
           py::array_t<double, py::array::c_style | py::array::forcecast> n_ice_act_cm3) {
            const auto b_k = k_pd_surf_s.request();
            const auto b_n = n_ice_act_cm3.request();
            if (b_k.size != b_n.size) {
                throw std::invalid_argument("co_photodesorption_R_field_cgs: input arrays must have same size");
            }
            if (b_k.ndim != b_n.ndim) {
                throw std::invalid_argument("co_photodesorption_R_field_cgs: input arrays must have same ndim");
            }
            for (py::ssize_t d = 0; d < b_k.ndim; ++d) {
                if (b_k.shape[d] != b_n.shape[d]) {
                    throw std::invalid_argument("co_photodesorption_R_field_cgs: input arrays must have same shape");
                }
            }

            std::vector<py::ssize_t> out_shape(static_cast<size_t>(b_k.ndim));
            for (py::ssize_t d = 0; d < b_k.ndim; ++d) {
                out_shape[static_cast<size_t>(d)] = b_k.shape[d];
            }
            py::array_t<double> out(out_shape);
            auto b_out = out.request();
            const double *k_ptr = static_cast<const double *>(b_k.ptr);
            const double *n_ptr = static_cast<const double *>(b_n.ptr);
            double *out_ptr = static_cast<double *>(b_out.ptr);
            for (py::ssize_t i = 0; i < b_k.size; ++i) {
                out_ptr[i] = co_phase::co_photodesorption_R(k_ptr[i], n_ptr[i]);
            }
            return out;
        },
        py::arg("k_pd_surf_s"),
        py::arg("n_ice_act_cm3"));
}
