#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "cvodeDense.h"
#include "gow17.h"
#include "slab.h"

namespace py = pybind11;

static constexpr int N_Y = 14;
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

    ode.SetfH2gr(fH2gr);
    ode.SetfHplusgr(fHplusgr);
    ode.SetfCplusgr(fCplusgr);
    ode.SetfHeplusgr(fHeplusgr);
    ode.SetfSplusgr(fSplusgr);
    ode.SetfSiplusgr(fSiplusgr);
    ode.SetfCplusCR(fCplusCR);

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
    const double gradv,
    const double Leff_CO_max,
    const bool isDust_cooling,
    const bool isCoolingCOThin,
    const double fH2gr,
    const double fHplusgr,
    const double fCplusgr,
    const double fHeplusgr,
    const double fSplusgr,
    const double fSiplusgr,
    const double fCplusCR,
    const bool userJac,
    const bool verbose) {

    if (y0.ndim() != 2 || y0.shape(1) != N_Y) {
        throw std::invalid_argument("y0 must be 2D with shape (Ncells, 14)");
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
        throw std::invalid_argument("abstol must be 1D with length 14");
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
        ode.SetGradv(gradv);
        ode.SetTdust(Tdust_ptr[i]);
        ode.Leff_CO_max(Leff_CO_max);
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
        py::arg("Leff_CO_max") = 3.0e20,
        py::arg("isDust_cooling") = false,
        py::arg("isCoolingCOThin") = false,
        py::arg("fH2gr"),
        py::arg("fHplusgr"),
        py::arg("fCplusgr"),
        py::arg("fHeplusgr"),
        py::arg("fSplusgr"),
        py::arg("fSiplusgr"),
        py::arg("fCplusCR"),
        py::arg("userJac"),
        py::arg("verbose") = false);
}
