#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

#include "cvodeDense.h"
#include "gow17.h"
#include "slab.h"

namespace py = pybind11;

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

PYBIND11_MODULE(_gow17, m) {
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
}
