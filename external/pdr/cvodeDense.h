/*Date: Mar 11, 2015   Author: Munan Gong
 * CvodeDense class that uses the CVODE package to solve the ode class.
 * method similar to cvRoberts_dns sample in the CVODE example.
 */
#ifndef CVODEDENSE_H_
#define CVODEDENSE_H_

#include <sundials/sundials_types.h> /* realtype type*/
#include <nvector/nvector_serial.h> /* N_Vector type*/
#include <cvode/cvode.h>            /* CVODE solver fcts., consts. */
#include <cvode/cvode_ls.h>
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <math.h> /*fabs*/
#include <time.h> /*clock_t, clock, CLOCKS_PER_SEC*/
#include <algorithm>    /*std::max, std::max_element*/
#include "ode.h"
#include "sundial.h" /*CheckFlag function*/

class CvodeDense {
  public:
    CvodeDense(Ode &ode,
               const double reltol, const double *abstol,
							 const bool userJac=false);
    ~CvodeDense();
    /*Set the maximum number of steps*/
    void SetMxsteps(const int mxsteps);
    /*Set the initial timestep*/
    void SetInitStep(const double hinit);
    /* Turn on or off the stablity limit detection. Default off.*/
    void SetStabLimDet(const bool stldet);
    /* Set the maximum order, default 5*/
    void SetMaxOrd(const int maxord);
		/*Reinitialize the solver*/
		void ReInit();
    /* solve ode to tfinal, store soltuion in ode object*/
    void Solve(const double tfinal);
    /* solve ode to equalibrium.
     * tolfac: conservative factor of tolerance, default 10. The real torlerance
     *      is set as the relative torlerance times the tolfac.
     * tmax: maximum time for reaching equalibrium, default 3.16e13 s or * * 1Myr.*/
    void SolveEq(const double tolfac=10, const double tmax=3.16e13,
                 const bool verbose=false, const double tmin=3.16e10);
    void PrintStats() const;
    /*write t_solve_ and t_solve_eq to file*/
    void WriteRunTime(FILE *pf) const;
    /*Write nst_last_ and nst_eq_ to file*/
    void WriteNstep(FILE *pf) const;
    /* Write the first and last timestep for solver*/
    void WriteFirstLastStep(FILE *pf) const;
    /*get the total number of steps from solver*/
    int GetNstep() const;
		long long GetNegativeCorrectionCount() const;
		bool ReachedTevolMax() const;
		double GetTevolMaxResidual() const;
		/* get the last actual timestep taken*/
		double GethLast() const;
		/* Get the runtime the last time call Solve*/
		double GettSolve() const;
		/* Get the number of steps taken the last time call Solve*/
		double GetnstepLast() const;

  private:
    Ode &ode_;
    const int kDimen;
    sunrealtype reltol_;
    N_Vector abstol_;
    SUNContext sunctx_;
    SUNMatrix dense_matrix_;
    SUNLinearSolver dense_ls_;
    void *cvode_mem_;
    /*record the run time and number of steps for the last time calling CVode
     * solver in Solve function.*/
    double t_solve_;
    long int nst_last_;
    /*recored the run time and number of steps for the last time calling SolveEq
     * to evolve to equalibrium.*/
    double t_solve_eq_;
    long int nst_eq_;
		long long negative_correction_count_;
		bool reached_tevol_max_;
		double tevol_max_residual_;
};

#endif /*CVODEDENSE_H_*/
