#ifndef CO_PHASE_H_
#define CO_PHASE_H_

#include <cmath>

namespace co_phase {

inline double v_th_co(const double Tgas, const double kB, const double mCO) {
    if (Tgas <= 0.0) return 0.0;
    const double pi = 3.141592653589793;
    return std::sqrt(8.0 * kB * Tgas / (pi * mCO));
}

inline double co_freezeout_rate(const double nH, const double Tgas, const double sigma_d_per_H,
                               const double kB, const double mCO) {
    return (sigma_d_per_H * nH) * v_th_co(Tgas, kB, mCO);
}

inline double co_thermal_desorption_rate(const double Tdust, const double nu0, const double Ebind) {
    if (Tdust <= 0.0) return 0.0;
    return nu0 * std::exp(-Ebind / Tdust);
}

inline double co_photodesorption_surface_rate(const double chi, const double F_DRAINE, const double Y_CO,
                                             const double N_SURF, const int N_LAY) {
    return (chi * F_DRAINE) * (Y_CO / (4.0 * N_SURF * static_cast<double>(N_LAY)));
}

inline double co_active_ice_max(const double nH, const double sigma_d_per_H, const double N_SURF,
                               const int N_LAY) {
    return (sigma_d_per_H * nH) * N_SURF * static_cast<double>(N_LAY);
}

inline void co_active_ice(const double nH, const double sigma_d_per_H, const double nco_ice,
                          const double N_SURF, const int N_LAY,
                          double &n_act_max, double &n_act) {
    n_act_max = co_active_ice_max(nH, sigma_d_per_H, N_SURF, N_LAY);
    n_act = (nco_ice > n_act_max) ? n_act_max : nco_ice;
}

inline double co_photodesorption_R(const double k_pd_surf, const double n_ice_act) {
    return k_pd_surf * n_ice_act;
}

} // namespace co_phase

#endif
