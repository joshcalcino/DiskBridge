"""
Physical and model constants for DiskBridge.

This module loads all constants from config.toml at import time and converts
them to CGS floats for use in numerical code (especially Numba kernels).

Constants are organized into sections:
- Physical constants (from Pint registry)
- Astronomical constants (from config)
- Common chemistry parameters (shared across models)
- Model-specific parameters (pinte_switches, co_two_phase, thermal_balance_v1)

Users can override defaults by calling diskbridge.load_config(path) before
importing this module.
"""

from __future__ import annotations

from diskbridge._units import units, Quantity
from diskbridge._config import get_config

_cfg = get_config()

# =============================================================================
# PHYSICAL CONSTANTS (from Pint registry - not overridable)
# All converted to CGS base units as floats for Numba compatibility
# =============================================================================

K_B = units('k_B').to_base_units().magnitude           # erg/K (1.381e-16)
M_H = units('m_H').to_base_units().magnitude           # g (1.674e-24)
C_LIGHT = units('c').to_base_units().magnitude         # cm/s (2.998e10)
H_PLANCK = units('h').to_base_units().magnitude        # erg*s (6.626e-27)
EV_TO_ERG = units('eV').to('erg').magnitude            # erg/eV (1.602e-12)
SIGMA_SB = units('sigma_SB').to_base_units().magnitude # erg/(cm^2 s K^4)
G_GRAV = units('G').to_base_units().magnitude          # cm^3/(g s^2)

# =============================================================================
# ASTRONOMICAL CONSTANTS (from config)
# =============================================================================

_astro = _cfg['astronomical']
SOLAR_MASS = Quantity(_astro['solar_mass']).to_base_units().magnitude        # g
SOLAR_RADIUS = Quantity(_astro['solar_radius']).to_base_units().magnitude    # cm
SOLAR_LUMINOSITY = Quantity(_astro['solar_luminosity']).to_base_units().magnitude  # erg/s
SOLAR_TEMPERATURE = Quantity(_astro['solar_temperature']).to_base_units().magnitude  # K
EARTH_MASS = Quantity(_astro['earth_mass']).to_base_units().magnitude        # g
EARTH_RADIUS = Quantity(_astro['earth_radius']).to_base_units().magnitude    # cm
JUPITER_MASS = Quantity(_astro['jupiter_mass']).to_base_units().magnitude    # g
JUPITER_RADIUS = Quantity(_astro['jupiter_radius']).to_base_units().magnitude  # cm
PARSEC = Quantity(_astro['parsec']).to_base_units().magnitude                # cm
AU = Quantity(_astro['au']).to_base_units().magnitude                        # cm
T_CMB = Quantity(_astro['T_cmb']).to_base_units().magnitude                  # K

# =============================================================================
# COMMON CHEMISTRY PARAMETERS (shared across all chemistry models)
# =============================================================================

_chem_common = _cfg['chemistry']['common']

# Elemental abundances (dimensionless, relative to H)
X_C_TOT = float(_chem_common['X_C_tot'])
X_O_TOT = float(_chem_common['X_O_tot'])

# CO molecular properties
E_BIND_CO = Quantity(_chem_common['E_bind_co']).to_base_units().magnitude    # K
NU0_CO = Quantity(_chem_common['nu0_co']).to_base_units().magnitude          # Hz (s^-1)
SIGMA_D_PER_H = Quantity(_chem_common['sigma_d_per_H']).to_base_units().magnitude  # cm^2

# =============================================================================
# PINTE SWITCHES MODEL
# =============================================================================

_pinte = _cfg['chemistry']['pinte_switches']

T_FRZ = Quantity(_pinte['T_frz']).to_base_units().magnitude   # K
EPS_FRZ = float(_pinte['eps_frz'])                            # dimensionless
LOG_CHI_OVER_NH_PDISS = float(_pinte['log_chi_over_nH_pdiss'])  # dimensionless
LOG_CHI_OVER_NH_PDES = float(_pinte['log_chi_over_nH_pdes'])    # dimensionless
EPS_CHI = float(_pinte['eps_chi'])                            # dimensionless

# =============================================================================
# CO TWO-PHASE MODEL
# =============================================================================

_co2p = _cfg['chemistry']['co_two_phase']

TAU_CO_FORM = Quantity(_co2p['tau_co_form']).to_base_units().magnitude  # s
K0_CO = Quantity(_co2p['k0_co']).to_base_units().magnitude              # s^-1
ALPHA_PD_ICE = float(_co2p['alpha_pd_ice'])                             # dimensionless

# Density-capped tau_form model
TAU_FORM_N0 = Quantity(_co2p['tau_form_n0']).to_base_units().magnitude      # cm^-3
TAU_FORM_TAU0 = Quantity(_co2p['tau_form_tau0']).to_base_units().magnitude  # s
TAU_FORM_TAU_MIN = Quantity(_co2p['tau_form_tau_min']).to_base_units().magnitude  # s
TAU_FORM_ALPHA = float(_co2p['tau_form_alpha'])                         # dimensionless

# =============================================================================
# COMMON THERMAL PARAMETERS (shared across thermal models)
# =============================================================================

_therm_common = _cfg['thermal']['common']

# Carbon ionization and cooling
GAMMA_CII = Quantity(_therm_common['gamma_cii']).to_base_units().magnitude  # cm^3/s
E_CII = Quantity(_therm_common['E_cii']).to_base_units().magnitude          # K
N_CRIT_CII = Quantity(_therm_common['n_crit_cii']).to_base_units().magnitude  # cm^-3
ALPHA_REC_C0 = Quantity(_therm_common['alpha_rec_c0']).to_base_units().magnitude  # cm^3/s
T_REC_EXP = float(_therm_common['T_rec_exp'])                               # dimensionless

# Dust properties
SIGMA_DUST = Quantity(_therm_common['sigma_dust']).to_base_units().magnitude  # cm^2
F_DUST = float(_therm_common['f_dust'])                                     # dimensionless

# =============================================================================
# THERMAL BALANCE V1 MODEL
# =============================================================================

_thermal_v1 = _cfg['thermal']['thermal_balance_v1']

# Cosmic ray heating
ZETA_CR = Quantity(_thermal_v1['zeta_cr']).to_base_units().magnitude        # s^-1
HEATING_PER_CR = Quantity(_thermal_v1['heating_per_cr']).to_base_units().magnitude  # erg

# Photoelectric heating
PE_HEATING_RATE_0 = Quantity(_thermal_v1['pe_heating_rate_0']).to_base_units().magnitude  # erg/s
PAH_SCALE = float(_thermal_v1['pah_scale'])                                 # dimensionless

# Carbon photoionization
GAMMA_C0 = Quantity(_thermal_v1['Gamma_C0']).to_base_units().magnitude      # s^-1

# Accretion heating
ALPHA_ACC = float(_thermal_v1['alpha_acc'])                                 # dimensionless

# Solver parameters
T_MIN_SOLVE = Quantity(_thermal_v1['T_min_solve']).to_base_units().magnitude  # K
T_MAX_SOLVE = Quantity(_thermal_v1['T_max_solve']).to_base_units().magnitude  # K
N_ITER_THERMAL = int(_thermal_v1['n_iter'])
TOL_THERMAL = float(_thermal_v1['tol'])
BETA_CII = float(_thermal_v1['beta_cii'])
MAX_BISECT_ITER = int(_thermal_v1['max_bisect_iter'])
BISECT_TOL = float(_thermal_v1['bisect_tol'])
