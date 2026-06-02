"""
Physical and model constants for DiskBridge.

This module loads all constants from config.toml at import time and converts
them to CGS floats for use in numerical code (especially Numba kernels).

Constants are organized into sections:
- Physical constants (from Pint registry)
- Astronomical constants (from config)
- Common chemistry parameters (shared across models)
- Model-specific parameters (pinte_switches, layered_column_switches)

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

# Draine ISRF UV energy density (erg/cm^3)
# Reference: Draine 1978, ApJS, 36, 595
U_DRAINE = 9.0e-14

# Fixed half-cell contribution for the starting cell in HEALPix ray integrations.
HEALPIX_SELF_WEIGHT = 0.5

# =============================================================================
# RADMC-3D EXTERNAL BOUNDARY FIELD
# =============================================================================

# Scaling of the Draine/Leiden UV-optical-NIR ISRF is controlled only by
# params.external_uv_chi. The IR background is independent of chi and is
# normalized so its integrated intensity corresponds to EXTERNAL_IR_TBACK.
EXTERNAL_IR_BACKGROUND = True
EXTERNAL_IR_TBACK = 10.0
EXTERNAL_IR_TCOLOR = 18.0
EXTERNAL_IR_BETA = 1.7
EXTERNAL_IR_REFERENCE_WAVELENGTH_MICRON = 250.0
EXTERNAL_CMB = True

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

# CO surface / photodesorption (used by gow17 enable_co_phase)
F_DRAINE = Quantity(_chem_common['F_DRAINE']).to_base_units().magnitude      # 1/(cm^2 s)
N_LAY = int(_chem_common['N_LAY'])                                           # dimensionless
N_SURF = Quantity(_chem_common['n_surf']).to_base_units().magnitude          # 1/cm^2
Y_CO = float(_chem_common['Y_CO'])                                           # molecules/photon

_gow17_cfg = _cfg.get('chemistry', {}).get('gow17', {})
_gow17_dust = _gow17_cfg.get('dust', {})
_gow17_co_phase = _gow17_cfg.get('co_phase', {})
SIGMA_D_ISM_REF = Quantity(
    _gow17_dust.get('sigma_d_ISM_ref', '1.0e-21 cm^2')
).to_base_units().magnitude
F_CRUV_CO_PDES_REF = Quantity(
    _gow17_co_phase.get('F_CRUV_CO_pdes_ref', '1.0e4 1/(cm^2 s)')
).to_base_units().magnitude
ZETA_CRUV_REF = Quantity(
    _gow17_co_phase.get('zeta_ref', '1.0e-17 1/s')
).to_base_units().magnitude
K_CRDES_CO = Quantity(
    _gow17_co_phase.get('k_crdes_CO', '0.0 1/s')
).to_base_units().magnitude

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
# ABUNDANCE SWITCHES MODEL
# =============================================================================

_ab_switch = _cfg['chemistry']['layered_column_switches']

CD_THRESHOLD_PDES = Quantity(_ab_switch['CD_threshold_pdes']).to_base_units().magnitude  # cm^-2
CD_THRESHOLD_PDISS = Quantity(_ab_switch['CD_threshold_pdiss']).to_base_units().magnitude  # cm^-2
