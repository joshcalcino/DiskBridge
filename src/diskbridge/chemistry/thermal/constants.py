"""Physical constants and default parameters for thermal chemistry.

This module defines all constants and default values used in the thermal
balance calculations, including carbon ionization rates, cosmic ray ionization,
and temperature solver limits.

Two sets of constants are provided:
1. Pint Quantity versions for the Python API boundary
2. Pure float versions (suffix _val) for Numba kernels
"""

from diskbridge._units import Quantity

# -----------------------------------------------------------------------------
# C II fine-structure cooling
# -----------------------------------------------------------------------------
gamma_cii = Quantity(2.3e-3, 'cm^3/s')
E_cii = Quantity(91.2, 'K')

gamma_cii_val = 2.3e-3      # cm^3/s - collisional de-excitation rate
E_cii_val = 91.2            # K - energy of 158 um transition
n_crit_cii_val = 3e3        # cm^-3 - critical density for C II

# -----------------------------------------------------------------------------
# Carbon recombination (C+ + e- -> C)
# UMIST-style: alpha_rec = alpha0 * (T/300)^T_rec_exp
# -----------------------------------------------------------------------------
alpha_rec_c0 = 4.67e-12     # cm^3/s at 300 K
T_rec_exp = -0.6            # temperature exponent

# -----------------------------------------------------------------------------
# Physical constants (CGS, for kernels)
# -----------------------------------------------------------------------------
k_B_cgs = 1.380649e-16      # erg/K - Boltzmann constant
m_H_cgs = 1.6735575e-24     # g - hydrogen mass
eV_to_erg = 1.602176634e-12 # erg/eV

# -----------------------------------------------------------------------------
# Default parameter values (Pint versions for API)
# -----------------------------------------------------------------------------
zeta_cr_default = Quantity(1e-17, 's^-1')
pah_scale_default = 1.0
X_C_tot_default = 1.4e-4
Gamma_C0_default = Quantity(3.0e-10, 's^-1')

T_min_solve = Quantity(2.0, 'K')
T_max_solve = Quantity(5000.0, 'K')

# -----------------------------------------------------------------------------
# Default parameter values (pure floats for kernels)
# -----------------------------------------------------------------------------
zeta_cr_default_val = 1e-17       # s^-1
Gamma_C0_default_val = 3.0e-10    # s^-1
T_min_val = 2.0                   # K
T_max_val = 5000.0                # K

# Cosmic ray heating: ~20 eV deposited per ionization
heating_per_cr_ionization_val = 20.0 * eV_to_erg  # erg

# Photoelectric heating coefficient (Bakes & Tielens 1994)
pe_heating_rate_0_val = 1.3e-24   # erg s^-1 (base rate)

# Gas-dust coupling defaults
alpha_acc_default_val = 0.3       # accommodation coefficient
sigma_dust_val = 1e-21            # cm^2 - effective dust cross section
f_dust_val = 0.01                 # dust-to-gas mass ratio

# -----------------------------------------------------------------------------
# Term bitmask flags for kernel dispatch
# -----------------------------------------------------------------------------
TERM_CR = 1 << 0     # cosmic ray heating
TERM_PE = 1 << 1     # photoelectric heating
TERM_CII = 1 << 2    # C II 158 um cooling
TERM_GD = 1 << 3     # gas-dust exchange
