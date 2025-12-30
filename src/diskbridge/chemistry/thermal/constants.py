"""Physical constants and default parameters for thermal chemistry.

This module defines all constants and default values used in the thermal
balance calculations, including carbon ionization rates, cosmic ray ionization,
and temperature solver limits.
"""

from diskbridge._units import Quantity

gamma_cii = Quantity(2.3e-3, 'cm^3/s')
E_cii = Quantity(91.2, 'K')

alpha_rec_c0 = 4.67e-12
T_rec_exp = -0.6

zeta_cr_default = Quantity(1e-17, 's^-1')
pah_scale_default = 1.0
X_C_tot_default = 1.4e-4
Gamma_C0_default = Quantity(3.0e-10, 's^-1')

T_min_solve = Quantity(2.0, 'K')
T_max_solve = Quantity(5000.0, 'K')
