from __future__ import annotations

from diskbridge._units import Quantity


T_FRZ_DEFAULT = Quantity(21.0, 'K')
EPS_DEFAULT = Quantity(8e-5, 'dimensionless')
LOG_CHI_OVER_NH_PDISS = Quantity(-6.0, 'dimensionless')
LOG_CHI_OVER_NH_PDES = Quantity(-7.0, 'dimensionless')
eps_chi = 1.0e-99


XCO_TOT_DEFAULT = 1.0e-4
TAU_CO_FORM_DEFAULT = Quantity(1.0e5, 'yr')
E_BIND_CO_DEFAULT = 855.0
NU0_CO_DEFAULT = 1.0e12
SIGMA_D_PER_H_DEFAULT = 1.0e-21
ALPHA_PD_ICE_DEFAULT = 1.0e-12
K0_CO_DEFAULT = Quantity(2.0e-10, '1/s')
