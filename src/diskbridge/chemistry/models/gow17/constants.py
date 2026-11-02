"""GOW17 dust and CO-phase reference constants in CGS units."""

# db-keywords: gow17, chemistry, config, units
# db-role: canonical
# db-scope: package
# db-purpose: GOW17-specific dust and CO-phase reference constants.

from diskbridge._config import get_config
from diskbridge._units import Quantity

_cfg = get_config()

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
