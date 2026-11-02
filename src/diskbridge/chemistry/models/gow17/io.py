"""GOW17 result preparation for line transfer."""

# db-keywords: gow17, line-transfer, units, chemistry
# db-role: canonical
# db-scope: package
# db-purpose: Derive line-transfer collider densities from GOW17 results.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from diskbridge._units import Quantity

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel


def add_default_line_colliders_from_gow17(
    result,
    rad: Optional['RadModel'] = None,
    opr: float = 3.0,
):
    """Add GOW17-derived line-transfer collider densities to a result."""

    del rad
    opr_value = float(opr)
    if not np.isfinite(opr_value) or opr_value <= 0.0:
        raise ValueError("line_h2_opr must be a positive finite number")
    if "h2" not in result.number_densities:
        raise KeyError("GOW17 line collider derivation requires number_densities['h2']")

    n_h2 = result.number_densities["h2"].to("cm^-3")
    f_p = 1.0 / (1.0 + opr_value)
    f_o = opr_value / (1.0 + opr_value)
    result.number_densities["p-h2"] = Quantity(f_p * n_h2.magnitude, "cm^-3")
    result.number_densities["o-h2"] = Quantity(f_o * n_h2.magnitude, "cm^-3")
    if "h+" not in result.number_densities:
        raise KeyError("GOW17 line collider derivation requires number_densities['h+']")
    if "he" not in result.number_densities:
        raise KeyError("GOW17 line collider derivation requires number_densities['he']")
    result.meta["line_h2_opr"] = opr_value
    result.meta["derived_line_colliders"] = ["h", "e", "h+", "he", "p-h2", "o-h2"]
    return result
