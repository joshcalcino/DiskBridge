from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from .core import SubModel
from .field import Field

if TYPE_CHECKING:
    from .core import Model


class _RegionDustConfigurator:
    def __init__(self, global_dust, mask: Optional[Field], is_disk_region: bool):
        self._dust = global_dust
        self._mask = mask
        self._is_disk_region = is_disk_region

    def set_distribution(self, mode: str = "proportional", **kwargs):
        if self._mask is None:
            raise ValueError(
                "Region has no mask defined; call set_mask_from_geometry first"
            )

        if mode == "settling" and not self._is_disk_region:
            raise ValueError(
                "Settling mode can only be used on regions defined as disk. "
                "Use is_a_disk=True when creating the mask."
            )

        self._dust.add_component_from_mask(
            mask=self._mask,
            mode=mode,
            **kwargs,
        )


class Disk(SubModel):
    _required_parameters = [
        "aspectratio",
        "flaringindex",
        "r0",
    ]

    def __init__(self, parent: "Model", parameters: Dict[str, Any]):
        SubModel.__init__(self, parent)
        for key in self._required_parameters:
            if key not in parameters:
                raise ValueError(f"Missing required parameter in Disk class: {key}")
        self.parameters: Dict[str, Any] = parameters

    @property
    def gas(self) -> "SubModel":
        return self.parent.gas

    @property
    def dust(self):
        return _RegionDustConfigurator(
            global_dust=self.parent.dust,
            mask=self.mask,
            is_disk_region=self.is_disk_region,
        )

    def puff_up_disk(
        self,
        n: int,
        zmax_over_H: float = 5.0,
    ) -> "Model":
        r = self.parent.mesh.centers("r")
        h0 = self.parameters["aspectratio"]
        fl = self.parameters["flaringindex"]
        r0 = self.parameters["r0"]

        if "surface_density" not in self.parent.gas:
            raise KeyError("Missing gas field 'surface_density'")
        Sigma = self.parent.gas["surface_density"].data

        sph_mesh = self.parent.mesh.to_spherical_by_scale_height(
            ncol=n, aspect_ratio=h0, zmax_over_H=zmax_over_H
        )

        r_sph = sph_mesh.centers("r")
        theta_sph = sph_mesh.centers("theta")
        phi_sph = sph_mesh.centers("phi")

        r_mag = r_sph.magnitude
        theta_mag = theta_sph.magnitude
        phi_mag = phi_sph.magnitude
        r_units = r_sph.units

        r_grid, phi_grid, theta_grid = np.meshgrid(
            r_mag, phi_mag, theta_mag, indexing="ij"
        )
        r_grid = r_grid * r_units

        z_cyl = r_grid * np.cos(theta_grid)

        h_r = h0 * (r_sph / r0) ** fl
        H = h_r * r_sph
        H_3d = H[:, None, None] * np.ones_like(theta_grid)

        Sigma_3d = Sigma[:, :, None] * np.ones(len(theta_sph))

        rho_3d = Sigma_3d / (np.sqrt(2 * np.pi) * H_3d) * np.exp(
            -z_cyl**2 / (2 * H_3d**2)
        )

        ntheta = len(theta_sph)
        nr = len(r_sph)
        nphi = len(phi_sph)

        vr_sph = None
        vphi_sph = None
        if "vr" in self.parent.gas:
            vr_2d = self.parent.gas["vr"].data
            vr_sph = vr_2d[:, :, None] * np.ones(ntheta)
        if "vphi" in self.parent.gas:
            vphi_2d = self.parent.gas["vphi"].data
            vphi_sph = vphi_2d[:, :, None] * np.ones(ntheta)

        vunit = None
        if "vr" in self.parent.gas:
            vunit = getattr(self.parent.gas["vr"].data, "units", None)
        elif "vphi" in self.parent.gas:
            vunit = getattr(self.parent.gas["vphi"].data, "units", None)
        vtheta = np.zeros((nr, nphi, ntheta))
        if vunit is not None:
            vtheta = vtheta * vunit

        self.parent.mesh = sph_mesh
        self.mesh = sph_mesh
        if hasattr(self.parent, "gas") and self.parent.gas is not None:
            self.parent.gas.mesh = sph_mesh
        self.coord_system = "spherical"

        self.parent.gas.clear()
        self.parent.gas_register(
            "density",
            Field(
                quantity="density",
                data=rho_3d,
                axis_order=("r", "phi", "theta"),
            ),
        )
        if vr_sph is not None:
            self.parent.gas_register(
                "vr",
                Field(
                    quantity="vr",
                    data=vr_sph,
                    axis_order=("r", "phi", "theta"),
                ),
            )
        if vphi_sph is not None:
            self.parent.gas_register(
                "vphi",
                Field(
                    quantity="vphi",
                    data=vphi_sph,
                    axis_order=("r", "phi", "theta"),
                ),
            )
        self.parent.gas_register(
            "vtheta",
            Field(
                quantity="vtheta",
                data=vtheta,
                axis_order=("r", "phi", "theta"),
            ),
        )
        return self
