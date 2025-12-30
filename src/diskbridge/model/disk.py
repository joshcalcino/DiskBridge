"""Disk geometry and structure models.

This module provides disk geometry laws and the Disk component class.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from diskbridge._units import Quantity, units
from .core import SubModel
from .field import Field
from .coords import spherical_grids, cylindrical_from_spherical

if TYPE_CHECKING:
    from .core import Model


def aspect_ratio(R: Quantity, h0: Quantity, r0: Quantity, flaring: Quantity) -> Quantity:
    """Compute disk aspect ratio h = H/R as a function of radius.
    
    Parameters
    ----------
    R : Quantity
        Cylindrical radius [length]
    h0 : Quantity
        Aspect ratio at reference radius (dimensionless)
    r0 : Quantity
        Reference radius [length]
    flaring : Quantity
        Flaring index (dimensionless)
        
    Returns
    -------
    Quantity
        Aspect ratio h(R) = h0 * (R/r0)^flaring (dimensionless)
        
    Notes
    -----
    For a constant aspect ratio disk, set flaring = 0.
    """
    return h0 * (R / r0) ** flaring


def scale_height(R: Quantity, h0: Quantity, r0: Quantity, flaring: Quantity) -> Quantity:
    """Compute disk scale height H(R) = h(R) * R.
    
    Parameters
    ----------
    R : Quantity
        Cylindrical radius [length]
    h0 : Quantity
        Aspect ratio at reference radius (dimensionless)
    r0 : Quantity
        Reference radius [length]
    flaring : Quantity
        Flaring index (dimensionless)
        
    Returns
    -------
    Quantity
        Scale height H(R) = h0 * (R/r0)^flaring * R [length]
        
    Notes
    -----
    This is the canonical scale height definition. For flaring=0, H ~ R (constant h).
    """
    return aspect_ratio(R, h0, r0, flaring) * R


def rho_gaussian_from_sigma(Sigma: Quantity, z: Quantity, H: Quantity) -> Quantity:
    """Compute 3D density from surface density assuming Gaussian vertical profile.
    
    Parameters
    ----------
    Sigma : Quantity
        Surface density [mass/area]
    z : Quantity
        Height above midplane [length]
    H : Quantity
        Scale height [length]
        
    Returns
    -------
    Quantity
        Volume density rho(z) = Sigma/(sqrt(2pi)H) * exp(-z^2/(2H^2)) [mass/volume]
        
    Notes
    -----
    This assumes vertical hydrostatic equilibrium with an isothermal sound speed.
    The midplane density is rho_mid = Sigma / (sqrt(2pi) * H).
    """
    rho_mid = Sigma / (np.sqrt(2.0 * np.pi) * H)
    expo = (-(z**2) / (2.0 * H**2)).to("dimensionless").magnitude
    return rho_mid * np.exp(expo)


def midplane_theta_index(theta_centers: Quantity) -> int:
    """Find the theta index closest to the midplane (theta = pi/2).
    
    Parameters
    ----------
    theta_centers : Quantity
        Theta cell centers [angle]
        
    Returns
    -------
    int
        Index of theta cell closest to midplane
        
    Notes
    -----
    In spherical coordinates, the midplane is at theta = pi/2.
    """
    return int(np.argmin(np.abs(theta_centers.to("radian") - 0.5 * np.pi)))


def keplerian_frequency(r: Quantity, mstar: Quantity) -> Quantity:
    """Compute Keplerian orbital frequency Omega_K = sqrt(G M_star / r^3).
    
    Parameters
    ----------
    r : Quantity
        Spherical or cylindrical radius [length]
    mstar : Quantity
        Stellar mass [mass]
        
    Returns
    -------
    Quantity
        Keplerian frequency Omega_K [1/time]
    """
    G = units('G')
    return np.sqrt(G * mstar / r**3)


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

        r_grid, phi_grid, theta_grid = spherical_grids(r_sph, phi_sph, theta_sph)
        R_cyl, z_cyl = cylindrical_from_spherical(r_grid, theta_grid)

        H = scale_height(R_cyl, h0, r0, fl)
        Sigma_3d = Sigma[:, :, None] * np.ones(len(theta_sph))
        
        rho_3d = rho_gaussian_from_sigma(Sigma_3d, z_cyl, H)

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
