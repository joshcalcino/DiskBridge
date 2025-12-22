from __future__ import annotations

from dataclasses import dataclass, field as dcfield
from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import numpy as np
import pickle

from .mesh import Mesh, Axis
from .field import Field
from diskbridge._logging import logger
from diskbridge._units import Quantity, units
from .utils import validate_field_against_mesh
from .clipping import compute_clip_indexer

class Model:

    """
    Add in documentation
    """

    def __init__(self):
        self.coord_system: Optional[str] = None
        self.variables: Dict[str, Any] = {}
        self.compile_options: Dict[str, Optional[bool]] = {}
        self.macros: Dict[str, float] = {}
        self.mesh: Optional[Mesh] = None
        self.file_units: Optional[str] = None # 'kms', 'cgs', or 'code'
        self.directory: Optional[str] = None
        self.n_file: Optional[int] = None
        self.filename: Optional[str] = None 
        
        # Initialize submodels
        self.gas: SubModel = None
        self.disk: Disk = None
        self.dust: 'Dust' = None  # Dust submodel
        
    def get_variables(self) -> Dict[str, Any]:
        return dict(self.variables)

    def gas_register(self, name: str, field: Field) -> None:
        # validate against the model's single mesh
        validate_field_against_mesh(field, self.mesh)  # type: ignore[arg-type]
        self.gas.register(name, field)

    def gas_register_lazy(self, name: str, builder: Callable[[], Field]) -> None:
        self.gas.register_lazy(name, builder)

    def _apply_rescaling(self, length_scale=None, mass_scale=None) -> None:
        """
        Internal method to apply length and mass rescaling during initialization.
        
        Parameters
        ----------
        length_scale : float, optional
            Dimensionless multiplier for length (e.g., 10 means 1 code_length = 10 au)
        mass_scale : float, optional
            Dimensionless multiplier for mass (e.g., 2 means 1 code_mass = 2 M_sun)
            
        Notes
        -----
        Code units are already defined as:
        - code_length = 1 au
        - code_mass = 1 M_sun
        - code_time = sqrt(au^3 / (G * M_sun))

        """
        # Get dimensionless scaling factors (default to 1 if not provided)
        length_factor = float(length_scale) if length_scale is not None else 1.0
        mass_factor = float(mass_scale) if mass_scale is not None else 1.0
        
        # Derived scaling factors based on Keplerian dynamics
        # Time: T^2 ~ L^3/M -> time_factor = sqrt(length_factor^3 / mass_factor)
        time_factor = np.sqrt(length_factor**3 / mass_factor)
        
        # Velocity: V = L/T -> velocity_factor = length_factor / time_factor = sqrt(mass_factor / length_factor)
        velocity_factor = np.sqrt(mass_factor / length_factor)
        
        # Apply rescaling only if at least one scale is provided
        if length_scale is not None or mass_scale is not None:
            # Rescale mesh coordinates
            if self.mesh is not None:
                self.mesh = self.mesh.rescale_length(length_factor)
            
            # Rescale gas fields
            if self.gas is not None:
                for name, field in list(self.gas.items()):
                    new_data = field.data
                    
                    if 'density' in name or 'surface_density' in name:
                        # Volume density: rho = M/L^3 -> scale by mass_factor/length_factor^3
                        # Surface density: Sigma = M/L^2 -> scale by mass_factor/length_factor^2
                        if 'surface' in name:
                            new_data = field.data * (mass_factor / length_factor**2)
                        else:
                            new_data = field.data * (mass_factor / length_factor**3)
                    
                    elif name in ['vr', 'vphi', 'vz', 'vtheta']:
                        # Velocity: V = L/T -> scale by sqrt(M/L)
                        new_data = field.data * velocity_factor
                    
                    elif name == 'temperature':
                        # Temperature: cutemp ~ M/L, so T -> T * (M/L)
                        # When length increases, temperature decreases
                        temp_factor = mass_factor / length_factor
                        new_data = field.data * temp_factor
                    
                    # Replace field with rescaled version
                    if new_data is not field.data:
                        self.gas[name] = Field(
                            data=new_data,
                            quantity=field.quantity,
                            axis_order=field.axis_order,
                            attrs=field.attrs
                        )
            
            # Store the scaling factors
            self.length_scale = length_factor
            self.mass_scale = mass_factor
            self.time_scale = time_factor
            self.velocity_scale = velocity_factor

    def load_model(
        self,
        path: Union[str, Path],
        reader: str = "fargo",
        file_n: int = 0,
        file_units: str = "code",
        length_scale: Optional[float] = None,
        mass_scale: Optional[float] = None,
    ) -> "Model":
        """
        Load a hydro snapshot and populate this Model, then return self.
        
        Parameters
        ----------
        path : str or Path
            Path to the simulation data directory
        reader : str, optional
            Reader type (default: "fargo")
        file_n : int, optional
            File number to load (default: 0)
        file_units : str, optional
            Unit system of the files: 'code', 'cgs', or 'kms' (default: 'code')
        length_scale : float, optional
            Dimensionless multiplier for length units. For code units (1 au), 
            this rescales by this factor (e.g., 10 means 1 code_length = 10 au).
            Time is also rescaled following Keplerian dynamics.
        mass_scale : float, optional
            Dimensionless multiplier for mass units. For code units (1 M_sun),
            this rescales by this factor (e.g., 2 means 1 code_mass = 2 M_sun).
            Velocities are rescaled as V -> V*sqrt(mass_scale/length_scale).
        
        Returns
        -------
        Model
            The populated model instance
            
        Notes
        -----
        Rescaling follows Keplerian dynamics where T^2 ~ L^3/M:
        - Lengths scale by length_scale
        - Masses scale by mass_scale  
        - Times scale by sqrt(length_scale^3/mass_scale)
        - Velocities scale by sqrt(mass_scale/length_scale)
        - Densities scale by mass_scale/length_scale^3
        """
        if length_scale is None or mass_scale is None:
            try:
                from diskbridge import params as _global_params
            except Exception:
                _global_params = None
            if _global_params is not None:
                if length_scale is None and hasattr(_global_params, "length_scale"):
                    length_scale = float(_global_params.length_scale)
                if mass_scale is None and hasattr(_global_params, "mass_scale"):
                    mass_scale = float(_global_params.mass_scale)
        p = Path(path)
        if reader.lower() == "fargo":
            from .readers.fargo import read_fargo_snapshot

            snap = read_fargo_snapshot(p, file_n=file_n, file_units=file_units)
        else:
            raise ValueError(f"Unsupported reader: {reader}")

        # Populate instance
        self.coord_system = snap["coord_system"]
        self.variables = snap["variables"]
        self.compile_options = snap.get("compile_options", {})
        self.macros = snap.get("macros", {})
        self.mesh = snap["mesh"]
        self.file_units = file_units 
        self.directory = str(p)
        self.n_file = file_n
        self.filename = None
        
        # Initialize submodels
        self.gas = SubModel(self)
        
        # Initialize Dust submodel (lazy import to avoid circular dependency)
        from .dust import Dust
        self.dust = Dust(self)

        # Initialize Disk parameters if present
        if "disk_parameters" in snap:
            self.disk = Disk(self, snap["disk_parameters"])

        # Register gas fields
        for name, field in snap["gas_fields"].items():
            self.gas_register(name, field)
        
        # Apply rescaling if requested
        if length_scale is not None or mass_scale is not None:
            self._apply_rescaling(length_scale=length_scale, mass_scale=mass_scale)
            # Update SubModel mesh references after rescaling
            if self.gas is not None:
                self.gas.mesh = self.mesh
            if self.dust is not None:
                self.dust.mesh = self.mesh
            if hasattr(self, 'disk') and self.disk is not None:
                self.disk.mesh = self.mesh

        return self
    
    def set_mask_from_geometry(
        self,
        r_min: Optional[Quantity] = None,
        r_max: Optional[Quantity] = None,
        theta_min: Optional[Quantity] = None,
        theta_max: Optional[Quantity] = None,
        honrmax: Optional[float] = None,
        is_a_disk: bool = False,
    ) -> SubModel:
        """Define a masked region on the model.
        
        If is_a_disk=True and self.disk exists, configure self.disk's mask.
        Otherwise, create and return a new SubModel with the requested mask.
        
        Args:
            r_min: Minimum radius
            r_max: Maximum radius
            theta_min: Minimum colatitude (spherical)
            theta_max: Maximum colatitude (spherical)
            honrmax: Number of pressure scale heights (disk-specific, requires self.disk)
            is_a_disk: Mark region as disk (enables settling mode)
            
        Returns:
            SubModel (or Disk) with mask applied
        """
        mesh = self.mesh
        if mesh is None:
            raise ValueError("Model has no mesh")
            
        if mesh.coord_system != 'spherical':
            raise ValueError(
                f"set_mask_from_geometry only supports spherical coordinates, "
                f"got {mesh.coord_system}"
            )
        
        # Determine target region
        if is_a_disk and self.disk is not None:
            target_region = self.disk
        else:
            target_region = SubModel(self)
        
        # Get coordinate arrays
        r = mesh.centers('r')
        theta = mesh.centers('theta')
        phi = mesh.centers('phi')
        
        r_grid, phi_grid, theta_grid = np.meshgrid(r, phi, theta, indexing='ij')
        
        # Start with all True
        mask = np.ones_like(r_grid, dtype=bool)
        
        # Apply radial constraints
        if r_min is not None:
            mask &= (r_grid.magnitude >= r_min.to(r.units).magnitude)
        if r_max is not None:
            mask &= (r_grid.magnitude <= r_max.to(r.units).magnitude)
        
        # If honrmax is provided and we have disk parameters, compute hydrostatic mask
        if honrmax is not None and self.disk is not None:
            # Convert spherical to cylindrical for height calculation
            R_cyl = r_grid * np.sin(theta_grid)
            z_cyl = r_grid * np.cos(theta_grid)
            
            # Get disk parameters
            h0 = self.disk.parameters["aspectratio"]
            fl = self.disk.parameters["flaringindex"]
            r0 = self.disk.parameters["r0"]
            
            # h(R_cyl) = h0 * (R_cyl/r0)^fl, dimensionless
            h = h0 * (R_cyl / r0.to(R_cyl.units)) ** fl
            
            # H(R_cyl) = h * R_cyl, scale height with units
            H = h * R_cyl
            
            # Mask: |z| <= honrmax * H(R_cyl)
            z_max = honrmax * H
            mask &= (np.abs(z_cyl) <= z_max)
        
        # Apply theta constraints if given
        if theta_min is not None:
            mask &= (theta_grid.magnitude >= theta_min.to(theta.units).magnitude)
        if theta_max is not None:
            mask &= (theta_grid.magnitude <= theta_max.to(theta.units).magnitude)
        
        axis_order = ('r', 'phi', 'theta')
        
        mask_quantity = Quantity(mask, 'dimensionless')
        mask_field = Field(
            data=mask_quantity,
            quantity='mask',
            axis_order=axis_order,
        )
        
        target_region.mask = mask_field
        
        # Mark as disk region if requested
        if is_a_disk:
            target_region.is_disk_region = True
        
        logger.info(
            f"{target_region.__class__.__name__} mask set: {np.sum(mask)} / {mask.size} cells "
            f"({100*np.sum(mask)/mask.size:.1f}%)"
        )
        
        return target_region
    
    def set_mask_from_joos_disk(
        self,
        rho_disk_min: Quantity,
        *,
        fthres: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 2.0,
        fthres_vr: Optional[Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]] = None,
        rho_core_min: Optional[Quantity] = None,
        r_max_for_axis: Optional[Quantity] = None,
        n_r_bins: Optional[int] = None,
        n_theta_bins: Optional[int] = None,
        r_max: Optional[Quantity] = None,
    ) -> SubModel:
        mesh = self.mesh
        if mesh.coord_system != "spherical":
            raise ValueError(
                f"set_mask_from_joos_disk only supports spherical coordinates, got {mesh.coord_system}"
            )

        r_c = mesh.centers("r")
        theta_c = mesh.centers("theta")
        phi_c = mesh.centers("phi")
        r_e = mesh.edges("r")
        theta_e = mesh.edges("theta")
        phi_e = mesh.edges("phi")
        if r_c is None or theta_c is None or phi_c is None:
            raise ValueError("Mesh is missing one or more spherical center axes")
        if r_e is None or theta_e is None or phi_e is None:
            raise ValueError("Mesh is missing one or more spherical edge axes")

        n_bins_native = len(theta_c)

        r3 = (r_e[1:] ** 3 - r_e[:-1] ** 3) / 3.0
        theta_e_rad = theta_e.to("radian").magnitude
        dcos = np.cos(theta_e_rad[:-1]) - np.cos(theta_e_rad[1:])
        phi_e_rad = phi_e.to("radian").magnitude
        dphi = phi_e_rad[1:] - phi_e_rad[:-1]
        dV = r3[:, None, None] * dphi[None, :, None] * dcos[None, None, :]
        dV_mag = dV.to_base_units().magnitude

        rho = self.gas["density"].data.to_base_units()
        vr = self.gas["vr"].data.to_base_units()
        vphi = self.gas["vphi"].data.to_base_units()
        vtheta = self.gas["vtheta"].data.to_base_units()

        r_grid_mag, phi_grid_mag, theta_grid_mag = np.meshgrid(
            r_c.to_base_units().magnitude,
            phi_c.to("radian").magnitude,
            theta_c.to("radian").magnitude,
            indexing="ij",
        )
        r_grid = r_grid_mag * r_c.to_base_units().units

        if rho_core_min is None:
            rho_core_min = 10.0 * rho_disk_min

        rho_core_thr = rho_core_min.to(rho.units).magnitude
        core_mask = rho.magnitude >= rho_core_thr
        if r_max_for_axis is not None:
            core_mask &= (r_grid.magnitude <= r_max_for_axis.to(r_grid.units).magnitude)

        sin_t = np.sin(theta_grid_mag)
        cos_t = np.cos(theta_grid_mag)
        cos_p = np.cos(phi_grid_mag)
        sin_p = np.sin(phi_grid_mag)

        er_x = sin_t * cos_p
        er_y = sin_t * sin_p
        er_z = cos_t

        et_x = cos_t * cos_p
        et_y = cos_t * sin_p
        et_z = -sin_t

        ep_x = -sin_p
        ep_y = cos_p
        ep_z = 0.0

        r_base_mag = r_grid.to_base_units().magnitude
        x = r_base_mag * er_x
        y = r_base_mag * er_y
        z = r_base_mag * er_z

        vx = vr * er_x + vtheta * et_x + vphi * ep_x
        vy = vr * er_y + vtheta * et_y + vphi * ep_y
        vz = vr * er_z + vtheta * et_z + vphi * ep_z

        mcell = (rho * dV).to_base_units().magnitude
        if not np.any(core_mask):
            rho_disk_thr = rho_disk_min.to(rho.units).magnitude
            core_mask = rho.magnitude >= rho_disk_thr

        w = mcell * core_mask
        Lx = np.sum(w * (y * vz.magnitude - z * vy.magnitude))
        Ly = np.sum(w * (z * vx.magnitude - x * vz.magnitude))
        Lz = np.sum(w * (x * vy.magnitude - y * vx.magnitude))
        Lnorm = float(np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz))
        if Lnorm == 0.0 or not np.isfinite(Lnorm):
            k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            k_hat = np.array([Lx / Lnorm, Ly / Lnorm, Lz / Lnorm], dtype=float)

        z_d = x * k_hat[0] + y * k_hat[1] + z * k_hat[2]
        rx = x - z_d * k_hat[0]
        ry = y - z_d * k_hat[1]
        rz = z - z_d * k_hat[2]
        R_d = np.sqrt(rx * rx + ry * ry + rz * rz)

        invR = np.zeros_like(R_d)
        nz = R_d > 0.0
        invR[nz] = 1.0 / R_d[nz]
        rhatx = rx * invR
        rhaty = ry * invR
        rhatz = rz * invR

        phix = k_hat[1] * rhatz - k_hat[2] * rhaty
        phiy = k_hat[2] * rhatx - k_hat[0] * rhatz
        phiz = k_hat[0] * rhaty - k_hat[1] * rhatx

        vR_d = vx * rhatx + vy * rhaty + vz * rhatz
        vphi_d = vx * phix + vy * phiy + vz * phiz
        vz_d = vx * k_hat[0] + vy * k_hat[1] + vz * k_hat[2]

        if "pressure" in self.gas:
            Pth = self.gas["pressure"].data
        elif "temperature" in self.gas:
            if not hasattr(self, "variables") or "MU" not in self.variables:
                raise ValueError(
                    "Cannot compute thermal pressure from temperature without mean molecular weight. "
                    "Provide gas['pressure'] or ensure the reader sets model.variables['MU']."
                )
            T = self.gas["temperature"].data
            mu_val = float(getattr(self.variables["MU"], "magnitude", self.variables["MU"]))
            Pth = rho.to("g/cm^3") * units("k_B") * T.to("K") / (mu_val * units("m_H"))
        else:
            raise KeyError("Need either gas['pressure'] or gas['temperature'] to evaluate thermal support")
        Pth = Pth.to_base_units()

        r_edges_native = mesh.edges("r")
        if r_edges_native is None:
            raise ValueError("Mesh is missing radial edges")
        r_edges_native = r_edges_native.to_base_units()

        if r_max is not None:
            r_max_base = r_max.to_base_units().magnitude
            r_edges_native = r_edges_native[r_edges_native.magnitude <= r_max_base]
            if len(r_edges_native) < 2:
                raise ValueError("r_max is too small; no radial bins remain")

        nR_native = len(r_edges_native) - 1
        if nR_native < 1:
            raise ValueError("No radial bins available")

        if n_r_bins is None:
            r_edges = r_edges_native
        else:
            if int(n_r_bins) != n_r_bins or n_r_bins <= 0:
                raise ValueError("n_r_bins must be a positive integer")
            if n_r_bins > nR_native:
                raise ValueError(
                    f"n_r_bins={n_r_bins} exceeds native radial bin count nR={nR_native}. "
                    "Refining radial bins is not supported."
                )
            if n_r_bins == nR_native:
                r_edges = r_edges_native
            else:
                idx = np.array([(i * nR_native) // n_r_bins for i in range(n_r_bins + 1)], dtype=int)
                r_edges = r_edges_native[idx]

        nR = len(r_edges) - 1
        if n_theta_bins is None:
            n_theta_bins = n_bins_native

        if callable(fthres):
            r_mid = 0.5 * (r_edges[:-1].to("au").magnitude + r_edges[1:].to("au").magnitude)
            fthres_eval = np.asarray(fthres(r_mid), dtype=float)
            if fthres_eval.ndim != 1:
                raise ValueError("fthres callable must return a 1D array of thresholds")
            if fthres_eval.size != nR:
                raise ValueError(
                    f"fthres callable returned {fthres_eval.size} values, expected nR={nR}"
                )
            fthres_use: Union[float, np.ndarray] = fthres_eval[:, None]
        elif np.isscalar(fthres):
            fthres_use = float(fthres)
        else:
            fthres_arr = np.asarray(fthres, dtype=float)
            if fthres_arr.ndim != 1:
                raise ValueError("fthres must be a scalar or a 1D array of length nR")
            if fthres_arr.size != nR:
                raise ValueError(f"fthres array length {fthres_arr.size} does not match nR={nR}")
            fthres_use = fthres_arr[:, None]

        if fthres_vr is None:
            fthres_vr_use: Union[float, np.ndarray] = fthres_use
        elif callable(fthres_vr):
            r_mid = 0.5 * (r_edges[:-1].to("au").magnitude + r_edges[1:].to("au").magnitude)
            fthres_vr_eval = np.asarray(fthres_vr(r_mid), dtype=float)
            if fthres_vr_eval.ndim != 1:
                raise ValueError("fthres_vr callable must return a 1D array of thresholds")
            if fthres_vr_eval.size != nR:
                raise ValueError(
                    f"fthres_vr callable returned {fthres_vr_eval.size} values, expected nR={nR}"
                )
            fthres_vr_use = fthres_vr_eval[:, None]
        elif np.isscalar(fthres_vr):
            fthres_vr_use = float(fthres_vr)
        else:
            fthres_vr_arr = np.asarray(fthres_vr, dtype=float)
            if fthres_vr_arr.ndim != 1:
                raise ValueError("fthres_vr must be a scalar or a 1D array of length nR")
            if fthres_vr_arr.size != nR:
                raise ValueError(
                    f"fthres_vr array length {fthres_vr_arr.size} does not match nR={nR}"
                )
            fthres_vr_use = fthres_vr_arr[:, None]

        theta_sel = np.ones_like(z_d, dtype=bool)
        if r_max is not None:
            theta_sel &= (r_grid.to_base_units().magnitude <= r_max.to_base_units().magnitude)

        r_d = r_grid.to_base_units().magnitude
        ok = r_d > 0.0
        theta_from_midplane = np.zeros_like(z_d, dtype=float)
        if np.any(ok):
            cos_theta = np.clip((z_d / r_d)[ok], -1.0, 1.0)
            theta_d = np.arccos(cos_theta)
            theta_from_midplane[ok] = theta_d - (0.5 * np.pi)

        theta_extent_sel = theta_sel & ok
        if np.any(theta_extent_sel):
            theta_max = float(np.max(np.abs(theta_from_midplane[theta_extent_sel])))
        else:
            theta_max = float(np.max(np.abs(theta_from_midplane[ok])))
        if theta_max == 0.0 or not np.isfinite(theta_max):
            raise ValueError("Invalid disk-frame theta extent; cannot build theta bins")

        theta_edges = np.linspace(-theta_max, theta_max, n_theta_bins + 1)

        r_bin = np.digitize(r_grid.to_base_units().magnitude, r_edges.magnitude) - 1
        t_bin = np.digitize(theta_from_midplane, theta_edges) - 1
        valid = (r_bin >= 0) & (r_bin < nR) & (t_bin >= 0) & (t_bin < n_theta_bins)

        ring_index = (r_bin * n_theta_bins + t_bin).astype(np.int64)
        ring_index_flat = ring_index[valid].ravel()
        dV_w = dV_mag[valid].ravel()
        if dV_w.size == 0:
            raise ValueError("No valid cells for ring binning (check mesh/edges)")

        nbins = nR * n_theta_bins
        sum_w = np.bincount(ring_index_flat, weights=dV_w, minlength=nbins)

        def ring_avg(q_mag_flat: np.ndarray) -> np.ndarray:
            num = np.bincount(ring_index_flat, weights=(q_mag_flat * dV_w), minlength=nbins)
            out = np.zeros(nbins, dtype=float)
            ok_w = sum_w > 0.0
            out[ok_w] = num[ok_w] / sum_w[ok_w]
            return out.reshape(nR, n_theta_bins)

        vphi_avg = ring_avg(np.abs(vphi_d.to_base_units().magnitude)[valid].ravel())
        vR_avg = ring_avg(np.abs(vR_d.to_base_units().magnitude)[valid].ravel())
        vz_avg = ring_avg(np.abs(vz_d.to_base_units().magnitude)[valid].ravel())
        rho_avg = ring_avg(rho.magnitude[valid].ravel())

        rho_thr = rho_disk_min.to(rho.units).magnitude

        rot = (0.5 * rho * (vphi_d.to_base_units() ** 2)).to_base_units()
        rot_avg = ring_avg(rot.magnitude[valid].ravel())
        P_avg = ring_avg(Pth.magnitude[valid].ravel())

        c1 = vphi_avg > (fthres_vr_use * vR_avg)
        c2 = vphi_avg > (fthres_use * vz_avg)
        c3 = rot_avg > (fthres_use * P_avg)
        c5 = rho_avg > rho_thr
        ring_pass = c1 & c2 & c3 & c5

        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        sum_w_2d = sum_w.reshape(nR, n_theta_bins)

        connected = np.zeros_like(ring_pass, dtype=bool)
        for i in range(nR):
            populated_idx = np.flatnonzero(sum_w_2d[i] > 0.0)
            if populated_idx.size == 0:
                continue

            seed_candidates = populated_idx[ring_pass[i, populated_idx]]
            if seed_candidates.size == 0:
                continue

            mid_t = int(seed_candidates[np.argmin(np.abs(theta_centers[seed_candidates]))])

            pop_order = populated_idx[np.argsort(theta_centers[populated_idx])]
            k0 = int(np.flatnonzero(pop_order == mid_t)[0])

            k = k0
            while k < pop_order.size and ring_pass[i, pop_order[k]]:
                connected[i, pop_order[k]] = True
                k += 1

            k = k0 - 1
            while k >= 0 and ring_pass[i, pop_order[k]]:
                connected[i, pop_order[k]] = True
                k -= 1

        connected_flat = connected.reshape(-1)
        mask = np.zeros_like(r_grid_mag, dtype=bool)
        mask_valid = connected_flat[ring_index_flat]
        mask[valid] = mask_valid

        if r_max is not None:
            mask &= (r_grid.to_base_units().magnitude <= r_max.to_base_units().magnitude)

        return self.set_mask_from_array(mask, is_a_disk=True)
    
    def set_mask_from_array(
        self,
        mask_array: np.ndarray,
        is_a_disk: bool = False,
    ) -> SubModel:
        """Define a masked region from a custom boolean array.
        
        Args:
            mask_array: Boolean array matching grid shape
            is_a_disk: Mark region as disk (enables settling mode)
            
        Returns:
            SubModel (or Disk) with mask applied
        """
        mesh = self.mesh
        if mesh is None:
            raise ValueError("Model has no mesh")
        
        if mesh.coord_system != 'spherical':
            raise ValueError(
                f"set_mask_from_array only supports spherical coordinates, "
                f"got {mesh.coord_system}"
            )
        
        # Determine target region
        if is_a_disk and self.disk is not None:
            target_region = self.disk
        else:
            target_region = SubModel(self)
        
        # Verify shape
        expected_shape = (
            len(mesh.axes['r'].centers),
            len(mesh.axes['phi'].centers),
            len(mesh.axes['theta'].centers)
        )
        
        if mask_array.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask_array.shape} doesn't match grid shape {expected_shape}"
            )
        
        axis_order = ('r', 'phi', 'theta')
        
        mask_quantity = Quantity(mask_array.astype(bool), 'dimensionless')
        mask_field = Field(
            data=mask_quantity,
            quantity='mask',
            axis_order=axis_order,
        )
        
        target_region.mask = mask_field
        
        # Mark as disk region if requested
        if is_a_disk:
            target_region.is_disk_region = True
        
        logger.info(
            f"{target_region.__class__.__name__} mask set: {np.sum(mask_array)} / {mask_array.size} cells "
            f"({100*np.sum(mask_array)/mask_array.size:.1f}%)"
        )
        
        return target_region

    def puff_up_model(
        self, n: int, 
        zmax_over_H: float = 5.0
    ) -> "Model":
        """Puff 2D polar model to 3D spherical coordinates."""
        self.disk.puff_up_disk(n=n, zmax_over_H=zmax_over_H)
        return self

    def extend_spherical_grid_inwards(
        self,
        r_min: Quantity,
        *,
        spacing: Optional[str] = None,
        density_match: Optional[str] = None,
    ) -> "Model":
        mesh = self.mesh
        if mesh is None:
            raise ValueError("Model has no mesh")
        if mesh.coord_system != "spherical":
            raise ValueError(
                "extend_spherical_grid_inwards only supports spherical coordinates, "
                f"got {mesh.coord_system}"
            )

        r_edges_old = mesh.edges("r")
        theta_edges = mesh.edges("theta")
        phi_edges = mesh.edges("phi")
        if r_edges_old is None or theta_edges is None or phi_edges is None:
            raise ValueError("Mesh is missing one or more spherical edge axes")
        if r_edges_old.size < 2:
            raise ValueError("Radial edges must have at least 2 entries")

        r_min_use = r_min.to(r_edges_old.units)
        r0_old = r_edges_old[0]
        if float(r_min_use.magnitude) >= float(r0_old.magnitude):
            raise ValueError(
                f"Requested r_min={r_min_use} is not smaller than existing inner edge {r0_old}"
            )

        spacing_mode = (spacing or "").strip().lower()

        r_edges_old_mag = np.asarray(r_edges_old.magnitude, dtype=float)
        if not np.all(np.isfinite(r_edges_old_mag)):
            raise ValueError("Radial edges contain non-finite values")

        dr0 = float(r_edges_old_mag[1] - r_edges_old_mag[0])
        ratio0 = float(r_edges_old_mag[1] / r_edges_old_mag[0])
        if not spacing_mode:
            diffs = np.diff(r_edges_old_mag[: min(6, r_edges_old_mag.size)])
            ratios = (
                r_edges_old_mag[1 : min(6, r_edges_old_mag.size)]
                / r_edges_old_mag[: min(5, r_edges_old_mag.size - 1)]
            )
            eps = np.finfo(float).eps
            dr_spread = float(np.max(np.abs(diffs - diffs[0])))
            ratio_spread = float(np.max(np.abs(ratios - ratios[0])))
            if dr_spread <= 32.0 * eps * max(1.0, abs(diffs[0])):
                spacing_mode = "lin"
            elif ratio_spread <= 32.0 * eps * max(1.0, abs(ratios[0])):
                spacing_mode = "log"
            else:
                raise ValueError(
                    "Could not infer radial spacing pattern from existing edges; provide spacing='log' or spacing='lin'"
                )

        if spacing_mode.startswith("log"):
            ratio = float(r_edges_old_mag[1] / r_edges_old_mag[0])
            if not np.isfinite(ratio) or ratio <= 1.0:
                raise ValueError(
                    "Cannot extend logarithmic grid inward: failed to infer ratio from first two radial edges"
                )
            inner_edges_desc: list[float] = []
            r_next = float(r_edges_old_mag[0])
            while True:
                if r_next <= float(r_min_use.magnitude):
                    break
                r_next = r_next / ratio
                if not np.isfinite(r_next) or r_next <= 0.0:
                    raise ValueError("Failed to extend inward: encountered non-positive radial edge")
                inner_edges_desc.append(r_next)
            if not inner_edges_desc:
                return self
            inner_edges_mag = np.array(list(reversed(inner_edges_desc)), dtype=float)
        elif spacing_mode.startswith("lin"):
            dr = float(r_edges_old_mag[1] - r_edges_old_mag[0])
            if not np.isfinite(dr) or dr <= 0.0:
                raise ValueError(
                    "Cannot extend linear grid inward: failed to infer positive dr from first two radial edges"
                )
            inner_edges_desc: list[float] = []
            r_next = float(r_edges_old_mag[0])
            while True:
                if r_next <= float(r_min_use.magnitude):
                    break
                r_next = r_next - dr
                if not np.isfinite(r_next) or r_next <= 0.0:
                    raise ValueError("Failed to extend inward: encountered non-positive radial edge")
                inner_edges_desc.append(r_next)
            if not inner_edges_desc:
                return self
            inner_edges_mag = np.array(list(reversed(inner_edges_desc)), dtype=float)
        else:
            raise ValueError(
                f"Unsupported spacing mode '{spacing_mode}'. Use 'log' or 'lin'."
            )

        r_edges_new = (
            np.concatenate([inner_edges_mag, r_edges_old_mag]) * r_edges_old.units
        )
        if r_edges_new.size <= r_edges_old.size:
            return self

        new_mesh = Mesh.spherical(
            r=Axis(edges=r_edges_new),
            theta=Axis(edges=theta_edges),
            phi=Axis(edges=phi_edges),
        )

        old_nr = int(r_edges_old.size - 1)
        new_nr = int(r_edges_new.size - 1)
        n_add = new_nr - old_nr

        r_centers_new = new_mesh.centers("r")
        theta_centers = new_mesh.centers("theta")
        phi_centers = new_mesh.centers("phi")
        if r_centers_new is None or theta_centers is None or phi_centers is None:
            raise ValueError("Failed to compute spherical centers for new mesh")

        if getattr(self.gas, "_lazy", None):
            for name in list(self.gas._lazy.keys()):
                _ = self.gas[name]

        density_units = None
        try:
            if "density" in self.gas:
                density_units = self.gas["density"].data.units
        except Exception:
            density_units = None

        def _compute_initial_density_for_inner_region() -> Optional[Quantity]:
            if self.disk is None:
                return None
            sigma0 = self.disk.parameters.get("sigma0")
            sigmaslope = self.disk.parameters.get("sigmaslope")
            h0 = self.disk.parameters.get("aspectratio")
            fl = self.disk.parameters.get("flaringindex")
            r0 = self.disk.parameters.get("r0")
            if sigma0 is None or sigmaslope is None or h0 is None or fl is None or r0 is None:
                return None

            h0_f = float(getattr(h0, "magnitude", h0))
            fl_f = float(getattr(fl, "magnitude", fl))
            sigmaslope_f = float(getattr(sigmaslope, "magnitude", sigmaslope))

            r_inner = r_centers_new[:n_add]
            r_mag = np.asarray(r_inner.magnitude, dtype=float)
            theta_mag = np.asarray(theta_centers.to("radian").magnitude, dtype=float)
            phi_mag = np.asarray(phi_centers.to("radian").magnitude, dtype=float)

            r_grid, phi_grid, theta_grid = np.meshgrid(r_mag, phi_mag, theta_mag, indexing="ij")
            r_grid = r_grid * r_inner.units

            sin_t = np.sin(theta_grid)
            cos_t = np.cos(theta_grid)
            R_cyl = r_grid * sin_t
            z = r_grid * cos_t

            H = (h0_f * (r_grid / r0) ** fl_f) * r_grid
            Sigma = sigma0 * (r_grid / r0) ** (-sigmaslope_f)
            rho_mid = Sigma / (np.sqrt(2.0 * np.pi) * H)

            expo = (-(z ** 2) / (2.0 * H ** 2)).to("dimensionless").magnitude
            rho = rho_mid * np.exp(expo)

            rhofloor = self.variables.get("RHOFLOORGAS", None)
            if rhofloor is not None:
                try:
                    rho_units = density_units if density_units is not None else rho.units
                    rho_floor_q = float(rhofloor) * rho_units
                    rho = Quantity(
                        np.maximum(
                            np.asarray(rho.to(rho_units).magnitude, dtype=float),
                            float(rho_floor_q.to(rho_units).magnitude),
                        ),
                        rho_units,
                    )
                except Exception:
                    pass

            return rho

        def _scale_density_to_match_snapshot(rho: Quantity, units_f) -> Quantity:
            mode = (density_match or "").strip().lower()
            if not mode or mode == "none":
                return rho
            if mode not in ("join_midplane",):
                raise ValueError(
                    "density_match must be one of: None, 'none', 'join_midplane'"
                )
            if self.disk is None:
                raise ValueError("density_match requested but model has no disk")

            sigma0 = self.disk.parameters.get("sigma0")
            sigmaslope = self.disk.parameters.get("sigmaslope")
            h0 = self.disk.parameters.get("aspectratio")
            fl = self.disk.parameters.get("flaringindex")
            r0 = self.disk.parameters.get("r0")
            if sigma0 is None or sigmaslope is None or h0 is None or fl is None or r0 is None:
                raise ValueError("density_match requested but disk parameters are incomplete")

            h0_f = float(getattr(h0, "magnitude", h0))
            fl_f = float(getattr(fl, "magnitude", fl))
            sigmaslope_f = float(getattr(sigmaslope, "magnitude", sigmaslope))

            if "density" not in self.gas:
                raise ValueError("density_match requested but gas field 'density' is missing")

            f_density = self.gas["density"]
            if not {"r", "phi", "theta"}.issubset(set(f_density.axis_order)):
                raise ValueError(
                    "density_match currently requires density axis_order to include ('r','phi','theta')"
                )

            r_axis = f_density.axis_order.index("r")
            phi_axis = f_density.axis_order.index("phi")
            theta_axis = f_density.axis_order.index("theta")

            old_mag = np.asarray(f_density.data.to(units_f).magnitude, dtype=float)
            old_mag_rpt = np.moveaxis(old_mag, [r_axis, phi_axis, theta_axis], [0, 1, 2])
            if old_mag_rpt.shape[0] <= 0:
                raise ValueError("density_match failed: density has no radial cells")

            theta_vals = np.asarray(theta_centers.to("radian").magnitude, dtype=float)
            theta_mid_idx = int(np.argmin(np.abs(theta_vals - 0.5 * np.pi)))
            snap_ref = float(np.mean(old_mag_rpt[0, :, theta_mid_idx]))
            if not np.isfinite(snap_ref) or snap_ref <= 0.0:
                raise ValueError("density_match failed: snapshot reference density is non-positive")

            r_ref = r_centers_new[n_add]
            theta_mid = float(theta_vals[theta_mid_idx])
            z_ref = r_ref * np.cos(theta_mid)
            H_ref = (h0_f * (r_ref / r0) ** fl_f) * r_ref
            Sigma_ref = sigma0 * (r_ref / r0) ** (-sigmaslope_f)
            rho_mid_ref = Sigma_ref / (np.sqrt(2.0 * np.pi) * H_ref)
            expo_ref = (-(z_ref**2) / (2.0 * H_ref**2)).to("dimensionless").magnitude
            rho_ref = (rho_mid_ref * np.exp(expo_ref)).to(units_f)
            analytic_ref = float(getattr(rho_ref, "magnitude", rho_ref))
            if not np.isfinite(analytic_ref) or analytic_ref <= 0.0:
                raise ValueError("density_match failed: analytic reference density is non-positive")

            scale = snap_ref / analytic_ref
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("density_match failed: computed scale is non-positive")
            rho_scaled = rho * scale

            rhofloor = self.variables.get("RHOFLOORGAS", None)
            if rhofloor is not None:
                try:
                    rho_floor_q = float(rhofloor) * units_f
                    rho_scaled = Quantity(
                        np.maximum(
                            np.asarray(rho_scaled.to(units_f).magnitude, dtype=float),
                            float(rho_floor_q.to(units_f).magnitude),
                        ),
                        units_f,
                    )
                except Exception:
                    pass

            return rho_scaled

        def _compute_initial_vphi_for_inner_region(v_unit) -> Quantity:
            import diskbridge

            mstar = getattr(diskbridge.params, "mstar", None)
            if mstar is None:
                raise ValueError("diskbridge.params.mstar is required to compute initial vphi")

            r_inner = r_centers_new[:n_add]
            r_mag = np.asarray(r_inner.to_base_units().magnitude, dtype=float)
            theta_mag = np.asarray(theta_centers.to("radian").magnitude, dtype=float)
            phi_mag = np.asarray(phi_centers.to("radian").magnitude, dtype=float)

            r_grid, phi_grid, theta_grid = np.meshgrid(r_mag, phi_mag, theta_mag, indexing="ij")
            r_grid = r_grid * r_inner.to_base_units().units

            sin_t = np.sin(theta_grid)
            R_cyl = r_grid * sin_t

            if np.any(np.asarray(R_cyl.magnitude) <= 0.0):
                raise ValueError("Cannot compute Keplerian vphi where cylindrical radius is non-positive")

            G = units("G")
            vphi = np.sqrt((G * mstar.to("g")) / R_cyl)
            return vphi.to(v_unit)

        rho_inner = _compute_initial_density_for_inner_region()
        if rho_inner is not None:
            try:
                units_density = density_units if density_units is not None else rho_inner.units
                rho_inner = _scale_density_to_match_snapshot(rho_inner, units_density)
            except Exception:
                raise

        new_fields: Dict[str, Field] = {}
        for name, f in list(self.gas.items()):
            if "r" not in f.axis_order:
                new_fields[name] = f
                continue

            r_axis = f.axis_order.index("r")
            old_mag = np.asarray(f.data.magnitude)
            units_f = f.data.units

            new_shape = list(old_mag.shape)
            new_shape[r_axis] = new_shape[r_axis] + n_add
            new_mag = np.empty(new_shape, dtype=old_mag.dtype)

            dst = [slice(None)] * old_mag.ndim
            dst[r_axis] = slice(n_add, None)
            new_mag[tuple(dst)] = old_mag

            src0 = np.take(old_mag, indices=0, axis=r_axis)
            pad = np.repeat(np.expand_dims(src0, axis=r_axis), repeats=n_add, axis=r_axis)

            if name == "density" and rho_inner is not None:
                pad = np.asarray(rho_inner.to(units_f).magnitude, dtype=new_mag.dtype)
            elif name == "vphi":
                try:
                    vphi_inner = _compute_initial_vphi_for_inner_region(units_f)
                    pad = np.asarray(vphi_inner.to(units_f).magnitude, dtype=new_mag.dtype)
                except Exception:
                    pass
            elif name in ("vr", "vtheta"):
                pad = np.zeros_like(pad)

            dst_inner = [slice(None)] * old_mag.ndim
            dst_inner[r_axis] = slice(0, n_add)
            new_mag[tuple(dst_inner)] = pad

            new_data = Quantity(new_mag, units_f)
            new_fields[name] = Field(
                data=new_data,
                quantity=f.quantity,
                axis_order=f.axis_order,
                attrs=f.attrs,
            )

        self.mesh = new_mesh
        self.coord_system = "spherical"
        self.gas.mesh = new_mesh
        self.gas.coord_system = "spherical"
        if self.disk is not None:
            self.disk.mesh = new_mesh
            self.disk.coord_system = "spherical"
        if getattr(self, "dust", None) is not None:
            self.dust.mesh = new_mesh
            self.dust.coord_system = "spherical"
            try:
                self.dust._dust_fields.clear()
            except Exception:
                pass

        self.gas.clear()
        for name, field in new_fields.items():
            self.gas_register(name, field)

        if self.disk is not None and self.disk.mask is not None:
            mask_field = self.disk.mask
            if "r" in mask_field.axis_order:
                r_axis = mask_field.axis_order.index("r")
                mask_mag = np.asarray(mask_field.data.magnitude, dtype=bool)
                new_shape = list(mask_mag.shape)
                new_shape[r_axis] = new_shape[r_axis] + n_add
                new_mask = np.zeros(new_shape, dtype=bool)

                dst = [slice(None)] * mask_mag.ndim
                dst[r_axis] = slice(n_add, None)
                new_mask[tuple(dst)] = mask_mag

                src0 = np.take(mask_mag, indices=0, axis=r_axis)
                pad = np.repeat(np.expand_dims(src0, axis=r_axis), repeats=n_add, axis=r_axis)

                dst_inner = [slice(None)] * mask_mag.ndim
                dst_inner[r_axis] = slice(0, n_add)
                new_mask[tuple(dst_inner)] = pad

                self.disk.mask = Field(
                    data=Quantity(new_mask, "dimensionless"),
                    quantity=mask_field.quantity,
                    axis_order=mask_field.axis_order,
                    attrs=mask_field.attrs,
                )

        return self

    def clip_mesh(
        self,
        *,
        r_min: Optional[Quantity] = None,
        r_max: Optional[Quantity] = None,
        theta_min: Optional[Quantity] = None,
        theta_max: Optional[Quantity] = None,
        phi_min: Optional[Quantity] = None,
        phi_max: Optional[Quantity] = None,
        x_min: Optional[Quantity] = None,
        x_max: Optional[Quantity] = None,
        y_min: Optional[Quantity] = None,
        y_max: Optional[Quantity] = None,
        z_min: Optional[Quantity] = None,
        z_max: Optional[Quantity] = None,
    ) -> "Model":
        mesh0 = self.mesh
        if mesh0 is None:
            raise ValueError("Model has no mesh")

        bounds: Dict[str, tuple[Optional[Quantity], Optional[Quantity]]] = {
            "r": (r_min, r_max),
            "theta": (theta_min, theta_max),
            "phi": (phi_min, phi_max),
            "x": (x_min, x_max),
            "y": (y_min, y_max),
            "z": (z_min, z_max),
        }
        bounds = {
            k: (vmin, vmax)
            for k, (vmin, vmax) in bounds.items()
            if vmin is not None or vmax is not None
        }

        for axis_name in bounds:
            if axis_name not in mesh0.axes:
                raise ValueError(
                    f"Cannot clip axis '{axis_name}' for coord_system '{mesh0.coord_system}'"
                )

        indexer, new_axes = compute_clip_indexer(mesh0, bounds)
        axis_slices = indexer.axis_slices

        mesh1 = Mesh(mesh0.coord_system, new_axes)

        def _slice_field(field0: Field) -> Field:
            slicer = []
            for ax in field0.axis_order:
                slicer.append(axis_slices.get(ax, slice(None)))
            mag0 = np.asarray(field0.data.magnitude)
            mag1 = mag0[tuple(slicer)]
            data1 = Quantity(mag1, field0.data.units)
            return Field(
                data=data1,
                quantity=field0.quantity,
                axis_order=field0.axis_order,
                attrs=field0.attrs,
            )

        new = Model()
        new.coord_system = mesh1.coord_system
        new.variables = dict(self.variables)
        new.compile_options = dict(self.compile_options)
        new.macros = dict(self.macros)
        new.mesh = mesh1
        new.file_units = self.file_units
        new.directory = self.directory
        new.n_file = self.n_file
        new.filename = self.filename

        for attr in ("length_scale", "mass_scale", "time_scale", "velocity_scale"):
            if hasattr(self, attr):
                setattr(new, attr, getattr(self, attr))

        new.gas = SubModel(new)
        if getattr(self.gas, "_lazy", None):
            for name in list(self.gas._lazy.keys()):
                _ = self.gas[name]
        for name, field0 in list(self.gas.items()):
            new.gas_register(name, _slice_field(field0))

        if getattr(self, "disk", None) is not None:
            new.disk = Disk(new, dict(self.disk.parameters))
            new.disk.is_disk_region = bool(getattr(self.disk, "is_disk_region", False))
            if self.disk.mask is not None:
                new.disk.mask = _slice_field(self.disk.mask)

        if getattr(self, "dust", None) is not None:
            from .dust import Dust, DustComponent, DustBin

            old_dust = self.dust
            new.dust = Dust(new)
            new_dust = new.dust

            new_dust.mask = _slice_field(old_dust.mask) if old_dust.mask is not None else None
            new_dust.is_disk_region = bool(getattr(old_dust, "is_disk_region", False))

            new_dust.distribution = old_dust.distribution
            new_dust.dust_to_gas_ratio = old_dust.dust_to_gas_ratio
            new_dust.mode = old_dust.mode
            new_dust.alpha = old_dust.alpha
            new_dust.delta = old_dust.delta
            new_dust.mean_molecular_weight = old_dust.mean_molecular_weight

            new_dust._has_region_components = bool(getattr(old_dust, "_has_region_components", False))
            new_dust._components = []
            for comp0 in getattr(old_dust, "_components", []):
                comp_mask = _slice_field(comp0.mask) if comp0.mask is not None else None
                new_dust._components.append(
                    DustComponent(
                        distribution=comp0.distribution,
                        dust_to_gas_ratio=comp0.dust_to_gas_ratio,
                        mode=comp0.mode,
                        mask=comp_mask,
                        alpha=comp0.alpha,
                        delta=comp0.delta,
                        mean_molecular_weight=comp0.mean_molecular_weight,
                        species_base=comp0.species_base,
                        component_index=comp0.component_index,
                    )
                )

            new_dust._global_bins = {}
            new_dust._bins = {}
            for comp_idx, comp in enumerate(new_dust._components):
                nbin = int(comp.distribution.nbin)
                for local_idx in range(nbin):
                    global_bin_idx = len(new_dust._global_bins)
                    bin_name = f"bin_{global_bin_idx}"
                    new_dust._global_bins[bin_name] = (comp_idx, local_idx)
                    new_dust._bins[bin_name] = DustBin(
                        parent_dust=new_dust,
                        bin_index=global_bin_idx,
                        size=comp.distribution.bin_centers[local_idx],
                        size_min=comp.distribution.bin_edges[local_idx],
                        size_max=comp.distribution.bin_edges[local_idx + 1],
                        mass_fraction=comp.distribution.mass_fractions[local_idx],
                        density_material=comp.distribution.grain_density,
                    )

            new_dust._dust_fields = {}
            for name, field0 in getattr(old_dust, "_dust_fields", {}).items():
                new_dust._dust_fields[name] = _slice_field(field0)

        return new
        

class SubModel:
    """Component that holds/modifies fields in a region."""

    def __init__(self, parent: Model):
        self.parent = parent
        self.mesh = parent.mesh
        self.coord_system = parent.coord_system
        self._fields: Dict[str, Field] = {}
        self._lazy: Dict[str, Callable[[], Field]] = {}

        # SubModel specific properties
        self.mask: Optional[Field] = None
        self.is_disk_region: bool = False  # Marks if this region allows settling

    # Registry API
    def register(self, name: str, field: Field) -> None:
        validate_field_against_mesh(field, self.mesh)  # type: ignore[arg-type]
        self._fields[name] = field

    def register_lazy(self, name: str, builder: Callable[[], Field]) -> None:
        self._lazy[name] = builder

    # Mapping-like API
    def __getitem__(self, key: str) -> Field:
        # Guard: request for volume density requires 3D mesh
        if key == "density":
            try:
                nd = getattr(self.mesh, "ndims", None)
            except Exception:
                nd = None
            if nd is not None and nd != 3:
                logger.error("'density' is unavailable: simulation is %sd (use 'surface_density')", nd)
                raise KeyError("density not available: simulation is 2D; use 'surface_density'")

        if key in self._fields:
            return self._fields[key]
        if key in self._lazy:
            field = self._lazy[key]()
            self._fields[key] = field
            return field
        raise KeyError(key)

    def __setitem__(self, key: str, value: Field) -> None:
        self._fields[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._fields or key in self._lazy

    def keys(self):
        return self._fields.keys()

    def items(self):
        return self._fields.items()

    def clear(self) -> None:
        self._fields.clear()
    
    @property
    def dust(self):
        """Access to dust with region-aware configuration."""
        if not hasattr(self.parent, 'dust') or self.parent.dust is None:
            raise AttributeError("Parent model has no dust submodel")
        return _RegionDustConfigurator(
            global_dust=self.parent.dust,
            mask=self.mask,
            is_disk_region=self.is_disk_region,
        )
    
    def anti_mask(self) -> "SubModel":
        """Create a new SubModel with the complement of this submodel's mask.
        
        Returns:
            SubModel with inverted mask
        """
        if self.mask is None:
            raise ValueError("This SubModel has no mask, cannot create an anti-mask")
        
        # Compute boolean complement
        mask_bool = self.mask.data.magnitude.astype(bool)
        other_bool = ~mask_bool
        
        # Use parent model's set_mask_from_array method
        other = self.parent.set_mask_from_array(other_bool, is_a_disk=False)
        
        logger.info(
            f"Created complement region: {np.sum(other_bool)} / {other_bool.size} cells "
            f"({100*np.sum(other_bool)/other_bool.size:.1f}%)"
        )
        
        return other


class _RegionDustConfigurator:
    """Internal helper that provides region-aware dust configuration."""
    
    def __init__(self, global_dust, mask: Optional[Field], is_disk_region: bool):
        self._dust = global_dust
        self._mask = mask
        self._is_disk_region = is_disk_region
    
    def set_distribution(self, mode: str = "proportional", **kwargs):
        """Configure dust distribution for this region.
        
        Args:
            mode: 'proportional' or 'settling'
            **kwargs: Additional parameters (amin, amax, nbin, etc.)
        """
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
    """Disk component with disk-specific operations."""

    _required_parameters = [
        "aspectratio",
        "flaringindex",
        "r0"
    ]

    def __init__(self, parent: Model, parameters: Dict[str, Any]):
        super().__init__(parent)
        for key in self._required_parameters:
            if key not in parameters:
                raise ValueError(f"Missing required parameter in Disk class: {key}")
        self.parameters: Dict[str, Any] = parameters
    
    @property
    def gas(self) -> SubModel:
        """Access to gas fields (convenience accessor to parent.gas)."""
        return self.parent.gas
    
    @property
    def dust(self):
        """Access to dust with region-aware configuration."""
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
        """Puff 2D surface density directly to 3D spherical coordinates.
        
        Computes the Gaussian vertical profile directly in spherical coordinates
        without intermediate cylindrical conversion, ensuring mass conservation.
        """

        r = self.parent.mesh.centers("r")
        h0 = self.parameters["aspectratio"]
        fl = self.parameters["flaringindex"]
        r0 = self.parameters["r0"]
        
        # Get 2D surface density
        if "surface_density" not in self.parent.gas:
            raise KeyError("Missing gas field 'surface_density'")
        Sigma = self.parent.gas["surface_density"].data  # (nrad, nsec)
        
        # Build spherical mesh
        sph_mesh = self.parent.mesh.to_spherical_by_scale_height(
            ncol=n, aspect_ratio=h0, zmax_over_H=zmax_over_H
        )
        
        r_sph = sph_mesh.centers("r")
        theta_sph = sph_mesh.centers("theta")
        phi_sph = sph_mesh.centers("phi")
        
        # Create 3D grids
        r_mag = r_sph.magnitude
        theta_mag = theta_sph.magnitude
        phi_mag = phi_sph.magnitude
        r_units = r_sph.units
        
        # Shape: (n_r, n_phi, n_theta) matching typical axis order
        r_grid, phi_grid, theta_grid = np.meshgrid(r_mag, phi_mag, theta_mag, indexing='ij')
        r_grid = r_grid * r_units
        
        # Compute z = r * cos(theta) for each spherical cell
        z_cyl = r_grid * np.cos(theta_grid)
        
        # Scale height H(r) = h(r) * r where h(r) = h0 * (r/r0)^fl
        h_r = h0 * (r_sph / r0) ** fl  # dimensionless
        H = h_r * r_sph  # scale height with units
        H_3d = H[:, None, None] * np.ones_like(theta_grid)  # broadcast to 3D
        
        # Expand Sigma to 3D: (nrad, nsec) -> (nrad, nsec, ntheta)
        # Note: Sigma is (r, phi), we need (r, phi, theta)
        Sigma_3d = Sigma[:, :, None] * np.ones(len(theta_sph))
        
        # Compute Gaussian density: rho = Sigma / (sqrt(2*pi) * H) * exp(-z^2 / (2*H^2))
        rho_3d = Sigma_3d / (np.sqrt(2 * np.pi) * H_3d) * np.exp(-z_cyl**2 / (2 * H_3d**2))
        
        # Handle velocities - expand 2D to 3D (constant in theta)
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
        
        # vtheta = 0
        vunit = None
        if "vr" in self.parent.gas:
            vunit = getattr(self.parent.gas["vr"].data, "units", None)
        elif "vphi" in self.parent.gas:
            vunit = getattr(self.parent.gas["vphi"].data, "units", None)
        vtheta = np.zeros((nr, nphi, ntheta))
        if vunit is not None:
            vtheta = vtheta * vunit
        
        # Commit spherical mesh + fields
        self.parent.mesh = sph_mesh
        self.mesh = sph_mesh
        if hasattr(self.parent, "gas") and self.parent.gas is not None:
            self.parent.gas.mesh = sph_mesh
        self.coord_system = "spherical"
        
        self.parent.gas.clear()
        self.parent.gas_register("density", Field(quantity="density", data=rho_3d, axis_order=("r", "phi", "theta")))
        if vr_sph is not None:
            self.parent.gas_register("vr", Field(quantity="vr", data=vr_sph, axis_order=("r", "phi", "theta")))
        if vphi_sph is not None:
            self.parent.gas_register("vphi", Field(quantity="vphi", data=vphi_sph, axis_order=("r", "phi", "theta")))
        self.parent.gas_register("vtheta", Field(quantity="vtheta", data=vtheta, axis_order=("r", "phi", "theta")))
        return self


def puff_up_model(
    model: "Model",
    n: int,
    zmax_over_H: float = 5.0,
) -> "Model":
    """Puff a 2D polar model to 3D spherical coordinates."""
    new = Model()
    new.coord_system = model.mesh.coord_system
    new.variables = dict(model.variables)
    new.compile_options = dict(model.compile_options)
    new.macros = dict(model.macros)

    cs = model.mesh.coord_system

    if cs == "polar":
        new.mesh = Mesh.polar(
            r=Axis(edges=model.mesh.edges("r"), centers=model.mesh.centers("r")),
            phi=Axis(edges=model.mesh.edges("phi"), centers=model.mesh.centers("phi")),
        )
    elif cs == "spherical":
        new.mesh = Mesh.spherical(
            r=Axis(edges=model.mesh.edges("r"), centers=model.mesh.centers("r")),
            theta=(
                Axis(edges=model.mesh.edges("theta"), centers=model.mesh.centers("theta"))
                if model.mesh.ncell("theta") else None
            ),
            phi=Axis(edges=model.mesh.edges("phi"), centers=model.mesh.centers("phi")),
        )
    else:
        raise ValueError(f"Unsupported coordinate system for puffing: {cs}")

    new.file_units = model.file_units
    new.directory = model.directory
    new.n_file = model.n_file
    new.filename = model.filename

    new.gas = SubModel(new)
    if getattr(model, "disk", None) is not None:
        new.disk = Disk(new, model.disk.parameters)

    for name, f in model.gas.items():
        new.gas_register(name, Field(data=f.data, quantity=f.quantity, axis_order=f.axis_order))

    new.disk.puff_up_disk(n, zmax_over_H=zmax_over_H)
    return new


def extend_disk_inwards(
    model: "Model",
    *,
    r_min: Optional[Quantity] = None,
    r_min_factor: Optional[float] = None,
    spacing: Optional[str] = None,
    density_match: Optional[str] = None,
) -> int:
    if (r_min is None) == (r_min_factor is None):
        raise ValueError("Provide exactly one of r_min or r_min_factor")

    mesh0 = model.mesh
    if mesh0 is None:
        raise ValueError("Model mesh is not set")
    r_edges0 = mesh0.edges("r")
    if r_edges0 is None:
        raise ValueError("Model radial edges are not set")

    if r_min is None:
        r_min = float(r_min_factor) * r_edges0[0]

    model.extend_spherical_grid_inwards(r_min, spacing=spacing, density_match=density_match)

    mesh1 = model.mesh
    if mesh1 is None:
        raise ValueError("Model mesh is not set after extension")
    r_edges1 = mesh1.edges("r")
    if r_edges1 is None:
        raise ValueError("Model radial edges are not set after extension")

    return int(r_edges1.size - r_edges0.size)
