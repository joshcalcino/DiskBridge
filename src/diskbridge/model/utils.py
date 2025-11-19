import numpy as np
from typing import Optional
from .mesh import Mesh
from .field import Field


# Validators for mesh and field structures 
def validate_field_against_mesh(field: Field, mesh: Mesh) -> None:
    names = set(mesh.axis_names())
    missing = [a for a in field.axis_order if a not in names]
    if missing:
        raise ValueError(f"{field.quantity}: axis {missing} not in mesh axes {mesh.axis_names()}")
    expected = tuple(mesh.ncell(a) for a in field.axis_order)
    actual   = tuple(np.asarray(field.data.magnitude).shape)
    if actual != expected:
        raise ValueError(
            f"{field.quantity}: data shape {actual} != expected {expected} for {field.axis_order}"
        )


def _interp_sph_to_cyl(sph: Optional[np.ndarray], r_sph: np.ndarray, tmed_sph: np.ndarray,
                       r_cyl: np.ndarray, zmed: np.ndarray) -> Optional[np.ndarray]:
    """
    Interpolate from spherical to cylindrical coordinates using bilinear interpolation.
    
    Parameters
    ----------
    sph : array (ntheta, nrad, nsec) or None
        Data in spherical coordinates
    r_sph : array (nrad,)
        Spherical radial grid
    tmed_sph : array (ntheta,)
        Spherical colatitude grid
    r_cyl : array (nrad,)
        Cylindrical radial grid
    zmed : array (nz,)
        Cylindrical vertical grid
        
    Returns
    -------
    cyl : array (nz, nrad, nsec) or None
        Data in cylindrical coordinates
    """
    if sph is None:
        return None
    nt, nrad_sph, nsec = sph.shape
    nz = len(zmed)
    nrad_cyl = len(r_cyl)
    out = []
    for k in range(nz):
        z = zmed[k]
        slice_k = np.zeros((nrad_cyl, nsec))
        for i in range(nrad_cyl):
            r_c = r_cyl[i]
            # Convert cylindrical (r_c, z) to spherical (r, theta)
            r_s = np.sqrt(r_c**2 + z**2)
            theta = np.arctan2(r_c, z)
            
            # Find indices in spherical grid
            ir = int(np.clip(np.searchsorted(r_sph, r_s) - 1, 0, nrad_sph - 2))
            it = int(np.clip(np.searchsorted(tmed_sph, theta) - 1, 0, nt - 2))
            
            r0 = r_sph[ir]
            r1 = r_sph[ir + 1]
            t0 = tmed_sph[it]
            t1 = tmed_sph[it + 1]
            
            dr = (r_s - r0) / (r1 - r0) if r1 != r0 else 0.0
            dt = (theta - t0) / (t1 - t0) if t1 != t0 else 0.0
            
            # Bilinear interpolation
            c00 = sph[it, ir, :]
            c01 = sph[it, ir + 1, :]
            c10 = sph[it + 1, ir, :]
            c11 = sph[it + 1, ir + 1, :]
            c0 = c00 * (1 - dr) + c01 * dr
            c1 = c10 * (1 - dr) + c11 * dr
            slice_k[i, :] = c0 * (1 - dt) + c1 * dt
        out.append(slice_k)
    return np.stack(out, axis=0)


def _interp_cyl_to_sph(cyl: Optional[np.ndarray], r_cyl: np.ndarray, zmed: np.ndarray, 
                       r_sph: np.ndarray, tmed_sph: np.ndarray) -> Optional[np.ndarray]:
    """
    Interpolate from cylindrical to spherical coordinates using bilinear interpolation.
    
    Parameters
    ----------
    cyl : array (nz, nrad_cyl, nsec) or None
        Data in cylindrical coordinates
    r_cyl : array (nrad_cyl,)
        Cylindrical radial grid
    zmed : array (nz,)
        Cylindrical vertical grid
    r_sph : array (nrad_sph,)
        Spherical radial grid
    tmed_sph : array (ntheta,)
        Spherical colatitude grid
        
    Returns
    -------
    sph : array (ntheta, nrad_sph, nsec) or None
        Data in spherical coordinates
    """
    if cyl is None:
        return None
    nz, nrad_cyl, nsec = cyl.shape
    nrad_sph = len(r_sph)
    nt = len(tmed_sph)
    out = []
    for j in range(nt):
        theta = tmed_sph[j]
        R = r_sph * np.sin(theta)
        Z = r_sph * np.cos(theta)
        slice_j = np.zeros((nrad_sph, nsec))
        for i in range(nrad_sph):
            Ri = R[i]
            Zi = Z[i]
            ir = int(np.clip(np.searchsorted(r_cyl, Ri) - 1, 0, nrad_cyl - 2))
            iz = int(np.clip(np.searchsorted(zmed, Zi) - 1, 0, nz - 2))
            r0 = r_cyl[ir]
            r1 = r_cyl[ir + 1]
            z0 = zmed[iz]
            z1 = zmed[iz + 1]
            dr = (Ri - r0) / (r1 - r0) if r1 != r0 else 0.0
            dz = (Zi - z0) / (z1 - z0) if z1 != z0 else 0.0
            c00 = cyl[iz, ir, :]
            c01 = cyl[iz, ir + 1, :]
            c10 = cyl[iz + 1, ir, :]
            c11 = cyl[iz + 1, ir + 1, :]
            c0 = c00 * (1 - dr) + c01 * dr
            c1 = c10 * (1 - dr) + c11 * dr
            slice_j[i, :] = c0 * (1 - dz) + c1 * dz
        out.append(slice_j)
    return np.stack(out, axis=0)
