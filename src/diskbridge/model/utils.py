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


def _interp_cyl_to_sph(cyl: Optional[np.ndarray], r_cyl: np.ndarray, zmed: np.ndarray, 
                       r_sph: np.ndarray, tmed_sph: np.ndarray) -> Optional[np.ndarray]:
    """
    Interpolate from cylindrical to spherical coordinates using bilinear interpolation.
    
    Parameters
    ----------
    cyl : array (nz, nrad, nsec) or None
        Data in cylindrical coordinates
    r_cyl : array (nrad,)
        Cylindrical radial grid
    zmed : array (nz,)
        Cylindrical vertical grid
    r_sph : array (nrad,)
        Spherical radial grid
    tmed_sph : array (ntheta,)
        Spherical colatitude grid
        
    Returns
    -------
    sph : array (ntheta, nrad, nsec) or None
        Data in spherical coordinates
    """
    if cyl is None:
        return None
    nz, nrad, nsec = cyl.shape
    nt = len(tmed_sph)
    out = []
    for j in range(nt):
        theta = tmed_sph[j]
        R = r_sph * np.sin(theta)
        Z = r_sph * np.cos(theta)
        slice_j = np.zeros((nrad, nsec))
        for i in range(nrad):
            Ri = R[i]
            Zi = Z[i]
            ir = int(np.clip(np.searchsorted(r_cyl, Ri) - 1, 0, nrad - 2))
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
