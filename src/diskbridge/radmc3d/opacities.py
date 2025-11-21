"""Dust opacity calculations using Mie scattering theory.

This module provides dust opacity calculations using the Bohren & Huffman
Mie scattering code, adapted from the fargo2radmc3d package.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, Dict
from pathlib import Path
import numpy as np
import math
from scipy.interpolate import interp1d

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define a no-op decorator if numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

if TYPE_CHECKING:
    from diskbridge.model.dust import DustBin

from diskbridge._units import units
from diskbridge._logging import logger


@njit(cache=True)
def bhmie_numba(x: float, refrel: complex, theta: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Numba-optimized Bohren and Huffman Mie scattering calculation.
    
    This is a JIT-compiled version of the bhmie function for improved performance.
    
    Args:
        x: Size parameter (2*pi*radius_grain/lambda)
        refrel: Complex index of refraction (e.g., 1.5 + 0.01j)
        theta: Array of scattering angles between 0 and 180 degrees
        
    Returns:
        Tuple of (S1, S2, Qext, Qabs, Qsca, Qback, gsca)
    """
    # Check theta array orientation
    nang = len(theta)
    if theta[0] == 0.0:
        iang0 = 0
        iang180 = nang - 1
    else:
        iang0 = nang - 1
        iang180 = 0
    
    # Allocate complex phase functions
    S1 = np.zeros(nang, dtype=np.complex128)
    S2 = np.zeros(nang, dtype=np.complex128)
    
    # Initialize arrays for series expansion
    pi = np.zeros(nang, dtype=np.float64)
    pi0 = np.zeros(nang, dtype=np.float64)
    pi1 = np.zeros(nang, dtype=np.float64)
    pi1[:] = 1.0
    tau = np.zeros(nang, dtype=np.float64)
    
    # Compute alternative to x
    y = x * refrel
    
    # Determine termination point for series expansion
    xstop = x + 4 * x**0.3333 + 2.0
    nstop = int(math.floor(xstop))
    
    # Start of logarithmic derivatives iteration
    nmx = int(math.floor(max(xstop, abs(y))) + 15)
    
    # Compute mu = cos(theta)
    mu = np.cos(theta * math.pi / 180.)
    
    # Calculate logarithmic derivative by downward recurrence
    dlog = np.zeros(nmx, dtype=np.complex128)
    for n in range(nmx - 1):
        en = float(nmx - n)
        dlog[nmx - n - 2] = en / y - 1.0 / (dlog[nmx - n - 1] + en / y)
    
    # Prepare for series expansion
    psi0 = math.cos(x)
    psi1 = math.sin(x)
    chi0 = -math.sin(x)
    chi1 = math.cos(x)
    xi1 = psi1 - chi1 * 1j
    p = -1.0
    Qsca = 0.0
    gsca = 0.0
    an = 0j
    bn = 0j
    
    # Riccati-Bessel functions - series expansion
    for n in range(nstop):
        en = float(n + 1)
        fn = (2 * en + 1.0) / (en * (en + 1.0))
        psi = (2 * en - 1.0) * psi1 / x - psi0
        chi = (2 * en - 1.0) * chi1 / x - chi0
        xi = psi - chi * 1j
        an1 = an
        bn1 = bn
        dum = dlog[n] / refrel + en / x
        an = (dum * psi - psi1) / (dum * xi - xi1)
        dum = dlog[n] * refrel + en / x
        bn = (dum * psi - psi1) / (dum * xi - xi1)
        
        # Add contributions to Qsca and gsca
        Qsca += (2 * en + 1.0) * (abs(an)**2 + abs(bn)**2)
        dum = (2 * en + 1.0) / (en * (en + 1.0))
        gsca += dum * (an.real * bn.real + an.imag * bn.imag)
        dum = (en - 1.0) * (en + 1.0) / en
        gsca += dum * (an1.real * an.real + an1.imag * an.imag +
                       bn1.real * bn.real + bn1.imag * bn.imag)
        
        # Contribute to scattering intensity pattern
        pi[:] = pi1[:]
        tau[:] = en * np.abs(mu[:]) * pi[:] - (en + 1.0) * pi0[:]
        
        # For mu >= 0
        for idx in range(nang):
            if mu[idx] >= 0:
                S1[idx] += fn * (an * pi[idx] + bn * tau[idx])
                S2[idx] += fn * (an * tau[idx] + bn * pi[idx])
        
        # For mu < 0
        p = -p
        for idx in range(nang):
            if mu[idx] < 0:
                S1[idx] += fn * p * (an * pi[idx] - bn * tau[idx])
                S2[idx] += fn * p * (bn * pi[idx] - an * tau[idx])
        
        # Prepare for next iteration
        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = psi1 - chi1 * 1j
        pi1[:] = ((2 * en + 1.0) * np.abs(mu[:]) * pi[:] - (en + 1.0) * pi0[:]) / en
        pi0[:] = pi[:]
    
    # Final calculations
    gsca = 2 * gsca / Qsca
    Qsca = (2.0 / (x * x)) * Qsca
    Qext = (4.0 / (x * x)) * S1[iang0].real
    Qback = (abs(S1[iang180]) / x)**2 / math.pi
    Qabs = Qext - Qsca
    
    return S1, S2, Qext, Qabs, Qsca, Qback, gsca


def bhmie(x: float, refrel: complex, theta: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Bohren and Huffman Mie scattering calculation.
    
    This is a direct port from the fargo2radmc3d/dust/bhmie.py module,
    originally from Bruce Draine's f77 code.
    
    Args:
        x: Size parameter (2*pi*radius_grain/lambda)
        refrel: Complex index of refraction (e.g., 1.5 + 0.01j)
        theta: Array of scattering angles between 0 and 180 degrees
        
    Returns:
        Tuple of (S1, S2, Qext, Qabs, Qsca, Qback, gsca):
            S1: Complex phase function (E perp to scattering plane)
            S2: Complex phase function (E para to scattering plane)
            Qext: Efficiency factor for extinction
            Qabs: Efficiency factor for absorption
            Qsca: Efficiency factor for scattering
            Qback: Backscattering efficiency
            gsca: <cos(theta)> for scattering
    """
    # Check theta array orientation
    nang = len(theta)
    if theta[0] == 0.0:
        assert theta[nang-1] == 180, "Angle grid must extend from 0 to 180 degrees."
        iang0 = 0
        iang180 = nang - 1
    else:
        assert theta[0] == 180, "Angle grid must extend from 0 to 180 degrees."
        assert theta[nang-1] == 0, "Angle grid must extend from 0 to 180 degrees."
        iang0 = nang - 1
        iang180 = 0
    
    # Allocate complex phase functions
    S1 = np.zeros(nang, dtype=np.complex128)
    S2 = np.zeros(nang, dtype=np.complex128)
    
    # Initialize arrays for series expansion
    pi = np.zeros(nang, dtype=np.float64)
    pi0 = np.zeros(nang, dtype=np.float64)
    pi1 = np.zeros(nang, dtype=np.float64) + 1.0
    tau = np.zeros(nang, dtype=np.float64)
    
    # Compute alternative to x
    y = x * refrel
    
    # Determine termination point for series expansion
    xstop = x + 4 * x**0.3333 + 2.0
    nstop = int(math.floor(xstop))
    
    # Start of logarithmic derivatives iteration
    nmx = int(math.floor(np.max([xstop, abs(y)])) + 15)
    
    # Compute mu = cos(theta)
    mu = np.cos(theta * math.pi / 180.)
    
    # Calculate logarithmic derivative by downward recurrence
    dlog = np.zeros(nmx, dtype=np.complex128)
    for n in range(nmx - 1):
        en = float(nmx - n)
        dlog[nmx - n - 2] = en / y - 1.0 / (dlog[nmx - n - 1] + en / y)
    
    # Prepare for series expansion
    psi0 = math.cos(x)
    psi1 = math.sin(x)
    chi0 = -math.sin(x)
    chi1 = math.cos(x)
    xi1 = psi1 - chi1 * 1j
    p = -1.0
    Qsca = 0.0
    gsca = 0.0
    an = 0j
    bn = 0j
    
    # Riccati-Bessel functions - series expansion
    for n in range(nstop):
        en = float(n + 1)
        fn = (2 * en + 1.0) / (en * (en + 1.0))
        psi = (2 * en - 1.0) * psi1 / x - psi0
        chi = (2 * en - 1.0) * chi1 / x - chi0
        xi = psi - chi * 1j
        an1 = an
        bn1 = bn
        dum = dlog[n] / refrel + en / x
        an = (dum * psi - psi1) / (dum * xi - xi1)
        dum = dlog[n] * refrel + en / x
        bn = (dum * psi - psi1) / (dum * xi - xi1)
        
        # Add contributions to Qsca and gsca
        Qsca += (2 * en + 1.0) * (abs(an)**2 + abs(bn)**2)
        dum = (2 * en + 1.0) / (en * (en + 1.0))
        gsca += dum * (an.real * bn.real + an.imag * bn.imag)
        dum = (en - 1.0) * (en + 1.0) / en
        gsca += dum * (an1.real * an.real + an1.imag * an.imag +
                       bn1.real * bn.real + bn1.imag * bn.imag)
        
        # Contribute to scattering intensity pattern
        pi[:] = pi1[:]
        tau[:] = en * np.abs(mu[:]) * pi[:] - (en + 1.0) * pi0[:]
        
        # For mu >= 0
        idx = mu >= 0
        S1[idx] += fn * (an * pi[idx] + bn * tau[idx])
        S2[idx] += fn * (an * tau[idx] + bn * pi[idx])
        
        # For mu < 0
        p = -p
        idx = mu < 0
        S1[idx] += fn * p * (an * pi[idx] - bn * tau[idx])
        S2[idx] += fn * p * (bn * pi[idx] - an * tau[idx])
        
        # Prepare for next iteration
        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = psi1 - chi1 * 1j
        pi1[:] = ((2 * en + 1.0) * np.abs(mu[:]) * pi[:] - (en + 1.0) * pi0[:]) / en
        pi0[:] = pi[:]
    
    # Final calculations
    gsca = 2 * gsca / Qsca
    Qsca = (2.0 / (x * x)) * Qsca
    Qext = (4.0 / (x * x)) * S1[iang0].real
    Qback = (abs(S1[iang180]) / x)**2 / math.pi
    Qabs = Qext - Qsca
    
    return S1, S2, Qext, Qabs, Qsca, Qback, gsca


class DustOpacityCalculator:
    """Calculate dust opacities using Mie scattering theory.
    
    This class computes dust absorption and scattering opacities from
    optical constants files using Mie theory.
    
    Example:
        calc = DustOpacityCalculator()
        opac = calc.compute_opacity(
            optconst_file='silicate.lnk',
            grain_density=2.7,  # g/cm^3
            grain_size=1e-5,    # cm
            wavelengths=lamcm,  # wavelengths in cm
        )
    """
    
    def __init__(self, use_numba: bool = True):
        """Initialize the opacity calculator.
        
        Args:
            use_numba: If True and numba is available, use JIT-compiled functions
        """
        self.verbose = False
        self.use_numba = use_numba and NUMBA_AVAILABLE
        if use_numba and not NUMBA_AVAILABLE:
            logger.warning("Numba not available, falling back to pure Python implementation")
        
    def compute_opacity(
        self,
        optconst_file: str | Path,
        grain_density: float,
        grain_size: float,
        wavelengths: np.ndarray,
        theta: Optional[np.ndarray] = None,
        logawidth: Optional[float] = None,
        wfact: float = 3.0,
        na: int = 20,
        chopforward: float = 0.0,
        errtol: float = 0.01,
        extrapolate: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute dust opacity using Mie theory.
        
        Args:
            optconst_file: Path to optical constants file (wavelength[μm], n, k)
            grain_density: Material density in g/cm^3
            grain_size: Grain radius in cm
            wavelengths: Wavelength grid in cm
            theta: Optional angular grid (0-180 degrees) for scattering matrix
            logawidth: Optional width for size distribution smoothing
            wfact: Grid width in units of logawidth
            na: Number of size sampling points if logawidth is set
            chopforward: Angle (degrees) within which to remove forward scattering
            errtol: Tolerance for kscat vs integral check
            extrapolate: Whether to extrapolate optical constants
            
        Returns:
            Dictionary containing:
                'kabs': Absorption opacity [cm^2/g]
                'kscat': Scattering opacity [cm^2/g]
                'gscat': Asymmetry parameter <cos(theta)>
                'wav': Wavelength grid [cm]
                Optional (if theta provided):
                    'zscat': Scattering matrix elements
                    'theta': Scattering angles
        """
        # Load optical constants
        data = np.loadtxt(optconst_file)
        wavmic, ncoef, kcoef = data.T
        
        # Extrapolate if needed
        if extrapolate:
            wmin = np.min(wavelengths) * 1e4 * 0.999
            wmax = np.max(wavelengths) * 1e4 * 1.001
            
            if wmin < np.min(wavmic):
                if wavmic[0] < wavmic[1]:
                    ncoef = np.append([ncoef[0]], ncoef)
                    kcoef = np.append([kcoef[0]], kcoef)
                    wavmic = np.append([wmin], wavmic)
                else:
                    ncoef = np.append(ncoef, [ncoef[-1]])
                    kcoef = np.append(kcoef, [kcoef[-1]])
                    wavmic = np.append(wavmic, [wmin])
                    
            if wmax > np.max(wavmic):
                if wavmic[0] < wavmic[1]:
                    ncoef = np.append(ncoef, [ncoef[-1] * math.exp(
                        (math.log(wmax) - math.log(wavmic[-1])) *
                        (math.log(ncoef[-1]) - math.log(ncoef[-2])) /
                        (math.log(wavmic[-1]) - math.log(wavmic[-2])))])
                    kcoef = np.append(kcoef, [kcoef[-1] * math.exp(
                        (math.log(wmax) - math.log(wavmic[-1])) *
                        (math.log(kcoef[-1]) - math.log(kcoef[-2])) /
                        (math.log(wavmic[-1]) - math.log(wavmic[-2])))])
                    wavmic = np.append(wavmic, [wmax])
                else:
                    ncoef = np.append([ncoef[0] * math.exp(
                        (math.log(wmax) - math.log(wavmic[0])) *
                        (math.log(ncoef[0]) - math.log(ncoef[1])) /
                        (math.log(wavmic[0]) - math.log(wavmic[1])))], ncoef)
                    kcoef = np.append([kcoef[0] * math.exp(
                        (math.log(wmax) - math.log(wavmic[0])) *
                        (math.log(kcoef[0]) - math.log(kcoef[1])) /
                        (math.log(wavmic[0]) - math.log(wavmic[1])))], kcoef)
                    wavmic = np.append([wmax], wavmic)
        
        # Interpolate to wavelength grid
        f = interp1d(np.log(wavmic * 1e-4), np.log(ncoef))
        ncoefi = np.exp(f(np.log(wavelengths)))
        f = interp1d(np.log(wavmic * 1e-4), np.log(kcoef))
        kcoefi = np.exp(f(np.log(wavelengths)))
        
        # Complex index of refraction
        refidx = ncoefi + kcoefi * 1j
        
        # Angular grid
        if theta is None:
            angles = np.array([0., 90., 180.])
        else:
            angles = theta
        nang = angles.size
        
        # Size distribution
        if logawidth is None:
            agr = np.array([grain_size])
            wgt = np.array([1.0])
        else:
            agr = np.exp(np.linspace(
                math.log(grain_size) - wfact * logawidth,
                math.log(grain_size) + wfact * logawidth,
                na
            ))
            wgt = np.exp(-0.5 * ((np.log(agr / grain_size)) / logawidth)**2)
            wgt = wgt / wgt.sum()
        
        nagr = agr.size
        siggeom = math.pi * agr * agr
        mgrain = (4 * math.pi / 3.0) * grain_density * agr * agr * agr
        
        # Prepare output arrays
        nlam = wavelengths.size
        kabs = np.zeros(nlam)
        kscat = np.zeros(nlam)
        gscat = np.zeros(nlam)
        
        if theta is not None:
            zscat = np.zeros((nlam, nang, 6))
            S11 = np.zeros(nang)
            S12 = np.zeros(nang)
            S33 = np.zeros(nang)
            S34 = np.zeros(nang)
        
        # Choose implementation: always prefer the numba-optimized version when available. It should reproduce the reference bhmie results
        bhmie_func = bhmie_numba if self.use_numba else bhmie
        
        # Loop over wavelengths
        for i in range(nlam):
            if self.verbose:
                logger.info(f"Computing opacity at wavelength {wavelengths[i]:.6e} cm")
            
            # Loop over grain sizes
            for l in range(nagr):
                x = 2 * math.pi * agr[l] / wavelengths[i]
                S1, S2, Qext, Qabs, Qsca, Qback, gsca = bhmie_func(x, refidx[i], angles)
                
                # Average over size distribution
                kabs[i] += wgt[l] * Qabs * siggeom[l] / mgrain[l]
                kscat[i] += wgt[l] * Qsca * siggeom[l] / mgrain[l]
                gscat[i] += wgt[l] * gsca
                
                # Compute scattering matrix if theta provided
                if theta is not None:
                    factor = (wavelengths[i] / (2 * math.pi))**2 / mgrain[l]
                    S11[:] = 0.5 * (np.abs(S2[:])**2 + np.abs(S1[:])**2)
                    S12[:] = 0.5 * (np.abs(S2[:])**2 - np.abs(S1[:])**2)
                    S33[:] = (S2[:] * np.conj(S1[:])).real
                    S34[:] = (S2[:] * np.conj(S1[:])).imag
                    
                    zscat[i, :, 0] += wgt[l] * S11[:] * factor
                    zscat[i, :, 1] += wgt[l] * S12[:] * factor
                    zscat[i, :, 2] += wgt[l] * S11[:] * factor  # Z22 = Z11 for spheres
                    zscat[i, :, 3] += wgt[l] * S33[:] * factor
                    zscat[i, :, 4] += wgt[l] * S34[:] * factor
                    zscat[i, :, 5] += wgt[l] * S33[:] * factor  # Z44 = Z33 for spheres

        # If we have an angular grid, enforce consistency between kscat
        # and the angular integral of Z11, and optionally apply a
        # forward-scattering chop (chopforward) following
        # fargo2radmc3d's compute_opac_mie.
        if theta is not None:
            mu = np.cos(angles * math.pi / 180.0)
            dmu = np.abs(mu[1:] - mu[:-1])
            kscat_from_z11 = np.zeros(nlam)
            error = False
            errmax = 0.0

            # Consistency check between kscat and integral over Z11
            for i in range(nlam):
                zav = 0.5 * (zscat[i, 1:, 0] + zscat[i, :-1, 0])
                dum = 0.5 * zav * dmu
                integral = dum.sum() * 4.0 * math.pi
                kscat_from_z11[i] = integral
                if kscat[i] > 0.0:
                    rel_err = abs(integral / kscat[i] - 1.0)
                    if rel_err > errtol:
                        error = True
                        errmax = max(errmax, rel_err)

            # Apply forward-scattering chop if requested
            if chopforward > 0.0:
                for i in range(nlam):
                    iang = np.where(angles < chopforward)[0]
                    if iang.size == 0:
                        continue
                    if angles[0] == 0.0:
                        iiang = int(np.max(iang) + 1)
                        if iiang >= nang:
                            iiang = nang - 1
                    else:
                        iiang = int(np.min(iang) - 1)
                        if iiang < 0:
                            iiang = 0
                    # Replace Z elements for chopped angles
                    for k in range(6):
                        zscat[i, iang, k] = zscat[i, iiang, k]
                    # Recompute kscat from chopped Z11
                    zav = 0.5 * (zscat[i, 1:, 0] + zscat[i, :-1, 0])
                    dum = 0.5 * zav * dmu
                    kscat[i] = dum.sum() * 4.0 * math.pi

            if error and self.verbose:
                logger.warning(
                    "Angular integral of Z11 is not equal to kscat at all wavelengths. "
                    f"Maximum relative error = {errmax:.6e}"
                )

        # Build output dictionary
        result = {
            'kabs': kabs,
            'kscat': kscat,
            'gscat': gscat,
            'wav': wavelengths,
        }
        
        if theta is not None:
            result['zscat'] = zscat
            result['theta'] = angles
        
        return result
    
    def write_radmc3d_opacity_file(
        self,
        opacity_data: Dict[str, np.ndarray],
        output_path: str | Path,
        scattering_matrix: bool = False,
    ) -> None:
        """Write opacity data to RADMC-3D format files.
        
        Args:
            opacity_data: Dictionary from compute_opacity()
            output_path: Base path for output file (without extension)
            scattering_matrix: If True, write full scattering matrix
        """
        output_path = Path(output_path)
        
        if scattering_matrix and 'zscat' in opacity_data:
            # Write dustkapscatmat file
            filename = f"dustkapscatmat_{output_path.name}.inp"
            self._write_scatmat_file(opacity_data, output_path.parent / filename)
        else:
            # Write dustkappa file
            filename = f"dustkappa_{output_path.name}.inp"
            self._write_kappa_file(opacity_data, output_path.parent / filename)
    
    def _write_kappa_file(self, opac: Dict, filename: Path) -> None:
        """Write dustkappa_*.inp file."""
        with open(filename, 'w') as f:
            # Header
            f.write("# Opacity file for RADMC-3D\n")
            f.write("# Columns: lambda[micron]  kabs[cm^2/g]  kscat[cm^2/g]  g\n")
            f.write("1\n")  # Format number
            f.write(f"{len(opac['wav'])}\n")  # Number of wavelengths
            
            # Data
            for i, lam in enumerate(opac['wav']):
                lam_micron = lam * 1e4  # cm to micron
                f.write(f"{lam_micron:13.6e}  {opac['kabs'][i]:13.6e}  "
                       f"{opac['kscat'][i]:13.6e}  {opac['gscat'][i]:13.6e}\n")
        
        logger.debug(f"Wrote opacity file: {filename}")
    
    def _write_scatmat_file(self, opac: Dict, filename: Path) -> None:
        """Write dustkapscatmat_*.inp file."""
        nlam = len(opac['wav'])
        nang = len(opac['theta'])
        
        with open(filename, 'w') as f:
            # Header
            f.write("# Opacity and scattering matrix file for RADMC-3D\n")
            f.write("1\n")  # Format number
            f.write(f"{nlam}\n")  # Number of wavelengths
            f.write(f"{nang}\n")  # Number of angles
            
            # Wavelengths
            for lam in opac['wav']:
                f.write(f"{lam * 1e4:13.6e}\n")  # cm to micron
            
            # Angles
            for ang in opac['theta']:
                f.write(f"{ang:13.6e}\n")
            
            # Opacity and scattering matrix data
            for i in range(nlam):
                f.write(f"{opac['kabs'][i]:13.6e}  {opac['kscat'][i]:13.6e}\n")
                
                # Write scattering matrix elements
                for j in range(nang):
                    z = opac['zscat'][i, j, :]
                    f.write(f"{z[0]:13.6e}  {z[1]:13.6e}  {z[2]:13.6e}  "
                           f"{z[3]:13.6e}  {z[4]:13.6e}  {z[5]:13.6e}\n")
        
        logger.info(f"Wrote scattering matrix file: {filename}")
