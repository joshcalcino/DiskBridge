"""RADMC-3D image handler for DiskBridge.

This module provides the RadImage class for reading RADMC-3D image output
and writing FITS files with proper WCS headers.

This is adapted from radmc3dPy.image but simplified to focus on FITS output.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np

try:
    from astropy.io import fits
except ImportError:
    fits = None
    print("Warning: astropy.io.fits not available. FITS writing disabled.")

from diskbridge._logging import logger

# Physical constants (CGS)
C_LIGHT = 2.99792458e10  # cm/s
PC = 3.08567758e18  # cm
AU = 1.49597871e13  # cm


class RadImage:
    """RADMC-3D image class for reading and writing FITS files.
    
    This class handles reading RADMC-3D image.out files and writing
    FITS files with proper WCS headers for use with astronomical tools.
    
    Attributes
    ----------
    image : ndarray
        Image data in erg/s/cm^2/Hz/ster
    imageJyppix : ndarray
        Image data in Jy/pixel
    x : ndarray
        X coordinate grid in cm
    y : ndarray
        Y coordinate grid in cm
    nx : int
        Number of pixels in x direction
    ny : int
        Number of pixels in y direction
    sizepix_x : float
        Pixel size in x direction (cm)
    sizepix_y : float
        Pixel size in y direction (cm)
    nfreq : int
        Number of frequencies
    freq : ndarray
        Frequency grid (Hz)
    nwav : int
        Number of wavelengths (same as nfreq)
    wav : ndarray
        Wavelength grid (cm)
    stokes : bool
        Whether image contains Stokes parameters
    filename : str
        Name of the image file
        
    Examples
    --------
    >>> from diskbridge.radmc3d import RadImage
    >>> 
    >>> # Read an image
    >>> img = RadImage()
    >>> img.readImage('image.out')
    >>> 
    >>> # Write to FITS
    >>> img.writeFits('output.fits', dpc=140.0, coord='16h32m22s -24d28m30s')
    """
    
    def __init__(self):
        """Initialize empty image."""
        self.image: Optional[np.ndarray] = None
        self.imageJyppix: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.nx: int = 0
        self.ny: int = 0
        self.sizepix_x: float = 0.0
        self.sizepix_y: float = 0.0
        self.nfreq: int = 0
        self.freq: Optional[np.ndarray] = None
        self.nwav: int = 0
        self.wav: Optional[np.ndarray] = None
        self.stokes: bool = False
        self.filename: str = ''
        
    def readImage(self, fname: str | Path = 'image.out', binary: bool = False) -> None:
        """Read a RADMC-3D image file.
        
        Parameters
        ----------
        fname : str or Path, optional
            Path to the RADMC-3D image file (default: 'image.out')
        binary : bool, optional
            Whether to read binary format (default: False, reads ASCII)
            
        Notes
        -----
        This method reads RADMC-3D image output files in either ASCII or binary
        format. The image data is stored in erg/s/cm^2/Hz/ster units and also
        converted to Jy/pixel units.
        """
        fname = Path(fname)
        self.filename = str(fname)
        
        logger.info(f"Reading RADMC-3D image: {fname}")
        
        if binary:
            self._readImageBinary(fname)
        else:
            self._readImageASCII(fname)
            
        # Convert to Jy/pixel (at 1 pc distance for unit conversion)
        conv = self.sizepix_x * self.sizepix_y / PC**2 * 1e23
        self.imageJyppix = self.image * conv
        
        # Create coordinate grids
        self.x = ((np.arange(self.nx, dtype=np.float64) + 0.5) - self.nx / 2) * self.sizepix_x
        self.y = ((np.arange(self.ny, dtype=np.float64) + 0.5) - self.ny / 2) * self.sizepix_y
        
        logger.info(f"Image read: shape=({self.nx}, {self.ny}, {self.nfreq}), "
                   f"pixel_size=({self.sizepix_x/AU:.3f}, {self.sizepix_y/AU:.3f}) AU")
        
    def _readImageASCII(self, fname: Path) -> None:
        """Read ASCII format image.out file."""
        with open(fname, 'r') as f:
            # Format number
            iformat = int(f.readline())
            
            # Number of pixels
            line = f.readline().split()
            self.nx = int(line[0])
            self.ny = int(line[1])
            
            # Number of frequencies
            self.nfreq = int(f.readline())
            self.nwav = self.nfreq
            
            # Pixel sizes
            line = f.readline().split()
            self.sizepix_x = float(line[0])
            self.sizepix_y = float(line[1])
            
            # Wavelengths
            self.wav = np.zeros(self.nwav, dtype=np.float64)
            for iwav in range(self.nwav):
                self.wav[iwav] = float(f.readline())
            self.freq = C_LIGHT / self.wav * 1e4  # Convert cm to Hz
            
            # Read image data
            if iformat == 1:
                # Normal intensity image
                self.stokes = False
                self.image = np.zeros([self.nx, self.ny, self.nwav], dtype=np.float64)
                
                for iwav in range(self.nwav):
                    # Blank line
                    f.readline()
                    for iy in range(self.ny):
                        for ix in range(self.nx):
                            self.image[ix, iy, iwav] = float(f.readline())
                            
            elif iformat == 3:
                # Full Stokes image
                self.stokes = True
                self.image = np.zeros([self.nx, self.ny, 4, self.nwav], dtype=np.float64)
                
                for iwav in range(self.nwav):
                    # Blank line
                    f.readline()
                    for iy in range(self.ny):
                        for ix in range(self.nx):
                            line = f.readline().split()
                            self.image[ix, iy, 0, iwav] = float(line[0])  # I
                            self.image[ix, iy, 1, iwav] = float(line[1])  # Q
                            self.image[ix, iy, 2, iwav] = float(line[2])  # U
                            self.image[ix, iy, 3, iwav] = float(line[3])  # V
            else:
                raise ValueError(f"Unknown image format: {iformat}")
                
    def _readImageBinary(self, fname: Path) -> None:
        """Read binary format image.bout file."""
        with open(fname, 'rb') as f:
            # Read header
            header = np.fromfile(f, dtype=np.int32, count=4)
            iformat = header[0]
            self.nx = header[1]
            self.ny = header[2]
            self.nfreq = header[3]
            self.nwav = self.nfreq
            
            # Read rest of data
            data = np.fromfile(f, dtype=np.float64)
            
            self.sizepix_x = data[0]
            self.sizepix_y = data[1]
            self.wav = data[2:2 + self.nfreq]
            self.freq = C_LIGHT / self.wav * 1e4
            
            # Reverse frequency array so it's in increasing order (for positive CDELT)
            if self.nfreq > 1 and self.freq[0] > self.freq[1]:
                self.freq = self.freq[::-1]
                self.wav = self.wav[::-1]
            
            # Read image data
            if iformat == 1:
                self.stokes = False
                self.image = np.reshape(
                    data[2 + self.nfreq:],
                    [self.nfreq, self.ny, self.nx]
                )
                # Swap axes to match ASCII format
                self.image = np.swapaxes(self.image, 0, 2)
            elif iformat == 3:
                self.stokes = True
                self.image = np.reshape(
                    data[2 + self.nfreq:],
                    [self.nfreq, 4, self.ny, self.nx]
                )
                # Swap axes to match ASCII format
                self.image = np.swapaxes(self.image, 0, 3)
                self.image = np.swapaxes(self.image, 1, 2)
            else:
                raise ValueError(f"Unknown image format: {iformat}")
                
    def writeFits(
        self,
        fname: str | Path = 'image.fits',
        dpc: float = 1.0,
        coord: str = '0h0m0s 0d0m0s',
        inc: float = 0.0,
        pa: float = 0.0,
        stokes: str = 'I',
        bandwidthmhz: float = 2000.0,
        nu0: float = 0.0,
        object_name: str = '',
        casa: bool = False,
        overwrite: bool = True,
    ) -> None:
        """Write image to FITS file.
        
        Parameters
        ----------
        fname : str or Path, optional
            Output FITS filename (default: 'image.fits')
        dpc : float, optional
            Distance to source in parsecs (default: 1.0)
        coord : str, optional
            Image center coordinates as 'RAh RAm RAs DECd DECm DECs'
            (default: '0h0m0s 0d0m0s')
        inc : float, optional
            Inclination in degrees (for header only)
        pa : float, optional
            Position angle in degrees (for header only)
        stokes : str, optional
            Stokes parameter to write: 'I', 'Q', 'U', 'V' (default: 'I')
        bandwidthmhz : float, optional
            Bandwidth in MHz for single-frequency images (default: 2000.0)
        nu0 : float, optional
            Rest frequency in Hz (default: 0.0, auto-detect)
        object_name : str, optional
            Object name for FITS header
        casa : bool, optional
            Write CASA-compatible 4D cube (default: False)
        overwrite : bool, optional
            Overwrite existing file (default: True)
            
        Notes
        -----
        The output FITS file will have proper WCS headers for use with
        astronomical visualization and analysis tools.
        
        Examples
        --------
        >>> img = RadImage()
        >>> img.readImage('image.out')
        >>> img.writeFits('co_line.fits', dpc=140.0, 
        ...               coord='16h32m22s -24d28m30s',
        ...               object_name='HD_163296')
        """
        if fits is None:
            raise ImportError("astropy.io.fits is required for FITS writing")
            
        fname = Path(fname)
        logger.info(f"Writing FITS file: {fname}")
        
        # Parse coordinates
        target_ra, target_dec = self._parse_coords(coord)
        
        # Select Stokes parameter
        istokes = 0
        if self.stokes:
            stokes_map = {'I': 0, 'Q': 1, 'U': 2, 'V': 3}
            istokes = stokes_map.get(stokes.upper(), 0)
            
        # Convert to Jy/pixel at specified distance
        conv = self.sizepix_x * self.sizepix_y / (dpc * PC)**2 * 1e23
        
        # Prepare data array
        if casa:
            # CASA format: [Stokes, Freq, Dec, RA]
            data = np.zeros([1, self.nfreq, self.ny, self.nx], dtype=float)
            if self.nfreq == 1:
                if self.stokes:
                    data[0, 0, :, :] = self.image[:, :, istokes, 0] * conv
                else:
                    data[0, 0, :, :] = self.image[:, :, 0] * conv
            else:
                for inu in range(self.nfreq):
                    if self.stokes:
                        data[0, inu, :, :] = self.image[:, :, istokes, inu] * conv
                    else:
                        data[0, inu, :, :] = self.image[:, :, inu] * conv
        else:
            # Standard format: [Freq, RA, Dec]
            data = np.zeros([self.nfreq, self.nx, self.ny], dtype=float)
            if self.nfreq == 1:
                if self.stokes:
                    data[0, :, :] = self.image[:, :, istokes, 0] * conv
                else:
                    data[0, :, :] = self.image[:, :, 0] * conv
            else:
                for inu in range(self.nfreq):
                    if self.stokes:
                        data[inu, :, :] = self.image[:, :, istokes, inu] * conv
                    else:
                        data[inu, :, :] = self.image[:, :, inu] * conv
                        
        # Create FITS HDU
        naxis = len(data.shape)
        hdu = fits.PrimaryHDU(data.swapaxes(naxis - 1, naxis - 2))
        
        # Set WCS keywords
        # Axis 1: RA
        hdu.header['CRPIX1'] = (self.nx + 1.) / 2.
        hdu.header['CDELT1'] = -self.sizepix_x / AU / dpc / 3600.
        hdu.header['CRVAL1'] = target_ra
        hdu.header['CUNIT1'] = 'deg'
        hdu.header['CTYPE1'] = 'RA---SIN'
        
        # Axis 2: Dec
        hdu.header['CRPIX2'] = (self.ny + 1.) / 2.
        hdu.header['CDELT2'] = self.sizepix_y / AU / dpc / 3600.
        hdu.header['CRVAL2'] = target_dec
        hdu.header['CUNIT2'] = 'deg'
        hdu.header['CTYPE2'] = 'DEC--SIN'
        
        # Axis 3: Frequency
        if casa:
            # Axis 4 for CASA (Stokes)
            hdu.header['CRPIX4'] = 1.
            hdu.header['CDELT4'] = 1.
            hdu.header['CRVAL4'] = 1.
            hdu.header['CUNIT4'] = ''
            hdu.header['CTYPE4'] = 'STOKES'
            
            # Axis 3 for CASA (Frequency)
            hdu.header['CRPIX3'] = 1.0
            if self.nfreq == 1:
                hdu.header['CDELT3'] = bandwidthmhz * 1e6
                hdu.header['CRVAL3'] = self.freq[0]
            else:
                # Use absolute value for positive CDELT (freq increases with channel)
                # This matches fargo2radmc3d convention and standard FITS practice
                # Since RADMC-3D gives freq in decreasing order, use minimum freq as CRVAL
                hdu.header['CDELT3'] = abs(self.freq[1] - self.freq[0])
                hdu.header['CRVAL3'] = self.freq[-1]  # Lowest frequency (last element)
            hdu.header['CUNIT3'] = 'Hz'
            hdu.header['CTYPE3'] = 'FREQ'
        else:
            hdu.header['CRPIX3'] = 1.0
            if self.nfreq == 1:
                hdu.header['CDELT3'] = bandwidthmhz * 1e6
                hdu.header['CRVAL3'] = self.freq[0]
            else:
                # Use absolute value for positive CDELT (freq increases with channel)
                # This matches fargo2radmc3d convention and standard FITS practice
                # Since RADMC-3D gives freq in decreasing order, use minimum freq as CRVAL
                hdu.header['CDELT3'] = abs(self.freq[1] - self.freq[0])
                hdu.header['CRVAL3'] = self.freq[-1]  # Lowest frequency (last element)
            hdu.header['CUNIT3'] = 'Hz'
            hdu.header['CTYPE3'] = 'FREQ'
            
        # Rest frequency
        if nu0 > 0:
            hdu.header['RESTFRQ'] = nu0
        elif self.nfreq == 1:
            hdu.header['RESTFRQ'] = self.freq[0]
        else:
            hdu.header['RESTFRQ'] = self.freq[self.nfreq // 2]
            
        # Brightness units
        hdu.header['BUNIT'] = 'Jy/pixel'
        hdu.header['BTYPE'] = 'Intensity'
        
        # Additional metadata
        hdu.header['EPOCH'] = 2000.0
        hdu.header['EQUINOX'] = 2000.0
        if object_name:
            hdu.header['OBJECT'] = object_name
        hdu.header['DISTANCE'] = (dpc, 'Distance to source (pc)')
        hdu.header['INCL'] = (inc, 'Inclination (deg)')
        hdu.header['PA'] = (pa, 'Position angle (deg)')
        
        # Write file
        hdu.writeto(fname, overwrite=overwrite)
        logger.info(f"FITS file written: {fname}")
        
    def _parse_coords(self, coord: str) -> tuple[float, float]:
        """Parse coordinate string to RA and Dec in degrees.
        
        Parameters
        ----------
        coord : str
            Coordinate string in format 'RAh RAm RAs DECd DECm DECs'
            Example: '16h32m22s -24d28m30s'
            
        Returns
        -------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        """
        # Parse RA
        dum = coord
        ra = []
        delim = ['h', 'm', 's']
        for i in delim:
            ind = dum.find(i)
            if ind <= 0:
                raise ValueError(
                    f"Invalid coordinate format: {coord}. "
                    "Expected format: '0h10m05s -10d05m30s'"
                )
            ra.append(float(dum[:ind]))
            dum = dum[ind + 1:]
            
        # Parse Dec
        dec = []
        delim = ['d', 'm', 's']
        for i in delim:
            ind = dum.find(i)
            if ind <= 0:
                raise ValueError(
                    f"Invalid coordinate format: {coord}. "
                    "Expected format: '0h10m05s -10d05m30s'"
                )
            dec.append(float(dum[:ind]))
            dum = dum[ind + 1:]
            
        # Convert to degrees
        target_ra = (ra[0] + ra[1] / 60. + ra[2] / 3600.) * 15.
        if dec[0] >= 0:
            target_dec = dec[0] + dec[1] / 60. + dec[2] / 3600.
        else:
            target_dec = dec[0] - dec[1] / 60. - dec[2] / 3600.
            
        return target_ra, target_dec
