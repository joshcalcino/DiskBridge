"""RADMC-3D image handler for DiskBridge.

This module provides the RadImage class for reading RADMC-3D image output
and writing FITS files with proper WCS headers.

This is adapted from radmc3dPy.image but simplified to focus on FITS output.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import shutil
import datetime

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

from astropy.io import fits
from diskbridge._logging import logger
from .utils import _extract_radmc_errors, create_radmc3d_symlinks, cleanup_symlink_paths, run_radmc3d_and_log
import diskbridge
from .molecule import RadMolecule

# Physical constants (CGS)
C_LIGHT = diskbridge.units('c').to('cm/s').magnitude
PC = diskbridge.units('pc').to('cm').magnitude
AU = diskbridge.units('au').to('cm').magnitude


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
    
    def __init__(self, model_dir: str | Path = '.', model=None):
        """Initialize RadImage.
        
        Parameters
        ----------
        model_dir : str or Path, optional
            Directory containing RADMC-3D files (default: '.')
        model : Model, optional
            DiskBridge model (needed for auto-generating gas velocity)
        """
        self.model_dir = Path(model_dir)
        self.model = model
        
        # Get params from diskbridge global
        self.params = diskbridge.params
        
        # Subdirectory paths for organized file storage (matching RadModel/RadWriter)
        self.inputs_dir = self.model_dir / 'radmc3d_inputs'
        self.outputs_dir = self.model_dir / 'radmc3d_outputs'
        
        # Track active symlinks for cleanup
        self._active_symlinks: list[Path] = []
        
        # Image data
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
        
        # Multi-angle storage
        self.multi_angle_images: list[np.ndarray] = []
        self.viewing_angles: list[tuple[float, float, float]] = []  # (incl, PA, phi)
    
    def create_symlinks(self) -> None:
        """Create symlinks to organized input files in the model directory.
        
        RADMC-3D expects input files in the directory where it runs. This method
        creates symlinks from the model directory to the organized subdirectories.
        """
        # All input files from radmc3d_inputs
        input_files = [
            'amr_grid.inp', 'wavelength_micron.inp',
            'stars.inp', 'dustopac.inp',
            'dust_density.binp', 'dust_density.inp',
            'gas_velocity.binp', 'gas_velocity.inp',
            'numberdens_*.binp', 'numberdens_*.inp',
            'radmc3d.inp', 'lines.inp', 'molecule_*.inp',
            'external_source.inp'
        ]
        
        # All output files from radmc3d_outputs
        output_files = ['dust_temperature.*', 'mean_intensity.out']

        create_radmc3d_symlinks(
            model_dir=self.model_dir,
            inputs_dir=self.inputs_dir,
            input_files=input_files,
            active_symlinks=self._active_symlinks,
            outputs_dir=self.outputs_dir,
            output_files=output_files,
        )
    
    def cleanup_symlinks(self) -> None:
        """Remove all symlinks created by create_symlinks().
        
        This ensures the model directory stays clean after RADMC-3D runs.
        Only removes symlinks that were tracked by this instance.
        """
        cleanup_symlink_paths(self._active_symlinks)
        
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
    
    def make_line_image(
        self,
        molecule: str,
        transition: int,
        inclinations: Optional[list[float] | float] = None,
        posangs: Optional[list[float] | float] = None,
        phis: Optional[list[float] | float] = None,
        npix: Optional[int] = None,
        sizeau: Optional[float] = None,
        widthkms: Optional[float] = None,
        linenlam: Optional[int] = None,
        output_dir: Optional[str] = None,
        distance: Optional[float] = None,
        coord: Optional[str] = None,
        object_name: str = '',
        **kwargs
    ) -> None:
        """Make molecular line image(s) at specified viewing angles.
        
        Parameters
        ----------
        molecule : str
            Molecule name (e.g., 'co', '13co')
        transition : int
            Transition number (e.g., 3 for J=3-2)
        inclinations : float or list of float, optional
            Inclination(s) in degrees. If None, uses params
        posangs : float or list of float, optional
            Position angle(s) in degrees. If None, uses params
        phis : float or list of float, optional
            Azimuthal angle(s) in degrees. If None, uses params
        npix : int, optional
            Number of pixels. If None, uses params
        sizeau : float, optional
            Image size in AU. If None, uses params
        widthkms : float, optional
            Velocity width in km/s. If None, uses params
        linenlam : int, optional
            Number of velocity channels. If None, uses params
        output_dir : str, optional
            Output directory name (default: 'image_{molecule}_J{transition}')
        distance : float, optional
            Distance in pc. If None, uses params
        coord : str, optional
            Source coordinates. If None, uses params or default
        object_name : str, optional
            Object name for FITS header
        **kwargs : additional arguments passed to _run_radmc3d_image
        """
        # Get defaults from params
        if inclinations is None:
            inclinations = self.params.inclination
            if not isinstance(inclinations, list):
                inclinations = [inclinations]
        if posangs is None:
            # Read PA from params in standard convention (0=N, 90=E)
            # Convert to RADMC-3D convention by adding 90 degrees
            pa_standard = self.params.posangle
            if isinstance(pa_standard, list):
                posangs = [(pa + 90.0) % 360.0 for pa in pa_standard]
            else:
                posangs = [(pa_standard + 90.0) % 360.0]
        if phis is None:
            phis = [0.0]  # Not in params typically
        if npix is None:
            npix = self.params.nbpixels
        if sizeau is None:
            sizeau = self.params.mapsize.to('au').magnitude
        if widthkms is None:
            widthkms = self.params.width.to('km/s').magnitude
        if linenlam is None:
            linenlam = self.params.nline
        if distance is None:
            distance = self.params.distance.to('pc').magnitude
        if coord is None:
            coord = '0h0m0s 0d0m0s'  # Use hardcoded default
        
        logger.debug(f"Parameters after defaults: npix={npix}, sizeau={sizeau}, widthkms={widthkms}, linenlam={linenlam}, phis={phis}")
        if output_dir is None:
            base_name = f'image_{molecule}_J{transition}'
            output_dir = self._apply_name_templates(base_name)
        
        # Convert single values to lists
        if not isinstance(inclinations, (list, tuple, np.ndarray)):
            inclinations = [inclinations]
        if not isinstance(posangs, (list, tuple, np.ndarray)):
            posangs = [posangs]
        if not isinstance(phis, (list, tuple, np.ndarray)):
            phis = [phis]
        
        # Ensure all required files exist
        self._ensure_molecule_file(molecule)
        self._ensure_gas_velocity()
        self._ensure_radmc3d_inp_configured()
        self._ensure_lines_inp(molecule)
        
        # Determine iline from molecule data
        iline = self._get_iline(molecule, transition)
        
        # Generate viewing angle combinations
        viewing_angles = [(i, p, phi) for i in inclinations for p in posangs for phi in phis]
        
        logger.info(f"Making {molecule} J={transition}-{transition-1} line images at {len(viewing_angles)} viewing angles")
        
        # Store parameters for FITS header
        self.widthkms = widthkms
        self.linenlam = linenlam
        
        # Run RADMC-3D for each viewing angle
        self.multi_angle_images = []
        self.viewing_angles = []
        
        # Store pixel sizes from first image
        saved_sizepix_x = None
        saved_sizepix_y = None
        
        for incl, pa, phi in viewing_angles:
            phi_str = f"{phi:.1f} deg" if phi is not None else "None"
            logger.info(f"  incl={incl:.1f} deg, PA={pa:.1f} deg, phi={phi_str}")
            
            # Run radmc3d image
            self._run_radmc3d_image(
                npix=npix,
                incl=incl,
                posang=pa,
                phi=phi,
                sizeau=sizeau,
                widthkms=widthkms,
                linenlam=linenlam,
                iline=iline,
                **kwargs
            )
            
            # Read image
            self.readImage('image.out')
            
            image_file = self.model_dir / 'image.out'
            if image_file.exists() or image_file.is_symlink():
                image_file.unlink()
            
            logger.debug(f"After readImage: sizepix_x={self.sizepix_x}, sizepix_y={self.sizepix_y}")
            
            # Save pixel sizes from first image
            if saved_sizepix_x is None:
                saved_sizepix_x = self.sizepix_x
                saved_sizepix_y = self.sizepix_y
                logger.debug(f"Saved pixel sizes: x={saved_sizepix_x}, y={saved_sizepix_y}")
            
            # Store for multi-angle FITS
            self.multi_angle_images.append(self.image.copy())
            self.viewing_angles.append((incl, pa, phi))
        
        logger.debug(f"Loop finished. Restoring: saved_x={saved_sizepix_x}, saved_y={saved_sizepix_y}")
        
        # Restore pixel sizes for FITS writing
        self.sizepix_x = saved_sizepix_x
        self.sizepix_y = saved_sizepix_y
        
        logger.debug(f"After restore: sizepix_x={self.sizepix_x}, sizepix_y={self.sizepix_y}")
        
        if self.sizepix_x is None or self.sizepix_y is None:
            raise RuntimeError(f"Pixel sizes not set for line image! sizepix_x={self.sizepix_x}, sizepix_y={self.sizepix_y}. "
                             f"This usually means readImage() failed or no images were processed.")
        
        logger.debug(f"Using pixel sizes: sizepix_x={self.sizepix_x/AU:.3f} AU, sizepix_y={self.sizepix_y/AU:.3f} AU")
        
        # Write multi-angle FITS
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fits_path = output_path / 'image.fits'
        self._write_multiangle_fits(
            fits_path,
            distance=distance,
            coord=coord,
            object_name=object_name,
            nu0=self._get_rest_frequency(molecule, transition),
        )
        
        # Copy params and add metadata
        self._organize_image_output(output_path, f"radmc3d image iline {iline}")
        
        logger.info(f"Line images written to {output_path}/image.fits")
    
    def make_image(
        self,
        wavelength: float,
        inclinations: Optional[list[float] | float] = None,
        posangs: Optional[list[float] | float] = None,
        phis: Optional[list[float] | float] = None,
        npix: Optional[int] = None,
        sizeau: Optional[float] = None,
        stokes: bool = False,
        output_dir: Optional[str] = None,
        distance: Optional[float] = None,
        coord: Optional[str] = None,
        object_name: str = '',
        **kwargs
    ) -> None:
        """Make continuum/scattered light image(s) at specified viewing angles.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in microns
        inclinations : float or list of float, optional
            Inclination(s) in degrees. If None, uses params
        posangs : float or list of float, optional
            Position angle(s) in degrees. If None, uses params
        phis : float or list of float, optional
            Azimuthal angle(s) in degrees. If None, uses params
        npix : int, optional
            Number of pixels. If None, uses params
        sizeau : float, optional
            Image size in AU. If None, uses params
        stokes : bool, optional
            Compute full Stokes parameters (default: False)
        output_dir : str, optional
            Output directory name (default: 'image_{wavelength}um')
        distance : float, optional
            Distance in pc. If None, uses params
        coord : str, optional
            Source coordinates. If None, uses params or default
        object_name : str, optional
            Object name for FITS header
        **kwargs : additional arguments passed to _run_radmc3d_image
        """
        # Get defaults from params
        if inclinations is None:
            inclinations = self.params.inclination
            if not isinstance(inclinations, list):
                inclinations = [inclinations]
        if posangs is None:
            # Read PA from params in standard convention, convert to RADMC-3D
            pa_standard = self.params.posangle
            if isinstance(pa_standard, list):
                posangs = [(pa + 90.0) % 360.0 for pa in pa_standard]
            else:
                posangs = [(pa_standard + 90.0) % 360.0]
        if phis is None:
            phis = [0.0]
        if npix is None:
            npix = self.params.nbpixels
        if sizeau is None:
            sizeau = self.params.mapsize.to('au').magnitude
        if distance is None:
            distance = self.params.distance.to('pc').magnitude
        if coord is None:
            coord = '0h0m0s 0d0m0s'
        if output_dir is None:
            base_name = f'image_{wavelength:.2f}um'
            output_dir = self._apply_name_templates(base_name)
        
        # Convert single values to lists
        if not isinstance(inclinations, (list, tuple, np.ndarray)):
            inclinations = [inclinations]
        if not isinstance(posangs, (list, tuple, np.ndarray)):
            posangs = [posangs]
        if not isinstance(phis, (list, tuple, np.ndarray)):
            phis = [phis]
        
        # Generate viewing angle combinations
        viewing_angles = [(i, p, phi) for i in inclinations for p in posangs for phi in phis]
        
        logger.info(f"Making {wavelength}um images at {len(viewing_angles)} viewing angles")
        
        # Run RADMC-3D for each viewing angle
        self.multi_angle_images = []
        self.viewing_angles = []
        
        # Store pixel sizes from first image
        saved_sizepix_x = None
        saved_sizepix_y = None
        
        for incl, pa, phi in viewing_angles:
            phi_str = f"{phi:.1f} deg" if phi is not None else "None"
            logger.info(f"  incl={incl:.1f} deg, PA={pa:.1f} deg, phi={phi_str}")
            
            # Run radmc3d image
            self._run_radmc3d_image(
                npix=npix,
                incl=incl,
                posang=pa,
                phi=phi,
                sizeau=sizeau,
                wavelength=wavelength,
                stokes=stokes,
                **kwargs
            )
            
            # Read image
            self.readImage('image.out')
            
            image_file = self.model_dir / 'image.out'
            if image_file.exists() or image_file.is_symlink():
                image_file.unlink()
            
            # Save pixel sizes from first image
            if saved_sizepix_x is None:
                saved_sizepix_x = self.sizepix_x
                saved_sizepix_y = self.sizepix_y
            
            # Store for multi-angle FITS
            self.multi_angle_images.append(self.image.copy())
            self.viewing_angles.append((incl, pa, phi))
        
        # Restore pixel sizes for FITS writing
        self.sizepix_x = saved_sizepix_x
        self.sizepix_y = saved_sizepix_y
        
        if self.sizepix_x is None or self.sizepix_y is None:
            raise RuntimeError(f"Pixel sizes not set for continuum image! sizepix_x={self.sizepix_x}, sizepix_y={self.sizepix_y}. "
                             f"This usually means readImage() failed.")
        
        logger.debug(f"Using pixel sizes: sizepix_x={self.sizepix_x/AU:.3f} AU, sizepix_y={self.sizepix_y/AU:.3f} AU")
        
        # Write multi-angle FITS
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fits_path = output_path / 'image.fits'
        self._write_multiangle_fits(
            fits_path,
            distance=distance,
            coord=coord,
            object_name=object_name,
        )
        
        # Copy params and add metadata
        self._organize_image_output(output_path, f"radmc3d image lambda {wavelength}")
        
        logger.info(f"Images written to {output_path}/image.fits")
        
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
        # Use default if None
        if coord is None:
            coord = '0h0m0s 0d0m0s'
        
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
    
    def _run_radmc3d_image(
        self,
        npix: int,
        incl: float,
        posang: float = 0.0,
        phi: float = 0.0,
        sizeau: Optional[float] = None,
        wavelength: Optional[float] = None,
        widthkms: Optional[float] = None,
        linenlam: Optional[int] = None,
        iline: Optional[int] = None,
        stokes: bool = False,
        **kwargs
    ) -> None:
        """Run RADMC-3D image command.
        
        Parameters
        ----------
        npix : int
            Number of pixels
        incl : float
            Inclination in degrees
        posang : float, optional
            Position angle in degrees
        phi : float, optional
            Azimuthal angle in degrees
        sizeau : float, optional
            Image size in AU
        wavelength : float, optional
            Wavelength in microns (for continuum)
        widthkms : float, optional
            Velocity width in km/s (for lines)
        linenlam : int, optional
            Number of channels (for lines)
        iline : int, optional
            Line transition index (for lines)
        stokes : bool, optional
            Compute Stokes parameters
        **kwargs : additional flags for radmc3d
        """
        # Build command
        cmd = ['radmc3d', 'image']
        cmd += ['npix', str(npix)]
        cmd += ['incl', str(incl)]
        cmd += ['posang', str(posang)]
        cmd += ['phi', str(phi)]
        
        if sizeau is not None:
            cmd += ['sizeau', str(sizeau)]
        
        secondorder_flag = getattr(self.params, 'secondorder', False)
        if secondorder_flag:
            cmd.append('secondorder')
        
        noscat_flag = getattr(self.params, 'noscat', False)
        if noscat_flag:
            cmd.append('noscat')
        
        # Wavelength or line parameters
        if wavelength is not None:
            cmd += ['lambda', str(wavelength)]
        elif iline is not None:
            cmd += ['iline', str(iline)]
            if widthkms is not None:
                cmd += ['widthkms', str(widthkms)]
            if linenlam is not None:
                cmd += ['linenlam', str(linenlam)]
        
        if stokes:
            cmd += ['stokes']
        
        # Additional kwargs
        for key, value in kwargs.items():
            if isinstance(value, bool) and value:
                cmd.append(key)
            elif not isinstance(value, bool):
                cmd.extend([key, str(value)])
        
        # Run command with symlink management
        cmd_str = ' '.join(cmd)
        logger.info(f"Running: {cmd_str}")
        
        image_file = self.model_dir / 'image.out'
        if image_file.exists() or image_file.is_symlink():
            image_file.unlink()
        
        # Create symlinks to organized input files
        self.create_symlinks()
        
        try:
            returncode, stdout, stderr, _, _ = run_radmc3d_and_log(
                cmd,
                self.model_dir,
                section='image',
                log_path=self.model_dir / 'radmc3d.out',
                command_str=cmd_str,
                preserve_existing=False,
            )
        finally:
            # Always clean up symlinks
            self.cleanup_symlinks()

        log_path = self.model_dir / 'radmc3d.out'
        
        if returncode != 0:
            logger.error(f"RADMC-3D image failed (see {log_path})")
            errors = _extract_radmc_errors(log_path)
            if errors:
                logger.error(f"RADMC-3D errors:\n{errors}")
                raise RuntimeError(
                    f"radmc3d image failed with exit code {returncode} and errors:\n{errors}"
                )
            raise RuntimeError(f"radmc3d image failed with exit code {returncode}")
        
        logger.debug("RADMC-3D image completed successfully")
    
    def _get_iline(self, molecule: str, transition: int) -> int:
        """Get iline index from molecule and transition.
        
        Parameters
        ----------
        molecule : str
            Molecule name (e.g., 'co')
        transition : int
            Transition number (e.g., 3 for J=3-2)
            
        Returns
        -------
        int
            Line index for RADMC-3D (1-indexed for command line)
        """
        try:
            from .molecule import RadMolecule
            
            mol = RadMolecule()
            # Molecule files are read from inputs_dir (radmc3d_inputs)
            mol_file = self.inputs_dir / f'molecule_{molecule}.inp'
            if not mol_file.exists():
                logger.warning(f"molecule_{molecule}.inp not found in radmc3d_inputs, using transition as iline")
                return transition
            
            mol.read(fname=mol_file)
            # Get 0-indexed line, add 1 for RADMC-3D command line
            iline = mol.getTransitionIndex(transition) + 1
            
            label = mol.getTransitionLabel(iline - 1)
            logger.info(f"Transition {molecule} {label} -> iline={iline}")
            
            return iline
            
        except Exception as e:
            logger.warning(f"Failed to parse molecule file: {e}, using transition as iline")
            return transition
    
    def _get_rest_frequency(self, molecule: str, transition: int) -> float:
        """Get rest frequency for a molecular transition.
        
        Parameters
        ----------
        molecule : str
            Molecule name (e.g., 'co')
        transition : int
            Transition number (e.g., 3 for J=3-2)
            
        Returns
        -------
        float
            Rest frequency in Hz
        """

        mol = RadMolecule()
        # Molecule files are read from inputs_dir (radmc3d_inputs)
        mol_file = self.inputs_dir / f'molecule_{molecule}.inp'
        if not mol_file.exists():
            logger.warning(f"molecule_{molecule}.inp not found in radmc3d_inputs, cannot determine rest frequency")
            return 0.0
        
        mol.read(fname=mol_file)
        freq = mol.getRestFrequency(transition)
        
        logger.info(f"Rest frequency for {molecule} J={transition}: {freq/1e9:.6f} GHz")
        
        return freq

    
    def _write_multiangle_fits(
        self,
        fname: Path,
        distance: float,
        coord: str,
        object_name: str = '',
        nu0: float = 0.0,
    ) -> None:
        """Write multi-angle 6D FITS file.
        
        Parameters
        ----------
        fname : Path
            Output FITS filename
        distance : float
            Distance in pc
        coord : str
            Source coordinates
        object_name : str, optional
            Object name
        nu0 : float, optional
            Rest frequency in Hz
        """
        if not self.multi_angle_images:
            logger.warning("No multi-angle images to write")
            return
        
        # Parse coordinates
        target_ra, target_dec = self._parse_coords(coord)
        
        # Organize angles
        incls = sorted(set(a[0] for a in self.viewing_angles))
        pas = sorted(set(a[1] for a in self.viewing_angles))
        phis = sorted(set(a[2] for a in self.viewing_angles))
        
        n_incl = len(incls)
        n_pa = len(pas)
        n_phi = len(phis)
        
        # Build 6D array: (n_phi, n_pa, n_incl, nfreq, ny, nx)
        # This matches the MCFOST format from the example
        if self.nfreq > 1:
            # Line cube
            data_6d = np.zeros((n_phi, n_pa, n_incl, self.nfreq, self.ny, self.nx), dtype=np.float32)
        else:
            # Continuum (no freq axis)
            data_6d = np.zeros((n_phi, n_pa, n_incl, self.ny, self.nx), dtype=np.float32)
        
        # Fill array
        conv = self.sizepix_x * self.sizepix_y / (distance * PC)**2 * 1e23
        
        for idx, (incl, pa, phi) in enumerate(self.viewing_angles):
            i_incl = incls.index(incl)
            i_pa = pas.index(pa)
            i_phi = phis.index(phi)
            
            img = self.multi_angle_images[idx]
            
            if self.nfreq > 1:
                # Line cube: transpose and convert
                for ifreq in range(self.nfreq):
                    data_6d[i_phi, i_pa, i_incl, ifreq, :, :] = img[:, :, ifreq].T * conv
            else:
                # Continuum
                data_6d[i_phi, i_pa, i_incl, :, :] = img[:, :, 0].T * conv
        
        # Create FITS HDU
        hdu = fits.PrimaryHDU(data_6d)
        
        # WCS headers for spatial axes
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CRVAL1'] = target_ra
        hdu.header['CRPIX1'] = (self.nx + 1.) / 2.
        hdu.header['CDELT1'] = -self.sizepix_x / AU / distance / 3600.
        hdu.header['CUNIT1'] = 'deg'
        
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CRVAL2'] = target_dec
        hdu.header['CRPIX2'] = (self.ny + 1.) / 2.
        hdu.header['CDELT2'] = self.sizepix_y / AU / distance / 3600.
        hdu.header['CUNIT2'] = 'deg'
        
        # Frequency axis (if line cube)
        if self.nfreq > 1:
            hdu.header['CTYPE3'] = 'VELO-LSR'
            hdu.header['CRPIX3'] = 1
            hdu.header['CRVAL3'] = -self.widthkms / 2.0 if hasattr(self, 'widthkms') else 0.0
            hdu.header['CDELT3'] = self.widthkms / self.nfreq if hasattr(self, 'widthkms') else 0.5
            hdu.header['CUNIT3'] = 'km/s'
            if nu0 > 0:
                hdu.header['RESTFRQ'] = nu0
        
        # Viewing angle metadata
        hdu.header['N_INCL'] = n_incl
        hdu.header['N_PA'] = n_pa
        hdu.header['N_PHI'] = n_phi
        for i, incl in enumerate(incls):
            hdu.header[f'INCL_{i}'] = incl
        for i, pa in enumerate(pas):
            hdu.header[f'PA_{i}'] = pa
        for i, phi in enumerate(phis):
            hdu.header[f'PHI_{i}'] = phi
        
        # Additional metadata
        hdu.header['BUNIT'] = 'W.m-2.pixel-1'  # Match MCFOST
        hdu.header['DISTANCE'] = (distance, 'Distance (pc)')
        if object_name:
            hdu.header['OBJECT'] = object_name
        
        # Write file
        hdu.writeto(fname, overwrite=True)
        logger.info(f"Multi-angle FITS written: {fname} (shape={data_6d.shape})")
    
    def _organize_image_output(self, output_dir: Path, command: str) -> None:
        """Copy params file and add metadata to output directory.
        
        Parameters
        ----------
        output_dir : Path
            Output directory
        command : str
            Command that was run
        """
        # Copy params file if it exists
        params_file = self.model_dir / 'params.txt'
        if params_file.exists():
            dst_params = output_dir / 'params.txt'
            shutil.copy2(str(params_file), str(dst_params))
            
            # Append metadata
            with open(dst_params, 'a') as f:
                f.write('\n# --- Run metadata (auto-generated) ---\n')
                f.write(f'timestamp = {datetime.datetime.now().isoformat()}\n')
                f.write(f'radmc3d_command = {command}\n')
            
            logger.debug(f"Copied params.txt -> {output_dir}")
    
    def _format_name_template(self, template: str) -> str:
        if not template:
            return ''
        try:
            params_obj = getattr(self, 'params', None)
            if params_obj is None:
                return template
            context = dict(params_obj.__dict__)
            for key, value in list(context.items()):
                if isinstance(value, bool):
                    context[key] = 'T' if value else 'F'
            return template.format(**context)
        except Exception as e:
            logger.warning(f"Failed to format name template '{template}': {e}")
            return template
    
    def _apply_name_templates(self, base_name: str) -> str:
        prepend = getattr(self.params, 'prepend_name', '')
        append = getattr(self.params, 'append_name', '')
        prepend_fmt = self._format_name_template(prepend) if prepend else ''
        append_fmt = self._format_name_template(append) if append else ''
        full_name = f"{prepend_fmt}{base_name}{append_fmt}"
        
        # Sanitize for shell/filesystem safety
        # Remove quotes, apostrophes, and replace spaces with underscores
        sanitized = full_name.replace("'", "").replace('"', "").replace(' ', '_')
        # Remove other problematic characters
        sanitized = sanitized.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        sanitized = sanitized.replace('{', '').replace('}', '').replace('|', '_')
        sanitized = sanitized.replace('&', '_').replace(';', '_').replace('$', '')
        # Clean up any double underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        return sanitized
    
    def _ensure_molecule_file(self, molecule: str) -> None:
        """Ensure molecule data file exists, download if necessary.
        
        Parameters
        ----------
        molecule : str
            Molecule name (e.g., 'co')
        """
        # Create inputs_dir if it doesn't exist
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        
        mol_file = self.inputs_dir / f'molecule_{molecule}.inp'
        if mol_file.exists():
            return
        
        local_moldata_dir = REPO_ROOT / "data" / "moldata"
        local_dat = local_moldata_dir / f"{molecule}.dat"
        if local_dat.exists():
            shutil.copyfile(local_dat, mol_file)
            logger.info(f"Copied local moldata file {local_dat} to {mol_file}")
            return

        logger.info(f"Molecule file not found, attempting to download {molecule}...")
        
        # Try to download from LAMDA
        import urllib.request
        try:
            url = f'https://home.strw.leidenuniv.nl/~moldata/datafiles/{molecule}.dat'
            urllib.request.urlretrieve(url, mol_file)
            logger.info(f"Downloaded molecule_{molecule}.inp from LAMDA to {mol_file}")
        except Exception as e:
            logger.error(f"Failed to download molecule data: {e}")
            raise RuntimeError(f"molecule_{molecule}.inp not found and download failed")
    
    def _ensure_gas_velocity(self) -> None:
        """Ensure gas velocity file exists, create if necessary."""
        # Check if gas_velocity.binp already exists in inputs_dir
        vel_file = self.inputs_dir / 'gas_velocity.binp'
        if vel_file.exists():
            # Sanity check: verify file is complete (header + 3*ncells values)
            try:
                with open(vel_file, 'rb') as f:
                    header = np.fromfile(f, dtype=np.int64, count=3)
                if header.size == 3:
                    ncells = int(header[2])
                    expected_bytes = 3 * 8 + 3 * ncells * 8
                    actual_bytes = vel_file.stat().st_size
                    if actual_bytes == expected_bytes:
                        return
            except Exception:
                # If anything goes wrong, fall through and regenerate
                pass
            logger.info("Existing gas_velocity.binp is invalid or incomplete; regenerating from model...")
        
        if self.model is None:
            raise RuntimeError(
                "gas_velocity.binp not found and no model provided to RadImage. "
                "Either provide model in __init__ or create gas_velocity.binp manually."
            )
        
        logger.info("Gas velocity file not found, creating from model...")
        
        from .writer import RadWriter
        writer = RadWriter(self.model)
        writer.write_gas_velocity(
            self.model.gas['vr'].data,
            self.model.gas['vtheta'].data,
            self.model.gas['vphi'].data,
            binary=True,
            output_dir=str(self.model_dir)
        )
        logger.info("Created gas_velocity.binp")
    
    def _ensure_radmc3d_inp_configured(self) -> None:
        """Ensure radmc3d.inp is properly configured for line transfer."""
        # Use radmc3d_inputs/radmc3d.inp
        inp_file = self.inputs_dir / 'radmc3d.inp'
        
        # Read existing file
        if inp_file.exists():
            with open(inp_file, 'r') as f:
                content = f.read()
            
            # Check if already configured
            has_lines = 'incl_lines' in content and '= 1' in content
            has_tgas = 'tgas_eq_tdust' in content
            has_itemp = 'itempdecoup' in content
            has_rto = 'rto_style' in content
            
            if has_lines and has_tgas and has_itemp and has_rto:
                return  # Already configured
        else:
            content = ""
        
        logger.info("Updating radmc3d.inp for line transfer...")
        
        # If file doesn't exist, create it in inputs_dir
        if not inp_file.exists():
            self.inputs_dir.mkdir(parents=True, exist_ok=True)
            inp_file = self.inputs_dir / 'radmc3d.inp'
            
            # Write/update configuration
            from .writer import RadWriter
            writer = RadWriter(self.model, organize_files=True) if self.model else None
            
            if writer:
                writer.write_radmc3d_inp(
                    output_dir=str(self.model_dir),
                    incl_dust=1,
                    incl_lines=1,
                )
                # Re-read to get the file in inputs_dir
                inp_file = self.inputs_dir / 'radmc3d.inp'
                if inp_file.exists():
                    with open(inp_file, 'r') as f:
                        content = f.read()
        
        # Append additional settings
        with open(inp_file, 'a') as f:
            if 'tgas_eq_tdust' not in content:
                f.write('tgas_eq_tdust = 1\n')
            if 'itempdecoup' not in content:
                f.write('itempdecoup = 1\n')
            if 'rto_style' not in content:
                f.write('rto_style = 3\n')
        
        logger.info(f"Updated radmc3d.inp at {inp_file}")
    
    def _ensure_lines_inp(self, molecule: str) -> None:
        """Ensure lines.inp file exists for specified molecule.
        
        Parameters
        ----------
        molecule : str
            Molecule name
        """
        # Create inputs_dir if it doesn't exist
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        
        lines_file = self.inputs_dir / 'lines.inp'
        
        # Check if file exists and already has this molecule
        if lines_file.exists():
            with open(lines_file, 'r') as f:
                content = f.read()
            if molecule in content:
                return  # Already configured
        
        logger.info(f"Creating lines.inp for {molecule}...")
        
        # Write lines.inp
        with open(lines_file, 'w') as f:
            f.write('2\n')  # Format number
            f.write('1\n')  # Number of molecules
            f.write(f'{molecule}    leiden    0    0    0\n')
        
        logger.info(f"Created lines.inp to {lines_file}")
