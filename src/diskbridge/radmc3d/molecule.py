"""RADMC-3D molecule data handler.

This module provides the RadMolecule class for reading and handling 
molecular data files in the Leiden LAMDA format used by RADMC-3D.

The class handles:
- Reading molecule_<name>.inp files
- Storing energy levels, transitions, and line data
- Computing partition functions
- Providing molecular properties for radiative transfer

This mirrors the structure of radmc3dPy.molecule but adapted for DiskBridge.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np

from diskbridge._logging import logger

# Physical constants (CGS)
H_PLANCK = 6.62607015e-27  # erg s
C_LIGHT = 2.99792458e10    # cm/s
K_BOLTZMANN = 1.380649e-16  # erg/K


class RadMolecule:
    """RADMC-3D molecule class.
    
    Handles molecular data in the Leiden LAMDA database format.
    This includes energy levels, radiative transitions, and collision rates.
    
    For now, collision rates are read but not extensively used, focusing on
    levels and lines needed for radiative transfer.
    
    Attributes
    ----------
    name : str
        Molecule name from the data file
    molweight : float
        Molecular weight in units of proton mass
    nlev : int
        Number of energy levels
    nlin : int
        Number of radiative transitions
    energycminv : ndarray
        Energy of each level in cm^-1
    energy : ndarray
        Energy of each level in erg
    wgt : ndarray
        Statistical weight of each level
    jrot : ndarray
        Rotational quantum number J for each level
    iup : ndarray
        Upper level index for each transition (0-indexed)
    ilow : ndarray
        Lower level index for each transition (0-indexed)
    aud : ndarray
        Einstein A coefficient (s^-1) for each transition
    freq : ndarray
        Frequency of each transition in Hz
    lam : ndarray
        Wavelength of each transition in microns
    temp : ndarray or None
        Temperature grid for partition function
    pfunc : ndarray or None
        Partition function values
        
    Examples
    --------
    >>> from diskbridge.radmc3d import RadMolecule
    >>> 
    >>> # Read CO molecule data
    >>> mol = RadMolecule()
    >>> mol.read(mol='co')
    >>> 
    >>> # Compute partition function
    >>> mol.getPartitionFunction(tmin=5.0, tmax=300.0, ntemp=50)
    """
    
    def __init__(self):
        """Initialize empty molecule."""
        self.name: str = ""
        self.molweight: float = 0.0
        self.nlev: int = 0
        self.nlin: int = 0
        self.energycminv: np.ndarray = np.array([])
        self.energy: np.ndarray = np.array([])
        self.wgt: np.ndarray = np.array([])
        self.jrot: np.ndarray = np.array([])
        self.iup: np.ndarray = np.array([])
        self.ilow: np.ndarray = np.array([])
        self.aud: np.ndarray = np.array([])
        self.freq: np.ndarray = np.array([])
        self.lam: np.ndarray = np.array([])
        self.temp: Optional[np.ndarray] = None
        self.pfunc: Optional[np.ndarray] = None
        
        # Collision partner data (stored but not extensively used yet)
        self.npartner: int = 0
        self.partner_names: list = []
        
    def read(self, mol: Optional[str] = None, fname: Optional[str | Path] = None) -> bool:
        """Read molecule data file.
        
        Reads a molecule_<mol>.inp file in the Leiden LAMDA format.
        
        Parameters
        ----------
        mol : str, optional
            Molecule name (e.g., 'co') for file molecule_<mol>.inp
        fname : str or Path, optional
            Full filename if not using the standard naming convention
            
        Returns
        -------
        bool
            True if successful
            
        Raises
        ------
        ValueError
            If neither mol nor fname is provided
        FileNotFoundError
            If the specified file does not exist
        """
        if fname is None:
            if mol is None:
                raise ValueError('Must provide either mol or fname')
            fname = f'molecule_{mol}.inp'
        
        fpath = Path(fname)
        if not fpath.exists():
            raise FileNotFoundError(f"Molecule file not found: {fpath}")
        
        with open(fpath, 'r') as f:
            logger.info(f"Reading molecule file: {fpath}")
            
            # Line 1: Comment/identifier
            _ = f.readline()
            
            # Line 2: Molecule name
            self.name = f.readline().split()[0]
            
            # Line 3: Comment
            _ = f.readline()
            
            # Line 4: Molecular weight
            self.molweight = float(f.readline().strip())
            
            # Line 5: Comment
            _ = f.readline()
            
            # Line 6: Number of energy levels
            self.nlev = int(f.readline().strip())
            
            # Line 7: Comment
            _ = f.readline()
            
            # Energy levels
            self.energycminv = np.zeros(self.nlev, dtype=np.float64)
            self.energy = np.zeros(self.nlev, dtype=np.float64)
            self.wgt = np.zeros(self.nlev, dtype=np.float64)
            self.jrot = np.zeros(self.nlev, dtype=np.float64)
            
            for i in range(self.nlev):
                parts = f.readline().split()
                # Format: level_index energy[cm^-1] weight J
                self.energycminv[i] = float(parts[1])
                self.energy[i] = float(parts[1]) * H_PLANCK * C_LIGHT
                self.wgt[i] = float(parts[2])
                self.jrot[i] = float(parts[3])
            
            # Line: Comment
            _ = f.readline()
            
            # Line: Number of radiative transitions
            self.nlin = int(f.readline().strip())
            
            # Line: Comment
            _ = f.readline()
            
            # Radiative transitions
            self.iup = np.zeros(self.nlin, dtype=np.int64)
            self.ilow = np.zeros(self.nlin, dtype=np.int64)
            self.aud = np.zeros(self.nlin, dtype=np.float64)
            self.freq = np.zeros(self.nlin, dtype=np.float64)
            self.lam = np.zeros(self.nlin, dtype=np.float64)
            
            for i in range(self.nlin):
                parts = f.readline().split()
                # Format: transition iup ilow Aul freq[GHz] Eup[K]
                self.iup[i] = int(parts[1])    # Note: 1-indexed in file
                self.ilow[i] = int(parts[2])   # Note: 1-indexed in file
                self.aud[i] = float(parts[3])
                self.freq[i] = float(parts[4]) * 1e9  # GHz to Hz
                self.lam[i] = C_LIGHT / self.freq[i] * 1e4  # cm to micron
            
            # The rest of the file contains collision partner data
            # We read it but don't use it extensively yet
            try:
                _ = f.readline()  # Comment
                self.npartner = int(f.readline().strip())
                
                for _ in range(self.npartner):
                    _ = f.readline()  # Comment
                    partner_id = f.readline().strip()
                    self.partner_names.append(partner_id)
                    
                    _ = f.readline()  # Comment
                    ncoll = int(f.readline().strip())
                    
                    _ = f.readline()  # Comment
                    ntemp = int(f.readline().strip())
                    
                    _ = f.readline()  # Comment
                    _ = f.readline()  # Temperature grid
                    
                    _ = f.readline()  # Comment
                    # Skip collision rate data
                    for _ in range(ncoll):
                        _ = f.readline()
            except:
                # If collision data is incomplete or missing, that's ok
                pass
        
        logger.info(f"Successfully read molecule '{self.name}': "
                   f"{self.nlev} levels, {self.nlin} lines")
        
        return True
    
    def getPartitionFunction(
        self,
        temp: Optional[np.ndarray] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        ntemp: Optional[int] = None,
        tlog: bool = True
    ) -> None:
        """Calculate partition function at a temperature grid.
        
        Computes Q(T) = Σ_i g_i exp(-E_i / k_B T) where the sum is over
        all energy levels.
        
        Parameters
        ----------
        temp : ndarray, optional
            Temperature array in K. If provided, tmin/tmax/ntemp are ignored.
        tmin : float, optional
            Minimum temperature in K (used if temp is None)
        tmax : float, optional
            Maximum temperature in K (used if temp is None)
        ntemp : int, optional
            Number of temperature points (used if temp is None)
        tlog : bool, optional
            If True, use logarithmic spacing (default: True)
            
        Raises
        ------
        ValueError
            If neither temp nor (tmin, tmax, ntemp) is provided
        """
        if temp is None:
            if tmin is None or tmax is None or ntemp is None:
                raise ValueError(
                    'Must provide either temp or (tmin, tmax, ntemp)'
                )
            
            if tlog:
                self.temp = tmin * (tmax / tmin) ** (np.arange(ntemp) / (ntemp - 1))
            else:
                self.temp = tmin + (tmax - tmin) * (np.arange(ntemp) / (ntemp - 1))
        else:
            # Convert input to numpy array
            if isinstance(temp, (float, int)):
                self.temp = np.array([temp], dtype=np.float64)
            elif isinstance(temp, (list, tuple)):
                self.temp = np.array(temp, dtype=np.float64)
            elif isinstance(temp, np.ndarray):
                self.temp = temp.astype(np.float64)
            else:
                raise TypeError(
                    'temp must be a float, list, tuple, or numpy array'
                )
        
        # Compute partition function
        self.pfunc = np.zeros(self.temp.shape[0], dtype=np.float64)
        
        for it, T in enumerate(self.temp):
            self.pfunc[it] = np.sum(
                self.wgt * np.exp(-self.energy / (K_BOLTZMANN * T))
            )
        
        logger.info(f"Computed partition function for {len(self.temp)} temperatures: "
                   f"T=[{self.temp.min():.1f}, {self.temp.max():.1f}] K")
    
    def getLineInfo(self, iline: int) -> dict:
        """Get information about a specific line.
        
        Parameters
        ----------
        iline : int
            Line index (0-indexed)
            
        Returns
        -------
        dict
            Dictionary with line properties
        """
        if iline < 0 or iline >= self.nlin:
            raise ValueError(f"Line index {iline} out of range [0, {self.nlin})")
        
        return {
            'iline': iline,
            'iup': self.iup[iline],
            'ilow': self.ilow[iline],
            'frequency_Hz': self.freq[iline],
            'wavelength_micron': self.lam[iline],
            'Aul_Hz': self.aud[iline],
            'Eup_K': self.energycminv[self.iup[iline] - 1] * H_PLANCK * C_LIGHT / K_BOLTZMANN
        }
    
    def getLinesInRange(
        self,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        lam_min: Optional[float] = None,
        lam_max: Optional[float] = None
    ) -> np.ndarray:
        """Get line indices within a frequency or wavelength range.
        
        Parameters
        ----------
        freq_min : float, optional
            Minimum frequency in Hz
        freq_max : float, optional
            Maximum frequency in Hz
        lam_min : float, optional
            Minimum wavelength in microns
        lam_max : float, optional
            Maximum wavelength in microns
            
        Returns
        -------
        ndarray
            Array of line indices matching the criteria
            
        Raises
        ------
        ValueError
            If neither frequency nor wavelength range is specified
        """
        if freq_min is not None or freq_max is not None:
            # Use frequency range
            mask = np.ones(self.nlin, dtype=bool)
            if freq_min is not None:
                mask &= self.freq >= freq_min
            if freq_max is not None:
                mask &= self.freq <= freq_max
        elif lam_min is not None or lam_max is not None:
            # Use wavelength range
            mask = np.ones(self.nlin, dtype=bool)
            if lam_min is not None:
                mask &= self.lam >= lam_min
            if lam_max is not None:
                mask &= self.lam <= lam_max
        else:
            raise ValueError('Must specify either frequency or wavelength range')
        
        indices = np.where(mask)[0]
        logger.info(f"Found {len(indices)} lines in specified range")
        
        return indices
    
    def __repr__(self) -> str:
        """String representation of molecule."""
        if self.name:
            return (f"radmc3dMolecule(name='{self.name}', "
                   f"nlev={self.nlev}, nlin={self.nlin}, "
                   f"weight={self.molweight:.2f})")
        else:
            return "radmc3dMolecule(empty)"
