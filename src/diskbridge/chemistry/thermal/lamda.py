"""LAMDA file parser for statistical equilibrium calculations.

Parses LAMDA .dat files at import time and provides numpy arrays for use
in Numba-accelerated cooling functions.

References
----------
- Schoier et al. 2005, A&A 432, 369 (LAMDA database)
- van der Tak et al. 2020, Atoms 8, 15 (LAMDA update)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Physical constants for energy conversions
H_PLANCK = 6.62607015e-27  # erg s
C_LIGHT = 2.99792458e10    # cm/s
K_BOLTZMANN = 1.380649e-16 # erg/K

# LAMDA collider ID mapping
COLLIDER_NAMES = {
    1: 'H2',
    2: 'pH2',
    3: 'oH2',
    4: 'e',
    5: 'H',
    6: 'He',
    7: 'H+',
}


@dataclass
class CollisionPartner:
    """Collision partner data from LAMDA file."""
    id: int
    name: str
    n_trans: int
    n_temps: int
    temps: np.ndarray  # [n_temps]
    trans_u: np.ndarray  # [n_trans]
    trans_l: np.ndarray  # [n_trans]
    rates: np.ndarray  # [n_trans, n_temps]


@dataclass
class LAMDAData:
    """Parsed LAMDA molecular data."""
    name: str
    weight: float
    n_levels: int
    energies_cm: np.ndarray  # [n_levels] in cm^-1
    energies_K: np.ndarray   # [n_levels] in K
    weights: np.ndarray      # [n_levels] statistical weights
    n_trans: int
    trans_u: np.ndarray      # [n_trans] upper level (0-indexed)
    trans_l: np.ndarray      # [n_trans] lower level (0-indexed)
    A_ul: np.ndarray         # [n_trans] Einstein A coefficients
    freq_GHz: np.ndarray     # [n_trans] frequencies in GHz
    E_ul_K: np.ndarray       # [n_trans] transition energies in K
    h_nu: np.ndarray         # [n_trans] photon energies in erg
    colliders: Dict[str, CollisionPartner]


def parse_lamda_file(filepath: Path) -> LAMDAData:
    """Parse a LAMDA .dat file.
    
    Parameters
    ----------
    filepath : Path
        Path to the LAMDA .dat file
        
    Returns
    -------
    LAMDAData
        Parsed molecular data
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Filter out comments and empty lines, but keep track of positions
    def next_data_line(start_idx: int) -> tuple[str, int]:
        """Get next non-comment line and its index."""
        idx = start_idx
        while idx < len(lines):
            line = lines[idx].strip()
            if line and not line.startswith('!'):
                return line, idx + 1
            idx += 1
        raise ValueError(f"Unexpected end of file at line {start_idx}")
    
    idx = 0
    
    # Skip !MOLECULE header
    while idx < len(lines) and lines[idx].strip().startswith('!'):
        idx += 1
    
    # Molecule name
    name, idx = next_data_line(idx)
    
    # Skip !MOLECULAR WEIGHT
    while idx < len(lines) and lines[idx].strip().startswith('!'):
        idx += 1
    weight_str, idx = next_data_line(idx)
    weight = float(weight_str)
    
    # Skip !NUMBER OF ENERGY LEVELS
    while idx < len(lines) and lines[idx].strip().startswith('!'):
        idx += 1
    n_levels_str, idx = next_data_line(idx)
    n_levels = int(n_levels_str)
    
    # Skip level header
    while idx < len(lines) and lines[idx].strip().startswith('!'):
        idx += 1
    
    # Parse levels
    energies_cm = np.zeros(n_levels, dtype=np.float64)
    weights = np.zeros(n_levels, dtype=np.float64)
    
    for i in range(n_levels):
        line, idx = next_data_line(idx)
        parts = line.split()
        # Format: level_num energy weight J [comment]
        energies_cm[i] = float(parts[1])
        weights[i] = float(parts[2])
    
    # Convert energies to K
    energies_K = energies_cm * H_PLANCK * C_LIGHT / K_BOLTZMANN
    
    # Skip !NUMBER OF RADIATIVE TRANSITIONS
    while idx < len(lines) and lines[idx].strip().startswith('!'):
        idx += 1
    n_trans_str, idx = next_data_line(idx)
    n_trans = int(n_trans_str)
    
    # Skip transition header
    while idx < len(lines) and lines[idx].strip().startswith('!'):
        idx += 1
    
    # Parse transitions
    trans_u = np.zeros(n_trans, dtype=np.int32)
    trans_l = np.zeros(n_trans, dtype=np.int32)
    A_ul = np.zeros(n_trans, dtype=np.float64)
    freq_GHz = np.zeros(n_trans, dtype=np.float64)
    E_ul_K = np.zeros(n_trans, dtype=np.float64)
    
    for i in range(n_trans):
        line, idx = next_data_line(idx)
        parts = line.split()
        # Format: trans_num up low A freq E_u
        trans_u[i] = int(parts[1]) - 1  # Convert to 0-indexed
        trans_l[i] = int(parts[2]) - 1
        A_ul[i] = float(parts[3])
        freq_GHz[i] = float(parts[4])
        E_ul_K[i] = float(parts[5])
    
    # Compute photon energies
    h_nu = H_PLANCK * freq_GHz * 1e9  # erg
    
    # Skip !NUMBER OF COLL PARTNERS
    while idx < len(lines) and lines[idx].strip().startswith('!'):
        idx += 1
    n_partners_str, idx = next_data_line(idx)
    n_partners = int(n_partners_str)
    
    # Parse collision partners
    colliders: Dict[str, CollisionPartner] = {}
    
    for _ in range(n_partners):
        # Skip !COLLISIONS BETWEEN
        while idx < len(lines) and lines[idx].strip().startswith('!'):
            idx += 1
        partner_line, idx = next_data_line(idx)
        parts = partner_line.split()
        partner_id = int(parts[0])
        partner_name = COLLIDER_NAMES.get(partner_id, f'partner_{partner_id}')
        
        # Skip !NUMBER OF COLL TRANS
        while idx < len(lines) and lines[idx].strip().startswith('!'):
            idx += 1
        coll_n_trans_str, idx = next_data_line(idx)
        coll_n_trans = int(coll_n_trans_str)
        
        # Skip !NUMBER OF COLL TEMPS
        while idx < len(lines) and lines[idx].strip().startswith('!'):
            idx += 1
        coll_n_temps_str, idx = next_data_line(idx)
        coll_n_temps = int(coll_n_temps_str)
        
        # Skip !COLL TEMPS
        while idx < len(lines) and lines[idx].strip().startswith('!'):
            idx += 1
        temps_line, idx = next_data_line(idx)
        temps = np.array([float(x) for x in temps_line.split()], dtype=np.float64)
        
        # Skip !TRANS + UP + LOW + COLLRATES
        while idx < len(lines) and lines[idx].strip().startswith('!'):
            idx += 1
        
        # Parse collision rates
        coll_trans_u = np.zeros(coll_n_trans, dtype=np.int32)
        coll_trans_l = np.zeros(coll_n_trans, dtype=np.int32)
        coll_rates = np.zeros((coll_n_trans, coll_n_temps), dtype=np.float64)
        
        for j in range(coll_n_trans):
            line, idx = next_data_line(idx)
            parts = line.split()
            coll_trans_u[j] = int(parts[1]) - 1  # 0-indexed
            coll_trans_l[j] = int(parts[2]) - 1
            for k, rate_str in enumerate(parts[3:3+coll_n_temps]):
                coll_rates[j, k] = float(rate_str)
        
        colliders[partner_name] = CollisionPartner(
            id=partner_id,
            name=partner_name,
            n_trans=coll_n_trans,
            n_temps=coll_n_temps,
            temps=temps,
            trans_u=coll_trans_u,
            trans_l=coll_trans_l,
            rates=coll_rates,
        )
    
    return LAMDAData(
        name=name,
        weight=weight,
        n_levels=n_levels,
        energies_cm=energies_cm,
        energies_K=energies_K,
        weights=weights,
        n_trans=n_trans,
        trans_u=trans_u,
        trans_l=trans_l,
        A_ul=A_ul,
        freq_GHz=freq_GHz,
        E_ul_K=E_ul_K,
        h_nu=h_nu,
        colliders=colliders,
    )


def _get_data_dir() -> Path:
    """Get the path to the moldata directory."""
    # Navigate from this file to data/moldata
    this_file = Path(__file__)
    return this_file.parent.parent.parent.parent.parent / 'data' / 'moldata'


def _load_species(filename: str) -> LAMDAData:
    """Load a species from the moldata directory."""
    data_dir = _get_data_dir()
    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"LAMDA file not found: {filepath}")
    return parse_lamda_file(filepath)


# =============================================================================
# Load species data at import time
# =============================================================================

CPLUS = _load_species('c+.dat')
C = _load_species('catom.dat')
O = _load_species('oatom.dat')
CO = _load_species('co.dat')


# =============================================================================
# Export numpy arrays for Numba kernels
# =============================================================================

# C+ arrays
CPLUS_N_LEVELS = CPLUS.n_levels
CPLUS_E_LEVELS_K = CPLUS.energies_K
CPLUS_G_LEVELS = CPLUS.weights
CPLUS_N_TRANS = CPLUS.n_trans
CPLUS_TRANS_U = CPLUS.trans_u
CPLUS_TRANS_L = CPLUS.trans_l
CPLUS_A_UL = CPLUS.A_ul
CPLUS_E_UL_K = CPLUS.E_ul_K
CPLUS_HNU = CPLUS.h_nu

# C+ collision rates (need to handle different temp grids per collider)
CPLUS_COLL_PH2_T = CPLUS.colliders['pH2'].temps
CPLUS_COLL_PH2_Q = CPLUS.colliders['pH2'].rates
CPLUS_COLL_OH2_T = CPLUS.colliders['oH2'].temps
CPLUS_COLL_OH2_Q = CPLUS.colliders['oH2'].rates
CPLUS_COLL_H_T = CPLUS.colliders['H'].temps
CPLUS_COLL_H_Q = CPLUS.colliders['H'].rates
CPLUS_COLL_E_T = CPLUS.colliders['e'].temps
CPLUS_COLL_E_Q = CPLUS.colliders['e'].rates

# C arrays
C_N_LEVELS = C.n_levels
C_E_LEVELS_K = C.energies_K
C_G_LEVELS = C.weights
C_N_TRANS = C.n_trans
C_TRANS_U = C.trans_u
C_TRANS_L = C.trans_l
C_A_UL = C.A_ul
C_HNU = C.h_nu

# C collision rates (raw pH2/oH2)
C_COLL_PH2_T = C.colliders['pH2'].temps
C_COLL_PH2_Q = C.colliders['pH2'].rates
C_COLL_OH2_T = C.colliders['oH2'].temps
C_COLL_OH2_Q = C.colliders['oH2'].rates
C_COLL_H_T = C.colliders['H'].temps
C_COLL_H_Q = C.colliders['H'].rates
C_COLL_E_T = C.colliders['e'].temps
C_COLL_E_Q = C.colliders['e'].rates

# NOTE: pH2 and oH2 rates are NOT pre-mixed here.
# The blended OPR model mixes them at runtime based on T and fH2.
# See _kernels.py: f_ortho_h2() and interp_rate_mix()

# O arrays
O_N_LEVELS = O.n_levels
O_E_LEVELS_K = O.energies_K
O_G_LEVELS = O.weights
O_N_TRANS = O.n_trans
O_TRANS_U = O.trans_u
O_TRANS_L = O.trans_l
O_A_UL = O.A_ul
O_HNU = O.h_nu

# O collision rates (raw pH2/oH2)
O_COLL_PH2_T = O.colliders['pH2'].temps
O_COLL_PH2_Q = O.colliders['pH2'].rates
O_COLL_OH2_T = O.colliders['oH2'].temps
O_COLL_OH2_Q = O.colliders['oH2'].rates
O_COLL_H_T = O.colliders['H'].temps
O_COLL_H_Q = O.colliders['H'].rates
O_COLL_E_T = O.colliders['e'].temps
O_COLL_E_Q = O.colliders['e'].rates

# NOTE: pH2 and oH2 rates are NOT pre-mixed here.
# The blended OPR model mixes them at runtime based on T and fH2.

# CO arrays
CO_N_LEVELS = CO.n_levels
CO_E_LEVELS_K = CO.energies_K
CO_G_LEVELS = CO.weights
CO_N_TRANS = CO.n_trans
CO_TRANS_U = CO.trans_u
CO_TRANS_L = CO.trans_l
CO_A_UL = CO.A_ul
CO_HNU = CO.h_nu

# CO collision rates (only pH2 available in standard LAMDA CO file)
CO_COLL_PH2_T = CO.colliders['pH2'].temps
CO_COLL_PH2_Q = CO.colliders['pH2'].rates
