from __future__ import annotations

import numpy as np

from diskbridge.chemistry.astrochemistry.umist_io import UmistReaction
from diskbridge.chemistry.astrochemistry.species import parse_species_formula


REACTION_TYPES = {
    'AD': 1,   # Associative detachment
    'CD': 2,   # Collision-induced dissociation
    'CE': 3,   # Charge exchange
    'CP': 4,   # Cosmic ray proton
    'CR': 5,   # Cosmic ray
    'DR': 6,   # Dissociative recombination
    'GR': 7,   # Grain surface
    'IN': 8,   # Ion-neutral
    'MN': 9,   # Mutual neutralization
    'NN': 10,  # Neutral-neutral
    'PH': 11,  # Photoprocess
    'RA': 12,  # Radiative association
    'REA': 13, # Radiative electron attachment
    'RR': 14,  # Radiative recombination
}


def compile_network(
    reactions: list[UmistReaction],
    species: list[str],
) -> dict:
    """Compile reactions and species into NumPy arrays.
    
    Creates deterministic integer-indexed arrays for efficient rate computation.
    
    Parameters
    ----------
    reactions : list[UmistReaction]
        Filtered reaction list
    species : list[str]
        Species list (should be sorted deterministically)
        
    Returns
    -------
    dict
        Network arrays with keys:
        - species: ndarray[str] shape (ns,)
        - charge: ndarray[int16] shape (ns,)
        - ir1, ir2: ndarray[int32] shape (nr,) - reactant indices (-1 for none)
        - ip1, ip2, ip3, ip4: ndarray[int32] shape (nr,) - product indices (-1 for none)
        - rtype: ndarray[int16] shape (nr,) - reaction type code
        - alpha, beta, gamma: ndarray[float64] shape (nr,)
        - Tlow, Thigh: ndarray[float64] shape (nr,)
        - S: ndarray[int16] shape (ns, nr) - stoichiometry matrix
        - elements: ndarray[str] - unique elements
        - E: ndarray[int16] shape (nelem, ns) - element count matrix
        
    Raises
    ------
    ValueError
        If species index is invalid or reaction type unknown
    """
    ns = len(species)
    nr = len(reactions)
    
    species_to_idx = {s: i for i, s in enumerate(species)}
    
    species_arr = np.array(species, dtype='U16')
    
    charge = np.zeros(ns, dtype=np.int16)
    for i, s in enumerate(species):
        charge[i], _ = parse_species_formula(s)
    
    ir1 = np.full(nr, -1, dtype=np.int32)
    ir2 = np.full(nr, -1, dtype=np.int32)
    ip1 = np.full(nr, -1, dtype=np.int32)
    ip2 = np.full(nr, -1, dtype=np.int32)
    ip3 = np.full(nr, -1, dtype=np.int32)
    ip4 = np.full(nr, -1, dtype=np.int32)
    
    rtype = np.zeros(nr, dtype=np.int16)
    alpha = np.zeros(nr, dtype=np.float64)
    beta = np.zeros(nr, dtype=np.float64)
    gamma = np.zeros(nr, dtype=np.float64)
    Tlow = np.zeros(nr, dtype=np.float64)
    Thigh = np.zeros(nr, dtype=np.float64)
    
    S = np.zeros((ns, nr), dtype=np.int16)
    
    for r, rec in enumerate(reactions):
        R1, R2 = rec.R
        P1, P2, P3, P4 = rec.P
        
        if R1:
            ir1[r] = species_to_idx[R1]
            S[ir1[r], r] -= 1
        if R2:
            ir2[r] = species_to_idx[R2]
            S[ir2[r], r] -= 1
        
        if P1:
            ip1[r] = species_to_idx[P1]
            S[ip1[r], r] += 1
        if P2:
            ip2[r] = species_to_idx[P2]
            S[ip2[r], r] += 1
        if P3:
            ip3[r] = species_to_idx[P3]
            S[ip3[r], r] += 1
        if P4:
            ip4[r] = species_to_idx[P4]
            S[ip4[r], r] += 1
        
        if rec.rtype not in REACTION_TYPES:
            raise ValueError(f"Unknown reaction type: {rec.rtype!r}")
        rtype[r] = REACTION_TYPES[rec.rtype]
        
        alpha[r] = rec.alpha
        beta[r] = rec.beta
        gamma[r] = rec.gamma
        Tlow[r] = rec.Tlow
        Thigh[r] = rec.Thigh
    
    all_elements = set()
    species_elem_counts = []
    for s in species:
        _, elem_dict = parse_species_formula(s)
        species_elem_counts.append(elem_dict)
        all_elements.update(elem_dict.keys())
    
    elements = sorted(all_elements)
    nelem = len(elements)
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    E = np.zeros((nelem, ns), dtype=np.int16)
    for i, elem_dict in enumerate(species_elem_counts):
        for elem, count in elem_dict.items():
            E[elem_to_idx[elem], i] = count
    
    return {
        'species': species_arr,
        'charge': charge,
        'ir1': ir1,
        'ir2': ir2,
        'ip1': ip1,
        'ip2': ip2,
        'ip3': ip3,
        'ip4': ip4,
        'rtype': rtype,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'Tlow': Tlow,
        'Thigh': Thigh,
        'S': S,
        'elements': np.array(elements, dtype='U2'),
        'E': E,
    }


def sort_species_deterministic(species: list[str]) -> list[str]:
    """Sort species list deterministically.
    
    Forces 'e-' first, then lexicographic order for the rest.
    
    Parameters
    ----------
    species : list[str]
        Unsorted species list
        
    Returns
    -------
    list[str]
        Sorted species list
    """
    if 'e-' in species:
        return ['e-'] + sorted(s for s in species if s != 'e-')
    else:
        return sorted(species)


def validate_compiled_network(network: dict) -> tuple[bool, list[str]]:
    """Validate compiled network for correctness.
    
    Checks:
    - All indices are valid (-1 or < ns)
    - Stoichiometry matrix agrees with reactant/product indices
    
    Parameters
    ----------
    network : dict
        Compiled network from compile_network()
        
    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, error_messages)
    """
    errors = []
    ns = len(network['species'])
    nr = network['ir1'].shape[0]
    
    for idx_name in ['ir1', 'ir2', 'ip1', 'ip2', 'ip3', 'ip4']:
        idx_arr = network[idx_name]
        invalid = (idx_arr < -1) | (idx_arr >= ns)
        if np.any(invalid):
            bad_idx = np.where(invalid)[0]
            errors.append(
                f"{idx_name}: {len(bad_idx)} invalid indices (must be -1 or < {ns})"
            )
    
    S = network['S']
    for r in range(min(nr, 10)):
        S_expected = np.zeros(ns, dtype=np.int16)
        
        for idx in [network['ir1'][r], network['ir2'][r]]:
            if idx >= 0:
                S_expected[idx] -= 1
        
        for idx in [network['ip1'][r], network['ip2'][r], network['ip3'][r], network['ip4'][r]]:
            if idx >= 0:
                S_expected[idx] += 1
        
        if not np.array_equal(S[:, r], S_expected):
            errors.append(f"Stoichiometry mismatch for reaction {r}")
    
    return len(errors) == 0, errors
