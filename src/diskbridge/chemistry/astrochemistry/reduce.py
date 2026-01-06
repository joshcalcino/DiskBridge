from __future__ import annotations

from typing import Optional

from diskbridge.chemistry.astrochemistry.umist_io import UmistReaction, reaction_species
from diskbridge.chemistry.astrochemistry.species import element_set as get_element_set, parse_species_formula


def reduce_network(
    reactions: list[UmistReaction],
    species_init: list[str],
    mode: str = "strict",
    max_species: int = 64,
    allowed_elements: Optional[set[str]] = None,
) -> tuple[list[UmistReaction], list[str]]:
    """Reduce UMIST network to reactions involving specified species.
    
    Parameters
    ----------
    reactions : list[UmistReaction]
        Full UMIST reaction list
    species_init : list[str]
        Initial species set
    mode : str, optional
        Reduction mode:
        - 'strict': Keep only reactions where all species are in whitelist
        - 'closure': Iteratively add species connected to initial set (bounded)
    max_species : int, optional
        Maximum total species (for closure mode)
    allowed_elements : set[str], optional
        Allowed elements (for closure mode). If None, derived from species_init.
        
    Returns
    -------
    tuple[list[UmistReaction], list[str]]
        (filtered_reactions, final_species_list)
        
    Raises
    ------
    ValueError
        If mode is invalid or species count exceeds max_species
    """
    if mode == "strict":
        return _reduce_strict(reactions, set(species_init))
    elif mode == "closure":
        if allowed_elements is None:
            allowed_elements = get_element_set(species_init)
        return _reduce_closure(reactions, species_init, allowed_elements, max_species)
    else:
        raise ValueError(f"Invalid reduction mode: {mode!r}. Use 'strict' or 'closure'.")


def _reduce_strict(
    reactions: list[UmistReaction],
    species_whitelist: set[str],
) -> tuple[list[UmistReaction], list[str]]:
    """Strict whitelist reduction.
    
    Keep only reactions where all chemical species are in the whitelist.
    
    Parameters
    ----------
    reactions : list[UmistReaction]
        Full reaction list
    species_whitelist : set[str]
        Allowed species
        
    Returns
    -------
    tuple[list[UmistReaction], list[str]]
        (filtered_reactions, species_in_network)
    """
    filtered = []
    species_used = set()
    
    for rec in reactions:
        rec_species = reaction_species(rec)
        if rec_species.issubset(species_whitelist):
            filtered.append(rec)
            species_used.update(rec_species)
    
    species_list = sorted(species_used)
    
    return filtered, species_list


def _reduce_closure(
    reactions: list[UmistReaction],
    species_init: list[str],
    allowed_elements: set[str],
    max_species: int,
) -> tuple[list[UmistReaction], list[str]]:
    """Bounded closure reduction.
    
    Start from initial species and iteratively add species from connected reactions,
    subject to element and size constraints.
    
    Parameters
    ----------
    reactions : list[UmistReaction]
        Full reaction list
    species_init : list[str]
        Initial species seed
    allowed_elements : set[str]
        Allowed element symbols
    max_species : int
        Maximum species count
        
    Returns
    -------
    tuple[list[UmistReaction], list[str]]
        (filtered_reactions, final_species_list)
        
    Raises
    ------
    ValueError
        If initial species exceed max_species
    """
    if len(species_init) > max_species:
        raise ValueError(
            f"Initial species count {len(species_init)} exceeds max_species {max_species}"
        )
    
    current_species = set(species_init)
    
    converged = False
    iteration = 0
    max_iterations = 100
    
    while not converged and iteration < max_iterations:
        iteration += 1
        new_species = set()
        
        for rec in reactions:
            rec_species = reaction_species(rec)
            
            if not rec_species:
                continue
            
            if rec_species.issubset(current_species):
                continue
            
            if rec_species & current_species:
                candidate_species = rec_species - current_species
                
                for s in candidate_species:
                    if len(current_species) + len(new_species) >= max_species:
                        break
                    
                    _, elem_dict = parse_species_formula(s)
                    if set(elem_dict.keys()).issubset(allowed_elements):
                        new_species.add(s)
        
        if not new_species:
            converged = True
        else:
            if len(current_species) + len(new_species) > max_species:
                new_species_sorted = sorted(new_species)
                n_to_add = max_species - len(current_species)
                new_species = set(new_species_sorted[:n_to_add])
            
            current_species.update(new_species)
    
    filtered, species_list = _reduce_strict(reactions, current_species)
    
    return filtered, species_list
