from __future__ import annotations


def parse_species_formula(name: str) -> tuple[int, dict[str, int]]:
    """Parse species formula to extract charge and elemental composition.
    
    Based on AstroChemistry.jl's breakdown_species function.
    Parses from right to left handling:
    - Trailing +/- for charge (can be repeated)
    - Element symbols (uppercase + optional lowercase)
    - Digits for multiplicity
    
    Parameters
    ----------
    name : str
        Species name (e.g., 'H2O', 'C+', 'e-', 'CH3OH')
        
    Returns
    -------
    tuple[int, dict[str, int]]
        (charge, element_counts)
        For 'e-' returns (-1, {})
        For 'C+' returns (+1, {'C': 1})
        For 'H2CO' returns (0, {'H': 2, 'C': 1, 'O': 1})
        
    Examples
    --------
    >>> parse_species_formula('e-')
    (-1, {})
    >>> parse_species_formula('H2O')
    (0, {'H': 2, 'O': 1})
    >>> parse_species_formula('C+')
    (1, {'C': 1})
    >>> parse_species_formula('HCO+')
    (1, {'H': 1, 'C': 1, 'O': 1})
    """
    if name == 'e-':
        return -1, {}
    
    charge = 0
    elements = []
    counts = []
    mult = 1
    skip_next = False
    
    for i in range(len(name) - 1, -1, -1):
        if skip_next:
            skip_next = False
            continue
        
        char = name[i]
        
        if char == '+':
            charge += 1
        elif char == '-':
            charge -= 1
        elif char.isalpha():
            if char.islower():
                if i == 0:
                    raise ValueError(f"Invalid species formula '{name}': lowercase at start")
                elem = name[i-1:i+1]
                skip_next = True
            else:
                elem = char
            
            elements.append(elem)
            counts.append(mult)
            mult = 1
        elif char.isdigit():
            mult = int(char)
        else:
            raise ValueError(f"Invalid character '{char}' in species formula '{name}'")
    
    elem_dict = {}
    for elem, count in zip(elements, counts):
        elem_dict[elem] = elem_dict.get(elem, 0) + count
    
    return charge, elem_dict


def charge_of(species: list[str]) -> list[int]:
    """Get charge for each species.
    
    Parameters
    ----------
    species : list[str]
        List of species names
        
    Returns
    -------
    list[int]
        Charge for each species
    """
    return [parse_species_formula(s)[0] for s in species]


def element_counts_of(species: list[str]) -> list[dict[str, int]]:
    """Get element counts for each species.
    
    Parameters
    ----------
    species : list[str]
        List of species names
        
    Returns
    -------
    list[dict[str, int]]
        Element counts for each species
    """
    return [parse_species_formula(s)[1] for s in species]


def element_set(species: list[str]) -> set[str]:
    """Get unique set of elements present in species list.
    
    Parameters
    ----------
    species : list[str]
        List of species names
        
    Returns
    -------
    set[str]
        Set of element symbols
    """
    elements = set()
    for s in species:
        _, elem_dict = parse_species_formula(s)
        elements.update(elem_dict.keys())
    return elements


def validate_element_conservation(
    reactants: tuple[str, str],
    products: tuple[str, str, str, str],
) -> tuple[bool, str]:
    """Check if a reaction conserves elements.
    
    Parameters
    ----------
    reactants : tuple[str, str]
        Reactant species (can contain empty strings)
    products : tuple[str, str, str, str]
        Product species (can contain empty strings)
        
    Returns
    -------
    tuple[bool, str]
        (is_conserved, message)
    """
    R_species = [s for s in reactants if s]
    P_species = [s for s in products if s]
    
    R_elem_total = {}
    for s in R_species:
        _, elem_dict = parse_species_formula(s)
        for elem, count in elem_dict.items():
            R_elem_total[elem] = R_elem_total.get(elem, 0) + count
    
    P_elem_total = {}
    for s in P_species:
        _, elem_dict = parse_species_formula(s)
        for elem, count in elem_dict.items():
            P_elem_total[elem] = P_elem_total.get(elem, 0) + count
    
    if R_elem_total != P_elem_total:
        return False, f"Reactants {R_elem_total} != Products {P_elem_total}"
    
    return True, "OK"
