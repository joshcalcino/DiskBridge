from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple


SPECIAL_TOKENS = {"PHOTON", "CRP", "CRPHOT"}


class UmistReaction(NamedTuple):
    """Parsed UMIST RATE12 reaction record.
    
    Attributes
    ----------
    rid : int
        Reaction number
    rtype : str
        Reaction type code (e.g., 'AD', 'PH', 'CR', 'CP', 'RA', 'NN', 'DR', 'GR')
    R : tuple[str, str]
        Two reactants (second can be empty string)
    P : tuple[str, str, str, str]
        Up to four products (empty strings allowed)
    alpha : float
        Rate coefficient alpha parameter
    beta : float
        Rate coefficient beta parameter
    gamma : float
        Rate coefficient gamma parameter
    Tlow : float
        Lower temperature limit (K)
    Thigh : float
        Upper temperature limit (K)
    st : str
        Source type (raw)
    acc : str
        Accuracy class (raw)
    """
    rid: int
    rtype: str
    R: tuple[str, str]
    P: tuple[str, str, str, str]
    alpha: float
    beta: float
    gamma: float
    Tlow: float
    Thigh: float
    st: str
    acc: str


def norm_species(s: str) -> str:
    """Normalize a species token.
    
    Strips whitespace and converts special tokens (PHOTON, CRP, CRPHOT)
    to empty strings since they are not chemical species.
    
    Parameters
    ----------
    s : str
        Raw species token
        
    Returns
    -------
    str
        Normalized species (empty string for special tokens)
    """
    s = s.strip()
    return "" if s in SPECIAL_TOKENS else s


def read_rate12(path: str | Path) -> list[UmistReaction]:
    """Read UMIST RATE12 format file.
    
    The RATE12 format is colon-separated with fields:
    num:type:R1:R2:P1:P2:P3:P4:nr:alpha:beta:gamma:Tl:Tu:ST:ACC:refs...
    
    Parameters
    ----------
    path : str or Path
        Path to RATE12.dist.txt file
        
    Returns
    -------
    list[UmistReaction]
        Parsed reactions
        
    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If parsing fails for a line
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"UMIST RATE12 file not found: {path}")
    
    reactions = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=':', quotechar='"')
        
        for line_num, fields in enumerate(reader, start=1):
            if not fields or (len(fields) == 1 and not fields[0].strip()):
                continue
            
            if len(fields) < 16:
                raise ValueError(
                    f"Line {line_num}: Expected at least 16 fields, got {len(fields)}"
                )
            
            try:
                rid = int(fields[0])
                rtype = fields[1].strip()
                
                R1 = norm_species(fields[2])
                R2 = norm_species(fields[3])
                
                P1 = norm_species(fields[4])
                P2 = norm_species(fields[5])
                P3 = norm_species(fields[6])
                P4 = norm_species(fields[7])
                
                alpha = float(fields[9])
                beta = float(fields[10])
                gamma = float(fields[11])
                
                Tlow = float(fields[12])
                Thigh = float(fields[13])
                
                st = fields[14].strip()
                acc = fields[15].strip()
                
                reactions.append(UmistReaction(
                    rid=rid,
                    rtype=rtype,
                    R=(R1, R2),
                    P=(P1, P2, P3, P4),
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    Tlow=Tlow,
                    Thigh=Thigh,
                    st=st,
                    acc=acc,
                ))
                
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Line {line_num}: Failed to parse reaction {fields[0] if fields else '?'}: {e}"
                ) from e
    
    return reactions


def reaction_species(rec: UmistReaction) -> set[str]:
    """Get all chemical species in a reaction (excluding specials and empty).
    
    Parameters
    ----------
    rec : UmistReaction
        Reaction record
        
    Returns
    -------
    set[str]
        Set of species names
    """
    return {s for s in rec.R + rec.P if s}
