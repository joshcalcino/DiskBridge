from __future__ import annotations

from pathlib import Path
from typing import Optional

from diskbridge._logging import logger
from diskbridge.chemistry.astrochemistry.umist_io import read_rate12
from diskbridge.chemistry.astrochemistry.reduce import reduce_network
from diskbridge.chemistry.astrochemistry.compile import (
    compile_network,
    sort_species_deterministic,
    validate_compiled_network,
)
from diskbridge.chemistry.astrochemistry.cache import save_network_npz, compute_file_hash
from diskbridge.chemistry.astrochemistry.species import validate_element_conservation


def build_and_cache_network(
    umist_path: str | Path,
    out_path: str | Path,
    species: list[str],
    reduce_mode: str = "strict",
    allowed_elements: Optional[set[str]] = None,
    max_species: int = 64,
    check_conservation: bool = True,
) -> str:
    """Build and cache a reduced UMIST network.
    
    This is the main entry point for compiling UMIST networks.
    
    Parameters
    ----------
    umist_path : str or Path
        Path to UMIST RATE12.dist.txt file
    out_path : str or Path
        Output .npz cache file path
    species : list[str]
        Initial species list for network reduction
    reduce_mode : str, optional
        Reduction mode ('strict' or 'closure'), default 'strict'
    allowed_elements : set[str], optional
        Allowed elements for closure mode. If None, derived from species.
    max_species : int, optional
        Maximum species count (for closure mode), default 64
    check_conservation : bool, optional
        Whether to validate element conservation, default True
        
    Returns
    -------
    str
        Path to created cache file
        
    Raises
    ------
    ValueError
        If network validation fails
        
    Examples
    --------
    >>> build_and_cache_network(
    ...     'RATE12.dist.txt',
    ...     'umist_CHO.npz',
    ...     species=['H', 'H2', 'C', 'C+', 'O', 'CO', 'e-'],
    ...     reduce_mode='strict'
    ... )
    """
    umist_path = Path(umist_path)
    out_path = Path(out_path)
    
    logger.info(f"Reading UMIST RATE12 file: {umist_path}")
    all_reactions = read_rate12(umist_path)
    logger.info(f"Loaded {len(all_reactions)} reactions from UMIST")
    
    umist_hash = compute_file_hash(umist_path)
    logger.info(f"UMIST file SHA256: {umist_hash[:16]}...")
    
    logger.info(f"Reducing network (mode={reduce_mode}, initial species={len(species)})")
    filtered_reactions, final_species = reduce_network(
        all_reactions,
        species,
        mode=reduce_mode,
        max_species=max_species,
        allowed_elements=allowed_elements,
    )
    logger.info(f"Reduced to {len(filtered_reactions)} reactions, {len(final_species)} species")
    
    if check_conservation:
        logger.info("Checking element conservation...")
        n_violations = 0
        for rec in filtered_reactions[:100]:
            conserved, msg = validate_element_conservation(rec.R, rec.P)
            if not conserved:
                n_violations += 1
                if n_violations <= 5:
                    logger.warning(
                        f"Reaction {rec.rid} ({rec.rtype}): {rec.R} -> {rec.P}: {msg}"
                    )
        
        if n_violations > 0:
            logger.warning(
                f"Found {n_violations} reactions with element conservation issues "
                f"(checked first 100)"
            )
    
    final_species_sorted = sort_species_deterministic(final_species)
    logger.info(f"Species (sorted): {final_species_sorted[:10]}...")
    
    logger.info("Compiling network to arrays...")
    network = compile_network(filtered_reactions, final_species_sorted)
    
    logger.info("Validating compiled network...")
    is_valid, errors = validate_compiled_network(network)
    if not is_valid:
        for err in errors:
            logger.error(f"Validation error: {err}")
        raise ValueError("Network validation failed")
    
    logger.info("Validation passed")
    
    meta = {
        'umist_path': str(umist_path.absolute()),
        'umist_sha256': umist_hash,
        'species_init': species,
        'reduce_mode': reduce_mode,
        'max_species': max_species,
        'n_reactions': len(filtered_reactions),
        'n_species': len(final_species_sorted),
    }
    
    if allowed_elements is not None:
        meta['allowed_elements'] = sorted(allowed_elements)
    
    try:
        import diskbridge
        meta['diskbridge_version'] = diskbridge.__version__
    except:
        pass
    
    logger.info(f"Saving network to {out_path}...")
    save_network_npz(out_path, network, meta)
    logger.info(f"Network cache created: {out_path}")
    
    return str(out_path.absolute())
