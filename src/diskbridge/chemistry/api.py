from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from pathlib import Path

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.registry import REGISTRY
from diskbridge.chemistry.io import write_many


def run_chemistry(
    rad: 'RadModel',
    model: str,
    config: Optional[dict] = None,
    write: bool = False,
    output_dir: Optional[Path] = None,
) -> ChemistryResult:
    """Run chemistry computation with the specified model.
    
    This is the main entry point for all chemistry workflows. It:
    1. Ensures required RAD fields exist (temperature, nH, chi via RadModel methods)
    2. Calls the model callable from the registry
    3. Optionally writes outputs
    4. Returns a ChemistryResult
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper (provides ensure_temperature, ensure_nH, ensure_chi)
    model : str
        Chemistry model name (must be in REGISTRY)
    config : dict, optional
        Model-specific configuration parameters (default: empty dict)
    write : bool, optional
        Whether to write number density outputs (default: False)
    output_dir : Path, optional
        Output directory for writing (default: rad.model_dir)
        
    Returns
    -------
    ChemistryResult
        Result object containing abundances, number_densities, fields, and metadata
        
    Raises
    ------
    ValueError
        If model name is not found in registry
        
    Examples
    --------
    >>> res = run_chemistry(
    ...     rad,
    ...     model="co_two_phase_steady",
    ...     config=dict(Xco_tot=1e-4, nside=8, b_kms=0.3),
    ...     write=True,
    ... )
    >>> nco = res.number_densities["co"]
    """
    if config is None:
        config = {}
    
    model_lower = model.lower()
    
    if model_lower not in REGISTRY:
        available = ', '.join(REGISTRY.keys())
        raise ValueError(
            f"Unknown chemistry model: {model!r}. "
            f"Available models: {available}"
        )
    
    model_fn = REGISTRY[model_lower]
    result = model_fn(rad, config)
    
    if write and result.number_densities:
        write_many(rad, result.number_densities, output_dir)
    
    return result
