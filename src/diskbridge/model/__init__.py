from pathlib import Path
from typing import Optional, Union

from .mesh import Mesh
from .field import Field
from .model import Model, puff_up_model, extend_disk_inwards
from .clipping import ClipIndexer, compute_clip_indexer

def load_model(
    path: Union[str, Path],
    reader: str = "fargo",
    file_n: int = 0,
    file_units: str = "code",
    length_scale: Optional[float] = None,
    mass_scale: Optional[float] = None,
):
    """
    Load a hydro simulation snapshot or a saved DiskBridge model.
    
    This function automatically detects whether `path` points to:
    1. A saved DiskBridge model file (.pkl) - loads using Model.load()
    2. A simulation data directory - loads using Model.load_model()
    
    Parameters
    ----------
    path : str or Path
        Path to either:
        - A saved DiskBridge model file (.pkl)
        - A simulation data directory
    reader : str, optional
        Reader type for simulation data (default: "fargo")
        Ignored when loading a saved model file
    file_n : int, optional
        File number to load for simulation data (default: 0)
        Ignored when loading a saved model file
    file_units : str, optional
        Unit system of the files: 'code', 'cgs', or 'kms' (default: 'code')
        Ignored when loading a saved model file
    length_scale : float, optional
        Dimensionless multiplier for length units. For code units (1 au),
        this rescales by this factor (e.g., 10 means 1 code_length = 10 au).
        Time is also rescaled following Keplerian dynamics.
        Ignored when loading a saved model file
    mass_scale : float, optional
        Dimensionless multiplier for mass units. For code units (1 M_sun),
        this rescales by this factor (e.g., 2 means 1 code_mass = 2 M_sun).
        Velocities are rescaled as V -> V*sqrt(mass_scale/length_scale).
        Ignored when loading a saved model file
    
    Returns
    -------
    Model
        The loaded model instance
        
    Notes
    -----
    Code units are already defined as 1 au, 1 M_sun, and code_time = sqrt(au^3/(G*M_sun)).
    Rescaling follows Keplerian dynamics where T^2 ~ L^3/M:
    - Lengths scale by length_scale
    - Masses scale by mass_scale
    - Times scale by sqrt(length_scale^3/mass_scale)
    - Velocities scale by sqrt(mass_scale/length_scale)
    - Densities scale by mass_scale/length_scale^3
        
    Examples
    --------
    >>> import diskbridge
    >>> # Load simulation with rescaling: 1 code unit = 10 au, 2 M_sun
    >>> model = diskbridge.load_model('data/', file_units='code', 
    ...                               length_scale=10.0, 
    ...                               mass_scale=2.0)
    >>> 
    >>> # Save the model for later
    >>> model.save('my_model.pkl')
    >>> 
    >>> # Load the saved model
    >>> model2 = diskbridge.load_model('my_model.pkl')
    """
    path = Path(path)
    
    # Check if path is a file (saved model) or directory (simulation data)
    if path.is_file():
        # Load saved model file
        if not path.suffix == '.pkl':
            # Allow loading but warn if not .pkl extension
            from diskbridge._logging import logger
            logger.warning(
                f"Loading file '{path}' which does not have .pkl extension. "
                "Expected a pickled DiskBridge model."
            )
        return Model.load(path)
    else:
        # Load simulation data directory
        return Model().load_model(
            path=path, 
            reader=reader, 
            file_n=file_n, 
            file_units=file_units,
            length_scale=length_scale,
            mass_scale=mass_scale
        )


__all__ = [
    "Mesh",
    "Field",
    "Model",
    "ClipIndexer",
    "compute_clip_indexer",
    "load_model",
    "puff_up_model",
    "extend_disk_inwards",
]
