from pathlib import Path
from typing import Optional, Union

from .mesh import Mesh
from .field import Field
from .model import Model, puff_up_model


def load_model(
    path: Union[str, Path],
    reader: str = "fargo",
    file_n: int = 0,
    file_units: str = "code",
    length_scale: Optional[float] = None,
    mass_scale: Optional[float] = None,
):
    """
    Load a hydro simulation snapshot into a Model.
    
    Parameters
    ----------
    path : str or Path
        Path to the simulation data directory
    reader : str, optional
        Reader type (default: "fargo")
    file_n : int, optional
        File number to load (default: 0)
    file_units : str, optional
        Unit system of the files: 'code', 'cgs', or 'kms' (default: 'code')
    length_scale : float, optional
        Dimensionless multiplier for length units. For code units (1 au),
        this rescales by this factor (e.g., 10 means 1 code_length = 10 au).
        Time is also rescaled following Keplerian dynamics.
    mass_scale : float, optional
        Dimensionless multiplier for mass units. For code units (1 M_sun),
        this rescales by this factor (e.g., 2 means 1 code_mass = 2 M_sun).
        Velocities are rescaled as V -> V*sqrt(mass_scale/length_scale).
    
    Returns
    -------
    Model
        The loaded model instance
        
    Notes
    -----
    Code units are already defined as 1 au, 1 M_sun, and code_time = sqrt(au³/(G*M_sun)).
    Rescaling follows Keplerian dynamics where T² ∝ L³/M:
    - Lengths scale by length_scale
    - Masses scale by mass_scale
    - Times scale by sqrt(length_scale³/mass_scale)
    - Velocities scale by sqrt(mass_scale/length_scale)
    - Densities scale by mass_scale/length_scale³
        
    Examples
    --------
    >>> import diskbridge
    >>> # Load model with rescaling: 1 code unit = 10 au, 2 M_sun
    >>> model = diskbridge.load_model('data/', file_units='code', 
    ...                               length_scale=10.0, 
    ...                               mass_scale=2.0)
    """
    return Model().load_model(
        path=path, 
        reader=reader, 
        file_n=file_n, 
        file_units=file_units,
        length_scale=length_scale,
        mass_scale=mass_scale
    )


__all__ = ["Mesh", "Field", "Model", "load_model", "puff_up_model"]
