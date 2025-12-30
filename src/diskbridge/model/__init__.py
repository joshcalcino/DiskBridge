from pathlib import Path
from typing import Optional, Union

from .mesh import Mesh
from .field import Field
from .core import Model, puff_up_model
from .clipping import ClipIndexer, compute_clip_indexer
from .masking import set_mask_from_joos_disk

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
    
    Parameters
    ----------
    path : str or Path
        Path to simulation data directory or saved model file
    reader : str, optional
        Reader type (default: "fargo"). Ignored for saved models.
    file_n : int, optional
        File number to load (default: 0). Ignored for saved models.
    file_units : str, optional
        Unit system: 'code', 'cgs', or 'kms' (default: 'code'). Ignored for saved models.
    length_scale : float, optional
        Length units multiplier. Ignored for saved models.
    mass_scale : float, optional
        Mass units multiplier. Ignored for saved models.
    
    Returns
    -------
    Model
        The loaded model instance
    """
    return Model.load(
        path=path,
        reader=reader,
        file_n=file_n,
        file_units=file_units,
        length_scale=length_scale,
        mass_scale=mass_scale,
    )


__all__ = [
    "Mesh",
    "Field",
    "Model",
    "ClipIndexer",
    "compute_clip_indexer",
    "load_model",
    "puff_up_model",
    "set_mask_from_joos_disk",
]
