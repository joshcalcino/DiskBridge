from pathlib import Path
from typing import Union

from .mesh import Mesh
from .field import Field
from .model import Model, puff_up_model


def load_model(
    path: Union[str, Path],
    reader: str = "fargo",
    file_n: int = 0,
    file_units: str = "code",
):
    return Model().load_model(path=path, reader=reader, file_n=file_n, file_units=file_units)


__all__ = ["Mesh", "Field", "Model", "load_model", "puff_up_model"]
