"""Simple I/O utilities."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_arrays(file_path: str | Path, *arrays: FloatArray) -> None:
    """Save numeric arrays as stacked columns."""
    data = np.column_stack(arrays)
    np.savetxt(Path(file_path), data, delimiter=",")
