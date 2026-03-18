"""Loading helpers for SHG datasets."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def load_columns(file_path: str | Path, delimiter: str = ",") -> FloatArray:
    """Load numeric columns from a text file."""
    return np.loadtxt(Path(file_path), delimiter=delimiter, dtype=np.float64)
