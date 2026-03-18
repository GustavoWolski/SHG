"""Loading helpers for SHG datasets."""

from pathlib import Path
import json

import numpy as np
import numpy.typing as npt

from src.data.synthetic_generator import SyntheticSHGDataset

FloatArray = npt.NDArray[np.float64]


def load_columns(file_path: str | Path, delimiter: str = ",") -> FloatArray:
    """Load numeric columns from a text file."""
    return np.loadtxt(Path(file_path), delimiter=delimiter, dtype=np.float64)


def load_synthetic_dataset(file_path: str | Path) -> SyntheticSHGDataset:
    """Load a synthetic SHG dataset saved as NPZ."""
    with np.load(Path(file_path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        seed_value = int(data["seed"].item())
        return SyntheticSHGDataset(
            d_nm=np.asarray(data["d_nm"], dtype=np.float64),
            i3=np.asarray(data["i3"], dtype=np.float64),
            i1=np.asarray(data["i1"], dtype=np.float64),
            curves=np.asarray(data["curves"], dtype=np.float64),
            parameters=np.asarray(data["parameters"], dtype=np.float64),
            lambda_m=float(data["lambda_m"].item()),
            bounds={name: tuple(values) for name, values in metadata["bounds"].items()},
            normalization=str(data["normalization"].item()),
            seed=None if seed_value < 0 else seed_value,
        )
