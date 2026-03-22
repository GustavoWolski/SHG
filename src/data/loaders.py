"""Loading helpers for SHG datasets."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import numpy.typing as npt

from src.data.synthetic_generator import SyntheticSHGDataset

FloatArray = npt.NDArray[np.float64]


@dataclass
class ExperimentalSHGData:
    """Experimental SHG curves loaded from an external text file."""

    d_nm: FloatArray
    i3: FloatArray
    i1: FloatArray
    i3_mask: npt.NDArray[np.bool_]
    i1_mask: npt.NDArray[np.bool_]


def load_columns(file_path: str | Path, delimiter: str = ",", skiprows: int = 0) -> FloatArray:
    """Load numeric columns from a text file."""
    return np.loadtxt(Path(file_path), delimiter=delimiter, dtype=np.float64, skiprows=skiprows)


def load_experimental_shg_data(
    file_path: str | Path,
    delimiter: str = ",",
    skiprows: int = 0,
) -> ExperimentalSHGData:
    """Load experimental SHG data with optional missing i3/i1 values."""
    columns = np.genfromtxt(
        Path(file_path),
        delimiter=delimiter,
        dtype=np.float64,
        skip_header=skiprows,
        filling_values=np.nan,
    )
    columns = np.asarray(np.atleast_2d(columns), dtype=np.float64)
    if columns.ndim != 2 or columns.shape[1] != 3:
        raise ValueError("Experimental data file must contain exactly 3 numeric columns: d_nm, i3, i1.")

    d_nm = np.asarray(columns[:, 0], dtype=np.float64)
    i3 = np.asarray(columns[:, 1], dtype=np.float64)
    i1 = np.asarray(columns[:, 2], dtype=np.float64)
    i3_mask = np.isfinite(i3)
    i1_mask = np.isfinite(i1)

    if not np.all(np.isfinite(d_nm)):
        raise ValueError("Experimental thickness values d_nm must be finite.")
    if not np.any(i3_mask | i1_mask):
        raise ValueError("Experimental data must contain at least one finite i3 or i1 value.")

    return ExperimentalSHGData(
        d_nm=d_nm,
        i3=i3,
        i1=i1,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
    )


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
