"""Dataset containers for ML experiments."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass
class SHGDataset:
    """Container for SHG supervised data."""

    features: FloatArray
    targets: FloatArray


def as_dataset(features: FloatArray, targets: FloatArray) -> SHGDataset:
    """Build a typed SHG dataset."""
    return SHGDataset(
        features=np.asarray(features, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
    )
