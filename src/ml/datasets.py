"""Dataset containers, splitting and masking helpers for SHG ML experiments."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from src.data.synthetic_generator import SyntheticSHGDataset
from src.utils.io import ensure_directory

FloatArray = npt.NDArray[np.float64]
MaskArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

MASK_BOTH = np.array([1.0, 1.0], dtype=np.float64)
MASK_I3_ONLY = np.array([1.0, 0.0], dtype=np.float64)
MASK_I1_ONLY = np.array([0.0, 1.0], dtype=np.float64)


@dataclass
class SHGDataset:
    """Dataset with SHG curves and parameter targets."""

    d_nm: FloatArray
    lambda_m: float
    i3: FloatArray
    i1: FloatArray
    targets: FloatArray

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""
        return int(self.targets.shape[0])

    @property
    def curve_length(self) -> int:
        """Return the number of points per SHG curve."""
        return int(self.d_nm.size)

    @property
    def input_dim(self) -> int:
        """Return the MLP input size for masked concatenated curves."""
        return int(2 * self.curve_length + 2)

    @property
    def output_dim(self) -> int:
        """Return the number of regression targets."""
        return int(self.targets.shape[1])


@dataclass
class DatasetSplit:
    """Train/validation/test split for an SHG dataset."""

    train: SHGDataset
    validation: SHGDataset
    test: SHGDataset
    train_indices: IntArray
    validation_indices: IntArray
    test_indices: IntArray

    def summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable description of the split."""
        return {
            "train_count": int(self.train_indices.size),
            "validation_count": int(self.validation_indices.size),
            "test_count": int(self.test_indices.size),
            "train_indices": self.train_indices.tolist(),
            "validation_indices": self.validation_indices.tolist(),
            "test_indices": self.test_indices.tolist(),
        }


def from_synthetic_dataset(dataset: SyntheticSHGDataset) -> SHGDataset:
    """Convert a saved/generated synthetic dataset into the ML dataset format."""
    return SHGDataset(
        d_nm=np.asarray(dataset.d_nm, dtype=np.float64),
        lambda_m=float(dataset.lambda_m),
        i3=np.asarray(dataset.i3, dtype=np.float64),
        i1=np.asarray(dataset.i1, dtype=np.float64),
        targets=np.asarray(dataset.parameters, dtype=np.float64),
    )


def full_observation_masks(num_samples: int) -> MaskArray:
    """Return masks indicating that both curves are present."""
    return np.tile(MASK_BOTH, (num_samples, 1))


def single_channel_masks(num_samples: int, keep: str) -> MaskArray:
    """Return masks for one-channel-only evaluation scenarios."""
    if keep == "i3":
        return np.tile(MASK_I3_ONLY, (num_samples, 1))
    if keep == "i1":
        return np.tile(MASK_I1_ONLY, (num_samples, 1))
    raise ValueError("keep must be 'i3' or 'i1'.")


def sample_augmentation_masks(num_samples: int, rng: np.random.Generator) -> MaskArray:
    """Sample random masks for training-time missing-channel augmentation."""
    scenario_index = rng.integers(0, 3, size=num_samples)
    masks = np.zeros((num_samples, 2), dtype=np.float64)
    masks[scenario_index == 0] = MASK_BOTH
    masks[scenario_index == 1] = MASK_I3_ONLY
    masks[scenario_index == 2] = MASK_I1_ONLY
    return masks


def build_input_features(i3: FloatArray, i1: FloatArray, masks: MaskArray) -> FloatArray:
    """Build masked MLP inputs as [i3, i1, mask]."""
    i3_array = np.asarray(i3, dtype=np.float64)
    i1_array = np.asarray(i1, dtype=np.float64)
    mask_array = np.asarray(masks, dtype=np.float64)

    if i3_array.shape != i1_array.shape:
        raise ValueError("i3 and i1 must have the same shape.")
    if i3_array.ndim != 2:
        raise ValueError("i3 and i1 must be 2D arrays with shape (samples, points).")
    if mask_array.shape != (i3_array.shape[0], 2):
        raise ValueError("masks must have shape (samples, 2).")

    masked_i3 = i3_array * mask_array[:, [0]]
    masked_i1 = i1_array * mask_array[:, [1]]
    return np.concatenate((masked_i3, masked_i1, mask_array), axis=1)


def subset_dataset(dataset: SHGDataset, indices: np.ndarray) -> SHGDataset:
    """Create a dataset view restricted to the selected sample indices."""
    selected_indices = np.asarray(indices, dtype=np.int64)
    return SHGDataset(
        d_nm=np.asarray(dataset.d_nm, dtype=np.float64),
        lambda_m=float(dataset.lambda_m),
        i3=np.asarray(dataset.i3[selected_indices], dtype=np.float64),
        i1=np.asarray(dataset.i1[selected_indices], dtype=np.float64),
        targets=np.asarray(dataset.targets[selected_indices], dtype=np.float64),
    )


def _normalized_split_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> FloatArray:
    """Validate and normalize train/validation/test fractions."""
    fractions = np.array([train_fraction, validation_fraction, test_fraction], dtype=np.float64)
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0.0):
        raise ValueError("Split fractions must be finite and strictly positive.")
    return fractions / np.sum(fractions)


def _split_counts(num_samples: int, fractions: FloatArray) -> tuple[int, int, int]:
    """Compute integer split sizes while keeping all splits non-empty."""
    if num_samples < 3:
        raise ValueError("At least 3 samples are required for train/validation/test split.")

    raw_counts = fractions * num_samples
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(num_samples - np.sum(counts))
    order = np.argsort(-(raw_counts - counts))
    for split_index in order[:remainder]:
        counts[split_index] += 1

    for split_index in range(counts.size):
        if counts[split_index] > 0:
            continue
        donor_index = int(np.argmax(counts))
        if counts[donor_index] <= 1:
            raise ValueError("Could not build non-empty train/validation/test splits.")
        counts[donor_index] -= 1
        counts[split_index] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def split_dataset(
    dataset: SHGDataset,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> DatasetSplit:
    """Split an SHG dataset into train, validation and test subsets."""
    fractions = _normalized_split_fractions(train_fraction, validation_fraction, test_fraction)
    train_count, validation_count, test_count = _split_counts(dataset.num_samples, fractions)

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(dataset.num_samples).astype(np.int64)
    else:
        indices = np.arange(dataset.num_samples, dtype=np.int64)

    train_end = train_count
    validation_end = train_count + validation_count
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end : validation_end + test_count]

    return DatasetSplit(
        train=subset_dataset(dataset, train_indices),
        validation=subset_dataset(dataset, validation_indices),
        test=subset_dataset(dataset, test_indices),
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def save_dataset_split(split: DatasetSplit, file_path: str | Path) -> Path:
    """Save the train/validation/test split indices as JSON."""
    output_path = Path(file_path)
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(split.summary_dict(), indent=2), encoding="utf-8")
    return output_path
