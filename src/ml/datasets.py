"""Dataset containers and masking helpers for SHG ML experiments."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.data.synthetic_generator import SyntheticSHGDataset

FloatArray = npt.NDArray[np.float64]
MaskArray = npt.NDArray[np.float64]

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
