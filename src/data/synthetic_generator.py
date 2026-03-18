"""Synthetic data generation helpers."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.physics.shg_model import SHGParams, simulate_shg

FloatArray = npt.NDArray[np.float64]


def generate_synthetic_shg(
    params: SHGParams,
    d_nm: Sequence[float],
    noise_level: float = 0.0,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate synthetic SHG data with optional Gaussian noise."""
    d_array = np.asarray(d_nm, dtype=np.float64)
    i3, i1 = simulate_shg(params, d_array)

    if noise_level <= 0.0:
        return d_array, i3, i1

    rng = np.random.default_rng(seed)
    i3_noisy = i3 + rng.normal(0.0, noise_level, size=i3.shape)
    i1_noisy = i1 + rng.normal(0.0, noise_level, size=i1.shape)
    return d_array, i3_noisy.astype(np.float64), i1_noisy.astype(np.float64)
