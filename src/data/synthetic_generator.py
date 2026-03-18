"""Synthetic SHG dataset generation helpers."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from src.physics.shg_model import SHGParams, simulate_shg
from src.utils.io import ensure_directory

FloatArray = npt.NDArray[np.float64]
NormalizationMode = Literal["none", "global", "separate"]

DEFAULT_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "n21w": (3.0, 7.0),
    "k21w": (0.0, 1.0),
    "n22w": (2.0, 5.0),
    "k22w": (0.0, 1.0),
}
PARAMETER_NAMES: tuple[str, str, str, str] = ("n21w", "k21w", "n22w", "k22w")


@dataclass
class SyntheticSHGDataset:
    """Synthetic SHG dataset ready for ML pipelines."""

    d_nm: FloatArray
    i3: FloatArray
    i1: FloatArray
    curves: FloatArray
    parameters: FloatArray
    lambda_m: float
    bounds: dict[str, tuple[float, float]]
    normalization: NormalizationMode
    seed: Optional[int]


def _validate_bounds(bounds: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Validate and normalize the parameter bounds."""
    validated_bounds: dict[str, tuple[float, float]] = {}
    for parameter_name in PARAMETER_NAMES:
        if parameter_name not in bounds:
            raise ValueError(f"Missing bounds for parameter '{parameter_name}'.")

        lower, upper = bounds[parameter_name]
        lower = float(lower)
        upper = float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid bounds for parameter '{parameter_name}'.")
        validated_bounds[parameter_name] = (lower, upper)
    return validated_bounds


def _validate_d_nm(d_nm: FloatArray) -> FloatArray:
    """Validate the shared thickness vector."""
    thickness_nm = np.asarray(d_nm, dtype=np.float64)
    if thickness_nm.ndim != 1 or thickness_nm.size == 0:
        raise ValueError("d_nm must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(thickness_nm)):
        raise ValueError("d_nm contains NaN or inf.")
    if np.any(thickness_nm < 0.0):
        raise ValueError("d_nm must be non-negative.")
    return thickness_nm


def _sample_params(
    rng: np.random.Generator,
    lambda_m: float,
    bounds: dict[str, tuple[float, float]],
) -> tuple[SHGParams, FloatArray]:
    """Sample one physically-parameterized SHG configuration."""
    parameter_vector = np.array(
        [rng.uniform(*bounds[parameter_name]) for parameter_name in PARAMETER_NAMES],
        dtype=np.float64,
    )
    params = SHGParams(
        lambda_m=lambda_m,
        n21w=complex(parameter_vector[0], parameter_vector[1]),
        n22w=complex(parameter_vector[2], parameter_vector[3]),
    )
    return params, parameter_vector


def _normalize_pair(i3: FloatArray, i1: FloatArray, mode: NormalizationMode) -> tuple[FloatArray, FloatArray]:
    """Normalize one pair of SHG curves before saving."""
    if mode == "none":
        return i3, i1

    if mode == "global":
        scale = float(max(np.max(i3), np.max(i1)))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Global normalization requires strictly positive finite curves.")
        return i3 / scale, i1 / scale

    if mode == "separate":
        i3_scale = float(np.max(i3))
        i1_scale = float(np.max(i1))
        if not np.isfinite(i3_scale) or i3_scale <= 0.0:
            raise ValueError("Separate normalization requires strictly positive finite i3 curves.")
        if not np.isfinite(i1_scale) or i1_scale <= 0.0:
            raise ValueError("Separate normalization requires strictly positive finite i1 curves.")
        return i3 / i3_scale, i1 / i1_scale

    raise ValueError(f"Unknown normalization mode: {mode}")


def _print_progress(current: int, total: int) -> None:
    """Print a simple progress indicator."""
    progress_percent = (100.0 * current) / total
    print(f"\rGenerating dataset: {current}/{total} ({progress_percent:5.1f}%)", end="", flush=True)
    if current == total:
        print()


def generate_synthetic_shg(params: SHGParams, d_nm: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate one synthetic SHG sample from explicit physical parameters."""
    thickness_nm = _validate_d_nm(d_nm)
    i3, i1 = simulate_shg(params, thickness_nm)
    return thickness_nm, i3, i1


def generate_synthetic_dataset(
    num_samples: int,
    d_nm: FloatArray,
    lambda_m: float,
    bounds: Optional[dict[str, tuple[float, float]]] = None,
    seed: Optional[int] = None,
    normalization: NormalizationMode = "none",
    show_progress: bool = True,
    max_attempts: Optional[int] = None,
) -> SyntheticSHGDataset:
    """Generate many SHG simulations for ML training."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not np.isfinite(lambda_m) or lambda_m <= 0.0:
        raise ValueError("lambda_m must be positive and finite.")

    thickness_nm = _validate_d_nm(d_nm)
    parameter_bounds = _validate_bounds(bounds or DEFAULT_PARAMETER_BOUNDS)
    rng = np.random.default_rng(seed)

    n_points = thickness_nm.size
    i3_dataset = np.zeros((num_samples, n_points), dtype=np.float64)
    i1_dataset = np.zeros((num_samples, n_points), dtype=np.float64)
    parameter_dataset = np.zeros((num_samples, len(PARAMETER_NAMES)), dtype=np.float64)

    maximum_attempts = max_attempts if max_attempts is not None else num_samples * 20
    sample_index = 0
    attempt_count = 0

    while sample_index < num_samples:
        if attempt_count >= maximum_attempts:
            raise RuntimeError("Could not generate the requested number of stable SHG samples.")

        params, parameter_vector = _sample_params(rng, lambda_m, parameter_bounds)
        attempt_count += 1

        try:
            _, i3_sample, i1_sample = generate_synthetic_shg(params, thickness_nm)
            i3_sample, i1_sample = _normalize_pair(i3_sample, i1_sample, normalization)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            continue

        i3_dataset[sample_index] = i3_sample
        i1_dataset[sample_index] = i1_sample
        parameter_dataset[sample_index] = parameter_vector
        sample_index += 1

        if show_progress:
            _print_progress(sample_index, num_samples)

    curves = np.stack((i3_dataset, i1_dataset), axis=1)
    return SyntheticSHGDataset(
        d_nm=thickness_nm,
        i3=i3_dataset,
        i1=i1_dataset,
        curves=curves,
        parameters=parameter_dataset,
        lambda_m=float(lambda_m),
        bounds=parameter_bounds,
        normalization=normalization,
        seed=seed,
    )


def save_synthetic_dataset(dataset: SyntheticSHGDataset, file_path: str | Path) -> Path:
    """Save a synthetic SHG dataset as a compressed NPZ archive."""
    output_path = Path(file_path)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")

    ensure_directory(output_path.parent)

    metadata = {
        "lambda_m": dataset.lambda_m,
        "bounds": {name: list(dataset.bounds[name]) for name in PARAMETER_NAMES},
        "parameter_names": list(PARAMETER_NAMES),
        "normalization": dataset.normalization,
        "seed": dataset.seed,
        "num_samples": int(dataset.parameters.shape[0]),
        "num_points": int(dataset.d_nm.size),
    }

    lower_bounds = np.array([dataset.bounds[name][0] for name in PARAMETER_NAMES], dtype=np.float64)
    upper_bounds = np.array([dataset.bounds[name][1] for name in PARAMETER_NAMES], dtype=np.float64)

    np.savez_compressed(
        output_path,
        d_nm=dataset.d_nm,
        i3=dataset.i3,
        i1=dataset.i1,
        curves=dataset.curves,
        parameters=dataset.parameters,
        lambda_m=np.array(dataset.lambda_m, dtype=np.float64),
        bounds_low=lower_bounds,
        bounds_high=upper_bounds,
        parameter_names=np.array(PARAMETER_NAMES),
        normalization=np.array(dataset.normalization),
        seed=np.array(-1 if dataset.seed is None else dataset.seed, dtype=np.int64),
        metadata_json=np.array(json.dumps(metadata)),
    )
    return output_path
