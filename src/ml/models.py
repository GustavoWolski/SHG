"""Simple MLP definitions for SHG inverse regression."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from src.utils.io import ensure_directory

FloatArray = npt.NDArray[np.float64]


@dataclass
class ModelConfig:
    """Configuration for a masked-input SHG MLP."""

    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...] = (256, 128)


@dataclass
class MLPRegressor:
    """Simple MLP regressor that consumes [i3, i1, mask] features."""

    weights: list[FloatArray]
    biases: list[FloatArray]
    input_mean: FloatArray
    input_std: FloatArray
    target_mean: FloatArray
    target_std: FloatArray
    config: ModelConfig

    def predict(self, features: FloatArray) -> FloatArray:
        """Predict SHG physical parameters from masked features."""
        feature_array = np.asarray(features, dtype=np.float64)
        normalized_features = (feature_array - self.input_mean) / self.input_std

        activations = normalized_features
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activations = activations @ weight + bias
            if layer_index < len(self.weights) - 1:
                activations = np.maximum(activations, 0.0)

        return activations * self.target_std + self.target_mean


def build_model(config: ModelConfig, seed: Optional[int] = None) -> MLPRegressor:
    """Build a simple MLP with He-style initialization."""
    rng = np.random.default_rng(seed)
    layer_dims = (config.input_dim,) + config.hidden_dims + (config.output_dim,)

    weights: list[FloatArray] = []
    biases: list[FloatArray] = []
    for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:]):
        weight_scale = np.sqrt(2.0 / input_dim)
        weights.append(rng.normal(0.0, weight_scale, size=(input_dim, output_dim)).astype(np.float64))
        biases.append(np.zeros(output_dim, dtype=np.float64))

    return MLPRegressor(
        weights=weights,
        biases=biases,
        input_mean=np.zeros(config.input_dim, dtype=np.float64),
        input_std=np.ones(config.input_dim, dtype=np.float64),
        target_mean=np.zeros(config.output_dim, dtype=np.float64),
        target_std=np.ones(config.output_dim, dtype=np.float64),
        config=config,
    )


def save_model(model: MLPRegressor, file_path: str | Path) -> Path:
    """Save a trained MLP regressor as a compressed NPZ archive."""
    output_path = Path(file_path)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    ensure_directory(output_path.parent)

    metadata = {
        "input_dim": model.config.input_dim,
        "output_dim": model.config.output_dim,
        "hidden_dims": list(model.config.hidden_dims),
        "num_layers": len(model.weights),
    }
    arrays: dict[str, np.ndarray] = {
        "input_mean": model.input_mean,
        "input_std": model.input_std,
        "target_mean": model.target_mean,
        "target_std": model.target_std,
        "metadata_json": np.array(json.dumps(metadata)),
    }

    for layer_index, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        arrays[f"weight_{layer_index}"] = weight
        arrays[f"bias_{layer_index}"] = bias

    np.savez_compressed(output_path, **arrays)
    return output_path


def load_model(file_path: str | Path) -> MLPRegressor:
    """Load a trained MLP regressor from a compressed NPZ archive."""
    with np.load(Path(file_path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        config = ModelConfig(
            input_dim=int(metadata["input_dim"]),
            output_dim=int(metadata["output_dim"]),
            hidden_dims=tuple(int(value) for value in metadata["hidden_dims"]),
        )

        weights: list[FloatArray] = []
        biases: list[FloatArray] = []
        for layer_index in range(int(metadata["num_layers"])):
            weights.append(np.asarray(data[f"weight_{layer_index}"], dtype=np.float64))
            biases.append(np.asarray(data[f"bias_{layer_index}"], dtype=np.float64))

        return MLPRegressor(
            weights=weights,
            biases=biases,
            input_mean=np.asarray(data["input_mean"], dtype=np.float64),
            input_std=np.asarray(data["input_std"], dtype=np.float64),
            target_mean=np.asarray(data["target_mean"], dtype=np.float64),
            target_std=np.asarray(data["target_std"], dtype=np.float64),
            config=config,
        )
