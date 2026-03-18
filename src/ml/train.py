"""Training entrypoints for masked-input SHG MLP models."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from src.ml.datasets import SHGDataset, build_input_features, sample_augmentation_masks
from src.ml.models import MLPRegressor, ModelConfig, build_model
from src.utils.io import ensure_directory

FloatArray = npt.NDArray[np.float64]


@dataclass
class TrainingConfig:
    """Optimization configuration for the SHG MLP."""

    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    seed: Optional[int] = None
    verbose: bool = False


@dataclass
class TrainingResult:
    """Trained model and compact loss history."""

    model: MLPRegressor
    train_loss_history: list[float]


@dataclass
class TrainingSummary:
    """Compact JSON-friendly summary of one MLP training run."""

    dataset_path: Optional[str]
    model_path: Optional[str]
    num_samples: int
    curve_length: int
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    seed: Optional[int]
    final_train_loss: float
    train_loss_history: list[float]

    def to_dict(self) -> dict[str, object]:
        """Convert the training summary into a JSON-serializable dictionary."""
        return {
            "dataset_path": self.dataset_path,
            "model_path": self.model_path,
            "num_samples": self.num_samples,
            "curve_length": self.curve_length,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "seed": self.seed,
            "final_train_loss": self.final_train_loss,
            "train_loss_history": self.train_loss_history,
        }


def build_training_summary(
    dataset: SHGDataset,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    training_result: TrainingResult,
    dataset_path: str | None = None,
    model_path: str | None = None,
) -> TrainingSummary:
    """Build a compact summary of one SHG MLP training run."""
    if not training_result.train_loss_history:
        raise ValueError("training_result.train_loss_history must not be empty.")

    return TrainingSummary(
        dataset_path=dataset_path,
        model_path=model_path,
        num_samples=dataset.num_samples,
        curve_length=dataset.curve_length,
        input_dim=model_config.input_dim,
        output_dim=model_config.output_dim,
        hidden_dims=model_config.hidden_dims,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        gradient_clip=training_config.gradient_clip,
        seed=training_config.seed,
        final_train_loss=float(training_result.train_loss_history[-1]),
        train_loss_history=[float(loss) for loss in training_result.train_loss_history],
    )


def save_training_summary(summary: TrainingSummary, file_path: str | Path) -> Path:
    """Save a compact MLP training summary as JSON."""
    output_path = Path(file_path)
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return output_path


def _normalize_columns(values: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Normalize features or targets column-wise."""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (values - mean) / std
    return normalized, mean.astype(np.float64), std.astype(np.float64)


def _forward_with_cache(
    model: MLPRegressor,
    normalized_features: FloatArray,
) -> tuple[FloatArray, list[FloatArray], list[FloatArray]]:
    """Run a forward pass and keep intermediate activations for backprop."""
    activations: list[FloatArray] = [normalized_features]
    pre_activations: list[FloatArray] = []
    current = normalized_features

    for layer_index, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        current = current @ weight + bias
        pre_activations.append(current)
        if layer_index < len(model.weights) - 1:
            current = np.maximum(current, 0.0)
        activations.append(current)

    return current, activations, pre_activations


def _backward_pass(
    model: MLPRegressor,
    predictions: FloatArray,
    targets: FloatArray,
    activations: list[FloatArray],
    pre_activations: list[FloatArray],
    weight_decay: float,
) -> tuple[list[FloatArray], list[FloatArray], float]:
    """Compute gradients for the MLP with MSE loss."""
    batch_size = targets.shape[0]
    delta = (2.0 / batch_size) * (predictions - targets)
    weight_grads: list[FloatArray] = [np.zeros_like(weight) for weight in model.weights]
    bias_grads: list[FloatArray] = [np.zeros_like(bias) for bias in model.biases]

    for layer_index in range(len(model.weights) - 1, -1, -1):
        weight_grads[layer_index] = activations[layer_index].T @ delta + weight_decay * model.weights[layer_index]
        bias_grads[layer_index] = np.sum(delta, axis=0)

        if layer_index > 0:
            delta = delta @ model.weights[layer_index].T
            delta = delta * (pre_activations[layer_index - 1] > 0.0)

    loss = float(np.mean((predictions - targets) ** 2))
    return weight_grads, bias_grads, loss


def _clip_gradients(weight_grads: list[FloatArray], bias_grads: list[FloatArray], clip_value: float) -> None:
    """Clip gradients by global norm for stable training."""
    if clip_value <= 0.0:
        return

    squared_norm = 0.0
    for gradient in weight_grads + bias_grads:
        squared_norm += float(np.sum(gradient ** 2))

    global_norm = np.sqrt(squared_norm)
    if global_norm <= clip_value or global_norm == 0.0:
        return

    scale = clip_value / global_norm
    for gradient in weight_grads:
        gradient *= scale
    for gradient in bias_grads:
        gradient *= scale


def _iterate_minibatches(num_samples: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Return shuffled batch indices."""
    indices = rng.permutation(num_samples)
    return [indices[start : start + batch_size] for start in range(0, num_samples, batch_size)]


def train_model(
    dataset: SHGDataset,
    model_config: ModelConfig,
    training_config: Optional[TrainingConfig] = None,
) -> TrainingResult:
    """Train an MLP on SHG curves with random missing-channel augmentation."""
    config = training_config or TrainingConfig()
    if dataset.num_samples <= 0:
        raise ValueError("dataset must contain at least one sample.")
    if model_config.input_dim != dataset.input_dim:
        raise ValueError("model_config.input_dim does not match the dataset feature size.")
    if model_config.output_dim != dataset.output_dim:
        raise ValueError("model_config.output_dim does not match the dataset target size.")
    if any(hidden_dim <= 0 for hidden_dim in model_config.hidden_dims):
        raise ValueError("model_config.hidden_dims must contain only positive values.")
    if config.epochs <= 0:
        raise ValueError("training_config.epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("training_config.batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("training_config.learning_rate must be positive.")
    if config.weight_decay < 0.0:
        raise ValueError("training_config.weight_decay must be non-negative.")
    if config.gradient_clip < 0.0:
        raise ValueError("training_config.gradient_clip must be non-negative.")

    rng = np.random.default_rng(config.seed)
    model = build_model(model_config, seed=config.seed)

    base_masks = np.ones((dataset.num_samples, 2), dtype=np.float64)
    base_features = build_input_features(dataset.i3, dataset.i1, base_masks)
    _, input_mean, input_std = _normalize_columns(base_features)
    normalized_targets, target_mean, target_std = _normalize_columns(dataset.targets)

    model.input_mean = input_mean
    model.input_std = input_std
    model.target_mean = target_mean
    model.target_std = target_std

    first_moments_w = [np.zeros_like(weight) for weight in model.weights]
    second_moments_w = [np.zeros_like(weight) for weight in model.weights]
    first_moments_b = [np.zeros_like(bias) for bias in model.biases]
    second_moments_b = [np.zeros_like(bias) for bias in model.biases]

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    train_loss_history: list[float] = []
    update_step = 0

    for epoch_index in range(config.epochs):
        epoch_losses: list[float] = []
        for batch_indices in _iterate_minibatches(dataset.num_samples, config.batch_size, rng):
            batch_i3 = dataset.i3[batch_indices]
            batch_i1 = dataset.i1[batch_indices]
            batch_targets = normalized_targets[batch_indices]
            batch_masks = sample_augmentation_masks(batch_indices.size, rng)

            batch_features = build_input_features(batch_i3, batch_i1, batch_masks)
            normalized_features = (batch_features - model.input_mean) / model.input_std

            predictions, activations, pre_activations = _forward_with_cache(model, normalized_features)
            weight_grads, bias_grads, batch_loss = _backward_pass(
                model=model,
                predictions=predictions,
                targets=batch_targets,
                activations=activations,
                pre_activations=pre_activations,
                weight_decay=config.weight_decay,
            )
            _clip_gradients(weight_grads, bias_grads, config.gradient_clip)

            update_step += 1
            for layer_index in range(len(model.weights)):
                first_moments_w[layer_index] = beta1 * first_moments_w[layer_index] + (1.0 - beta1) * weight_grads[layer_index]
                second_moments_w[layer_index] = beta2 * second_moments_w[layer_index] + (1.0 - beta2) * (weight_grads[layer_index] ** 2)
                first_moments_b[layer_index] = beta1 * first_moments_b[layer_index] + (1.0 - beta1) * bias_grads[layer_index]
                second_moments_b[layer_index] = beta2 * second_moments_b[layer_index] + (1.0 - beta2) * (bias_grads[layer_index] ** 2)

                weight_m_hat = first_moments_w[layer_index] / (1.0 - beta1 ** update_step)
                weight_v_hat = second_moments_w[layer_index] / (1.0 - beta2 ** update_step)
                bias_m_hat = first_moments_b[layer_index] / (1.0 - beta1 ** update_step)
                bias_v_hat = second_moments_b[layer_index] / (1.0 - beta2 ** update_step)

                model.weights[layer_index] -= config.learning_rate * weight_m_hat / (np.sqrt(weight_v_hat) + epsilon)
                model.biases[layer_index] -= config.learning_rate * bias_m_hat / (np.sqrt(bias_v_hat) + epsilon)

            epoch_losses.append(batch_loss)

        epoch_loss = float(np.mean(epoch_losses))
        train_loss_history.append(epoch_loss)
        if config.verbose and ((epoch_index + 1) % 25 == 0 or epoch_index == 0):
            print(f"Epoch {epoch_index + 1}/{config.epochs} - loss={epoch_loss:.6f}")

    return TrainingResult(model=model, train_loss_history=train_loss_history)
