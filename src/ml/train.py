"""Training entrypoints for ML experiments."""

from src.ml.datasets import SHGDataset
from src.ml.models import ModelConfig, build_model


def train_model(dataset: SHGDataset, config: ModelConfig) -> dict[str, int]:
    """Return a placeholder trained model."""
    _ = dataset
    return build_model(config)
