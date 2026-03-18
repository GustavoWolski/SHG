"""Evaluation helpers for ML experiments."""

from src.ml.datasets import SHGDataset


def evaluate_model(model: object, dataset: SHGDataset) -> dict[str, float]:
    """Return placeholder evaluation metrics."""
    _ = model
    _ = dataset
    return {"loss": 0.0}
