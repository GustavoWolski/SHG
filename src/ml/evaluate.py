"""Evaluation helpers for masked-input SHG inverse models."""

from dataclasses import dataclass

import numpy as np

from src.ml.datasets import SHGDataset, build_input_features, full_observation_masks, single_channel_masks
from src.ml.models import MLPRegressor


@dataclass
class ScenarioMetrics:
    """Regression metrics for one SHG observation scenario."""

    mse: float
    mae: float


def _scenario_metrics(model: MLPRegressor, dataset: SHGDataset, masks: np.ndarray) -> ScenarioMetrics:
    """Evaluate the model under a specific missing-data mask."""
    features = build_input_features(dataset.i3, dataset.i1, masks)
    predictions = model.predict(features)
    errors = predictions - dataset.targets
    return ScenarioMetrics(
        mse=float(np.mean(errors ** 2)),
        mae=float(np.mean(np.abs(errors))),
    )


def evaluate_model(model: MLPRegressor, dataset: SHGDataset) -> dict[str, ScenarioMetrics]:
    """Evaluate the model with both curves, only i3, and only i1."""
    return {
        "i3_i1": _scenario_metrics(model, dataset, full_observation_masks(dataset.num_samples)),
        "i3_only": _scenario_metrics(model, dataset, single_channel_masks(dataset.num_samples, keep="i3")),
        "i1_only": _scenario_metrics(model, dataset, single_channel_masks(dataset.num_samples, keep="i1")),
    }
