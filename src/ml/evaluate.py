"""Evaluation helpers for masked-input SHG inverse models."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from src.data.synthetic_generator import PARAMETER_NAMES
from src.ml.datasets import SHGDataset, build_input_features, full_observation_masks, single_channel_masks
from src.ml.models import MLPRegressor
from src.physics.shg_model import simulate_shg
from src.inverse.objective import build_shg_params
from src.utils.io import ensure_directory

FloatArray = npt.NDArray[np.float64]


@dataclass
class ParameterMetrics:
    """Regression metrics for one predicted physical parameter."""

    mae: float
    rmse: float
    r2: float


@dataclass
class ReconstructionMetrics:
    """Aggregate metrics for SHG curve reconstruction."""

    mse_i3: float
    mse_i1: float
    mean_total_error: float
    simulation_failures: int


@dataclass
class ScenarioEvaluation:
    """Full evaluation result for one SHG observation scenario."""

    parameter_metrics: dict[str, ParameterMetrics]
    reconstruction_metrics: ReconstructionMetrics
    predictions: FloatArray
    parameter_errors: FloatArray
    reconstructed_i3: FloatArray
    reconstructed_i1: FloatArray
    sample_mse_i3: FloatArray
    sample_mse_i1: FloatArray
    sample_total_reconstruction_error: FloatArray

    def summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "parameter_metrics": {name: asdict(metrics) for name, metrics in self.parameter_metrics.items()},
            "reconstruction_metrics": asdict(self.reconstruction_metrics),
        }


@dataclass
class EvaluationReport:
    """Evaluation report across all missing-data scenarios."""

    scenarios: dict[str, ScenarioEvaluation]

    def summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary of the evaluation."""
        return {scenario_name: scenario.summary_dict() for scenario_name, scenario in self.scenarios.items()}


def _safe_r2(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Compute a numerically safe R² score."""
    residual_sum = float(np.sum((y_true - y_pred) ** 2))
    total_sum = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total_sum <= 1e-12:
        return 1.0 if residual_sum <= 1e-12 else 0.0
    return float(1.0 - residual_sum / total_sum)


def compute_parameter_metrics(y_true: FloatArray, y_pred: FloatArray) -> dict[str, ParameterMetrics]:
    """Compute MAE, RMSE and R² for each predicted physical parameter."""
    metrics: dict[str, ParameterMetrics] = {}
    for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
        true_values = y_true[:, parameter_index]
        predicted_values = y_pred[:, parameter_index]
        errors = predicted_values - true_values
        metrics[parameter_name] = ParameterMetrics(
            mae=float(np.mean(np.abs(errors))),
            rmse=float(np.sqrt(np.mean(errors ** 2))),
            r2=_safe_r2(true_values, predicted_values),
        )
    return metrics


def predict_parameters_for_scenario(model: MLPRegressor, dataset: SHGDataset, masks: FloatArray) -> FloatArray:
    """Predict SHG physical parameters under a given observation mask."""
    features = build_input_features(dataset.i3, dataset.i1, masks)
    return model.predict(features)


def reconstruct_curves_from_predictions(
    dataset: SHGDataset,
    predictions: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, int]:
    """Reconstruct SHG curves by feeding predicted parameters into the forward model."""
    num_samples = dataset.num_samples
    curve_length = dataset.curve_length

    reconstructed_i3 = np.full((num_samples, curve_length), np.nan, dtype=np.float64)
    reconstructed_i1 = np.full((num_samples, curve_length), np.nan, dtype=np.float64)
    sample_mse_i3 = np.full(num_samples, np.inf, dtype=np.float64)
    sample_mse_i1 = np.full(num_samples, np.inf, dtype=np.float64)
    sample_total_error = np.full(num_samples, np.inf, dtype=np.float64)
    simulation_failures = 0

    for sample_index in range(num_samples):
        params = build_shg_params(predictions[sample_index], dataset.lambda_m)
        try:
            i3_reconstructed, i1_reconstructed = simulate_shg(params, dataset.d_nm)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            simulation_failures += 1
            continue

        reconstructed_i3[sample_index] = i3_reconstructed
        reconstructed_i1[sample_index] = i1_reconstructed

        sample_mse_i3[sample_index] = float(np.mean((i3_reconstructed - dataset.i3[sample_index]) ** 2))
        sample_mse_i1[sample_index] = float(np.mean((i1_reconstructed - dataset.i1[sample_index]) ** 2))
        sample_total_error[sample_index] = 0.5 * (sample_mse_i3[sample_index] + sample_mse_i1[sample_index])

    return (
        reconstructed_i3,
        reconstructed_i1,
        sample_mse_i3,
        sample_mse_i1,
        sample_total_error,
        simulation_failures,
    )


def compute_reconstruction_metrics(
    sample_mse_i3: FloatArray,
    sample_mse_i1: FloatArray,
    sample_total_error: FloatArray,
    simulation_failures: int,
) -> ReconstructionMetrics:
    """Aggregate reconstruction errors across the test set."""
    finite_i3 = sample_mse_i3[np.isfinite(sample_mse_i3)]
    finite_i1 = sample_mse_i1[np.isfinite(sample_mse_i1)]
    finite_total = sample_total_error[np.isfinite(sample_total_error)]

    return ReconstructionMetrics(
        mse_i3=float(np.mean(finite_i3)) if finite_i3.size > 0 else float("inf"),
        mse_i1=float(np.mean(finite_i1)) if finite_i1.size > 0 else float("inf"),
        mean_total_error=float(np.mean(finite_total)) if finite_total.size > 0 else float("inf"),
        simulation_failures=simulation_failures,
    )


def evaluate_scenario(model: MLPRegressor, dataset: SHGDataset, masks: FloatArray) -> ScenarioEvaluation:
    """Run parameter and reconstruction evaluation for one mask scenario."""
    predictions = predict_parameters_for_scenario(model, dataset, masks)
    parameter_errors = predictions - dataset.targets
    parameter_metrics = compute_parameter_metrics(dataset.targets, predictions)
    (
        reconstructed_i3,
        reconstructed_i1,
        sample_mse_i3,
        sample_mse_i1,
        sample_total_error,
        simulation_failures,
    ) = reconstruct_curves_from_predictions(dataset, predictions)
    reconstruction_metrics = compute_reconstruction_metrics(
        sample_mse_i3=sample_mse_i3,
        sample_mse_i1=sample_mse_i1,
        sample_total_error=sample_total_error,
        simulation_failures=simulation_failures,
    )
    return ScenarioEvaluation(
        parameter_metrics=parameter_metrics,
        reconstruction_metrics=reconstruction_metrics,
        predictions=predictions,
        parameter_errors=parameter_errors,
        reconstructed_i3=reconstructed_i3,
        reconstructed_i1=reconstructed_i1,
        sample_mse_i3=sample_mse_i3,
        sample_mse_i1=sample_mse_i1,
        sample_total_reconstruction_error=sample_total_error,
    )


def _maybe_import_matplotlib() -> tuple[object, object]:
    """Import matplotlib lazily for plotting."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate evaluation figures.") from exc
    return plt, plt


def plot_predicted_vs_true(
    y_true: FloatArray,
    y_pred: FloatArray,
    output_path: str | Path,
    scenario_name: str,
) -> Path:
    """Save predicted-vs-true scatter plots for all physical parameters."""
    plt, _ = _maybe_import_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axis_list = axes.flatten()

    for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
        axis = axis_list[parameter_index]
        true_values = y_true[:, parameter_index]
        predicted_values = y_pred[:, parameter_index]
        min_value = min(float(np.min(true_values)), float(np.min(predicted_values)))
        max_value = max(float(np.max(true_values)), float(np.max(predicted_values)))
        axis.scatter(true_values, predicted_values, s=18, alpha=0.7)
        axis.plot([min_value, max_value], [min_value, max_value], "--k", linewidth=1.0)
        axis.set_title(parameter_name)
        axis.set_xlabel("Real")
        axis.set_ylabel("Previsto")
        axis.grid(True, alpha=0.3)

    figure.suptitle(f"Previsto vs real | {scenario_name}")
    figure.tight_layout()
    output = Path(output_path)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_parameter_error_histograms(
    parameter_errors: FloatArray,
    output_path: str | Path,
    scenario_name: str,
) -> Path:
    """Save histograms of parameter prediction errors."""
    plt, _ = _maybe_import_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axis_list = axes.flatten()

    for parameter_index, parameter_name in enumerate(PARAMETER_NAMES):
        axis = axis_list[parameter_index]
        axis.hist(parameter_errors[:, parameter_index], bins=30, alpha=0.8, color="steelblue")
        axis.set_title(parameter_name)
        axis.set_xlabel("Erro previsto - real")
        axis.set_ylabel("Contagem")
        axis.grid(True, alpha=0.3)

    figure.suptitle(f"Histogramas de erro | {scenario_name}")
    figure.tight_layout()
    output = Path(output_path)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _select_example_indices(errors: FloatArray, examples_per_group: int) -> list[tuple[str, int]]:
    """Select good, median and bad reconstruction examples."""
    finite_indices = np.flatnonzero(np.isfinite(errors))
    if finite_indices.size == 0:
        return []

    sorted_indices = finite_indices[np.argsort(errors[finite_indices])]
    selection: list[tuple[str, int]] = []

    good_indices = sorted_indices[:examples_per_group]
    median_center = sorted_indices.size // 2
    median_start = max(0, median_center - examples_per_group // 2)
    median_indices = sorted_indices[median_start : median_start + examples_per_group]
    bad_indices = sorted_indices[-examples_per_group:]

    for sample_index in good_indices:
        selection.append(("good", int(sample_index)))
    for sample_index in median_indices:
        selection.append(("median", int(sample_index)))
    for sample_index in bad_indices:
        selection.append(("bad", int(sample_index)))
    return selection


def plot_reconstruction_examples(
    dataset: SHGDataset,
    scenario_evaluation: ScenarioEvaluation,
    output_path: str | Path,
    scenario_name: str,
    examples_per_group: int = 2,
) -> Optional[Path]:
    """Plot representative good, median and bad SHG reconstructions."""
    selected_examples = _select_example_indices(
        scenario_evaluation.sample_total_reconstruction_error,
        examples_per_group=examples_per_group,
    )
    if not selected_examples:
        return None

    plt, _ = _maybe_import_matplotlib()
    figure, axes = plt.subplots(len(selected_examples), 2, figsize=(12, 3.4 * len(selected_examples)), sharex=True)
    if len(selected_examples) == 1:
        axes = np.asarray([axes], dtype=object)

    for row_index, (quality_label, sample_index) in enumerate(selected_examples):
        axis_i3 = axes[row_index, 0]
        axis_i1 = axes[row_index, 1]

        axis_i3.plot(dataset.d_nm, dataset.i3[sample_index], "-k", label="True")
        axis_i3.plot(dataset.d_nm, scenario_evaluation.reconstructed_i3[sample_index], "--r", label="Recon")
        axis_i3.set_ylabel(f"{quality_label} #{sample_index}")
        axis_i3.set_title(f"i3 | total error={scenario_evaluation.sample_total_reconstruction_error[sample_index]:.3e}")
        axis_i3.grid(True, alpha=0.3)
        axis_i3.legend()

        axis_i1.plot(dataset.d_nm, dataset.i1[sample_index], "-k", label="True")
        axis_i1.plot(dataset.d_nm, scenario_evaluation.reconstructed_i1[sample_index], "--b", label="Recon")
        axis_i1.set_title(f"i1 | total error={scenario_evaluation.sample_total_reconstruction_error[sample_index]:.3e}")
        axis_i1.grid(True, alpha=0.3)
        axis_i1.legend()

    axes[-1, 0].set_xlabel("d (nm)")
    axes[-1, 1].set_xlabel("d (nm)")
    figure.suptitle(f"Exemplos de reconstrucao | {scenario_name}")
    figure.tight_layout()
    output = Path(output_path)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def save_evaluation_summary(report: EvaluationReport, output_path: str | Path) -> Path:
    """Save the evaluation summary as JSON."""
    output = Path(output_path)
    output.write_text(json.dumps(report.summary_dict(), indent=2), encoding="utf-8")
    return output


def evaluate_model(
    model: MLPRegressor,
    dataset: SHGDataset,
    output_dir: str | Path | None = None,
    save_figures: bool = True,
    examples_per_group: int = 2,
) -> EvaluationReport:
    """Run the full SHG inverse-model evaluation pipeline."""
    scenario_masks = {
        "i3_i1": full_observation_masks(dataset.num_samples),
        "i3_only": single_channel_masks(dataset.num_samples, keep="i3"),
        "i1_only": single_channel_masks(dataset.num_samples, keep="i1"),
    }

    report = EvaluationReport(
        scenarios={
            scenario_name: evaluate_scenario(model, dataset, masks)
            for scenario_name, masks in scenario_masks.items()
        }
    )

    if output_dir is not None:
        base_output_dir = ensure_directory(output_dir)
        save_evaluation_summary(report, base_output_dir / "evaluation_summary.json")

        if save_figures:
            for scenario_name, scenario in report.scenarios.items():
                scenario_dir = ensure_directory(base_output_dir / scenario_name)
                plot_predicted_vs_true(
                    y_true=dataset.targets,
                    y_pred=scenario.predictions,
                    output_path=scenario_dir / "predicted_vs_true.png",
                    scenario_name=scenario_name,
                )
                plot_parameter_error_histograms(
                    parameter_errors=scenario.parameter_errors,
                    output_path=scenario_dir / "parameter_error_histograms.png",
                    scenario_name=scenario_name,
                )
                plot_reconstruction_examples(
                    dataset=dataset,
                    scenario_evaluation=scenario,
                    output_path=scenario_dir / "reconstruction_examples.png",
                    scenario_name=scenario_name,
                    examples_per_group=examples_per_group,
                )

    return report
