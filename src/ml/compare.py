"""Comparison utilities for SHG inverse methods."""

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import time
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from src.data.synthetic_generator import PARAMETER_NAMES
from src.inverse.fitters import DEFAULT_BOUNDS, refine_fit_locally, run_fit, run_natural_fit
from src.inverse.objective import NormalizationStrategy
from src.ml.datasets import SHGDataset, build_input_features, full_observation_masks
from src.ml.evaluate import (
    ParameterMetrics,
    ReconstructionMetrics,
    compute_parameter_metrics,
    compute_reconstruction_metrics,
    reconstruct_curves_from_predictions,
)
from src.ml.models import MLPRegressor
from src.utils.io import ensure_directory

FloatArray = npt.NDArray[np.float64]
LocalBoundsMode = Literal["global", "neighborhood"]


@dataclass
class TimingMetrics:
    """Timing statistics for one inverse method."""

    total_seconds: float
    mean_seconds_per_sample: float
    per_sample_seconds: FloatArray

    def summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable timing summary."""
        return {
            "total_seconds": self.total_seconds,
            "mean_seconds_per_sample": self.mean_seconds_per_sample,
        }


@dataclass
class MethodComparisonResult:
    """Aggregated result for one SHG inversion method."""

    parameter_metrics: dict[str, ParameterMetrics]
    reconstruction_metrics: ReconstructionMetrics
    timing: TimingMetrics
    predictions: FloatArray
    reconstructed_i3: FloatArray
    reconstructed_i1: FloatArray
    sample_mse_i3: FloatArray
    sample_mse_i1: FloatArray
    sample_total_reconstruction_error: FloatArray

    def summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable method summary."""
        return {
            "parameter_metrics": {name: asdict(metrics) for name, metrics in self.parameter_metrics.items()},
            "reconstruction_metrics": asdict(self.reconstruction_metrics),
            "timing": self.timing.summary_dict(),
        }


@dataclass
class ComparisonReport:
    """Comparison report across classical, natural, ML and hybrid SHG inversion methods."""

    methods: dict[str, MethodComparisonResult]

    def summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary of all methods."""
        return {method_name: result.summary_dict() for method_name, result in self.methods.items()}


def _print_progress(label: str, current: int, total: int) -> None:
    """Print a simple progress indicator for per-sample routines."""
    progress_percent = 100.0 * current / total
    print(f"\r{label}: {current}/{total} ({progress_percent:5.1f}%)", end="", flush=True)
    if current == total:
        print()


def _build_method_result(
    dataset: SHGDataset,
    predictions: FloatArray,
    per_sample_seconds: FloatArray,
) -> MethodComparisonResult:
    """Aggregate parameter, reconstruction and timing metrics for one method."""
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
    timing = TimingMetrics(
        total_seconds=float(np.sum(per_sample_seconds)),
        mean_seconds_per_sample=float(np.mean(per_sample_seconds)),
        per_sample_seconds=per_sample_seconds,
    )
    return MethodComparisonResult(
        parameter_metrics=parameter_metrics,
        reconstruction_metrics=reconstruction_metrics,
        timing=timing,
        predictions=predictions,
        reconstructed_i3=reconstructed_i3,
        reconstructed_i1=reconstructed_i1,
        sample_mse_i3=sample_mse_i3,
        sample_mse_i1=sample_mse_i1,
        sample_total_reconstruction_error=sample_total_error,
    )


def _run_classical_method(
    dataset: SHGDataset,
    normalization_strategy: NormalizationStrategy,
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> MethodComparisonResult:
    """Run sample-wise classical SHG inversion with differential evolution."""
    num_samples = dataset.num_samples
    predictions = np.zeros_like(dataset.targets)
    per_sample_seconds = np.zeros(num_samples, dtype=np.float64)

    for sample_index in range(num_samples):
        start_time = time.perf_counter()
        fit_result = run_fit(
            d_exp=dataset.d_nm,
            i3_exp=dataset.i3[sample_index],
            i1_exp=dataset.i1[sample_index],
            lambda_m=dataset.lambda_m,
            normalization_strategy=normalization_strategy,
            seed=None if seed is None else seed + sample_index,
            verbose=False,
        )
        per_sample_seconds[sample_index] = time.perf_counter() - start_time
        predictions[sample_index] = fit_result.parameter_vector

        if show_progress:
            _print_progress("Classical fit", sample_index + 1, num_samples)

    return _build_method_result(dataset, predictions, per_sample_seconds)


def _run_ml_method(dataset: SHGDataset, model: MLPRegressor) -> MethodComparisonResult:
    """Run direct SHG parameter prediction with the trained MLP."""
    full_masks = full_observation_masks(dataset.num_samples)
    features = build_input_features(dataset.i3, dataset.i1, full_masks)

    start_time = time.perf_counter()
    predictions = model.predict(features)
    total_seconds = time.perf_counter() - start_time

    per_sample_seconds = np.full(dataset.num_samples, total_seconds / dataset.num_samples, dtype=np.float64)
    return _build_method_result(dataset, predictions, per_sample_seconds)


def _run_natural_method(
    dataset: SHGDataset,
    normalization_strategy: NormalizationStrategy,
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> MethodComparisonResult:
    """Run sample-wise natural-computation SHG inversion with dual annealing."""
    num_samples = dataset.num_samples
    predictions = np.zeros_like(dataset.targets)
    per_sample_seconds = np.zeros(num_samples, dtype=np.float64)

    for sample_index in range(num_samples):
        start_time = time.perf_counter()
        fit_result = run_natural_fit(
            d_exp=dataset.d_nm,
            i3_exp=dataset.i3[sample_index],
            i1_exp=dataset.i1[sample_index],
            lambda_m=dataset.lambda_m,
            normalization_strategy=normalization_strategy,
            seed=None if seed is None else seed + sample_index,
            verbose=False,
        )
        per_sample_seconds[sample_index] = time.perf_counter() - start_time
        predictions[sample_index] = fit_result.parameter_vector

        if show_progress:
            _print_progress("Natural fit", sample_index + 1, num_samples)

    return _build_method_result(dataset, predictions, per_sample_seconds)


def _compute_local_bounds(
    initial_guess: FloatArray,
    global_bounds: list[tuple[float, float]],
    mode: LocalBoundsMode,
    neighborhood_fraction: float,
) -> list[tuple[float, float]]:
    """Build physically valid local bounds for the hybrid refinement."""
    if mode == "global":
        return global_bounds

    if mode != "neighborhood":
        raise ValueError(f"Unknown local bounds mode: {mode}")

    local_bounds: list[tuple[float, float]] = []
    for parameter_index, (lower, upper) in enumerate(global_bounds):
        radius = neighborhood_fraction * (upper - lower)
        center = float(initial_guess[parameter_index])
        local_lower = max(lower, center - radius)
        local_upper = min(upper, center + radius)
        if local_lower >= local_upper:
            local_lower, local_upper = lower, upper
        local_bounds.append((float(local_lower), float(local_upper)))
    return local_bounds


def _run_hybrid_method(
    dataset: SHGDataset,
    model: MLPRegressor,
    normalization_strategy: NormalizationStrategy,
    local_bounds_mode: LocalBoundsMode,
    neighborhood_fraction: float,
    show_progress: bool = True,
) -> MethodComparisonResult:
    """Run MLP prediction followed by bounded local physical refinement."""
    full_masks = full_observation_masks(dataset.num_samples)
    features = build_input_features(dataset.i3, dataset.i1, full_masks)

    ml_start = time.perf_counter()
    ml_predictions = model.predict(features)
    ml_total_seconds = time.perf_counter() - ml_start

    num_samples = dataset.num_samples
    refined_predictions = np.zeros_like(dataset.targets)
    per_sample_seconds = np.full(num_samples, ml_total_seconds / num_samples, dtype=np.float64)

    for sample_index in range(num_samples):
        local_bounds = _compute_local_bounds(
            initial_guess=ml_predictions[sample_index],
            global_bounds=DEFAULT_BOUNDS,
            mode=local_bounds_mode,
            neighborhood_fraction=neighborhood_fraction,
        )
        start_time = time.perf_counter()
        fit_result = refine_fit_locally(
            d_exp=dataset.d_nm,
            i3_exp=dataset.i3[sample_index],
            i1_exp=dataset.i1[sample_index],
            lambda_m=dataset.lambda_m,
            initial_guess=ml_predictions[sample_index],
            bounds=local_bounds,
            normalization_strategy=normalization_strategy,
            verbose=False,
        )
        per_sample_seconds[sample_index] += time.perf_counter() - start_time
        refined_predictions[sample_index] = fit_result.parameter_vector

        if show_progress:
            _print_progress("Hybrid refinement", sample_index + 1, num_samples)

    return _build_method_result(dataset, refined_predictions, per_sample_seconds)


def save_comparison_summary_json(report: ComparisonReport, output_path: str | Path) -> Path:
    """Save the comparison summary as JSON."""
    output = Path(output_path)
    output.write_text(json.dumps(report.summary_dict(), indent=2), encoding="utf-8")
    return output


def save_comparison_summary_csv(report: ComparisonReport, output_path: str | Path) -> Path:
    """Save the comparison summary as CSV."""
    output = Path(output_path)
    fieldnames = [
        "method",
        "total_seconds",
        "mean_seconds_per_sample",
        "mse_i3",
        "mse_i1",
        "mean_total_reconstruction_error",
        "simulation_failures",
    ]
    for parameter_name in PARAMETER_NAMES:
        fieldnames.extend(
            [
                f"{parameter_name}_mae",
                f"{parameter_name}_rmse",
                f"{parameter_name}_r2",
            ]
        )

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for method_name, result in report.methods.items():
            row = {
                "method": method_name,
                "total_seconds": result.timing.total_seconds,
                "mean_seconds_per_sample": result.timing.mean_seconds_per_sample,
                "mse_i3": result.reconstruction_metrics.mse_i3,
                "mse_i1": result.reconstruction_metrics.mse_i1,
                "mean_total_reconstruction_error": result.reconstruction_metrics.mean_total_error,
                "simulation_failures": result.reconstruction_metrics.simulation_failures,
            }
            for parameter_name, metrics in result.parameter_metrics.items():
                row[f"{parameter_name}_mae"] = metrics.mae
                row[f"{parameter_name}_rmse"] = metrics.rmse
                row[f"{parameter_name}_r2"] = metrics.r2
            writer.writerow(row)

    return output


def _import_matplotlib() -> object:
    """Import matplotlib lazily for comparison figures."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate comparison figures.") from exc
    return plt


def plot_timing_comparison(report: ComparisonReport, output_path: str | Path) -> Path:
    """Plot mean runtime comparison across inversion methods."""
    plt = _import_matplotlib()
    method_names = list(report.methods.keys())
    mean_times = [report.methods[method_name].timing.mean_seconds_per_sample for method_name in method_names]

    method_colors = {
        "classical": "black",
        "natural": "seagreen",
        "ml": "royalblue",
        "hybrid": "darkorange",
    }
    colors = [method_colors.get(method_name, "gray") for method_name in method_names]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(method_names, mean_times, color=colors)
    axis.set_ylabel("Tempo medio por amostra (s)")
    axis.set_title("Comparacao de custo computacional")
    axis.grid(True, axis="y", alpha=0.3)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_parameter_rmse_comparison(report: ComparisonReport, output_path: str | Path) -> Path:
    """Plot per-parameter RMSE for all methods."""
    plt = _import_matplotlib()
    method_names = list(report.methods.keys())
    parameter_indices = np.arange(len(PARAMETER_NAMES))
    bar_width = 0.22

    figure, axis = plt.subplots(figsize=(10, 5))
    for method_offset, method_name in enumerate(method_names):
        rmse_values = [report.methods[method_name].parameter_metrics[name].rmse for name in PARAMETER_NAMES]
        axis.bar(parameter_indices + method_offset * bar_width, rmse_values, width=bar_width, label=method_name)

    axis.set_xticks(parameter_indices + bar_width)
    axis.set_xticklabels(PARAMETER_NAMES)
    axis.set_ylabel("RMSE")
    axis.set_title("Comparacao de erro parametrico")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_reconstruction_error_comparison(report: ComparisonReport, output_path: str | Path) -> Path:
    """Plot mean reconstruction error across inversion methods."""
    plt = _import_matplotlib()
    method_names = list(report.methods.keys())
    total_errors = [report.methods[method_name].reconstruction_metrics.mean_total_error for method_name in method_names]

    method_colors = {
        "classical": "black",
        "natural": "seagreen",
        "ml": "royalblue",
        "hybrid": "darkorange",
    }
    colors = [method_colors.get(method_name, "gray") for method_name in method_names]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(method_names, total_errors, color=colors)
    axis.set_ylabel("Erro medio total de reconstrucao")
    axis.set_title("Comparacao de reconstrucao fisica")
    axis.grid(True, axis="y", alpha=0.3)

    output = Path(output_path)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _select_example_indices(report: ComparisonReport, examples_per_group: int) -> list[tuple[str, int]]:
    """Select representative samples using mean reconstruction error across methods."""
    method_errors = []
    for result in report.methods.values():
        safe_errors = np.where(np.isfinite(result.sample_total_reconstruction_error), result.sample_total_reconstruction_error, np.nan)
        method_errors.append(safe_errors)

    stacked_errors = np.vstack(method_errors)
    mean_errors = np.nanmean(stacked_errors, axis=0)
    finite_indices = np.flatnonzero(np.isfinite(mean_errors))
    if finite_indices.size == 0:
        return []

    sorted_indices = finite_indices[np.argsort(mean_errors[finite_indices])]
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


def plot_method_reconstruction_examples(
    dataset: SHGDataset,
    report: ComparisonReport,
    output_path: str | Path,
    examples_per_group: int = 1,
) -> Optional[Path]:
    """Plot true and reconstructed SHG curves for representative samples."""
    selected_examples = _select_example_indices(report, examples_per_group)
    if not selected_examples:
        return None

    plt = _import_matplotlib()
    figure, axes = plt.subplots(len(selected_examples), 2, figsize=(13, 3.5 * len(selected_examples)), sharex=True)
    if len(selected_examples) == 1:
        axes = np.asarray([axes], dtype=object)

    method_styles = {
        "classical": ("--", "black"),
        "natural": ((0, (3, 1, 1, 1)), "seagreen"),
        "ml": ("-.", "royalblue"),
        "hybrid": (":", "darkorange"),
    }

    for row_index, (quality_label, sample_index) in enumerate(selected_examples):
        axis_i3 = axes[row_index, 0]
        axis_i1 = axes[row_index, 1]

        axis_i3.plot(dataset.d_nm, dataset.i3[sample_index], "-", color="forestgreen", linewidth=2.0, label="True")
        axis_i1.plot(dataset.d_nm, dataset.i1[sample_index], "-", color="forestgreen", linewidth=2.0, label="True")

        for method_name, result in report.methods.items():
            line_style, color = method_styles.get(method_name, ("--", None))
            axis_i3.plot(
                dataset.d_nm,
                result.reconstructed_i3[sample_index],
                line_style,
                color=color,
                label=method_name,
            )
            axis_i1.plot(
                dataset.d_nm,
                result.reconstructed_i1[sample_index],
                line_style,
                color=color,
                label=method_name,
            )

        axis_i3.set_ylabel(f"{quality_label} #{sample_index}")
        axis_i3.set_title("i3")
        axis_i3.grid(True, alpha=0.3)
        axis_i3.legend()

        axis_i1.set_title("i1")
        axis_i1.grid(True, alpha=0.3)
        axis_i1.legend()

    axes[-1, 0].set_xlabel("d (nm)")
    axes[-1, 1].set_xlabel("d (nm)")
    figure.suptitle("Exemplos comparativos de reconstrucao SHG")
    figure.tight_layout()
    output = Path(output_path)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def compare_methods(
    model: MLPRegressor,
    dataset: SHGDataset,
    output_dir: str | Path | None = None,
    normalization_strategy: NormalizationStrategy = "global",
    local_bounds_mode: LocalBoundsMode = "neighborhood",
    neighborhood_fraction: float = 0.1,
    classical_seed: Optional[int] = None,
    save_figures: bool = True,
    examples_per_group: int = 1,
    show_progress: bool = True,
) -> ComparisonReport:
    """Compare classical, natural, ML and hybrid SHG inverse methods on a test dataset."""
    report = ComparisonReport(
        methods={
            "classical": _run_classical_method(
                dataset=dataset,
                normalization_strategy=normalization_strategy,
                seed=classical_seed,
                show_progress=show_progress,
            ),
            "natural": _run_natural_method(
                dataset=dataset,
                normalization_strategy=normalization_strategy,
                seed=classical_seed,
                show_progress=show_progress,
            ),
            "ml": _run_ml_method(dataset=dataset, model=model),
            "hybrid": _run_hybrid_method(
                dataset=dataset,
                model=model,
                normalization_strategy=normalization_strategy,
                local_bounds_mode=local_bounds_mode,
                neighborhood_fraction=neighborhood_fraction,
                show_progress=show_progress,
            ),
        }
    )

    if output_dir is not None:
        base_output_dir = ensure_directory(output_dir)
        save_comparison_summary_json(report, base_output_dir / "comparison_summary.json")
        save_comparison_summary_csv(report, base_output_dir / "comparison_summary.csv")

        if save_figures:
            plot_timing_comparison(report, base_output_dir / "timing_comparison.png")
            plot_parameter_rmse_comparison(report, base_output_dir / "parameter_rmse_comparison.png")
            plot_reconstruction_error_comparison(report, base_output_dir / "reconstruction_error_comparison.png")
            plot_method_reconstruction_examples(
                dataset=dataset,
                report=report,
                output_path=base_output_dir / "method_reconstruction_examples.png",
                examples_per_group=examples_per_group,
            )

    return report
