"""Experimental inverse-method runners for SHG fitting.

This module applies the available inverse approaches to a single
experimental SHG curve pair. The physical forward model is unchanged;
the only extra step for ML-based inference is a numerical completion of
missing points so the fixed-size MLP input can still be assembled.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from src.inverse.fitters import DEFAULT_BOUNDS, refine_fit_locally, run_fit, run_natural_fit
from src.inverse.objective import ChannelWeights, NormalizationStrategy, build_shg_params, error_function, normalize_shg_curves
from src.ml.datasets import build_input_features
from src.ml.models import MLPRegressor
from src.physics.shg_model import SHGParams, simulate_shg
from src.utils.io import ensure_directory

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
MethodName = Literal["classical", "natural", "ml", "hybrid"]
LocalBoundsMode = Literal["global", "neighborhood"]


def _compute_channel_metrics(exp_values: FloatArray, sim_values: FloatArray, mask: BoolArray) -> Optional[dict[str, float]]:
    """Compute per-channel error metrics on observed (masked) points.

    Returns ``None`` when no observed points exist for the channel.
    """
    observed_mask = np.asarray(mask, dtype=bool)
    if not np.any(observed_mask):
        return None
    residuals = exp_values[observed_mask] - sim_values[observed_mask]
    mse = float((residuals ** 2).mean())
    mae = float(np.abs(residuals).mean())
    rmse = float(mse ** 0.5)
    max_abs_error = float(np.abs(residuals).max())
    return {"mse": mse, "mae": mae, "rmse": rmse, "max_abs_error": max_abs_error}


@dataclass
class ExperimentalMethodResult:
    """One inverse-method result on a single experimental SHG sample."""

    method_name: MethodName
    fitted_params: SHGParams
    parameter_vector: FloatArray
    objective_error: float
    runtime_seconds: float
    reconstructed_i3: FloatArray
    reconstructed_i1: FloatArray
    normalization_strategy: NormalizationStrategy
    channel_mask: tuple[bool, bool]
    used_interpolation: bool
    d_exp: FloatArray = None
    i3_exp: FloatArray = None
    i1_exp: FloatArray = None
    i3_mask: BoolArray = None
    i1_mask: BoolArray = None
    message: str = ""

    def _channel_metrics_dict(self) -> dict[str, object]:
        """Compute per-channel and global metrics using normalized curves."""
        if self.d_exp is None or self.i3_exp is None or self.i1_exp is None:
            return {}
        normalized = normalize_shg_curves(
            i3_exp=self.i3_exp,
            i1_exp=self.i1_exp,
            i3_sim=self.reconstructed_i3,
            i1_sim=self.reconstructed_i1,
            strategy=self.normalization_strategy,
            i3_mask=self.i3_mask,
            i1_mask=self.i1_mask,
        )
        if normalized is None:
            return {}
        i3_exp_n, i1_exp_n, i3_sim_n, i1_sim_n = normalized
        i3_obs = np.ones(len(i3_exp_n), dtype=bool) if self.i3_mask is None else np.asarray(self.i3_mask, dtype=bool)
        i1_obs = np.ones(len(i1_exp_n), dtype=bool) if self.i1_mask is None else np.asarray(self.i1_mask, dtype=bool)
        i3_metrics = _compute_channel_metrics(i3_exp_n, i3_sim_n, i3_obs)
        i1_metrics = _compute_channel_metrics(i1_exp_n, i1_sim_n, i1_obs)
        available_mses = [m["mse"] for m in (i3_metrics, i1_metrics) if m is not None]
        result: dict[str, object] = {}
        if available_mses:
            result["mean_channel_mse"] = sum(available_mses) / len(available_mses)
        if i3_metrics is not None:
            result["i3"] = i3_metrics
        if i1_metrics is not None:
            result["i1"] = i1_metrics
        return result

    def summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary of the method result."""
        metrics = self._channel_metrics_dict()
        result: dict[str, object] = {
            "method_name": self.method_name,
            "objective_error": self.objective_error,
            "runtime_seconds": self.runtime_seconds,
            "parameter_vector": self.parameter_vector.tolist(),
            "fitted_params": {
                "n21w": {"real": float(self.fitted_params.n21w.real), "imag": float(self.fitted_params.n21w.imag)},
                "n22w": {"real": float(self.fitted_params.n22w.real), "imag": float(self.fitted_params.n22w.imag)},
            },
            "normalization_strategy": self.normalization_strategy,
            "channel_mask": list(self.channel_mask),
            "used_interpolation": self.used_interpolation,
            "message": self.message,
        }
        if metrics:
            result["metrics"] = metrics
        return result


@dataclass
class ExperimentalComparisonReport:
    """Comparison report for single-sample experimental inversion."""

    results: dict[str, ExperimentalMethodResult]
    best_method_name: str

    def summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report summary."""
        return {
            "best_method_name": self.best_method_name,
            "results": {method_name: result.summary_dict() for method_name, result in self.results.items()},
        }


def _channel_observed(mask: BoolArray) -> bool:
    """Return whether at least one point exists for the channel."""
    return bool(np.any(mask))


def _fill_missing_curve(d_nm: FloatArray, values: FloatArray, mask: BoolArray) -> tuple[FloatArray, bool]:
    """Fill sporadic missing points so the fixed-size MLP can consume the curve."""
    completed_values = np.asarray(values, dtype=np.float64).copy()
    observed_mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    missing_mask = ~observed_mask

    if not np.any(missing_mask):
        return completed_values, False
    if not np.any(observed_mask):
        return np.zeros_like(completed_values), False

    observed_indices = np.flatnonzero(observed_mask)
    if observed_indices.size == 1:
        completed_values[missing_mask] = completed_values[observed_indices[0]]
        return completed_values, True

    completed_values[missing_mask] = np.interp(
        d_nm[missing_mask],
        d_nm[observed_mask],
        completed_values[observed_mask],
    )
    return completed_values, True


def _build_ml_features(
    d_nm: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    i3_mask: BoolArray,
    i1_mask: BoolArray,
) -> tuple[FloatArray, tuple[bool, bool], bool]:
    """Build one-sample MLP features from incomplete experimental curves."""
    completed_i3, i3_interpolated = _fill_missing_curve(d_nm, i3_exp, i3_mask)
    completed_i1, i1_interpolated = _fill_missing_curve(d_nm, i1_exp, i1_mask)
    channel_mask = (_channel_observed(i3_mask), _channel_observed(i1_mask))
    if not any(channel_mask):
        raise ValueError("At least one of i3 or i1 must contain observed experimental values.")

    features = build_input_features(
        completed_i3.reshape(1, -1),
        completed_i1.reshape(1, -1),
        np.array([[float(channel_mask[0]), float(channel_mask[1])]], dtype=np.float64),
    )
    return features, channel_mask, bool(i3_interpolated or i1_interpolated)


def _build_result(
    method_name: MethodName,
    parameter_vector: FloatArray,
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    normalization_strategy: NormalizationStrategy,
    runtime_seconds: float,
    i3_mask: BoolArray,
    i1_mask: BoolArray,
    channel_mask: tuple[bool, bool],
    used_interpolation: bool,
    message: str = "",
    channel_weights: ChannelWeights = None,
) -> ExperimentalMethodResult:
    """Build a typed inverse-method result from a parameter vector."""
    fitted_params = build_shg_params(parameter_vector, lambda_m)
    reconstructed_i3, reconstructed_i1 = simulate_shg(fitted_params, d_exp)
    objective_value = error_function(
        parameter_vector,
        d_exp,
        i3_exp,
        i1_exp,
        lambda_m,
        normalization_strategy=normalization_strategy,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
        channel_weights=channel_weights,
    )
    return ExperimentalMethodResult(
        method_name=method_name,
        fitted_params=fitted_params,
        parameter_vector=np.asarray(parameter_vector, dtype=np.float64),
        objective_error=float(objective_value),
        runtime_seconds=float(runtime_seconds),
        reconstructed_i3=reconstructed_i3,
        reconstructed_i1=reconstructed_i1,
        normalization_strategy=normalization_strategy,
        channel_mask=channel_mask,
        used_interpolation=used_interpolation,
        d_exp=np.asarray(d_exp, dtype=np.float64),
        i3_exp=np.asarray(i3_exp, dtype=np.float64),
        i1_exp=np.asarray(i1_exp, dtype=np.float64),
        i3_mask=np.asarray(i3_mask, dtype=bool),
        i1_mask=np.asarray(i1_mask, dtype=bool),
        message=message,
    )


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


def _clip_parameter_vector_to_bounds(
    parameter_vector: FloatArray,
    bounds: list[tuple[float, float]],
) -> tuple[FloatArray, bool]:
    """Clip a parameter vector to the physical bounds used by the project."""
    clipped = np.array(
        [np.clip(parameter_vector[index], lower, upper) for index, (lower, upper) in enumerate(bounds)],
        dtype=np.float64,
    )
    return clipped, bool(not np.allclose(clipped, parameter_vector))


def run_classical_inverse_method(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    normalization_strategy: NormalizationStrategy,
    i3_mask: BoolArray,
    i1_mask: BoolArray,
    seed: Optional[int] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    channel_weights: ChannelWeights = None,
) -> ExperimentalMethodResult:
    """Run the baseline classical SHG inverse method on one experiment."""
    start_time = time.perf_counter()
    fit_result = run_fit(
        d_exp=d_exp,
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        lambda_m=lambda_m,
        bounds=bounds,
        normalization_strategy=normalization_strategy,
        seed=seed,
        verbose=False,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
        channel_weights=channel_weights,
    )
    runtime_seconds = time.perf_counter() - start_time
    return _build_result(
        method_name="classical",
        parameter_vector=fit_result.parameter_vector,
        d_exp=d_exp,
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        lambda_m=lambda_m,
        normalization_strategy=normalization_strategy,
        runtime_seconds=runtime_seconds,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
        channel_mask=(_channel_observed(i3_mask), _channel_observed(i1_mask)),
        used_interpolation=False,
        message=fit_result.message,
        channel_weights=channel_weights,
    )


def run_ml_inverse_method(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    model: MLPRegressor,
    normalization_strategy: NormalizationStrategy,
    i3_mask: BoolArray,
    i1_mask: BoolArray,
    channel_weights: ChannelWeights = None,
) -> ExperimentalMethodResult:
    """Run direct MLP-based parameter prediction on one experiment."""
    features, channel_mask, used_interpolation = _build_ml_features(d_exp, i3_exp, i1_exp, i3_mask, i1_mask)
    start_time = time.perf_counter()
    prediction = model.predict(features)[0]
    runtime_seconds = time.perf_counter() - start_time
    clipped_prediction, used_clipping = _clip_parameter_vector_to_bounds(
        np.asarray(prediction, dtype=np.float64),
        DEFAULT_BOUNDS,
    )
    message = ""
    if used_interpolation:
        message = "Missing points inside an available channel were linearly interpolated for MLP inference."
    if used_clipping:
        clipping_note = "Direct MLP output was clipped to the physical bounds used by the project."
        message = f"{message} {clipping_note}".strip()

    return _build_result(
        method_name="ml",
        parameter_vector=clipped_prediction,
        d_exp=d_exp,
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        lambda_m=lambda_m,
        normalization_strategy=normalization_strategy,
        runtime_seconds=runtime_seconds,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
        channel_mask=channel_mask,
        used_interpolation=used_interpolation,
        message=message,
        channel_weights=channel_weights,
    )


def run_natural_inverse_method(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    normalization_strategy: NormalizationStrategy,
    i3_mask: BoolArray,
    i1_mask: BoolArray,
    seed: Optional[int] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    channel_weights: ChannelWeights = None,
) -> ExperimentalMethodResult:
    """Run natural-computation SHG inversion on one experiment."""
    start_time = time.perf_counter()
    fit_result = run_natural_fit(
        d_exp=d_exp,
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        lambda_m=lambda_m,
        bounds=bounds,
        normalization_strategy=normalization_strategy,
        seed=seed,
        verbose=False,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
        channel_weights=channel_weights,
    )
    runtime_seconds = time.perf_counter() - start_time
    return _build_result(
        method_name="natural",
        parameter_vector=fit_result.parameter_vector,
        d_exp=d_exp,
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        lambda_m=lambda_m,
        normalization_strategy=normalization_strategy,
        runtime_seconds=runtime_seconds,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
        channel_mask=(_channel_observed(i3_mask), _channel_observed(i1_mask)),
        used_interpolation=False,
        message=fit_result.message,
        channel_weights=channel_weights,
    )


def run_hybrid_inverse_method(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    model: MLPRegressor,
    normalization_strategy: NormalizationStrategy,
    i3_mask: BoolArray,
    i1_mask: BoolArray,
    local_bounds_mode: LocalBoundsMode = "neighborhood",
    neighborhood_fraction: float = 0.1,
    channel_weights: ChannelWeights = None,
) -> ExperimentalMethodResult:
    """Run MLP initialization followed by bounded physical local refinement."""
    features, channel_mask, used_interpolation = _build_ml_features(d_exp, i3_exp, i1_exp, i3_mask, i1_mask)
    start_time = time.perf_counter()
    initial_guess = model.predict(features)[0]
    local_bounds = _compute_local_bounds(
        initial_guess=np.asarray(initial_guess, dtype=np.float64),
        global_bounds=DEFAULT_BOUNDS,
        mode=local_bounds_mode,
        neighborhood_fraction=neighborhood_fraction,
    )
    fit_result = refine_fit_locally(
        d_exp=d_exp,
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        lambda_m=lambda_m,
        initial_guess=np.asarray(initial_guess, dtype=np.float64),
        bounds=local_bounds,
        normalization_strategy=normalization_strategy,
        verbose=False,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
        channel_weights=channel_weights,
    )
    runtime_seconds = time.perf_counter() - start_time
    message = fit_result.message
    if used_interpolation:
        interpolation_note = "Missing points inside an available channel were linearly interpolated for MLP initialization."
        message = f"{interpolation_note} {message}".strip()

    return _build_result(
        method_name="hybrid",
        parameter_vector=fit_result.parameter_vector,
        d_exp=d_exp,
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        lambda_m=lambda_m,
        normalization_strategy=normalization_strategy,
        runtime_seconds=runtime_seconds,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
        channel_mask=channel_mask,
        used_interpolation=used_interpolation,
        message=message,
        channel_weights=channel_weights,
    )


def compare_experimental_methods(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    normalization_strategy: NormalizationStrategy,
    i3_mask: BoolArray,
    i1_mask: BoolArray,
    model: Optional[MLPRegressor] = None,
    seed: Optional[int] = None,
    local_bounds_mode: LocalBoundsMode = "neighborhood",
    neighborhood_fraction: float = 0.1,
    bounds: Optional[list[tuple[float, float]]] = None,
    channel_weights: ChannelWeights = None,
) -> ExperimentalComparisonReport:
    """Compare the inverse methods on one experimental SHG sample."""
    if model is None:
        raise ValueError("A trained model is required to compare ML and hybrid inverse methods.")

    results = {
        "classical": run_classical_inverse_method(
            d_exp=d_exp,
            i3_exp=i3_exp,
            i1_exp=i1_exp,
            lambda_m=lambda_m,
            normalization_strategy=normalization_strategy,
            i3_mask=i3_mask,
            i1_mask=i1_mask,
            seed=seed,
            bounds=bounds,
            channel_weights=channel_weights,
        ),
        "natural": run_natural_inverse_method(
            d_exp=d_exp,
            i3_exp=i3_exp,
            i1_exp=i1_exp,
            lambda_m=lambda_m,
            normalization_strategy=normalization_strategy,
            i3_mask=i3_mask,
            i1_mask=i1_mask,
            seed=seed,
            bounds=bounds,
            channel_weights=channel_weights,
        ),
        "ml": run_ml_inverse_method(
            d_exp=d_exp,
            i3_exp=i3_exp,
            i1_exp=i1_exp,
            lambda_m=lambda_m,
            model=model,
            normalization_strategy=normalization_strategy,
            i3_mask=i3_mask,
            i1_mask=i1_mask,
            channel_weights=channel_weights,
        ),
        "hybrid": run_hybrid_inverse_method(
            d_exp=d_exp,
            i3_exp=i3_exp,
            i1_exp=i1_exp,
            lambda_m=lambda_m,
            model=model,
            normalization_strategy=normalization_strategy,
            i3_mask=i3_mask,
            i1_mask=i1_mask,
            local_bounds_mode=local_bounds_mode,
            neighborhood_fraction=neighborhood_fraction,
            channel_weights=channel_weights,
        ),
    }
    best_method_name = min(results, key=lambda method_name: results[method_name].objective_error)
    return ExperimentalComparisonReport(results=results, best_method_name=best_method_name)


def save_experimental_comparison_summary(
    report: ExperimentalComparisonReport,
    output_path: str | Path,
) -> Path:
    """Save the experimental inverse-method summary as JSON."""
    output = Path(output_path)
    ensure_directory(output.parent)
    output.write_text(json.dumps(report.summary_dict(), indent=2), encoding="utf-8")
    return output


def save_experimental_method_summary(
    result: ExperimentalMethodResult,
    output_path: str | Path,
) -> Path:
    """Save the summary of one experimental inverse-method result as JSON."""
    output = Path(output_path)
    ensure_directory(output.parent)
    output.write_text(json.dumps(result.summary_dict(), indent=2), encoding="utf-8")
    return output
