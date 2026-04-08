"""Objective functions for SHG inverse fitting.

This module keeps the inverse-problem logic simple: the optimizer proposes
material parameters, the direct SHG model simulates the curves, and the
objective function measures the mismatch against the experimental data.
"""

from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from src.physics.shg_model import SHGParams, simulate_shg

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
NormalizationStrategy = Literal["global", "separate"]
LARGE_ERROR_PENALTY: float = 1e9


def build_shg_params(parameter_vector: Sequence[float], lambda_m: float) -> SHGParams:
    """Build SHG parameters from the optimizer vector."""
    return SHGParams(
        lambda_m=lambda_m,
        n21w=complex(parameter_vector[0], parameter_vector[1]),
        n22w=complex(parameter_vector[2], parameter_vector[3]),
    )


def _safe_channel_scale(values: FloatArray) -> Optional[float]:
    """Return a valid normalization scale for one curve."""
    scale = float(np.max(values))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def _resolve_observation_mask(values: FloatArray, mask: Optional[BoolArray]) -> BoolArray:
    """Build a boolean mask for the experimentally observed points."""
    finite_mask = np.isfinite(values)
    if mask is None:
        return finite_mask

    observation_mask = np.asarray(mask, dtype=bool)
    if observation_mask.shape != values.shape:
        raise ValueError("Observation masks must match the curve shape.")
    return observation_mask & finite_mask


def normalize_shg_curves(
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    i3_sim: FloatArray,
    i1_sim: FloatArray,
    strategy: NormalizationStrategy = "global",
    i3_mask: Optional[BoolArray] = None,
    i1_mask: Optional[BoolArray] = None,
) -> Optional[tuple[FloatArray, FloatArray, FloatArray, FloatArray]]:
    """Normalize SHG curves using only experimentally observed points."""
    observed_i3_mask = _resolve_observation_mask(i3_exp, i3_mask)
    observed_i1_mask = _resolve_observation_mask(i1_exp, i1_mask)

    if strategy == "global":
        experimental_values: list[FloatArray] = []
        simulated_values: list[FloatArray] = []
        if np.any(observed_i3_mask):
            experimental_values.append(np.asarray(i3_exp[observed_i3_mask], dtype=np.float64))
            simulated_values.append(np.asarray(i3_sim[observed_i3_mask], dtype=np.float64))
        if np.any(observed_i1_mask):
            experimental_values.append(np.asarray(i1_exp[observed_i1_mask], dtype=np.float64))
            simulated_values.append(np.asarray(i1_sim[observed_i1_mask], dtype=np.float64))
        if not experimental_values or not simulated_values:
            return None

        sim_scale = _safe_channel_scale(np.concatenate(simulated_values))
        exp_scale = _safe_channel_scale(np.concatenate(experimental_values))
        if sim_scale is None or exp_scale is None:
            return None
        return i3_exp / exp_scale, i1_exp / exp_scale, i3_sim / sim_scale, i1_sim / sim_scale

    if strategy == "separate":
        i3_exp_scale = _safe_channel_scale(i3_exp[observed_i3_mask]) if np.any(observed_i3_mask) else 1.0
        i1_exp_scale = _safe_channel_scale(i1_exp[observed_i1_mask]) if np.any(observed_i1_mask) else 1.0
        i3_sim_scale = _safe_channel_scale(i3_sim[observed_i3_mask]) if np.any(observed_i3_mask) else 1.0
        i1_sim_scale = _safe_channel_scale(i1_sim[observed_i1_mask]) if np.any(observed_i1_mask) else 1.0
        if None in (i3_exp_scale, i1_exp_scale, i3_sim_scale, i1_sim_scale):
            return None
        return (
            i3_exp / i3_exp_scale,
            i1_exp / i1_exp_scale,
            i3_sim / i3_sim_scale,
            i1_sim / i1_sim_scale,
        )

    raise ValueError(f"Unknown normalization strategy: {strategy}")


ChannelWeights = Optional[tuple[float, float]]


def error_function(
    x: Sequence[float],
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    normalization_strategy: NormalizationStrategy = "global",
    i3_mask: Optional[BoolArray] = None,
    i1_mask: Optional[BoolArray] = None,
    channel_weights: ChannelWeights = None,
) -> float:
    """Compute the fitting error, ignoring missing experimental samples via masks.

    When *channel_weights* is supplied as ``(w_i3, w_i1)``, each channel
    MSE is multiplied by the corresponding weight before summation.  The
    default ``None`` is equivalent to ``(1.0, 1.0)``.
    """
    params = build_shg_params(x, lambda_m)
    w_i3, w_i1 = channel_weights if channel_weights is not None else (1.0, 1.0)

    try:
        i3_sim, i1_sim = simulate_shg(params, d_exp)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return LARGE_ERROR_PENALTY

    normalized_curves = normalize_shg_curves(
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        i3_sim=i3_sim,
        i1_sim=i1_sim,
        strategy=normalization_strategy,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
    )
    if normalized_curves is None:
        return LARGE_ERROR_PENALTY

    i3_exp_norm, i1_exp_norm, i3_sim_norm, i1_sim_norm = normalized_curves
    observed_i3_mask = _resolve_observation_mask(i3_exp, i3_mask)
    observed_i1_mask = _resolve_observation_mask(i1_exp, i1_mask)

    channel_errors: list[float] = []
    if np.any(observed_i3_mask):
        transmission_error = np.mean((i3_exp_norm[observed_i3_mask] - i3_sim_norm[observed_i3_mask]) ** 2)
        channel_errors.append(float(w_i3 * transmission_error))
    if np.any(observed_i1_mask):
        reflection_error = np.mean((i1_exp_norm[observed_i1_mask] - i1_sim_norm[observed_i1_mask]) ** 2)
        channel_errors.append(float(w_i1 * reflection_error))
    if not channel_errors:
        return LARGE_ERROR_PENALTY
    return float(sum(channel_errors))
