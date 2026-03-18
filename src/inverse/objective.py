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


def normalize_shg_curves(
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    i3_sim: FloatArray,
    i1_sim: FloatArray,
    strategy: NormalizationStrategy = "global",
) -> Optional[tuple[FloatArray, FloatArray, FloatArray, FloatArray]]:
    """Normalize experimental and simulated SHG curves for error evaluation."""
    if strategy == "global":
        sim_scale = _safe_channel_scale(np.array([np.max(i3_sim), np.max(i1_sim)], dtype=np.float64))
        exp_scale = _safe_channel_scale(np.array([np.max(i3_exp), np.max(i1_exp)], dtype=np.float64))
        if sim_scale is None or exp_scale is None:
            return None
        return i3_exp / exp_scale, i1_exp / exp_scale, i3_sim / sim_scale, i1_sim / sim_scale

    if strategy == "separate":
        i3_exp_scale = _safe_channel_scale(i3_exp)
        i1_exp_scale = _safe_channel_scale(i1_exp)
        i3_sim_scale = _safe_channel_scale(i3_sim)
        i1_sim_scale = _safe_channel_scale(i1_sim)
        if None in (i3_exp_scale, i1_exp_scale, i3_sim_scale, i1_sim_scale):
            return None
        return (
            i3_exp / i3_exp_scale,
            i1_exp / i1_exp_scale,
            i3_sim / i3_sim_scale,
            i1_sim / i1_sim_scale,
        )

    raise ValueError(f"Unknown normalization strategy: {strategy}")


def error_function(
    x: Sequence[float],
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    normalization_strategy: NormalizationStrategy = "global",
) -> float:
    """Compute the inverse-fitting error for SHG transmission and reflection."""
    params = build_shg_params(x, lambda_m)

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
    )
    if normalized_curves is None:
        return LARGE_ERROR_PENALTY

    i3_exp_norm, i1_exp_norm, i3_sim_norm, i1_sim_norm = normalized_curves
    transmission_error = np.mean((i3_exp_norm - i3_sim_norm) ** 2)
    reflection_error = np.mean((i1_exp_norm - i1_sim_norm) ** 2)
    return float(transmission_error + reflection_error)
