"""Inverse fitting routines for SHG.

The fitting workflow solves an inverse problem: a direct physical model
simulates SHG curves for a candidate parameter set, and an optimizer searches
the parameter space for the lowest mismatch with the experiment.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from src.inverse.objective import ChannelWeights, NormalizationStrategy, build_shg_params, error_function, residual_vector
from src.physics.shg_model import SHGParams, simulate_shg
from src.data.synthetic_generator import DEFAULT_PARAMETER_BOUNDS, PARAMETER_NAMES

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

DEFAULT_BOUNDS: list[tuple[float, float]] = [
    DEFAULT_PARAMETER_BOUNDS[name] for name in PARAMETER_NAMES
]


@dataclass
class FitResult:
    """Result of SHG inverse fitting with an optimizer."""

    fitted_params: SHGParams
    final_error: float
    success: bool
    n_evaluations: int
    parameter_vector: FloatArray
    normalization_strategy: NormalizationStrategy
    seed: Optional[int]
    message: str
    raw_result: Any
    optimizer_name: str


def simulate_fit_result(fit_result: FitResult, d_nm: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Simulate SHG curves using the best-fit parameters."""
    thickness_nm = np.asarray(d_nm, dtype=np.float64)
    return simulate_shg(fit_result.fitted_params, thickness_nm)


def _build_fit_result(
    optimizer_result: Any,
    lambda_m: float,
    normalization_strategy: NormalizationStrategy,
    seed: Optional[int],
    optimizer_name: str,
    final_error: Optional[float] = None,
) -> FitResult:
    """Convert the raw optimizer output into a typed fitting result."""
    parameter_vector = np.asarray(optimizer_result.x, dtype=np.float64)
    fitted_params = build_shg_params(parameter_vector, lambda_m)
    return FitResult(
        fitted_params=fitted_params,
        final_error=float(optimizer_result.fun if final_error is None else final_error),
        success=bool(getattr(optimizer_result, "success", False)),
        n_evaluations=int(getattr(optimizer_result, "nfev", 0)),
        parameter_vector=parameter_vector,
        normalization_strategy=normalization_strategy,
        seed=seed,
        message=str(getattr(optimizer_result, "message", "")),
        raw_result=optimizer_result,
        optimizer_name=optimizer_name,
    )


def print_fit_summary(fit_result: FitResult) -> None:
    """Print a compact summary of the inverse-fitting result."""
    print("\n=== RESULTADO DO FIT ===")
    print(f"Otimizador = {fit_result.optimizer_name}")
    print(f"n21w = {fit_result.fitted_params.n21w.real:.4f}")
    print(f"k21w = {fit_result.fitted_params.n21w.imag:.4f}")
    print(f"n22w = {fit_result.fitted_params.n22w.real:.4f}")
    print(f"k22w = {fit_result.fitted_params.n22w.imag:.4f}")
    print(f"Erro final = {fit_result.final_error:.6f}")
    print(f"Sucesso = {fit_result.success}")
    print(f"Avaliacoes = {fit_result.n_evaluations}")
    print(f"Normalizacao = {fit_result.normalization_strategy}")
    print(f"Seed = {fit_result.seed}")


def _candidate_from_fractions(
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    fractions: tuple[float, float, float, float],
) -> FloatArray:
    """Build one bounded initial guess from per-parameter interval fractions."""
    fraction_vector = np.asarray(fractions, dtype=np.float64)
    return lower_bounds + fraction_vector * (upper_bounds - lower_bounds)


def _choose_least_squares_initial_guess(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    normalization_strategy: NormalizationStrategy,
    i3_mask: Optional[BoolArray],
    i1_mask: Optional[BoolArray],
    channel_weights: ChannelWeights,
) -> FloatArray:
    """Choose a non-degenerate deterministic starting point for local fitting."""
    candidate_fractions = [
        (0.50, 0.50, 0.50, 0.50),
        (0.35, 0.25, 0.65, 0.25),
        (0.65, 0.25, 0.35, 0.25),
        (0.25, 0.10, 0.75, 0.40),
        (0.75, 0.40, 0.25, 0.10),
    ]
    candidates = [
        _candidate_from_fractions(lower_bounds, upper_bounds, fractions)
        for fractions in candidate_fractions
    ]
    errors = [
        error_function(
            candidate,
            d_exp,
            i3_exp,
            i1_exp,
            lambda_m,
            normalization_strategy,
            i3_mask,
            i1_mask,
            channel_weights,
        )
        for candidate in candidates
    ]
    best_index = int(np.argmin(errors))
    return candidates[best_index]


def run_fit(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    bounds: Optional[list[tuple[float, float]]] = None,
    normalization_strategy: NormalizationStrategy = "global",
    seed: Optional[int] = None,
    verbose: bool = True,
    i3_mask: Optional[BoolArray] = None,
    i1_mask: Optional[BoolArray] = None,
    channel_weights: ChannelWeights = None,
) -> FitResult:
    """Run least squares as the classical local SHG inverse solver."""
    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scipy is required to run the inverse fitting routine.") from exc

    local_bounds = bounds or DEFAULT_BOUNDS
    lower_bounds = np.asarray([lower for lower, _ in local_bounds], dtype=np.float64)
    upper_bounds = np.asarray([upper for _, upper in local_bounds], dtype=np.float64)
    initial_guess = _choose_least_squares_initial_guess(
        d_exp=d_exp,
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        lambda_m=lambda_m,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        normalization_strategy=normalization_strategy,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
        channel_weights=channel_weights,
    )

    optimizer_result = least_squares(
        residual_vector,
        x0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        args=(d_exp, i3_exp, i1_exp, lambda_m, normalization_strategy, i3_mask, i1_mask, channel_weights),
        max_nfev=500,
    )
    final_error = error_function(
        optimizer_result.x,
        d_exp,
        i3_exp,
        i1_exp,
        lambda_m,
        normalization_strategy,
        i3_mask,
        i1_mask,
        channel_weights,
    )

    fit_result = _build_fit_result(
        optimizer_result=optimizer_result,
        lambda_m=lambda_m,
        normalization_strategy=normalization_strategy,
        seed=None,
        optimizer_name="least_squares",
        final_error=final_error,
    )
    if verbose:
        print_fit_summary(fit_result)
    return fit_result


def run_natural_fit(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    bounds: Optional[list[tuple[float, float]]] = None,
    normalization_strategy: NormalizationStrategy = "global",
    seed: Optional[int] = None,
    verbose: bool = True,
    i3_mask: Optional[BoolArray] = None,
    i1_mask: Optional[BoolArray] = None,
    channel_weights: ChannelWeights = None,
) -> FitResult:
    """Run differential evolution as a natural-computation SHG inverse solver."""
    try:
        from scipy.optimize import differential_evolution
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scipy is required to run the inverse fitting routine.") from exc

    optimizer_result = differential_evolution(
        error_function,
        bounds=bounds or DEFAULT_BOUNDS,
        args=(d_exp, i3_exp, i1_exp, lambda_m, normalization_strategy, i3_mask, i1_mask, channel_weights),
        strategy="best1bin",
        maxiter=100,
        popsize=20,
        polish=True,
        seed=seed,
    )

    fit_result = _build_fit_result(
        optimizer_result=optimizer_result,
        lambda_m=lambda_m,
        normalization_strategy=normalization_strategy,
        seed=seed,
        optimizer_name="differential_evolution",
    )
    if verbose:
        print_fit_summary(fit_result)
    return fit_result


def refine_fit_locally(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    initial_guess: FloatArray,
    bounds: Optional[list[tuple[float, float]]] = None,
    normalization_strategy: NormalizationStrategy = "global",
    verbose: bool = False,
    i3_mask: Optional[BoolArray] = None,
    i1_mask: Optional[BoolArray] = None,
    channel_weights: ChannelWeights = None,
) -> FitResult:
    """Refine SHG parameters locally starting from an informed initial guess."""
    try:
        from scipy.optimize import minimize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scipy is required to run the local refinement routine.") from exc

    local_bounds = bounds or DEFAULT_BOUNDS
    clipped_initial_guess = np.array(
        [np.clip(initial_guess[index], local_bounds[index][0], local_bounds[index][1]) for index in range(len(local_bounds))],
        dtype=np.float64,
    )

    optimizer_result = minimize(
        error_function,
        x0=clipped_initial_guess,
        args=(d_exp, i3_exp, i1_exp, lambda_m, normalization_strategy, i3_mask, i1_mask, channel_weights),
        method="L-BFGS-B",
        bounds=local_bounds,
    )

    fit_result = _build_fit_result(
        optimizer_result=optimizer_result,
        lambda_m=lambda_m,
        normalization_strategy=normalization_strategy,
        seed=None,
        optimizer_name="L-BFGS-B",
    )
    if verbose:
        print_fit_summary(fit_result)
    return fit_result
