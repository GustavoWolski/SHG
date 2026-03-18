"""Fitting routines for SHG parameters."""

import numpy as np
import numpy.typing as npt
from typing import Any

from src.inverse.objective import error_function

FloatArray = npt.NDArray[np.float64]

DEFAULT_BOUNDS: list[tuple[float, float]] = [
    (3.0, 7.0),
    (0.0, 1.0),
    (2.0, 5.0),
    (0.0, 1.0),
]


def run_fit(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    bounds: list[tuple[float, float]] | None = None,
) -> Any:
    """Run differential evolution for SHG parameters."""
    try:
        from scipy.optimize import differential_evolution
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scipy is required to run the inverse fitting routine.") from exc

    result = differential_evolution(
        error_function,
        bounds=bounds or DEFAULT_BOUNDS,
        args=(d_exp, i3_exp, i1_exp, lambda_m),
        strategy="best1bin",
        maxiter=100,
        popsize=20,
        polish=True,
    )

    print("\n=== RESULTADO DO FIT ===")
    print(f"n21w = {result.x[0]:.4f}")
    print(f"k21w = {result.x[1]:.4f}")
    print(f"n22w = {result.x[2]:.4f}")
    print(f"k22w = {result.x[3]:.4f}")
    print(f"Erro final = {result.fun:.6f}")

    return result
