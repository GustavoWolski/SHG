"""Objective functions for SHG fitting."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.physics.shg_model import SHGParams, simulate_shg

FloatArray = npt.NDArray[np.float64]


def error_function(
    x: Sequence[float],
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
) -> float:
    """Compute the normalized SHG fitting error."""
    n21w = complex(x[0], x[1])
    n22w = complex(x[2], x[3])

    params = SHGParams(
        lambda_m=lambda_m,
        n21w=n21w,
        n22w=n22w,
    )

    i3_sim, i1_sim = simulate_shg(params, d_exp)

    max_sim = max(i3_sim.max(), i1_sim.max())
    max_exp = max(i3_exp.max(), i1_exp.max())

    if max_sim == 0 or max_exp == 0:
        return 1e9

    i3_sim = i3_sim / max_sim
    i1_sim = i1_sim / max_sim
    i3_exp_n = i3_exp / max_exp
    i1_exp_n = i1_exp / max_exp

    erro_t = np.mean((i3_exp_n - i3_sim) ** 2)
    erro_r = np.mean((i1_exp_n - i1_sim) ** 2)

    return float(erro_t + erro_r)
