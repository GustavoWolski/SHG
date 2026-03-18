"""Plotting helpers for SHG analysis."""

import numpy as np
import numpy.typing as npt

from src.inverse.objective import error_function

FloatArray = npt.NDArray[np.float64]


def plot_shg_curves(d_nm: FloatArray, i3: FloatArray, i1: FloatArray) -> None:
    """Plot normalized transmitted and reflected SHG curves."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate SHG plots.") from exc

    max_gen = max(i3.max(), i1.max())
    i3n = i3 / max_gen if max_gen > 0 else i3
    i1n = i1 / max_gen if max_gen > 0 else i1

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(d_nm, i3n, "-k")
    axes[0].set_ylabel("T (norm)")
    axes[0].grid(True)

    axes[1].plot(d_nm, i1n, "-k")
    axes[1].set_xlabel("d (nm)")
    axes[1].set_ylabel("R (norm)")
    axes[1].grid(True)

    fig.suptitle("SHG sim (T/R)")
    plt.tight_layout()
    plt.show()


def plot_error_map(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    n22_fixed: float = 2.8,
    k22_fixed: float = 0.4,
) -> None:
    """Plot the error map varying n21w and k21w."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate SHG plots.") from exc

    n_vals = np.linspace(3.5, 7.0, 80)
    k_vals = np.linspace(0.0, 1.0, 80)

    error_map = np.zeros((len(k_vals), len(n_vals)))

    for i, k in enumerate(k_vals):
        for j, n in enumerate(n_vals):
            x = [n, k, n22_fixed, k22_fixed]
            error_map[i, j] = error_function(
                x,
                d_exp,
                i3_exp,
                i1_exp,
                lambda_m,
            )

    plt.figure(figsize=(7, 6))
    plt.imshow(
        error_map,
        origin="lower",
        extent=[n_vals.min(), n_vals.max(), k_vals.min(), k_vals.max()],
        aspect="auto",
    )
    plt.colorbar(label="Erro")
    plt.xlabel("n21w")
    plt.ylabel("k21w")
    plt.title("Mapa do Erro")
    plt.show()
