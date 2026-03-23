"""Plotting helpers for SHG analysis."""

from typing import Any

import numpy as np
import numpy.typing as npt

from src.inverse.fitters import FitResult, simulate_fit_result
from src.inverse.methods import ExperimentalMethodResult
from src.inverse.objective import NormalizationStrategy, error_function, normalize_shg_curves

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _show_or_close_figure(plt: Any, figure: Any) -> None:
    """Show figures only on interactive backends and always release resources."""
    backend_name = str(plt.get_backend()).lower()
    if "agg" not in backend_name:
        plt.show()
    plt.close(figure)


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
    _show_or_close_figure(plt, fig)


def plot_error_map(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    lambda_m: float,
    n22_fixed: float = 2.8,
    k22_fixed: float = 0.4,
    normalization_strategy: NormalizationStrategy = "global",
    i3_mask: BoolArray | None = None,
    i1_mask: BoolArray | None = None,
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
                normalization_strategy=normalization_strategy,
                i3_mask=i3_mask,
                i1_mask=i1_mask,
            )

    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(
        error_map,
        origin="lower",
        extent=[n_vals.min(), n_vals.max(), k_vals.min(), k_vals.max()],
        aspect="auto",
    )
    fig.colorbar(image, ax=axis, label="Erro")
    axis.set_xlabel("n21w")
    axis.set_ylabel("k21w")
    axis.set_title("Mapa do Erro")
    _show_or_close_figure(plt, fig)


def plot_fit_comparison(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    fit_result: FitResult,
    i3_mask: BoolArray | None = None,
    i1_mask: BoolArray | None = None,
) -> None:
    """Plot experimental and best-fit simulated SHG curves."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate SHG plots.") from exc

    i3_sim, i1_sim = simulate_fit_result(fit_result, d_exp)
    normalized_curves = normalize_shg_curves(
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        i3_sim=i3_sim,
        i1_sim=i1_sim,
        strategy=fit_result.normalization_strategy,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
    )
    if normalized_curves is None:
        raise ValueError("Could not normalize the experimental and fitted curves for plotting.")

    i3_exp_norm, i1_exp_norm, i3_sim_norm, i1_sim_norm = normalized_curves
    observed_i3_mask = np.isfinite(i3_exp) if i3_mask is None else np.asarray(i3_mask, dtype=bool) & np.isfinite(i3_exp)
    observed_i1_mask = np.isfinite(i1_exp) if i1_mask is None else np.asarray(i1_mask, dtype=bool) & np.isfinite(i1_exp)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    if np.any(observed_i3_mask):
        axes[0].plot(d_exp[observed_i3_mask], i3_exp_norm[observed_i3_mask], "ok", label="Exp")
    axes[0].plot(d_exp, i3_sim_norm, "-r", label="Sim")
    axes[0].set_ylabel("T (norm)")
    axes[0].grid(True)
    axes[0].legend()

    if np.any(observed_i1_mask):
        axes[1].plot(d_exp[observed_i1_mask], i1_exp_norm[observed_i1_mask], "ok", label="Exp")
    axes[1].plot(d_exp, i1_sim_norm, "-b", label="Sim")
    axes[1].set_xlabel("d (nm)")
    axes[1].set_ylabel("R (norm)")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(
        "Melhor ajuste SHG "
        f"| erro={fit_result.final_error:.4e} "
        f"| norm={fit_result.normalization_strategy}"
    )
    plt.tight_layout()
    _show_or_close_figure(plt, fig)


def plot_inverse_method_comparison(
    d_exp: FloatArray,
    i3_exp: FloatArray,
    i1_exp: FloatArray,
    method_results: dict[str, ExperimentalMethodResult],
    i3_mask: BoolArray | None = None,
    i1_mask: BoolArray | None = None,
) -> None:
    """Plot experimental SHG points against multiple inverse-method reconstructions."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate SHG plots.") from exc

    observed_i3_mask = np.isfinite(i3_exp) if i3_mask is None else np.asarray(i3_mask, dtype=bool) & np.isfinite(i3_exp)
    observed_i1_mask = np.isfinite(i1_exp) if i1_mask is None else np.asarray(i1_mask, dtype=bool) & np.isfinite(i1_exp)
    if not method_results:
        raise ValueError("method_results must contain at least one inverse-method result.")

    reference_result = next(iter(method_results.values()))
    normalized_reference = normalize_shg_curves(
        i3_exp=i3_exp,
        i1_exp=i1_exp,
        i3_sim=reference_result.reconstructed_i3,
        i1_sim=reference_result.reconstructed_i1,
        strategy=reference_result.normalization_strategy,
        i3_mask=i3_mask,
        i1_mask=i1_mask,
    )
    if normalized_reference is None:
        raise ValueError("Could not normalize the experimental and reconstructed curves for plotting.")
    i3_exp_norm, i1_exp_norm, _, _ = normalized_reference

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    if np.any(observed_i3_mask):
        axes[0].plot(d_exp[observed_i3_mask], i3_exp_norm[observed_i3_mask], "ok", label="Exp")
    if np.any(observed_i1_mask):
        axes[1].plot(d_exp[observed_i1_mask], i1_exp_norm[observed_i1_mask], "ok", label="Exp")

    method_styles = {
        "classical": ("-", "black"),
        "ml": ("--", "royalblue"),
        "hybrid": (":", "darkorange"),
    }
    for method_name, result in method_results.items():
        normalized_curves = normalize_shg_curves(
            i3_exp=i3_exp,
            i1_exp=i1_exp,
            i3_sim=result.reconstructed_i3,
            i1_sim=result.reconstructed_i1,
            strategy=result.normalization_strategy,
            i3_mask=i3_mask,
            i1_mask=i1_mask,
        )
        if normalized_curves is None:
            continue
        _, _, i3_sim_norm, i1_sim_norm = normalized_curves
        line_style, color = method_styles.get(method_name, ("-", None))
        axes[0].plot(d_exp, i3_sim_norm, line_style, color=color, linewidth=1.8, label=method_name)
        axes[1].plot(d_exp, i1_sim_norm, line_style, color=color, linewidth=1.8, label=method_name)

    axes[0].set_ylabel("T (norm)")
    axes[0].set_title("Comparacao de metodos inversos em i3")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("d (nm)")
    axes[1].set_ylabel("R (norm)")
    axes[1].set_title("Comparacao de metodos inversos em i1")
    axes[1].grid(True)
    axes[1].legend()

    best_method_name = min(method_results, key=lambda method_name: method_results[method_name].objective_error)
    figure.suptitle(
        "Comparacao de metodos SHG "
        f"| melhor={best_method_name} "
        f"| erro={method_results[best_method_name].objective_error:.4e}"
    )
    plt.tight_layout()
    _show_or_close_figure(plt, figure)
