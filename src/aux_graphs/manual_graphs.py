from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_experimental_shg_data
from src.inverse.objective import normalize_shg_curves
from src.physics.shg_model import SHGParams, simulate_shg


def _compute_channel_metrics(exp_values, sim_values, mask) -> dict[str, float]:
    observed_mask = mask.astype(bool)
    residuals = exp_values[observed_mask] - sim_values[observed_mask]
    mse = float((residuals ** 2).mean())
    mae = float(abs(residuals).mean())
    rmse = float(mse ** 0.5)
    max_abs_error = float(abs(residuals).max())
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
    }


def main() -> None:
    experimental = load_experimental_shg_data(
        PROJECT_ROOT / "src" / "data" / "experimental_fit.csv",
        delimiter=",",
        skiprows=0,
    )

    params = SHGParams(
        lambda_m=1560e-9,
        n21w=complex(2.04, 0.7),
        n22w=complex(1.42, 0.8),
    )

    i3_sim, i1_sim = simulate_shg(params, experimental.d_nm)
    normalized = normalize_shg_curves(
        i3_exp=experimental.i3,
        i1_exp=experimental.i1,
        i3_sim=i3_sim,
        i1_sim=i1_sim,
        strategy="global",
        i3_mask=experimental.i3_mask,
        i1_mask=experimental.i1_mask,
    )
    if normalized is None:
        raise RuntimeError("Falha ao normalizar curvas para comparacao.")

    i3_exp_norm, i1_exp_norm, i3_sim_norm, i1_sim_norm = normalized

    i3_metrics = _compute_channel_metrics(i3_exp_norm, i3_sim_norm, experimental.i3_mask)
    i1_metrics = _compute_channel_metrics(i1_exp_norm, i1_sim_norm, experimental.i1_mask)
    objective_error = i3_metrics["mse"] + i1_metrics["mse"]
    mean_channel_mse = 0.5 * objective_error

    output_path = PROJECT_ROOT / "outputs" / "valores_idealizados.png"
    summary_path = PROJECT_ROOT / "outputs" / "valores_idealizados_error_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(experimental.d_nm[experimental.i3_mask], i3_exp_norm[experimental.i3_mask], "ok", label="Exp")
    axes[0].plot(experimental.d_nm, i3_sim_norm, "-r", linewidth=1.8, label="Sim")
    axes[0].set_ylabel("T - i3 (norm)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(experimental.d_nm[experimental.i1_mask], i1_exp_norm[experimental.i1_mask], "ok", label="Exp")
    axes[1].plot(experimental.d_nm, i1_sim_norm, "-b", linewidth=1.8, label="Sim")
    axes[1].set_xlabel("d (nm)")
    axes[1].set_ylabel("R - i1 (norm)")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Experimental vs simulacao manual | norm=global")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "normalization": "global",
        "parameters": {
            "lambda_nm": 1560.0,
            "n21w": 2.04,
            "k21w": 0.7,
            "n22w": 1.42,
            "k22w": 0.8,
        },
        "metrics": {
            "objective_error": objective_error,
            "mean_channel_mse": mean_channel_mse,
            "i3": i3_metrics,
            "i1": i1_metrics,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(output_path)
    print(summary_path)
    print(f"objective_error = {objective_error:.6f}")
    print(f"mean_channel_mse = {mean_channel_mse:.6f}")
    print(
        "i3 -> "
        f"mse={i3_metrics['mse']:.6f}, "
        f"mae={i3_metrics['mae']:.6f}, "
        f"rmse={i3_metrics['rmse']:.6f}, "
        f"max_abs_error={i3_metrics['max_abs_error']:.6f}"
    )
    print(
        "i1 -> "
        f"mse={i1_metrics['mse']:.6f}, "
        f"mae={i1_metrics['mae']:.6f}, "
        f"rmse={i1_metrics['rmse']:.6f}, "
        f"max_abs_error={i1_metrics['max_abs_error']:.6f}"
    )


if __name__ == "__main__":
    main()