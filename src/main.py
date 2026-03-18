"""CLI entrypoint for SHG simulation and inverse fitting."""

import argparse

import numpy as np
import numpy.typing as npt

from src.inverse.fitters import run_fit
from src.physics.shg_model import SHGParams, simulate_shg
from src.utils.plotting import plot_error_map, plot_shg_curves

FloatArray = npt.NDArray[np.float64]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="SHG sim (adaptado do gerador em MATLAB).")
    parser.add_argument("--lambda-nm", type=float, default=1560.0, help="Comprimento de onda (nm).")
    parser.add_argument("--n21w", type=float, default=5.6428, help="Parte real de n21w.")
    parser.add_argument("--k21w", type=float, default=0.0849, help="Parte imag de n21w.")
    parser.add_argument("--n22w", type=float, default=2.8698, help="Parte real de n22w.")
    parser.add_argument("--k22w", type=float, default=0.4492, help="Parte imag de n22w.")
    parser.add_argument("--d-max-nm", type=float, default=600.0, help="Espessura maxima (nm).")
    parser.add_argument("--d-step-nm", type=float, default=1.0, help="Passo de espessura (nm).")
    return parser


def sample_experimental_data() -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Return the current sample experimental data."""
    d_exp = np.array([0, 50, 100, 150, 200], dtype=np.float64)
    i3_exp = np.array([0.1, 0.8, 0.3, 1.0, 0.4], dtype=np.float64)
    i1_exp = np.array([0.5, 0.2, 0.7, 0.1, 0.6], dtype=np.float64)
    lambda_m = 1560e-9
    return d_exp, i3_exp, i1_exp, lambda_m


def main() -> None:
    """Run the SHG command-line workflow."""
    parser = build_parser()
    args = parser.parse_args()

    d_nm = np.arange(0.0, args.d_max_nm + args.d_step_nm, args.d_step_nm)
    params = SHGParams(
        lambda_m=args.lambda_nm * 1e-9,
        n21w=complex(args.n21w, args.k21w),
        n22w=complex(args.n22w, args.k22w),
    )

    i3, i1 = simulate_shg(params, d_nm)
    plot_shg_curves(d_nm, i3, i1)

    d_exp, i3_exp, i1_exp, lambda_m = sample_experimental_data()
    result = run_fit(d_exp, i3_exp, i1_exp, lambda_m)
    plot_error_map(
        d_exp,
        i3_exp,
        i1_exp,
        lambda_m,
        n22_fixed=float(result.x[2]),
        k22_fixed=float(result.x[3]),
    )


if __name__ == "__main__":
    main()
