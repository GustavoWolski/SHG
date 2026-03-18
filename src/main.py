"""CLI entrypoint for SHG simulation, inversion and ML workflows."""

import argparse
from collections.abc import Callable
from typing import Optional

import numpy as np
import numpy.typing as npt

from src.data.synthetic_generator import DEFAULT_PARAMETER_BOUNDS
from src.data.loaders import load_synthetic_dataset
from src.ml.datasets import from_synthetic_dataset, subset_dataset
from src.ml.models import load_model
from src.physics.shg_model import SHGParams, simulate_shg
from src.utils.plotting import plot_error_map, plot_fit_comparison, plot_shg_curves

FloatArray = npt.NDArray[np.float64]
CommandHandler = Callable[[argparse.Namespace], None]


def build_simulate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the simulate subcommand."""
    parser = subparsers.add_parser("simulate", help="Executa a simulacao SHG.")
    parser.add_argument("--lambda-nm", type=float, default=1560.0, help="Comprimento de onda (nm).")
    parser.add_argument("--n21w", type=float, default=5.6428, help="Parte real de n21w.")
    parser.add_argument("--k21w", type=float, default=0.0849, help="Parte imag de n21w.")
    parser.add_argument("--n22w", type=float, default=2.8698, help="Parte real de n22w.")
    parser.add_argument("--k22w", type=float, default=0.4492, help="Parte imag de n22w.")
    parser.add_argument("--d-max-nm", type=float, default=600.0, help="Espessura maxima (nm).")
    parser.add_argument("--d-step-nm", type=float, default=1.0, help="Passo de espessura (nm).")
    parser.set_defaults(handler=handle_simulate)


def build_fit_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the fit subcommand."""
    parser = subparsers.add_parser("fit", help="Executa o ajuste inverso com dados internos.")
    parser.add_argument(
        "--normalization",
        choices=["global", "separate"],
        default="global",
        help="Estrategia de normalizacao do erro.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional para reprodutibilidade do otimizador.",
    )
    parser.set_defaults(handler=handle_fit)


def build_generate_dataset_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the synthetic dataset generation subcommand."""
    parser = subparsers.add_parser("generate-dataset", help="Gera dataset sintetico em formato NPZ.")
    parser.add_argument("--num-samples", type=int, default=100, help="Numero de amostras sinteticas.")
    parser.add_argument("--output", type=str, default="data/shg_synthetic_dataset.npz", help="Arquivo NPZ de saida.")
    parser.add_argument("--lambda-nm", type=float, default=1560.0, help="Comprimento de onda (nm).")
    parser.add_argument("--d-max-nm", type=float, default=600.0, help="Espessura maxima (nm).")
    parser.add_argument("--d-step-nm", type=float, default=1.0, help="Passo de espessura (nm).")
    parser.add_argument("--n21w-min", type=float, default=DEFAULT_PARAMETER_BOUNDS["n21w"][0], help="Minimo de n21w.")
    parser.add_argument("--n21w-max", type=float, default=DEFAULT_PARAMETER_BOUNDS["n21w"][1], help="Maximo de n21w.")
    parser.add_argument("--k21w-min", type=float, default=DEFAULT_PARAMETER_BOUNDS["k21w"][0], help="Minimo de k21w.")
    parser.add_argument("--k21w-max", type=float, default=DEFAULT_PARAMETER_BOUNDS["k21w"][1], help="Maximo de k21w.")
    parser.add_argument("--n22w-min", type=float, default=DEFAULT_PARAMETER_BOUNDS["n22w"][0], help="Minimo de n22w.")
    parser.add_argument("--n22w-max", type=float, default=DEFAULT_PARAMETER_BOUNDS["n22w"][1], help="Maximo de n22w.")
    parser.add_argument("--k22w-min", type=float, default=DEFAULT_PARAMETER_BOUNDS["k22w"][0], help="Minimo de k22w.")
    parser.add_argument("--k22w-max", type=float, default=DEFAULT_PARAMETER_BOUNDS["k22w"][1], help="Maximo de k22w.")
    parser.add_argument(
        "--normalization",
        choices=["none", "global", "separate"],
        default="none",
        help="Normalizacao opcional das curvas antes de salvar.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed opcional para reprodutibilidade.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Desabilita a barra simples de progresso no terminal.",
    )
    parser.set_defaults(handler=handle_generate_dataset)


def build_train_ml_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the MLP training subcommand."""
    parser = subparsers.add_parser("train-ml", help="Treina um modelo de ML a partir de dataset sintetico.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Arquivo NPZ com o dataset sintetico.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/shg_mlp.npz",
        help="Arquivo NPZ de saida para o modelo treinado.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="outputs/train_ml/training_summary.json",
        help="Arquivo JSON para salvar o resumo do treinamento.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Dimensoes das camadas ocultas da MLP.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Numero de epocas de treinamento.")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do mini-batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Taxa de aprendizado.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Regularizacao L2.")
    parser.add_argument("--gradient-clip", type=float, default=5.0, help="Clip global do gradiente.")
    parser.add_argument("--seed", type=int, default=None, help="Seed opcional para reprodutibilidade.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe o log resumido do treinamento ao longo das epocas.",
    )
    parser.set_defaults(handler=handle_train_ml)


def build_evaluate_ml_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ML evaluation subcommand."""
    parser = subparsers.add_parser("evaluate-ml", help="Avalia um modelo de ML em dataset de teste.")
    parser.add_argument("--model-path", type=str, required=True, help="Arquivo NPZ com o modelo treinado.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Arquivo NPZ com o dataset de teste.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluate_ml",
        help="Diretorio para salvar metricas e figuras.",
    )
    parser.add_argument(
        "--examples-per-group",
        type=int,
        default=2,
        help="Numero de exemplos bons/medianos/ruins para plotar por cenario.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Nao salva figuras, apenas metricas numericas.",
    )
    parser.set_defaults(handler=handle_evaluate_ml)


def build_compare_methods_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the method-comparison subcommand."""
    parser = subparsers.add_parser("compare-methods", help="Compara metodos de inversao SHG.")
    parser.add_argument("--model-path", type=str, required=True, help="Arquivo NPZ com o modelo treinado.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Arquivo NPZ com o dataset de teste.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/compare_methods",
        help="Diretorio para salvar metricas e figuras da comparacao.",
    )
    parser.add_argument(
        "--normalization",
        choices=["global", "separate"],
        default="global",
        help="Estrategia de normalizacao da parte fisica.",
    )
    parser.add_argument(
        "--local-bounds",
        choices=["global", "neighborhood"],
        default="neighborhood",
        help="Tipo de bounds usados no refinamento hibrido.",
    )
    parser.add_argument(
        "--neighborhood-fraction",
        type=float,
        default=0.1,
        help="Fracao da largura dos bounds globais usada na vizinhanca local.",
    )
    parser.add_argument("--classical-seed", type=int, default=None, help="Seed base para o fitting classico.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita o numero de amostras do teste para a comparacao.",
    )
    parser.add_argument(
        "--examples-per-group",
        type=int,
        default=1,
        help="Numero de exemplos bons/medianos/ruins a ilustrar.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Nao salva figuras comparativas.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Desabilita o progresso do fitting classico e do refinamento hibrido.",
    )
    parser.set_defaults(handler=handle_compare_methods)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser with subcommands."""
    parser = argparse.ArgumentParser(description="SHG sim (adaptado do gerador em MATLAB).")
    subparsers = parser.add_subparsers(dest="command")

    build_simulate_parser(subparsers)
    build_fit_parser(subparsers)
    build_generate_dataset_parser(subparsers)
    build_train_ml_parser(subparsers)
    build_evaluate_ml_parser(subparsers)
    build_compare_methods_parser(subparsers)

    return parser


def sample_experimental_data() -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Return the current sample experimental data."""
    d_exp = np.array([0, 50, 100, 150, 200], dtype=np.float64)
    i3_exp = np.array([0.1, 0.8, 0.3, 1.0, 0.4], dtype=np.float64)
    i1_exp = np.array([0.5, 0.2, 0.7, 0.1, 0.6], dtype=np.float64)
    lambda_m = 1560e-9
    return d_exp, i3_exp, i1_exp, lambda_m


def handle_simulate(args: argparse.Namespace) -> None:
    """Run the forward SHG simulation."""
    d_nm = np.arange(0.0, args.d_max_nm + args.d_step_nm, args.d_step_nm)
    params = SHGParams(
        lambda_m=args.lambda_nm * 1e-9,
        n21w=complex(args.n21w, args.k21w),
        n22w=complex(args.n22w, args.k22w),
    )

    i3, i1 = simulate_shg(params, d_nm)
    plot_shg_curves(d_nm, i3, i1)


def handle_fit(args: argparse.Namespace) -> None:
    """Run the inverse fitting workflow with in-code experimental data."""
    from src.inverse.fitters import run_fit

    d_exp, i3_exp, i1_exp, lambda_m = sample_experimental_data()
    normalization_strategy = args.normalization
    result = run_fit(
        d_exp,
        i3_exp,
        i1_exp,
        lambda_m,
        normalization_strategy=normalization_strategy,
        seed=args.seed,
    )
    plot_fit_comparison(d_exp, i3_exp, i1_exp, result)
    plot_error_map(
        d_exp,
        i3_exp,
        i1_exp,
        lambda_m,
        n22_fixed=float(result.fitted_params.n22w.real),
        k22_fixed=float(result.fitted_params.n22w.imag),
        normalization_strategy=normalization_strategy,
    )


def handle_generate_dataset(args: argparse.Namespace) -> None:
    """Generate and save a synthetic SHG dataset for ML."""
    from src.data.synthetic_generator import generate_synthetic_dataset, save_synthetic_dataset

    bounds = {
        "n21w": (args.n21w_min, args.n21w_max),
        "k21w": (args.k21w_min, args.k21w_max),
        "n22w": (args.n22w_min, args.n22w_max),
        "k22w": (args.k22w_min, args.k22w_max),
    }
    d_nm = np.arange(0.0, args.d_max_nm + args.d_step_nm, args.d_step_nm, dtype=np.float64)
    dataset = generate_synthetic_dataset(
        num_samples=args.num_samples,
        d_nm=d_nm,
        lambda_m=args.lambda_nm * 1e-9,
        bounds=bounds,
        seed=args.seed,
        normalization=args.normalization,
        show_progress=not args.no_progress,
    )
    output_path = save_synthetic_dataset(dataset, args.output)
    print(f"Dataset salvo em: {output_path}")
    print(f"Formato curves: {dataset.curves.shape}")
    print(f"Formato parameters: {dataset.parameters.shape}")


def handle_train_ml(args: argparse.Namespace) -> None:
    """Train a masked-input MLP from a synthetic SHG dataset."""
    from src.ml.models import ModelConfig, save_model
    from src.ml.train import (
        TrainingConfig,
        build_training_summary,
        save_training_summary,
        train_model,
    )

    if any(hidden_dim <= 0 for hidden_dim in args.hidden_dims):
        raise ValueError("--hidden-dims must contain only positive integers.")

    synthetic_dataset = load_synthetic_dataset(args.dataset_path)
    dataset = from_synthetic_dataset(synthetic_dataset)

    model_config = ModelConfig(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        hidden_dims=tuple(args.hidden_dims),
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        verbose=args.verbose,
    )

    training_result = train_model(
        dataset=dataset,
        model_config=model_config,
        training_config=training_config,
    )
    model_path = save_model(training_result.model, args.model_path)
    summary = build_training_summary(
        dataset=dataset,
        model_config=model_config,
        training_config=training_config,
        training_result=training_result,
        dataset_path=args.dataset_path,
        model_path=str(model_path),
    )
    summary_path = save_training_summary(summary, args.summary_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"Resumo salvo em: {summary_path}")
    print(f"Amostras usadas: {dataset.num_samples}")
    print(f"Entrada={dataset.input_dim} | Saida={dataset.output_dim}")
    print(f"Camadas ocultas: {model_config.hidden_dims}")
    print(f"Loss final de treino: {summary.final_train_loss:.6f}")


def handle_evaluate_ml(args: argparse.Namespace) -> None:
    """Evaluate a trained ML model on a test SHG dataset."""
    from src.ml.evaluate import evaluate_model

    dataset = from_synthetic_dataset(load_synthetic_dataset(args.dataset_path))
    model = load_model(args.model_path)
    report = evaluate_model(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        save_figures=not args.no_figures,
        examples_per_group=args.examples_per_group,
    )

    for scenario_name, scenario in report.scenarios.items():
        print(f"\n=== {scenario_name} ===")
        for parameter_name, metrics in scenario.parameter_metrics.items():
            print(
                f"{parameter_name}: "
                f"MAE={metrics.mae:.6f} "
                f"RMSE={metrics.rmse:.6f} "
                f"R2={metrics.r2:.6f}"
            )
        reconstruction = scenario.reconstruction_metrics
        print(
            "Reconstrucao: "
            f"MSE_i3={reconstruction.mse_i3:.6e} "
            f"MSE_i1={reconstruction.mse_i1:.6e} "
            f"Erro_total={reconstruction.mean_total_error:.6e} "
            f"Falhas={reconstruction.simulation_failures}"
        )


def handle_compare_methods(args: argparse.Namespace) -> None:
    """Compare classical, ML and hybrid SHG inversion methods."""
    from src.ml.compare import compare_methods

    dataset = from_synthetic_dataset(load_synthetic_dataset(args.dataset_path))
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        selected_indices = np.arange(min(args.max_samples, dataset.num_samples), dtype=np.int64)
        dataset = subset_dataset(dataset, selected_indices)

    model = load_model(args.model_path)
    report = compare_methods(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        normalization_strategy=args.normalization,
        local_bounds_mode=args.local_bounds,
        neighborhood_fraction=args.neighborhood_fraction,
        classical_seed=args.classical_seed,
        save_figures=not args.no_figures,
        examples_per_group=args.examples_per_group,
        show_progress=not args.no_progress,
    )

    for method_name, result in report.methods.items():
        print(f"\n=== {method_name} ===")
        print(
            f"Tempo total={result.timing.total_seconds:.4f}s "
            f"| Tempo medio={result.timing.mean_seconds_per_sample:.4f}s"
        )
        for parameter_name, metrics in result.parameter_metrics.items():
            print(
                f"{parameter_name}: "
                f"MAE={metrics.mae:.6f} "
                f"RMSE={metrics.rmse:.6f} "
                f"R2={metrics.r2:.6f}"
            )
        reconstruction = result.reconstruction_metrics
        print(
            "Reconstrucao: "
            f"MSE_i3={reconstruction.mse_i3:.6e} "
            f"MSE_i1={reconstruction.mse_i1:.6e} "
            f"Erro_total={reconstruction.mean_total_error:.6e} "
            f"Falhas={reconstruction.simulation_failures}"
        )


def resolve_handler(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Optional[CommandHandler]:
    """Resolve the selected subcommand handler."""
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return None
    return handler


def main() -> None:
    """Run the SHG command-line interface."""
    parser = build_parser()
    args = parser.parse_args()
    handler = resolve_handler(args, parser)

    if handler is None:
        return

    handler(args)


if __name__ == "__main__":
    main()
