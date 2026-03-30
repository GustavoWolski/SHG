"""Fast smoke tests for the SHG inverse project."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data.loaders import load_experimental_shg_data, load_synthetic_dataset
from src.data.synthetic_generator import generate_synthetic_dataset, save_synthetic_dataset
from src.inverse.objective import error_function
from src.inverse.methods import run_hybrid_inverse_method, run_ml_inverse_method
from src.main import build_parser, resolve_handler
from src.ml.datasets import from_synthetic_dataset, split_dataset
from src.ml.evaluate import evaluate_model
from src.ml.models import ModelConfig, load_model, save_model
from src.ml.train import TrainingConfig, train_model
from src.physics.shg_model import SHGParams, simulate_shg
from src.physics.shg_model import validate_default_simulation


class SmokeTests(unittest.TestCase):
    """Minimal regression checks for the executable project pieces."""

    def test_default_physics_validation(self) -> None:
        """The default SHG simulation should remain finite and shape-consistent."""
        i3, i1 = validate_default_simulation()
        self.assertEqual(i3.shape, (601,))
        self.assertEqual(i1.shape, (601,))
        self.assertTrue(np.all(np.isfinite(i3)))
        self.assertTrue(np.all(np.isfinite(i1)))
        self.assertGreaterEqual(float(i3.min()), -1e-12)
        self.assertGreaterEqual(float(i1.min()), -1e-12)

    def test_synthetic_dataset_roundtrip(self) -> None:
        """Synthetic dataset generation and loading should preserve array structure."""
        thickness_nm = np.arange(0.0, 55.0, 5.0, dtype=np.float64)
        dataset = generate_synthetic_dataset(
            num_samples=6,
            d_nm=thickness_nm,
            lambda_m=1560e-9,
            seed=11,
            normalization="global",
            show_progress=False,
        )

        self.assertEqual(dataset.curves.shape, (6, 2, thickness_nm.size))
        self.assertEqual(dataset.parameters.shape, (6, 4))

        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = save_synthetic_dataset(dataset, Path(temporary_directory) / "dataset.npz")
            reloaded = load_synthetic_dataset(dataset_path)

        self.assertEqual(reloaded.curves.shape, dataset.curves.shape)
        self.assertEqual(reloaded.parameters.shape, dataset.parameters.shape)
        self.assertEqual(reloaded.normalization, "global")

    def test_experimental_loader_and_split(self) -> None:
        """External experimental files and automatic dataset split should be valid."""
        thickness_nm = np.arange(0.0, 65.0, 5.0, dtype=np.float64)
        synthetic_dataset = generate_synthetic_dataset(
            num_samples=12,
            d_nm=thickness_nm,
            lambda_m=1560e-9,
            seed=17,
            normalization="global",
            show_progress=False,
        )
        dataset = from_synthetic_dataset(synthetic_dataset)
        dataset_split = split_dataset(dataset, seed=17)

        self.assertEqual(dataset_split.train.num_samples + dataset_split.validation.num_samples + dataset_split.test.num_samples, dataset.num_samples)
        self.assertGreater(dataset_split.train.num_samples, 0)
        self.assertGreater(dataset_split.validation.num_samples, 0)
        self.assertGreater(dataset_split.test.num_samples, 0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "experimental.csv"
            data_path.write_text("0,0.10,\n5,,0.20\n10,0.30,0.40\n15,,\n", encoding="utf-8")
            loaded = load_experimental_shg_data(data_path)

        self.assertEqual(loaded.d_nm.shape, (4,))
        self.assertEqual(loaded.i3.shape, (4,))
        self.assertEqual(loaded.i1.shape, (4,))
        self.assertTrue(np.array_equal(loaded.i3_mask, np.array([True, False, True, False], dtype=bool)))
        self.assertTrue(np.array_equal(loaded.i1_mask, np.array([False, True, True, False], dtype=bool)))

    def test_objective_supports_missing_experimental_values(self) -> None:
        """The inverse objective should ignore missing i3/i1 samples through masks."""
        thickness_nm = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
        params = SHGParams(
            lambda_m=1560e-9,
            n21w=complex(5.6428, 0.0849),
            n22w=complex(2.8698, 0.4492),
        )
        i3_exp, i1_exp = simulate_shg(params, thickness_nm)
        i3_mask = np.array([True, False, True, False], dtype=bool)
        i1_mask = np.array([False, True, True, False], dtype=bool)
        i3_with_gaps = i3_exp.copy()
        i1_with_gaps = i1_exp.copy()
        i3_with_gaps[~i3_mask] = np.nan
        i1_with_gaps[~i1_mask] = np.nan

        objective_value = error_function(
            [5.6428, 0.0849, 2.8698, 0.4492],
            thickness_nm,
            i3_with_gaps,
            i1_with_gaps,
            1560e-9,
            normalization_strategy="global",
            i3_mask=i3_mask,
            i1_mask=i1_mask,
        )

        self.assertTrue(np.isfinite(objective_value))
        self.assertLess(objective_value, 1e-12)

    def test_train_and_evaluate_pipeline(self) -> None:
        """A short ML pipeline should train, save, load and evaluate successfully."""
        thickness_nm = np.arange(0.0, 65.0, 5.0, dtype=np.float64)
        synthetic_dataset = generate_synthetic_dataset(
            num_samples=12,
            d_nm=thickness_nm,
            lambda_m=1560e-9,
            seed=22,
            normalization="global",
            show_progress=False,
        )
        dataset = from_synthetic_dataset(synthetic_dataset)

        model_config = ModelConfig(
            input_dim=dataset.input_dim,
            output_dim=dataset.output_dim,
            hidden_dims=(32, 16),
        )
        training_config = TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            seed=22,
            verbose=False,
        )
        training_result = train_model(dataset, model_config, training_config)

        self.assertEqual(len(training_result.train_loss_history), 2)
        self.assertEqual(training_result.best_epoch, 2)
        self.assertTrue(np.isfinite(training_result.train_loss_history[-1]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = save_model(training_result.model, Path(temporary_directory) / "model.npz")
            loaded_model = load_model(model_path)
            report = evaluate_model(
                model=loaded_model,
                dataset=dataset,
                output_dir=None,
                save_figures=False,
                examples_per_group=1,
            )

        self.assertEqual(set(report.scenarios.keys()), {"i3_i1", "i3_only", "i1_only"})
        for scenario in report.scenarios.values():
            self.assertEqual(scenario.predictions.shape, dataset.targets.shape)
            self.assertEqual(scenario.reconstructed_i3.shape, dataset.i3.shape)
            self.assertEqual(scenario.reconstructed_i1.shape, dataset.i1.shape)
            self.assertEqual(scenario.reconstruction_metrics.simulation_failures, 0)

    def test_ml_and_hybrid_experimental_fit_modes(self) -> None:
        """ML and hybrid inverse methods should work on one experimental sample with gaps."""
        thickness_nm = np.arange(0.0, 65.0, 5.0, dtype=np.float64)
        synthetic_dataset = generate_synthetic_dataset(
            num_samples=10,
            d_nm=thickness_nm,
            lambda_m=1560e-9,
            seed=25,
            normalization="global",
            show_progress=False,
        )
        dataset = from_synthetic_dataset(synthetic_dataset)
        model_config = ModelConfig(
            input_dim=dataset.input_dim,
            output_dim=dataset.output_dim,
            hidden_dims=(32, 16),
        )
        training_config = TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            seed=25,
            verbose=False,
        )
        training_result = train_model(dataset, model_config, training_config)

        i3_exp = synthetic_dataset.i3[0].copy()
        i1_exp = synthetic_dataset.i1[0].copy()
        i3_mask = np.ones_like(i3_exp, dtype=bool)
        i1_mask = np.ones_like(i1_exp, dtype=bool)
        i3_mask[[1, 3]] = False
        i1_mask[[0, 4]] = False
        i3_exp[~i3_mask] = np.nan
        i1_exp[~i1_mask] = np.nan

        ml_result = run_ml_inverse_method(
            d_exp=thickness_nm,
            i3_exp=i3_exp,
            i1_exp=i1_exp,
            lambda_m=1560e-9,
            model=training_result.model,
            normalization_strategy="global",
            i3_mask=i3_mask,
            i1_mask=i1_mask,
        )
        hybrid_result = run_hybrid_inverse_method(
            d_exp=thickness_nm,
            i3_exp=i3_exp,
            i1_exp=i1_exp,
            lambda_m=1560e-9,
            model=training_result.model,
            normalization_strategy="global",
            i3_mask=i3_mask,
            i1_mask=i1_mask,
            local_bounds_mode="neighborhood",
            neighborhood_fraction=0.1,
        )

        self.assertEqual(ml_result.parameter_vector.shape, (4,))
        self.assertEqual(hybrid_result.parameter_vector.shape, (4,))
        self.assertTrue(np.isfinite(ml_result.objective_error))
        self.assertTrue(np.isfinite(hybrid_result.objective_error))
        self.assertEqual(ml_result.reconstructed_i3.shape, thickness_nm.shape)
        self.assertEqual(hybrid_result.reconstructed_i1.shape, thickness_nm.shape)

    def test_train_ml_cli_handler(self) -> None:
        """The train-ml subcommand should train and persist artifacts successfully."""
        thickness_nm = np.arange(0.0, 45.0, 5.0, dtype=np.float64)
        synthetic_dataset = generate_synthetic_dataset(
            num_samples=8,
            d_nm=thickness_nm,
            lambda_m=1560e-9,
            seed=33,
            normalization="global",
            show_progress=False,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = save_synthetic_dataset(synthetic_dataset, Path(temporary_directory) / "dataset.npz")
            model_path = Path(temporary_directory) / "trained_model.npz"
            summary_path = Path(temporary_directory) / "training_summary.json"
            output_dir = Path(temporary_directory) / "train_pipeline"

            parser = build_parser()
            args = parser.parse_args(
                [
                    "train-ml",
                    "--dataset-path",
                    str(dataset_path),
                    "--model-path",
                    str(model_path),
                    "--summary-path",
                    str(summary_path),
                    "--output-dir",
                    str(output_dir),
                    "--hidden-dims",
                    "32",
                    "16",
                    "--epochs",
                    "2",
                    "--batch-size",
                    "4",
                    "--seed",
                    "33",
                    "--no-figures",
                ]
            )
            handler = resolve_handler(args, parser)
            self.assertIsNotNone(handler)
            assert handler is not None
            handler(args)

            self.assertTrue(model_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue((output_dir / "dataset_split.json").exists())
            self.assertTrue((output_dir / "validation" / "evaluation_summary.json").exists())
            self.assertTrue((output_dir / "test" / "evaluation_summary.json").exists())
            loaded_model = load_model(model_path)
            self.assertEqual(loaded_model.config.hidden_dims, (32, 16))

    def test_generate_dataset_from_experimental_grid(self) -> None:
        """The dataset generator should reuse d_nm from an experimental CSV when requested."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            experimental_path = Path(temporary_directory) / "experimental.csv"
            experimental_path.write_text(
                "d_nm,i3,i1\n65,0.1,\n80,0.2,0.3\n100,,0.4\n150,0.5,0.6\n",
                encoding="utf-8",
            )
            output_path = Path(temporary_directory) / "synthetic_from_grid.npz"

            parser = build_parser()
            args = parser.parse_args(
                [
                    "generate-dataset",
                    "--num-samples",
                    "5",
                    "--output",
                    str(output_path),
                    "--experimental-grid-path",
                    str(experimental_path),
                    "--grid-delimiter",
                    ",",
                    "--grid-skiprows",
                    "1",
                    "--seed",
                    "19",
                    "--normalization",
                    "global",
                    "--no-progress",
                ]
            )
            handler = resolve_handler(args, parser)
            self.assertIsNotNone(handler)
            assert handler is not None
            handler(args)

            dataset = load_synthetic_dataset(output_path)
            self.assertTrue(np.array_equal(dataset.d_nm, np.array([65.0, 80.0, 100.0, 150.0], dtype=np.float64)))
            self.assertEqual(dataset.i3.shape, (5, 4))
            self.assertEqual(dataset.i1.shape, (5, 4))

    def test_fit_ml_cli_saves_summary_and_figure(self) -> None:
        """The fit subcommand should save summary and plot artifacts in ML mode."""
        thickness_nm = np.array([65.0, 80.0, 100.0, 150.0, 190.0], dtype=np.float64)
        synthetic_dataset = generate_synthetic_dataset(
            num_samples=10,
            d_nm=thickness_nm,
            lambda_m=1560e-9,
            seed=41,
            normalization="global",
            show_progress=False,
        )
        dataset = from_synthetic_dataset(synthetic_dataset)
        model_config = ModelConfig(
            input_dim=dataset.input_dim,
            output_dim=dataset.output_dim,
            hidden_dims=(32, 16),
        )
        training_config = TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            seed=41,
            verbose=False,
        )
        training_result = train_model(dataset, model_config, training_config)

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = save_model(training_result.model, Path(temporary_directory) / "model.npz")
            experimental_path = Path(temporary_directory) / "experimental.csv"
            experimental_path.write_text(
                "d_nm,i3,i1\n65,0.10,\n80,0.20,0.30\n100,,0.40\n150,0.50,0.60\n190,0.70,0.80\n",
                encoding="utf-8",
            )
            output_dir = Path(temporary_directory) / "fit_ml_outputs"

            parser = build_parser()
            args = parser.parse_args(
                [
                    "fit",
                    "--method",
                    "ml",
                    "--model-path",
                    str(model_path),
                    "--data-path",
                    str(experimental_path),
                    "--lambda-nm",
                    "1560",
                    "--delimiter",
                    ",",
                    "--skiprows",
                    "1",
                    "--normalization",
                    "global",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            handler = resolve_handler(args, parser)
            self.assertIsNotNone(handler)
            assert handler is not None
            handler(args)

            self.assertTrue((output_dir / "fit_ml_summary.json").exists())
            self.assertTrue((output_dir / "fit_ml_curves.png").exists())
            self.assertTrue((output_dir / "fit_ml_simulation.png").exists())


if __name__ == "__main__":
    unittest.main()
