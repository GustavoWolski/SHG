"""Fast smoke tests for the SHG inverse project."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data.loaders import load_synthetic_dataset
from src.data.synthetic_generator import generate_synthetic_dataset, save_synthetic_dataset
from src.main import build_parser, resolve_handler
from src.ml.datasets import from_synthetic_dataset
from src.ml.evaluate import evaluate_model
from src.ml.models import ModelConfig, load_model, save_model
from src.ml.train import TrainingConfig, train_model
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
                    "--hidden-dims",
                    "32",
                    "16",
                    "--epochs",
                    "2",
                    "--batch-size",
                    "4",
                    "--seed",
                    "33",
                ]
            )
            handler = resolve_handler(args, parser)
            self.assertIsNotNone(handler)
            assert handler is not None
            handler(args)

            self.assertTrue(model_path.exists())
            self.assertTrue(summary_path.exists())
            loaded_model = load_model(model_path)
            self.assertEqual(loaded_model.config.hidden_dims, (32, 16))


if __name__ == "__main__":
    unittest.main()
