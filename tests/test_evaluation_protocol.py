"""Artificial-data checks for split provenance and held-out evaluation."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from eval import evaluation
from eval.evaluation import get_utility_metrics, privacy_metrics, stat_sim
from model.data_splits import prepare_data_splits
from model.model import TabDAT


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.source_path = root / "artificial.csv"
        self.split_dir = root / "splits"
        frame = pd.DataFrame({
            "id": np.arange(40),
            "value": np.arange(40, dtype=float) + 1,
            "label": ["a", "b"] * 20,
        })
        frame.to_csv(self.source_path, index=False)

    def test_split_reuse_and_train_only_preprocessing(self):
        paths = prepare_data_splits(
            self.source_path, self.split_dir,
            validation_ratio=0.1, test_ratio=0.2, stratify_col="label",
        )
        splits = {name: pd.read_csv(path) for name, path in paths.items()}
        self.assertEqual([len(splits[name]) for name in ("train", "validation", "test")],
                         [28, 4, 8])
        ids = [set(split["id"]) for split in splits.values()]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(set.union(*ids), set(range(40)))
        self.assertEqual(paths, prepare_data_splits(
            self.source_path, self.split_dir,
            validation_ratio=0.1, test_ratio=0.2, stratify_col="label",
        ))
        manifest = json.loads((self.split_dir / "split_manifest.json").read_text())
        self.assertEqual(manifest["rows"], {"train": 28, "validation": 4, "test": 8})
        with self.assertRaises(ValueError):
            prepare_data_splits(self.source_path, self.split_dir, seed=43,
                                stratify_col="label")

        with contextlib.redirect_stdout(io.StringIO()):
            model = TabDAT(
                embed_dim=8, num_heads=2, num_layers=1,
                file_path=str(paths["train"]), cat_cols=[2], device="cpu",
            )
        self.assertEqual(len(model.data), 28)
        self.assertAlmostEqual(
            model.scalers[1].mean_[0], splits["train"]["value"].mean()
        )
        with self.assertRaisesRegex(ValueError, "Split the raw CSV first"):
            model.fit(epochs=0, test_split_ratio=0.2)

    def test_utility_uses_explicit_real_holdout_and_all_fake_rows(self):
        paths = prepare_data_splits(self.source_path, self.split_dir)
        fake_path = Path(self.temp_dir.name) / "fake.csv"
        pd.read_csv(paths["train"]).to_csv(fake_path, index=False)
        lengths = []

        def record_lengths(x_train, y_train, x_test, y_test, model_name, problem_type):
            lengths.append((len(x_train), len(x_test)))
            return [0.0, 0.0, 0.0]

        with patch("eval.evaluation.supervised_model_training", side_effect=record_lengths):
            get_utility_metrics(
                str(paths["train"]), [str(fake_path)],
                type={"Regression": ["l_reg"]},
                cat_cols=["label"], target_col="value",
                real_test_path=str(paths["test"]),
            )
        self.assertEqual(lengths, [(28, 8), (28, 8)])
        with self.assertRaisesRegex(ValueError, "real_test_path"):
            get_utility_metrics(str(paths["train"]), [str(fake_path)])

    def test_utility_real_regression_smoke_on_artificial_rows(self):
        paths = prepare_data_splits(self.source_path, self.split_dir)
        fake_path = Path(self.temp_dir.name) / "fake.csv"
        pd.read_csv(paths["train"]).to_csv(fake_path, index=False)
        real, fake = get_utility_metrics(
            str(paths["train"]), [str(fake_path)],
            type={"Regression": ["l_reg"]},
            cat_cols=["label"], target_col="value",
            real_test_path=str(paths["test"]),
        )
        np.testing.assert_allclose(real.ravel(), fake.ravel())
        self.assertEqual(real.shape, (1, 3))
        self.assertTrue(np.isfinite(real).all())

    def test_statistic_uses_mean_unique_pair_error(self):
        fake_path = Path(self.temp_dir.name) / "fake.csv"
        pd.read_csv(self.source_path).to_csv(fake_path, index=False)
        real_assoc = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.3],
                               [0.4, 0.3, 1.0]])
        fake_assoc = np.array([[1.0, 0.5, 0.4], [0.5, 1.0, 0.6],
                               [0.4, 0.6, 1.0]])
        with patch("eval.evaluation.compute_associations",
                   side_effect=[real_assoc, fake_assoc]):
            result = stat_sim(str(self.source_path), str(fake_path),
                              cat_cols=["label"])
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)
        self.assertAlmostEqual(result[3], 0.2)

    def test_statistic_with_real_dython_on_artificial_rows(self):
        if evaluation.compute_associations is None:
            self.skipTest("Optional dython dependency is not installed")
        result = stat_sim(
            str(self.source_path), str(self.source_path), cat_cols=["label"]
        )
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)
        self.assertAlmostEqual(result[3], 0.0)

    def test_privacy_metrics_are_paused(self):
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            privacy_metrics("unused.csv", "unused.csv", [])


if __name__ == "__main__":
    unittest.main()
