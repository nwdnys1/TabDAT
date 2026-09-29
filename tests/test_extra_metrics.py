"""Artificial-table checks for optional quality metrics without heavy packages."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from eval import evaluation


class ExtraMetricTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.real_path = root / "real.csv"
        self.fake_path = root / "fake.csv"
        pd.DataFrame({
            "amount": [1, 2, 3, 4, 5, 6],
            "group": ["a", "b", "a", "b", "a", "b"],
        }).to_csv(self.real_path, index=False)
        pd.DataFrame({
            "amount": [2, 3, 4, 5],
            "group": ["a", "b", "new", "b"],
        }).to_csv(self.fake_path, index=False)

    def test_missing_optional_dependencies_are_explicit(self):
        with patch.object(evaluation, "eval_statistical", None), \
             patch.object(evaluation, "GenericDataLoader", None), \
             patch.object(evaluation, "LogisticDetection", None):
            with self.assertRaisesRegex(ImportError, "synthcity==0.2.12.*sdmetrics==0.21.0"):
                evaluation.get_extra_metrics(self.real_path, self.fake_path)

    def test_equal_seeded_rows_and_metadata(self):
        seen = {}

        class FakeAlphaPrecision:
            def evaluate(self, real, fake):
                seen["real_encoded"] = real
                seen["fake_encoded"] = fake
                return {
                    "delta_precision_alpha_naive": 0.7,
                    "delta_coverage_beta_naive": 0.6,
                }

        class FakeDetection:
            @staticmethod
            def compute(*, real_data, synthetic_data, metadata):
                seen["real"] = real_data
                seen["fake"] = synthetic_data
                seen["metadata"] = metadata
                return 0.8

        with patch.object(evaluation, "eval_statistical",
                          SimpleNamespace(AlphaPrecision=FakeAlphaPrecision)), \
             patch.object(evaluation, "GenericDataLoader", lambda frame: frame), \
             patch.object(evaluation, "LogisticDetection", FakeDetection):
            result = evaluation.get_extra_metrics(
                self.real_path, self.fake_path, cat_cols=["group"],
                max_rows=3, seed=17,
            )
            again = evaluation.get_extra_metrics(
                self.real_path, self.fake_path, cat_cols=["group"],
                max_rows=3, seed=17,
            )

        self.assertEqual(result, again)
        self.assertEqual(result, {
            "alpha_precision": 0.7, "beta_recall": 0.6,
            "c2st": 0.8, "n_rows": 3,
        })
        self.assertEqual(len(seen["real"]), 3)
        self.assertEqual(len(seen["fake"]), 3)
        self.assertEqual(seen["metadata"], {"columns": {
            "amount": {"sdtype": "numerical"},
            "group": {"sdtype": "categorical"},
        }})
        self.assertEqual(seen["real_encoded"].shape, seen["fake_encoded"].shape)
        self.assertTrue(np.isfinite(seen["fake_encoded"].to_numpy()).all())

    def test_rejects_bad_sample_limit(self):
        with patch.object(evaluation, "eval_statistical", SimpleNamespace()), \
             patch.object(evaluation, "GenericDataLoader", object()), \
             patch.object(evaluation, "LogisticDetection", object()):
            with self.assertRaisesRegex(ValueError, "max_rows"):
                evaluation.get_extra_metrics(self.real_path, self.fake_path, max_rows=1)


if __name__ == "__main__":
    unittest.main()
