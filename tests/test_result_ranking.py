"""Check new quality metric labels and higher-is-better ranking."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from eval import exp_results


class ResultRankingTests(unittest.TestCase):
    def test_new_extra_columns_remain_separate_from_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adult_extra_results.csv"
            pd.DataFrame([{
                "Dataset": "sampled_adult_mine_42.csv",
                "Alpha-Precision (naive)": 0.7,
                "Beta-Recall (naive)": 0.6,
                "C2ST LogisticDetection": 0.8,
                "Metric Rows": 100,
            }]).to_csv(path, index=False)
            with patch.object(exp_results, "BASE_PATHS", [temporary]):
                data, _ = exp_results.process_results()
        metrics = data["mine"]["classification"][42]
        self.assertEqual(metrics["AP naive"], [0.7])
        self.assertEqual(metrics["BR naive"], [0.6])
        self.assertEqual(metrics["C2ST LogisticDetection"], [0.8])
        self.assertNotIn("AP (legacy)", metrics)

    def test_higher_detection_score_ranks_first(self):
        values = {
            "low": {"C2ST LogisticDetection": {"val": 0.2, "str": "0.2"}},
            "high": {"C2ST LogisticDetection": {"val": 0.9, "str": "0.9"}},
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exp_results.print_table(values, ["C2ST LogisticDetection"])
        self.assertIn("**0.9**", output.getvalue())
        self.assertNotIn("**0.2**", output.getvalue())


if __name__ == "__main__":
    unittest.main()
