"""Small CPU-only checkpoint-selection checks on artificial rows."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from model.checkpoint_selection import build_monitor_bank, evaluate_monitor
from model.model import TabDAT


class CheckpointSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.train = pd.DataFrame({
            "group": ["a", "b"] * 6,
            "value": [-2.0, -1.0, 1.0, 2.0] * 3,
        })
        self.validation = pd.DataFrame({
            "group": ["b", "a", "b", "a"],
            "value": [3.0, 4.0, -3.0, 0.0],
        })
        self.train_path = self.root / "train.csv"
        self.train.to_csv(self.train_path, index=False)

    def make_model(self, head="gaussian", dropout=0.1):
        with contextlib.redirect_stdout(io.StringIO()):
            return TabDAT(
                embed_dim=8, num_heads=2, num_layers=1,
                file_path=str(self.train_path), cat_cols=[0],
                dropout=dropout, device="cpu", continuous_head=head,
                diffusion_steps=8, diffusion_hidden_dim=16,
            )

    def make_bank(self, model, data, seed=17):
        return build_monitor_bank(
            data, num_vars=model.num_vars, strategy="bernoulli_all",
            mask_prob=1.0, order=None, continuous_head=model.continuous_head,
            diffusion_steps=model.diffusion_steps, max_rows=4,
            repeats=3, seed=seed,
        )

    def test_holdout_uses_train_scaler_and_rejects_schema_or_unseen_category(self):
        model = self.make_model()
        before = model.scalers[1].mean_.copy()
        transformed = model.transform_holdout(self.validation)
        self.assertEqual(transformed.shape, (4, 2))
        self.assertAlmostEqual(
            transformed[0, 1].item(),
            (3.0 - before[0]) / model.scalers[1].scale_[0],
        )
        self.assertEqual(model.scalers[1].mean_.tolist(), before.tolist())
        with self.assertRaisesRegex(ValueError, "columns must match"):
            model.transform_holdout(self.validation[["value", "group"]])
        unknown = self.validation.copy()
        unknown.loc[0, "group"] = "new"
        with self.assertRaisesRegex(ValueError, "unseen categories"):
            model.transform_holdout(unknown)

    def test_fixed_bank_and_ddpm_noise_are_reproducible_without_rng_side_effects(self):
        model = self.make_model(head="ddpm")
        validation_data = model.transform_holdout(self.validation)
        before = torch.random.get_rng_state().clone()
        bank = self.make_bank(model, validation_data)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertTrue(torch.all(bank.target_mask <= bank.input_mask))
        self.assertIsNotNone(bank.timesteps)
        first = evaluate_monitor(model, bank, batch_size=5, device="cpu")
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertTrue(model.training)
        torch.manual_seed(938)
        second = evaluate_monitor(model, bank, batch_size=4, device="cpu")
        self.assertAlmostEqual(first["loss"], second["loss"], places=6)
        self.assertEqual(first["target_counts"], second["target_counts"])
        self.assertAlmostEqual(sum(first["per_column"]), first["loss"], places=5)
        self.assertEqual(bank.digest, self.make_bank(model, validation_data).digest)

    def test_three_checkpoint_rules_and_legacy_run_are_separate(self):
        model = self.make_model()
        validation_data = model.transform_holdout(self.validation)
        run_dir = self.root / "new_run"
        scores = [3.0, 3.0, 2.0, 1.0, 1.0, 2.0]

        def fake_monitor(*args, **kwargs):
            return {"loss": scores.pop(0), "per_column": [0.0, 0.0],
                    "target_counts": [1, 1], "task_rows": 4,
                    "categorical": 0.0, "continuous": 0.0}

        with patch("model.training.evaluate_monitor", side_effect=fake_monitor), \
                contextlib.redirect_stdout(io.StringIO()):
            model.fit(
                epochs=3, batch_size=4, lr=1e-3, mask_prob=1.0,
                validation_data=validation_data, checkpoint_dir=str(run_dir),
                eval_every=1, monitor_max_rows=4, monitor_repeats=1,
                run_info={"dataset": "artificial"},
            )
        self.assertEqual(model.checkpoint_selection_summary["best_epochs"],
                         {"best_train": 3, "best_val": 2})
        history = [json.loads(line) for line in
                   (run_dir / "monitor_history.jsonl").read_text().splitlines()]
        self.assertEqual(len(history), 3)
        for kind, epoch in (("best_train", 3), ("best_val", 2), ("final", 3)):
            path = run_dir / f"{kind}.pth"
            self.assertTrue(path.is_file())
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = TabDAT.load(str(path), device="cpu")
            self.assertEqual(loaded.checkpoint_selection["epoch"], epoch)
            self.assertEqual(loaded.checkpoint_selection["rule"], kind)
            self.assertEqual(len(loaded.checkpoint_selection["w_order_at_save"]), 2)
            self.assertEqual(loaded.training_config["checkpoint_monitor"]["repeats"], 1)
        with self.assertRaises(FileExistsError):
            model.fit(
                epochs=1, validation_data=validation_data,
                checkpoint_dir=str(run_dir),
            )

    def test_actual_monitor_scores_are_batch_weighted(self):
        model = self.make_model(dropout=0.0)
        bank = self.make_bank(model, model.transform_holdout(self.validation))
        small = evaluate_monitor(model, bank, batch_size=5, device="cpu")
        large = evaluate_monitor(model, bank, batch_size=len(bank.data), device="cpu")
        self.assertAlmostEqual(small["loss"], large["loss"], places=6)
        self.assertAlmostEqual(sum(small["per_column"]), small["loss"], places=6)

    def test_ordered_gmm_ddpm_monitor_and_dp_isolation(self):
        for head in ("gmm", "ddpm"):
            with self.subTest(head=head):
                model = self.make_model(head=head)
                validation_data = model.transform_holdout(self.validation)
                run_dir = self.root / head
                with contextlib.redirect_stdout(io.StringIO()):
                    model.fit(
                        epochs=1, batch_size=4, mask_strategy="prefix_next",
                        training_order=[1, 0], validation_data=validation_data,
                        checkpoint_dir=str(run_dir), eval_every=1,
                        monitor_max_rows=4, monitor_repeats=2,
                    )
                self.assertTrue((run_dir / "best_val.pth").is_file())
                record = model.monitor_history[0]["validation_monitor"]
                self.assertEqual(sum(record["target_counts"]), record["task_rows"])
                self.assertEqual(model.training_config["training_order"], [1, 0])
                with self.assertRaises(NotImplementedError):
                    model.fit(
                        dp=True, validation_data=validation_data,
                        checkpoint_dir=str(self.root / "dp"),
                    )

    def test_monitoring_does_not_change_training_rng_trajectory(self):
        torch.manual_seed(404)
        legacy = self.make_model()
        with contextlib.redirect_stdout(io.StringIO()):
            legacy.fit(epochs=2, batch_size=4, mask_prob=0.5)

        torch.manual_seed(404)
        monitored = self.make_model()
        validation_data = monitored.transform_holdout(self.validation)
        with contextlib.redirect_stdout(io.StringIO()):
            monitored.fit(
                epochs=2, batch_size=4, mask_prob=0.5,
                validation_data=validation_data,
                checkpoint_dir=str(self.root / "rng_run"),
                eval_every=1, monitor_max_rows=4, monitor_repeats=2,
            )
        for name, original in legacy.state_dict().items():
            torch.testing.assert_close(original, monitored.state_dict()[name])


if __name__ == "__main__":
    unittest.main()
