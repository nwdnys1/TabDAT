"""File-free CPU checks for training task masks and explicit orders."""

import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from model.mask_strategies import MASK_STRATEGIES, ORDERED_STRATEGIES, sample_training_masks
from model.model import TabDAT


class MaskStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(23)

    def test_legacy_bernoulli_has_identical_rng_and_targets(self):
        seed = 57
        expected_generator = torch.Generator().manual_seed(seed)
        actual_generator = torch.Generator().manual_seed(seed)
        expected = torch.rand(32, 4, generator=expected_generator) < 0.3
        input_mask, target_mask = sample_training_masks(
            32, 4, mask_prob=0.3, generator=actual_generator
        )
        torch.testing.assert_close(input_mask, expected)
        torch.testing.assert_close(target_mask, expected)

    def test_all_strategies_hide_every_target(self):
        order = [2, 0, 3, 1]
        for strategy in MASK_STRATEGIES:
            with self.subTest(strategy=strategy):
                input_mask, target_mask = sample_training_masks(
                    128, 4, strategy=strategy, mask_prob=0.5,
                    order=order if strategy in ORDERED_STRATEGIES else None,
                )
                self.assertEqual(input_mask.shape, (128, 4))
                self.assertTrue(torch.all(target_mask <= input_mask))
                if strategy in ("prefix_next", "ordered_completion",
                                "canonical_subset", "random_permutation_next"):
                    self.assertTrue(torch.all(target_mask.sum(1) == 1))
                elif strategy == "uniform_count_all":
                    torch.testing.assert_close(input_mask, target_mask)
                    self.assertTrue(torch.all(target_mask.sum(1) >= 1))

    def test_prefix_next_uses_only_true_prefix(self):
        order = [2, 0, 3, 1]
        hidden, target = sample_training_masks(
            100, 4, strategy="prefix_next", order=order
        )
        for row in range(len(hidden)):
            step = order.index(target[row].nonzero().item())
            self.assertEqual(
                hidden[row].nonzero().flatten().tolist(), sorted(order[step:])
            )

    def test_ordered_completion_targets_first_hidden(self):
        order = [2, 0, 3, 1]
        hidden, target = sample_training_masks(
            100, 4, strategy="ordered_completion", order=order
        )
        for row in range(len(hidden)):
            expected = next(index for index in order if hidden[row, index])
            self.assertEqual(target[row].nonzero().item(), expected)

    def test_canonical_subset_targets_last_query_member(self):
        order = [2, 0, 3, 1]
        hidden, target = sample_training_masks(
            100, 4, strategy="canonical_subset", order=order
        )
        for row in range(len(hidden)):
            target_index = target[row].nonzero().item()
            visible = (~hidden[row]).nonzero().flatten().tolist()
            self.assertTrue(all(order.index(i) < order.index(target_index)
                                for i in visible))

    def test_invalid_orders_fail_before_training(self):
        for bad in (None, [0, 1, 1], [0, 1], [0, 1, 2.0]):
            with self.subTest(order=bad), self.assertRaises(ValueError):
                sample_training_masks(2, 3, strategy="prefix_next", order=bad)
        with self.assertRaises(ValueError):
            sample_training_masks(2, 3, strategy="uniform_count_all",
                                  order=[0, 1, 2])


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(23)
        frame = pd.DataFrame({
            "group": ["a", "b", "a", "b"],
            "value": [-2.0, -1.0, 1.0, 2.0],
            "other": [1.0, 2.0, 3.0, 4.0],
        })
        csv_patch = patch("pandas.read_csv", return_value=frame)
        csv_patch.start()
        self.addCleanup(csv_patch.stop)

    def make_model(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return TabDAT(
                embed_dim=8, num_heads=2, num_layers=1, file_path="artificial.csv",
                cat_cols=[0], dropout=0.0, device="cpu",
            )

    def test_every_strategy_trains_on_artificial_data(self):
        for strategy in sorted(MASK_STRATEGIES):
            with self.subTest(strategy=strategy):
                model = self.make_model()
                order = [2, 0, 1] if strategy in ORDERED_STRATEGIES else None
                with contextlib.redirect_stdout(io.StringIO()):
                    model.fit(
                        epochs=1, batch_size=4, test_split_ratio=0.0,
                        mask_strategy=strategy, training_order=order,
                    )
                self.assertEqual(model.training_config["mask_strategy"], strategy)
                self.assertEqual(model.training_config["training_order"], order)

    def test_explicit_sampling_order_and_checkpoint_metadata(self):
        model = self.make_model()
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(epochs=1, batch_size=4, test_split_ratio=0.0,
                      mask_strategy="prefix_next", training_order=[2, 0, 1])
            result = model.sample(2, device="cpu", order=[1, 2, 0])
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(model.last_sampling_order, [1, 2, 0])
        with self.assertRaises(ValueError):
            model.sample(2, device="cpu", order=[0, 0, 2])
        with (
            patch("model.checkpoint.os.makedirs"),
            patch("model.checkpoint.torch.save") as save_mock,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            model.save("/unused/model.pth")
        checkpoint = save_mock.call_args.args[0]
        with (
            patch("model.checkpoint.torch.load", return_value=checkpoint),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            loaded = TabDAT.load("/unused/model.pth", device="cpu")
        self.assertEqual(loaded.training_config, model.training_config)
        self.assertEqual(loaded.last_sampling_order, [1, 2, 0])

    def test_dp_compatibility_dispatch_is_isolated(self):
        model = self.make_model()
        with patch.object(model, "fit_dp", return_value="dp path") as fit_dp:
            self.assertEqual(model.fit(dp=True, epochs=0), "dp path")
            fit_dp.assert_called_once()
        with self.assertRaises(NotImplementedError):
            model.fit(dp=True, mask_strategy="prefix_next",
                      training_order=[0, 1, 2], epochs=0)

    def test_legacy_dp_loop_runs_separately_on_artificial_data(self):
        model = self.make_model()
        with (
            patch("model.dp_training.compute_dp_epsilon", return_value=0.0),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            model.fit(dp=True, epochs=1, batch_size=4, test_split_ratio=0.0,
                      mask_prob=1.0)
        self.assertTrue(model.training_config["dp"])
        self.assertEqual(model.training_config["mask_strategy"], "bernoulli_all")


if __name__ == "__main__":
    unittest.main()
