"""Small, file-free CPU checks for continuous output heads."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model.model import TabDAT


class OutputHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.frame = pd.DataFrame(
            {"group": ["a", "b", "a", "b"], "value": [-2.0, -1.0, 1.0, 2.0]}
        )
        self.read_csv = patch("pandas.read_csv", return_value=self.frame)
        self.read_csv.start()
        self.addCleanup(self.read_csv.stop)

    def make_model(self, head, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return TabDAT(
                embed_dim=8,
                num_heads=2,
                num_layers=1,
                file_path="artificial.csv",
                cat_cols=[0],
                dropout=0.0,
                device="cpu",
                continuous_head=head,
                diffusion_steps=8,
                diffusion_hidden_dim=16,
                **kwargs,
            )

    def outputs_for(self, model):
        embeddings = model.mask_token.repeat(len(model.data), model.num_vars, 1)
        return model.forward(embeddings)

    def test_gaussian_default_matches_legacy_formula(self):
        model = self.make_model("gaussian")
        self.assertEqual(model.prediction_heads["1"][2].out_features, 2)
        self.assertIn("prediction_heads.1.0.weight", model.state_dict())
        outputs = self.outputs_for(model)
        selected = torch.ones_like(model.data, dtype=torch.bool)
        actual = model.compute_loss(model.data, outputs, selected)

        mu, log_sigma = outputs[1].chunk(2, dim=-1)
        normal = torch.distributions.Normal(
            mu.squeeze(-1), torch.exp(log_sigma).clamp(min=1e-5).squeeze(-1)
        )
        expected = (
            F.cross_entropy(outputs[0], model.data[:, 0].long(), reduction="sum")
            - normal.log_prob(model.data[:, 1]).sum()
        ) / len(model.data)
        torch.testing.assert_close(actual, expected)

    def test_all_heads_train_and_sample_without_files(self):
        for head in ("gaussian", "gmm", "ddpm"):
            with self.subTest(head=head):
                model = self.make_model(head)
                if head == "ddpm":
                    self.assertLess(model.prediction_heads["1"].alpha_bars[-1].item(), 1e-3)
                with contextlib.redirect_stdout(io.StringIO()):
                    model.fit(epochs=1, batch_size=4, mask_prob=1.0, test_split_ratio=0.0)
                    sampled = model.sample(3, device="cpu")
                self.assertEqual(sampled.shape, (3, 2))
                self.assertTrue(np.isfinite(sampled["value"].to_numpy()).all())
                if head == "ddpm":
                    denoiser = model.prediction_heads["1"].denoiser
                    self.assertTrue(any(p.grad is not None for p in denoiser.parameters()))

    def test_only_selected_targets_contribute(self):
        selected = torch.tensor(
            [[True, False], [False, True], [True, False], [False, True]]
        )
        for head in ("gaussian", "gmm", "ddpm"):
            with self.subTest(head=head):
                model = self.make_model(head)
                outputs = self.outputs_for(model)
                changed = model.data.clone()
                changed[~selected] += 10
                torch.manual_seed(31)
                original_loss = model.compute_loss(model.data, outputs, selected)
                torch.manual_seed(31)
                changed_loss = model.compute_loss(changed, outputs, selected)
                torch.testing.assert_close(original_loss, changed_loss)

    def test_in_memory_checkpoint_roundtrip_and_legacy_fallback(self):
        for head in ("gaussian", "gmm", "ddpm"):
            with self.subTest(head=head):
                model = self.make_model(head)
                with (
                    patch("model.checkpoint.os.makedirs"),
                    patch("model.checkpoint.torch.save") as save_mock,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    model.save("/unused/model.pth")
                checkpoint = save_mock.call_args.args[0]
                if head == "gaussian":
                    for key in (
                        "continuous_head", "gmm_components", "diffusion_steps",
                        "diffusion_hidden_dim",
                    ):
                        checkpoint["config"].pop(key)
                with (
                    patch("model.checkpoint.torch.load", return_value=checkpoint),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    loaded = TabDAT.load("/unused/model.pth", device="cpu")
                self.assertEqual(loaded.continuous_head, head)
                self.assertEqual(model.state_dict().keys(), loaded.state_dict().keys())
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
