"""TabDAT methods extracted without changing their implementation."""

import torch
import torch.nn.functional as F


class LossMixin:
    def compute_loss(self, x, outputs, mask_indices):
        """
        Computes masked-target negative log-likelihood, summed over columns
        and averaged over rows (the original batch normalization).

        Args:
            x (torch.Tensor): The input data.
            outputs (dict): The output from the forward pass.
            mask_indices (Tensor): Boolean [batch, num_vars] mask; True
                identifies an unknown target included in the loss.

        Returns:
            float: The total loss.
        """

        # Negative Log-Likelihood
        nll = x.new_zeros(())
        for i in range(self.num_vars):
            selected = mask_indices[:, i]
            target = x[selected, i]
            prediction = outputs[i][selected]
            if prediction.shape[0] == 0:
                # Differentiable zero, including for a fully unmasked batch.
                # An empty sum avoids evaluating invalid unselected values.
                nll = nll + prediction.sum()
                continue
            if i in self.var_types.get("cat", {}):
                # Categorical Cross-Entropy Loss
                nll += F.cross_entropy(prediction, target.long(), reduction="sum")
            else:
                # Gaussian Negative Log-Likelihood
                mu, log_sigma = prediction.chunk(2, dim=-1)
                sigma = torch.exp(log_sigma)
                # Clamp sigma to avoid numerical instability
                sigma = torch.clamp(sigma, min=1e-5)
                dist = torch.distributions.Normal(mu.squeeze(-1), sigma.squeeze(-1))
                nll -= dist.log_prob(target).sum()

        nll /= x.shape[0]  # Average over batch

        return nll

