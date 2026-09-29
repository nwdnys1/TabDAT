"""TabDAT methods extracted without changing their implementation."""

import torch
import torch.nn.functional as F

if __package__:
    from .output_heads import gmm_nll
else:
    from output_heads import gmm_nll


class LossMixin:
    def compute_loss(
        self, x, outputs, mask_indices, return_components=False, *,
        ddpm_timesteps=None, ddpm_noise=None, return_per_column=False,
    ):
        """
        Computes masked-target loss, summed over columns and averaged over rows
        (the original batch normalization). DDPM uses noise-prediction MSE;
        Gaussian and GMM use negative log-likelihood.

        Args:
            x (torch.Tensor): The input data.
            outputs (dict): The output from the forward pass.
            mask_indices (Tensor): Boolean [batch, num_vars] mask; True
                identifies an unknown target included in the loss.

        Returns:
            Tensor, or (Tensor, dict) when return_components is True.
        """

        # Keep the legacy Gaussian accumulation and batch-size normalization.
        # The DDPM term below is a denoising MSE, not a log likelihood.
        nll = x.new_zeros(())
        categorical_loss = x.new_zeros(())
        continuous_loss = x.new_zeros(())
        column_sums = [] if return_per_column else None
        for i in range(self.num_vars):
            selected = mask_indices[:, i]
            target = x[selected, i]
            prediction = outputs[i][selected]
            if prediction.shape[0] == 0:
                # Differentiable zero, including for a fully unmasked batch.
                # An empty sum avoids evaluating invalid unselected values.
                nll = nll + prediction.sum()
                if return_per_column:
                    column_sums.append(x.new_zeros(()))
                continue
            if i in self.var_types.get("cat", {}):
                # Categorical Cross-Entropy Loss
                contribution = F.cross_entropy(prediction, target.long(), reduction="sum")
                nll += contribution
                categorical_loss += contribution.detach()
            elif self.continuous_head == "gmm":
                contribution = gmm_nll(prediction, target, self.gmm_components).sum()
                nll += contribution
                continuous_loss += contribution.detach()
            elif self.continuous_head == "ddpm":
                contribution = self.prediction_heads[str(i)].loss(
                    prediction, target,
                    timesteps=None if ddpm_timesteps is None else ddpm_timesteps[selected, i],
                    noise=None if ddpm_noise is None else ddpm_noise[selected, i],
                ).sum()
                nll += contribution
                continuous_loss += contribution.detach()
            else:
                # Gaussian Negative Log-Likelihood
                mu, log_sigma = prediction.chunk(2, dim=-1)
                sigma = torch.exp(log_sigma)
                # Clamp sigma to avoid numerical instability
                sigma = torch.clamp(sigma, min=1e-5)
                dist = torch.distributions.Normal(mu.squeeze(-1), sigma.squeeze(-1))
                contribution = -dist.log_prob(target).sum()
                nll += contribution
                continuous_loss += contribution.detach()

            if return_per_column:
                column_sums.append(contribution.detach())

        nll /= x.shape[0]  # Average over batch

        if return_components:
            parts = {
                "categorical": categorical_loss / x.shape[0],
                "continuous": continuous_loss / x.shape[0],
            }
            if return_per_column:
                parts["per_column"] = torch.stack(column_sums) / x.shape[0]
                parts["target_counts"] = mask_indices.sum(dim=0)
            return nll, parts
        return nll
