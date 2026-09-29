"""TabDAT methods extracted without changing their implementation."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

if __package__:
    from .mask_strategies import validate_order
    from .output_heads import sample_gmm
else:
    from mask_strategies import validate_order
    from output_heads import sample_gmm


class SamplingMixin:
    def _topological_sort(self, w):
        """
        Performs a topological sort based on continuous edge weights.
        Iteratively picks the node with the minimum weighted in-degree.

        Args:
            w (torch.Tensor): The adjacency matrix (d x d) with values in [0, 1].
                              w[i, j] represents the strength of edge i -> j.
        Returns:
            list: A list of node indices in topological order.
        """
        # Work on a copy to avoid modifying the original matrix
        w_curr = w.clone()
        w_curr.fill_diagonal_(0)  # Ensure no self-loops
        d = self.num_vars

        # Calculate initial weighted in-degrees (sum of incoming edge weights)
        in_degree = w_curr.sum(dim=0)
        sampling_order = []

        for _ in range(d):
            u = torch.argmin(in_degree).item()
            sampling_order.append(u)
            in_degree -= w_curr[u, :]
            in_degree[u] = float("inf")

        # give an random order
        # import random
        # sampling_order = random.sample(range(self.num_vars), self.num_vars)
        
        print(f"Sampling order: {sampling_order}")
        return sampling_order

    @torch.no_grad()
    def sample(self, n_samples, device="cuda", order=None):
        """
        Generates synthetic data samples from the learned Bayesian network.

        Args:
            n_samples (int): The number of samples to generate.
            device (str): The device to perform computation on ('cpu' or 'cuda').
            order (sequence[int], optional): A complete explicit generation
                order. If absent, use the existing W-based order.

        Returns:
            pd.DataFrame: A DataFrame containing the generated data in its
                          original scale.
        """
        self.to(device)
        self.eval()

        if order is None:
            w = torch.sigmoid(self.W.data)
            sampling_order = self._topological_sort(w)
        else:
            sampling_order = validate_order(order, self.num_vars)
            print(f"Sampling order: {sampling_order}")
        self.last_sampling_order = sampling_order.copy()

        samples = torch.zeros(n_samples, self.num_vars, device=device)
        embeddings = self.mask_token.repeat(n_samples, self.num_vars, 1)
        for var_idx in sampling_order:
            predictions = self.forward(embeddings=embeddings)

            var_pred = predictions[var_idx]

            if var_idx in self.var_types.get("cat", {}):
                # Categorical: sample from the predicted distribution
                probs = F.softmax(var_pred, dim=-1)
                sampled_values = torch.multinomial(probs, num_samples=1).squeeze(-1)
            elif self.continuous_head == "gmm":
                sampled_values = sample_gmm(var_pred, self.gmm_components)
            elif self.continuous_head == "ddpm":
                # The Transformer context is computed once for this column;
                # only its small scalar denoiser runs at each reverse step.
                sampled_values = self.prediction_heads[str(var_idx)].sample(var_pred)
            else:
                # Continuous: sample from the predicted Gaussian
                mu, log_sigma = var_pred.chunk(2, dim=-1)
                # Add a small clamp to log_sigma to prevent extreme values from causing NaNs
                log_sigma = torch.clamp(log_sigma, max=10)
                sigma = torch.exp(log_sigma).clamp(min=1e-5)
                dist = torch.distributions.Normal(mu.squeeze(-1), sigma.squeeze(-1))
                sampled_values = dist.sample()

            # Update the samples tensor with the newly generated values
            samples[:, var_idx] = sampled_values

            # Update the embeddings for the next iteration
            if var_idx in self.var_types.get("cat", {}):
                embeddings[:, var_idx, :] = self.embedding_layers[str(var_idx)](
                    sampled_values.long()
                )
            else:
                embeddings[:, var_idx, :] = self.embedding_layers[str(var_idx)](
                    sampled_values.unsqueeze(1).float()
                )

        # Post-processing (Inverse-scaling)
        final_samples_df = pd.DataFrame(index=range(n_samples))

        # Use the raw numpy data for transformations
        raw_samples = samples.cpu().numpy()

        for i in range(self.num_vars):
            col = self.col_names[i]
            if i in self.scalers:
                scaler = self.scalers[i]
                # Reshape for scaler/encoder
                col_data = raw_samples[:, i].reshape(-1, 1)

                if isinstance(scaler, LabelEncoder):
                    # Inverse transform categorical data
                    col_data_int = col_data.astype(int)
                    final_samples_df[col] = scaler.inverse_transform(
                        col_data_int.ravel()
                    )
                else:
                    # Inverse transform continuous data
                    final_samples_df[col] = scaler.inverse_transform(col_data)
            else:
                # If a column has no scaler, copy it directly
                final_samples_df[col] = raw_samples[:, i]

        # for log columns inverse transform
        if self.log_cols:
            for col_idx in self.log_cols:
                col = self.col_names[col_idx]
                lower_bound = self.lower_bounds[col]
                if lower_bound > 0:
                    final_samples_df[col] = final_samples_df[col].apply(
                        lambda x: np.exp(x)
                    )
                elif lower_bound == 0:
                    final_samples_df[col] = final_samples_df[col].apply(
                        lambda x: (
                            np.ceil(np.exp(x) - 1)
                            if (np.exp(x) - 1) < 0
                            else (np.exp(x) - 1)
                        )
                    )
                else:
                    final_samples_df[col] = final_samples_df[col].apply(
                        lambda x: np.exp(x) - 1 + lower_bound
                    )

        return final_samples_df
