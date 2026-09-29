"""Continuous output distributions used by TabDAT.

The Gaussian head remains in ``model.py`` to preserve legacy state-dict keys.
All values here are in the model's preprocessed (scaled) feature space.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gmm_nll(parameters, targets, components):
    """Per-row negative log likelihood of a scalar Gaussian mixture."""
    logits, means, log_stds = parameters.split(components, dim=-1)
    log_stds = log_stds.clamp(min=-7.0, max=7.0)
    normal = torch.distributions.Normal(means, log_stds.exp())
    component_log_probs = normal.log_prob(targets.unsqueeze(-1))
    return -torch.logsumexp(
        F.log_softmax(logits, dim=-1) + component_log_probs, dim=-1
    )


def sample_gmm(parameters, components):
    """Draw one scalar per row from a Gaussian mixture."""
    logits, means, log_stds = parameters.split(components, dim=-1)
    chosen = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    mean = means.gather(1, chosen).squeeze(1)
    log_std = log_stds.gather(1, chosen).squeeze(1).clamp(min=-7.0, max=7.0)
    return mean + log_std.exp() * torch.randn_like(mean)


class ConditionalDDPMHead(nn.Module):
    """A scalar epsilon-prediction DDPM conditioned on a column representation.

    ``forward`` passes the condition through so the model's existing
    target-independent forward API is preserved. The denoiser runs only for
    selected training targets or during this column's sampling step.
    """

    def __init__(self, context_dim, hidden_dim, steps, time_dim=32):
        super().__init__()
        if steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if hidden_dim < 1 or time_dim < 4 or time_dim % 2:
            raise ValueError("Invalid diffusion hidden or time dimension")

        self.steps = steps
        self.time_dim = time_dim
        self.denoiser = nn.Sequential(
            nn.Linear(context_dim + time_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # A cosine schedule reaches near-pure noise even with relatively few
        # steps. Build in float64, then keep float32 buffers on the model device.
        grid = torch.arange(steps + 1, dtype=torch.float64) / steps
        alpha_bar_curve = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
        alpha_bar_curve = alpha_bar_curve / alpha_bar_curve[0]
        betas = (1.0 - alpha_bar_curve[1:] / alpha_bar_curve[:-1]).clamp(max=0.999)
        betas = betas.float()
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)

        # The schedule is reconstructed from checkpoint config, not duplicated
        # in every continuous column's state_dict.
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        self.register_buffer("posterior_variance", posterior_variance, persistent=False)

    def forward(self, context):
        return context

    def _time_embedding(self, timesteps, dtype):
        half = self.time_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / (half - 1)
        )
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1).to(dtype=dtype)

    def predict_noise(self, noisy_values, timesteps, context):
        inputs = torch.cat(
            (
                noisy_values.unsqueeze(-1),
                self._time_embedding(timesteps, context.dtype),
                context,
            ),
            dim=-1,
        )
        return self.denoiser(inputs).squeeze(-1)

    def loss(self, context, targets):
        """Return one noise-prediction squared error per selected row."""
        timesteps = torch.randint(self.steps, (targets.shape[0],), device=targets.device)
        noise = torch.randn_like(targets)
        alpha_bar = self.alpha_bars[timesteps].to(dtype=targets.dtype)
        noisy = alpha_bar.sqrt() * targets + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self.predict_noise(noisy, timesteps, context)
        return (predicted_noise - noise).square()

    @torch.no_grad()
    def sample(self, context):
        """Draw one scalar per row, reusing the same context at every step."""
        values = torch.randn(context.shape[0], device=context.device, dtype=context.dtype)
        for step in range(self.steps - 1, -1, -1):
            timesteps = torch.full(
                (context.shape[0],), step, device=context.device, dtype=torch.long
            )
            predicted_noise = self.predict_noise(values, timesteps, context)
            beta = self.betas[step].to(dtype=values.dtype)
            alpha = self.alphas[step].to(dtype=values.dtype)
            alpha_bar = self.alpha_bars[step].to(dtype=values.dtype)
            values = (
                values - beta * predicted_noise / (1.0 - alpha_bar).sqrt()
            ) / alpha.sqrt()
            if step > 0:
                variance = self.posterior_variance[step].to(dtype=values.dtype)
                values = values + variance.sqrt() * torch.randn_like(values)
        return values
