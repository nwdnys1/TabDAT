"""Fixed conditional tasks for non-DP checkpoint monitoring."""

from dataclasses import dataclass
import hashlib
import math

import torch

if __package__:
    from .mask_strategies import sample_training_masks
else:
    from mask_strategies import sample_training_masks


@dataclass(frozen=True)
class MonitorBank:
    data: torch.Tensor
    input_mask: torch.Tensor
    target_mask: torch.Tensor
    timesteps: torch.Tensor | None
    noise: torch.Tensor | None
    row_count: int
    repeats: int
    digest: str


def build_monitor_bank(
    data, *, num_vars, strategy, mask_prob, order, continuous_head,
    diffusion_steps, max_rows, repeats, seed,
):
    """Sample tasks once, on CPU, without advancing the training RNG."""
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[1] != num_vars:
        raise ValueError("Monitor data must be a [rows, num_vars] tensor")
    if data.shape[0] < 1 or max_rows < 1 or repeats < 1:
        raise ValueError("Monitor data, max_rows and repeats must be positive")
    if not torch.isfinite(data).all():
        raise ValueError("Monitor data contains non-finite values")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    row_count = min(len(data), max_rows)
    indices = torch.randperm(len(data), generator=generator)[:row_count]
    selected = data.detach().to("cpu")[indices].repeat_interleave(repeats, dim=0)
    input_mask, target_mask = sample_training_masks(
        len(selected), num_vars, strategy=strategy, mask_prob=mask_prob,
        order=order, device="cpu", generator=generator,
    )
    if continuous_head == "ddpm":
        timesteps = torch.randint(
            diffusion_steps, (len(selected), num_vars), generator=generator
        )
        noise = torch.randn((len(selected), num_vars), generator=generator)
    else:
        timesteps = noise = None

    digest = hashlib.sha256()
    for tensor in (selected, input_mask, target_mask, timesteps, noise):
        if tensor is not None:
            digest.update(tensor.contiguous().numpy().tobytes())
    return MonitorBank(
        selected, input_mask, target_mask, timesteps, noise,
        row_count, repeats, digest.hexdigest(),
    )


@torch.inference_mode()
def evaluate_monitor(model, bank, *, batch_size, device):
    """Evaluate exactly the same row/task pairs at every checkpoint epoch."""
    if batch_size < 1:
        raise ValueError("Monitor batch_size must be positive")
    was_training = model.training
    model.eval()
    total = categorical = continuous = 0.0
    column_sums = torch.zeros(model.num_vars, dtype=torch.float64)
    target_counts = torch.zeros(model.num_vars, dtype=torch.long)
    try:
        for start in range(0, len(bank.data), batch_size):
            stop = min(start + batch_size, len(bank.data))
            batch = bank.data[start:stop].to(device)
            input_mask = bank.input_mask[start:stop].to(device)
            target_mask = bank.target_mask[start:stop].to(device)
            embeddings = torch.zeros(
                len(batch), model.num_vars, model.embed_dim, device=device
            )
            for i in range(model.num_vars):
                values = batch[:, i]
                if i in model.var_types.get("cat", {}):
                    embeddings[:, i, :] = model.embedding_layers[str(i)](values.long())
                else:
                    embeddings[:, i, :] = model.embedding_layers[str(i)](
                        values.unsqueeze(1).float()
                    )
            masked = torch.where(
                input_mask.unsqueeze(-1),
                model.mask_token.expand(len(batch), model.num_vars, -1),
                embeddings,
            )
            outputs = model.forward(embeddings=masked)
            loss, parts = model.compute_loss(
                batch, outputs, target_mask, return_components=True,
                ddpm_timesteps=None if bank.timesteps is None else bank.timesteps[start:stop].to(device),
                ddpm_noise=None if bank.noise is None else bank.noise[start:stop].to(device),
                return_per_column=True,
            )
            total += float(loss) * len(batch)
            categorical += float(parts["categorical"]) * len(batch)
            continuous += float(parts["continuous"]) * len(batch)
            column_sums += (parts["per_column"] * len(batch)).to("cpu", dtype=torch.float64)
            target_counts += parts["target_counts"].to("cpu")
    finally:
        model.train(was_training)

    count = len(bank.data)
    if not math.isfinite(total):
        raise FloatingPointError("Non-finite checkpoint monitor loss")
    return {
        "loss": total / count,
        "categorical": categorical / count,
        "continuous": continuous / count,
        "per_column": (column_sums / count).tolist(),
        "target_counts": target_counts.tolist(),
        "task_rows": count,
    }
