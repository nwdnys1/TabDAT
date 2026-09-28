"""Legacy DP training step and privacy accounting, extracted from training.py.

This module preserves the existing implementation; it is not a privacy audit.
"""

import math

import torch

if __package__:
    from .utils.rdp_accountant import compute_rdp, get_privacy_spent
else:
    from utils.rdp_accountant import compute_rdp, get_privacy_spent


def dp_training_step(
    model,
    batch_data,
    optimizer,
    mask_prob,
    dp_sigma,
    dp_clip,
    dp_micro_batch_size,
    device,
):
    """Run one batch using the original micro-batch DP update."""
    saved_grads = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    current_batch_size = batch_data.size(0)
    num_micro_batches = math.ceil(current_batch_size / dp_micro_batch_size)

    batch_loss_accum = 0

    for k in range(num_micro_batches):
        start_idx = k * dp_micro_batch_size
        end_idx = min((k + 1) * dp_micro_batch_size, current_batch_size)
        micro_data = batch_data[start_idx:end_idx]
        embeddings = torch.zeros(
            micro_data.shape[0],
            model.num_vars,
            model.embed_dim,
            device=device,
        )
        for i in range(model.num_vars):
            data_slice = micro_data[:, i].unsqueeze(1)
            if i in model.var_types.get("cat", {}):
                embeddings[:, i, :] = model.embedding_layers[str(i)](
                    data_slice.long().squeeze(1)
                )
            else:
                embeddings[:, i, :] = model.embedding_layers[str(i)](
                    data_slice.float()
                )

        mask_indices = torch.rand(embeddings.shape[:2], device=device) < mask_prob
        mask_embed = model.mask_token.repeat(
            embeddings.shape[0], model.num_vars, 1
        )
        masked_embeddings = torch.where(
            mask_indices.unsqueeze(-1), mask_embed, embeddings
        )

        outputs = model.forward(embeddings=masked_embeddings)

        loss = model.compute_loss(micro_data, outputs, mask_indices)

        batch_loss_accum += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), dp_clip)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                saved_grads[name] += param.grad

        model.zero_grad()

    for name, param in model.named_parameters():
        if param.requires_grad:
            noise = torch.normal(
                0, dp_sigma * dp_clip, size=param.shape, device=device
            )
            param.grad = (saved_grads[name] + noise) / num_micro_batches

    optimizer.step()

    return batch_loss_accum / num_micro_batches


def compute_dp_epsilon(batch_size, train_dataset_size, dp_sigma, dp_steps):
    """Compute epsilon with the original per-epoch accounting parameters."""
    lmbds = range(2, 4096)
    rdp = compute_rdp(
        batch_size / train_dataset_size,
        dp_sigma,
        dp_steps,
        lmbds,
    )
    epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5)
    return epsilon
