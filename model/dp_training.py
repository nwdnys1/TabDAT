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


def fit_dp_legacy(
    model,
    data=None,
    epochs=1000,
    batch_size=1024,
    lr=1e-3,
    mask_prob=0.15,
    test_split_ratio=0.2,
    sample=None,
    lr_decay_gamma=0.999,
    dp_epsilon=1.0,
    dp_sigma=1.02,
    dp_clip=1.0,
    dp_micro_batch_size=1,
):
    """Keep the old DP loop separate from ordinary training.

    This is a compatibility extraction, not a privacy guarantee or audit.
    """
    if model.continuous_head != "gaussian":
        raise NotImplementedError(
            "Legacy DP training has not been audited for GMM or DDPM heads"
        )
    if data is None:
        if model.data is not None:
            data = model.data
        else:
            raise ValueError(
                "Training data not found. Please provide data to the `fit` method or specify `file_path` during model initialization."
            )

    dataset_size = len(data)
    test_size = int(test_split_ratio * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        data, [train_size, test_size]
    )
    device = model.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=lr_decay_gamma
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    model.training_config = {
        "mask_strategy": "bernoulli_all",
        "mask_prob": mask_prob,
        "training_order": None,
        "dp": True,
    }

    print("--- Starting Training ---")
    print(
        f"DPSGD Enabled: Sigma={dp_sigma}, Clip={dp_clip}, MicroBatch={dp_micro_batch_size}"
    )
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    dp_steps = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            total_loss += dp_training_step(
                model, batch_data, optimizer, mask_prob, dp_sigma, dp_clip,
                dp_micro_batch_size, device,
            )
            dp_steps += 1

        scheduler.step()
        epsilon = compute_dp_epsilon(
            batch_size, len(train_dataset), dp_sigma, dp_steps
        )
        print(f"Epoch {epoch+1}: ε = {epsilon:.4f} for δ = 1e-5")
        if epsilon > dp_epsilon:
            print(
                f"Reached privacy budget of epsilon = {dp_epsilon}. Stopping training."
            )
            break
        if (epoch + 1) % 10 == 0:
            total_loss /= len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f} ")
        if sample and (epoch + 1) % sample == 0:
            model.sample(model.data.shape[0], device=device)

    print("--- Training Finished ---")
