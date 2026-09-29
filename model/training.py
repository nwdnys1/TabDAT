"""Ordinary TabDAT training, with a compatibility entry for legacy DP."""

import torch

if __package__:
    from .mask_strategies import (
        MASK_STRATEGIES,
        ORDERED_STRATEGIES,
        sample_training_masks,
        validate_order,
    )
else:
    from mask_strategies import (
        MASK_STRATEGIES,
        ORDERED_STRATEGIES,
        sample_training_masks,
        validate_order,
    )


class TrainingMixin:
    def fit(
        self,
        data=None,
        epochs=1000,
        batch_size=1024,
        lr=1e-3,
        mask_prob=0.15,
        test_split_ratio=0.2,
        sample=None,
        lr_decay_gamma=0.999,
        dp=False,
        dp_epsilon=1.0,
        dp_sigma=1.02,
        dp_clip=1.0,
        dp_micro_batch_size=1,
        mask_strategy="bernoulli_all",
        training_order=None,
    ):
        """
        Trains the TabDAT model.

        Args:
            data (torch.Tensor, optional): The training data tensor. If None, the model will use
                                           the data pre-loaded during initialization.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
            mask_prob (float): Mask probability for bernoulli_all only.
            test_split_ratio (float): The ratio of the dataset to be used for testing.
            lr_decay_gamma (float): Gamma for exponential learning rate decay.
            dp (bool): Whether to enable differential privacy.
            dp_epsilon (float): Epsilon parameter for differential privacy.
            dp_sigma (float): Noise multiplier for differential privacy.
            dp_clip (float): Clipping threshold for gradients.
            dp_micro_batch_size (int): Micro-batch size for differential privacy.
            mask_strategy (str): Training task sampler; default retains the old
                independent Bernoulli masking.
            training_order (sequence[int], optional): Complete column order
                required by ordered strategies, chosen before training.
        """
        if dp:
            if mask_strategy != "bernoulli_all" or training_order is not None:
                raise NotImplementedError(
                    "Legacy DP training only supports bernoulli_all without a fixed order"
                )
            return self.fit_dp(
                data=data, epochs=epochs, batch_size=batch_size, lr=lr,
                mask_prob=mask_prob, test_split_ratio=test_split_ratio,
                sample=sample, lr_decay_gamma=lr_decay_gamma,
                dp_epsilon=dp_epsilon, dp_sigma=dp_sigma, dp_clip=dp_clip,
                dp_micro_batch_size=dp_micro_batch_size,
            )

        if mask_strategy not in MASK_STRATEGIES:
            raise ValueError(f"Unknown mask strategy: {mask_strategy}")
        if mask_strategy in ORDERED_STRATEGIES:
            training_order = validate_order(training_order, self.num_vars)
        elif training_order is not None:
            raise ValueError(f"{mask_strategy} does not use a fixed training order")
        if mask_strategy == "bernoulli_all" and not 0 <= mask_prob <= 1:
            raise ValueError("mask_prob must be in [0, 1]")

        # Use pre-loaded data if no new data is provided
        if data is None:
            if self.data is not None:
                data = self.data
            else:
                raise ValueError(
                    "Training data not found. Please provide data to the `fit` method or specify `file_path` during model initialization."
                )

        # --- Dataset Splitting ---
        dataset_size = len(data)
        test_size = int(test_split_ratio * dataset_size)
        train_size = dataset_size - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            data, [train_size, test_size]
        )

        device = self.device
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_decay_gamma
        )

        # Using TensorDataset and DataLoader for batching on the training set
        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.training_config = {
            "mask_strategy": mask_strategy,
            "mask_prob": mask_prob,
            "training_order": training_order,
            "dp": False,
        }

        print("--- Starting Training ---")
        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        for epoch in range(epochs):

            total_loss = 0
            categorical_loss = 0.0
            continuous_loss = 0.0
            self.train()
            for batch_data in dataloader:
                batch_data = batch_data.to(device)

                optimizer.zero_grad()

                # Build one embedding for every column, then hide input columns.
                embeddings = torch.zeros(
                    batch_data.shape[0], self.num_vars, self.embed_dim, device=device
                )
                for i in range(self.num_vars):
                    data_slice = batch_data[:, i].unsqueeze(1)
                    if i in self.var_types.get("cat", {}):
                        embeddings[:, i, :] = self.embedding_layers[str(i)](
                            data_slice.long().squeeze(1)
                        )
                    else:
                        embeddings[:, i, :] = self.embedding_layers[str(i)](
                            data_slice.float()
                        )

                input_mask, target_mask = sample_training_masks(
                    embeddings.shape[0], self.num_vars, strategy=mask_strategy,
                    mask_prob=mask_prob, order=training_order, device=device,
                )
                mask_embed = self.mask_token.repeat(
                    embeddings.shape[0], self.num_vars, 1
                )
                masked_embeddings = torch.where(
                    input_mask.unsqueeze(-1), mask_embed, embeddings
                )
                outputs = self.forward(embeddings=masked_embeddings)
                loss, loss_parts = self.compute_loss(
                    batch_data, outputs, target_mask, return_components=True
                )
                total_loss += loss.item()
                categorical_loss += loss_parts["categorical"].item()
                continuous_loss += loss_parts["continuous"].item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                total_loss /= len(dataloader)
                print(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f} "
                    f"(cat: {categorical_loss / len(dataloader):.4f}, "
                    f"cont: {continuous_loss / len(dataloader):.4f})"
                )

            if sample and (epoch + 1) % sample == 0:
                self.sample(self.data.shape[0], device=device)

        print("--- Training Finished ---")

    def fit_dp(self, **kwargs):
        """Opt in to the isolated legacy DP procedure."""
        if __package__:
            from .dp_training import fit_dp_legacy
        else:
            from dp_training import fit_dp_legacy
        return fit_dp_legacy(self, **kwargs)
