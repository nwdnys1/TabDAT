"""TabDAT methods extracted without changing their implementation."""

import torch

if __package__:
    from .dp_training import compute_dp_epsilon, dp_training_step
else:
    from dp_training import compute_dp_epsilon, dp_training_step


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
    ):
        """
        Trains the TabDAT model.

        Args:
            data (torch.Tensor, optional): The training data tensor. If None, the model will use
                                           the data pre-loaded during initialization.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
            mask_prob (float): Probability of masking an input token during training.
            test_split_ratio (float): The ratio of the dataset to be used for testing.
            lr_decay_gamma (float): Gamma for exponential learning rate decay.
            dp (bool): Whether to enable differential privacy.
            dp_epsilon (float): Epsilon parameter for differential privacy.
            dp_sigma (float): Noise multiplier for differential privacy.
            dp_clip (float): Clipping threshold for gradients.
            dp_micro_batch_size (int): Micro-batch size for differential privacy.
        """
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
        train_dataset, _ = torch.utils.data.random_split(data, [train_size, test_size])

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

        print("--- Starting Training ---")
        if dp:
            print(
                f"DPSGD Enabled: Sigma={dp_sigma}, Clip={dp_clip}, MicroBatch={dp_micro_batch_size}"
            )

        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(_)}")

        dp_steps = 0

        for epoch in range(epochs):

            total_loss = 0
            self.train()
            for batch_data in dataloader:
                batch_data = batch_data.to(device)

                optimizer.zero_grad()

                if dp:
                    total_loss += dp_training_step(
                        self,
                        batch_data,
                        optimizer,
                        mask_prob,
                        dp_sigma,
                        dp_clip,
                        dp_micro_batch_size,
                        device,
                    )

                    dp_steps += 1

                else:

                    # --- Masking Augmentation ---
                    # 1. Get original embeddings
                    embeddings = torch.zeros(
                        batch_data.shape[0],
                        self.num_vars,
                        self.embed_dim,
                        device=device,
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

                    # 2. Create mask and apply it
                    mask_indices = (
                        torch.rand(embeddings.shape[:2], device=device) < mask_prob
                    )
                    mask_embed = self.mask_token.repeat(
                        embeddings.shape[0], self.num_vars, 1
                    )
                    masked_embeddings = torch.where(
                        mask_indices.unsqueeze(-1), mask_embed, embeddings
                    )
                    outputs = self.forward(embeddings=masked_embeddings)

                    loss = self.compute_loss(
                        batch_data,
                        outputs,
                        mask_indices,
                    )
                    total_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

            scheduler.step()

            if dp:
                # --- Privacy Accounting ---
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
                df = self.sample(
                    self.data.shape[0], device=device
                )  # sampling during training

        print("--- Training Finished ---")
