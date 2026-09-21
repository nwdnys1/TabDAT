import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import os
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    QuantileTransformer,
    MinMaxScaler,
)
from utils.rdp_accountant import compute_rdp, get_privacy_spent

torch.backends.mha.set_fastpath_enabled(False)


class TabDAT(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        cont_scaler="standard",
        file_path=None,
        cat_cols=None,
        log_cols=None,
        dropout=0.1,
        device="cuda",
    ):
        """
        Transformer-based Hybrid Bayesian Network.

        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of heads in the multi-head attention models.
            num_layers (int): The number of sub-encoder-layers in the encoder.
            cont_scaler (str): The type of scaler to use for continuous variables.
            file_path (str, optional): Path to the data file (e.g., CSV).
                                       If provided, `num_vars`, `var_types`, and data are automatically loaded and processed.
            cat_cols (list, optional): List of categorical column indices.
            log_cols (list, optional): List of logarithmic transformation column indices.
            dropout (float): The dropout value.
        """
        super(TabDAT, self).__init__()

        self.data = None
        self.file_path = file_path
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cont_scaler = cont_scaler
        self.dropout = dropout
        self.scalers = {}
        self.device = device
        self.cat_cols = cat_cols
        self.log_cols = log_cols

        self._load_and_preprocess_data(file_path)

        # 1. Trainable Adjacency Matrix (W)
        self.W = nn.Parameter(torch.randn(self.num_vars, self.num_vars))

        # 2. Input Embedding Layers (E)
        self.embedding_layers = nn.ModuleDict()
        # Discrete variable embeddings
        for idx, cardinality in self.var_types.get("cat", {}).items():
            self.embedding_layers[str(idx)] = nn.Embedding(cardinality, embed_dim)
        # Continuous variable linear projections
        for idx in self.var_types.get("cont", []):
            self.embedding_layers[str(idx)] = nn.Linear(1, embed_dim)

        # 3. Shared Transformer Encoder (T)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Special [MASK] token embedding for generation
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 4. Prediction Heads (H)
        self.prediction_heads = nn.ModuleDict()
        for i in range(self.num_vars):
            # For discrete variables, output logits for each category
            if i in self.var_types.get("cat", {}):
                cardinality = self.var_types["cat"][i]
                self.prediction_heads[str(i)] = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 2, cardinality),
                )
            # For continuous variables, output mean and log_std
            else:
                self.prediction_heads[str(i)] = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 2, 2),  # mu and log_sigma
                )

    def _load_and_preprocess_data(self, file_path):
        """
        Loads data from a file, infers types, preprocesses it, and stores it.
        """
        df = pd.read_csv(file_path).dropna()
        self.col_names = df.columns.tolist()
        self.num_vars = len(df.columns)

        # --- Infer Variable Types ---
        cat_cols = {}
        cont_cols = []
        if self.cat_cols is None:
            for i, col_name in enumerate(df.columns):
                col = df[col_name]
                # 1. Explicitly check for object or bool types first.
                if col.dtype == "object" or col.dtype == "bool":
                    cat_cols[i] = col.nunique(dropna=False)
                    continue
                numeric_col = pd.to_numeric(df[col_name], errors="coerce")
                if numeric_col.isnull().sum() > 0 or df[col_name].dtype == "object":
                    cardinality = df[col_name].nunique(dropna=False)
                    cat_cols[i] = cardinality
                else:
                    cont_cols.append(i)
        else:
            for idx in self.cat_cols:
                cat_cols[idx] = df.iloc[:, idx].nunique(dropna=False)
            for i in range(self.num_vars):
                if i not in cat_cols:
                    cont_cols.append(i)
        self.var_types = {"cat": cat_cols, "cont": cont_cols}

        # for log columns
        if self.log_cols:
            self.lower_bounds = {}
            for col_idx in self.log_cols:
                col = df.columns[col_idx]
                lower = np.min(df[col].values)
                # store lower bound for inverse transform
                self.lower_bounds[col] = lower
                if lower > 0:
                    df[col] = df[col].apply(lambda x: np.log(x))
                elif lower == 0:
                    df[col] = df[col].apply(lambda x: np.log(x + 1))
                else:
                    df[col] = df[col].apply(lambda x: np.log(x - lower + 1))

        # --- Fit Scalers and Transform Data ---
        processed_data = df.copy()
        for i in range(self.num_vars):
            col_name = df.columns[i]
            if i in self.var_types.get("cont", []):
                if self.cont_scaler == "minmax":
                    scaler = MinMaxScaler()
                elif self.cont_scaler == "quantile":
                    scaler = QuantileTransformer(output_distribution="normal")
                elif self.cont_scaler == "standard":
                    scaler = StandardScaler()
                processed_data[col_name] = scaler.fit_transform(
                    df[[col_name]].values
                ).flatten()
                self.scalers[i] = scaler
            elif i in self.var_types.get("cat", {}):
                encoder = LabelEncoder()
                processed_data[col_name] = encoder.fit_transform(df[col_name].values)
                self.scalers[i] = encoder

        # Store the processed data as a tensor
        self.data = torch.tensor(processed_data.values, dtype=torch.float32)
        print(self.data)

    def _create_mask(self):
        """
        Creates a attention mask from the adjacency matrix W.
        The mask uses log(sigmoid(W)) to modulate attention scores,
        and manually ensures that self-attention is always forbidden.
        """
        mask = F.logsigmoid(self.W.t())

        mask.fill_diagonal_(float("-inf"))

        return mask

    def forward(self, embeddings):
        """
        Forward pass of the model.

        Args:
            embeddings (torch.Tensor, optional): Pre-computed embeddings of shape
                                                 (batch_size, num_vars, embed_dim).

        Returns:
            dict: A dictionary containing the predicted distributions for each variable.
        """

        # 1. Causal Attention Masking
        mask = self._create_mask()

        # 2. Transformer Encoding
        # Input shape: (batch_size, seq_len, embed_dim)
        # Mask shape: (seq_len, seq_len)
        contextual_repr = self.transformer_encoder(embeddings, mask)

        # 3. Conditional Probability Distribution Prediction
        outputs = {}
        for i in range(self.num_vars):
            head_input = contextual_repr[:, i, :]
            outputs[i] = self.prediction_heads[str(i)](head_input)

        return outputs

    def compute_loss(self, x, outputs):
        """
        Computes the total loss.
        L = Negative Log-Likelihood

        Args:
            x (torch.Tensor): The input data.
            outputs (dict): The output from the forward pass.

        Returns:
            float: The total loss.
        """

        # Negative Log-Likelihood
        nll = 0
        for i in range(self.num_vars):
            target = x[:, i]
            prediction = outputs[i]
            if i in self.var_types.get("cat", {}):
                # Categorical Cross-Entropy Loss
                nll += F.cross_entropy(prediction, target.long(), reduction="sum")
            else:
                # Gaussian Negative Log-Likelihood
                mu, log_sigma = prediction.chunk(2, dim=-1)
                sigma = torch.exp(log_sigma)
                # Clamp sigma to avoid numerical instability
                sigma = torch.clamp(sigma, min=1e-5)
                dist = torch.distributions.Normal(mu.squeeze(), sigma.squeeze())
                nll -= dist.log_prob(target).sum()

        nll /= x.shape[0]  # Average over batch

        return nll

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
        # print(f"Sampling order: {sampling_order}")
        # return sampling_order

    @torch.no_grad()
    def sample(self, n_samples, device="cuda"):
        """
        Generates synthetic data samples from the learned Bayesian network.

        Args:
            n_samples (int): The number of samples to generate.
            device (str): The device to perform computation on ('cpu' or 'cuda').

        Returns:
            pd.DataFrame: A DataFrame containing the generated data in its
                          original scale.
        """
        self.to(device)
        self.eval()

        w = torch.sigmoid(self.W.data)
        sampling_order = self._topological_sort(w)

        samples = torch.zeros(n_samples, self.num_vars, device=device)
        embeddings = self.mask_token.repeat(n_samples, self.num_vars, 1)
        for var_idx in sampling_order:
            predictions = self.forward(embeddings=embeddings)

            var_pred = predictions[var_idx]

            if var_idx in self.var_types.get("cat", {}):
                # Categorical: sample from the predicted distribution
                probs = F.softmax(var_pred, dim=-1)
                sampled_values = torch.multinomial(probs, num_samples=1).squeeze(-1)
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
                    saved_grads = {
                        name: torch.zeros_like(param)
                        for name, param in self.named_parameters()
                        if param.requires_grad
                    }

                    current_batch_size = batch_data.size(0)
                    num_micro_batches = math.ceil(
                        current_batch_size / dp_micro_batch_size
                    )

                    batch_loss_accum = 0

                    for k in range(num_micro_batches):
                        start_idx = k * dp_micro_batch_size
                        end_idx = min((k + 1) * dp_micro_batch_size, current_batch_size)
                        micro_data = batch_data[start_idx:end_idx]
                        embeddings = torch.zeros(
                            micro_data.shape[0],
                            self.num_vars,
                            self.embed_dim,
                            device=device,
                        )
                        for i in range(self.num_vars):
                            data_slice = micro_data[:, i].unsqueeze(1)
                            if i in self.var_types.get("cat", {}):
                                embeddings[:, i, :] = self.embedding_layers[str(i)](
                                    data_slice.long().squeeze(1)
                                )
                            else:
                                embeddings[:, i, :] = self.embedding_layers[str(i)](
                                    data_slice.float()
                                )

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

                        loss = self.compute_loss(micro_data, outputs)

                        batch_loss_accum += loss.item()

                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.parameters(), dp_clip)

                        for name, param in self.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                saved_grads[name] += param.grad

                        self.zero_grad()

                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            noise = torch.normal(
                                0, dp_sigma * dp_clip, size=param.shape, device=device
                            )
                            param.grad = (saved_grads[name] + noise) / num_micro_batches

                    optimizer.step()

                    total_loss += batch_loss_accum / num_micro_batches

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
                    )
                    total_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

            scheduler.step()

            if dp:
                # --- Privacy Accounting ---
                lmbds = range(2, 4096)
                rdp = compute_rdp(
                    batch_size / len(train_dataset),
                    dp_sigma,
                    dp_steps,
                    lmbds,
                )
                epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5)
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

    def save(self, path):
        """
        Saves the model state, configuration, and scalers to a file.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "file_path": self.file_path,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "cat_cols": self.cat_cols,
                "log_cols": self.log_cols,
                "dropout": self.dropout,
                "cont_scaler": self.cont_scaler,
                "col_names": self.col_names,  # Save column names
            },
            "scalers": self.scalers,  # Save sklearn scalers (pickle)
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, device="cuda"):
        """
        Loads a model from a file.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]

        # Re-instantiate the model using the saved configuration
        model = cls(
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            cat_cols=config["cat_cols"],
            log_cols=config["log_cols"],
            dropout=config["dropout"],
            cont_scaler=config["cont_scaler"],
            device=device,
            file_path=config["file_path"],
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load scalers and column names
        model.scalers = checkpoint["scalers"]
        model.col_names = config["col_names"]

        model.to(device)
        model.eval()
        print(f"Model loaded from {path}")
        return model
