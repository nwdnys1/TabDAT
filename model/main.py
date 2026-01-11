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
from rdp_accountant import compute_rdp, get_privacy_spent

torch.backends.mha.set_fastpath_enabled(False)


class TransformerBN(nn.Module):
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
            file_path (str, optional): Path to the data file (e.g., CSV).
                                       If provided, `num_vars`, `var_types`, and data are automatically loaded and processed.
            num_vars (int, optional): Number of variables (d). Required if `file_path` is not provided.
            var_types (dict, optional): Dictionary specifying variable types. Required if `file_path` is not provided.
                                        Example: {'cat': {0: 2, 2: 5}, 'cont': [1, 3]}
            dropout (float): The dropout value.
        """
        super(TransformerBN, self).__init__()

        self.data = None  # To store preprocessed data
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
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

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
        if self.cat_cols is None:  # 自动推断类别变量
            for i, col_name in enumerate(df.columns):
                col = df[col_name]
                # 1. Explicitly check for object or bool types first.
                if col.dtype == "object" or col.dtype == "bool":
                    cat_cols[i] = col.nunique(
                        dropna=False
                    )  # 和LabelEncoder逻辑保持一致 nan也作为一个类别
                    continue
                numeric_col = pd.to_numeric(df[col_name], errors="coerce")
                if numeric_col.isnull().sum() > 0 or df[col_name].dtype == "object":
                    cardinality = df[col_name].nunique(dropna=False)
                    cat_cols[i] = cardinality
                else:
                    cont_cols.append(i)
        else:  # 使用用户提供的类别变量列表
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

    def _create_causal_mask(self, tau=1.0):
        """
        Creates a causal attention mask from the adjacency matrix W.
        The mask uses log(sigmoid(W)) to modulate attention scores,
        and manually ensures that self-attention is always allowed.
        """
        # Requirement 2: Use log(sigmoid(W)) as the attention mask
        w_prob = torch.sigmoid(self.W)

        # Add a small epsilon for numerical stability
        # mask = torch.log(w_prob.t())
        mask = F.logsigmoid(self.W.t())
        # mask = (w_prob.t() - 1.0) * 1e9  # Large negative value for disallowed edges

        # Manually allow self-attention by setting the diagonal to 0
        # This decouples the graph's acyclicity constraint from the Transformer's self-attention need.
        # mask.fill_diagonal_(0)
        mask.fill_diagonal_(float("-inf"))

        return mask

    def forward(self, embeddings, tau=1.0, mask=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, num_vars).
                              Used for calculating embeddings if they are not provided.
            embeddings (torch.Tensor, optional): Pre-computed embeddings of shape
                                                 (batch_size, num_vars, embed_dim).
                                                 If provided, `x` is ignored for embedding calculation.

        Returns:
            dict: A dictionary containing the predicted distributions for each variable.
        """

        # 1. Causal Attention Masking
        # The mask is shared across the batch.
        # TransformerEncoderLayer expects mask of shape (L, S) or (N*num_heads, L, S)
        # where L is target sequence length, S is source sequence length, N is batch size.
        # Here L=S=num_vars.

        if mask is None:
            causal_mask = self._create_causal_mask(tau=tau)
        else:
            causal_mask = mask

        # 2. Transformer Encoding
        # Input shape: (batch_size, seq_len, embed_dim)
        # Mask shape: (seq_len, seq_len)
        contextual_repr = self.transformer_encoder(embeddings, mask=causal_mask)

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
                # TODO:也可以用F.nll_loss
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
        # in_degree[j] = sum(w[:, j])
        in_degree = w_curr.sum(dim=0)
        topo_order = []

        for _ in range(d):
            # Find the node with the minimum in-degree among unvisited nodes
            # argmin will always return an index, even if min value > 0 (handling cycles gracefully)
            u = torch.argmin(in_degree).item()
            topo_order.append(u)

            # "Remove" node u: its outgoing edges no longer contribute to others' in-degrees
            # w_curr[u, :] contains weights of edges u -> v.
            # Subtracting this row removes u's contribution to v's in-degree.
            in_degree -= w_curr[u, :]

            # Mark u as visited by setting its in-degree to infinity so it's not picked again
            in_degree[u] = float("inf")

        print(f"Topological order for sampling: {topo_order}")
        return topo_order

    @torch.no_grad()
    def sample(self, n_samples, device="cpu"):
        """
        Generates synthetic data samples from the learned Bayesian network.

        Args:
            n_samples (int): The number of samples to generate.
            w_threshold (float): The threshold to prune the adjacency matrix W.
            device (str): The device to perform computation on ('cpu' or 'cuda').

        Returns:
            pd.DataFrame: A DataFrame containing the generated data in its
                          original scale.
        """
        self.to(device)
        self.eval()

        # 1. Preparation Phase
        w_prob = torch.sigmoid(self.W.data)
        # # 采样出确定的W作为mask 大于threshold的位置为1，其余为0
        # w_binary = (w_prob > w_threshold).float()
        # topo_order = self._topological_sort(w_binary)
        topo_order = self._topological_sort(w_prob)

        # 2. Sequential Generation
        # Initialize samples with zeros.
        samples = torch.zeros(n_samples, self.num_vars, device=device)
        # Initialize all input embeddings with the mask token
        embeddings = self.mask_token.repeat(n_samples, self.num_vars, 1)
        # Iterate through variables in topological order
        for var_idx in topo_order:
            # Pass the current embeddings to the model
            # The causal mask ensures that the prediction for var_idx only depends on
            # its parents, which have already been sampled.
            predictions = self.forward(embeddings=embeddings)

            # Get the prediction for the current variable
            var_pred = predictions[var_idx]

            # 4. Sample and Update
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
            # The variable we just sampled now has a known value, so we replace
            # its MASK embedding with the true embedding of its sampled value.
            if var_idx in self.var_types.get("cat", {}):
                embeddings[:, var_idx, :] = self.embedding_layers[str(var_idx)](
                    sampled_values.long()
                )
            else:
                embeddings[:, var_idx, :] = self.embedding_layers[str(var_idx)](
                    sampled_values.unsqueeze(1).float()
                )

        # 3. Post-processing (Inverse-scaling)
        # Create a new, empty DataFrame to build the results.
        # This avoids the dtype incompatibility warning by allowing pandas to infer dtypes column by column.
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
                    # Ensure data is integer type for inverse_transform
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

    @torch.no_grad()
    def evaluate(
        self, test_data, batch_size, lambda1, lambda2, mask_prob, device="cpu"
    ):
        """
        Evaluates the model on the test set.

        Args:
            test_data (torch.Tensor): The test data tensor.
            batch_size (int): Batch size for evaluation.
            device (str): The device to perform computation on.

        Returns:
            float: The average Negative Log-Likelihood on the test set.
        """
        self.to(device)
        self.eval()  # Set the model to evaluation mode

        test_dataset = torch.utils.data.TensorDataset(test_data)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size
        )

        total_nll = 0
        total_loss = 0
        total_comp = 0
        total_acyclic = 0

        for (batch_data,) in test_dataloader:
            batch_data = batch_data.to(device)

            # 1. Get original embeddings
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

            # 2. Create mask and apply it
            # Probability of masking is mask_prob
            mask_indices = torch.rand(embeddings.shape[:2], device=device) < mask_prob
            # Expand mask_token to match batch dimensions
            mask_embed = self.mask_token.repeat(embeddings.shape[0], self.num_vars, 1)
            # Replace embeddings with mask token where mask is True
            masked_embeddings = torch.where(
                mask_indices.unsqueeze(-1), mask_embed, embeddings
            )

            # Forward pass with masked embeddings
            outputs = self.forward(embeddings=masked_embeddings)

            # Compute loss (still using original batch_data for targets)
            loss, nll, comp, acyclic = self.compute_loss(
                batch_data, outputs, lambda1=lambda1, lambda2=lambda2
            )

            total_nll += nll.item()
            total_loss += loss.item()
            total_comp += comp.item()
            total_acyclic += acyclic.item()

        total_nll /= len(test_dataloader)
        total_loss /= len(test_dataloader)
        total_comp /= len(test_dataloader)
        total_acyclic /= len(test_dataloader)

        return total_nll, total_loss, total_comp, total_acyclic

    def fit(
        self,
        data=None,
        epochs=100,
        batch_size=32,
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
        Trains the TransformerBN model.

        Args:
            data (torch.Tensor, optional): The training data tensor. If None, the model will use
                                           the data pre-loaded during initialization.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
            lambda1 (float): Regularization parameter for complexity.
            lambda2 (float): Regularization parameter for acyclicity.
            device (str): The device to perform computation on.
            mask_prob (float): Probability of masking an input token during training.
            test_split_ratio (float): The ratio of the dataset to be used for testing.
            lambda2_max (float): Maximum value for lambda2.
            lambda2_update_freq (int): Frequency (in epochs) to update lambda2.
            lambda2_growth_rate (float): Factor to multiply lambda2 by.
            lr_decay_gamma (float): Gamma for exponential learning rate decay.
            tau_start (float): Initial temperature for Gumbel-Softmax.
            tau_end (float): Final temperature for Gumbel-Softmax.
            dp (bool): Whether to enable differential privacy.
            dp_sigma (float): Noise multiplier for differential privacy.
            dp_clip (float): Clipping threshold for gradients.
            dp_micro_batch_size (int): Micro-batch size for differential privacy.
        """
        # dp_micro_batch_size = batch_size
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
        test_data = test_dataset.dataset[test_dataset.indices]

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
        print(f"Test set size: {len(test_dataset)}")

        dp_steps = 0

        for epoch in range(epochs):

            total_loss = 0
            self.train()  # Set model to training mode for each epoch
            for batch_data in dataloader:
                batch_data = batch_data.to(device)

                optimizer.zero_grad()

                if dp:
                    # ================= DPSGD 逻辑 =================

                    # 1. 初始化梯度累加器
                    saved_grads = {
                        name: torch.zeros_like(param)
                        for name, param in self.named_parameters()
                        if param.requires_grad
                    }

                    # 计算微批次数量
                    current_batch_size = batch_data.size(0)
                    num_micro_batches = math.ceil(
                        current_batch_size / dp_micro_batch_size
                    )

                    batch_loss_accum = 0

                    for k in range(num_micro_batches):
                        # 2. 获取微批次数据
                        start_idx = k * dp_micro_batch_size
                        end_idx = min((k + 1) * dp_micro_batch_size, current_batch_size)
                        micro_data = batch_data[start_idx:end_idx]

                        # --- Masking Augmentation (针对微批次) ---
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

                        # 3. 前向传播与 Loss 计算
                        outputs = self.forward(embeddings=masked_embeddings)

                        loss = self.compute_loss(micro_data, outputs)

                        # 记录 Loss (用于日志)
                        batch_loss_accum += loss.item()

                        # 4. 反向传播
                        loss.backward()

                        # 5. 梯度裁剪 (Per-micro-batch Clipping)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), dp_clip)

                        # 6. 累加裁剪后的梯度
                        for name, param in self.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                saved_grads[name] += param.grad

                        # 清空梯度，为下一个微批次做准备
                        self.zero_grad()

                    # 7. 加噪与平均 (Add Noise & Average)
                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            # 生成高斯噪声
                            noise = torch.normal(
                                0, dp_sigma * dp_clip, size=param.shape, device=device
                            )
                            # 恢复梯度：(累加的裁剪梯度 + 噪声) / 微批次数量
                            # 注意：这里除以 num_micro_batches 是因为 compute_loss 内部已经是 mean reduction
                            # 如果 compute_loss 是 sum reduction，这里逻辑会有所不同
                            param.grad = (saved_grads[name] + noise) / num_micro_batches

                    # 8. 更新参数
                    optimizer.step()

                    # 更新 Epoch 统计数据
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
                    # Probability of masking is mask_prob
                    mask_indices = (
                        torch.rand(embeddings.shape[:2], device=device) < mask_prob
                    )

                    # Expand mask_token to match batch dimensions
                    mask_embed = self.mask_token.repeat(
                        embeddings.shape[0], self.num_vars, 1
                    )
                    # Replace embeddings with mask token where mask is True
                    masked_embeddings = torch.where(
                        mask_indices.unsqueeze(-1), mask_embed, embeddings
                    )

                    # Forward pass with masked embeddings
                    outputs = self.forward(embeddings=masked_embeddings)

                    # Compute loss (still using original batch_data for targets)
                    loss = self.compute_loss(
                        batch_data,
                        outputs,
                    )
                    total_loss += loss.item()

                    # Backward pass and optimization
                    loss.backward()
                    # Gradient clipping to prevent explosion
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
                    df = self.sample(
                        self.data.shape[0], device=device
                    )  # Test sampling during training
                    os.makedirs(
                        f"TransformerBN/eval/fake_datasets/{dataset}/eps{dp_epsilon}",
                        exist_ok=True,
                    )
                    df.to_csv(
                        f"TransformerBN/eval/fake_datasets/{dataset}/eps{dp_epsilon}/sampled_{dataset}_bs{batch_size}_sigma{dp_sigma}.csv",
                        index=False,
                    )
                    break

            if (epoch + 1) % 10 == 0:
                total_loss /= len(dataloader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f} | ")

                # if (epoch + 1) % 5 == 0:
                # --- Periodic Evaluation on Test Set ---
                # test_nll, test_loss, test_comp, test_acyclic = self.evaluate(
                #     test_data, batch_size, lambda1, lambda2, mask_prob, device=device
                # )
                # print(
                #     f"Test for {epoch+1}, Loss: {test_loss:.4f} | "
                #     f"NLL: {test_nll:.4f} | Comp: {test_comp:.4f} | Acyclic: {test_acyclic:.4f}"
                # )

            if sample and (epoch + 1) % sample == 0:
                df = self.sample(
                    self.data.shape[0], device=device
                )  # Test sampling during training
                os.makedirs(
                    f"TransformerBN/eval/fake_datasets/{dataset}_tf", exist_ok=True
                )
                df.to_csv(
                    f"TransformerBN/eval/fake_datasets/{dataset}_tf/sampled_{dataset}_{epoch+1}.csv",
                    index=False,
                )
                self.save(f"TransformerBN/model/ckpt/{dataset}/model_{epoch+1}.pth")

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


def hyperparameter_tuning():
    bs = [64, 128, 256, 512, 1024]
    sig = [1.5, 2.0, 3.0, 5.0]

    for b in bs:
        for s in sig:
            print(f"Batch Size: {b}, Sigma: {s}")

            # 1. Load data to determine properties
            data_path = f"datasets/{dataset}.csv"

            # 2. Model parameters
            embedding_dimension = 64
            attention_heads = 8
            transformer_layers = 2
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            # 3. Create the model by providing the data path
            # The model will now automatically infer num_vars and var_types
            model = TransformerBN(
                file_path=data_path,
                embed_dim=embedding_dimension,
                num_heads=attention_heads,
                num_layers=transformer_layers,
                cat_cols=cat_cols.get(dataset, None),
                log_cols=log_cols.get(dataset, None),
                dropout=0.1,
                cont_scaler="standard",
                device=device,
            )

            print(f"Number of variables (inferred): {model.num_vars}")
            print(f"Variable types (inferred): {model.var_types}")
            print("\nTrainable Adjacency Matrix W (initial):\n", model.W.data)

            # 4. Training
            model.fit(
                epochs=10000,
                batch_size=b,
                lr=1e-3,
                lambda1=0.0,
                lambda2=0.0,
                mask_prob=0.75,
                test_split_ratio=0.0,
                # sample=1000,
                lambda2_max=1e4,
                lambda2_update_freq=100,
                lambda2_growth_rate=2,
                lr_decay_gamma=1,
                tau_start=0.1,
                tau_end=0.1,
                dp=True,
                dp_epsilon=1.0,
                dp_sigma=s,
                dp_clip=1.0,
                dp_micro_batch_size=b,
            )

            print("\nTrainable Adjacency Matrix W (after training):\n", model.W.data)

            model.save(f"TransformerBN/model/ckpt/{dataset}/model.pth")


# Example Usage:
if __name__ == "__main__":

    cat_cols = {
        "adult": [1, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14],  # 将两个captain视为分类变量
        "ecoli70": [],  # 全部为连续型变量
        "barley": None,
        "credit": [30],  # Class amount是长尾分布
        "loan": [3, 6, 10, 11, 12, 7, 8, 9, 4],  # 其中mortagage是混合变量 视为分类变量
        "insurance": [1, 3, 4, 5],  # children在ctabgan+中被视为分类变量
        "king": [6, 13, 14, 1, 2, 5, 7, 8, 9],  # renovated, waterfront, zipcode
        "covertype": [
            i for i in range(10, 55)
        ],  # 44个binary variables 以及1个目标covertype
        "pm25": [0, 1, 2, 3, 8, 10, 11],
    }
    log_cols = {
        "credit": [29],  # amount是长尾分布 取log
    }

    # hyperparameter_tuning()

    for dataset in [
        # "adult",
        # "ecoli70",  # 256
        # "barley",  #
        # "loan",  # 50
        # "credit",  # 1024
        # "insurance",
        # "king",
        "covertype",  # 32 1.1
        # "pm25",  # 32 1.1
    ]:

        # 1. Load data to determine properties
        data_path = f"datasets/{dataset}.csv"

        # 2. Model parameters
        embedding_dimension = 64
        attention_heads = 8
        transformer_layers = 2
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # 3. Create the model by providing the data path
        # The model will now automatically infer num_vars and var_types
        model = TransformerBN(
            file_path=data_path,
            embed_dim=embedding_dimension,
            num_heads=attention_heads,
            num_layers=transformer_layers,
            cat_cols=cat_cols.get(dataset, None),
            log_cols=log_cols.get(dataset, None),
            dropout=0.1,
            cont_scaler="standard",
            device=device,
        )

        # model = TransformerBN.load(
        #     f"TransformerBN/model/ckpt/ablation/0.6/{dataset}/model.pth", device=device
        # )

        print(f"Number of variables (inferred): {model.num_vars}")
        print(f"Variable types (inferred): {model.var_types}")

        # df = pd.read_csv(data_path)
        # for i in range(5):
        #     df = model.sample(
        #         df.shape[0], device=device
        #     )  # Test sampling during training
        #     os.makedirs(f"TransformerBN/eval/fake_datasets/{dataset}/ablation/0.6", exist_ok=True)
        #     df.to_csv(
        #         f"TransformerBN/eval/fake_datasets/{dataset}/ablation/0.6/sampled_{dataset}_mine_{i+1}.csv",
        #         index=False,
        #     )

        # 4. Training
        model.fit(
            epochs=10000,
            batch_size=1024,
            lr=1e-3,
            mask_prob=0.75,
            test_split_ratio=0.0,
            # sample=1000,
            lr_decay_gamma=1,
            dp=True,
            dp_epsilon=100.0,
            dp_sigma=0.8,
            dp_clip=1.0,
            dp_micro_batch_size=4,
        )

        model.save(f"TransformerBN/model/ckpt/eps100.0/{dataset}/model.pth")
