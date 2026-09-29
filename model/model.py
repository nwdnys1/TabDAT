import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .preprocessing import PreprocessingMixin
    from .losses import LossMixin
    from .sampling import SamplingMixin
    from .training import TrainingMixin
    from .checkpoint import CheckpointMixin
    from .output_heads import ConditionalDDPMHead
else:
    from preprocessing import PreprocessingMixin
    from losses import LossMixin
    from sampling import SamplingMixin
    from training import TrainingMixin
    from checkpoint import CheckpointMixin
    from output_heads import ConditionalDDPMHead

torch.backends.mha.set_fastpath_enabled(False)


class TabDAT(
    PreprocessingMixin, LossMixin, SamplingMixin, TrainingMixin, CheckpointMixin, nn.Module
):
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
        continuous_head="gaussian",
        gmm_components=5,
        diffusion_steps=100,
        diffusion_hidden_dim=None,
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
            continuous_head (str): One of 'gaussian', 'gmm', or 'ddpm'.
            gmm_components (int): Number of Gaussian components for the GMM head.
            diffusion_steps (int): Number of DDPM noise and sampling steps.
            diffusion_hidden_dim (int, optional): Width of the scalar denoiser.
        """
        super(TabDAT, self).__init__()

        if continuous_head not in {"gaussian", "gmm", "ddpm"}:
            raise ValueError("continuous_head must be 'gaussian', 'gmm', or 'ddpm'")
        if gmm_components < 1:
            raise ValueError("gmm_components must be positive")
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if diffusion_hidden_dim is not None and diffusion_hidden_dim < 1:
            raise ValueError("diffusion_hidden_dim must be positive")

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
        self.continuous_head = continuous_head
        self.gmm_components = gmm_components
        self.diffusion_steps = diffusion_steps
        self.diffusion_hidden_dim = (
            diffusion_hidden_dim if diffusion_hidden_dim is not None else max(32, embed_dim)
        )

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
            # Preserve the exact legacy Gaussian module structure and keys.
            elif continuous_head == "gaussian":
                self.prediction_heads[str(i)] = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 2, 2),  # mu and log_sigma
                )
            elif continuous_head == "gmm":
                self.prediction_heads[str(i)] = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 2, 3 * gmm_components),
                )
            else:
                self.prediction_heads[str(i)] = ConditionalDDPMHead(
                    context_dim=embed_dim,
                    hidden_dim=self.diffusion_hidden_dim,
                    steps=diffusion_steps,
                )

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
