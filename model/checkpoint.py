"""TabDAT methods extracted without changing their implementation."""

import os
import tempfile
import torch


class CheckpointMixin:
    def save(self, path, *, selection=None, atomic=False):
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
                "continuous_head": self.continuous_head,
                "gmm_components": self.gmm_components,
                "diffusion_steps": self.diffusion_steps,
                "diffusion_hidden_dim": self.diffusion_hidden_dim,
                "col_names": self.col_names,  # Save column names
                "training_config": getattr(self, "training_config", None),
                "last_sampling_order": getattr(self, "last_sampling_order", None),
            },
            "scalers": self.scalers,  # Save sklearn scalers (pickle)
        }
        if selection is not None:
            checkpoint["selection"] = selection
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        if atomic:
            descriptor, temporary = tempfile.mkstemp(prefix="checkpoint-", suffix=".pth", dir=directory)
            os.close(descriptor)
            try:
                torch.save(checkpoint, temporary)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        else:
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
            continuous_head=config.get("continuous_head", "gaussian"),
            gmm_components=config.get("gmm_components", 5),
            diffusion_steps=config.get("diffusion_steps", 100),
            diffusion_hidden_dim=config.get("diffusion_hidden_dim"),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load scalers and column names
        model.scalers = checkpoint["scalers"]
        model.col_names = config["col_names"]
        model.training_config = config.get("training_config")
        model.last_sampling_order = config.get("last_sampling_order")
        model.checkpoint_selection = checkpoint.get("selection")

        model.to(device)
        model.eval()
        print(f"Model loaded from {path}")
        return model
