"""Ordinary TabDAT training, with a compatibility entry for legacy DP."""

import json
from pathlib import Path

import torch

if __package__:
    from .checkpoint_selection import build_monitor_bank, evaluate_monitor
    from .mask_strategies import (
        MASK_STRATEGIES,
        ORDERED_STRATEGIES,
        sample_training_masks,
        validate_order,
    )
else:
    from checkpoint_selection import build_monitor_bank, evaluate_monitor
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
        test_split_ratio=0.0,
        sample=None,
        lr_decay_gamma=0.999,
        dp=False,
        dp_epsilon=1.0,
        dp_sigma=1.02,
        dp_clip=1.0,
        dp_micro_batch_size=1,
        mask_strategy="bernoulli_all",
        training_order=None,
        validation_data=None,
        checkpoint_dir=None,
        eval_every=50,
        monitor_max_rows=4096,
        monitor_repeats=4,
        monitor_seed=42,
        monitor_batch_size=None,
        run_info=None,
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
            test_split_ratio (float): Deprecated for ordinary training. Must be 0;
                split raw data before constructing the model so preprocessors see
                training rows only. Legacy DP uses its separate training path.
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
            validation_data (Tensor, optional): Held-out rows already transformed
                with the training-fitted preprocessors.
            checkpoint_dir (str, optional): New run directory; required with
                validation_data. Existing directories are never reused.
            eval_every (int): Fixed epoch interval for checkpoint comparison.
            monitor_max_rows (int): Maximum fixed rows per train/validation bank.
            monitor_repeats (int): Fixed masking tasks per monitored row.
            monitor_seed (int): Seed for monitor row, mask, and DDPM noise banks.
            run_info (dict, optional): Split provenance recorded in checkpoints.
        """
        if dp:
            if validation_data is not None or checkpoint_dir is not None:
                raise NotImplementedError("Checkpoint selection is not enabled for legacy DP")
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

        if test_split_ratio != 0:
            raise ValueError(
                "fit() cannot safely create a holdout after preprocessing. "
                "Split the raw CSV first and construct TabDAT from the training CSV."
            )
        train_dataset = data
        monitor_enabled = validation_data is not None or checkpoint_dir is not None
        if monitor_enabled:
            if validation_data is None or checkpoint_dir is None:
                raise ValueError("validation_data and checkpoint_dir must be supplied together")
            if epochs < 1 or eval_every < 1:
                raise ValueError("epochs and eval_every must be positive")
            if monitor_batch_size is None:
                monitor_batch_size = batch_size
            if monitor_batch_size < 1:
                raise ValueError("monitor_batch_size must be positive")
            if run_info is not None and not isinstance(run_info, dict):
                raise TypeError("run_info must be a dictionary")
            checkpoint_dir = Path(checkpoint_dir)
            if checkpoint_dir.exists():
                raise FileExistsError(f"Checkpoint run already exists: {checkpoint_dir}")
            bank_options = dict(
                num_vars=self.num_vars, strategy=mask_strategy, mask_prob=mask_prob,
                order=training_order, continuous_head=self.continuous_head,
                diffusion_steps=self.diffusion_steps, max_rows=monitor_max_rows,
                repeats=monitor_repeats,
            )
            train_bank = build_monitor_bank(train_dataset, seed=monitor_seed + 1, **bank_options)
            validation_bank = build_monitor_bank(
                validation_data, seed=monitor_seed, **bank_options
            )
            best_scores = {"best_train": float("inf"), "best_val": float("inf")}
            best_epochs = {}
            self.monitor_history = []

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
        if monitor_enabled:
            self.training_config["checkpoint_monitor"] = {
                "eval_every": eval_every,
                "max_rows": monitor_max_rows,
                "repeats": monitor_repeats,
                "seed": monitor_seed,
                "train_bank_sha256": train_bank.digest,
                "validation_bank_sha256": validation_bank.digest,
                "run_info": run_info or {},
            }
            checkpoint_dir.mkdir(parents=True, exist_ok=False)

        print("--- Starting Training ---")
        print(f"Training set size: {len(train_dataset)}")
        print("Validation and test rows must be held out before model construction")

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

            if monitor_enabled and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs):
                train_scores = evaluate_monitor(
                    self, train_bank, batch_size=monitor_batch_size, device=device
                )
                validation_scores = evaluate_monitor(
                    self, validation_bank, batch_size=monitor_batch_size, device=device
                )
                w_order = self._topological_sort(torch.sigmoid(self.W.detach()), verbose=False)
                record = {
                    "epoch": epoch + 1,
                    "train_monitor": train_scores,
                    "validation_monitor": validation_scores,
                    "w_order": w_order,
                }
                self.monitor_history.append(record)
                selection_base = {
                    "epoch": epoch + 1,
                    "train_monitor_loss": train_scores["loss"],
                    "validation_monitor_loss": validation_scores["loss"],
                    "w_order_at_save": w_order,
                    "run_info": run_info or {},
                }
                for kind, score in (
                    ("best_train", train_scores["loss"]),
                    ("best_val", validation_scores["loss"]),
                ):
                    if score < best_scores[kind]:
                        best_scores[kind] = score
                        best_epochs[kind] = epoch + 1
                        self.save(
                            str(checkpoint_dir / f"{kind}.pth"),
                            selection={**selection_base, "rule": kind, "score": score},
                            atomic=True,
                        )
                with (checkpoint_dir / "monitor_history.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                print(
                    f"Epoch [{epoch+1}/{epochs}], monitor train={train_scores['loss']:.4f}, "
                    f"val={validation_scores['loss']:.4f}"
                )

            if sample and (epoch + 1) % sample == 0:
                self.sample(len(train_dataset), device=device)

        if monitor_enabled:
            final_record = self.monitor_history[-1]
            self.save(
                str(checkpoint_dir / "final.pth"),
                selection={
                    "rule": "final",
                    "epoch": epochs,
                    "score": final_record["validation_monitor"]["loss"],
                    "train_monitor_loss": final_record["train_monitor"]["loss"],
                    "validation_monitor_loss": final_record["validation_monitor"]["loss"],
                    "w_order_at_save": final_record["w_order"],
                    "run_info": run_info or {},
                },
                atomic=True,
            )
            self.checkpoint_selection_summary = {
                "final_epoch": epochs,
                "best_epochs": best_epochs,
                "best_scores": best_scores,
                "train_bank_sha256": train_bank.digest,
                "validation_bank_sha256": validation_bank.digest,
            }
            with (checkpoint_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(self.checkpoint_selection_summary, handle, indent=2)

        print("--- Training Finished ---")

    def fit_dp(self, **kwargs):
        """Opt in to the isolated legacy DP procedure."""
        if __package__:
            from .dp_training import fit_dp_legacy
        else:
            from dp_training import fit_dp_legacy
        return fit_dp_legacy(self, **kwargs)
