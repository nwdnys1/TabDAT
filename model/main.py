import random

import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path

if __package__:
    from .model import TabDAT
    from .data_splits import _file_sha256, prepare_data_splits
else:
    from model import TabDAT
    from data_splits import _file_sha256, prepare_data_splits


def _run_name(run_id):
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one non-empty path component")
    return run_id


def train(dataset, *, seed=42, run_id=None, eval_every=50):
    cat_cols = {
        "adult": [1, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14],
        "ecoli70": [],
        "barley": None,
        "credit": [30],
        "loan": [3, 6, 10, 11, 12, 7, 8, 9, 4],
        "insurance": [1, 3, 4, 5],
        "king": [6, 13, 14, 1, 2, 5, 7, 8, 9],
        "covertype": [i for i in range(10, 55)],
        "pm25": [0, 1, 2, 3, 8, 10, 11],
    }
    if dataset not in cat_cols:
        raise ValueError(
            "This example supports only the original benchmarks; datasets with "
            "official, temporal, or grouped splits need a separate protocol"
        )

    # Split raw rows before fitting encoders, scalers, or the generator.
    target_cols = {
        "adult": "class",
        "barley": "rokap",
        "credit": "Class",
        "loan": "Personal Loan",
        "covertype": "Covertype",
    }
    split_paths = prepare_data_splits(
        f"datasets/{dataset}.csv",
        f"TabDAT/eval/real_datasets/{dataset}",
        stratify_col=target_cols.get(dataset),
    )
    data_path = str(split_paths["train"])
    run_id = _run_name(run_id or f"ckpt_v1_seed{seed}")
    checkpoint_dir = Path(f"TabDAT/model/ckpt/holdout_v1/{dataset}") / run_id
    if checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint run already exists: {checkpoint_dir}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Model parameters
    embedding_dimension = 64
    attention_heads = 8
    transformer_layers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Create the model by providing the data path
    # The model will now automatically infer num_vars and var_types
    model = TabDAT(
        file_path=data_path,
        embed_dim=embedding_dimension,
        num_heads=attention_heads,
        num_layers=transformer_layers,
        cat_cols=cat_cols.get(dataset, None),
        log_cols=None,
        dropout=0.1,
        cont_scaler="standard",
        device=device,
    )

    print(f"Number of variables (inferred): {model.num_vars}")
    print(f"Variable types (inferred): {model.var_types}")
    validation_data = model.transform_holdout(pd.read_csv(split_paths["validation"]))
    run_info = {
        "dataset": dataset,
        "seed": seed,
        "train_path": str(Path(data_path).resolve()),
        "train_sha256": _file_sha256(Path(data_path)),
        "validation_path": str(Path(split_paths["validation"]).resolve()),
        "validation_sha256": _file_sha256(Path(split_paths["validation"])),
        "split_manifest": str((Path(data_path).parent / "split_manifest.json").resolve()),
    }
    # 4. Training
    model.fit(
        epochs=10000,
        batch_size=4096,
        lr=1e-3,
        mask_prob=0.75,
        test_split_ratio=0.0,
        lr_decay_gamma=1,
        dp=False,
        dp_epsilon=1.0,
        dp_sigma=1.0,
        dp_clip=1.0,
        dp_micro_batch_size=1,
        validation_data=validation_data,
        checkpoint_dir=str(checkpoint_dir),
        eval_every=eval_every,
        monitor_seed=seed,
        run_info=run_info,
    )

    # This run contains final, best_train and best_val; legacy model.pth is untouched.
    return model.checkpoint_selection_summary


def sample(dataset, n_samples=None, *, run_id=None, checkpoint_kind=None, sample_seed=5):
    if (run_id is None) != (checkpoint_kind is None):
        raise ValueError("run_id and checkpoint_kind must be supplied together")
    if checkpoint_kind is not None and checkpoint_kind not in {
        "final", "best_train", "best_val"
    }:
        raise ValueError("checkpoint_kind must be final, best_train or best_val")
    if run_id is not None:
        run_id = _run_name(run_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_path = Path(f"TabDAT/model/ckpt/holdout_v1/{dataset}")
    if run_id is None:
        checkpoint_path = checkpoint_path / "model.pth"
    else:
        checkpoint_path = checkpoint_path / run_id / f"{checkpoint_kind}.pth"
    model = TabDAT.load(str(checkpoint_path), device=device)
    expected_train = Path(f"TabDAT/eval/real_datasets/{dataset}/train.csv").resolve()
    if Path(model.file_path).resolve() != expected_train:
        raise ValueError("Checkpoint was not trained on the recorded training split")
    provenance = (model.checkpoint_selection or {}).get("run_info", {})
    if provenance:
        if provenance.get("train_sha256") != _file_sha256(expected_train):
            raise ValueError("Training split content differs from checkpoint provenance")
        validation_path = expected_train.parent / "validation.csv"
        if provenance.get("validation_sha256") != _file_sha256(validation_path):
            raise ValueError("Validation split content differs from checkpoint provenance")

    print(f"Number of variables (inferred): {model.num_vars}")
    print(f"Variable types (inferred): {model.var_types}")

    name = (
        f"sampled_{dataset}_{sample_seed}.csv" if run_id is None else
        f"sampled_{dataset}_{run_id}_{checkpoint_kind}_{sample_seed}.csv"
    )
    output_path = Path(f"TabDAT/eval/fake_datasets/{dataset}/holdout_v1") / name
    if output_path.exists():
        raise FileExistsError(f"Generated CSV already exists: {output_path}")
    # Use a synthetic training set of the same size as the real training split.
    if run_id is not None:
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
    df = model.sample(len(model.data) if n_samples is None else n_samples, device=device)
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    for dataset in [
        "adult",
        "ecoli70",
        "barley",
        "loan",
        "credit",
        "insurance",
        "king",
        "covertype",
        "pm25",
    ]:
        # train(dataset)
        sample(dataset)
