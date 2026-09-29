import torch
import os
from pathlib import Path

if __package__:
    from .model import TabDAT
    from .data_splits import prepare_data_splits
else:
    from model import TabDAT
    from data_splits import prepare_data_splits


def train(dataset):
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
    checkpoint_path = Path(f"TabDAT/model/ckpt/holdout_v1/{dataset}/model.pth")
    if checkpoint_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")

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
    )

    # Keep historical full-data checkpoints untouched.
    model.save(str(checkpoint_path))


def sample(dataset, n_samples=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = TabDAT.load(f"TabDAT/model/ckpt/holdout_v1/{dataset}/model.pth", device=device)
    expected_train = Path(f"TabDAT/eval/real_datasets/{dataset}/train.csv").resolve()
    if Path(model.file_path).resolve() != expected_train:
        raise ValueError("Checkpoint was not trained on the recorded training split")

    print(f"Number of variables (inferred): {model.num_vars}")
    print(f"Variable types (inferred): {model.var_types}")

    output_path = Path(
        f"TabDAT/eval/fake_datasets/{dataset}/holdout_v1/sampled_{dataset}_5.csv"
    )
    if output_path.exists():
        raise FileExistsError(f"Generated CSV already exists: {output_path}")
    # Use a synthetic training set of the same size as the real training split.
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
