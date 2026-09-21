import torch
import pandas as pd
import os
from model import TabDAT


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

    # 1. Load data to determine properties
    data_path = f"datasets/{dataset}.csv"

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

    model.save(f"TabDAT/model/ckpt/random_order/{dataset}/model.pth")


def sample(dataset):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = TabDAT.load(f"TabDAT/model/ckpt/non-dp/{dataset}/model.pth", device=device)

    print(f"Number of variables (inferred): {model.num_vars}")
    print(f"Variable types (inferred): {model.var_types}")

    df = model.sample(1000, device=device)
    os.makedirs(f"TabDAT/eval/fake_datasets/{dataset}/random_order", exist_ok=True)
    df.to_csv(
        f"TabDAT/eval/fake_datasets/{dataset}/random_order/sampled_{dataset}_5.csv",
        index=False,
    )


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
