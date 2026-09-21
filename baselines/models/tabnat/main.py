import time
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils_train import preprocess, TabularDataset
from models.tabnat.model import TabNAT

warnings.filterwarnings("ignore")


def main(args):

    dataname = args.dataname
    data_dir = f"datasets/{dataname}.csv"
    device = "cuda"
    info_path = f"data_profile/{dataname}.json"

    with open(info_path, "r") as f:
        info = json.load(f)

    task_type = info["task_type"]
    num_col_idx = info["num_col_idx"]

    X_num, X_cat, categories, n_num, num_inverse, cat_inverse = preprocess(
        raw_config=info,
        dataset_path=data_dir,
        task_type=task_type,
        inverse=True,
        concat=True,
    )
    n_cat = len(categories)
    len_ori_cat = len(categories)

    has_num = n_num > 0
    has_cat = n_cat > 0
    print(f"has_num: {has_num}, has_cat: {has_cat}")

    if has_num:
        X_train_num, X_test_num = X_num
    else:
        X_train_num, X_test_num = None, None
    if has_cat:
        X_train_cat, X_test_cat = X_cat
    else:
        X_train_cat, X_test_cat = None, None

    if has_num:
        X_train_num = X_train_num[:, :n_num]
        X_test_num = X_test_num[:, :n_num]
        X_train_num = torch.tensor(X_train_num).float()
        X_test_num = torch.tensor(X_test_num).float()

        mean, std = X_train_num.mean(0), X_train_num.std(0)
        X_train_num = (X_train_num - mean) / std / 2
        X_test_num = (X_test_num - mean) / std / 2

        X_train_num = X_train_num.float()

    if has_cat:
        X_train_cat = X_train_cat[:, :n_cat]
        X_test_cat = X_test_cat[:, :n_cat]
        X_train_cat = torch.tensor(X_train_cat)
        X_test_cat = torch.tensor(X_test_cat)
        categories = categories[:n_cat]

    train_data = TabularDataset(X_train_num, X_train_cat)

    n_cat_model = n_cat

    model = TabNAT(
        n_num=n_num,
        n_cat=n_cat_model,
        categories=categories,
        norm_layer=nn.LayerNorm,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=50)

    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)

    best_loss = np.inf
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0
        epoch_loss_num = 0
        epoch_loss_cat = 0
        epoch_num = 0

        for batch in train_loader:
            if not has_cat:
                x_num = batch
                x_cat = None
            elif not has_num:
                x_cat = batch
                x_num = None
            else:
                x_num, x_cat = batch

            # =================================
            if n_num >= 1:
                x_num = x_num.to(device)
            else:
                x_num = None
            if n_cat >= 1:
                x_cat = x_cat.to(device)
            else:
                x_cat = None

            repeat = 1

            optimizer.zero_grad()

            if n_num >= 1:
                x_num = x_num.repeat(repeat, 1)
            if n_cat >= 1:
                x_cat = x_cat.repeat(repeat, 1)

            loss, loss_num, loss_cat = model(x_num, x_cat)

            loss.backward()
            optimizer.step()

            batch_len = x_cat.shape[0] if has_cat else x_num.shape[0]

            epoch_loss += loss.item() * batch_len
            if has_cat:
                epoch_loss_cat += loss_cat.item() * batch_len
            if has_num:
                epoch_loss_num += loss_num.item() * batch_len

            epoch_num += batch_len
            # =================================

        scheduler.step(epoch_loss)
        epoch_loss /= epoch_num
        epoch_loss_cat /= epoch_num
        epoch_loss_num /= epoch_num

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1

        if epoch % 1 == 0:
            print(
                f"Epoch [{epoch}/{args.epochs}], Epoch Loss: {epoch_loss:.4f}, Epoch Loss Cat: {epoch_loss_cat:.4f}, Epoch Loss Num: {epoch_loss_num:.4f}, Best Loss: {best_loss:.4f}, patience: {patience}"
            )

    end_time = time.time()

    print(f"Training time: {end_time - start_time:.2f} seconds")

    print(f"Training finished. Best Loss: {best_loss}")

    model.eval()

    # Sampling
    start_time = time.time()
    with torch.no_grad():

        B = X_train_cat.shape[0] if has_cat else X_train_num.shape[0]

        syn_num, syn_cat = model.sample(B, device=device)

        if syn_num is not None:
            syn_num = (syn_num.cpu() * 2 * std) + mean
            syn_num = syn_num.numpy()

            syn_num = num_inverse(syn_num)

        if syn_cat is not None:
            syn_cat = syn_cat.cpu().numpy()
            syn_cat = cat_inverse(syn_cat[:, -len_ori_cat:])

    if task_type == "regression":
        syn_target = syn_num[:, 0:1]
        syn_num = syn_num[:, 1:]
    else:
        syn_target = syn_cat[:, 0:1]
        syn_cat = syn_cat[:, 1:]

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    # 初始化合并数组
    syn_merged = np.empty(
        (B, max(num_col_idx + cat_col_idx + target_col_idx) + 1), dtype=object
    )

    # 按原始列索引填充
    syn_merged[:, num_col_idx] = syn_num  # 填充数值列
    syn_merged[:, cat_col_idx] = syn_cat  # 填充分类列
    syn_merged[:, target_col_idx] = syn_target  # 填充目标列

    syn_df = pd.DataFrame(syn_merged, columns=info["column_names"])

    # 若分类列是数值，则转换为整数类型
    for col in cat_col_idx:
        cat_col = info["column_names"][col]
        try:
            syn_df[cat_col] = syn_df[cat_col].astype(int)
        except:
            pass
    if task_type == "binclass" or task_type == "multiclass":
        target_col = info["column_names"][target_col_idx[0]]
        try:
            syn_df[target_col] = syn_df[target_col].astype(int)
        except:
            pass

    end_time = time.time()

    print("Sampling time:", end_time - start_time)

    syn_df.to_csv(args.save_path, index=False)

    print("Saving sampled data to {}".format(args.save_path))
