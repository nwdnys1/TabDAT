import torch
import numpy as np
import pandas as pd
import os
import json
import time

from models.tabddpm.models.gaussian_multinomial_distribution import (
    GaussianMultinomialDiffusion,
)
from models.tabddpm.models.modules import MLPDiffusion

import src
from src.utils_train import make_dataset


@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse):
    task_type = info["task_type"]

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == "regression":
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    if info["task_type"] == "regression":
        syn_target = syn_num[:, : len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx) :]

    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, : len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx) :]

    return syn_num, syn_cat, syn_target


def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping = info["idx_mapping"]
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info["task_type"] == "regression":
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[
                    :, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)
                ]

    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[
                    :, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)
                ]

    return syn_df


def get_model(model_name, model_params, n_num_features, category_sizes):
    print(model_name)
    if model_name == "mlp":
        model = MLPDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model


def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1] : indices[i]], axis=1)
        t = X[:, indices[i - 1] : indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


def sample(
    raw_config,
    model_save_path,
    sample_save_path,
    real_data_path,
    batch_size=2000,
    num_samples=0,
    task_type="binclass",
    model_type="mlp",
    model_params=None,
    num_timesteps=1000,
    gaussian_loss_type="mse",
    scheduler="cosine",
    T_dict=None,
    # num_numerical_features = 0,
    disbalance=None,
    device=torch.device("cuda:0"),
    change_val=False,
    ddim=False,
    steps=1000,
    save_path=None,
):

    T = src.Transformations(**T_dict)

    D = make_dataset(
        raw_config,
        real_data_path,
        T,
        task_type=task_type,
        n_classes=model_params["num_classes"],
        change_val=False,
    )

    K = np.array(D.get_category_sizes("train"))
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])

    num_numerical_features_ = D.X_num["train"].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    print(K, num_numerical_features_)
    model_params["d_in"] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes("train"),
    )

    model_path = f"{model_save_path}/model.pt"

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        device=device,
    )

    diffusion.to(device)
    diffusion.eval()

    start_time = time.time()
    if not ddim:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False)
    else:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=True, steps=steps)

    syn_data = x_gen
    num_inverse = D.num_transform.inverse_transform if D.num_transform else lambda x: x
    cat_inverse = D.cat_transform.inverse_transform if D.cat_transform else lambda x: x

    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, raw_config, num_inverse, cat_inverse
    )
    num_col_idx = raw_config["num_col_idx"]
    cat_col_idx = raw_config["cat_col_idx"]
    target_col_idx = raw_config["target_col_idx"]
    # 初始化合并数组
    syn_merged = np.empty(
        (num_samples, max(num_col_idx + cat_col_idx + target_col_idx) + 1), dtype=object
    )

    # 按原始列索引填充
    syn_merged[:, num_col_idx] = syn_num  # 填充数值列
    syn_merged[:, cat_col_idx] = syn_cat  # 填充分类列
    syn_merged[:, target_col_idx] = syn_target  # 填充目标列

    syn_df = pd.DataFrame(syn_merged, columns=raw_config["column_names"])

    # 若分类列是数值，则转换为整数类型
    for col in cat_col_idx:
        cat_col = raw_config["column_names"][col]
        try:
            syn_df[cat_col] = syn_df[cat_col].astype(int)
        except:
            pass
    if task_type == "binclass" or task_type == "multiclass":
        target_col = raw_config["column_names"][target_col_idx[0]]
        try:
            syn_df[target_col] = syn_df[target_col].astype(int)
        except:
            pass

    end_time = time.time()

    print("Sampling time:", end_time - start_time)

    syn_df.to_csv(save_path, index=False)
