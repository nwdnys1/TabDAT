import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
import json

from tqdm import tqdm
from models.tabsyn.model import MLPDiffusion, Model
from models.tabsyn.latent_utils import get_input_train
from src.utils_train import preprocess, TabularDataset
from models.tabsyn.latent_utils import (
    get_input_generate,
    recover_data,
    split_num_cat_target,
)
from models.tabsyn.diffusion_utils import sample

from models.tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model

warnings.filterwarnings("ignore")


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    if X_num is not None:
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
    else:
        mse_loss = torch.tensor(0.0)
    if X_cat is not None:
        ce_loss = 0
        acc = 0
        total_num = 0

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat.argmax(dim=-1)
            acc += (x_hat == X_cat[:, idx]).float().sum()
            total_num += x_hat.shape[0]

        ce_loss /= idx + 1
    else:
        ce_loss = torch.tensor(0.0)
        acc = torch.tensor(0.0)
        total_num = 1  # to avoid division by zero
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def main(args):
    dataname = args.dataname
    data_dir = f"datasets/{dataname}.csv"
    max_beta = 1e-2
    min_beta = 1e-5
    lambd = 0.7

    device = "cuda"

    info_path = f"data_profile/{dataname}.json"

    with open(info_path, "r") as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f"{curr_dir}/vae/ckpt/{dataname}"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f"{ckpt_dir}/model.pt"
    encoder_save_path = f"{ckpt_dir}/encoder.pt"
    decoder_save_path = f"{ckpt_dir}/decoder.pt"

    X_num, X_cat, categories, d_numerical = preprocess(
        raw_config=info, dataset_path=data_dir, task_type=info["task_type"]
    )
    if X_num is not None:
        X_train_num, _ = X_num
    else:
        X_train_num = None
    if X_cat is not None:
        X_train_cat, _ = X_cat
    else:
        X_train_cat = None

    if X_train_num is not None:
        X_train_num = torch.tensor(X_train_num).float()

    if X_train_cat is not None:
        X_train_cat = torch.tensor(X_train_cat)

    train_data = TabularDataset(X_train_num, X_train_cat)

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    LR = 1e-3
    WD = 0
    D_TOKEN = 4
    TOKEN_BIAS = True

    N_HEAD = 1
    FACTOR = 32
    NUM_LAYERS = 2

    model = Model_VAE(
        NUM_LAYERS,
        d_numerical,
        categories,
        D_TOKEN,
        n_head=N_HEAD,
        factor=FACTOR,
        bias=True,
    )
    model = model.to(device)

    pre_encoder = Encoder_model(
        NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR
    ).to(device)
    pre_decoder = Decoder_model(
        NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR
    ).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.95, patience=10)

    num_epochs = 400
    best_train_loss = float("inf")

    current_lr = optimizer.param_groups[0]["lr"]
    patience = 0

    beta = max_beta
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        for batch in pbar:
            if X_train_num is None:
                batch_num, batch_cat = None, batch
            elif X_train_cat is None:
                batch_num, batch_cat = batch, None
            else:
                batch_num, batch_cat = batch

            model.train()
            optimizer.zero_grad()

            if batch_num is not None:
                batch_num = batch_num.to(device)
            if batch_cat is not None:
                batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)

            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(
                batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
            )

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch[0].shape[0] if batch_num is not None else batch.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count

    end_time = time.time()
    print("Training time: {:.4f} mins".format((end_time - start_time) / 60))

    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        if X_train_num is not None:
            X_train_num = X_train_num.to(device)
        if X_train_cat is not None:
            X_train_cat = X_train_cat.to(device)

        print("Successfully load and save the model!")

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        np.save(f"{ckpt_dir}/train_z.npy", train_z)

        print("Successfully save pretrained embeddings in disk!")

    train_z, _, _, ckpt_path, _ = get_input_train(args)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    num_epochs = 500

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=20)

    model.train()

    best_loss = float("inf")
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):

        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)

            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
        else:
            patience += 1
            if patience == 500:
                print("Early stopping")
                break

    end_time = time.time()
    print("Time: ", end_time - start_time)

    """
        Generating samples    
    """
    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)

    start_time = time.time()

    num_samples = train_z.shape[0]
    sample_dim = in_dim

    x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, info, num_inverse, cat_inverse, args.device
    )

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    # 初始化合并数组
    syn_merged = np.empty(
        (num_samples, max(num_col_idx + cat_col_idx + target_col_idx) + 1), dtype=object
    )

    # 按原始列索引填充
    syn_merged[:, num_col_idx] = syn_num  # 填充数值列
    syn_merged[:, cat_col_idx] = syn_cat  # 填充分类列
    syn_merged[:, target_col_idx] = syn_target  # 填充目标列

    syn_df = pd.DataFrame(syn_merged, columns=info["column_names"])

    # 若分类列是数值，则转换为整数类型
    task_type = info["task_type"]
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
