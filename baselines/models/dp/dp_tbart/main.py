import os
import time
import json
import warnings
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import KBinsDiscretizer

from src.utils_train import preprocess, TabularDataset
from models.dp.dp_tbart.models.model import TabMAR
from models.dp.utils.rdp_accountant import compute_rdp, get_privacy_spent

warnings.filterwarnings("ignore")


def main(args):

    dataname = args.dataname
    data_dir = f"datasets/{dataname}.csv"
    device = "cuda"
    info_path = f"data_profile/{dataname}.json"
    batch_size = 1024

    dp = getattr(args, "dp", False)
    dp_epsilon = getattr(args, "eps", 1.0)
    dp_sigma = getattr(args, "sigma", 0.8)
    dp_clip = getattr(args, "clip", 1.0)
    dp_micro_batch_size = getattr(args, "micro_batch_size", 1024)
    dp_steps = 0

    with open(info_path, "r") as f:
        info = json.load(f)

    task_type = info["task_type"]
    X_num, X_cat, categories, n_num, num_inverse, cat_inverse = preprocess(
        raw_config=info,
        dataset_path=data_dir,
        task_type=task_type,
        inverse=True,
        concat=True,
    )
    len_ori_cat = len(categories)
    

    if X_cat is not None:
        X_cat_train = X_cat[0]
    else:
        X_cat_train = np.empty((X_num[0].shape[0], 0))

    if X_num is not None:
        X_num = X_num[0]
        # discretize numerical features
        enc = KBinsDiscretizer(
            n_bins=[100 for _ in range(n_num)],
            encode="ordinal",
            strategy="kmeans",
            subsample=None,
        )
        X_num_disc = enc.fit_transform(X_num).astype(np.int32)

        X_cat_train = np.concatenate([X_num_disc, X_cat_train], axis=1)
        categories = [100 for _ in range(n_num)] + categories

    n_cat = len(categories)
    n_num = 0
    
    X_train_num = None
    X_train_cat = torch.tensor(X_cat_train)

    train_data = TabularDataset(X_train_num, X_train_cat)

    model = TabMAR(
        n_num,
        n_cat,
        categories,
        norm_layer=nn.LayerNorm,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=200)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)

    best_loss = np.inf
    patience = 1000
    if not os.path.exists(f"TabART/checkpoints/{dataname}"):
        os.makedirs(f"TabART/checkpoints/{dataname}")

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0
        epoch_loss_cat = 0
        epoch_num = 0

        for batch in train_loader:
            x_cat = batch
            x_num = None

            # =================================
            if n_num >= 1:
                x_num = x_num.to(device)
            else:
                x_num = None
            if n_cat >= 1:
                x_cat = x_cat.to(device)
            else:
                x_cat = None

            if dp:
                # ================= DPSGD 逻辑 =================
                # 1. 初始化梯度累加器
                saved_grads = {
                    name: torch.zeros_like(param)
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }

                current_batch_size = x_cat.size(0)
                num_micro_batches = math.ceil(current_batch_size / dp_micro_batch_size)

                for k in range(num_micro_batches):
                    # 2. 获取微批次数据
                    start_idx = k * dp_micro_batch_size
                    end_idx = min((k + 1) * dp_micro_batch_size, current_batch_size)

                    m_x_cat = x_cat[start_idx:end_idx]
                    m_x_num = x_num[start_idx:end_idx] if x_num is not None else None

                    optimizer.zero_grad()

                    # 3. 前向传播 (TabMAR 内部已包含 Masking Augmentation)
                    m_x_cat = m_x_cat.long()
                    loss, loss_num, loss_cat = model(m_x_num, m_x_cat)

                    # 4. 反向传播
                    loss.backward()

                    # 5. 梯度裁剪 (Per-micro-batch Clipping)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), dp_clip)

                    # 6. 累加裁剪后的梯度
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            saved_grads[name] += param.grad

                # 7. 加噪与平均 (Add Noise & Average)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        noise = torch.normal(
                            0, dp_sigma * dp_clip, size=param.shape, device=device
                        )
                        param.grad = (saved_grads[name] + noise) / num_micro_batches

                # 8. 更新参数
                optimizer.step()

                # 更新 Epoch 统计
                epoch_loss += loss.item() * current_batch_size
                epoch_loss_cat += loss_cat.item() * current_batch_size
                epoch_num += current_batch_size
                dp_steps += 1

            repeat = 1

            optimizer.zero_grad()

            if n_num >= 1:
                x_num = x_num.repeat(repeat, 1)
            if n_cat >= 1:
                x_cat = x_cat.repeat(repeat, 1)
                
            x_cat = x_cat.long()
            loss, loss_num, loss_cat = model(x_num, x_cat)

            loss.backward()
            optimizer.step()

            batch_len = x_cat.shape[0]

            epoch_loss += loss.item() * batch_len
            # epoch_loss_num += loss_num.item() * batch_len
            epoch_loss_cat += loss_cat.item() * batch_len

            epoch_num += batch_len

            # =================================

        scheduler.step(epoch_loss)
        # --- Privacy Accounting ---
        if dp:
            lmbds = range(2, 4096)
            rdp = compute_rdp(batch_size / len(train_data), dp_sigma, dp_steps, lmbds)
            epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5)
            print(f"Step {dp_steps}: ε = {epsilon:.4f} for δ = 1e-5")

            if epsilon > dp_epsilon:
                print(
                    f"Reached privacy budget of epsilon = {dp_epsilon}. Stopping training."
                )
                break

        epoch_loss /= epoch_num
        # epoch_loss_num /= epoch_num
        epoch_loss_cat /= epoch_num

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0

        else:
            patience += 1

        if epoch % 1 == 0:
            print(
                f"Epoch {epoch}, Epoch Loss: {epoch_loss:.4f}, Epoch Loss Cat: {epoch_loss_cat:.4f}, Best Loss: {best_loss:.4f}, patience: {patience}"
            )

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Training finished. Best Loss: {best_loss}")

    model.eval()

    # Sampling
    start_time = time.time()
    with torch.no_grad():
        B = X_cat_train.shape[0]
        syn_num, syn_cat = model.sample(B, device=device)

        assert syn_num is None
        syn_cat = syn_cat.cpu().numpy()

        if X_num is not None:
            # reverse KBinsDiscretizer
            if len_ori_cat == 0:
                syn_num_disc = enc.inverse_transform(syn_cat).astype(np.float64)
            else:
                syn_num_disc = enc.inverse_transform(syn_cat[:, :-len_ori_cat]).astype(
                    np.float64
                )
            syn_num = num_inverse(syn_num_disc)
        if X_cat is not None:
            syn_cat = cat_inverse(syn_cat[:, -len_ori_cat:])
        else:
            syn_cat = np.empty((syn_num.shape[0], 0))

    task_type = info["task_type"]
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
