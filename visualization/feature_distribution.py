import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'AR PL UMing CN', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_feature_distribution(
    dataset, column, cat_cols, real_path, fake_dir, save_path, models_to_plot=None
):
    if not os.path.exists(real_path):
        print(f"Real data path not found: {real_path}")
        return
    real_df = pd.read_csv(real_path)

    all_fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.csv")))

    fake_files = []
    model_info = []

    for fpath in all_fake_files:
        fname = os.path.basename(fpath)
        parts = fname.replace(".csv", "").split("_")
        if len(parts) >= 4:
            model_name = parts[2]
            index = parts[3]

            if models_to_plot is not None and model_name not in models_to_plot:
                continue

            if index == "1":
                fake_files.append(fpath)
                model_info.append((fpath, model_name))

    for i, (fpath, model_name) in enumerate(model_info):
        if model_name.lower() == "mine":
            item = model_info.pop(i)
            model_info.insert(0, item)
            break

    if not fake_files:
        print(
            f"No valid fake data files (index 1) found in: {fake_dir} for models {models_to_plot}"
        )
        return

    is_categorical = (
        column in cat_cols.get(dataset, []) or cat_cols.get(dataset) == "all"
    )

    plt.figure(figsize=(10, 6))

    if not is_categorical:
        x_min, x_max = real_df[column].min(), real_df[column].max()

        sns.kdeplot(
            data=real_df[column],
            label="Real",
            fill=True,
            color="blue",
            alpha=0.1,
            linewidth=2.5,
            bw_adjust=1.5,
            gridsize=500,
        )

        colors = ["red", "orange", "green", "purple", "brown", "pink"]
        i = 0
        for fpath, model_name in model_info:
            color = colors[i]
            f_df = pd.read_csv(fpath)
            if column in f_df.columns:
                sns.kdeplot(
                    data=f_df[column],
                    label=model_name,
                    fill=True,
                    color=color,
                    alpha=0.1,
                    linewidth=2.5,
                    bw_adjust=1.5,
                    gridsize=500,
                )
            i += 1

        plt.xlim(x_min, x_max)
        plt.ylabel("概率密度", fontsize=30)
    else:

        plot_data_list = []

        temp_real = real_df[[column]].copy()
        temp_real["Source"] = "Real"
        plot_data_list.append(temp_real)

        for fpath, model_name in model_info:
            f_df = pd.read_csv(fpath)
            if column in f_df.columns:
                temp_fake = f_df[[column]].copy()
                temp_fake["Source"] = model_name
                plot_data_list.append(temp_fake)

        if plot_data_list:
            all_data = pd.concat(plot_data_list, axis=0)
            sns.countplot(
                data=all_data,
                x=column,
                hue="Source",
                palette="muted",
                edgecolor="black",
                linewidth=1,
            )
            plt.ylabel("频数", fontsize=30)

    plt.title(f"{dataset}", fontsize=30)
    plt.xlabel(column, fontsize=30)
    plt.legend(fontsize=15, shadow=False)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    full_save_path = os.path.join(save_path, f"{dataset}/{column}_dist.png")
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    plt.savefig(full_save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved feature distribution plot to {full_save_path}")


def plot_multiple_feature_distributions(
    feature_list,
    cat_cols,
    real_datasets_dir,
    fake_datasets_base_dir,
    save_path,
    models_to_plot=None,
    n_cols=2,
):
    import math

    n_features = len(feature_list)
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 8))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (dataset, column) in enumerate(feature_list):
        ax = axes[idx]
        real_path = os.path.join(real_datasets_dir, f"{dataset}.csv")
        fake_dir = os.path.join(fake_datasets_base_dir, f"{dataset}/baselines")

        if not os.path.exists(real_path):
            print(f"Real data path not found: {real_path}")
            ax.axis("off")
            continue

        real_df = pd.read_csv(real_path)
        all_fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.csv")))
        fake_files = []
        model_info = []

        for fpath in all_fake_files:
            fname = os.path.basename(fpath)
            parts = fname.replace(".csv", "").split("_")
            if len(parts) >= 4:
                model_name = parts[2]
                index = parts[3]
                if models_to_plot is not None and model_name not in models_to_plot:
                    continue
                if index == "1":
                    fake_files.append(fpath)
                    model_info.append((fpath, model_name))

        for i, (fpath, model_name) in enumerate(model_info):
            if model_name.lower() == "mine":
                item = model_info.pop(i)
                model_info.insert(0, item)
                break

        for i in range(len(model_info)):
            fpath, model_name = model_info[i]
            model_name = model_name.lower()
            if model_name == "mine":
                model_name = "TabDAT"
            elif model_name == "ctabganplus":
                model_name = "CTAB-GAN+"
            elif model_name == "ttvae":
                model_name = "TTVAE"
            model_info[i] = (fpath, model_name)

        is_categorical = (
            column in cat_cols.get(dataset, []) or cat_cols.get(dataset) == "all"
        )

        if not is_categorical:
            x_min, x_max = real_df[column].min(), real_df[column].max()
            sns.kdeplot(
                data=real_df[column],
                label="Real",
                fill=True,
                color="blue",
                alpha=0.1,
                linewidth=2.5,
                bw_adjust=1.5,
                gridsize=500,
                ax=ax,
            )

            colors = ["red", "orange", "green", "purple", "brown", "pink"]
            for i, (fpath, model_name) in enumerate(model_info):
                color = colors[i % len(colors)]
                f_df = pd.read_csv(fpath)
                if column in f_df.columns:
                    sns.kdeplot(
                        data=f_df[column],
                        label=model_name,
                        fill=True,
                        color=color,
                        alpha=0.1,
                        linewidth=2.5,
                        bw_adjust=1.5,
                        gridsize=500,
                        ax=ax,
                    )
            ax.set_xlim(x_min, x_max)
            ax.set_ylabel("概率密度", fontsize=40)
        else:
            plot_data_list = []
            temp_real = real_df[[column]].copy()
            temp_real["Source"] = "Real"
            plot_data_list.append(temp_real)
            for fpath, model_name in model_info:
                f_df = pd.read_csv(fpath)
                if column in f_df.columns:
                    temp_fake = f_df[[column]].copy()
                    temp_fake["Source"] = model_name
                    plot_data_list.append(temp_fake)
            if plot_data_list:
                all_data = pd.concat(plot_data_list, axis=0)
                sns.countplot(
                    data=all_data,
                    x=column,
                    hue="Source",
                    palette="muted",
                    edgecolor="black",
                    linewidth=1,
                    ax=ax,
                )
                ax.set_xticklabels(
                    ax.get_xticklabels(),
                    rotation=20,
                    ha="center",
                )
                ax.set_ylabel("频数", fontsize=40)

        ax.set_title(f"{dataset.capitalize()}", fontsize=40)
        ax.set_xlabel(column, fontsize=40)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.legend(fontsize=23, shadow=False)

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    plt.savefig(save_path, bbox_inches="tight", dpi=300, format="svg")
    plt.close()
    print(f"Saved multiple feature distributions plot to {save_path}")


def plot_feature_distribution_ablation(
    dataset, column, cat_cols, real_path, fake_dir, save_path
):
    """
    可视化消融实验中不同配置下 mine 模型的分布。
    涉及 5 个路径：ablation/0.3, ablation/0.45, ablation/0.6, ablation/0.9, baselines/
    """
    if not os.path.exists(real_path):
        print(f"Real data path not found: {real_path}")
        return
    real_df = pd.read_csv(real_path)

    subdirs = [
        "ablation/0.9",
        "baselines",
        "ablation/0.6",
        "ablation/0.45",
        "ablation/0.3",
    ]
    labels = ["p=0.9", "p=0.75", "p=0.6", "p=0.45", "p=0.3"]

    model_info = []
    for subdir, label in zip(subdirs, labels):
        pattern = os.path.join(fake_dir, subdir, f"sampled_{dataset}_mine_1.csv")
        files = glob.glob(pattern)
        if files:
            model_info.append((files[0], label))
        else:
            print(f"Warning: No mine model found in {os.path.join(fake_dir, subdir)}")

    if not model_info:
        print(f"No valid mine data files found in any of the specified subdirectories.")
        return

    is_categorical = (
        column in cat_cols.get(dataset, []) or cat_cols.get(dataset) == "all"
    )

    plt.figure(figsize=(10, 6))

    if not is_categorical:
        x_min, x_max = real_df[column].min(), real_df[column].max()
        sns.kdeplot(
            data=real_df[column],
            label="Real",
            fill=True,
            color="blue",
            alpha=0.1,
            linewidth=2.5,
            bw_adjust=1.5,
            gridsize=500,
        )

        colors = ["red", "orange", "green", "purple", "brown"]
        for i, (fpath, label) in enumerate(model_info):
            color = colors[i % len(colors)]
            f_df = pd.read_csv(fpath)
            if column in f_df.columns:
                sns.kdeplot(
                    data=f_df[column],
                    label=label,
                    fill=True,
                    color=color,
                    alpha=0.1,
                    linewidth=2.5,
                    bw_adjust=1.5,
                    gridsize=500,
                )

        plt.xlim(x_min, x_max)
        plt.ylabel("概率密度", fontsize=30)
    else:
        plot_data_list = []
        temp_real = real_df[[column]].copy()
        temp_real["Source"] = "Real"
        plot_data_list.append(temp_real)

        for fpath, label in model_info:
            f_df = pd.read_csv(fpath)
            if column in f_df.columns:
                temp_fake = f_df[[column]].copy()
                temp_fake["Source"] = label
                plot_data_list.append(temp_fake)

        if plot_data_list:
            all_data = pd.concat(plot_data_list, axis=0)
            sns.countplot(
                data=all_data,
                x=column,
                hue="Source",
                palette="muted",
                edgecolor="black",
                linewidth=1,
            )
            ax = plt.gca()
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=10,
                ha="center",
            )
            plt.ylabel("频数", fontsize=30)

    plt.title(f"{dataset.capitalize()}", fontsize=30)
    plt.xlabel(column, fontsize=30)
    plt.legend(fontsize=15, shadow=False)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    full_save_path = os.path.join(save_path, f"{dataset}_ablation/{column}_dist.svg")
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    plt.savefig(full_save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved ablation distribution plot to {full_save_path}")
