import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.gridspec import GridSpec


def plot_joint_distribution(
    dataset, col1, col2, cat_cols, real_path, fake_dir, save_path, models_to_plot=None
):
    if not os.path.exists(real_path):
        print(f"Real data path not found: {real_path}")
        return
    real_df = pd.read_csv(real_path)

    all_fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.csv")))
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
        elif model_name == "tabnat":
            model_name = "TabNAT"
        elif model_name == "tabsyn":
            model_name = "TabSyn"
        elif model_name == "tabddpm":
            model_name = "TabDDPM"
        elif model_name == "ctgan":
            model_name = "CTGAN"
        elif model_name == "tvae":
            model_name = "TVAE"
        elif model_name == "tabmt":
            model_name = "TabMT"

        model_info[i] = (fpath, model_name)

    plot_items = []
    plot_items.append((real_df, "Real"))
    for fpath, m_name in model_info:
        f_df = pd.read_csv(fpath)
        if col1 in f_df.columns and col2 in f_df.columns:
            plot_items.append((f_df, m_name))

    if len(plot_items) <= 1:
        print(f"No valid fake data found for models {models_to_plot}")
        return

    is_col1_cat = col1 in cat_cols.get(dataset, []) or cat_cols.get(dataset) == "all"
    is_col2_cat = col2 in cat_cols.get(dataset, []) or cat_cols.get(dataset) == "all"
    use_kde = not (is_col1_cat or is_col2_cat)

    x_range = (real_df[col1].min(), real_df[col1].max()) if not is_col1_cat else None
    y_range = (real_df[col2].min(), real_df[col2].max()) if not is_col2_cat else None

    num_plots = len(plot_items)
    n_cols = 5
    n_rows = (num_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))

    outer_gs = GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.2)

    for i, (df, name) in enumerate(plot_items):
        row_idx = i // n_cols
        col_idx = i % n_cols

        inner_gs = outer_gs[row_idx, col_idx].subgridspec(
            4, 4, hspace=0.05, wspace=0.05
        )

        ax_main = fig.add_subplot(inner_gs[1:4, 0:3])
        ax_top = fig.add_subplot(inner_gs[0:1, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(inner_gs[1:4, 3:4], sharey=ax_main)

        if use_kde:
            sns.kdeplot(data=df, x=col1, y=col2, ax=ax_main, fill=True, cmap="Blues")
            sns.kdeplot(data=df, x=col1, ax=ax_top, fill=True, color="steelblue")
            sns.kdeplot(data=df, y=col2, ax=ax_right, fill=True, color="steelblue")
        else:
            sns.histplot(
                data=df,
                x=col1,
                y=col2,
                ax=ax_main,
                cmap="Blues",
                cbar=False,
                discrete=(is_col1_cat, is_col2_cat),
            )
            sns.histplot(
                data=df, x=col1, ax=ax_top, color="steelblue", discrete=is_col1_cat
            )
            sns.histplot(
                data=df, y=col2, ax=ax_right, color="steelblue", discrete=is_col2_cat
            )

        ax_top.set_title(name, fontsize=20, fontweight="bold")
        ax_top.axis("off")
        ax_right.axis("off")

        ax_main.set_xlabel(col1, fontsize=20)
        if col_idx == 0:
            ax_main.set_ylabel(col2, fontsize=20)
        else:
            ax_main.set_ylabel("")
            plt.setp(ax_main.get_yticklabels(), visible=False)

        if x_range:
            ax_main.set_xlim(x_range)
        if y_range:
            ax_main.set_ylim(y_range)

    if save_path:
        full_save_path = os.path.join(save_path, f"{dataset}/joint_{col1}_{col2}.svg")
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(full_save_path, bbox_inches="tight", dpi=300, format="svg")
        plt.close()
        print(f"Saved marginal joint plot to {full_save_path}")
    else:
        plt.show()
