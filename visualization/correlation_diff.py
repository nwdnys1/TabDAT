import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from dython.nominal import compute_associations


def plot_correlation_difference(datasets, models_to_plot, cat_cols, save_path):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    num_datasets = len(datasets)
    num_models = len(models_to_plot)

    fig, axes = plt.subplots(
        num_datasets,
        num_models,
        figsize=(3 * num_models + 1, 3 * num_datasets),
        sharex=False,
        sharey=False,
    )

    if num_datasets == 1 and num_models == 1:
        axes = np.array([[axes]])
    elif num_datasets == 1:
        axes = axes[np.newaxis, :]
    elif num_models == 1:
        axes = axes[:, np.newaxis]

    vmin, vmax = 0, 0.4

    for d_idx, dataset in enumerate(datasets):
        real_path = f"{cur_dir}/../eval/real_datasets/{dataset}.csv"
        if not os.path.exists(real_path):
            for m_idx in range(num_models):
                axes[d_idx, m_idx].axis("off")
            continue

        real_df = pd.read_csv(real_path)
        dataset_cat_cols = cat_cols.get(dataset, [])
        real_corr = compute_associations(real_df, nominal_columns=dataset_cat_cols)
        cols_to_use = real_corr.columns.tolist()

        fake_base_dir = f"{cur_dir}/../eval/fake_datasets/{dataset}/baselines/"
        all_fake_files = sorted(glob.glob(os.path.join(fake_base_dir, "*.csv")))

        for m_idx, model_name in enumerate(models_to_plot):
            ax = axes[d_idx, m_idx]

            target_fpath = None
            for fpath in all_fake_files:
                fname = os.path.basename(fpath)
                parts = fname.replace(".csv", "").split("_")
                if (
                    len(parts) >= 4
                    and parts[1] == dataset
                    and parts[2] == model_name
                    and parts[3] == "1"
                ):
                    target_fpath = fpath
                    break

            if target_fpath:
                try:
                    synth_df = pd.read_csv(target_fpath)
                    synth_corr = compute_associations(
                        synth_df, nominal_columns=dataset_cat_cols
                    )
                    synth_corr = synth_corr.reindex(
                        index=cols_to_use, columns=cols_to_use
                    )
                    diff_corr = (real_corr - synth_corr).abs().astype(float)

                    sns.heatmap(
                        diff_corr,
                        ax=ax,
                        cmap="Greens",
                        vmin=vmin,
                        vmax=vmax,
                        cbar=False,
                        xticklabels=False,
                        yticklabels=False,
                    )
                except Exception:
                    ax.axis("off")
            else:
                ax.axis("off")

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

            if d_idx == 0:
                ax.set_title(model_name, fontsize=20, fontweight="bold")
            if m_idx == 0:
                ax.set_ylabel(dataset.capitalize(), fontsize=20, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap="Greens", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
        shrink=0.85,
        aspect=30,
    )
    cbar.ax.tick_params(labelsize=25)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, "correlation_diff_grid.png")
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(full_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved correlation grid plot with single colorbar to {full_path}")
