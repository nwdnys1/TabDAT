import os
import glob
import pandas as pd
from dython.nominal import compute_associations
from feature_distribution import (
    plot_feature_distribution,
    plot_feature_distribution_ablation,
    plot_multiple_feature_distributions,
)
from joint_distribution import plot_joint_distribution
from correlation_diff import plot_correlation_difference

cat_cols = {
    "adult": [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "class",
    ],
    "barley": "all",
    "ecoli70": [],
    "loan": [
        "ZIP Code",
        "Education",
        "Personal Loan",
        "Securities Account",
        "CD Account",
        "Online",
        "CreditCard",
        "Mortgage",
    ],
    "credit": ["Class"],
    "insurance": [
        "sex",
        "smoker",
        "region",
    ],
    "king": [
        "waterfront",
        "yr_renovated",
        "zipcode",
    ],
    "covertype": [f"Soil_Type{i}" for i in range(1, 41)]
    + [f"Wilderness_Area{i}" for i in range(1, 5)]
    + ["Covertype"],
    "pm25": ["cbwd"],
}

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(cur_dir, "../../results/visualization")
    models_to_plot = [
        "mine",
        "ctabganplus",
        "ttvae",
        # "ctgan",
        # "tvae",
        # "tabmt",
        # "tabsyn",
        # "tabnat",
        # "tabddpm",
    ]

    # for dataset in [
    #     "adult",
    #     "ecoli70",
    #     "barley",
    #     "loan",
    #     "credit",
    #     "insurance",
    #     "king",
    #     "covertype",
    #     "pm25",
    # ]:

    #     real_path = f"{cur_dir}/../eval/real_datasets/{dataset}.csv"
    #     fake_dir = f"{cur_dir}/../eval/fake_datasets/{dataset}/baselines"

    #     cols = pd.read_csv(real_path).columns.tolist()
    #     for col in cols:
    #         plot_feature_distribution(
    #             dataset, col, cat_cols, real_path, fake_dir, save_path, models_to_plot
    #         )

    # feature_list = [
    #     ("adult", "education-num"),
    #     ("ecoli70", "lacY"),
    #     ("adult", "relationship"),
    #     ("pm25", "cbwd"),
    # ]
    # plot_multiple_feature_distributions(
    #     feature_list=feature_list,
    #     cat_cols=cat_cols,
    #     real_datasets_dir=f"{cur_dir}/../eval/real_datasets",
    #     fake_datasets_base_dir=f"{cur_dir}/../eval/fake_datasets",
    #     save_path=os.path.join(save_path, "combined_dist.svg"),
    #     models_to_plot=[
    #         "mine",
    #         "ctabganplus",
    #         "ttvae",
    #     ],
    #     n_cols=4,
    # )

    # plot_joint_distribution(
    #     "king",
    #     "bathrooms",
    #     "condition",
    #     cat_cols,
    #     f"{cur_dir}/../eval/real_datasets/king.csv",
    #     f"{cur_dir}/../eval/fake_datasets/king/baselines",
    #     save_path,
    #     models_to_plot,
    # )

    # plot_joint_distribution(
    #     "pm25",
    #     "hour",
    #     "pm2.5",
    #     cat_cols,
    #     f"{cur_dir}/../eval/real_datasets/pm25.csv",
    #     f"{cur_dir}/../eval/fake_datasets/pm25/baselines",
    #     save_path,
    #     models_to_plot,
    # )

    # datasets = [
    #     "adult",
    #     "ecoli70",
    #     "barley",
    #     "loan",
    #     "credit",
    #     "insurance",
    #     "king",
    #     "covertype",
    #     "pm25",
    # ]
    # plot_correlation_difference(
    #     datasets,
    #     models_to_plot,
    #     cat_cols,
    #     save_path,
    # )

    dataset= "adult"
    plot_feature_distribution_ablation(
        dataset,
        "race",
        cat_cols,
        f"{cur_dir}/../eval/real_datasets/{dataset}.csv",
        f"{cur_dir}/../eval/fake_datasets/{dataset}/",
        save_path,
    )
