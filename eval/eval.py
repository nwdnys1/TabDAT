from evaluation import (
    get_utility_metrics,
    stat_sim,
    privacy_metrics,
    get_adr_metric,
    get_eps_metric,
    get_mia_metric,
    get_hit_metric,
    get_nnaa_metric,
)
import numpy as np
import pandas as pd
import glob
import os

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
        # "children",
    ],
    "king": [
        "waterfront",
        "yr_renovated",
        "zipcode",
        # "bedrooms",
        # "bathrooms",
        # "floors",
        # "view",
        # "condition",
        # "grade",
    ],
    "covertype": [f"Soil_Type{i}" for i in range(1, 41)]
    + [f"Wilderness_Area{i}" for i in range(1, 5)]
    + ["Covertype"],  # 44个binary variables 以及1个目标covertype
    "pm25": ["cbwd"],
}

problem_types = {
    "adult": "Classification",
    "barley": "Classification",
    "ecoli70": "Regression",
    "loan": "Classification",
    "credit": "Classification",
    "insurance": "Regression",
    "king": "Regression",
    "covertype": "Classification",
    "pm25": "Regression",
}

tar_cols = {
    "loan": "Personal Loan",
    "credit": "Class",
    "insurance": "charges",
    "king": "price",
    "pm25": "pm2.5",
}


def eval_ss(dataset):
    # 统计相似性评估
    stat_res_avg = []
    stat_columns = ["Dataset", "Average WD", "Average JSD", "Correlation Distance"]

    for fake_path in fake_paths:
        print(f"  Evaluating fake dataset: {fake_path}")
        stat_res = stat_sim(real_path, fake_path, cat_cols=cat_cols.get(dataset, []))
        stat_res_avg.append(stat_res)

    stat_results = pd.DataFrame(np.array(stat_res_avg), columns=stat_columns)
    os.makedirs(f"results/{suffix}", exist_ok=True)
    stat_results.to_csv(f"results/{suffix}/{dataset}_stat_results.csv", index=False)
    print(stat_results.to_string(index=False))


def eval_ml(dataset):
    problem_type = problem_types[dataset]

    if problem_type == "Classification":
        model_dict = {"Classification": ["lr", "dt", "rf", "mlp", "svm"]}
        metric_names = ["Acc", "AUC", "F1_Score"]
    else:
        model_dict = {"Regression": ["l_reg", "ridge","lasso" ,"B_ridge"]}
        metric_names = ["MSE", "EVS", "R2"]
    # 机器学习效用评估
    ml_res_avg = []
    models = list(model_dict.values())[0]

    # Update column names
    ml_columns = ["Dataset"]
    for model in models:
        for metric in metric_names:
            ml_columns.append(f"{model}_{metric}_real")
            ml_columns.append(f"{model}_{metric}_fake")
    for metric in metric_names:
        ml_columns.append(f"avg_{metric}_real")
        ml_columns.append(f"avg_{metric}_fake")

    for fake_path_single in fake_paths:
        real_results, fake_results = get_utility_metrics(
            real_path,
            [fake_path_single],
            "MinMax",
            model_dict,
            test_ratio=0.20,
            cat_cols=cat_cols.get(dataset, []),
            target_col=tar_cols.get(dataset, None),
        )

        row = [fake_path_single]
        # Add real and fake metrics for each model
        for i in range(len(models)):
            for j in range(len(metric_names)):
                row.append(real_results[i, j])
                row.append(fake_results[i, j])

        # Calculate and add average metrics
        avg_real = np.mean(real_results, axis=0)
        avg_fake = np.mean(fake_results, axis=0)
        for i in range(len(metric_names)):
            row.append(avg_real[i])
            row.append(avg_fake[i])

        ml_res_avg.append(row)

    ml_results = pd.DataFrame(ml_res_avg, columns=ml_columns)
    os.makedirs(f"results/{suffix}", exist_ok=True)
    ml_results.to_csv(f"results/{suffix}/{dataset}_ml_results.csv", index=False)
    print("Machine Learning Utility Results:")
    print(ml_results.to_string(index=False))


def eval_priv(dataset):
    # 隐私保护评估
    priv_res_avg = []
    privacy_columns = [
        "Dataset",
        "DCRsr",
        "DCRrr",
        "DCRss",
        "DCRgain",
        "NNDRsr",
        "NNDRrr",
        "NNDRss",
        "NNDRgain",
    ]
    for fake_path_single in fake_paths:
        priv_res = privacy_metrics(
            real_path, fake_path_single, cat_cols.get(dataset, [])
        )
        priv_res_avg.append([fake_path_single] + priv_res.flatten().tolist())

    privacy_results = pd.DataFrame(priv_res_avg, columns=privacy_columns)
    privacy_results.to_csv(f"results/{dataset}_privacy_results.csv", index=False)
    print("\nPrivacy Metrics Results:")
    print(privacy_results.to_string(index=False))


if __name__ == "__main__":

    for dataset in [
        "adult",
        "barley",
        "ecoli70",
        "loan",
        "credit",
        "insurance",
        "king",
        "covertype",
        "pm25"
    ]:
        suffix = "eps100.0"

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        real_path = f"{cur_dir}/real_datasets/{dataset}.csv"
        fake_path = f"{cur_dir}/fake_datasets/{dataset}/{suffix}"
        fake_paths = glob.glob(f"{fake_path}/*.csv")

        print(f"Evaluating dataset: {dataset}")
        eval_ss(dataset)
        eval_ml(dataset)
        # eval_priv(dataset)

        # # ADR evaluation
        # adr_res_avg = []
        # adr_columns = ["Dataset", "ADR"]
        # for fake_path_single in fake_paths:
        #     # "all" might not be the correct value for cat_cols, using what was used for privacy
        #     adr_res = get_adr_metric(real_path, fake_path_single, "all")
        #     adr_res_avg.append(adr_res)

        # adr_results = pd.DataFrame(adr_res_avg, columns=adr_columns)
        # adr_results.to_csv(f"results/{dataset}_adr_results.csv", index=False)
        # print("\nADR Metric Results:")
        # print(adr_results.to_string(index=False))

        # # Epsilon evaluation
        # eps_res_avg = []
        # eps_columns = ["Dataset", "Epsilon"]
        # for fake_path_single in fake_paths:
        #     # "all" might not be the correct value for cat_cols, using what was used for privacy
        #     eps_res = get_eps_metric(
        #         real_path, fake_path_single, categoricals.get(dataset, [])
        #     )
        #     eps_res_avg.append(eps_res)
        # eps_results = pd.DataFrame(eps_res_avg, columns=eps_columns)
        # eps_results.to_csv(f"results/{dataset}_eps_results.csv", index=False)
        # print("\nEpsilon Metric Results:")
        # print(eps_results.to_string(index=False))

        # # MIA
        # mia_res_avg = []
        # mia_columns = ["Dataset", "MIA_Precision"]
        # for fake_path_single in fake_paths:
        #     # "all" might not be the correct value for cat_cols, using what was used for privacy
        #     mia_res = get_mia_metric(
        #         real_path, fake_path_single, categoricals.get(dataset, [])
        #     )
        #     mia_res_avg.append(mia_res)  # Placeholder for MIA metric function
        # mia_results = pd.DataFrame(mia_res_avg, columns=mia_columns)
        # mia_results.to_csv(f"results/{dataset}_mia_results.csv", index=False)
        # print("\nMIA Metric Results:")
        # print(mia_results.to_string(index=False))

        # # hitting rate
        # hr_res_avg = []
        # hr_columns = ["Dataset", "Hitting_Rate"]
        # for fake_path_single in fake_paths:
        #     # "all" might not be the correct value for cat_cols, using what was used for privacy
        #     hr_res = get_hit_metric(
        #         real_path, fake_path_single, categoricals.get(dataset, [])
        #     )
        #     hr_res_avg.append(hr_res)  # Placeholder for Hitting Rate metric function
        # hr_results = pd.DataFrame(hr_res_avg, columns=hr_columns)
        # hr_results.to_csv(f"results/{dataset}_hr_results.csv", index=False)
        # print("\nHitting Rate Metric Results:")
        # print(hr_results.to_string(index=False))

        # # nnaa
        # nnaa_res_avg = []
        # nnaa_columns = ["Dataset", "NNAA"]
        # for fake_path_single in fake_paths:
        #     # "all" might not be the correct value for cat_cols, using what was used for privacy
        #     nnaa_res = get_nnaa_metric(
        #         real_path, fake_path_single, categoricals.get(dataset, [])
        #     )
        #     nnaa_res_avg.append(nnaa_res)  # Placeholder for NNAA metric function
        # nnaa_results = pd.DataFrame(nnaa_res_avg, columns=nnaa_columns)
        # nnaa_results.to_csv(f"results/{dataset}_nnaa_results.csv", index=False)
        # print("\nNNAA Metric Results:")
        # print(nnaa_results.to_string(index=False))
