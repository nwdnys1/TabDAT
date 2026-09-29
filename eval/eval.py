from evaluation import (
    get_utility_metrics,
    stat_sim,
    # Privacy metrics are paused while the non-DP evaluation protocol is revised.
    # privacy_metrics,
    # get_adr_metric,
    # get_eps_metric,
    # get_mia_metric,
    # get_hit_metric,
    # get_nnaa_metric,
    get_extra_metrics
)
import numpy as np
import pandas as pd
import glob
import os
import argparse

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
    stat_res_avg = []
    stat_columns = ["Dataset", "Average WD", "Average JS Distance", "Mean Association Error"]

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
        metric_names = ["Acc", "AUC", "F1_macro"]
    else:
        model_dict = {"Regression": ["l_reg", "ridge","lasso" ,"B_ridge"]}
        metric_names = ["MAPE", "EVS", "R2"]

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
            real_train_path,
            [fake_path_single],
            "MinMax",
            model_dict,
            cat_cols=cat_cols.get(dataset, []),
            target_col=tar_cols.get(dataset, None),
            real_test_path=real_path,
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

def eval_extra(dataset, *, max_rows=2000, seed=42):
    extra_rows = []
    columns = ["Dataset", "Alpha-Precision (naive)", "Beta-Recall (naive)",
               "C2ST LogisticDetection", "Metric Rows"]

    for fake_path in fake_paths:
        print(f"  Evaluating extra metrics for: {fake_path}")
        result = get_extra_metrics(
            real_path, fake_path, cat_cols=cat_cols.get(dataset, []),
            max_rows=max_rows, seed=seed,
        )
        extra_rows.append([
            fake_path, result["alpha_precision"], result["beta_recall"],
            result["c2st"], result["n_rows"],
        ])

    output = pd.DataFrame(extra_rows, columns=columns)
    os.makedirs(f"results/{suffix}", exist_ok=True)
    output.to_csv(f"results/{suffix}/{dataset}_extra_results.csv", index=False)
    print(output.to_string(index=False))


def eval_constraint(dataset):
    """
    Evaluates logical constraints for specific datasets (e.g., adult).
    For adult: checks (sex='Female', relationship='Husband') or (sex='Male', relationship='Wife').
    """
    if dataset != "adult":
        return

    constraint_res_avg = []
    columns = ["Dataset", "Constraint_Violation_Rate"]

    for fake_path in fake_paths:
        print(f"  Evaluating constraints for: {fake_path}")
        try:
            df = pd.read_csv(fake_path)
            # Standardize column names to avoid case/space issues
            df.columns = [c.strip() for c in df.columns]
            
            if 'sex' in df.columns and 'relationship' in df.columns:
                # Count violations: (Female + Husband) OR (Male + Wife)
                # Note: Assuming values are strings like 'Female', 'Male', 'Husband', 'Wife'
                # We strip and convert to lower for robust matching
                invalid_mask = (
                    ((df['sex'].astype(str).str.strip().str.lower() == 'female') & 
                     (df['relationship'].astype(str).str.strip().str.lower() == 'husband')) |
                    ((df['sex'].astype(str).str.strip().str.lower() == 'male') & 
                     (df['relationship'].astype(str).str.strip().str.lower() == 'wife'))
                )
                violation_count = invalid_mask.sum()
                violation_rate = violation_count / len(df) if len(df) > 0 else 0
                constraint_res_avg.append([fake_path, violation_rate])
            else:
                print(f"  Columns 'sex' or 'relationship' not found in {fake_path}")
                constraint_res_avg.append([fake_path, np.nan])
        except Exception as e:
            print(f"  Error evaluating constraints for {fake_path}: {e}")
            constraint_res_avg.append([fake_path, np.nan])

    constraint_results = pd.DataFrame(constraint_res_avg, columns=columns)
    os.makedirs(f"results/{suffix}", exist_ok=True)
    constraint_results.to_csv(f"results/{suffix}/{dataset}_constraint_results.csv", index=False)
    print("Constraint Violation Results:")
    print(constraint_results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TabDAT synthetic tables")
    parser.add_argument("--datasets", nargs="+", choices=problem_types.keys(),
                        default=list(problem_types))
    parser.add_argument("--extra", action="store_true",
                        help="Compute alpha/beta and C2ST (optional dependencies)")
    parser.add_argument("--extra-max-rows", type=int, default=2000)
    parser.add_argument("--extra-seed", type=int, default=42)
    args = parser.parse_args()

    for dataset in args.datasets:
        suffix = "holdout_v1"

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        split_dir = f"{cur_dir}/real_datasets/{dataset}"
        real_train_path = f"{split_dir}/train.csv"
        real_path = f"{split_dir}/test.csv"
        if not os.path.isfile(real_train_path) or not os.path.isfile(real_path):
            raise FileNotFoundError(
                f"Missing train/test split in {split_dir}; prepare it before evaluating"
            )
        fake_path = f"{cur_dir}/fake_datasets/{dataset}/{suffix}"
        fake_paths = sorted(glob.glob(f"{fake_path}/*.csv"))
        if not fake_paths:
            raise FileNotFoundError(f"No generated CSVs found in {fake_path}")

        print(f"Evaluating dataset: {dataset}")
        eval_ss(dataset)
        eval_ml(dataset)
        if args.extra:
            eval_extra(dataset, max_rows=args.extra_max_rows, seed=args.extra_seed)
        # eval_constraint(dataset)
