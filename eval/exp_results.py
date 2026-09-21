import os
import glob
import pandas as pd
import numpy as np
import re

DATASET_TYPES = {
    "adult": "classification",
    "barley": "classification",
    "ecoli70": "regression",
    "loan": "classification",
    "credit": "classification",
    "insurance": "regression",
    "king": "regression",
    "covertype": "classification",
    "pm25": "regression",
}

BASE_PATHS = [
    # "results/",
    # "results/mine",
    "results/baselines",
    # "results/eps1.0",
    # "results/eps2.0",
    # "results/eps4.0",
    # "results/eps8.0",
    # "results/eps16.0",
    # "results/eps100.0",
    # "results/ablation/W",
    # "results/ablation/0.3",
    # "results/ablation/0.45",
    # "results/ablation/0.6",
    # "results/ablation/0.9",
    # "results/ablation/order",
    # "results/random_order",
    # "results/ablation/1",
    # "results/ablation/3",
    # "results/ablation/4",
    # "results/ablation/16",
    # "results/ablation/32",
    # "results/ablation/8",
    # "results/ablation/attn",
]


def parse_dataset_info(dataset_path_str):
    filename = os.path.basename(dataset_path_str)

    match = re.match(r"sampled_(.+?)_(.+?)_(\d+)\.csv", filename)
    if match:
        dataset = match.group(1)
        model = match.group(2)
        seed = int(match.group(3))
        return dataset, model, seed

    match = re.match(r"sampled_(.+?)_(\d+)\.csv", filename)
    if match:
        dataset = match.group(1)
        model = "mine"
        seed = int(match.group(2))
        return dataset, model, seed

    return None, None, None


def get_dataset_type(dataset_name):
    return DATASET_TYPES.get(dataset_name, None)


def process_results():
    data_store = {}
    detailed_store = {}

    all_files = []
    for base_path in BASE_PATHS:
        if os.path.exists(base_path):
            csvs = glob.glob(os.path.join(base_path, "*.csv"))
            all_files.extend(csvs)

    for file_path in all_files:
        filename = os.path.basename(file_path)

        is_stat = "stat" in filename
        is_ml = "ml" in filename
        is_extra = "extra" in filename

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"failed: {file_path}, {e}")
            continue

        for _, row in df.iterrows():
            dataset_path = row["Dataset"]
            dataset_name, model_name, seed = parse_dataset_info(dataset_path)

            if dataset_name is None:
                continue

            task_type = get_dataset_type(dataset_name)

            if task_type == None:
                continue

            if model_name not in data_store:
                data_store[model_name] = {"classification": {}, "regression": {}}
            if seed not in data_store[model_name][task_type]:
                data_store[model_name][task_type][seed] = {}

            if dataset_name not in detailed_store:
                detailed_store[dataset_name] = {}
            if model_name not in detailed_store[dataset_name]:
                detailed_store[dataset_name][model_name] = {}
            if seed not in detailed_store[dataset_name][model_name]:
                detailed_store[dataset_name][model_name][seed] = {}

            target_dict = data_store[model_name][task_type][seed]
            ds_target = detailed_store[dataset_name][model_name][seed]

            if is_stat:
                wd = row.get("Average WD")
                jsd = row.get("Average JSD")
                corr = row.get("Correlation Distance")

                for k, v in [("WD", wd), ("JSD", jsd), ("Corr", corr)]:
                    for d in [target_dict, ds_target]:
                        if k not in d:
                            d[k] = []
                        d[k].append(v)
            
            elif is_extra:
                ap = row.get("Alpha-Precision")
                br = row.get("Beta-Recall")
                c2st = row.get("C2ST")
                
                for k, v in [("AP", ap), ("BR", br), ("C2ST", c2st)]:
                    if not pd.isna(v):
                        for d in [target_dict, ds_target]:
                            if k not in d:
                                d[k] = []
                            d[k].append(v)

            elif is_ml:
                if task_type == "classification":
                    metrics = [
                        ("ACC", "avg_Acc_real", "avg_Acc_fake"),
                        ("AUC", "avg_AUC_real", "avg_AUC_fake"),
                        ("F1", "avg_F1_Score_real", "avg_F1_Score_fake"),
                    ]
                    for m_name, real_col, fake_col in metrics:
                        if real_col in row and fake_col in row:
                            diff = row[real_col] - row[fake_col]
                            for d in [target_dict, ds_target]:
                                if m_name not in d:
                                    d[m_name] = []
                                d[m_name].append(diff)

                elif task_type == "regression":
                    if "avg_MSE_real" in row and "avg_MSE_fake" in row:
                        diff = row["avg_MSE_fake"] - row["avg_MSE_real"]
                        for d in [target_dict, ds_target]:
                            if "MSE" not in d:
                                d["MSE"] = []
                            d["MSE"].append(diff)

                    for m_name, r_col, f_col in [
                        ("EVS", "avg_EVS_real", "avg_EVS_fake"),
                        ("R2", "avg_R2_real", "avg_R2_fake"),
                    ]:
                        if r_col in row and f_col in row:
                            diff = row[r_col] - row[f_col]
                            for d in [target_dict, ds_target]:
                                if m_name not in d:
                                    d[m_name] = []
                                d[m_name].append(diff)
                                
            elif "constraint" in filename:
                rate = row.get("Constraint_Violation_Rate")*100.0
                if rate is not None:
                    for d in [target_dict, ds_target]:
                        if "Constraint" not in d:
                            d["Constraint"] = []
                        d["Constraint"].append(rate)


    return data_store, detailed_store


def aggregate_and_format(data_store):
    final_table = {}

    for model, tasks in data_store.items():
        if model not in final_table:
            final_table[model] = {}

        seed_metrics = {}

        for task_type, seeds_data in tasks.items():
            for seed, metrics in seeds_data.items():
                for metric, values in metrics.items():
                    if metric not in seed_metrics:
                        seed_metrics[metric] = {}
                    if seed not in seed_metrics[metric]:
                        seed_metrics[metric][seed] = []

                    valid_values = [v for v in values if not pd.isna(v)]
                    seed_metrics[metric][seed].extend(valid_values)

        for metric, seeds_dict in seed_metrics.items():
            seed_means = []
            for seed, vals in seeds_dict.items():
                if vals:
                    seed_means.append(np.mean(vals))

            if seed_means:
                m_mean = np.mean(seed_means)
                m_std = np.std(seed_means)
                final_table[model][metric] = {
                    "val": m_mean,
                    "str": f"{m_mean:.3f}±{m_std:.3f}",
                }

    return final_table


def aggregate_detailed(detailed_store):
    final_detailed = {}
    for ds_name, models_data in detailed_store.items():
        final_detailed[ds_name] = {}
        for model, seeds_data in models_data.items():
            final_detailed[ds_name][model] = {}
            metric_seeds = {}
            for seed, metrics in seeds_data.items():
                for m_name, vals in metrics.items():
                    if m_name not in metric_seeds:
                        metric_seeds[m_name] = []
                    if vals:
                        metric_seeds[m_name].append(np.mean(vals))

            for m_name, seed_means in metric_seeds.items():
                if seed_means:
                    m_mean = np.mean(seed_means)
                    m_std = np.std(seed_means)
                    final_detailed[ds_name][model][m_name] = {
                        "val": m_mean,
                        "str": f"{m_mean:.3f}±{m_std:.3f}",
                    }
    return final_detailed


def print_table(final_table, metrics_order, title="EXPERIMENT RESULTS"):
    print(f"\n{'='*30} {title} {'='*30}")

    if not final_table:
        print("无数据。")
        return

    models = sorted(list(final_table.keys()))

    headers = ["Model"] + metrics_order

    rows = []

    col_values = {m: [] for m in metrics_order}
    for model in models:
        for m in metrics_order:
            item = final_table[model].get(m)
            if item:
                col_values[m].append(item["val"])
            else:
                col_values[m].append(np.inf)

    best_indices = {}
    second_indices = {}
    improvement_vals = {}

    for m in metrics_order:
        vals = col_values[m]
        sorted_unique_vals = sorted(list(set([v for v in vals if v != np.inf])))

        if len(sorted_unique_vals) >= 1:
            best_val = sorted_unique_vals[0]
            best_indices[m] = [i for i, v in enumerate(vals) if v == best_val]

            if len(sorted_unique_vals) >= 2:
                second_val = sorted_unique_vals[1]
                second_indices[m] = [i for i, v in enumerate(vals) if v == second_val]
                if abs(second_val) > 1e-6:
                    improvement_vals[m] = (
                        f"{(second_val - best_val) / abs(second_val) * 100:.1f}%"
                    )
                else:
                    improvement_vals[m] = f"{second_val - best_val:.3f}"

    for i, model in enumerate(models):
        row_str = [model]
        for m in metrics_order:
            item = final_table[model].get(m)
            if item:
                val_str = item["str"]
                if i in best_indices.get(m, []):
                    val_str = f"**{val_str}**"
                elif i in second_indices.get(m, []):
                    val_str = f"*{val_str}*"
                row_str.append(val_str)
            else:
                row_str.append("-")
        rows.append(row_str)

    imp_row = ["improvement"]
    for m in metrics_order:
        imp_row.append(improvement_vals.get(m, "-"))
    rows.append(imp_row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    header_row = " | ".join([h.ljust(w) for h, w in zip(headers, col_widths)])
    print(header_row)
    print("-" * len(header_row))

    for row in rows:
        print(" | ".join([c.ljust(w) for c, w in zip(row, col_widths)]))


def print_detailed_results(final_detailed, all_metrics):
    for ds_name in sorted(final_detailed.keys()):
        task_type = get_dataset_type(ds_name)
        if task_type == "classification":
            ds_metrics = [m for m in all_metrics if m not in ["MSE", "EVS", "R2"]]
        else:
            ds_metrics = [m for m in all_metrics if m not in ["ACC", "AUC", "F1"]]

        print_table(
            final_detailed[ds_name], ds_metrics, title=f"DATASET: {ds_name.upper()}"
        )


if __name__ == "__main__":
    raw_data, detailed_data = process_results()

    final_results = aggregate_and_format(raw_data)
    final_detailed = aggregate_detailed(detailed_data)

    all_metrics = ["AP","BR","C2ST"]
    print_table(final_results, all_metrics)

    # print_detailed_results(final_detailed, all_metrics)
