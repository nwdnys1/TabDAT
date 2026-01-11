import os
import glob
import pandas as pd
import numpy as np
import re

# =================配置区域=================

# 定义数据集类型字典 (请根据实际情况补充完整)
# 键是数据集名称，值是 'classification' 或 'regression'
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

# 定义结果路径
BASE_PATHS = [
    # "results/mine",  # 自己的模型
    # "results/baselines",  # 基线模型
    "results/eps1.0",  # 差分隐私模型
    # "results/eps100.0",
    # "results/ablation/W",
    # "results/ablation/0.3",
    # "results/ablation/0.45",
    # "results/ablation/0.6",
    # "results/ablation/0.9",
]

# =================数据处理逻辑=================


def parse_dataset_info(dataset_path_str):
    """
    从 CSV 的 Dataset 列（文件路径）中解析 数据集名、模型名、采样次数
    例如: /home/zh/.../sampled_adult_tvae_3.csv -> adult, tvae, 3
    """
    filename = os.path.basename(dataset_path_str)
    # 假设文件名格式固定为: sampled_{dataset}_{model}_{seed}.csv
    # 使用非贪婪匹配提取
    match = re.match(r"sampled_(.+?)_(.+?)_(\d+)\.csv", filename)
    if match:
        dataset = match.group(1)
        model = match.group(2)
        seed = int(match.group(3))
        return dataset, model, seed

    # 否则文件名应该形如sampled_loan_1000.csv
    match = re.match(r"sampled_(.+?)_(\d+)\.csv", filename)
    if match:
        dataset = match.group(1)
        model = "mine"
        seed = int(match.group(2))
        return dataset, model, seed

    return None, None, None


def get_dataset_type(dataset_name):
    """获取数据集类型，默认为 classification 以防字典缺失"""
    return DATASET_TYPES.get(dataset_name, None)


def process_results():
    # 数据结构: data[model][task_type][seed][metric] = [value1, value2, ...]
    # task_type: 'classification' or 'regression'
    data_store = {}

    # 1. 遍历所有 CSV 文件
    all_files = []
    for base_path in BASE_PATHS:
        if os.path.exists(base_path):
            # 查找该路径下的所有 csv (不递归，只看当前层)
            csvs = glob.glob(os.path.join(base_path, "*.csv"))
            all_files.extend(csvs)

    print(f"找到 {len(all_files)} 个结果文件，开始处理...")

    for file_path in all_files:
        filename = os.path.basename(file_path)

        # 判断是 stat 还是 ml
        is_stat = "stat" in filename
        is_ml = "ml" in filename

        if not (is_stat or is_ml):
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"读取文件失败: {file_path}, {e}")
            continue

        for _, row in df.iterrows():
            dataset_path = row["Dataset"]
            dataset_name, model_name, seed = parse_dataset_info(dataset_path)

            if dataset_name is None:
                continue

            task_type = get_dataset_type(dataset_name)

            if task_type == None:
                continue

            # 初始化字典结构
            if model_name not in data_store:
                data_store[model_name] = {"classification": {}, "regression": {}}
            if seed not in data_store[model_name][task_type]:
                data_store[model_name][task_type][seed] = {}

            target_dict = data_store[model_name][task_type][seed]

            # --- 处理统计相似性指标 (Stat) ---
            if is_stat:
                # 对应列: Average WD, Average JSD, Correlation Distance
                # 注意：CSV列名可能带有空格，建议strip
                wd = row.get("Average WD")
                jsd = row.get("Average JSD")
                corr = row.get("Correlation Distance")

                for k, v in [("WD", wd), ("JSD", jsd), ("Corr", corr)]:
                    if k not in target_dict:
                        target_dict[k] = []
                    target_dict[k].append(v)

            # --- 处理机器学习效用指标 (ML) ---
            elif is_ml:
                if task_type == "classification":
                    # ACC, AUC, F1. 越大越好，所以 Gap = Real - Fake
                    metrics = [
                        ("ACC", "avg_Acc_real", "avg_Acc_fake"),
                        ("AUC", "avg_AUC_real", "avg_AUC_fake"),
                        ("F1", "avg_F1_Score_real", "avg_F1_Score_fake"),
                    ]
                    for m_name, real_col, fake_col in metrics:
                        if real_col in row and fake_col in row:
                            diff = row[real_col] - row[fake_col]  # Real - Fake
                            if m_name not in target_dict:
                                target_dict[m_name] = []
                            target_dict[m_name].append(diff)

                elif task_type == "regression":
                    # MSE, EVS, R2.
                    # MSE: 越小越好. Gap = Fake - Real (如果Fake误差大，结果为正)
                    # EVS, R2: 越大越好. Gap = Real - Fake

                    # 注意：需求12提到 avg_MSE, 但需求13要求输出 MAPE。
                    # 这里优先读取 MSE 列，如果 CSV 中实际是 MAPE 请修改列名

                    # 处理 MSE (或 MAPE)
                    if "avg_MSE_real" in row and "avg_MSE_fake" in row:
                        diff = row["avg_MSE_fake"] - row["avg_MSE_real"]  # Fake - Real
                        if "MSE" not in target_dict:
                            target_dict["MSE"] = []
                        target_dict["MSE"].append(diff)

                    # 处理 EVS
                    if "avg_EVS_real" in row and "avg_EVS_fake" in row:
                        diff = row["avg_EVS_real"] - row["avg_EVS_fake"]  # Real - Fake
                        if "EVS" not in target_dict:
                            target_dict["EVS"] = []
                        target_dict["EVS"].append(diff)

                    # 处理 R2
                    if "avg_R2_real" in row and "avg_R2_fake" in row:
                        diff = row["avg_R2_real"] - row["avg_R2_fake"]  # Real - Fake
                        if "R2" not in target_dict:
                            target_dict["R2"] = []
                        target_dict["R2"].append(diff)

    return data_store


def aggregate_and_format(data_store):
    # 最终结果: final_table[task_type][model][metric] = {'mean': x, 'std': y, 'str': "x±y"}
    final_table = {"classification": {}, "regression": {}}

    for model, tasks in data_store.items():
        for task_type, seeds_data in tasks.items():
            if not seeds_data:
                continue

            # 1. 先计算每个 Seed 内部的平均值 (跨数据集平均)
            # seed_means[metric] = [mean_seed_1, mean_seed_2, ..., mean_seed_5]
            seed_means = {}

            for seed, metrics in seeds_data.items():
                for metric, values in metrics.items():
                    if metric not in seed_means:
                        seed_means[metric] = []
                    # 对该 seed 下所有数据集的该指标求平均 若遇到 NaN 则跳过
                    valid_values = [v for v in values if not pd.isna(v)]

                    if valid_values:
                        # 只有当存在有效值时才计算平均
                        seed_means[metric].append(np.mean(valid_values))

            # 2. 计算 5 次采样的 Mean 和 Std
            if model not in final_table[task_type]:
                final_table[task_type][model] = {}

            for metric, values in seed_means.items():
                m_mean = np.mean(values)
                m_std = np.std(values)
                final_table[task_type][model][metric] = {
                    "val": m_mean,  # 用于排序
                    "str": f"{m_mean:.3f}±{m_std:.3f}",
                }

    return final_table


def print_table(final_table, task_type, metrics_order):
    print(f"\n{'='*20} {task_type.upper()} DATASETS RESULTS {'='*20}")

    if not final_table[task_type]:
        print("无数据。")
        return

    models = list(final_table[task_type].keys())

    # 准备表头
    headers = ["Model"] + metrics_order

    # 准备每一行数据
    rows = []

    # 为了标注最优和次优，我们需要先收集每一列的所有值
    # col_values[metric] = [val_model_A, val_model_B, ...]
    col_values = {m: [] for m in metrics_order}
    for model in models:
        for m in metrics_order:
            item = final_table[task_type][model].get(m)
            if item:
                col_values[m].append(item["val"])
            else:
                col_values[m].append(np.inf)  # 缺失值设为无穷大，避免被选为最优

    # 确定每一列的最优（最小值）和次优
    # 假设所有指标（包括ML的Gap和Stat的距离）都是越小越好 (接近0)
    best_indices = {}
    second_indices = {}

    for m in metrics_order:
        vals = col_values[m]
        if not vals:
            continue

        # 排序并去重以找到第一小和第二小的值
        sorted_vals = sorted(list(set([v for v in vals if v != np.inf])))

        if len(sorted_vals) >= 1:
            best_val = sorted_vals[0]
            # 找到所有等于 best_val 的索引
            best_indices[m] = [i for i, v in enumerate(vals) if v == best_val]

        if len(sorted_vals) >= 2:
            second_val = sorted_vals[1]
            second_indices[m] = [i for i, v in enumerate(vals) if v == second_val]

    # 构建打印行
    for i, model in enumerate(models):
        row_str = [model]
        for m in metrics_order:
            item = final_table[task_type][model].get(m)
            if item:
                val_str = item["str"]
                # 添加星号
                if i in best_indices.get(m, []):
                    val_str = f"**{val_str}**"
                elif i in second_indices.get(m, []):
                    val_str = f"*{val_str}*"
                row_str.append(val_str)
            else:
                row_str.append("-")
        rows.append(row_str)

    # 简单的对齐打印
    # 计算每列最大宽度
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    # 打印表头
    header_row = " | ".join([h.ljust(w) for h, w in zip(headers, col_widths)])
    print(header_row)
    print("-" * len(header_row))

    # 打印数据行
    for row in rows:
        print(" | ".join([c.ljust(w) for c, w in zip(row, col_widths)]))


# =================主程序=================

if __name__ == "__main__":
    # 1. 处理数据
    raw_data = process_results()

    # 2. 聚合统计
    final_results = aggregate_and_format(raw_data)

    # 3. 打印分类结果
    # 指标顺序: ACC, AUC, F1, JSD, WD, Corr
    class_metrics = ["ACC", "AUC", "F1", "JSD", "WD", "Corr"]
    print_table(final_results, "classification", class_metrics)

    # 4. 打印回归结果
    # 指标顺序: MSE(对应MAPE位置), EVS, R2, JSD, WD, Corr
    # 注意：代码中处理的是 MSE，如果需要显示为 MAPE，请确保输入文件包含 MAPE 或在此理解为 MSE
    reg_metrics = ["MSE", "EVS", "R2", "JSD", "WD", "Corr"]
    print_table(final_results, "regression", reg_metrics)
