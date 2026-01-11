import pandas as pd
import numpy as np
import os
import re


def extract_model_name(filename):
    """
    从文件名中提取模型名称。
    例如:
    - /path/to/sampled_syn_smote_2.csv -> smote
    - /path/to/sampled_syn_ttvae_5.csv -> ttvae
    - /path/to/sampled_syn_304.csv -> ctabdm (默认)
    """
    basename = os.path.basename(filename)
    match = re.search(r"sampled_\w+_(.+?)_\d+\.csv", basename)
    if match:
        return match.group(1)
    # 如果不匹配特定模型格式，则假定为默认模型
    return "FedBN"


def load_and_merge_results(filename, results_dir):
    """
    尝试从 results_dir 和 results_dir/baselines 读取文件并合并。
    """
    baselines_dir = os.path.join(results_dir, "baselines")
    dfs = []
    
    main_file = os.path.join(results_dir, filename)
    baseline_file = os.path.join(baselines_dir, filename)
    
    if os.path.exists(main_file):
        dfs.append(pd.read_csv(main_file))
    if os.path.exists(baseline_file):
        dfs.append(pd.read_csv(baseline_file))
        
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def highlight_best(s, higher_is_better=True):
    """
    高亮Series中的最佳和次佳值。
    - 最佳值加粗并加两个星号 (e.g., **0.123**)
    - 次佳值加粗并加一个星号 (e.g., *0.456*)
    """
    # 在函数开始时就转换为字符串，以确保返回类型始终是字符串
    styled_s = s.astype(str)

    s_numeric = pd.to_numeric(s, errors="coerce")
    if s_numeric.isnull().all():
        return styled_s  # 返回字符串版本的Series

    if higher_is_better:
        sorted_values = s_numeric.dropna().sort_values(ascending=False)
    else:
        sorted_values = s_numeric.dropna().sort_values(ascending=True)

    if len(sorted_values) < 1:
        return styled_s  # 返回字符串版本的Series

    best_val = sorted_values.iloc[0]
    second_best_val = sorted_values.iloc[1] if len(sorted_values) > 1 else None

    # 使用isclose处理浮点数比较问题
    # np.isclose要求输入不能有nan
    valid_indices = s_numeric.notna()

    # 找到最佳值的索引
    best_indices = np.zeros_like(s_numeric, dtype=bool)
    if valid_indices.any():
        best_indices[valid_indices] = np.isclose(s_numeric[valid_indices], best_val)

    # 找到次佳值的索引
    second_best_indices = np.zeros_like(s_numeric, dtype=bool)
    if (
        second_best_val is not None
        and not np.isclose(best_val, second_best_val)
        and valid_indices.any()
    ):
        second_best_indices[valid_indices] = np.isclose(
            s_numeric[valid_indices], second_best_val
        )

    # 应用样式
    # 次佳值
    styled_s.loc[second_best_indices] = "*" + styled_s.loc[second_best_indices] + "*"
    # 最佳值 (后应用以覆盖)
    styled_s.loc[best_indices] = (
        "**" + styled_s.loc[best_indices].str.replace("*", "") + "**"
    )

    return styled_s


def process_and_display_results(dataset_name, results_dir="results"):
    """
    处理给定数据集的统计、机器学习和隐私评估结果，
    并以格式化的表格形式打印。
    """
    print(f"--- Processing results for dataset: {dataset_name} ---")

    task_type_map = {
        "shuttle": "classification",
        "statlog": "classification",
        "cancer": "classification",
        "asia": "classification",
        "adult": "classification",
        "syn": "regression",
        "auto": "regression",
        "airfoil": "regression",
        "concrete": "regression",
        "abalone": "regression",
        "pm25": "regression",
        "barley": "classification",
        "ecoli70": "regression",
        "diabetes": "classification",
        "hailfinder": "classification",
         "loan": "classification",
        "credit": "classification",
        "insurance": "regression",
        "king": "regression",
        "covertype": "classification",
    }
    task_type = task_type_map.get(dataset_name, "classification")  # 默认为分类

    # 1. 处理统计结果
    stat_filename = f"{dataset_name}_stat_results.csv"
    df_stat = load_and_merge_results(stat_filename, results_dir)

    if df_stat is not None:
        df_stat["model"] = df_stat.iloc[:, 0].apply(extract_model_name)

        stat_metrics = ["Average WD", "Average JSD", "Correlation Distance"]

        # 计算均值和标准差
        agg_funcs = {metric: ["mean", "std"] for metric in stat_metrics}
        stat_summary = df_stat.groupby("model").agg(agg_funcs).reset_index()
        stat_summary.columns = [
            "_".join(col).strip() for col in stat_summary.columns.values
        ]

        # 格式化为 "mean ± std"
        final_stat_table = pd.DataFrame()
        final_stat_table["Model"] = stat_summary["model_"]
        for metric in stat_metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            # 高亮
            styled_mean_col = highlight_best(
                stat_summary[mean_col], higher_is_better=False
            )

            # 构建最终的列，将高亮后的均值与标准差合并
            final_column_data = []
            for styled, mean, std in zip(
                styled_mean_col, stat_summary[mean_col], stat_summary[std_col]
            ):
                # 检查高亮标记，并相应地格式化字符串
                if "**" in styled:
                    num_str = styled.replace("**", "")
                    final_column_data.append(f"**{float(num_str):.4f}** ± {std:.4f}")
                elif "*" in styled:
                    num_str = styled.replace("*", "")
                    final_column_data.append(f"*{float(num_str):.4f}* ± {std:.4f}")
                else:
                    final_column_data.append(f"{mean:.4f} ± {std:.4f}")
            final_stat_table[metric] = final_column_data

        print("\n📊 Statistical Similarity Results:")
        print(final_stat_table.to_markdown(index=False))
    else:
        print(f"\n[!] Statistical results file not found: {stat_filename}")

    # 2. 处理机器学习效用结果
    ml_filename = f"{dataset_name}_ml_results.csv"
    df_ml = load_and_merge_results(ml_filename, results_dir)

    if df_ml is not None:
        df_ml["model"] = df_ml.iloc[:, 0].apply(extract_model_name)

        # 根据任务类型筛选指标
        all_ml_metrics = [
            col
            for col in df_ml.columns
            if col.endswith("_fake") and not col.startswith("avg_")
        ]
        if task_type == "regression":
            ml_metrics = [m for m in all_ml_metrics if "r2" in m.lower()]
        else:  # classification
            ml_metrics = [m for m in all_ml_metrics if "f1" in m.lower()]

        if not ml_metrics:
            print(
                f"\n[!] No suitable ML utility metrics found for task type '{task_type}'."
            )
        else:
            agg_funcs = {metric: ["mean", "std"] for metric in ml_metrics}
            ml_summary = df_ml.groupby("model").agg(agg_funcs).reset_index()
            ml_summary.columns = [
                "_".join(col).strip() for col in ml_summary.columns.values
            ]

            final_ml_table = pd.DataFrame()
            final_ml_table["Model"] = ml_summary["model_"]

            # 为每个具体任务指标创建 "mean ± std" 列并高亮
            for metric in ml_metrics:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"
                metric_name = metric.replace("_fake", "")

                # 对均值列进行高亮处理
                # 注意：对于MSE（均方误差），值越小越好
                is_mse = "mse" in metric_name.lower()
                styled_mean_col = highlight_best(
                    ml_summary[mean_col], higher_is_better=not is_mse
                )

                # 构建最终的列，将高亮后的均值与标准差合并
                final_column_data = []
                for styled, mean, std in zip(
                    styled_mean_col, ml_summary[mean_col], ml_summary[std_col]
                ):
                    # 检查高亮标记，并相应地格式化字符串
                    if "**" in styled:
                        num_str = styled.replace("**", "")
                        final_column_data.append(
                            f"**{float(num_str):.4f}** ± {std:.4f}"
                        )
                    elif "*" in styled:
                        num_str = styled.replace("*", "")
                        final_column_data.append(f"*{float(num_str):.4f}* ± {std:.4f}")
                    else:
                        final_column_data.append(f"{mean:.4f} ± {std:.4f}")
                final_ml_table[metric_name] = final_column_data

            print("\n🤖 Machine Learning Utility Results (on Fake Data):")
            print(final_ml_table.to_markdown(index=False))
    else:
        print(f"\n[!] ML utility results file not found: {ml_filename}")

    # 3. 处理隐私结果
    privacy_filename = f"{dataset_name}_privacy_results.csv"
    df_privacy = load_and_merge_results(privacy_filename, results_dir)

    if df_privacy is not None:
        df_privacy["model"] = df_privacy.iloc[:, 0].apply(extract_model_name)

        privacy_metrics = ["DCRsr", "NNDRsr"]

        agg_funcs = {metric: ["mean", "std"] for metric in privacy_metrics}
        privacy_summary = df_privacy.groupby("model").agg(agg_funcs).reset_index()
        privacy_summary.columns = [
            "_".join(col).strip() for col in privacy_summary.columns.values
        ]

        final_privacy_table = pd.DataFrame()
        final_privacy_table["Model"] = privacy_summary["model_"]
        for metric in privacy_metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            # 高亮 (假设值越高越好)
            styled_mean_col = highlight_best(
                privacy_summary[mean_col], higher_is_better=True
            )

            # 构建最终的列，将高亮后的均值与标准差合并
            final_column_data = []
            for styled, mean, std in zip(
                styled_mean_col, privacy_summary[mean_col], privacy_summary[std_col]
            ):
                # 检查高亮标记，并相应地格式化字符串
                if "**" in styled:
                    num_str = styled.replace("**", "")
                    final_column_data.append(f"**{float(num_str):.4f}** ± {std:.4f}")
                elif "*" in styled:
                    num_str = styled.replace("*", "")
                    final_column_data.append(f"*{float(num_str):.4f}* ± {std:.4f}")
                else:
                    final_column_data.append(f"{mean:.4f} ± {std:.4f}")
            final_privacy_table[metric] = final_column_data

        print("\n🔒 Privacy Protection Results:")
        print(final_privacy_table.to_markdown(index=False))
    else:
        print(f"\n[!] Privacy results file not found: {privacy_filename}")

    # 4. 处理ADR结果
    adr_filename = f"{dataset_name}_adr_results.csv"
    df_adr = load_and_merge_results(adr_filename, results_dir)

    if df_adr is not None:
        df_adr["model"] = df_adr.iloc[:, 0].apply(extract_model_name)

        adr_metrics = ["ADR"]

        agg_funcs = {metric: ["mean", "std"] for metric in adr_metrics}
        adr_summary = df_adr.groupby("model").agg(agg_funcs).reset_index()
        adr_summary.columns = [
            "_".join(col).strip() for col in adr_summary.columns.values
        ]

        final_adr_table = pd.DataFrame()
        final_adr_table["Model"] = adr_summary["model_"]
        for metric in adr_metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            # 高亮 (ADR越低越好)
            styled_mean_col = highlight_best(
                adr_summary[mean_col], higher_is_better=False
            )

            # 构建最终的列，将高亮后的均值与标准差合并
            final_column_data = []
            for styled, mean, std in zip(
                styled_mean_col, adr_summary[mean_col], adr_summary[std_col]
            ):
                # 检查高亮标记，并相应地格式化字符串
                if "**" in styled:
                    num_str = styled.replace("**", "")
                    final_column_data.append(f"**{float(num_str):.4f}** ± {std:.4f}")
                elif "*" in styled:
                    num_str = styled.replace("*", "")
                    final_column_data.append(f"*{float(num_str):.4f}* ± {std:.4f}")
                else:
                    final_column_data.append(f"{mean:.4f} ± {std:.4f}")
            final_adr_table[metric] = final_column_data

        print("\n🛡️ Adversarial Disclosure Risk (ADR) Results:")
        print(final_adr_table.to_markdown(index=False))
    else:
        print(f"\n[!] ADR results file not found: {adr_filename}")


if __name__ == "__main__":
    # 使用方法示例：
    # 假设您的结果CSV文件位于项目根目录下的 'results' 文件夹中
    # 且您想处理 'syn' 数据集的结果

    # 获取当前脚本所在的目录，并假定'results'目录在项目根目录
    # CTabDM/eval/process_results.py -> CTabDM/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join("results")

    # 要处理的数据集列表
    datasets_to_process = [
        "adult",
        "barley",
        "ecoli70",
        "loan",
        "credit",
        "insurance",
        "king",
        "covertype",
        "pm25",
    ]  # 您可以添加更多, e.g., ['syn', 'adult', 'cancer']

    for dataset in datasets_to_process:
        process_and_display_results(dataset, results_dir=results_path)
        print("\n" + "=" * 50 + "\n")
