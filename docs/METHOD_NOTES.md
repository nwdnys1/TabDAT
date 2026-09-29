# TabDAT 方法与实现备忘

本文件保存不必挤在[研究计划](RESEARCH_PLAN.md)中的实现细节。以下是代码现状和候选方案，不是质量结论。

## 掩码与顺序

- 已实现 `bernoulli_all`、`uniform_count_all`、`prefix_next`、`ordered_completion`、`canonical_subset`、`random_permutation_next`。输入中隐藏的列与参与损失的目标列是独立概念；目标绝不能作为可见输入。
- 默认 `fit()` 的独立掩码概率是 0.15，`model/main.py` 的示例显式传入 0.75；全掩码概率为 `p^D`。只有 `bernoulli_all` 使用 `mask_prob`，其他策略按各自任务分布抽样。
- `prefix_next`：训练前固定顺序，均匀抽其中一步，只用真实前缀预测当前一列。它适合“各顺序分别匹配训练”的对照，不能用训练完成后 W 导出的顺序倒称此前训练已匹配。
- `ordered_completion`：先均匀抽隐藏列数和集合，再只训练固定顺序中的首个未知列。它是条件填补任务分布的对照，不是实现填补采样接口的前提。
- `canonical_subset`：固定规范总次序，抽查询子集 E，仅训练 E 中最后一列给定其余 E 列的条件项。它只借鉴 MAC 的一类规范边，不复现 MAC 的完整查询分解与频率加权；规范次序也不必是 DAG 拓扑序。
- `random_permutation_next`：抽排列与步骤，训练该步骤的一列；用于任意整表顺序对照。若未来限制为 DAG 拓扑序，必须先定义并验证 DAG，当前软矩阵 W 不满足这一条件。
- 条件生成／缺失列填补还缺少“锁定已知列、跳过已知列、生成其余列”的采样接口。仅前缀训练可能没有学到非前缀证据下的预测；先评估已有任意子集掩码，再决定是否用 `ordered_completion`。

## 输出头与损失

- 连续头已支持默认 `gaussian`、可选 `gmm`、标量条件 `ddpm`；后者以条件噪声预测 MSE 训练，100 步 cosine 日程，采样时每列只运行一次 Transformer。分类头保持原状；DP 兼容入口只支持旧 Gaussian 路径。
- masked-only 损失当前按 batch 行数归一化，对每个被选中目标项给相同系数。逐行除以目标列数会降低“大掩码数”状态中的单个条件项权重；不能在比较掩码策略时顺便改动。后续再分别研究按目标项平均、分类／连续分别平均、逐列权重和辅助损失。
- Gaussian／GMM 的 NLL 与 DDPM 的去噪 MSE 量纲不同，不直接比较其数值，也不未经验证地相加。应同时记录逐列损失、各头对共享骨干的梯度尺度，以及独立的合成质量指标。

## checkpoint 监测与选择

仅在非 DP 的 `fit(validation_data=..., checkpoint_dir=...)` 中启用。验证行须先由训练集拟合的 `transform_holdout()` 转换；列顺序、缺失值、未见类别都严格检查。每个 run 使用新的目录，不覆盖旧 `model.pth`。默认每 50 epoch（及最终 epoch）测一次，固定种子选最多 4096 行、每行 4 个掩码任务；训练监测库与验证库分开。DDPM 的时间步与噪声也固定，评估不消耗训练 RNG。未启用时 `fit()` 的训练路径保持原状。

每次记录与当前掩码策略匹配的 train／validation masked-only 损失、逐列贡献与目标数、当时 W 导出的采样顺序。`best_train` 和 `best_val` 分别取固定监测任务上的最低损失，等分时保留较早 epoch；`final` 总保存最终参数。三个推理 checkpoint、`monitor_history.jsonl` 与 `selection_summary.json` 位于独立 run 目录。训练中不早停，`best_train` 不是逐 batch 在线训练日志的最低值。不同输出头或不同掩码／固定顺序的原始损失不能直接跨 run 排名；合成质量仍须另行验证。

`model/main.py` 的示例训练入口默认写入 `TabDAT/model/ckpt/holdout_v1/{dataset}/ckpt_v1_seed42/`。已有 `sample(dataset)` 仍读取旧的 `model.pth`；对新 run 显式调用 `sample(dataset, run_id="ckpt_v1_seed42", checkpoint_kind="best_val")` 等。运行任何真实数据实验前仍须与用户确认数据、预算和协议。

## 可选合成质量指标

`python eval/eval.py --datasets adult --extra --extra-max-rows 2000 --extra-seed 42` 会在常规统计／效用评估后运行额外指标；仅在对应真实划分和合成 CSV 已准备好、用户确认可运行时使用。`sdmetrics==0.21.0` 与 `synthcity==0.2.12` 顶层包已装入本机 `tabdat`，但其依赖尚不完整；目前导入分别缺 `copulas` 和 `optuna`，`pip check` 还显示 synthcity 与 Torch 2.7／NetworkX 3.4 的版本约束冲突。不要仅凭安装成功就启动正式指标计算；依赖问题解决后再做人工数据的真实库检查。

α-precision／β-recall 使用与 TabSyn 评估代码一致的 synthcity **naive** 版本；C2ST 使用 SDMetrics `LogisticDetection` 的“难以区分”得分，三者均越大越好。每对表先按固定种子从完整行中等量抽样，默认上限 2000 行，并写出实际 `Metric Rows`。正式比较时须预先固定每个数据集相同样本数、种子及参考真实划分；指标不能单独证明隐私或完整联合分布一致。

已通过人工数据的掩码、输出头、旧 checkpoint 兼容及新 checkpoint 选择接口检查；新指标仅完成模拟依赖的接口检查，待真实库可导入后再验证。尚无真实数据上的质量优劣结论。正式实验前仍须固定生成种子、样本数、顺序和报告指标。
