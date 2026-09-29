# TabDAT 文献借鉴与主张边界

更新：2026-09-29。本文档记录我们**如何使用**相关论文，而非仅列参考文献。下列联系是当前研究设计，除特别注明外，不代表 TabDAT 已在实验中验证相应结论。研究任务与状态见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

## 顺序、依赖与条件任务

| 论文（原文） | 可借鉴的内容 | 对 TabDAT 的用途与边界 |
| --- | --- | --- |
| [PAFT：*Are LLMs Naturally Good at Synthetic Tabular Data Generation?*](https://arxiv.org/abs/2406.14541)（后续版本题为 *Why LLMs Are Bad at Synthetic Table Generation (and what to do about it)*） | 从表中发现功能依赖，再利用列排列辅助 LLM 微调；关注逻辑约束。 | 支持“顺序值得研究”的动机，启发约束违反率及排序对照。它依赖文本化表格与 LLM；既不能证明 TabDAT 的 W 学到真实依赖，也不能把其收益直接归因于我们的机制。作为不同模型家族的参考／必要时基线。 |
| [MAC：*Training and Inference on Any-Order Autoregressive Models the Right Way*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/123fd8a56501194823c8e0dca00733df-Abstract-Conference.html) | 根据查询与确定性的分解规则，减少冗余的单变量条件任务，并让训练任务分布贴近推理。 | 启发“先定义采样协议，再设计输入 mask、目标 mask 与任务权重”。仅固定前缀是其思想的一个受限特例；任意证据下按固定顺序补全也只是 MAC 风格协议，不等于复现其完整方法。需另证其对 TabDAT 有效。 |
| [DEformer：*The DEformer: An Order-Agnostic Distribution Estimating Transformer*](https://arxiv.org/abs/2106.06989) | 显式表示特征身份和值，使同一模型能处理多种顺序与条件任务。 | 任意顺序能力的架构对照；检查 TabDAT 的列身份表达与顺序切换的训练覆盖。不能用它的结果推断“固定顺序更好”或“W 是依赖图”。 |

## 掩码训练与表格输出分布

| 论文（原文） | 可借鉴的内容 | 对 TabDAT 的用途与边界 |
| --- | --- | --- |
| [TabDAR：*Diffusion-nested Auto-Regressive Synthesis of Heterogeneous Tabular Data*](https://arxiv.org/abs/2410.21523) | 在任意顺序的掩码式 Transformer 中，对连续目标使用条件扩散；训练时先均匀抽掩码数量，再抽列。 | `uniform_count` 掩码与逐连续列扩散头的直接相关工作；应比较相同条件任务下的质量、计算和采样成本。TabDAT 加入类似条件扩散本身不是新颖主张，也不自动解决顺序问题。 |
| [TabNAT：*A Continuous-Discrete Joint Generative Framework for Tabular Data*](https://proceedings.mlr.press/v267/zhang25t.html) | 联合考虑连续和离散生成及灵活列顺序。 | 作为异构输出与顺序灵活性的重要对照；核对其训练目标和推理协议后再做细粒度架构比较，避免把“用了 diffusion”当成差异。 |
| [TabMT：*Generating tabular data with masked transformers*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/90debc7cedb5cac83145fc8d18378dc5-Abstract-Conference.html) | 掩码式 Transformer 生成、缺失字段处理及不同规模表格的评估。 | 掩码训练与缺失列填补的相关基线；具体训练／采样协议须逐项核对，不能笼统归为 TabDAT 的顺序对照。 |
| [TabDDPM：*Modelling Tabular Data with Diffusion Models*](https://proceedings.mlr.press/v202/kotelnikov23a.html) | 在异构表格数据上使用扩散生成。 | 作为 diffusion 类整体表格生成基线；不同于 TabDAT 的“Transformer 条件向量 + 单连续列输出头”，不能只比较损失数值。 |

## 边界性参考与暂缓方向

| 论文（原文） | 可借鉴的内容 | 对 TabDAT 的用途与边界 |
| --- | --- | --- |
| [GReaT：*Language Models Are Realistic Tabular Data Generators*](https://openreview.net/pdf?id=cEygmQNOeI) | 将表格行序列化并用语言模型生成，可处理条件生成。 | 属 LLM 路线；参数量、预训练与表示方式不同。若论文声称宽泛的 SOTA，可增加代表性基线并报告资源成本；不把它作为顺序机制的同构对照。 |
| [CTSyn：*A Foundation Model for Cross Tabular Data Generation*](https://proceedings.iclr.cc/paper_files/paper/2025/hash/778055b859a3731bbe0d92fa61655e80-Abstract-Conference.html) | 以 schema 条件化的表格潜空间自编码与扩散进行跨表预训练。 | 帮助界定跨表／元数据路线；当前暂缓，不纳入单表顺序机制主线，也不把单表训练延长称为“预训练收益”。 |

## 从文献到可检验问题

1. **顺序是否真的降低有限容量模型的拟合难度？** PAFT 给动机，MAC 给条件任务分配视角，DEformer 给任意顺序架构参照。TabDAT 仍需在相同预算下比较顺序、逐条件误差和联合合成质量；同时报告“同模型换采样顺序”与“各顺序分别匹配训练”，不能混成一个因果结论。
2. **训练任务是否匹配采样协议？** 比较当前 `bernoulli_all`、均匀掩码数量的 `uniform_count_all`、无条件固定顺序的 `prefix_next`，以及任意已知列下的 `ordered_completion`。`prefix_next` 随机选一步，只用真实前缀预测这一列；`ordered_completion` 随机选已知列集合，按固定顺序预测第一列未知变量，推理再逐步补齐。后者借鉴 MAC“确定查询后规定唯一分解”的思想；单纯固定前缀自回归不能算 MAC 方法。两者训练时均不需要用生成值完整滚动后缀。输入 mask 定义可见上下文，目标 mask 指定本步预测列；掩码策略的收益可能来自训练／推理匹配或任务权重，而非顺序本身，须做交叉对照。
3. **连续输出头是否限制了顺序效应？** 对照 Gaussian、GMM、条件 DDPM，并与 TabDAR／TabNAT 的任务及成本明确区分。DDPM 的噪声预测 MSE 不是与 Gaussian NLL 可直接数值比较的似然。
4. **结论覆盖何种使用场景？** 无条件合成、给定部分列的条件合成／缺失列填补分别评估；逻辑约束违反率可借鉴 PAFT 的关注点，但不能代替统计质量或下游效用。

文献状态：PAFT、MAC、DEformer 已做初步讨论；其余为设计相关工作索引。实现或撰写论文前，应针对要借用的具体公式、实验协议和最新版本再核对原文。新增条目时记录：论文链接、借用的具体设计、不能借用的结论、拟开展的对照与阅读状态。
