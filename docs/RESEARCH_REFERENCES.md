# TabDAT 文献借鉴与主张边界

更新：2026-09-29。本文档记录我们**如何使用**相关论文，而非仅列参考文献。下列联系是当前研究设计，除特别注明外，不代表 TabDAT 已在实验中验证相应结论。研究任务与状态见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

## 顺序、依赖与条件任务

| 论文（原文） | 可借鉴的内容 | 对 TabDAT 的用途与边界 |
| --- | --- | --- |
| [PAFT：*Are LLMs Naturally Good at Synthetic Tabular Data Generation?*](https://arxiv.org/abs/2406.14541)（后续版本题为 *Why LLMs Are Bad at Synthetic Table Generation (and what to do about it)*） | 从表中发现功能依赖，再利用列排列辅助 LLM 微调；关注逻辑约束。 | 支持“顺序值得研究”的动机，启发约束违反率及排序对照。它依赖文本化表格与 LLM；既不能证明 TabDAT 的 W 学到真实依赖，也不能把其收益直接归因于我们的机制。作为不同模型家族的参考／必要时基线。 |
| [MAC：*Training and Inference on Any-Order Autoregressive Models the Right Way*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/123fd8a56501194823c8e0dca00733df-Abstract-Conference.html) | 对任意子集的**边缘概率查询**，按规范顺序确定性地移除变量，将查询分解为较少的单变量条件项；按查询使用频率调整训练任务分布。 | 启发“训练任务应与实际推理查询匹配”这一研究视角。它不是“给定任意列后直接生成第一列未知变量”的算法；固定顺序 `prefix_next` 也不是 MAC 的特例实现。若只研究无条件合成，不必为了引用 MAC 新增条件补全策略。 |
| [DEformer：*The DEformer: An Order-Agnostic Distribution Estimating Transformer*](https://arxiv.org/abs/2106.06989) | 显式表示特征身份和值，使同一模型能处理多种顺序与条件任务。 | 任意顺序能力的架构对照；检查 TabDAT 的列身份表达与顺序切换的训练覆盖。不能用它的结果推断“固定顺序更好”或“W 是依赖图”。 |

MAC 的规范次序是全局列排列，不必是 DAG 的拓扑序。对子集 `E={A,C}`，若规范次序为 `A<B<C`，其分解可写为 `p(A,C)=p(A)p(C|A)`；全表 `E={A,B,C}` 则仍按 `A→B→C` 分解。这是“任意子集查询”，不等于整表的任意生成排列，也不直接训练 `p(A|C)` 这类以非前缀列为证据的前向补全条件项。完整 MAC 还要按目标查询的使用频率设计训练任务权重。

## 掩码训练与表格输出分布

| 论文（原文） | 可借鉴的内容 | 对 TabDAT 的用途与边界 |
| --- | --- | --- |
| [TabDAR：*Diffusion-nested Auto-Regressive Synthesis of Heterogeneous Tabular Data*](https://arxiv.org/abs/2410.21523) | 在任意顺序的掩码式 Transformer 中，对连续目标使用条件扩散；训练时先均匀抽掩码数量，再抽列。 | `uniform_count` 掩码与逐连续列扩散头的直接相关工作；应比较相同条件任务下的质量、计算和采样成本。TabDAT 加入类似条件扩散本身不是新颖主张，也不自动解决顺序问题。 |
| [TabNAT：*A Continuous-Discrete Joint Generative Framework for Tabular Data*](https://proceedings.mlr.press/v267/zhang25t.html) | 联合考虑连续和离散生成及灵活列顺序。 | 作为异构输出与顺序灵活性的重要对照；核对其训练目标和推理协议后再做细粒度架构比较，避免把“用了 diffusion”当成差异。 |
| [TabMT：*Generating tabular data with masked transformers*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/90debc7cedb5cac83145fc8d18378dc5-Abstract-Conference.html) | 掩码式 Transformer 生成、缺失字段处理及不同规模表格的评估。 | 掩码训练与缺失列填补的相关基线；具体训练／采样协议须逐项核对，不能笼统归为 TabDAT 的顺序对照。 |
| [TabDDPM：*Modelling Tabular Data with Diffusion Models*](https://proceedings.mlr.press/v202/kotelnikov23a.html) | 在异构表格数据上使用扩散生成。 | 作为 diffusion 类整体表格生成基线；不同于 TabDAT 的“Transformer 条件向量 + 单连续列输出头”，不能只比较损失数值。 |

## 原论文基线的 checkpoint 与生成器预处理核查

此处的“预处理”指**生成器输入**，不是下游效用模型或统计指标的特征缩放；“调超参数”也不是“从同一次训练中挑 epoch”。论文没写清的细节以官方代码辅助核查；本仓库的 `baselines/` 有适配改动，不能自动当作论文原始协议。

| 方法 | checkpoint 选择：可确认的证据 | 生成器输入：可确认的证据 |
| --- | --- | --- |
| CTGAN / TVAE | [官方 CTGAN](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/ctgan.py) 与 [TVAE](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py) 的 `fit` 按设定 epoch 训练，未见验证集选 epoch；随后使用当前参数。 | [官方 DataTransformer](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/data_transformer.py) 对连续列用 Bayesian GMM / mode-specific normalization，对分类列 one-hot；不是整表 MinMax/Standard。 |
| CTAB-GAN+ | [官方训练代码](https://github.com/Team-TUD/CTAB-GAN-Plus/blob/main/model/synthesizer/ctabgan_synthesizer.py) 固定轮数后以当前生成器采样，未见验证集选 epoch。 | [官方 transformer](https://github.com/Team-TUD/CTAB-GAN-Plus/blob/main/model/synthesizer/transformer.py) 一般连续列用 GMM 分量内缩放和分量 one-hot；`general_columns` 则映到 [-1,1]，混合列有单独分支。 |
| TabDDPM | [论文](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf) 用保留验证集上的 CatBoost 效用调**超参数**；[官方训练](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/train.py) 在训练结束保存 `model.pt`/EMA，[默认采样管道](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/pipeline.py) 读取最终 `model.pt`，非验证最优 epoch。 | [论文](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf) 明确数值列 Gaussian quantile transformation，分类列 one-hot；[示例训练配置](https://github.com/yandex-research/tab-ddpm/blob/main/exp/churn2/config.toml) 的 `normalization=quantile` 与评估配置不同。 |
| TabSyn | [官方 VAE](https://github.com/amazon-science/tabsyn/blob/main/tabsyn/vae/main.py) 用名为 `X_test` 的留出数组上的**分类重建 CE** 存最优 VAE；但训练结束导出潜变量时从内存中的最终模型导出，未重新载入该最优 VAE。[官方 diffusion](https://github.com/amazon-science/tabsyn/blob/main/tabsyn/main.py) 按最小**训练损失**覆盖保存 `model.pt`，[采样](https://github.com/amazon-science/tabsyn/blob/main/tabsyn/sample.py) 加载该文件。 | [官方预处理](https://github.com/amazon-science/tabsyn/blob/main/utils_train.py) 数值列用 quantile；第二阶段将潜变量逐维中心化后除以 2（不是除以标准差）。 |
| TabNAT | [官方训练](https://github.com/fangliancheng/TabNAT/blob/main/tabnat/main.py) 记录最佳训练损失，但只定期存档及在结束时保存默认 `model.pt`；未见按 `best_loss` 保存最优 epoch。 | [官方预处理](https://github.com/fangliancheng/TabNAT/blob/main/utils_train.py) 数值列用 quantile，主训练入口还做 `(x-mean)/std/2`。 |
| TabMT | [论文](https://papers.nips.cc/paper_files/paper/2023/file/90debc7cedb5cac83145fc8d18378dc5-Paper-Conference.pdf) 给出训练预算、超参数搜索及结果用的验证集，但未明确描述生成器 epoch 选择；本仓库实现记录训练 `best_loss`，却未据此保存，直接用最终参数采样。 | 论文默认先以 K-Means 量化连续列，再对**聚类中心比例**做 min-max 以构造有序 embedding；这不等于将原始连续数据简单 MinMax。**本仓库适配版不同**：先 quantile，再用 uniform `KBinsDiscretizer` 分箱。 |
| TTVAE | [官方代码](https://github.com/coksvictoria/TTVAE/blob/main/ttvae/model.py) 按训练损失下降保存 `model.pt`，但比较的是 epoch 均损失、更新 `best_loss` 时却用了最后一个 batch 的损失；因此不能笼统称为可靠的最优训练损失选择。本仓库采样入口会加载该文件。 | [官方 DataTransformer](https://github.com/coksvictoria/TTVAE/blob/main/ttvae/util.py) 与 CTGAN 类似：连续列 GMM 分量归一化加 one-hot，离散列 one-hot。 |

实验含义：不能仅因其它方法默认 `final` 就强迫 TabDAT 使用 `final`，也不能让 TabDAT 从测试集合成指标挑 epoch、而其它方法无同等机会。后续应先固定原始 train/validation/test、预算和候选 checkpoint，生成器预处理仅在训练集拟合；分别报告“官方／复现默认协议”与“统一验证规则”两个比较口径，并公开每个模型的实际采用文件与 epoch。本仓库 `baselines/models/tabddpm/train.py` 会在循环内按训练损失暂存 `model.pt`，但在结束时以最终参数覆盖；`baselines/models/tabsyn/main.py` 的 diffusion 同样只记录最优训练损失而以当前参数采样。复现实验前需要逐个审计入口，不能只看 `best_loss` 日志。

## 边界性参考与暂缓方向

| 论文（原文） | 可借鉴的内容 | 对 TabDAT 的用途与边界 |
| --- | --- | --- |
| [GReaT：*Language Models Are Realistic Tabular Data Generators*](https://openreview.net/pdf?id=cEygmQNOeI) | 将表格行序列化并用语言模型生成，可处理条件生成。 | 属 LLM 路线；参数量、预训练与表示方式不同。若论文声称宽泛的 SOTA，可增加代表性基线并报告资源成本；不把它作为顺序机制的同构对照。 |
| [CTSyn：*A Foundation Model for Cross Tabular Data Generation*](https://proceedings.iclr.cc/paper_files/paper/2025/hash/778055b859a3731bbe0d92fa61655e80-Abstract-Conference.html) | 以 schema 条件化的表格潜空间自编码与扩散进行跨表预训练。 | 帮助界定跨表／元数据路线；当前暂缓，不纳入单表顺序机制主线，也不把单表训练延长称为“预训练收益”。 |

## 从文献到可检验问题

1. **顺序是否真的降低有限容量模型的拟合难度？** PAFT 给动机，MAC 给条件任务分配视角，DEformer 给任意顺序架构参照。TabDAT 仍需在相同预算下比较顺序、逐条件误差和联合合成质量；同时报告“同模型换采样顺序”与“各顺序分别匹配训练”，不能混成一个因果结论。
2. **训练任务是否匹配采样协议？** 代码现有六种可选策略：`bernoulli_all`、`uniform_count_all`、`prefix_next`、`ordered_completion`、`canonical_subset`、`random_permutation_next`；实现不等于正式实验全部采用。前两者覆盖任意可见列集合；`prefix_next` 聚焦已定顺序的无条件生成路径；`random_permutation_next` 覆盖随机排列和步骤。给定部分列后按固定顺序跳过已知列属于尚未实现的条件采样规则，不依赖 `ordered_completion`；后者是条件填补路径的单目标训练对照。`canonical_subset` 只实现 MAC 启发的单条规范边采样，没有复现其查询频率权重或完整分解。仅前缀训练未必覆盖非前缀证据条件项。掩码收益与顺序收益须做交叉对照。
3. **连续输出头是否限制了顺序效应？** 对照 Gaussian、GMM、条件 DDPM，并与 TabDAR／TabNAT 的任务及成本明确区分。DDPM 的噪声预测 MSE 不是与 Gaussian NLL 可直接数值比较的似然。
4. **结论覆盖何种使用场景？** 无条件合成、给定部分列的条件合成／缺失列填补分别评估；逻辑约束违反率可借鉴 PAFT 的关注点，但不能代替统计质量或下游效用。

文献状态：PAFT、MAC、DEformer 已做初步讨论；其余为设计相关工作索引。实现或撰写论文前，应针对要借用的具体公式、实验协议和最新版本再核对原文。新增条目时记录：论文链接、借用的具体设计、不能借用的结论、拟开展的对照与阅读状态。
