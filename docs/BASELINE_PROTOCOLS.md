# 基线 checkpoint 与生成器预处理核查

更新：2026-09-29。本表区分论文、官方代码和本仓库适配版。“生成器预处理”不是下游效用评估器的缩放；超参数搜索也不是同一次训练内的 epoch 选择。论文未写明时，仅描述官方代码中能核实的默认行为。

| 方法 | checkpoint | 生成器输入 |
| --- | --- | --- |
| CTGAN / TVAE | [官方 CTGAN](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/ctgan.py)、[TVAE](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py)按固定轮数训练后使用当前参数，未见验证集选 epoch。 | [官方变换器](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/data_transformer.py)：连续列 GMM 分量内归一化，分类列 one-hot；不是整表 MinMax/Standard。 |
| CTAB-GAN+ | [官方训练](https://github.com/Team-TUD/CTAB-GAN-Plus/blob/main/model/synthesizer/ctabgan_synthesizer.py)固定轮数后采样当前生成器。 | [官方变换器](https://github.com/Team-TUD/CTAB-GAN-Plus/blob/main/model/synthesizer/transformer.py)：一般连续列用 GMM 分量内缩放及分量 one-hot；特殊 `general_columns` 映到 [-1,1]，混合列有单独分支。 |
| TabDDPM | [论文](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf)以验证集 CatBoost 效用调超参数；[官方训练](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/train.py)结束时存 `model.pt` 和 EMA，[默认采样](https://github.com/yandex-research/tab-ddpm/blob/main/scripts/pipeline.py)读取最终 `model.pt`，非验证最优 epoch。 | 论文明确数值列 Gaussian quantile transformation，分类变量在扩散输入中 one-hot；[训练配置示例](https://github.com/yandex-research/tab-ddpm/blob/main/exp/churn2/config.toml)也采用 quantile，评估配置另设。 |
| TabSyn | [官方 VAE](https://github.com/amazon-science/tabsyn/blob/main/tabsyn/vae/main.py)用名为 `X_test` 的留出数组上的分类重建 CE 存最佳文件，但导出训练潜变量时使用内存中的最终模型；[官方扩散](https://github.com/amazon-science/tabsyn/blob/main/tabsyn/main.py)按最小训练损失保存， [采样](https://github.com/amazon-science/tabsyn/blob/main/tabsyn/sample.py)加载此文件。 | [官方预处理](https://github.com/amazon-science/tabsyn/blob/main/utils_train.py)数值列用 quantile；扩散阶段将潜变量中心化后除以 2。 |
| TabNAT | [官方训练](https://github.com/fangliancheng/TabNAT/blob/main/tabnat/main.py)虽记录最佳训练损失，默认 `model.pt` 仍在训练结束时保存。 | [官方预处理](https://github.com/fangliancheng/TabNAT/blob/main/utils_train.py)数值列用 quantile，主训练入口还做 `(x-mean)/std/2`。 |
| TabMT | [论文](https://papers.nips.cc/paper_files/paper/2023/file/90debc7cedb5cac83145fc8d18378dc5-Paper-Conference.pdf)未明确生成器的 epoch 选择；本仓库适配版记录训练 `best_loss`，但采样最终参数。 | 论文默认 K-Means 量化连续列，再对聚类中心比例做 min-max 构造 embedding，非直接缩放原始连续列；本仓库适配版先 quantile，再均匀分箱。 |
| TTVAE | [官方代码](https://github.com/coksvictoria/TTVAE/blob/main/ttvae/model.py)尝试按训练损失存最佳文件，但比较 epoch 均损失后把 `best_loss` 更新为最后一个 batch 的损失，不能视为严格最优规则。 | [官方变换器](https://github.com/coksvictoria/TTVAE/blob/main/ttvae/util.py)连续列用 GMM 分量归一化及 one-hot，分类列 one-hot。 |

复现时记录实际采用的文件、epoch、预处理器及其拟合数据。不能从 `best_loss` 日志推断采样文件：本仓库的 TabDDPM 适配版会在结束时覆盖中途保存的 `model.pt`，TabSyn 适配版扩散阶段也只记录最优训练损失而采样当前参数。生成器预处理仅在训练集拟合；可同时报告各方法原生协议与统一验证规则，但不从测试集挑 epoch。
