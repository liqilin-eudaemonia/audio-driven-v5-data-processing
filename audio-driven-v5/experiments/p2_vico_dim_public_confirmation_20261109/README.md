# DIM / ViCo 未确认项：进一步公开证据核查

2026-11-09 已检查官方仓库当前文件及首次发布历史、全部公开 issue 与回复、项目页、ECCV 正式补充材料、官方 EMOCA 模型包，并检查后续论文和有改动的 fork 是否提供原表执行证据。没有向作者发送消息，没有恢复暂停的十次生成队列。

**结论：公开评测规则已能确定；仍缺的是“与 DIM Table 1 绑定的实际数据底稿和运行配置”。不能把公开代码的默认行为直接写成作者已经确认的主表执行过程。**

[机器可读结论与文件 SHA](CONFIRMED.json) · [首次发布／当前版本对照](HISTORY_PARITY.json) · [本轮网络取证](FETCH.json) · [补充取证](FETCH_MORE.json) · [两版单次结果及两种 Var](PUBLIC_RULES_SINGLE_SAMPLE.csv)。前一轮[43 条／548 数值复算](../p2_vico_dim_alignment_recheck_20261109/CHECKED.json)最大误差为 0。

## 已从公开证据确定的规则

| 项目 | 确认结果 | 证据 |
|---|---|---|
| 测试划分 | `get_vico_dataloaders` 内部读取 `mode='test'`；字典键 `valid` 不表示另一个 validation 划分 | `dataset/data_loader.py` |
| 十次采样 | README 列出的 `test_s2s_pretrain.py` 实际调用 `evaluate_test_epoch`，生成 10 次；每段按完整 56 维 GT Fréchet 选一个候选，再统一报告指标 | `test_s2s_pretrain.py` → `x_engine_pt.py` |
| FD | pose6、exp50 分开，以原始系数计算逐片段 Fréchet，再对片段等权平均 | `mymetrics.py`、`metrics/eval_utils.py` |
| SID | 全测试 GT 拟合 KMeans；表情 40 类、姿态 20 类，所有预测帧合并统计直方图，再求 log2 熵 | `metrics/eval_utils.py:calcuate_sid` |
| 发布代码 Var | 先合并片段，再把帧和系数拉平计算 `np.var` | `mymetrics.py:print_metrics` |
| EMOCA 型号 | 预处理点名 `EMOCA_v2_lr_mse_20`；官方同名压缩包只有一个 checkpoint，我们已严格加载其编码器 | `vico_preprocessing.py`；官方模型包 |
| 时间索引 | 预训练评测入口使用 listener 的 t1…T−1 与 speaker 的 t0…T−2 | `x_engine_pt.py` |

以上关键文件自首次代码发布提交 `cfa6485d3d342d104a37f1ed3ac70586e38493c5`（2024-06-28）至当前版本，7 个相关文件的 AST 均一致。因此，十次采样、SID／Var 行为和加载器问题不是近期版本更新造成的。

历史对照保存了原始文件，而不只保存文字结论，见 [first_release/](first_release/)。

## 仍无法从查到的公开材料确定的两项

### 1. 原表实际使用的 GT 特征和有效帧名单

已确认论文和公开入口使用 ViCo 的 test 划分，但没有找到绑定 Table 1 的 GT EMOCA 文件、最终 listener ID／有效帧列表，以及相应裁脸和漏脸处理配置。官方仓库及 release 中没有提供这套底稿；项目页也没有相应下载链接。

作者仓库 [issue #1](https://github.com/Boese0601/Dyadic-Interaction-Modeling/issues/1)早在 2024-07-16 就询问 EMOCA／HuBERT 特征提取步骤，本轮 API 查询仍为 **0 条回复**。本轮检查全部 3 个 issue 的回复，作者唯一相关维护回复是补回 `models/lib`，没有给出评测配置。

我们维持官方 43 个独立 listener 和 16,238 个共同有效评分帧。公开加载器的 audio-key 覆盖和视频长度投影可得到 41 条记录／39 个 listener，但缺少作者实际 HuBERT／EMOCA 文件，不能把投影当成作者原表名单。

EMOCA 的模型名称已经确认；尚未确认的是作者当时对该模型使用的完整数据处理过程。不能把它概括为“提取器未知”。

### 2. 哪个评测分支与 Table 1 逐行对应

预训练模型评测入口的十次择优已经确定。但 README 同时列出 `test_s2s.py` 等其他入口，其中也有单次、基于 logits 的评价路径；没有找到把这些入口、checkpoint 和配置逐一绑定到 Table 1 各行的运行清单。

因此，当前可以准确写“按 DIM 发布的预训练评测入口使用十次 GT 择优”，不能进一步写“已经确认论文所有基线均使用这套十次规则”。同样，公开代码中的 SID／Var 汇总已经明确，但发表时是否完全采用该分支仍没有直接运行证据。

## rPCC 与真实 speaker 的处理

论文把 P-FD／rPCC 定义为真实 speaker 与 listener 的关系，但发布默认 `ViCoDataset` 将 speaker 动作替换为全一常量。常量的 Pearson 相关系数无定义，该路径不能直接解释原表有限的 rPCC。

我们继续使用真实 speaker 动作进行评分，依据是论文的指标定义。这个选择是有依据的复现决定，不能说成已取得作者对原表运行分支的确认。它不影响 FD。

## 已知差异的处理

- 当前模型结果仍为 seed137 单次生成；十次生成补测尚未完成，队列仍按此前要求暂停。确认了代码规则，不等于已经完成十次补测。
- 保留用户指定的逐系数 Var 作为当前展示，同时提供发布代码的拉平 Var；两版均可从现有特征得到，见 [CSV](PUBLIC_RULES_SINGLE_SAMPLE.csv)。没有改写原始封存成绩。
- 我们从生成 RGB 视频提取动作，DIM 原生模型直接预测 EMOCA 系数。这是测量路径差异，不能靠确认公式消除，也不能继续列成“FD 算法未确定”。

## 排除的材料

- 检查了 10 个 fork 的元数据，并查看其中有实际新增提交的分支：Seamless 51 维适配属于独立改造，不作为作者原表证据。
- 本地 `listening_head.zip` 中的 `inputs/test.csv` 有 30 条不透明 ID 记录，不是本轮的 43-listener 清单，未替换当前测试集。
- 同作者后续 DiTaiListener 论文有另外的模型、训练与评分设置，未将其细节直接当成 DIM Table 1 的执行证明。

## 继续执行时的配置依据

固定官方 test43、同名官方 EMOCA、原始 pose6＋exp50、DIM 发布指标函数、真实 speaker 和已披露的帧索引。SID、FD 公式及公开 Var 行为不再作为待研究卡点。按已固定队列完成十次补测后，将单次与十次择优分别报告；用户指定 Var 与发布代码 Var 保留清楚的标签。

剩余两项只限制“严格复现作者原表执行”的表述，不以作者回复作为继续计算的前置条件。
