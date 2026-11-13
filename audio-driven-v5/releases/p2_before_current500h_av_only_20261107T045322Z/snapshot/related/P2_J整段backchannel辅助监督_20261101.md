# J 整段 backchannel 多标签辅助监督

2026-11-01。已按用户确认实现：监督供给 LTX 的条件 hidden states U 包含整段行为信息，不要求精确事件时间。类别采用当前开放行为统计名称，不映射回旧七类。首轮 guided＋AV＋Event S1 已在四机32卡启动，进度见运行目录。

当前代码和八组训练配置已切换为最低正例数 5,000 的 **19 类（视觉 16／音频 3）**，保留全部训练样本。完整 2,726 类标签数据库继续保留，训练端按冻结词表选择这 19 类，不重新打标或重提缓存。类别依据见下方“类别收敛与数据保留讨论”，实现与验证见“19 类训练实现”。

## 模型与损失

### Connector：1024 维、4 层

已将 J 数据的八组配置（直接／guided × AV／AV＋Event × S1／S2）统一设为 `fusion_width=1024`、`fusion_depth=4`、`fusion_heads=16`。每个注意力头为 64 维，Connector 共 96,688,128 个参数。结构如下：

```text
Qwen 第 8／18／28／35 层 hidden（每层 2048 维）
  → 各层 LayerNorm → 共享 Linear(2048,1024) → 加层标识 → 拼接为 memory
外部视频时空查询＋音频时间查询
  → Fourier 坐标编码 → Linear(96,1024)＋模态嵌入
  → 4 × [Cross-Attention(memory) → Self-Attention(音视频查询联合) → FFN]
  → 视频输出投影 1024→4096→4096／音频输出投影 1024→2048→2048
  → 条件 U：追加到 LTX 文本上下文；同时供整段多标签分支池化
```

每层注意力沿用 pre-norm、残差连接、有效 token mask，FFN 中间维度为 4096。输出投影保留 GELU、RMSNorm 和最后线性层零初始化。多层 hidden 提取、外部查询网格、直接／guided 的 prompt 组织及整段多标签损失均沿用现有实现。S2 的 Qwen hidden 保留梯度。

训练结构单独写在实验配置的 `connector` 字段，`MatrixModel` 通过复制配置覆盖这三个参数实例化 Connector；缓存 manifest 中的 extractor 配置仍用于核验原缓存。已有 Qwen／Gemma／VAE 缓存无需重提。省略 `connector` 的历史实验继续沿用自己的原结构。严格恢复绑定完整实验配置；S1→S2 还显式核对 Connector 配置，防止注意力头数不同但参数形状相同的检查点混用。旧 512 维、2 层权重不能直接作为新结构的 S2 父检查点。

结构改动时的验证：46 项测试通过，覆盖配置隔离、八组梯度路径、probe 隔离及检查点迁移；从生产缓存读取 24 条样本的直接／guided 两种条件，共 48 次读取，完整 1024 维、4 层模型完成 4 次辅助损失反传和优化步，梯度非零且参数有限。验证使用 CPU，未运行 22B LTX 完整 GPU 训练。该阶段仍用 2,726 类，证据位于 `experiments/p2_connector1024_validation_20261101/`；后续已完成 19 类切换，验证见下文。

视频和音频 U 分别按有效 token mask 对整段平均池化，再经过 `LayerNorm → Linear(D,128) → GELU → Linear(128,C)`。每类独立计算 BCEWithLogitsLoss；同一段可以有多种或重叠行为，同类重复出现仍只监督“出现”，不监督时间和次数。

AV＋Event 的辅助损失回传到 Connector，第二阶段继续回传到 Qwen LoRA，不直接更新下游 LTX LoRA。AV 模式沿用 detach(U) 的诊断 probe，不影响生成链路。新配置明确设置 `event_supervision.mode=clip_multilabel`，旧配置保持七类时间格监督。

辅助头能够读出行为信息，不保证生成器已经执行该行为，仍需通过同条件生成对比验证。

## 标签与权重

`experiments/p2_j_clip_labels_20261101/` 覆盖冻结清单全部 **285,638 条 train**。SQLite 保留样本顺序、正类、不确定类、caption 和冻结样本指纹。原 caption、行为统计、音视频缓存及上游审核状态保持不变。

- 正类来自已有 `counted_types`，保留名称，不新增同义合并。词表仅从 train 正类建立，共 **2,625 个视觉、101 个音频名称**，属于开放候选行为名称，不等于人工确认的 backchannel 类别。
- 正类目标 1、权重 1；未提及类别为弱负例，目标 0、默认权重 **0.1**，由 `prepare --absence-weight` 设置。这是训练假设，不是确认不存在。
- 待核、不可用或抽取异常的记录不提供省略类别的弱负例，保留可用正类。非 counted/baseline 证据对应类别屏蔽。否定语句可能只描述局部时段，不能升级成整段确定负例；正类与此类证据冲突也屏蔽，共 **4,849 个样本-类别冲突**。
- BCE 按每类有效样本数归一化，再平均有效类别；分母不使用权重总和，避免纯弱负例批次抵消 0.1 权重。正例权重由 train 加权负例/正例数计算并限制在 `[1,10]`。无正负对比的类别禁用。本批 2,726 类均通过这个基础检查，但它没有最低正例数量要求，不代表所有类别都具有充分学习依据；后续收敛讨论见下节。
- 总损失为 `L_AV + 0.05 * L_clip`。类别顺序、标签及数据库哈希进入训练身份；恢复与 S1→S2 迁移会检查。

## 类别收敛与数据保留讨论

记录日期：2026-11-01。用户希望辅助监督采用十几至二十种有足够正例支持的行为；继续保留开放名称，不强行映射回旧七类。已按确认方案采用最低正例数 **5,000**，得到 **19 类（16 个视觉、3 个音频）**，训练代码和配置已落实，首轮 S1 已启动。

### 长尾分布与门槛比较

统计来自 `experiments/p2_j_clip_labels_20261101/manifest.json` 的 `statistics[*].positive`，是既有派生规则处理后采用的正类样本数。全量有 825,413 条“样本—类别”正例记录；各类正例数中位数仅 2。1,111 类只有 1 条正例，675 类有 2～4 条，469 类有 5～19 条，193 类有 20～49 条，278 类至少 50 条。

因此，loss 下降不等于稀有类学会了；多数类别可能主要学习预测低概率，少量正例也可能被记忆。已通过的代码测试和梯度验证不能证明长尾类别可泛化。讨论中依次考虑了 278 类、81 类及当前 19 类方案，前两者不作为最终已执行选择。

| 最低正例数 | 类别数 | 视觉／音频 | 正类记录覆盖率 |
|---|---:|---:|---:|
| 50 | 278 | 267／11 | 98.38% |
| 200 | 137 | 131／6 | 96.74% |
| 500 | 81 | 75／6 | 94.66% |
| 1,000 | 55 | 50／5 | 92.38% |
| 2,000 | 32 | 27／5 | 88.52% |
| 3,000 | 23 | 19／4 | 85.79% |
| 4,000 | 22 | 18／4 | 85.32% |
| **5,000** | **19** | **16／3** | **83.63%** |
| 8,000 | 17 | 14／3 | 82.01% |
| 10,000 | 14 | 11／3 | 78.93% |
| 15,000 | 11 | 9／2 | 74.47% |
| 20,000 | 9 | 8／1 | 70.06% |

覆盖率分母为全部 825,413 条正类记录，不是片段、时长或人工确认行为的覆盖率。

### 19 类训练清单

| 原类别键 | 中文说明 | 正例数 |
|---|---|---:|
| `visual:head nodding` | 点头 | 191,270 |
| `visual:smiling` | 微笑 | 106,925 |
| `visual:gaze shifting` | 视线转移 | 72,375 |
| `visual:blinking` | 眨眼 | 54,641 |
| `audio:speech` | 说话 | 41,041 |
| `visual:mouth opening` | 张嘴 | 39,558 |
| `visual:head tilting` | 歪头 | 25,966 |
| `visual:mouth movement` | 嘴部运动 | 23,701 |
| `visual:eye closing` | 闭眼 | 22,785 |
| `visual:hand raising` | 抬手 | 18,574 |
| `audio:vocalization` | 发声 | 17,844 |
| `audio:audible laughter` | 可听见的笑声 | 13,372 |
| `visual:eyebrow raising` | 抬眉 | 13,316 |
| `visual:eye opening` | 睁眼 | 10,139 |
| `visual:lip pursing` | 抿嘴 | 8,580 |
| `visual:hand lowering` | 放下手 | 8,406 |
| `visual:head turning` | 转头 | 8,402 |
| `visual:frowning` | 皱眉 | 6,791 |
| `visual:mouth closing` | 闭嘴 | 6,580 |

这些是回应相关的动作与发声候选，不都等价于具有交流意义的 backchannel；`speech` 与 `vocalization` 等可以重叠，多标签形式允许这种情况。每类至少 6,580 个正例，数量足以支持首轮学习，但标签质量、人物／会话多样性仍决定有效监督量。

guided 分支学习从包含 P 的条件中保留行为信息；direct 分支只能从 speaker 音频与参考图预测可能的回应，真实行为存在一对多的不确定性。分类头表现良好不代表 LTX 已执行这些行为。主判断仍是同条件音视频生成效果，辅助诊断观察独立验证集的逐类正例召回与 PR-AUC，不只看训练 loss 或总体准确率。

### 全部训练样本保留

用户认可：限制辅助监督类别，不据此删减训练样本。完整 **285,638 条、约 500h** 清单继续使用，原 caption、其他类别记录、音视频目标和 split 保持不变。根据 `labels.sqlite` 每行的 `positive` 与上述 19 类求交，实际构成为：

| 情况 | 样本数 | 占比 |
|---|---:|---:|
| 至少含一个入选的 19 类正例 | 268,334 | 93.94% |
| 只有其他类别的正例 | 5,081 | 1.78% |
| 没有当前采用的正类标签 | 12,223 | 4.28% |

后两类合计 17,304 条，占 6.06%。它们仍参与 AV 训练：其他类别包含 humming、exhalation 等候选；真实安静倾听片段可提供姿态、细微运动和静音监督；标注缺失或待核的样本仍可能有有效生成目标。没有采用的正类标签不能解释为全程静止、静音或无回应。

辅助分支只使用有效信息：已采用正类为 1；未提及类别沿用权重 0.1 的弱负例假设；不确定、待核或冲突类别 mask 掉。不能把所有“无 19 类正例”样本升级为确定的全负例。数据是否删除应由画质、音轨或配对质量决定。

切换到 19 类不需要重新打标、人工补时间区间或重提 Qwen／Gemma／VAE 缓存。代码已冻结类别顺序，按所选类别读取标签并保留原有效 mask，配置、损失统计和检查点身份同步使用 19 类。完整 2,726 类标签保留。直接／guided、有／无辅助监督的实验共用完整数据清单。

### 19 类训练实现

`labels.select_classes` 只使用完整标签 manifest 中 train 的已采用正例数，按 `positive >= min_positive` 筛选，并沿用原词表顺序。每组实验明确记录 `event_supervision.min_positive=5000` 及 19 个完整名称；视频头输出 16 个 logit、音频头输出 3 个 logit，不增加 `other` 类。`emit --min-positive 5000` 为默认生成方式。

数据读取仍遍历原 SQLite 的全部行，不按所选正例过滤。仅生成这 19 类的目标与 mask；预检查只输出相应 19 类的统计、正例权重和启用状态，损失、跨卡计数、日志沿用同一词表。权重仍根据全部 train 样本统计，而不是只看含所选正例的样本。类别名称、顺序和阈值随实验身份进入检查点；旧 2,726 类实验不能直接恢复为 19 类，S1→S2 必须使用同一监督配置。省略筛选字段的历史完整词表配置仍可读取原数据库。

验证证据位于 `experiments/p2_clip19_validation_20261101/`：

- `selection_audit.json`：数据库 SHA 与原 manifest 一致，逐行核对全部 285,638 条及样本顺序；268,334 条含所选正例，5,081 条只有其他正例，12,223 条无已采用正例。19 类覆盖 690,266／825,413 条正类记录，即 83.6267%。
- `tests.log`：55 项测试通过，含阈值边界、类别身份、19 类统计、弱负例／未知 mask、无所选正例时的 AV 梯度、两种条件 S1／S2 梯度和检查点验证。
- `real_cache_check.json`：真实 train 前 24 条、两种条件共 48 次读取，确认完整训练集仍为 285,638 条、每条仅有 19 类目标、辅助头为 16＋3。1024 维／4 层 Connector 完成 4 次辅助反传与优化，参数有限、梯度非零。这 24 条也包含无正类及只有其他类别正例的样本。

验证在 CPU 执行，未启动真实 22B LTX 完整训练。

## 实现与使用

### 首轮已确定：guided＋AV＋Event，S1 四机 32 卡

2026-11-01 启动记录：缓存全量验收已成功。原单进程 preflight 造成 GPU 等待，已改为64个CPU进程执行相同的逐样本检查，285,638条耗时69.65秒通过；24条真实输入串并行结果一致，3项回归含损坏标记阻断测试通过。证据见运行目录 `queue/preflight_parallel_fix/`。32卡真实训练已经产生优化步；前16步验证及整轮续跑状态以队列文件为准。

用户已确认先训练该单组 S1，预算为完整训练集 1 轮。配置为 `configs/p2_j_clip_events_20261101/guided_av_event_s1.json`，输出目录为 `experiments/p2_j_guided_av_event_s1_32gpu_20261101/`。global batch 为 32，每卡 1 条、无额外梯度累积，预算 `ceil(285638/32)=8927` 步。共 285,664 个样本位置，末尾 26 个位置按现有确定性采样器从下一轮补齐；保留完整片段，不按所选事件类别删样本。

S1 冻结 Qwen，读取 guided hidden、Gemma 文本和 AV latent 缓存；训练 1024 维／4 层 Connector、19 类整段辅助头和 LTX 文本注意力 LoRA。沿用 BF16、全层重计算、FP64 梯度累加／汇总、每 100 步保存和 TensorBoard。其余实验组与 S2 本轮不自动启动。

`scripts/p2_j_s1_queue.py` 提供本轮自动接续：等待缓存 `COMPLETE.json` 通过、四机缓存 launcher 全部退出后，运行训练 preflight；再以完整 32 卡执行前 16 步，核验损失有限、Connector／Event／文本 LoRA 梯度非零及检查点身份；随后严格恢复第 16 步检查点，继续完成第 8,927 步。前 16 步计入正式预算。任何检查失败都会记录错误并停止接续，不会跳过失败检查继续训练。

实际状态看运行目录的 `queue/state.json`，初始训练验证看 `queue/INITIAL_TRAINING_VERIFIED.json`，训练进度看 `status.json` 与 `metrics.jsonl`。`plan.json` 为排队时的静态计划，不代表训练已启动。停止请求可写运行目录的 `STOP`，训练会在步边界保存退出。

### 32 卡时长对齐实测与接续

2026-11-01，按用户确认实测加速方案。参考既有384窗口／24卡方案，本轮使用512个样本位置（16个global batch）的窗口；窗口内按完整片段时长分组，再打乱batch顺序。模型结构、音视频输入、损失、global batch和8,927步预算不变。仅调整第100步之后的样本顺序；原3,200个已消费位置保留，全部285,638条样本和285,664个样本位置的成员、原随机ordinal均一致。批次组合改变会改变优化轨迹，不要求其权重与随机组批相同。

四组均从原第100步的同一检查点开始，使用相同的后续512个样本位置，每组四机32卡、16步。按随机A1→对齐B1→对齐B2→随机A2执行；B2在第8步保存退出后恢复。计时为训练日志中的每步耗时，包含B2恢复后的首步，不包含模型加载和检查点保存。

| 对照 | 平均秒/步 |
| --- | ---: |
| 随机 A1 | 6.6960 |
| 对齐 B1，连续运行 | 3.9509 |
| 对齐 B2，中途恢复 | 3.9707 |
| 随机 A2 | 6.7007 |

随机均值6.6983秒/步，对齐均值3.9608秒/步，**步时降低40.87%，吞吐提高约69.12%**。B1与B2最终模型、优化器和32个rank的随机状态全部逐位一致，达到事先设定的至少5%提速及恢复一致性门槛，自动采用。该结果是同512个位置的短测，后续速度以正式日志为准；不据此判断音视频生成质量。

正式续跑目录改为 `experiments/p2_j_guided_av_event_s1_32gpu_duration512_20261101/`，配置为旧运行目录下 `performance/duration512/selected_duration.json`。原第100步模型、优化器和随机状态原样迁移，仅增加采样计划及其源码身份、修改输出目录；四组测试更新均丢弃，不计入正式预算。旧目录的 `DURATION_REPLACEMENT.json` 指向新目录，旧 `queue/state.json` 保留为 `stopped`；当前接续状态应查看 `performance/duration512/state.json`，实际进度查看新目录的 `status.json`、`metrics.jsonl`。新训练仍每100步保存，可通过新输出目录的 `STOP` 请求在步边界保存退出。不要重新运行旧队列脚本从前16步开始。

北京时间2026-11-01 16:28检查：新正式训练已实际完成第108步／8,927步，32卡身份正确，损失有限且Connector、Event、文本LoRA梯度非零，后台继续运行。

代码为 `scripts/p2_j_duration_execution.py`（采样及源码身份）、`scripts/p2_j_duration_train.py`（接入原训练循环）、`scripts/p2_j_duration_benchmark.py`（ABBA、恢复验证及自动接续）。没有修改已冻结的缓存或原训练核心源码。12项采样相关测试通过。证据位于原运行目录的 `performance/duration512/`：`RESULT.json`、`membership_audit.json`、`ORIGINAL_SAVED.json`、`tests.log`，以及各组的 `metrics.jsonl`、`migration.json` 和检查点。

准备验证：5 项队列测试通过；四机实际 32 个 CPU rank 的 Gloo 通信、all-reduce 和 rank 映射通过，证据为运行目录 `cpu32_connectivity/RESULT.json`。该通信测试未占用 GPU；真实 NCCL／LTX 多卡训练由上述前 16 步验证，不能以 CPU 通信通过替代。

`src/p2_clip_events/model.py` 实现池化和损失，`labels.py` 派生标签，`backend.py` 接入新缓存、预检查与配置。`scripts/p2_clip_events.py` 提供统一入口；训练循环复用 `p2_matrix16`，增加动态类别统计和跨卡归一化。

`configs/p2_j_clip_events_20261101/` 已生成 fixed/guided × AV/AV＋Event 的 S1、S2 配置。fixed 是直接生成，guided 是同一原 caption 同时进入 Qwen 和 Gemma；首轮 `guided_av_event_s1` 已设 8,927 步，其余配置预算仍为空。现有 Qwen、Gemma、VAE 缓存直接复用；本轮训练配置与队列准备未重启后台提取。S2 在线重算 Qwen hidden，只复用文本与 AV latent。

标签与配置已经生成，无需重复 prepare/emit。复现时使用新的输出目录：

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
$PY scripts/p2_clip_events.py prepare \
  --cache experiments/p2_j_train500h_cache_20261101 \
  --output experiments/NEW_CLIP_LABELS --processes 16 --absence-weight 0.1
$PY scripts/p2_clip_events.py emit \
  --labels experiments/NEW_CLIP_LABELS --output configs/NEW_CLIP_CONFIGS --min-positive 5000
```

全量缓存的 `COMPLETE.json` 验收成功后，才允许正式全量训练前检查：

```bash
CONFIG=configs/p2_j_clip_events_20261101/guided_av_event_s1.json
$PY scripts/p2_clip_events.py preflight --config "$CONFIG"
# 确定预算并设置 TRAIN_STEPS 后启动；本轮未执行。
torchrun --nproc_per_node=8 scripts/p2_clip_events.py train \
  --config "$CONFIG" --steps "$TRAIN_STEPS" --global-batch 32
```

多机使用同一入口及 torchrun rendezvous 参数。严格恢复传 `--resume`；S2 选择同文本、同损失配置，传 `--parent` 指向 S1 检查点。各配置的 `output` 指定检查点目录。推理沿用 `p2_matrix16.inference.restore(..., backend=p2_clip_events.backend)`，guided 的文本来源仍须标明预测或目标标注。

## 验证

以下为初版完整 2,726 类实现的历史验证，证据目录为 `experiments/p2_clip_event_validation_20261101/`；当前 19 类验证见上方“19 类训练实现”：

- `tests.log`：**45 项 CPU 测试通过**，覆盖池化 mask、多正类、不确定类、弱负例权重、跨 rank 归一化、S1/S2 梯度、旧训练兼容、检查点恢复和类别顺序检查。
- `real_cache_check.json`：正式 train 清单前 **24 条已完成缓存**，直接/双通路共 **48 次读取**；正式 **2,726 类**配置、生产规格 Connector 完成两种条件各两次辅助反传/更新，梯度非零、参数有限。
- `online_reader_check.json`：两种条件 S2 真实读取通过，布局与缓存一致；未读冻结 Qwen hidden，guided 在线 P 与 Gemma P 相同。
- 全量标签 `manifest.json` 保存覆盖数、词表、权重统计和源身份。

真实缓存验证在 CPU 执行，小模型回归覆盖原生 Qwen/LTX 梯度路径。本轮没有运行真实 22B LTX 新训练，也没有生成质量结论。
