# V8 倾听数据处理代码导航

## 提速调度更新（2026-10-23 06:19 UTC）

用户要求最大限度提速。发现旧调度器对单次 CPU > 80% 的采样直接从每卡 4 进程降到 1，且不自动恢复；实际每卡 4 进程下 CPU 多数在 70%–80%，没有明显 I/O 等待，三机仍有其他任务占用资源。因此新增 [独立调度器](run_listen_v8_audio_governor.py)，只替换原 coordinator，保留原 node supervisor、worker、共享队列、已完成结果和切分/音质契约。

并发上限仍为每卡 4 个模型进程、每进程 2 路准备，即三机最多 96 个模型进程。先运行 4×1，预热后每台机器观察约 4 分钟的 pair 吞吐；资源有余量时试 4×2，若实测增益不足 5% 则恢复 4×1。随机来源不同，因此这是生产负载下的观测速率，不是相同输入的严格加速比。

CPU 约 60 秒窗口持续超过 92%，或连续约 30 秒达到 97%，才按“准备 2→1、模型 4→2→1”降档；单次 80% 尖峰不触发降档。约 2 分钟稳定余量并满足 3 分钟恢复冷却后允许 1→2→4。内存保护不等待 CPU 窗口：每卡空闲显存至少 16 GiB、系统可用内存至少 10%；持续严重内存压力 120 秒会排空本运行任务。不会停止其他任务。完整阈值以 [调度契约](../data_corpora/seamless_listen_v8_audio_final_full/cluster/governor/contract.json) 为准。

7 项针对性测试通过：[test_listen_v8_audio_governor.py](../tests/test_listen_v8_audio_governor.py)。[调度切换记录](../data_corpora/seamless_listen_v8_audio_final_full/cluster/governor/launch.json)、[分档吞吐实验](../data_corpora/seamless_listen_v8_audio_final_full/cluster/governor/trials.json)、[当前运行状态](../data_corpora/seamless_listen_v8_audio_final_full/runtime.json) 可追溯。恢复中断的全量任务时，先执行原入口的 `resume`，确认三机运行后再执行 `run_listen_v8_audio_governor.py handoff`，避免恢复到旧的单次采样降档逻辑。不得对已有 governor 重复 handoff；脚本会校验被替换进程的实际命令行。

## 当前全量：切分与音质、人脸只统计（2026-10-23）

用户已授权按已备份标准全量处理，并明确暂不做人脸区域裁剪筛选。新运行版本 `v8-audio-final-full-no-face-filter-1`；音质仍为 `v8-audio-severity-review-2`。范围沿用完整来源目录的 64,087 对及原身份隔离，不使用旧版裁剪通过名单作为输入。启动时间 2026-10-23 06:01 UTC；动态状态以 [runtime.json](../data_corpora/seamless_listen_v8_audio_final_full/runtime.json) 为准。

- [全量入口](run_listen_v8_audio_full.py)：独立契约、三机共享队列、每卡最多 4 进程/每进程 2 路准备；有资源压力自动降档，不升档或干预其他任务。按 pair 原子写结果，可恢复；模型按需加载。
- [无视觉门槛规划](plan_listen_v8_audio_full.py)：复用 V8 完整语句和已审查的 ASR 区间修复，移除旧裁框 448px、wide_crop、关键点有效率及方向准入。AV 不同步、缺少视频流/转写仍单列来源问题；缺少/异常关键点仅记为未测量。没有输出最终裁框或导出裁剪视频。
- [人脸统计](listen_v8_face_statistics.py)：使用原视频坐标中的面部关键点 23:91，置信度 > 0.4、每帧至少 45 个有效点，以 3%/97% 分位跨度度量宽高。这是面部关键点范围，不是旧头部裁框尺寸。统计原始宽高/短边像素、占原画面面积比例、片段内分位数、有效帧比例；无测量值单列，不记为小脸或淘汰。
- [规则测试](../tests/test_listen_v8_audio_full.py)：7 项通过，覆盖小脸/缺失人脸仍进入音质、像素/面积口径、缺失帧分母、汇总和缓存匹配。历史 84 段的音质池及原因与备份规则一致；正式试跑 20 段（13 复用、7 重检）成功。

输出根目录：`data_corpora/seamless_listen_v8_audio_final_full/`。`results/` 保存完整证据，`audit/` 保存所有切分与未决原因，`summaries/` 保存逐 pair 统计，`errors/` 保留处理错误。旧 62 对报错也在此次全量队列中重试。旧 QC4 结果及其冻结代码保持原样。

人脸分布：[face_size_distribution.json](../data_corpora/seamless_listen_v8_audio_final_full/face_size_distribution.json)；进度/各池时长：[screening_summary.json](../data_corpora/seamless_listen_v8_audio_final_full/screening_summary.json)。按 split / primary 或 onset_reserve / 音质池分组，再分 speaker/listener。像素分桶边界为 64、96、128、160、192、256、320、384、448、512、768、1024；这些仅是统计边界，不是准入阈值。分别报告片段数量、片段时长、有效帧分布、未测量数量和时长。按片段有效帧 p10 短边计算的各阈值覆盖时长只供比较，必须结合有效帧比例阅读，不能自动当作可用量。

统计约每 5 分钟更新；`final=false` 是进行中快照。结束后统一导出各池 JSONL 清单并更新最终统计；音质 `audio_review` 仍不放行，`keep_candidate` 只代表切分与音质候选，人脸裁剪阈值尚未选择。最终有错误时以 `error_pairs` 单列，不能归为音质淘汰。

复用音质证据必须双角色、原始来源、起止区间完全一致，且归一化音轨 SHA256 一致；人工作用范围和任务证据按新计划刷新。新增/变更区间运行原 QC3/QC4 检测。运行契约：[contract.json](../data_corpora/seamless_listen_v8_audio_final_full/contract.json)；历史样本一致性：[routing_regression.json](../data_corpora/seamless_listen_v8_audio_final_full/routing_regression.json)。

从项目根目录使用现有 Python 环境：`scripts/run_listen_v8_audio_full.py report` 更新统计；`report --export` 另导出当前清单；`verify` 校验运行契约。启动使用 `init` 后 `detach`，已在运行时勿重复启动。中断恢复使用 `resume`，它先检查三机无本运行的存活进程，再重置队列游标、保留结果并重试未完成/报错 pair。保留旧统计中的历史阶段状态，以下基线记录的“未全量切换”描述对应本次授权之前。

## 当前音质筛选基线（2026-10-23）

用户决定：“关于音质的筛选先这样，你记录好筛选规则和重要代码”。当前工作基线为 **`v8-audio-severity-review-2`**，后续裁剪讨论沿用此音质方案。阶段决定记录在 [checkpoint](../data_reviews/listener_v8_audio_policy_checkpoint_20261023.json)。这条决定没有新增逐条人工标签；最新 24 段的自动分组及既有人工反馈原样保留。

本版已完成小批重分流与展示，尚未接入旧 QC4 全量入口。`thresholds_validated=false`、`training_admitted=false` 保持原值：参数尚未经过独立大规模标注校准，候选不等于训练已准入。本次仅归档，未启动处理或调整裁剪。

### 筛选原则与分组

音质优先于轻微边界疑点。明显串音、严重底噪或突发杂音排除；轻微底噪、短暂低音量的弱匹配和孤立分类器提示可降为提示。单独的身份不确定提示不作为音质差的证据，但清晰的 listener 正式接话、实际 speaker 词越界、无效区间和历史人工拒绝仍有约束。

| 输出池 | 条件与用途 |
|---|---|
| `keep_candidate` | 无硬性音质问题、无待复核音质证据且通过原任务约束，或命中既有精确人工认可；保留软提示供追溯 |
| `audio_reject` | 命中硬性音质条件；保留来源与原因作为后续修复补充池，不删除原始数据。自动判定仍可能有误 |
| `audio_review` | 孤立匹配、环境声或旧排除证据无法明确解释；单独保留，暂不计入可用数据 |
| 原任务/人工 hold | 保留原路由结果，不将角色、区间或人工拒绝统一解释成音质差 |

实际顺序：既有人工决定 → 硬性音质排除 → 音质待复核 → 原任务约束 → 保留候选。人工负例按 listener 身份和区间重叠生效，优先于正例；正例只覆盖相同 sample ID、双角色身份及起止区间，不能扩展到整对来源或新切分。依据：[原约束](listen_v8_audio_first.py)、[最终路由](listen_v8_audio_final.py)。

### 当前阈值与证据

以下条件对应 `listen_v8_audio_final.py` 的 `POLICY`、`evidence()` 和 `route()`；实现中也有显式常量，不能只修改参数字典。`RMS` 和 `floor20` 使用现有分别归一化的音轨，不能当作原始录音的串音能量比；模型分数也不是校准概率。

| 检测项 | 当前条件 | 处理 |
|---|---|---|
| 重复延迟匹配 | 至少两个不重叠 listener 事件，各相关度 ≥ 0.30、时长 ≥ 0.12s、事件 RMS ≥ −35 dBFS；匹配窗口的两轨 RMS 均 > −45；开始时间相隔 ≥ 0.25s，延迟差 ≤ 10ms | 硬排除 `repeated_audible_delayed_match`；重叠事件不重复计数 |
| 明显局部匹配 | 相关度 ≥ 0.45、事件时长 ≥ 0.25s、事件 RMS ≥ −20，参考轨 RMS > −45 | 硬排除 `loud_local_speaker_match` |
| 既有强证据 | QC3 `strong_local_speaker_match` / `two_model_environmental_contamination`；QC4 `strong_reference_match` / `consistent_reference_leakage` | 硬排除 |
| 短暂弱局部匹配 | 已触发 `possible_local_speaker_match`，listener 事件时长 ≤ 0.25s 且 RMS ≤ −28 | 软提示；否则待复核 |
| 孤立参考匹配 | QC4 孤立提示，且存在局部相关度 ≥ 0.45 的事件，并且这些事件全部时长 ≤ 0.25s、RMS ≤ −28 | 软提示；否则待复核。没有局部证据也不能直接放行 |
| 高 listener 底噪 | listener `floor20_dbfs` > −40 | 硬排除；floor20 为 20ms RMS 窗的第 20 百分位 |
| 轻微平稳底噪/宽带提示 | listener floor20 ≤ −42、噪声分数 < 0.35，且窗口 RMS ≤ −35 或 Speech 分数 ≥ 0.60 | 软提示；其他情况待复核。低底噪不能单独豁免响亮噪声窗口 |
| 单独的 elevated floor 提示 | listener floor20 ≤ −42 | 软提示；较高但 ≤ −40 或 speaker 侧待复核，> −40 listener 已被硬拦截 |
| 孤立 speaker Music | speaker 背景中恰好一个 Music 环境提示窗口、Music 分数 < 0.35，human label 为 Speech 且分数 ≥ 0.60 | 软提示；不能推广为所有 Music 都保留 |
| 突发噪声 | 已触发 QC3 环境声提示且类别为点击、爆裂、敲击、刮擦、失真等 `TRANSIENT` 集合 | 硬排除；其他无法解释的环境预测待复核 |
| 未解释的旧证据 | 未被窗口解释的 QC3/QC4 原因，或旧 `audio_qc_status=exclude` 且没有硬证据 | 待复核，不因新规则未命中而自动放行 |

平稳类别集合：White noise、Pink noise、Static、Hum、Mains hum、Air conditioning、Mechanical fan。突发类别集合：Clicking、Clickety-clack、Crackle、Crack、Burst, pop、Bang、Knock、Tap、Scratch、Scrape、Distortion。事件和背景窗口均检查 speaker、listener 两轨；“轻微平稳底噪”的放宽只适用于 listener。

上游证据保持原实现：QC3 局部相关度 0.75 为强、0.45 为疑似；环境声使用 AST/PANNs 两模型和音量门限。QC4 使用 1s 窗、0.5s 步长、±1s 延迟搜索；宽带相关度 0.45、子带 0.55，至少两个子带在 25ms 内一致；强参考阈值 0.65，重复参考疑点按 25ms 延迟一致性判断。QC4 重叠滑窗不代表独立重复事件；本版额外使用上表的不重叠事件规则。完整上游条件以以下源文件及快照为准。

### 重要代码入口与依赖

| 文件 | 职责 |
|---|---|
| [listen_v8_audio_final.py](listen_v8_audio_final.py) | 本版唯一音质分级规则：`POLICY`、`evidence`、`route`；保留 hard/review/soft 原因、疑点窗口与延迟匹配证据 |
| [listen_v8_audio_first.py](listen_v8_audio_first.py) | 最终路由必需依赖：继承人工区间、角色和任务约束；其旧版“所有音质提示都拒绝”逻辑不代表当前规则 |
| [listen_audio_quality_v3.py](listen_audio_quality_v3.py)、[listen_audio_quality_v4.py](listen_audio_quality_v4.py) | 双模型噪声、背景底噪、局部匹配及参考串音扫描；输出原始证据 |
| [listen_v8_recovery.py](listen_v8_recovery.py)、[evaluate_listen_v8_recovery.py](evaluate_listen_v8_recovery.py) | ASR 区间误拦修复、边界分级、恢复区间与审计；可靠词时间及带 80ms 保护的词/VAD 零重叠是 ASR 包络豁免前提 |
| [prepare_listen_v8_audio_final.py](prepare_listen_v8_audio_final.py) | 合并人工反馈，复用同双角色、同区间的 QC 证据，重新分流和分层抽样；不重算 GPU 模型 |
| [publish_listen_v8_audio_final.py](publish_listen_v8_audio_final.py)、[页面模板](listen_v8_audio_final_review.html) | 4 路媒体导出，双轨/单轨、约 ±8s 上下文、疑点跳转和反馈导出；复用 recovery → v7 sample 导出器 |
| [check_listen_v8_audio_final_browser.py](check_listen_v8_audio_final_browser.py)、[针对性测试](../tests/test_listen_v8_audio_final.py) | 网页播放与反馈验证；串音、放宽、人工范围、原任务约束和输入证据不变的回归 |
| [listen_v8_release.py](listen_v8_release.py)、[run_listen_v8_qc4_full.py](run_listen_v8_qc4_full.py) | 历史 QC4 生产路由和入口；不是本版最终音质入口，未被本次归档修改 |

### 人工依据、验证与版本快照

人工反馈保留在 [原 QC4 反馈](../data_reviews/listener_v8_qc4_feedback.json)、[恢复小批反馈](../data_reviews/listener_v8_recovery_feedback_20261023.json)、[最近 5 条明确反馈](../data_reviews/listener_v8_audio_first_feedback_20261023.json)。没有把未逐条反馈的片段补写为人工通过。

重要回归：旧音质小批第 6 段的重复弱串音能自动拦截；18/22/23 自动保留，区别分别为短而弱的事件、speaker 单个低分 Music 提示、较低的平稳底噪。旧 19 自动证据仍待复核，只对人工认可的精确区间保留。恢复小批 33/34 的自动结果仍是待复核，人工负例继续拦截，不能声称所有已知坏例都已自动硬拒绝。详见 [对比证据](../experiments/p3_listen_v8_audio_first_20261023/feedback_analysis/comparison.json) 与 [人工/自动分列回归](../experiments/p3_listen_v8_audio_final_20261023/reviewed_regression.json)。

固定 1,024 对来源的 `train/primary` 分流：保留 2,725 段 / 5.1223h，排除 270 段 / 0.5718h，待复核 372 段 / 0.7650h；其他 split 和 onset reserve 单列。这不是全量产量估计。[统计](../experiments/p3_listen_v8_audio_final_20261023/summary.json)。最新 [24 段预览](http://127.0.0.1:8875/listen_v8_audio_final/) 为 12 保留、8 排除、4 待复核，来自此前未展示的 24 个 pair 和身份组，无人工覆盖。分层抽样比例不代表自然通过率。

已有验证：10 项针对性测试通过，24 × 3 = 72 组网页音轨播放通过；筛选、疑点跳转、上下文 Range、反馈导出及手机布局检查通过，无页面脚本错误。记录见 [sample_validation](../experiments/p3_listen_v8_audio_final_20261023/sample_validation.json)、[browser_validation](../data_previews/seamless_listen_v8_audio_final/browser_validation.json)。本次仅核对归档完整性，不将历史测试描述成本次重跑。

已保存 **35 个关键代码文件、14 个契约/反馈/验证证据文件**的逐字节快照及 SHA256：[recorded_baseline/manifest.json](../experiments/p3_listen_v8_audio_final_20261023/recorded_baseline/manifest.json)。本版契约的 3 个代码文件、旧生产契约的 23 个代码文件均核对一致；另含恢复、发布、网页与测试组件。该快照用于追溯，不包含全部传递依赖、模型、源媒体或 Python 环境。完整抽样契约保留在 [contract.json](../experiments/p3_listen_v8_audio_final_20261023/contract.json)。

### 复用方式

从项目根目录执行针对性测试：

```bash
OPENBLAS_NUM_THREADS=1 /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -m unittest discover -s tests -p test_listen_v8_audio_final.py -v
```

准备、发布入口分别为 `scripts/prepare_listen_v8_audio_final.py` 和 `scripts/publish_listen_v8_audio_final.py`，使用相同 Python 环境。**prepare 会根据运行当时的历史展示清单重新选样，不能用它重放原来的 24 段。** 精确回放应使用已归档的 `selection.json`、`review_selection.json` 和对应 results/媒体；新增实验使用新目录并检查证据缓存，修改区间后必须重新计算对应音质证据。

本轮没有降噪、串音分离、片段独立增益调整或裁剪修改。既有三机配置仍为每卡 4 个模型进程、每进程 2 路准备，自动升档关闭；本次未调整或启动调度。下一项讨论是裁剪框：合适人脸占比、向肩膀方向多取景、减少头顶空白。

## 历史阶段记录（以下状态与命令按各自日期理解）

**三机调度（2026-10-23 02:58 UTC）：** [run_listen_v8_cluster.py](run_listen_v8_cluster.py) 以共享队列复用原 QC4 处理函数，node17 / node12 / node4 各 8 卡，固定每卡 4 个模型进程、每进程 2 路准备，总计 96 个模型进程。自动升档关闭，每机模型上限 32、准备上限 64；共享状态文件短暂缺失/无效 JSON 有限重试，27 项相关回归通过。筛查全部节点结束后使用 [render_listen_v8_cluster.py](render_listen_v8_cluster.py) 三机导出正常候选，编码规则不变。`guard_listen_v8_cluster.py` 阻止旧单机入口混跑。质量契约不变；[配置](../configs/p3_listen_v8_cluster.json)、[运行与限制](../related/P3_V8_QC4_第二批音质验收.md)。下文为历史阶段记录。

**2026-10-23 02:08 UTC 最新：** 用户验收通过并指定音质补充池，已用 `run_listen_v8_qc4_full.py` 启动独立全量，8 卡 / 8 路准备，候选媒体 4 路导出。新目录 `data_corpora/seamless_listen_v8_qc4_full/`；`listen_v8_release.py` 实施人工区间与各池隔离。80 项测试、8 对正式入口试跑通过。见 [验收及运行记录](../related/P3_V8_QC4_第二批音质验收.md)。旧入口保持冻结，下文为历史记录。

**2026-10-23 当前状态：** 原 9 对报错已修复处理；8 卡、8 路 CPU 准备的小批 128 对完成（127 对结果、1 对响度未达标），全量等待人工验收。新增 [publish_listen_v8_repair_sample.py](publish_listen_v8_repair_sample.py) 导出 24 段复核媒体，仅用 4 路媒体任务；[页面模板](listen_v8_repair_review.html)。全量入口默认 8 个模型进程；媒体导出 `--render-workers` 默认 16，已取消写死的 128 并发。`review_hold.json` 会阻止未验收时启动无数量限制的全量。冻结契约修订和完整证据见 [运行说明](../related/P3_V8_全量处理与运行.md)。

当前状态：2026-10-22 用户认可 24 组对照中的整体切分效果，7 组无新版的片段暂不采用。科学处理代码保持验收时版本；这里只整理职责、依赖和运行方式。

**后续已获全量执行授权：** 新增 [run_listen_v8_full.py](run_listen_v8_full.py) 与 [render_listen_v8_full.py](render_listen_v8_full.py)，接入 7 条人工记录、独立运行契约、共享队列、8 卡/128 路准备、资源监控、清单和实际媒体导出。见 [全量处理与运行](../related/P3_V8_全量处理与运行.md)。下文的 24 组入口与其“尚无全量调度”描述仅对应最初对照阶段。

规则与结果：[V8 完整句子与声学边界修复](../related/P3_V8_完整句子与声学边界修复.md)。总入口：[项目 README](../README.md)。人工反馈：[listener_v8_boundary_feedback.json](../data_reviews/listener_v8_boundary_feedback.json)。版本证据：[reviewed_code_manifest.json](../experiments/p3_listen_v8_boundary_validation/reviewed_code_manifest.json)。

## 模块职责

| 模块 | 主要接口 | 职责 |
|---|---|---|
| [listen_turns_v8.py](listen_turns_v8.py) | `prepare`、`track_floors`、`full_turn_crop` | 原始词/段准备、组合反馈识别、话轮与裁框；作为库使用 |
| [listen_policy_v8.py](listen_policy_v8.py) | `POLICY`、`sentence_units`、`listener_compatibility`、`group_units`、`acoustic_cut` | 完整语句、listener 正式接话约束、时长组合、原始音轨声学切口 |
| [plan_listen_v8.py](plan_listen_v8.py) | `inspect_pair(item)`、`raw_rms(source)` | 从一对来源生成 clips / unresolved / floors / decisions；无全量命令行调度 |
| [preview_listen_v8.py](preview_listen_v8.py) | `plan` → `render` → `qc` → `publish` | 固定 24 组对照的唯一编排入口，阶段结果落盘 |
| [listen_v8_boundaries.html](listen_v8_boundaries.html) | 浏览器页面模板 | V7 / V8 / 连续上下文、声道、筛选与反馈导出 |
| [check_listen_v8_browser.py](check_listen_v8_browser.py) | 独立执行 | 最终页面播放、seek、声道、Range、笔记与窄屏验证 |
| [test_listen_policy_v8.py](../tests/test_listen_policy_v8.py) | unittest | 16 项语句、反馈及声学边界回归测试 |

`POLICY` 是规则参数说明；实现内还有显式边界常量，修改参数时需同步审查函数实现及测试，不能假定只改字典就会改变全部行为。

## 复用组件及边界

- `prepare_listen_300h.py` / `build_seamless_listening_preview.py`：只复用来源读取、媒体工具等基础能力，不执行旧版数据规划主入口。
- `listen_policy_v6.py`：复用方向检查、已知人工噪声区间匹配；`listen_policy_v7.py`：复用边界活动描述。V8 不调用 V7 的旧时间窗口组合逻辑。
- `publish_listen_v7_sample.py`：复用实际媒体导出、源文件校验、完整解码校验，输出目录由 V8 指定。
- `run_listen_v7_bulk.py`：复用 `verify_contract()` 与 `scan()` 做新时间区间质检，不启动 V7 全量规划。其 AST 来自 `listen_audio_quality.py`，PANNs 与音质判定来自 `listen_audio_quality_v3.py`。
- `src/p3_listen/subturns.py`：原整轨固定增益归一化实现。当前对照复用已有归一化音轨，不在每条片段上重新调增益。音轨实际路径以样本字段为准，可能跨旧产物目录。

V7 冻结文件、全量清单与人工反馈快照不修改。新人工记录单独保存，未来 V8 全量准入必须显式接入；不能只运行旧 V7 `scan()` 就认为这 7 条新反馈已被加载。排除范围是已展示片段区间，不是 pair 全部时间。

## 命令与环境

以下命令从 `/private/liqi_workspace/project_code` 执行。需要可访问原 TAR、已有 V7 catalog/归一化 WAV、AST/PANNs 模型缓存、FFmpeg、8 张 CUDA 卡及发布使用的字体。

```bash
# CPU 规划；24 组对照最多 24 个任务同时运行，默认并发上限 32。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 LISTEN_V8_CPU_WORKERS=32 \
  Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/preview_listen_v8.py plan

# 视频导出；新片段/上下文独立排队，最多 32 个媒体任务；可复用校验通过的缓存。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 LISTEN_V8_CPU_WORKERS=32 \
  Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/preview_listen_v8.py render

# 8 卡音质校验：物理卡 0–7，每卡一个进程、AST/PANNs batch size 16。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/preview_listen_v8.py qc

# 发布已完成结果；不会启动训练。
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/preview_listen_v8.py publish

# 边界回归测试。
OPENBLAS_NUM_THREADS=1 Omni-LTX-stage2/.venv/bin/python -m unittest discover \
  -s audio-driven-v5/tests -p test_listen_policy_v8.py -v

# 预览服务 8875 已启动时，验证最终页面。
LD_LIBRARY_PATH=/tmp/seamless_browser_libs/extracted/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/tmp/seamless_browser_deps \
  python audio-driven-v5/scripts/check_listen_v8_browser.py
```

`all` 按顺序执行四阶段，会重新运行 GPU 音频质检。以上是固定对照集入口，不是全量重处理入口；不要将其重复执行理解为扩大数据覆盖。

## 缓存、审计和后续修改

- `experiments/p3_listen_v8_boundary_validation/selection.json`：24 组及选择种子；`audit/`：逐对完整候选和失败原因；`planned.json`、`rendered.json`：阶段结果；`qc/`、`logs/qc_0.log` 至 `qc_7.log`：音质结果与 8 卡完成证据。
- `data_previews/seamless_listen_v8_boundaries/samples.json`：页面实际使用的样本、自动结论和策略；`media/`：真实导出的样本及上下文；`browser_validation.json`：65 个版本的播放验证。
- `reviewed_code_manifest.json` 保存验收时核心文件与直接复用组件的 SHA256，不是包含所有传递依赖、权重和数据的独立软件包。
- 规划缓存以三个 V8 科学处理模块的 SHA256 标识；如果改变复用依赖或人工规则，应创建新的实验目录或显式失效相关缓存，不能把旧结果直接当作新结果。
- 保留 `boundary_review_flags`、`audio_qc_status` 和人工意见三个维度；unresolved 原因、角色冲突与处理报错不能统一归为音质不合格。
- 后续全量处理还需要独立调度、人工反馈准入、失败重试、去重时长汇总及训练帧数适配。帧数适配不得再次硬截完整句子。当前整理没有启动这些任务。
