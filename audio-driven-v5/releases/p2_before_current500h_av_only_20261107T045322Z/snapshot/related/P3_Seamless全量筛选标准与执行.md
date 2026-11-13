# P3 / Bernini：Seamless 全量筛选标准与执行

更新：2026-10-22。本文件保留 V6/QC3 严格基线的历史标准。该基线剩余进程已停止，新规则已落实为 [V7 原始数据全量筛选](P3_V7_全量切分与448裁框筛选.md)：2–10 秒自然切分、448 原裁框、开头反馈备用池，以及完整来源重建。现行输出为 `data_corpora/seamless_listen_v7_full/`。下文“正在运行”等状态均指旧基线启动时。

**后续诊断已发现时间规则过严。** 见 [QC3 淘汰归因与裁框分布](P3_QC3_淘汰归因与裁框分布.md)：固定 18,880 对会话中，裁框合格的 106.90 小时父话轮仅留下 17.51 小时；静音凑帧、跨话轮完整句及 4 秒下限造成额外损失。本文仍准确记录正在运行的严格基线，不代表这些条件已适合作为最终数据标准。裁框门槛和切分规则尚未因诊断而更改。

用户已认可切分边界，完成 QC3 试听并明确保留 5 段、排除 4 段，现要求整理标准后对原始 Seamless 全量处理，统计实际可用时长。**启动全量筛选；不在本轮自动启动训练。** 音质模型仍可能误判，保留项是候选，待复核项隔离。未承诺筛选后一定有 300 小时。

## 1. 数据范围与来源隔离

- 原始索引：`/private/liqi_workspace/data/pipeline/seamless/outputs_demo100/pair_index_all.json`，共 64,619 对会话。原始音视频及标注来自 `/public/0_DATA/2_Avatar/seamless_interaction/` 的 TAR。
- 按 session 和 participant 构建连通组，保留官方 train / dev 中不跨集合的完整组，防止同一身份或会话跨训练/开发/测试集合泄漏。
- 本轮实际筛查全部 **30,615 对 train + 645 对 dev = 31,260 对**，不受旧 300h 选取名单限制，不达到 300h 就提前结束。
- 其余 33,359 对在来源隔离阶段已剔除：train/dev/test 混合组 30,672；dev/test 431；dev/train 1,688；test/train 23；纯 test 499；跨子集配对 46。它们不是本轮音质失败样本，不纳入训练可用时长。测试数据不用于规则调优或训练。
- 复用的是已建立的来源分组目录和经 SHA256 校验的原始整轨响度归一化音频；**所有切分、画面与音质资格重新检查**，不复用旧训练窗口或旧特征缓存。

## 2. 时间边界与话轮标准

1. 以 speaker 的转写词时间与话轮为基础，保留中间停顿；listener 在 speaker 范围外的 VAD 不延长片段。
2. 优先完整句末，其次自然停顿（当前阈值 ≥0.65 秒，并避开明显未完成的功能词结尾）。语义无法安全切开的长句舍弃，不按 10 秒直接切断。
3. 当前片段 **4–9.96 秒、25 fps、8k+1 帧、最多 249 帧**。短片段可在同一有效话轮内合并完整语义单元，不跨过被排除的区间。
4. 帧数对齐只允许在边界真实静音中延伸不足 0.32 秒，不能裁掉正在说的词、复制帧或补虚构音频。speaker 与 listener 音轨、listener 视频使用同一源时间区间，最终等长；不再二次滑窗截短。
5. speaker 与 listener 同时出声本身不构成排除依据。首尾重叠的短回应按 speaker 范围保留；为避免从短反应中间切开，找不到安全语义/帧边界的候选可以舍弃，不反向延长话轮。
6. 排除转写支持的 listener 实质发言/抢话；短附和、笑声、鼻音可保留。无转写、VAD 密集或较长的未知活动仅送音频复核，不直接认定为杂音或真实回应。

## 3. 画面标准

- 原始 listener 父话轮固定裁框边长 ≥512 像素；不得放大裁框或先插值来凑门槛。合格源裁框缩放到 512×512 输出。
- 排除宽泛上半身裁框（沿用 `wide_crop` 判定），保持头部/肩部画面；裁框沿整个父话轮计算，子片段继承固定裁框。
- 原始像素方向错误、带非零旋转元数据、关键点方向不确定均舍弃；不旋转修复后冒充合格原视频。双方原视频都检查。
- 检查原视频帧数、关键点数量与独立 WAV 时长的一致性。处理错误单独记账，不当作合格或音质拒绝。

## 4. 音量与音质标准

### 音量处理

沿用试听时的整条原始 participant 音轨归一化：目标 −18 LUFS、峰值限制约 −1 dBTP；统一增益作用于该来源所有话轮，不逐片段强拉静音。使用原始独立 WAV，输出 48 kHz 单声道 PCM；补偿限幅器延迟，核对样本数和同步。此轮不增加降噪、门控静音或声音重建。

**音质检查在归一化后的音频上进行**，因此被放大的源噪声也要接受检查。响度达标不等于音质合格。

### 三类分流

| 结果 | 标准与后续处理 |
|---|---|
| 保留候选 keep | 时间、画面、归一化通过，双方音轨均未触发污染/不确定提示；或同一来源同一时间片段获得明确人工保留。区分“有人声证据”和“安静/弱活动”统计，避免把大量沉默当成回应数据。 |
| 排除 exclude | 人工确认的噪声/串音源区间；或 QC3 的强局部串音/双模型环境声证据。自动排除保留证据与媒体来源供抽检，不删除原始文件，也不把模型判断当人工标签。 |
| 待复核 review | 可能串音、明显宽带背景声、反应身份不明确、模型意见不一致等。暂不进入训练候选合格清单；保留以便恢复正常笑声/附和，不能整组永久丢弃。 |

实现使用 **AST AudioSet + PANNs Cnn14**、声学特征和双轨局部波形匹配。主要冻结阈值：

- 双轨 200–3500 Hz 带通，±150 ms 时延搜索；相关 ≥0.75 且达到可听电平，建议排除；0.45–0.75 待复核。不是检测“同时说话”，而是检测局部声音高度相似。
- 局部可听事件 PANNs 环境声分 ≥0.45，且 AST top-8 中环境声分 ≥0.20：建议排除；仅 PANNs ≥0.20：复核。
- 人声证据：两模型均 ≥0.20，或 PANNs ≥0.60 且 AST ≥0.10。人声高分不能抵消同时存在的污染证据。
- 整轨以 2 秒窗口、1 秒步长检查到结尾，覆盖 VAD 间隙；宽带噪声、背景电平等提示进入复核。
- 没有转写、频谱集中、短促突变，不能单独硬删；“嗯”“yeah”、笑声不要求逐词转写。

这些模型分数未经准确率校准，现有试听支持使用它们做全量分流，不代表所有自动 keep 已经人工确认。

## 5. 人工反馈如何应用

- [`listener_qc3_feedback.json`](../data_reviews/listener_qc3_feedback.json)：本轮 5 段保留、4 段排除。
- [`listener_noise_rejections.json`](../data_reviews/listener_noise_rejections.json)：所有已确认污染的 listener 源时间区间。重切后只要相交仍排除，不能靠更换样本 ID 绕过。
- 人工保留只覆盖同 pair、同 listener、同起止时刻的完整片段，不能扩展到整个会话或所有相似声音；排除区间优先于保留。
- 自动判断、人工判断、最终分流分别记录。保留正常回应的人工决定不伪装成模型自动通过，也不转成逐事件训练标签。
- 本轮运行冻结反馈快照，运行中变更名单需新版本重筛，避免前后规则不一致。

## 6. 全量处理步骤及代码

| 步骤 | 代码 | 输入 / 输出 |
|---|---|---|
| 原始来源、连通组隔离 | `scripts/prepare_listen_300h.py::catalog` | 原始 pair index → 来源目录；不调用其中旧版切窗、冻结或训练流程 |
| 语义切分、方向与原裁框检查 | `scripts/preview_listen_v6.py::inspect_pair`；`scripts/listen_policy_v6.py`；`scripts/build_seamless_rule_turns.py` | 原始标注、视频、关键点 → 每对会话候选与拒绝原因；全量调用使用 `defer_audio_uncertainty=True` |
| 整轨音量归一化 | `src/p3_listen/subturns.py::normalize` | 原始 WAV → 48 kHz PCM 和增益/响度/峰值记录；只复用归一化函数，不使用旧切分函数 |
| AST 事件证据 | `scripts/listen_audio_quality.py::assess` | VAD、词和归一化音频 → 局部特征、模型分数；QC1 的旧硬拒绝结论不用于准入 |
| 串音、环境声、全背景检查 | `scripts/listen_audio_quality_v3.py::scan_track` | AST 证据＋PANNs＋成对音轨 → 双方音质判断 |
| 全量编排、反馈合并、断点与统计 | [`scripts/run_listen_qc3_bulk.py`](../scripts/run_listen_qc3_bulk.py) | 全部合格来源目录 → keep/review/exclude、逐对结果、错误、各 rank 进度和时长汇总 |
| 每分钟累计时长报告 | [`scripts/report_listen_qc3_bulk.py`](../scripts/report_listen_qc3_bulk.py) | 只读逐对结果，增量更新 `screening_summary.json`，分开统计边界拒绝原因与边界种类；不修改冻结的筛选代码 |
| 回归验证 | `tests/test_listen_policy_v6.py`、`tests/test_listen_audio_quality_v3.py`、`tests/test_qc3_bulk.py` | 边界、方向、噪声证据、保留/排除范围与优先级 |

独立输出目录：`data_corpora/seamless_listen_qc3_full/`。运行契约 `contract.json` 冻结代码、目录、反馈、模型来源和参数。使用三台机器各 8 个 GPU worker，共 24 路；每个 worker 一个 GPU，AST/PANNs batch 各 16，CPU 预取下一对会话。该任务不运行 DDP 训练。

```bash
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/run_listen_qc3_bulk.py init
# 分别在 node17 / node12 / node16 后台执行，node 为 0 / 1 / 2：
Omni-LTX-stage2/.venv/bin/python -u audio-driven-v5/scripts/run_listen_qc3_bulk.py launch-node --node 0
# 汇总当前已经完成的结果；三机结束后也会自动生成：
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/run_listen_qc3_bulk.py summary
```

该入口只筛选，没有缓存/训练命令。每对会话原子写入结果；进程失败的源记录单独落盘，可重启同节点跳过已完成项。连续 10 对处理错误停止该 worker，避免系统性失败被当成数据淘汰。旧 `BOUNDARY_REVIEW_HOLD.json` 文件继续阻止旧编排误启动，文件内容更新为新全量筛查已授权、训练仍未放行。

## 7. 时长统计和最终验收

- `results/<pair>.json`：候选、原始区间、裁框、归一化来源、自动检查与人工决定。
- `audit/<pair>.json`：边界、画面、实质插话、已知噪声等前置筛查。
- `errors/<pair>.json`：I/O、推理等处理失败；不得计入完成时长。
- `progress/rank_XX.json`、`logs/rank_XX.log`、`node_N.json`：24 路实际进度、PID、运行状态。
- **`screening_summary.json`：最新累计报告，每分钟更新**。按 train/dev × keep/review/exclude 的片段数和时长，另列含人声证据的片段数、来源隔离数、前置拒绝原因和未完成/失败数量。独立报告进程的 PID 在 `report_pid.json`，日志在 `logs/report.log`。
- `summary.json`：编排器自带的按需/结束汇总；时长口径相同，其旧 `geometry_boundary_rejection_reasons` 字段会混入句末/停顿等边界种类。分析淘汰原因以独立报告的 `screening_summary.json` 为准，该报告优先读取真实 `rejection` 字段。报告修正未改变任何筛查、候选或模型决策。

**主时长口径为每段 speaker 起止跨度（含内部停顿），排除帧对齐额外静音；同对话区间不能重叠计数。** 同时另报实际媒体时长。不能把双方轨道相加、把重叠窗口重复计时，或把 review 加入可训练时长。自动 keep 的汇总称“保留候选时长”，全量结束并完成媒体导出/同步验证、按来源分组抽检后，才能称“最终合格时长”。

后续根据实际产量决定是否满足目标 300h。249 帧正式训练配置与新缓存仍须独立适配校验，旧 201 帧缓存不能直接继承。

## 8. 执行记录

- 2026-10-22 07:13:13 UTC：已启动 node17 / node12 / node16 三个独立后台 supervisor，各 8 个 GPU worker，共 24 路。进程记录在 `launch_0.json`～`launch_2.json`、`node_0.json`～`node_2.json`；终端退出不停止任务。
- 启动前通过 19 项测试：V6 边界/方向 10 项、QC3 音质 6 项、人工准入范围与优先级 3 项。
- 实际模型回归：`S0847_I00000064` 正常回应仍为自动 keep；`S0065_I00000126` 串音仍为自动 exclude，与原试听结果一致。
- 从原始 WAV 重新处理 S0847 的双方整轨音频，两条归一化输出 SHA256 与试听版本完全一致。验证记录：`experiments/p3_listen_qc3_review/bulk_smoke/validation.json`。
- 07:14:06 UTC 首批汇总：完成 437 / 31,260 对，处理错误 0；train 保留候选 73 段、0.141788 小时（约 8.5 分钟），待复核 111 段、0.223934 小时，自动排除 1 段、0.002234 小时。其余拒绝发生在边界/画面等前置阶段；首批尚未覆盖全部 dev 来源。这是启动快照，不是最终产量或全局通过率。
- 07:17:30 UTC：完成 3,455 对，train 保留候选 748 段、约 1.47 小时；dev 保留候选 9 段、约 1.08 分钟。24 个 worker 均已进入实际筛查。随后检查到少量响度未达目标/峰值超过容差的失败，已记录到 `errors/` 并隔离，不计入候选合格时长；没有为增加产量而放宽阈值。

最终时长以 `screening_summary.json` 为准，不能从小样本比例直接推算。`progress/` 实时记录增量，独立报告每分钟累计；三机退出后自动生成最终报告。单次读取汇总命令仍可使用上面的 `summary`，两者读取的候选与时长一致。
