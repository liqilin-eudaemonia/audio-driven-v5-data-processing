# P2 弱背景监督：GPU 验证与数值稳定性修复

> 最新修复（2026-10-25）：严格单卡/双卡一致性问题已解决。确定性反向＋FP64 梯度累加/求和后，105 项 CPU 测试、真实 GPU E0/E1 单/双卡各 7 步、恢复重放及独立生成全部通过；E0/E1 最终参数差均为 0，原阈值 2e-5 未放宽。详见[单卡/双卡数值修复与验证](P2_单卡双卡数值一致性修复与验证.md)。下文此前测试数量及失败结论为历史阶段记录；正式效果实验未启动。

日期：2026-10-25。用户授权在 `172.29.104.220` 后四张卡叠加验证；使用 GPU 4/5/6，不停止其他任务。**单卡工程验收通过；双卡训练可运行、事件头梯度等价通过，但严格单卡/双卡累计参数一致性未通过。没有将整轮标为全部通过。** 最终记录见[summary.json](../experiments/p2_listen_weak_background_gpu_fp32_20261025/summary.json)。

## 第一轮发现的问题

[第一轮目录](../experiments/p2_listen_weak_background_gpu_20261025/README.md)实际完成单卡 E0/E1 各 7 步、E1 第 5→7 步恢复、双卡 E1 7 步，以及新 E1 检查点的独立生成。所有训练损失有限，标签覆盖和 E0/E1 梯度路径符合约定。但以下数值验收未通过，不能据此宣布全部 GPU 测试通过：

1. [恢复重放](../experiments/p2_listen_weak_background_gpu_20261025/replay.json)：模型、优化器、RNG 状态精确恢复，更新后的模型/优化器误差均满足既定容差；事件 forward loss 却与保存记录相差约 `1.17e-5`，原脚本要求其完全相等。
2. [重复重放诊断](../experiments/p2_listen_weak_background_gpu_20261025/replay_diagnostic.json)：两次均从相同状态精确恢复，事件 loss 分别为 `0.3117147588`、`0.3117227557`，相差约 `8.0e-6`。这证明当前 GPU 计算存在浮点波动，不是简单的检查点状态没恢复。
3. [单/双卡对照](../experiments/p2_listen_weak_background_gpu_20261025/ddp/comparison.json)：相同初始化、样本序号与全局监督计数成立，首步事件 loss 差约 `1.99e-8`；7 步后最大参数差约 `4.43e-5`，超过预先设定的 `2e-5`，最大差位于视觉事件头。

第一轮结果完整保留，不覆盖为“通过”。

## 修复内容与不变的契约

[events.py](../src/p2_listen/events.py) 将小型事件头的时间聚合、LayerNorm/MLP 计算放在 FP32 中，显式关闭这一区域的 autocast。此前空间 scatter 聚合与 MLP 随主训练使用 BF16，对归约顺序和梯度数值较敏感；新计算保留到原始 U 的梯度，E0 仍 detach、E1 仍 joint。标签、七类结构、BCE、`pos_weight`、背景权重及有效 bin 归一化均不改。

[恢复检查脚本](../scripts/check_p2_listen_gpu_replay.py) 保留精确状态/RNG/有效监督计数检查；对浮点 forward loss 显式采用 `atol=1e-6, rtol=1e-5`，并同时比较第一次重放、第二次重放和原训练记录。模型/优化器更新容差仍是 `1e-6 / 1e-5`，单/双卡参数容差仍是 `2e-5`。原 BF16 约 `1.17e-5` 的事件 loss 偏差仍超过新 forward 容差，新增回归测试对此有明确检查。

新增[精度回归测试](../tests/test_p2_listen_precision.py)覆盖 autocast 下 FP32 输出一致性、E0/E1 梯度、微小舍入容差、旧漂移拒绝、监督计数严格一致、非有限值/过大差异及非法容差拒绝。**97 项 CPU 测试通过**，记录见[unit_tests.json](../experiments/p2_listen_weak_background_gpu_fp32_20261025/unit_tests.json)。

## 修复后复测

使用[独立复测目录](../experiments/p2_listen_weak_background_gpu_fp32_20261025/README.md)，从相同初始化重跑，未从旧精度配方恢复权重。仍为 50 条真实样本、global batch 8、seed 20261027、单/双卡各 7 步；不使用 `--debug-positive-events`。新标签和所有历史输入/缓存保留哈希核对。

共享机器其他任务在复测前增载，GPU 4–6 可用显存约 60–63 GiB；根据既有约 50.5 GiB 单卡实测，将本轮启动阈值由 65 GiB 调整为 60 GiB。只重启当时尚未启动任何 GPU 子任务的本轮调度器；没有终止其他任务。GPU 7 未使用。

## 最终结果与验收边界

| 检查 | 结果 |
|---|---|
| 单卡 E0/E1 | 各 7 步，覆盖全部 50 条；六类正/背景监督有效；E0 事件到 U 梯度为 0，E1 两模态梯度有效 |
| 断点恢复 | E1 第 5→7 步成功 |
| 相邻步重复重放 | 从第 6 步两次重放第 7 步：状态/RNG 精确恢复；base/event forward loss 与原记录**完全相等**；参数/优化器更新误差通过原容差，见[replay.json](../experiments/p2_listen_weak_background_gpu_fp32_20261025/replay.json) |
| 双卡完整训练 | E0、E1 各 7 步，NCCL、样本序号、全局有效监督计数正常；没有 OOM/非有限损失 |
| 固定状态 GPU 事件梯度等价 | 用真实弱标签/时间网格、固定随机 U 隔离事件目标，单/双卡 E0/E1 均通过；最大参数梯度差约 7.5e-9，见[组件检查](../experiments/p2_listen_weak_background_gpu_fp32_20261025/event_gpu_ddp_equivalence.json)。该检查不包含 AV renderer，不冒充完整训练等价证明 |
| 独立生成 | 512×512、25 FPS、51 帧；WAV 48 kHz、96048 样点，真实支持 2.001 秒；实际音视频解码/时长/有限值检查通过，禁止 GT 文件读取的审计通过，见[媒体核验](../experiments/p2_listen_weak_background_gpu_fp32_20261025/generated_source_only/media_validation.json) |
| **严格单/双卡累计参数一致性** | **未通过**：E1 最大差 5.76e-5，E0 对照最大差 8.64e-5，均超过原定 2e-5；[E1 对照](../experiments/p2_listen_weak_background_gpu_fp32_20261025/ddp/comparison.json)、[E0 对照](../experiments/p2_listen_weak_background_gpu_fp32_20261025/ddp/E0_control_comparison.json) |

E0 没有事件梯度进入 U，仍出现跨卡累计差异；固定状态事件目标的 GPU 梯度等价检查通过。现有证据支持将事件目标的归一化与完整混合精度训练的跨卡累计差异分开诊断，**不能据此宣称累计差异已全部解释或严格等价已验收**。没有放宽 2e-5 参数阈值、删除失败记录或修改标签来获得通过。

因此 `summary.json` 明确记录 `passed=false`、`functional_pipeline_passed=true`、`single_gpu_ready=true`、`strict_single_ddp_match=false`，并列出剩余检查。`job_status.json` 的任务进程成功仅表示命令正常退出，不代表所有数值断言通过。执行 `validate_results.py --report-only` 只是同时收集已通过项与未通过项，默认严格检查仍会拒绝这轮跨卡参数比较。

**首轮效果实验先使用匹配的单卡 E0/E1。** 若要把单/双卡作为严格可互换的训练方式，仍需继续定位完整 AV/feature 更新的跨卡累计数值差异，不能将本轮双卡结果当作已完成严格验收。原始标签、历史 S0 和既有缓存哈希保持不变；本轮 GPU 子任务均已结束，不停止其他任务。

以上结果只证明所列工程检查，不证明生成反应已更符合 speaker 内容与节奏；正式训练与效果评价未启动，后续 audio_swap 及盲评仍按[实验计划](P2_倾听事件监督实验计划.md)执行。
