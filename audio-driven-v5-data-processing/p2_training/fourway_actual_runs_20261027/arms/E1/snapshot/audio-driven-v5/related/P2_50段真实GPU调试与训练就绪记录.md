# P2 七类事件监督：50 段真实 GPU 调试与训练就绪记录

日期：2026-10-25。运行机器：`172.29.104.220`，NVIDIA H200；使用检查时空闲的 GPU 4、5、6。真实模型缓存、单卡 E0/E1、双卡 E1、断点恢复和独立生成均已完成。本轮验收范围内没有未处理的运行错误；正式训练和效果验证尚未开始。

入口：[总索引](../README.md)、[完整运行说明](P2_倾听事件监督实现与运行.md)。证据目录：[p2_listen_gpu_debug50_20261025](../experiments/p2_listen_gpu_debug50_20261025/)，机器可读汇总：[summary.json](../experiments/p2_listen_gpu_debug50_20261025/summary.json)。

## 1. 数据和模型

用户提供的 [50 段数据](../experiments/listen_debug50_fullraw_20261025/) 共 **213.0865 秒**，真实片段长度 **2.001–7.872 秒**，全部属于 train。未拆出伪 dev/test，未修改媒体、原文候选或输入标签。

使用 `observations.json`、`targets.debug.json`、`labels.debug-positive.jsonl`。整合候选共 257 条；训练网格按现有 coverage/unknown 规则生成，不能直接把候选条数当有效监督 bin 数。

| 类别 | 已知正例 bin | 已知负例 bin |
|---|---:|---:|
| head_nod | 30 | 0 |
| head_shake | 0 | 0 |
| smile | 244 | 0 |
| laugh | 51 | 0 |
| frown | 37 | 0 |
| surprise | 23 | 0 |
| listener_vocalization | 147 | 0 |

使用实际训练基座 Qwen2.5-Omni-3B、LTX-2.3-22B 和 Gemma 文本编码器，模型配置、资产身份和代码指纹保存在各 run 的 `run.json`；没有用小模型或合成特征替代真实 S0。环境见 [environment.json](../experiments/p2_listen_gpu_debug50_20261025/environment.json)。

## 2. 本轮代码修正

原正式统计策略会禁用没有负例支持的类别。这批只有正例的 debug 标签若直接沿用该策略，所有事件监督都会被关闭，无法验证实际事件反向路径。

新增 `train --s0 --debug-positive-events`：只对明确标为 debug 的标签启用已知正例，`pos_weight=1`；unknown 保持屏蔽，摇头继续无监督。开关、原始统计、实际启用类别均写入 run/checkpoint 身份，恢复时必须一致。它只用于检查代码，不产生可用于判断实验优劣的指标。

同时增加正式训练保护：标签带 `debug_only=true` 或 `formal_training_admitted=false` 时，不允许进入非 S0 训练。已通过实际 CLI 拒绝检查，没有创建正式 run。没有删除 debug 标记或伪造负例。

相关文件：

- [training.py](../src/p2_listen/training.py)：S0 监督策略与正式训练保护。
- [p2_listen.py](../scripts/p2_listen.py)：显式 debug 参数。
- [test_p2_listen_gpu_guards.py](../tests/test_p2_listen_gpu_guards.py)：4 项针对性测试。
- [check_p2_listen_gpu_replay.py](../scripts/check_p2_listen_gpu_replay.py)：真实 GPU 恢复重放、状态精确检查、重复更新及数值容差检查。
- [check_p2_listen_source_only.py](../scripts/check_p2_listen_source_only.py)：生成时启用 Python 文件读取审计，检查目标/标签/目标缓存隔离；这是回归检查，不是操作系统沙箱。
- [check_p2_listen_code.py](../scripts/check_p2_listen_code.py)：支持独立 `--output`，保留历史验证记录。

## 3. 实际验收结果

| 检查 | 结果与证据 |
|---|---|
| 媒体与 split 导入 | 50/50 通过；最终再次核验媒体与身份，[data_audit_final.json](../experiments/p2_listen_gpu_debug50_20261025/data_audit_final.json) |
| 五类真实缓存 | input/teacher/vae/events 各 50 份，固定 text 1 份；身份、SHA256、张量有限值和标签统计全部通过，[cache_validation.json](../experiments/p2_listen_gpu_debug50_20261025/cache_validation.json) |
| 全尺寸单样本 S0 | E1 5 步；AV 到两个 U 的梯度非零，feature 路径在零初始化开始更新后非零，已知事件路径非零；冻结 renderer 未更新 |
| 完整 50 段 E0/E1 | 相同 seed `20261026`、global batch 8，各 7 步，覆盖全部样本；E0 事件梯度不进入 U，E1 接入；各步 loss/梯度有限 |
| 实际 CLI 恢复 | E1 按 7 步预算在第 5 步停止，从 `step_000005.pt` 恢复并完成第 6、7 步；日志 ordinal 连续，无重复或跳样 |
| 下一步重放 | 单样本第 4 步恢复后重放第 5 步；载入模型/优化器/RNG 状态精确一致，forward loss 与 RNG 一致；参数最大误差 `2.11e-7`、优化器最大误差 `2.90e-6`，分别低于 `1e-6` / `1e-5` 容差，[replay_validated.json](../experiments/p2_listen_gpu_debug50_20261025/smoke/replay_validated.json) |
| 真实双卡 NCCL | E1 7 步，global batch 8；单/双卡初始化、样本 ordinal、全局逐类 bin 计数一致，各 rank 梯度范数一致；首步事件 loss 差为 0，7 步后参数最大差 `1.65e-5 < 2e-5`，[comparison.json](../experiments/p2_listen_gpu_debug50_20261025/ddp/comparison.json) |
| 独立生成 | 2.167 秒和 7.872 秒两个片段，均实际执行 30 步采样、VAE 解码、MP4 与独立 WAV 导出；文件读取审计未触发目标/标签访问 |
| 生成媒体核验 | 两份 MP4 均完整解码，WAV 有限且长度准确；最长片段输出 197 帧、48 kHz 音频 377,856 samples，原生 201 帧对齐输出另行保留 |
| CPU 回归 | **72 项通过，0 失败、0 错误、0 跳过**，[unit_tests.json](../experiments/p2_listen_gpu_debug50_20261025/unit_tests.json) |
| 正式训练隔离与原始输入 | debug 正式训练被预期拒绝，[formal_guard_validation.json](../experiments/p2_listen_gpu_debug50_20261025/formal_guard_validation.json)；四份输入清单哈希不变，[input_integrity_final.json](../experiments/p2_listen_gpu_debug50_20261025/input_integrity_final.json) |

恢复检查的边界：GPU 混合精度更新不保证逐比特一致。首次逐比特比较发现差异后，增加了“同一状态重复更新”对照；该对照也存在 `2.10e-7` 参数、`3.66e-6` 优化器差异，RNG 和 forward loss 不变。因此报告采用明确的数值容差，不宣称 GPU 位级复现。重复重放脚本还修正了 AdamW CPU step 张量引用导致测试基准被首轮更新修改的问题；修正后的双次重放通过。早期诊断记录保留，最终结论以 `replay_validated.json` 为准。

本轮 **没有摇头真实正例**，其七类接口、mask 与梯度由 CPU 合成测试覆盖；不能声称已验证摇头真实监督效果。没有独立人工核验的 dev/test，也没有自然度或音质盲评结论。短 S0 的事件 loss 下降不能当作模型效果。

## 4. 缓存与资源

缓存共约 **516.6 MiB**：input 217.9 MiB、teacher 226.0 MiB、VAE 47.9 MiB、events 0.8 MiB、固定 text 24.0 MiB。最终 U、事件头输出、带随机噪声的训练状态不能预先固定缓存，训练时实时计算。

本次 512×512、有效 batch 8、每次处理一个样本并累积梯度，单卡训练计算每步约 **12–16 秒**；双卡约 **6–10 秒**。这些时间不包含预训练模型加载、训练前缓存核验、样本搬运和 checkpoint I/O。一次训练中 GPU 使用量采样约 **50.5 GiB**，不是峰值显存保证。保存模型与 AdamW 状态的单份 checkpoint 约 **2.69 GB（2.50 GiB）**；生产时应根据磁盘预算设置保存间隔。

GPU 任务均正常结束，使用的 4、5、6 卡已释放；未终止其他用户/任务进程。见 [gpu_released.csv](../experiments/p2_listen_gpu_debug50_20261025/gpu_released.csv)。

## 5. 正式数据就绪后的接入

训练代码已实际跑通，正式数据到位后无需继续补写缓存或训练主链路。新数据仍需要完成导入和一次特征缓存，不能把这批 50 段缓存当作全量缓存。

1. 建立新的实验配置和 `output`，不要在本 debug run 上改清单后续训。保持七类 schema/order，`mapping_version`、`admission_policy` 与正式标签一致。
2. `prepare` 接入 speaker 音频、listener 首帧、listener 目标视频/独立音轨，以及真实 participant/session 分组、train/dev/test 划分和准入状态。当前已验收范围为 25 FPS、512×512、对齐后不超过 201 帧。
3. `import-labels` 接入正式准入标签。**2026-10-25 后续方案更新：** 正常完成标注的有效训练窗口可按明确策略派生弱背景 0，不要求另建人工负例库；真实 unknown、失败和 padding 保留屏蔽。见 [新监督计划 §5.4](P2_倾听事件监督实验计划.md#54-已确认事件区间与弱背景的派生契约)。本页历史正例-only 标签及 S0 结果保持不变；新派生已另存并通过验证，见[弱背景派生实现与验证](P2_弱背景标签派生实现与验证.md)。类别启用按新标签的实际正/背景支持统计，不能只删除禁用判断或擦掉 debug 标记。
4. `cache --part all` 生成五类缓存；可用 `--shards N --shard-index i` 分卡运行，固定文本由 shard 0 写入。标签修订只要求重建事件缓存，其他缓存需经过身份校验后复用。
5. 使用正式数据先跑 E0/E1 短 S0，再用同一正式配置/seed 开始配对训练。正式训练不带 `--s0` 或 `--debug-positive-events`；不使用本批 S0 权重替代新的配对初始化。

命令接口和多卡/恢复/生成示例见 [完整运行说明](P2_倾听事件监督实现与运行.md)。本批复现参数、检查点和日志索引见 [调试目录 README](../experiments/p2_listen_gpu_debug50_20261025/README.md)。
