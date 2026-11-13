# Qwen H＋Whisper，仅AV：256MiB通信桶正式接续

从第209步完整检查点接续，四机32卡。原500h／285,638条、最后裁框、listener净化音轨目标、原Qwen H／Gemma／speaker Whisper缓存、模型结构、学习率、Adam状态、32卡RNG和全部样本／噪声顺序不变。继续从第3步就启用的duration512，预算8,927步、每500步和最终步保存。

本次唯一训练执行变更为FP64流式SUM通信桶从32MiB增至256MiB。保留全层梯度重计算、GPU驻留FP32 Adam状态、BF16前后向、FP32可训练参数、既有NCCL环境和mmap恢复。

## 同批32卡实测

同一后续512个样本位置，固定权重、零学习率Adam、每组预热后计时16步；不计入正式更新。

| 方案 | 平均秒／步 | 峰值预留显存GiB | 结论 |
|---|---:|---:|---|
| 全重算＋32MiB | 5.4357 | 102.844 | 同批基线 |
| 全重算＋256MiB | 5.0634 | 102.900 | **采用，步时减少6.85%** |
| 四分之三重算＋256MiB | 4.8221 | 133.998 | 超过预设128GiB余量门槛，未采用 |

三组首个计时batch的32卡完整裁剪梯度SHA一致，16步全局损失逐值一致。四分之三重算没有写成OOM失败：实测跑完，但预留显存贴近134GiB allocator上限，不满足预先设定的余量要求。

记录：[不可修改的测速报告](../p2_j_whisper_execution_tune_20261104/RESULT.json)、[逐值迁移验收](migration.json)、[配置](config.json)、[实际进度](status.json)、[训练指标](metrics.jsonl)、[接续状态](queue/state.json)。测速和正式训练的参数更新方式不同，且正式训练含梯度分组诊断、日志等额外开销；预计结束时间以恢复后的完整窗口更新。

## 接续与代码

- [训练入口](../../scripts/p2_j_whisper_av.py)、[有界FP64归约](../../src/p2_whisper_av/execution.py)、[迁移校验](../../src/p2_whisper_av/migration.py)、[一次性测速驱动器](../../scripts/p2_whisper_execution_tune.py)、[GPU测速入口](../../scripts/p2_whisper_execution_benchmark.py)。
- 迁移逐值核对模型、优化器和32卡RNG；原第209步检查点及旧运行不改，旧队列STOP保留。
- [接续脚本](../../scripts/p2_whisper_train_after_cache.py)启动后执行完整来源检查、32卡最长样本前后向和严格恢复。`STOP`放在训练根目录表示保存完整检查点后平稳停止；`queue/STOP`表示立即取消队列所属进程。已用跨过15秒轮询周期的真实等待测试确认平稳停止不会被监督线程提前打断。
- 一次性测速驱动器已拒绝对已有`RESULT.json`重复执行，避免再次暂停已接续的训练。恢复依据为当前目录自己的完整检查点。

相关11项CPU回归通过；独立进程bootstrap检查与32卡真实同批测试通过。正式运行状态见上方实时文件。
