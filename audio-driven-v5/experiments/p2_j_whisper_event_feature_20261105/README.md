# 原始 H＋AV＋Event＋Feature＋speaker Whisper

四机32卡测试与执行优化已完成，2026-11-05 11:41 UTC通过完整第18步检查点验收后停下。**没有自动进入完整8927步训练。**

- 损失：AV 1＋Event 0.05＋视频 Feature 0.025＋音频 Feature 0.025。
- 输入：原始冻结 Qwen H＋Gemma 指导文本＋listener 首帧＋speaker Whisper 独立双路 attention。
- 数据：十二组对照的原500h／285,638条train，复用最终裁框、净化listener音轨、H、Gemma、VAE、teacher与speaker Whisper缓存；不是新筛选的train＋dev 284,218条版本。
- 实际配置：32卡／global32，duration512，全层重算，GPU Adam，FP64 SUM **512MiB桶＋后台CPU预取2批**；正式每500步保存。
- 固定512条同样本测速：5.734607→5.471520秒／步，步时减少4.59%，完整梯度／损失逐值一致。
- 18步真实更新通过，分组16步平均5.821668秒／步；剩余预算粗估14–16小时。
- 73项CPU回归、285,638条全量收据、真实线程预取和32rank最长片段均通过。最长样本完整Adam检查最大rank96.734375GiB，测速峰值最大107.355469GiB。

## 入口与结果

- [完整结构、缓存和代码交接](../../related/P2_AV_Event_Feature_Whisper联合实验代码交接_20261105.md)
- [四机实测、加速方案和继续命令](OPTIMIZATION_32GPU.md)
- [实验配置](../../configs/p2_j_qwen_whisper_event_feature_20261105.json) · [32卡集群](../../configs/p2_j_qwen_whisper_event_feature_32gpu_20261105.json) · [统一入口](../../scripts/p2_j_whisper_event_feature.py)
- [实际18步与检查点验收](test32/COMPLETE.json) · [检查点32份RNG／五组Adam／当前源码身份](test32/CHECKPOINT_METADATA_CHECK.json)
- [同512条测速完整结果](test32/benchmark/RESULT.json) · [执行选择](test32/EXECUTION_SELECTED.json)
- [73项CPU测试](cpu_tests_optimization.log) · [全量preflight](preflight.json) · [真实预取](CPU_PREFETCH_CHECK.json)
- [训练日志](metrics.jsonl) · [TensorBoard](tensorboard/)
- [旧Whisper模块SHA未改](LEGACY_SOURCE_CHECK.json) · [当前源码快照](optimized_snapshot/) · [初版快照（历史）](code_snapshot/)

检查点为本目录 `step_000018.pt`，配套 `.complete.json` 保存完整SHA。第0／18步是本次测试退出所需额外检查点，后续仍按500步保存。当前已存在run，必须按交接中的 `--resume .../step_000018.pt` 严格恢复，不能覆盖重开。本轮没有修改旧纯AV Whisper源码或归档，没有新增媒体缓存，也未推送GitHub。
