# 原始 H＋AV＋Event＋Feature＋speaker Whisper

代码已实现，尚未启动训练。完整说明见 [联合实验代码交接](../../related/P2_AV_Event_Feature_Whisper联合实验代码交接_20261105.md)。

- 损失：AV 1＋Event 0.05＋视频 Feature 0.025＋音频 Feature 0.025。
- 输入：原始冻结 Qwen H＋Gemma 指导文本＋listener 首帧＋speaker Whisper 独立双路 attention。
- 数据：沿用十二组对照的原 500h／285,638 条 train 及全部现有缓存；不是新筛选的 train＋dev 284,218 条。
- 预算：32 卡／global batch32／8,927 步，每 500 步保存；同源 duration512 调度与全层重算、FP64 256MiB 通信桶、GPU Adam。
- [实验配置](../../configs/p2_j_qwen_whisper_event_feature_20261105.json)、[集群配置](../../configs/p2_j_qwen_whisper_event_feature_32gpu_20261105.json)、[入口](../../scripts/p2_j_whisper_event_feature.py)。
- [CPU 回归日志](cpu_tests.log)：65 项通过。
- [真实 3 条联合缓存读取](CPU_CACHE_CHECK.json)：含最长片段、来源和时长计划校验，未初始化 CUDA。
- [四机启动命令（未执行）](commands_32gpu.json)。
- [旧模块未改动核对](LEGACY_SOURCE_CHECK.json)、[本地代码快照](code_snapshot/)。

开训还需全量 preflight、真实最长样本显存及 32 卡检查。入口 `launch` 会先 preflight，训练器在正式第 0 步保存前运行并恢复最长样本两次更新；失败不继续。
