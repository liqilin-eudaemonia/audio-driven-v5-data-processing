# Qwen H＋Whisper，仅AV：立即时长分组接续

> 当前已从完整第209步转移到[256MiB通信桶正式运行](../p2_j_qwen_whisper_av_tuned_20261104/README.md)。本目录及STOP保留，后续实时步数请看新目录。

**已实际接续并通过首个完整窗口检查。** 第3–18步共512个样本位置，平均5.444秒／步，范围3.419–9.165秒；32卡带完整Adam状态的最长片段检查通过，最大预留显存87.133GiB。三个训练组梯度非零、损失有限，Event／Feature均为0。按首窗口粗估剩余计算约13.5小时，连同存盘余量约14–16小时，需随后续实测更新。切换前2步为27–29秒；两次批次不同且同时取消CPU搬运，不作为单独时长分组的严格AB提速比例。[运行验收与阶段耗时](PERFORMANCE_VERIFIED.json) · [140文件固定快照](../../releases/p2_whisper_duration_now_20261104/manifest.json)。

2026-11-04用户要求立即采用时长分组，并复用总索引已有优化。原运行的前2步完整保留，从第3步采用512样本窗口内时长排序、batch顺序打乱；保留全部样本、原随机ordinal、global batch32和8,927步预算，每500步保存。

原数据、最后裁框、净化listener目标、原Qwen H／Gemma／Whisper缓存、模型结构、损失及学习率均不变。新增Whisper两路cross-attention，目标仍只有AV。

- [实际配置](config.json)、[时长计划](duration.schedule.json)、[迁移验收](migration.json)。迁移逐值检查模型、Adam状态和32卡RNG，并核对消费前缀及完整样本／noise ordinal多重集。
- [当前训练状态](status.json)、[训练指标](metrics.jsonl)、[接续状态](queue/state.json)。前2步指标继承原运行；从第3步才是本配置的耗时。
- [最长片段各rank检查](memory_probe/)：用迁移后的权重及驻留GPU的Adam状态执行无更新前后向，恢复RNG后开始正式训练。
- 继承全层梯度重计算、32MiB FP64流式归约、已有NCCL环境、预计算条件／目标缓存、mmap恢复。关闭每步Adam动量CPU往返，FP32状态及Adam公式不变。前16个新步骤记录load／forward_backward／gradient_reduce／gradient_norm／optimizer耗时，不用单纯显存或GPU利用率判断提速。
- 原检查点和旧运行完整保留在 `../p2_j_qwen_whisper_av_20261104/`；旧队列STOP保留，避免双份训练。

参考经验：[32卡时长对齐](../../related/P2_J整段backchannel辅助监督_20261101.md#32-卡时长对齐实测与接续)、[重计算及通信分桶](../../related/P2_24卡联合训练GPU验证与提速.md)、[原Feature训练长片段恢复经验](../p2_j_guided_av_feature_base025_20261102/duration_memorysafe_save500/README.md)。旧实验的加速比例不能直接套到新增约27.8亿可训练参数的本实验，实际步速以本目录指标为准。

代码：[迁移工具](../../scripts/p2_whisper_duration_now.py)、[执行约束](../../src/p2_whisper_av/migration.py)、[训练入口](../../scripts/p2_j_whisper_av.py)。本次31项CPU回归通过，包含迁移样本前缀／多重集、训练配置边界、来源变更白名单、架构梯度和分布式一致性。
