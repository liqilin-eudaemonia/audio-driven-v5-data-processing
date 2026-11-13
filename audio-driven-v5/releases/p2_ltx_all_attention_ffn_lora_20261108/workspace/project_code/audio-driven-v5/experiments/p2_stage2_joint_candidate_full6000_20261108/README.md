# Qwen＋Connector＋LTX联合微调候选：GPU debug已完成

用户授权的四机31卡debug已完成，原完整模型6000＋SFT2752，仅沿用原text LoRA范围。node11与用户要求保留的vLLM叠加；未使用后来新增的全attention／FFN LoRA版本。

- 最长实际片段两次完整更新成功，AV／Feature／整段Event及规划CE都接入；生成梯度可达原SFT Qwen LoRA。
- 全局峰值106.457GiB；node11训练进程上限108GiB，其余134GiB。
- 测试更新已回滚，step0模型、fresh Adam及31份RNG完整保存；SHA及worker退出已检查。
- 当前为`awaiting_baseline_evaluation`，不会自动进入100步。正式64条端到端评测尚未执行。
- 候选规划权重0.1的Qwen梯度在两次探针中分别为生成项147.7／227.7倍，正式训练前需评估损失平衡，不以这两条样本推断整个分布。

[GPU调试验收](debug_launch_20261108/GPU_DEBUG_ACCEPTED.json) · [真实更新／梯度／显存](LONGEST_SAMPLE_CHECK.json) · [step0完整标记](step_000000.complete.json) · [worker退出](debug_launch_20261108/DEBUG_WORKERS_EXITED.json) · [代码与完整方案](../../related/P2_第二阶段联合微调实现与验收_20261108.md)。
