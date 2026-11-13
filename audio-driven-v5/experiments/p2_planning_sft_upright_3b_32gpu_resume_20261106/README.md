# 新版 Qwen LoRA 四机32卡接续

2026-11-06，用户要求将原node8 GPU7训练迁移至node17、node8、node12、node4各8卡。已实际完成32卡反传、保存和恢复；当前训练仍在运行，以[状态](STATUS.json)及[逐步指标](training/step_metrics.jsonl)为准。

- 最新标签筛选train 704,368条／1,191.9879小时；dev/test不进梯度，不运行dev评估。
- 原始Qwen2.5-Omni-3B，语言层r32/alpha64/dropout0 LoRA；模型、损失、音视频读取与原运行一致。
- 32卡每卡8、累积1，全局batch256；总预算仍2,752步，峰值LR1e-4、原cosine进度及Adam状态接续。
- 原单卡正常工作至1,670步，最近完整检查点为1,500；未保存的1,501—1,670步在32卡重算。原输出与检查点保留。
- 先从1,500步恢复，32个rank均验证优化器／scheduler／RNG及梯度，实际更新至1,503并保存完整32卡检查点；CPU验收通过后从1,503继续正式训练。
- 全局length grouping沿用原单卡8×32=256的分组尺度。CPU重放原Trainer采样器与实际32卡采样顺序哈希一致。最后112条按各rank3／4条分配，不补齐、不丢弃。
- 严格复用既有首帧PNG、隔离AUTO4解码、worker4／prefetch4、BF16／SDPA／TF32／fused AdamW及已有四机NCCL配置。
- world size变化带来分布式归约差异，不承诺后续数值轨迹逐位相同。单卡CUDA RNG按相同状态扩展到各rank的8个可见设备，映射及实际恢复均有记录。

[实际启动验收](LAUNCH_ACCEPTED.json) · [32卡3步验收](CANARY_ACCEPTED.json) · [原采样器重放](PARENT_SAMPLER_REPLAY.json) · [全部批次分配核验](BATCH_PARTITION.json) · [迁移预检](PREFLIGHT.json)

[训练配置](config.json) · [四机配置](cluster.json) · [控制日志](controller.log) · [四机训练日志](training/launch_logs/)

最终权重将写入 `training/final_adapter/`。只有原生 `training/COMPLETE.json` 和控制器 `COMPLETE.json` 均成功才表示完整训练结束。原单卡守护和队列已按此次迁移停止；此前暂停的会话goal没有恢复。
