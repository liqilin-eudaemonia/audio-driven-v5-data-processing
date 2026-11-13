# 缓存验收后自动四机训练：Qwen H＋Whisper，仅AV

> 当前已转到[立即时长分组运行](../../p2_j_qwen_whisper_av_duration_now_20261104/README.md)。本目录STOP保留；完整第2步权重、Adam与32卡RNG经过逐值核对迁移，原文件不改。后续实际状态与步速请看新目录。

2026-11-04用户已授权：缓存结束后四机联合训练。后台守护已挂上，实际状态见 [state.json](state.json)。

- [实际训练配置](../../../configs/p2_j_qwen_whisper_av_train32_20261104.json)：原始未SFT Qwen H＋原Gemma文本＋新增Whisper双路cross-attention，**仅AV loss**，无Event／Feature损失。
- [实际四机配置](../../../configs/p2_j_qwen_whisper_av_train32_cluster_20261104.json)：node17／node8／node12／node4各8卡，world32、global batch32，继承既有NCCL参数，使用已核实的known_hosts。
- 原500h、285,638条，同一最终裁框、listener净化音频latent、Qwen H、Gemma和数据顺序。Whisper缓存为 `experiments/p2_j_speaker_whisper_cache_20261104_prefetch_v2/`。
- 总预算8,927步，保存周期500步及最终步。第2步是同一正式训练的启动检查点，验收后严格恢复继续，不重置优化器和步数。

## 自动顺序

1. 等完整speaker缓存 `COMPLETE.json` 成功，核对样本数、角色和协议SHA。
2. 训练数据preflight使用32个CPU进程核对原缓存及新缓存的全部来源／完成收据；复用历史进程池经验，避免Python GIL使线程校验串行。
3. 确认32卡实际释放，不终止其他任务。
4. 每rank对最大时长真实样本做一次完整前后向，不更新参数，恢复CPU／GPU RNG后进入正式训练。结果在上一级 `memory_probe/rankNNN.json`。
5. 四机先执行正式第1–2步，核对损失有限、仅AV、Connector／文本LoRA／Whisper梯度、32卡最长样本检查及完整第2步检查点。
6. 从第2步完整权重严格恢复，继续到8,927步。

训练入口：[p2_j_whisper_av.py](../../../scripts/p2_j_whisper_av.py)；接续入口：[p2_whisper_train_after_cache.py](../../../scripts/p2_whisper_train_after_cache.py)。

## 执行优化与检查

- 继承原时长分组、预计算H／Gemma／AV缓存、全局batch和确定性归约协议。
- 使用全层非重入梯度重计算，先保证新增约27.8亿参数下的最大时长显存安全。
- 32卡每rank一个microbatch时，FP64梯度按32MiB流式SUM归约，去掉整份FP64累积及整份舍入缓冲；多microbatch时回退原FP64累积。loss归一化、参数、数据顺序不改。
- FP32 Adam动量在前后向期间暂存CPU，更新前回到原GPU；step标量保持原位置，不改Adam公式和状态dtype。
- 大检查点采用流式SHA检查和mmap读取。恢复绑定代码、数据、缓存、配置、world和rank RNG。
- CPU回归覆盖流式／整份FP64一致、跨bucket与未使用参数、多microbatch拒绝、Adam状态／更新一致、缓存准入与纯AV启动门控；两进程Gloo流式更新对照串行原归约，两次完整更新后参数逐值一致。真实GPU启动检查在接续中自动执行，未用CPU测试代替GPU验收。

`controller.json / guard.json`记录进程身份与心跳；`preflight.log`及`initial2_* / continue_epoch_*`保留日志；停止可在本目录创建 `STOP`。守护仅恢复本任务，最多5次，使用最新完整且匹配身份的检查点，不跨版本假装严格恢复。

原1,107,323条打标队列已排空；原打标CPU审计仍单独记录。当前500h训练输入已冻结，不将新打标审计的候选或失败结果自动替换进该训练集。
