# 已运行全LoRA完整组／去Event组复核

2026-11-09，复核对象是正在运行的完整组与停在660的去Event组，不是刚补齐的去Feature／去Whisper。最终证据见 [REVIEW_ACCEPTED.json](REVIEW_ACCEPTED.json)。本次没有修改训练源码、停止完整组或恢复去Event组。

完整组：AV＋0.05 LTX整段Event＋0.025视频Feature＋0.025音频Feature，Qwen缓存与speaker Whisper条件；去Event组：AV＋双路各0.025 Feature，保留同样两路条件，Event头／标签reader／优化器组均无，实际指标Event恒为0。两组都是48层1632处rank8／alpha8 LoRA，新增两路Whisper attention全参数训练，原生LTX基座冻结。

检查了运行身份绑定的295／319份源码，无改动；数据路径、sample list、最新caption缓存、原始Qwen H、teacher、Whisper、LoRA、Connector和除batch／总步数外的训练参数一致。完整6500与去Event660检查点重新执行SHA、模型／Adam有限值、schema／identity／优化器分组／RNG检查。去Event状态仍为660且STOP存在。

43项原生模型和执行测试通过：LoRA覆盖、冻结基座、梯度路径、无Event分支、共享初始化、精确下一步恢复、旧范围拒绝、batch31在24卡不均匀分配的样本权重及梯度归约、28卡后续样本/noise计划与尾批归一化。各16条额外真实样本（train12＋dev4）经过后台预取，同ID／同完整caption，teacher与Whisper均存在，Event仅完整组存在。两组此前最长11.95秒两次真实GPU更新及恢复证明也重新核对通过；这次没有另开GPU测试抢占当前任务。

当前完整组已从6428改batch28，去Event仍是batch31、只训练660步。两组共享组件与数据设计正确，但现有检查点训练预算不同，不能直接作为等预算去Event消融结论。后续应按累计样本展示量对齐，并如实记录batch变化。
