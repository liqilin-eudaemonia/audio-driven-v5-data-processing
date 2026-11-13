# TensorBoard 接入设计

用户已要求补齐训练监控。采用训练时直接写 TensorBoard，保留 JSONL；相较离线转换，可以实时观察训练，并在同一恢复点处理曲线和日志。

每个 seed/arm 只有 rank 0 的 writer。记录全局 AV/base/优化目标、AV 视频/音频分项、全局 Event 分类和模态贡献、监督计数、优化器分组 LR、裁剪前梯度范数、数据/计算/步耗时、吞吐及最大 rank 显存。事件分母继续用现有全局计数；没有监督的类别不伪造零损失。E0 的 Event 是 detached probe 指标，生成目标另列。

独立可选配置 enabled/log_every/flush_secs；当前 LoRA/H200 配置默认开启。恢复从 saved_step+1 写入，TensorBoard purge_step 清理未提交尾部；JSONL 同步按检查点裁去尾部。每次检查点后和结束时 flush，finally close。日志仅使用 detached 数值，不消费训练 RNG。保存频率 100、损失和梯度逻辑不变。

验证：真实 event accumulator 读取、恢复清理、CPU 两 rank 不同损失/监督计数聚合、已有 CPU 回归，以及真实 GPU 单/双卡和恢复短测。保持严格源码/配置身份校验，不放宽旧版本 checkpoint 的恢复边界。
