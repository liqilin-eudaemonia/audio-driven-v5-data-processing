# Event 组四分之三重计算切换

详细加速方案、参数表、实测口径与多机复用边界见 [加速交接](../../related/P2_Whisper训练四分之三重计算加速交接_20261106.md)。正式训练窗口快照见 `ACCELERATION_RECORD.json`。

用户已确认接受实测 128.1367 GiB 的 reserved 峰值，替代旧的 128 GiB 筛选门槛。保留每进程 134 GiB 分配上限。

采用 Event-only `three_quarters`；后续 Feature 仍为 `all`，未将 Event 测速结论推广到未测试的 Feature 分支。

- 新队列：`scripts/p2_whisper_recompute_profile30_20261106.py run`。
- 新 Event / Feature 训练入口和缓存路径分别见本目录 `event/config.json`、`feature/config.json`。
- 原队列 `../p2_whisper_continue30_20261106` 保留完整检查点及停止标记，避免双启动；`SUPERSEDED_BY.json` 指向本目录。
- 拓扑：node8/node12/node4 各8卡，node11仅0–5卡，共30卡；6、7卡不参与。
- global batch30，原始H，原有512修正版裁框、净化listener音轨、speaker Whisper与损失保持；同一 remaining schedule、样本/noise ordinal、学习率、总预算与每500步保存策略均保持。
- 原训练在当前完整更新后立即保存，迁移模型、Adam及30份rank RNG逐值验证。仅版本化实验配置及执行源码身份发生变化，未覆盖已绑定的旧训练代码。
- 新输出 Event 预算8134步；之后自动开始 Feature，预算8267步。原训练开始至切换前的指标仍在原目录，新目录只追加接续指标。

证据：`CPU_VALIDATION.json`（配置边界、调度、损失与拓扑）；`event/MIGRATION.json`（完整迁移）；`event/RESUME_VERIFIED.json`（实际接续更新）；`state.json`、`event/status.json`（当前进度）。源码备份及摘要见 `code_snapshot/MANIFEST.json`。

速度依据：`../p2_whisper_recompute30_trial_20261106_retry/partial_results.json`。30卡、同480条样本，all 5.1944 秒/步，three_quarters 4.8601 秒/步，步时减少约6.4%。最长11.95秒样本、两次真实Adam更新通过，损失、梯度、模型、Adam与诊断逐值一致。完整ABBA复测按用户要求中止，此结果是候选测速，不保证整程稳定提速6.4%。

操作：在本目录根部创建 `STOP` 可在完整更新及保存后停止队列；恢复时确认用户停训意图后再移除相应STOP，运行上述入口。勿同时重启原队列。

加载这次的新配置／权重时，先把 `src` 和 `scripts` 加入Python搜索路径，并调用 `p2_whisper_recompute_profile_20261106.install()`，再调用现有模型／训练读取入口。它先安装冻结源兼容，再启用新执行配置的严格身份校验。不得跳过身份校验或把新权重冒充旧实验。
