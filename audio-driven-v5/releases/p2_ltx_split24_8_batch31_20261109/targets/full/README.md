# 完整Event组：node11全部8卡

已实际接续并通过验收。继承原模型、Adam和step，从6000继续，验收到6004；global batch31，学习率／损失／样本-noise计划／每500步保存保持。迁移证据MIGRATION.json，最长样本两次真实更新及恢复证据LONGEST_UNEVEN_CHECK.json。

当前状态见status.json，指标metrics.jsonl／tensorboard，逐卡日志rank_logs/。原检查点保留。停止可在本目录创建STOP，原生训练会在当前更新完成后保存退出；再次启动必须从最新完整检查点严格恢复，不能重跑初次迁移控制器。

[两组最终验收](../p2_ltx_split24_8_batch31_20261109/FINAL_ACCEPTED.json) · [执行与恢复说明](../p2_ltx_split24_8_batch31_20261109/README.md)。
