# TensorBoard 接入验收

已完成；真实 GPU 测试进程全部结束，没有启动正式效果训练。

- [运行说明与指标口径](../../related/P2_TensorBoard训练监控与恢复.md)
- [验收汇总](summary.json)、[GPU 指标/恢复/等价性](gpu_verification.json)
- [146 项 CPU 测试日志](cpu_tests.log)、[两进程监控聚合](cpu_metrics_ddp/summary.json)、[训练 CPU DDP](cpu_training_ddp.json)
- [E0 实际事件文件](single/runs/s0_20261026/E0/tensorboard/)、[E1 实际事件文件](single/runs/s0_20261026/E1/tensorboard/)
- [双卡 E1 事件文件](ddp/runs/s0_20261026/E1/tensorboard/)
- [修改前源码](code_before/)、[修改后源码](code_after/)、[当前校验清单](after_manifest.json)

E1 单卡和双卡的事件目录各有三个文件：第一次训练、人工插入未提交尾部、恢复训练。这是有意进行的恢复测试，不是多个 rank 重复写入。TensorBoard 读取后只显示正确的 1–5 步，AV=999 的旧第 4 步已被 purge；JSONL 同步清理。
