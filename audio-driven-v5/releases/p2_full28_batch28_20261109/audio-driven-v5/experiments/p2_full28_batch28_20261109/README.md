# 当前完整组四机28卡、global batch28续训

用户明确要求前三机各8卡、node11仅0–3号卡，留出4–7号卡，batch与卡数相等。模型为当前LTX全attention/FFN rank8 LoRA＋Whisper两路全参数＋AV＋整段Event＋双路Feature，使用原冻结Qwen hidden条件。非在线Qwen联合微调。最终启动状态见 `ACCEPTED.json`。

父输出 `../p2_ltx_all_lora_full31_resumed_20261109/`，完整恢复点step6428；新输出 `../p2_ltx_all_lora_full28_batch28_20261109/`。模型、Adam、更新步数保留；父31份RNG保留原处，新28rank从父rank0启动RNG初始化，每个样本仍按原noise ordinal独立设种子。

只把余下84971个原计划pairs按28连续分组，已消费199268个pairs不动；不重新shuffle，不额外补样。最后19条按实际样本数归一化AV和Feature，9个空rank仍参与FP64 SUM；Event按全局实际known标签计数归一化。新一轮终点9463，原总样本量不增加。batch改变会改变优化轨迹，不宣称同batch消融。

四机8＋8＋8＋4；node11 GPU4–7不放本训练。保持已验证四机单网卡、原学习率、all重计算、GPU Adam、512MiB streaming FP64、每500更新完整保存与每小时观察。启动前实际资源检查记录在 `RESOURCES_BEFORE.json`。

## 实现和验收

独立 `runtime28.py`/`training.py`/`worker.py`/`canary.py`，原源码和6428权重不修改。配置记录真实batch28与新步数；模型和冻结缓存读取器用规范化原配置构建，确保结构、数据来源和标签不变。

全剩余计划逐条核对；尾批、后续恢复的绝对偏移、AV/Feature均值和Event有效计数检查；3rank实际Gloo检查包含空rank和未知Event标签时的全局归约、梯度及Adam对照。启动前迁移逐值核验模型及Adam，并执行两次最长11.95秒真实更新、严格恢复6428后才继续正式更新。

## 入口及恢复

`manager.py prepare`/`migrate`/`launch`用于本次独立6428接续；已经有后续进度时，不可盲目重新迁移或执行固定6428启动命令，应使用新输出最新完整checkpoint保持当前身份恢复。原输出STOP保持，旧6500停止预约已经取消，不应重新启动。

`watch_hourly.py`为只读小时观察器，明确忽略node11预留GPU4–7上的其他任务；不杀其他进程。未安排自动切换其他pipeline。

## 实际启动结果

已从6428恢复，验收到6440，12次正式更新平均9.02秒／步（短窗口）。最长两次更新峰值94.64GiB。28个进程卡位、loss／梯度、batch28及消耗样本计数核验通过，小时监控已启动并完成首轮检查。详见 `ACCEPTED.json`、`HOURLY_LATEST.json`。
