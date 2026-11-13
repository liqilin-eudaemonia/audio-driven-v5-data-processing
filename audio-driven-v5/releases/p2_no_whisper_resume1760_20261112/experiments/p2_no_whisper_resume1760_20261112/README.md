# 去Whisper全LoRA消融：1760步续训至6643

2026-11-12用户明确要求继续。复用已有`p2_ablation28_guard9h_20261111/recover.py`恢复入口和原冻结runner/runtime；不修改训练源码、配置、数据、损失、优化器、batch或样本/noise调度，不重新初始化。

- 组：AV＋双路Feature＋LTX hidden整段Event；无Whisper分支，Qwen冻结并读取原始Qwen hidden。
- 最终train＋dev500h及最新caption；LTX原生48层全attention／FFN rank8 LoRA。
- 恢复：`../p2_all_lora_ablations28_20261111/no_whisper/step_001760.pt`，完整模型、Adam及28份RNG；原生SHA／有限值／身份检查后恢复。
- 目标：6643，剩余4883次更新，按原完整组6000×31的样本展示预算近似对齐。
- 四机28卡，node12／node8／node4各0–7，node11仅0–3；其4–7卡保留。batch28，每卡1条。
- 沿用每500步与最终6643保存；下一个定期点2000。学习率及原预热进度接续。
- 用户新指令授权恢复，原总队列／本组STOP已留档在本目录，见`USER_RESUME_AUTHORIZED.json`。

启动器：`LAUNCHER.json`、`CONTROLLER.json`；日志`launch.log`；实际训练进度、TensorBoard和新检查点仍写原`../p2_all_lora_ablations28_20261111/no_whisper/`。每小时只读记录沿用原队列`hourly.jsonl`；达到6643后原生完整验收并结束，不启动其他训练或新评测。

`reliable_entry.py`只修复原多机启动器对SSH瞬时断开的重试竞态；保留原进程组保护，不修改训练函数或检查点身份。正式进度以实际更新及`RESUME_ACCEPTED.json`为准。
