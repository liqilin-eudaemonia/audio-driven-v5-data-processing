# 当前消融结束后，完整全LoRA从5500接续至7000

**2026-11-13 UTC00:42最终更新：** step7000及15份每100步检查点均已完成，全部训练进程退出；7000模型／Adam／28RNG和SHA重新核验通过，见[最终权重复核](FINAL_7000_RECHECK_20261113.json)。下述6704状态为历史快照，不应据此重复启动。

**UTC23:52更新：** 当前6704步，6700已有原生完整保存标记；6600另经SHA／模型／Adam／28RNG完整验收。九小时巡检时段已结束，巡检进程退出，训练仍在继续，目标7000、每100步保存及全部数据／损失保持。node4新SSH连接持续被拒，四机完整硬件验收受阻，详见[九小时验收记录](../p2_full_chain_guard9h_20261112/FINAL_INCOMPLETE.json)。当前训练控制器存活，无需因巡检受阻重复启动训练。

**UTC2026-11-12 20:38已实际接续：** 去Whisper6643完成完整保存／原生校验并退出，本组四机28卡已从完整5500模型／Adam／28RNG恢复，真实更新超过5505。最长11.95秒样本两次更新及检查点回滚通过，峰值94.64GiB；Connector、LTX LoRA、Whisper、Feature、Event梯度均非零且有限。每100步保存设置保持，首个新保存点5600。[实际接续验收](../p2_full_chain_guard9h_20261112/HANDOFF_ACCEPTED.json)。下文等待安排为原执行计划，实时进度以`train/status.json`及`train/metrics.jsonl`为准。

用户2026-11-12明确要求接在当前训练之后；当前去Whisper组继续到6643，完整保存、验收且所有训练进程退出后，自动启动本任务。失败或提前停止不跳过前置任务。

- 起点：`../p2_ltx_all_attention_ffn_lora31_20261108/step_005500.pt`，SHA `7343755f189f401d38b746ba40f57241b3f1728e336057a354e7e8b2824e4244`。
- 配方：全LTX原生attention／FFN rank8、alpha8、48层1632处LoRA；新增Whisper→音频／视频attention全参数训练；原始Qwen hidden冻结。
- 数据：沿用5500对应最终train＋dev 284218段、500h及最新caption／缓存／数值标签，不重新打标或抽样。
- 损失：AV=1，LTX hidden整段Event=0.05，双路Feature各0.025；LR、Adam状态及预热进度保持。
- 资源：node12／node8／node4各GPU0–7，node11仅GPU0–3；四机28卡、batch28，每卡1条。node11 GPU4–7保留。
- 目标为绝对step7000，即新增1500次更新；每100步完整保存：5600、5700……7000，共15个新检查点；7000保存核验后退出，不接新评测。
- 单独输出到本目录`train/`，不覆盖历史5500／6000／6500／7000／7500。

31→28迁移沿用此前验证方式：模型及Adam逐值保持，原31份RNG仍保存在源权重中，源rank0启动RNG复制为28份；每个样本仍使用原ordinal确定noise seed。原5500消费170500条pair，下一步5501从该位置继续取28条，保持后续样本／noise顺序，不跳样本。batch改变，因此不声称与历史31卡逐步轨迹相同。

`prepare.py`在CPU核验源检查点SHA、源码、500h数据身份、完整模型／Adam／RNG、连续样本计划及100步保存设置，生成`READY.json`与`train/MIGRATION.json`。复用此前完整组batch28原生训练循环及最长样本两次更新／回滚检查，不修改正在运行的消融代码。

实际衔接控制器为`continue_after.py`，经此前`reliable_entry.py`启动，保留原生多机进程保护并处理已知SSH瞬时重试竞态。每30秒检查前置结束及资源释放条件，每小时记录训练进度。`STATE.json`区分等待前置、等待选定28卡、训练和完成；实际训练进度见`train/metrics.jsonl`，TensorBoard见`train/tensorboard/`。

原生`train/status.json`的budget仍为完整样本计划终点9563；本任务在7000停下，以本目录`COMPLETE.json`表示本次目标完成，不把完整epoch标成完成。每个检查点均含模型、Adam、28份RNG与SHA收据。

启动／重启：使用`cluster.json`中的Python执行

```bash
/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -u experiments/p2_stage2_joint_full6000_trial28_20261112/reliable_entry.py experiments/p2_full_resume5500_to7000_after_ablation_20261112/continue_after.py
```

控制器单实例锁防止重复启动；已有本组训练时不得再次手动拉起。创建本目录`STOP`取消等待；训练中创建`train/STOP`由原生训练完整保存并退出。当前消融的STOP独立，不会被本任务清除。
