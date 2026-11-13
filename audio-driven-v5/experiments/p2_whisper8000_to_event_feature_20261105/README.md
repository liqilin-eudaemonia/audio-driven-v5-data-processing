**本28卡等待队列已按用户后续指令关闭，改为三机24卡：** [当前训练入口](../p2_j_base_h_av_event_feature_continue24_20261105/README.md)。本目录保留Whisper8000步保存与node8故障历史。

# Whisper 8000步保存后恢复原始H＋AV＋Event＋Feature

2026-11-05用户授权：当前Whisper训练保存step8000后，转回尚未完成的组合损失实验。接续已启动，实际阶段以[state.json](state.json)为准。

- 来源：`../p2_j_qwen_whisper_av_tuned_20261104/`。8000步指标写出后，仅创建训练器的`output/STOP`；训练器先完成既定8000步检查点，再在下一轮退出。校验完整标记和权重SHA、确认原控制器正常退出后，才写`queue/STOP`关闭原守护重启入口。保留Whisper模型、Adam、32rank随机状态及全部缓存。
- 目标：`../p2_j_base_h_av_event_feature_20261103/step_005500.pt`。严格沿用原配置、源码、数据、teacher、448时长分组及28rank随机状态；旧5514步日志另存`metrics_before_resume.jsonl`，由原训练器按5500步截断并重放尚未保存的14步。
- 资源：node17/node8/node12各8卡，node4的0–3卡，共28卡；global batch28、总10202步，每500步及最终保存。保留原异构rank映射和已验收的单HCA网络配置，未切成32rank/global32。
- 目标损失：`AV + 0.05 Event + 0.025 video Feature + 0.025 audio Feature`；冻结原始未SFT Qwen H，无Whisper条件。

入口为[`../../scripts/p2_whisper8000_then_event_feature.py`](../../scripts/p2_whisper8000_then_event_feature.py)。`spec.json`固定双方配置摘要；`PRECHECK.json`记录真实检查点、28rank RNG、104源码文件、data identity、teacher身份和确定性时长计划校验。四项CPU测试覆盖保存与停止顺序、坏检查点不能暂停当前训练、辅助损失梯度检查。没有改动现有两组训练核心源码。

接续控制器记录在`controller.json`，日志`controller.log`；持有本目录及目标原队列两个排他锁，不能重复启动。原队列旧`state=training`并非实时状态，恢复启动后会更新。

状态顺序：`waiting_whisper_step8000` → `waiting_whisper_checkpoint_and_graceful_exit` → `waiting_selected_28_gpus` → `target_training` → `complete`。校验失败为`needs_attention`，不会跳过检查来启动。node8 SSH当前在认证前断开，资源检查会重试并等待，不能据此认定当前训练失败。

`SOURCE_SAVED.json`只在8000步完整权重SHA验证后写出；`RESUME_VERIFIED.json`需真实恢复更新并核验损失公式、四组梯度后才写出。实际恢复日志位于`attempt_*/`。启动失败最多重试5次，每次仅从目标最新完整检查点恢复；不停止其他任务。创建本目录`STOP`会请求已启动的目标训练完整保存后退出；首次等待阶段的`STOP`不会修改当前训练。


## 当前实际状态（2026-11-05 UTC 02:19之后）

Whisper第8000步完整权重已保存并独立SHA通过，见`SOURCE_SAVED.json`和`STEP8000_SHA_VERIFIED.json`。node8在保存完成后的NCCL退出阶段因`pthread_create: Resource temporarily unavailable`异常退出，原守护尝试重新做CPU预检；确认完整权重后，已停止该重复CPU预检并关闭原队列重启入口。来源权重和数据没有丢失。

目标首次启动已因node8资源隐患主动停止，仅停止本接续持有的启动进程；未发生新的有效训练更新。调查发现node8有627297个/proc进程项，分层抽样301项中300项为PID1（`sleep infinity`）的僵尸子进程；cgroup计数628942/629145，仅余203个进程／线程名额。SSH在旧GPU进程退出后暂时恢复，但不足以启动8个训练rank。容器cgroup只读且无SYS_PTRACE权限，不能在容器内安全回收PID1僵尸，需要平台重启该环境。诊断见`NODE8_PID_DIAGNOSTIC.json`。

接续控制器已加入四机PID配额门控：每机空余至少4096个名额，且28张卡空闲才可启动。新控制器已运行，等待node8恢复；请看`controller.json`、`state.json`、`resources.json`与`RELAUNCHED_PID_GATE.json`，不要凭旧attempt日志判定已恢复。恢复后仍严格读取5500步模型／Adam／28rank随机状态。新门控第4项CPU测试验证了203名额会被拒绝、4096边界与无限配额处理。
