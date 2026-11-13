# 联合微调：从100步接续至1000步

**已完成（2026-11-12）：** 1000步训练、300／600／900／1000完整保存、最新v3同64dev评测与对比全部完成，进程正常退出。本轮LPIPS 0.2433→0.2563、共同可判定视觉F1 0.5659→0.5070、静音参考片段额外发声2/39→5/39；发声F1 0→0.1053，但综合未见收益，暂保留原完整6000模型。不自动追加训练。[比较表](RESULTS.md) · [完整数据](FINAL_COMPARISON.json) · [完成验收](COMPLETION_ACCEPTED.json)。

用户2026-11-12要求先继续整体LoRA联合训练，并最终改为每300步保存。四机28卡：node12／node8／node4各GPU0–7，node11仅GPU0–3；batch28，每卡1条。目标为联合阶段总步数1000，包含已完成100步，继续900次更新；保存300、600、900和最终1000，不在300步插入评测。

## 模型与完整恢复

从`../p2_stage2_joint_full6000_trial28_20261112/run/step_000100.pt`完整恢复模型、Adam及28份RNG，SHA为`0c782b79e1420bac610acbcb3f727defb12044a31ea47d09ce82d18a3f65cf61`。原初始化为全LoRA完整6000＋规划SFT2752。

Qwen语言LoRA、Connector、原生LTX全attention／FFN LoRA、两路新增Whisper attention、Feature decoder及整段Event头的范围不改；AV=1、Event=0.05、双路Feature各0.025、规划CE=0.0001。Qwen LR1e-6，其余组第一阶段LR的1/3；不重做预热。最终500h／最新caption及原样本-noise前缀调度不改，100步前的2800条顺序逐项相同，101步接下一条。

`runtime.py`只接入预算、保存间隔、严格继承和恢复路径；前向、损失、梯度归约及更新仍用原生v2函数。初始恢复直接读已验收joint100，避免先加载后又覆盖第一阶段6000的训练参数；不复制36GB父检查点。原16条GPU梯度校准作为历史证据继承，明确标记未重新测量，不把继承当作新校准。`run/CONTINUATION_ACCEPTED.json`绑定父状态、最新v3安全评分、数据与源码。

`CONTINUATION_RUNTIME_ACCEPTED.json`已核对真实恢复后的28卡身份、Qwen参数顺序、数据、数值设置及初始化沿革与父组一致；最终评测的64条参考协议连接在CPU上通过。

## 状态、保存及恢复

- 当前进度：`run/status.json`；训练记录与TensorBoard：`run/metrics.jsonl`、`run/tensorboard`。
- 控制器：`LAUNCHER.json`、`CONTROLLER.json`；日志`launch.log`和`run/rank_logs`。
- 输出权重：`run/step_000300.pt`、`step_000600.pt`、`step_000900.pt`、`step_001000.pt`，各带完整标记和SHA。
- `run/STOP`在更新边界保存并退出；其他任务特别是node11 GPU4–7不受影响。

重新拉起前确认旧控制器／rank已退出；使用本目录`continue.py`，不要使用旧300步CLI。示例：

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
ROOT=/private/liqi_workspace/project_code/audio-driven-v5
"$PY" "$ROOT/experiments/p2_stage2_joint_full6000_trial28_20261112/reliable_entry.py" \
  "$ROOT/experiments/p2_stage2_joint_continue1000_20261112/continue.py" launch \
  --resume "$ROOT/experiments/p2_stage2_joint_continue1000_20261112/run/step_000300.pt"
```

## 1000步后的收益比较

`finish.py`每小时记录训练进度。1000完整保存、资源释放后自动四机28卡评测，与原0／100步同64条独立dev、seed137、30采样步及最新版v3评分比较；不重训基线。`FINISH_LAUNCHER.json`为接续控制器，输出`evaluation1000.log`、`run/evaluation/step_001000`，最终`FINAL_COMPARISON.json`和`RESULTS.md`。

沿用用户要求的当前最新版v3提示词、smile/brow补充及Seamless reader，版本清单在父目录`LATEST_EVALUATION_ACTIVE.json`。每个已更新Qwen检查点重新生成自己的计划、hidden和Gemma，旧0／100视频和评分按SHA复用。1000的新状态与父基线分属不同检查点配置，评测显式记录父基线出处，严格核对相同参考与检测协议，不伪造新run的step0。

部分类别unknown时，同时给出三组相同可判定集合上的原生F1及覆盖信息；单seed64条用于实际模型效果初筛，不表示全数据最优或统计显著。当前尚无1000步效果结论。
