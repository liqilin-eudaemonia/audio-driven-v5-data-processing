# 第二阶段联合LoRA v2：生成优先、小权重规划CE

2026-11-12，用户要求实现。独立包`src/p2_stage2_joint_v2/`，入口`scripts/p2_stage2_joint_v2_train.py`，评测worker`scripts/p2_stage2_v2_eval_worker.py`。旧`p2_stage2_joint`及现有训练／测评源码不改；旧源码快照在`before/`，逐文件SHA在`ORIGINAL_SOURCE.json`。

## 已准备候选

`candidate_full6000/config.json`：原始权重完整**全attention／FFN LoRA6000步**＋规划SFT2752。Qwen原语言层rank32/alpha64 LoRA继续训练；LTX原生48层1632处rank8/alpha8 LoRA；Connector、两路speaker Whisper cross-attention、Feature decoder和Event头随原目标更新，基座冻结。生成端严格从同一个6000检查点初始化，第二阶段Adam重新初始化。

- AV=1；Event=0.05；视觉和音频Feature各0.025。
- 候选规划CE系数0.0001；Qwen LR=1e-6，其余组第一阶段LR的1/3，预热20步。
- 原31卡模板，batch31，每卡1条；node12/node8/node4各0–7，node11 0–6，GPU7预留。未启动训练，也未抢占现有测评。
- 最终train+dev 284218段／500.001259h及最新完整caption；原始冻结Qwen H payload禁止用于联合训练，在线hidden保留梯度。AV/Gemma/Whisper/teacher缓存继续复用。
- 规划CE仅对批准SFT训练清单与500h交集计算，沿用190987条支持范围；其他样本继续生成训练。回答token及EOS按全局有效token数归一化。

## 启动校准

第一次launch在任何优化更新之前，执行已冻结`calibration_manifest.json`的16条训练样本：有声／全静音×<4／4–8／≥8秒共6个分层，包含最长11.95秒片段，全部来自批准规划标签交集。标签及样本索引固定SHA，独立64条dev验证集不用于校准。

在同一组Qwen LoRA参数上分别计算生成目标（AV+Event+Feature）及**未乘权重**的规划CE梯度；生成目标按16条样本平均，CE按这批有效token平均，Event按全局known计数。随后对梯度向量做真实全局SUM，再计算范数、方向余弦；不把各rank梯度范数相加。FP64使用16MiB CUDA临时分块，CPU承载两路完整诊断向量。

`CALIBRATION.json`记录全局及逐样本诊断、0.1/0.01/0.001/0.0001系数比例估算和建议系数。多系数只对同一份梯度做线性换算，不做四次前向。方向余弦是诊断，不作为必须同向的硬条件。

0.1–0.3是**参考比例带**、0.2是建议系数的目标比率，不是已验证最优。当前工程保护条件为：生成和规划梯度均非零且有限，已加权的**全局**规划梯度范数不超过生成项。失败则停在`awaiting_calibration_review`，不保存虚假的step0、不做更新，也不自动改配置；用户确认新系数后在新独立目录prepare。8–16条可通过prepare参数选择，默认16。

校准不调用optimizer.step，恢复本rank CPU/CUDA RNG，不保留参数.grad。31rank校准16条时15rank贡献零向量并参与相同归约，正式训练仍是31条／31卡。

## 短程训练及评测

校准通过后沿用最长样本两次真实更新／完整回滚检查，保存step0并退出。step0完成64条独立dev端到端评测后，显式恢复到100步，评测通过再显式恢复到300步；不自动扩大1000步。规划CE、LPIPS、误发声、格式错误和反应指标沿用旧工程策略。

每个检查点都重新用**当前Qwen**生成规划、hidden及对应Gemma条件，不能跨已更新Qwen的阶段共用这些缓存。评测仍区分计划执行诊断和GT事件类别一致性，不冒充人工语境正确率。

恢复同时核对当前配置、源码、校准报告、完整Adam/RNG及评测门槛；不能从旧v1 step0直接恢复本v2，也不能改系数后继续冒用旧身份。

## 命令

以下为入口说明，未执行GPU命令：

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
ROOT=/private/liqi_workspace/project_code/audio-driven-v5
CFG=$ROOT/experiments/p2_stage2_joint_balance_v2_20261112/candidate_full6000/config.json
OUT=$ROOT/experiments/p2_stage2_joint_balance_v2_20261112/candidate_full6000
cd "$ROOT"

# 当前候选已prepare和preflight；首个launch仅校准、最长样本检查并保存step0。
"$PY" scripts/p2_stage2_joint_v2_train.py commands --config "$CFG"
"$PY" scripts/p2_stage2_joint_v2_train.py launch --config "$CFG" --logs "$OUT/launch_step0"
# 评测在node11 GPU7；需该卡可用。
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/p2_stage2_joint_v2_train.py evaluate --config "$CFG" --step 0
"$PY" scripts/p2_stage2_joint_v2_train.py launch --config "$CFG" --resume "$OUT/step_000000.pt" --logs "$OUT/launch_to100"
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/p2_stage2_joint_v2_train.py evaluate --config "$CFG" --step 100
"$PY" scripts/p2_stage2_joint_v2_train.py decide --config "$CFG" --step 100
# 只有continue_to_300才允许接续。
"$PY" scripts/p2_stage2_joint_v2_train.py launch --config "$CFG" --resume "$OUT/step_000100.pt" --logs "$OUT/launch_to300"
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/p2_stage2_joint_v2_train.py evaluate --config "$CFG" --step 300
"$PY" scripts/p2_stage2_joint_v2_train.py decide --config "$CFG" --step 300
```

若prepare其他候选，必须新output，并显式给parent-config、checkpoint、adapter、validation；可通过`--planning-weight`、`--calibration-samples`设置候选值，prepare后重新preflight。

## 验证边界

CPU测试覆盖原生小型Qwen+PEFT+LTX全LoRA的梯度、严格完整恢复、两进程真实更新与单进程累积逐值一致、四进程含空rank的全局梯度SUM及方向、校准不修改参数和RNG、失效报告拒绝、评测门槛和CLI。完整22B仅meta检查，1632处／84221952个LoRA参数，Whisper全参数2779450368，4809个生成端状态键／shape／dtype与真实6000权重严格一致。

真实32条缓存回读和16条校准输入检查另见验收记录。**没有运行完整规模GPU校准／训练，尚无实际校准比率、显存或步速结论。0.0001仍是候选，不是已验证最优值。** 最终完成范围见`CODE_ACCEPTED.json`。
