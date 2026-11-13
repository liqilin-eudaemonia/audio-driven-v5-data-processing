**已自动切换并正式训练（2026-11-07）：** AV-only第6000步完整保存并校验，旧31rank已退出。AV＋Feature已从原始预训练初始化启动四机31卡，31rank最长样本反传和实际更新验收通过；三组梯度、组合损失正常，持续性能监控启用。每500步保存，node11 GPU7保留推理。当前步数见[status.json](status.json)，启动验收见[RUNNING_VERIFIED.json](RUNNING_VERIFIED.json)，[TensorBoard6008](http://172.29.252.37:6008)。

**已挂后台自动接续（2026-11-07）：** AV-only第6000步完整保存、检查点SHA通过且旧31rank退出后，确认所选31卡释放，再从原始预训练初始化启动本实验；node11 GPU7保留。控制器PID见CONTROLLER.json，阶段见QUEUE_STATE.json；切换等待／保存门槛测试通过。新实验启动后自动验收31rank最长样本反传、实际更新、损失及三组梯度，并持续记录速度和GPU利用率。每500步保存。[授权](QUEUE_AUTHORIZED.json) · [实时队列状态](QUEUE_STATE.json) · [TensorBoard（6008，等待新训练事件）](http://172.29.252.37:6008)。

**再次全量复核通过：** 284,218份caption实际SHA／原文及Qwen/Gemma缓存合同逐条一致；teacher全部成员与媒体身份／复用合同通过。35项测试通过，已修正31卡短测停止参数传递，并强化最长样本完整损失／梯度检查。GPU训练尚未启动。[完整复核说明](RECHECK.md) · [最终验收](RECHECK_ACCEPTED.json)。

# 最终500h：Qwen-only AV＋Feature 消融

已准备，未启动GPU训练；当前AV-only继续运行。本实验按用户要求复用原有AV＋Feature pipeline，仅更新到最终500h与对应最新完整caption。

- 数据：284,218段，500.001259126小时；train 281,060＋dev 3,158，test 0。与当前AV-only完全相同的选择、Qwen/Gemma文本缓存和AV缓存。
- 条件：冻结的原始、未做规划SFT的Qwen2.5-Omni-3B hidden＋最新完整caption的Gemma条件。不加speaker Whisper。
- 损失：`L = L_video_AV + L_audio_AV + 0.025 L_video_feature + 0.025 L_audio_feature`。Event权重显式为0，不读取Event标签、不执行Event头。
- 沿用旧AV＋Feature的EmbeddingDecoder与teacher语义，U由同一次Connector前向产生，供Feature Decoder和LTX共同使用。仅训练Connector、LTX text-attention LoRA和两路EmbeddingDecoder；冻结Qwen本体、LTX主干与历史占位Event头。
- 原始预训练初始化，seed=20261023，不继承联合模型或AV-only的训练权重。Feature Decoder初始化使用fork_rng，Connector／LoRA初始化保持与原流程一致。
- 31卡配置：node12/node8/node4各8卡，node11仅0–6卡；GPU7留给推理。batch31，完整一轮9,169步，尾部补齐21条；真实样本及noise ordinal与当前AV-only31逐项一致。每500步及最终步保存完整模型、Adam和31份RNG。
- 提速沿用已稳定方案：all全重计算、CPU预取2批、512MiB streaming FP64梯度归约、GPU Adam。独立rendezvous端口29854，不占用当前训练的端口。

## 数据验收

`preflight.json`：核对全量成员顺序／train+dev划分、最终选择与各缓存完整验收收据、teacher全量映射覆盖；实际读取32条Qwen/Gemma/AV/teacher缓存，检查完整caption原文和teacher时间几何。teacher沿用已验收的276,780条双路复用＋7,438条增量结果，无需重新提取。完整caption审计沿用 [RECHECK_FULL.json](../p2_current500h_av_only_20261107/RECHECK_FULL.json)。

`tests.xml`：新实验测试及已有AV-only回归；覆盖原生小尺寸LTX真实反传、Feature单独反传到Connector、原有初始化一致、完整Adam/RNG断点恢复、禁止Event／Whisper、全量调度一致与分布式归一化。

**目前仅完成CPU模型测试和真实缓存预检，尚未做本实验31卡最长样本显存／真实更新验收。** 启动入口会先在所有rank用最长样本执行完整AV＋Feature反传，保持optimizer不更新，然后正式训练。GPU启动验收通过后才能报告正式运行。

## 入口

使用 `/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`，设置 `PYTHONPATH=src:scripts`。

```bash
python scripts/p2_current500h_av_feature.py preflight --output experiments/p2_current500h_av_feature31_20261107
python scripts/p2_current500h_av_feature.py commands --output experiments/p2_current500h_av_feature31_20261107
```

四机命令已写入COMMANDS.json。待分配机器后显式调用launch；现已按用户指令加入AV-only6000步保存后自动接续队列；以下命令保留用于人工入口，队列运行期间勿重复launch：

```bash
python scripts/p2_current500h_av_feature.py launch --output experiments/p2_current500h_av_feature31_20261107 --logs experiments/p2_current500h_av_feature31_20261107/rank_logs
```

如需停止，在本目录创建STOP，训练会在当前更新结束后完整保存并退出；仅允许从本实验目录内且身份一致的完整checkpoint恢复。当前AV-only与本实验能够单独比较Feature增量；与历史联合模型相比还同时改变了Event／Whisper条件。
