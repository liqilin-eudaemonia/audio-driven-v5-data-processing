**当前恢复运行：[四分之三梯度重算＋duration512＋每500步保存](duration_memorysafe_save500/README.md)。原运行在第162步长片段发生停顿，从最近完整第70步恢复。**

**当前运行已迁移：[立即时长分组＋每500步保存](duration_now_save500/README.md)。从第71步起生效，原70步全部保留。下文保留初次启动记录，当前配置以子目录config.json为准。**

# 当前500h：原始Qwen H＋AV＋Feature（0.025／0.025）

用户在2026-11-02确认：本轮先使用未经规划SFT的原始Qwen H，视觉和音频Feature系数各改为0.025。新SFT H保留供后续独立对照。56项CPU回归、3条真实缓存读取和全量285,638条preflight均通过。已按用户要求启动四机32卡，前3步属于正式训练预算；完成检查点和梯度验证后，控制器自动严格续跑至8,927步。实时状态见 `queue/state.json`、`status.json` 和 `controller.log`。

## 实际配置

- 配置：[p2_j_guided_av_feature_base025_20261102.json](../../configs/p2_j_guided_av_feature_base025_20261102.json)。
- 损失：`L_AV + 0.025*L_video_feature + 0.025*L_audio_feature`，等价于 `L_AV + 0.05*mean(L_video_feature,L_audio_feature)`；没有再次除以2。
- `qwen_hidden_source=base_pre_sft`、S1、`qwen_lora.enabled=false`。读取原始Qwen2.5-Omni-3B的H，训练时Qwen冻结，不加载最新规划adapter。
- 原H、原文Gemma和AV latent：`../p2_j_train500h_cache_20261101/`；guided H为该目录的 `cache/input_plan/`。输入沿用speaker音频＋最终listener裁框首帧＋原打标描述。
- 独立冻结teacher：`../p2_j_teacher500h_node8_20261102/production/`。其视觉与音频目标已使用最终裁框和净化listener音轨；全量285,638条验收通过。
- 新SFT H：`../p2_j_train500h_sft_hidden_20261102/`，完整保留，本轮不读取。
- Event head冻结并跳过计算；配置保留的event=0.05不参与本轮优化。Feature更新自身decoder与共享Connector；AV更新Connector及LTX LoRA。

继承此前AV对照的500h成员、原文、目标、seed、Connector 1024／4层／16头、global batch32、8,927步预算、学习率、alternating梯度重算和duration512时长分组计划。保存间隔沿用该AV配置的200步。本轮没有变更已完成训练的配置或权重。四机32卡前3步完整反传、检查点SHA及恢复身份已通过，见 `GPU_VALIDATION.json`：损失有限、Connector/Feature/LTX LoRA梯度有效，Event梯度为0；rank0峰值预留显存107.58GiB。控制器已接续剩余预算。该显存值是rank0，不代表所有rank峰值。

## 代码与验证

- `src/p2_matrix16/config.py`：支持新0.025／0.025及历史0.1／0.1；后续新生成matrix配置默认0.025，已有历史JSON不重写。
- `src/p2_clip_events/backend.py`：接入已验收teacher，绑定来源与完成协议，读原始H并校验teacher几何；全量preflight包含teacher缓存。
- `src/p2_matrix16/model.py`：Feature沿用原生EmbeddingDecoder；本轮显式关闭Event probe计算和参数更新。
- `scripts/p2_j_duration_train.py`：复用时长分组调度，增加同一源码身份下的preflight入口。
- `cpu_tests.xml`：权重算式与梯度、原始H读取、teacher来源拒绝及已有训练回归。
- `reader_check.json`：实际500h清单首／中／末3条，backend读取H与原始input_plan逐值一致，teacher及AV联合读取通过。
- `preflight.json`：全量检查完成后生成，包含冻结的数据、teacher和训练源码身份；`preflight.log`与`preflight_progress.json`记录进度。

新实验应从原始LTX＋新初始化Connector/Feature decoder开始，与已有基线按相同预算比较。不能把新增Feature结构及改权重当成旧实验的严格断点续训；历史运行的严格恢复仍需其对应源码快照。

## 命令与读取

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
CONFIG=configs/p2_j_guided_av_feature_base025_20261102.json
$PY scripts/p2_j_duration_train.py preflight --config "$CONFIG" --processes 32
# 后续启动分布式训练时，torchrun调用此入口，以保留原时长分组：
# scripts/p2_j_duration_train.py train --config "$CONFIG"
```

四机实际命令保存在 `launch_commands.json`；使用 node17、node8、node12、node4 各8卡。控制器为 `scripts/p2_j_feature_launch.py`，集群配置为 `configs/p2_j_av_feature_base025_32gpu_20261102.json`。初始3步与后续训练日志分别位于 `queue/initial3/`、`queue/continue_epoch/`。前3步保留完整模型、优化器、32个rank随机数状态，不重置步数。不要重复启动已有控制器。不要使用纯AV专用入口 `p2_j_av_only.py`，该入口明确拒绝Feature teacher。

```python
from p2_matrix16.config import load
from p2_clip_events.backend import Dataset

c = load("configs/p2_j_guided_av_feature_base025_20261102.json", formal=True)
sample = Dataset(c).sample(0, device="cpu")
# sample['condition']: 原始未SFT Qwen H
# sample['teacher']: 独立冻结的真实listener视觉/音频特征
# sample['text'] / sample['av']: 原文Gemma和原训练目标缓存
```

## 本次启动与恢复

控制器PID记录在 `controller.pid`，状态为 `queue/state.json`。启动命令（已有任务运行时不要重复执行）：

```bash
/private/liqi_workspace/project_code/audio-driven-v5/../Omni-LTX-stage2/.venv/bin/python scripts/p2_j_feature_launch.py \
  --config configs/p2_j_guided_av_feature_base025_20261102.json \
  --cluster configs/p2_j_av_feature_base025_32gpu_20261102.json
```

前3步完整检查点 `step_000003.pt` 已验收；自动恢复使用相同32卡、配置、样本计划、优化器和各rank随机数状态。后续每200步保存完整检查点，并在第8927步保存最终检查点。故障时以 `.complete.json` 配对且SHA有效的最近检查点恢复，使用 `launch_commands.json` 的 continuation 命令修改 `--resume` 路径，另选日志目录；先确认旧控制器已退出。`queue/continue_epoch/completed.json` 仅表明进程退出，完整训练须同时检查 `status.json` 的 `complete=true` 和 `step=8927`。
