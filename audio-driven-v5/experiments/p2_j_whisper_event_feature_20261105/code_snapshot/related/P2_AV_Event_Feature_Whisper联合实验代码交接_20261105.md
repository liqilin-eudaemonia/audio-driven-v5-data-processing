# AV＋Event＋Feature＋Whisper 联合实验

本次新增独立代码与配置，尚未启动训练。目标是在现有原始 Qwen H＋AV＋Event＋Feature 上，加上此前已经训练、评测过的 speaker Whisper 双路 cross-attention。固定原有损失权重，不把 Whisper 当作新增损失。

## 1. 条件、结构与损失

```text
原始未 SFT 的冻结 Qwen H ── Connector ── U ──┬─ LTX 原有文本 cross-attention
标注自然语言 ── 冻结 Gemma ────────────────┘                 │
speaker audio ── 冻结 Whisper-large-v3 ── adapter ── 独立音频/视频 cross-attention
                                                           │
listener 首帧 + 目标 AV latent ──────────────────────────── LTX ── AV loss

U ── clip Event heads ── 19 类多标签 BCE
U ── video/audio Feature decoders ── 冻结 listener teacher 特征监督
```

总目标：

`L = L_AV + 0.05 L_Event + 0.025 L_Feature_video + 0.025 L_Feature_audio`

- **AV**：保持原有视频与音频生成目标。
- **Event**：沿用前几轮真实对照的片段级多标签监督，16 类视觉＋3 类音频。对 U 做有效 token 均值池化，再由 LayerNorm／Linear／GELU／Linear 分类。没有新增事件时间定位头；未知标签和类别有效性继续使用已有 mask、正例权重与弱负例权重。
- **Feature**：沿用原有两路 EmbeddingDecoder 的 flow-matching 特征重建损失，teacher 是冻结模型从真实 listener 媒体提取的目标特征。不是简单地把 Qwen H 与 teacher 做 MSE。
- **Whisper**：输入只有同一个 pair 的 speaker audio，冻结 large-v3 编码器，使用已有 1280 维缓存。LayerNorm＋Linear adapter 后进入 LTX 每层的独立视频、音频残差 cross-attention；输出投影从零初始化。沿用共同秒数时间 RoPE，时间原点为 pair 开始，保留静音与完整时长。
- **梯度**：Event／Feature 更新各自的头和共享 Connector；Whisper adapter、Whisper cross-attention 和 LTX 文本 LoRA 通过 AV 更新。原始 Qwen、Whisper encoder、Gemma、LTX 主干和 teacher 都冻结。推理时不读取真实 listener teacher 或 Event 标签。

五组优化器参数：Connector、Event heads、Feature decoders、LTX text LoRA、Whisper adapter＋两路 attention。对应学习率依次为 `3e-5 / 1e-4 / 3e-5 / 1e-5 / 3e-5`。

## 2. 数据和缓存版本

为对齐刚完成的十二组实验，本实验使用其原有 **285,638 条 train／约 500h**，不切换到 11 月 5 日重新筛选的 train＋dev 284,218 条版本。已有最终裁框、listener 官方时间并集净化音轨、指导文本及来源契约由缓存读取器严格校验，训练不重新解码媒体。

| 内容 | 复用位置（相对项目根目录） |
|---|---|
| 冻结原始 Qwen H、Gemma、AV VAE | `experiments/p2_j_train500h_cache_20261101/` |
| 片段级 Event 标签 | `experiments/p2_j_clip_labels_20261101/` |
| 冻结 listener Feature teacher | `experiments/p2_j_teacher500h_node8_20261102/production/` |
| 冻结 speaker Whisper | `experiments/p2_j_speaker_whisper_cache_20261104_prefetch_v2/` |
| 新实验结果目录 | `experiments/p2_j_whisper_event_feature_20261105/` |

**不需要补充缓存。** 数据读取按样本 ID 核对 Event source digest、H／AV 来源、teacher 来源、Whisper source 与 encoder 身份。私有 LTX 目录中的 Gemma 路径仅允许已有、字节完全相同的三处代码迁移；不放宽其他来源校验。

## 3. 新代码与原代码边界

- [统一命令入口](../scripts/p2_j_whisper_event_feature.py)
- [模型与梯度归属](../src/p2_whisper_event_feature/model.py)
- [联合缓存读取](../src/p2_whisper_event_feature/data.py)
- [训练、全局归一化与最长样本检查](../src/p2_whisper_event_feature/training.py)
- [完整检查点与严格恢复](../src/p2_whisper_event_feature/checkpoint.py)
- [推理加载与 CFG 采样](../src/p2_whisper_event_feature/inference.py)
- [实验配置](../configs/p2_j_qwen_whisper_event_feature_20261105.json)
- [四机 32 卡配置](../configs/p2_j_qwen_whisper_event_feature_32gpu_20261105.json)
- [回归测试](../tests/test_p2_whisper_event_feature.py)

新增模块 `src/p2_whisper_event_feature/` 复用原联合损失算法及 Whisper 分支实现。没有修改 `src/p2_whisper_av/` 或其私有 LTX 副本，避免破坏旧 Whisper 权重的源码身份；不能用旧纯 AV Whisper 的模型加载器加载新检查点。

新模型与原联合损失模型的共享参数（含 Feature decoder），以及与纯 AV Whisper 模型的共享参数，初始化逐值一致。完整检查点保存模型、Adam、每个 rank RNG、数据与代码身份。只接受本实验自身的严格恢复，不从旧实验继续训练，以保证消融起点一致。

## 4. 沿用的加速与训练预算

- 四机 32 卡，global batch 32，每卡一条，完整预算 **8,927 步**，每 **500 步**保存完整权重和最终步。
- 同一个 seed `20261023`、同一个样本及噪声 ordinal、同一个 duration512 计划；为保持旧 Whisper 对照完全一致，第 1–2 步保留其随机顺序，之后按时长分组。
- 全层 gradient checkpointing、BF16 autocast、256 MiB streaming FP64 SUM 通信、GPU 驻留 Adam；沿用原数值与 NCCL 环境配置。
- AV／Feature 按全局样本数平均；Event 按全局已知样本数逐类归一化后取有效类均值。梯度通信执行 SUM，不能再额外除一次 world size。
- 主训练参数与 Event heads 分别按原规则裁剪梯度范数 1。
- 新增联合损失会增加计算量与显存；此前纯 Whisper 的步时不能直接当成本组实测速度。

启动前会对全局最长片段运行两次完整 forward／backward／Adam，包含 Event／Feature 和真实 Adam 状态，汇总最大 rank 显存峰值。结束后恢复初始参数、优化器和 RNG，不计入正式训练步数。该检查失败就退出，不继续全量训练。

## 5. 命令

在项目根目录执行，Python 为 `../Omni-LTX-stage2/.venv/bin/python`：

```bash
# 查看配置，不启动训练
../Omni-LTX-stage2/.venv/bin/python scripts/p2_j_whisper_event_feature.py inspect

# CPU 实际读取短、中、长样本，同时核对实际时长调度计划
../Omni-LTX-stage2/.venv/bin/python scripts/p2_j_whisper_event_feature.py verify-cache --limit 3

# 开训前必做：全量缓存收据检查，默认 8 个 CPU 进程
../Omni-LTX-stage2/.venv/bin/python scripts/p2_j_whisper_event_feature.py preflight --processes 8

# 仅输出四机启动命令
../Omni-LTX-stage2/.venv/bin/python scripts/p2_j_whisper_event_feature.py commands

# 确定开训后：先全量 preflight，再四机联合训练
../Omni-LTX-stage2/.venv/bin/python scripts/p2_j_whisper_event_feature.py launch \
  --logs experiments/p2_j_whisper_event_feature_20261105/launch32

# 同一配置、同一 world size 严格恢复
../Omni-LTX-stage2/.venv/bin/python scripts/p2_j_whisper_event_feature.py launch \
  --resume experiments/p2_j_whisper_event_feature_20261105/step_000500.pt \
  --logs experiments/p2_j_whisper_event_feature_20261105/resume500
```

集群配置写有 node17／node12／node4／node8 各 8 卡，其中 node8 为 `172.29.104.237`。仅生成了命令，未连接机器或启动进程；实际开训前应核对资源和地址。已有输出目录的 `STOP` 文件会阻止启动；训练中检测到 STOP 时保存完整当前步再退出。

后续推理使用 `p2_whisper_event_feature.inference.load_model()` 和 `.sample()`。采样复用十二组对照的原始 `p2_listen.runtime.guided_sample`，保持相同 FP32 视频初始噪声，CFG 正负路径均保留 Qwen 和 speaker Whisper 条件。当前十二组评测清单未改动，待新权重产生后再注册新组。

## 6. 已验收与尚待实测

- **65 项 CPU 回归通过**：联合模型、旧纯 Whisper 模型、梯度执行、时长调度和片段 Event 回归；包含两次真实 tiny LTX 更新、五组梯度、冻结主干、全层重算、跨 rank 的损失归一化、严格恢复复现下一步、最长样本检查后参数／Adam／RNG 恢复、推理不使用 teacher／Event、CFG 两路时间条件和首帧保持。
- 实际读取 **3 条缓存样本**：2.0s、3.115s、11.95s；H、Gemma、AV、19 类 Event、teacher 与 speaker Whisper 均通过来源／形状校验，实际时长调度计划通过。报告 [CPU_CACHE_CHECK.json](../experiments/p2_j_whisper_event_feature_20261105/CPU_CACHE_CHECK.json)。
- 旧 Whisper 模块 SHA256 未变；见 [LEGACY_SOURCE_CHECK.json](../experiments/p2_j_whisper_event_feature_20261105/LEGACY_SOURCE_CHECK.json)。
- **尚未执行**本实验的 285,638 条全量 preflight、真实 GPU 最长样本测试、32 卡 NCCL 与完整模型训练。CPU canary 回归使用 tiny LTX 和模拟显存接口，不能替代真实 GPU 显存验收。
- 本地代码快照与哈希：[code_snapshot](../experiments/p2_j_whisper_event_feature_20261105/code_snapshot/)。未发起 GitHub 推送。
