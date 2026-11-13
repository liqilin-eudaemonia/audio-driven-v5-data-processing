# Qwen H＋Whisper 双路条件，仅 AV loss

2026-11-04。用户已确认：保留现有 Qwen H 和 Gemma 文本，额外加入旧 Whisper 方案的独立音频／视频 cross-attention 残差。代码已实现；CPU 回归及真实样本记录见下文。尚未提取全量 Whisper 缓存，未启动正式训练，完整 GPU 显存／吞吐和多机 NCCL 仍待实测。

## 1. 实验与基线

- 基线：[仅 AV 配置](../configs/p2_j_av_only_20261101.json)。新配置：[Qwen＋Whisper 配置](../configs/p2_j_qwen_whisper_av_20261104.json)。
- 数据仍为同一份 **285,638 条／500h**，相同最终 listener 修正版裁框、净化后的 listener 音频目标、标注文本、划分和时长分组顺序。
- Qwen H 使用 **原生未经过规划 SFT 的 Qwen2.5-Omni-3B** 缓存。Qwen 原输入及提取方式不变：speaker audio＋listener 首帧＋标注自然语言计划。
- 原 H→Connector→U 与 Gemma 条件继续拼接后进入 LTX 原生文本 cross-attention，不被替换。
- 新增：speaker audio→冻结 Whisper-large-v3 Encoder 最后一层→LayerNorm＋Linear→各层独立视频／音频 cross-attention 残差。Whisper 不运行 ASR decoder。
- LTX 内部复用历史命名 `qwen_attn2`／`audio_qwen_attn2`；**该新增分支的 K/V 是 Whisper，原 Qwen U 仍在原生 `attn2`／`audio_attn2` 中**。
- 该对照检验“新增 Whisper 条件与分支的整体收益”；新增参数量也是实验差异，不能据此单独断言 Whisper 编码器更强。

未选方案：用 Whisper 替换 H、仅拼接 Whisper 与文本、只用固定提示。没有引入旧 Bernini 的阶段切换、Feature decoder 或旧训练权重。

## 2. 时间、音轨和训练范围

Whisper 使用与基线 Qwen 相同的 **speaker** 独立归一化音轨和源片段起止，经固定重采样至 16kHz。保留前导静音，pair 起点为 0，不裁走静音，不使用 listener 官方时间掩码门控 speaker，不读 listener 目标音频来构建 Whisper 条件。

特征为 FP32 `[1,T,1280]`、50Hz、20ms 网格；30秒编码窗、两侧最多2秒上下文、26秒互不重叠保留核心，长输入无截断、无重复时间 token。全零 speaker 波形仍是有效观测，不当作缺失输入。

使用旧 W1：Whisper K 和 LTX Q 按共同秒数构建时间 RoPE，时间尺度沿用 LTX 音频预处理器；视频 Q 只用时间坐标。保留 padding mask。离线可观察整段 speaker 音频，**不是流式因果模型**。

| 模块 | 训练状态 |
|---|---|
| 原 Connector，1024宽／4层／16 heads | 可训练，沿用基线学习率 |
| 原生文本 attention LoRA，rank8／alpha8 | 可训练，沿用基线学习率 |
| Whisper Adapter＋每层新增两路 attention | 可训练，FP32 master，学习率3e-5 |
| Qwen、Whisper Encoder、LTX基座、Gemma、VAE | 冻结 |
| Event／Feature | 不运行、不计损失；Event头仅保留初始化和状态键兼容，冻结且不进优化器 |

新增 attention 的输出投影零初始化；其构造使用独立 RNG 上下文，保持原 Connector／LoRA 初始化。损失严格为现有 masked 视频 AV loss＋音频 AV loss。配置保留矩阵格式的 Event／Feature 权重字段，但该入口不会使用它们。

保持全局 batch32、种子、时长分组和8927步预算；每500步及最终步保存，6,000／8,927步用于后续对照。初始权重与仅 AV 基线相同，不继承已训练的 AV 检查点。恢复必须使用本实验自己的完整检查点，绑定代码、配置、数据、缓存、world size和各rank RNG。

## 3. 代码与隔离

统一入口：[scripts/p2_j_whisper_av.py](../scripts/p2_j_whisper_av.py)。

**仅提取 speaker audio 缓存的专用入口：** [scripts/p2_j_speaker_whisper_cache.py](../scripts/p2_j_speaker_whisper_cache.py)。只提供 `prepare / extract / finalize / commands`，没有训练或切换音轨选项；读取的是冻结样本的 `item.speaker`，并校验 speaker／listener 身份、原始来源和片段时间。其专用5项CLI回归通过，未启动全量提取。

| 文件 | 职责 |
|---|---|
| [config.py](../src/p2_whisper_av/config.py) | 仅AV／原始H约束、基线比较、来源身份 |
| [features.py](../src/p2_whisper_av/features.py) | 冻结Whisper、批量编码、长音频时间网格 |
| [cache.py](../src/p2_whisper_av/cache.py) | speaker来源、分rank续跑、完整性验收 |
| [compat.py](../src/p2_whisper_av/compat.py) | 三个逐字节相同的Gemma文件重定位兼容 |
| [data.py](../src/p2_whisper_av/data.py) | 原缓存与新增Whisper联合读取，固定train顺序 |
| [model.py](../src/p2_whisper_av/model.py) | 双路残差、原条件保留、参数归属及AV forward |
| [training.py](../src/p2_whisper_av/training.py) | preflight、时长分组、FP64梯度累积和SUM归约 |
| [checkpoint.py](../src/p2_whisper_av/checkpoint.py) | 新schema、哈希完成标记、严格恢复 |
| [inference.py](../src/p2_whisper_av/inference.py) | 新权重读取、CFG两侧均保留Qwen U及Whisper |

私有 LTX 源码在 [external_sources/p2_whisper_av](../external_sources/p2_whisper_av/UPSTREAM.json)，从当前实际安装版本复制111个Python文件，只在 Modality／TransformerArgs／两路attention调用处加入时间信息。没有修改共享安装的 LTX，也没有修改现有训练模块。入口在任何 LTX import 前激活私有副本；同一进程已加载别的副本时拒绝继续。

Gemma 缓存依赖记录包含源码绝对路径。私有副本会改变路径；兼容器只允许 `embeddings_processor.py`、`embeddings_connector.py`、`tokenizer.py` 在 SHA256 与复制记录完全一致时映射回旧路径，随后仍执行完整依赖比较及原缓存 payload 哈希校验。内容变化不能利用重定位绕过校验。

## 4. 数据与缓存位置

- 原缓存：`experiments/p2_j_train500h_cache_20261101/`；Qwen H、Gemma、AV latent 均复用。
- 新 Whisper 缓存：`experiments/p2_j_speaker_whisper_cache_20261104_prefetch_v2/`，待全量提取；原v1目录保留协议和接续队列。[新版提速配置、验证及迁移说明](P2_Whisper缓存流水线提速与接续_20261104.md)。
- 新训练输出：`experiments/p2_j_qwen_whisper_av_20261104/`，未开训。
- 真实单条代码检查：[real_sample.json](../experiments/p2_j_qwen_whisper_av_codecheck_20261104/real_sample.json)。该单条缓存不是全量训练缓存。

不保存新 WAV 或视频。新增 FP32 Whisper hidden 仍需磁盘：500h×50Hz×1280×4字节约 **460.8GB／429GiB**，另有少量向上取整、时间和校验元数据。不要将“复用旧缓存”理解成新增 Whisper 不占存储。

## 5. 使用顺序

**2026-11-04 全量缓存已挂自动接续：** 等当前1,107,323条打标完成固定清单验收并释放四机32卡，自动提取当前500h／285,638条的speaker Whisper，结束执行全量缓存验收；不接续训练。[队列说明、状态和停止方式](../experiments/p2_j_speaker_whisper_cache_20261104/queue/README.md)。已有后台队列时，不要另开同一套提取任务。

从 `audio-driven-v5` 执行，Python 沿用 `/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python`。

只做 speaker 缓存时可用专用入口（与下方统一入口调用相同的缓存实现）：

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
CFG=configs/p2_j_qwen_whisper_av_20261104.json
$PY scripts/p2_j_speaker_whisper_cache.py prepare --config "$CFG"

# 仅打印4机32卡提取命令；不启动任务。默认独立rendezvous端口29844。
$PY scripts/p2_j_speaker_whisper_cache.py commands --config "$CFG" \
  --batch-size 16 --read-workers 4 --cpu-threads 2

# 按打印出的各机命令完成提取后，再执行全量校验。
$PY scripts/p2_j_speaker_whisper_cache.py finalize --config "$CFG"
```

每rank读取 `RANK/WORLD_SIZE` 分片，同一份冻结清单覆盖32卡；每卡用 `LOCAL_RANK` 选GPU。支持批量编码、并行读取、断点复用和STOP；已存在缓存仍按来源和payload哈希核对。此入口不会加载Gemma、Qwen、listener PCM或视频，不会新建WAV。

```bash
PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
CFG=configs/p2_j_qwen_whisper_av_20261104.json

# 只冻结来源和模型身份，不开始提取或训练。
$PY scripts/p2_j_whisper_av.py cache-prepare --config "$CFG"

# 单机示例；4机时使用torchrun的nnodes/node_rank/master_addr参数。
# 自动使用RANK/WORLD_SIZE/LOCAL_RANK，32rank共同写共享缓存。
# 编码batch16、读取线程4；已接跨批次预取和后台写盘。
$PY -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/p2_j_whisper_av.py cache-extract --config "$CFG" \
  --batch-size 16 --read-workers 4

# 必须所有rank完成后执行；任何缺失／损坏都会失败，不发布成功。
$PY scripts/p2_j_whisper_av.py cache-finalize --config "$CFG"
$PY scripts/p2_j_whisper_av.py preflight --config "$CFG" --processes 16

# 只打印四机命令，不启动。
$PY scripts/p2_j_whisper_av.py commands --config "$CFG"

# 获得实际空闲资源后，先2步GPU验收，完成后严格恢复同一运行。
$PY scripts/p2_j_whisper_av.py launch --config "$CFG" \
  --stop-after 2 --logs experiments/p2_j_qwen_whisper_av_20261104/launch_first2
$PY scripts/p2_j_whisper_av.py launch --config "$CFG" \
  --resume experiments/p2_j_qwen_whisper_av_20261104/step_000002.pt \
  --logs experiments/p2_j_qwen_whisper_av_20261104/launch_continue
```

正式配置：[四机配置](../configs/p2_j_qwen_whisper_av_32gpu_20261104.json)。它只描述节点与既有网络参数，不会自动抢占或停止其他任务。全局batch32；不足32卡时需要world size整除32。显存配置尚未通过真实新架构验收。

`cache-extract --limit N` 只用于定位问题，不会把部分结果视为全量成功。独立试点可通过新配置指定新的 `whisper_cache` 并用 `cache-prepare --sample-ids <JSON数组>`；正式训练拒绝缺少任何train成员或成员顺序不同的缓存。

## 6. 验收、资源及评测边界

回归入口：[tests/test_p2_whisper_av.py](../tests/test_p2_whisper_av.py)。记录位于 [代码检查目录](../experiments/p2_j_qwen_whisper_av_codecheck_20261104/README.md)。覆盖零残差与原初始化、两次AV更新、冻结范围、padding无效性、W1时间偏移效应、梯度重计算、CFG双路条件、长窗覆盖、真实tiny Whisper批量／串行、缓存损坏、原始speaker重采样与严格恢复。

真实片段 `improvised_V00_S0822_I00000920_local_1ebda932114f`：Whisper `[1,156,1280]`、0–3.115秒，原 Qwen H `[1,4,481,2048]`；原Gemma、AV缓存可读，未新增媒体文件。该检查在CPU执行，不代表22B模型GPU完整反传或多机验收。

旧48层双路全参数结构约新增 **27.8亿参数**。FP32权重、梯度、Adam状态约41.4GiB；本训练器沿用FP64梯度累积，还会有额外累积／转换缓冲，不能把41.4GiB当作总显存。需要用实际长短样本检查显存、保存恢复与吞吐后，确定重计算／内存策略；不为压显存静默改变为LoRA、删层或移除音频分支。

后续比较沿用 ResponseNet200、相同seed／采样／规划器与文本来源。训练使用标注描述；自主推理时必须用同一规划器生成的计划，不能只给新实验使用GT描述。当前提供新架构的权重读取与采样接口，**未自动接入旧批量评测队列，也未启动评测**。
