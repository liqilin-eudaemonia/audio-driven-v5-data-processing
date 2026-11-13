# P2：OmniResponse 音视频与可选文本评测

2026-10-26。用户已确认默认 **FVD-32：32 帧连续窗口**。本次把本地 [OmniResponse/evaluation](../../OmniResponse/evaluation/README.md) 的可用指标接入 P2，提供独立入口、数据绑定、缓存、报告和配对比较。训练源码与训练 Python 环境未修改；E0 继续运行。正式 dev/test 数据仍待用户提供，未启动正式生成或效果评测。

## 已准备好的指标

| 指标 | 本版计算方式 | 输入与边界 |
| --- | --- | --- |
| FVD-32 ↓ | 参考 TorchScript I3D、一次 bilinear 224×224 缩放、uint8 RGB、rescale=True / resize=False；population covariance、原 eps=1e-6 sqrtm | 25 FPS、32 帧连续窗口；sample_id + crop_seed 固定起点，GT、所有生成 seed、E0/E1 使用同一窗口。每个生成 seed 单独形成样本分布 |
| FVD-256 ↓ | 同一实现，显式 `--fvd-frames 256` | 短于 256 帧的 GT 标不适用；报告覆盖率，不补帧，也不与 FVD-32 混比 |
| FD ↓ | 参考 `compute_fd_features.py`，拼接所有提供的 T×C 特征，sample covariance | 需明确 feature_schema 与同语义特征；不是图像 FID。未新增 3DMM 提取器 |
| UTMOSv2 ↑ | 原离线模型、原模型预处理/TTA、固定随机种子、逐段 MOS 后等权平均 | 使用独立 listener 音频，适配层统一为 16 kHz 单声道 PCM16、无响度归一化；分别输出 GT 与生成分数。RMS<1e-5 标静音不适用。模型内部有 trim/tile/TTA，不声称内部完全不补音频 |
| LSE-D ↓ / LSE-C ↑ | 原 SyncNet 模型、MFCC、5 帧视频/20 帧 MFCC、vshift=15、参考 calc_pdist | 对应 listener 音视频；参考总体均值按有效人脸轨迹等权，另给每片段等权均值、offset 与覆盖率 |
| METEOR、ROUGE-L、BERTScore ↑ | 参考 METEOR、beta=1.2 ROUGE-L；相同本地 RoBERTa-large、第 17 层、无 baseline rescale | 可选，需要真实生成/转写文本与 listener GT 文本；不自动 ASR |
| Distinct-1 / 2 ↑ | 参考代码的 corpus-level unique ngrams / all ngrams | 可选，在明确支持的配对文本集合上统计，不能平均逐句 Distinct |

调用或适配的是参考实现的公式与模型。FVD-32 的窗口、共享裁窗、同步预处理及适用集是本项目明确规定的协议，**不能把这些结果声称为论文数值的完全复现**。原目录没有 PSNR、SSIM、图像 FID、LPIPS、Var、rPCC、DI-Sync；本次没有用名字相似的其他指标代替它们。

Fréchet 在小样本、低秩协方差时可能出现很小的负舍入误差。保留 `raw_reference_value`，只有负值幅度不超过 `1e-6 × max(1, covariance traces + mean-distance²)` 才归零；超过容差直接报错。每项记录是否修正及容差。FVD 本身仍受样本量影响，两段自配对只验证工程链路。

## SyncNet 的必要适配

参考 `combine_av.py` 没有显式选轨；这里使用 `-map 0:v:0 -map 1:a:0`，强制选择独立 listener 音频。拒绝起点偏移和明显时长不匹配，不用 `-shortest`。视频必须通过真实帧/PTS 校验后才进入 tracker。

真实测试发现上游转 AVI 的隐式 MP3 编码会引入音频延迟，把原 50 帧扩成 52 帧。适配器只在当前子进程替换这条 ffmpeg 调用：保留 PCM 音频、真实帧数，并在跟踪后再次核对视频几何/时间轴。参考源码保持原样。

本项目短片较多，`min_track=5` 要求至少 6 次检测；13 点中值平滑改为不超过真实轨迹长度的奇数窗口。默认要求单 listener，多个同时出现的人脸不强行选一个。评分保留 vshift=15；窗口数少于 31 时标 `low_support`，LSE-C 仅作为原始诊断值，不宣称具有充分置信支持。无有效人脸、静音、过短轨迹都明确保留原因。

CPU 模式会对子进程设置 `CUDA_VISIBLE_DEVICES=''`，避免 tracker/BERTScore 自动使用训练 GPU。GPU 模式只接受显式 `cuda:N`，启动前确认卡上没有其他计算进程。

## 环境与源码固定

入口：[scripts/p2_omni_eval.py](../scripts/p2_omni_eval.py)。实现：[src/p2_omni_eval](../src/p2_omni_eval/)。输入示例：[examples/p2_omni_eval](../examples/p2_omni_eval/)。

本机直接复用已有两个独立评测 Python、UTMOS/I3D/BERT/SyncNet/S3FD 权重、NLTK 语料和辅助模型缓存，没有向训练环境安装包。实际路径、环境版本、参考源码和所有模型文件 SHA256 位于 [runtime lock](../configs/p2_omni_runtime.lock.json)。还固定 ffmpeg/ffprobe 路径、二进制哈希与版本。离线环境设置 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE，METEOR 缺少资源时直接报错，不回退另一种 tokenizer。

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
P2_PY=/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python
"$P2_PY" scripts/p2_omni_eval.py check
```

迁移机器时先准备相同依赖与权重，按原 catalog 的 `metrics_python` 和 `metrics` 字段提供新路径，再运行 `runtime --catalog /path/catalog.json --output /path/runtime.lock.json`。`runtime` 是盘点与冻结入口，不会自动联网下载模型。已冻结的 lock 不可静默覆盖；变更版本应另建 lock 和评测 work。

## 已有生成结果的评测

先按示例填写 paired manifest。默认只启用可直接用于音视频输出的 FVD-32、UTMOS、SyncNet：

```bash
"$P2_PY" scripts/p2_omni_eval.py prepare \
  --manifest /absolute/path/e1_pairs.json --output /absolute/path/e1_omni_eval
"$P2_PY" scripts/p2_omni_eval.py evaluate \
  --work /absolute/path/e1_omni_eval --device cpu
# 训练释放 GPU 后可显式使用 --device cuda:0
```

有额外特征和文本时启用全部指标：

```bash
"$P2_PY" scripts/p2_omni_eval.py prepare \
  --manifest /absolute/path/all_inputs.json --output /absolute/path/all_metrics \
  --metrics fvd fd utmos sync text --fvd-frames 32 --crop-seed 42
```

没有特征或文本时保留默认三项即可；不能用空数据冒充 FD/文本结果。每个 seed 至少两个有效片段才能计算 FVD。

## 对接 E1 / 新版 E0 生成

E1 与预检查修复版 E0 具有不同的真实源码身份。新增 [scripts/p2_eval_generate.py](../scripts/p2_eval_generate.py) 自动根据 run.json / generation_protocol.json 选择原训练包或 `releases/p2_preflight_v1`，其余复用原 GLARE 生成接口，不改检查点、不放宽 source 校验。

```bash
"$P2_PY" scripts/p2_eval_generate.py prepare-generation \
  --run /absolute/path/E0_or_E1_run \
  --observations /absolute/path/dev_observations.json \
  --targets /absolute/path/dev_targets.json \
  --output /absolute/path/generation --seeds 137 251
"$P2_PY" scripts/p2_eval_generate.py generate \
  --work /absolute/path/generation \
  --checkpoint /absolute/path/E0_or_E1_run/step_001268.pt \
  --device cuda:0 --wait
"$P2_PY" scripts/p2_omni_eval.py export-generation \
  --work /absolute/path/generation \
  --gt-targets /absolute/path/dev_targets.json \
  --output /absolute/path/omni_pairs.json
```

生成继续要求独立 dev/test、与 train 身份隔离、最终完整检查点；生成阶段不读 GT。支持原有 `--shards/--shard-index` 多卡分片。这里没有自动启动正式生成，因为用户尚未提供正式评测集。

## 缓存、覆盖率与 E0/E1 比较

`prepare` 冻结清单、媒体哈希、评测源码、模型、环境和选项。`evaluate` 执行前后核对输入；逐行成功完成标记最后写出，第二次执行复用已有结果；失败行保留错误后可重试，损坏的成功缓存直接拒绝。技术失败令 CLI 非零退出。

每项区分 planned、eligible、succeeded、inapplicable、failed，附样本 ID。`full_value` 仅在全部计划样本支持且完成时给出，否则为 null；`supported_subset_value` 明确是支持子集。各 seed 分开评分，`full_set_across_seeds` 给完整集合分数的等权均值/标准差，任何 seed 缺失时不偷偷只平均其他 seed。标准差不是置信区间。

比较时要求共同 GT、样本、seed、协议、特征定义、环境与设备一致，再在各指标的共同支持集合上重新汇总：

```bash
"$P2_PY" scripts/p2_omni_eval.py compare \
  --left /absolute/path/e0_omni_eval --right /absolute/path/e1_omni_eval \
  --output /absolute/path/e0_e1_comparison.json
```

同时报告 common_count/planned。不要直接比较两个不同支持子集的均值；静音和检测失败可能与生成质量相关，覆盖率本身必须一起看。

主要输出为 `protocol.json`、`progress.json`、`rows/*/result.json`、逐次 attempts 与日志、`results.json`、`report.md`。SyncNet 的裁脸和逐轨迹结果也保留，可回查每个分数。

## 验证

71 项回归测试通过，真实 CPU 完整链路、自配对、成功缓存复用和共同支持集合比较均已验收。见 [工程验收记录](../experiments/p2_omni_eval_validation_20261026/README.md)。真实 CPU 路径使用实际 I3D、UTMOSv2、S3FD/SyncNet、RoBERTa 和 NLTK；FD 使用明确标记的已知特征样例验证公式。真实样例是已有训练数据的工程自配对，文本也是工程样例，绝不作为正式 dev/test 效果报告。
