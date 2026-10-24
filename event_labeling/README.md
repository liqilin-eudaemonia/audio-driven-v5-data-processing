# Listener 开放事件打标：已认可的小样本方案

2026-10-24，用户认可当前标注，要求整理代码并备份。当前基线为 **Qwen3-VL 视觉＋Qwen3-Omni 纯音频＋Qwen3-ASR / ForcedAligner 逐词时间 → 候选时间线 → 开放描述整合**。本目录独立保存这条流程的源码、prompt、配置、原始响应、结果及复现证据；源码按原文件逐字节复制，没有修改模型行为。

50 段样本的 **1,089 条候选 → 682 条开放事件**；709 种原描述完成规范化，0 格式错误、0 回退。42 个 ASR 词事件的边界保持原值。321 条排除建议单独保存在来源追踪中。用户的整体认可见 [USER_ACCEPTANCE.json](USER_ACCEPTANCE.json)，没有将其改写成逐事件人工真值或直接训练准入。±0.2 秒是整合容差及目标要求，不是已测得的时间精度。

## 从哪里开始

| 内容 | 入口 |
| --- | --- |
| 已认可的 50 条最终标签 | [labels.jsonl](experiments/listen_open_merge_20261024T084253Z/labels.jsonl) |
| 来源、内部动作、排除原因 | [provenance.json](experiments/listen_open_merge_20261024T084253Z/provenance.json) |
| 原始视觉／音频响应与候选 | [input_review.json](experiments/listen_open_merge_20261024T084253Z/input_review.json) |
| 整合报告与历史浏览器检查 | [integration_report.json](experiments/listen_open_merge_20261024T084253Z/integration_report.json)、[browser_check.json](experiments/listen_open_merge_20261024T084253Z/browser_check.json) |
| 打标 prompt | [prompts.py](src/listen_benchmark/prompts.py)，本轮使用 `backchannel_visual_simple_v2`、`backchannel_audio_simple_v2` |
| 多卡推理、逐词对齐 | [run_listen_benchmark_vllm.py](scripts/run_listen_benchmark_vllm.py)、[run_listen_word_alignment.py](scripts/run_listen_word_alignment.py) |
| 窗口候选整合 | [integration.py](src/listen_benchmark/integration.py) |
| 描述规范化 prompt 与推理 | [run_listen_atomic_normalize.py](scripts/run_listen_atomic_normalize.py) |
| 开放事件整合算法 | [open_merge.py](src/listen_benchmark/open_merge.py) 中的 `atomic_consolidate` |
| 导出标签与 HTML | [build_listen_open_merge.py](scripts/build_listen_open_merge.py)、[HTML 模板](scripts/listen_open_merge_review.html) |
| 模型、窗口及实验参数 | [split config](experiments/listen_split_validation_20261024T074733Z/config.json)、[plan](experiments/listen_split_validation_20261024T074733Z/plan.json) |
| 复制校验与重放验证 | [BACKUP_MANIFEST.json](BACKUP_MANIFEST.json)、[REPLAY_VALIDATION.json](REPLAY_VALIDATION.json)、[VALIDATION.json](VALIDATION.json) |

原机器的审核页：<http://127.0.0.1:8876/listen_benchmark_review/open_merge/>。在本机浏览需转发远端 8876 端口；客户端分配的本地端口可能不同。

## 无 GPU 重现当前标签与 HTML

在仓库根目录执行。先复制到临时目录，避免覆盖归档结果；Python 3.10+ 的标准库即可完成最终整合重放，不需要模型权重或媒体。

```bash
replay_root=$(mktemp -d /tmp/listen-open-replay.XXXXXX)
cp -a event_labeling "$replay_root/event_labeling"
cd "$replay_root/event_labeling"
python scripts/build_listen_open_merge.py \
  --run experiments/listen_open_merge_20261024T084253Z
python -m unittest discover -s tests -p 'test_listen_open_merge.py'
python -m unittest discover -s tests -p 'test_listen_integration.py'
python -m http.server 8877 --bind 127.0.0.1 --directory data_previews
```

打开 <http://127.0.0.1:8877/listen_benchmark_review/open_merge/>。新生成的 `labels.jsonl`、`labels.json`、`provenance.json`、`summary.json` 应与归档逐字节一致，校验值见重放验证文件。HTML 的事件对照和下载可用；播放需要原媒体目录 `data_previews/listen_benchmark_review/media/<sample_id>/`，其中含 `video.mp4` 和 `audio.wav`。本仓库没有上传媒体、模型权重或机器凭据。可从源项目复制／软链接 `media` 目录；未附带的 `review.woff` 会回退系统字体。

## 供后续训练整理的格式

`labels.jsonl` 每行一个样本；`events` 中保留 `start`、`end`、`modality`、开放的 `behavior`、`event_id` 和 `review_flags`。时间单位为秒，相对该 clip 起点。行为没有固定类别表，例如：

```json
{"start": 1.78, "end": 2.10, "modality": "audio", "behavior": "Says \"Okay\".", "event_id": "event-xxxx", "review_flags": ["asr_transcription_unverified"]}
```

上例展示字段，不作为独立样本记录。完整样本还记录 `duration_s`、`time_origin`、`human_ground_truth:false`、`training_admitted:false`、`unlabeled_time_policy:"unknown"`。模型未标出的时间不自动视为负例。`provenance.json` 用 `sample_id + event_id` 连接来源；训练标签和溯源内容分开存放，不把冗长模型解释混入行为描述。

整合先将每个原描述独立拆成可观察动作，再按语义与时间证据处理重复：持续表情可合并；重复点头、不同话语、相反动作保持独立；音频和视觉保留各自边界。ASR 词边界为保留的锚点，Omni 同词证据只作补充。少量连续嘴部微动作概述仍保留内部原子动作供回溯。本批 [merge_rules.json](experiments/listen_open_merge_20261024T084253Z/merge_rules.json) 是实际描述的同义规则，新动作不受该词表限制。

## 重新进行模型打标

使用原媒体、模型和兼容环境。下列路径是这次实际运行的位置，迁移机器需对应修改配置和 manifest；Python 环境没有打包进 Git。环境证据保存在 [environment](environment)，每种系统各保留一份运行时输入、模型清单和版本记录。历史执行使用 vLLM 0.19.0，记录中 Torch 为 2.10.0+cu129、Transformers 为 4.57.6；ASR 使用单独的 `ovi` 环境及其 `qwen_asr` 依赖。

| 用途 | 模型位置／环境 |
| --- | --- |
| 视觉及纯文本描述规范化 | `/private/ckpts/Qwen3-VL-8B-Instruct`；`/private/software/conda/envs/vllm/bin/python` |
| 音频开放事件 | `/private/ckpts/Qwen3-Omni/Instruct`；同上 vLLM 环境 |
| 逐词识别与对齐 | `/private/ckpts/Qwen3-ASR-1.7B`、`/private/ckpts/Qwen3-ForcedAligner-0.6B`；`/private/software/conda/envs/ovi/bin/python` |
| 无 GPU 完整单元测试 | 原项目 `Omni-LTX-stage2/.venv/bin/python`；除标准库外需要 torch、numpy、scipy、av、Pillow、soundfile、requests |

从 `event_labeling` 目录运行单个视觉 worker 的示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/private/software/conda/envs/vllm/lib:${LD_LIBRARY_PATH:-} \
/private/software/conda/envs/vllm/bin/python scripts/run_listen_benchmark_vllm.py \
  --config experiments/listen_split_validation_20261024T074733Z/config.json \
  --manifest experiments/listen_split_validation_20261024T074733Z/windows50.json \
  --output /tmp/listen-new-inference/results --systems visual_windows \
  --worker-id visual-g0 --batch-size 8 --clients 16 --limit 257
```

音频使用另一 GPU、唯一 worker-id，并改 `--systems audio_windows`。相同模型的多个 worker 可共享输出目录抢占任务；原 8 卡分配及后续加速记录见 [jobs.json](experiments/listen_split_validation_20261024T074733Z/jobs.json)、[visual_extra_jobs.json](experiments/listen_split_validation_20261024T074733Z/visual_extra_jobs.json)。这些是历史命令记录，重跑应换新输出目录并按当前空闲卡安排。整段 50 条推理用于对照，使用 `manifest50.json`、`visual_full` / `audio_full` 和 `--limit 50`；最终候选来自 3 秒窗口、1 秒步长的 257 个窗口。

ASR 入口接受 `--manifest alignment_manifest.json --output <新目录> --dependency-path <qwen_asr 附加依赖目录>`，原始 307 个结果保存在实验目录的 `word_alignment/`。该脚本当前在代码内指定上述两项模型路径及 `cuda:0`，迁移需调整模型路径、用 `CUDA_VISIBLE_DEVICES` 指定物理卡。规范化脚本同样保留原模型路径，接受 `--run <包含 normalize_jobs.json 的新目录> --shard 0 --shards 8`，8 个独立进程分配不同卡和 shard 编号；归档的 60 份规范化响应含完整 prompt，可直接用于确定性重放。

历史 [build_listen_split_review.py](scripts/build_listen_split_review.py) 还依赖源项目的 `data_previews/listen_benchmark_review/review.json` 和 `localization/review.json`，以读取媒体身份、旧对照及头部轨迹；这两份历史整页没有重复打包。需要从新推理重建该旧对照页时，从源项目恢复它们。当前 `input_review.json` 已冻结所有 50 条样本的窗口响应、prompt、ASR 与候选，最终整合重放完整独立于这些历史页面。`src/listen_labeling` 和通用 benchmark 中保留早期固定类别协议供兼容；本轮使用上文两个 simple_v2 prompt 和开放事件导出。

## 归档边界与验证

源码来自 `/private/liqi_workspace/project_code/audio-driven-v5`；`BACKUP_MANIFEST.json` 记录每个复制文件的源路径、大小和 SHA256。`final_code/` 另外保留产出这批最终标签的原始源码快照。首次尝试的全局语义合并未用于最终标签，仅保留 `first_pass_validation.json` 作为历史问题证据；最终入口是 `atomic_consolidate`，不是模块中的旧 `consolidate`。

备份验证包括 50 条原始窗口候选重建、最终四个输出文件逐字节重放、44 项无 GPU 单元测试及源文件哈希核对。历史浏览器验证与本次重放验证分开记录；本次整理没有重新执行 GPU 推理，也没有重启全量打标。
