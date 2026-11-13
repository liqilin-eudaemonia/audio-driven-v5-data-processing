# 独立 test 24 条快速验证子集

已完成：**24 条，154.5935 秒，平均 6.441 秒**，来自原有独立 **test** 划分，与新的 50h train 在 pair、会话、参与者及连通组上均无交集。没有从 train 中划出测试样本。

原全量标注池只覆盖 train，因此先从 test 的音质候选池选出 512 条人脸较大的片段，补做同一套 Qwen3-VL 视觉、Qwen3-Omni 音频、Qwen3-ASR＋对齐、语义归类及原文整合。之后应用官方 listener 词／句／VAD 并集静音，以及多字中文 ASR 去除规则，再按剩余事件与原始人脸大小筛选。模型输入沿用预裁框裁成 512×512，音频为独立 listener 音轨。

## 结果

| 类别 | 含该类的片段数 |
|---|---:|
| 点头 | 18 |
| 摇头 | 1 |
| 微笑 | 18 |
| 大笑／笑声 | 6 |
| 皱眉 | 10 |
| 惊讶 | 18 |
| 音频回应／发声 | 14 |

一段可包含多类。所有入选片段均无记录的原始推理或语义归类错误，人脸有效帧比例≥98%，且至少有一个非微笑事件。

- 全部原始人脸短边 p10≥310.82 像素；p10 中位数 328.81，p50 中位数 336.10 像素。
- 同一评分公式、单类计数封顶、最多 8 段／pair，以及源区间不重叠约束，与 50h 保持一致。
- 先选 21 条主体，再检查稀有类别；主体已经覆盖七类，因此剩余 3 条仍从主体门槛补足，无需降低人脸门槛。
- 共 17 个 pair、6 个连通组。候选事件数中位数 16，非微笑候选数中位数 6.5；不是人工确认的动作次数。

这是按大脸、事件丰富度挑选的快速诊断集合，不代表整个 test 的自然分布。七类标签是模型弱标注，用于选样和定性复核；没有将其自动提升为人工真值或正式事件指标参考。

## 使用入口

- [manifests/observations.json](manifests/observations.json)：推理输入，包括配对 speaker 音频、listener 首帧、时长与真实 test 身份。
- [manifests/targets.json](manifests/targets.json)：512×512 listener 视频和原始 listener 裁片音轨。
- [integrated_labels.jsonl](integrated_labels.jsonl)：保留原描述的最终整合结果、七类归属、过滤后时间支持及排除证据。
- `audio_postprocess/recipes/<sample_id>.json`：与每条 listener 音轨绑定的虚拟静音配方。
- [samples.csv](samples.csv)、[summary.json](summary.json)、[selection.json](selection.json)：逐条人脸尺寸、事件统计、排序及入选原因。
- [media_references.json](media_references.json)、[dataset_contract.json](dataset_contract.json)、[validation.json](validation.json)、[COMPLETE.json](COMPLETE.json)：媒体身份和验证结果。

`media/<sample_id>/` 保存裁片视频、首帧、speaker.wav、listener.wav 与裁剪元数据。这里的 WAV 是原始音轨的裁片；**没有保存另一份静音处理后的 WAV**，也没有为这 24 条创建训练 latent 缓存。

需要处理后的参考波形时，通过配方读取，不直接把 `targets.json` 中的原 WAV 当作静音后音频：

```python
import sys
from pathlib import Path

project = Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path.insert(0, str(project / 'src'))
from p2_listen.contracts import read
from listen_audio_postprocess.recipes import read_listener_audio

root = project / 'experiments/p2_test24_backchannel_face_20261027'
observations = read(root / 'manifests/observations.json')
sid = observations[0]['sample_id']
recipe = read(root / 'audio_postprocess/recipes' / f'{sid}.json')
waveform, sample_rate, metadata = read_listener_audio(recipe)
```

本目录 `config.json` 是媒体导出配置，不是训练配置。训练使用另一个目录中的 [50h train 配置](../p2_train50h_postprocessed_reselected_20261027/config.json)。`targets.json` 的 `admitted=true` 仅表示媒体通过结构验证，不把 test 变成训练数据，也不把弱标注变成人工真值。

## 追溯与验证

`raw_run/` 保留模型原始响应、输入证据、各视窗请求及工作进程记录；`fusion/` 保留语义归类与原文整合；`postprocessed/` 保留候选的官方时间证据、音频后处理结果和统计。

逐条验证了导出媒体、文件校验值、裁剪尺寸、音画时长与虚拟音频读取。24 条 listener 裁片 WAV 的校验值都与音频模型当时收到的 WAV 一致。全部入选源区间不重叠，train/test 身份交集为 0。没有启动训练；补标和整合工作进程在完成后自动退出。

实现入口：[候选准备](../../scripts/prepare_independent_test_candidates.py)、[补标进程启动](launch_worker.py)、[接续整合](ingest.py)、[后处理、最终选择与媒体导出](finalize.py)。保留固定候选和原始响应，可以直接重跑 `finalize.py` 验证及续导出，无需再次请求模型。
