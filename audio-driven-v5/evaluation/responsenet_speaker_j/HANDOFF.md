# ResponseNet 观察对象 pair 交接说明

**测评框架接入（2026-11-02）：** 已接入 OmniResponse 的 PSNR / SSIM / FID / FVD / LPIPS / Emo-Acc / Emo-Div / ΔResp 配对视频入口。[接入说明、生成映射模板及运行命令](../../../OmniResponse/evaluation/responsenet_pairs/README.md)。参考清单共 586 pair，GT 固定为下述切分 listener.mp4；当前尚未生成模型结果，正式跑分待填映射后执行。

测评时用切开的 pair，不要用官方整段代替。说话人固定为 Speaker / Human · 观察对象。Agent 自己说话的区间没有反向切成 pair。音质模型没有跑。

140 条官方样本里，135 条有观察对象 pair，共 **586** 条，合计 **4392.07** 秒。单条时长约 2.02–11.95 秒。没有 pair 的 5 条是 `QeSaNfa`、`L6QFFp6`、`twUBM0l`、`eSbIsX8`、`n7zd2b8`，只有官方整段，没有 `cuts/<样本id>/` 目录。

| 词时间 | 样本 | pair |
|---|---:|---:|
| 官方 ASR | 87 | 514 |
| 本地 Whisper large-v3，打在官方分轨上 | 53 | 72 |

本地词时间的样本 `paper_comparable` 为 false。官方 `test/text` 没有改写。这 72 条不能当作论文可比的官方 GT。

## 测评时读哪一份

一条 pair 要观察对象和 Agent 各一份画面、音轨、ASR。机器可读清单里的路径已经是绝对路径，按它读。

清单：

`/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/index.json`

逐条人类可读清单：

`/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/CUTS.md`

切分判断（时间边界、是否官方词时间）：

`/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/results/<样本id>.json`

目录编号是两位，从 `pair01` 开始。一条样本最多 26 条 pair。

```text
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/<样本id>/pair01/speaker.mp4
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/<样本id>/pair01/speaker.wav
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/<样本id>/pair01/speaker_asr.json
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/<样本id>/pair01/listener.mp4
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/<样本id>/pair01/listener.wav
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/<样本id>/pair01/listener_asr.json
```

| 文件 | 角色 | 内容 |
|---|---|---|
| `speaker.mp4` | 观察对象 | 这段 pair 的画面和声音。H.264，640×640，音轨 AAC。帧率随官方原视频。 |
| `speaker.wav` | 观察对象 | 同一段的官方音轨切片。PCM s16le。采样率和声道数随该条官方 wav，不统一重采样。 |
| `speaker_asr.json` | 观察对象 | 这段用到的词时间。 |
| `listener.mp4` | Agent | 同一时间窗的真实参考画面和声音。格式与观察对象画面相同。 |
| `listener.wav` | Agent | 同一时间窗的官方 Agent 音轨切片。 |
| `listener_asr.json` | Agent | 同一时间窗里 Agent 轨上的词。Agent 这段没说话时 `text` 为空、`words` 为空。 |

画面里的声音和旁边的 wav 是同一时间窗。测评需要无损音轨时读 wav；需要看口型时读 mp4。

## 读取方式

`cuts/index.json` 顶层有 `sample_count`、`pair_count`、`pair_seconds`、`samples_without_pairs`。`samples` 里每一条：

| 字段 | 含义 |
|---|---|
| `id` | 样本 id |
| `pair_count` | 这条切出的 pair 数。0 表示没有切开的文件。 |
| `paper_comparable` | true 表示切分用的是官方词时间。false 表示用了本地 Whisper。 |
| `official.speaker.video` / `audio` / `asr` | 观察对象官方整段的绝对路径 |
| `official.listener.video` / `audio` / `asr` | Agent 官方整段的绝对路径 |
| `asr_documents.speaker.path` | 实际用来切分的观察对象词时间文件 |
| `asr_documents.listener.path` | 实际用来切分的 Agent 词时间文件 |
| `asr_documents.*.source` | `official`，或 `whisper_large_v3_on_official_separated_wav` |
| `pairs[].start` / `end` / `duration_s` / `text` | pair 在官方整段时间轴上的起止和观察对象文本 |
| `pairs[].speaker.video` / `audio` / `asr` | 这条 pair 的观察对象文件 |
| `pairs[].listener.video` / `audio` / `asr` | 这条 pair 的 Agent 文件 |

```python
import json
from pathlib import Path

index = json.loads(Path(
    "/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/index.json"
).read_text())

for sample in index["samples"]:
    if sample["pair_count"] == 0:
        continue
    for pair in sample["pairs"]:
        speaker_video = pair["speaker"]["video"]
        speaker_wav = pair["speaker"]["audio"]
        speaker_asr = json.loads(Path(pair["speaker"]["asr"]).read_text())
        listener_video = pair["listener"]["video"]
        listener_wav = pair["listener"]["audio"]
        listener_asr = json.loads(Path(pair["listener"]["asr"]).read_text())
        # speaker_asr["text"] 是观察对象这段话
        # listener_asr["text"] 是同一时间窗里 Agent 轨上的词，可以是空字符串
```

pair 的 ASR JSON：

| 字段 | 含义 |
|---|---|
| `source` | 这份词时间来自哪个文件 |
| `word_time_source` | 有这个字段时，`official` 或 `whisper_large_v3_on_official_separated_wav`。早先导出的官方 pair 可能没有该字段，以清单里的 `paper_comparable` 为准。 |
| `pair_start` / `pair_end` | 与清单里的 `start` / `end` 相同，单位秒，官方整段时间轴 |
| `text` | 落在这个时间窗里的词连成的文本 |
| `words[].text` / `start` / `end` | 词和词时间，单位秒，仍是官方整段时间轴 |

切开的 mp4 / wav 自身从 0 秒开始。词时间没有改成文件内的相对时间。把一个词对齐到切开的 wav 时，用 `word["start"] - pair_start`。

官方整段 JSON 和本地整段 JSON 的词时间字段名是 `timestamps.words[].start_time` / `end_time`。pair 的 `*_asr.json` 已经改成 `words[].start` / `end`。测评 pair 时读 pair 目录里的 ASR，不要再去解析整段 JSON 的字段名。

## 官方整段和本地词时间

官方整段根目录：

`/public/0_DATA/2_Avatar/ResponseNet/test/`

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/<样本id>.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/<样本id>.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/<样本id>.json` |
| Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/<样本id>.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/<样本id>.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/<样本id>.json` |

140 条官方清单：

`/private/liqi_workspace/project_code/audio-driven-v4/evaluation/benchmark_alignment/assets/responsenet_inputs/manifest.json`

`paper_comparable == false` 时，切分用的词时间在：

```text
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/<样本id>/speaker.json
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/<样本id>/listener.json
```

本地 JSON 的词在 `timestamps.words`，字段是 `text`、`start_time`、`end_time`。它打在对应角色的官方 wav 上。长于 1.5 秒的词在切分时按该音轨的实际声音收短；收短发生在内存里，`local_asr` 文件仍是 Whisper 原始词时间。pair 目录里的 `*_asr.json` 用的是收短之后、并且和这条 pair 时间窗相交的词。

`MxQ5sAK` 的观察对象开头是对前 3.2 秒再转写后写回 `local_asr/MxQ5sAK/speaker.json` 的。对应 pair 是：

```text
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/speaker.mp4
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/speaker.wav
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/speaker_asr.json
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/listener.mp4
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/listener.wav
/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/listener_asr.json
```

时间是官方整段上的 0–2.82 秒，文本是 “What do you remember from the first time we met?”。

## 规则和代码

规则是冻结的 J v1，`listen-pair-segmentation-j-v1`，revision `sentence-integrity-two-source-trial-6`。规则文件没有改：

`/private/liqi_workspace/project_code/audio-driven-v5/scripts/listen_local_recovery_semantic_trial.py`

ResponseNet 只保留 `speaker_index == 0` 的适配在：

`/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/segment.py`

导出 pair 文件和两份清单：

`/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/export_cuts.py`

总览页上的 8 条抽样：

http://localhost:18780/omniresponse-media/benchmark_guide/0#response
