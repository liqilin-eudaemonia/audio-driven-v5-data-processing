# ResponseNet：只切观察对象的说话片段

冻结的 J v1 不改。这里只处理 ResponseNet 官方 140 条：说话人固定为 **Speaker / Human · 观察对象**，Agent 自己说话的区间不反向切。音质模型不跑。

官方材料本来就是分开的：

| 内容 | 官方位置 |
|---|---|
| 观察对象画面 / 音轨 / ASR | `test/video/speaker/`、`test/audio/speaker/`、`test/text/speaker/` |
| Agent 画面 / 音轨 / ASR | `test/video/listener/`、`test/audio/listener/`、`test/text/listener/` |
| 140 条清单 | `audio-driven-v4/evaluation/benchmark_alignment/assets/responsenet_inputs/manifest.json` |

## 代码

| 步骤 | 文件 |
|---|---|
| 切分 | [segment.py](segment.py)。规则调用 [listen_local_recovery_semantic_trial.py](../../scripts/listen_local_recovery_semantic_trial.py) 的 `SourceContext.propose()`，然后只保留 `speaker_index == 0` |
| 文本映射检查 | [test_segment.py](test_segment.py) |
| 放到原视频审核页 | [publish_review.py](publish_review.py)。写左右并排原视频、进度条、官方 ASR、每条 pair 分开的画面和音轨 |
| 审核页上的 ResponseNet 展示 | [review_page.js](review_page.js)。发布时复制到审核页目录 |

切分判断在 `results/<样本id>.json`。`pairs` 是初始切分。观察对象文本没有词时 `pair_count` 为 0。官方包没有 VAD，未转写的尾音不会按 VAD 延长。

官方 ASR 没有可用词、因此没有 pair 的样本，用 `transcribe_remaining.py` 在空闲 GPU 上对官方分轨另打 Whisper large-v3 词时间，结果写到 `local_asr/<样本id>/{speaker,listener}.json`。官方 `test/text` 不改。观察对象和 Agent 仍按各自音轨分开转写。这些样本的 `paper_comparable` 为 false。补切命令是 `segment.py --local-only`。

140 条都已处理。135 条有观察对象 pair，共 586 条、合计 4392.07 秒。没有 pair 的 5 条是 `QeSaNfa`、`L6QFFp6`、`twUBM0l`、`eSbIsX8`、`n7zd2b8`。

| 内容 | 位置 |
|---|---|
| 测评交接：完整路径和读取方式 | [HANDOFF.md](HANDOFF.md) |
| 逐条路径 | [CUTS.md](CUTS.md) |
| 机器可读清单 | [cuts/index.json](cuts/index.json) |
| 切分判断 | `results/<样本id>.json` |
| 切开的画面、音轨、ASR | `cuts/<样本id>/pairNN/`，观察对象和 Agent 各一份 |
| 本地词时间 | `local_asr/<样本id>/{speaker,listener}.json` |
| 官方整段 | `/public/0_DATA/2_Avatar/ResponseNet/test/{video,audio,text}/{speaker,listener}/<样本id>.{mp4,wav,json}` |

每条 pair 目录里的文件：

```text
cuts/<样本id>/pair01/speaker.mp4
cuts/<样本id>/pair01/speaker.wav
cuts/<样本id>/pair01/speaker_asr.json
cuts/<样本id>/pair01/listener.mp4
cuts/<样本id>/pair01/listener.wav
cuts/<样本id>/pair01/listener_asr.json
```

官方整段文件仍在 `/public/0_DATA/2_Avatar/ResponseNet/test/`：

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `video/speaker/<样本id>.mp4` | `audio/speaker/<样本id>.wav` | `text/speaker/<样本id>.json` |
| Agent | `video/listener/<样本id>.mp4` | `audio/listener/<样本id>.wav` | `text/listener/<样本id>.json` |

## 运行

在 `audio-driven-v5` 根目录：

```bash
python evaluation/responsenet_speaker_j/test_segment.py
python evaluation/responsenet_speaker_j/segment.py --all
python evaluation/responsenet_speaker_j/export_cuts.py
python evaluation/responsenet_speaker_j/publish_review.py 6V1HXfo 9lhidn7
```

`segment.py --all` 会切全部 140 条，不自动发布。发布只处理命令行给出的样本；不写参数时发布已经看过的 `6V1HXfo` 和 `9lhidn7`。

页面：

- http://localhost:8876/seamless_face_listening_v1/listen_full_source/?pair=6V1HXfo
- http://localhost:8876/seamless_face_listening_v1/listen_full_source/?pair=9lhidn7

A 是观察对象，B 是 Agent。进度条和原视频是同一条时间线。进度条下面是按 pair 切开的画面和音轨。
