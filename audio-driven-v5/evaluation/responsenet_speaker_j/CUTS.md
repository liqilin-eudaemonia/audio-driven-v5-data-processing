# ResponseNet 切分结果位置

说话人固定为 Speaker / Human · 观察对象。Agent 自己说话的区间没有反向切。音质模型没有跑。

每条样本先给出官方整段文件，再给出按 pair 切开的画面、音轨和 ASR。

样本 140 条，pair 586 条，合计 4392.07 秒。
有 pair 的样本 135 条。没有 pair 的 5 条：`L6QFFp6`、`QeSaNfa`、`eSbIsX8`、`n7zd2b8`、`twUBM0l`。

位置：

- 切分判断：`results/<样本id>.json`
- 切开的画面、音轨、ASR：`cuts/<样本id>/pairNN/`
- 机器可读清单：`cuts/index.json`
- 本地词时间：`local_asr/<样本id>/{speaker,listener}.json`
- 官方整段：`/public/0_DATA/2_Avatar/ResponseNet/test`

官方 ASR 没有可用词的样本，词时间来自对官方分轨打的 Whisper large-v3，写在 `local_asr/<样本id>/`。官方 `test/text` 没有改。这些样本的 `paper_comparable` 为 false。

## 1PiVxUn

时间轴 30.98 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/1PiVxUn.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/1PiVxUn.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/1PiVxUn.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/1PiVxUn/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/1PiVxUn.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/1PiVxUn.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/1PiVxUn.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/1PiVxUn/listener.json` |

### pair 01 · 0.000–6.350 秒

When you think about the first time we met, what do you remember feeling?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/1PiVxUn/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/1PiVxUn/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/1PiVxUn/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/1PiVxUn/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/1PiVxUn/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/1PiVxUn/pair01/listener_asr.json` |

## 27hyMWh

时间轴 47.27 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/27hyMWh.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/27hyMWh.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/27hyMWh.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/27hyMWh/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/27hyMWh.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/27hyMWh.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/27hyMWh.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/27hyMWh/listener.json` |

### pair 01 · 0.000–4.840 秒

What's your greatest insecurity in our relationship, and how can I support you even more with that?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/27hyMWh/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/27hyMWh/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/27hyMWh/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/27hyMWh/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/27hyMWh/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/27hyMWh/pair01/listener_asr.json` |

## 2U9wCYx

时间轴 61.09 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/2U9wCYx.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/2U9wCYx.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/2U9wCYx.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/2U9wCYx/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/2U9wCYx.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/2U9wCYx.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/2U9wCYx.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/2U9wCYx/listener.json` |

### pair 01 · 0.000–9.840 秒

When were you the proudest of us? Does anything come to mind?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/2U9wCYx/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/2U9wCYx/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/2U9wCYx/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/2U9wCYx/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/2U9wCYx/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/2U9wCYx/pair01/listener_asr.json` |

## 3ISQD1Z

时间轴 74.88 秒，观察对象 pair 8 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/3ISQD1Z.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/3ISQD1Z.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/3ISQD1Z.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/3ISQD1Z.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/3ISQD1Z.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/3ISQD1Z.json` |

### pair 01 · 0.680–8.560 秒

She was not in a happy mood. And she had the opportunity to grill him. I mean, he was, she was still, she was, they were both still there.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair01/listener_asr.json` |

### pair 02 · 8.560–18.160 秒

And he can't one of the things she was looking for evidence that he knew what he had done, or that he had that just how premeditated it all had been.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair02/listener_asr.json` |

### pair 03 · 18.160–27.120 秒

But she was also looking for like how much does he care? And she was she came away feeling like he actually has no ability to feel.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair03/listener_asr.json` |

### pair 04 · 27.120–33.760 秒

Like I can't get him to generate, I can't get him to even fake a kind of compassion.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair04/listener_asr.json` |

### pair 05 · 33.760–40.880 秒

And she was thinking about the pain he'd caused her, like lost all her life savings and lost her job and her reputation, all that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair05/listener_asr.json` |

### pair 06 · 40.880–46.380 秒

She couldn't even get directly from him the sense that she felt sorry for her.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair06/listener_asr.json` |

### pair 07 · 46.520–55.040 秒

So the only thing I'd say in his defense is I don't think he'd ever pretend that he had any of those feelings.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair07/listener_asr.json` |

### pair 08 · 55.040–64.010 秒

I think if you asked him, if you had him on the show and said, Do you feel badly? He might actually say, I don't feel anything.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3ISQD1Z/pair08/listener_asr.json` |

## 3Lq4Qm4

时间轴 30.33 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/3Lq4Qm4.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/3Lq4Qm4.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/3Lq4Qm4.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/3Lq4Qm4.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/3Lq4Qm4.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/3Lq4Qm4.json` |

### pair 01 · 0.080–2.800 秒

Now it's actually the country's land. So you're absolutely right.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair01/listener_asr.json` |

### pair 02 · 2.840–11.600 秒

This is, and this is what [NAME] [NAME] talks about a lot: the idea of how important it is to have property rights and the ability to invest in something like Bitcoin for people who can't, you know,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair02/listener_asr.json` |

### pair 03 · 11.620–20.560 秒

buy the SP 500 and how immense of an opportunity this is. Um, [NAME] can you talk a little bit about [REDACTED] [REDACTED] and also stable coins?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair03/listener_asr.json` |

### pair 04 · 20.620–27.980 秒

Because it seems like in [LOC] stable coins are important as well. But there are so many questions surrounding some of the stable coins that are out there.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3Lq4Qm4/pair04/listener_asr.json` |

## 3sAU2dP

时间轴 36.17 秒，观察对象 pair 5 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/3sAU2dP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/3sAU2dP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/3sAU2dP.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/3sAU2dP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/3sAU2dP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/3sAU2dP.json` |

### pair 01 · 0.080–9.520 秒

Promise, I was reading it promise, but one of the most fascinating parts of your book. Um, this is software was created by a former [REDACTED] official and his wife.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair01/listener_asr.json` |

### pair 02 · 9.520–14.480 秒

And then you write that it ended up in the hands of [REDACTED] intelligence, where there was a trapdoor installed.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair02/listener_asr.json` |

### pair 03 · 14.500–24.720 秒

And then that version was marketed to governments, corporations, high security laboratories, and it could essentially track anything, but it was particularly good at tracking money and people.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair03/listener_asr.json` |

### pair 04 · 24.720–31.040 秒

And so there was a bugged version of the software. [LOC] had a version and the [LOC] had a version. Can you sort of spell that out?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair04/listener_asr.json` |

### pair 05 · 31.060–36.045 秒

Because as you mentioned, [NAME] [NAME] ended up marketing this software and it allowed what, like a dub, it was

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3sAU2dP/pair05/listener_asr.json` |

## 3xwdCKZ

时间轴 55.59 秒，观察对象 pair 7 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/3xwdCKZ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/3xwdCKZ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/3xwdCKZ.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/3xwdCKZ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/3xwdCKZ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/3xwdCKZ.json` |

### pair 01 · 0.080–5.920 秒

Get charged because it was my business, and then to be able to, you know, move it around.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair01/listener_asr.json` |

### pair 02 · 5.940–13.920 秒

And it's a lot more flexible than the sort of the crazy Bitcoin maximalist crazies on thing, oh, gold is terrible, it's going to zero.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair02/listener_asr.json` |

### pair 03 · 13.940–24.320 秒

It has its purpose, and I think it's actually going to do very well right now. Um, but store of value, which I think is very important to understand.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair03/listener_asr.json` |

### pair 04 · 24.320–31.600 秒

You know, [NAME] talks about um Bitcoin as a savings technology a lot. I love that concept.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair04/listener_asr.json` |

### pair 05 · 31.600–36.080 秒

Um, gold is the savings technology for the traditional world.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair05/listener_asr.json` |

### pair 06 · 36.080–46.160 秒

And so that was in [DATE] My gold business integrated with a company called [REDACTED] which um [DATE] is the uphold wallet.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair06/listener_asr.json` |

### pair 07 · 46.160–53.500 秒

And we were the first place that you could buy or sell Bitcoin or [REDACTED] actually, to buy or sell gold.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/3xwdCKZ/pair07/listener_asr.json` |

## 418SUcw

时间轴 88.89 秒，观察对象 pair 9 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/418SUcw.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/418SUcw.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/418SUcw.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/418SUcw.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/418SUcw.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/418SUcw.json` |

### pair 01 · 0.040–3.680 秒

We're just continue to do this as long as Yeah.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair01/listener_asr.json` |

### pair 02 · 3.680–13.600 秒

And so they're gonna and they're gonna do that until the reverse repo facility is drained, and then they're gonna have no choice but to go further out on the curve or just keep

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair02/listener_asr.json` |

### pair 03 · 13.600–20.960 秒

cycling and recycling these these T bills and hopefully into notes and hopefully further on on the curve as rates come down.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair03/listener_asr.json` |

### pair 04 · 20.980–30.720 秒

But they're kind of playing a game of chicken with the economy and the [REDACTED] hoping that rates will come down soon enough, and the the

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair04/listener_asr.json` |

### pair 05 · 30.720–41.500 秒

landing will be soft enough that it doesn't cause the you know the the recession doesn't cause the deficits to blow out and then all of a sudden they have to borrow more money.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair05/listener_asr.json` |

### pair 06 · 41.500–51.040 秒

So that's kind of what they're hoping on. I I I think it's it's unlikely. I think we're going to have a hard recession. I think they're gonna have to borrow more. And so that is a problem.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair06/listener_asr.json` |

### pair 07 · 51.080–57.750 秒

But I would expect the [REDACTED] to hustle and get that bomb buyback program up and running ASAP.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair07/listener_asr.json` |

### pair 08 · 57.750–67.600 秒

So like right, we hit the ground running in [DATE] They're gonna be buying as many off the run USTs as they can. And like I said, that's quasi yield curve control.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair08/listener_asr.json` |

### pair 09 · 67.620–70.380 秒

I don't I don't see it any other way. So

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/418SUcw/pair09/listener_asr.json` |

## 4LBZmlb

时间轴 90.27 秒，观察对象 pair 2 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/4LBZmlb.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/4LBZmlb.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/4LBZmlb.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/4LBZmlb/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/4LBZmlb.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/4LBZmlb.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/4LBZmlb.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/4LBZmlb/listener.json` |

### pair 01 · 3.250–5.880 秒

What's a pain in me you wish you could heal?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair01/listener_asr.json` |

### pair 02 · 86.220–90.165 秒

think that, um, I've done enough

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4LBZmlb/pair02/listener_asr.json` |

## 4ZpywVG

时间轴 58.58 秒，观察对象 pair 7 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/4ZpywVG.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/4ZpywVG.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/4ZpywVG.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/4ZpywVG.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/4ZpywVG.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/4ZpywVG.json` |

### pair 01 · 0.000–8.640 秒

The greatest thing about it is that I actually don't feel any different, but people around me, people who love me, people who are my best friends and my family

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair01/listener_asr.json` |

### pair 02 · 8.640–17.680 秒

tell me that they notice a big difference in me and that they notice that I'm more outgoing and and um you know eager to talk to people and sort of like um

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair02/listener_asr.json` |

### pair 03 · 17.680–28.880 秒

eager to share ideas and eager to share things about myself and so all of these things that that I was terrified about losing have only been augmented and and become sort of greater in in the wake of that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair03/listener_asr.json` |

### pair 04 · 28.900–39.520 秒

And I think that it's made me a more social person, so it's made me want to be around people more, it's made me it's made me um less socially anxious and so and it's made me, I think, you know,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair04/listener_asr.json` |

### pair 05 · 39.520–42.480 秒

just a just a a better human being to be around.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair05/listener_asr.json` |

### pair 06 · 42.500–49.420 秒

And so I think that uh those allowing pathways into sort of like the human experience um

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair06/listener_asr.json` |

### pair 07 · 49.720–58.490 秒

allowing allowing me to encounter new people and talk to new people and share myself and and then you know in turn have people share themselves with me.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4ZpywVG/pair07/listener_asr.json` |

## 4k11zSs

时间轴 31.83 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/4k11zSs.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/4k11zSs.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/4k11zSs.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/4k11zSs.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/4k11zSs.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/4k11zSs.json` |

### pair 01 · 0.000–8.140 秒

I'm having a daughter and I'm just really grateful that my daughter will learn a different way.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair01/listener_asr.json` |

### pair 02 · 8.140–17.920 秒

That she will from the time, you know, even now in my womb, I talk to her about how beautiful it is to be us,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair02/listener_asr.json` |

### pair 03 · 17.920–28.320 秒

how beautiful it is to be [REDACTED] to be of indigenous roots, um, and how she will learn a different way and how she will have

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair03/listener_asr.json` |

### pair 04 · 28.320–31.750 秒

mirrors where she will see her reflection.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/4k11zSs/pair04/listener_asr.json` |

## 6GmhKyA

时间轴 63.51 秒，观察对象 pair 7 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/6GmhKyA.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/6GmhKyA.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/6GmhKyA.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/6GmhKyA.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/6GmhKyA.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/6GmhKyA.json` |

### pair 01 · 0.080–10.000 秒

A few examples. And all of our donors, all of our recipients, all of this data was used for harassment for [REDACTED] campaigns in more than 20 languages.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair01/listener_asr.json` |

### pair 02 · 10.000–19.520 秒

And this is the reason why we have lost our bank account. And even with political support, as I said, you I had quite significant political support and protection.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair02/listener_asr.json` |

### pair 03 · 19.520–26.960 秒

I had I granted a residence permitting in [LOC] and also I had protection there, the same my colleague here.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair03/listener_asr.json` |

### pair 04 · 27.000–37.280 秒

But authoritarian regimes were able to abuse banking data because of abuse of financial action task forces recommendations.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair04/listener_asr.json` |

### pair 05 · 37.280–46.560 秒

And this is a moment when we started to do campaigns against this kind of transnational repression. We call it transnational financial repressions.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair05/listener_asr.json` |

### pair 06 · 46.560–55.040 秒

And we started to use as a defense tool, as the only financial tool, a payment instrument, fundraising instrument, Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair06/listener_asr.json` |

### pair 07 · 55.060–63.395 秒

And this is the reason we say Bitcoin is a human right. And you use it human rights not only in authoritarian states, peer-to-peer transactions specifically, but also.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6GmhKyA/pair07/listener_asr.json` |

## 6IV8As8

时间轴 31.27 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/6IV8As8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/6IV8As8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/6IV8As8.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/6IV8As8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/6IV8As8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/6IV8As8.json` |

### pair 01 · 0.120–7.120 秒

Around the time of the [NAME] collapse, we we moved everything back into cold storage just while we let the industry kind of find its footing.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair01/listener_asr.json` |

### pair 02 · 7.120–16.480 秒

And we're always looking at opportunities to kind of revisit how do we work that work that stack and um in a more meaningful way,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair02/listener_asr.json` |

### pair 03 · 16.500–25.010 秒

but of course without taking unnecessary counterparty risk, which is critically, critically important. So for now, our Bitcoin uh remains remains in custody.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair03/listener_asr.json` |

### pair 04 · 25.010–31.155 秒

Um but when we think about you referenced earlier, we are in the middle of a

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6IV8As8/pair04/listener_asr.json` |

## 6V1HXfo

时间轴 28.17 秒，观察对象 pair 2 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/6V1HXfo.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/6V1HXfo.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/6V1HXfo.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/6V1HXfo.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/6V1HXfo.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/6V1HXfo.json` |

### pair 01 · 0.040–9.920 秒

Uh what are your thoughts on it? It's a bit of a movement right now, and it's compelling when you hear about it, and I understand when they're talking about it, and then I'm like, this feels like it's it it almost feels sort of borderline

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair01/listener_asr.json` |

### pair 02 · 9.920–13.290 秒

unhealthy that we don't have self-determination. What are your thoughts about it?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/6V1HXfo/pair02/listener_asr.json` |

## 7p8Hrss

时间轴 54.19 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/7p8Hrss.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/7p8Hrss.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/7p8Hrss.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/7p8Hrss.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/7p8Hrss.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/7p8Hrss.json` |

### pair 01 · 0.120–9.440 秒

And one of the metrics I've seen recently is about 75 or 80% of holders have not sold. They're holding through this bear market, which I think is pretty remarkable as well.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair01/listener_asr.json` |

### pair 02 · 9.440–21.360 秒

Going back to your earlier point where you talked about the idea of Bitcoin eating into some of these other asset classes like equities and gold, you know, one thing that I didn't see in your report is any sort of price prediction, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair02/listener_asr.json` |

### pair 03 · 21.380–30.320 秒

Like if Bitcoin were to take 1% of this uh this uh asset class's market cap, then the price would be X or Y. Um is there a reason?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair03/listener_asr.json` |

### pair 04 · 30.340–34.320 秒

And does [REDACTED] kind of think of price predictions in the future?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair04/listener_asr.json` |

### pair 05 · 34.320–45.050 秒

I I just had [NAME] [NAME] on, and I know a lot of people were excited and also in disbelief that her company said um $1.4 million Bitcoin in [DATE] Yeah,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair05/listener_asr.json` |

### pair 06 · 45.060–54.015 秒

uh we don't do price predictions. Uh and because we think that's probably a little bit hard to say the least for

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/7p8Hrss/pair06/listener_asr.json` |

## 84oJQVu

时间轴 78.05 秒，观察对象 pair 7 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/84oJQVu.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/84oJQVu.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/84oJQVu.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/84oJQVu.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/84oJQVu.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/84oJQVu.json` |

### pair 01 · 0.360–10.060 秒

I mean, we're so far away from that. But also, on the other hand, I would never bet against [NAME] ever.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair01/listener_asr.json` |

### pair 02 · 10.190–20.080 秒

So, you know, that keeps me balanced as well. I'm, you know, why can't some of these bigger guys get involved? And [DATE] [DATE] [DATE] [DATE] [NAME] was not involved.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair02/listener_asr.json` |

### pair 03 · 20.100–27.600 秒

And I think he said something to the extent of, you know, [REDACTED] is much more interesting, and I'm not really involved in Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair03/listener_asr.json` |

### pair 04 · 27.620–33.180 秒

And then he recently came out and said, well, maybe I should be long Bitcoin, right? Because he's a great trader.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair04/listener_asr.json` |

### pair 05 · 33.180–43.760 秒

And if it's going to go up a lot, he's probably long and making money. He's obviously one of the great money makers of all time.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair05/listener_asr.json` |

### pair 06 · 61.600–71.680 秒

I know, but I don't think he has a resolution. So I don't, I mean, yes, we see that, but what's the answer?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair06/listener_asr.json` |

### pair 07 · 71.680–77.945 秒

And I think the answer I've come up with, it's the best I could do. Um, and I think maybe.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/84oJQVu/pair07/listener_asr.json` |

## 87qRqu5

时间轴 34.57 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/87qRqu5.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/87qRqu5.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/87qRqu5.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/87qRqu5.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/87qRqu5.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/87qRqu5.json` |

### pair 01 · 0.080–9.840 秒

And still, I had not really learned all of it myself. Like I had just scratched the surface. But as I was scratching the surface for myself, I was then relaying that to my people at work and my friends.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair01/listener_asr.json` |

### pair 02 · 9.840–13.440 秒

Um, and my friends, you know, some of them, actually many of them are teachers.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair02/listener_asr.json` |

### pair 03 · 13.460–24.160 秒

I have this just community of teachers here in [LOC] And um, they were saying, Oh, some of my kids are getting interested in Bitcoin and they hear about this blockchain thing. What does all this mean? I said, Okay, I'll come give a talk.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair03/listener_asr.json` |

### pair 04 · 24.160–34.000 秒

Let me try to explain it to them. Uh, and so I started doing this in I want to say [DATE] Um, I think that was my first, my maybe [DATE]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/87qRqu5/pair04/listener_asr.json` |

## 8zCQ67V

时间轴 50.13 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/8zCQ67V.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/8zCQ67V.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/8zCQ67V.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/8zCQ67V/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/8zCQ67V.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/8zCQ67V.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/8zCQ67V.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/8zCQ67V/listener.json` |

### pair 01 · 0.000–3.780 秒

What's the pain in me you wish you could heal?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/8zCQ67V/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/8zCQ67V/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/8zCQ67V/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/8zCQ67V/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/8zCQ67V/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/8zCQ67V/pair01/listener_asr.json` |

## 9Reve57

时间轴 103.75 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/9Reve57.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/9Reve57.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/9Reve57.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/9Reve57/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/9Reve57.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/9Reve57.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/9Reve57.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/9Reve57/listener.json` |

### pair 01 · 0.000–3.840 秒

What is the pain in me that you'd like to heal? Oh no, I didn't know that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9Reve57/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9Reve57/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9Reve57/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9Reve57/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9Reve57/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9Reve57/pair01/listener_asr.json` |

## 9U3k7Wy

时间轴 52.35 秒，观察对象 pair 5 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/9U3k7Wy.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/9U3k7Wy.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/9U3k7Wy.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/9U3k7Wy.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/9U3k7Wy.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/9U3k7Wy.json` |

### pair 01 · 0.520–11.190 秒

Welcome back. Joining me [DATE] [DATE] is [NAME] [NAME] He's the co-founder and [REDACTED] at Swan Bitcoin and the author of one of my favorite Bitcoin books, Inventing Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair01/listener_asr.json` |

### pair 02 · 11.190–19.740 秒

This is amazing. It's a must-read, especially if you want to understand Bitcoin mining and you're not super technical like I am not. Jan, thank you so much.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair02/listener_asr.json` |

### pair 03 · 19.740–22.320 秒

It's such a pleasure to have you on the show for the very first

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair03/listener_asr.json` |

### pair 04 · 25.200–34.000 秒

Well, I'm excited to learn a little bit more about you and how you got to write this book and help uh create what is now [NAME] Bitcoin. So let's start at the beginning.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair04/listener_asr.json` |

### pair 05 · 34.020–44.520 秒

I know your family's originally from [LOC] right? And you had some financial pain points that ultimately led you to why you believe Bitcoin is so important. So tell us that story.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9U3k7Wy/pair05/listener_asr.json` |

## 9UB94rx

时间轴 59.70 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/9UB94rx.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/9UB94rx.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/9UB94rx.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/9UB94rx/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/9UB94rx.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/9UB94rx.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/9UB94rx.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/9UB94rx/listener.json` |

### pair 01 · 0.000–2.580 秒

When do you feel closest to me?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9UB94rx/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9UB94rx/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9UB94rx/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9UB94rx/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9UB94rx/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9UB94rx/pair01/listener_asr.json` |

## 9lhidn7

时间轴 30.54 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/9lhidn7.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/9lhidn7.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/9lhidn7.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/9lhidn7.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/9lhidn7.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/9lhidn7.json` |

### pair 01 · 0.080–10.140 秒

How can the person then cut them off in a safe and healthy way for themselves? But then how can they also process the grief of losing someone? Because I think that can also be one of the real challenges of this.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9lhidn7/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9lhidn7/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9lhidn7/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9lhidn7/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9lhidn7/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/9lhidn7/pair01/listener_asr.json` |

## A6urxiC

时间轴 132.10 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/A6urxiC.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/A6urxiC.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/A6urxiC.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/A6urxiC.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/A6urxiC.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/A6urxiC.json` |

### pair 01 · 0.120–2.590 秒

The greatest impact in the future.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair01/listener_asr.json` |

### pair 02 · 2.680–7.600 秒

So, yes, I think this is just a really critical age group to make sure they understand Bitcoin,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair02/listener_asr.json` |

### pair 03 · 117.220–120.540 秒

And I'll preface with this is such a rabbit hole for me.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair03/listener_asr.json` |

### pair 04 · 120.810–131.970 秒

But if I was not working on Bitcoin, what I wrote all of my college applications about was solving the question of how do we code ethics?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/A6urxiC/pair04/listener_asr.json` |

## AfHzPEl

时间轴 36.64 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/AfHzPEl.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/AfHzPEl.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/AfHzPEl.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/AfHzPEl.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/AfHzPEl.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/AfHzPEl.json` |

### pair 01 · 0.160–9.120 秒

out of uh a a system that you didn't expect. And what the you know the report outlines, well, you know, we were wrong. There's no emergent properties.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair01/listener_asr.json` |

### pair 02 · 9.120–20.240 秒

This is just the low blood numbers happening again. It's like it it's just it's so hilarious to watch all the nerds like frothing about like their new uh AI gods are gonna pop up out of the circuits.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair02/listener_asr.json` |

### pair 03 · 20.240–29.190 秒

Complete scam. It's total bullshit. You know, the like I said, it it's it's the same story again. We're talking about high powered automation.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair03/listener_asr.json` |

### pair 04 · 29.190–35.680 秒

And fundamentally at [DATE] [DATE] [DATE] [DATE] [DATE] the biggest risk is not that the AI is going to become sentient and make decisions on its own.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/AfHzPEl/pair04/listener_asr.json` |

## Ano4Jkc

时间轴 129.40 秒，观察对象 pair 2 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/Ano4Jkc.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/Ano4Jkc.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/Ano4Jkc.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Ano4Jkc/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/Ano4Jkc.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/Ano4Jkc.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/Ano4Jkc.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Ano4Jkc/listener.json` |

### pair 01 · 1.520–7.500 秒

this is my question where do you think we'd be now if we never met wait I

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair01/listener_asr.json` |

### pair 02 · 8.440–16.280 秒

said my own question where do you think you'd be now if we never met let's say I didn't graduate high school

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ano4Jkc/pair02/listener_asr.json` |

## Aud2Wt4

时间轴 75.10 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/Aud2Wt4.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/Aud2Wt4.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/Aud2Wt4.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Aud2Wt4/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/Aud2Wt4.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/Aud2Wt4.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/Aud2Wt4.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Aud2Wt4/listener.json` |

### pair 01 · 0.000–4.850 秒

In what ways has being a survivor of human trafficking most affected the way you raised me?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Aud2Wt4/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Aud2Wt4/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Aud2Wt4/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Aud2Wt4/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Aud2Wt4/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Aud2Wt4/pair01/listener_asr.json` |

## B5zW2Mg

时间轴 37.27 秒，观察对象 pair 2 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/B5zW2Mg.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/B5zW2Mg.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/B5zW2Mg.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/B5zW2Mg.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/B5zW2Mg.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/B5zW2Mg.json` |

### pair 01 · 0.120–5.600 秒

civil society, an effective society and good economy, unless we do those things.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair01/listener_asr.json` |

### pair 02 · 30.920–37.155 秒

First of all, tolerating the intolerable, which is what we're we're doing and have been doing for a while, is

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/B5zW2Mg/pair02/listener_asr.json` |

## BPzouDn

时间轴 49.72 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/BPzouDn.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/BPzouDn.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/BPzouDn.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/BPzouDn.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/BPzouDn.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/BPzouDn.json` |

### pair 01 · 0.330–8.790 秒

I don't know, [DATE] [DATE] after I started my job at [REDACTED] [REDACTED] and uh and did well enough that I haven't really had to worry about making a living as a writer ever since.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair01/listener_asr.json` |

### pair 02 · 8.790–15.340 秒

Um it was it was accidental.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair02/listener_asr.json` |

### pair 03 · 16.040–21.840 秒

It's it's felt it's felt like the thing I should be doing ever since I started doing it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair03/listener_asr.json` |

### pair 04 · 21.840–28.240 秒

I've had mostly nothing but encouragement since I started doing it, but I can't completely explain it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair04/listener_asr.json` |

### pair 05 · 28.240–36.720 秒

It's not in my blood, it's not in my family, it's not in my personal history. It's just kind of like it became the thing I wanted to do, and I did it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair05/listener_asr.json` |

### pair 06 · 36.720–41.600 秒

It's an inspiring story, especially what you shared with. I'm not sure it's inspiring.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/BPzouDn/pair06/listener_asr.json` |

## DUpeSZx

时间轴 45.67 秒，观察对象 pair 5 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/DUpeSZx.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/DUpeSZx.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/DUpeSZx.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/DUpeSZx.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/DUpeSZx.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/DUpeSZx.json` |

### pair 01 · 0.000–10.800 秒

It's another thing that I saw in sort of like the I think it's like died down a little bit, but there was a time when you had all these websites that when I was still writing for the internet, so this was like back in like, you know,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair01/listener_asr.json` |

### pair 02 · 10.800–20.480 秒

[DATE] [DATE] [DATE] where there was this kind of scourge of personal essays about the most horrific, humiliating thing that happened to a human being.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair02/listener_asr.json` |

### pair 03 · 20.500–28.580 秒

Like it and and sort of like and so there I'm just gonna publish like the most humiliating or degrading or awful thing that I experienced in my life.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair03/listener_asr.json` |

### pair 04 · 28.600–35.660 秒

And, you know, I would read those things sometimes and I'd be like, It's exactly what you're saying, like

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair04/listener_asr.json` |

### pair 05 · 35.800–45.590 秒

you were writing about this before you've actually processed the full meaning of this in your life. And and you are you're sort of like you're just like exposing a wound.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DUpeSZx/pair05/listener_asr.json` |

## DmahEyA

时间轴 78.78 秒，观察对象 pair 9 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/DmahEyA.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/DmahEyA.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/DmahEyA.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/DmahEyA.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/DmahEyA.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/DmahEyA.json` |

### pair 01 · 0.000–6.800 秒

And so for me, it was all learning experience. That was when the Occupy Wall Street movement began, and or shortly in the wake of that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair01/listener_asr.json` |

### pair 02 · 6.800–14.240 秒

And what was interesting was the complaint there again was that a bunch of the big banks and financial institutions got bailed out at the

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair02/listener_asr.json` |

### pair 03 · 16.420–26.800 秒

you know the risks turned sour, but when the times were good, the people who, you know, weren't who ended up getting bailed out later ended up still coming out of it, you know, wealthier and the better off for it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair03/listener_asr.json` |

### pair 04 · 26.800–38.240 秒

And uh, it's actually when I first encountered one of my friends who was working in another hedge fund at the time, he reminded me of a saying that sticks to me to [DATE] [DATE] He's like, look, if you're a teacher, you come home with a few extra pencils.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair04/listener_asr.json` |

### pair 05 · 38.240–47.200 秒

If you're if you're a nurse, you might come home with a few extra latex gloves. If you're a banker, you come home with a few extra green pieces of paper. It's just a part of the perk of being on the job.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair05/listener_asr.json` |

### pair 06 · 47.220–57.520 秒

And so it was always a very fascinated, cynical, you know, view, but that informed my view of not real capitalism, but what was the form of crony capitalism that we lived in.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair06/listener_asr.json` |

### pair 07 · 57.520–65.520 秒

And so I was investing in biotech, and one of the areas of crony capitalism is any regulated industry. You know, pharma is one of those highly regulated industries.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair07/listener_asr.json` |

### pair 08 · 65.520–75.200 秒

And those regulated industries, especially that have cozy relationships, mutual relationships with government, that often the very regulations that they're quote unquote encumbered by

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair08/listener_asr.json` |

### pair 09 · 75.200–78.670 秒

were created by some of the biggest companies in a space to be able to.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/DmahEyA/pair09/listener_asr.json` |

## EZoxiBE

时间轴 42.01 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/EZoxiBE.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/EZoxiBE.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/EZoxiBE.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/EZoxiBE.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/EZoxiBE.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/EZoxiBE.json` |

### pair 01 · 0.090–10.960 秒

And so, and then obviously Bitcoin's a part of it, but that's one of those hard assets. So that's kind of what I look at. Um, certain commodities um will probably do well in this environment as well.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair01/listener_asr.json` |

### pair 02 · 11.010–21.680 秒

So, you know, those are the kind of the things that I look at that I I look at history um and just and I say, okay, these things could do well, and then try to diversify my risk as as much as I can.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair02/listener_asr.json` |

### pair 03 · 21.680–30.240 秒

Um, and that's really all you can ask for, you know. Um, and so it's it's a challenging environment. It's because we've never been in an environment like this.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair03/listener_asr.json` |

### pair 04 · 30.240–41.610 秒

And so if anybody's being really certain about their forecast or where you should put your money, I think you should really question them because there's going to be a lot of um, you know, you got to be humble in this environment that's so new.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EZoxiBE/pair04/listener_asr.json` |

## EnQG7ZQ

时间轴 36.00 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/EnQG7ZQ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/EnQG7ZQ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/EnQG7ZQ.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/EnQG7ZQ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/EnQG7ZQ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/EnQG7ZQ.json` |

### pair 01 · 0.120–4.960 秒

That back. And if you ask honest questions, you have no choice but to be orange-pilled.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair01/listener_asr.json` |

### pair 02 · 4.980–12.560 秒

Like if you ask the questions and are willing to disseminate the information properly, it will win you over. Truth always does that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair02/listener_asr.json` |

### pair 03 · 12.560–16.360 秒

And so that's that's kind of why I'm here [DATE]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair03/listener_asr.json` |

### pair 04 · 27.960–35.840 秒

So I as you said to the audience, my last name is Dement. So it means [REDACTED] My somewhere in my past, we were probably evil banker.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/EnQG7ZQ/pair04/listener_asr.json` |

## FKbQrgu

时间轴 70.60 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/FKbQrgu.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/FKbQrgu.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/FKbQrgu.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/FKbQrgu/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/FKbQrgu.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/FKbQrgu.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/FKbQrgu.json` |

### pair 01 · 0.000–2.940 秒

or quantum computing crack Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/FKbQrgu/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/FKbQrgu/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/FKbQrgu/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/FKbQrgu/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/FKbQrgu/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/FKbQrgu/pair01/listener_asr.json` |

## HoiwFoQ

时间轴 49.75 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/HoiwFoQ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/HoiwFoQ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/HoiwFoQ.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/HoiwFoQ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/HoiwFoQ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/HoiwFoQ.json` |

### pair 01 · 0.280–5.680 秒

Um, but if it's not Bitcoin, I don't really know what else it would be. So I think that's that's how it phrase that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair01/listener_asr.json` |

### pair 02 · 25.170–37.120 秒

Yeah, I think I mean there are technologies and uh sometimes there are things that we think of as basic [DATE] like running water, for example, or refrigeration, where you know you don't like you just have to show people it exists and then they want it, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair02/listener_asr.json` |

### pair 03 · 37.140–40.960 秒

You don't have to like convince them, you don't have to uh force them to get it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair03/listener_asr.json` |

### pair 04 · 40.980–49.635 秒

It's just like it engineers can sometimes make something so good that that just it permanently solves a problem almost for everyone.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/HoiwFoQ/pair04/listener_asr.json` |

## II25rmR

时间轴 35.71 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/II25rmR.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/II25rmR.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/II25rmR.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/II25rmR.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/II25rmR.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/II25rmR.json` |

### pair 01 · 0.040–9.200 秒

It it really was a really big indication of how things were changing in that town and how things were changing in [LOC] Um and it

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair01/listener_asr.json` |

### pair 02 · 9.200–18.240 秒

it's still so relevant [DATE] Um, you know, what what not not so much like the quota thing, because of course we know now the rules wouldn't be so overtly racist,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair02/listener_asr.json` |

### pair 03 · 18.260–28.480 秒

but there's always small ways in which um people can find to sort of still control, still put people in their place, uh, still keep us from our history, you know.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair03/listener_asr.json` |

### pair 04 · 28.500–35.575 秒

I mean, you see what's happening in [LOC] with like the [REDACTED] [REDACTED] American studies. Um and it's all sort of related.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/II25rmR/pair04/listener_asr.json` |

## Idkn5ei

时间轴 31.03 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/Idkn5ei.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/Idkn5ei.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/Idkn5ei.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Idkn5ei/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/Idkn5ei.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/Idkn5ei.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/Idkn5ei.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Idkn5ei/listener.json` |

### pair 01 · 0.000–2.160 秒

what makes us family.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Idkn5ei/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Idkn5ei/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Idkn5ei/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Idkn5ei/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Idkn5ei/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Idkn5ei/pair01/listener_asr.json` |

## J0CnWGr

时间轴 37.60 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/J0CnWGr.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/J0CnWGr.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/J0CnWGr.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/J0CnWGr/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/J0CnWGr.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/J0CnWGr.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/J0CnWGr.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/J0CnWGr/listener.json` |

### pair 01 · 0.000–5.540 秒

When you think of the first time we met, what do you remember feeling?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/J0CnWGr/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/J0CnWGr/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/J0CnWGr/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/J0CnWGr/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/J0CnWGr/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/J0CnWGr/pair01/listener_asr.json` |

## KKhXHI4

时间轴 47.21 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/KKhXHI4.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/KKhXHI4.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/KKhXHI4.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/KKhXHI4/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/KKhXHI4.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/KKhXHI4.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/KKhXHI4.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/KKhXHI4/listener.json` |

### pair 01 · 0.000–5.320 秒

What do you feel is the most valuable lesson you've taught me and why do you think that is?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/KKhXHI4/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/KKhXHI4/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/KKhXHI4/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/KKhXHI4/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/KKhXHI4/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/KKhXHI4/pair01/listener_asr.json` |

## L6QFFp6

时间轴 48.24 秒，观察对象 pair 0 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/L6QFFp6.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/L6QFFp6.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/L6QFFp6.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/L6QFFp6/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/L6QFFp6.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/L6QFFp6.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/L6QFFp6.json` |

这条没有切出观察对象 pair。

## LhAva6g

时间轴 33.83 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/LhAva6g.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/LhAva6g.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/LhAva6g.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/LhAva6g.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/LhAva6g.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/LhAva6g.json` |

### pair 01 · 0.080–10.080 秒

Uh Professor, it's always good to speak to you. And uh I I I came late to writing books. It's I wish I'd started earlier. It's been something wonderful to me.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair01/listener_asr.json` |

### pair 02 · 10.100–20.160 秒

And your book, uh predictably, I think it was predictably irrational, was motivation for me to write the book because your agent reached out to me and said,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair02/listener_asr.json` |

### pair 03 · 20.160–23.520 秒

You remind me of Professor [NAME] and you should write a book.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair03/listener_asr.json` |

### pair 04 · 31.540–33.555 秒

Thanks again. Best of luck to you.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LhAva6g/pair04/listener_asr.json` |

## LwtflSs

时间轴 31.20 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/LwtflSs.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/LwtflSs.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/LwtflSs.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/LwtflSs/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/LwtflSs.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/LwtflSs.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/LwtflSs.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/LwtflSs/listener.json` |

### pair 01 · 0.000–4.080 秒

What do you think I'm hesitant to tell you? Why?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LwtflSs/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LwtflSs/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LwtflSs/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LwtflSs/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LwtflSs/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/LwtflSs/pair01/listener_asr.json` |

## M5GDyF2

时间轴 80.08 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/M5GDyF2.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/M5GDyF2.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/M5GDyF2.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/M5GDyF2.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/M5GDyF2.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/M5GDyF2.json` |

### pair 01 · 0.080–9.590 秒

Colonel, something that's always amazed me is how far we're able to kick the can down the road. Um, you know, the [LOC] dollar still used in about eighty percent of global transactions.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair01/listener_asr.json` |

### pair 02 · 9.640–17.320 秒

We're so deeply entrenched in in this financial system that we helped prop up and now it's basically built on a foundation of debt.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair02/listener_asr.json` |

### pair 03 · 17.320–22.950 秒

So you mentioned that you think it's going to crumble pretty soon, but how do you actually see it playing out?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair03/listener_asr.json` |

### pair 04 · 22.950–33.420 秒

Because I think a lot of people would argue that it's it's probably going to be the [LOC] dollar dominating for a bit longer as other currencies start to rise, including many of us believe Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/M5GDyF2/pair04/listener_asr.json` |

## MKjU54l

时间轴 32.93 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/MKjU54l.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/MKjU54l.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/MKjU54l.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/MKjU54l.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/MKjU54l.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/MKjU54l.json` |

### pair 01 · 0.040–11.830 秒

In your article, The Rise of Jamie Diamond, you said that, you know, some of the figures involved were engaged in actions that would intentionally provoke the collapse of certain banks to further consolidate the banking sector for their benefit.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair01/listener_asr.json` |

### pair 02 · 11.850–18.800 秒

The goal, both then and now, seems to be a move toward the logical conclusion of the too big to fail banking model, the eventual

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair02/listener_asr.json` |

### pair 03 · 18.810–25.760 秒

creation of a centralized cartel of mega banks that dominate not only commercial banking, but also central banking.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair03/listener_asr.json` |

### pair 04 · 25.760–32.825 秒

Um, you know, a lot of us in Bitcoin talk about the creature of [NAME] [NAME] that there is this idea of a cartel, um, and that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MKjU54l/pair04/listener_asr.json` |

## MNcdBFN

时间轴 56.76 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/MNcdBFN.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/MNcdBFN.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/MNcdBFN.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/MNcdBFN.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/MNcdBFN.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/MNcdBFN.json` |

### pair 01 · 0.160–9.360 秒

You did such a good job explaining that. And I know that a lot of my viewers and listeners, they might be owning uh GBTC. And it at one time there was a massive premium.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair01/listener_asr.json` |

### pair 02 · 9.380–18.560 秒

So there was that arbitrage trade that people have heard about with some of the exchanges that have since uh gone under. Now it's been trading at a massive, massive discount.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair02/listener_asr.json` |

### pair 03 · 18.560–26.240 秒

And I was trying to really understand that process of how you match the underlying assets value.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair03/listener_asr.json` |

### pair 04 · 26.260–35.280 秒

So if you have, let's say, a hundred Bitcoin in the vault and each Bitcoin is trading at $30,000, that's $3 million. So all the shares have to be worth that $3 million, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair04/listener_asr.json` |

### pair 05 · 35.300–38.480 秒

Otherwise, you can sell, you can redeem, you can buy back.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair05/listener_asr.json` |

### pair 06 · 38.480–44.400 秒

Is that essentially how it works and the benefits of something like this compared to the GBTC or [REDACTED] [REDACTED]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MNcdBFN/pair06/listener_asr.json` |

## MsfRfIJ

时间轴 118.35 秒，观察对象 pair 9 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/MsfRfIJ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/MsfRfIJ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/MsfRfIJ.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/MsfRfIJ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/MsfRfIJ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/MsfRfIJ.json` |

### pair 01 · 0.120–10.640 秒

Was um that was the revelatory moment for me where I was like, oh, it's like rolling a really big dice over and over and over and over again until you hit that target number,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair01/listener_asr.json` |

### pair 02 · 10.640–21.040 秒

and then you're able to um potentially verify the transactions and add a block to the blockchain, the ledger. Yeah. I've recently been then. Oh, I think we have a delay. What'd you say?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair02/listener_asr.json` |

### pair 03 · 50.800–59.440 秒

is such a good analogy. That is such a good illustration. You know, I think that the when you can illustrate it so that someone can picture it in their mind, it may it makes a lot more sense.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair03/listener_asr.json` |

### pair 04 · 59.460–68.800 秒

Like for me, also the analogy that helped was how how is it something that's difficult to um determine or difficult to guess, but easy to verify,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair04/listener_asr.json` |

### pair 05 · 68.800–78.480 秒

so that everyone can agree that yes, this miner won the block. And the best analogy that I found for that was the idea of a bike lock, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair05/listener_asr.json` |

### pair 06 · 78.540–88.960 秒

There's a combination on it. And let's say this bike lock is very, very big, the combination is massive, but as soon as you hit the right combination, the bike, the bike lock unlocks,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair06/listener_asr.json` |

### pair 07 · 88.980–95.920 秒

and everyone can verify that hey, you you were able to unlock that bike. Um, and so these are the types of things that actually really help me.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair07/listener_asr.json` |

### pair 08 · 95.940–106.640 秒

But can you talk a little bit more about this target number and how it fits into um hashes, which ultimately makes it very difficult if you were to try to say um uh

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair08/listener_asr.json` |

### pair 09 · 106.640–114.300 秒

you know put a fraudulent transaction or block into the into the chain, you're not gonna have an easy time doing it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MsfRfIJ/pair09/listener_asr.json` |

## MxQ5sAK

时间轴 64.77 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/MxQ5sAK.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/MxQ5sAK.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/MxQ5sAK.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/MxQ5sAK/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/MxQ5sAK.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/MxQ5sAK.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/MxQ5sAK.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/MxQ5sAK/listener.json` |

### pair 01 · 0.000–2.820 秒

What do you remember from the first time we met?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/MxQ5sAK/pair01/listener_asr.json` |

## Mzp5FCj

时间轴 62.46 秒，观察对象 pair 2 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/Mzp5FCj.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/Mzp5FCj.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/Mzp5FCj.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/Mzp5FCj.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/Mzp5FCj.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/Mzp5FCj.json` |

### pair 01 · 0.120–10.400 秒

Lower prices mean lower profits. Can you talk a little bit more about that? Because you basically argue that Wall Street firms have quietly seized control of the [REDACTED] economy.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair01/listener_asr.json` |

### pair 02 · 10.400–13.440 秒

So can you explain that? And if you were elected president,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Mzp5FCj/pair02/listener_asr.json` |

## NKrgNNz

时间轴 41.64 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/NKrgNNz.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/NKrgNNz.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/NKrgNNz.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/NKrgNNz.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/NKrgNNz.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/NKrgNNz.json` |

### pair 01 · 0.120–5.360 秒

Of these big companies are. And that's also, yeah, very interesting to see.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair01/listener_asr.json` |

### pair 02 · 5.380–12.000 秒

Like, you know, people when you are in this type of market, you're stepping back as a resource professional.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair02/listener_asr.json` |

### pair 03 · 12.020–20.700 秒

Like if I was internal in a company right now, you might might be hiring less, but you're being very thoughtful about how do you solve the challenges right now.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair03/listener_asr.json` |

### pair 04 · 20.700–25.970 秒

And that might not be hiring more people, but it might be pivoting to doing something new.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair04/listener_asr.json` |

### pair 05 · 25.970–29.840 秒

So seeing that with a lot of more traditional finance companies as

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair05/listener_asr.json` |

### pair 06 · 37.640–39.680 秒

Adoption, I would say.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NKrgNNz/pair06/listener_asr.json` |

## NUJ6bTd

时间轴 37.54 秒，观察对象 pair 5 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/NUJ6bTd.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/NUJ6bTd.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/NUJ6bTd.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/NUJ6bTd.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/NUJ6bTd.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/NUJ6bTd.json` |

### pair 01 · 0.080–4.000 秒

Even from the [LOC] government's perspective, between Bitcoin and other cryptocurrencies, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair01/listener_asr.json` |

### pair 02 · 4.000–13.200 秒

[NAME] [NAME] of the [REDACTED] refers to Bitcoin as a commodity, but every other cryptocurrency is a security from the [REDACTED] perspective. That's pretty significant.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair02/listener_asr.json` |

### pair 03 · 13.200–23.500 秒

So seeing Bitcoin as like this commodity from the [REDACTED] perspective, perspective, the only commodity in the digital currency space, they obviously want to use that

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair03/listener_asr.json` |

### pair 04 · 23.500–29.870 秒

to their benefit if they can. And it seems like they probably can and will. Yeah.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair04/listener_asr.json` |

### pair 05 · 30.200–36.000 秒

So the conflict here as I see it is why was Bitcoin originally made?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/NUJ6bTd/pair05/listener_asr.json` |

## Nl1iMid

时间轴 68.74 秒，观察对象 pair 9 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/Nl1iMid.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/Nl1iMid.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/Nl1iMid.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/Nl1iMid.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/Nl1iMid.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/Nl1iMid.json` |

### pair 01 · 0.120–9.440 秒

That I think that bonds are the real melting ice cube over [DATE] [DATE] [DATE] because of the money printing that will be necessary

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair01/listener_asr.json` |

### pair 02 · 9.440–16.460 秒

when we have a debt to GDP of 130% and amount of unfunded liabilities come and do.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair02/listener_asr.json` |

### pair 03 · 16.460–22.560 秒

They're going to print an astronomical amount, and the nominal yield on bonds cannot keep up with that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair03/listener_asr.json` |

### pair 04 · 22.600–30.940 秒

So you're going to be in real terms losing money [DATE] [DATE] you hold bonds. That's my rough estimate.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair04/listener_asr.json` |

### pair 05 · 31.000–40.160 秒

By contrast, I think that Bitcoin's inherent properties make it such that if you hold for [DATE] [DATE] or more, you're going to grow your value.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair05/listener_asr.json` |

### pair 06 · 40.160–49.200 秒

So that's the choice that everybody holding bonds is facing. And they're currently making that choice [DATE] [DATE] [DATE] whether or not they know it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair06/listener_asr.json` |

### pair 07 · 49.200–54.530 秒

They are holding their value in bonds rather than in Bitcoin because they don't understand Bitcoin yet.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair07/listener_asr.json` |

### pair 08 · 54.530–64.420 秒

And so as more time plays out, as more people catch on, I think a large portion of the bond market will

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair08/listener_asr.json` |

### pair 09 · 64.420–68.490 秒

realize that it's lower risk, and actually it's lower risk.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Nl1iMid/pair09/listener_asr.json` |

## OmpucWA

时间轴 55.15 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/OmpucWA.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/OmpucWA.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/OmpucWA.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/OmpucWA/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/OmpucWA.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/OmpucWA.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/OmpucWA.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/OmpucWA/listener.json` |

### pair 01 · 0.000–2.120 秒

How do you describe our breakup to others?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/OmpucWA/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/OmpucWA/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/OmpucWA/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/OmpucWA/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/OmpucWA/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/OmpucWA/pair01/listener_asr.json` |

## PE3p2rf

时间轴 36.17 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/PE3p2rf.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/PE3p2rf.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/PE3p2rf.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/PE3p2rf.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/PE3p2rf.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/PE3p2rf.json` |

### pair 01 · 0.080–6.960 秒

Including like [NAME] [NAME] [NAME] [NAME] and others who say the era of [REDACTED] is over. They're going to run a tight ship.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair01/listener_asr.json` |

### pair 02 · 6.960–16.460 秒

And then there are others who are saying, no, the next time the money printer goes, it's even going to be faster and harder than before. And we could even see double digit inflation.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair02/listener_asr.json` |

### pair 03 · 16.460–19.320 秒

Where are you at? And tell me.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PE3p2rf/pair03/listener_asr.json` |

## PJMXXO9

时间轴 141.01 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/PJMXXO9.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/PJMXXO9.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/PJMXXO9.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/PJMXXO9.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/PJMXXO9.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/PJMXXO9.json` |

### pair 01 · 0.080–5.740 秒

To 100k. Um, it was wrong in this confidence, even that we would go to 55k.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair01/listener_asr.json` |

### pair 02 · 5.740–16.320 秒

But I think directionally it's true, it just might take more time than four years for uh you know, collectively every market participant

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair02/listener_asr.json` |

### pair 03 · 16.320–20.320 秒

to understand the new value of this asset.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair03/listener_asr.json` |

### pair 04 · 20.320–29.570 秒

When you have this dislocation that happens [DATE] [DATE] [DATE] it takes and it takes generations for people to fully incorporate the changed world.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PJMXXO9/pair04/listener_asr.json` |

## PxlIH74

时间轴 44.33 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/PxlIH74.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/PxlIH74.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/PxlIH74.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/PxlIH74/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/PxlIH74.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/PxlIH74.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/PxlIH74.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/PxlIH74/listener.json` |

### pair 01 · 0.000–7.740 秒

What do you think is going to happen, like what is making you nervous about it, besides the fact of us living together?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PxlIH74/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PxlIH74/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PxlIH74/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PxlIH74/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PxlIH74/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/PxlIH74/pair01/listener_asr.json` |

## Q4RMVpn

时间轴 43.13 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/Q4RMVpn.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/Q4RMVpn.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/Q4RMVpn.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Q4RMVpn/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/Q4RMVpn.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/Q4RMVpn.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/Q4RMVpn.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Q4RMVpn/listener.json` |

### pair 01 · 0.000–3.660 秒

What quality of mind do you wish you had and why?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Q4RMVpn/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Q4RMVpn/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Q4RMVpn/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Q4RMVpn/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Q4RMVpn/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Q4RMVpn/pair01/listener_asr.json` |

## QeSaNfa

时间轴 51.83 秒，观察对象 pair 0 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/QeSaNfa.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/QeSaNfa.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/QeSaNfa.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/QeSaNfa/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/QeSaNfa.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/QeSaNfa.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/QeSaNfa.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/QeSaNfa/listener.json` |

这条没有切出观察对象 pair。

## Ql102D5

时间轴 156.80 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/Ql102D5.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/Ql102D5.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/Ql102D5.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Ql102D5/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/Ql102D5.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/Ql102D5.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/Ql102D5.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/Ql102D5/listener.json` |

### pair 01 · 0.000–5.440 秒

If this were to be our last conversation, what is something you'd never want me to forget?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair01/listener_asr.json` |

### pair 02 · 111.520–120.200 秒

not feel guilty about any sacrifices I've made. That I would do it all again, and do even more if I could.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair02/listener_asr.json` |

### pair 03 · 128.440–135.920 秒

I wish your life could be different i wish it could be better it pains me every day to see you have to struggle through so much

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/Ql102D5/pair03/listener_asr.json` |

## SChFtvs

时间轴 34.32 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/SChFtvs.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/SChFtvs.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/SChFtvs.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/SChFtvs/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/SChFtvs.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/SChFtvs.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/SChFtvs.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/SChFtvs/listener.json` |

### pair 01 · 0.000–4.150 秒

Look into my eyes, describe what you see.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SChFtvs/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SChFtvs/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SChFtvs/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SChFtvs/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SChFtvs/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SChFtvs/pair01/listener_asr.json` |

## SbnRTCu

时间轴 52.32 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/SbnRTCu.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/SbnRTCu.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/SbnRTCu.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/SbnRTCu.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/SbnRTCu.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/SbnRTCu.json` |

### pair 01 · 0.080–8.640 秒

Or whatever. So it was a pretty messed up economy. We left in [DATE] and there was also a restriction on leaving, by the way.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair01/listener_asr.json` |

### pair 02 · 8.660–18.800 秒

So there were many periods where you just couldn't even leave. And in [DATE] was one of those windows where they'd opened up as long as you had somebody from the outside kind of like request you, you know, uh sponsor you essentially.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair02/listener_asr.json` |

### pair 03 · 18.820–21.680 秒

So they had to write in and say, we'll take these people in.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair03/listener_asr.json` |

### pair 04 · 21.680–31.280 秒

Um so we left us essentially as religious refugees uh and kind of you know went to the [LOC] And um yeah, for the longest time I didn't really know.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair04/listener_asr.json` |

### pair 05 · 31.280–40.480 秒

I knew I mean obviously I remember it because I was I was a kid, uh, but I was about seven, so it's a little bit fuzzy. Uh but most of what I know about [LOC] [LOC] [LOC] is from my parents, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair05/listener_asr.json` |

### pair 06 · 48.100–50.720 秒

that's actually something I found out quite late in life.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/SbnRTCu/pair06/listener_asr.json` |

## T83453B

时间轴 97.57 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/T83453B.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/T83453B.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/T83453B.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/T83453B/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/T83453B.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/T83453B.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/T83453B.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/T83453B/listener.json` |

### pair 01 · 0.000–4.320 秒

What quality or trait of mind do you wish you had?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair01/listener_asr.json` |

### pair 02 · 53.660–62.460 秒

feel like you have that same goodness, and that's why we never

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair02/listener_asr.json` |

### pair 03 · 63.580–73.560 秒

have drama between us or anything because we're both like, we need to cut that stuff and not have to do that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/T83453B/pair03/listener_asr.json` |

## TH4jxzE

时间轴 30.02 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/TH4jxzE.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/TH4jxzE.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/TH4jxzE.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/TH4jxzE/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/TH4jxzE.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/TH4jxzE.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/TH4jxzE.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/TH4jxzE/listener.json` |

### pair 01 · 0.000–7.260 秒

If you could go back to our first date, what advice would you give yourself about being in a relationship with me?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TH4jxzE/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TH4jxzE/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TH4jxzE/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TH4jxzE/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TH4jxzE/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TH4jxzE/pair01/listener_asr.json` |

## TXYN2qw

时间轴 36.33 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/TXYN2qw.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/TXYN2qw.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/TXYN2qw.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/TXYN2qw.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/TXYN2qw.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/TXYN2qw.json` |

### pair 01 · 0.080–11.020 秒

And you, I mean, you you're pretty transparent around your own struggles with mental health in your book. How does it manifest in your life? Like I from the outside, [NAME] you look like you're just killing it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair01/listener_asr.json` |

### pair 02 · 11.020–20.320 秒

I mean, you look like if this guy can't be happy, I mean, you look as if your life is just uh I'm seeing you everywhere. You look like you're in amazing shape.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair02/listener_asr.json` |

### pair 03 · 20.370–25.120 秒

From an outsider standpoint, it looks like you've just got it, you're you're just doing so well.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair03/listener_asr.json` |

### pair 04 · 25.120–31.900 秒

What are your own means of identifying when you're struggling and are do you have any immediate go-tos?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TXYN2qw/pair04/listener_asr.json` |

## TwIj7o9

时间轴 31.07 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/TwIj7o9.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/TwIj7o9.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/TwIj7o9.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/TwIj7o9/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/TwIj7o9.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/TwIj7o9.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/TwIj7o9.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/TwIj7o9/listener.json` |

### pair 01 · 0.000–6.620 秒

When you think of me, what's the first memory that comes to mind and

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TwIj7o9/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TwIj7o9/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TwIj7o9/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TwIj7o9/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TwIj7o9/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/TwIj7o9/pair01/listener_asr.json` |

## UPBBbCR

时间轴 27.13 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/UPBBbCR.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/UPBBbCR.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/UPBBbCR.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/UPBBbCR/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/UPBBbCR.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/UPBBbCR.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/UPBBbCR.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/UPBBbCR/listener.json` |

### pair 01 · 0.000–6.420 秒

How do you love me? I can answer that so much better.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UPBBbCR/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UPBBbCR/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UPBBbCR/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UPBBbCR/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UPBBbCR/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UPBBbCR/pair01/listener_asr.json` |

## UcTatbq

时间轴 81.12 秒，观察对象 pair 9 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/UcTatbq.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/UcTatbq.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/UcTatbq.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/UcTatbq.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/UcTatbq.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/UcTatbq.json` |

### pair 01 · 0.040–8.000 秒

To believe that we are. But I think it's going to take somebody whose best days in life are still ahead. I hope they are for me.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair01/listener_asr.json` |

### pair 02 · 8.020–14.800 秒

I'm [DATE] I'd be the youngest president ever elected if elected, the youngest person to run for the [REDACTED] nomination right now I am.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair02/listener_asr.json` |

### pair 03 · 14.800–25.710 秒

I think it's going to take somebody from the next generation from the outside to break the broken system that has given us everything we've talked about in our in our chat right now, [NAME]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair03/listener_asr.json` |

### pair 04 · 25.710–35.680 秒

And so if you share that belief with me, I think it's going to take somebody with fresh legs, then you know, help boost us there. And you know, you guys do your part. I promise you I'm going to do mine.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair04/listener_asr.json` |

### pair 05 · 39.350–47.920 秒

If it's a dollar, let it be a dollar, but you still send a signal. And if this country's been as kind to you as it's been to me, then let it be the maximum of what you're able.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair05/listener_asr.json` |

### pair 06 · 47.920–56.960 秒

But that's how we actually do this is we're not going to be saved by somebody else coming and let it get be somebody else's problem. If we're going to be saved, it's going to be because we save ourselves.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair06/listener_asr.json` |

### pair 07 · 56.960–62.240 秒

And so that's the spirit between now and [DATE] [DATE] when the [LOC] caucus happens.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair07/listener_asr.json` |

### pair 08 · 62.240–71.760 秒

Now is the window where if we step on the gas, I I can be your next president, shut down the administrative state, and you know, restore the liberty that we know and love in this country.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair08/listener_asr.json` |

### pair 09 · 71.760–80.000 秒

But I'm going to, you know, require everybody who's in our movement to do their part too, and that's how we're going to make this happen. Thank you.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/UcTatbq/pair09/listener_asr.json` |

## VVWnl4P

时间轴 56.77 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/VVWnl4P.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/VVWnl4P.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/VVWnl4P.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/VVWnl4P/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/VVWnl4P.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/VVWnl4P.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/VVWnl4P.json` |

### pair 01 · 0.000–2.380 秒

When do you feel the closest

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VVWnl4P/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VVWnl4P/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VVWnl4P/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VVWnl4P/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VVWnl4P/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VVWnl4P/pair01/listener_asr.json` |

## VXQZZ5K

时间轴 119.17 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/VXQZZ5K.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/VXQZZ5K.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/VXQZZ5K.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/VXQZZ5K/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/VXQZZ5K.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/VXQZZ5K.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/VXQZZ5K.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/VXQZZ5K/listener.json` |

### pair 01 · 0.000–4.060 秒

How does my hearing ability affect our relationship?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair01/listener_asr.json` |

### pair 02 · 67.840–73.000 秒

this is why i'm so surprised that you said um this is like one of the most critical

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair02/listener_asr.json` |

### pair 03 · 73.000–83.260 秒

things because you go out of your way so much like again more than any person other than like my family that i've known in my entire life

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair03/listener_asr.json` |

### pair 04 · 84.320–90.840 秒

to be, like, critical, like, that stuff that I can do for myself, I advocate, and so, like, the, even

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair04/listener_asr.json` |

### pair 05 · 92.020–101.660 秒

or, like, even the fact that you're thinking about that actively is, is, it blows my mind, because no one's ever done that before for me,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair05/listener_asr.json` |

### pair 06 · 102.000–108.240 秒

my family my mom but Yeah, you just know oh

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/VXQZZ5K/pair06/listener_asr.json` |

## W9zIKoG

时间轴 30.03 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/W9zIKoG.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/W9zIKoG.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/W9zIKoG.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/W9zIKoG.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/W9zIKoG.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/W9zIKoG.json` |

### pair 01 · 0.080–3.350 秒

Truth and it has this ledger where it writes down all the transactions.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair01/listener_asr.json` |

### pair 02 · 3.350–14.000 秒

Um, and the point of Bitcoin is to really take that bank out of the equation and to create a system that is decentralized, has no center, no bank, um, and is trustless and is permissionless.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair02/listener_asr.json` |

### pair 03 · 14.020–23.360 秒

Anybody can do it, you don't have to trust the other party, anybody can join the network, right? Um, and and network is just again, it's a it's a bunch of people that talk to each other, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair03/listener_asr.json` |

### pair 04 · 23.410–29.935 秒

That's the simplest thing you can think about it as a network. So when we have these two phones and we want to send Bitcoin between phone A and phone B.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/W9zIKoG/pair04/listener_asr.json` |

## WckeX4i

时间轴 302.84 秒，观察对象 pair 26 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/WckeX4i.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/WckeX4i.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/WckeX4i.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/WckeX4i.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/WckeX4i.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/WckeX4i.json` |

### pair 01 · 0.040–8.350 秒

uh continuing to do my studies on Bitcoin and all of a sudden it it connected with the econ side of wow this could be a potential um market

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair01/listener_asr.json` |

### pair 02 · 8.350–17.520 秒

based money that is emerging from the marketplace organically right something that we hadn't really seen except for some of these these free banking episodes and stuff like that so

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair02/listener_asr.json` |

### pair 03 · 17.520–27.920 秒

I continue to follow that path um a lot of people out of that program go on to do policy work I quickly realized that was not for me so I went back to finance

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair03/listener_asr.json` |

### pair 04 · 27.920–38.560 秒

and I went back to being an equity analyst and so then I started covering the payment companies so the the squares the PayPals the Visa Mastercards of the world as well as the financial exchanges so

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair04/listener_asr.json` |

### pair 05 · 38.560–45.680 秒

NASX CME all those companies uh were under my uh stable of companies that I had to research

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair05/listener_asr.json` |

### pair 06 · 45.680–55.680 秒

and of course they're starting to do stuff with Bitcoin and crypto by this point and I just started writing reports on this and also writing uh an investor's guide to Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair06/listener_asr.json` |

### pair 07 · 55.700–61.120 秒

So a lot of our clients at the job I was at were RIAs. They were interested their clients were coming to them.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair07/listener_asr.json` |

### pair 08 · 61.120–70.920 秒

And so I started bridging these two worlds together my traditional finance background with my interest in in crypto right or Bitcoin specifically.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair08/listener_asr.json` |

### pair 09 · 70.920–80.560 秒

And then lo and behold this opportunity came up at [REDACTED] again I said I'm happy where I'm at but I got to throw my hat in the ring see what happens and of course I did and and made it through uh

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair09/listener_asr.json` |

### pair 10 · 80.560–89.440 秒

their process and was offered I said I absolutely can't turn this down to make this my my full time job so here I am at [REDACTED] director of research couldn't be happier.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair10/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair10/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair10/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair10/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair10/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair10/listener_asr.json` |

### pair 11 · 89.460–99.040 秒

It's it's kind of a cliche but I am very fortunate to say I'm in one of those jobs where it just seems like all these things come together all of your background all of your experience

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair11/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair11/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair11/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair11/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair11/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair11/listener_asr.json` |

### pair 12 · 99.080–106.480 秒

and I I'm using all those things my traditional finance experience my interest in in economics and then of course Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair12/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair12/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair12/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair12/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair12/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair12/listener_asr.json` |

### pair 13 · 154.040–160.640 秒

Yeah I mean on the tech side it was probably just going back to my youth when we had a [REDACTED] one

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair13/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair13/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair13/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair13/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair13/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair13/listener_asr.json` |

### pair 14 · 160.640–168.300 秒

computer and it fascinated me and I love to play games and toy around in it and I've always been interested in computers and technology ever since

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair14/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair14/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair14/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair14/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair14/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair14/listener_asr.json` |

### pair 15 · 168.300–178.540 秒

maybe in another life I would have gone down that that route somehow but uh the finance world was a little bit of a stronger call I think and then on the on the other part um

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair15/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair15/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair15/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair15/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair15/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair15/listener_asr.json` |

### pair 16 · 178.540–180.560 秒

I don't know you know how it really started.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair16/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair16/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair16/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair16/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair16/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair16/listener_asr.json` |

### pair 17 · 180.580–186.800 秒

It was definitely definitely in college where I started to become aware of um some of these uh economists that really interested me

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair17/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair17/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair17/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair17/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair17/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair17/listener_asr.json` |

### pair 18 · 186.800–195.120 秒

but I think a real key moment was reading uh [NAME] denationalization of money essay right and it was kind of just an aha moment where it says look

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair18/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair18/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair18/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair18/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair18/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair18/listener_asr.json` |

### pair 19 · 195.120–206.080 秒

we look at the marketplace and it produces really good goods and services um shoes, cars, um all kinds of things why can't the market produce uh money

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair19/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair19/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair19/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair19/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair19/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair19/listener_asr.json` |

### pair 20 · 206.080–215.360 秒

and it was a really interesting question to me right and so you kind of go through the arguments and you think about what this would entail and then you realize wow there is something here to that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair20/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair20/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair20/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair20/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair20/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair20/listener_asr.json` |

### pair 21 · 215.360–223.840 秒

So that had been written you know [DATE] before and then of course here we are [DATE] where we we almost see that playing out in real time where you have this organic market

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair21/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair21/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair21/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair21/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair21/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair21/listener_asr.json` |

### pair 22 · 223.880–232.600 秒

based money emerging uh rather than it coming from the top down it has at as it has traditionally done uh through all of history really

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair22/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair22/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair22/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair22/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair22/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair22/listener_asr.json` |

### pair 23 · 271.320–283.040 秒

absolutely it's a it's a great story and something that obviously predates me uh coming on but I love to tell the story and it was one of the things that really attracted me to [REDACTED] and this position.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair23/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair23/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair23/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair23/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair23/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair23/listener_asr.json` |

### pair 24 · 283.060–291.010 秒

And so you're correct you know [REDACTED] it's this this long standing financial institution, private company. It's been [DATE] [DATE] [DATE] [DATE] now.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair24/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair24/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair24/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair24/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair24/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair24/listener_asr.json` |

### pair 25 · 291.010–300.560 秒

And to go back to where their story started with with Bitcoin and digital assets, you actually have to go back to [DATE] before any of this even existed.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair25/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair25/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair25/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair25/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair25/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair25/listener_asr.json` |

### pair 26 · 300.560–302.740 秒

And that is when [REDACTED] started something called the

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair26/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair26/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair26/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair26/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair26/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WckeX4i/pair26/listener_asr.json` |

## WmwCUML

时间轴 31.67 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/WmwCUML.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/WmwCUML.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/WmwCUML.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/WmwCUML.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/WmwCUML.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/WmwCUML.json` |

### pair 01 · 0.080–8.640 秒

didn't want me to speak [REDACTED] They wanted me to speak English. They were proud that I was going to get to learn English. And I have seen that shift a little bit.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair01/listener_asr.json` |

### pair 02 · 8.660–18.080 秒

And I don't, you know, I almost wish that I had more of my [REDACTED] speaking skills, but it was again, it was that pride that we are here, we are [REDACTED] And now,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair02/listener_asr.json` |

### pair 03 · 18.100–23.760 秒

especially if you live in certain parts of the country, you might not find someone who actually speaks English.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/WmwCUML/pair03/listener_asr.json` |

## X5BQM3K

时间轴 72.64 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/X5BQM3K.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/X5BQM3K.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/X5BQM3K.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/X5BQM3K.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/X5BQM3K.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/X5BQM3K.json` |

### pair 01 · 0.190–8.940 秒

This is you know it takes [DATE] and it's can be about 500 Rand. What's that? You know, that's like $30.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair01/listener_asr.json` |

### pair 02 · 9.240–14.620 秒

Just to do you know, just to get the transaction going, never mind what the banks take from it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair02/listener_asr.json` |

### pair 03 · 14.840–24.240 秒

Um individuals can take can take money out up to a certain limit, beyond which you have to have a tax audit and uh all sorts of difficult things.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair03/listener_asr.json` |

### pair 04 · 24.260–28.320 秒

So it is possible to get money out, but it's just there's lots of friction.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/X5BQM3K/pair04/listener_asr.json` |

## XK8hxRU

时间轴 67.07 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/XK8hxRU.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/XK8hxRU.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/XK8hxRU.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/XK8hxRU/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/XK8hxRU.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/XK8hxRU.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/XK8hxRU.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/XK8hxRU/listener.json` |

### pair 01 · 0.000–2.820 秒

What are your three favorite memories we share?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/XK8hxRU/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/XK8hxRU/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/XK8hxRU/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/XK8hxRU/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/XK8hxRU/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/XK8hxRU/pair01/listener_asr.json` |

## YUr85M3

时间轴 38.41 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/YUr85M3.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/YUr85M3.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/YUr85M3.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/YUr85M3.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/YUr85M3.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/YUr85M3.json` |

### pair 01 · 0.080–8.080 秒

And equities and the big tech stocks. Well, you're basically just kind of treading water and keeping up with inflation, not actually making as great of gains as you want.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair01/listener_asr.json` |

### pair 02 · 8.080–17.520 秒

But with Bitcoin, it's been the best performing asset of [DATE] [DATE] [DATE] So can you just can you talk a little bit more before we wrap up about that $10 trillion

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair02/listener_asr.json` |

### pair 03 · 17.520–20.480 秒

market cap and how you see Bitcoin actually reaching that?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair03/listener_asr.json` |

### pair 04 · 20.500–23.290 秒

And do you think it'll happen in [DATE] [DATE] [DATE] [DATE] [DATE]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YUr85M3/pair04/listener_asr.json` |

## YXBtyDB

时间轴 36.87 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/YXBtyDB.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/YXBtyDB.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/YXBtyDB.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/YXBtyDB/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/YXBtyDB.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/YXBtyDB.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/YXBtyDB.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/YXBtyDB/listener.json` |

### pair 01 · 0.000–9.300 秒

Okay. What's my biggest blind spot and what advice do you have for me regarding it? Don't bring up boys again. Don't do it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXBtyDB/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXBtyDB/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXBtyDB/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXBtyDB/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXBtyDB/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXBtyDB/pair01/listener_asr.json` |

## YXYrLed

时间轴 43.58 秒，观察对象 pair 5 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/YXYrLed.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/YXYrLed.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/YXYrLed.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/YXYrLed.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/YXYrLed.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/YXYrLed.json` |

### pair 01 · 0.040–5.120 秒

That right there is a human behavior, a paternal and maternal feel, the caveman feel, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair01/listener_asr.json` |

### pair 02 · 5.170–17.120 秒

That and I think [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED] has done a great job, you know, trying to convince us that we're not a real man or a real woman or real parents or real [REDACTED] until we own a home.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair02/listener_asr.json` |

### pair 03 · 17.140–25.900 秒

And I quite frankly, I think it's bullshit. I think there's there's a lot of instances where you're just the smarter, more responsible thing to do is to rent.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair03/listener_asr.json` |

### pair 04 · 26.000–35.040 秒

What are some of the other kind of key attributes or um psych you know, behaviors, emotions that you see emerging

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair04/listener_asr.json` |

### pair 05 · 35.040–43.470 秒

in what is kind of a weird time right now, and how how it's impacting the economy and the way people think about investing and approach their financial life.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/YXYrLed/pair05/listener_asr.json` |

## ZsE5VCw

时间轴 44.65 秒，观察对象 pair 5 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/ZsE5VCw.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/ZsE5VCw.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/ZsE5VCw.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/ZsE5VCw.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/ZsE5VCw.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/ZsE5VCw.json` |

### pair 01 · 0.080–11.580 秒

Something new. And for that matter, it takes us several generations to adapt our personal wealth mantras.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair01/listener_asr.json` |

### pair 02 · 11.880–23.180 秒

You know, I think that the climate that was set in motion since the since [DATE] when interest rates peaked at [DATE] 17%,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair02/listener_asr.json` |

### pair 03 · 23.180–26.860 秒

something like that, and have been trending down ever since.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair03/listener_asr.json` |

### pair 04 · 27.080–37.760 秒

That downward trend of interest rates is what really drove asset valuations, financial asset valuations going through the roof.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair04/listener_asr.json` |

### pair 05 · 37.760–44.525 秒

That means stocks and real estate. And what we've, as a culture, taken away from that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ZsE5VCw/pair05/listener_asr.json` |

## afUreE9

时间轴 69.47 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/afUreE9.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/afUreE9.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/afUreE9.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/afUreE9/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/afUreE9.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/afUreE9.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/afUreE9.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/afUreE9/listener.json` |

### pair 01 · 0.000–4.860 秒

When was the when was the moment you realized you could trust me?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair01/listener_asr.json` |

### pair 02 · 42.700–45.320 秒

if I if you me.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair02/listener_asr.json` |

### pair 03 · 66.510–69.440 秒

I don't know if you me. I don't know hear me. I me.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/afUreE9/pair03/listener_asr.json` |

## b0vOGMc

时间轴 240.21 秒，观察对象 pair 23 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/b0vOGMc.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/b0vOGMc.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/b0vOGMc.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/b0vOGMc.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/b0vOGMc.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/b0vOGMc.json` |

### pair 01 · 0.000–6.560 秒

It's taking a little time, but it's also relative, right? Some people say you're not as fast as these crypto native firms out there.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair01/listener_asr.json` |

### pair 02 · 6.560–15.600 秒

But on the other hand, we're probably one of the most forward-looking and um fast growing in terms of all the traditional financial institutions out

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair02/listener_asr.json` |

### pair 03 · 72.130–80.240 秒

Yeah. So to be clear, the revisited part is because we actually released this original report uh over [DATE] [DATE] [DATE] maybe even longer.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair03/listener_asr.json` |

### pair 04 · 80.250–89.600 秒

So this is kind of a refresh where we refresh some of the data and some of the stuff in there. But it's been one of our most successful reports, shared most widely, downloaded and viewed the most.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair04/listener_asr.json` |

### pair 05 · 89.600–93.680 秒

And I think that's a testament to how evergreen the content is, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair05/listener_asr.json` |

### pair 06 · 93.680–102.000 秒

And as I was alluding to earlier, you know, the main question we wanted to tackle that we put in the beginning is people hear about Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair06/listener_asr.json` |

### pair 07 · 102.000–105.600 秒

Obviously, it's a new technology, it's a breakthrough technology.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair07/listener_asr.json` |

### pair 08 · 105.600–117.280 秒

And most traditional investors hear technology and they put on these lenses or frameworks of okay, that's great, but usually the first mover in a technological space is not the winner, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair08/listener_asr.json` |

### pair 09 · 117.280–126.390 秒

And so we see [REDACTED] replacing my space. We see the iPhone replacing [REDACTED] right? And so I I was there, I was a tech investor.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair09/listener_asr.json` |

### pair 10 · 126.400–128.640 秒

Those were the lenses I put on as well.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair10/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair10/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair10/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair10/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair10/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair10/listener_asr.json` |

### pair 11 · 128.640–140.320 秒

Uh, but it was through a lot of other writings, standing on the shoulders of giants in this space, as you know, that really got me to look at this more from an economic perspective and the monetary perspective, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair11/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair11/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair11/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair11/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair11/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair11/listener_asr.json` |

### pair 12 · 140.320–150.320 秒

And so once you come around to view Bitcoin, the asset, the token, as an emerging monetary good, not just the technology of the network or the payment system,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair12/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair12/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair12/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair12/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair12/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair12/listener_asr.json` |

### pair 13 · 150.320–158.240 秒

then you start to realize ah this focuses on decentralization and security, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair13/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair13/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair13/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair13/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair13/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair13/listener_asr.json` |

### pair 14 · 158.240–166.720 秒

And those are the core, the core things you want in a monetary good. And then falling back on my economics, you go through the properties of what make good money, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair14/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair14/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair14/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair14/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair14/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair14/listener_asr.json` |

### pair 15 · 166.720–175.680 秒

Portable, divisible, homogenous, easily transportable. Ah, Bitcoin fulfills all those things too. So now you've got a superior form of money.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair15/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair15/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair15/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair15/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair15/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair15/listener_asr.json` |

### pair 16 · 175.680–183.840 秒

And then as you study forms of money and network effects, you realize anyone that tries to quote improve upon this is going to make a trade-off.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair16/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair16/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair16/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair16/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair16/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair16/listener_asr.json` |

### pair 17 · 183.840–192.160 秒

They're going to make a trade-off in terms of that decentralization or that security. Now, in the report, we're very clear that's not necessarily a bad thing per se.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair17/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair17/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair17/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair17/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair17/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair17/listener_asr.json` |

### pair 18 · 192.160–200.640 秒

Maybe you want trade-offs for different use cases. But if you're talking about an emerging form of money, a monetary good, you probably don't want trade-offs there.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair18/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair18/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair18/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair18/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair18/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair18/listener_asr.json` |

### pair 19 · 200.660–209.660 秒

And so that's why we think you have to put Bitcoin in a completely separate bucket from everything else. So again, we write to institutional investors.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair19/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair19/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair19/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair19/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair19/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair19/listener_asr.json` |

### pair 20 · 210.040–219.520 秒

Our main point is you need to have the right lens or framework on. So view Bitcoin as the first and view view Bitcoin as emerging monetary good.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair20/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair20/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair20/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair20/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair20/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair20/listener_asr.json` |

### pair 21 · 219.540–221.760 秒

And then if you want to look at other stuff, that's fine.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair21/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair21/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair21/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair21/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair21/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair21/listener_asr.json` |

### pair 22 · 221.780–232.080 秒

But make sure you view that more like a venture capital bet, something that's going to have a lot more competition, something that's making trade-offs, something that needs to find their product market use case fit.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair22/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair22/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair22/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair22/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair22/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair22/listener_asr.json` |

### pair 23 · 232.080–240.105 秒

And that's, and that's a lot harder for them, right? And so you need to be aware of those risks. So that was the main impetus for this piece and kind of how it evolved.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair23/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair23/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair23/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair23/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair23/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b0vOGMc/pair23/listener_asr.json` |

## b3VScXm

时间轴 50.40 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/b3VScXm.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/b3VScXm.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/b3VScXm.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/b3VScXm/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/b3VScXm.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/b3VScXm.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/b3VScXm.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/b3VScXm/listener.json` |

### pair 01 · 0.000–3.160 秒

What makes us a family?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b3VScXm/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b3VScXm/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b3VScXm/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b3VScXm/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b3VScXm/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/b3VScXm/pair01/listener_asr.json` |

## bFd1pKz

时间轴 31.17 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/bFd1pKz.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/bFd1pKz.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/bFd1pKz.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/bFd1pKz.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/bFd1pKz.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/bFd1pKz.json` |

### pair 01 · 0.230–9.420 秒

sometimes sadly happens, then it's time to move on. If I if I'm running out of things to say, it sort of means the therapy's over.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bFd1pKz/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bFd1pKz/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bFd1pKz/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bFd1pKz/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bFd1pKz/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bFd1pKz/pair01/listener_asr.json` |

## bHYCRUP

时间轴 57.84 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/bHYCRUP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/bHYCRUP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/bHYCRUP.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/bHYCRUP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/bHYCRUP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/bHYCRUP.json` |

### pair 01 · 29.080–35.120 秒

yeah. I guess the finite number, like how many Bitcoin there are, gives me a lot of hope.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair01/listener_asr.json` |

### pair 02 · 35.120–46.000 秒

Um comparing to other cryptocurrencies where there's like a CEO, there's a team of people, like knowing that Bitcoin is just distributed to us who own it, and

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair02/listener_asr.json` |

### pair 03 · 46.000–49.440 秒

that who that's who stands to gain.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair03/listener_asr.json` |

### pair 04 · 53.320–57.560 秒

Got to go there a few times uh [DATE] [DATE] [DATE] [DATE]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bHYCRUP/pair04/listener_asr.json` |

## bsRWRaQ

时间轴 45.67 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/bsRWRaQ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/bsRWRaQ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/bsRWRaQ.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/bsRWRaQ/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/bsRWRaQ.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/bsRWRaQ.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/bsRWRaQ.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/bsRWRaQ/listener.json` |

### pair 01 · 43.320–45.640 秒

I think family is, yeah.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bsRWRaQ/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bsRWRaQ/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bsRWRaQ/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bsRWRaQ/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bsRWRaQ/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/bsRWRaQ/pair01/listener_asr.json` |

## ceq9tRp

时间轴 50.22 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/ceq9tRp.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/ceq9tRp.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/ceq9tRp.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/ceq9tRp.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/ceq9tRp.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/ceq9tRp.json` |

### pair 01 · 0.080–9.520 秒

[DATE] [DATE] [DATE] And the [REDACTED] happened in [DATE] [DATE] [DATE] [DATE] [DATE] So, you know, to me, [REDACTED] [REDACTED] [REDACTED] probably kind of lines up with um,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair01/listener_asr.json` |

### pair 02 · 9.520–18.000 秒

and there was a tightening cycle going on at the same time, uh, very similar. They had rapidly uh raised rates to try and contain what they knew was a housing bubble.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair02/listener_asr.json` |

### pair 03 · 18.000–28.400 秒

Uh and so to me, [REDACTED] [REDACTED] [REDACTED] is kind of maybe the bare stearns event. And I I don't see how we make it a year without a really big dislocation in the credit market.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair03/listener_asr.json` |

### pair 04 · 28.420–35.200 秒

I just don't see how we get there. Um, my partner thinks maybe it's [DATE] [DATE] [DATE] [DATE] [DATE] It could be. I mean, they are good at kicking the can, I'll give them that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair04/listener_asr.json` |

### pair 05 · 35.200–45.680 秒

But you know, it could also be [DATE] [DATE] It could happen [DATE] You know, I mean, we did see, you know, two three big banks fail and [REDACTED] [REDACTED] you know, was forced into a merger. So there's a lot of trouble there.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair05/listener_asr.json` |

### pair 06 · 45.700–50.030 秒

And also there's a signal here. I mean, this this this skip uh a meeting and

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ceq9tRp/pair06/listener_asr.json` |

## cv9uGsL

时间轴 33.58 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/cv9uGsL.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/cv9uGsL.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/cv9uGsL.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/cv9uGsL.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/cv9uGsL.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/cv9uGsL.json` |

### pair 01 · 0.040–2.320 秒

I 1.2 or 1.4 million views.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair01/listener_asr.json` |

### pair 02 · 2.340–12.560 秒

And I was I was ribbing you that more people watched you on this podcast than watch any [REDACTED] show, which is a different talk show about how the world of media is changing.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair02/listener_asr.json` |

### pair 03 · 12.560–22.640 秒

But what in [DATE] [DATE] [DATE] [DATE] uh in terms of the war in [LOC] what w give us a sense for the state of play, how it's unfolded,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair03/listener_asr.json` |

### pair 04 · 22.640–26.780 秒

and what has surprised you and where do you think it's headed?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/cv9uGsL/pair04/listener_asr.json` |

## dSSvPkF

时间轴 121.56 秒，观察对象 pair 14 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/dSSvPkF.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/dSSvPkF.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/dSSvPkF.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/dSSvPkF.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/dSSvPkF.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/dSSvPkF.json` |

### pair 01 · 0.080–10.960 秒

Ban uh Bitcoin miners, and all those Bitcoin miners went to [LOC] and [LOC] for the most part. Um, and of course, there's still clandestine mining operations in [LOC] It's not like they actually shut anything down.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair01/listener_asr.json` |

### pair 02 · 10.960–14.720 秒

Um, but yeah, but most of the hash rate left [LOC] and went to other places.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair02/listener_asr.json` |

### pair 03 · 14.740–23.440 秒

So that's one answer to what happens when there's too much centralization from a jurisdictional standpoint, like when a country is really against something, it'll cause it to go elsewhere.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair03/listener_asr.json` |

### pair 04 · 23.440–33.920 秒

Um, with the pools, the answer to centralization is that pools aren't miners, pools are pools, and a miner can literally hop between pools every second if they wanted to. It's a piece of software, they can point themselves somewhere else.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair04/listener_asr.json` |

### pair 05 · 33.920–38.920 秒

So if a pool is misbehaving or acting in a way that the miner doesn't like, they can switch pretty easily.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair05/listener_asr.json` |

### pair 06 · 38.920–48.480 秒

Um, there is now push from the [LOC] government specifically, and probably other governments over time, that these pools should be like [REDACTED] people

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair06/listener_asr.json` |

### pair 07 · 48.480–55.520 秒

or following [REDACTED] restrictions, which are basically saying, you know, you can't send this Bitcoin to [LOC] or [LOC] [LOC] or wherever, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair07/listener_asr.json` |

### pair 08 · 63.280–71.140 秒

been levels of compliance for sure, especially like if you're a public company in the [LOC] you've got to comply with that kind of stuff. I mean, that's the government. Um, so that is a risk.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair08/listener_asr.json` |

### pair 09 · 71.170–82.160 秒

There's a risk that you know, governments again, there's some jurisdictional arbitrage, like if you really don't want to comply, okay, go and operate out of you know [LOC] [LOC] or somewhere else that doesn't have this restriction.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair09/listener_asr.json` |

### pair 10 · 82.160–84.720 秒

So there's always going to be some jurisdictional arbitrage.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair10/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair10/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair10/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair10/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair10/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair10/listener_asr.json` |

### pair 11 · 84.740–94.720 秒

But I also think we have to be realistic and say, you know, places like the [LOC] and and you know, leaders in sort of uh where the hash rate is is probably where the pools are gonna end up,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair11/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair11/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair11/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair11/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair11/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair11/listener_asr.json` |

### pair 12 · 94.720–104.800 秒

and there's gonna be some amount of that compliance for sure. Um, so that's a risk. I don't know what we do about it, other than it's you always have a choice, right? You don't want to be here, leave the country, go somewhere else.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair12/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair12/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair12/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair12/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair12/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair12/listener_asr.json` |

### pair 13 · 104.800–113.040 秒

That's actually the point of Bitcoin is to give you a choice to leave the country. Because at [DATE] [DATE] [DATE] [DATE] [DATE] that's like that's the pressure valve that we have against our government.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair13/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair13/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair13/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair13/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair13/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair13/listener_asr.json` |

### pair 14 · 113.060–121.460 秒

Is is if we're not happy, yeah, you know, we take to the streets or we just leave and we have brain drain, and then the governments at [DATE] [DATE] [DATE] [DATE] [DATE] you know, they crumble because like that's what happens to.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair14/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair14/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair14/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair14/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair14/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/dSSvPkF/pair14/listener_asr.json` |

## ddeureR

时间轴 102.30 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/ddeureR.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/ddeureR.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/ddeureR.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/ddeureR/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/ddeureR.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/ddeureR.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/ddeureR.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/ddeureR/listener.json` |

### pair 01 · 1.640–8.880 秒

When was the last time you seriously questioned the way that you think and how did you grow from it?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ddeureR/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ddeureR/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ddeureR/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ddeureR/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ddeureR/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ddeureR/pair01/listener_asr.json` |

## e0BpX0j

时间轴 42.91 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/e0BpX0j.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/e0BpX0j.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/e0BpX0j.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/e0BpX0j.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/e0BpX0j.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/e0BpX0j.json` |

### pair 01 · 0.160–7.310 秒

We can't just think and be curious. We are so focused on, you know, how do we just support ourselves to live our life?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair01/listener_asr.json` |

### pair 02 · 7.310–14.400 秒

So I think there's a lot of societal implications that are are coming at [REDACTED] [REDACTED] and we're just trying to do our best to keep up, but it's

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair02/listener_asr.json` |

### pair 03 · 26.120–36.800 秒

No, no, I think there's a lot of work to be done. And I think definitely in the in the [LOC] and especially in the [LOC] For instance, in [LOC] other parts of the world, not the case.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair03/listener_asr.json` |

### pair 04 · 36.800–42.775 秒

But my peer group, so for instance, I'm in [REDACTED] [REDACTED] [REDACTED] [REDACTED] and there's 60 of us.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e0BpX0j/pair04/listener_asr.json` |

## e2jiLyg

时间轴 32.60 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/e2jiLyg.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/e2jiLyg.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/e2jiLyg.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/e2jiLyg.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/e2jiLyg.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/e2jiLyg.json` |

### pair 01 · 0.080–10.640 秒

Versus them, and there's this evil cabal running the world. I don't think that that um brings about any type of hope that we're able to change anything when we do have the numbers and the power.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair01/listener_asr.json` |

### pair 02 · 10.640–20.880 秒

And to your earlier point that all the people that are in [LOC] right now are uh motivated by self-interest, well, we are we all are. That's kind of the basis of capitalism. That's not a bad thing.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair02/listener_asr.json` |

### pair 03 · 20.880–32.501 秒

But the problem is when you have a unit of account that can be so manipulated at the core of the whole system, which allows for these incentive structures to basically pool capital in a very, very small group and then.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/e2jiLyg/pair03/listener_asr.json` |

## eSbIsX8

时间轴 56.77 秒，观察对象 pair 0 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/eSbIsX8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/eSbIsX8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/eSbIsX8.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/eSbIsX8/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/eSbIsX8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/eSbIsX8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/eSbIsX8.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/eSbIsX8/listener.json` |

这条没有切出观察对象 pair。

## eeSzp4P

时间轴 32.77 秒，观察对象 pair 5 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/eeSzp4P.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/eeSzp4P.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/eeSzp4P.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/eeSzp4P.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/eeSzp4P.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/eeSzp4P.json` |

### pair 01 · 0.000–10.000 秒

And so my field site was actually in [LOC] You know, I had a um a convent picked out of nuns that I was, you know, ready to study with.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair01/listener_asr.json` |

### pair 02 · 10.000–16.720 秒

And uh I submitted my dissertation prospectus and my committee completely rejected it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair02/listener_asr.json` |

### pair 03 · 16.720–27.440 秒

Like wow, why um it was, you know, I was in an interdisciplinary program, but my dissertation was

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair03/listener_asr.json` |

### pair 04 · 27.440–30.640 秒

too interdisciplinary even for that program.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair04/listener_asr.json` |

### pair 05 · 30.640–32.665 秒

You know, I drew on things like

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/eeSzp4P/pair05/listener_asr.json` |

## ffjnvZP

时间轴 53.60 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/ffjnvZP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/ffjnvZP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/ffjnvZP.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/ffjnvZP/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/ffjnvZP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/ffjnvZP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/ffjnvZP.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/ffjnvZP/listener.json` |

### pair 01 · 0.000–6.480 秒

How do you think the recent calls to defund the police have affected me differently than you and why?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ffjnvZP/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ffjnvZP/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ffjnvZP/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ffjnvZP/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ffjnvZP/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ffjnvZP/pair01/listener_asr.json` |

## ghKqLx0

时间轴 119.65 秒，观察对象 pair 14 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/ghKqLx0.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/ghKqLx0.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/ghKqLx0.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/ghKqLx0.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/ghKqLx0.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/ghKqLx0.json` |

### pair 01 · 0.120–9.280 秒

Unsettling. It is a [LOC] presidential election near [DATE] [DATE] but it's also a presidential election [DATE] [DATE] in about half of the countries of the world.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair01/listener_asr.json` |

### pair 02 · 9.280–16.800 秒

So I'm sure [DATE] [DATE] is going to be much more turbulent than [DATE] [DATE] was in terms of stuff on that scale.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair02/listener_asr.json` |

### pair 03 · 16.840–25.520 秒

And then you have these groups like the one of the top guys at the World Economic Forum saying a massive cyber attack is likely is not likely,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair03/listener_asr.json` |

### pair 04 · 25.540–32.960 秒

but is going to come before [DATE] [DATE] which of course, or in [DATE] [DATE] that implies

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair04/listener_asr.json` |

### pair 05 · 32.960–42.320 秒

it's going to be in [DATE] And this is the same group that during COVID was like, well, just wait for the cyber pandemic and it's going to be worse than COVID and all of this stuff.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair05/listener_asr.json` |

### pair 06 · 42.340–52.850 秒

So which, by the way, the uh World Economic Forum has a partnership against cybercrime that I've talked a lot, including at the Bitcoin conference [DATE] [DATE] [DATE] in [DATE] And on our show, yeah.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair06/listener_asr.json` |

### pair 07 · 52.850–62.400 秒

And here as well. It's run by a career [REDACTED] intelligence guy named [NAME] [NAME] Um, it's very tied up with the biggest banks in the world, including some of the too big to fail banks from the [LOC]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair07/listener_asr.json` |

### pair 08 · 62.440–72.400 秒

[REDACTED] [REDACTED] [REDACTED] um, and intelligence agencies of the [LOC] [LOC] and [LOC] that have these explicit plans to end both online and financial anonymity online

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair08/listener_asr.json` |

### pair 09 · 72.400–83.040 秒

and specifically attack crypto as a terror financing tool, which has been circulated in mainstream media and then debunked, but it still creates the impression

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair09/listener_asr.json` |

### pair 10 · 83.040–90.640 秒

specifically that Bitcoin is the money of terrorists. And this is an overt effort to criminalize even further the use of mixers.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair10/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair10/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair10/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair10/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair10/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair10/listener_asr.json` |

### pair 11 · 90.660–100.720 秒

So, of course, you have people that have created mixers that have gotten in legal trouble, but the goal is to have anyone that uses a mixer get in legal trouble, which is a significant significant escalation.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair11/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair11/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair11/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair11/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair11/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair11/listener_asr.json` |

### pair 12 · 100.740–107.120 秒

And we're seeing efforts worldwide to end uh the use of encryption by individuals.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair12/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair12/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair12/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair12/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair12/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair12/listener_asr.json` |

### pair 13 · 107.130–113.200 秒

All of this is very significant, and there's a reason they're criminalizing Bitcoin specifically for that.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair13/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair13/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair13/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair13/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair13/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair13/listener_asr.json` |

### pair 14 · 113.220–119.545 秒

And this is part of this broader effort we spoke about earlier to co opt Bitcoin, uh, and they want to use the

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair14/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair14/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair14/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair14/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair14/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ghKqLx0/pair14/listener_asr.json` |

## hD0OtHp

时间轴 34.44 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/hD0OtHp.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/hD0OtHp.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/hD0OtHp.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/hD0OtHp.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/hD0OtHp.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/hD0OtHp.json` |

### pair 01 · 0.160–10.380 秒

[REDACTED] and money launderers out there are able to use the legacy banking system with these regulations in place like 99% of the time.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair01/listener_asr.json` |

### pair 02 · 10.440–13.680 秒

And so who ends up actually getting penalized?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair02/listener_asr.json` |

### pair 03 · 13.680–24.160 秒

Ordinary people who are doing nothing wrong, who are not suspected of a crime, who there's no reason to even suspect them of having committed a crime.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair03/listener_asr.json` |

### pair 04 · 24.160–26.440 秒

So it's a it's a real problem.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hD0OtHp/pair04/listener_asr.json` |

## hLbEq5U

时间轴 31.24 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/hLbEq5U.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/hLbEq5U.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/hLbEq5U.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/hLbEq5U.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/hLbEq5U.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/hLbEq5U.json` |

### pair 01 · 0.120–6.570 秒

Well, that's really exciting and congratulations that it's already going so well. I'm not, I'm not surprised given your own talent.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair01/listener_asr.json` |

### pair 02 · 6.580–13.760 秒

So tell me a little bit more because right now we're in a bear market, and you would think that maybe even hiring slows down.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair02/listener_asr.json` |

### pair 03 · 13.760–18.080 秒

But it sounds like, you know, lightning companies, Bitcoin companies, they are looking for people.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair03/listener_asr.json` |

### pair 04 · 18.100–25.120 秒

So can you can you share what are they looking for and how are they growing despite the financial conditions that that surround

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hLbEq5U/pair04/listener_asr.json` |

## hsYrrx9

时间轴 31.77 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/hsYrrx9.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/hsYrrx9.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/hsYrrx9.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/hsYrrx9.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/hsYrrx9.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/hsYrrx9.json` |

### pair 01 · 0.360–10.800 秒

Which is you know how most people will likely go uh having having the seed offline on metal in a safe and having a harder

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair01/listener_asr.json` |

### pair 02 · 10.800–14.240 秒

wallet on how you that's how you handle the [LOC] right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair02/listener_asr.json` |

### pair 03 · 14.240–19.820 秒

Uh I I think you're already in like a fantastic place, better than you know, 999%

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/hsYrrx9/pair03/listener_asr.json` |

## i8evlJc

时间轴 84.05 秒，观察对象 pair 10 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/i8evlJc.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/i8evlJc.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/i8evlJc.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/i8evlJc.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/i8evlJc.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/i8evlJc.json` |

### pair 01 · 0.080–8.400 秒

People who lose are the everyday citizens whose money was being abused, weaponized, to advance these agendas. Again, that's another, it's not capitalism.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair01/listener_asr.json` |

### pair 02 · 8.400–19.040 秒

What you'll hear from the proponents of the ESG movement is this is just capitalism. So what the owners of capital say they want, that's a farce. The owners of capital don't know how their money's actually being used in the market.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair02/listener_asr.json` |

### pair 03 · 19.040–29.080 秒

If most of them did know, they would say hell no to this to this practice. That's not capitalism. It's again a perverted form of cronyism that masquerades as capitalism.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair03/listener_asr.json` |

### pair 04 · 29.080–38.400 秒

And so I'm a guy, whenever I see a bureaucracy that rotten, I tear down that bureaucracy, and I tend to do it through competition. First best solutions are through the market.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair04/listener_asr.json` |

### pair 05 · 38.400–47.200 秒

I created an alternative firm that offered the same kinds of index funds that [REDACTED] and State Street and [REDACTED] do, but with a different approach, saying that hell no,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair05/listener_asr.json` |

### pair 06 · 47.220–53.120 秒

we won't use other people's money to advance social or environmental goals. That was the premise when I founded it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair06/listener_asr.json` |

### pair 07 · 53.120–62.560 秒

And that ended up, you know, becoming again another another success in shaping the terms of the direction of the ESG movement, which has now had its wings clipped a little bit. That's for the better.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair07/listener_asr.json` |

### pair 08 · 62.560–70.480 秒

And market competition was one way we did it. I mean, the firm I founded crossed a billion dollars in assets under management [DATE] [DATE] [DATE] after launching its first fund.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair08/listener_asr.json` |

### pair 09 · 70.480–77.200 秒

That's twice as quickly as it took [REDACTED] [REDACTED] to get to the same place of a billion dollars when it got into the same line of business.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair09/listener_asr.json` |

### pair 10 · 77.200–83.945 秒

And so it's funny that I'm in the world of politics now because I have great disdain for the intervention and the distortions created by.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair10/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair10/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair10/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair10/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair10/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/i8evlJc/pair10/listener_asr.json` |

## iPr0oCN

时间轴 68.77 秒，观察对象 pair 2 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/iPr0oCN.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/iPr0oCN.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/iPr0oCN.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/iPr0oCN/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/iPr0oCN.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/iPr0oCN.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/iPr0oCN.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/iPr0oCN/listener.json` |

### pair 01 · 0.000–6.600 秒

How do you feel we experience our positions as police chiefs differently and why?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair01/listener_asr.json` |

### pair 02 · 57.680–68.740 秒

it's interesting that you say that because the conversation I had before ours was with my oldest son and we talked about the talk and he talked about how I didn't have that conversation with him.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iPr0oCN/pair02/listener_asr.json` |

## iQrJ1zh

时间轴 72.44 秒，观察对象 pair 9 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/iQrJ1zh.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/iQrJ1zh.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/iQrJ1zh.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/iQrJ1zh.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/iQrJ1zh.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/iQrJ1zh.json` |

### pair 01 · 0.160–9.680 秒

Think about it this way. If I had a bucket, okay, and everybody puts in one dollar bill, right? And then I mix the bucket, and then everybody takes a dollar bill out.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair01/listener_asr.json` |

### pair 02 · 9.700–18.960 秒

We don't know whose dollar bill was that, but everybody came out with a dollar bill. Right? So we just mix the UTXOs together. Um, I'm oversimplifying.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair02/listener_asr.json` |

### pair 03 · 18.960–26.240 秒

But uh essentially through statistical reverse analysis, right? We can come up with a schema where that comes out uh anonymized.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair03/listener_asr.json` |

### pair 04 · 26.240–35.520 秒

Um the issue with all of them is um if you don't if you're not very careful on how you handle the coins on the outset, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair04/listener_asr.json` |

### pair 05 · 35.520–42.400 秒

You can dox yourself again. So it's very tricky. It's uh it's unfortunately it's very advanced.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair05/listener_asr.json` |

### pair 06 · 42.400–52.320 秒

Um the best privacy right now is for people simply to not spend. Uh if you don't spend, nobody's gonna know. Uh that's one way of thinking about it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair06/listener_asr.json` |

### pair 07 · 52.320–57.040 秒

Uh and these tools are improving, they are improving and they will get a lot better.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair07/listener_asr.json` |

### pair 08 · 57.040–67.360 秒

Now, uh some exchanges in some jurisdictions uh will uh block your account if you send to those addresses or if you receive from those addresses, uh,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair08/listener_asr.json` |

### pair 09 · 67.360–72.300 秒

because they are kind of known that they are in a mixer, they don't know they can't.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/iQrJ1zh/pair09/listener_asr.json` |

## inLkPhA

时间轴 136.99 秒，观察对象 pair 13 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/inLkPhA.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/inLkPhA.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/inLkPhA.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/inLkPhA.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/inLkPhA.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/inLkPhA.json` |

### pair 01 · 0.080–8.750 秒

Change of narrative. And we can do it together. Only if we work and educate as much as possible media, and it's huge also role what you are doing.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair01/listener_asr.json` |

### pair 02 · 8.760–16.960 秒

We can educate human rights defenders and we can educate regulators. This is people who makes our life for the moment so hard, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair02/listener_asr.json` |

### pair 03 · 16.960–28.140 秒

Because if it's banned right now, if it's labeled right now, we lose our the last instrument, and it's also going to be weaponized in authoritarian regimes immediately against all of our network of activists.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair03/listener_asr.json` |

### pair 04 · 28.190–35.160 秒

So this is so crucial not to allow democratic countries to turn onto [LOC] style

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair04/listener_asr.json` |

### pair 05 · 35.160–45.800 秒

governance when they want to ban something which is not even present in [REDACTED] [REDACTED] Because there is almost no miners, Bitcoin miners in [REDACTED] [REDACTED] for obvious reason for the price of energy.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair05/listener_asr.json` |

### pair 06 · 45.800–53.610 秒

And uh we that is the reason also why it's so easy to ban it, uh or actually, you know, to move forward

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair06/listener_asr.json` |

### pair 07 · 53.640–63.120 秒

regulation which will discourage investment in Bitcoin mining in a global impactful way, just because there is no voice of Bitcoin miners and Bitcoin advocates.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair07/listener_asr.json` |

### pair 08 · 63.140–72.240 秒

So this is the reason why we ask, and why I also try to encourage everyone in [LOC] in every whenever you are, just join our campaign, please,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair08/listener_asr.json` |

### pair 09 · 72.260–79.120 秒

and help us to defend Bitcoin because it's 27 countries, it's established global standards. We don't want it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair09/listener_asr.json` |

### pair 10 · 79.120–88.640 秒

We don't want our the bank of last resort, our human rights, be left uh without defense and without actually protection. We need to defend it almost human rights.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair10/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair10/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair10/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair10/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair10/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair10/listener_asr.json` |

### pair 11 · 88.690–96.720 秒

If you don't protect them, you lose them. So and Bitcoin the same. Maybe Bitcoin doesn't need us, but we need Bitcoin. We need is a payment instrument.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair11/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair11/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair11/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair11/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair11/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair11/listener_asr.json` |

### pair 12 · 96.740–103.360 秒

We need it right now, not in [DATE] [DATE] We cannot allow to ban it a label, it's whenever this fake

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair12/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair12/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair12/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair12/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair12/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair12/listener_asr.json` |

### pair 13 · 132.920–136.855 秒

Thank you, thank you, [NAME] and thank you everyone for listening. See you soon.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair13/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair13/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair13/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair13/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair13/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/inLkPhA/pair13/listener_asr.json` |

## j8hO2TC

时间轴 27.33 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/j8hO2TC.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/j8hO2TC.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/j8hO2TC.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/j8hO2TC/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/j8hO2TC.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/j8hO2TC.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/j8hO2TC.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/j8hO2TC/listener.json` |

### pair 01 · 0.000–3.920 秒

What was the hardest thing for you at my age?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/j8hO2TC/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/j8hO2TC/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/j8hO2TC/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/j8hO2TC/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/j8hO2TC/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/j8hO2TC/pair01/listener_asr.json` |

## jD9XJrn

时间轴 55.72 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/jD9XJrn.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/jD9XJrn.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/jD9XJrn.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/jD9XJrn.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/jD9XJrn.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/jD9XJrn.json` |

### pair 01 · 1.560–11.290 秒

How do we lead our lives so that our every move maximizes the number of human beings who will exist? It's just it leads you to different weird places.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair01/listener_asr.json` |

### pair 02 · 19.560–28.240 秒

Well, those are two different questions. Start with the ego. Massive, uh, massive grandiosity.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair02/listener_asr.json` |

### pair 03 · 28.260–34.780 秒

Like it, and I think it was a byproduct of the isolation in which he spent his childhood.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair03/listener_asr.json` |

### pair 04 · 34.780–43.360 秒

I think that he was never his view of himself was never seriously challenged.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair04/listener_asr.json` |

### pair 05 · 43.380–53.120 秒

He could have a romanticized view of himself because he didn't have kids in the he didn't have friends to tease him. You know, he didn't have friends to bring him down to earth. And his parents didn't do it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair05/listener_asr.json` |

### pair 06 · 53.120–55.620 秒

His parents were, as a friend said of them.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jD9XJrn/pair06/listener_asr.json` |

## jErsaIY

时间轴 54.19 秒，观察对象 pair 7 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/jErsaIY.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/jErsaIY.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/jErsaIY.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/jErsaIY.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/jErsaIY.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/jErsaIY.json` |

### pair 01 · 0.120–5.440 秒

Just looking for anywhere that they could plug in Bitcoin mining servers.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair01/listener_asr.json` |

### pair 02 · 5.460–12.160 秒

And they were willing to pay kind of traditional data center pricing just to get these machines plugged in.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair02/listener_asr.json` |

### pair 03 · 12.180–18.960 秒

So I think that confluence of events is really what took me all the way down the rabbit hole when it with respect to Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair03/listener_asr.json` |

### pair 04 · 18.960–23.760 秒

I had the opportunity to meet some great people in the space.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair04/listener_asr.json` |

### pair 05 · 23.760–33.520 秒

I read Blockchain Revolution, which is written by a great [REDACTED] [NAME] [NAME] [NAME] and his son [NAME] And then it just kind of kept going. Eventually I found the Bitcoin standard.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair05/listener_asr.json` |

### pair 06 · 33.520–45.290 秒

And from that moment on, I was really always trying to find a way to figure out how to come into this space or find a convergence narrative within tech.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair06/listener_asr.json` |

### pair 07 · 45.720–53.060 秒

And anyway, long story short, that moment came when the board reached out to me in [DATE] [DATE] [DATE] [DATE] to ultimately join [REDACTED]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jErsaIY/pair07/listener_asr.json` |

## jHNdbRD

时间轴 66.47 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/jHNdbRD.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/jHNdbRD.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/jHNdbRD.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/jHNdbRD/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/jHNdbRD.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/jHNdbRD.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/jHNdbRD.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/jHNdbRD/listener.json` |

### pair 01 · 0.000–3.840 秒

What is one thing I could do to improve our relationship?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jHNdbRD/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jHNdbRD/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jHNdbRD/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jHNdbRD/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jHNdbRD/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jHNdbRD/pair01/listener_asr.json` |

## jLbj84E

时间轴 153.59 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/jLbj84E.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/jLbj84E.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/jLbj84E.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/jLbj84E.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/jLbj84E.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/jLbj84E.json` |

### pair 01 · 0.440–10.800 秒

I want to dig into some of that because in [DATE] not only did we decouple from gold, but that was [DATE] [DATE] of the [REDACTED] And I loved your chapter about Heavy is the head who wears that

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair01/listener_asr.json` |

### pair 02 · 10.800–21.810 秒

wears the crown, because this idea that like we must remain the global reserve currency, you are so thoughtful in painting the picture of actually a bright future

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair02/listener_asr.json` |

### pair 03 · 21.810–32.560 秒

in a multipolar world that we don't have to hold on to this [LOC] the empire, which has actually caused the hollowing out of the middle class and so many of the negative financial effects that we felt,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair03/listener_asr.json` |

### pair 04 · 32.580–38.300 秒

including the financialization that you talked about, where everyone's now investing and risking as opposed to saving.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/jLbj84E/pair04/listener_asr.json` |

## k7NsDZS

时间轴 41.33 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/k7NsDZS.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/k7NsDZS.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/k7NsDZS.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/k7NsDZS.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/k7NsDZS.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/k7NsDZS.json` |

### pair 01 · 0.080–6.320 秒

uh of this human thing of seeing and being seen and all these sort of crossed you know and and

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair01/listener_asr.json` |

### pair 02 · 6.320–15.500 秒

in that moment there's yeah there is just absurdity and humor and and also embarrassment and shame and and disappointment all these sorts of things

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair02/listener_asr.json` |

### pair 03 · 34.200–41.225 秒

absolutely absolutely I mean that is exactly what difficult beauty is is asking us to pay attention to is

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/k7NsDZS/pair03/listener_asr.json` |

## kIVp3Gv

时间轴 41.41 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/kIVp3Gv.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/kIVp3Gv.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/kIVp3Gv.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/kIVp3Gv.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/kIVp3Gv.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/kIVp3Gv.json` |

### pair 01 · 0.080–9.920 秒

That idea that even though [LOC] imperfect at creating those conditions, we're about the best darn nation known to human history that allows you, no matter your God-given potential,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair01/listener_asr.json` |

### pair 02 · 9.940–17.200 秒

to achieve everything you can with your unique gifts. And that's what I went on to do in my adulthood as well.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair02/listener_asr.json` |

### pair 03 · 17.220–26.810 秒

And it's something that I want to create for every kid in this country. That's one of the motivators for me. And my upbringing definitely had a big part to do with it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/kIVp3Gv/pair03/listener_asr.json` |

## llHHjhT

时间轴 82.52 秒，观察对象 pair 7 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/llHHjhT.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/llHHjhT.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/llHHjhT.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/llHHjhT.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/llHHjhT.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/llHHjhT.json` |

### pair 01 · 0.170–8.800 秒

Yeah, it w it's amazing to uh to learn from someone like [NAME] because he does break it down as Bitcoin being the internet of money.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair01/listener_asr.json` |

### pair 02 · 8.800–14.080 秒

And I didn't I didn't realize I did not learn prior to Bitcoin really how the internet works.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair02/listener_asr.json` |

### pair 03 · 14.100–22.450 秒

I didn't know about [REDACTED] And I think the the majority of people who use this technology [DATE] [DATE] [DATE] they really don't know how it works.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair03/listener_asr.json` |

### pair 04 · 22.450–28.170 秒

But for people who want to understand Bitcoin, like I said, your book has been really, really incredible.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair04/listener_asr.json` |

### pair 05 · 28.210–40.160 秒

So I wanna um I want to kind of play, not devil's advocate, but I want to kind of um put on the the veil of I don't understand Bitcoin and sort of ask you about some of the most important themes in this book.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair05/listener_asr.json` |

### pair 06 · 40.180–42.930 秒

But first of all, how did you come to write it?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair06/listener_asr.json` |

### pair 07 · 42.930–47.440 秒

Because I know we were talking offline and really interesting story you were teaching high schoolers,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/llHHjhT/pair07/listener_asr.json` |

## mIPMSk1

时间轴 67.74 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/mIPMSk1.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/mIPMSk1.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/mIPMSk1.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/mIPMSk1.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/mIPMSk1.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/mIPMSk1.json` |

### pair 01 · 0.080–8.960 秒

Forced from reality on every level, economically, financially, politically, culturally, strategically, militarily.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair01/listener_asr.json` |

### pair 02 · 22.710–27.920 秒

Oh, absolutely. There's no question about that. And this is what's really disturbing most of all.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair02/listener_asr.json` |

### pair 03 · 27.920–36.100 秒

If you recognize that it's only a matter of time until your own financial system falls apart here at home,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair03/listener_asr.json` |

### pair 04 · 36.110–46.540 秒

and potentially that we slide into a very serious recession, if not a deep depression, which is by no means uh out of bounds right now for us,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair04/listener_asr.json` |

### pair 05 · 46.600–58.550 秒

you understand that one of the things that you've got to do if you're going to get your creditors to cooperate with you is demonstrate your willingness to tighten the purse strings, to cut spending, to cut back.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair05/listener_asr.json` |

### pair 06 · 58.550–67.630 秒

What is this administration doing? It's almost as though you're on a roller coaster. It's no longer connected to the track, you're about to go over a giant hump.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/mIPMSk1/pair06/listener_asr.json` |

## n7zd2b8

时间轴 50.83 秒，观察对象 pair 0 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/n7zd2b8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/n7zd2b8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/n7zd2b8.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/n7zd2b8/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/n7zd2b8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/n7zd2b8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/n7zd2b8.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/n7zd2b8/listener.json` |

这条没有切出观察对象 pair。

## nOeXTrs

时间轴 193.66 秒，观察对象 pair 20 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/nOeXTrs.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/nOeXTrs.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/nOeXTrs.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/nOeXTrs.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/nOeXTrs.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/nOeXTrs.json` |

### pair 01 · 0.080–4.270 秒

Expeditiously as possible because we're really not prepared to fight.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair01/listener_asr.json` |

### pair 02 · 4.280–14.640 秒

You can't make commitments to do a whole variety of things when the means to meet those commitments don't exist. Right now, those means are no longer available.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair02/listener_asr.json` |

### pair 03 · 14.640–16.880 秒

We have serious problems here at home.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair03/listener_asr.json` |

### pair 04 · 16.880–26.480 秒

That doesn't mean abandoning [LOC] or anybody else, but it means impressing on them the criticality of finding a third way forward.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair04/listener_asr.json` |

### pair 05 · 26.480–35.710 秒

Get out of this either or, either we win or they win, and there's nothing in between. If we can do that, then I think there's hope.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair05/listener_asr.json` |

### pair 06 · 35.720–45.740 秒

But I'm not sure we can. I wish I could sit here and say these two irreconcilable sides can find a way forward. I I don't know.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair06/listener_asr.json` |

### pair 07 · 82.550–90.240 秒

Well, we use the word democracy, but in reality, we are a republic. People vote for people to represent their interests.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair07/listener_asr.json` |

### pair 08 · 90.240–96.480 秒

Now we know that the people we voted into office in most cases have not represented our interests.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair08/listener_asr.json` |

### pair 09 · 96.480–106.720 秒

If I were to pull, say, 5,000 [REDACTED] randomly into a stadium and ask them, do you want to go to war here? Do you want to go to war there or somewhere else?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair09/listener_asr.json` |

### pair 10 · 106.720–109.360 秒

I think they would overwhelmingly say no.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair10/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair10/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair10/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair10/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair10/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair10/listener_asr.json` |

### pair 11 · 109.360–119.600 秒

We can look at the polling data over [DATE] [DATE] [DATE] In virtually every case, [REDACTED] were opposed to overseas entanglements and war. So that's not really the issue.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair11/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair11/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair11/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair11/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair11/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair11/listener_asr.json` |

### pair 12 · 119.600–126.800 秒

The issue is how do we get control of our government? How do we get people in [LOC] who represent our interests?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair12/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair12/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair12/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair12/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair12/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair12/listener_asr.json` |

### pair 13 · 126.820–134.640 秒

And one of the things that you have to do if you're going to represent other people's interests is set aside your own emotion. Control it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair13/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair13/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair13/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair13/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair13/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair13/listener_asr.json` |

### pair 14 · 134.640–141.200 秒

You know, I remember sitting in a in a meeting with several four stars, [REDACTED] [REDACTED] and [REDACTED]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair14/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair14/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair14/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair14/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair14/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair14/listener_asr.json` |

### pair 15 · 141.200–152.220 秒

and this particular officer stood up and gave this very passionate speech about what he thought was appropriate, and the [REDACTED] four star said, control yourself.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair15/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair15/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair15/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair15/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair15/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair15/listener_asr.json` |

### pair 16 · 152.360–156.540 秒

We're not interested in emotion. We want facts.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair16/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair16/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair16/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair16/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair16/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair16/listener_asr.json` |

### pair 17 · 160.680–171.600 秒

And instead of condemning one side or the other, if we can get the facts, then there's a chance that we can find this middle road, at least temporarily.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair17/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair17/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair17/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair17/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair17/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair17/listener_asr.json` |

### pair 18 · 171.600–178.560 秒

Once you get people to stop killing each other, at least for a while, things can improve.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair18/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair18/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair18/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair18/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair18/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair18/listener_asr.json` |

### pair 19 · 178.560–184.700 秒

It's not a perfect outcome, but until you stop killing each other, it's hopeless.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair19/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair19/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair19/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair19/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair19/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair19/listener_asr.json` |

### pair 20 · 191.410–193.550 秒

Sure. Thank you, [NAME] Happy birth.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair20/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair20/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair20/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair20/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair20/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nOeXTrs/pair20/listener_asr.json` |

## ni5RYOf

时间轴 31.53 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/ni5RYOf.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/ni5RYOf.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/ni5RYOf.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/ni5RYOf.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/ni5RYOf.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/ni5RYOf.json` |

### pair 01 · 0.200–10.480 秒

To be that next president. And the [REDACTED] establishment, I mean, the things I'm talking about right now, they view that as dangerous, as a threat, as

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair01/listener_asr.json` |

### pair 02 · 10.520–12.640 秒

something they have an anaphylactic response to.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair02/listener_asr.json` |

### pair 03 · 12.640–23.040 秒

The the uh stooge of a clown who's the uh a chairwoman of [REDACTED] [REDACTED] [REDACTED] [REDACTED] a woman by the name of [NAME] [NAME] you have no reason to know who that is, and you know, all the better for you if you if you don't.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair03/listener_asr.json` |

### pair 04 · 23.040–31.405 秒

But she's publicly said after I criticized [REDACTED] [REDACTED] [REDACTED] [REDACTED] on that last debate stage that I would get not another cent of funding from the [REDACTED]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ni5RYOf/pair04/listener_asr.json` |

## nsHRyEf

时间轴 34.57 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/nsHRyEf.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/nsHRyEf.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/nsHRyEf.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/nsHRyEf/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/nsHRyEf.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/nsHRyEf.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/nsHRyEf.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/nsHRyEf/listener.json` |

### pair 01 · 8.040–14.280 秒

don't think there's anything like really important that I hadn't told you it would probably be very little but

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nsHRyEf/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nsHRyEf/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nsHRyEf/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nsHRyEf/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nsHRyEf/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/nsHRyEf/pair01/listener_asr.json` |

## oTTdbR6

时间轴 95.30 秒，观察对象 pair 5 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/oTTdbR6.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/oTTdbR6.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/oTTdbR6.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/oTTdbR6.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/oTTdbR6.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/oTTdbR6.json` |

### pair 01 · 0.040–3.040 秒

Somewhere else, that's what happens. Um, but yeah, that's that's a

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair01/listener_asr.json` |

### pair 02 · 63.060–71.040 秒

I think that's that's a really good call out. Uh the Bitcoin blockchain is completely public. You can go and look it up on a box explorer, you can download it on your own node.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair02/listener_asr.json` |

### pair 03 · 71.040–78.960 秒

Um, Bitcoin was designed with that in mind for transparency of supply. I think that's like the auditability of Bitcoin was number one.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair03/listener_asr.json` |

### pair 04 · 78.960–87.130 秒

We want to make sure we can follow transactions, like we know that no fake money entered into the system, nobody was double spent, and so on. The trade-off is privacy.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair04/listener_asr.json` |

### pair 05 · 87.160–94.700 秒

So with cash, I would say, you know, back in [DATE] [DATE] we used to have these like green pieces of paper, which probably nobody remembers anymore as we speak.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/oTTdbR6/pair05/listener_asr.json` |

## owudSAI

时间轴 71.17 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/owudSAI.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/owudSAI.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/owudSAI.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/owudSAI.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/owudSAI.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/owudSAI.json` |

### pair 01 · 0.080–7.440 秒

You wanted to be an art historian, right? And you studied archaeology, then you studied economics. So how did you get to be where you are?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/owudSAI/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/owudSAI/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/owudSAI/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/owudSAI/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/owudSAI/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/owudSAI/pair01/listener_asr.json` |

## pSSy7yy

时间轴 43.33 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/pSSy7yy.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/pSSy7yy.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/pSSy7yy.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/pSSy7yy.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/pSSy7yy.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/pSSy7yy.json` |

### pair 01 · 0.080–8.680 秒

time period we're in. It's [DATE] We're before [DATE] [DATE] [DATE] We're in a split [REDACTED] Uh the naive but hopeful part of me is thinking

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair01/listener_asr.json` |

### pair 02 · 8.680–19.280 秒

this is the time when education can actually make a difference and a time when people are willing to listen because [DATE] [DATE] [DATE] [DATE] uh all people are going to be focused on is getting reelected.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair02/listener_asr.json` |

### pair 03 · 19.300–23.680 秒

And at that point, uh they're not necessarily going to be listening to a bunch of new ideas.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pSSy7yy/pair03/listener_asr.json` |

## pp26ca6

时间轴 120.67 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/pp26ca6.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/pp26ca6.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/pp26ca6.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/pp26ca6/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/pp26ca6.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/pp26ca6.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/pp26ca6.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/pp26ca6/listener.json` |

### pair 01 · 0.000–2.660 秒

Okay, so what would your family warn me about if we were going to

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair01/listener_asr.json` |

### pair 02 · 66.900–73.020 秒

cruised like through like my sister had to deal with all of the she's my older sister I have an older sister and a younger brother

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair02/listener_asr.json` |

### pair 03 · 73.100–84.080 秒

um sister had to deal with all the like the like breaking down the barriers of like can I get a later curfew um you know uh when am I allowed to

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair03/listener_asr.json` |

### pair 04 · 84.080–92.920 秒

do this stuff and then by the time like she kind of like worked my parents in so then when I was like oh I got a cell phone in eighth grade instead of high school like oh like I can have a little later curfew

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/pp26ca6/pair04/listener_asr.json` |

## qWRcSCu

时间轴 83.05 秒，观察对象 pair 5 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/qWRcSCu.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/qWRcSCu.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/qWRcSCu.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/qWRcSCu.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/qWRcSCu.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/qWRcSCu.json` |

### pair 01 · 0.120–3.360 秒

That you just came back from the [LOC] conference in [LOC] How was

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair01/listener_asr.json` |

### pair 02 · 7.000–16.960 秒

It's on my list. I really want to go. Um, okay, I know we have little time, a lot to talk about, so let's get right into it because there are a lot of concerns surrounding security.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair02/listener_asr.json` |

### pair 03 · 16.960–26.830 秒

We've been hearing just these heartbreaking, devastating stories of people putting their seed phrases onto password managers, and then poof, all of their Bitcoin is gone.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair03/listener_asr.json` |

### pair 04 · 26.840–34.080 秒

So I it's something we talk about a lot is self-custody, the importance of that, but also where you put your seed phrase is also critical.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair04/listener_asr.json` |

### pair 05 · 34.100–39.370 秒

So, can you talk a little bit about the lessons that we're learning from some of the anecdotes that are in the space?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/qWRcSCu/pair05/listener_asr.json` |

## rc1Awgw

时间轴 56.60 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/rc1Awgw.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/rc1Awgw.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/rc1Awgw.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/rc1Awgw/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/rc1Awgw.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/rc1Awgw.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/rc1Awgw.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/rc1Awgw/listener.json` |

### pair 01 · 0.000–3.280 秒

Where do you see us five years from now?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair01/listener_asr.json` |

### pair 02 · 4.960–8.060 秒

I see us on like a beach somewhere.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair02/listener_asr.json` |

### pair 03 · 23.820–26.280 秒

Yeah, you're to be old, bro. I

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/rc1Awgw/pair03/listener_asr.json` |

## sVStK1M

时间轴 37.81 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/sVStK1M.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/sVStK1M.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/sVStK1M.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/sVStK1M.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/sVStK1M.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/sVStK1M.json` |

### pair 01 · 0.120–10.140 秒

So, like the fact that there is 21 million, the fact that like well once you understand that this is a a beast or a thing that is um

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair01/listener_asr.json` |

### pair 02 · 10.140–20.880 秒

got more to do with uh using the inefficiency of social consensus, like use it in a really unique way, which is once something has like

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair02/listener_asr.json` |

### pair 03 · 20.880–24.800 秒

got too many people uh required to make a decision, nothing changes.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair03/listener_asr.json` |

### pair 04 · 24.800–35.200 秒

Once you realize that, it's a very unique way to create a network that doesn't change, and that's Bitcoin's most uh useful power. So therefore it doesn't matter who created it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/sVStK1M/pair04/listener_asr.json` |

## swLFTvk

时间轴 95.10 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/swLFTvk.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/swLFTvk.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/swLFTvk.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/swLFTvk.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/swLFTvk.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/swLFTvk.json` |

### pair 01 · 0.120–10.730 秒

[NAME] how do you think that the forces of freedom, which I think represent Bitcoin, this return to sound money, hardness, individual sovereignty will intersect

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair01/listener_asr.json` |

### pair 02 · 10.740–14.300 秒

with the forces of control, being the regulators, the government.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair02/listener_asr.json` |

### pair 03 · 36.040–38.460 秒

So so what do you think about that?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swLFTvk/pair03/listener_asr.json` |

## swbqUS8

时间轴 140.68 秒，观察对象 pair 6 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/swbqUS8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/swbqUS8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/swbqUS8.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/swbqUS8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/swbqUS8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/swbqUS8.json` |

### pair 01 · 0.120–10.460 秒

I want to actually dig into a piece of that. And you know, one thing that I do find interesting is that Bitcoin was first. And uh [NAME] [NAME] talks about this in her new book, Broken Money.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair01/listener_asr.json` |

### pair 02 · 10.460–18.960 秒

It could have been a project that isn't that great, right? That the creator runs away with the coins that they mined, and you know, somehow the network doesn't work.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair02/listener_asr.json` |

### pair 03 · 18.980–30.240 秒

It's amazing that the first one is actually this beautiful, perfect, powerful, uh, you know, original blockchain that actually could serve as a superior form of money and a global reserve asset.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair03/listener_asr.json` |

### pair 04 · 30.260–39.760 秒

So um, but one thing that she also talks about is really separating the difference between a digital uh store of value and a superior monetary network versus what

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair04/listener_asr.json` |

### pair 05 · 39.780–49.600 秒

maybe are interesting ideas, but ultimately are essentially digital equities. So you said in the report, other non-bitcoin projects should be evaluated from a different perspective than Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair05/listener_asr.json` |

### pair 06 · 49.620–52.700 秒

And you started to kind of go into why.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/swbqUS8/pair06/listener_asr.json` |

## t2htkuW

时间轴 67.27 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/t2htkuW.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/t2htkuW.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/t2htkuW.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/t2htkuW.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/t2htkuW.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/t2htkuW.json` |

### pair 01 · 0.080–5.760 秒

Driven by grassroots movement leaders because they are already fighting the fight for freedom.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair01/listener_asr.json` |

### pair 02 · 5.760–12.800 秒

And once they understand that Bitcoin is a tool of freedom, they will make it their own and will use it to win their struggles.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair02/listener_asr.json` |

### pair 03 · 12.800–15.760 秒

So that is literally how I see this thing going and what I've been

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair03/listener_asr.json` |

### pair 04 · 59.110–67.155 秒

When you look at a country like [LOC] [DATE] [DATE] their currency locked at some points up to 70% of its value. When you look at the Nigerian Naira, the currency lost.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/t2htkuW/pair04/listener_asr.json` |

## tp1LyoD

时间轴 106.87 秒，观察对象 pair 7 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/tp1LyoD.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/tp1LyoD.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/tp1LyoD.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/tp1LyoD.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/tp1LyoD.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/tp1LyoD.json` |

### pair 01 · 0.080–10.320 秒

going to a bank and the bank doesn't have the money, uh instead we have to worry about the fact that our the value of our assets in our bank just keeps going down because we've taken away like the

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair01/listener_asr.json` |

### pair 02 · 10.320–16.240 秒

the prospect for a loss, but instead we just continually print and smooth it out over the course.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair02/listener_asr.json` |

### pair 03 · 16.280–27.710 秒

So free banking is one of those things where both critics and defenders of it, I think have certain points because the inherent kind of function of banks in that environment is just unstable.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair03/listener_asr.json` |

### pair 04 · 27.710–37.440 秒

But it's not always the instability that a central bank would argue it was, especially if you cherry pick the the the jurisdictions where it had the most problems and ironically

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair04/listener_asr.json` |

### pair 05 · 37.440–40.540 秒

wasn't as free as other jurisdictions.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair05/listener_asr.json` |

### pair 06 · 92.680–99.520 秒

If you kind of look at wealth concentration you had, it was it was rising through you know up until [DATE] [DATE] [DATE] and then

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair06/listener_asr.json` |

### pair 07 · 99.520–106.755 秒

starting with the Great Depression and the and the World War II and then kind of the the redistribution that happened, it started falling into [DATE] [DATE] and

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/tp1LyoD/pair07/listener_asr.json` |

## twUBM0l

时间轴 81.37 秒，观察对象 pair 0 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/twUBM0l.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/twUBM0l.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/twUBM0l.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/twUBM0l/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/twUBM0l.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/twUBM0l.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/twUBM0l.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/twUBM0l/listener.json` |

这条没有切出观察对象 pair。

## u4TjE5S

时间轴 113.77 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/u4TjE5S.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/u4TjE5S.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/u4TjE5S.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/u4TjE5S/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/u4TjE5S.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/u4TjE5S.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/u4TjE5S.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/u4TjE5S/listener.json` |

### pair 01 · 0.000–5.420 秒

A secret.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair01/listener_asr.json` |

### pair 02 · 16.700–21.520 秒

You could have been too afraid to tell you.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair02/listener_asr.json` |

### pair 03 · 27.440–31.640 秒

know, um, it's got to be something,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair03/listener_asr.json` |

### pair 04 · 43.520–48.470 秒

right, sure, but, I mean, um, I pick my booger sometimes,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/u4TjE5S/pair04/listener_asr.json` |

## uMFYbA2

时间轴 45.67 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/uMFYbA2.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/uMFYbA2.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/uMFYbA2.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/uMFYbA2.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/uMFYbA2.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/uMFYbA2.json` |

### pair 01 · 0.080–8.120 秒

But let's just press press pause there. A financial meltdown, lashing out at people important to you, going into a dark place.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/uMFYbA2/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/uMFYbA2/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/uMFYbA2/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/uMFYbA2/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/uMFYbA2/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/uMFYbA2/pair01/listener_asr.json` |

## ueXQsPb

时间轴 122.19 秒，观察对象 pair 15 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/ueXQsPb.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/ueXQsPb.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/ueXQsPb.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/ueXQsPb.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/ueXQsPb.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/ueXQsPb.json` |

### pair 01 · 0.640–8.560 秒

So what does it mean? Well, I I expect, first of all, I think it means the [REDACTED] going to have to tread really

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair01/listener_asr.json` |

### pair 02 · 8.560–15.120 秒

lightly and be careful not to compete with the [REDACTED] on the long end of the curve. And what do I mean by that?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair02/listener_asr.json` |

### pair 03 · 15.120–25.680 秒

Well, they've been they've been uh executing [NAME] In reality, mostly what they've been doing is just letting bonds that they hold on their

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair03/listener_asr.json` |

### pair 04 · 25.720–29.140 秒

on the balance sheet mature off the books.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair04/listener_asr.json` |

### pair 05 · 29.480–37.600 秒

But if the I mean, this is gonna take you know, that'll take [DATE] for them to just let this stuff mature off the books.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair05/listener_asr.json` |

### pair 06 · 37.620–42.480 秒

I mean, so they can't, if they want to get this off, they're gonna have to actually be selling things.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair06/listener_asr.json` |

### pair 07 · 42.500–51.680 秒

But what I'm saying is that they still have $7.9 trillion of uh bonds left in their books, so of assets left on their books.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair07/listener_asr.json` |

### pair 08 · 51.680–58.400 秒

So I don't see that going anywhere fast. So I think that active [NAME] is kind of off the table.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair08/listener_asr.json` |

### pair 09 · 58.450–70.400 秒

Uh and and just as [NAME] [NAME] said to me [DATE] [DATE] was the [REDACTED] is they're they're just trying to execute this and make it so boring it's like paint drying, right?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair09/listener_asr.json` |

### pair 10 · 70.420–72.880 秒

And so okay, I agree.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair10/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair10/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair10/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair10/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair10/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair10/listener_asr.json` |

### pair 11 · 72.900–84.400 秒

Uh, but the reality is there's no way that they're going to be able to step in and execute active [REDACTED] It's just like this, this is a big signal, in my opinion.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair11/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair11/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair11/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair11/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair11/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair11/listener_asr.json` |

### pair 12 · 84.400–94.090 秒

So that's the second thing is I would expect the [REDACTED] to continue to lean heavily on the short end of the yield curve in their issuance.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair12/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair12/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair12/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair12/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair12/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair12/listener_asr.json` |

### pair 13 · 94.120–102.160 秒

So what like we talked about, the the reverse repo facility has about 900 billion dollars in it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair13/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair13/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair13/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair13/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair13/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair13/listener_asr.json` |

### pair 14 · 102.160–112.720 秒

And so that 900 billion dollars is that's going to be drawn out for funding the government. Like they're they're just drawing that out and putting in the T bills.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair14/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair14/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair14/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair14/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair14/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair14/listener_asr.json` |

### pair 15 · 112.740–122.095 秒

But those are continuously maturing. So they're gonna have to continue to issue T bills, and and they're issuing these T bills at what? The highest rate on the curve.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair15/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair15/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair15/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair15/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair15/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ueXQsPb/pair15/listener_asr.json` |

## vMyQurP

时间轴 54.30 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/vMyQurP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/vMyQurP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/vMyQurP.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/vMyQurP/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/vMyQurP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/vMyQurP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/vMyQurP.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/vMyQurP/listener.json` |

### pair 01 · 0.000–4.700 秒

When was the last time you considered ending this relationship and why didn't you?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vMyQurP/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vMyQurP/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vMyQurP/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vMyQurP/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vMyQurP/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vMyQurP/pair01/listener_asr.json` |

## vdugLXv

时间轴 46.37 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/vdugLXv.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/vdugLXv.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/vdugLXv.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/vdugLXv/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/vdugLXv.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/vdugLXv.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/vdugLXv.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/vdugLXv/listener.json` |

### pair 01 · 0.000–3.060 秒

What are you hesitant to tell me?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vdugLXv/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vdugLXv/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vdugLXv/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vdugLXv/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vdugLXv/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vdugLXv/pair01/listener_asr.json` |

## vwORVmd

时间轴 32.20 秒，观察对象 pair 2 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/vwORVmd.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/vwORVmd.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/vwORVmd.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/vwORVmd.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/vwORVmd.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/vwORVmd.json` |

### pair 01 · 15.520–24.720 秒

Yeah, so they're 575 in one at least that takes into account like all the ETFs they've ever launched, including the ones they've thus like closed or liquidated, and they've only been denied once.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair01/listener_asr.json` |

### pair 02 · 24.740–30.240 秒

And subsequent, that application basically, they were um they were applying for a certain way to structure an ETF.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwORVmd/pair02/listener_asr.json` |

## vwTzvfP

时间轴 49.09 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/vwTzvfP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/vwTzvfP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/vwTzvfP.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/vwTzvfP/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/vwTzvfP.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/vwTzvfP.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/vwTzvfP.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/vwTzvfP/listener.json` |

### pair 01 · 0.000–2.940 秒

What do you remember about the first time we met?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwTzvfP/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwTzvfP/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwTzvfP/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwTzvfP/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwTzvfP/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vwTzvfP/pair01/listener_asr.json` |

## vypBLvi

时间轴 40.88 秒，观察对象 pair 3 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/vypBLvi.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/vypBLvi.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/vypBLvi.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/vypBLvi.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/vypBLvi.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/vypBLvi.json` |

### pair 01 · 0.080–5.420 秒

With this with this merger, and uh we are very, very excited to get to the other side of it.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair01/listener_asr.json` |

### pair 02 · 30.760–35.600 秒

Yeah, I think it's gonna be really interesting uh to see how things evolve.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair02/listener_asr.json` |

### pair 03 · 35.620–40.641 秒

Uh you [LOC] [LOC] [LOC] really did become the the net beneficiary of the of the

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/vypBLvi/pair03/listener_asr.json` |

## wqJFJpy

时间轴 33.23 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/wqJFJpy.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/wqJFJpy.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/wqJFJpy.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/wqJFJpy.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/wqJFJpy.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/wqJFJpy.json` |

### pair 01 · 0.120–11.040 秒

It's so amazing that we are at a point where all of this, I mean, [DATE] of research and development and work in cryptography culminated in something as perfect as Bitcoin.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair01/listener_asr.json` |

### pair 02 · 11.060–20.990 秒

Because I think it's [NAME] that talks about this in broken money. It didn't, it maybe it maybe the perfect one would have been the third or fourth iteration of a cryptocurrency, but it was really the first.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair02/listener_asr.json` |

### pair 03 · 20.990–28.080 秒

I mean, Bitcoin, the first one, is the best. It's the most perfectly engineered form of money.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair03/listener_asr.json` |

### pair 04 · 28.100–33.135 秒

And going back to what you mentioned earlier and why ultimately you decided that Bitcoin's the one you want to focus on.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/wqJFJpy/pair04/listener_asr.json` |

## xHulOn8

时间轴 111.21 秒，观察对象 pair 4 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/xHulOn8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/xHulOn8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/xHulOn8.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/xHulOn8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/xHulOn8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/xHulOn8.json` |

### pair 01 · 40.640–46.160 秒

[LOC] and our allies, by the way, which I believe need to step up in much more significant manners.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair01/listener_asr.json` |

### pair 02 · 46.160–56.220 秒

I believe [LOC] and our allies who believe in the preservation of democracy and free nations have to support [LOC] until they win.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair02/listener_asr.json` |

### pair 03 · 56.220–65.760 秒

And anything short of that, [NAME] in my estimation, is exactly the hall pass that was given to [NAME] [NAME] when he took [LOC] [LOC] is watching,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair03/listener_asr.json` |

### pair 04 · 65.760–70.560 秒

[LOC] [LOC] is watching, uh, [LOC] is watching, especially as it relates to

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xHulOn8/pair04/listener_asr.json` |

## xrbASOb

时间轴 87.07 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/xrbASOb.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/xrbASOb.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/xrbASOb.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/xrbASOb/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/xrbASOb.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/xrbASOb.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/xrbASOb.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/xrbASOb/listener.json` |

### pair 01 · 1.440–4.260 秒

When do you worry for me the most?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xrbASOb/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xrbASOb/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xrbASOb/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xrbASOb/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xrbASOb/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/xrbASOb/pair01/listener_asr.json` |

## yLysrQ2

时间轴 69.54 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/yLysrQ2.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/yLysrQ2.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/yLysrQ2.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/yLysrQ2/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/yLysrQ2.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/yLysrQ2.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/yLysrQ2.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/yLysrQ2/listener.json` |

### pair 01 · 0.000–2.320 秒

What can I do to be a better father?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yLysrQ2/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yLysrQ2/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yLysrQ2/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yLysrQ2/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yLysrQ2/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yLysrQ2/pair01/listener_asr.json` |

## ybrqds8

时间轴 114.45 秒，观察对象 pair 14 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/ybrqds8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/ybrqds8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/ybrqds8.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/ybrqds8.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/ybrqds8.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/ybrqds8.json` |

### pair 01 · 0.080–10.240 秒

Of that was just from four accounts, and 50 million of that was one account tied to [NAME] [NAME] [NAME] that was titled [REDACTED] [REDACTED] [REDACTED] So you have [NAME]

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair01/listener_asr.json` |

### pair 02 · 10.260–16.800 秒

[NAME] [REDACTED] [REDACTED] you have [REDACTED] and [REDACTED] and [REDACTED] [REDACTED] and [REDACTED] are obviously connected.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair02/listener_asr.json` |

### pair 03 · 16.800–22.560 秒

And they're basically uh using this trying to use this tiny bank for something.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair03/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair03/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair03/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair03/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair03/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair03/listener_asr.json` |

### pair 04 · 22.560–34.220 秒

And based on just what was what is documentable, um, there's no way this bank should have been approved by the [REDACTED] It should never have been made part of the [REDACTED] system.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair04/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair04/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair04/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair04/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair04/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair04/listener_asr.json` |

### pair 05 · 34.220–43.120 秒

Um, and the [REDACTED] [REDACTED] system can't explain, refuses to explain still how they approved it and why.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair05/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair05/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair05/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair05/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair05/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair05/listener_asr.json` |

### pair 06 · 43.140–53.070 秒

Uh, and there's also the same sort of like no comment, nothing to see here, even though, like, yes, local reporter, you bring up good points.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair06/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair06/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair06/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair06/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair06/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair06/listener_asr.json` |

### pair 07 · 53.080–59.600 秒

Uh there are they were irrelevant to us at the time or something from like [LOC] financial regulators.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair07/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair07/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair07/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair07/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair07/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair07/listener_asr.json` |

### pair 08 · 59.600–66.880 秒

And uh, if you want more details on that, you can look in the in the two pieces I've I've written about it. But there's definitely something very weird there.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair08/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair08/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair08/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair08/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair08/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair08/listener_asr.json` |

### pair 09 · 66.880–77.520 秒

Like there was somebody in the [REDACTED] system that got this approved when it should not have been approved, and even high people up at the [REDACTED] are covering that up.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair09/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair09/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair09/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair09/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair09/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair09/listener_asr.json` |

### pair 10 · 77.520–79.690 秒

That's essentially what's happening here.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair10/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair10/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair10/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair10/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair10/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair10/listener_asr.json` |

### pair 11 · 79.810–91.760 秒

Um, and so what was you know, it [REDACTED] collapses, and all this scrutiny obviously gets placed on this tiny bank, even from mainstream media and also from congressmen.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair11/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair11/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair11/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair11/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair11/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair11/listener_asr.json` |

### pair 12 · 91.890–97.360 秒

Like it was obviously like something was going on there, and it was very obvious to see.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair12/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair12/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair12/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair12/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair12/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair12/listener_asr.json` |

### pair 13 · 97.360–106.400 秒

And then the [REDACTED] has since like issued like enforcement actions against [LOC] and like forced it to close down, and so it doesn't exist anymore.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair13/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair13/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair13/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair13/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair13/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair13/listener_asr.json` |

### pair 14 · 106.400–113.440 秒

Yeah, but they had big plans for it before [REDACTED] unraveled, and that's essentially what this piece is about.

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair14/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair14/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair14/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair14/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair14/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ybrqds8/pair14/listener_asr.json` |

## yc5XJ6Z

时间轴 99.10 秒，观察对象 pair 1 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/yc5XJ6Z.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/yc5XJ6Z.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/yc5XJ6Z.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/yc5XJ6Z/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/yc5XJ6Z.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/yc5XJ6Z.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/yc5XJ6Z.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/yc5XJ6Z/listener.json` |

### pair 01 · 1.260–4.640 秒

What's one experience you wish I never had?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yc5XJ6Z/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yc5XJ6Z/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yc5XJ6Z/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yc5XJ6Z/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yc5XJ6Z/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/yc5XJ6Z/pair01/listener_asr.json` |

## zQbLB6Q

时间轴 30.47 秒，观察对象 pair 2 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/zQbLB6Q.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/zQbLB6Q.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/zQbLB6Q.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/zQbLB6Q/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/zQbLB6Q.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/zQbLB6Q.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/zQbLB6Q.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/zQbLB6Q/listener.json` |

### pair 01 · 0.000–4.780 秒

Aww, this is sweet. Okay, what does my friendship feel like?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair01/listener_asr.json` |

### pair 02 · 14.000–18.170 秒

No, you're gonna, I'm just crying real quick. No, what do you, what?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/zQbLB6Q/pair02/listener_asr.json` |

## ztN2H5z

时间轴 48.80 秒，观察对象 pair 2 条。

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 官方整段 · 观察对象 | `/public/0_DATA/2_Avatar/ResponseNet/test/video/speaker/ztN2H5z.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/speaker/ztN2H5z.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/speaker/ztN2H5z.json` |
| 本次切分用的词时间 · 观察对象 |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/ztN2H5z/speaker.json` |
| 官方整段 · Agent | `/public/0_DATA/2_Avatar/ResponseNet/test/video/listener/ztN2H5z.mp4` | `/public/0_DATA/2_Avatar/ResponseNet/test/audio/listener/ztN2H5z.wav` | `/public/0_DATA/2_Avatar/ResponseNet/test/text/listener/ztN2H5z.json` |
| 本次切分用的词时间 · Agent |  |  | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/local_asr/ztN2H5z/listener.json` |

### pair 01 · 0.000–2.880 秒

All right, what does my style say about me?

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair01/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair01/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair01/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair01/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair01/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair01/listener_asr.json` |

### pair 02 · 29.980–32.240 秒

what, like, is there something specific that, like,

| 角色 | 画面 | 音轨 | ASR |
|---|---|---|---|
| 观察对象 | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair02/speaker.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair02/speaker.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair02/speaker_asr.json` |
| Agent | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair02/listener.mp4` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair02/listener.wav` | `/private/liqi_workspace/project_code/audio-driven-v5/evaluation/responsenet_speaker_j/cuts/ztN2H5z/pair02/listener_asr.json` |
