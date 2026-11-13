# 左右同框片段的官方声音处理

37 段左右同时在画面里的测试对话，声音按 SpeakerVid-5M 第一阶段的规则处理。画面仍是 Human 那一段原片时间。每个人只保留自己正在说的时段，其余是静音。

听的页面：http://localhost:18780/omniresponse-media/benchmark_guide/0#vidchat  
筛选：「左右同框 · 同一时间」。

这不是把重叠人声拆成两条干净音轨。官方仓库里没有 Demucs 这类源分离。两个人同时说话的区间，贴回去的仍是混在一起的原声。

## 官方规则

代码在 `SpeakerVid-5M-Code/cal_base_anno/auto_single_yolo.py` 的 `split_audio_by_speakers`。第一阶段先用 3D-Speaker 做说话人日志，再用 SyncNet 把人脸框对上某个说话人。写出音轨时从一段等长静音开始，只在这个说话人的时间段里贴上原片声音。

```143:153:project_code/SpeakerVid-5M-Code/cal_base_anno/auto_single_yolo.py
def split_audio_by_speakers(wav,fs, start, speaker_times):
    # audio = AudioSegment.from_file(input_file)
    extracted_audio = from_torchaudio_to_AudioSegment(wav, fs)
    silence = AudioSegment.silent(duration=len(extracted_audio))
    speaker_audio = silence
    for start_speak, end_speak in speaker_times:
        start_speak_ms = (start_speak - start) * 1000
        end_speak_ms = (end_speak - start) * 1000
        speaker_audio = speaker_audio[:start_speak_ms] + extracted_audio[start_speak_ms:end_speak_ms] + speaker_audio[end_speak_ms:]
    wav_from_AudioSegment,fs_from_AudioSegment = from_AudioSegment_to_torchaudio(speaker_audio)
    return wav_from_AudioSegment
```

`readme.md` 第 6 步 `cal_scene_and_audio_diarization` 是同一套日志，说明里写这一步已经并进第一阶段。`split_clip_from_anno.py` 不走这套，它把时间窗里的原片音轨整段截下来，背景声和其他人的声音都还在。

## 这次用的材料

日志用官网已发布的结果，没有重跑 3D-Speaker，也没有重跑 SyncNet。

- 数据集 `dorni/SpeakerVid-5M-Dataset`，revision `4a7e26f2660b89321b038d465a16140e03397351`
- 文件 `raw_labels/speaker_json/tar_batch_00.tar.gz` 到 `tar_batch_19.tar.gz`
- 本地目录 `/public/0_DATA/2_Avatar/Speakervid/hf_raw_labels/speaker_json/`
- 成员名 `{youtube_id}_{宽x高}_full_video.json`

每个 JSON 里要读的是 `speaker_json`。它的每一项是 `{start, stop, speaker}`，`speaker` 为 `A`、`B` 或其他字母。时间是原片上的秒。

顶层还有一个键 `A`，它是全部时段的列表，旁边的 `speaker` 列表才是每一段的字母。不能把顶层的 `A` 当成说话人 A 的时段。

片段名里的字母就是日志里的说话人。`..._full_video_026_A_00` 用 `A`，`..._027_B_00` 用 `B`。字母在文件名倒数第二段。

混合音轨用已经切好的 Human 片段 wav：16 kHz、单声道、这一段原片时间里的完整原声。Agent 不另切一条混合音轨，两边都从这条上取自己的时段。

## 处理步骤

1. 只处理页面上左右同框的 37 段。正反打、镜头里只有一个人的不改。
2. 时间窗用 Human 标注的 `start_seconds` 到 `end_seconds`。Agent 画面已经按这个时间、用 Agent 自己的框裁过。
3. 从 `speaker_json` 取出 Human 字母的时段，和 Agent 字母的时段。
4. 时段和这个时间窗求交。窗外的不留。
5. 各自从静音开始，只把交集里的原声贴回去。音轨长度保持和时间窗一致。
6. 无声画面和这条 wav 合成页面播放用的 mp4。原始无声 mp4 不动。

写出的文件：

- Human / Agent 的 wav：`/public/0_DATA/2_Avatar/Speakervid/testclips_sametime/{id}_{宽x高}/`
- 带声音的播放文件：`/public/0_DATA/2_Avatar/Speakervid/testclips_sametime_view/`
- 页面目录 `omniresponse_media.json` 里，这 37 段的 `svtesthwav`、`svtestawav`、`svtesthplay`、`svtestaplay` 指向上述文件。

## 代码

```python
import json, tarfile, wave
from pathlib import Path
import numpy as np

ROOT = Path("/public/0_DATA/2_Avatar/Speakervid")
SPEAKER_JSON = ROOT / "hf_raw_labels/speaker_json"
OUT = ROOT / "testclips_sametime"

def speaker_letter(stem):
    # 7n4..._full_video_026_A_00 -> A
    return stem.split("_")[-2]

def load_segments(doc):
    segs = {}
    for item in doc["speaker_json"].values():
        segs.setdefault(item["speaker"], []).append((item["start"], item["stop"]))
    return segs

def gate(src_wav, dst_wav, start, end, segs):
    with wave.open(str(src_wav)) as w:
        fs = w.getframerate()
        channels = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        data = data.reshape(-1, channels).copy()
    out = np.zeros_like(data)
    for a, b in segs:
        if a >= end or b <= start:
            continue
        a, b = max(a, start), min(b, end)
        i0 = max(0, int(round((a - start) * fs)))
        i1 = min(len(out), int(round((b - start) * fs)))
        if i1 > i0:
            out[i0:i1] = data[i0:i1]
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst_wav), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(out.tobytes())

def open_speaker_doc(name):
    for tar_path in sorted(SPEAKER_JSON.glob("tar_batch_*.tar.gz")):
        with tarfile.open(tar_path) as tar:
            if name in tar.getnames():
                return json.loads(tar.extractfile(name).read())
    raise FileNotFoundError(name)

# human_stem / agent_stem 来自片段文件名
# mix_wav 是 Human 时间窗上的 16 kHz 单声道原声
# start, end 是 Human 标注的 start_seconds, end_seconds
def gate_pair(human_stem, agent_stem, mix_wav, start, end):
    doc_name = human_stem.split("_full_video_")[0] + "_full_video.json"
    segs = load_segments(open_speaker_doc(doc_name))
    parent = human_stem.split("_full_video_")[0]
    gate(mix_wav, OUT / parent / f"{human_stem}.wav", start, end, segs.get(speaker_letter(human_stem), []))
    gate(mix_wav, OUT / parent / f"{agent_stem}.wav", start, end, segs.get(speaker_letter(agent_stem), []))
```

播放文件用这条命令合成。视频轨直接拷贝，不重新编码。

```bash
ffmpeg -y -i 无声画面.mp4 -i 处理后.wav \
  -c:v copy -c:a aac -b:a 128k -shortest -movflags +faststart \
  播放.mp4
```

## 这 37 段的结果

22 段两边都有声音。12 段 Agent 整段是静音，因为这段时间的日志里没有 Agent 那个字母在说话。`h6eIc6PQkSo_1920x1080_full_video_060_A_02` 的 549.62–559.59 秒两边都是静音，这段日志标的是说话人 C，不是画面上 A 和 B。

第一条 `SMi3JO-hsyY` 的 217.92–223.53 秒：Human（A）留下约 0.75 秒，Agent（B）留下约 4.86 秒。
