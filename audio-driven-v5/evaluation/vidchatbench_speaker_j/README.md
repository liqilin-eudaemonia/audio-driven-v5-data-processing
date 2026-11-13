# VidChatBench：观察对象固定的 J 切分

画面和声音从 `/public/0_DATA/2_Avatar/Speakervid/videos_fps24` 的 24fps 单人片段读取。观察对象用 human 那条，Agent 用另一条。官方 ASR 没有词时间，词时间是对这两条音轨重新转写的。音质模型不跑。

说话人固定为 human，页面上是 Speaker / Human · 观察对象。agent 自己说话的区间不反向切。音质模型不跑。冻结的 J v1 规则文件不改。

官方 `official_asr.json` 只有句级起止，没有词时间，不能拿来下刀。切分用的词时间来自本地修复时已经落盘的 Whisper large-v3 词级转写：`speakervid_repair_20261021/asr/<id>/{human,agent}.json`。页面上的官方 ASR 仍是原文，不改字。

## 命令

```bash
python evaluation/vidchatbench_speaker_j/segment.py --all
python evaluation/vidchatbench_speaker_j/publish_review.py
```

8 条里只有 2 条切出了观察对象 pair。第二条命令默认把这两条放上原视频审核页：

- http://localhost:8876/seamless_face_listening_v1/listen_full_source/?pair=P8ZY3udZ7no_1920x1080_full_video_118_A_00
- http://localhost:8876/seamless_face_listening_v1/listen_full_source/?pair=ep3yBD3MjM8_1920x1080_full_video_057_A_01
