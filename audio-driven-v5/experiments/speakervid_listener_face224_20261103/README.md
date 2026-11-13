# SpeakerVid 测试原片 · 大脸 listener 池

2026-11-03。从已下载的 SpeakerVid 测试原片中，留下口型能分开、清晰度保留、listener 脸短边 p10≥224 的片段。从中抽出 400 条作为测试子集。

## 清单

[pool.jsonl](pool.jsonl)：1514 条，2.503 小时，37 条原片。每行一个片段，字段包括 `video_id`、`start`、`end`、`duration_s`、`face_short_p10`、`speaker_conf`、`listener_conf`、`listener_mp4`、`source_mp4`。

[summary.json](summary.json)：段数、时长和不少于 5 秒的子集统计。

[selected_400.jsonl](selected_400.jsonl)：400 条，播放 0.974 小时，27 条原片。播放文件用 Seamless 最终头肩裁剪框重新裁过，头顶留在画面内；5 秒按这个文件的实际时长计。同一原片起点至少隔开 25 秒，并优先留下更清晰的脸。`file_dur` 是播放时长，`duration_s` 是原片窗口，`crop_x/y/size` 是最终框。按原片的条数见 [selected_400_summary.json](selected_400_summary.json)。

## 媒体

- listener 脸：`/public/0_DATA/2_Avatar/Speakervid/realtalk_fullscan/full_front/listener/{id}.mp4`
- 切之前的整幅画面：`/public/0_DATA/2_Avatar/Speakervid/realtalk_fullscan/full_front/clips/{id}.mp4`
- 页面上可翻看的是更大的「已分开」集合（7368 段／11.23 小时／85 条原片），不是这 1514 条：[已分开的 listener](http://localhost:18780/omniresponse-media/benchmark_guide/0#lispool)

## 进入这 1514 条的条件

1. 场景切分后去掉黑屏，片段内说话人日志只有一个人，单段不超过 11 秒。
2. 片段内 1 秒处正好两张够大的正脸，左右分开。正脸候选共 7967 段／11.61 小时。
3. SyncNet 口型分数能分开：较高分不低于 1.0，且比另一张脸至少高 0.5。分出来的是 7368 段／11.23 小时／85 条原片。播放文件是分数较低的那张脸。
4. 沿用 `listen_visual_quality_v1`：相对本段 P80 没有连续 2 秒或累计 25% 的明显变糊，判定为 retain。
5. listener 脸在原图像素上的短边 p10≥224。没有一条达到 384。

清晰度通过但脸短边不足 224 的片段不在本清单。本清单未再按来源去重，相邻的 10 秒片段可以来自同一次说话。
