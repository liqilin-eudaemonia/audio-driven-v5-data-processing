# V5：不超过 10 秒的子话轮与统一音量

日期：2026-10-21。由 `seamless_speaker_span_v4` 的 50 个 interaction、457 个父区间产生。

## 时间切分

872 个子话轮，最长 10.0 秒，总时长 6574.949 秒。208 个长父区间被进一步切分；子话轮连续、不重叠，完整保留父区间时间，不跨原话轮排除区间。

优先使用 speaker 停顿或词间隙，并尽量避开 listener 短反馈。33 个切点没有合适词间隙，按 10 秒硬切；1 个切点截断 listener 活动，记录在 `subturn_audit.json`。原始词/VAD 时间戳为自动标注，不是人工确认标签。

每人视频 512×512、25 FPS；双人对照 1024×512。子话轮时间不超过 10 秒，视频向下量化到帧，较对应音轨/时间区间最多短一个帧周期（40 ms）。双方视频与双人对照帧数相同。

## 音量处理

100 条原始参与者完整 WAV 分别测量 EBU R128 响度。按目标 −18 LUFS 计算固定增益、限制峰值并复测，最终响度范围 −18.29～−18.02 LUFS，最大真峰值 −0.97 dBTP。增益上限 45 dB，用于原始语音特别轻的音轨；无足够转写语音或近乎数字静音的轨道不盲目放大。

同一原始音轨在所有子话轮和角色中共享同一结果，保留静音与轨内相对动态。限幅器的前视延迟已补偿，全部源音轨长度核验通过；低/中/高原始响度三条音轨的相关分析均为 0 样本偏移。`normalized_audio/` 中保存无损 WAV 及增益、处理前后测量、来源与哈希。

子话轮预览使用新音量；完整会话上下文仍是原始音量，页面明确区分。正式训练重新提取原始 MP4，音频截取上述归一化 WAV，不用预览 AAC 做训练。

## 入口与复现

当前预览已发布到 <http://127.0.0.1:8875/rule_turns/> 和 <http://127.0.0.1:8875/speaker_span/>，不新增旧版转发入口。

从项目父目录执行：

```bash
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/prepare_listen_subturns.py prepare --workers 6
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/prepare_listen_subturns.py render --workers 8
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/audit_listen_subturns.py
python audio-driven-v5/scripts/publish_listen_subturns.py
```

- `manifest.json`：子话轮、父区间、裁剪框、音轨归一化来源。
- `summary.json`、`subturn_audit.json`：统计与所有切点。
- `audio_boundary_validation.json`：完整时间覆盖、音轨时长、响度/峰值及抽样同步检查。
- `media_validation.json`：5282 个媒体文件检查通过，50 个最长对照片段完整解码通过。
- `browser_validation.json`：播放、音轨切换、上下文、筛选、审阅导出、Range 下载和移动端宽度检查通过。

导出已增加第二遍完整音轨封装，解决旧 FFmpeg 在限制视频帧数时截断 MP4 音频尾部的问题。视频流直接复制，独立 M4A 和归一化 WAV 不受影响。

这些检查不替代人工试听和反应适宜性判断。
