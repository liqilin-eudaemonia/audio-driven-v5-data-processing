# Speaker 起止 V4

基于 V2 同一批 50 个 interaction、457 个连续倾听区间，按用户确认的规则统一裁剪双方视频及独立音轨：

- 起点：原 V2 区间内 speaker 第一个已转写词的开始时间，受原区间起点限制。
- 终点：原 V2 区间内 speaker 最后一个已转写词的结束时间，受原区间终点限制。
- 不额外留首尾静音。保留 speaker 内部停顿，不将有效发声时长相加后拼接。
- 区间外的 listener 发声不保留；与 speaker 首尾重叠的部分仍保留，不为移除 listener VAD 截短 speaker。
- 沿用 V2 主话轮与插话排除区间，不重新判断话轮、不跨排除区间扩展或拼接。
- 未运行新 ASR、VAD 或其他模型；词级时间戳为数据集自动标注，不是人工确认的声学端点。

## 结果

185 段收紧开头，共去掉 115.491 秒；324 段收紧结尾，共去掉 347.732 秒。总长 6574.949 秒，约 109.6 分钟。37 段裁短后不足 4 秒，仍保留用于审阅。

每人保持 512×512、25 FPS，双人对照 1024×512；使用原 V2 固定空间裁剪框。双方音视频使用同一时间区间，实际视频时长受帧周期量化限制（25 FPS 为 40 毫秒），AAC 也有编码帧粒度。

页面独立显示 listener 转写与 VAD，父话轮中的事件标明是否在本段范围内。情绪曲线保留已有采样点及其新相对时间，不插值生成标签。区间外事件仍能通过完整会话回看。

## 入口与复现

- 新版：<http://127.0.0.1:8875/speaker_span/>。
- 默认入口也已更新至 V4：<http://127.0.0.1:8875/rule_turns/>。
- V1/V2/V3 页面和媒体访问已按用户要求停用（HTTP 410），文件仍保留在本地。8875 继续提供 V4 及其他现用入口。
- `manifest.json`：原/新起止、裁剪量、speaker 首尾词锚点、角色及所有媒体路径。
- `boundary_event_audit.json`：被裁掉或截短的 listener 事件。
- `boundary_validation.json`、`media_validation.json`、`browser_validation.json`：边界、媒体与页面检查记录。

```bash
python scripts/trim_seamless_speaker_span.py --workers 12
python scripts/publish_speaker_span_preview.py
python scripts/validate_rule_turns_media.py --directory data_previews/seamless_speaker_span_v4
```

本版属于审阅预览：精确同边界的文件复用 V2/V3 媒体；需要重新裁开头的文件直接从 V2 媒体裁剪并重编码一次，避免在 V3 上继续累积编码。完整会话复用原文件。最终训练数据应按本版 manifest 从原始视频和独立 WAV 提取。需保留 V2/V3 目录以维持媒体引用。

## 完成验证

全部 457 段边界与相对时间标注检查通过。2792 个媒体文件的流信息、尺寸、帧率和时长检查通过；每个会话最长对照片段完整解码，共 50 条。逐段核对 speaker、listener 和双人对照视频，三者的视频流时长完全一致；相对目标时长最大视频误差 39 毫秒、最大音频误差 22 毫秒。开头裁剪量最大的 5 段另抽查双方音轨，新音频与按正确偏移截取的 V2 音频相关系数均超过 0.987。

浏览器播放、listener 独立音轨、上下文跳转、筛选、审阅导出、下载 Range 请求和移动端宽度检查通过。综合结果见 `validation.json`。这些检查确认裁剪与媒体可用性，不等同于人工确认语义标签或原始音画同步。
