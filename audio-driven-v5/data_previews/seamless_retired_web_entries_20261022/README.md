# Seamless 倾听脸部音视频：首批审阅

这批用于人工检查分段、裁脸和音轨质量，不是已确认的 backchannel 训练集。

完成结果：50 个 interaction，来自 16 个录制 session，共 85 段、998.373 秒（约 16 分 38 秒）。其中 47 段含 listener 短发言（待区分反馈、插话及正式接话），38 段为较安静候选。

根据首轮反馈，脸部输出更新为每人 512×512，并增加单独的 listener/speaker 音视频下载。原有 85 段的配对、时间边界和裁剪框保持一致，以便对照审阅。第一版分段仍是下文的停顿/长度规则；新的主话轮与插话处理方式尚在讨论，没有把显示名称修改当作重新完成分段。

初版验证：340 个输出音视频文件均通过完整解码和时长检查；浏览器实测播放、独立 listener 音轨、原画面同步、筛选、跳转、备注保存和 JSON 导出正常；手机宽度下无横向溢出。512 版本另对重新生成的双人及单人视频检查分辨率、帧率、音轨、时长与解码，结果记录在 `validation.json` 的 `revision_512`。这些工程检查不代表分段语义或声音标签已经人工核验。

## 查看方式

打开同目录 `index.html`，或在工作机运行：

```bash
python /private/liqi_workspace/project_code/audio-driven-v5/scripts/serve_seamless_preview.py --port 8875
```

在 Cursor 的 Ports（端口）面板转发远程 `8875` 到本机，然后访问 `http://127.0.0.1:8875/`（如果本地端口被占用，以 Cursor 显示的 Local Address 为准）。不要直接使用工作机的 `11.1.*` 内部网卡地址。本次已通过 Cursor 远程 CLI 的 `--openExternal http://127.0.0.1:8875/` 触发访问，服务日志确认收到页面、缩略图及视频请求。

HTML 内嵌样本清单，没有外部 CDN 依赖；若复制到另一台机器，需要保留 `media/`、`assets/` 和 HTML 的相对位置。

- 左侧筛选、搜索与选择片段；每个 interaction 有 1–2 段。
- 左侧视频为 speaker，右侧为 listener。可听双方混音，也可单独听其中一人。
- 原画面对照展开后跟随主播放器播放，绿色方框显示实际裁剪范围。
- 时间轴分别展示双方原有 VAD 和 listener 转写，不把两类信息混为一个标签。
- 情绪曲线显示 listener 的 valence/arousal 模型原始值，没有将未知类别顺序的 emotion_scores 解释为情绪类别。
- 标签和文字备注保存在当前浏览器 localStorage，可以导出为 JSON 供后续讨论。

## 抽样范围与配对

- 原始数据根目录：`/public/0_DATA/2_Avatar/seamless_interaction/`。
- 使用此前已有的 `audio-driven-v4/evaluation/benchmark_alignment/assets/seamless_test_index.json`，没有重新扫描全部原始数据。
- 当前批次来自 `naturalistic / test / V00`，固定随机种子 `20261021`。此前试抽的多条 V01/V03 缺少请求的 emotion 字段，而这个 test 索引中的 V00 仅位于 naturalistic，因此没有强求两类各半。
- 按相同 vendor/session/interaction、不同 participant 配对；要求恰好两名参与者，并有视频、独立 WAV、现有转写和 VAD。
- 每个最终 pair 的双方 NPZ 都实际检查存在 `movement:emotion_scores`、`movement:emotion_valence`、`movement:emotion_arousal`。
- 目标是 50 个不同 interaction（会话视频对），不是 50 个不同人物组或录制 session；同一组人物可能在不同会话中重复出现。
- 随机打乱候选后应用质量和分段条件，跳过不合格项，直到目标数。它是筛选后的小规模预览，不代表全数据集分布。
- 只读取选中候选相关的归档目录项和必要的媒体/特征。视频通过 FFmpeg subfile 按 TAR 内偏移读取，没有全库解包。

## 倾听区间规则（v1）

1. 使用已有词级时间戳。间隔不超过 0.55 秒的词合并为发声 bout，忽略时间戳为空的词。
2. 将 speaker 的 VAD 与词组区间合并，允许最多 1.5 秒停顿，形成候选主话轮区间。
3. 排除 listener 连续转写超过 2.5 秒或超过 5 词的区间，并留 0.25 秒边界；同时排除 listener 原有 VAD 连续超过 3 秒的区间，留 0.15 秒边界。
4. 在剩余位置提取约 10–12 秒的窗口；要求 speaker 至少有 10 个起点落在窗口内的词，VAD 占比至少 50%；listener 的 VAD 占比不超过 25%。
5. 每个会话优先选择一段含 listener 短转写的候选和一段较安静的候选；缺少某类时用其他合格片段补充。同一会话选取的两段不重叠，角色可互换。
6. 不删除片段内的静音，不单独压缩双方时间轴。

标记含义：

- `short_feedback_candidate`：历史字段名保留以兼容现有清单；页面改称“短发言待分类”。它只表示存在短 listener 转写，不能证明是 backchannel，可能包含插话、短回答或短提问。
- `vocal_candidate`：没有 listener 转写，但其原有 VAD 占比大于 1%。未进一步确认是否为笑声、其他非语言声音或噪声。
- `quiet_candidate`：没有 listener 转写且 VAD 占比不超过 1%。这不证明完全无声，可能仍有漏检。

现有 ASR 和 VAD 不是人工真值。本批没有重新运行 ASR、笑声识别、语义话轮分类或人工逐条听音审核。页面提供孤立音轨和原始时间，供本轮评审发现误分。

## 裁脸与音视频

- 使用 NPZ 的 133 点 WholeBody 关键点中面部部分（23:91），置信度大于 0.4 的点参与估计；要求至少 80% 的片段帧有不少于 45 个合格面部点。
- 根据逐帧面部范围的稳健分位数确定该片段的固定正方形窗口，给头顶、侧边留余量。窗口不逐帧追随面部，保留点头、摇头的空间变化。
- 如果所需窗口超过典型面部宽高最大值的 3.2 倍，跳过该片段，避免大幅侧移让“脸部视频”变成半身视频。部分会话因此只保留一段。
- 检查独立 WAV 与视频时长差不超过 0.15 秒、配对双方时长差不超过 0.2 秒、逐帧关键点长度与视频帧数差不超过 1。
- 两人按相同原始起止秒裁切。脸部从源视频裁切为每人 512×512、25 FPS，合成 1024×512；原画面对照为 640×360、15 FPS。展示分辨率不代表原始人脸具有同样有效细节。
- 使用独立逐人 WAV，不使用 MP4 内混音作为生成目标。试听混音为双方各乘 0.5 后相加；单人试听不做响度归一化。
- MP4/H.264 和 M4A/AAC 为浏览器审阅编码，不能直接替代正式训练用无损目标数据。
- 长度对齐检查不等于已经确认声画精确同步，也不保证没有残余串音。

## 文件与复现

- `manifest.json`：最终会话、角色、原始时间、裁剪框、情绪曲线及源媒体定位。
- `selection.json`：本轮实际尝试的候选、原索引来源及现有转写/VAD。
- `media/<pair>/pair.json`：逐会话记录；`clip_XX/metadata.json` 为逐片段记录。
- `faces.mp4`：双人脸部对照和试听混音；`context.mp4`：静音原画面对照。
- `listener_512.mp4`、`speaker_512.mp4`：各自 512×512 的脸部视频，携带对应人物的独立音轨。
- `speaker.m4a`、`listener.m4a`：独立试听音轨；`poster.jpg`：缩略图。
- `build.log`：构建与跳过原因。

```bash
python scripts/build_seamless_listening_preview.py --count 50 --workers 4
python scripts/publish_seamless_preview.py
```

此次使用 test 数据仅进行人工审阅；未来训练集制作需要另外确定正式 split，不能把本批测试样本默认加入训练。
