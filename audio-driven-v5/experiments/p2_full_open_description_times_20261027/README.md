# 全量开放描述与事件时间统计（无图）

范围：722,706 段，全部保留，包括无七类事件及原先未获训练准入的记录。原始标签、准入标记和缓存不变。

本报告不再把七类当作全部行为。按 **模态＋逐字保留的原文描述** 建立开放目录；同义句、大小写和不同详细程度仍可能分列，不能把原文类型数当成语义类别数。

现有模型只完成了七类语义归类，没有“所有语义类别”的可靠统一分类。本次完整列出原文，不用关键词硬造新类别。额外视觉记录是模型观察候选，可能是普通动作、状态或跨窗重复；不自动认定为 backchannel。

| 统计层 | 涉及片段数 | 标注条目数 | 不同原文描述数 |
|---|---:|---:|---:|
| 未纳入七类的视觉证据 | 635,851 | 10,967,755 | 2,354,624 |
| 后处理整合结果 | 503,895 | 5,527,413 | 1,452,276 |

至少有一条上述记录：711,283 段；两层均无记录：11,423 段。这是有无模型观察，不是有无真实回应的判定。

## 数据入口

- [后处理整合结果：全部原文类型与时长](retained_全部原文类型.csv)
- [额外视觉证据：全部原文类型与时长](unassigned_visual_全部原文类型.csv)
- [原七类时间对照](七类时间对照.csv)
- [可查询的完整原文目录](description_catalog.sqlite)
- [汇总及高频原文](summary.json)
- `shards/*.events.jsonl.gz`：逐事件起止时间，支持流式读取；`retained-*` 为整合后结果，`evidence-*` 为未纳入七类的视觉证据。

## 时间口径

- `intervals`：相对当前片段的起止秒；`source_intervals`：相对原始源视频的起止秒。分离的音频保留区间保持分离，不用外包区间填补静音。
- 每一原文类型、每一片段先求时间并集，再累计为该类型覆盖时长。同一条记录跨越多个保留区间时，持续时长为这些区间之和。
- 原七类表同样在片段内按类求并集；不同类别可以同时发生，各类别时长不能直接相加当总时长。
- 原文目录同义描述可能重叠，目录各行时长也不能简单相加。P10/P50/P90 是标注条目持续时长，使用最近秩定义、1 毫秒分辨率；不代表边界具有毫秒级准确度。
- 条目数不是物理动作次数；额外视觉证据没有重新跨窗整合。统计没有把它们强行并入训练标签。
- 音频只使用后处理保留结果，没有复活被静音规则排除的音频标注。视觉保留原证据中的所有未纳入条目。

## 原七类时间对照（用于与旧报告核对，不是类别上限）

| 类别 | 覆盖片段数 | 累计覆盖小时 | 条目持续 P50 秒 |
|---|---:|---:|---:|
| 点头 | 172,084 | 54.886 | 0.133 |
| 摇头 | 980 | 0.161 | 0.233 |
| 微笑 | 351,385 | 328.761 | 0.133 |
| 大笑／笑声 | 62,228 | 31.494 | 0.367 |
| 皱眉 | 68,581 | 32.241 | 0.367 |
| 惊讶 | 105,972 | 31.345 | 0.133 |
| 音频回应／发声 | 162,953 | 27.930 | 0.300 |

## 后处理整合结果：高频原文示例

| 原文描述 | 片段数 | 区间并集小时 |
|---|---:|---:|
| Yeah | 52,980 | 5.487 |
| Smiling and maintaining eye contact, indicating positive engagement and attentiveness. | 17,648 | 8.894 |
| Laughter | 17,445 | 3.954 |
| Smiling broadly, indicating positive engagement or amusement. | 16,190 | 2.059 |
| Okay | 12,093 | 1.258 |
| Uh huh | 11,799 | 1.244 |
| Smiling with eyes closed, indicating enjoyment or amusement. | 11,577 | 1.085 |
| Nodding head slightly while listening, indicating engagement. | 11,455 | 0.670 |
| Listener maintains steady eye contact and a slight smile, indicating attentive listening and positive engagement. | 11,381 | 8.006 |
| 嗯 | 10,952 | 0.836 |
| Right | 10,753 | 1.069 |
| Smiling while listening, indicating positive engagement. | 10,744 | 0.619 |
| Affirmative vocalization | 9,797 | 0.674 |
| Smiling consistently, conveying positive affect and receptiveness. | 8,226 | 7.920 |
| Smiling broadly, indicating positive affect and engagement. | 8,178 | 2.586 |
| Nodding head slightly while speaking, indicating active listening and engagement. | 7,689 | 0.454 |
| Eyes widen slightly, indicating surprise or heightened attention. | 7,324 | 0.393 |
| Smiling broadly, indicating positive engagement and enjoyment. | 7,264 | 1.423 |
| Smiling while speaking, indicating positive engagement. | 6,836 | 0.334 |
| Smiling slightly, indicating positive engagement or understanding. | 6,795 | 0.452 |
| Smiling while looking down, indicating positive engagement or amusement. | 6,697 | 0.930 |
| Smiling while looking slightly upward, indicating positive engagement or amusement. | 6,612 | 0.563 |
| Smiling slightly, indicating positive engagement or amusement. | 6,352 | 0.328 |
| Yes | 5,821 | 0.692 |
| Smiling while looking upward, indicating positive engagement or amusement. | 5,613 | 0.802 |

## 未纳入七类的视觉证据：高频原文示例

| 原文描述 | 片段数 | 区间并集小时 |
|---|---:|---:|
| Listener maintains steady eye contact with speaker, indicating active listening and engagement. | 114,331 | 104.436 |
| Closing eyes briefly, indicating attentiveness or processing information. | 97,011 | 9.336 |
| Maintaining steady eye contact with the camera, indicating active listening and engagement. | 65,986 | 66.079 |
| Maintaining steady eye contact with the speaker, indicating active listening and engagement. | 51,807 | 49.455 |
| Listener maintains steady eye contact with the speaker, indicating active listening and engagement. | 46,273 | 44.074 |
| Listener maintains steady eye contact with speaker, indicating active listening. | 38,964 | 4.598 |
| Listener maintains steady eye contact with speaker, indicating attentive listening. | 37,617 | 28.491 |
| Listener maintains steady eye contact with the camera, indicating active listening and engagement. | 16,804 | 16.498 |
| Briefly closing eyes, indicating attentiveness or processing information. | 16,657 | 0.845 |
| Eyes closing briefly, indicating attentiveness or processing information. | 16,639 | 0.963 |
| Minimal head movement, indicating focused attention on the speaker. | 16,378 | 15.544 |
| Listener maintains steady eye contact with the speaker, indicating active listening and engagement throughout the clip. | 16,306 | 14.826 |
| Closing eyes briefly, possibly indicating listening intently or processing information. | 16,257 | 0.991 |
| Blinking (eyes closing briefly). | 14,774 | 1.639 |
| Blinking (eyes closing briefly) | 14,302 | 1.352 |
| Blinking (eye closure) - a brief, natural blink indicating attention or processing. | 14,210 | 0.580 |
| The listener maintains steady eye contact with the speaker, indicating active listening and engagement throughout the clip. | 14,131 | 13.840 |
| Closing eyes briefly, possibly indicating listening or processing information. | 13,571 | 1.590 |
| Blinking eyes, indicating attentive listening. | 13,316 | 1.141 |
| Listener blinks slowly, a natural backchannel behavior indicating attentiveness. | 12,752 | 0.764 |
| Blinking (eye closure) - a brief, natural blink indicating attentive listening. | 12,623 | 0.496 |
| Listener maintains steady eye contact with speaker, indicating active listening and engagement throughout the clip. | 12,178 | 11.994 |
| Neutral facial expression with slightly parted lips, suggesting attentive listening without strong emotional reaction. | 10,762 | 10.153 |
| The listener maintains steady eye contact with the speaker, indicating active listening and engagement. | 10,717 | 10.744 |
| Brief eye closure, possibly indicating thought, processing, or mild disengagement. | 10,302 | 0.481 |

## 复现

```bash
python scripts/report_open_backchannel_times.py \
  --postprocess experiments/p2_full_audio_postprocess_crop384_20261026 \
  --fusion experiments/listen_full_fusion_crop384_20261025 \
  --baseline-report experiments/p2_full_backchannel_report_20261027 \
  --output experiments/p2_full_open_description_times_20261027 --workers 16
```

仅 CPU 统计，不运行模型，不生成图。输入 SHA、逐分片输出 SHA 和集合核对保存在 manifest / shards / COMPLETE。
