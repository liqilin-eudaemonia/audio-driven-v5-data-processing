# 同100条：原文保留与官方时间对齐

[规则与实现条件](../../related/P2_官方时间对齐与原文保留规则.md) · [Cursor转发8876复核页](http://127.0.0.1:8876/listen_benchmark_review/audio_official_aligned100/)

沿用上次固定100条与已确认的处理后音轨，仅更新独立派生标签与展示。视频在左，音轨和标签在右；标签页可以查看剩余、移出、视觉与官方时间记录。词不一致时保留我们的原文。

| 结果 | 数量 |
| --- | ---: |
| 仍剩余发声标注的片段 | 41 |
| 没有发声标注的片段 | 59 |
| 其中无发声标注、但音频仍有保留区间 | 7 |
| 原音频事件 | 158 |
| 派生标签保留的音频事件 | 66 |
| 使用官方词时间 | 15 |
| 使用单次VAD时间 | 1 |
| 使用原模型时间与保留范围交集 | 50 |
| 移出的音频事件 | 92 |
| 保留事件中时间发生调整 | 40 |

例如第19条 `Yeah` 从1.040–1.760秒改成官方词的1.156–1.656秒；第30条 `Right` 从3.840–4.080秒改成3.881–4.141秒，补回原时间未覆盖的词尾。第55条我们的 `Um` 原文保留，官方 `Yeah.` 只作为对照。第20条两次 `Yeah` 仍分别为0.209–0.449和1.510–1.730秒。

“没有发声标注”不等于“没有声音”或“没有视觉反应”。其中7条有官方保留音频，但没有可保留的原模型音频事件；本轮没有根据官方记录补写新事件。

## 文件

- [`results.json`](results.json)：全部100条展示与审计记录。
- [`summary.csv`](summary.csv)、[`summary.json`](summary.json)：统计。
- `labels/<sample_id>.json`：新标签、原始区间、依据及排除记录。
- [`policy.json`](policy.json)：匹配阈值。
- [`verification.json`](verification.json)、[`full_numeric_check.json`](full_numeric_check.json)：全部100条原文、类别、视觉事件、来源哈希、处理后WAV哈希及有效区间校验。
- [`browser_check.json`](browser_check.json)：100条页面切换、分组、左右布局、音视频同步、区间试听、文本冲突、下载和反馈检查。

在项目根目录复现：

```bash
python scripts/build_official_aligned100.py
python -m unittest discover -s tests -p 'test_listen_official*.py' -v
python experiments/listen_official_aligned100_20261026/check_processed.py
```

测试覆盖官方词尾延伸、原文冲突保留、两次okay分别保留、多个事件竞争同一词、重复词歧义、非语言单次VAD、多事件共享VAD、不连续区间及视觉不变。波形未重新处理，仍引用上一版试听页的同一批文件。

尚未替换原始标签、训练清单或缓存。未来用于训练时，音频监督须读取`intervals`；原`start/end`已另存到`source_interval`。不连续的子区间不意味着新增事件次数。
