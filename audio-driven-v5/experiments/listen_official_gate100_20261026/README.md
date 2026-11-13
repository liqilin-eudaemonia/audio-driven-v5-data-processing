# 固定100条：官方 listener 区间并集静音试听

2026-10-26。使用上一轮已经抽出的同一批100条，顺序不变。只生成处理副本，未替换训练数据、原始标签或缓存。

[规则文档](../../related/P2_官方音频时间并集静音规则.md) · [Cursor 转发8876试听页](http://127.0.0.1:8876/listen_benchmark_review/audio_official_gate100/) · [逐样本统计](summary.csv)

## 实际结果

| 项目 | 数量 |
| --- | ---: |
| 片段 | 100 |
| 总时长 | 592.2655秒 |
| 官方词、句、VAD在片段内均为空，输出全静音 | 52条 |
| 有保留区间 | 48条 |
| 官方三类时间并集 | 44.1965秒 |
| 加保护边界后的保留区间 | 54.6915秒 |
| 原有音频事件 | 158条 |
| 完整保留支持 | 39条 |
| 部分保留，记录实际保留子区间 | 27条 |
| 完全静音，从派生候选列表移出 | 92条 |

保留区间时长含保护边界、淡变和句内停顿，不等于实际发声时长。“完整保留支持”只说明事件时间没有被静音，不代表事件描述已验证正确。52条全静音是按已确认规则产生的结果，不是音频加载失败。视觉事件全部保留。

## 处理方式与复核

只用官方 listener 的词、句、VAD时间：先与片段相交，再前后各扩展0.1秒，合并，内部边界加10毫秒淡入淡出，其余波形置零。原长度、采样率、通道与响度尺度不变。speaker参考轨道不参与门控。

页面默认播放处理后音频。切换“原音／处理后／被移除部分”可在同一时间位置对比。灰色波形是原音，青色是当前版本，使用共同尺度。可筛选全静音、部分受影响事件、无VAD但词/句保留的样本。反馈填写“处理后杂音”和“listener回应”，可导出JSON。

该方法只移除区间外声音；保留区间内的串音不会自动消除。“被移除部分”也不等于已确认的噪声。试听结果尚未由本轮用户反馈确认。

## 文件与复现

- 核心实现：[`official_audio_gate.py`](../../src/listen_benchmark/official_audio_gate.py)。
- 批处理入口：[`apply_official_audio_gate100.py`](../../scripts/apply_official_audio_gate100.py)。
- 原始固定输入：[`comparison.json`](../listen_official_random100_20261026/comparison.json)。
- 全部结果与来源：[`results.json`](results.json)、[`policy.json`](policy.json)、[`summary.json`](summary.json)。
- 派生候选：`labels/<sample_id>.json`；原文、原始时间和排除记录均保留。部分保留事件附实际保留子区间，不能继续把原始整个区间当作连续发声监督。
- 媒体：`../../data_previews/listen_benchmark_review/audio_official_gate100/media/<sample_id>/processed.wav`、`removed.wav`、`gate.json`、`gated_candidates.json`。原音与视频引用上一轮页面的原始文件。

在项目根目录运行：

```bash
python scripts/apply_official_audio_gate100.py
python -m unittest discover -s tests -p test_listen_official_audio_gate.py -v
python experiments/listen_official_gate100_20261026/check_processed.py
```

9项边界和标签单元测试通过；100条原音哈希、时长、精确门控、残差、官方核心区间采样点不变及原文/视觉事件保留检查通过，见[`full_numeric_check.json`](full_numeric_check.json)。浏览器检查见[`browser_check.json`](browser_check.json)。

处理副本尚未接入训练。后续如应用到训练，目标音频相关缓存需按新音频身份重建，边界监督需读取实际保留子区间。
