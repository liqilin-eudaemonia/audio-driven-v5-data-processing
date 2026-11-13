# 固定100条：多字中文ASR筛选结果

[规则](../../related/P2_多字中文ASR移除与静音规则.md) · [试听页](http://127.0.0.1:8876/listen_benchmark_review/audio_official_chinese100/) · [统计](summary.json) · [逐片段CSV](summary.csv)

按用户确认，保留单字中文ASR，移出多字中文并静音对应时间。第23条“南京地铁”、第58条“我唔记得咗”、第77条“不能”共3个事件已移出；9个单字事件完整保留。剩余63个音频事件，分为38条有发声标注、62条没有发声标注。

音轨切换：**处理后**为本轮结果，**筛选前**为仅官方区间门控的上一版，**原音**为原listener，**被移除部分**为原音减本轮处理后音频。移除部分并不等于已确认噪声。

媒体仅新写3条：`data_previews/listen_benchmark_review/audio_official_chinese100/media/<sample_id>/`。其他97条引用此前文件。原始文件、上一版标签、全量训练清单和缓存没有修改。

- `results.json`：全部100条结果与展示信息。
- `labels/<sample_id>.json`：本轮候选标签、语言筛选策略和排除记录。
- `verification.json`、`full_numeric_check.json`：来源哈希、时长、精确置零、非处理区采样一致、单字事件和视觉保留验证。
- `browser_check.json`：分组、100条展示、3条排除记录、单字标注、音轨切换与布局检查。

复现（项目根目录）：

```bash
python scripts/apply_chinese_asr_filter100.py
python -m unittest discover -s tests -p test_listen_chinese_asr_filter.py -v
python experiments/listen_official_chinese100_20261026/check_results.py
```

原来的 `/audio_official_aligned100/` 入口将跳转至本轮页面，方便直接刷新；旧页面备份为同目录 `index_before_chinese_filter.html`，其旧JSON及旧标签保留不变。
