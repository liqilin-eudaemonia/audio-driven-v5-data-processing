# 已有打标结果原文整合（当前方案，2026-10-25）

本目录独立备份当前CPU整合流程：保留模型原文，减少跨窗重复，选择已有时间，保留来源分歧。删除“嘿”改为笑声、Ed替换为Yeah和Says模板等内容改写。人工反馈只用于检查通用规则，不参与填写输出。

- [完整规则、代码导航与限制](RULES.md)
- [整合代码](src/listen_benchmark/existing_result_fusion.py)、[构建入口](scripts/build_listen_existing_fusion.py)、[HTML模板](scripts/listen_existing_fusion.html)
- [175条候选结果](results/labels.candidate.jsonl)、[完整对照与来源](results/review.json)、[统计](results/summary.json)
- [682条原始输入观测](inputs/semantic_observations/observations.jsonl)、[模型来源输入](inputs/semantic_observations/review.json)、[307份ASR输入](inputs/word_alignment)
- [复制哈希](BACKUP_MANIFEST.json)、[独立重放验证](REPLAY_VALIDATION.json)、[原机浏览器及结果检查](results/validation.json)、[原项目总索引快照](SOURCE_PROJECT_README.md)

50段得到175条候选（134视觉＋41音频）。输出候选都有有效区间，另2个词无法定位单列；15段无候选不能作为负例。七类字段和候选范围沿用上游，不在此重新做类别硬映射。当前未应用到全量，未准入训练，±0.2秒是目标与匹配容差，并非实测精度保证。

## 克隆后用CPU重放

Python 3.10+标准库即可完成以下命令，不依赖原项目路径、GPU、模型或媒体。先进入仓库根目录，再运行：

```bash
cd event_labeling/integration_verbatim
replay_dir=$(mktemp -d /tmp/listen-verbatim-replay.XXXXXX)
python scripts/build_listen_existing_fusion.py \
  --source inputs/semantic_observations \
  --align inputs/word_alignment \
  --out "$replay_dir/results" \
  --preview "$replay_dir/data_previews/listen_benchmark_review/existing_fusion_verbatim"
python -m unittest discover -s tests -p test_listen_existing_fusion.py
cmp results/labels.candidate.jsonl "$replay_dir/results/labels.candidate.jsonl"
cmp results/summary.json "$replay_dir/results/summary.json"
cmp inputs/semantic_observations/observations.jsonl "$replay_dir/results/observations.jsonl"
python -m http.server 8877 --bind 127.0.0.1 --directory "$replay_dir/data_previews"
```

浏览 http://127.0.0.1:8877/listen_benchmark_review/existing_fusion_verbatim/ 。文本、来源和下载可直接查看。音视频位于`data_previews/listen_benchmark_review/media/<sample_id>/{video.mp4,audio.wav}`，未打包；播放时须从源项目复制或链接media目录，并使用支持HTTP Range的服务器。可选`--font <字体文件>`需要fontTools；默认使用系统字体。浏览器自动检查另需Playwright和Chromium。

`review.json`的run_id取输出目录名，所以整页JSON比较需排除此字段；其余数据应一致。`manifest.json`包含本次环境的绝对路径和生成时间，不能逐字节比较。归档results/manifest.json记录原机产出，不是克隆后的执行路径配置。REPLAY_VALIDATION.json记录从本备份独立构建的验证结果。

本目录保留当前算法所需的纯Python依赖。其中integration_pilot.py含历史兼容功能，但当前fuse只以correction=None调用，不读取人工裁定。上层event_labeling目录保留早期开放事件整合基线，当前入口以本目录为准。
