# 新标签人工抽检：自动通过组24条／复核组36条

- [检查自动通过组 P01–P24](http://localhost:8876/listen_benchmark_review/j_caption_quality_20261105/?group=pass)
- [检查复核组 R01–R36](http://localhost:8876/listen_benchmark_review/j_caption_quality_20261105/?group=review)
- [全部60条](http://localhost:8876/listen_benchmark_review/j_caption_quality_20261105/)

沿用此前 [J最终裁框素材复核页](../listen_j_face224_training_review_20261101/README.md) 的布局和精确float32双音轨同步播放器。使用本轮全量新caption，不使用旧caption或旧行为统计。访问时转发当前机器8876端口。

已收到用户反馈：“我觉得自动通过组的都挺好的。”已按 P01–P24 整组一般性认可保存到 [user_feedback.json](user_feedback.json)，仅对应这24条，未据此推断全量质量或逐事件真值。R组继续待用户检查。文本核对已确认 R31、R32、R33、R36 出现上百次句子循环并因长度上限结束，见 [文字检查记录](text_inspection_notes.json)；R34、R35 仍需结合实际动作判断重复是否合理。

后续用户反馈：打标基本无大问题，但指出 R25 整幅画面侧转90度。核查原帧和预览一致，48个既有roll测量的中位数为−94.62度；现行判定未按roll自动排除，只对已登记的来源绑定人工证据应用 `sideways_video`。R25的清晰度和检测支持均通过，实际进入旧500h有声补充及规划SFT清单。问题与范围见 [R25方向漏检核查](orientation_issue_R25.json)，新增 [来源绑定排除证据](orientation_reviewed_evidence.json) 仅作为下一版选集依据，尚未修改生产规则或冻结清单。

## 抽样

源清单为 [全量新标签统一入口](../listen_j_labels_manifest_20261105/README.md) 的 `labels.jsonl`，1,107,322条，SHA256为 `b7dc75f51c9f408ff779ad9b7c4f5e42c9eee3b3a2bfdc34eda501269547ddda`。

固定种子 `j-caption-quality-review-20261105-v1`，使用种子、分层键和sample_id的SHA256优先级抽样；每个pair最多一条，共60个不同pair。本轮抽train/dev，原split保持不变，test未用于校准。

| 组别 | 数量 | 抽样方式 |
|---|---:|---|
| 原自动通过组 | 24 | train 20＋dev 4；在各split内覆盖非零音频／全静音、人脸p10≥224／较小脸四格；train每格5条、dev每格1条 |
| 原带复核标记组 | 36 | 以下6个主类别各6条；前5类各含4条单一标记、2条多标记样本 |

| 审阅编号 | 抽样主类别 |
|---|---|
| R01–R06 | `invented_vocal_on_silence`：静音中发声提示 |
| R07–R12 | `coverage_uncertain`：动作次数／区间覆盖待核 |
| R13–R18 | `missing_reaction_cue`：其他反应线索遗漏提示 |
| R19–R24 | `missing_nod_cue`：点头遗漏提示 |
| R25–R30 | `missing_vocal_cue`：发声线索遗漏提示 |
| R31–R36 | `repeated_generation`／`generation_not_finished`：生成重复或未结束 |

抽样只读取split、原检查标记、音频是否全零及人脸大小等字段，没有按caption内容或观看效果挑换样本。一个样本可能同时具有多个标记，页面原因筛选会展示全部命中项。分层和少见异常被有意超额抽取；本批不能直接估计全量准确率或误报比例。

[固定抽样及分层容量](sampling.json) · [样本清单CSV](sample_list.csv)。

## 如何检查

1. 默认同步播放listener音视频，先核对新caption中的动作、发声、否定和时序。
2. 需要上下文时切换speaker或双方同时听；每次切换样本会恢复为仅listener。
3. 查看原检查原因、剩余线索及可点击的点头／发声／其他反应时间；检测和ASR是辅助证据，实际音视频是核对依据。
4. 选择“基本可用／存在实质标签问题／需要进一步复核”，并可单独记录原检查提示是否误报。意见自动保存在当前浏览器，点击“导出全部复核意见”保存JSON。

也可直接在对话中按编号反馈，例如“P03漏写了后半段的mhm”“R02是检查器误报”。尚未检查的样本保持未判断。

## 媒体与来源

- 标签正文及原始JSON逐字保留，绑定清单行和原始标签SHA256。
- 视频从原TAR读取，使用新caption实际采用的最终修正版裁框；512×512、25fps预览，记录原始PTS、编码前RGB哈希和同帧参考图。原打标视频输入为8fps。
- listener按新caption内官方词／句／VAD并集配方重放；float32波形哈希与本轮打标记录逐条一致。
- speaker来自配对参与者同一源时间区间的规范化音轨。浏览器同步播放器使用原float32波形；下载MP4/M4A为AAC试听副本。
- 点头、ASR和其他反应线索直接来自新标签的 `input_evidence`，没有再次运行检测或生成新caption。

页面数据位于 `data_previews/listen_benchmark_review/j_caption_quality_20261105/data.json`。原标签、配方、清单行及媒体来源均可从页面展开查看。

## 构建与验收

[构建来源验收](build_audit.json) · [真实浏览器与媒体验收](review_audit.json)。浏览器检查覆盖所有60条视频加载、120路PCM哈希、两组和各原因筛选、播放／音轨切换／拖动、意见保存导出及移动端布局。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/build_j_caption_quality_review.py --workers 2

/tmp/native-fps-review-qa/bin/python scripts/check_j_caption_quality_review.py
```

[构建脚本](../../scripts/build_j_caption_quality_review.py) · [复用模板](../../scripts/j_caption_quality_review.html) · [验收脚本](../../scripts/check_j_caption_quality_review.py)。固定抽样支持继续构建，已完成媒体按caption身份复用。
