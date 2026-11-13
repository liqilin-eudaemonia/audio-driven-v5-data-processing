# J 全量打标结果：最终画质复筛（2026-11-01）

已完成。此前1,119,665段表示有caption和行为统计，不等于全部画质通过。早期caption保留原样，但本次已重新关联全量60,869份来源画质回执，对1,151,805段逐条核对来源后生成独立筛选清单。

## 最终数据

| 处理结果 | 片段数 | 时长（小时） |
|---|---:|---:|
| 画质通过且有标注，最终保留 | 1,107,323 | 1,856.25 |
| 画质明确不通过，排除 | 22,295 | 35.55 |
| 画质测量不完整，单独待核、不进保留集 | 21,896 | 31.72 |
| 点头检测无人脸，按用户要求排除 | 291 | 0.35 |

四组互斥，总计1,151,805段、约1,923.88小时。无人脸优先按用户确认的剔除规则处理，其中231段同时画质exclude、50段画质review、10段画质retain，不重复计数。

本次从已有1,119,665段caption中额外排除6,237段（10.10小时），另将6,105段（8.81小时）移入待核。其余原先已分流的16,058段画质不通过、15,791段画质待核和291段无人脸保持不可入选。

| 原划分 | 最终保留片段数 | 时长（小时） |
|---|---:|---:|
| train | 1,086,217 | 1,820.42 |
| dev | 8,822 | 15.32 |
| test | 12,284 | 20.51 |

沿用原train/dev/test划分、人物和来源时间。没有重新执行50h/300h的人脸大小排序、丰富度或稀有类别补充筛选。

## 采用的规则

- 使用已完成的画质回执，冻结契约为343c387629f05961589339fa4ab020f2a3d87a5cc4c874ea4a52206b16cfe778；没有重解码视频或修改阈值。
- 每秒5帧测量原生人脸区域清晰度。低清晰度连续至少2秒，或累计占比至少25%，按现有OR规则排除；完全无人脸也排除。
- 测量支持不足等review单独留存，不自动判成好数据，也不永久删除。
- 点头no_detected_face全部排除，不因画质retain/review重新放行。
- 不以是否有动作、是否发声、是否有可计数行为为入选条件。44,745段采用类别数为0的片段仍保留；这个0不是经过人工确认的“无回应”。
- 1,072,881段文本统计通过严格校验；34,442段局部待核统计仍保留，不因字段校验问题再删数据。
- 原始caption、音频净化配方、点头证据、实际裁框来源与原统计内容均不改写。旧框标注按此前授权继续保留实际来源。

这是现有相对清晰度规则的复筛，不保证识别整段均匀模糊或全部其他画质缺陷。最终保留表示本轮画质条件通过，不自动覆盖原有training_admitted标志或其他训练准入要求。

## 文件与读取

- [retained.jsonl](retained.jsonl)：最终可供后续选取的清单。每行包含sample_id、原split、时长、caption_file、原统计文件及字节位置、完整画质决定。
- [selection.sqlite](selection.sqlite)：selection表含全部分流；samples和behaviors仅含最终保留集，可按人脸大小、行为丰富度等继续查询。samples.file/offset指向原统计JSONL。
- [behavior_catalog.csv](behavior_catalog.csv)：仅对最终保留集重新汇总类型、涉及片段数、可计次数、未知次数与持续状态组；4,764个有采用记录的开放候选名称含同义细分，不是4,764个互斥真值类别。covered_clip_seconds是含此类别的片段时长，不是动作精确时长。
- [quality_excluded.jsonl](quality_excluded.jsonl)、[quality_review.jsonl](quality_review.jsonl)、[no_face_excluded.jsonl](no_face_excluded.jsonl)：分流名单和原因。
- [summary.json](summary.json)：全部数量、时长、划分、原因和文件hash。
- [audit.json](audit.json)：独立验收；[complete.json](complete.json)：完整导出成功标记。

没有新增WAV、视频或latent。通过caption_file可读取原整段描述、实际视频裁框和净化音频配方；通过statistics_file和statistics_byte_offset读取原文与全部行为证据：

```python
import json
with open(row['statistics_file'], 'rb') as f:
    f.seek(row['statistics_byte_offset'])
    annotation = json.loads(f.readline())
assert annotation['sample_id'] == row['sample_id']
```

## 复现与校验

代码：[filter_listen_richness_visual_quality.py](../../scripts/filter_listen_richness_visual_quality.py)。只按索引读取画质回执，两个读取线程，无GPU任务。重跑应指定新的输出目录，不覆盖本结果。

```bash
python scripts/filter_listen_richness_visual_quality.py \
  --source-work experiments/listen_richness_full_j_20261031 \
  --output experiments/listen_j_final_quality_selection_REPRO
```

检查了60,869份画质回执契约、计数及完整ID覆盖；全部1,151,805段原来源hash和回执来源hash一致。独立复核五份导出清单的全部ID、划分、时长及SHA256；抽查111条原文、实际裁框来源和字节位置读取完全一致。SQLite完整性检查通过，无漏失的画质记录或缺标待核项。原始全量结果不删除。
