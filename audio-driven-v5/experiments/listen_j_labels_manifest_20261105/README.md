> 后续转正与重复标签修复已完成并通过独立验收。请使用[新的全量标签统一入口](../listen_j_labels_upright_20261105/README.md)，共1,107,313条。本目录保留父版本追溯。

# J 全量新标签统一入口（2026-11-05）

后续行为统计、数据筛选与新缓存准备统一从 [labels.jsonl](labels.jsonl) 读取本轮新标签。每行一条引用记录，含完整 caption、原始标签文件及 SHA256、原 split、来源时间、人物、最终裁框、音频净化配方和原生成复核标记。

来源为 `experiments/listen_j_reaction_vocal_omni_20261103/`；固定标签身份为 `e0fa13b3987ff1585c53a6a6074982d5d705fb7db2cfa44fec715f975c1eab39`，实际原始标签位于该目录的 `captions/e0fa13b3987ff158/` 下。

[本轮人工抽检：通过组24条／复核组36条](../listen_j_caption_quality_review_20261105/README.md) 已提供，等待用户检查；分层抽样、媒体和反馈入口见该页。

## 覆盖范围

| 项目 | 数量 |
|---|---:|
| 原定目标样本 | 1,107,323 |
| 已导出新 caption | 1,107,322 |
| 无新 caption、单列例外 | 1 |
| 重复 ID、来源元数据不一致 | 0 |

| 原 split | 新标签 | 单列例外 |
|---|---:|---:|
| train | 1,086,216 | 1 |
| dev | 8,822 | 0 |
| test | 12,284 | 0 |

唯一例外为 `naturalistic_V03_S1672_I00000219_local_9172c091cbd4`，train、9.725 秒。新版反应／点头证据检测返回 `no_detected_face`，上游将其置为 held，未生成新 caption。诊断文件及哈希见 [exceptions.jsonl](exceptions.jsonl)。该条未回填旧 caption。

独立覆盖核对已通过：全部 1,107,323 个目标逐行对齐，重复／遗漏／多余 ID、来源元数据错配和 group 跨 split 均为 0；全部导出 JSONL 的完整哈希通过，686 条原始标签分层回读及唯一例外诊断核对通过。

`source_coverage_complete=true` 表示全部目标 ID 均已归账；`label_coverage_complete=false` 明确记录其中 1 条没有新标签。

## 文件

| 文件 | 用途 |
|---|---|
| [labels.jsonl](labels.jsonl) | **后续处理的数据入口**；1,107,322 条新标签，逐行读取，约 8.12 GB |
| [coverage.jsonl](coverage.jsonl) | 全部 1,107,323 个目标的覆盖记录，包含唯一 held 例外 |
| [exceptions.jsonl](exceptions.jsonl) | 无新标签的样本、原因和诊断来源 |
| [summary.json](summary.json) | 版本、数量、split、复核原因、源文件和导出文件 SHA256 |
| [EXPORT_COMPLETE.json](EXPORT_COMPLETE.json) | 导出完成标记，同时保留完整覆盖和标签缺项状态 |
| [COVERAGE_AUDIT.json](COVERAGE_AUDIT.json) | 独立全量覆盖核对；正式使用前确认 `passed=true` |
| [verification.json](verification.json) | 真实小批量、错误注入及无脸例外分流检查记录 |

## 标签与统计口径

- 正文与本轮原始标签中的 `caption` 完全一致；每份原始 JSON 已计算 SHA256，并校验当前代码身份、检测证据身份、来源、split、起止、裁框和音频归属。
- 977,240 条原生成检查通过、130,082 条带原生成复核标记，全部保留在 `labels.jsonl`。本次只导出与核对覆盖，未执行新的语义验收或基于复核标记的删除。
- `training_admitted=false`、`human_ground_truth=false` 保留来源性质。本清单是新标签池，后续实际训练／测试选集另行生成。
- 人脸 p10/p50 和画质证据复用既有物理测量。每条 `behavior_statistics_status=not_recomputed_for_this_caption_version`；不导出旧 caption 的行为类别、次数或旧 `statistics_file` 引用。后续行为统计应读取这里的新正文。
- 原始视频、音频与标注文件由引用定位；没有复制媒体。完整原始生成尝试、检测与 ASR 证据可从每条 `caption_file` 读取。

## 最小读取示例

```python
import json
from pathlib import Path

data = Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/listen_j_labels_manifest_20261105')
assert json.loads((data / 'COVERAGE_AUDIT.json').read_text())['passed']
with (data / 'labels.jsonl').open() as stream:
    for line in stream:
        row = json.loads(line)
        # row['sample_id'], row['split'], row['caption'], row['caption_file']
```

## 复现与验收

[导出脚本](../../scripts/export_j_caption_manifest.py)按固定保留清单流式分组读取，并逐条验证当前身份的新标签；使用新输出目录重跑。全部目标由新标签和例外两路互斥归账。

```bash
python scripts/export_j_caption_manifest.py \
  --work experiments/listen_j_reaction_vocal_omni_20261103 \
  --output experiments/listen_j_labels_manifest_REPRO \
  --workers 24

python scripts/audit_j_caption_manifest.py \
  --data experiments/listen_j_labels_manifest_REPRO
```

[独立校验器](../../scripts/audit_j_caption_manifest.py)重读全部导出行，对照全部权威 ID 及数据库元数据，检查每份导出 JSONL 的完整哈希、split／人物分组、例外诊断，并分层加确定性抽样回读原始标签正文和哈希。实际核对数量与抽样数量见 `COVERAGE_AUDIT.json`。这些检查验证导出及来源一致性，媒体语义质量需要后续单独验收。
