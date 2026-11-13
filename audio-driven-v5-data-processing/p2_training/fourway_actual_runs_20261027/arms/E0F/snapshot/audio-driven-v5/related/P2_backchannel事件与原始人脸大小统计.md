# P2：逐片段 backchannel 事件与原始人脸大小统计

用户已选择「先做逐片段统计与筛选」。目的是从已经完成整合的 722,706 段中，筛出 backchannel 候选更多、原始画面中 listener 人脸更大的训练片段。本工具只读取既有整合结果与人脸测量，不重新打标、不改写描述、不合并事件、不自动确定训练集或授予标签人工确认状态。

## 入口与产物

- [命令入口](../scripts/listen_backchannel_face_stats.py)
- [实现](../src/listen_benchmark/backchannel_face_stats.py)
- [回归测试](../tests/test_listen_backchannel_face_stats.py)
- [本次全量统计结果、门槛对照与实际限制](../experiments/listen_backchannel_face_stats_20261026/README.md)
- [全量逐片段 CSV](../experiments/listen_backchannel_face_stats_20261026/samples.csv)：一段一行，适合分析；体积较大，建议用 SQLite 查询。
- [SQLite](../experiments/listen_backchannel_face_stats_20261026/stats.sqlite)：`samples` 表，`sample_id` 主键。
- [汇总](../experiments/listen_backchannel_face_stats_20261026/summary.json)：事件、人脸分布、数据问题计数。
- [门槛对照表](../experiments/listen_backchannel_face_stats_20261026/threshold_grid.csv)：不同人脸短边、事件数门槛对应的段数、时长、事件数。
- [输入身份记录](../experiments/listen_backchannel_face_stats_20261026/provenance.json)、[完成与输出校验](../experiments/listen_backchannel_face_stats_20261026/COMPLETE.json)。

## 统计口径

### 事件

`event_count` 是整合结果 `events` 中的候选条数，同一 `event_id` 只允许出现一次。一个事件有多个类别时，总数仍只加一；各类别列可同时加一，因此类别之和可能超过总数。`response_groups`、原始观察条目和支持证据不再计为额外事件。没有二次关键词映射，也没有新的事件去重操作。

保留原有七类计数：`head_nod_count`、`head_shake_count`、`smile_count`、`laugh_count`、`frown_count`、`surprise_count`、`listener_vocalization_count`。`category_counts_json` 同时保留所有实际出现的类别，即使未来类别不在这七个名字中。`non_smile_event_count` 表示至少有一个非 smile 类别的候选条数，方便分析结果是否主要由微笑描述构成；它不更改任何标签。

- `visual_event_count` / `audio_event_count`：分别按原始 `modality` 计数。
- `events_per_10s`：候选总数 / 片段秒数 × 10；便于比较长短片段，但不代表人工确认的动作频率。
- `valid_timed_event_count`、`timed_events_per_10s`：仅计时间有效的候选，要求有限数值、`0 ≤ start < end ≤ duration`，只容忍 1e-6 秒浮点误差。
- `event_coverage_s` / `event_coverage_fraction`：有效区间的并集，视听重叠不重复累加；另有各模态并集时长。不会把时间并集重新写回标签。
- `invalid_timed_event_count`：越界或无效区间单列，不截成“正确标签”。

这些都是模型候选统计，不保证一条候选就等于一次真实发生的 backchannel。持续微笑的多条描述等仍可能影响事件数；筛选时应结合类别、密度和覆盖时长。零条候选不等于已确认负例。

### 原始人脸大小

读取原始 manifest 的 `face_geometry.listener`，测量范围是**整个片段中检测有效的帧**，没有重新读取视频/逐帧关键点，也不是“某条事件发生期间”的测量。

已有测量基于 listener 的 facial landmarks 23:91，关键点置信度高于 0.4，每帧至少 45 个有效点，以横纵坐标第 3 / 97 百分位计算宽高。尺寸是原始编码画面的像素范围，不是包含头发的整头框，也不是缩放后的 512×512 尺寸。

| 字段 | 含义 |
| --- | --- |
| `face_short_side_p10` | 各有效帧人脸短边的第 10 百分位，单位原始像素；近似反映偏小情况下仍有多大 |
| `face_short_side_p50` | 各有效帧人脸短边中位数，单位原始像素 |
| `face_width_p10/p50`、`face_height_p10/p50` | 原始人脸宽、高的分位数 |
| `face_area_fraction_p10/p50` | 人脸面积占原始完整画面的比例 |
| `face_area_px2_p10/p50` | 上述面积分位数 × 原始画面面积；不使用独立宽高分位数的乘积冒充面积分位数 |
| `face_valid_fraction`、`face_valid_frames`、`face_expected_frames` | 检测有效程度；p10/p50 只统计有效帧，必须同时关注此项 |
| `crop_width/height`、`crop_x/y` | 当时模型输入所采用的整数预裁框，含脸外区域 |
| `crop_proposal_size` | 原始浮点 proposal 尺寸，可能因取整与实际裁框差少量像素 |
| `native_width/height` | 人脸测量对应的原始画面分辨率 |

原先“预裁框 ≥384×384”不能推出“人脸 ≥384×384”。例如样本 `naturalistic_V00_S0886_I00000514_local_b32ff176d988` 的实际预裁框为 396×396，人脸短边 p10 约 110.45、p50 约 125.62 像素。缺失或非 measured 的人脸测量输出空值，绝不拿裁框尺寸替代。

### 标注问题与身份

`raw_issue_count`、`semantic_issue_count` 和 `unresolved_observation_count` 分别保留。`label_issue_flag` 在原始问题、语义问题或无效时间区间任一存在时为 1。未归属观察也可能只是 out-of-scope，因此不会仅因存在 unresolved 项就排除片段。

`--exclude-label-issues` 是按这些已记录技术问题排除，不是“标注一定正确”的保证。

保留 `pair`、`group_id`、`listener_id`、`split`、`admission_pool`、源时间、manifest 行号与 SHA256，以及整合标签文件、字节偏移、长度和单行 SHA256。`show` 根据这些引用返回**完整原始整合记录**和完整人脸几何统计，事件原文与支持证据不受统计过程影响。

## 重建统计

在项目根目录执行：

```bash
python scripts/listen_backchannel_face_stats.py build \
  --fusion-run experiments/listen_full_fusion_crop384_20261025 \
  --feature-run experiments/p2_features_full_crop384_20261025 \
  --output experiments/listen_backchannel_face_stats_20261026 \
  --workers 16
```

已存在的输出不会默认覆盖。同输入、同代码中断恢复加 `--resume`，已完成分片会校验输出 SHA256 后复用；完整结果恢复会核验产物 SHA256。换输入或代码时使用新输出目录。

构建会检查整合与特征完成标记、索引 SHA256、源行索引数据库 SHA256；每段核对源 manifest 行 SHA256、样本与人物身份、listener 视频与音频、源时间跨度。每个整合分片核对文件 SHA256、样本数与事件数；最终检查样本主键唯一、总样本数和 SQLite 完整性。只读取现有源文件，不改动其内容。

## 如何筛选

建议先用 `face_short_side_p10` 设置人脸门槛，并同时要求较高的 `face_valid_fraction`，再按事件数或密度排序。p50 可作为补充：p50 只要求典型帧的人脸大，p10 更严格地约束偏小帧。此处不预先指定最终训练阈值。

`threshold_grid.csv` 对照 p10 / p50、短边 64–384 像素、事件数至少 0/1/2/3/5/8/10，分别列出保留与排除已记录标注问题的结果。该表的 `face_valid_fraction` 下限统一为 0.9。时长为片段时长相加，重叠源时间没有去重，不应称为唯一视频小时数。

下面仅是可复现的筛选命令示例，**不是已经决定的训练集门槛**：

```bash
python scripts/listen_backchannel_face_stats.py select \
  --stats experiments/listen_backchannel_face_stats_20261026 \
  --output experiments/listen_selection_example_face128_event3 \
  --min-face-p10 128 --min-face-valid 0.9 \
  --min-events 3 --exclude-label-issues --split train \
  --sort density
```

还支持 `--min-face-p50`、`--min-density`（每 10 秒）、`--sort events|density|face`、`--limit`。这些参数默认为不加对应限制；相同分数按人脸 p10、事件数、sample_id 确定性排序。

输出 `samples.csv`、`sample_ids.txt` 和 `selection.json`；后者保存确切门槛、SQL、统计完成标记 SHA256、样本/事件/时长和既有特征目录，供训练代码按 sample_id 连接缓存。不会复制大体积媒体或缓存，不移动或删除未选数据，也不改动已有 split。导出的 `training_admitted=false` 说明本工具没有把弱标签自动提升为人工确认标签。

查看某段完整事件原文和人脸信息：

```bash
python scripts/listen_backchannel_face_stats.py show \
  --stats experiments/listen_backchannel_face_stats_20261026 \
  --sample-id naturalistic_V00_S0886_I00000514_local_b32ff176d988
```

## 验证

```bash
python -m pytest -q tests/test_listen_backchannel_face_stats.py
```

涵盖事件/类别/回应组计数、重叠区间并集、原文不变、真实人脸与裁框区别、面积分位数、缺失人脸、无效时间、重复 event_id、源身份与哈希、完整构建/筛选/回查/恢复、阈值边界及损坏分片不能写完成标记。
