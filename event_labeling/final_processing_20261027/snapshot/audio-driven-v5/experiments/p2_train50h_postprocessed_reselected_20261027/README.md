# 静音后重新筛选的 50h 训练子集

已完成并通过验收：**29,975 段，180,000.532312494 秒（50 小时零 0.532 秒）**，全部为独立划分中的 train。保留完整片段，平均 6.005 秒。未启动训练。

这次从已完成后处理的全量 722,706 段中重新选择，不是对旧的 30,415 段名单直接换音轨。评分使用静音后实际保留的七类弱标签；已删除的音频事件不计入丰富度，同一事件因静音被切成多个支持区间时只计一次。

## 数据入口

- [config.json](config.json)：新数据视图配置，`output` 已指向本目录。
- [manifests/observations.json](manifests/observations.json)：speaker 音频、listener 首帧、时长、真实 train 划分。
- [manifests/targets.json](manifests/targets.json)：listener 视频及原始 listener 音频引用。
- [integrated_labels.jsonl](integrated_labels.jsonl)：全量后处理后的开放式整合原文，逐条原样复制，未改写描述。
- `labels/canonical/<sample_id>.json`：对应七类训练弱标签，原样复制。
- `audio_postprocess/recipes/<sample_id>.json`：官方词／句／VAD 支持与中文过滤的虚拟静音配方。
- [samples.csv](samples.csv)：逐片段类别数、人脸尺寸、评分、入选原因及原始来源。
- [dataset.jsonl](dataset.jsonl)：逐条标签、配方和缓存校验值及位置。
- [summary.json](summary.json)、[selection_policy.json](selection_policy.json)、[split_audit.json](split_audit.json)、[validation.json](validation.json)、[COMPLETE.json](COMPLETE.json)：统计、筛选规则及验收。

`samples.csv` 的 `label_file/label_offset/label_sha256` 是继承的后处理前来源定位，供追溯；当前标签以本目录 `labels/canonical`、`integrated_labels.jsonl` 及 `current_canonical_sha256` 为准。`pre_postprocess_*` 列仅作前后对照，不参与本轮排序。

## 选择标准与结果

沿用“主体集＋稀有类别补充”。原始人脸短边 p10≥224 像素作为主体；人脸有效帧比例≥98%，原始及语义标注无记录错误，至少含一个非微笑事件。45h 主体之后优先补充摇头（最低 p10≥160），其他类别如不足 2,000 段可从 p10≥192 的候选补充，最后补足 50h。实际其他六类均已超过 2,000 段，无需降低门槛。

评分：`0.5×min(face_p10/384,1) + 0.3×类别数/7 + 0.2×min(封顶密度/12,1)`。每类最多计 3 个事件，微笑最多计 1 个；密度分母至少 5 秒。这些封顶只作用于排序，不删除或重写实际标签。

- 99.17%（29,726 段）人脸 p10≥224；249 段低于主体门槛，均来自摇头补充。
- 人脸 p10 中位数约 282 像素，p50 中位数约 296 像素，都是原始画面真实人脸尺寸，不是 512×512 裁框尺寸。
- 事件数中位数 9，非微笑事件中位数 4；这里是整合后模型候选事件数，不是人工确认的动作次数。
- 同一源 pair 最多 8 段；源时间区间不重叠，即使 listener 方向相反也检查重叠。
- 共 9,527 个 pair、313 个连通组；与现有 dev/test 的会话、参与者及连通组无交集。

| 类别 | 含该类的片段数 |
|---|---:|
| 点头 | 14,048 |
| 摇头 | 352 |
| 微笑 | 23,660 |
| 大笑／笑声 | 6,909 |
| 皱眉 | 6,767 |
| 惊讶 | 12,127 |
| 音频回应／发声 | 15,059 |

一段可以包含多类。没有发声标签的片段仍可因视觉事件入选；实际音频是否静音由配方决定，不用“有无模型标签”替代官方支持区间。

## 读取与缓存

上游为 [全量音频后处理](../p2_full_audio_postprocess_crop384_20261026/COMPLETE.json)。本目录复用已完成的新音频 latent、事件缓存和原视频缓存；没有新编码，也没有保存处理后的 WAV。原始 speaker 音频仍然保留。

训练使用本目录 `config.json`，通过现有 `listen_audio_postprocess.overlay.load_training_sample` 读取，自动组合原视频与新音频 latent。`targets.json` 中的 `target_audio` 是配方绑定的原文件，**不要绕过配方直接用它作为清洗后的音频**。需要波形时调用 `read_listener_audio(recipe)`。

`cache/input`、`cache/text`、`cache/audio_postprocess`、`cache/events` 为上游目录链接；原视频 VAE 通过 `data.audio_postprocess.base_output` 读取。不要删除上游缓存。复制的上游 `policy_manifest` 描述全量处理政策，具体子集成员只由本目录两个 manifests 确定。

当前配置 `feature_loss_enabled=false`。本数据视图可沿用现有 AV／Event 读取；它不表示已准备静音后音频的 Feature teacher，不能直接打开 Feature loss。

## 验证与复现

逐条验证了全部 29,975 段的标签、配方、缓存合同以及媒体存在性；抽取 34 段覆盖七类，实际走 CPU 训练读取并验证缓存载荷。大规模载荷完整性继承上游全量验收。源区间无重叠，train/dev/test 身份交集为 0。筛选及片段计数相关 7 项测试通过。

在项目根目录执行（新建输出目录，不能覆盖已完成版本）：

```bash
python scripts/select_postprocessed_listen_subset.py \
  --full experiments/p2_full_audio_postprocess_crop384_20261026 \
  --stats experiments/listen_backchannel_face_stats_20261026 \
  --catalog data_corpora/seamless_listen_local_full/catalog.json \
  --output experiments/p2_train50h_postprocessed_reselected_NEW \
  --workers 32

/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/audit_postprocessed_listen_subset.py \
  experiments/p2_train50h_postprocessed_reselected_NEW --workers 24
```

第一条仅写 `export_complete.json`，第二条全部通过才写最终 `COMPLETE.json`。`--finish-export` 可在完整导出 `dataset.jsonl` 后续做开放式标签导出；仍必须执行最终审计。实现见 [postprocessed_subset.py](../../src/listen_benchmark/postprocessed_subset.py)。

独立 test 24 条的准备目录是 [p2_test24_backchannel_face_20261027](../p2_test24_backchannel_face_20261027/)，它不属于这 50h，也不会从本训练集中随机切出。
