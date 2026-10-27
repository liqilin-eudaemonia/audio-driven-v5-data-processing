# 300h 全部原文描述训练子集

**已完成并验收：159,056 段，300 小时零 2.934 秒，全部为 train。** 保留完整片段。[COMPLETE.json](COMPLETE.json)、[validation.json](validation.json) 已生成，未启动训练。

按用户最新要求，直接保留原始打标信息，沿用已有代码去重整合。**不新增语义归类，不改写原句，不把七类之外的视觉观察排除。** 音频使用已完成的静音后处理结果，已被排除的音频标注不会重新恢复。标签仍是模型候选，不提升为人工真值。

## 数据入口

- [integrated_labels.jsonl](integrated_labels.jsonl)：本版主要标签，`listen-open-all-verbatim-v1`，每段包含全部整合后原文事件。
- [open_label_index.sqlite](open_label_index.sqlite)：按 `sample_id` 定位原文与校验值。
- [manifests/observations.json](manifests/observations.json)、[manifests/targets.json](manifests/targets.json)：speaker 音频、listener 视频／原始音频、裁框和时长。
- [samples.csv](samples.csv)、[selection.json](selection.json)：原始人脸尺寸、原文丰富度、分数及入选原因。
- [dataset.jsonl](dataset.jsonl)：媒体、配方、缓存和原文校验值；[dataset_contract.json](dataset_contract.json)：明确的开放标签读取合同。
- [media_cache_config.json](media_cache_config.json)：媒体缓存配置，不能当作现有七类监督头的开放标签配置。
- [summary.json](summary.json)、[selection_policy.json](selection_policy.json)、[split_audit.json](split_audit.json)：统计、规则与隔离检查。

## 保留与去重规则

视觉直接读取原始 `evidence.original_events`，包括已有 `out_of_scope/needs_review` 和没有七类归属的观察。调用原 `integration_pilot.key/clusters/make_event` 去重，沿用同内容、跨窗口、边界接近且重叠的条件；同一窗口内重复发生保留。代表区间来自某条真实来源记录，不扩成连续长区间。

每个事件保留 `description`、`original_descriptions`、`source_observation_ids`；来源文件、偏移、长度及SHA在 `source_evidence` 中，可查全部原始时间和文本。已有 `categories` 是来源元数据，不要求每条都归入七类。`response_groups` 只表示可能的时间共现，不表示持续动作。

音频事件逐条等于全量后处理的 `records/<sample_id>.json.audio_events`。原 `intervals` 保留，继续使用官方词／句／VAD并集及中文过滤的虚拟静音配方。

## 筛选结果

与 [七类版](../p2_train300h_seven_20261027/README.md) 独立筛选。原始人脸短边p10≥160，有效帧≥98%，每pair最多24段，源时间不重叠，与dev/test保持身份隔离。先取270h主体、补充已有稀有类别，再补足300h。

人脸、封顶原文描述多样性、封顶事件密度分别占0.5/0.3/0.2；已有仅微笑描述整体计分封顶1次，其余每种描述封顶3次。文字规范化只用于排名，保存的原文不变。文字多样性不等于语义类别数量。

- 20,415个pair，438个连通组；人脸p10中位数225.945像素，最小160.049像素。
- 4,589,610条整合候选，其中3,305,458条视觉候选包含原先七类之外／待判断的观察。
- 44,825段没有原七类canonical事件，但保留其他原始视觉观察；不能将它们误读成“无标签”。
- 与七类版共有95,174段。原文版仍包含全部已知七类：点头49,686段、摇头353段、微笑72,934段、大笑／笑声11,525段、皱眉18,335段、惊讶33,387段、音频回应35,882段；这只是已有七类的对照，不是开放标签总类数。

本版保证入选片段的原始视觉观察全部保留，不声称这300h涵盖全量每一种语义类别。新语义类型清单已按用户纠正取消，试跑输出未用于筛选或标签。

`summary.json` 的 `per_class`、`seven_class_reference_events` 及 `quantiles` 中事件／类别字段是旧七类对照；本版实际标签数量以 `open_events` 和 `samples.csv` 的 `open_*` 字段为准。原文层 `training_admitted=false` 表示未经人工确认、未自动变成固定类别监督真值；本子集成员以两个manifest为准，读取器仍返回这些完整候选。

## 读取方式

仅取原文标签：

```python
from listen_benchmark.open_subset_reader import load_open_label
label = load_open_label(dataset_root, sample_id)
events = label["events"]
```

读媒体缓存和全部原文：

```python
from listen_benchmark.open_subset_reader import load_open_training_sample
x, layout, _, av_target, label = load_open_training_sample(
    dataset_root, observation, target, device="cpu"
)
```

最后一个返回值是原文事件字典，**不是固定七类事件张量**。现有七类监督头尚未改造，本数据准备任务只交付开放标签和读取接口；后续训练代码需明确消费原文及区间。`labels/canonical` 和 `seven_class_reference_cache` 仅用于既有缓存身份验证／对照，不作为本版开放标签输出。

媒体复用原视频及已处理音频latent，没有新WAV或重新编码。需要清洗后波形时使用配方读取器，不能直接播放 `targets.json` 的原始音轨冒充处理结果。保留上游缓存和原始来源文件。当前Feature loss保持关闭。

## 代码与验收

[完整筛选约定](../../related/P2_双300h训练子集筛选方案.md)、[原文整合](../../src/listen_benchmark/open_verbatim_subset.py)、[导出入口](../../scripts/select_open_verbatim_subset.py)、[读取器](../../src/listen_benchmark/open_subset_reader.py)、[独立验收](../../scripts/audit_open_verbatim_subset.py)。

验收逐条核对每条原始视觉观察都有且只有一个去重去向、原句和时间来自真实来源、合并符合旧代码条件；音频事件与后处理结果完全一致。同时检查来源校验值、媒体缓存合同、质量门槛、源时间不重叠和身份隔离，并抽样实际运行CPU开放标签读取。

159,056段逐条检查全部通过：4,597,547条原始视觉观察按旧规则整合为4,524,472条视觉候选，65,138条后处理音频事件保持不变。39条实际CPU开放读取通过（含仅有七类之外观察的片段），源区间重叠及跨划分身份交集均为0。筛选／原文整合相关10项回归测试通过。

在项目根目录执行，必须使用新输出目录：

```bash
python scripts/prepare_open_verbatim_subset.py \
  --fusion experiments/listen_full_fusion_crop384_20261025 \
  --full experiments/p2_full_audio_postprocess_crop384_20261026 \
  --stats experiments/listen_backchannel_face_stats_20261026 \
  --report experiments/p2_full_backchannel_report_20261027 \
  --output experiments/p2_train300h_open_NEW --workers 32

python scripts/select_open_verbatim_subset.py \
  --full experiments/p2_full_audio_postprocess_crop384_20261026 \
  --stats experiments/listen_backchannel_face_stats_20261026 \
  --report experiments/p2_full_backchannel_report_20261027 \
  --catalog data_corpora/seamless_listen_local_full/catalog.json \
  --output experiments/p2_train300h_open_NEW --hours 300 --workers 48

/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/audit_open_verbatim_subset.py \
  experiments/p2_train300h_open_NEW --workers 32
```

`prepared/` 是候选中间产物，训练成员由本目录两个manifest确定。Python读取示例需让项目 `src` 位于 `PYTHONPATH`。
