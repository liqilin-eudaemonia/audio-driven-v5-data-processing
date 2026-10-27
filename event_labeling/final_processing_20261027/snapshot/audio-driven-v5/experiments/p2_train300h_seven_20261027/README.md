# 300h 七类训练子集

**已完成并验收：176,147 段，300 小时零 3.631 秒，全部为 train。** 保留完整片段；平均 6.131 秒。[COMPLETE.json](COMPLETE.json)、[validation.json](validation.json) 已生成，未启动训练。

本版保留原六类视觉＋一类音频监督，按静音后实际保留的事件重新选取。与 [300h 原文版](../p2_train300h_open_20261027/README.md) 独立评分，两份名单可以不同。

## 数据与读取入口

- [config.json](config.json)：现有七类训练读取配置。
- [manifests/observations.json](manifests/observations.json)、[manifests/targets.json](manifests/targets.json)：speaker 音频、listener 首帧／视频、原始音轨与片段时长。
- [integrated_labels.jsonl](integrated_labels.jsonl)：上游已整合、已音频后处理的原文，逐条复制。
- `labels/canonical/<sample_id>.json`：七类弱标签；`audio_postprocess/recipes/<sample_id>.json`：虚拟静音配方。
- [samples.csv](samples.csv)、[selection.json](selection.json)：评分、入选原因、类别与原始人脸尺寸；[summary.json](summary.json)：汇总。
- [dataset.jsonl](dataset.jsonl)：标签、配方和缓存位置及校验值；[dataset_contract.json](dataset_contract.json)：数据合同。
- [selection_policy.json](selection_policy.json)、[split_audit.json](split_audit.json)：选择政策与隔离检查。

训练通过 `listen_audio_postprocess.overlay.load_training_sample` 读取本配置，组合原视频 latent 与已处理音频 latent；已有七类事件缓存可直接复用。没有新 WAV，也没有重新编码 latent。`targets.json` 的音频路径指向配方绑定的原始文件，不代表可绕过配方读取清洗后波形。需要波形时使用 `listen_audio_postprocess.recipes.read_listener_audio(recipe)`。

`cache/` 链接及配置中的 `data.audio_postprocess.base_output` 依赖上游 [全量后处理数据](../p2_full_audio_postprocess_crop384_20261026/README.md)，请保留源文件和缓存。配置保持 `feature_loss_enabled=false`；本轮不新建静音后音频的 Feature teacher。

## 筛选标准与统计

按用户确认，原始人脸短边 p10≥160 像素，有效人脸帧≥98%，每 pair 最多24段；源时间不重叠，保持 train/dev/test 会话、参与者和连通组隔离。先取270h主体，补稀有类别，再补足300h。质量标记有效，至少有一个非微笑候选事件。

沿用原评分：人脸0.5、七类丰富度0.3、封顶事件密度0.2；每类计分最多3次，微笑最多1次，密度分母至少5秒。封顶仅用于排名，不删改标签。[完整约定](../../related/P2_双300h训练子集筛选方案.md)。

- 27,712 个 pair，553 个连通组。
- 原始人脸短边 p10 中位数210.217像素，p50中位数222.574像素；最小p10为160.001像素。这里是真实人脸尺寸，不是512×512裁框尺寸。
- 共2,091,342条 canonical 候选事件；同一事件的静音分片不重复计数。这不是人工确认的动作次数。

| 类别 | 含该类片段数 |
|---|---:|
| 点头 | 79,457 |
| 摇头 | 356 |
| 微笑 | 109,829 |
| 大笑／笑声 | 28,011 |
| 皱眉 | 30,698 |
| 惊讶 | 49,605 |
| 音频回应／发声 | 72,112 |

一段可以有多类。与原文版共有95,174段；两版按各自丰富度独立筛选，不能把差异仅归因于标签方式。

`samples.csv` 中继承的 `label_file/label_offset/label_sha256` 是历史来源定位，当前标签以本目录整合文件、canonical 文件及 `current_canonical_sha256` 为准。`below_224_face_p10` 是旧224门槛的对照统计，当前批准门槛是160。

## 复现与验收

项目根目录执行；新输出目录不可覆盖本目录：

```bash
python scripts/select_postprocessed_listen_subset.py \
  --full experiments/p2_full_audio_postprocess_crop384_20261026 \
  --stats experiments/listen_backchannel_face_stats_20261026 \
  --catalog data_corpora/seamless_listen_local_full/catalog.json \
  --pool-source experiments/p2_train50h_postprocessed_reselected_20261027 \
  --hours 300 --core-face-floor 160 --max-per-pair 24 \
  --output experiments/p2_train300h_seven_NEW --workers 48

/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/audit_postprocessed_listen_subset.py \
  experiments/p2_train300h_seven_NEW --workers 40
```

选择器检查来源隔离；验收逐条检查标签、配方、计数和缓存合同，并抽取样本实际进行CPU训练读取。载荷完整性继承上游全量验收，再对实际读取样本复核。`export_complete.json` 只代表导出结束；最终 `COMPLETE.json` 才代表独立验收通过。

本版176,147条检查全部通过，34条实际CPU训练读取通过，源区间重叠及跨划分身份交集均为0。筛选／原文整合相关10项回归测试通过。
