# 当前AV500h选集：train＋dev来源范围（2026-11-05）

**已完成并独立验收。** 按用户明确确认，主集和有声补充都从原train＋dev统一筛选，test不加入；源split字段保留，没有把dev改名为train。复用已完成的全部1,107,313条冻结v3统计，没有重新打标、模型推理或媒体处理。

实际读取：[selection/selected.jsonl](selection/selected.jsonl)，配套[CSV](selection/sample_list.csv)、[SQLite字节索引](selection/index.sqlite)、[摘要](selection/SUMMARY.json)、[独立验收](selection/INDEPENDENT_FINAL_AUDIT.json)及[完成标记](DELIVERY_COMPLETE.json)。

| 原来源split | 段数 | 小时 |
|---|---:|---:|
| train | 281,060 | 494.17009744 |
| dev | 3,158 | 5.83116169 |
| 合计 | **284,218** | **500.00125913** |

| 组成 | 段数 | 小时 |
|---|---:|---:|
| p10≥224px主集 | 267,079 | 468.18205332 |
| 有声补充 | 17,139 | 31.81920581 |

主集包含train 264,063＋dev 3,016；补充包含train 16,997＋dev 142。保留完整片段，超出500h约4.53秒。补充仍按184≤p10<224、指定5类采用的音频正向标签、净化波形非零、官方core>0筛取，按p10／p50／采用类别数降序、sample ID升序统一排序；实际截断p10为192.09939981px。未叠加每pair配额、丰富度门槛或SFT严格文本筛查。

## 当前类别和数据概况

共有2,605个不同开放标签名称（视觉2,525、音频80），平均每段2.955个采用标签。名称包含近义词和低频细分，不能解释为已经统一定义的交际性backchannel类别。[完整类别CSV](selection/category_distribution.csv) · [类别及数据摘要](selection/CATEGORY_SUMMARY.json)。次数仍按用户接受的粗粒度尺度作参考，本次只调整来源范围，未启动细节重标或统计规则修订。

保留普通caption复核标记33,337段、已转正1,066段。用户此前确认“大致满足即可，不再继续细节打标”，此次继续沿用该尺度。

## 版本和验收

历史文件既有train＋dev候选统计，也有train-only旧缓存选择。本次以用户最新明确指示的train＋dev作为本轮范围，见[SCOPE_CORRECTION.json](SCOPE_CORRECTION.json)。原train-only版284,264段／500.00069849h保留在[旧目录](../listen_j_upright_av500h_v3_20261105/README.md)，不再作为当前选集。两版共用该目录下的冻结全量统计。

相较旧train-only版，新增3,158段、移出3,204段，共同保留281,060段，见[成员差异](selection/DIFF_FROM_TRAIN_ONLY.json)。不能在旧500h后直接追加dev，否则总量超出500h。

独立验收重新按Python排序复算补充成员，对照原始全量manifest验证完整train＋dev主集、源split、原caption／媒体／音频字段、全部CSV字段及SQLite偏移；无test、无重复。未重写任何原标签、原统计、旧缓存或正在运行的实验。

五项训练缓存已完成，统一读取见下方缓存索引。后续成员以本目录`selected.jsonl`为准；旧行中的`training_admitted`及`behavior_statistics_status`保留导出时原值，用于来源追溯。新行`source_split`与原`split`一致，实际类别／次数来源绑定在`statistics_file`和`statistics_offset`。

复现入口：`scripts/reselect_j_av500h_train_dev_v3.py`，源码SHA见[CODE_SNAPSHOT.json](CODE_SNAPSHOT.json)。

原始Qwen H已按本选集完成并验收，见[四机运行与读取入口](../p2_j_train_dev500h_qwen_hidden_20261105/README.md)。Gemma、AV latent及teacher不在本次范围，两种H各284,218条，完成回执见该运行COMPLETE.json。

**训练缓存统一入口已更新：** Qwen H、Gemma、AV VAE、speaker Whisper、listener Feature teacher均完成，覆盖本选集全部284,218段；[统一说明与读取](CACHE_README.md) · [机器可读索引](TRAINING_CACHE_INDEX.json)。各项旧文件引用和增量完成均有独立回执。
