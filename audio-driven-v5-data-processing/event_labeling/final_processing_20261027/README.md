# 最终数据处理、整合、筛选与缓存代码备份

2026-10-27。覆盖上游数据处理、原始打标、原文整合、官方区间静音／中文过滤、全量缓存、统计、50h／双300h筛选及test48导出。旧备份保留；此次没有重新处理数据或启动训练。

固定版本：[`p2-final-data-processing-20261027`](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/p2-final-data-processing-20261027/event_labeling/final_processing_20261027)。

## 检查与验证

此前备份未完整覆盖最终版本。对364个当前项目Python文件按SHA比对，197个有同版本副本，167个未找到当前版本副本（含新增和更新），现已补齐。逐文件对照见 [prior_code_coverage.json](evidence/prior_code_coverage.json)。

本次文件范围见 [BACKUP_MANIFEST.json](BACKUP_MANIFEST.json)，包括源码、依赖、配置、文档、完成收据和压缩子集名单。备份副本 **140项CPU测试、9个子测试通过**，关键包导入位置均在快照内；见 [测试日志](validation/restored_cpu.log) 和 [验收汇总](BACKUP_VALIDATION.json)。

## 当前代码入口

以下均位于 `snapshot/audio-driven-v5/`，保持原文件内容与目录层级：

| 阶段 | 入口 |
|---|---|
| 上游切分／音质／人脸统计 | `scripts/listen_policy_v8.py`、`listen_v8_audio_final.py`、`run_listen_local_full.py`等已有入口 |
| 原始打标和调度 | `scripts/run_listen_full_raw.py`、`supervise_listen_full_raw.py`、`src/listen_benchmark/full_raw.py`、`prompts.py` |
| 原七类元数据与原文整合 | `scripts/manage_listen_full_fusion.py`、`run_listen_full_fusion_semantic.py`、`src/listen_benchmark/full_fusion.py`、`existing_result_fusion.py`、`integration_pilot.py` |
| 官方时间支持／中文过滤 | `src/listen_benchmark/official_audio_gate.py`、`official_event_alignment.py`、`chinese_asr_filter.py` |
| 音频后处理及训练读取 | `scripts/p2_audio_postprocess.py`、`src/listen_audio_postprocess/`、`src/p2_listen/training_data.py` |
| 全量后处理接续 | `scripts/p2_full_audio_postprocess.py`、`src/listen_audio_full/` |
| 特征抽取和缓存调度 | `scripts/p2_listen.py`、`run_p2_full_features.py`、`manage_p2_full_features.py`、`src/p2_listen/features.py`、`src/listen_benchmark/feature_pipeline.py`、`resident_features.py` |
| 人脸／事件／时间统计 | `scripts/listen_backchannel_face_stats.py`、`report_postprocessed_backchannels.py`、`report_open_backchannel_times.py` |
| 新50h／300h七类选择 | `scripts/select_postprocessed_listen_subset.py`、`src/listen_benchmark/train_subset.py`、`postprocessed_subset.py` |
| 300h全部原文版 | `scripts/prepare_open_verbatim_subset.py`、`select_open_verbatim_subset.py`、`src/listen_benchmark/open_verbatim_subset.py`、`open_subset_reader.py` |
| test48导出 | `scripts/select_processed_test_panel.py` |
| 子集验收 | `scripts/audit_postprocessed_listen_subset.py`、`audit_open_verbatim_subset.py`、`src/listen_benchmark/audit_identity_snapshot.py` |

300h原文版沿用已有去重整合，不新增语义类型或改写原句。已取消的新语义类别清单试跑不属于最终流程，未纳入本快照；上游已使用的七类元数据代码仍保留。

## 生产运行时的冻结版本

| 阶段 | 精确匹配文件数 | 位置 |
|---|---:|---|
| 原始打标 | 48 | `production_stage_snapshots/listen_raw_crop384_20261024T131300Z/` |
| 全量整合 | 12 | `production_stage_snapshots/listen_full_fusion_crop384_20261025/` |
| 全量特征抽取 | 29 | `production_stage_snapshots/p2_features_full_crop384_20261025/` |
| 全量音频后处理 | 74 | `production_full_snapshot/audio-driven-v5/` |

共163个文件与生产任务代码清单SHA完全一致。绑定见 [production_stage_bindings.json](evidence/production_stage_bindings.json) 和 [production_full_bindings.json](evidence/production_full_bindings.json)。例如筛选器后来为300h修改过，不能用当前文件冒充旧运行代码；历史版本从已有运行快照或已验证副本恢复。

`snapshot/audio-driven-v4/`、`snapshot/Omni-LTX-stage2/` 保存本流程依赖源码与锁文件，须与 `audio-driven-v5` 保持同级。未打包Python环境或模型。

## 数据和存储边界

[最终数据与缓存位置总览](snapshot/audio-driven-v5/related/P2_最终数据集与缓存位置总览.md)随快照保存。各数据集的配置、完成收据和外部文件引用见 [dataset_references.json](evidence/dataset_references.json)。`membership/` 保存新50h、两份300h和test24/test48的ID名单gzip，解压后字节与原名单一致。

不包含媒体、模型权重、训练checkpoint、latent、完整标注表、全量官方metadata或凭据。原数据仍在 `/private/liqi_workspace/project_code/audio-driven-v5/`；配置绝对路径和历史文档媒体链接指向原机器，代码备份不是完整数据集镜像。迁移需要相应数据和路径合同，不能删除哈希校验强行复用缓存。

## 检查与恢复

文件校验不需要GPU或原数据：

```bash
python verify_backup.py
```

在已安装依赖的Python环境运行CPU回归，脚本隐藏CUDA，检查导入位置，不加载生产模型或启动训练：

```bash
P2_PYTHON=/path/to/python bash restore_cpu_check.sh
```

测试覆盖打标组织、原文去重、静音及事件对齐、中文过滤、虚拟波形和增量缓存、全量接续、特征调度、统计、子集筛选及读取隔离；不替代生成效果评估。当前后处理读取关闭Feature loss，原文版仍需训练端明确消费开放标签。

`build_backup.py` 保存生成逻辑，存在 `BACKUP_MANIFEST.json` 时拒绝覆盖，后续更新使用新版本目录。
