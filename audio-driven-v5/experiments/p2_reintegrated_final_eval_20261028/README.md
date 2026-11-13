# 当前正式评测：所有pipeline统一原独立test24

以用户最新明确的范围为准：上一批四组原本评测的独立test24保持不变，只替换为重新整合后的英文指导文本；新版也使用完全相同的test24。不加入train样本，不扩充为test48。

| 版本 | Pipeline | 权重 | 样本与seed | 输出数 |
|---|---|---|---|---:|
| 上一批固定prompt训练 | AV、AV＋Event、AV＋Feature、AV＋Event＋Feature | 各组最终1268步 | 同一原test24 × seed137/251 | 每组48，共192 |
| 新版标注指导训练 | AV、AV＋Event | 各组最终1249步 | 同上 | 每组48，共96 |
| 新版标注指导训练 | AV＋Feature | 等待最终1249步 | 同上 | 待补48 |

**当前6组可用最终权重共288个输出，全部7组计划共336个输出。** 6组288输出及指标均已完成并发布。新版AV＋Feature按用户新指令从第587步恢复三机24卡训练，已推进并保存第600步；不使用587步或200步替代最终权重。原后续训练及LoRA加速测试接续已重新挂接。

## 固定数据与对照条件

- [原test24输入](test24/observations.json)：与上一轮 `p2_test24_fourway_eval_20261027/observations.json` 完全一致。
- [新整合标注](test24/integrated_labels.jsonl)：从已完成的完整test48整合结果中取出对应原24条，成员不变。模型二次审核只记录，硬处理失败部分保留原描述和时间，不丢样本。
- 上一批四组保留原Qwen条件、权重、speaker音频、listener首帧、采样设置，只更新Gemma正向指导文本。新版Qwen与Gemma共同使用同一份指导文本。全部24条两条路径的实际Gemma文本已核对一致。
- GT继续使用上一轮test24的原视频与 `reference_audio` WAV，逐条核对它们与当前静音配方渲染的音频完全一致；不引入额外GT变化。
- [正式任务清单](jobs.test24.json)、[范围与成员校验](cluster24_test24/scope.json)、[复用产物审计](cluster24_test24/reuse_audit.json)。仅符合这24条、seed、权重和文件哈希的旧产物可复用。扩展范围产生的train或额外test输出不计入正式结果。

## 三机24卡运行

- [当前调度入口](cluster_test24.py)、[实时状态](eval_status.json)、[控制器日志](cluster_test24_controller.log)、[任务日志与状态](cluster24_test24/)。旧的 `cluster24` 宽范围队列已停止，勿重新启动。
- 每卡独立领取生成任务，按时长调度；编码器缓存复用，生成无需DDP同步。每组48个输出齐全后，自动执行FVD-32、UTMOSv2、SyncNet LSE-D/LSE-C，以及GLARE四项的严格和无额外质检门槛版本。音视频指标分3片并行；GT与检测缓存用锁保护共享复用。
- [推理适配](eval_worker.py)、[指标实现](eval_metrics.py)、[测试](test_eval_chain.py)、[本地源码备份](code_backup/MANIFEST.json)。已有最终1249步的真实GPU生成及指标试跑通过。所有结果属于目标标注指导诊断，不宣称是仅凭speaker自主规划的结果。
- 完成标记：`cluster24_test24/AVAILABLE_COMPLETE.json`。待最终权重项：`cluster24_test24/waiting_final_checkpoints.json`。停止当前评估：创建 `cluster24_test24/STOP`。
- [训练暂停及恢复权重](TRAINING_PAUSED.json)：历史暂停记录；模型、优化器和24卡RNG齐全。用户现已要求恢复，见[恢复记录](TRAINING_RESUMED.json)。

此前的test48、train24及step200试运行产物仅保留追溯。历史总数576/672均已被本页的288/336替代，不能混入当前对比结论。

## 指标阶段的空卡与收尾加速

288个视频生成完成后，GPU负载下降：小型音视频指标模型间歇使用GPU，人脸检测、视频解码、跟踪和部分统计计算主要在CPU。固定三分片曾造成短任务先完成、长任务等待，现另增加 [单样本指标与CPU检测辅助执行器](metric_helpers.py)：用非阻塞行锁领取未完成指标，已完成或被其他进程计算的行直接跳过；三机各增加4个持久CPU检测进程，共12个。共享同一行／视频锁和已锁定的原指标实现，不改样本、seed、公式或检测规则。

[辅助检测进度与日志](metric_helpers/)；GPU单样本辅助任务记录在 `cluster24_test24/states/row_helper_*.json`。任务完成数包含跳过已被领取的行，正式指标完成数以各组 `omni/rows/*/complete.json` 为准。

## HTML发布与训练恢复

[历史HTML总入口](http://127.0.0.1:8876/p2_av_review/)已加入[本批结果](http://127.0.0.1:8876/p2_av_review/reintegrated_20261028/)。6组×24条×2seed＝288输出；有声GT、speaker与ASR、listener独立/叠加音轨、逐样本全英文实际Gemma文本、完整指标表均可查看。旧HTML数据未覆盖。

[构建脚本](publish_review.py)、[指标展示](review_metrics.js)、[发布记录](HTML_PUBLISHED.json)。AV三类指标在六条件和两个seed之间取共同支持：FVD 24条、UTMOS/SyncNet各14条；GLARE去额外门槛版全24条，严格版逐条件报告自身支持数，不混作统一排名。

训练恢复使用原配置与原完整检查点；三个S1控制器及三个S2备用观察器共享锁，避免重复调度。S1仍按每组24卡顺序训练；四个S1完成后由原加速接续器测试并选择S2并行或串行。历史完成标记中的training_remains_paused仅记录评测完成时状态，当前状态以TRAINING_RESUMED.json和训练实时日志为准。
