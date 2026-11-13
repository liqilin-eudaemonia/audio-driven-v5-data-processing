> 此版将人工反馈用于直接裁定并留空边界，已被用户纠正。当前使用 [CPU自动整合版](../listen_existing_fusion_cpu_20261025/README.md)，本页保留历史记录。

# 50 段开放事件整合试验（2026-10-25）

已按用户确认的流程在现有50段实施，供复核；没有应用到全量原始打标，也没有自动准入训练。

页面：[整合复核](http://127.0.0.1:8876/listen_benchmark_review/integration_pilot/)。远端8876需由编辑器转发到本机；默认全部50段，可筛选5个人工回归案例。

## 当前结果与口径

- 682条原始观测逐字节保留；其中210条七类已归类观测，经人工修正、证据关联和争议单列后，得到182个独立事件候选：134视觉、48音频。不能把减少的28条全部解释为跨窗去重。
- ASR原始词观测130条，保守匹配为85个候选索引；45条跨窗重复观测不再重复计入索引，但所有成员和词序仍保存。30条时间无效记录仍保留。85不是实际发声次数。
- 已有语义观测层此前做过整合；本轮没有额外满足严格事件级匹配条件的观测合并（0条），没有为了压缩数量放松条件。
- 9个事件时间未知，start/end明确为null。15段当前没有七类独立候选，属于未知覆盖，不能作为无事件负例。
- 本次不重新运行模型或声学定位，不声称所有边界达到±0.2秒。匹配容差0.2秒只是寻找重复候选的参数。

## 实现规则

复用已有模型语义签名、开放描述及七类语义映射，不新增关键词类别映射。同语义且边界接近、有重叠、来源不冲突的跨窗观测，采用完整组内互相匹配；同一来源的独立观测不合并，不做传递式区间并集。宽音频描述可关联多个词作为证据，不扩大事件区间或增加独立次数。与ASR同词但时间不相交的描述保留为潜在额外回应争议。

全量整段/滑窗词序单独保存，包括零时长词。整段参考不覆盖更具体的滑窗边界。视觉继承的旧区间并集显式标记边界待审，不声称重新定位。每个事件保留各自模态边界；上层组只表达可能同时发生，不输出持续活动掩码。单独视觉事件也有独立组引用。

5个人工案例的修正是明确的逐样本裁定，不是可泛化到其他数据的字符串规则。原始反馈文件完整保留，模型原文不改写。

## 回归案例

- S2043：一条“嘿嘿式笑声”候选，同时关联已有Laughs观测；不重复生成两条笑声，也不把三个嘿当三个词。笑声段数和边界未确认。
- S0805：Right×2、Yeah×5；七个独立词事件保留顺序，但时间全部留空。已提交的三个区间保存在反馈原文中，没有强行分配到某几个词。额外模型音频证据仍可见，并未据此判为不存在。
- S0835：Yeah及尾音；旧Ed仍在原始侧，不作为整合内容；3.88–3.96只作核心参考，尾音结束未知。
- S3358：两次嗯，分别保留0.32–1.44、1.44–1.52的提交区间，精度未单独核验。
- S5193：Uh huh、Yeah两个回应；采用用户3–4、7–8的大致时间，标记旧ASR时间分歧。
- S0699：两次Okay保持1.78–2.10、4.08–4.40；整段2.08–4.40的宽词记录只是冲突证据，不桥接两次回应。

## 文件与训练接口

[候选标签](labels.candidate.jsonl)、[完整复核数据](review.json)、[原始观测](observations.jsonl)、[人工修正](human_corrections.json)、[原始反馈](human_feedback.original.json)、[统计](summary.json)、[验证](validation.json)、[源文件和代码哈希](manifest.json)。

候选包含events、response_groups、evidence_links、unresolved_observations、token_clusters、token_event_links。事件的content/count/boundary三个状态分开，缺边界用null，unlabeled_time_policy=unknown，negative_coverage为空。training_admitted=false；当前不生成可直接监督的精确时间掩码。后续训练导入器须按已确认字段与任务要求决定监督，不能把组范围或未标时间自动填成正/负样本。

## 如何复核

选择样本和模态，左侧整合候选、右侧原始观测。时间未知的事件播放的是明确标出的参考上下文。可分别确认内容或内容＋区间，也可报告需要修改。没有候选的段仍可点击“报告漏标／整段问题”。保存后导出listen-integration-human-review-v1 JSON发回；支持导入恢复。浏览器自动化测试的反馈只存在独立临时浏览器，不是人工结论。

## 代码与重放

核心src/listen_benchmark/integration_pilot.py；入口scripts/build_listen_integration_pilot.py；页面scripts/listen_integration_pilot.html；回归tests/test_listen_integration_pilot.py与scripts/check_listen_integration_pilot.py。代码快照在code/。

构建参数：--source experiments/listen_response_episodes_20261024T120000Z --align experiments/listen_split_validation_20261024T074733Z/word_alignment --feedback experiments/listen_response_review_20261025/human_feedback/20261025T022241Z.original.json --out <新目录> --preview data_previews/listen_benchmark_review/<新目录>。可选--font NotoSansCJK-Regular.ttc生成完整当前页面所需中文子集字体。输出目录须为空。

完成13项相关单元测试及真实50段回归验证，浏览器检查媒体同步、Range206、保存导出导入、未知边界拒绝确认、空候选段反馈与手机布局。最终应用全量前仍须用户复核认可。
