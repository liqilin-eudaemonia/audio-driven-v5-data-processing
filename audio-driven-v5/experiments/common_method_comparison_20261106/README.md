# 本地方法比较：结果与暂停状态

**新增ViCo43共同输入实测已完成（2026-11-07）：** [我们／DyStream／Avatar Forcing三模型结果、逐条视频、配对区间及验收](../p2_vico43_unified_20261107/README.md)。新结果单独保存，不覆盖本目录旧结果；L2L人物迁移与文献仍独立分表，没有恢复暂停中的L2L训练。

**当前：统一实测及展示已完成；全部L2L训练停止，本地重训没有正式基线。VQ停27749步、码本诊断3000步结束、短预热predictor停569步，均未进入正式test43。先核对论文／补充与发布代码的协议差异。**

[完整调研、进度、协议与后续方案](../../related/P2_参考方法统一测评与L2L重训暂停交接_20261106.md) · [机器可读状态](HANDOFF_STATE.json)

[最新论文依据与纠正](../l2l_literature_protocol_audit_20261106/PAPER_GROUNDED_REPRODUCTION_AUDIT_20261107.md) · [VQ保存/停止](../p2_l2l_reliable_retrain_20261107/README.md) · [predictor诊断已停止](../p2_l2l_bounded_predictor_check_20261107/README.md)

## 查看结果

- [三个测试集指标与独立文献参考](http://localhost:8876/listen_benchmark_review/common_method_comparison_20261106/)
- [ResponseNet/Seamless全部400对GT、联合8000、DyStream视频](http://localhost:8876/listen_benchmark_review/common_method_comparison_20261106/video_pairs.html)
- [ViCo43视频及共同DECA总体/逐片段评分](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/)
- [汇总JSON](RESULTS.json) · [视频CSV](video_metrics.csv) · [共同DECA CSV](common_deca_metrics.csv) · [文献引用](published_reference.json) · [展示验收](FINAL_ACCEPTANCE.json)

## 已完成范围

| 项目 | 已完成 |
|---|---|
| ViCo | 我们8000步与DyStream各43条视频，原官方画质/身份五项，共同DECA六组动作结果 |
| ResponseNet | DyStream固定200条，与14个既有我们模型arm汇总 |
| Seamless | DyStream固定200条，与14个既有我们模型arm汇总；图像200/200、情绪/ΔResp199/200 |
| 官方L2L | Conan/Fallon/Stephen/Trevor各43条原生动作；无写实renderer，图像指标N/A |
| 文献参考 | DIM第11页Table1的8行独立转录，不加入本地分数或跨表排名 |

复用已封存视频，不重新生成。ViCo为512×512/25fps生成→抗混叠256×256/30fps评分；DyStream自身音频全零、speaker进入other分支、同一首帧、不二次裁脸、官方5步、seed137。

共同DECA为原始表情50＋头旋转3＋下颌3、30fps，无平移。旧speaker/GT86路复用，新提取我们及DyStream86路/32576帧；四套L2L172条保留。RTLCC延迟±60帧，RWTLCC完整120帧窗口/30帧步长，分别覆盖42、41条，其他43条。旧L2L逐条分数回归误差0，独立复算258条预测L1误差0。

DECA、旧BFM、文献EMOCA不跨空间排名；视频反提取与原生动作、训练集和输入模态仍有差异。训练来源和有限范围的去重见 [TRAINING_PROVENANCE.json](TRAINING_PROVENANCE.json)。400对按原清单完整展示；Seamless GT预览从已校验评分RGB缓存制作，ResponseNet GT复用原始640×640，不改变评分输入。

## 数据与实际训练覆盖

用户选择从官方train398隔离1条与test共享speaker时间段的样本，剩397条按源对话、seed137划为train351/validation46；test43与OOD42不用于训练、拟合统计或选权重。[划分](l2l_retrained_v2/split.json)

397对特征全部完成，实际缓存位于`l2l_retrained/features/`，v2的features是符号链接复用。64帧窗口，train步长32、validation步长64，无短片段补帧；实际有窗口的片段为326/351及41/46，窗口数3637、336。[窗口与短片段](l2l_retrained/WINDOWS.json)

音频按metadata.audio读WAV；一条WAV比4秒视频短约0.3545秒，尾补零保持时间轴，逐条调整量在receipt。保留官方`mean_std_swap`，仅拟合训练窗口；是否适合当前分布尚待检查。

## 两轮结果与当前暂停点

| 版本 | 执行结果 | 正式比较资格 |
|---|---|---|
| [pilot](l2l_training_pilot/RESULTS.json) | train8/val4，两阶段各100步，链路通过 | 不是收敛模型 |
| [v1](l2l_retrained/QUALITY_GATE.json) | VQ3750步、predictor4250步、test43执行完成；VQ最优第1步 | 未收敛诊断，已移出正式比较 |
| [v2](l2l_retrained_v2/vq_COMPLETE.json) | VQ20000步跑满；末次验证38.144559，最优仍第1步14.636833 | 用户要求VQ后暂停，predictor/test43未启动，无有效重训基线 |

保持原架构及loss，从头初始化，batch64、lr1e−4、AdamW/Adam。恒定学习率是本地方案，不等于原仓库warmup或DIM配置。v2只取消原预算内提前停止；延长步数未解决问题，不能确定根因仅为早停。v2自动codec有效性检查未执行。

[暂停凭证](l2l_retrained_v2/PAUSED_AFTER_VQ.json)确认本轮进程退出、GPU上下文释放。其他LTX任务随后接用GPU7；这只是当时快照，不表示卡当前空闲或仍为本轮预留。**保持暂停，后续仅在用户明确要求恢复后执行。**

## 权重与恢复限制

- v2的[vq_best.pth](l2l_retrained_v2/vq_best.pth)是第1步模型及对应优化器，不是第20000步权重。
- 没有第20000步last模型/优化器/RNG完整checkpoint，不能从第20000步精确续训；[完整历史](l2l_retrained_v2/vq_history.json)、统计、划分、协议和397对特征均保留。
- [ACTIVE_L2L_RUN.json](ACTIVE_L2L_RUN.json)只指向当前版本，不表示进程仍运行。
- v1的`COMPLETE.json`和根目录`COMPLETE.execution_only_v1.json`只证明历史执行完成，不能覆盖质量不合格；v2没有整轮完成标记。

## 后续方案与代码

恢复前先核对暂停与资源。先在validation拆解VQ重建/量化loss、比较训练均值、检查归一化尺度与官方optimizer/warmup差异；新方案使用新目录，复用特征和划分，保留失败记录，补齐周期性last及RNG保存。VQ有效性通过后才接predictor，最后冻结checkpoint评test43，不能按test挑配置。

- [共同DECA](../../compare/deployment/vico_common_deca.py)
- [v1训练](../../compare/deployment/vico_l2l_retrain.py)、[v2训练](../../compare/deployment/vico_l2l_retrain_v2.py)：不是当前可直接执行的暂停恢复命令
- [汇总构建](../../scripts/build_common_method_comparison.py)、[ViCo构建](../../scripts/build_vico_three_method_review.py)、[验收](../../scripts/check_common_method_comparison.py)
- [论文及issue审计](../l2l_literature_protocol_audit_20261106/github_issues/FINDINGS.json)

只重建展示可使用既有构建脚本；不要根据历史“自动接续”描述重启GPU训练。完整资源缺口、证据和恢复边界见 [总交接](../../related/P2_参考方法统一测评与L2L重训暂停交接_20261106.md)。
