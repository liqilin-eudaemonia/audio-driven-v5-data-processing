**L2L复现依据重新核对（2026-11-07）：** 当前全部L2L训练已停止。原论文补充与公开代码在warmup、VQ隐藏维度和优化器上存在差异；官方README明确VQ loss可能先升后降，不能据此断定训练故障。短预热和码本初始化均只保留为诊断，不计作复现。[逐项证据、纠正与执行顺序](../experiments/l2l_literature_protocol_audit_20261106/PAPER_GROUNDED_REPRODUCTION_AUDIT_20261107.md)。

**L2L可靠重训已重新授权（2026-11-07）：** 用户允许临时使用node11 GPU7并与现有推理叠加，主种子137的VQ因量化项不稳定已在27749步保存并停止；3000步码本诊断已结束，另一个短预热predictor诊断已在569步停止；当前全部L2L训练停止，先核对论文协议。新轮已通过官方更新／断点恢复／小样本学习预检，恢复官方预热；通过验证学习检查后才进入test43；旧两轮第1步最优结果仍不算有效基线。[当前方案、进度和验收入口](../experiments/p2_l2l_reliable_retrain_20261107/README.md)。

**参考方法部署/测评交接（2026-11-06）：** 当前可用的本地统一结果为我们、DyStream及L2L四套人物迁移模型；本地L2L重训尚无合格基线。第二轮VQ20000步结束后已按用户要求暂停，后续需用户明确要求再恢复。[完整调研、方法缺口、协议和暂停后方案](../related/P2_参考方法统一测评与L2L重训暂停交接_20261106.md) · [结果与页面](../experiments/common_method_comparison_20261106/README.md) · [交接状态JSON](../experiments/common_method_comparison_20261106/HANDOFF_STATE.json)。

**七个参考方法真实训练接入已短跑通过（2026-11-09）：** GLARE、DyStream、Avatar Forcing、L2L、DIM、ViCo、ListenFormer 已用最终选集真实片段完成完整网络反向／更新与断点重放，七个方法的固定代码新进程恢复也全部通过。DyStream原生训练步骤在 `main.py`，先前“无训练循环”的判断已更正；Avatar Forcing是基于原生forward和scheduler补齐的训练适配器，未含论文偏好阶段。L2L／DIM的多阶段均实际更新，尚未完成500h正式训练。[代码、命令、验收及全量阶段剩余工作](../experiments/baseline_training_realdata_20261109/README.md)。

# 参考工作的部署与 ViCo test43 实测

**三方法统一展示已接入：** [同页查看联合8000与DyStream的视频和评分、Learning2Listen四套动作结果](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/)。联合与DyStream可并排同步播放，L2L没有写实推理视频，单独展示DECA动作评分／曲线；全部43组已验证。

2026-11-06 已完成当前可运行的两项工作：DyStream 43 条视频及画质／身份／BFM运动评分；Learning2Listen 四套人物模型各43条，共172条DECA动作结果及独立评分。

[打开全部视频与动作曲线](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/) · [完整结果与缺项说明](../experiments/p2_compare_vico43_20261106/RESULTS.md) · [最终验收](../experiments/p2_compare_vico43_20261106/FINAL_AUDIT.json)

| 工作 | 当前可运行范围 | 本轮ViCo完成情况 |
|---|---|---|
| DyStream | 视频生成；不生成listener声音 | 43/43条，5项画质／身份与6组运动指标 |
| Learning2Listen | 四套person-specific动作模型，无公开写实renderer | 4×43条，独立DECA50＋头旋转3＋下颌3评分 |
| DIM | 环境就绪，缺核心生成权重 | 未运行完整生成 |
| ViCo baseline | 入口可用，缺driver权重和训练统计 | 未运行完整生成 |
| MM2022-ViCoPerceptualHeadGeneration | 修改已集成，缺训练完成的核心driver权重 | 未运行完整生成 |
| ReactFace | 环境／FaceVerse可用，缺核心checkpoint及部分资源 | 未运行完整生成 |
| OmniResponse | 仓库未公开完整生成代码／权重 | 未运行论文生成模型 |
| SpeakerVid-5M-Code | 数据处理与评分工具可用，无对话生成器 | 未运行论文生成模型 |

DyStream采用用户确认的listener适配：target自身音频全零、speaker进入other分支、同一listener首帧且不二次裁脸；512/25生成、256/30评分。Learning2Listen额外使用speaker视频动作，在独立适配器修正音频4t对齐；四套权重全部测试，不按真值择优。六项缺资源的工作没有计为测试完成。

详细部署命令与环境见 [deployment/README.md](deployment/README.md)，机器可读状态见 [deployment/STATUS.json](deployment/STATUS.json)，本轮官方资源复查见 [current_generator_audit_20261106.json](deployment/logs/current_generator_audit_20261106.json)。统一仅使用node8物理GPU7，未动0–6卡训练。
