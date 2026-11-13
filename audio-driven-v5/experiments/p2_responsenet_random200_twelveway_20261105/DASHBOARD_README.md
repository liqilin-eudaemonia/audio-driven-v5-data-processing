**GLARE已加入总结果页（2026-11-12）：** Seamless固定200新增GLARE19050／32940，现22组／9组对照／41项差值；新增全LoRA6000、6428及原完整6000→GLARE19050、GLARE19050→32940。逐ID核验200条GT RGB、帧数、裁框与倾听区间一致；声音指标N/A，不记0、不参与音频排名，CSV同步。总页新增ViCo43、ResponseNet完整586、Seamless固定200及完整test6624独立汇总，第一名加粗、第二名下划线；ResponseNet随机200保持18组，完整586不混入随机200。19050全集仍在全局汇总／验收，32940全集按用户要求暂停，未完成不填成绩；完成后渲染入口自动同步总页表格。994项数值／差值、浏览器、移动端、CSV及原视频音轨检查通过。[GLARE方法对照](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=glare#ablation-glare) · [各测试范围汇总](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless#checkpoint-results)。

**当前Feature／Whisper6000评测完成并更新总结果页（2026-11-08）：** 两组均使用最新SFT2752，双200共800条生成／解码／视频与音频评分和最终验收完成；四机接续耗时约23分钟。总页现为两测试集各18组、30项对照；新增当前AV→Feature、AV→Whisper、Feature→Whisper及两组分别→联合6000，原值、B−A箭头、第一／第二名颜色、音频覆盖与CSV同步。840项数值／差值及浏览器验收通过。[打开总结果页](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-joint_whisper)。

**当前500h AV-only6000＋SFT2752双200评测已完成并加入总结果页（2026-11-08）：** 400条最终验收通过；ResponseNet／Seamless各16组、25项差值对照，新增“相同SFT2752条件：当前仅AV→联合6000”，历史AV基线独立保留。总体指标、音频覆盖、第一／第二名配色、B−A箭头及CSV同步更新；700项数值／差值和浏览器验收通过。[总结果页](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-joint_whisper)。

**最新step6000＋SFT2752已加入原测评页面（2026-11-07）：** ResponseNet／Seamless各15组，7组消融、24项对照；新增主干Event＋Feature＋Whisper6000＋SFT2752，与历史联合6000、仅AV6000及AV＋Event6000对比。原值、B−A、指标箭头、前两名配色和音频覆盖均保留；注明数据／Event读出／规划器多项变化，不能归因于单一组件。672项数值／差值及浏览器验收通过。[打开新增对照](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-whisper)。

**本轮联合Whisper已加入原可视化（2026-11-06）：** 两套测试集各14组，7组消融、21项对照。新增联合6000／8000，明确与仅AV、AV＋Event、AV＋Feature、原Event＋Feature和仅AV＋Whisper比较；含6000→8000及预算不等的终点参考。原值、B−A、箭头、第一／第二名颜色、音频及覆盖保留。 [ResponseNet联合消融](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=responsenet&ablation=joint_whisper) · [Seamless联合消融](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper) · [有声视频基线对照](http://localhost:8876/listen_benchmark_review/p2_seamless200_20261105/?preset=joint_av&metrics=1)。588项数值／差值检查、浏览器与32份新增媒体／PCM核验通过。

# 两测试集十二组数值与消融表

主页面：http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/

支持ResponseNet分层随机200条与Seamless固定200条分别切换。原Seamless视频页同步同一套消融表；历史ResponseNet精选视频页增加新入口，原视频／音轨保留。

每套12组总体结果，第一／第二名分别绿色／蓝色，按未舍入值并列排名。六组消融共11项对照：Event和Feature各6000／8927；组合7000相对AV／Event／Feature6000；SFT推理Qwen；SFT训练H；Whisper6000相对AV6000及Whisper6000→8000。

所有变化量为B−A，根据指标方向标绿色改善、橙色退化。百分比差值显示pp（百分点）。展开显示基线A、方案B、差值及音频指标。Emo-Div保留生成／GT原始均值，原值不单独判优，另展示与GT绝对差距。

CSV分别导出12组原值和全部消融变化。数据来自3份既有RESULTS.json，来源SHA和参考清单SHA记录在dashboard.json。ResponseNet保留各pair原始帧率（23.98、24、25、29.97、30fps），Seamless为25fps。未修改原始评测JSON、推理输出、裁框、音轨或标签。

比较限制：组合7000近似匹配6000组的训练展示数；SFT推理同时改变文本和H；SFT训练H对照还有训练调度差异；Whisper8000只作进展参考。页面均已注明。

构建：`/private/software/conda/bin/python scripts/build_p2_metrics_dashboard.py`。
检查：`/tmp/native-fps-review-qa/bin/python scripts/check_p2_metrics_dashboard.py`。

代码为上述两脚本与scripts/p2_metrics_dashboard.html、scripts/p2_ablation_tables.js。构建只读取既有评测JSON，不扫描原始媒体或调用GPU。308项原值／差值、两套12行总体表、6组／11项消融、CSV、移动布局、原Seamless视频与音轨全部核对通过；见DASHBOARD_BROWSER_AUDIT.json。代码快照保存在dashboard_code_snapshot/。
