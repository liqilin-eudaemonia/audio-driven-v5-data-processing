# 用户授权的十小时接续监控

首次检查：2026-10-31 16:08:37 UTC（北京时间11月1日00:08:37）。截止：2026-11-01 01:08:37 UTC（北京时间09:08:37）。任务验收完成即提前停止。

首次检查后每分钟检查接续进程身份；死进程在输入／源码检查通过后最多自动恢复3次，活进程不会因心跳陈旧而盲目重启。进入统计后立即做一次四机45秒采样，5分钟后复查，再每30分钟采样一次。既有接续控制器负责单卡恢复，异常来源／失败chunk保留并报告，主代理继续排查。

统计性能参考：D512，32卡端到端259.47段/秒，主测窗口GPU均值73.3%。加载、输入等待、尾部和CPU汇总分别解释，不用GPU占用率代替产出。

进程：child.json；状态：state.json；日志：monitor.log；检查报告：latest.json与report-*.json。原任务不因监控截止而停止。源码：scripts/monitor_listen_richness_goal.py。

## 首检与输入准备修复

首检报告report-1793462971.json：打标未完成，接续进程存活、没有失败。输入准备存在进度查询全表扫描，给coverage.status及sources.status增加覆盖索引并增加SQLite页缓存，保持FULL持久化及全部记录不变。真实541,208条caption准备记录等状态的计数完全一致；查询6.122秒降至0.0317秒，建索引6.96秒。60项CPU测试通过，准备程序自动恢复。验证在input_index_validation.json，修改快照在input_index_fix/。

## 两来源遗漏的补齐与覆盖核对

第一次收尾时，队列显示60,867来源／1,151,777段，原始清单为60,869／1,151,805。核对唯一job_id与片段数发现两个来源只有任务文件、未进入Omni追加索引。已验证原始索引SHA256、原始结果SHA256、点头来源及6＋22段数量，补回索引，按原规则完成画质检查和Omni打标。28段全部caption生成检查通过，source_records原始来源与音频配方核验通过。

补跑期间临时使用node17 GPU3，31卡统计继续，未停止其他任务；完成后已恢复32卡统计配置。修复证据：missing_source_recovery.json、missing_source_recovery_validation.json、omni_recovery.log。全量唯一来源ID及count与原清单一致；一条历史索引的额外recovery注释不影响ID、count或数据内容。

## 最终完成与停止

北京时间2026-11-01 01:24:58全量汇总完成，01:24:59监控自动退出，01:25:32独立全量验收通过。全部32卡模型进程结束并释放显存；没有留下周期监控。

完整范围60,869来源／1,151,805段：caption_ready 1,119,665，held无人脸291，visual_excluded 16,058，visual_review 15,791。输入来源错误和模型任务失败均为0。统计computed 1,084,936，recovered_with_review 34,729；0截断、0无法解析，局部问题没有导致删数据或当作零行为。

最后一次模型活跃窗口约393.80段/秒；node12／node8／node17／node4 GPU均值82.45%／92.09%／88.48%／91.46%，32卡平均88.62%。后续GPU释放属于CPU汇总阶段，不是卡等待输入。

全量独立验收见final_audit.json及audit_final.py：每段来源与原文一致、全部分块hash和ID一致、4,374段重放一致、最终JSONL拼接一致；CSV、coverage、类别表导出一致性见export_audit.json。gpu_release_after_extraction.json记录32卡显存释放。最终统计仍是模型候选，不等同于人工真值；没有重新筛选训练集。
