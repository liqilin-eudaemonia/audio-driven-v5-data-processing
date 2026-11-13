# AV-only6000评分改为GPU7叠加运行

用户2026-11-08明确要求叠加评分。切换前GPU7已有另一项任务，28332MiB／利用率约20–25%，可用115439MiB。此前卡住的原因是原空闲门槛（used<4096MiB、util<10），不是指标或模型失败。

只替换了本轮闲置评分协调器PID3262495；当时active=0。另一项任务及PID3987181不动，400条生成／解码与已通过的评分smoke全部保留。不修改原生评分源码或本轮模型、SFT条件、采样、缓存及指标定义。

新入口scripts/p2_current_av6000_score_overlay_20261108.py：仅node11 GPU7，评分最多8路，预留16GiB，按新增每worker预算8GiB计算可启动数量。GPU利用率不再是拒绝叠加的条件；新增进程仍受显存余量限制。后续资源突增可能减少新任务准入，不终止现有其他任务。

沿用CPU读取／像素计算流水线、batch64和原8个固定评分分片，最终用全200条特征重算FID/FVD。完成后调用原生final核验并输出RESULTS.json、COMPARISON.json及FINAL_AUDIT.json。STOP沿用；没有恢复Goal或聊天定时巡检。

当前状态metrics_status.json。RESOURCE.json、SLOT_BUDGET.json为作业调度快照，EXECUTION.json和CODE_SHA256.json记录执行来源。PREVIOUS_CONTROLLER.json记录被替换的本作业协调器，其他任务未停止。
