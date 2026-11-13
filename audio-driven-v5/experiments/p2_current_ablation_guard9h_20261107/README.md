# 四机31卡消融九小时巡检

**完整九小时已验收结束，训练继续运行。** hour00–09四机31rank实机记录均正常；Feature精确停在6000，完整模型／Adam／31RNG及SHA通过，旧进程全部退出，Whisper31卡自动从原始预训练启动。最终实查Whisper1609步，最近200步5.99秒／步；Whisper500／1000／1500完整检查点通过。全量连续更新、loss公式、梯度、源码和最终500h／最新caption身份复核通过，期间无训练故障、持续异常降速或恢复重启；保留node11 GPU7。[FINAL_ACCEPTED.json](FINAL_ACCEPTED.json)逐项列出目标证据。独立九小时observer已按期退出，Whisper coordinator常规监控与TensorBoard6009继续运行。

用户目标窗口：2026-11-07 14:46:23 UTC至23:46:23 UTC（北京时间11月7日22:46:23至11月8日07:46:23）。到期不停止训练。

每15秒核对实际metrics与Whisper接续控制器；每小时独立通过SSH/NVML和/proc核对四机的31个真实训练rank、GPU映射、配置/源码身份、损失/梯度和按片段时长分组的速度。保留node11 GPU7，不终止其他任务。保存允许600秒、新模型冷启动允许900秒宽限；查询超时不代表训练已死亡，不据此重启。

AV＋Feature在6000完整保存后，已由既有后台coordinator自动切换到新AV＋speaker Whisper，两个消融分别从原始预训练初始化。

- guardian.py / GUARDIAN.json：独立巡检代码和真实PID身份。
- reports/hour_00.json至hour_09.json：对应时段的实际完整巡检证据。
- status.json / events.jsonl：15秒检查及事件。
- NEEDS_AGENT.json：异常诊断请求，不能据此声称健康。
- LATEST_AGENT_AUDIT.json：代理当前推进及未完事项。
- STOP：仅停止此巡检，不能停止训练。

巡检和有界自动恢复完成了完整九小时窗口。10项保护测试通过，Feature6000完整检查点的身份、模型、Adam、31份RNG已只读验证，真实6000步退出及Whisper31卡接续已验收；没有为测试恢复而重启训练。最终验收检查完整窗口、全部逐小时实机证据及当前运行状态，见[FINAL_ACCEPTED.json](FINAL_ACCEPTED.json)。切换后的速度基线已补齐所有实际时长档位，固定已有档位避免吸收持续降速；只重载独立observer，训练没有中断，见[SPEED_COVERAGE_FIX.json](SPEED_COVERAGE_FIX.json)。


自动恢复：普通更新180秒无进展、保存超过600秒、冷启动超过900秒或确认训练控制器退出时，先验证全量节点观测和完整检查点，再仅终止本实验且PID启动身份、CLI/config、rank/mask均匹配的worker。保留所有外部任务及node11 GPU7，31张所选卡必须确实空闲后才严格恢复。严重按时长匹配降速必须持续180秒并经过20个新step，且无其他任务争用时才考虑本任务恢复。单次慢步、GPU瞬时0%或SSH失败不会触发重启。30分钟最多3次失败/重启，超过则留下NEEDS_AGENT供活跃goal诊断；明确STOP优先。

recovery.py负责故障恢复与失效接续控制器重启；RECOVERY_DRYRUN.json是当前完整检查点只读验证；LIVE_ACCEPTANCE.json是部署后31rank快照；HANDOFF_ACCEPTED.json在真实6000保存退出＋Whisper20步及31rank验收通过后才写出。

到期边界已补强：截止时间阻止新的干预，但已清理本任务进程的恢复事务必须完成复原；明确的训练STOP仍优先。见DEADLINE_RECOVERY_FIX.json。

6000步保存失败守护已补强：达到6000本身不能视为正常退出，必须存在完整保存标记，缺标记时仍检测死进程和真实停滞并允许恢复。见FAILED6000_GUARD_FIX.json；仅更新独立守护，训练源码未改。
