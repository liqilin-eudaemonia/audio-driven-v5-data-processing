# 当前完整LoRA训练四机九小时巡检

用户授权目标：每小时检查四机；训练退出、卡死或持续严重降速时及时恢复。当前目标窗口：2026-11-08T15:25:25+00:00至2026-11-09T00:25:25+00:00，北京时间结束为2026-11-09T08:25:25+08:00。部署或观察器重载不重置期限，到期不停止训练。

## 当前作用对象

固定`p2_ltx_all_attention_ffn_lora31_20261108`完整pipeline，四机31rank：node12/node8/node4 GPU0–7，node11 GPU1–7，GPU0保留。全重计算、单网卡、rank8、AV＋0.05整段Event＋两路各0.025 Feature、batch31、每500步保存、最终500h及最新caption保持。node11训练108GiB分配器上限，其余134GiB；保留已授权vLLM并存。

## 监测与恢复规则

- 每15秒检查实际metrics、loss公式、有限梯度、停步和按时长匹配的步速；每60秒通过SSH/NVML和/proc重新核对四机31rank身份与GPU上下文。
- 每小时保存完整巡检（hour00–09），含真实四机状态、配置/282源码和当前数据缓存身份。时长分组速度基线固定为本次并存训练的已有稳态样本，不以温度或单次0%利用率判断故障。
- 普通更新允许180秒无出步，正常checkpoint保存允许600秒；冷启动最多900秒。SSH超时或不完整观测不能触发终止；真实失败需要完整观测及两次确认。
- 按时长匹配的最近32步中位速度比超过1.5，持续180秒且新增20步，才触发严重降速恢复判断。正常保存中不做速度干预；已允许的vLLM上下文仍在且31rank、数值和出步健康时，不用隔离运行的速度基线触发重启。卡死、退出及数值错误的恢复保持，见[共享负载判断修正](SHARED_COMPUTE_POLICY.md)。
- 恢复前验证完整checkpoint SHA、模型、Adam和31份RNG。只终止CLI/config、rank/mask、PID启动身份全部匹配的本训练worker；保留vLLM、GPU0及其他任务。严重降速优先请求原生STOP完整保存当前进度；卡死或退出从最近有效完整检查点恢复。
- 任何用户训练STOP优先；不能清理新用户STOP。30分钟最多3次恢复，次数跨观察器重启保留；超过上限写NEEDS_AGENT，由活动goal继续诊断，避免反复重启。
- 恢复必须确认至少3步新更新后才算成功。监督器每30秒检查观察器真实PID和启动身份，只有确认观察器退出才重启；不会仅因观测超时杀观察器或训练。
- 截止时间后不发起新的常规干预，但已经开始清理本任务的恢复事务必须完成复原。完整窗口结束后另做最终实机验收；部署成功不等于九小时目标已完成。

## 已完成的部署验收

最初15项部署保护测试通过，后续修正后共19项通过（见tests.xml及[修正记录](SHARED_COMPUTE_POLICY.md)）：观察不完整、SSH失败、未知占卡、允许vLLM、PID复用／wrong config／wrong rank禁止发信号、速度持续门槛、STOP／截止边界。step1000完整检查点只读验证通过（模型／Adam有限、31份RNG、CPU随机状态恢复，无CUDA初始化）；没有为测试而故意停止／重启训练。

证据：LIVE_ACCEPTED.json、RECOVERY_DRYRUN.json、PYTEST.log、tests.xml、reports/hour_00.json。guardian.py／common.py及引用的历史观察／信号代码均已本地冻结，OBSERVER_SOURCE.json保存SHA。

实时：status.json、LATEST_CLUSTER.json、events.jsonl、GUARDIAN.json、SUPERVISOR.json。异常：NEEDS_AGENT.json／RECOVERY_ERROR.json。WORK/STOP只停止本守护，不停止训练；训练自己的STOP会阻止自动恢复。

完整九小时窗口及最终四机实机验收现已完成，见[FINAL_ACCEPTED.json](FINAL_ACCEPTED.json)。验收包含到期后新增更新、全部小时报告、恢复事务和完整检查点；没有仅凭WINDOW_FINISHED.json判定完成。

## 九小时结束与最终验收

原定窗口未变；最终验收时间为2026-11-09T00:31:06.886035+00:00。验收时step4337，四机31rank正常，最近50步平均7.80秒。九小时观察器与监督器已按期退出，原训练launcher及全部worker继续运行；到期后的两次实机确认之间又完成了35步新更新。

- hour00–09均有四机真实PID／启动身份、GPU上下文、配置／282源码及当前500h缓存身份记录。
- 当前metrics共4337条，步号连续；全量loss公式、损失及梯度范数有限性通过。
- 窗口内完整检查点1500／1909／2000／2500／3000／3040／3500／4000均通过完整模型、Adam及31份RNG校验。最新完整检查点为step4000，位于原训练目录。
- 共享负载的间歇变化曾让性能监控两次触发保留进度恢复；分别从1909和3040完整状态接续，均完成至少3步真实更新再验收，未丢失已完成step。该判断已修正，19项保护检查及启动宽限期后的真实运行观察通过。详细经过保留在两个incident目录与SHARED_COMPUTE_POLICY.md。
- 数据、最新caption、损失、batch31、训练源码和执行策略均保持；node11 GPU0最终NVML核对没有本训练PID，其他任务未被本守护终止。
- 无待处理NEEDS_AGENT、恢复错误或观察器错误；最终验收不代表整轮训练已经结束，训练仍在原目录继续。

最终证据：[逐项验收](FINAL_ACCEPTED.json)、[可复核审计代码](final_audit.py)、reports/hour_00.json至hour_09.json、checkpoints/、incident_1794158710/、incident_1794173353/。
