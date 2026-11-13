# 四机32卡稳定训练九小时守护

用户要求：稳定优先；优化不稳定就回到原方案。当前固定all全重计算，不重试three_quarters。训练目录：`../p2_ltx_clip_train32_after_gpu67_20261106`。

守护窗口2026-11-06 15:50:31 UTC至2026-11-07 00:50:31 UTC（北京时间11月6日23:50:31至11月7日08:50:31）。到期仅结束守护，不停止训练。

每15秒检查实际步数和有限损失，每半小时保存四节点GPU/进程/速度完整记录。按调度中的片段时长分组比较速度，避免长片段误报。普通更新120秒无进展触发恢复；明确checkpoint保存和冷启动允许有界宽限。严重持续降速也触发诊断与恢复；三次/30分钟失败后请求agent检查，避免无限重启。

自动恢复只终止命令、配置路径及rank环境都匹配的本任务worker；所有选定GPU释放后才从最新完整checkpoint重启。保持32rank、batch32、每300步保存；用户STOP始终优先，不覆盖。原训练源码保持不变。

- `GUARDIAN.json`：守护PID和启动标识。
- `protocol.json`：授权范围、期限、源码/配置摘要。
- `status.json`：当前检查状态与速度比较。
- `reports/`：每半小时完整巡检。
- `events.jsonl`：故障/恢复/降速记录。
- `incident_*/`：故障进程快照及重启命令日志。
- `NEEDS_AGENT.json`：需要进一步诊断，不能视为一切正常。
- `STOP`：仅停止守护；训练停止须另行明确操作。

[回退后实际训练验收](../p2_ltx_clip_train32_after_gpu67_20261106/RESTORE_ALL_VERIFIED.json)。


此前的资源争用记录见`RESOURCE_CONTENTION_REPORT.json`及`GPU7_RESOURCE_TIMELINE.jsonl`：node11 GPU7曾被另一项目共享，明显拖慢rank31；最新21:12至23:42的六次完整快照均未发现外部GPU进程，近期步时接近原all基线。未收到暂停外部项目的授权，也未向外部进程发信号。若再次共享，仅捕获诊断，不因共享反复重启健康训练；真实停滞仍恢复本训练，恢复前须有至少108GiB可用显存。实时情况以`status.json`和`reports/`为准。


600步周期保存已全量验收：[CHECKPOINT600_VERIFIED.json](CHECKPOINT600_VERIFIED.json)。37,046,673,651字节完整文件SHA匹配；模型及6,936份优化器张量全部有限，五组Adam步数均为600，32份RNG，运行身份精确匹配；CPU验证无CUDA初始化。保存后训练继续推进。


巡检查询已改为异步后台执行，GPU进程查询先筛选NVML PID再读取/proc，避免节点慢查询阻塞15秒心跳。实测完整四节点32rank查询约2.2秒，见[READONLY_PROBE_ASYNC_VERIFIED.json](READONLY_PROBE_ASYNC_VERIFIED.json)。训练源码不变。

恢复查询检查已补强：GPU查询错误不视为空闲；必须确认每节点8张物理卡；恢复前要求每卡可用显存至少108GiB（当前稳定all进程实测约101GiB，额外预留约7GiB）。每进程134GiB分配器上限不变。[验证](RECOVERY_QUERY_VALIDATION.json)。

900步周期保存已完整验收：[CHECKPOINT900_VERIFIED.json](CHECKPOINT900_VERIFIED.json)。文件SHA、所有模型/Adam张量、32份RNG结构/CPU可恢复性、运行身份均通过；五组Adam计数全部为900。保存后训练继续推进，下一周期1200。

1200步周期保存已全量验收：[CHECKPOINT1200_VERIFIED.json](CHECKPOINT1200_VERIFIED.json)。SHA、模型和全部Adam张量、32份RNG结构/CPU可恢复性、运行身份均通过；五组Adam计数均为1200。保存后训练继续推进，下一周期1500。

保存宽限已细分：写入未完成的周期检查点保留600秒宽限，完成标记已存在后恢复正常120秒停滞阈值；冷启动宽限独立保留。守护重启延续既有半小时巡检时间表。[验证](CHECKPOINT_GRACE_VALIDATION.json)。


[8小时阶段验收](EIGHT_HOUR_AUDIT.json)：16个半小时窗口均有完整32rank快照；原all恢复后无再次训练恢复事件，已记录的loss均有限。TensorBoard入口及转发方式见[本轮训练说明](../p2_ltx_clip_train32_after_gpu67_20261106/README.md#tensorboard本轮)。
