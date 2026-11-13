# 原权重全LoRA逐组件消融：四机28卡接续

2026-11-11用户授权继续训练三组。node12/node8/node4各0–7，node11仅0–3；node11 4–7保留。队列按去Event→去Feature→去Whisper运行。完整组不重训。

| 组 | 起点 | 目标 | 累计样本展示 |
|---|---|---|---|
| 去Event | 本组完整5818，恢复模型、Adam、28份RNG | 6573 | 186024 |
| 去Feature | 原预训练基座独立初始化 | 6643 | 186004 |
| 去Whisper | 原预训练基座独立初始化 | 6643 | 186004 |

完整组6000×31=186000作为预算参照。去Event前660步为batch31，此后batch28；新两组batch28、每卡一条。按样本量近似对齐，不声称更新次数／优化轨迹完全一致。新两组保留原31batch训练计划的样本及noise ordinal前缀，只按28重新切分；一轮尾部取原计划前284228项，正式本轮在186004项结束。

最终train+dev 284218段／500.001259h、最新caption、原始Qwen hidden、现有数值Event和特征缓存不变。原生LTX 48层1632处rank8 LoRA；启用Whisper时其两路cross-attention全参数。Event=0.05，启用Feature时视频、音频各0.025；关闭对应组件及其读取／优化器分组。

每500步和目标步完整保存。每组达到目标、完整SHA/模型/Adam/28RNG检查通过、launcher退出，才接下一组。现有全重计算、预取2、GPU Adam、512MiB FP64归约、单mlx5_0沿用。新两组启动原生最长样本两次更新并恢复；去Event沿用原恢复canary。不会停止其他任务。

独立适配入口 `runner.py`、`runtime.py`，原模型和训练文件不改。配置及真实32条缓存读取验收在各子目录preflight.json。runtime将原组件训练的Schedule导入替换为同前缀28batch版本，其余训练逻辑复用；配置严格归一化校验，仅允许batch和一轮步数变化。

- 队列状态：STATUS.json；异常：FAILURE.json；全部结束：COMPLETE.json。
- 每组通过：no_event_ACCEPTED.json、no_feature_ACCEPTED.json、no_whisper_ACCEPTED.json。
- 小时进度：hourly.jsonl，每3600秒记录一次。
- 去Event权重仍在 `../p2_ltx_all_lora_noevent28_batch28_20261109/`；另两组在本目录no_feature、no_whisper。
- 队列STOP阻止下一组启动；安全停止当前训练需在当前输出写STOP，原生循环保存后退出，不应直接kill。
- 当前run入口为一次性显式接续；故障后需核验最新完整检查点再显式恢复，不能盲目重跑以5818覆盖后续进度。

GPU正式推进验收另记LIVE_ACCEPTED.json；配置准备通过不等于GPU训练通过。

**2026-11-11九小时巡检：** UTC14:49:09–23:49:09，每小时四机完整实查、60秒轻量进度/存活检测；首轮28rank及318文件源码合同通过。独立恢复入口已补齐最新完整checkpoint选择，不修改本目录冻结源码；正式故障恢复须核验进程退出及模型/Adam/28RNG。[巡检与恢复说明](../p2_ablation28_guard9h_20261111/README.md)。

**九小时巡检已完成：** hour00–09全部通过，28rank正常、无额外同卡任务／持续降速／训练重启；5500完整模型／Adam／28RNG已复核，训练继续。[最终验收](../p2_ablation28_guard9h_20261111/FINAL_ACCEPTED.json)。

**2026-11-12用户测评暂停：** 去Whisper在1760步完整保存后退出，模型／Adam／28份RNG／SHA校验通过，四节点本轮训练worker及node进程全部退出；总队列和本组STOP保留，不自动接续。当前28卡上已有Seamless双seed测评进程，未干预。恢复应显式使用`no_whisper/step_001760.pt`，沿用本组配置及样本/noise顺序；不要重新初始化。原队列的目标步数断言退出属于本次提前停止，记录已归档为USER_STOP_EXPECTED_QUEUE_EXIT.json。[停止验收](USER_STOP_ACCEPTED.json)。

**2026-11-12已按用户新指令恢复去Whisper：** 从1760完整恢复模型／Adam／28RNG，原源码、配置、最终500h、caption、batch28及sample/noise计划保持；目标6643，每500步及终点保存。旧STOP已留档，四机28rank真实推进超过1780，loss及四组梯度正常。运行及新验收见[本轮恢复记录](../p2_no_whisper_resume1760_20261112/README.md)。
