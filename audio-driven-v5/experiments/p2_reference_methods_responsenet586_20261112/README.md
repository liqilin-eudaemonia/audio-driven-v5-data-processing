**已按用户要求暂停（北京时间2026-11-13 14:57:18）：** node11 GPU7任务、自动恢复和ResponseNet后续守护均已停止，已有产物保留。资源策略当前不允许任何GPU，需用户重新授权才能恢复。[暂停进度与恢复说明](../p2_joint_qwen_ltx_test64_16gpu_20261113/NODE11_PAUSE_README.md)。以下运行安排为暂停前历史记录。

# ResponseNet完整test586：五方法后续评测

**最新资源安排（2026-11-13）：仅使用node11物理GPU7，GPU4禁止用于本轮及后续队列。** 以下旧GPU4/7记录为历史安排，当前以共享`RESOURCE_POLICY.json`、运行时设备检查和最新Monitor为准。

用户要求在Seamless完整test6624之后，继续测**ResponseNet完整test586条样本对**，仅使用node11物理GPU7。当前为后续队列，未启动新增386条的GPU推理。

## 范围、模型和指标

沿用L2L四套官方人物权重、ListenFormer本地ViCo完整重训best.pt、DyStream/Avatar Forcing/DualTalk官方权重及已验收的必要适配。L2L四套分别评测各586条，视频标注GPAvatar公开渲染；DualTalk报告原生/代理动作，RGB仍为N/A。没有生成listener音轨，音频指标N/A。

完整586清单来自双200运行时已审计的ResponseNet完整来源。前200条沿用历史固定子集及原8个25条批次，其余386条按固定ID排序分批；共24批（23×25＋最后11条），seed137。复用200条的输入、特征、身份、模型输出及指纹一致的评分分片。`REUSE_COMPLETE.json`记录来源和哈希；旧双200结果不改。

统一视频八项沿用PSNR、SSIM、LPIPS、FID、FVD、Emo-Acc、Emo-Div、ΔResp。**ResponseNet face-padding64、原GT帧率、GT按PIL bicubic变为512×512**。FID/FVD在完整586范围的有效特征上汇总，不平均分片分数。缺脸/追踪无结果保留N/A和覆盖，技术错误重试或报错。原生DECA动作诊断单独列出。

## 后续Monitor接续

`WAIT_FOR.json`指定等待[Seamless完整6624](../p2_reference_methods_seamless6624_20261112/README.md)根目录`COMPLETE.json`。已注册在该目录的`FOLLOWUPS.json`，由独立CPU守护`followup_watch.py`持续检查；前序完整结束后自动启动本目录`monitor.py`，并监督意外退出后的恢复。无需重复手工提交GPU任务。

后续守护启动记录：
`../p2_vico_additional_baselines_20261107/jobs/reference_responsenet586_20261112_followup/`。

本目录Monitor也会核验前序依赖，因此提前手工启动仍只等待，不抢占GPU。原模型、批次与完成回执支持断点续跑。

## GPU资源规则

`RESOURCE_POLICY.json`链接到Seamless全量任务的同一份策略：

- 只使用node11 GPU7，GPU4禁止使用；连续无其他GPU作业180秒后准入。
- 每阶段模型加载一次后常驻，连续处理待办；输入未就绪时在同一进程内等待，不按小批次退出、不定时重启。
- GPU7预留40GiB，最多12个不同阶段worker，启动仍受实际显存约束。
- 前台任务出现时SIGSTOP暂停计算并保留模型；空闲稳定后SIGCONT恢复同一进程。暂停仍占模型显存。
- 单个常驻评分服务跨分片复用原Networks／Faces实例；通过真实片段与历史结果一致性检查后继续正式评分。
- 仅进程技术失败才重试／重新加载；已完成产物和原25条固定批次继续复用。

本轮仍等待Seamless6624全部结束。常驻实现与检查说明见[前序运行文档](../p2_reference_methods_seamless6624_20261112/README.md)。

## 看结果、状态和恢复

- [完整586结果与进度](RESULTS.md)、[CSV](image_metrics.csv)、[状态](status.json)。准备/缓存200条不代表完整586已测完。
- [后续队列状态](../p2_reference_methods_seamless6624_20261112/FOLLOWUPS_STATUS.json)。
- `protocol.json`、`RUN_SCOPE.json`、`BATCHES.json`：冻结清单与采样批次。
- `MODEL_MANIFEST.json`、`MODEL_VERIFIED.json`、`PREFLIGHT.json`：权重及输入核验。
- `monitor/`、`processes/`：未来实际运行时的日志、PID与退出码。
- `results/*_responsenet.json`：全586视频结果；`*_responsenet_native.json`：原生动作诊断。
- 只有根目录`COMPLETE.json`生成，才代表本轮全部计划结果完成。

模型原始推理入口见[方法交接索引](../../related/method_handoffs/README.md)，历史双200结果见[固定子集表](../p2_reference_methods_dualtest_20261112/RESULTS.md)。
