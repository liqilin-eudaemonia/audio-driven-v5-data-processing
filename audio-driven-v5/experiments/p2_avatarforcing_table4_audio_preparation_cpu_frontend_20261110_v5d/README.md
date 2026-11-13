**最新执行入口：** GPU部分批次完成22／86后因显存余量不足等待；CPU四进程严格预检通过后，仅本会话自己的旧调度器和验收等待进程已退出。完整86路统一在[CPU整批入口](../p2_avatarforcing_table4_audio_preparation_cpu_20261110_v5e/README.md)重算，不混入旧GPU波形。旧22路与运行记录均保留。

# IIANet输入准备：CPU人脸前端＋GPU音源分离

仍以复现AF Table4自身8项为目标，当前未复现。最新完整结果为：[参考图预处理对照1/8](../p2_avatarforcing_table4_reference_mean_20261110_v3/CONTROLLED_REFERENCE_RESULTS.md)，[改官方256 GT后0/8](../p2_avatarforcing_table4_canonical_gt_20261110_v4/RESULTS.md)。本目录补论文5.1明确要求的目标说话人音源分离，尚无分离后的新AF全43条成绩。

**用户最新边界：音源分离做好后停止。** 本轮仅完成43对／86路分离及独立文件验收，不启动后续AF生成或评分。完成后查看[完成与停止记录](COMPLETED_AND_STOPPED.json)、[音轨清单](separated_audio_manifest.jsonl)、[独立验收](SEPARATION_INDEPENDENT_AUDIT.json)；这些文件只在全量真实完成并验收后产生。

## 当前执行方式

同一套官方IIANet Vox2、视觉编码器、MTCNN／SFD／FAN4及裁嘴函数；人脸前端在CPU运行，每进程8线程，音视频分离网络仍在node11物理GPU7。节点实测160核配额、当时负载约34；四进程最多32个人脸前端CPU线程。只改变执行设备来解决前端跨进程复现问题，不改模型参数或裁剪几何规则，不根据Table4分数选择设置。

先让四个独立进程同时处理同一段CNN speaker。嘴部ROI数组必须完全相同，原始音频最大差≤1e-6。通过后自动接续全部43对／86路；每进程GPU allocator5%，计入上下文并保留16GiB余量。未通过则停在预检，不能把等待或失败写成完成。

[冻结分离协议](PROTOCOL.json) · [四进程执行协议](PARALLEL_PROTOCOL.json) · [启动回执](LAUNCH.json) · [实际控制器状态](PARALLEL_RUNNER_STATUS.json) · [严格预检](PARALLEL_PARITY.json) · [全量进度](PARALLEL_STATUS.json) · [最终分离验收](FULL_SEPARATION_COMPLETE.json)。阶段文件按实际产生。

## 前序问题与修正证据

| 版本 | 已验证的事实 | 后续处理 |
|---|---|---|
| V5官方GPU前端 | 2对／4路功能预检通过；277个音频状态、137个视觉状态严格加载；重复音频前向误差0。独立进程嘴部／音频不同，误差0.000147987 | 批量没有启动，输出保留诊断 |
| V5b固定cuDNN benchmark | 两进程严格预检误差0，但四进程完整任务遇到关键点NPZ损坏；额外核查四上下文首条音频也有1.396e-5差异 | 停止批量，保留1路完成及其他部分输出 |
| V5c全局确定性＋安全写入 | CPU回归重现旧ZIP损坏，修正后含None关键点可正常保存、加载、插值；四GPU进程前端仍不完全一致 | 未启动全量，保留失败预检 |
| 本轮V5d | 相同网络／权重，MTCNN／SFD／FAN4改CPU，分离网络仍GPU；四进程严格预检已通过：嘴部ROI完全一致，三次对照原始音频最大差均为0 | 全量控制器已启动，当前等待GPU7可用显存；尚未完成任何全量音轨 |

2026-11-10启动记录：原全量控制器PID3852802在0／86且没有工作子进程时，因固定32GiB下限一直等待。已核验该控制器及父进程退出，保留替换记录。首次资源调度修正漏掉了helper内同一个32GiB下限，因此在开始处理前退出；失败日志保留。V2资源执行器同时覆盖内外资源下限，仍保留每进程5%硬分配上限、1GiB上下文预算和16GiB余量，不改网络、输入或数值设置。V2控制器PID3865851已实际开始分离，首路已产出。后续实际进度以`PARALLEL_STATUS.json`与最终验收文件为准。

[资源调度协议](RESOURCE_SCHEDULER_V2_PROTOCOL.json) · [实际启动回执](RESOURCE_SCHEDULER_V2_LAUNCH.json) · [最终验收控制器](finish_separation.py)。完成文件验收后进程退出，没有AF接续命令。

[原始权重及源码来源](../p2_avatarforcing_table4_audio_preparation_20261110_v5/README.md) · [V5b四进程及ZIP故障记录](../p2_avatarforcing_table4_audio_preparation_stable_20261110_v5b/FULL_RUN_FAILURE_AUDIT.json) · [NPZ回归](NPZ_REGRESSION.json) · [本轮设备变更](CHANGE_FROM_V5C.json)。

NPZ问题来自我新增的兼容写法：先调用保存函数再捕获ragged-array错误，会留下未关闭的旧ZIP对象覆盖新文件头。现改为在打开ZIP前完成`np.asarray`／object转换，回归测试同时重现旧错误并确认新文件可读取、插值。所有失败数据保留，未放宽一致性容差。

## 后续AF输入条件

公开AF的Wav2Vec2预处理会归一化音轨。两条功能预检中低音量listener输出，归一化后RMS仍为0.465／0.994，全零则为0。因此两种条件不能混报：①分离speaker＋自身音轨全零；②分离speaker＋分离listener。后者使用目标listener音轨及离线视觉分离，输入信息不同。两套各覆盖43条，仍固定V3的参考图、权重、seed20、NFE10、CFG2／1、512／25和导出；不进行种子／CFG扫描。AF未公开Table4实际自身音轨策略，不能把任一假设称为作者确认。

[在新Table4分数前固定的两套条件](NEXT_AF_AUDIO_CONDITIONS.json)保留为历史方案。按用户最新指示，本轮不实施这两套AF生成／评分；86路音源分离验收后停止，不能把音频准备当作已经有了新成绩。
