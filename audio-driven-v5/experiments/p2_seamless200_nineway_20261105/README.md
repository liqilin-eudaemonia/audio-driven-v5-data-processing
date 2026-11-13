**已被三机24卡十组评测接替（2026-11-05）：** 本目录的node8自动等待已取消（`pause_for_other_job/CANCEL_AUTO_RESUME`），已完成产物保持不变并由[新运行](../p2_seamless200_tenway_24gpu_20261105/README.md)复用。不再在node8自动启动。本页下方暂停等待及九组运行说明为历史记录。

**当前已按用户要求暂停（2026-11-05）：** node8的本评测GPU worker已停止，8卡已释放；已完成结果保留。`pause_for_other_job`中的轻量等待进程先观察其他GPU任务开始，再等待其结束且8卡连续空闲120秒才自动接续。不是刚释放后立即重启；不停止其他任务，不重算已提交产物。取消自动接续可创建`pause_for_other_job/CANCEL_AUTO_RESUME`。资源查询异常会重新累计空闲时间。

# 新Seamless200：全部历史9组重测

用户2026-11-05指定重测历史全部9组，在新node8 `172.29.104.237:22` 的GPU0–7上运行。三机24卡组合损失训练继续，不占用它们的GPU。控制器已放到node8本机，避免每次GPU阶段及评分分片再次通过SSH握手；共享盘保留全部进度和结果。

数据为[最新标签200条诊断集](../seamless_quick200_latest_20261105/README.md)，合计19.99分钟、每个源pair一段；同一200条、相同seed与生成设置用于全部9组，共1800条输出。不是另选的ResponseNet200，也没有混用原Seamless6624条总量。

| 系统 | 训练条件／损失 | Qwen规划器 |
|---|---|---|
| av_only_6000 / av_only_8927 | 原始H，仅AV | 原始Qwen2.5-Omni-3B |
| av_event_6000 / av_event_8927 | 原始H，AV＋0.05 Event | 原始Qwen |
| base_feature_6000 / base_feature_8927 | 原始H，AV＋0.025视频Feature＋0.025音频Feature | 原始Qwen |
| sft_event_6000 / sft_event_8927 | 规划SFT H，AV＋0.05 Event | 对应规划SFT LoRA |
| sft_on_base_event_6000 | 原始H训练的Event6000，换规划器的交叉对照 | 同一规划SFT LoRA |

8份独立完整权重、9个系统；当前仍在训练的AV＋Event＋Feature及Whisper8000没有加入。各权重完整SHA、步数及代码来源已检查，见[PRECHECK.json](PRECHECK.json)、[checkpoint_hashes.json](checkpoint_hashes.json)与[protocol.json](protocol.json)。旧权重来源差异只接受此前GPU逐值重放已经审核过的精确文件／摘要变化，逐checkpoint写在`compatibility/`，没有关闭来源检查。

## 生成与提速

- 复用现有`p2_testsets_eval` GPU worker与原生采样，seed137、30步Euler、video CFG3／audio CFG7、512×512、25fps及各自完整pair时长；不降低采样步数或改CFG。
- 8卡按实际原生补帧数均衡分片，每卡25条。每卡1个常驻GPU进程，CPU四线程供数；Qwen/Gemma batch16（历史batch32曾OOM，未采用）。
- 基础4组、Feature2组、SFT Event与交叉3组按架构分成三个模型驻留阶段；同组多个checkpoint复用LTX/VAE模型加载。6个原始规划器系统共享同一套Qwen H／Gemma条件，3个SFT规划器系统共享另一套。两类缓存严格隔离。
- 推理只读取speaker音频＋listener首帧；最新GT caption仅用于选集和人工复核。输出为独立listener音视频，不混入speaker声音。
- `cache → render → decode`每阶段同时使用8卡。完成1800条生成后，原8卡自动切换评分，避免评分进程与LTX争抢同卡显存。

## 评测

沿用历史九项：PSNR、SSIM、FID、FVD、LPIPS、Emo-Acc、Emo-Div、ΔResp、UTMOSv2，并保留音频活动匹配、漏回应和误发声统计。不是本次新加GLARE评分；确保与此前历史组的既有主表口径一致。

采用已验收的GPU检脸／106点关键点、GPU表情、LPIPS/FID batch64、像素计算4线程；每GPU最多4个评分worker，共32路动态领取72个评分任务。GT特征由各系统共享；Seamless face-padding固定0。FID/FVD从同一200条全部分片特征重算，不平均分片标量。静音／短声音的UTMOS为N/A，保留有效覆盖率及所有组共同有效语音窗口。

## 状态、结果及恢复

- [status.json](status.json)：总阶段；`base_standard/`、`base_feature/`、`sft_event/`有各自GPU阶段状态和progress。
- `LAUNCHED_NODE8_LOCAL.json`、`controller.json`：node8本机进程身份；本机`/proc`不能用来判定远端PID存活。
- `generated/<系统>/seamless/<id>/generated.mp4`、`listener.wav`：生成结果。
- `metrics/<系统>/seamless/summary.json`：逐系统正式结果；最终`RESULTS.json`、`UTMOS_COMMON_WINDOWS.json`、`COMPLETE.json`。
- `GENERATION_COMPLETE.json`仅代表生成完成；正式完成还需`COMPLETE.json`及各项覆盖率。`partial`明确报告指标失败，不能冒充完整成功。

统一入口[`../../scripts/p2_seamless200_nineway.py`](../../scripts/p2_seamless200_nineway.py)。控制器持有queue锁，不得重复启动；各GPU阶段也有排他锁和完整产物标记。失败日志保留，已有模型条件／latent／视频按完成标记复用。首次调度出现间歇SSH错误，已在GPU推理开始前关闭旧的等待控制器并改成本机调度，没有停止正在生成的GPU worker。

用户请求停止时，在本目录放`STOP`阻止下一阶段；已运行阶段如需停止，先核对对应子目录的worker与锁。恢复前确认原控制器及子worker均已退出，避免重复提交。


## 后台守护与已完成输入验收

`watch`守护在node8本机运行，随本任务完成而退出，不另建定时汇报。它按PID/start_ticks核对本控制器，控制器异常退出时先等待尚存的本任务GPU worker或评分调度器，不启动重叠任务；旧阶段完整产物复用，最多重启3次。见`watch.json`及`WATCH_LAUNCHED_NODE8_LOCAL.json`。这不是之前已停止的20分钟巡检任务。

`BASE_CONDITIONING_VERIFIED.json`已逐条核对实际GPU生成的200条原始规划器条件：speaker音频、512×512首帧、source-only标记、未访问GT描述／listener未来媒体，以及Qwen/Gemma完整标记。连续3次GPU采样各卡99–100%，见`GPU_LIVE_SNAPSHOT.json`；这是该采样窗口的利用率，不代表包括加载／切换在内的全程均值。最终指标尚在生成后接续。


本轮累计CPU检查10项通过：新选集／来源与守护6项、原Seamless adapter4项；见`VALIDATION.json`。准备和输入验收不代表1800条生成／评分已完成，正式完成以运行目录`COMPLETE.json`为准。
