# 全LoRA6000／6500：ResponseNet完整586条

用户2026-11-11要求：把当前ViCo使用的两个完整模型，也在ResponseNet全测试集上评估。

**2026-11-12 00:01 UTC快照：6000已生成／导出70/586条，6500排队，尚无全量分数。** 586条条件全部齐全；23:27 UTC已从分块加载安全交回原常驻模型，GPU7利用率100%，资源许可时并行原VAE解码。九小时巡检到期，测评队列继续运行。完成以本目录`COMPLETE.json`和两组`FINAL_AUDIT.json`为准。

## 权重

| 步数 | 原始完整模型 | SHA256 |
|---|---|---|
| 6000 | `../p2_ltx_all_attention_ffn_lora31_20261108/step_006000.pt` | `3c2c78e8b1c81e3cf84efdc565b16a28565c0e923cacf8e902c3adcd6232eea6` |
| 6500 | `../p2_ltx_all_lora_full28_batch28_20261109/step_006500.pt` | `4bf7a44137130488fd60c0c2c5a16eae495dc6ce4fc8c2024cee212de2232a52` |

均为attention／FFN全LoRA＋Event＋双路Feature＋Whisper组。6500来自6428步后batch31→28的连续训练，迁移差异保留在协议中。直接复用已经逐张量验证的推理导出，原始完整权重保持。

## 全量范围与推理设置

- 沿用[既有ResponseNet配对入口](../../evaluation/responsenet_speaker_j/HANDOFF.md)：官方140个来源中135个有配对片段，共586条、73.201分钟；五个没有配对片段的来源见原始审计。本轮不是随机200条，也不改为140条整段视频。
- 冻结清单为`OmniResponse/evaluation/responsenet_pairs/prepared_20261102/references.jsonl`，SHA256为`65dbb308d04505c276fa64f84e89f0cc05013fb7919fd79b6da6907669f27be3`。
- 514条采用官方词时间，72条采用本地Whisper词时间；全586都保留，72条原有`paper_comparable=false`标记不改，不能据此宣称完全等同论文数据处理。
- 两个模型各586条，共1172个生成视频和对应无损listener音轨。SFT2752规划、H、Gemma和speaker Whisper完全共享；复用200条已验证缓存，补齐386条。
- 输入只有speaker音频和listener首帧；不输入listener未来画面、音频、ASR或GT描述。CPU检查已经逐条核验586条speaker波形、首帧和复用缓存。
- 生成512×512，按每条原帧率／帧数导出；帧率分布为29.97×429、30×71、24×49、23.98×23、25×14。VAE临时补帧在导出时移除。GT由640×640按原框架PIL bicubic缩至512×512；脸部padding64只用于检测器，不改变像素指标图像。
- seed137，Euler30步，video CFG3／audio CFG7，原生Whisper数值策略与采样逻辑。缺失缓存采用planner batch4、Gemma batch8、Whisper batch8以适应同卡资源，条件仍为同一SFT2752。

## 评分与加速

沿用目前随机200条的评分代码：PSNR、SSIM、FID、FVD、LPIPS、Emo-Acc、Emo-Div、ΔResp，另保留音频活动比较、UTMOS有效窗口／N/A覆盖。旧数据说明中的“音质模型不跑”为早期设置；本轮跟随后续正式200条队列的九项输出和音频活动规则，不新增GLARE、SyncNet或ViCo指标。

八个逻辑评分分片，最多两个评分进程同用GPU7；CPU有界预取、GT特征缓存及原生batch64保留。FID／FVD合并全586条特征后重新计算，不能平均分片FID／FVD。情绪、运动、音质支持不足时保留原N/A及覆盖数，不删样本凑满分。

模型使用mmap导出加载、eval／bf16、常驻VAE encoder、DiT与解码分开。显存足够时生成与一路原生VAE解码并行，生成结束后最多两路解码；基于实时空闲显存准入，不停止其他任务。所有GPU命令固定`CUDA_VISIBLE_DEVICES=7`并核对物理GPU UUID。

16:20 UTC资源准入v2：复用既有全LoRA Seamless200队列已使用的64GiB空闲生成准入和`min(96GiB, free−4GiB)`进程上限；若实际OOM则保留断点、改等80GiB空闲。原方案与协议存入`revisions/20261111_resource_admission_v2/`，仅改显存调度，不改样本、模型、规划或指标，避免沿用较长ViCo片段的80GiB门槛而无谓等待。当前约23GB空闲仍不足以启动完整模型。

### 显存受限阶段：官方分块加载，20GiB进程上限

保留原生BF16／FP32各权重的dtype和数值，把CPU固定内存中的模型块逐块拷到两个GPU缓冲槽，继续执行原forward。不量化、不融合LoRA，不在CPU上做模型矩阵计算；checkpoint、SFT、采样、seed和评分数学保持。原权重严格加载完成后才安装加载钩子。使用已有`ltx_core.block_streaming`实现，全部CPU权重拷贝逐值检查，额外源码绑定在[当前执行协议](STREAM_PROTOCOL.json)。

[6000短样本](STREAM_RENDER_SMOKE.json)与[6500短样本](STREAM_RENDER6500_SMOKE.json)分别对照同权重已完成ViCo产物，音频／视频latent均逐值相同，最大差0；短样本GPU峰值约4.1GiB。该结果验证这两个控制样本，不代表已经验证所有586条。

Gemma沿原生BF16前向，批次降到2以适应显存；89条与已有SFT缓存的prompt逐字相同，验证同一Gemma资产后复用，来源及payload SHA见`GEMMA_EXACT_PROMPT_REUSE.json`，其余297条已补齐。对历史batch16缓存的两条检查中，音频条件／mask逐值相同，视频条件存在少量浮点差异（平均绝对差约0.00035／0.00048，最大0.5625／2.0），完整记录见`STREAM_GEMMA_SMOKE.json`，不宣称整个条件链逐值相同。两模型共享完全同一份586条最终条件。

每次加载模型最多生成8条，再用原VAE和编码器解码这些输出，按原协议保留512×512／原fps／原时段。每组586条后绑定清单、逐个运行8个视频／音频评分分片，全量汇总FID／FVD。GPU任务使用20GiB上限；显存达到64GiB空闲时，在安全批次边界交回原常驻模型队列以提速，已有latent／视频保留。

21:02 UTC记录：6000已产出8份latent，首批完成后解码。最大帧数361的真实样本GPU峰值7,661MiB；后续资源profilev3试用两路各9.5GiB上限、总预算19GiB的原生生成，使用独立进度路径及逐样本锁，运行新样本而不重复已完成项。失败或生产吞吐明显变差时自动回退单路。每路最多8条，随后统一解码。GPU模型不重启；当前控制器会完成在途生成和解码后，通过共享文件锁交给修订控制器，交接证据见`STREAM_FILE_HANDOFF*.json`，不依赖SSH观察成功与否。旧代码和协议在`stream_validation/v3_parallel/`保留。

当前入口：[分块调度](../../scripts/p2_rn586_streaming_gpu7_20261111.py)与[官方缓冲器适配](../../src/p2_eval_block_streaming_20261111.py)。`STREAM_STOP`可停止分块调度；本目录临时`STOP`的owner标记仅用于防止旧控制器同时启动，正常交回旧队列或全部完成后自动移除。用户显式停止和九小时守护的停止标记始终优先。

## 进度、代码与结果

23:27 UTC，空闲显存达到原队列准入条件，分块控制器在批次生成与解码结束后自动交回常驻模型队列，证据见`STREAM_NATIVE_HANDOFF.json`。保留此前58条视频和全部latent，原渲染进程继续从断点生成；23:46 UTC检查为64条latent、58条视频，GPU7利用率100%，其余显存由其他任务占用。当前空闲显存不足以安全并行VAE，解码随调度条件继续。6500仍排队，尚无全量评分。

- [进度](status.json)、[启动PID](LAUNCH.json)、[控制器PID](CONTROLLER.json)、[控制器日志](controller.log)。
- 当前已交回原常驻模型控制器，见`CONTROLLER.json`、`STREAM_NATIVE_HANDOFF.json`和每组`processes/`、`progress/render/00.json`。`STREAM_CONTROLLER.json`及`streaming_logs/`保留此前分块阶段记录，不能据此重启另一路生成。
- [已导出视频／GT／speaker／音轨快照](VIDEOS_PROGRESS.md)。此表是阶段快照，不代表全586已完成。
- 22:06 UTC：6000已有26条视频；双进程首批16条耗时2640.8秒，平均165.05秒／条，前一批单进程中位182.78秒／条（不同生产样本、同最大帧数范围）。两条真实视频的原视频八项及音频入口验证通过，执行错误0，见`STREAM_METRIC_SMOKE_COMPLETE.json`与`step_006000/metric_smoke/`，这些不是全测试集分数。后续吞吐比较固定使用单进程参考，避免把并发时增大的单任务延迟当成单进程基准。
- [全量清单准备](PREPARED.json)、[CPU逐条核验](CPU_ACCEPTANCE.json)。
- 低显存预提取：`SMALL_CONDITIONS_LAUNCH.json`、`SMALL_CONDITIONS_POLICY.json`、`small_conditions.log`；使用原Whisper FP32／batch8及SFT2752／batch4，不改模型／提示／H提取数学。共享目录`WHISPER_COMPLETE.json`／`SFT2752_COMPLETE.json`分别表示该项586条齐全，Gemma由原队列后续补齐。[独立适配器](../../scripts/p2_rn586_small_memory_conditions_20261111.py)。
- [执行入口](../../scripts/p2_responsenet_full586_all_lora_20261111.py)、[CPU核验入口](../../scripts/review_responsenet_full586_all_lora_20261111.py)。
- 共享条件：`conditioning/`；两组协议和输出：`step_006000/`、`step_006500/`。
- 视频位置：`step_00XXXX/generated/all_loraXXXX_sft2752/responsenet/<pair_id>/generated.mp4`；同目录`listener.wav`。
- 每组评分：`step_00XXXX/RESULTS.json`；每组完成核验：`step_00XXXX/FINAL_AUDIT.json`。
- 两组最终汇总：`RESULTS.json`；全部完成标记：`COMPLETE.json`。尚未生成的结果文件不可视作已完成。

控制器以文件锁和逐样本提交恢复，后台自动等待ViCo6500的完成标记及其登记进程退出，再接GPU7。发生执行错误写入`status.json`并保留日志；不把失败或缺失项记成有效分数。

用户最新要求已改为**每小时巡检，持续九小时**，取代此前每20分钟。已启动独立存活守护，异常退出且无存活子进程时按断点恢复，仍在运行的模型不盲目重启。[九小时窗口、检查与修复证据](../p2_vico_responsenet_watch9h_20261111/README.md)。
