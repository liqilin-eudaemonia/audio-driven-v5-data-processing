> 已于 2026-10-24 停止并冻结，当前任务已收紧为原始预裁框 ≥384 的 722,706 段，见 [新运行](../listen_raw_crop384_20261024T131300Z/README.md)。以下为旧范围历史记录。

# 三机全量原始 backchannel 打标

本轮范围为 accepted_audio_candidates/index.json 指定的两个 train 清单：1,419,124 段、2,335.944 小时。两个清单逐字节 SHA256 已校验；源切分、音质筛选、起始活动标记不变。没有纳入 dev/test、音质 review/reject 或其他数据集。

用户已确认使用 crop_proposals.listener。原画按 x/y/size 四舍五入到原生整数像素裁剪，再 bicubic 缩放到 512×512，8 fps，保留真实帧 PTS。音频只读取 normalized_audio.listener 的独立 WAV、原片段时间区间；不使用混音、speaker 转写、VAD 裁短或二次响度归一化。7,846 段无有效裁框，视觉单列 preparation_failed，音频和 ASR 仍执行。

Qwen3-VL-8B-Instruct 视觉、Qwen3-Omni-30B-A3B-Instruct 纯音频、Qwen3-ASR-1.7B＋ForcedAligner-0.6B 沿用已认可方案。整段＋3 秒窗/1 秒步长；整段与短窗输入完全相同时只计算一次，不是事件合并。每分支约 7,429,882 个请求，不能把滑窗请求数当作 pair 数。视觉和音频 prompt 保持“所有识别到的 backchannel＋每次起止时间＋开放描述”，不限定七类，不做事件归并，不生成训练准入。

## 当前进度与资源

- [实时状态](status.json)、[页面](http://127.0.0.1:8876/listen_benchmark_review/full_raw/)
- [冻结输入索引](queue/index.json)、[运行配置](config.json)、[模型资产检查](model_assets.json)
- [逐节点监控](nodes/)、[吞吐历史](throughput_history.jsonl)、[工作进程状态](workers/)

初始每节点 GPU 0–4 为视觉（batch 32）、5–6 为音频（batch 64）、7 为 ASR（batch 32）。GPU 内存目标为 90%；CPU 准备线程每视觉卡 16、音频和 ASR 每卡 8。已完成单卡验证并推广：视觉 batch 64、Omni 音频 batch 128、ASR batch 128；视觉和 Omni 启用 CUDA Graph，模型内存预留调为 85%，为编码器及图运行留出空间。control.json 记录实际批量及图加速试验，workers 中的 batch 是最近运行值。显存预留不等于 GPU 计算利用率；以吞吐和等待时间判断提速。

## 原始结果读取

- raw/visual/results/*.jsonl：视觉原始回答及验证结果。
- raw/audio/results/*.jsonl：Omni 音频原始回答及验证结果。
- raw/asr/results/*.jsonl：ASR 原始转写先写入 asr_transcript，再写带词时间的 result；二者不能重复计数。
- raw/asr_alignment_repairs/results.jsonl：启动阶段缺语言分词依赖的 73 条 alignment-only 补跑，原始转写不变，原失败保留。读取时以 request_id / source_recorded_at 对应补跑，不把它当成新样本。

每条 result 保存 sample_id、request_id、view、window_start_s/window_end_s、原始 source、裁框、listener 身份、起始活动、来源行哈希、输入证据及原始文本。模型时间是窗口局部秒；回到片段时间需加 window_start_s，回到源视频时间再加源片段起点。归并规则后续讨论，不在本轮实施。

同一 request_id 的重试记录以最后一次 result 为当前执行结果，旧原文不删除。batch_error 是执行诊断，不是无事件标签；preparation_failed、inference_failed、validation_failed、alignment_failed 均不能当成阴性。早期部分 validation_failed 来自窗口浮点相减与模型输出小数的极微小差异，原回答已经保存，不能据此断言感知失败，也不需要修改原文。

## 运行与恢复

代码快照在 code/，哈希在 code_manifest.json；改动前快照在 code_revisions/。每个分支有 1,024 个动态领取分片；文件锁保证每分片单写者，JSONL 持久化后再更新 state，重启恢复时忽略已完整记录的请求，并仅截去中断的末行。不同分支可独立完成；不要将前台终端退出当作任务停止。

停止领取并在当前批完成后退出：在此运行目录创建 STOP 文件。只停止监控：创建 STOP_MONITOR。不要 kill 其他任务。节点 supervisor 对自己启动的异常退出工作进程最多自动重试两次，反复失败单独报警；启动时检查目标 GPU 不被其他任务占用。

入口：scripts/index_listen_full_raw.py、scripts/run_listen_full_raw.py、scripts/supervise_listen_full_raw.py、scripts/monitor_listen_full_raw.py。全量索引 276 秒完成；6 项针对性测试通过，首条源媒体验证视觉 72 张 512×512 帧和 listener 独立音轨，证据见 preflight_first_source.json。

音频/ASR 工作进程完成可领取分片后，如仍有足够视觉工作，监控进程会启动对应 GPU 的视觉接续进程（再次验证空闲显存，不终止其他任务）；记录见 handoffs.json。

## 吞吐优化后的资源配比

首轮图加速稳定后，根据实际瓶颈改为 21 路视觉、2 路 Omni 音频、1 路 ASR。node12 GPU 5 和 node17 GPU 5 跑 Omni；node12 GPU 7 跑 ASR；其余 21 卡跑视觉。实际分配在 allocations.json，批量与图加速在 control.json；supervisor 验证原工作进程命令和进程组归属后交接，只操作本运行自己的进程。已提交原始结果不重复推理；中断的未提交批会重做。ASR/音频完成后再转视觉。

全量时长审计：1,419,124 段全部在约 2–11.95 秒范围，无超过 16 秒预算的样本（duration_audit.json）。
