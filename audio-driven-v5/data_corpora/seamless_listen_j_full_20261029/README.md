# Seamless 原始全量 J v1 切分：四机 CPU 持续队列

用户于 2026-10-29 将任务扩大为：四台机器直接挂全量切分，并在约八小时内拿到结果。正式清单覆盖原始索引的 **64,622 条双人来源**，包括旧目录的 64,087 条和此前因身份重叠隔离的 535 条。后者在 manifest 中保留 `previously_excluded_identity_overlap` 与原始 split；所有结果仍 `training_admitted=false`，本任务不改变训练准入。

## 当前入口

- [实时状态与 ETA](status.json)：`completed`、`remaining`、`failed`、`enabled`、`active`、`eta`、`deadline_at_risk`、训练中位数与 P95、调度事件。
- [冻结清单](manifest.json)：来源、原始范围、代码指纹、输入、截止时间。
- [分片队列](queue.json)：四分片游标、正在处理项、失败项、完成数。
- [控制配置](../../configs/listen_pair_j_full_cpu_cluster.json)、[持续运行入口](../../scripts/run_listen_pair_j_full_cluster.py)、[全量输入适配](../../scripts/listen_pair_j_full_runtime.py)。
- [固定 J 规则交接](../../releases/listen_pair_segmentation_j_v1_20261029/README.md)、[最初低并发试跑与验证](../../experiments/pair_j_cluster_pilot_20261029/README.md)。

目标时间从此次全量指令起计，deadline 固定为 **2026-10-29 22:19 UTC**（北京时间 10 月 30 日 06:19）。这是完成目标，是否能按时完成须看持续吞吐；状态中的 ETA 按最近最多十分钟实际完成量估计，不将缓存命中率或空闲进程数当成完成速度。

## 并发与训练保护

首批 32 条的平均处理耗时为 37.52 秒／来源，音质尝试复用 776 次、新算 110 次。仅按旧目录估算，八小时约需 84 个双线程 worker；考虑准备时间、535 条缺少旧缓存的来源和余量，运行层以四机每机约 24 个为目标，最多每机 32 个。

四台同步从每机 4 个逐步升至 8、16、24 个，每 90 秒最多升一档；随后依据剩余量、剩余时间和实测每 worker 吞吐调整，保留 15% 的容量余量。每个 worker 主要计算 2 线程，BLAS 1 线程，CPU 亲和性限定在两个逻辑核，同机不同槽位使用不同的核，`nice=10`，不使用 GPU、不生成审核视频。全量不扫描所有媒体目录，仅按已有完整原始索引领取来源。

训练监控沿用中位数／P95：连续两个不重叠 40 步窗口任一指标相对基线超过约 5%，降低切分 worker 数；训练进度超过 60 秒停滞时暂停切分，恢复出步后从低并发重启。该保护可能与八小时时限冲突，因此状态明确标记预计逾期，不会伪报完成。若确实到时未完成，任务继续处理并记录超时事件。

## 运行、恢复和停止

从项目根目录运行：

```bash
python scripts/run_listen_pair_j_full_cluster.py run \
  --work data_corpora/seamless_listen_j_full_20261029

python scripts/run_listen_pair_j_full_cluster.py status \
  --work data_corpora/seamless_listen_j_full_20261029

python scripts/run_listen_pair_j_full_cluster.py verify \
  --work data_corpora/seamless_listen_j_full_20261029

python scripts/run_listen_pair_j_full_cluster.py stop \
  --work data_corpora/seamless_listen_j_full_20261029
```

同一 work 只允许一个 controller。恢复执行原 `run` 命令，保留原始截止时间、完成结果与逐候选检查点；旧停止请求不会阻止新 session。controller 失联超过 120 秒，worker 在安全候选边界退出。worker 崩溃后可以重新领取该来源，来源锁防止重复处理，完成回执带结果校验和。

本目录独立于旧生产结果。`source_inputs/<来源>.json` 在领取来源时冻结所用基线与归一化音频描述，缺少旧缓存时只在本目录生成必要音频；不预先并发扫读 64,622 条的全部结果。首批 32 条已完成 J 结果在校验后复用。

## 读取结果

每条来源的输出为 `results/<来源ID>/`：

- `initial.json`：音质筛选前完整候选。
- `qc/`：本轮逐段音质证据；复用整条已完成 J 来源时，来源记录见 `result.json.reused_completed_j_result`。
- `checkpoint.json`：尚未完成的候选、已处理区间、父子恢复关系。
- `result.json`：全部终端 `clips`、被替代父候选、音质路由、诊断和统计。
- `complete.json`：只有最终结果原子提交后才存在，不能只凭目录存在判断已完成。

`audio_final.pool == keep_candidate` 是自动通过，其他候选仍保留复核／排除状态。没有合法候选或原始素材缺少必要信息的来源也输出明确诊断；运行异常在 `errors/` 与 `queue.json.failed` 中明确列出，不计为成功完成。全量结果中的身份隔离信息应结合 manifest 的来源行读取。

原 J v1 的切分边界、音质阈值及局部恢复顺序保持不变。四机运行层的验收与实际启动状态见总索引及试跑记录。
