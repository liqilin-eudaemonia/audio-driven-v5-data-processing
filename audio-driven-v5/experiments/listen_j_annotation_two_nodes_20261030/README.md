# J 全量打标：node17＋node8，16卡

> **2026-10-30后续状态：** 旧VLM分支已按用户授权停止继续领取，已完成结果保留；CPU点头分支继续运行。GPU转入[J Omni自然语言接续流水线](../listen_j_omni_20261030/README.md)。本文下方记录原16卡VLM部署过程。

2026-10-30，用户授权清空 node8 并与 node17 联合打标。原填写地址 `1172.29.104.233` 为笔误，实际成功连接及核验地址为 **172.29.104.233**，主机名 `dedicated-developjob-zz-dev-machine-1-l1o08`。两机均8×H200、CPU quota160核、内存2TiB。

## 范围与结果

| 节点 | 独立工作目录 | 来源数 | 分片段数 |
|---|---|---:|---:|
| node17 / 172.29.15.91 | [原工作目录](../listen_j_annotation_node17_20261029/) | 34,331 | 649,665（含此前已完成） |
| node8 / 172.29.104.233 | [新工作目录](../listen_j_annotation_node8_20261030/) | 26,538 | 502,140 |
| 合计 | 两节点结果合并读取 | 60,869 | 1,151,805 |

两分片来源互斥、并集与原范围一致：[范围清单](nodes.json)、[分片审计](partition_audit.json)。**node17原来的 ingest_complete.json 是导入历史总量，不能再当作单节点当前分片分母。** 上游194个失败来源不在上述可打标范围内，沿用原记录。

各节点工作目录下：

- `vlm/results/<hash前2位>/<sample_id>.json`：视觉原始结果。
- `nod/results/<hash前2位>/<sample_id>.json`：GLARE recall点头候选。
- `{vlm,nod}/completed/<source_id>.json`：来源完成凭证。
- `queues/{vlm,nod}/state.json`：各分支进度及失败队列。
- `workers/`、`supervisor.json`、`watchdog.json`：运行状态。

两节点完成数可以按分支相加，因为来源互斥；视觉和点头两分支不能相加当作完整片段数。未来整理结果应读取两个结果根目录，以 `nodes.json` 为入口，不能只读取原node17目录。

## 已验证加速配置

沿用[固定加速交接](../../related/P2_打标加速固定配置与多机复用交接_20261030.md)：每卡batch512、调度token32768、帧预算16384、每组50段、CPU解码16进程、2个renderer、一个CPU预取组、decode-only CUDA Graph。每节点112共享解码槽、80个CPU点头worker，GPU显存比例0.80。

node8八卡统一2组连续调度、graph上限128。node17保持原进程及其有效配置，GPU7原4组/256图配置仅为历史遗留，不推广。RGB复用及hybrid硬解均未启用。模型、观察式C prompt、8fps、512裁框、整段＋3秒/1秒窗口、输出4096上限与25fps点头规格不变。

## 迁移方式及恢复

先确认源导入已完成，同时持有 ingest锁和视觉/点头两队列状态锁。取两分支已读偏移的最大值，保留此前所有字节和来源；保护inflight/failed来源；只迁移两分支都未领取的后缀来源。按片段数平衡分配，保留来源内部完整性。

node17不重启、不重算、不复制已完成结果；其队列偏移仍有效。node8拥有独立队列、锁、临时缓存与守护状态，只读复用 `jobs/` 的源规格。没有跨节点使用PID回收或共享一个工作目录。

- [事务凭证](partition.json)：原始、迁移后两份journal SHA及迁移来源列表。
- `original_ready.jsonl`、`original_queue_states.json`：迁移前证据。
- [迁移实现](../../src/listen_benchmark/j_queue_partition.py)：幂等提交；未提交事务若已经出现新领取则拒绝套用旧计划。
- [代码快照与SHA](code_snapshot/SHA256.json)、`config_snapshot/`：本次部署代码与配置。

**生产继续后不要直接恢复旧ready.jsonl，否则会重复分发node8来源。** 回迁需先停止目标分片领取并核对完成凭证，不能覆盖正在变化的队列。

启动器保留精确主机名保护，现在支持 `config.json.expected_hostname`；未设置时仍仅允许原node17。node8先显式初始化supervisor，再由watchdog负责恢复。启动前已停止用户指定旧GPU任务及其 `recovery.py` 自动恢复进程；记录见 [旧任务](node8_preemption.json)、[恢复进程](node8_recovery_stopped.json)，没有删除训练文件。

node8点头环境由node17的 `/tmp/glare-nod-pilot-env` 原样复制并验证：Python3.10、MediaPipe0.10.21、PyAV16.0.1、numpy1.26.4。机器重建后须恢复该环境，不能忽略缺失环境导致点头分支不启动。

## 查看与验收

只读检查两机记录的进程身份、GPU状态、队列计数：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python experiments/listen_j_annotation_two_nodes_20261030/check_status.py
```

输出保存至 [status.json](status.json)。该工具不扫描结果树、不创建任务、不优化参数，也不恢复已停用的定时消息。

部署前27项队列、分片、主机保护和执行配置测试通过；分片真实范围审计通过。运行验收另存node8的 `monitor_snapshots/`，须区分输入/来源检查与响应结构有效率，不能据结构通过宣称语义准确。goal保持暂停；后台两机任务及各自故障恢复持续运行。


启动验收：node8 [首100段视觉输入/来源审计](../listen_j_annotation_node8_20261030/monitor_snapshots/initial_100_audit.json)通过，432次窗口响应全部结构有效、零截断；[20段点头来源/裁框/25fps检查](../listen_j_annotation_node8_20261030/monitor_snapshots/initial_nod20_audit.json)通过。node17 [88个原工作进程身份保持不变](node17_preserved.json)。两个节点各88个worker、监督及守护已检查存活；全量仍在处理。启动阶段尚未测得两机稳定吞吐，不将单机速度直接当作两机实测。
