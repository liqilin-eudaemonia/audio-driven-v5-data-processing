# Seamless：三机 19 卡打标，与降低负载的画质筛查并行

**最新：本机GPU停用后，剩余GPU标签已交给node17＋node8两机16卡。** 本机CPU点头继续运行，最终按分支归属关联标签。完整待打标范围没有缩减，已有标签复用，见 [两机接管完整剩余GPU队列](local_vlm_delegation_20261030/HANDOFF.md)。下文24卡和19卡配置为历史阶段。

**后续扩容覆盖本页的卡数、来源分片与 CPU 配额：** 用户确认释放后，本机 GPU4～7 已加入，当前三机各八卡，共 24 卡。本机点头 worker80、共享解码槽112，另有32个 CPU 画质辅助 worker。仅迁移未领取来源后，node17／node8／node12 的分片分别为488,281／359,479／304,045段，剩余工作量各约28.5万段。见 [24 卡部署、分片与恢复](expansion_24/HANDOFF.md)；以下19卡部署是原始阶段记录。

2026-10-30。用户将流程改为：减少 CPU 画质筛查占用，同时恢复原 GPU 打标；最后按画质结果选取保留数据。另授权本机 node12 的 0～2 卡加入，合计 19 卡。未恢复 goal 自动续行或定时消息。

## 当前工作目录与分片

| 节点 | GPU | 标注工作目录 | 来源数 | 片段数（含已有完成项） |
|---|---|---|---:|---:|
| node17 / 172.29.15.91 | 0–7 | `../listen_j_annotation_node17_20261029` | 32,846 | 577,922 |
| node8 / 172.29.104.233 | 0–7 | `../listen_j_annotation_node8_20261030` | 25,070 | 431,029 |
| node12 / 172.29.252.37，本机 | 0–2 | `../listen_j_annotation_node12_20261030` | 2,953 | 142,854 |
| 总计 | 19 卡 | [nodes.json](nodes.json) | 60,869 | 1,151,805 |

迁移仅来自两机视觉和点头分支均未领取的来源。保护两分支最大游标前的字节前缀、inflight、failed 和已有完成记录。分别从两机未领取池按约 3/19 片段容量迁移：71,743＋71,111 段。三分片互斥且并集不变，见 [partition_audit.json](partition_audit.json)。两个事务目录 `transfer_from_node17/`、`transfer_from_node8/` 保存旧 journal、旧队列状态和提交凭证；不能在继续运行后直接覆盖回旧 journal。

node12 `jobs/` 只读链接到原始完整 job 存储，`vlm/results`、`nod/results` 使用本机新工作目录中的独立结果。旧两机结果与完成凭证保留，继续使用原队列断点；已完成 pair 不重复打标。

## 原加速配置保持

沿用已验证每卡 batch512、token32768、frame budget16384、每组50段、CPU进程解码16路、两个renderer、连续补入、输入预处理重叠、decode-only CUDA Graph、显存比例0.80。Qwen模型、观察式C prompt、8 fps、512裁框、JPEG85、整段＋3秒/1秒窗口、4096输出上限保持。

- node17、node8 原 selected/config/resource_tuning 的加速字段未改变；node17 原有个别逐卡覆盖也保留。
- node12 使用 node8 已验证配置，仅限定 `gpu_ids=[0,1,2]`；节点级共享解码槽按卡数设为42、点头 worker30。每卡模型与执行参数相同。
- supervisor 新增显式 GPU 列表，默认仍为八卡；本机只管理 0～2 卡。其他 GPU 任务未停止或重配。
- 本机点头独立环境 `/tmp/listen-node12-nod-env`：PyAV16.0.1、NumPy1.26.4、MediaPipe0.10.21、SoundFile0.13.1、SciPy1.15.3、Pillow12.3.0，与原点头环境一致。本机最初缺 SoundFile 的384个来源失败及30个未完成来源已保留记录、补齐依赖、通过真实片段验证后重新入队；不是画质失败，不剔除数据。见本机 `nod/dependency_recovery.json`。

## 画质筛查改为并行，最后过滤

画质工作仍在 `data_corpora/seamless_listen_j_visual_quality_v1_20261030/{node17,node8}`，范围仍覆盖全部1,151,805段，与标注来源重新分片互不混淆。

- 每机32个画质worker，nice=10；保留原5fps、连续解码复用、检测器单线程和已确认裁框规则。
- 原串行流程的画质守护和controller已经停止，`STOP` 标记继续保留，防止筛后再启动一套重复标注。
- 新入口：`scripts/listen_j_quality_parallel.py`；`parallel_watchdog.json` 与 `controller.json` 记录并行状态。模型检测代码和已完成画质缓存不变。
- 原GPU标注使用原裁框继续进行；画质通过清单应用新裁框并保留原框。按照用户“已完成标签不重算”的要求，所有旧框标注保留来源，后续验收可按裁框差异单列。
- 画质完成后自动导出保留／排除／待复核清单；不启动新标注队列。
- 三机标注队列结束后，按最新三机来源归属汇总每个画质通过pair的视觉／点头结果路径，保存为各画质节点目录的 `retained_annotation_index.jsonl`。`parallel_complete.json` 记录完整／缺失标签和裁框差异计数。存在失败或缺失时明确单列，不把文件存在当作语义质量通过，仍 `training_admitted=false`。
- 画质根目录的 `annotation_nodes.json` 指向当前三机标注范围。筛后历史导出的 `listen_j_annotation_quality_*` 目录不会启动任务；三机最终标签应以本页 nodes 与最终索引读取。

## 状态与恢复

[最近三机状态](status.json)；执行一次只读更新：

```bash
python experiments/listen_j_annotation_three_nodes_20261030/check_status.py
python data_corpora/seamless_listen_j_visual_quality_v1_20261030/check_status.py
```

标注完成数按视觉、点头分支分别相加，不能相加成为完整pair数。画质独立计数。

恢复某节点标注时，在对应节点运行原入口，使用该节点工作目录：

```bash
/private/software/conda/envs/vllm/bin/python scripts/listen_j_annotation_watchdog.py \
  --work experiments/listen_j_annotation_node12_20261030 \
  --upstream data_corpora/seamless_listen_j_full_20261029
```

恢复画质并行守护时，在node17/node8运行（另一个节点替换路径）：

```bash
/tmp/listen-j-quality-env/bin/python scripts/listen_j_quality_parallel.py watchdog \
  --work data_corpora/seamless_listen_j_visual_quality_v1_20261030/node17
```

不要恢复旧的 `listen_j_quality_pipeline.py watchdog` 串行调度，也不要启动筛后导出目录的标注守护，否则会与原任务重复。
