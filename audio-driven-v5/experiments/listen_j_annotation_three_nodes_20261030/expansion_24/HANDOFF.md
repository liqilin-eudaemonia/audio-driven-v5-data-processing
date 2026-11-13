# 三机 24 卡打标与 CPU 画质筛查

用户授权：本机 4～7 卡释放后，将原已验证打标方案扩至三机各 8 卡；保持机器有任务处理。原任务仍运行时不终止或抢占。

2026-10-30 先启用本机 GPU3 至20卡。用户随后暂停等待 Goal，原打标和画质进程保持运行。用户确认 GPU4～7 已释放后，再次实测各卡无计算进程、显存仅1MiB，已启用这四卡：node17、node8、node12 各8卡，共24卡。见 [启用凭证](full8_activation.json)。继续沿用原生产队列与故障恢复守护，无新增定时消息或 Goal 自动等待。

## 自动接入

入口 `scripts/listen_j_expand_local_capacity.py`，记录 [capacity.json](capacity.json)，启动凭证 [launch.json](launch.json)，日志 `capacity.log`。每 15 秒检查本机 GPU；连续两次确认无计算进程且显存低于 1500 MiB，写配置前再复查，随后加入原 annotation supervisor。不会杀死其他任务或改动已有 GPU 的执行配置。

新增卡直接复用 node8 固定快照的逐卡配置：batch512、token32768、process decode、两个 renderer、两组连续补入、原 CUDA Graph 与显存比例。共享解码槽按每卡 14 个扩至 112；点头 worker 按每卡 10 个扩至 80。scope 文件 GPU 列表同步更新，pair 总量和现有标签不变。已配置 GPU 的 worker 恢复仍由原 watchdog/supervisor 负责。

## 本机 CPU 画质工作

本机增加 32 个 nice=10 worker，分别协助原 node17、node8 画质队列各 16 个。使用各队列专用槽号 1000～1015，原两机仍使用 0～31。合计三机 96 个画质 worker。

共享文件系统的锁在 node17、node8 上实测阻塞成功，见 [cross_host_lock_validation.json](cross_host_lock_validation.json)。任务领取只依赖原 JobQueue 的文件锁与逐来源 claim 锁，不通过远端 PID 判断任务失效；三个节点不会同时领取同一来源。原 controller 在所有来源完成后继续负责导出结果和最终标签关联。

没有迁移画质队列或修改范围指纹。本机使用 `/tmp/listen-j-quality-env/bin/python`，PyAV17.1.0、NumPy1.26.4、MediaPipe0.10.21、OpenCV4.11.0；启动前逐项比对原代码、模型、依赖和范围 contract，必须完全一致。保持原裁框与筛查规则、连续片段复用解码器、检测器单线程。结果直接保存原两机画质目录，不需要另做结果合并。

`capacity.json` 记录本机辅助 worker 的 PID 与启动时间，原画质 status 的来源完成数已包含这些 worker 的贡献；其原有 `quality_workers_live` 仅统计原两机进程，本机 32 个需查 capacity 记录，不能据原状态漏算。

扩容时本机原分片仅剩约12.4万段，原两机各剩35～37万段。已通过 `rebalance.py` 从两机迁移双方标注分支均未领取的来源，共161,191段。三机当前分片为：

| 节点 | 来源数 | 总片段数（含已完成） |
|---|---:|---:|
| node17 | 30,027 | 488,281 |
| node8 | 22,900 | 359,479 |
| node12 | 7,942 | 304,045 |

调整后三机剩余工作量各约28.5万段。总计60,869来源、1,151,805片段，来源互斥、并集及每来源片段数完全不变，见 [balance_audit.json](balance_audit.json)。node17 journal中原有一个重试重复条目，统计按来源去重，保留该条目及队列前缀。

事务 `balance_from_node17/`、`balance_from_node8/` 保存原 journal、受保护游标、源队列状态和目标追加凭证。追加本机 journal 时锁定 ingest及视觉／点头队列，仅追加未领取来源，不修改任何已完成标签、游标或计数。6项原迁移测试与2项追加／恢复测试通过。该脚本为本次事务，不要直接用于下一次新的负载分配。

## 恢复

在本机运行以下命令；`capacity.lock` 和 `capacity_watchdog.lock` 保证单实例。守护通过 PID 与启动时间检查扩容控制器，缺失时才恢复，保留仍在运行的画质子进程：

```bash
/tmp/listen-j-quality-env/bin/python scripts/listen_j_expand_local_capacity.py --watchdog
```

两项测试覆盖“即使显存低也不能接入有进程的卡”和“扩卡后保留原卡配置并同步 24 卡规模”：`tests/test_j_expand_local_capacity.py`。

实时卡数、PID与完成数见上级 `status.json`；实际24卡产出与加速配置验收见 `full24_validation.json`（生成后才代表验收完成）。
