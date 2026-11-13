# J v1 四机 CPU 分片、断点续跑与训练共存试跑

**后续用户已明确改为八小时内完成原始全量。** 有限试跑已停止，128 条补充试跑未启动；当前持续任务请看 [64,622 条正式队列](../../data_corpora/seamless_listen_j_full_20261029/README.md) 与 [实时状态](../../data_corpora/seamless_listen_j_full_20261029/status.json)。下面保留每机 1 worker 的初始试验记录，不能将其并发误当作正式全量配置。

本次只增加运行层，固定 J v1 切分规则及音质规则不变。每台最多一个 worker，主要计算 2 线程、CPU 亲和性限定两个逻辑核、`nice=10`，音质模型只用 CPU，优先复用精确区间缓存，不生成审核视频。新结果写入独立工作目录，不写旧全量结果或训练输入。

## 入口

| 用途 | 文件 |
|---|---|
| 初始化、试跑、状态、校验、停止 | [run_listen_pair_j_cluster.py](../../scripts/run_listen_pair_j_cluster.py) |
| 四机地址与观察窗口配置 | [listen_pair_j_cpu_cluster.json](../../configs/listen_pair_j_cpu_cluster.json) |
| 分片、跨机锁、失败恢复、完成回执 | [listen_pair_j_queue.py](../../scripts/listen_pair_j_queue.py) |
| 原 J 规则的 CPU 运行适配与逐段检查点 | [listen_pair_j_runtime.py](../../scripts/listen_pair_j_runtime.py) |
| 固定算法与旧样本对照 | [J v1 交接](../../releases/listen_pair_segmentation_j_v1_20261029/README.md) |
| 设计／执行计划 | [设计](../../docs/superpowers/specs/2026-10-29-pair-j-cluster-design.md)、[计划](../../docs/superpowers/plans/2026-10-29-pair-j-cluster.md) |

内部 SSH 均使用 22 端口；node8 地址为已核实的 `172.29.104.233`，不是原消息中的 `1172.29.104.233`。node12 是当前机器。连接配置不包含私钥内容。

## 有限试跑与恢复

从项目根目录执行：

```bash
# 建立新的一批固定来源；同一 work 已存在时只校验，不重选或清空。
python scripts/run_listen_pair_j_cluster.py init \
  --work experiments/pair_j_cluster_pilot_20261029/run128 --count 128 --seed 20261102

# 启动或继续同一工作目录：先 node8/node4，稳定后加入 node17/node12。
python scripts/run_listen_pair_j_cluster.py pilot \
  --work experiments/pair_j_cluster_pilot_20261029/run128

python scripts/run_listen_pair_j_cluster.py status \
  --work experiments/pair_j_cluster_pilot_20261029/run128

python scripts/run_listen_pair_j_cluster.py verify \
  --work experiments/pair_j_cluster_pilot_20261029/run128

# 请求本轮 controller 与 worker 在安全边界退出，保留检查点。
python scripts/run_listen_pair_j_cluster.py stop \
  --work experiments/pair_j_cluster_pilot_20261029/run128
```

`pilot` 是有时间上限的训练共存试跑，不是无限期守护任务。每阶段至少观察 300 秒、最多 600 秒；结束时停止自己启动的 worker。重新执行同一条 `pilot` 可继续未完成来源，已提交完成回执的来源直接跳过。上轮停止请求不会阻止新一轮。`init --all` 只建立全量清单，本次未执行；当前入口没有自动长期全量运行命令。

试跑以训练日志最近 100 个正常训练步为基线，每阶段比较最近 80 步的中位数和 P95；比较窗口内每个节点至少 80% 采样须在实际处理来源，不能用队列已空闲的训练速度认定四机稳定。另以不重叠的 40 步窗口判断持续变慢：连续两窗任一指标较基线超过约 5%，就暂停本轮全部切分负载，保留检查点。训练超过 60 秒没有新进度也会保守停止，不直接归因为切分。

## 分片与检查点

- 来源 ID 的 SHA256 固定映射到四个分片；worker 优先处理本分片，可领取其他空闲分片，避免一个节点提前空闲。
- 共享目录上同时持有来源锁和节点槽位锁；领取未完成来源时使用非阻塞锁，不接管仍被活跃进程占用的来源。四机各一个 worker，不在节点内再开来源级并发。
- `manifest.json` 固定来源、代码、反馈输入与配置指纹；更换规则或输入时应建立新 work，不能混用旧检查点。
- 每处理完一个候选，原子保存 pending、已处理区间、父子恢复关系、音质证据与统计；重启从该候选边界续跑。模型前向中断时，最多重算尚未提交的当前候选。
- `result.json` 写完后才写带校验和的 `complete.json`；提交完成但队列计数尚未更新时可恢复，避免重复出结果。每个来源最多重试三次，失败明确登记，不能冒充完成。
- controller 心跳租约 60 秒；失联或过期时 worker 在候选边界退出。正常停止先 SIGTERM；超时清理只针对 PID、启动时间、脚本、work 与 session 全部匹配的本轮进程。

## 输出与验证

工作目录包含 `manifest.json`、`queue.json`、`control.json`、`pilot_report.json`、节点状态和日志。每个来源在 `results/<来源>/` 下保存 `initial.json`、`qc/`、`checkpoint.json`、`result.json`、`complete.json`。终端 `clips` 仍包含自动通过、待复核或排除，不能把候选总数当作训练可用数。

预检已完成：

- [跨机共享锁实测](lock_validation.json)：node8/node4 同时竞争仅一个获得锁，另一台持锁期间不能重复领取。
- [真实中断恢复与原 J 结果对照](regression_validation.json)：4 条代表来源的起止、原文、角色、音质路由、恢复父子关系与原 J 一致；node8 上实际 SIGTERM 后恢复成功。128 次音质尝试均精确复用。
- 14 项队列／控制器／原切分测试通过，覆盖进程退出恢复、完成回执、重复领取防护、重试隔离、持续变慢窗口与空闲负载防误判。
- [首次资源核查](resource_audit.json)：四台切分 PID 均不在 GPU 计算进程列表，CPU 亲和性为两个逻辑核。进程包含心跳及库辅助线程，不能将 `/proc` 总线程数等同于主要计算线程数。

首批 [32 条试跑](run/pilot_report.json) 全部完成且无失败。双机阶段观察 300.74 秒，最近 80 步中位数 2.5922 秒、P95 2.9294 秒；相对无切分时 100 步基线的 2.6207 秒／2.9725 秒分别为 −1.09%／−1.45%。四机阶段因来源很快耗尽，不能作为持续四机负载验收。已准备的 [128 条独立来源](run128/manifest.json) 与首批无重合，但用户改为直接全量，故取消启动，转入正式持续队列。

全量输入适配补验：[4 条原 J 结果逐字段等价、2 条没有旧缓存的原始来源](full_adapter_validation.json) 全部通过；两条新增来源生成 20／26 个终端 pair，46 次 CPU 音质计算完成。队列、期限容量估算、槽位与原切分合计 **17 项测试通过**。固定 J 及归档 **46 份源码指纹仍一致**。
