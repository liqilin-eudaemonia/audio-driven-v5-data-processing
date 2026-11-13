# Seamless J：两机 CPU 画质筛查，再继续原加速打标

**后续指令已改为并行模式：每机32个低优先级画质worker，原标注恢复并扩展为node17＋node8＋本机node12的0～2卡，共19卡；最后按画质结果过滤。以下串行启动说明为历史记录，不再用于恢复任务。当前入口、三机结果与恢复见 [19卡并行交接](../../experiments/listen_j_annotation_three_nodes_20261030/HANDOFF.md)。**

2026-10-30。用户授权两机 CPU 高并发筛查、采用已验证裁框、已完成标签不重复打标、筛后通过项补齐缺失标签。未恢复 goal 自动续行或定时消息。

## 范围与阶段

- 原 J 切分来源：`../seamless_listen_j_full_20261029/results/<source>/result.json`。
- 本轮范围沿用两机打标有效输入：60,869 个来源、1,151,805 段。node17 为 649,665 段；node8 为 502,140 段。上游 194 个切分失败来源未混入。
- `nodes.json` 同时记录旧标注工作目录、画质工作目录、新标注工作目录。node17 原 ready journal 有一条重复来源，本轮去重后按原分片审计核对来源与片段总数；不重复筛查。
- 两机旧守护／监督已安全停止，旧 worker 接到 SIGTERM 后保留断点退出。记录在各节点 `pause_receipt.json`；旧目录的 `STOP_SUPERVISION` 防止旧队列自动继续。
- CPU 阶段结束、两机筛后清单均导出后，自动启动新标注目录的原 supervisor/watchdog。CPU 阶段使用两机资源，GPU 暂不打标。
- 较早完成的一机可协助另一机处理剩余画质来源，使用同一来源锁，结果仍写入原分片；不会改变标注分片身份。

## 规则与裁框

沿用 [视觉规则 v1](../../releases/listen_visual_quality_v1_20261030/HANDOFF.md)：5 fps 原图测量；连续低清晰度 ≥2 秒 **或** 累计 ≥25% 排除；无法确认的测量不足单列待复核；已确认横转等使用来源绑定依据。未知横转、全程均匀模糊仍有既有覆盖限制，没有宣称本轮补齐。

通过项应用最小平移优先、边长最多 +5% 的固定裁框；无法满足条件时保留旧框。只更新新输入行 `source.video_transform.crop`，保留 pair ID、音视频时间、原始源定位与身份隔离字段。`source.quality_screening` 保存原框、原提案、测量位置及规则。旧 J 结果及原媒体不改写。

## 已完成标签如何复用

- 新标注工作目录分别为 `experiments/listen_j_annotation_quality_node17_20261030`、`experiments/listen_j_annotation_quality_node8_20261030`。
- 新目录的 `vlm/results`、`nod/results` 直接链接到对应旧目录的结果存储。原 worker 的“结果存在即跳过”逻辑继续适用，按视觉／点头分支分别补齐缺失项。
- 筛查时核对已完成标签的来源、参与者、起止时间；记录旧标签文件 SHA256、原裁框和 `crop_matches_final`。
- **按用户明确要求，旧裁框产生的完整标签也不重算。** 它们保持原始来源记录，不伪装为新裁框标注；后续标签验收可按裁框变化单列。缺失标签使用筛后新裁框。
- 未完成的旧窗口级 partial 不挪到新裁框任务；新目录采用独立 partial，避免将旧输入的半段输出拼接进新输入。
- 被画质排除的已完成标签仍留存，但不进入新打标 ready 队列或最终通过清单。待复核项同样暂不派发。

## 保持原来的打标加速配置

各节点的 `original_config.json`、`original_selected.json`、`original_resource_tuning.json` 是切换前快照。导出新队列时原样继承 selected、resource_tuning 和 config 的全部加速字段，仅补充新输入来源、主机校验与独立临时缓存路径。

维持 batch512、调度 token32768、原生 8 fps、CPU 进程解码、连续补入、预处理重叠、两个 renderer 和原逐卡 CUDA Graph 配置；点头仍用原 25 fps GLARE recall。没有在本轮重新探索 GPU 配置。原节点个别逐卡覆盖参数同样保留。

## CPU 提速与等价验证

- 独立环境 `/tmp/listen-j-quality-env/bin/python`：PyAV17.1.0、MediaPipe0.10.21、NumPy1.26.4、OpenCV4.11.0.86；复用本机既有依赖，不修改旧打标环境。两机原点头环境的 PyAV16 会导致小幅检测与裁框差异，已通过独立环境消除；两机各 5 段复测完全一致。
- 每个 worker 绑定一个允许的逻辑 CPU，OpenCV／解码线程保持 1，MediaPipe graph executor 设为 1。模型、推理选项和阈值不变。默认检测器约 63 线程，原实现高并发会造成严重线程争抢。
- 同一来源按 listener、开始时间处理，最多缓存两个解码器。相邻且向前的请求间隔 ≤1 秒时继续解码；重叠、回跳、长间隔正常 seek。保留原生像素转换、取帧规则和每段检测器重置。缓存只保留解码器及边界帧，不缓存整段 4K RGB。
- 16 段真实对照逐项相同；连续片段组 69.07→54.12 秒，另一组 3.31→3.27 秒。该结果是局部测速，不代表全量固定加速比例。
- 紧凑原子 JSON 降低落盘体积；示例缩至缩进格式约 70.5%。以来源分配任务、来源结果哈希每来源校验一次；逐段成功测量可复用，失败来源通过队列重试。
- 8 段组合实测验证新裁框、筛选结果和旧标签复用。回归测试及测速文件位于 `experiments/j_quality_pipeline_validation_20261030/`。
- `worker_control.json` 控制各机 worker 数。并发采用同一批真实片段测试，判断有效吞吐；不以线程或进程数量冒充加速。

## 文件与状态

每个节点目录包含：

| 路径 | 含义 |
|---|---|
| `pipeline_config.json` / `contract.json` | 分片、节点、规则／源码／依赖指纹 |
| `worker_control.json` | CPU 并发配置 |
| `controller.json` / `watchdog.json` | 阶段、存活身份、来源进度和调度记录 |
| `telemetry.jsonl` | 吞吐、CPU 配额使用率、worker 数；来源整组提交会有统计延迟 |
| `measurements/<hash>/<sample_id>.json` | 原始检测、清晰度区间、裁框测量；可续跑 |
| `approved_jobs/<source>.json` | 本来源所有画质决定、通过的新裁框行、旧标签复用证据 |
| `quality/completed/<source>.json` | 来源完成凭证与解码复用统计 |
| `queues/quality/state.json` / `errors/` | 队列、未完成项、失败原因 |
| `decisions.jsonl` / `retained_rows.jsonl` | 本节点导出后的决定与通过行 |
| `excluded_rows.jsonl` / `review_rows.jsonl` | 本节点导出后的排除／复核清单 |
| `quality_complete.json` | 本节点筛查与新打标队列导出完成凭证 |

整合读取两个节点的输出，不能只读取 node17。`retain` 仍不是人工确认训练准入，`training_admitted=false`。

## 代码与恢复

- [运行入口](../../scripts/listen_j_quality_pipeline.py)
- [队列、筛查、导出与自动衔接](../../src/listen_benchmark/j_quality_pipeline.py)
- [连续解码复用](../../src/listen_benchmark/quality_decode_session.py)
- [检测器线程限制](../../src/listen_benchmark/quality_cpu_execution.py)

在对应节点恢复画质守护（具备单例锁）：

```bash
/tmp/listen-j-quality-env/bin/python scripts/listen_j_quality_pipeline.py watchdog \
  --work data_corpora/seamless_listen_j_visual_quality_v1_20261030/node17
```

node8 改为相应目录。保留节点 `ready.jsonl`、contract、队列和逐段测量，不要启动旧标注目录的守护。画质结束后，新目录的原标注守护负责持续补齐标签；旧分片 status 不再代表新流程进度。
