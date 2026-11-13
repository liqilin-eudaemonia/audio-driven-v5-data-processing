# J 新切分 → 官方并集净化＋点头补充 → Omni 整段自然语言

> 2026-10-31 更新：画质关口已覆盖32卡Omni和128个点头worker。自适应请求提交已通过同样本与四机生产对照并完成推广；画质CPU现为256进程（node8释放CPU后新增32），已验证并停止继续试参，后台生产和故障恢复继续。半小时巡检继续至06:07:27UTC。详见[当前运行交接](../../related/P2_画质前置筛查与四机32卡运行_20261031.md)。


2026-10-30，用户认可“第一版描述方式＋简短点头／静音说明”，授权代码验证通过后启动多机。用户进一步明确：**直接使用官方 listener 词、句、VAD 并集净化；不补 ASR，不叠加旧中文过滤区间。**

## 数据入口

- 新 J 切分：`data_corpora/seamless_listen_j_full_20261029/results/<pair>/result.json`；同目录`complete.json`提供完成凭证与哈希。
- 当前打标范围：`experiments/listen_j_annotation_two_nodes_20261030/nodes.json`，60,869个来源、1,151,805个片段；不与旧722,706段或旧141.9万片段混为一套。
- node17：`experiments/listen_j_annotation_node17_20261029/`。
- node8：`experiments/listen_j_annotation_node8_20261030/`。
- 每节点`ready.jsonl`给来源范围，`jobs/<pair>.json`给片段及原始媒体引用。
- 点头：每节点`nod/results/<SHA256(sample_id)前2位>/<sample_id>.json`；`nod/completed/<pair>.json`提供来源完成凭证。

生产者只领取点头已完成的来源，不遍历百万级结果目录。两个来源分片沿用既有互斥划分；node4仅增加算力、不增加来源分片；新的Omni队列由三机共享，按可用worker动态领取来源，减少节点尾部不均衡。

资源：node17 `172.29.15.91`、node8 `172.29.104.233`、node4 `172.29.225.51`各使用0–7卡，共24卡。node4按用户追加授权接入，不使用本机显卡。

## 输入与输出约定

- 模型 `/private/ckpts/Qwen3-Omni/Instruct`，使用已认可的`first_plus_evidence` prompt。完整说明见[prompt.txt](prompt.txt)、冻结参数见[config.json](config.json)。
- 同一个 listener、原始预裁框缩放512×512、原生视频8fps，保留完整时长；复用J视觉解码的时间网格和JPEG85像素路径。点头25fps结果必须与同片段的人物、裁框、起止和来源一致。
- 独立listener长音轨按`round(start*sr):round(end*sr)`读取，不使用混音或视频内嵌声轨。官方词／句／VAD换算到该精确起点，按已验收规则取并集、100ms扩边、10ms淡入淡出；区间外置零，不删时间。
- `listener_audio_is_silent`由实际波形全零检查产生。官方字段缺失不当作静音；点头未完成不当作空候选；检测不到人脸记为证据不可用并单列。
- 每次独立生成一段英文，无旧caption或ASR转写输入。原始prompt、caption、每次尝试、视频帧／音频波形哈希、点头来源均保留。
- 每卡一个Omni副本；完整prefill、单请求、decode-only CUDA Graph。当前node17/node8各卡4个CPU准备进程、预取8；node4各卡8进程、预取16，均就绪先提交。temperature=0，4096 token，重复惩罚1.05；空、截断或长循环最多重试一次，惩罚1.12。不通过输出仍留原文和失败标记。
- 不新增WAV，不改变旧训练缓存；新音频是可复现的segment recipe。读取入口为`listen_benchmark.j_omni.read_audio`，不能直接传给旧的整段WAV recipe读取器。

## 本目录下结果

| 路径 | 内容 |
| --- | --- |
| `caption/results/<hash2>/<sample_id>.json` | 整段原始描述、prompt、来源、输入证据、recipe和生成状态 |
| `attempts/<hash2>/<sample_id>/<n>.json` | 每次原始生成；恢复时相同输入直接复用 |
| `audio_recipes/results/<hash2>/<sample_id>.json` | 原长音轨路径、精确样本范围、官方区间与净化规则 |
| `official_metadata/<hash2>/<key>.json` | 官方原始JSON及归档身份，按来源缓存 |
| `held/results/<hash2>/<sample_id>.json` | 点头证据不可用等需另行处理的片段，不伪造空候选 |
| `errors/results/<hash2>/<sample_id>.json` | 输入准备异常；来源队列自动重试，连续失败单列 |
| `caption/completed/<pair>.json` | 来源完成凭证，分别计caption、生成失败、held |
| `queues/caption/state.json` | 已处理来源、进行中来源、失败来源与领取游标 |
| `nodes/<node>/`、`workers/` | 监督、守护、日志和逐卡心跳 |
| `status.json` | 最近一次手动状态检查；完成数以来源凭证为口径，正在处理的来源尚未计入 |

`resolved_clips`包括已完成来源中的held和生成异常，不能直接当作合格caption数。`receipt_counts.valid_generation_clips`只表示正常生成，不代表语义完全正确。所有结果保留`training_admitted=false`，原split/group不变；本次没有启动训练或全量缓存重算。

## 验证与部署

- [小批验收](validation/canary_audit.json)：16条真实J片段，6条有声、10条静音，13条有点头候选、3条无候选，全部首次正常生成；有声／静音各一条音视频重放哈希相同。
- `tests/test_j_omni_pipeline.py`：12项测试，覆盖分段静音与旧已验收读取器逐样本一致、空／缺失官方区间、越界邻居不误扩边、源文件变更、点头缺失／空／不可用、跨worker抢占和断点恢复、生产者等待凭证、CPU-only监督。
- [跨机器锁验证](validation/cross_host_locks.json)：node17→node8及反向均验证同一锁排他，队列不依赖远端PID猜测。
- [最初切换记录](transitions/)：两机最初各保留80个原CPU点头worker；旧VLM完成当前批次后退出，显卡释放后才启动Omni。后续CPU并发平衡另见下节和`revisions/nod_cpu_balance/`。其他任务不受停机操作影响。
- 16条canary输出已原样复用，见[复用凭证](validation/promoted_canary.json)。
- `code/`保存运行时源码，`delivery_code/`保存交付脚本、测试与哈希。这里只做本地源码备份，未宣称推送GitHub。

## 状态与恢复

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
python experiments/listen_j_omni_20261030/check_status.py
```

每机已有watchdog维持新Omni supervisor；同名worker有排他锁，崩溃后会恢复未完成来源。每卡每小时最多自动启动4次，超过上限记录blocked_gpus等待排查，避免无限失败重启。数据源连续失败由队列单列，不能自动宣称完成。

手动恢复监督（在对应节点执行，已有实例会被排他锁阻止重复启动）：

```bash
LD_PRELOAD=/private/software/conda/envs/vllm/lib/libstdc++.so.6 /private/software/conda/envs/vllm/bin/python scripts/listen_j_omni.py watchdog --work experiments/listen_j_omni_20261030 --node node17
# node8上使用 --node node8
```

旧点头监督继续使用原工作目录。旧配置`visual_enabled=false`只停用VLM调度，不停止点头；不要直接改回true，否则会与Omni争用显卡。切换脚本`transition.py`已执行，拒绝盲目重复执行。修改核心代码后须显式更新并验证冻结配置，不能绕过启动哈希校验。

本流水线有后台故障恢复和本地状态记录；没有恢复用户已停止的20分钟聊天汇报。2026-10-30另外按用户明确授权启动了下述十小时goal及半小时本地巡检。

## 生产预处理修复（execution revision v2）

真实运行发现：多个全静音片段使用相同音频数组，vLLM按模态缓存时只命中音频，造成视频单独进入要求音视频配对的Omni处理器，抛出StopIteration。直接关闭处理器缓存也会在当前安装版本触发音频占位符计数错误。

采用`llm.reset_mm_cache()`，在每次同步生成前清空前后端多模态缓存，始终成对预处理。音视频数据、prompt、采样、生成参数与pipeline_id不变。执行版本`paired-av-cache-reset-v2`写入worker心跳和新结果。CPU定向回归见[缓存回归报告](validation/processor_cache_regression.json)：默认缓存复现错误，完全关闭缓存仍失败，逐请求重置缓存3/3通过。

`revisions/`保留修复前后配置、源码快照、精确进程滚动替换记录和失败重入队前记录。`repair_runtime.py`只对本流水线的worker发SIGTERM，保留当前结果，由监督拉起新执行版本；失败来源重新入队并复用已完成片段。旧点头worker不受影响。源文件校验、音视频时轴、生成结束检查与语义人工验收是不同层次，不能相互替代。

新增node4与node17的跨机锁双向验证通过，见[第三节点锁验证](validation/node4_lock.json)。`audit_runtime.py`逐卡抽样核验实际输出与输入证据，同时要求此前10个失败来源都获得完成凭证，完成报告保存为`validation/production_v2_24gpu_audit.json`。

三机24卡上线验收已通过：[逐卡实际输出与失败恢复验收](validation/production_v2_24gpu_audit.json)。截至该报告，来源凭证累计处理4,054段，10个历史预处理失败来源全部恢复，失败来源队列为0。逐卡检查只验证来源、净化时轴和生成有效性，不等同全部语义已人工确认。

## 2026-10-30用户授权的提速与十小时巡检

第一轮同样本端到端试验得到2.034倍吞吐，第二轮8进程相比4进程又提高21.9%；每轮四阶段96次生成均正常且逐条caption一致。node17/node8保持4个准备进程，node4的8卡已推广8个准备进程，24卡来源与音频抽查通过；模型与输入规则不变。[执行窗口、完整证据、回退与巡检说明](../../related/P2_Omni24卡提速与10小时巡检_20261030.md)。新profile在`config.json.preparation_profiles`和逐卡心跳中记录；不要只按旧的“双线程”文字复现执行配置。

CPU点头与Omni准备会争抢CPU。node17点头并发已从80调整为48，前后180秒窗口Omni由2.08升至2.46段/秒，点头由3.94变为3.72段/秒，仍有大量领先结果供接续。node8相同检查仅增2.14%，已自动恢复80；结果看`performance_check/node8_cpu_balance/complete.json`，最终快照为`performance_check/optimization_frozen.json`。本轮参数调优收尾，十小时巡检继续。顺序生产窗口不是同样本对照，后续继续检查稳定吞吐。


另做过node4 GPU7的batch1/4/4/1推理对照：96次生成正常，输入媒体和prompt token一致，推理阶段2.055倍，但存在动作/发声描述分歧，未证明质量等价及生产端到端收益，因此没有推广batch4。试验覆盖已撤销，单请求恢复及新结果来源/音频复验通过：`validation/batch4_restore_audit.json`。详情和原始差异保留在`performance_check/batch4_canary/`。当前所有生产请求仍逐条生成，CPU预取配置保持已验证设置。


### 17:46 UTC 有效加速配置

node17与node4均为每卡CPU准备8进程/预取16；node8为4进程/预取8。CPU点头并发分别为48、80（node4无点头分片）。24卡生产来源和音频抽查通过，见`validation/prefetch_production_audit.json`。GPU生成仍为单请求；批处理候选未推广。以`performance_check/optimization_frozen.json`和`config.json`为当前参数入口，前文早期配置保留为实验历史。十小时巡检与故障恢复继续。

2026-10-30 21:00 UTC：视频尾帧元数据误拒绝修复已在24卡完成更新及逐卡来源/音频验收。配置不变，详见[巡检交接](../../related/P2_Omni24卡提速与10小时巡检_20261030.md)与`validation/pts_tail_cluster_audit.json`。
