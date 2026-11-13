# J 数据：node17 八卡打标与持续监督

**2026-10-30两机扩展：** 本节点当前来源分片649,665段（含已完成），另502,140段已移交node8；原导入总量仍为历史记录。node17工作进程未重启。[两机范围、进度、代码与结果位置](../listen_j_annotation_two_nodes_20261030/README.md)。

本轮用户授权：node17 空闲后，以八张 H200 测速并接续 J 数据打标；保持八小时持续监督。任务仍在运行，不是完成报告。

监督窗口：2026-10-29 15:57:32～23:57:32 UTC（北京时间 10 月 29 日 23:57:32～10 月 30 日 07:57:32）。到达八小时不会自动杀掉尚未完成的打标。不要把八小时监督承诺解释为全量完成时间。

## 数据范围和不可改变的语义

- 上游：[J 全量结果](../../data_corpora/seamless_listen_j_full_20261029/)。读取 `result.json` 的终端 `clips`，验证 `complete.json` 的 contract 和 SHA256；不读取 `initial.json` 的旧候选作为最终片段。
- 已提交来源 64,428 条，上游失败 194 条，总计 64,622 条。失败来源原样记录，未冒充成功。
- `audio_final.pool=keep_candidate` 且 listener 预裁框、媒体身份和时长有效的片段进入本轮队列：60,869 个来源任务、1,151,805 个片段。数量由 `ready.jsonl` 唯一来源计数确认。
- 保留 train/dev/test、参与者和隔离组信息；没有重新套用旧版“只取 train／裁框≥384”的范围，也没有自动放行 `audio_review` 或 `audio_reject`。
- listener `crop_proposals.listener` 使用原来的整数裁框规则，缩放 512×512；不用双人原画替代。原始音轨路径及身份继续保存。本轮视觉不输入音频，也不宣称重新做了 ASR。
- Qwen3-VL-8B-Instruct、观察式 C prompt、原生视频 8 fps、JPEG 85、整段＋3 秒窗口／1 秒步长不变。temperature=0、max_tokens=4096、max_model_len=49152。
- GLARE recall 点头补充读取相同片段与裁框的 25 fps 姿态，CPU 独立队列。专用摇头检测关闭。全部输出保留原文和来源，`training_admitted=false`。

## 本轮八卡重复测速

每个配置重复两次，每次 50 个相同片段、293 次请求。所有八组结构有效 293/293，整段有效 50/50，无 token 截断。以下吞吐合并重复耗时，避免选择偶然最快的一张卡。

| batch | token budget | frame budget | 两次吞吐，段/秒/卡 | 合并吞吐 |
|---:|---:|---:|---|---:|
| 256 | 16384 | 8192 | 0.40595、0.42260 | 0.41411 |
| 512 | 16384 | 16384 | 0.45079、0.35041 | 0.39431 |
| 512 | 32768 | 16384 | 0.44347、0.44438 | **0.44392** |
| 256 | 32768 | 8192 | 0.32616、0.32401 | 0.32508 |

本轮选择 **batch512／32768 token／16384 frame budget**，memory=0.85，50 段组成跨片段批次，后台预取。相比本轮 batch256／16384，合并吞吐约提高 7.2%。这仅是本次已测配置中的最佳结果，不是所有配置的全局最优；不是精度提升证明。

最初自动选择单次最快的 batch512／16384，随后修正为按两次总耗时选择，通过逐卡安全退出与恢复换用稳定配置；已完成结果保留。冷启动、提前构建帧缓存时间不计入表内吞吐，正式任务还包含解码和落盘。

历史 9 条“越界”是 `2.78 > 2.7799999999999994` 这类浮点舍入误差。本轮验证容差为 `1e-9` 秒，未改写任何模型时间或描述；真正越界仍标为无效。

## 运行状态与输出

- [监督程序状态](supervisor.json)：真实 PID＋进程启动标识、阶段、GPU 指标、子进程、错误与调整记录。
- [监督程序守护状态](watchdog.json)：监督进程退出后重启，单例锁避免重复启动。
- [基础执行配置](selected.json)、[逐卡覆盖与 worker 配置](config.json)、[当前 CPU 解码配置](resource_tuning.json)。配置更新后以各 `workers/gpu-N.json` 的 execution 和活动 CPU 参数为已生效依据；不能只看 selected.json 判断单卡试验参数。
- [导入结果](ingest_complete.json)、`ready.jsonl`、`imported.jsonl`：来源与队列记录。早期导入统计有重启后的计数修正，唯一来源／片段数应以日志与任务文件核验。
- `jobs/<source>.json`：紧凑输入、原结果哈希、排除原因。
- `vlm/results/<sample_id的SHA256前两位>/<sample_id>.json`：全部原始 VLM 响应、时间窗口、解析状态、输入证据。
- `nod/results/<相同分桶>/<sample_id>.json`：点头候选、姿态／分数、裁剪和帧来源。不是手工合并后的最终标签。
- `vlm/partial/`：可恢复的窗口级检查点；最终结果原子落盘后删除对应检查点。
- `queues/{vlm,nod}/state.json`、`{vlm,nod}/completed/`：独立队列与按来源的完成凭证。`failed` 不是成功；来源完成数不能直接当作片段完成数。
- `logs/`、`telemetry.jsonl`：进程日志及约 30 秒一次的资源记录。

## 监督与资源策略

每 15 秒检查真实进程身份；进程退出后从队列和窗口检查点恢复。连续无进展超过 10 分钟且 GPU 低于 5% 持续约 5 分钟时才恢复该 worker，避免误杀正常长批次。仅操作本任务记录的 PID／进程组，不停止其他任务。

执行配置逐卡更新，等待前一个新 worker 产出后再更新下一个。先请求安全退出；超过 10 分钟仍未退出才结束本任务进程组，保存的检查点继续有效。

CPU 解码按机器共享槽位限制，后台预取；有 CPU 余量且供料阶段持续低利用率时允许从每 worker 4 路／总 32 路提高到 8 路／总 64 路。不能把显存占用率等同于计算利用率，也不能用无效运算填满 GPU。

以上为初始自动调整范围。后续已测过 8／80、12／96 等配置；截至 2026-10-30 00:08 UTC，[八卡实际产出检查](monitor_snapshots/process_all_eight_producing.json)确认全部 GPU 已采用连续补入、decode-only CUDA Graph（捕获1～128）、独立 CPU 进程解码、2个 renderer、显存比例0.80；batch512／32768 token／16384 frame budget 不变。每 GPU 16路解码、全机112共享槽、64路点头。四线程 renderer 未启用。全卡稳定测速仍待完成，部署完成不等于已证明全局最优。后续变化以 config.json、resource_tuning.json 和实际 worker 状态为准，完整试验依据见下方记录。自动控制器不会为了提高占用率增加无效计算。

解码帧临时放在 node17 `/dev/shm/listen_j_annotation_prime_20261029`，正常新 worker 在最终结果落盘后删除已消费帧缓存。GLARE 的临时 25 fps 视频只用于姿态提取，处理后删除；没有新增训练 WAV。

## 代码与恢复

- [导入、预取、GPU worker](../../scripts/listen_j_annotation.py)
- [点头补充](../../scripts/listen_j_annotation_nod.py)
- [节点监督程序](../../scripts/listen_j_annotation_supervisor.py)
- [监督进程守护](../../scripts/listen_j_annotation_watchdog.py)
- [校验与动态队列](../../src/listen_benchmark/j_annotation.py)
- [跨片段批处理](../../src/listen_benchmark/visual_batch.py)、[输入协议](../../src/listen_benchmark/visual_native8.py)
- [逐卡执行配置](../../src/listen_benchmark/execution_profile.py)、[连续补入](../../src/listen_benchmark/rolling_inference.py)、[独立 CPU 解码](../../src/listen_benchmark/process_decode.py)

在 node17 上恢复守护程序（有单例锁；不要删除生产锁文件或已有输出）：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
nohup /private/software/conda/envs/vllm/bin/python scripts/listen_j_annotation_watchdog.py \
  --work experiments/listen_j_annotation_node17_20261029 \
  --upstream data_corpora/seamless_listen_j_full_20261029 \
  > experiments/listen_j_annotation_node17_20261029/logs/watchdog-manual.log 2>&1 < /dev/null &
```

监督状态、队列与已有结果必须保留。若人为停止监督，需要先创建工作目录内 `STOP_SUPERVISION` 标记，再停止该任务守护进程和监督进程，防止被自动恢复；此操作不等于终止正在运行的 worker。

验证：队列／断点／浮点时间检查 8 项，批处理与预取 13 项，重复测速选择及 PID 身份检查 2 项通过；真实 J 片段点头 smoke 已通过。正式八卡结果仍需按产出持续验收，不以单元测试替代全量完成验收。

正式结果的有限抽查见 [bounded_output_audit.json](bounded_output_audit.json)：20 个完成来源、342 段、1,825 次响应，1,821 次结构有效，4 次无效（3 次零长度区间、1 次生成截断）。无效响应原文和错误状态保留，不改时间、不伪造成通过；队列“完成”表示原始处理结果已落盘，不等于所有窗口有效或已获训练准入。GPU 和 nod worker 已全部使用新版计数；早期计数已按完成凭证校准，校准时间保存在队列状态的 `count_reconciled_at`。

CPU 点头补充从 8 路逐步提高到 16、32、48 路，以追赶 VLM。升至 48 路前五分钟，VLM 为 3.45 段/秒、点头为 2.56 段/秒，192 核机器负载约 96；依据保存在 [resource_adjustments.jsonl](resource_adjustments.jsonl)。各 GPU 解码 8 路、共享上限 64 路，受机器 CPU 和实际供料情况约束。代码快照为本地备份，未宣称已推送 GitHub。

2026-10-29 点头异常修正：全程未检测到人脸时，保存 `status=unavailable`、`reason=no_detected_face`、`events=null` 和逐帧检测证据；这表示缺乏点头判断证据，不是负例。修复前重复三次失败的真实片段已验证并重新入队；[该片段记录](nod/no_face_recovery.json)。正常点头片段重新运行后，事件、逐帧分数、姿态、来源、帧数与参数全部逐项一致：[正例回归](nod/revision2_positive_regression.json)。CPU worker 安全退出不再消耗输入错误重试额度；所有 32 个 worker 已更新，GPU 进程未重启。

## 后台准备模型输入（GPU worker revision 3）

除提前解码外，把 `prepare_input` 的窗口数组、时间元数据和 chat template 也放到后台预取线程。仍调用原函数；不改变像素、时间、prompt、模型或推理参数。`visual_batch` 读取已准备好的输入，并检查其窗口起止一致，避免前台重复组装。

验证包括：批处理测试由 13 项增加至 15 项；[3 段／16 个窗口的像素数组、元数据和 prompt 一致性](precompile_input_equivalence.json)；[真实单卡首批 66 段／313 次响应全部结构有效](precompile_canary_first_group.json)。首批后台输入准备 9.36 秒，前台读取准备输入不足 0.001 秒。该数字代表准备工作已转移到后台，不代表端到端每批必然节省 9.36 秒；是否完全隐藏仍取决于前一批推理和下一批解码时间。

已设置 `config.json:gpu_worker_revision=3`、`resource_tuning.json:precompile_inputs=true`，由监督程序逐卡更新，等待前一张卡的新 worker 实际产出后再继续。是否全部生效以 `workers/gpu-N.json:revision` 为准。结果新增 `input_performance` 和 `prefetch_group_performance` 以记录后台开销。旧窗口检查点保留并可继续使用。

该版本代码快照：[code_precompile_v3/SHA256.json](code_precompile_v3/SHA256.json)。先前 [code_current/SHA256.json](code_current/SHA256.json) 为推广前版本，保留便于追溯。

滚动更新等待条件随后修复：旧 worker 退出后，其残留心跳曾导致提前更新下一张卡。现在必须同时满足真实 PID／启动标识存活、心跳属于该进程、执行参数和代码版本符合要求、已经产出新结果；任何新 worker 仍在准备首批时，都不更新其他旧 worker。加入了旧进程死亡、版本过旧、PID 不符、尚未产出和成功就绪的回归检查，监督测试共 3 项通过。更新不删除原结果或检查点。最新监督修复快照：[code_supervision_guard/SHA256.json](code_supervision_guard/SHA256.json)。

点头新版有限验收：[最近 30 个完成来源、541 段的来源与状态检查](nod/revision2_bounded_audit.json)。这是有限抽查，不是全量标签正确率证明。

## 运行诊断补充（17:30 UTC 后）

实际 CPU 配额已核实：宿主机 `os.cpu_count()` 为 192，但容器 affinity 为 160 核，`cpu.max=16000000 100000`，即 160 核配额。[真实进程及 CPU 配额快照](monitor_snapshots/cpu_allocation_and_live.json)。先前扩并发记录中的 192 指宿主机数量；后续资源判断应采用 160。当前保持 48 路点头，不继续盲目扩并发。

GPU3 的首批曾耗时偏长，实际完成 73 段、449 次响应，全部结构有效；解码 228.35 秒、后台输入准备 69.59 秒，两次推理分别 276.27／63.29 秒。输出 token 最大 1,876，没有触及 4,096 上限，也不是进程卡死。[诊断明细](slow_first_batch_diagnostic.json)。不同批次输入时长和媒体不同，不能仅用该批与早期 66 段试用批的耗时宣称加速或退化。

点头另用逐 worker 累计计数测量，避免来源任务整组完成造成短时速度失真：[最近固定 PID 窗口](monitor_snapshots/nod48_rate_latest.json)。该窗口 48 个 PID 均未变，约 271 秒新增 588 段，约 2.17 段/秒；它与此前来源完成计数的 2.56 段/秒不是严格对照，不能据此认定 48 路一定更快。更新结束后继续观察稳定窗口。

八张卡全部更新并各自产出后的稳定观测见 [stable_throughput_latest.json](monitor_snapshots/stable_throughput_latest.json)：约 484 秒，8 个 GPU worker 与 48 个点头 worker 的 PID 均未变，分别新增 1,777／1,016 段，约 3.67／2.10 段每秒。这是现有配置的持续吞吐，不是与旧配置的受控精度或速度对照。

同期采样中，GPU 父进程合计约用 39.55 个 CPU 核、点头父进程约 37.56 核；前者未包含独立的 vLLM EngineCore 子进程，不能当作机器总 CPU 使用量。`/proc/locks` 未观测到分布式文件系统解码锁，不能据此断言没有锁竞争。点头主线程的有限状态采样保存在 [nod_wait_states.json](monitor_snapshots/nod_wait_states.json)，后续优化需继续结合实际阶段耗时判断，暂不继续增加并发。

## 精确 RGB 裁剪优化（CPU revision 3／GPU revision 4）

仍然完整解码、转换成 RGB24，并使用原来的整数裁框、PIL bicubic 和 JPEG 85；只省掉了整帧构造 PIL 图像的步骤。现在先在 RGB 数组中取裁框，再把小图转换成 PIL 后缩放。

实现：[frame_crop.py](../../src/listen_benchmark/frame_crop.py)。`visual_native8.prepare_media` 增加可选 `frame_crop` 回调，默认仍为原路径；`listen_j_annotation.py` 由 `visual_fast_frame_crop` 控制，点头入口由 `nod_fast_frame_crop` 控制。两者执行方法写入结果执行信息，不改标注描述或时间。

4 段各 8 帧、每帧交替重复 3 次，全部 RGB 像素与 JPEG 字节一致。[测速原始结果](rgb_crop_benchmark.json) · [复验脚本](benchmark_rgb_crop.py)。单帧裁剪步骤的耗时中位数如下，**不代表整条流水线加速倍数**：

| 原图尺寸 | 原路径 | 数组裁剪路径 |
|---|---:|---:|
| 2160×3840，样本 1 | 83.20 ms | 14.24 ms |
| 2160×3840，样本 2 | 107.18 ms | 17.12 ms |
| 1080×1920，样本 3 | 15.53 ms | 11.39 ms |
| 1080×1920，样本 4 | 13.15 ms | 7.75 ms |

进一步验证：[整段一致性报告](fast_crop_full_parity.json) · [复验脚本](validate_fast_crop.py)。两段 8 fps 视频共 111 帧，全部像素、PTS、JPEG 记录与元数据一致；真实点头正例的编码视频、姿态、逐帧分数、事件和来源也完全一致。4K 样本完整解码由 7.34 秒降到 4.48 秒；1080p 样本 2.53／2.62 秒，存在运行波动，不能声称所有样本端到端都更快。

新增 2 项裁剪测试，覆盖 24 组 RGB／YUV420／YUV444／10-bit YUV、奇数坐标、边缘裁框及越界拒绝；原 15 项批处理测试通过。运行端有限检查见 [fast_crop_live_nod_check.json](fast_crop_live_nod_check.json)。

CPU 分支已启用并按每批最多 4 个 worker 更新，等待新进程实际产出再继续；使用 `nod_worker_revision=3`，无脸仍为 `unavailable`。GPU 先仅在 GPU7 试用 revision 4，其余卡保持运行；是否推广及当前生效情况以配置和各 worker 状态为准。原始结果与检查点保留。最新代码备份见 [code_fast_crop/SHA256.json](code_fast_crop/SHA256.json)。

GPU7 正式试用已完成首批 **80 段、419 次响应，全部结构有效、无截断**；来源、8 fps 和纯视觉输入核对一致：[结果检查](fast_crop_gpu_canary_result.json)。已将 `gpu_worker_revision` 设置为 4，由监督程序逐卡推广。结构检查和输入字节一致不等同于人工标注正确率，原始模型输出及无效窗口处理规则仍保持不变。

48 个点头 worker 均已使用精确裁剪并产出。固定 PID 的初步观测约 231 秒新增 752 段，约 3.25 段/秒：[观测](monitor_snapshots/fast_crop_rate_latest.json)。早前稳定窗口为 2.10 段/秒，但样本不同，不能视为受控提速倍数。GPU 仍在逐卡更新，当前窗口不能用于完整新旧配置对照。[33 秒资源采样](monitor_snapshots/fast_crop_resource_window.json)记录 CPU 平均约 93 核、各卡 GPU 利用率与更新阶段；并未达到持续满载。

## 重叠窗口共享帧数组（GPU revision 5，逐卡推广中）

CPU 定位显示 `np.stack` 和 PIL 转数组占据多数输入准备时间：[分析](monitor_snapshots/input_prepare_profile.json)。新增可选 [shared_video_inputs.py](../../src/listen_benchmark/shared_video_inputs.py)，整段帧只转换一次，窗口引用连续切片；通用函数默认路径保持不变。J worker revision 5 由 `shared_frame_arrays` 配置控制调用，执行信息和准备耗时均记录是否启用。

[两条真实片段交替测速](shared_input_preparation_benchmark.json)：3 个窗口从 46.7 ms 降到 19.3 ms，10 个窗口从 206.5 ms 降到 59.8 ms；窗口数组存储分别从约 56.6/231.2 MB 降至 22.0/65.3 MB。这是 CPU 准备步骤的结果，不是整条打标流水线加速倍数。

像素、窗口时间元数据、prompt 完全一致；2 项新增测试覆盖短片段、刚超过 3 秒、非整数结尾、奇数帧及错误数组拒绝，原 15 项批处理测试通过。进一步运行实际安装的 Qwen3-VL video processor，整段、首窗、末窗的 `pixel_values_videos` 和 `video_grid_thw` 逐元素完全相同，源数组未被修改：[报告](shared_input_processor_parity.json) · [复验脚本](validate_shared_input_processor.py)。CPU 预处理一致尚不替代真实 GPU 持续吞吐验证。

候选 helper 本地快照：[code_shared_inputs_candidate/SHA256.json](code_shared_inputs_candidate/SHA256.json)。随后将其接入 GPU3 原本已经排队的 worker 更新，避免额外重启：GPU3 PID 393585，revision 5，共享数组已启用。验证期间目标版本保持 4；首批完成后已将目标版本改为 5，逐卡推广，仍等待前一张卡的新 worker 实际产出。

[GPU3 完整首组结果](shared_input_canary_result.json)：59 段、353 次响应，352 次结构有效，1 次无效，无截断；来源、8 fps、纯视觉协议及共享数组启用信息核对一致。无效项是模型把短暂闭眼写成 `start=end=2.8`，原文及无效状态保留，不修补时间、不当作有效标注。此类零长度事件在旧生产输出中也存在。推广依据是输入和实际处理器张量一致、运行正常，**不是全部响应通过，也不是精度提升结论**。本组后台输入准备 15.56 秒，不能直接与不同样本批次相比声称端到端提速。

## 解码槽位统一与 80 路试运行

检查发现 GPU 运行配置已是 64 个共享解码槽，但点头每段读取的 `selected.json` 仍为 32，只能竞争前 32 个槽位。[非阻塞探测](monitor_snapshots/decode_slot_probe.json)的 4 次观测共占用 60–64/64 槽。结合前述 CPU 平均约 93/160 核，已将两分支统一为 80 个共享槽位，GPU 解码线程仍为每 worker 8 路，点头仍为 48 个 worker，不触发 GPU 重启。两分支在下次准备组／片段时读取新配置。

原测速解码参数保存在 `selected.json:benchmark_decode_parameters`；`measured_clips_per_s` 仍是原对照实测，不代表更新后的速度。更新前状态：[decode80_baseline.json](monitor_snapshots/decode80_baseline.json)。需要结合之后的 CPU 压力、锁等待和实际吞吐决定是否保留 80，不能仅凭槽位增加认定提速。

初步运行证据：[约 53 秒资源窗口](monitor_snapshots/decode80_first_window.json)中 CPU 平均约 106/160 核，没有 OOM；[约 197 秒逐 worker 计数](monitor_snapshots/decode80_nod_first_rate.json)中固定 48 个点头 PID 新增 854 段，约 4.34 段/秒。当前来源的有限 89 条完成记录，解码锁等待中位数约 0.012 秒、渲染 9.69 秒、人脸提取 2.82 秒。该阶段同时有 GPU 更新和不同媒体，暂不作为严格 A/B 加速结论；维持 80 继续观察。

共享数组接入版代码快照：[code_shared_inputs_worker/SHA256.json](code_shared_inputs_worker/SHA256.json)。首批选定来源固定记录在 [shared_input_canary_jobs.json](shared_input_canary_jobs.json)，避免 worker 领取下一组后错把不同批次作为首批验收。

后续正常更新的 GPU worker 额外设置 `disable_log_stats=False`，在 `logs/gpu-N.log` 记录 vLLM 的 prompt/generation token 吞吐、排队请求及 KV 缓存占用；结果 execution 中以 `engine_metrics_enabled` 标明。仅增加运行诊断，不改变模型输入、采样或请求批次，不单独触发重启。旧进程未开启该日志，不能用缺少日志判断其没有工作。

## vLLM 内部预处理（GPU revision 6，GPU7 单卡试验）

GPU0 的引擎日志出现过只有 1–8 个运行请求、没有等待请求、KV 缓存占用很低的采样；这可能包含批次尾部，不能仅据此判断应加大 GPU batch。进一步在 CPU 上直接运行实际安装的 vLLM renderer：[定位记录](monitor_snapshots/vllm_renderer_cpu_profile.json)。这条 83 帧样本的短窗口预处理约 0.2–0.7 秒，位于原 `prepare_input` 之后，包含 HuggingFace 视频处理、token 化和多模态布局。

vLLM 0.19.0 原生支持 `renderer_num_workers>1`＋`render_cmpl_async`，但明确要求 `mm_processor_cache_gb=0`，因为该缓存不支持多线程并发。当前仅 GPU7 试用 2 个 renderer worker、关闭该处理缓存；模型的 prefix caching 保持开启，所有 engine 提交仍由原主线程执行。

[CPU 对照脚本](benchmark_renderer_parallel.py) · [结果](renderer_parallel_cpu_benchmark.json)：同一片段 10 个窗口，双方均关闭该处理缓存、开启与生产相同的 prefix caching，先预热，再交替重复 4 次；串行中位数 2.812 秒，两路官方异步预处理中位数 1.717 秒。全部 EngineInput 递归比较一致，包括 token、张量、哈希、占位布局和描述；只排除本来不同的 `arrival_time`。没有加载模型权重或使用 GPU，也没有写入生产标签。先前未显式开启 prefix caching 的 CPU 初试为 3.065／1.713 秒，原记录保留在 [prefix-off 初试](renderer_parallel_cpu_benchmark_prefix_off.json)。

这只是 CPU 步骤对照，不代表端到端提高同样比例。接入实现：[engine_input_renderer.py](../../src/listen_benchmark/engine_input_renderer.py)。后台在解码尚未结束时逐片段提交官方异步 renderer，最多 4 个待渲染片段；处理后的 EngineInput 经版本固定的 vLLM 0.19.0 `_render_and_run_requests` 交给原 engine，保留原请求顺序、参数、分批、窗口检查点及输入证据。每组预估最多 12,288 帧；超过时整组回退到原串行输入路径，不裁帧、不减少窗口，并记录 `renderer_skip_reason=group_frame_memory_limit`。该上限用于限制归一化视频张量的 RAM，占用并非只看 GPU 显存。

真实后台线程验证：[复验脚本](validate_engine_renderer_bridge.py) · [结果](engine_renderer_bridge_cpu_parity.json)。串行处理缓存 4GB 与并行处理缓存关闭时，实际 token、张量、时间与占位布局完全相同；只有到达时间及 2 个奇数帧窗口的缓存键不同。已定位到安装版 HuggingFace `Qwen3VLProcessor._calculate_timestamps` 会原地追加最后一个帧索引，而 vLLM 缓存开关使哈希计算分别发生在预处理之前／之后。验证直接计算补帧前后键值，精确解释该差异。输入内容未改变，也不强行把缓存键伪造成相同。

桥接层仅复制元数据后交给 HF，避免它的补帧操作修改原始证据；帧数组继续共享。两路的原始输入均未被修改，线程及事件循环关闭也通过验证。新增 6 项桥接检查及 17 项批处理／续跑检查通过。

生产单卡请求见 [engine_renderer_canary_requested.json](engine_renderer_canary_requested.json)：GPU7 原 worker 完成当前组后安全退出，由守护程序启动 revision 6；其余卡保持 revision 5。当前 `gpu_worker_revision` 目标仍为 5，`renderer_canary_gpus=[7]`，尚未推广到全部卡。实际首批结果及持续吞吐还需验证，不能把 CPU 试验当成生产提速结论。[代码快照](code_engine_renderer_canary/SHA256.json)。

精确裁剪的点头分支另有约 892 秒固定 48 PID 的逐片段计数观测，新增 3,050 段，约 3.42 段/秒：[fast_nod_worker_rate.json](monitor_snapshots/fast_nod_worker_rate.json)。该数不受来源整组结算影响，但仍非相同样本的新旧受控对照。


## 后台并行模型预处理与大来源批次限制（单卡验证中）

新增 `engine_input_renderer.py`，使用已安装的 vLLM 0.19.0 原生 renderer，在后台两路准备 EngineInput，与解码及上一组 GPU 推理重叠。模型、像素、8 fps、全部窗口、prompt、输出上限和采样设置不变。启用时关闭 CPU 多模态处理缓存，仍保留前缀缓存。版本和结果映射均校验，过大的预处理组回退到原路径，不丢帧或窗口。

真实处理器验证：除到达时间与缓存键外，token、像素张量和多模态位置完全一致。奇数帧时，原处理器会补最后一个帧索引；缓存开关改变了哈希发生在补帧前还是补帧后，因此部分缓存键不同，已按实际规则核实，不能宣称所有字段完全相同。新路径复制时间元数据，原始输入证据不会被原处理器原地修改。[CPU 一致性](engine_renderer_bridge_cpu_parity.json)。

CPU 同输入交替测速的中位数为串行 2.81 秒、两路并行 1.72 秒，仅代表该预处理步骤。[测速](renderer_parallel_cpu_benchmark.json)。GPU7 前两组共 149 段、808 次响应，全部结构有效、无截断：[首组](engine_renderer_canary_result.json)、[第二组](engine_renderer_second_group.json)。不是标注准确率证明。

随后六分钟固定 PID 观测，GPU7 约 0.536 段/秒，另外六张正常推进的旧版卡平均约 0.489 段/秒；GPU1 因大来源准备无新增产出，单独列出，不把它计入上述正常卡均值。样本不同，不能当作受控端到端提速倍数。[观测](monitor_snapshots/renderer6_live_window.json)。CPU 平均约 105/160 核，无 OOM。

发现 GPU1 一次取到含 327 段的来源，旧逻辑允许来源把 50 段准备上限撑大，导致 GPU 长时间等整组准备。`bounded_claim_groups` 现在严格每组至多 50 段，同一来源的锁跨组持有，只有最后一组才检查该来源的全部输出并写完成凭证。停止后释放未完成来源，恢复时跳过已保存片段；不改片段、时间或标注内容。

新增四项回归覆盖 327 段拆分、来源完成时机、预取异常释放、停止与断点恢复；已有队列 8 项和批处理 17 项均通过。该改动为 GPU worker revision 7，先在 GPU7 验证；全局目标版本仍为 5，避免尚未验收就自动推广。执行版本以 worker 心跳为准。[候选代码本地快照](code_bounded_groups_candidate/SHA256.json)。


### Revision 7 验收与逐卡推广

[首批 50 段验收](bounded_groups_canary_result.json)：263 次响应中 262 次结构有效、1 次模型重复描述导致 4096 token 截断；原文和无效状态保留。所有来源、窗口、8 fps、prompt 与纯视觉协议检查通过。不能将处理完成当成所有响应有效。

跨组来源 `improvised_V00_S0876_I00000384` 首组完成 7/10 时没有完成凭证；下一组完成余下 3 段后才提交 10/10 凭证，10 段共 50 个窗口全部有效：[接续检查](monitor_snapshots/bounded_source_continuation.json)。GPU7 已连续产出 100 段，准备和推理组均为 50 段。

已将全局目标版本设为 revision 7，后台两路 renderer＋严格 50 段准备组。由现有监督程序逐卡更新，等待新卡实际产出后再更新下一张；其余卡继续生产。[推广记录](monitor_snapshots/revision7_promotion.json)。现行推理参数仍为 batch512／32768 token／16384 frame budget；没有降低采样帧率或输出上限。

本次生产代码合并快照：[code_revision7/SHA256.json](code_revision7/SHA256.json)，包含 worker、监督、队列、renderer、裁剪、帧共享和相关回归测试；这是本地备份，未推送 GitHub。


### 推广期间的来源接续和第二张卡验收

旧版 GPU1 等待较久的 327 段来源现已全部保存并提交完成凭证，无漏段或来源错配；2,124 次响应中 2,122 次结构有效，含 1 次截断，未把无效记录改为有效。原批次完整解码耗时 839 秒、输入准备约 36 秒，进一步确认严格限制准备组大小的必要性。[该来源验收](monitor_snapshots/large_source_327_completed.json)。

推广卡 GPU0 的首批 50 段、234 次响应全部结构有效，无截断，来源、完整窗口和执行参数检查通过。[第二张卡检查](monitor_snapshots/revision7_gpu0_result.json)。监督程序随后才启动 GPU1 更新；逐卡真实进程身份和监督／守护进程检查记录在 [revision7_rollout_live.jsonl](monitor_snapshots/revision7_rollout_live.jsonl)。该记录不是所有卡已经更新的声明。

Revision 7 单卡固定 PID 的约 411 秒窗口新增 250 段，约 0.608 段/秒，无 OOM：[实测窗口](monitor_snapshots/revision7_first_rate.json)。这是不同来源连续处理、其他卡仍在更新时的观测，不是受控新旧加速倍数，也不能外推为八卡已经达到的总吞吐。


### 部分推广时的供料检查

[有限输入耗时抽查](monitor_snapshots/revision7_pipeline_spotcheck.json)仅读取更新卡当前来源的 6 个结果文件，覆盖 3 个批次：后台准备约 66–77 秒，抽到的单片段解码锁等待约 0.01–0.84 秒。该抽查不是全量平均值，不据此单独判定全机瓶颈。

[40 秒资源测量](monitor_snapshots/revision7_partial_rollout_cpu.json)中，8 个 GPU worker 和 48 个点头 worker 的真实进程身份均通过检查，CPU 平均约 91.6/160 核、I/O pressure 很低、无 OOM；当时仍有卡处于新版首批准备，不把它当作全八卡稳定运行的 CPU 基线。后续并发试验应等全八卡版本一致且有固定 PID 稳定窗口，再对比解码并发变化；当前未提高每卡 8 路／共享 80 槽。


### 固定进程的运行窗口测量

新增只读脚本 [measure_runtime_window.py](measure_runtime_window.py)，在 node17 运行。它记录 8 个 GPU 与配置数量的 nod worker 的真实 PID／启动标识、心跳对应关系、版本、已生效解码参数、各进程片段计数、CPU 使用量和采样 GPU 利用率。每 30 秒采样，不修改参数、不重启 worker、不扫描结果目录。

存在滚动更新、尚未首批产出、进程身份变化、心跳错误、版本／运行参数不一致或 OOM kill 时，该窗口不会产生可用于比较的吞吐值。五项回归覆盖固定进程计数、重启后计数误判、中途进程退出、部分参数未生效和完成时间窗归属等情形。代码快照：[code_runtime_measurement/SHA256.json](code_runtime_measurement/SHA256.json)。

待八卡全部 revision 7 且均产出后，可采集基线（tag 唯一，已有报告不会覆盖）：

```bash
python3 experiments/listen_j_annotation_node17_20261029/measure_runtime_window.py \
  --tag revision7_decode8_slots80_baseline --seconds 600
```

该结果是持续生产的实测速度，输入片段不同，不能称为相同样本受控加速倍数。更高并发只在稳定基线和资源余量确认后测试；当前仍为每 GPU 8 路解码、共享 80 槽、48 个点头 worker。

窗口结束后补充记录片段总时长、原始画面像素量、请求／输入 token／输出 token 量、结构有效数与截断数。只读取窗口起始在途来源及该窗口队列日志中新增领取的来源，不扫描整个结果目录；按结果完成时间归属，排除窗口外结果并去重。该信息用于识别片段长度／输出量变化造成的吞吐偏差。

工作量收集器已用真实生产窗口验证：[runtime_workload_smoke.json](monitor_snapshots/runtime_workload_smoke.json)。读取 60 个相关来源中的 272 个已存在结果，时间窗内 200 段，与 worker 增量一致；不读取其他来源。因当时仍在更新，测量明确标为不可用于稳定吞吐比较，未将短窗口的批次集中落盘当作提速结论。

CPU 供料参数是动态读取的：worker/result 的 `execution.decode_workers`、`execution.decode_slots` 是该进程启动时的配置快照；热调整后的实际准备参数应查看心跳 `decode_workers_active`／`decode_slots_active`、`resource_tuning.json`、`selected.json` 及对应测速窗口。模型 batch／token／frame／memory 则仍以实际 worker execution 为准。后续热调整将保存调整时间与原配置；跨越调整的窗口不作为稳定对照。


### 更高解码并发的文件句柄前置检查

[8 个实际 GPU worker 的限制检查](monitor_snapshots/decode12_resource_precheck.json)：当前软上限 1,024，硬上限 1,048,576，已有约 638–725 个打开句柄（准备空档更低）。每路解码在等待共享槽时可能持有全部槽文件，因此 12×96 本身就需要 1,152 个句柄，不能直接增加并发。

已加入启动保护 `ensure_decode_fd_capacity`：在硬上限允许范围内优先把本进程软上限提高到 8,192，并验证解码锁加余量所需容量；硬上限不足则在开始解码前明确拒绝。隔离子进程实测从 1,024 提到 8,192，硬上限不变；硬上限只有 1,024 时正确拒绝 12×96。生产进程尚未因此重启，当前基线参数未改变。下一步热调整前还须对正在运行的本任务 worker 提高软上限，并记录实际 PID／启动标识与调整结果。[代码快照](code_fd_capacity/SHA256.json)。


### 完整八卡基线与 12 路／96 槽试验

[十分钟稳定基线](monitor_snapshots/revision7_decode8_slots80_baseline.json)通过全部进程、版本、运行参数和监督检查：600.54 秒，worker 增量为 VLM 2,550 段／4.246 段每秒、nod 2,276 段／3.790 段每秒；采样 GPU 平均利用率 58.58%，CPU 平均 110.53/160 核。按原始结果精确完成时间统计为 2,534 段、15,221.64 秒视频、13,498 次请求，其中 13,479 次结构有效、4 次截断。worker 计数与文件时间窗的少量差异来自心跳采样时点，不能把二者混作同一统计口径；原始无效响应继续保留。

确认资源余量后，已验证 PID／启动标识和命令身份，仅将本任务 8 个 GPU worker 的文件句柄软上限提高到 8,192，硬上限不变：[调整记录](monitor_snapshots/decode12_fd_limits.json)。随后把 `resource_tuning.json` 与 `selected.json` 统一为每 GPU 12 路解码、共享 96 槽，nod 数量仍为 48；模型、全部窗口、帧率、prompt 和输出上限保持不变，未重启模型。

[调整前配置](monitor_snapshots/decode12_trial_before.json) · [应用记录](monitor_snapshots/decode12_trial_applied.json) · [参数生效与预取过渡检查](monitor_snapshots/decode12_adoption.json)。每张卡确认实际读取新参数后，再等累计新增 100 段（两个 50 段批次）才开始候选窗口，以排除旧预取组。此时只是试验中，尚未根据完整候选窗口选择最终并发；候选报告将保存为 `monitor_snapshots/revision7_decode12_slots96_trial.json`。

热调整后已核实八卡实际使用 12／96、所有原 PID 保持存活、软上限均为 8,192，多个 worker 打开句柄达到 1,205–1,244 个，超过原先 1,024 的软限制；无 OOM 或来源任务失败：[实际运行检查](monitor_snapshots/decode12_live_resource_check.json)。这证明增加并发前的句柄调整是必要条件，不代表候选方案已经通过速度比较。

### 12 路试验结果与原配置复测

[12 路／96 槽完整窗口](monitor_snapshots/revision7_decode12_slots96_trial.json)通过进程与参数稳定性检查：601.32 秒，VLM 增量 2,600 段、4.324 段/秒；nod 增量 2,220 段、3.692 段/秒。GPU 采样平均利用率 59.11%，CPU 平均 111.35/160 核。相比 8／80 基线，视觉片段吞吐约高 1.8%，nod 约低 2.6%，尚不能认定整体提速。

按结果完成时间统计，候选窗口有 2,650 段、15,883.11 秒视频，14,088 次响应中 14,078 次结构有效、5 次截断；原始帧平均像素量约 381 万，基线约 361 万。两个窗口处理不同片段，token 量也不同，不能把该差异当作相同输入下的加速或质量提升。

已热恢复每 GPU 8 路解码／共享 80 槽，模型与 48 个 nod 进程继续运行，文件句柄软上限保留 8,192。再次等待八卡全部采用配置并各完成两个后续准备组，再采集十分钟复测，检查来源变化带来的波动：[切换记录](monitor_snapshots/decode8_repeat_applied.json)、[过渡检查](monitor_snapshots/decode8_repeat_adoption.json)。复测报告目标为 `monitor_snapshots/revision7_decode8_slots80_repeat.json`；未完成前不宣称已经选定最优 CPU 并发。

### 连续补入请求的单卡候选代码（尚待 GPU 验证）

[引擎日志抽查](monitor_snapshots/decode12_engine_tail_samples.json)只读取每卡日志尾部 80 KB，并限制在 12／96 测速窗口：各卡多次出现仅剩 1–4 条运行请求、等待队列为空。这是批次长尾的证据，日志采样数量不能直接换算为空闲时间比例。单纯增加 CPU 解码并发无法消除等待整批返回的同步屏障。

新增 [rolling_inference.py](../../src/listen_benchmark/rolling_inference.py) 候选路径：最多两个标注组加一个 CPU 预取组；实际未完成请求总数和对应输入帧数仍分别不超过 512／16,384。引擎只由主线程调用；两个标注线程复用已有 `annotate_group` 的原文保存、窗口检查和断点逻辑。按外部请求 ID 归还结果，避免混淆 vLLM 随机化的内部 ID。来源组仍按领取顺序落盘和提交完成凭证，防止后一个来源分块提前结束。

该路径由 `rolling_inference=true` 开启，可用 `rolling_canary_gpus=[7]` 限定单卡；开启卡报告 revision 8，其余卡保持 revision 7。当前配置尚未启用，等待完整 CPU 并发复测结束。此方案没有减少窗口、帧率、输出长度或改动 prompt，也不以截断长回答换速度。模型输出的正确性仍须后续验收，不能由调度测试推出。

本机与 node17 实际 Python 环境均通过 35 项 CPU 回归，覆盖请求乱序归还、长尾期间补入、容量上限、停止和异常退出、原文／窗口对应、截断保留、断点续跑及来源队列。代码快照：[code_rolling_canary/SHA256.json](code_rolling_canary/SHA256.json)。这是本地备份，GPU 试验尚未开始。

更新：8／80 的第二个完整窗口已结束，VLM 4.163 段/秒、nod 3.835 段/秒，全部进程和参数稳定性检查通过。[三轮比较](monitor_snapshots/decode_concurrency_comparison.json)显示 12／96 比两轮 8／80 的加权视觉吞吐高约 2.8%，nod 低约 3.2%；视频来源不同，该幅度不能证明受控提速。暂时保留 8／80，优先验证上述批次长尾优化。

已仅为 GPU7 开启 rolling 候选，向本任务核实过 PID／启动标识的旧 worker 发送 SIGTERM，使其完成当前批次后由现有监督器接续；其余七卡不切换。[切换前](monitor_snapshots/rolling_canary_before.json)、[应用记录](monitor_snapshots/rolling_canary_applied.json)、[实际进程观察](monitor_snapshots/rolling_canary_live.json)。候选出现明确 worker 错误时，观察脚本会关闭此单卡候选并交回原路径；观察超时本身不触发重启。等待新进程至少产出 100 段后再检查原始窗口与来源，不把模型加载成功当成打标验收通过。

### Rolling 候选首批输出验收

GPU7 新进程 PID 1877091 已完成前 100 段：[原始结果验收](monitor_snapshots/rolling_canary_first100_audit.json)。544 次响应全部结构有效，无截断；逐片段来源、完整窗口、prompt、8 fps、512 裁剪、原始帧证据及来源完成凭证检查均通过。检查程序只读取队列时间窗涉及的 126 个来源，没有扫描全量结果树；代码为 [audit_rolling_canary.py](audit_rolling_canary.py)。该检查不是人工语义准确率评估。

对首批中的一段再次独立读取原始 listener 视频，重新裁剪、采样、JPEG 编码与构建 11 个窗口，帧哈希、PTS／时间元数据、prompt 与渲染后的完整提示文本全部精确一致：[独立解码核对](monitor_snapshots/rolling_canary_input_redecode.json)。没有重跑模型推理。

前两批还没有发生推理期间补入，下一批 CPU 输入尚未就绪是当时的限制；首批准备耗时约 108.49 秒，其中也包含新进程首次输入准备。当前只能确认候选路径和保存结果通过检查，不能据此宣称提速或推广到所有 GPU。其余七卡与 48 个 nod worker 持续生产，后续须以稳定窗口与实际补入次数决定是否推广。

更新：候选已连续完成 200 段，观察到 2 次引擎仍有未完成请求时补入新批次，实际峰值 330 请求／9,488 帧，低于既定 512／16,384 上限。已开始排除首批预热后的十分钟固定进程窗口，报告目标为 `monitor_snapshots/rolling_canary_stable_window.json`，分别记录单卡候选与七卡原路径的完成数和视频／token 工作量；在完整报告出来前不认定提速。

测量脚本现支持显式指定单卡版本，例如 `--revision 7 --worker-revision gpu-7=8`。它仍拒绝其他卡版本变化、进程重启和推理路径中途切换；另外保存逐卡吞吐及原路径／rolling 路径分别处理的视频和 token 工作量。7 项回归通过，快照见 [code_runtime_measurement_canary/SHA256.json](code_runtime_measurement_canary/SHA256.json)。该扩展不修改当前正在运行的测量进程或生产配置。

### 连续补入后的检查及第二张试验卡

[后续 200 段检查](monitor_snapshots/rolling_canary_overlap_audit.json)覆盖首批之后已发生实际补入的阶段：1,141 次响应，1,140 次结构有效，1 次事件区间无效、无截断；来源、完整窗口与输入规格检查通过，原始无效记录保留。相关来源完成凭证均在其全部片段保存后存在。检查支持 `--skip`／`--count`／`--tag`，不会覆盖已有报告。

[十分钟单卡稳定窗口](monitor_snapshots/rolling_canary_stable_window.json)中，GPU7 完成 350 段、约 0.583 段/秒；其余七卡合计 2,000 段、平均每卡约 0.476 段/秒。该窗口内候选处理 2,192.86 秒视频、1,962 个请求、395,929 个输出 token，其余七卡合计 11,984.69 秒视频、10,613 个请求、1,979,857 个输出 token。按每卡计算，候选的视频时长处理速率约高 28.1%，输出 token 速率约高 40.0%；候选的平均片段更长、原始像素量也略高。全部进程与运行参数保持稳定，无 OOM。[工作量比较](monitor_snapshots/rolling_first_window_comparison.json)。这是不同来源的生产观测，不是相同输入下的因果加速倍数；也不以结构有效比例宣称语义准确率提升。

因此只增加 GPU0 作为第二张候选卡，配置为 `rolling_canary_gpus=[0,7]`，全局目标版本暂时仍为 7，其余六卡继续原路径。对核实的本任务 GPU0 进程请求完成当前批次后退出，由监督器接续；[切换记录](monitor_snapshots/rolling_gpu0_applied.json)。新 worker 的结果增加 `execution.worker_id`／`worker_pid` 来源字段，便于用 `--worker gpu-0` 单独审阅。准备在新卡产出 100 段时检查、200 段后再测十分钟，报告目标为 `monitor_snapshots/rolling_two_card_stable_window.json`。未完成此验证前，不宣称已推广到全部八卡。

GPU0 的前 100 段已独立通过检查，447 次响应全部结构有效，无截断或来源／窗口错误：[GPU0 首批验收](monitor_snapshots/rolling_gpu0_first100_audit.json)。它实际完成 250 段、发生连续补入后，已启动两卡稳定窗口；此窗口内继续保持现有参数。

### 后续可选的 decode-only CUDA Graph 准备（关闭状态）

已确认当前 vLLM 0.19.0 支持 `CompilationConfig(mode=0, cudagraph_mode="FULL_DECODE_ONLY")`，并在禁用 CUDA 设备的配置检查中验证构造成功。准备的捕获大小为 `[1,2,4,8,16,32,64,128]`；模型的最大并发仍为 512、token 预算仍为 32,768。只对纯生成批次捕获图，不编译或改变视觉编码／混合预填充路径。拟从显存比例 0.80 的单卡候选开始留出图捕获空间；是否实际安全、是否提速仍需 GPU 实测，当前没有启用。

共用配置入口为 [execution_profile.py](../../src/listen_benchmark/execution_profile.py)，通过 `config.json.gpu_execution_overrides["7"]` 等指定某张卡的 `memory`、`eager` 与 `decode_graph_capture_sizes`，拒绝修改帧率等数据语义。worker 与监督程序使用同一目标配置，避免覆盖参数被监督程序误认为异常而反复重启。候选代码、监督匹配与已有 rolling 回归共 17 项检查通过，[代码备份](code_decode_graph_preparation/SHA256.json)。图配置启用的 worker 将报告 revision 9；现行 worker 仍为 7／8。

**部署前置条件：当前正在运行的监督进程尚未重载这些配置解析代码。必须等当前稳定测速窗口结束，先让监督进程安全重载并确认原 worker 全部保持存活，才能设置任何单卡执行覆盖。**不得直接向旧监督进程写入显存覆盖；当前 `gpu_execution_overrides` 未设置，没有启动图捕获试验。

### 第二轮结果与 CPU 供料试验

[两卡稳定窗口](monitor_snapshots/rolling_two_card_stable_window.json)已完成且通过进程、版本和运行参数检查。GPU0 完成 350 段（0.583 段/秒），GPU7 仅 150 段（0.250 段/秒）；六张原路径卡合计 1,900 段（平均每卡 0.527 段/秒）。因此没有直接推广到全部八卡。

[按来源与 worker 的核查](monitor_snapshots/rolling_two_card_workload_diagnosis.json)显示，GPU7 这 150 段全部来自 2160×3840 原片，平均 7.02 秒，每组输入准备耗时中位数 181.60 秒；GPU0 原片平均约 520 万像素、准备 83.19 秒；六张原路径卡平均约 404 万像素、准备 76.16 秒。三者解码锁等待中位数都只有约 0.1 秒。该窗口输入差异很大，不把低片段吞吐直接归因于 rolling；更明显的限制是重原片的输入准备速度。

监督程序已在窗口结束后安全重载：PID 1286692 → 4065359，原来 8 GPU＋48 nod 的全部 56 个真实进程身份保持不变，八小时截止时间也保持不变：[重载验证](monitor_snapshots/supervisor_profile_reload_verified.json)。前述“尚未重载”的前置步骤已完成，但 CUDA Graph 配置仍关闭。

随后热调整到每 GPU 16 路解码、全机共享 112 槽，并把 nod worker 数从 48 增到 64。模型参数、窗口、输入帧率、原始描述处理不变，GPU worker 不重启，仍只有 GPU0／7 使用 rolling。调整前检查了 CPU 160 核配额和 8 个 GPU worker 的 8,192 文件句柄软上限，足以覆盖 16×112 槽加余量。当前 CPU 并发仍是候选，不是已证明最快配置。

[调整前状态](monitor_snapshots/cpu16_nod64_before.json) · [应用记录](monitor_snapshots/cpu16_nod64_applied.json) · [生效／预取过渡检查](monitor_snapshots/cpu16_nod64_adoption.json)。等待八卡全部生效并各再完成两个组，且全部 64 个 nod worker 均产出至少 3 段，再开始十分钟稳定窗口，报告目标为 `monitor_snapshots/cpu16_slots112_nod64_stable_window.json`。只读测量器能记录新 worker 尚未写首条心跳的状态，不会因短暂缺少文件而误报进程退出。

当前代码与运行配置快照：[code_cpu16_trial/SHA256.json](code_cpu16_trial/SHA256.json)。相关 50 项回归检查通过；这是本地备份，不是 GitHub 推送记录。


### CPU 并发窗口完成；单卡 CUDA Graph 实测开始

[16／112、64 nod 稳定窗口](monitor_snapshots/cpu16_slots112_nod64_stable_window.json)完成 600.54 秒：VLM 2,250 段（3.747 段/秒），nod 2,382 段（3.966 段/秒），平均 GPU 利用率 49.1%，CPU 用量约 119.6 核；进程、配置均稳定，无 OOM。视觉速度未优于前一窗口，nod 较快，但来源不同，不能归因于单一参数。当前保留此配置作为下一项单变量试验的背景，不宣称为最优。

已于 2026-10-29 22:30 UTC 仅为 GPU7 配置 decode-only CUDA Graph（显存比例 0.80，捕获大小 1–128，最大请求数仍 512）。原 GPU7 完成当前工作后由已更新的监督器接续；其余七卡继续原配置。记录见 [切换前状态](monitor_snapshots/decode_graph_canary_before.json)。首次至少 100 段输出通过来源、完整窗口、8 fps 与原文保留检查后，才启动 600 秒稳定测量。worker 在验收前反复退出或首次输出超时将回退该单卡覆盖；没有改变打标语义。

测量器新增图捕获配置的一致性检查，并单列图执行的工作量，避免与另一张普通 rolling 卡混算；原有 7 项测量回归通过。当前只是已启动试验，尚无 CUDA Graph 提速结论。


切换中发现并修复停止时的部分预处理问题：旧 GPU7 在停止请求到达输入准备中途时产生混合输入，原先拒绝混用而退出；未完成来源通过已有队列与 partial 检查点续跑。现在只补做缺失输入的处理，已经处理好的输入及原始顺序保留，并保护帧时间元数据；同时在等待预取结束后再次检查停止状态，避免误提交一个已要求停止的预取组。首次加载图候选恰逢模块更新，因旧模块缺少新 helper 退出一次；监督器已启动完整加载新版代码的 worker。此部署问题不归为 CUDA Graph 性能或显存失败。

修复包含部分预处理时的来源顺序保护、缺失 renderer 拒绝、缺失处理结果拒绝以及停止边界回归。快照：[code_partial_render_recovery/SHA256.json](code_partial_render_recovery/SHA256.json)。运行中的其余 worker 保留已加载版本，后续正常接续时采用修复，不为此同时重启全部 GPU。

更新：node17 实际 vLLM Python 环境的 rolling、visual schedule、renderer 共 37 项回归通过。最终修复快照为 [code_partial_render_recovery_verified/SHA256.json](code_partial_render_recovery_verified/SHA256.json)。GPU7 新进程 PID 666460 已进入真实推理；首批结果验收和稳定测速尚在等待完成。


### CUDA Graph 首批与四线程 CPU 预处理候选

[图执行首批 100 段](monitor_snapshots/decode_graph_first100_audit.json)完成来源、窗口、输入规格和原始响应保留检查：661 次响应，659 次结构有效，2 次无效（含 1 次截断），异常原文保留。模型结果不是人工准确率结论。GPU 日志显示图捕获增加约 0.16 GiB；稳定窗口见 `monitor_snapshots/decode_graph_stable_window.json`，未完成前不推广。

[首批输入成本](monitor_snapshots/decode_graph_first100_input_cost.json)：70 段 4K、30 段 1080p；两个 50 段准备组分别耗时约 102.46／107.72 秒。单段解码中位数 18.16 秒、等待解码槽 4.60 秒、CPU renderer 5.34 秒，各并行耗时不能直接相加视为流水线时间。[固定 PID 的十秒线程采样](monitor_snapshots/input_wait_probe.json)中 GPU0／4 父进程分别约 3.55／2.30 核，主要 futex 等待，没有观察到 D 状态磁盘阻塞；这只是短窗口，不能排除其他时刻 I/O 限制。

据此准备每 GPU 四个 CPU renderer 的单卡候选；当前生产仍为两个。profile 只增加 `renderer_workers` 的 0／2／4 白名单，worker 与监督器共享匹配逻辑；四线程版本报告 revision 10。输入数组、预取数量、最大待渲染片段数、缓存策略及原文输出规则不变。51 项检查在 node17 实际环境通过，备份：[code_renderer4_preparation/SHA256.json](code_renderer4_preparation/SHA256.json)。启用前需等稳定窗口结束、安全重载监督器并核对原 72 个 worker 存活；不要直接向旧监督器写入新覆盖字段。


### 解码线程竞争：相同输入对照与进程候选设计

[原始解码定位](monitor_snapshots/slow_source_decode_probe.json)核对同一个片段：生产曾记录 51.72 秒；独立 CPU 进程重读为 9.16 秒，打开文件 0.16 秒、seek 0.0006 秒，实际解码 203 帧、前滚 139 帧，定位正确。因此未发现 seek 失效，不更改时间定位策略。

[8 并发、16 段、线程／进程 ABBA 对照](monitor_snapshots/decode_executor_probe.json)：线程两轮 41.07／37.00 秒，进程两轮 28.63／30.06 秒，全部帧哈希和时间元数据一致。均在生产运行时使用相同代码与原始输入，仅比较 CPU 解码，不把约 25% 的解码耗时降低等同于端到端提速。

下一步仅为 GPU7 准备 `spawn` CPU 进程池解码候选，其他 GPU 保持线程池。子进程禁用 CUDA、保持单线程解码和原裁剪／采样／JPEG 代码；沿用全机解码槽限制。子进程把媒体写入本任务原有受信任临时缓存，父进程校验样本与缓存位置后读取，避免通过进程管道复制大数组。进程池随 GPU worker 关闭；损坏、缺失或不匹配的结果必须报错，由已有断点与监督恢复。只改变 CPU 执行方式，不修改原始标注或输入语义。四线程 renderer 候选暂不启用。


进程解码实现为 [process_decode.py](../../src/listen_benchmark/process_decode.py)，CPU 子进程采用 `spawn`、清空 CUDA 可见设备，原始媒体经已有可信临时缓存传回。56 项回归在本机和 node17 通过；[实际后台线程启动／缓存回传验证](monitor_snapshots/process_decode_parity.json)覆盖 16 段的原始帧与时间元数据，以及首段逐字节像素一致性，临时测试缓存已清理。

生产候选仅给 GPU7 增加 `decode_backend=process`，其余执行配置（显存 0.80、CUDA Graph、2 个 renderer、16 路解码和 112 共享槽）保持不变，版本为 revision 11。监督器需安全重载后匹配新字段；[重载前](monitor_snapshots/process_decoder_supervisor_reload_before.json)、[重载验证](monitor_snapshots/process_decoder_supervisor_reload_verified.json)、[候选应用](monitor_snapshots/process_decode_canary_applied.json)。前 100 段还需检查每条记录的 CPU 子进程来源，再测十分钟。未获得生产结果前不宣称端到端提速。

代码备份：[code_process_decode_canary/SHA256.json](code_process_decode_canary/SHA256.json)。四线程 renderer 仍未启用。


更新：监督器已安全重载至 PID 2450670，原 72 个 GPU／nod worker 身份不变，覆盖截止仍为 1793318252。随后仅 GPU7 正常接续到 revision 11（PID 3136659），旧 worker 完成批次后退出，未再出现部分预处理停止错误。[子进程核查](monitor_snapshots/process_decode_child_live.json)确认 16 个 CPU 解码进程，GPU 上仅有八个模型引擎，无 OOM。

[进程解码首批 100 段验收](monitor_snapshots/process_decode_first100_audit.json)：519 次响应全部结构有效，无截断；来源、窗口、原文保留和独立 CPU 子进程证据检查通过。这不代表人工语义准确率。于约 23:11:37 UTC 开始固定进程十分钟测量，报告为 `monitor_snapshots/process_decode_stable_window.json`；测量期间不修改生产配置，尚未推广其余 GPU。


### 进程解码生产窗口完成；第二张卡接续

[完整生产对比](monitor_snapshots/decode_backend_production_comparison.json)：相同 GPU7、相同图执行配置，线程解码窗口完成 350 段／1,948.33 秒视频，进程解码窗口完成 450 段／2,816.10 秒视频。后者平均片段更长，4K 占比从 39/350 上升到 198/450，且两轮都没有帧缓存命中；准备组耗时中位数从 85.79 降至 70.72 秒。观察到片段吞吐约高 28.6%、视频时长吞吐约高 44.5%，但生产输入不同，不能当作同输入端到端加速倍数。

进程解码窗口全机视觉 4.163 段/秒、nod 4.188 段/秒，CPU 约 124.93/160 核，无进程／配置变化或 OOM。结合相同输入 CPU ABBA 试验，支持继续逐卡验证，暂未宣称全部八卡配置最优。

已仅将 GPU7 的已验证组合复制到 GPU0（显存 0.80、decode-only CUDA Graph、连续补入、2 renderer、独立 CPU 解码），其余六卡不变：[GPU0 切换前](monitor_snapshots/process_gpu0_before.json)。等待新 GPU0 的 100 段验收后，再做 300 秒两卡稳定窗口，目标报告 `monitor_snapshots/process_two_card_stable_window.json`。首次验收新增 `--queue-snapshot`，只读取本次切换时未完成来源与后续队列片段，避免重复读取数小时之前的结果；仍执行相同来源、时间窗口和 CPU 子进程检查。监督窗口不延长或重置。

新增分析与验收工具备份：[code_second_process_trial/SHA256.json](code_second_process_trial/SHA256.json)。


### 已请求将已验证组合逐卡推广到八卡（尚未全部生效）

[GPU0 首批验收](monitor_snapshots/process_gpu0_first100_audit.json)覆盖 100 段、620 次响应，全部结构有效、无截断，来源和 CPU 子进程证据通过。新的队列快照把读取范围限制为 166 个来源。

[两卡 300 秒稳定窗口](monitor_snapshots/process_two_card_stable_window.json)中，全机 VLM 4.160 段/秒、nod 4.133 段/秒，CPU 127.44/160 核，无进程／参数变化或 OOM；两个新配置 worker 共完成 300 段、2,008.08 秒视频，六个原配置 worker 共完成 950 段、5,950.14 秒视频。窗口较短且输入不同，不据此声称固定的两卡加速倍数，但未观察到节点资源挤占。

基于同输入 CPU 对照、GPU7 完整生产窗口和两卡稳定性，已请求其余六卡采用同一组合：batch512、32768 token、16384 frame budget、50 段来源组、连续补入、memory=0.80、decode-only CUDA Graph（最大捕获128）、2 renderer、独立 CPU 进程解码。每 GPU 16 解码并发、全机112槽、64 nod 保持不变。

[推广前状态](monitor_snapshots/process_all_rollout_before.json)保留原配置与全部72个 worker 身份。监督器仍一次只切换一张卡，前一个新 worker 产出后再继续；[推广观察脚本](observe_all_process_rollout.py)为每张新卡保存自己的队列快照，待至少150条新增产出后验收其中100条，再启动全八卡600秒稳定窗口。配置写入不等于全部已生效：看 `monitor_snapshots/process_all_rollout_status.json` 和实际 worker 状态。观察器异常会写明错误，不会因为观察超时停止生产。

备份：[code_full_process_rollout/SHA256.json](code_full_process_rollout/SHA256.json)。八小时截止仍为 23:57:32 UTC，未延长或重置，未完成打标不会到时自动终止。


### 末帧可变时长校验修复与八小时节点检查

来源 `improvised_V01_S0354_I00001171` 中一段 6.645 秒片段被旧校验误拒：最后一帧的实际显示区间覆盖到 6.70 秒，超过片段终点约 55 毫秒；旧校验把超过一个平均帧周期的尾端覆盖当成缺帧。该数据没有缺少末尾画面。现在仅检查是否缺少终点覆盖，保留原来的一个平均帧周期容差，不裁改帧、不改变采样或时间戳。

新实现 [visual_media_coverage.py](../../src/listen_benchmark/visual_media_coverage.py) 保持冻结版原始帧构造；原 `visual_native8.py` 及缓存身份未修改，避免运行中改变缓存键。[真实帧验证](monitor_snapshots/vfr_coverage_parity_and_recovery_cache.json)确认问题片段的54帧与原构造完全一致，并对正常对照片段验证元数据和像素相同。59项相关回归在 node17 的实际 Python 环境通过。

仅将这个失败来源重新加入队列，其他已完成结果保留。[重排记录](monitor_snapshots/vfr_source_requeued.json)保留旧失败与三次尝试；修复后的完整结果和来源提交记录均已生成，原 `input_errors` 记录附加解决信息，没有删除历史错误。

[CPU进程刷新](refresh_cpu_decoders.py)临时将16路调为17路，以使旧的 CPU 进程池载入修复；各目标 worker 再产出100段后恢复16路，1200秒观察超时也恢复。模型 worker 不因此重启。该操作只是代码更新传播，不是额外吞吐试验；完成状态见 `monitor_snapshots/cpu_decoder_refresh_complete.json`。

[八小时节点检查](monitor_snapshots/eight_hour_coverage_checkpoint.json)在原截止时间之后确认监督器、守护进程及72个 GPU／nod worker 均存活，两个队列没有未恢复失败；VLM已提交103,253段，nod已提交91,030段。八小时并不等于全量完成，也不代表每张卡一直100%利用；正在进行最后两卡逐卡切换与后续全卡稳定测速，生产不会到时停止。

本次代码快照：[code_vfr_coverage_recovery/SHA256.json](code_vfr_coverage_recovery/SHA256.json)，为本地备份。

更新（2026-10-30 00:04 UTC）：[CPU进程刷新完成](monitor_snapshots/cpu_decoder_refresh_complete.json)，并发配置已恢复16／112。[新产出证据](monitor_snapshots/coverage_validator_live_outputs.json)只读取112个已知来源，验证GPU0／1／2／7旧CPU池以及GPU3／4／5均有携带新版校验标记的真实结果；未扫描全量结果目录。GPU6正在最后一次逐卡接续，全卡稳定窗口尚未完成，不能把当前切换阶段的吞吐当作最终结论。

GPU5扩展验收：100段／586个窗口响应，其中584个结构有效、2个整段响应截断；[原文核查](monitor_snapshots/process_gpu5_truncation_review.json)显示同一来源的两段反复生成视线左右变化描述，达到既定4096 token上限。原始响应和invalid状态保留，局部窗口结果独立保留；通过输入／来源审计不等同于全部响应有效或语义准确。未为了提速降低输出上限或修改prompt。

全卡运行后的[10秒CPU分组抽查](monitor_snapshots/all_process_cpu_probe.json)：8个GPU父进程约16.07核、128个CPU解码子进程约62.03核、8个模型引擎约10.42核、64个点头worker约57.26核，已知进程合计约145.77核，接近160核配额。只按已知父子进程采样，没有扫描全系统进程；短窗口不代表持续平均值，暂不据此增加解码并发，等待完整稳定窗口。

八张卡均已完成新配置首100段输入验收：[合并审计记录](monitor_snapshots/process_eight_card_input_audit_summary.json)。共800个不同片段、4,164个窗口响应，4,158个结构有效；2个截断、4个零长度事件区间原样保留为invalid。不同时间与来源构成的部署抽查不能当作全量准确率。全8卡十分钟固定配置测速已于约00:11:30 UTC启动，输出路径 `monitor_snapshots/process_all_card_stable_window.json`。


### 全8卡稳定结果；点头80路容量试验

[全8卡600.55秒稳定测量](monitor_snapshots/process_all_card_stable_window.json)完成：视觉3,100段（5.162段/秒）、点头2,390段（3.980段/秒）；平均GPU利用率60.73%、CPU用量135.18/160核。全部进程身份和参数稳定，没有OOM。视觉处理18,669.73秒视频、16,548个窗口响应，其中16,518个结构有效、15个截断；其余无效记录同样保留。不把不同来源的生产窗口比较当作固定加速倍数，也不把结构有效率当作语义准确率。

[引擎压力抽查](monitor_snapshots/all_card_engine_log_pressure.json)收集窗口后半程334条周期日志，210条的运行请求超过128，7条超过256；KV占用中位数85.95%，147条超过95%，未发现日志报告preemption或OOM。该周期记录不能确定每一步是否用了CUDA Graph，暂不据此改动图捕获或显存参数。

点头已成为较慢分支，因此在全卡窗口结束后将 `nod_workers` 从64调到80，视觉8卡、每卡16路解码和112共享槽均不变。当前80只是候选，尚无提速结论。[调整前配置与GPU身份](monitor_snapshots/nod80_before.json)、[观察脚本](observe_nod80_trial.py)保存恢复依据；新增16个worker每个至少产出3段后，自动启动600秒测量 `monitor_snapshots/process_all_nod80_stable_window.json`。观察器不因为等待超时停止生产。

若资源竞争导致整体吞吐下降，退回64路：先按现有监督恢复约定安全停止本任务监督与守护（GPU／原64个nod继续运行），恢复配置64，只向本次新增、已核对PID和启动标识的nod-64～79发送SIGTERM，让其完成当前片段并释放来源锁；确认退出后从持久监督状态移除这些额外子进程记录，保留原状态备份，再恢复守护。不得结束GPU、原64个nod或其他任务，也不得删除队列或结果来回滚容量。

另已准备[选择性裁剪实验脚本](probe_nod_selective_crop.py)，尚未执行媒体测试、未接入生产。它只在独立副本中跳过最终25Hz采样不用的源帧裁剪，保留解码、原始帧时间、末帧补用和x264参数。测试将对两个实际输入按旧／新／新／旧顺序比较完整媒体元数据和MP4字节哈希；必须等80路稳定窗口完成后才允许运行，避免CPU测试干扰本轮容量对照。即使通过，也仅证明这两个输入的等价性与渲染耗时，不能直接推广全量。快照：[code_nod_selective_crop_probe/SHA256.json](code_nod_selective_crop_probe/SHA256.json)。

测速器另增加点头队列偏移和来源负载记录，后续窗口将统计实际视频时长、分辨率、render／face耗时及无脸证据状态。只读取起始在途来源和窗口内队列片段，去重并严格过滤完成时间；不扫描全量结果树。旧窗口缺少该信息时明确返回available=false，不补造统计。node17的9项测量回归通过；当前已启动的80路窗口仍使用启动时载入版本，不因磁盘更新改变过程。快照：[code_nod_workload_measurement/SHA256.json](code_nod_workload_measurement/SHA256.json)。

80路点头试验完成：[64／80路对照](monitor_snapshots/nod64_nod80_production_comparison.json)。80路窗口600.60秒：视觉2,650段（4.412段/秒）、点头2,785段（4.637段/秒），CPU约140.44/160核、GPU均值61.5%，进程身份稳定、无OOM。较慢分支速度从3.980升至4.412段/秒，观察值约高10.9%；视觉输入平均长度从6.022增至6.138秒、平均原始帧像素从381万增至405万，且缺少旧窗口点头来源负载，不能当作同输入因果加速倍数。目前保留80路作为更均衡的生产配置，8卡视觉执行配置和16／112解码容量均不变。

在两轮稳定测量均结束之后才启动单个CPU裁剪对照，不与稳定窗口混用。实验仍没有接入生产。

[选择性裁剪实测完成](monitor_snapshots/nod_selective_crop_probe_analysis.json)：两个输入、各旧／新／新／旧四轮，MP4字节哈希及完整媒体元数据完全一致。4K样本旧版均值25.04秒、新版23.36秒，约少6.7%渲染耗时；1080p旧版首轮5.89秒、末轮4.17秒，新版两轮4.15／4.11秒，存在明显首轮或负载差异，不能把均值差当作通用提速。当前仍未部署该优化，后续需更多边界和真实输入覆盖。生产保持80路点头与现有8卡视觉配置。


### GPU7 图捕获上限256的单卡试验（进行中）

全卡128图捕获与80路点头窗口已完成，选择GPU7单独把捕获列表由1～128扩为1～256。batch512、32768 token、16384 frame budget、显存比例0.80、CPU解码16／112、80路点头和所有输入／输出规范均不变；其余7卡覆盖配置逐项验证无变化。周期日志中较多运行请求数超过128只是试验依据，不证明这些步骤都能使用图捕获，不能提前声称加速。

[切换前状态与目标](monitor_snapshots/graph256_before.json) · [只读观察脚本](observe_graph256_trial.py)。监督器让原GPU7完成已开始工作再接续，候选首100段通过来源／完整窗口验收后，再进行600秒稳定窗口 `monitor_snapshots/graph256_gpu7_stable_window.json`；本轮包含新增的点头输入负载统计。若候选退出、OOM或验证失败，仅把GPU7覆盖恢复为切换前配置，监督器逐卡恢复，保留全部结果和异常记录，不影响其他GPU／点头进程。

快照：[code_graph256_trial/SHA256.json](code_graph256_trial/SHA256.json)。选择性裁剪仍只是独立实验，没有部署进生产。

GPU7的256图捕获已成功，实际图池约0.24 GiB（128版本约0.16 GiB）。[首100段验收](monitor_snapshots/graph256_gpu7_first100_audit.json)共507个响应，506个结构有效、1个截断，输入／来源／窗口检查通过。[响应长度分析](monitor_snapshots/graph256_first100_tail_analysis.json)显示中位数120 token、p95为495、最长4096；含最长响应的263请求批次耗时163.67秒，另244请求批次66.06秒。这提示等待少数长响应的可能性，但不是GPU空闲时间的直接证据，也不证明增加并行组一定提速。稳定测速已开始，结果完成前不推广256图捕获。


### GPU7捕获256窗口完成；128共享解码槽试验

[图捕获128／256生产对比](monitor_snapshots/graph128_graph256_comparison.json)：GPU7两轮均300段；256窗口视频时长从1,817.96增至1,909.93秒，4K从55/300增至151/300，输出token从294,205增至350,464。模型与其他参数固定，但来源不同，不能宣称精确加速倍数。256窗口全机视觉4.662段/秒、nod4.634段/秒、CPU140.72/160核、GPU均值66.76%，进程与配置稳定。暂保留GPU7的256，不推广其他卡。

[7次解码槽探测](monitor_snapshots/graph256_decode_slot_probe.json)中112槽占用111～112，均值111.71；这是非原子的短时探测，不能替代排队时间统计。完整nod窗口记录的解码槽等待均值0.444秒、中位数0.020秒。[88个进程句柄上限检查](monitor_snapshots/slots128_fd_preflight.json)确认满足16×128＋余量以及nod侧需求。

在图捕获测量完成后，已热调整为每GPU16路／共享128槽，同时修改selected与resource_tuning保证两分支一致，模型进程无需重启。80路点头、GPU7捕获256／其他卡128及所有输入语义不变。[调整前快照](monitor_snapshots/slots128_before.json)保留基线和全部进程身份；[观察器](observe_slots128_trial.py)等各卡采用128槽并再完成150段，以排空旧准备组，再启动600秒固定窗口 `monitor_snapshots/decode_slots128_stable_window.json`。若整体吞吐下降，可仅恢复两份配置的112槽，无需重启GPU。

快照：[code_slots128_trial/SHA256.json](code_slots128_trial/SHA256.json)。此时128是试验配置，尚未证明更快；选择性裁剪仍未部署。


### 四组并行候选已准备，尚未启用

`rolling_groups`增加显式2／4组容量，默认仍2组；两种模式都只保留一个CPU预取组，提交顺序不变，engine的512请求／16384帧预算不变。新进程可记录活动组数、预取是否已完成及等待中的预取，便于区分输入不足与等待长响应。候选4组不会减少窗口、截断描述或提前终止长响应。

逐卡覆盖增加 `rolling_annotation_groups`，仅允许开启rolling的GPU选择2或4；worker采用4时为revision12，否则仍沿用原版本。监督器共享目标检查、验收器增加 `--annotation-groups 4`，测量器会记录容量变化。当前config没有该覆盖，运行中的所有GPU仍为2组，原进程未因此重启；128槽测速期间不会启用候选或重载监督器。

66项相关回归在node17通过，包括四组在首组长响应未结束时继续接纳后续组、严格请求／帧预算、来源提交顺序、停止后只排空已开始组、预取不误提交，以及默认两组兼容。代码快照：[code_four_group_candidate_integrated/SHA256.json](code_four_group_candidate_integrated/SHA256.json)。这只是代码验证，不是实测提速。

启用前须等当前容量测速结束，再安全重载监督器并确认 `rolling_group_capacity_support=true`、原GPU／nod身份保留；然后只对GPU7写入4组覆盖，按单卡首100段验收和稳定窗口流程试验。初步RSS检查GPU7父进程约37.7 GiB、节点内存使用约1.41TB／2.20TB；四组会增加宿主内存，必须先观察单卡，不能未经测量直接推广八卡。

### 2026-10-30 01:20 UTC 后续提速实验

- 128 解码槽位的完整 600 秒生产窗口：视觉 4.245 段/秒、点头 4.418 段/秒，GPU 57.95%、CPU 143.17 核。相比先前 112 槽位窗口（4.662/4.634 段/秒），没有观察到更高吞吐；但视觉原图平均像素多约25%、点头多约4%，不能宣称并发本身导致退化。已热恢复112槽位，16个解码进程/卡、80点头进程保持。证据：`monitor_snapshots/slots128_decision.json`。
- GPU 解码探索没有上线：PyAV17.0.1 暴露 CUDA HWAccel，但真实源 `improvised_V00_S0833_I00000933_local_302d3a42b3ed` 是 H.264 High 10。NVDEC 明确返回 format supported: no / Hardware is lacking required capabilities，顺序读取和 seek 都失败。软件解码正常。不能把“安装了硬件解码库”当成这些源视频能硬解。证据：`monitor_snapshots/cuda_decode_probe.json`、`cuda_decode_diagnosis.json`。没有创建转码文件或修改输入。
- 下一实验为 GPU7 四组流水线 canary，其他7卡保持两组；全局请求上限512、帧预算16384、token调度32768、每请求最大输出4096均保持。先由现有watchdog重新加载supervisor（工作进程不停止），再仅更新GPU7配置；先审核100段输入及输出来源，再测600秒稳定窗口。观察器在失败时移除GPU7四组覆盖配置，交给原supervisor安全回退。此处仅为试验方案，不代表已验证提速。
- 实验脚本本地 SHA256 备份：`code_cuda_decode_groups4_probe/`；未声称已推送 GitHub。

### CUDA 解码像素一致性补充（未接入生产）

有界源头检查同时发现8-bit H.264 High和10-bit High 10，不能一概判定所有源均不支持。4K样本 `naturalistic_V03_S1929_I00000314_local_ab319fda71c4` 可硬解；直接NV12转RGB与原YUV420P转RGB不逐像素相等（首帧RGB MAE约0.67），已拒绝该路径。先将NV12无损布局转回YUV420P，再用原RGB裁剪/JPEG流程，整段87帧的JPEG哈希、RGB像素、PTS、裁框与所有媒体元信息均与CPU相同。ABBA均值 CPU 16.935s / CUDA 7.972s；仅单段渲染观察比值 2.12，不是全流水线提速结论。GPU解码仍未启用，下一步需扩大编码/裁框/时间覆盖并测并发。详情 `monitor_snapshots/cuda_decode_8bit_normalized_probe.json`、`cuda_decode_pixel_diagnosis.json`、`cuda_decode_normalized_analysis.json`。

扩展CUDA一致性探针已准备：`probe_cuda_decode_extended.py`，固定清单 `monitor_snapshots/cuda_decode_extended_manifest.json`。16段来自8个不同源（4个4K自然对话、2个较小画幅自然对话、2个即兴对话），每个源取当前任务中的首末片段。仅对H264/yuv420p选择硬解，High10等继续原CPU路径；硬解NV12先转回YUV420P再裁剪。每段CPU和候选交替执行，校验全部媒体元信息、JPEG哈希与图像像素；记录wall/CPU时间。脚本要求四组600秒实验完成标记存在才运行，避免污染吞吐测量。当前仅准备，结果以`monitor_snapshots/cuda_decode_extended.json`的success/complete为准；未上线。

### 可选混合解码实现（尚未启用）

`src/listen_benchmark/hybrid_video_decode.py` 已实现只为 H264/yuv420p 尝试 NVDEC、NV12→YUV420P 后使用原 RGB 裁剪、不支持编码继续原 CPU 读取、硬解失败整段丢弃后完整 CPU 回退及显式原因记录。`visual_media_coverage.prepare_media_covered` 仅增加可选 source_opener 注入，默认仍为原 CPU；冻结的 `visual_native8.py` 未改动。生产的 process 解码器与配置均没有调用此模块。

12项检查通过：`PYTHONPATH=tests:src python -m unittest test_hybrid_video_decode test_visual_media_coverage test_process_decode -q`（本机Python3.13；初次使用 tests. 前缀因包发现失败，已改为平面测试模块名重新通过）。扩展16段实测改为直接调用该模块，并在报告记录源码SHA。实测前不接入正式工作进程。备份 `code_hybrid_decode_optin/SHA256.json`。

混合进程解码接入代码已准备但配置仍关闭：`hybrid_process_decode.py` 仅允许单个明确GPU mask、spawn隔离子进程、原可信缓存路径交付、来源/设备校验与缓存复用记录。worker 支持显式 hybrid 后端（revision13），supervisor 支持监控其变化，canary审核支持核验CUDA/CPU回退与像素布局归一化证据。全部8卡目标仍是 process；supervisor 未为此再次重载。75项相关测试通过，部署前仍需扩展真实像素一致性、并发显存检查、单卡试运行。代码备份 `code_hybrid_decode_integration/`。

### 四组流水线结果（2026-10-30 01:39 UTC）

600.58秒稳定窗口全机3050视觉/2699点头，约5.078/4.494段每秒，GPU均值64.0%、CPU142.0核。GPU7四组完成350段；旧两组窗口300段。但四组窗口片段均长5.73秒（原6.37秒），4K仅37/350（原151/300），输入准备平均83.38秒（原97.16秒）。21个状态采样中的活跃组最大仅1，不能说四组上限带来提速。保留GPU7现有容量避免无收益重载，其他七卡不推广；后续优先验证输入解码。比较及决策：`monitor_snapshots/groups2_groups4_comparison.json`、`groups4_decision.json`。首100段输入来源与窗口审核通过，655响应中654结构有效，1个截断原样保留。

已准备 `probe_hybrid_process_pool.py --workers 4|8|16` 并发验证（需16段顺序对照success才允许运行）。同一16段做CPU/hybrid/hybrid/CPU，spawn进程、独立 `/dev/shm/listen-hybrid-pool-*` 临时缓存、逐帧/元信息核对；NVML仅记录本探针解码子进程显存与GPU6总显存峰值。初始空闲显存不足4GiB拒绝启动。所有探针进程正常退出后临时缓存自动清理。测试模型batch/token参数不变，也不操作正式结果。

### 混合解码16段实测通过

8个源的16段、713帧全部通过RGB像素、JPEG哈希、PTS、裁框和媒体元信息完全相同的对照。8段4K 8-bit实际走CUDA，418帧一致：CPU合计106.79秒、hybrid48.998秒（观察比值2.18），进程CPU时间102.92→56.26秒；8段High10继续CPU，295帧一致，32.51→34.01秒（生产负载下有波动）。没有硬件失败回退。该结果是单解码器对照，不能当作8卡吞吐提升。详情 `monitor_snapshots/cuda_decode_extended.json`、`cuda_decode_extended_analysis.json`。下一阶段独立4进程池ABBA验证已启动，尚未部署。

4进程池ABBA通过（相同16段、每轮独立缓存、四轮像素与元信息均一致）：CPU平均44.873秒，hybrid平均37.347秒，观察吞吐比约1.20；新增解码进程显存峰值2.71GiB。NVML已直接观测解码子进程在GPU6上，未使用其他卡。当前未部署，继续8进程并发验证。结果 `monitor_snapshots/hybrid_pool4_probe.json`、`hybrid_pool4_analysis.json`。

8进程池ABBA同样16段、像素/元信息全通过，但CPU平均24.033秒，hybrid27.186秒；该并发下混合解码更慢，不能将单进程约2.18倍当作生产收益。新增解码进程峰值5.42GiB。暂不部署当前自动下载全部源帧的NVDEC路径。安装的PyAV17源码 `video/reformatter.py` 支持硬件帧在reformat时下载，下一探针测试 `HWAccel(is_hw_owned=True)`，仅在已有8fps采样选中的帧上下载，并恢复原像素格式/色彩属性。生产仍process后端。

### 输入转换后续探索：当前均未部署

- CUDA硬件帧延迟下载两段ABBA逐像素通过，只下载既有8fps选中帧；未证明8进程收益，未替换正式路径。
- YUV提前裁框：四个8-bit首帧一致，四个10-bit首帧有像素差异，拒绝通用部署。其重复同一帧的计时受原PyAV内部转换器预热影响，不用于生产吞吐结论。
- 复用完整RGB转换器：八个源128个不同采样帧均逐像素一致，旋转调用顺序的裁剪阶段计时合计5.886→3.522秒；同一流程显式threads=1的另一窗口7.689→3.385秒。两个窗口负载不同，不能把二者比值直接解读成线程参数收益。
- 已安装PyAV17 `video/reformatter.py`明确threads默认0自动选择CPU数量；stream.codec_context.thread_count=1只控制解码，不控制swscale。nod环境PyAV16.0.1不接受threads参数。
- `crop_original_frame_reused` 是新加的可选函数，原 `crop_original_frame` 保持原样，生产尚不调用新函数。新函数每线程保留独立转换器，PyAV17显式单线程，PyAV16严格识别不支持threads参数后使用原默认方式。两环境均通过格式/尺寸/色彩矩阵切换、线程隔离、无效裁框等3项真实PyAV检查。
- 已启动 `probe_rgb_reuse_pool.py --workers 16`：同16段ABBA，独立临时缓存，仅基准子进程替换裁剪回调，正式进程完全不变。完整结果以 `monitor_snapshots/rgb_reuse_pool16_probe.json` 为准。
- 本地SHA备份：`code_rgb_reuse_candidate/`。

### 正在运行：GPU6 的 CPU RGB 转换单卡试用

16进程、同16段ABBA完整准备链路（含解码/裁剪/JPEG/独立缓存/IPC）逐帧与元信息全通过；原路径平均25.585秒，新路径20.349秒，约20.5%耗时下降。仍不能直接当作打标全流水线提升。

已通过80项相关回归及新增测速配置变更校验。`process_decode.init_cpu_decoder` 会在隐藏CUDA前保留所属GPU编号，仅作CPU任务归属；仍强制CUDA_VISIBLE_DEVICES为空。`resource_tuning.json` 的 `rgb_reuse_gpus:[6]` 仅为GPU6的CPU输入准备启用新转换器。默认空列表保持原实现；其他七卡保持原转换器，GPU硬解没有启用。

为加载新CPU子进程代码，已请求decode_workers 16→17，全部工作进程采纳后观察器自动恢复16，等旧准备组排空后审核GPU6首100段并测600秒。GPU模型进程不重载，88个模型/点头进程身份必须保持；失配、输入错误或超时会取消RGB单卡配置并恢复16。原输入采样、512裁框、JPEG、prompt、窗口、输出预算均不变。

观察器：`observe_rgb_reuse_trial.py`；前态与配置备份 `monitor_snapshots/rgb_reuse_before.json`；进度 `rgb_reuse_trial_status.json`；成功/异常分别 `rgb_reuse_trial_complete.json` / `rgb_reuse_trial_error.json`。新结果input_performance记录CPU decoder revision2、所属GPU、实际RGB路径及是否复用缓存。代码SHA备份 `code_rgb_hot_canary/`。


## 2026-10-30：停止探索优化，固定生产配置

按用户最新要求，不再追加参数搜索或新性能试验，继续完成全量打标并保留监控、故障恢复。采用 batch 512、调度 token budget 32768、每卡 16 个 CPU 解码进程、112 个共享解码槽及 80 个点头 worker；视觉输入、prompt、8 fps 和窗口规格不变。GPU7 现有 4 组调度保留以避免重启，但没有证明优于 2 组，不推广。

最近已完成的 600.6 秒生产窗口：视觉 5.078 段/秒，点头 4.494 段/秒；输入不同，不能作为同样本加速比。同样本 batch 对照为 0.41411→0.44392 段/秒/卡，约提升 7.2%。

RGB 复用的 CPU 基准通过像素等价验证，但尚未完成生产验证，因此已停止该试验观察进程，rgb_reuse_gpus 恢复为空；未停止或重启 GPU 模型及点头 worker。已进入处理中的试验输入保留原有来源记录。详见 monitor_snapshots/optimization_freeze.json。后续只处理生产异常，不自动恢复优化探索。


固定策略与多机复用交接已集中整理：[加速配置交接](../../related/P2_打标加速固定配置与多机复用交接_20261030.md)。[配置快照](frozen_config_20261030/recommended_reference.json)区分推荐基线与当前GPU7试验遗留项；原始配置、逐卡实况和SHA256同目录保存。
