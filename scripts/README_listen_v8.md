# V8 倾听数据处理代码导航

## 完整源视频全量运行（2026-10-23）

用户已认可两条长视频与前后 3 秒边界审核，授权备份后全量处理 Seamless。代码已推送至 GitHub `0e99354`，2026-10-23 10:35:55 UTC 启动三机队列，已核实三机各 32 个存活工作进程及每卡 4 个绑定。新入口 [run_listen_local_full.py](run_listen_local_full.py)，版本 `full-source-local-recovery-crop-statistics-1`。使用 [listen_local_recovery.py](listen_local_recovery.py) 的已审完整源视频扫描；音质继续使用 `v8-audio-severity-review-2`，不新增人工放行，不按人脸大小排除。

完整目录含 64,087 对来源，保持已有 train/dev/test 和身份隔离。旧全量错误也重新尝试。局部音质问题只截断有证据的区间，新子片段重新质检；未知/全局问题不擅自放行。被子片段替代的父候选保存于 `superseded_attempts`，最终各池与清单只统计互不重叠的终端片段，避免重复计算。每个来源的 `coverage` 同时记录相对旧版增加和减少的时间区间；净增长不等于所有旧切片原样保留。

输出独立保存于 `data_corpora/seamless_listen_local_full/`：

- `contract.json` 固定代码、反馈、目录及集群配置哈希；`results/` 保存候选与完整音质证据，`audit/` 保存原视频角色区间、切镜、未覆盖原因。
- `screening_summary.json` 为各 split / admission pool / 音质池进度和时长；`errors/` 是处理错误，不能视作音质不合格；来源异常另计。
- `face_size_distribution.json` 统计原生面部关键点宽高、短边及有效帧；`crop_size_distribution.json` 统计向肩部扩展的固定参考方框边长、框内面部面积占比和缩放到 512 后的面部高度。speaker/listener 分开统计，数量和时长分别加权；缺失/异常框单列，仍参与音质结果。
- `crop_proposals` 是统计和构图参考，不是最终裁框。预计头部放不下也只标记；没有尺寸准入、最终裁剪或训练媒体导出。统计每约 5 分钟更新，`final=false` 为未完成快照。

三机配置 [p3_listen_local_full_cluster.json](../configs/p3_listen_local_full_cluster.json)：24 卡共最多 96 个模型进程、192 路准备，每卡同时最多 4 路镜头解码。独立共享队列及逐对原子结果支持恢复。复用持续负载 governor，GPU 空闲显存保留至少 12 GiB、系统空闲内存至少 10%，持续 CPU 压力优先减少准备并发；不停止其他任务。活跃 GPU 进程数、完成数、控制档位以 `cluster/nodes/`、`runtime.json` 为准，GPU 总利用率包含其他任务，不能全部归为此作业。

[listen_full_source_runtime.py](listen_full_source_runtime.py) 仅加速执行：NVDEC 解码后仍采用原软件缩放及 scene > 0.35，解码不支持则明确退回原 CPU 路径。170 秒样本的 5,100 帧缩放输出逐帧 MD5 一致，单轨耗时 34.79→23.10 秒；这不是全量整体加速比。镜头结果按媒体位置、文件 stat 与算法版本缓存。真实两条审核样本没有硬切，镜头切点两侧继续提取的验证来自合成硬切测试，尚不能宣称所有真实切镜类型均已验证。

使用原项目 Python 环境，项目根目录运行：

```bash
python scripts/run_listen_local_full.py verify
python scripts/run_listen_local_full.py report
# 首次运行 init 后 detach；已启动时勿重复启动
python scripts/run_listen_local_full.py init
python scripts/run_listen_local_full.py detach
# 中断后使用 resume，先检查三机无本作业活跃进程，保留已有结果、重新认领未完成来源
python scripts/run_listen_local_full.py resume
```

相关验证：新全量/切分 16 项、无视觉准入/人脸统计 7 项、持续负载调度 12 项测试通过。正式入口另回归已审核的两条原视频，并用未参与审核来源实际运行音质模型；证据保存于 `experiments/p3_listen_local_full_20261023/`。历史每分钟 650 对是旧候选流程，不能当作此版重新扫描全视频的速度预测。

## 完整源视频连续提取与审核（2026-10-23）

**切分边界上下文：** 逐段审核页默认展示每段起点前 3 秒至终点后 3 秒。原片起止时间、角色、判定和编号不变；源首尾不足 3 秒时明确展示实际长度。上下文 MP4 烧录“切分前上下文 / 切分范围内 / 切分后上下文”、切入切出提示及范围条，下载后仍可辨认边界。页面另有可点击的起止标记、查看起点/终点前 3 秒按钮、三种音轨，以及“仅原切片”对照。上下文不计入该片段时长，也不继承片段音质结论；角色名称按待审片段固定，前后文可能换说话人。入口：[extend_listen_clip_context.py](extend_listen_clip_context.py)，复用源 WAV 按采样点取上下文，原切片文件不变，验证记录见预览目录的 `context_export_summary.json` 与 `context_validation.json`。重新运行独立片段导出入口时也会自动生成上下文。

**逐段独立导出：** 按用户要求，将本节两条源会话的全部 33 个质检候选单独导出，[打开逐段审核页](http://127.0.0.1:8875/listen_full_source_clips/)。第一组 20 段（19 段音质通过），第二组 13 段（9 段通过）；包含被拒绝的父候选和回收子片段，重叠不重复计入可用时长。每段提供双人完整画面、参考框、双音轨/speaker/listener 三种独立 MP4，以及从已校验归一化源 WAV 按样本精确截取的两条 PCM WAV；页面可按组下载 ZIP、记录和导出反馈。视频使用已有连续审核合成画面进行准确寻址重编码，属于审核预览，不是最终裁剪训练媒体；原切点、筛选结论与波形处理均未修改。导出入口：[export_listen_full_source_clips.py](export_listen_full_source_clips.py)，输出目录 `data_previews/seamless_listen_full_source_clips/`，逐文件音视频时长、解码及哈希检查见该目录的 `clips.json`，输入结果哈希与导出汇总见 `export_summary.json`。

用户指出：一个原始长视频应沿完整时间线生成多个小 pair，局部接话、镜头切换或音质问题不能导致其前后都不再提取；审核必须保留原时间线，展示未提取区间。此修订独立于冻结的 `v8-audio-final-full-no-face-filter-1` 全量运行，先提供两条实际原数据验收，不自动替换全量队列。

- 算法：[listen_local_recovery.py](listen_local_recovery.py)，版本 `local-listening-recovery-1`。扫描源视频全部逐词时间；主发言方为 speaker，另一方为 listener。短附和不抢占主发言角色；双方实质性发言同时出现时仅局部断开。旧 `track_floors` 只保留诊断用途，不再控制候选提名。
- 两轨视频逐帧检测硬切（缩放分析宽度 320，FFmpeg scene > 0.35），将两轨切点合并；每个切点前后都进入后续提取，候选不能跨切点。该检测不是镜头真值，闪光可误报，渐变可漏检；本次两条真实原片均未检出切点。专门的三镜头合成视频测试验证 2/4 秒硬切检出及两侧继续分段，不能代替真实切镜验收。
- 优先完整句界，其次自然停顿/标点，再其次完整词之间。长句可产生多个 2–11.95 秒子片段；组合失败后尝试更早的终点。句内切分显式标记 `recovered_partial_sentence`，仍做原始 PCM 双轨声学切点检查，不截断已标注的词。此处是根据用户新要求对旧“完整句优先且不可拆”策略的明确调整。
- 对已定位的音质事件，按源时间加 150ms 保护区后局部避开。最终分池未带出的 `local_match_unresolved` 窗口从 QC3 的 `possible_local_speaker_match` 事件恢复；参考匹配原因从 QC4 的完整扫描 `suspicious` 窗口恢复。所有有效拒绝/复核原因都必须能定位，才允许回收；一个局部原因不能覆盖另一个全局原因。持续底噪、无法定位的音质原因、人工拒绝和任务 hold 不自动放行。音质证据为片段相对时间，使用时必须加片段起点；原角色的拒绝不直接继承给交换后的角色。
- 每个新切分重新跑冻结 QC3/QC4，并按已验收音质规则分池；只有角色、来源、精确区间及归一化音频 SHA 均一致才可复用 QC。新切片不继承父片段的音质失败结论；新的局部失败可继续产生子片段并重新质检，最多 6 层，达到上限记录为未通过而不是自动放行。
- 入口：[review_listen_full_source.py](review_listen_full_source.py)。默认处理 `naturalistic_V02_S3610_I00000088`（170 秒）及 `improvised_V01_S0122_I00000125`（196 秒），可重复 `--pair` 指定其他完整源会话；`--prepare-only` 仅做源扫描和候选规划。复用原目录只读来源与响度归一化，输出到 `experiments/p3_listen_full_source_recovery_20261023/`，不修改旧结果。
- 展示：[listen_full_source_review.html](listen_full_source_review.html)，[打开审核页](http://127.0.0.1:8875/listen_full_source/)。整条 A/B 原视频同步播放，A/B 音轨保持分离，可切换当前 speaker/listener 或混音；旧结果、新完整覆盖、所有小 pair 三条时间轴均可定位。无输出区间也保留在原视频与表格中。所有时间覆盖区间合计必须等于完整源时长，不重叠、不留未解释的时间洞。片段结束不会停止原视频播放。
- 蓝色裁框为当前片段的固定头肩构图提案；未提取处采用 10 秒参考窗口的虚线框。显示脸占裁框面积及缩放到 512 后的人脸高度，缺失则明确未测量；不按框大小或关键点有效率筛数据。下载的视频保留原时间线、完整双画面和双通道音频；动态框和边界在页面叠加，可保存当前标注截图、按原时间导出反馈。

本次两条选择是定位问题的诊断样本，不是随机产量估计：

| 原始会话 | 旧音质通过 | 新音质通过 | 新覆盖 / 旧覆盖不再保留 |
|---|---:|---:|---:|
| naturalistic_V02_S3610_I00000088 | 8 段 / 53.519 秒 | 19 段 / 96.826 秒 | +43.4375 / −0.1305 秒 |
| improvised_V01_S0122_I00000125 | 3 段 / 18.861 秒 | 9 段 / 48.428 秒 | +29.6070 / −0.0400 秒 |

第二条 9.712–19.179 秒已重新形成候选并独立质检，但父片段仍被音质规则拒绝；补齐局部证据后，进一步回收的 11.373–13.960 秒子片段重新通过音质。不能把整个父片段算作新增合格时长。28 段通过候选中 14 段为句内切分，需重点听切点自然度。源 ASR/角色归属仍可能错误；VAD 与关键点不是真人发言或真实可见性的充分证明。通过音质也不等于人工或训练准入。

运行方式（在项目根目录，使用已安装项目依赖的环境）：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 \
  /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python \
  scripts/review_listen_full_source.py
```

资源只增加 1 个模型进程；运行前应确认所选卡留有模型加载与原任务保护所需余量。11 项新增测试覆盖局部区间差集、角色切换、镜头两侧连续提取、真实 PCM 切点、真实 FFmpeg 硬切检测、长句拆分、组合失败重试、音质相对时间偏移、分池遗漏窗口恢复与人工/全局阻断，原无视觉筛选及缓存回归 7 项也通过。浏览器检查入口：[check_listen_full_source_review.py](check_listen_full_source_review.py)，依赖此前安装在 `/tmp/v8_browser_deps` 的 Playwright 与 `/tmp/v8_browser_libs` 的系统库。

## 二次提速与故障恢复（2026-10-23 08:00 UTC）

当前使用 `sustained-pressure-governor-3`，三机每卡 4 模型进程、每进程 2 路准备；GPU 模型实例数和 batch16 不变。机器上的其他任务转为较低 CPU / 较高显存占用后，原先 CPU 受限的 4×2 档位已有余量。07:59:50 UTC 的最近 5 分钟实测约 647 对/分钟（3 分钟约 669），当时完成 31,835 对；这是当前资源条件下的观测速率，不是相同输入的严格加速比。

本次修复了两次调度停止：07:12:52 因节点目录一次列举不完整误报心跳缺失，停前后节点遥测仍正常；07:51:39 因 GPU 空闲略低于旧 16 GiB 预留线触发停止，没有 CUDA OOM。已完成结果保留，并从共享队列恢复。节点读取改为已知节点文件名直接读取、短暂读取失败沿用最近心跳、超过 120 秒才认定实际缺失；已完成节点不要求继续发布心跳。GPU 运行预留调整为 **12 GiB**，持续低余量仍会排空本运行任务；未增加 GPU 进程数或模型批大小。两次停止和旧契约保存在 `cluster/governor_history/`，不计入提速效果。

[调度器](run_listen_v8_audio_governor.py) 12 项针对性测试通过；新增 [运行监控与受限恢复](watch_listen_v8_audio_full.py)，2 项恢复边界测试通过。恢复仅限已知心跳/内存保护停止，先确认三机本运行进程已退出、每卡至少 30 GiB 可用于重新加载模型且系统内存足够；半小时最多尝试 3 次。未知模型错误、代码契约不一致等不自动重试，不停止其他任务。任务完成后监控退出。[监控状态](../data_corpora/seamless_listen_v8_audio_final_full/cluster/watchdog/status.json)。

注意：`runtime.json` 的 `progress` 统计当前恢复轮次；累计处理量请读 [screening_summary.json](../data_corpora/seamless_listen_v8_audio_final_full/screening_summary.json) 或监控中的 `complete_pairs_lower_bound`。最终时长、人脸分布仍按全部保留结果汇总；切分、音质、无视觉筛选规则均未改变。以下为此前调度与各阶段历史记录。

## 提速调度更新（2026-10-23 06:19 UTC）

**首轮实测结论：** 三机 4×1 合计约 325.8 对/分钟，相比此前约 163.4 对/分钟约提高一倍。node12/node17 的 4×2 准备并发测试因 CPU 持续超过 92% 分别于 06:27:13 / 06:27:23 UTC 回退为 4×1，模型进程没有降为 1；node4 暂无足够 CPU 余量测试 4×2。当前维持每卡 4 模型进程、每进程 1 路准备，后续仍由持续负载保护控制。[实测结论](../data_corpora/seamless_listen_v8_audio_final_full/cluster/governor/tuning_conclusion.json)。实现已备份到 GitHub [71cdbd0](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/commit/71cdbd023b07740e51ad8901860277f6deb9b8d5)。

用户要求最大限度提速。发现旧调度器对单次 CPU > 80% 的采样直接从每卡 4 进程降到 1，且不自动恢复；实际每卡 4 进程下 CPU 多数在 70%–80%，没有明显 I/O 等待，三机仍有其他任务占用资源。因此新增 [独立调度器](run_listen_v8_audio_governor.py)，只替换原 coordinator，保留原 node supervisor、worker、共享队列、已完成结果和切分/音质契约。

并发上限仍为每卡 4 个模型进程、每进程 2 路准备，即三机最多 96 个模型进程。先运行 4×1，预热后每台机器观察约 4 分钟的 pair 吞吐；资源有余量时试 4×2，若实测增益不足 5% 则恢复 4×1。随机来源不同，因此这是生产负载下的观测速率，不是相同输入的严格加速比。

CPU 约 60 秒窗口持续超过 92%，或连续约 30 秒达到 97%，才按“准备 2→1、模型 4→2→1”降档；单次 80% 尖峰不触发降档。约 2 分钟稳定余量并满足 3 分钟恢复冷却后允许 1→2→4。内存保护不等待 CPU 窗口：每卡空闲显存至少 16 GiB、系统可用内存至少 10%；持续严重内存压力 120 秒会排空本运行任务。不会停止其他任务。完整阈值以 [调度契约](../data_corpora/seamless_listen_v8_audio_final_full/cluster/governor/contract.json) 为准。

7 项针对性测试通过：[test_listen_v8_audio_governor.py](../tests/test_listen_v8_audio_governor.py)。[调度切换记录](../data_corpora/seamless_listen_v8_audio_final_full/cluster/governor/launch.json)、[分档吞吐实验](../data_corpora/seamless_listen_v8_audio_final_full/cluster/governor/trials.json)、[当前运行状态](../data_corpora/seamless_listen_v8_audio_final_full/runtime.json) 可追溯。恢复中断的全量任务时，先执行原入口的 `resume`，确认三机运行后再执行 `run_listen_v8_audio_governor.py handoff`，避免恢复到旧的单次采样降档逻辑。不得对已有 governor 重复 handoff；脚本会校验被替换进程的实际命令行。

## 当前全量：切分与音质、人脸只统计（2026-10-23）

用户已授权按已备份标准全量处理，并明确暂不做人脸区域裁剪筛选。新运行版本 `v8-audio-final-full-no-face-filter-1`；音质仍为 `v8-audio-severity-review-2`。范围沿用完整来源目录的 64,087 对及原身份隔离，不使用旧版裁剪通过名单作为输入。启动时间 2026-10-23 06:01 UTC；动态状态以 [runtime.json](../data_corpora/seamless_listen_v8_audio_final_full/runtime.json) 为准。

- [全量入口](run_listen_v8_audio_full.py)：独立契约、三机共享队列、每卡最多 4 进程/每进程 2 路准备；有资源压力自动降档，不升档或干预其他任务。按 pair 原子写结果，可恢复；模型按需加载。
- [无视觉门槛规划](plan_listen_v8_audio_full.py)：复用 V8 完整语句和已审查的 ASR 区间修复，移除旧裁框 448px、wide_crop、关键点有效率及方向准入。AV 不同步、缺少视频流/转写仍单列来源问题；缺少/异常关键点仅记为未测量。没有输出最终裁框或导出裁剪视频。
- [人脸统计](listen_v8_face_statistics.py)：使用原视频坐标中的面部关键点 23:91，置信度 > 0.4、每帧至少 45 个有效点，以 3%/97% 分位跨度度量宽高。这是面部关键点范围，不是旧头部裁框尺寸。统计原始宽高/短边像素、占原画面面积比例、片段内分位数、有效帧比例；无测量值单列，不记为小脸或淘汰。
- [规则测试](../tests/test_listen_v8_audio_full.py)：7 项通过，覆盖小脸/缺失人脸仍进入音质、像素/面积口径、缺失帧分母、汇总和缓存匹配。历史 84 段的音质池及原因与备份规则一致；正式试跑 20 段（13 复用、7 重检）成功。

输出根目录：`data_corpora/seamless_listen_v8_audio_final_full/`。`results/` 保存完整证据，`audit/` 保存所有切分与未决原因，`summaries/` 保存逐 pair 统计，`errors/` 保留处理错误。旧 62 对报错也在此次全量队列中重试。旧 QC4 结果及其冻结代码保持原样。

人脸分布：[face_size_distribution.json](../data_corpora/seamless_listen_v8_audio_final_full/face_size_distribution.json)；进度/各池时长：[screening_summary.json](../data_corpora/seamless_listen_v8_audio_final_full/screening_summary.json)。按 split / primary 或 onset_reserve / 音质池分组，再分 speaker/listener。像素分桶边界为 64、96、128、160、192、256、320、384、448、512、768、1024；这些仅是统计边界，不是准入阈值。分别报告片段数量、片段时长、有效帧分布、未测量数量和时长。按片段有效帧 p10 短边计算的各阈值覆盖时长只供比较，必须结合有效帧比例阅读，不能自动当作可用量。

统计约每 5 分钟更新；`final=false` 是进行中快照。结束后统一导出各池 JSONL 清单并更新最终统计；音质 `audio_review` 仍不放行，`keep_candidate` 只代表切分与音质候选，人脸裁剪阈值尚未选择。最终有错误时以 `error_pairs` 单列，不能归为音质淘汰。

复用音质证据必须双角色、原始来源、起止区间完全一致，且归一化音轨 SHA256 一致；人工作用范围和任务证据按新计划刷新。新增/变更区间运行原 QC3/QC4 检测。运行契约：[contract.json](../data_corpora/seamless_listen_v8_audio_final_full/contract.json)；历史样本一致性：[routing_regression.json](../data_corpora/seamless_listen_v8_audio_final_full/routing_regression.json)。

从项目根目录使用现有 Python 环境：`scripts/run_listen_v8_audio_full.py report` 更新统计；`report --export` 另导出当前清单；`verify` 校验运行契约。启动使用 `init` 后 `detach`，已在运行时勿重复启动。中断恢复使用 `resume`，它先检查三机无本运行的存活进程，再重置队列游标、保留结果并重试未完成/报错 pair。保留旧统计中的历史阶段状态，以下基线记录的“未全量切换”描述对应本次授权之前。

## 当前音质筛选基线（2026-10-23）

用户决定：“关于音质的筛选先这样，你记录好筛选规则和重要代码”。当前工作基线为 **`v8-audio-severity-review-2`**，后续裁剪讨论沿用此音质方案。阶段决定记录在 [checkpoint](../data_reviews/listener_v8_audio_policy_checkpoint_20261023.json)。这条决定没有新增逐条人工标签；最新 24 段的自动分组及既有人工反馈原样保留。

本版已完成小批重分流与展示，尚未接入旧 QC4 全量入口。`thresholds_validated=false`、`training_admitted=false` 保持原值：参数尚未经过独立大规模标注校准，候选不等于训练已准入。本次仅归档，未启动处理或调整裁剪。

### 筛选原则与分组

音质优先于轻微边界疑点。明显串音、严重底噪或突发杂音排除；轻微底噪、短暂低音量的弱匹配和孤立分类器提示可降为提示。单独的身份不确定提示不作为音质差的证据，但清晰的 listener 正式接话、实际 speaker 词越界、无效区间和历史人工拒绝仍有约束。

| 输出池 | 条件与用途 |
|---|---|
| `keep_candidate` | 无硬性音质问题、无待复核音质证据且通过原任务约束，或命中既有精确人工认可；保留软提示供追溯 |
| `audio_reject` | 命中硬性音质条件；保留来源与原因作为后续修复补充池，不删除原始数据。自动判定仍可能有误 |
| `audio_review` | 孤立匹配、环境声或旧排除证据无法明确解释；单独保留，暂不计入可用数据 |
| 原任务/人工 hold | 保留原路由结果，不将角色、区间或人工拒绝统一解释成音质差 |

实际顺序：既有人工决定 → 硬性音质排除 → 音质待复核 → 原任务约束 → 保留候选。人工负例按 listener 身份和区间重叠生效，优先于正例；正例只覆盖相同 sample ID、双角色身份及起止区间，不能扩展到整对来源或新切分。依据：[原约束](listen_v8_audio_first.py)、[最终路由](listen_v8_audio_final.py)。

### 当前阈值与证据

以下条件对应 `listen_v8_audio_final.py` 的 `POLICY`、`evidence()` 和 `route()`；实现中也有显式常量，不能只修改参数字典。`RMS` 和 `floor20` 使用现有分别归一化的音轨，不能当作原始录音的串音能量比；模型分数也不是校准概率。

| 检测项 | 当前条件 | 处理 |
|---|---|---|
| 重复延迟匹配 | 至少两个不重叠 listener 事件，各相关度 ≥ 0.30、时长 ≥ 0.12s、事件 RMS ≥ −35 dBFS；匹配窗口的两轨 RMS 均 > −45；开始时间相隔 ≥ 0.25s，延迟差 ≤ 10ms | 硬排除 `repeated_audible_delayed_match`；重叠事件不重复计数 |
| 明显局部匹配 | 相关度 ≥ 0.45、事件时长 ≥ 0.25s、事件 RMS ≥ −20，参考轨 RMS > −45 | 硬排除 `loud_local_speaker_match` |
| 既有强证据 | QC3 `strong_local_speaker_match` / `two_model_environmental_contamination`；QC4 `strong_reference_match` / `consistent_reference_leakage` | 硬排除 |
| 短暂弱局部匹配 | 已触发 `possible_local_speaker_match`，listener 事件时长 ≤ 0.25s 且 RMS ≤ −28 | 软提示；否则待复核 |
| 孤立参考匹配 | QC4 孤立提示，且存在局部相关度 ≥ 0.45 的事件，并且这些事件全部时长 ≤ 0.25s、RMS ≤ −28 | 软提示；否则待复核。没有局部证据也不能直接放行 |
| 高 listener 底噪 | listener `floor20_dbfs` > −40 | 硬排除；floor20 为 20ms RMS 窗的第 20 百分位 |
| 轻微平稳底噪/宽带提示 | listener floor20 ≤ −42、噪声分数 < 0.35，且窗口 RMS ≤ −35 或 Speech 分数 ≥ 0.60 | 软提示；其他情况待复核。低底噪不能单独豁免响亮噪声窗口 |
| 单独的 elevated floor 提示 | listener floor20 ≤ −42 | 软提示；较高但 ≤ −40 或 speaker 侧待复核，> −40 listener 已被硬拦截 |
| 孤立 speaker Music | speaker 背景中恰好一个 Music 环境提示窗口、Music 分数 < 0.35，human label 为 Speech 且分数 ≥ 0.60 | 软提示；不能推广为所有 Music 都保留 |
| 突发噪声 | 已触发 QC3 环境声提示且类别为点击、爆裂、敲击、刮擦、失真等 `TRANSIENT` 集合 | 硬排除；其他无法解释的环境预测待复核 |
| 未解释的旧证据 | 未被窗口解释的 QC3/QC4 原因，或旧 `audio_qc_status=exclude` 且没有硬证据 | 待复核，不因新规则未命中而自动放行 |

平稳类别集合：White noise、Pink noise、Static、Hum、Mains hum、Air conditioning、Mechanical fan。突发类别集合：Clicking、Clickety-clack、Crackle、Crack、Burst, pop、Bang、Knock、Tap、Scratch、Scrape、Distortion。事件和背景窗口均检查 speaker、listener 两轨；“轻微平稳底噪”的放宽只适用于 listener。

上游证据保持原实现：QC3 局部相关度 0.75 为强、0.45 为疑似；环境声使用 AST/PANNs 两模型和音量门限。QC4 使用 1s 窗、0.5s 步长、±1s 延迟搜索；宽带相关度 0.45、子带 0.55，至少两个子带在 25ms 内一致；强参考阈值 0.65，重复参考疑点按 25ms 延迟一致性判断。QC4 重叠滑窗不代表独立重复事件；本版额外使用上表的不重叠事件规则。完整上游条件以以下源文件及快照为准。

### 重要代码入口与依赖

| 文件 | 职责 |
|---|---|
| [listen_v8_audio_final.py](listen_v8_audio_final.py) | 本版唯一音质分级规则：`POLICY`、`evidence`、`route`；保留 hard/review/soft 原因、疑点窗口与延迟匹配证据 |
| [listen_v8_audio_first.py](listen_v8_audio_first.py) | 最终路由必需依赖：继承人工区间、角色和任务约束；其旧版“所有音质提示都拒绝”逻辑不代表当前规则 |
| [listen_audio_quality_v3.py](listen_audio_quality_v3.py)、[listen_audio_quality_v4.py](listen_audio_quality_v4.py) | 双模型噪声、背景底噪、局部匹配及参考串音扫描；输出原始证据 |
| [listen_v8_recovery.py](listen_v8_recovery.py)、[evaluate_listen_v8_recovery.py](evaluate_listen_v8_recovery.py) | ASR 区间误拦修复、边界分级、恢复区间与审计；可靠词时间及带 80ms 保护的词/VAD 零重叠是 ASR 包络豁免前提 |
| [prepare_listen_v8_audio_final.py](prepare_listen_v8_audio_final.py) | 合并人工反馈，复用同双角色、同区间的 QC 证据，重新分流和分层抽样；不重算 GPU 模型 |
| [publish_listen_v8_audio_final.py](publish_listen_v8_audio_final.py)、[页面模板](listen_v8_audio_final_review.html) | 4 路媒体导出，双轨/单轨、约 ±8s 上下文、疑点跳转和反馈导出；复用 recovery → v7 sample 导出器 |
| [check_listen_v8_audio_final_browser.py](check_listen_v8_audio_final_browser.py)、[针对性测试](../tests/test_listen_v8_audio_final.py) | 网页播放与反馈验证；串音、放宽、人工范围、原任务约束和输入证据不变的回归 |
| [listen_v8_release.py](listen_v8_release.py)、[run_listen_v8_qc4_full.py](run_listen_v8_qc4_full.py) | 历史 QC4 生产路由和入口；不是本版最终音质入口，未被本次归档修改 |

### 人工依据、验证与版本快照

人工反馈保留在 [原 QC4 反馈](../data_reviews/listener_v8_qc4_feedback.json)、[恢复小批反馈](../data_reviews/listener_v8_recovery_feedback_20261023.json)、[最近 5 条明确反馈](../data_reviews/listener_v8_audio_first_feedback_20261023.json)。没有把未逐条反馈的片段补写为人工通过。

重要回归：旧音质小批第 6 段的重复弱串音能自动拦截；18/22/23 自动保留，区别分别为短而弱的事件、speaker 单个低分 Music 提示、较低的平稳底噪。旧 19 自动证据仍待复核，只对人工认可的精确区间保留。恢复小批 33/34 的自动结果仍是待复核，人工负例继续拦截，不能声称所有已知坏例都已自动硬拒绝。详见 [对比证据](../experiments/p3_listen_v8_audio_first_20261023/feedback_analysis/comparison.json) 与 [人工/自动分列回归](../experiments/p3_listen_v8_audio_final_20261023/reviewed_regression.json)。

固定 1,024 对来源的 `train/primary` 分流：保留 2,725 段 / 5.1223h，排除 270 段 / 0.5718h，待复核 372 段 / 0.7650h；其他 split 和 onset reserve 单列。这不是全量产量估计。[统计](../experiments/p3_listen_v8_audio_final_20261023/summary.json)。最新 [24 段预览](http://127.0.0.1:8875/listen_v8_audio_final/) 为 12 保留、8 排除、4 待复核，来自此前未展示的 24 个 pair 和身份组，无人工覆盖。分层抽样比例不代表自然通过率。

已有验证：10 项针对性测试通过，24 × 3 = 72 组网页音轨播放通过；筛选、疑点跳转、上下文 Range、反馈导出及手机布局检查通过，无页面脚本错误。记录见 [sample_validation](../experiments/p3_listen_v8_audio_final_20261023/sample_validation.json)、[browser_validation](../data_previews/seamless_listen_v8_audio_final/browser_validation.json)。本次仅核对归档完整性，不将历史测试描述成本次重跑。

已保存 **35 个关键代码文件、14 个契约/反馈/验证证据文件**的逐字节快照及 SHA256：[recorded_baseline/manifest.json](../experiments/p3_listen_v8_audio_final_20261023/recorded_baseline/manifest.json)。本版契约的 3 个代码文件、旧生产契约的 23 个代码文件均核对一致；另含恢复、发布、网页与测试组件。该快照用于追溯，不包含全部传递依赖、模型、源媒体或 Python 环境。完整抽样契约保留在 [contract.json](../experiments/p3_listen_v8_audio_final_20261023/contract.json)。

### 复用方式

从项目根目录执行针对性测试：

```bash
OPENBLAS_NUM_THREADS=1 /private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -m unittest discover -s tests -p test_listen_v8_audio_final.py -v
```

准备、发布入口分别为 `scripts/prepare_listen_v8_audio_final.py` 和 `scripts/publish_listen_v8_audio_final.py`，使用相同 Python 环境。**prepare 会根据运行当时的历史展示清单重新选样，不能用它重放原来的 24 段。** 精确回放应使用已归档的 `selection.json`、`review_selection.json` 和对应 results/媒体；新增实验使用新目录并检查证据缓存，修改区间后必须重新计算对应音质证据。

本轮没有降噪、串音分离、片段独立增益调整或裁剪修改。既有三机配置仍为每卡 4 个模型进程、每进程 2 路准备，自动升档关闭；本次未调整或启动调度。下一项讨论是裁剪框：合适人脸占比、向肩膀方向多取景、减少头顶空白。

## 历史阶段记录（以下状态与命令按各自日期理解）

**三机调度（2026-10-23 02:58 UTC）：** [run_listen_v8_cluster.py](run_listen_v8_cluster.py) 以共享队列复用原 QC4 处理函数，node17 / node12 / node4 各 8 卡，固定每卡 4 个模型进程、每进程 2 路准备，总计 96 个模型进程。自动升档关闭，每机模型上限 32、准备上限 64；共享状态文件短暂缺失/无效 JSON 有限重试，27 项相关回归通过。筛查全部节点结束后使用 [render_listen_v8_cluster.py](render_listen_v8_cluster.py) 三机导出正常候选，编码规则不变。`guard_listen_v8_cluster.py` 阻止旧单机入口混跑。质量契约不变；[配置](../configs/p3_listen_v8_cluster.json)、[运行与限制](../related/P3_V8_QC4_第二批音质验收.md)。下文为历史阶段记录。

**2026-10-23 02:08 UTC 最新：** 用户验收通过并指定音质补充池，已用 `run_listen_v8_qc4_full.py` 启动独立全量，8 卡 / 8 路准备，候选媒体 4 路导出。新目录 `data_corpora/seamless_listen_v8_qc4_full/`；`listen_v8_release.py` 实施人工区间与各池隔离。80 项测试、8 对正式入口试跑通过。见 [验收及运行记录](../related/P3_V8_QC4_第二批音质验收.md)。旧入口保持冻结，下文为历史记录。

**2026-10-23 当前状态：** 原 9 对报错已修复处理；8 卡、8 路 CPU 准备的小批 128 对完成（127 对结果、1 对响度未达标），全量等待人工验收。新增 [publish_listen_v8_repair_sample.py](publish_listen_v8_repair_sample.py) 导出 24 段复核媒体，仅用 4 路媒体任务；[页面模板](listen_v8_repair_review.html)。全量入口默认 8 个模型进程；媒体导出 `--render-workers` 默认 16，已取消写死的 128 并发。`review_hold.json` 会阻止未验收时启动无数量限制的全量。冻结契约修订和完整证据见 [运行说明](../related/P3_V8_全量处理与运行.md)。

当前状态：2026-10-22 用户认可 24 组对照中的整体切分效果，7 组无新版的片段暂不采用。科学处理代码保持验收时版本；这里只整理职责、依赖和运行方式。

**后续已获全量执行授权：** 新增 [run_listen_v8_full.py](run_listen_v8_full.py) 与 [render_listen_v8_full.py](render_listen_v8_full.py)，接入 7 条人工记录、独立运行契约、共享队列、8 卡/128 路准备、资源监控、清单和实际媒体导出。见 [全量处理与运行](../related/P3_V8_全量处理与运行.md)。下文的 24 组入口与其“尚无全量调度”描述仅对应最初对照阶段。

规则与结果：[V8 完整句子与声学边界修复](../related/P3_V8_完整句子与声学边界修复.md)。总入口：[项目 README](../README.md)。人工反馈：[listener_v8_boundary_feedback.json](../data_reviews/listener_v8_boundary_feedback.json)。版本证据：[reviewed_code_manifest.json](../experiments/p3_listen_v8_boundary_validation/reviewed_code_manifest.json)。

## 模块职责

| 模块 | 主要接口 | 职责 |
|---|---|---|
| [listen_turns_v8.py](listen_turns_v8.py) | `prepare`、`track_floors`、`full_turn_crop` | 原始词/段准备、组合反馈识别、话轮与裁框；作为库使用 |
| [listen_policy_v8.py](listen_policy_v8.py) | `POLICY`、`sentence_units`、`listener_compatibility`、`group_units`、`acoustic_cut` | 完整语句、listener 正式接话约束、时长组合、原始音轨声学切口 |
| [plan_listen_v8.py](plan_listen_v8.py) | `inspect_pair(item)`、`raw_rms(source)` | 从一对来源生成 clips / unresolved / floors / decisions；无全量命令行调度 |
| [preview_listen_v8.py](preview_listen_v8.py) | `plan` → `render` → `qc` → `publish` | 固定 24 组对照的唯一编排入口，阶段结果落盘 |
| [listen_v8_boundaries.html](listen_v8_boundaries.html) | 浏览器页面模板 | V7 / V8 / 连续上下文、声道、筛选与反馈导出 |
| [check_listen_v8_browser.py](check_listen_v8_browser.py) | 独立执行 | 最终页面播放、seek、声道、Range、笔记与窄屏验证 |
| [test_listen_policy_v8.py](../tests/test_listen_policy_v8.py) | unittest | 16 项语句、反馈及声学边界回归测试 |

`POLICY` 是规则参数说明；实现内还有显式边界常量，修改参数时需同步审查函数实现及测试，不能假定只改字典就会改变全部行为。

## 复用组件及边界

- `prepare_listen_300h.py` / `build_seamless_listening_preview.py`：只复用来源读取、媒体工具等基础能力，不执行旧版数据规划主入口。
- `listen_policy_v6.py`：复用方向检查、已知人工噪声区间匹配；`listen_policy_v7.py`：复用边界活动描述。V8 不调用 V7 的旧时间窗口组合逻辑。
- `publish_listen_v7_sample.py`：复用实际媒体导出、源文件校验、完整解码校验，输出目录由 V8 指定。
- `run_listen_v7_bulk.py`：复用 `verify_contract()` 与 `scan()` 做新时间区间质检，不启动 V7 全量规划。其 AST 来自 `listen_audio_quality.py`，PANNs 与音质判定来自 `listen_audio_quality_v3.py`。
- `src/p3_listen/subturns.py`：原整轨固定增益归一化实现。当前对照复用已有归一化音轨，不在每条片段上重新调增益。音轨实际路径以样本字段为准，可能跨旧产物目录。

V7 冻结文件、全量清单与人工反馈快照不修改。新人工记录单独保存，未来 V8 全量准入必须显式接入；不能只运行旧 V7 `scan()` 就认为这 7 条新反馈已被加载。排除范围是已展示片段区间，不是 pair 全部时间。

## 命令与环境

以下命令从 `/private/liqi_workspace/project_code` 执行。需要可访问原 TAR、已有 V7 catalog/归一化 WAV、AST/PANNs 模型缓存、FFmpeg、8 张 CUDA 卡及发布使用的字体。

```bash
# CPU 规划；24 组对照最多 24 个任务同时运行，默认并发上限 32。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 LISTEN_V8_CPU_WORKERS=32 \
  Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/preview_listen_v8.py plan

# 视频导出；新片段/上下文独立排队，最多 32 个媒体任务；可复用校验通过的缓存。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 LISTEN_V8_CPU_WORKERS=32 \
  Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/preview_listen_v8.py render

# 8 卡音质校验：物理卡 0–7，每卡一个进程、AST/PANNs batch size 16。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/preview_listen_v8.py qc

# 发布已完成结果；不会启动训练。
Omni-LTX-stage2/.venv/bin/python audio-driven-v5/scripts/preview_listen_v8.py publish

# 边界回归测试。
OPENBLAS_NUM_THREADS=1 Omni-LTX-stage2/.venv/bin/python -m unittest discover \
  -s audio-driven-v5/tests -p test_listen_policy_v8.py -v

# 预览服务 8875 已启动时，验证最终页面。
LD_LIBRARY_PATH=/tmp/seamless_browser_libs/extracted/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/tmp/seamless_browser_deps \
  python audio-driven-v5/scripts/check_listen_v8_browser.py
```

`all` 按顺序执行四阶段，会重新运行 GPU 音频质检。以上是固定对照集入口，不是全量重处理入口；不要将其重复执行理解为扩大数据覆盖。

## 缓存、审计和后续修改

- `experiments/p3_listen_v8_boundary_validation/selection.json`：24 组及选择种子；`audit/`：逐对完整候选和失败原因；`planned.json`、`rendered.json`：阶段结果；`qc/`、`logs/qc_0.log` 至 `qc_7.log`：音质结果与 8 卡完成证据。
- `data_previews/seamless_listen_v8_boundaries/samples.json`：页面实际使用的样本、自动结论和策略；`media/`：真实导出的样本及上下文；`browser_validation.json`：65 个版本的播放验证。
- `reviewed_code_manifest.json` 保存验收时核心文件与直接复用组件的 SHA256，不是包含所有传递依赖、权重和数据的独立软件包。
- 规划缓存以三个 V8 科学处理模块的 SHA256 标识；如果改变复用依赖或人工规则，应创建新的实验目录或显式失效相关缓存，不能把旧结果直接当作新结果。
- 保留 `boundary_review_flags`、`audio_qc_status` 和人工意见三个维度；unresolved 原因、角色冲突与处理报错不能统一归为音质不合格。
- 后续全量处理还需要独立调度、人工反馈准入、失败重试、去重时长汇总及训练帧数适配。帧数适配不得再次硬截完整句子。当前整理没有启动这些任务。
