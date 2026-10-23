# audio-driven-v5 总索引

**当前 Seamless 全量版本（2026-10-23 10:35:55 UTC 启动）：** 已按用户验收的完整源视频逻辑启动独立全量入口，覆盖 64,087 对来源；音质基线保持 `v8-audio-severity-review-2`，裁框只统计、不筛选。三机 24 卡共享队列，每卡最多 4 个模型进程、每进程 2 路准备；逐帧镜头扫描使用已验证的 NVDEC 解码和原缩放算法。详见 [全量入口与恢复说明](scripts/README_listen_v8.md#完整源视频全量运行2026-10-23)、[运行状态](data_corpora/seamless_listen_local_full/runtime.json)、[音质进度](data_corpora/seamless_listen_local_full/screening_summary.json)、[原生人脸分布](data_corpora/seamless_listen_local_full/face_size_distribution.json)、[头肩参考框分布](data_corpora/seamless_listen_local_full/crop_size_distribution.json)。[GitHub 代码备份 `0e99354`](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/commit/0e993542fca3714994c3501f1a3943139e9a24da)。以下旧全量状态为历史记录；新运行不会覆盖旧结果。

**RealTalk 全量切分与人脸尺寸统计（2026-10-23）：** [处理规则与代码](../../realtalk_processing/full_20261023/README.md)、[可恢复备份说明](../audio-driven-v5-data-processing/realtalk/README.md)、[GitHub 备份目录](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/main/realtalk)、[本地统计页](http://127.0.0.1:8798/)。版本 `realtalk-pairs-10s-max11-v1`：镜头/角色边界两侧分别提取，目标 10 秒、最长 11 秒，尽量沿句末切分且保留短尾段。692 条源视频产生 129,094 个候选 pair、110.7779 小时；仅做全量清单和原生人脸/预裁剪框时长统计，未应用分辨率或音质筛选，未导出全量裁剪视频。

更新时间：2026-10-23。

**完整源视频切分修复与连续审核（2026-10-23）：** 新增独立 `local-listening-recovery-1`，从整条源会话扫描发言角色，不再用旧整段话轮提名限制候选；镜头边界两侧分别提取、局部问题仅阻断对应区间、失败后尝试更小单元，新增切点重新做 QC3/QC4。两条 170/196 秒真实源会话共得到 28 段音质通过候选、145.254 秒，旧版为 11 段、72.380 秒；只代表这两条诊断样本，不外推全量。两条均未检出硬切，真实切镜场景尚未通过这批样本验证。[连续原视频审核](http://127.0.0.1:8875/listen_full_source/)展示全时间线、所有候选及未覆盖原因、双人原画面与裁框参考；[实现与限制](scripts/README_listen_v8.md#完整源视频连续提取与审核2026-10-23)。该小批已获用户认可；后续独立全量入口见本页顶部，旧冻结结果保留，裁框仍不作为准入条件。

**最新速度与恢复（2026-10-23 08:00 UTC）：** 已修复共享文件系统节点列举造成的误停，并按实测资源恢复三机 4×2；最近 5 分钟约 647 对/分钟，当时累计完成 31,835 对。GPU 进程数与模型批大小不变，运行显存预留调整为 12 GiB；新增有资源检查和重试次数限制的自动恢复。切分、音质及人脸只统计规则不变。[修复与统计口径](scripts/README_listen_v8.md#二次提速与故障恢复2026-10-23-0800-utc)、[累计进度](data_corpora/seamless_listen_v8_audio_final_full/screening_summary.json)、[运行监控](data_corpora/seamless_listen_v8_audio_final_full/cluster/watchdog/status.json)。以下速率、并发及保护阈值为此前阶段记录。

**调度提速（2026-10-23 06:19 UTC）：** 根据用户“最大限度提高速度”的要求，已替换过于敏感的调度保护：持续高负载才分级降档，有稳定余量自动恢复；三机恢复每卡 4 进程，并测量每进程 1/2 路准备的实际吞吐。切分、音质、人脸统计规则及已有结果不变，工作进程未重启。见 [调度规则](scripts/README_listen_v8.md#提速调度更新2026-10-23-0619-utc)、[分档实测](data_corpora/seamless_listen_v8_audio_final_full/cluster/governor/trials.json)。

**最新全量处理（2026-10-23 06:01 UTC 启动）：** 按用户新授权，64,087 对来源重新规划完整切分，采用已备份音质规则；暂不做人脸尺寸、裁框或关键点有效率准入，先统计原视频中的人脸大小分布。三机共享队列，每卡最多 4 个模型进程，有压力自动降档。新目录 `data_corpora/seamless_listen_v8_audio_final_full/`；[动态运行状态](data_corpora/seamless_listen_v8_audio_final_full/runtime.json)、[音质各池进度](data_corpora/seamless_listen_v8_audio_final_full/screening_summary.json)、[人脸大小分布](data_corpora/seamless_listen_v8_audio_final_full/face_size_distribution.json)、[代码与统计口径](scripts/README_listen_v8.md#当前全量切分与音质人脸只统计2026-10-23)。全量实现已备份到 GitHub 提交 [ee5b30a](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/commit/ee5b30a7be27a3b8fb2dfeeeb377871c3920b8be)。旧结果保留，最终裁框及媒体导出待阈值讨论后确定。以下“尚未全量切换”等文字为此次授权之前的记录。

**切分与音质代码 GitHub 备份（2026-10-23）：** 已推送到私有仓库 [audio-driven-v5-data-processing](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing)，提交 [`c736e3b`](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/commit/c736e3bce69edadae73f88b153200d1f495f3c1d)。独立本地副本：`../audio-driven-v5-data-processing/`。包含 148 个源代码/文档/审计文件及 4 个备份说明文件；逐文件 SHA256 核对通过，备份目录内切分、恢复、最终音质共 36 项测试通过，远端 main 与本地提交一致。未上传媒体、模型或凭据；运行依赖与限制见该仓库 README。

**音质筛选阶段已记录（2026-10-23）：** 按用户“音质的筛选先这样”，以 `v8-audio-severity-review-2` 作为后续裁剪讨论的工作基线。明显串音/严重杂音排除，轻微底噪/短暂弱匹配按条件保留，证据不明确者单列待复核。规则、阈值、人工覆盖范围、代码入口和验证状态统一见 [音质基线与代码导航](scripts/README_listen_v8.md#当前音质筛选基线2026-10-23)；已保存 [35 个代码文件及证据快照](experiments/p3_listen_v8_audio_final_20261023/recorded_baseline/manifest.json) 和 [阶段决定](data_reviews/listener_v8_audio_policy_checkpoint_20261023.json)。本次仅归档；新规则尚未全量切换，最新小批的待复核分组保持原样。以下为此前各阶段记录。

**最后一版音质小批（2026-10-23）：** [打开 24 段新样本](http://127.0.0.1:8875/listen_v8_audio_final/)。1–12 为保留候选（常规、轻微底噪/单个弱提示、短暂弱匹配各 4 段），13–20 为串音/杂音排除对照，21–24 为证据不明确的待复核组，后者暂不计入可用数据。重新分流 1,024 对未展示来源，使用固定参数补查重复弱匹配，并降低轻微底噪和孤立 speaker Music 提示的权重；裁剪沿用原框。10 项针对性测试通过；24 段均来自历史未展示的 pair 和身份组，无人工认可覆盖。旧第 6 段可自动拦截，旧 18/22/23 可自动保留，旧 19 的自动证据仍待复核、其精确样本按人工认可保留。规则尚未切换全量，待本批检查后讨论裁剪。[规则与记录](related/P3_V8_QC4_第二批音质验收.md#最后一版音质小批2026-10-23)。

**音质优先小批反馈更新（2026-10-23）：** 新批第 6 段漏掉 speaker 串音；第 18、19、22、23 段用户认为可保留。已保存 5 条明确反馈，未推断其他未提及样本全部通过。对比确认：22 仅 speaker 单个 Music 模型提示；23 的底噪门槛过敏；18 的疑点短且较弱；19 仍不能靠现有指标充分解释。第 6 的相关性低于阈值但存在重复近似延迟，说明简单调相关性阈值无法解决全部误杀/漏检。新规则尚未获全批验收，生产规则未更改。[对比证据](experiments/p3_listen_v8_audio_first_20261023/feedback_analysis/comparison.json)。

**音质优先的新一批复核（2026-10-23）：** 已记录上一批 36 段反馈：33 段认可，旧第 28、33、34 段因串音/音质排除。新规则把身份不确定、边界疑点和未确认短回应降为提示，串音/环境杂音仍优先拦截，明确角色错误及原人工拒绝仍保留。对 512 对未展示来源复用同区间 QC3/QC4 证据重新分流，抽出 24 个新 pair/身份组：[打开音质优先检查页](http://127.0.0.1:8875/listen_v8_audio_first/)。1–16 为保留候选，17–24 为疑似串音/杂音筛掉对照；裁剪沿用现状，新增规则暂未切换全量。详情见 [音质优先小批](related/P3_V8_QC4_第二批音质验收.md#音质优先小批2026-10-23)。

**增量复核与裁剪小样（2026-10-23）：** [打开 36 段检查页](http://127.0.0.1:8875/listen_v8_recovery/)，包含 6 段头肩裁剪、6 段 ASR 误拦修复新增切分、6 段边界软疑点、12 段身份/串音等对照，以及 6 段仍被筛掉的内容、音质、边界样本。每段可单听双人音轨并查看前后约 8 秒上下文；拒绝原因可见，可标注误杀。裁剪仅作构图预览，3 段有显式记录的视觉校正，不代表自动头部检测已经完成。生产筛查已完成 64,025 对，另有 62 对错误待处理；冻结生产规则未替换。训练主池原自动候选 50.27h；全量边界软疑点候选 60.67h，ASR 修复在 2,007 对训练样本净新增 5.64 分钟、粗略外推约 2.93h，身份不确定池 112.06h 仍需复核，均不能直接算作新增可用时长。详情见 [增量复核记录](related/P3_V8_QC4_第二批音质验收.md#增量复核与裁剪小样2026-10-23)。以下日期较早的状态为历史记录。

**P2 listen-only 事件监督代码已准备，等待用户提供真实数据。** 新增独立数据/缓存/训练/推理/评测入口，E0/E1 共享最终 U，事件梯度分别关闭/开启；CPU 梯度、恢复、时间轴、CFG 和两进程分布式检查已通过。详见 [实现与运行说明](related/P2_倾听事件监督实现与运行.md)、[数据格式示例](examples/p2_listen/README.md)、[详细实验计划](related/P2_倾听事件监督实验计划.md) 与 [代码修改要求](related/P2_倾听事件监督代码修改要求.md)。当前只使用合成张量/临时媒体验证，未加载预训练权重，未开展真实 GPU S0 或正式训练；QC4 数据任务与本实验独立。

**产量审计更新（2026-10-23 03:07 UTC）：** QC4 已统计 44,020 对，训练主池自动候选 33.69h；按当前比例外推约 49h，不能把音质储备等池合并称作“约 300h 可用”。2,048 对随机来源审计发现，裁框门槛、整句冲突、边界疑点及人声身份不确定共同造成大量转出；回查原始转写证实存在整段 ASR 包络覆盖无词空档、从而误拦候选的机制。尚未量化修复可回收时长，未修改生产规则。证据与限制见 [产量审计](related/P3_V8_QC4_第二批音质验收.md#产量审计2026-10-23-0307-utc)。三机固定每卡 4 进程继续处理。

**当前（2026-10-23 02:58 UTC）：QC4 全量固定为三机 24 卡、每卡 4 个模型进程、每进程 2 路准备，已恢复共享队列处理。** 合计 96 个模型进程、每机最多 64 路准备；停止自动升档。恢复前已完成 27,496 / 64,087 对，剩余 36,565 对与 26 对错误一起重新排队；已完成结果保留。4×2 完整测量约 2,866 对/分钟，剩余筛查按该短窗口速率约 13 分钟，模型启动、尾部任务和视频导出另计。8×1 测量因共享状态文件短暂读取失败而中断；已增加有限重试，27 项相关回归通过。训练主池自动候选暂为 21.67h，音质修复储备主池 103.39h 单独隔离。[实时状态](data_corpora/seamless_listen_v8_qc4_full/runtime.json)、[节点负载](data_corpora/seamless_listen_v8_qc4_full/cluster/latest.json)、[并发测量](data_corpora/seamless_listen_v8_qc4_full/cluster/benchmark.json)、[处理统计](data_corpora/seamless_listen_v8_qc4_full/screening_summary.json)。筛查后仅正常候选进入三机媒体导出，训练未启动；详见 [QC4 验收与三机运行记录](related/P3_V8_QC4_第二批音质验收.md)。下文为历史阶段记录。

**当前：QC4 验收通过并按问题分组，已于 2026-10-23 02:08 UTC 启动独立全量处理。** 用户确认其余新样本无问题：第 15 段原始双人转写重复，列入说话归属待修复；第 18、20、21、22、24 段单列音质修复补充池，第 19、23 段保留内容问题。补充池不计入可用时长，暂不修复。全量覆盖 64,087 对来源，8 卡各 1 进程、总计 8 路准备；80 项测试及 8 对正式入口试跑通过，候选媒体导出另用 4 路。新输出为 `data_corpora/seamless_listen_v8_qc4_full/`，见 [实时进度](data_corpora/seamless_listen_v8_qc4_full/screening_summary.json)、[运行状态](data_corpora/seamless_listen_v8_qc4_full/runtime.json)、[完整规则与验收记录](related/P3_V8_QC4_第二批音质验收.md)。只导出正常候选成品，训练未启动；以下为此前阶段记录。

**当前：V8 修复后小批验收，全量等待用户确认。** [打开 24 段验收样本](http://127.0.0.1:8875/listen_v8_repair/)。已修复反馈起点越过前一片段终点的兜底错误，并明确审计无视频流来源；原 9 对报错已处理，3 对恢复 50 段候选、6 对记为源数据缺失。原验收 24 对的规划结果不变，26 项测试通过。随后仅处理 128 对来源：127 对完成、1 对响度归一化未达标，未放行；8 卡各 1 个模型进程、总计 8 路 CPU 准备，用时 134 秒。用户要求先验收再全量，`review_hold.json` 已阻止无数量限制的启动。下方 64 进程/128 并发和“全量执行”描述为历史记录。详情见 [V8 全量处理与运行](related/P3_V8_全量处理与运行.md)，负载实测见 [load_report.json](experiments/p3_listen_v8_load_tuning_20261023/load_report.json)。

**V8 全量执行入口已建立：** 用户已要求应用至 Seamless 全量。处理 64,087 对来源，采用本机 8 卡、默认 64 个模型进程与 128 路 CPU 准备；筛查后自动导出非排除候选的裁剪视频和双路 WAV。当前运行状态见 [runtime.json](data_corpora/seamless_listen_v8_full/runtime.json)，统计见 [screening_summary.json](data_corpora/seamless_listen_v8_full/screening_summary.json)，资源实测见 [telemetry/latest.json](data_corpora/seamless_listen_v8_full/telemetry/latest.json)。规则、命令及完整路径见 [V8 全量处理与运行](related/P3_V8_全量处理与运行.md)。V7 结果保留，训练不随本次数据任务自动启动。

**V8 切分已获用户认可，规则与代码已整理：** [当前处理逻辑与验收结论](related/P3_V8_完整句子与声学边界修复.md)记录完整 speaker 语句、backchannel、2–10 秒及最多 11.95 秒例外、声学边界、448 裁框、音量与音质步骤；[代码导航及运行说明](scripts/README_listen_v8.md)列出模块职责、复用依赖、分阶段命令和产物。用户确认“未生成新版”的 7 组先不要，精确样本及时间区间见 [人工记录](data_reviews/listener_v8_boundary_feedback.json)，验收版本见 [代码清单](experiments/p3_listen_v8_boundary_validation/reviewed_code_manifest.json)。这是切分规则验收，不代表其余样本音质全部通过；全量 V8 与训练尚未启动。

**V8 完整句子边界对照已导出：** [打开 24 组新旧与上下文试听](http://127.0.0.1:8875/listen_v8/)。先补齐 speaker 语句，再找双方音轨的安静切口，保留 backchannel；为句子完整允许到 11.95 秒，不延伸进 listener 正式回应。17 组有新候选、7 组未强行切分；新候选仍有边界复查标记，不能当作已合格。规则、代码、结果和 8 卡/32 路 CPU 调度见 [V8 完整句子与声学边界修复](related/P3_V8_完整句子与声学边界修复.md)。仅对照样本完成处理，V7 全量未替换，训练未启动。以下 V7 与更早记录为对应阶段状态。

**V7 六类数据抽样复核：** [打开 80 段预览](http://127.0.0.1:8875/listen_v7/)，旧 `/listen_qc3/` 同样跳转。包含通过/待复核/建议排除 × 主池/开头反馈备用池，保留原 24 段主池并新增 56 段；页面显示分流原因及可疑音频时间窗口，可记录并导出意见。当前全量分类不变。抽样方法、范围、代码与文件见 [六类抽样复核说明](related/P3_V7_六类数据抽样复核.md)。仅抽样媒体已实际导出，全量视频尚未导出。

**对外交付说明：** [V7 244 小时主池数据交付说明](related/P3_V7_244小时主池数据交付说明.md)可直接转交接收方，包含已完成与待完成事项、主池完整路径、所需音视频依赖、清单字段及时间/裁框坐标解释、跨机器迁移和最终片段导出约定。当前交付形态为清单、归一化整轨音频和原视频引用，尚未打包为逐片段成品。

**最新状态（2026-10-22 10:02 UTC）：V7 本轮全量筛选已结束，清单已导出。** 64,087 对来源中，63,814 对完成筛选，273 对处理报错，待处理为 0；所有工作进程组已正常结束。训练集自动通过主池 126,748 段、244.12h，开头反馈备用池另有 9.37h，合计 253.49h；待复核主池 360.33h、备用池 39.03h，均不计入合格时长。尚未达到 400–500h 目标。全量视频重编码、模型帧数适配和特征缓存尚未完成，训练未启动。结果见 [最终筛选报告](data_corpora/seamless_listen_v7_full/screening_summary.json)及[导出清单索引](data_corpora/seamless_listen_v7_full/manifests/index.json)。下方运行快照为历史记录。

**性能更新（09:51 UTC）：** V7 已改成三机共享任务队列，选定 node17/node12/node16 分别 192/192/224 路 CPU 准备并发，本机 256 路试验组正在平滑退出。三机实测约 1,600 对/分钟，前两台 CPU 约 97–98%；模型、切分、归一化和筛选阈值未改变。09:50:53 已完成 53,010/64,087 对，全量筛选仍在进行。处理步骤、模型名称、性能证据和运行入口见 [V7 全量筛选性能优化](related/P3_V7_全量筛选性能优化.md)；下方早期启动方式作为历史记录，勿与动态队列混跑。

**当前执行：V7 原始 Seamless 全量筛选已于 2026-10-22 08:42 UTC 启动，三机 24 卡；训练未启动。** [V7 全量切分与 448 裁框筛选](related/P3_V7_全量切分与448裁框筛选.md)记录已落地的规则、来源核验、处理代码、启动检查和结果口径。原始目录 21,959 个 TAR 已重新清点；本轮处理 62,474 对 train、695 对 dev、918 对 test。自然切分为 2–10 秒，取消凑帧淘汰，原像素裁框至少 448；结尾反馈保留，开头活动单列备用池。实时 [筛选报告](data_corpora/seamless_listen_v7_full/screening_summary.json)分别统计 2 秒/3 秒门槛与是否纳入开头备用池的时长。旧 QC3 严格基线已停止并保留快照；下方关于旧基线“正在运行”的描述为历史状态。

文档路径：`related/P3_V7_全量切分与448裁框筛选.md`；输出目录：`data_corpora/seamless_listen_v7_full/`。本轮先完成源时间区间、裁框、归一化和音质筛查；全量视频重编码及模型帧数适配尚未开始。

**原始来源与覆盖范围核验（2026-10-22）：** 实际音视频读取自 `/public/0_DATA/2_Avatar/seamless_interaction`。旧配对索引含 64,619 对会话，引用的 21,383 个 TAR 均存在于该目录；当前严格基线仅处理分组筛选后的 31,260 对，不代表整个原始目录全量。原目录共 21,959 个 TAR，未被索引引用的 576 个中，531 个属于 `extras`，45 个是 dev/test 普通包。所有 train 普通包均被索引引用，但尚未证明每个包内的所有有效会话均已入索引。核验记录：[original_source_coverage.json](experiments/p3_listen_qc3_attrition/original_source_coverage.json)。此前 62,474 对 train 的容量结论限于旧索引，不能当作原始目录完整会话量或最终合格时长。

**预览服务已精简：8875 仅保留 [QC3](http://127.0.0.1:8875/listen_qc3/) 及其标注/回归页面。旧版在线入口已关闭，历史数据文件保留；下方历史文档中的旧预览链接不再对外提供。**

**当前：用户已确认切分及 QC3 试听结论，授权全量筛选并统计实际可用时长；训练尚未启动。**

**数据量目标更新：最终至少约 400–500h，按 500h 合格训练集规划，开发/测试另计。** [400–500 小时扩量建议](related/P3_400至500小时数据扩量建议.md)记录恢复被混合 split 连通组整体排除的官方 train 来源、修正时间层、384/448 裁框对照和质量分流建议。来源审计显示可将原始 train 候选从 30,615 对扩展到 62,474 对；这只是来源容量，不是已合格时长。新方案尚未替换正在运行的严格基线。

**最新诊断：[QC3 淘汰归因与裁框分布](related/P3_QC3_淘汰归因与裁框分布.md)。** 对固定快照的 18,880 对会话逐对统计原始裁框分布、各阶段时长损失和切分覆盖对照，分析无失败。原裁框中位数 402 像素；裁框合格的 106.90 小时父话轮，被现行时间规则压缩为 17.51 小时，仅取消静音凑帧要求即可覆盖 68.55 小时候选。当前切分实现明显过严，应修正后再评估最终可用量；报告给出 384/448/512 阈值比较，尚未更改生产阈值。路径：`related/P3_QC3_淘汰归因与裁框分布.md`。

- **现行处理标准、步骤与代码指路：[P3_Seamless全量筛选标准与执行.md](related/P3_Seamless全量筛选标准与执行.md)**。
- 简介：统一记录 speaker 完整句/自然停顿边界、4–9.96 秒与 249 帧对齐、原裁框 ≥512、错误方向舍弃、整轨响度归一化、AST/PANNs 串音与噪声三类筛查、人工反馈应用、全量执行入口及去重时长口径。
- 路径：`/private/liqi_workspace/project_code/audio-driven-v5/related/P3_Seamless全量筛选标准与执行.md`。
- 全量入口：[run_listen_qc3_bulk.py](scripts/run_listen_qc3_bulk.py)；独立输出：`data_corpora/seamless_listen_qc3_full/`；三机 24 路已于 2026-10-22 07:13 UTC 启动，覆盖来源隔离后全部 31,260 对 train/dev 会话。实际时长见每分钟更新的 [累计报告](data_corpora/seamless_listen_qc3_full/screening_summary.json)。
- 本轮确认 5 段保留、4 段排除，见 [QC3 人工记录](data_reviews/listener_qc3_feedback.json)。待复核隔离，不计入训练合格时长。旧 300h 清单、旧窗口和旧缓存不作为新规则下的合格结果。

以下 V6/QC1/QC2/QC3 小样本段落为历史检查过程；其中旧网页已下线，旧暂停状态由上述现行执行文档更新。[V6 小样本审阅说明](related/P3_V6_切分与质量规则小样本审阅.md)保留边界与画面规则的推导。

用户已认可 V6 切分边界；后续降噪 A/B/C 仍有源杂音，已转为筛查并剔除。[QC1 随机音质筛查小样本](related/P3_QC1_随机音质筛查小样本.md)记录新一轮 48 个分组、59 段候选的检查：19 段自动通过、40 段剔除。[最新试听页面](http://127.0.0.1:8875/listen_qc/)展示 10 段通过候选和 4 段剔除反例，保留音量归一化、不额外降噪。**等待本批人工检查，再决定全量处理与训练。**

[音质筛查与剔除方案](related/P3_listener音质筛查与剔除方案.md)及[人工拒绝清单](data_reviews/listener_noise_rejections.json)保留前次结论。6 段已知噪声自动回归全部拒绝，但不代表独立准确率；当前尚缺经人工确认的干净笑声验证。

**最新人工复核：QC1 展示的 4 段拒绝样本，1 段确为杂音、3 段是应保留的真实倾听反应。旧硬拒绝规则已否决，未确认的拒绝项转为待复核，不得直接全量使用。** 反馈见 [QC1 人工记录](data_reviews/listener_qc1_feedback.json)，修正方向见 [QC1 文档](related/P3_QC1_随机音质筛查小样本.md)。

已完成 [QC2 独立抽样与微反应复核](related/P3_QC2_独立抽样与微反应复核.md)：冻结新规则后选取 48 个未检查过的分组，56 段音质候选中 28 段保留候选、28 段待复核。[QC2 试听](http://127.0.0.1:8875/listen_qc2/)展示 16 段独立样本。无转写、窄带集中和瞬态不再单独硬删；改为保留候选/待复核/疑似杂音。已知例回归仍有 2 段杂音进入自动保留候选，故此轮仅作人工审阅分流，不能直接启动全量训练。

[QC3 环境噪声与串音筛查](related/P3_QC3_环境噪声与串音筛查.md)在现有 56 段上执行，自动结果为 20 段保留候选、35 段待复核、1 段建议排除。[试听页面](http://127.0.0.1:8875/listen_qc3/)含 20 段及逐段人工结论；[已知例回归](http://127.0.0.1:8875/listen_qc3/known_examples.html)同时展示成功与漏放。未新增降噪，240 个事件仍是待标注队列；本轮已获准按三类分流开展全量筛选，训练需等待新数据验收。

本项目研究 listen 阶段的音视频生成：根据 speaker 的语音和 listener 参考图像，生成 listener 的面部视频及其自身声音，包括沉默、短促附和和笑声等倾听反应。已有 P3 / Bernini 固定文本工程原型；当前新增 P2 事件监督首轮实验设计，先检验事件监督的生成收益，再考虑 P3 规划机制的额外价值。

## 当前扩展：300 小时、三机 24 卡

- [300 小时与三机 24 卡执行说明](related/P3_300小时_三机24卡训练.md)：新的节点配置、去重时长口径、数据隔离、处理与自动训练顺序、日志和恢复方式。
- 节点为 node17 `172.29.15.91`（master）、node12 `172.29.252.11`、node16 `172.29.112.17`，各 8 张 H200；[拓扑配置](configs/p3_listen_3nodes.json)、[训练配置](configs/p3_listen_300h.json)。
- 旧版编排输出在 `experiments/p3_listen_300h/`，当前仍被 hold 文件阻止。新全量筛选使用 `data_corpora/seamless_listen_qc3_full/` 的进度与汇总；旧编排状态、872 条数据和 8 卡结果只代表历史工程验证，不代表新规则下的正式训练状态。

## 当前实验计划

### P2 固定文本倾听与事件监督（代码已准备，待真实数据 S0）

- 事件自动标注（v2 开放发现）：[实现与运行说明](related/P2_倾听事件自动标注实现与运行.md)；[入口](scripts/label_listen_events.py)、[配置](configs/listen_labeling.json)、[清单与自由描述示例](examples/listen_labeling/README.md)。Joy 静音视觉 + 音频模型完整波形先自由发现行为、保留证据，再独立查看原媒体检查点头/发声；支持 Omni 两阶段对照、分阶段续跑、人工修订和 P2 导入。23 项合成媒体/模拟 HTTP 测试通过，真实模型联调和数据标注尚未开展；[验证记录](experiments/listen_labeling_code_validation/result.json)。
- 实验计划：[P2_倾听事件监督实验计划.md](related/P2_倾听事件监督实验计划.md)。覆盖路线选择、V8/QC4 小规模数据准入、发声/点头标注、共享 U 监督、成对训练预算、生成事件评测、继续条件和失败分流。
- 代码要求：[P2_倾听事件监督代码修改要求.md](related/P2_倾听事件监督代码修改要求.md)。覆盖旧 P2 复用边界、新模块职责、输入/目标隔离、schema、时间轴与缓存、梯度/DDP/恢复、拟定 CLI 和验收测试。
- 来源明确为 [audio-driven-v4 中的 Pipeline 2](../audio-driven-v4/docs/PIPELINE2_PLAN.md)：冻结 Omni/DiT，融合 Thinker 8/18/28/35 层，最终 U 共享特征 FM 与 AV 监督。不是 P2.1，也不继承旧 GT 文字计划作为当前任务输入。
- 首轮主比较：E0 的事件头读取停止梯度的 U，仅作 probe；E1 的事件 loss 反馈到同一份生成条件 U。两组保留相同的特征/AV 损失、初始化、数据、预算和采样协议。
- 默认 pilot 建议 train/dev/test 为 2048/128/128 段、每组 1000 步，先做一个训练 seed 的成对实验，有信号后复验第二个 seed；这些是待实施的起始预算，不是已完成的数据或训练数量。
- 实现与运行：[P2_倾听事件监督实现与运行.md](related/P2_倾听事件监督实现与运行.md)；入口：[scripts/p2_listen.py](scripts/p2_listen.py)；代码：[src/p2_listen/](src/p2_listen/)；配置：[基础配置](configs/p2_listen_event_base.json)、[配对协议](configs/p2_listen_event_pair.json)。
- 数据接口：[示例与说明](examples/p2_listen/README.md)。用户提供数据后先形成独立小集清单和事件标注，再做真实 S0；不自动使用旧数据或启动训练。
- 验证：[CPU 测试结果](experiments/p2_listen_code_validation/unit_tests.json)、[梯度与恢复](experiments/p2_listen_code_validation/cpu_audit.json)、[两进程 CPU DDP](experiments/p2_listen_code_validation/cpu_ddp.json)。这些是代码验证，不是预训练 Omni/LTX 的真实 S0 或生成质量结论。P3 原有实现和 QC4 处理保留。

### P3 / Bernini 固定文本倾听实验

- 文档：[P3_Bernini_固定文本倾听实验计划.md](related/P3_Bernini_固定文本倾听实验计划.md)
- 相对路径：`related/P3_Bernini_固定文本倾听实验计划.md`
- 绝对路径：`/private/liqi_workspace/project_code/audio-driven-v5/related/P3_Bernini_固定文本倾听实验计划.md`
- 状态：已实现独立数据/缓存/训练/推理入口；单卡 P/C0/A、检查点恢复、8 卡 DDP 和 8 路完整生成验证均已通过。正式训练和效果评估尚未完成。
- 实现与运行：[P3_Bernini_固定文本实现与运行.md](related/P3_Bernini_固定文本实现与运行.md)，包含代码结构、命令、数据修订与验证记录。
- 配置：[configs/p3_listen.json](configs/p3_listen.json)；入口：[scripts/p3_listen.py](scripts/p3_listen.py)；代码：[src/p3_listen/](src/p3_listen/)。

实验检验：不提供逐样本 GT 回复文本、动作描述或反应时间表，仅用固定文本和音频驱动的 hidden states，能否生成自然、时机合理的倾听音视频。Omni 使用固定任务提示，Gemma 使用固定场景提示，两者均在训练和推理时保持一致。首轮为离线、给定长度的生成，输入为 speaker 音频、listener 首帧与窗口长度，输出为等长的 listener 音视频。

计划包含数据分组与切窗、固定提示词全文、旧 P3 接口的必要改造、特征缓存与时间分辨率、S0 小样本检查 → P 规划器 → C0 Connector → 可选 A 联合适配的训练阶段及预算。对照包括固定文本基线、预测 hidden 主模型、GT 特征诊断、移除目标 hidden 和替换 speaker 音频；同时规定开发集选点、测试集隔离、人工评价以及继续/停止条件。

当前 V5 将原 457 个片段切为 872 个不超过 10 秒的连续子话轮，约 109.6 分钟，音轨响度已统一。14 个参与者/session 连通组已固定为 10/2/2 组，训练/开发/测试窗口为 1160/224/228。训练窗口最大约 8 秒，避免人物和重叠窗口跨集合。这些是数据与工程产物，不代表模型效果结论。

## 任务与参考资料

| 文档 / 目录 | 简介 |
|---|---|
| [倾听数据任务设定与分段讨论](related/倾听数据任务设定与分段讨论.md) | 倾听生成任务的初步设定与数据分段讨论。 |
| [Seamless 本地数据小样本检查](related/Seamless_本地数据小样本检查.md) | 本地 Seamless 数据的小样本检查与初步处理记录。 |
| [GLARE 论文与代码详细分析](related/GLARE_论文与代码详细分析.md) | GLARE 参考工作的论文、模型和代码分析。 |
| [GLARE 本地论文](related/00_GLARE_local.pdf) | GLARE 论文 PDF。 |
| [GLARE 代码库](related/glare/) | 本地参考代码与数据处理实现。 |
| [OmniPersona 本地论文](related/OmniPerson.pdf) | Query 提取多模态上下文并指导音视频生成的强相关参考；其 memory query 不等于本方案的逐时段事件监督。 |
| [上一版项目总览](../audio-driven-v4/README.md) | 历史 P1、P2、P3 等 pipeline 的尝试记录，任务包含全身及 listen+response；不直接视为当前实验结果。 |
| [Bernini 参考代码库](../Bernini/) | 原始 Bernini 实现，供核对特征规划与生成条件机制。 |

## 当前数据入口

当前新实验的数据准入以本文顶部 V8/QC4 规则与 [P2 计划的数据部分](related/P2_倾听事件监督实验计划.md#4-数据先做独立小规模证据不等待全量大训练) 为准。以下 V5 数据和工程验证是已有 P3 的历史输入/证据，不自动视为新 P2 pilot 的合格训练集。

| 入口 | 说明 |
|---|---|
| [V5 子话轮说明](data_previews/seamless_subturns_v5/README.md) | 当前数据：不超过 10 秒的子话轮与统一音量。 |
| [V5 manifest](data_previews/seamless_subturns_v5/manifest.json) | 当前实验数据准备入口，含子话轮、原始来源和归一化音轨来源。 |
| [V5 边界与音频验证](data_previews/seamless_subturns_v5/audio_boundary_validation.json) | 完整时间覆盖、响度、峰值、样本数与抽样同步检查。 |
| [8 卡 DDP 验证](experiments/p3_fixed_text_v2_subturns/validation_8gpu/ddp/completed.json) | 两次真实 A 阶段联合更新；每步各 rank 参数哈希一致，冻结参数未变化。 |
| [8 路生成验证](experiments/p3_fixed_text_v2_subturns/validation_8gpu/generation/completed.json) | 8/8 成功，覆盖不同条件路径；仅为工程检查。 |
| [V4 数据说明](data_previews/seamless_speaker_span_v4/README.md) | V5 继承的 speaker 首尾词边界与固定裁剪框说明。 |
| [V4 manifest](data_previews/seamless_speaker_span_v4/manifest.json) | 历史父区间索引，保留用于追溯。 |
| [数据处理脚本](scripts/) | 数据检查、裁剪、预览发布及验证脚本。 |

父区间边界以 speaker 起止为准，保留中间停顿，以及 listener 与 speaker 首尾重叠的声音；V5 在其内部进一步划分连续子话轮。正式训练依据 V5 边界及继承的裁剪框重新提取原始视频，使用统一音量后的原始独立 WAV。

阅读顺序建议：先看当前实验计划，再按需查阅数据说明、任务讨论和参考工作。后续实施记录、检查点与实验结果应在本索引中另列入口，并注明状态。
