# audio-driven-v5 总索引

**TensorBoard 训练监控已接入（2026-10-26）：** [指标口径、查看命令及恢复说明](related/P2_TensorBoard训练监控与恢复.md)、[真实 GPU 验收](experiments/p2_tensorboard_20261026/gpu_verification.json)。当前 LoRA/H200 配置默认开启，rank 0 记录全局 AV 与视频/音频分项、Event 各类/模态、学习率、梯度、吞吐和显存；每 100 step 保存及断点恢复保留。146 项 CPU 测试、真实单/双卡和曲线恢复检查通过；日志开关前后参数、优化器、RNG 逐位一致。

**50 小时快速训练子集已完成（2026-10-26）：** [数据入口、七类覆盖与使用说明](experiments/p2_train50h_backchannel_face_20261026/README.md)。按“大脸主体集＋稀有类别补充”选出 **30,415 段、50 小时零 4 秒**，全部来自 train；七类均覆盖，含摇头 351 段。99.185% 片段的人脸短边 p10≥224 原始像素，p50 中位数约 303 像素。与现有独立 dev/test 的会话、规范化参与者和连通组无交集；原生输入/目标清单、原文整合结果、七类弱标签和缓存引用已准备，未启动训练。[选择器](scripts/select_listen_train_subset.py)、[完整校验](experiments/p2_train50h_backchannel_face_20261026/validation.json)。

**24 卡真实联合训练与 H200 提速（2026-10-26）：** [GPU 验证、速度对比与启动说明](related/P2_24卡联合训练GPU验证与提速.md)、[H200 配置](configs/p2_listen_av_event_text_lora_h200_v1.json)、[两 seed 配对协议](configs/p2_listen_av_event_text_lora_h200_pair_v1.json)、[跨机网络环境](configs/p2_listen_h200_multinode.env)。3×8 H200 完成真实 LoRA 联合训练；140 项 CPU 测试、E0/E1 单/双卡、实际续训、恢复重算和独立生成通过。隔层检查点＋32 MiB 通信桶与原方式训练 7 步后参数/优化器/RNG 逐位一致。batch=24 时 24 卡约 3.60 秒/步；11.948 秒真实样本约 114.2 GiB。正式 batch=8 保留，推荐四份实验按 8＋8＋4＋4 卡分配；尚未开始正式效果实验。

**逐片段 backchannel 与原始人脸大小统计（2026-10-26）：** [统计口径、代码和筛选命令](related/P2_backchannel事件与原始人脸大小统计.md)、[结果目录](experiments/listen_backchannel_face_stats_20261026/)。将全量整合事件关联到 listener 原始画面的真实人脸短边 p10/p50、宽高、面积和检测有效比例，同时保留裁框尺寸；输出逐段 CSV、SQLite 和“事件数 × 人脸大小”门槛对照表。原始描述不变，按 sample_id 可回查完整事件，并可导出筛选清单。**此前 722,706 段的打标、整合、特征抽取及缓存校验均已完成**；整合问题标记保留，以下“尚未完成”条目是历史进度。

**当前首轮：AV / AV＋Event＋匹配 text-attention LoRA（2026-10-26）：** [实现与验证](related/P2_TextAttention_LoRA实现与验证.md)、[训练配置](configs/p2_listen_av_event_text_lora_v1.json)、[配对协议](configs/p2_listen_av_event_text_lora_pair_v1.json)。E0/E1 均训练原生视频/音频 text-attention LoRA；Event loss 只直接反馈到 E1 的 Connector。仍关闭 feature decoder/teacher。136 项 CPU 测试、LoRA 单/双进程参数差 0、50 段缓存和真实架构 meta 检查通过；本版真实 GPU S0 后续已通过，最新 24 卡及提速结果见[GPU 验证报告](related/P2_24卡联合训练GPU验证与提速.md)。[验证记录](experiments/p2_listen_text_lora_validation_20261026/README.md)。

**此前无 LoRA 简化版 GitHub 备份（2026-10-26）：** [固定版本源码、配置与恢复入口](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/p2-av-event-no-feature-20261026/p2_training/av_event_20261026)、[本地备份说明](../audio-driven-v5-data-processing/p2_training/av_event_20261026/README.md)。版本标签 `p2-av-event-no-feature-20261026`，保存无 feature、冻结 LTX 的 AV / AV＋Event 版本及 v4 复用依赖、测试和验证记录；备份时文件 SHA256 与源目录一致，**不包含后续 text-attention LoRA 修改**。模型、媒体、训练检查点和特征缓存仍保留在源机器。

**P2 首轮已切换为 AV / AV＋Event 对照（2026-10-26，代码与 CPU 验证完成）：** [简化实现与运行](related/P2_AV与事件监督简化实现.md)、[新配置](configs/p2_listen_av_event_v1.json)。显式关闭 feature 分支后不构建 decoder、不读 teacher、不计算 feature loss、不建立 features 优化器组；E0 detach 事件梯度，E1 回传共享 U。119 项 CPU 测试、两种模式的单/双进程对照及 50 段无 teacher 真实缓存检查通过。完整版本已备份，既有缓存身份和全量抽取任务保留；指定机器暂无空卡，本轮尚未重跑简化结构的真实 GPU S0。[验证记录](experiments/p2_listen_av_event_validation_20261026/README.md)。

**运行通知设置更新：** 按用户最新要求，已停止全量特征任务的每 20 分钟自动巡检与会话汇报；抽取、校验和运行状态页面继续工作。[停止记录](experiments/p2_features_full_crop384_20261025/monitor/DISABLED.json)。

**全量接续进展（2026-10-25）：** 722,706 段打标与信息整合已完成最终审计（`completed_with_issues`，问题记录保留），28 卡已开始特征抽取，共 56 个常驻 GPU 进程。本轮修复完成标记写出阻塞，并优化 CPU 媒体供给；[接续、媒体等价验证与恢复说明](related/P2_全量特征抽取接续与资源巡检.md)、[本轮巡检处理记录](experiments/p2_features_full_crop384_20261025/monitor/optimization-1792951862.json)、[实时状态](experiments/p2_features_full_crop384_20261025/status.json)。

**全量特征抽取已接续部署（2026-10-25，扩至28卡）：** 信息整合完成后自动接手 node17／node12／node4 各 8 卡和 `172.29.104.220` 的 GPU 4–7，共 28 卡。已验证每卡 2 个常驻进程、每进程 2 个 CPU 线程、预取 2 段及原生 CPU 输入预取；小样本复测比初始单进程快约 62%，40 份缓存逐元素一致，305 帧双进程峰值约 109 GiB。覆盖 722,706 段，完整保留至 11.95 秒／305 对齐帧。**原始打标全部结束后才启动20分钟定时巡检与报告。** [监控](http://127.0.0.1:8876/listen_benchmark_review/full_features/)、[结果与运行入口](experiments/p2_features_full_crop384_20261025/README.md)、[规则与恢复说明](related/P2_全量特征抽取接续与资源巡检.md)、[28卡配置压测](related/P2_特征抽取28卡配置压测.md)。新增四卡已产出原始打标结果，后续自动接整合与特征；全量特征计算和训练尚未启动。

**全量打标已接自动整合（2026-10-25）：** 722,706 段按已确认原文整合规则自动处理。CPU 提前准备已完成分片；四机 28 卡 GPU 释放后自动接手七类语义归属，随后 CPU 去重整合，描述和原始来源保留。已部署控制器及四机守护、断点恢复、异常重试和全量完整性审计。[实时整合监控](http://127.0.0.1:8876/listen_benchmark_review/full_fusion/)、[运行及结果目录](experiments/listen_full_fusion_crop384_20261025/README.md)、[规则与恢复说明](related/P2_全量打标自动整合接续说明.md)。最终候选位于该运行目录 `labels/`，全量审计完成后生成 `index.json` 和 `COMPLETE.json`；当前尚未全部完成。下方“仅存原始结果／尚未应用全量”是此前阶段记录。

**P2 特征抽取交接（2026-10-25）：** [代码、缓存与运行交接文档](related/P2_特征抽取代码与缓存交接.md)。覆盖 Omni 多层 hidden H、listener 目标特征、LTX VAE、Gemma 固定文本、训练时 H→U 融合的代码位置与依赖，附 50 条真实缓存路径、实测张量形状、CPU 检查及单卡/分卡重建命令。当前训练特征基座为 Qwen2.5-Omni-3B；请与打标模型区分。

**P2 单/双卡数值问题已修复（2026-10-25），本轮全部严格验收通过。** 在 `172.29.104.220` 的 GPU 4/5/6 上完成 50 条真实弱标签的 E0/E1 单卡与双卡各 7 步：两组最终参数最大差均为 **0**，原阈值 `2e-5` 未放宽；恢复重放最大差也为 0，独立生成及媒体解码通过。修复为严格确定性反向＋FP64 梯度累加/跨卡求和，保留 FP32 事件头；105 项 CPU 回归及 CPU 双进程对照通过。标签、BCE 和损失权重不变，旧失败记录保留。详见[修复与验证](related/P2_单卡双卡数值一致性修复与验证.md)、[最终汇总](experiments/p2_listen_ddp_fix_v2_20261025/summary.json)、[复测目录与命令](experiments/p2_listen_ddp_fix_v2_20261025/README.md)。这是 S0 工程验收，正式效果训练未启动；下方历史阶段结果保留。

**P2 首轮监督方案已确认（2026-10-25）：** 事件类型＋起止时间生成“事件 1／可用背景弱监督 0／不确定区域 mask”，跨边界按占用比例；复用现有 MLP＋BCE 和 E0/E1，不新增动作强度标注，不以人工负例库、Smooth-L1 或背景单独降权为前置条件。评价重点包含生成反应随 speaker 节奏与内容合理变化。见 [实验计划](related/P2_倾听事件监督实验计划.md#54-已确认事件区间与弱背景的派生契约)、[标签接入说明](related/P2_倾听事件监督实现与运行.md#31-本次弱背景方案怎样接入)、[最小接入要求](related/P2_倾听事件监督代码修改要求.md#0-2026-10-25-已确认方案与最小接入范围)。**派生接入已实现，91 项 CPU 测试和 50 条真实样本的派生→导入→事件缓存→梯度检查通过。** 见[弱背景实现与验证](related/P2_弱背景标签派生实现与验证.md)、[新配置](configs/p2_listen_weak_background_v1.json)。原始标签与历史 S0 保留；正式 E0/E1 效果实验尚未启动。

**P2七类真实 GPU 调试已通过（2026-10-25）：** [50段调试与训练就绪记录](related/P2_50段真实GPU调试与训练就绪记录.md)。在 `172.29.104.220` 空闲 GPU 4/5/6 上完成 50 段、213 秒数据的五类真实缓存，单卡 E0/E1 各 7 步、双卡 E1 7 步、断点恢复和两段独立音视频生成；72 项 CPU 测试通过。补齐仅供 S0 的弱正例检查及 debug 标签正式训练保护，unknown 未改为负例。正式数据到位后可按既有入口导入、缓存并训练；本批无摇头正例，正式训练与效果评测未开始。[七类迁移接口](related/P2_七类事件监督迁移实现与验证.md)、[基础配置](configs/p2_listen_event_v2.json)、[验收汇总](experiments/p2_listen_gpu_debug50_20261025/summary.json)。

**已有结果CPU原文整合（2026-10-25，当前入口）：** [打开50段原文结果](http://127.0.0.1:8876/listen_benchmark_review/existing_fusion_verbatim/)、[规则与代码](experiments/listen_existing_fusion_verbatim_20261025/README.md)。175个候选（134视觉＋41音频），保留模型原文、ASR原词及来源差异；已去掉“嘿”转笑声、Ed换Yeah和Says模板等内容改写。只整理已有时间、跨窗重复与证据关系，不用人工反馈填标签、不用GPU。682条输入观测原样保留；2个无法定位的词单列。尚未应用全量。

- **整合代码与规则备份：** [详细规则、字段与恢复说明](related/P2_打标信息原文整合说明.md)、[GitHub原文整合目录](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/main/event_labeling/integration_verbatim)、[本地备份及CPU重放](../audio-driven-v5-data-processing/event_labeling/integration_verbatim/README.md)。独立保存当前整合代码、10项测试、50段完整输入与候选结果，旧版本另行保留。 已推送提交 [`dff56f9`](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/commit/dff56f9f8806dc0d4e61dd9b1592a767b489423a)；326个复制文件哈希核对、10项测试和175条候选逐字节CPU重放通过。

**50段独立事件整合试验（2026-10-25，历史版本）：** [打开整合复核页](http://127.0.0.1:8876/listen_benchmark_review/integration_pilot/)、[规则／回归结果／代码](experiments/listen_integration_pilot_20261025/README.md)、[候选标签](experiments/listen_integration_pilot_20261025/labels.candidate.jsonl)。纳入5组人工反馈，保留全部682条原始观测，整理为182个独立候选（134视觉＋48音频）；9个事件时间留空，未知不作负例。ASR词证据130条→85个候选索引，真实发声次数另行确认。两次okay仍分开。页面覆盖全部50段，支持事件复核、漏标反馈和JSON导出。本轮尚未应用到全量、未准入训练。

**回应次数与内部边界复核（2026-10-25）：** [打开复核页](http://127.0.0.1:8876/listen_benchmark_review/response_review/)、[处理规则与数据](experiments/listen_response_review_20261025/README.md)。默认复核旧版发生合并的5组，也可切换全部50段。保留完整ASR词序与无效时间记录，同次转写的多个词不再合并；上层组不作为持续发声标签。S0805整段ASR的5个yeah中2个零时长，真实次数与边界仍待听音确认。页面支持填写区间、保存及导出人工复核JSON；不影响当前全量原始打标。

**原始预裁框 ≥384×384 打标（三机 24 卡，2026-10-24）：** 本轮已按用户要求收紧为 **722,706 段 / 1,224.604 小时**（原始、未取整的 listener 框边长 ≥384）；保持 512×512 裁剪、8 FPS、整段＋滑窗、独立 listener 音频和现有模型／prompt。旧结果按身份原样复用，当前 21 卡视觉＋2 卡 Omni 音频＋1 卡 ASR，运行中；优先保证质量再优化速度。[实时监控](http://127.0.0.1:8876/listen_benchmark_review/full_raw/)、[运行／原始结果／恢复说明](experiments/listen_raw_crop384_20261024T131300Z/README.md)、[状态 JSON](experiments/listen_raw_crop384_20261024T131300Z/status.json)。旧 1,419,124 段任务已停止并保留，不再作为本轮完成分母。本轮只存原始结果，不合并事件。

**连续音频回应合并（2026-10-24，历史版本）：** [查看整合后的 HTML](http://127.0.0.1:8876/listen_benchmark_review/response_episodes/)、[回应区间标签](experiments/listen_response_episodes_20261024T120000Z/labels.jsonl)、[完整原始观测](experiments/listen_response_episodes_20261024T120000Z/observations.jsonl)、[验证结果](experiments/listen_response_episodes_20261024T120000Z/validation.json)。50 段的音频已归类观测从 76 条整合为 **58 个回应区间**；视觉事件不变，七类候选合计 192 条。S0805 的连续 yeah 共 14 条重叠／相接观测合为 3.32–4.96 秒一个区间，点头独立保留。全部 682 条原始观测和 42 条词对齐记录原样保存；内部词记录不是重复发声次数。该层依据已有时间与相同内容合并，不改变模型语义归类，未重新分析媒体，也未确认 ±0.2 秒边界精度。

- **实现与重放：** [回应合并](src/listen_benchmark/response_episodes.py)、[导出](scripts/build_listen_response_episodes.py)、[回归测试](tests/test_listen_response_episodes.py)。运行 `python scripts/build_listen_response_episodes.py --source experiments/listen_semantic_labels_20261024T104629Z --output experiments/listen_response_episodes_new --preview data_previews/listen_benchmark_review/response_episodes_new`，两个输出目录须为空。七维训练导入器仍待适配，本次未启动训练。

**自然语言保留＋模型语义归类（2026-10-24，上游观测层）：** [查看结果 HTML](http://127.0.0.1:8876/listen_benchmark_review/seven_class_semantic/)、[完整自然语言标签 JSONL](experiments/listen_semantic_labels_20261024T104629Z/labels.jsonl)、[仅已归类候选](experiments/listen_semantic_labels_20261024T104629Z/category_candidates.jsonl)、[统计](experiments/listen_semantic_labels_20261024T104629Z/summary.json)。50 段的 **682 条开放事件全部保留**，Qwen3-VL 逐条理解描述后附加类别：210 条已归类、20 条待核验、452 条七类之外；每条保留原文、原始来源、时间和模型判断理由。42 个 ASR 词不变，未自动设为训练真值。惊讶等类别仍需结合画面核验，数量变化不等于准确率改善。

- **实现：** [语义 prompt／结构验证](src/listen_benchmark/semantic_mapping.py)、[准备请求](scripts/prepare_listen_semantic_mapping.py)、[8 卡批量推理](scripts/run_listen_semantic_mapping.py)、[导出入口](scripts/build_listen_seven_class.py)、[完整语义导出](scripts/build_listen_semantic_seven_class.py)。默认导出要求 `--semantic-run`，旧关键词规则仅通过 `--legacy-rules` 显式复现。
- **无 GPU 重放：** `python scripts/build_listen_seven_class.py --semantic-run experiments/listen_semantic_mapping_20261024T104527Z --output experiments/listen_semantic_labels_new_run`。输出目录必须为空；加 `--preview data_previews/listen_benchmark_review/<新目录>` 生成同层页面。本机查看使用远端 8876 转发。


**RealTalk 全量 v2（2026-10-25）：** [流程与结果目录](../../realtalk_processing/full_v2_20261025/README.md)、[全量进度](http://127.0.0.1:8798/full_v2/)。基于用户确认的边界版本重新处理692条原始视频；listener分离/优化音轨分别做SyncNet，未应用分辨率阈值；实际运行状态以进度页为准。全量代码已备份提交 [`3af6f00`](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/commit/3af6f005abad2a5b2647564cf59d4a8b881f69d9)。 并发扩容控制器已启用，按显存预算动态增加任务进程，[实时并发与资源](http://127.0.0.1:8798/full_v2/performance.json)；扩容代码备份 `b4188d4`。

**RealTalk 切分边界 v2（2026-10-25，用户已确认）：** [完整长视频切分审核](http://127.0.0.1:8798/boundary_v2_review/)、[生产代码与规则](../../realtalk_processing/boundary_v2_pilot_20261025/README.md)、[本地备份与恢复](../audio-driven-v5-data-processing/realtalk/boundary_v2/README.md)、[GitHub备份](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/main/realtalk/boundary_v2)。完整处理 `-1wCGAxqT_4`（611.64秒），137条候选／401.56秒，最长10.44秒；旧候选帧全部保留。正式接话前后继续提取、保留短回应，修复几何切点、词级句末、黑屏末帧、音频终点及缓存问题。206.72秒角色未知完整单列。用户确认本批切分；审核用原混音，全量清单未替换，未进行干净listener音轨认证或训练准入。 已备份提交 [`ae0b3c6`](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/commit/ae0b3c6cb77dd21c33a4842422628b8e953f2ee0)，49个源文件SHA256一致，备份及恢复目录边界测试通过。

**RealTalk 全量listener分离＋SyncNet（运行中）：** [实时进度与完整统计](http://127.0.0.1:8798/full_listener_sync/)、[规则及断点续跑说明](../../realtalk_processing/full_listener_sync_20261024/README.md)。692视频、129094个原pair全部入队，短片段/未知框均尝试补救；8张GPU并行，原角色与边界不变。口型过线、双方身份可见、短时暂定及待确认分开统计；当前非最终产量。

**历史关键词规则版（2026-10-24，已由上方语义版取代）：** [查看七类候选 HTML](http://127.0.0.1:8876/listen_benchmark_review/seven_class/)、[七类标签 JSONL](experiments/listen_seven_class_20261024T102123Z/labels.jsonl)、[统计与待核验数量](experiments/listen_seven_class_20261024T102123Z/summary.json)。50 段的 682 条开放事件中，115 条可映射、57 条待复核、510 条七类之外；原始描述及时间全部保留，42 个 ASR 词边界不变。零条类别不等于确认不存在该动作，候选尚未自动获得训练准入。

- **代码：** [七类映射](src/listen_benchmark/seven_class.py)、[独立导出入口](scripts/build_listen_seven_class.py)、[浏览器检查](scripts/check_listen_seven_class_review.py)、[58 项相关测试与重放验证](experiments/listen_seven_class_20261024T102123Z/validation.json)。
- **复现：** 在项目目录运行 `python scripts/build_listen_seven_class.py --legacy-rules --run experiments/listen_open_merge_20261024T084253Z --output experiments/listen_seven_class_new_run`；输出目录须新建。加 `--preview data_previews/listen_benchmark_review/<新目录>` 可生成同层复核页；本机查看需转发远端 8876 端口。原整合入口支持 `--seven-class-output <新目录> --semantic-run <已完成语义运行>` 连续导出，原开放标签协议不变。

**Listener 开放事件打标已认可并归档（2026-10-24）：** 采用 **Qwen3-VL 视觉＋Qwen3-Omni 纯音频＋ASR 逐词时间 → 开放事件整合**；50 段样本的 1,089 条候选整合为 682 条事件，保留每次动作／发声的起止时间和开放描述。用户认可当前标注，逐事件人工真值状态仍单独保存。

- **代码备份与恢复：** [GitHub 事件打标目录](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/main/event_labeling)（已推送版本 [`0d44862`](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/commit/0d448621beb968436cd4072b5a975f32d3f08fea)）、[本地使用与复现说明](../audio-driven-v5-data-processing/event_labeling/README.md)。包含 prompt、推理入口、整合算法、HTML 模板、测试、原始响应和结果；媒体与权重留在原机器。
- **代码入口：** [打标 prompt](src/listen_benchmark/prompts.py)、[多卡推理](scripts/run_listen_benchmark_vllm.py)、[逐词对齐](scripts/run_listen_word_alignment.py)、[描述规范化](scripts/run_listen_atomic_normalize.py)、[开放事件整合](src/listen_benchmark/open_merge.py)、[标签与 HTML 导出](scripts/build_listen_open_merge.py)。
- **结果与复核：** [整合前后 HTML](http://127.0.0.1:8876/listen_benchmark_review/open_merge/)、[标签 JSONL](experiments/listen_open_merge_20261024T084253Z/labels.jsonl)、[来源追踪](experiments/listen_open_merge_20261024T084253Z/provenance.json)、[整合报告](experiments/listen_open_merge_20261024T084253Z/integration_report.json)。本机查看 HTML 时转发远端 8876 端口。

**当前小批量审核入口（严格使用原全量pair）：** [12条双列对照：分离原音轨／进一步处理](http://127.0.0.1:8798/av_baseline_pairs/)、[批次来源与验证](../../realtalk_processing/av_baseline_pairs_20261024/README.md)。前6条新增、后6条原随机50对照，共93秒；角色、起止、crop直接沿用 full_20261023，未扩窗寻找回应。每条可同位置核对双人画面＋原混音。原混音口型检查12条支持清单speaker侧，仅作线索；目标音轨仍有分离不足，训练准入0。

**Listener反应直接展示（2026-10-24）：** [039裁剪视频＋本人处理音轨](http://127.0.0.1:8798/listener_reactions/)、[来源与审核状态](../../realtalk_processing/listener_reactions_review_20261024/README.md)。仅列已明确确认存在yeah反应的039，默认584×584 listener视频＋自适应音轨；不混入全静音或未确认短轮次，尚未认证完全无串音。

**RealTalk 自适应门限扩样16条（2026-10-24，已撤回，保留问题记录）：** [样本与复跑说明](../../realtalk_processing/av_adaptive_batch_20261024/README.md)、[按人物／类型筛选试听](http://127.0.0.1:8798/av_adaptive_batch/)。新增五对人物独立校准，15条新审核＋039回归，含短轮次与持续讲话对照；全静音和抑制不足结果均保留。35段新模型推理、70路输出，训练准入0。

**RealTalk 自适应门限试验（2026-10-24）：** [算法与校准限制](../../realtalk_processing/av_adaptive_gate_20261024/README.md)、[四段双人试听](http://127.0.0.1:8798/av_adaptive_gate/)。四段参考校准、两段新增留出，按人物分布和原录音音量调整门限；两人发声/串音分布重叠，启用保护上限，未判定分离合格。039右侧约−60.06dBFS，±12dB增益测试掩码一致，训练准入0。

**RealTalk 039更强门限与短缓冲（2026-10-24）：** [参数与结果](../../realtalk_processing/av_gate039_stronger_20261024/README.md)、[基线／新三档／单因素对照试听](http://127.0.0.1:8798/av_gate039_stronger/)。已审核−62档逐样本保留；短缓冲−60／−58／−56分别静音2.70／2.99／3.17秒，4.0–4.5秒均不变，5.5–6.5秒全静音。其他轻声位置可能受影响，待听辨，训练准入0。

**RealTalk 第一条039串音门限复核（2026-10-24）：** [参数与副作用](../../realtalk_processing/av_gate039_trial_20261024/README.md)、[−64／−62 dBFS试听](http://127.0.0.1:8798/av_gate039_trial/)。仅处理第一条右侧p1：−62档主要静音5.59–6.93秒，4.0–4.5秒不变，但3.5–3.63秒附近也被改动，待核查yeah起音。仅本例调参，训练准入0。

**RealTalk AV整段输出门限试验（2026-10-24）：** [参数与结果](../../realtalk_processing/av_gate_trial_20261024/README.md)、[原输出／轻度／中度试听及静音时间轴](http://127.0.0.1:8798/av_gate_trial/)。用户授权的低幅度静音小试：039两档逐样本不变；连续讲话对照左侧轻度静音3.42秒、中度5秒全静音。仅两例试验，未认定全量安全，训练准入0。

**RealTalk AV推理窗口对照（2026-10-24）：** [运行与限制](../../realtalk_processing/av_window_trial_20261024/README.md)、[3秒／5秒／整段试听](http://127.0.0.1:8798/av_window_trial/)。固定权重与输入，12路实际输出；3秒基线逐样本复现，较长窗口降低部分问题区间输出幅度，但仍有残留迹象，yeah完整性待审核，训练准入0。

**RealTalk WeSep 参考声音提取试验（2026-10-24）：** [代码与诊断](../../realtalk_processing/wesep_trial_20261024/README.md)、[六例试听](http://127.0.0.1:8798/wesep_trial/)。BSRNN＋ECAPA实际推理，751张量严格核验；合成持续重叠有效，但真实039两路几乎相同、目标缺席对照失败，训练准入0。

**RealTalk 039 音视频目标声音提取试验（2026-10-24）：** [运行与限制](../../realtalk_processing/av_listener_trial_20261024/README.md)、[试听页](http://127.0.0.1:8798/av_listener_trial/)。实际运行AV_MossFormer2_TSE_16K，原混音配两人独立嘴部视频；1308张量严格核验。039与p1连续发言对照均保留直接模型输出，另做视觉时序对齐对照；自动检查仍提示speaker残留，短应答保留待听辨，训练准入0。


**RealTalk 039 官方版本最小流程审计（2026-10-24）：** [复跑与定位说明](../../realtalk_processing/pyannote_official332_20261024/README.md)、[原混音／speaker／listener三轨试听](http://127.0.0.1:8798/pyannote_332_minimal/)。实际重跑pyannote.audio 3.3.2，关闭可选活动静音；完整两路输出与3.4.0逐样本完全相同，原生浮点WAV读回也完全相同，排除了此例的版本差异和导出改写。人物绑定复用相同波形的SyncNet证据；未宣称音质修复，训练准入0。


**RealTalk listener 短应答保留修正（2026-10-24）：** [代码与限制](../../realtalk_processing/listener_preserve_20261024/README.md)、[039优先试听](http://127.0.0.1:8798/listener_preserve/)。用户确认旧活动抑制删掉039的“yeah”，该版已否决；6条默认轨逐字节恢复到活动抑制前。另列DeepFilterNet3轻度降噪试验，039等仅表现为整体变小，未判作改善；剩余speaker串音未解决，训练准入0。


**当前候选范围更新（2026-10-24）：** 用户认为起始活动池抽样基本可保留；音质通过的主池与起始活动池统一纳入后续视觉筛选，共 **1,419,124 段 / 2,335.944 小时**。起始活动标记保留，原音质复核/拒绝/人工待定项未放行，尚未视觉准入。[统一候选索引](data_corpora/seamless_listen_local_full/accepted_audio_candidates/index.json)、[读取说明及合并尺寸统计](data_corpora/seamless_listen_local_full/accepted_audio_candidates/README.md)、[用户反馈记录](data_reviews/listener_onset_acceptance_20261024.json)。下方 2,002.627h 主池及 333.317h 储备池是原来源分组。

**RealTalk listener 串音后处理复核（2026-10-24）：** [规则与代码](../../realtalk_processing/listener_cleanup_20261024/README.md)、[listener裁剪／原画三路对照](http://127.0.0.1:8798/listener_cleanup/)。双人场景中另一人物明确为listener；对剩余候选波形增加模型活动抑制，复查speaker口型残留与listener发声保留。6条中3条变为全静音，未认定成功；全部仍待音质审核，训练准入0，不改变全量pair。


**起始活动储备池抽样（2026-10-24）：** [打开 32 条储备池审核](http://127.0.0.1:8875/listen_size_review/?group=reserve_all)：24 条均匀随机 + 4 条跨切入点人声 + 4 条切入后前 0.5 秒起声。源 VAD 区间、前后 3 秒上下文、单独音轨及可导出反馈均已接入原 HTML。储备池 333.317h 与主池 2002.627h 的标签和统计分开；未改变生产分组。

**RealTalk pyannote + SyncNet 音轨人物绑定复测（2026-10-24）：** [模型、代码与规则](../../realtalk_processing/pyannote_separation_20261024/README.md)、[带人物框的试听页](http://127.0.0.1:8798/syncnet_binding/)、[绑定清单](../../realtalk_processing/full_20261023/reports/syncnet_binding/binding.json)。6条失败样本经真实波形分离与135次口型交叉打分，3条具备双人绑定证据、3条仅绑定当前pair的speaker；通道身份变化与疑似串音保留待复核，未作音质认证或训练准入，未修改全量pair。


**全量抽样与人脸尺寸审核（2026-10-24）：** [打开 HTML](http://127.0.0.1:8875/listen_size_review/)：50 条随机主池样本 + 48 条人脸/裁框大小分档样本。双人原画幅、512 裁框预览、前后 3 秒切分边界、单独音轨和可导出反馈；阈值仅供预览、不修改筛选。[抽样、指标与代码说明](related/P3_全量抽样与脸部尺寸审核_20261024.md)。

**RealTalk 音轨提取 v2（试验，2026-10-23）：** [代码与规则](../../realtalk_processing/audio_separation_v2_20261023/README.md)、[新旧试听对照](http://127.0.0.1:8798/audio_separation_v2/)。原生16kHz参考人物条件提取、多参考一致性、目标缺席及局部身份检查；6条复测中未有完整段通过全部检查，局部身份支持候选6段/12.56秒，串音尚未认证，训练准入为0。所有原pair保留，未替换全量流程。

**标注代码测试包（随机 50 条，2026-10-23）：** 已从训练主池均匀随机抽取 50 条，提供双人完整画幅片段、前后各最多 3 秒上下文、归一化 WAV、预裁框和质检元数据。[目录及字段说明](data_exports/seamless_annotation_sample50_20261023/README.md)、[打包下载](data_exports/seamless_annotation_sample50_20261023.zip)。媒体路径相对包根目录，200 个视频解码/尺寸/时长校验通过；仅供标注代码测试，未新增视觉筛选。

**Seamless 全量完成（2026-10-23 13:49:46 UTC）：** 本轮关闭切镜检测，沿用已确认的完整源对话切分与音质规则，人脸/裁框只统计。64,087 对来源全部有结论：63,895 对完成处理（含 2,074 对源数据无效的诊断结果），192 对归一化异常隔离，剩余 0。训练主池音质通过候选 **1,200,212 段 / 2,002.627 小时**，起始反应储备池另有 **333.317 小时**；dev/test 单独保留。四机 32 卡从 12:48:46 到 13:49:46 共 **61 分钟**，包含 node4 显存保护停机、检查恢复及最终导出；导出耗时 100.46 秒。各节点工作进程正常退出。未执行最终裁剪或训练准入。[尺寸分布报告](data_corpora/seamless_listen_local_full/size_distribution_report.md)、[最终统计](data_corpora/seamless_listen_local_full/screening_summary.json)、[清单索引](data_corpora/seamless_listen_local_full/manifests/index.json)、[完成校验](experiments/p3_listen_local_full_20261023/completion_verification.json)、[代码备份](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing)。下方均为历史过程记录。

**Seamless 四机 32 卡提速（2026-10-23）：** 用户明确 Seamless 不需要 RealTalk 的切镜检测，正式入口已取消整段视频逐帧扫描；完整词时间线、角色切换、接话边界、音质规则与人脸只统计保持不变。新增 node8（zz-dev，实际内网 IP `172.29.104.220`）；原三机每卡 4 个模型进程，node8 目标每卡 16 个，其第 2 卡因其他作业显存占用较高上限 8 个。共最多 216 个活跃模型进程，按持续资源压力降档。保留 993 对无切镜影响的已完成结果，4 对受旧切镜边界影响者归档重跑。[实施细节](scripts/README_listen_v8.md#seamless-四机无切镜运行2026-10-23)、[实时进度](data_corpora/seamless_listen_local_full/screening_summary.json)。半小时至一小时为用户目标，是否达标以切换后的实测吞吐判断。

**Seamless 运行恢复（2026-10-23）：** 检查发现全量曾因 node4 一次 GPU 监控查询超时停止，实际已完成 475 对。已修复短暂遥测超时处理，新增有资源检查和次数上限的自动恢复；切分/音质/裁框只统计规则不变。[故障与修复说明](scripts/README_listen_v8.md#监控超时修复与自动恢复2026-10-23)、[实时状态](data_corpora/seamless_listen_local_full/runtime.json)、[自动恢复监控](data_corpora/seamless_listen_local_full/cluster/watchdog/status.json)。以下完成数量和启动记录为历史快照。

**历史 Seamless 启动版本（2026-10-23 10:35:55 UTC，已被无切镜版本取代）：** 已按用户验收的完整源视频逻辑启动独立全量入口，覆盖 64,087 对来源；音质基线保持 `v8-audio-severity-review-2`，裁框只统计、不筛选。三机 24 卡共享队列，每卡最多 4 个模型进程、每进程 2 路准备；逐帧镜头扫描使用已验证的 NVDEC 解码和原缩放算法。详见 [全量入口与恢复说明](scripts/README_listen_v8.md#完整源视频全量运行2026-10-23)、[运行状态](data_corpora/seamless_listen_local_full/runtime.json)、[音质进度](data_corpora/seamless_listen_local_full/screening_summary.json)、[原生人脸分布](data_corpora/seamless_listen_local_full/face_size_distribution.json)、[头肩参考框分布](data_corpora/seamless_listen_local_full/crop_size_distribution.json)。[GitHub 当前代码备份 `01318e6`](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/commit/01318e690928543a917a8ebaf1a479b3fbeb37b5)。以下旧全量状态为历史记录；新运行不会覆盖旧结果。

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

**P2 listen-only 七类事件监督已完成真实 S0。** 50 段 debug 数据的真实缓存、单卡/双卡训练、恢复与 source-only 生成已通过，详见 [GPU 验收记录](related/P2_50段真实GPU调试与训练就绪记录.md)。入口与数据格式见 [实现与运行说明](related/P2_倾听事件监督实现与运行.md)、[数据格式示例](examples/p2_listen/README.md)、[详细实验计划](related/P2_倾听事件监督实验计划.md) 与 [代码修改要求](related/P2_倾听事件监督代码修改要求.md)。本批仅用于代码测试，正式数据和独立效果评测另行接入；QC4 数据任务与本实验独立。

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

### P2 固定文本倾听与事件监督（七类代码与真实 S0 已通过，待正式数据）

- **七类迁移与当前接口（2026-10-25）**：[实现与验证记录](related/P2_七类事件监督迁移实现与验证.md)。候选仍需显式接受和逐类已知覆盖，CPU验证不替代真实S0。

- **真实 GPU 验收（2026-10-25）**：[50段真实模型调试与正式接入](related/P2_50段真实GPU调试与训练就绪记录.md)。全量缓存、E0/E1、双卡、恢复和短/长片段生成通过；仅 debug 弱正例，无正式效果结论。

- **事件打标交接检查入口（2026-10-24）**：[代码、模型、测试数据与复验说明](related/P2_事件打标交接检查说明.md)。集中列出四模型权重和实际端口、prompt/裁剪/校验代码、Seamless 与 RealTalk 小样本位置、逐请求原始响应、对照结果、离线检查和小样本复验命令，以及已知质量与续跑问题。
- **完整对照测试代码（2026-10-24）**：[测试矩阵、原生视听输入与运行说明](related/P2_事件打标完整对照测试.md)。独立入口覆盖九个主配置、三个辅助配置和四个输入控制配置；支持原始回答留存、断点恢复、盲审模板及人工 GT 事件评测。node17/node12/node4 共 24 张 H200 已完成 50 条×16 配置对照：[完整结果与分路组合对比](experiments/listen_benchmark_cluster_20261024T014809Z/最终对照结果.md)。最终纳入 1450 次请求，代码回归 57 项通过，显卡已释放。没有人工 GT，本轮结果仍是待审草稿，未续跑原全量生产打标。
- **开放描述结果分析**：HTML 首页现为 discovery 分析，提供 13 种事件配置的独立统计、5 个可播放复核样本、原始描述与 audit 切换。[分析说明](experiments/listen_benchmark_retest_20261024T032648Z/开放描述分析.md)。Qwen3-VL 有 254 条开放描述，其中 125 条文本涉及眼部开闭；Omni 视觉存在重复切片，原生视听存在覆盖整段的粗时间区间。计数不等于有效 backchannel 或准确率。
- **HTML 模型对照与选型分析（已完成 24 卡重测）**：[打开交互页面](http://127.0.0.1:8875/listen_benchmark_review/) · [上轮页面](http://127.0.0.1:8875/listen_benchmark_review_initial/) · [重测分析](experiments/listen_benchmark_retest_20261024T032648Z/选型分析.md)。16 配置、50 样本、1450 次请求全部重新执行。修正时序后 Qwen 原生视听发声事件 4→14、点头仍 1；Joy 点头 41→40，Qwen3-VL 仍 2。当前没有通过人工质量验收的生产方案。页面提供前后变化表、旧原生视听回答、真实媒体同步对照和资源曲线。
- **2026-10-23 八小时全量事件打标任务已结束**：Joy 视觉 + Qwen3-Omni 独立 listener 音频、64 worker 共处理 57,038 条，结构完整 54,820 条；最终快照含失败 2,218 条及剩余 1,160,908 条。22:45 停止领取新样本，约 22:51 完成导出，当前未续跑；[方案、质量限制与故障记录](related/P2_Seamless全量事件打标_8小时运行.md)、[最后检测状态](experiments/listen_labeling_full_20261023/watchdog/latest.json)、[最终结果导出索引](experiments/listen_labeling_full_20261023/exports/latest.json)。所有结果仍待人工审核；结构通过不等于事件准确。

- Seamless 随机 50 条已接入完整视听输入：50/50 离线核验通过，2,661 张原画裁剪后送模图均为 512×512，使用独立 listener WAV；3 条真实媒体的 12 个模拟 HTTP 请求通过。早期 node8 的 Joy、Qwen3-Omni 和 MiniCPM-o-4.5 联调发现格式、有效覆盖及联合输入/推理耗时问题；不视为标签质量验收通过。Joy 权重已下载；后续全量打标进展见上条，尚未训练。早期实测及复现见 [真实数据与模型小样验证](related/P2_事件打标_真实数据与模型小样验证.md)。见 [核验及联调说明](related/P2_倾听事件自动标注实现与运行.md#10-seamless-随机-50-条完整双路输入与实模型联调)、[双路清单](experiments/listen_labeling_seamless_sample50_512/av_manifest.json)。
- RealTalk 随机 50 条已做打标输入兼容性检查：原画按固定 crop 在线裁剪后 resize 为 512×512；25 条有框通过，622 张采样图均为 512×512，25 条缺框单列。包内是混合音轨，当前只准备视觉模式；未调用模型、未准入训练。见 [处理说明与命令](related/P2_倾听事件自动标注实现与运行.md#9-realtalk-随机-50-条原画坐标裁剪与-512-输入核验)、[核验报告](experiments/listen_labeling_realtalk_random50_512/preparation.json)、[裁剪对照图](experiments/listen_labeling_realtalk_random50_512/previews/020_comparison.jpg)。
- 事件自动标注（v2 开放发现）：[实现与运行说明](related/P2_倾听事件自动标注实现与运行.md)；[入口](scripts/label_listen_events.py)、[通用配置模板](configs/listen_labeling.json)、[清单与自由描述示例](examples/listen_labeling/README.md)。Joy 静音视觉 + 音频模型完整波形先自由发现行为、保留证据，再独立查看原媒体检查点头/发声；支持 Omni 两阶段对照、分阶段续跑、人工修订和 P2 导入。含批量源读取与恢复用例共 37 项合成媒体/模拟 HTTP 测试通过，node8 实模型结果已单独记录，标注质量待验收；[验证记录](experiments/listen_labeling_code_validation/result.json)。
- 实验计划：[P2_倾听事件监督实验计划.md](related/P2_倾听事件监督实验计划.md)。覆盖路线选择、V8/QC4 小规模数据准入、六类视觉＋一类音频标注、共享 U 监督、成对训练预算、生成事件评测、继续条件和失败分流。
- 代码要求：[P2_倾听事件监督代码修改要求.md](related/P2_倾听事件监督代码修改要求.md)。覆盖旧 P2 复用边界、新模块职责、输入/目标隔离、schema、时间轴与缓存、梯度/DDP/恢复、拟定 CLI 和验收测试。
- 来源为 [audio-driven-v4 中的 Pipeline 2](../audio-driven-v4/docs/PIPELINE2_PLAN.md) 的多层条件提取；当前增加 P2.1 的原生 text-attention LoRA 能力，冻结基座原始参数。融合 Thinker 8/18/28/35 层，不继承旧 GT 文字计划作为输入。
- 首轮主比较：两组关闭 feature decoder 和 feature loss，只读取 input、VAE、text、events；两组均由 AV loss 更新相同范围的 text-attention LoRA。E0 用 AV loss 更新 Connector，事件头读取停止梯度的 U，仅作 probe；E1 用 AV＋0.05×Event loss 更新同一份生成条件 U。两组保持相同的初始化、数据、预算和采样协议。
- 默认 pilot 建议 train/dev/test 为 2048/128/128 段、每组 1000 步，先做一个训练 seed 的成对实验，有信号后复验第二个 seed；这些是待实施的起始预算，不是已完成的数据或训练数量。
- 实现与运行：[AV / AV＋Event＋text-attention LoRA](related/P2_TextAttention_LoRA实现与验证.md)；入口：[scripts/p2_listen.py](scripts/p2_listen.py)；代码：[src/p2_listen/](src/p2_listen/)；首轮请显式使用[新 LoRA 配置](configs/p2_listen_av_event_text_lora_v1.json)和[配对协议](configs/p2_listen_av_event_text_lora_pair_v1.json)。[无 feature 冻结 LTX 版](related/P2_AV与事件监督简化实现.md)、[含 feature 版](related/P2_倾听事件监督实现与运行.md)与旧配置保留供后续消融。
- 数据接口：[示例与说明](examples/p2_listen/README.md)。50 段已完成完整版本的真实 S0；简化版已通过无 teacher 的真实缓存检查，尚未重跑 GPU S0。正式数据使用新的实验配置/输出，并核验 coverage、准入及分组划分。
- 验证：[简化版 119 项 CPU 测试、单/双进程对照与 50 段缓存检查](experiments/p2_listen_av_event_validation_20261026/README.md)。历史完整版本：[GPU 汇总](experiments/p2_listen_gpu_debug50_20261025/summary.json)、[GPU 恢复重放](experiments/p2_listen_gpu_debug50_20261025/smoke/replay_validated.json)、[双卡对照](experiments/p2_listen_gpu_debug50_20261025/ddp/comparison.json)。真实 S0 验证运行链路，不代表生成质量结论。P3 原有实现和 QC4 处理保留。

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
