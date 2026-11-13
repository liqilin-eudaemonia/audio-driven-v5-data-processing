**两项Whisper辅助损失消融已接四机31卡顺序队列（2026-11-06）：** 等当前规划LoRA释放资源后，依次训练AV＋0.05 Event＋Whisper、AV＋两路0.025 Feature＋Whisper；node8 GPU7预留。沿用旧500h／原始H、时长分组、prefetch2／512MiB通信桶／GPU Adam，每组8000步、每500步保存。18项单项消融／队列CPU回归及缓存校验通过（本次连同共享模块共65项），无需新缓存；首组Event＋Whisper已自动开训，31卡启动、最长片段反传及500步完整检查点（模型／Adam／31份RNG）验收通过，Feature组后续接续。[代码、配置、验收与运行交接](related/P2_Whisper单项辅助损失消融代码交接_20261106.md) · [当前队列状态](experiments/p2_whisper_single_aux31_20261106/state.json) · [实测步速](experiments/p2_whisper_single_aux31_20261106/speed.json)。

**新版 Qwen LoRA 已切换四机32卡接续（2026-11-06）：** node17／node8／node12／node4各8卡，32个rank的NCCL、Adam／scheduler／RNG恢复、有限梯度及完整保存验收通过；从原完整checkpoint-1500接续，3步验收到1503后继续正式训练，已超过原单卡停止时的1670步。全局batch256、训练704,368条、总预算2,752步、数据／模型／loss保持，末尾112条不补齐或丢弃；原采样器重放与实际32卡顺序一致。完整训练尚未结束。[运行、验证与配置](experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106/README.md) · [实际32卡启动验收](experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106/LAUNCH_ACCEPTED.json) · [逐步指标](experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106/training/step_metrics.jsonl)。

**本轮联合Whisper已加入原可视化（2026-11-06）：** 两套测试集各14组，7组消融、21项对照。新增联合6000／8000，明确与仅AV、AV＋Event、AV＋Feature、原Event＋Feature和仅AV＋Whisper比较；含6000→8000及预算不等的终点参考。原值、B−A、箭头、第一／第二名颜色、音频及覆盖保留。 [ResponseNet联合消融](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=responsenet&ablation=joint_whisper) · [Seamless联合消融](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper) · [有声视频基线对照](http://localhost:8876/listen_benchmark_review/p2_seamless200_20261105/?preset=joint_av&metrics=1)。588项数值／差值检查、浏览器与32份新增媒体／PCM核验通过。

**已完成（2026-11-06 02:23 UTC，北京时间10:23）：** 6000／8000步×Seamless200／ResponseNet200，共800条生成、导出、62个评分分片及全局FID/FVD汇总完成，最终逐条验收通过。Seamless两组各1条情绪／运动支持不足（199/200），其余图像指标200/200；音频执行错误0，UTMOS因短回应为N/A。训练保持停在8000步，巡检goal保持暂停。 [结果与验收](experiments/p2_combined_whisper_eval_20261106/RESULTS.md)。

**新版 LoRA 第1,500步检查点已独立验收（2026-11-06 01:27 UTC）：** 训练保存后继续推进，仍为原训练进程；CPU重读504份语言LoRA权重（59,867,136参数）及504份Adam状态均有限，优化器步数与scheduler均为1,500，Python/NumPy/CPU RNG可恢复、CUDA RNG载荷可读；全部6个关键文件SHA已记录，封存数据/代码一致。检查未创建GPU上下文，也未启动第二份训练。正式训练继续，尚未完成。[1,500步验收](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/checkpoint_audits/checkpoint-1500.json) · [1,000步验收](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/checkpoint_audits/checkpoint-1000.json) · [全部检查点验收](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/checkpoint_audits/) · [小时巡检](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/latest_full_check.json)。

**新版 LoRA 40分钟首检已通过（2026-11-05 15:20 UTC）：** 筛选后已自动衔接node8 GPU7正式训练；首检核对真实PID/启动时间/命令、连续步进、有限loss与梯度、冻结代码/配置，未发生本轮训练重启或恢复。首检后继续每小时完整检查，下一次16:20:16 UTC；本机30秒守护持续运行，完整训练尚未结束。[首检及后续小时报告](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/latest_full_check.json) · [全部报告](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/reports/) · [实时守护](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/state.json)。

**新版全量筛选完成，GPU7规划 LoRA 已正式接上（2026-11-05）：** 1,107,313条标签全量严格筛查、train/dev发声恢复和完整导出/覆盖/媒体检查均通过；最终train704,368条／1,191.9879h，dev5,296条，独立test7,087条，未决2条不进训练。最终输入封存及12条真实processor/音视频对照通过，真实GPU反传、完整保存及adapter重载通过；node8 GPU7正式训练已实际更新，原始Qwen2.5-Omni-3B新建r32 LoRA、有效batch256、一轮2,752步，每500步保存、不跑dev评估。巡检按15:20:16 UTC首检及其后每小时继续，完整训练尚未结束。[筛选最终摘要](experiments/listen_j_upright_response_sft_20261105/final/summary.json) · [真实开训验收](experiments/p2_planning_sft_upright_3b_gpu7_20261105/TRAINING_STARTUP_ACCEPTANCE.json) · [逐步指标](experiments/p2_planning_sft_upright_3b_gpu7_20261105/training/step_metrics.jsonl) · [巡检状态](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/state.json)。

**新版规划 LoRA 定时巡检已启用（2026-11-05）：** 按用户要求，首次完整检查定于15:20:16 UTC（请求后约40分钟）；顺利开训后每小时完整检查，未接上则每5分钟复查，另有node8本机每30秒进程／步进／GPU健康守护。核对PID＋启动时间＋命令，观察失败不重启；持续10分钟无步进且至少2分钟GPU为0才确认停滞，写检查点期间豁免。仅恢复本轮筛选／LoRA进程，优先完整检查点；无检查点则保存故障目录后按相同数据/seed重放，不跳样本。巡检覆盖至完整训练及final_adapter验收完成。[巡检状态](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/state.json) · [首次及小时报告](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/latest_full_check.json) · [事件与恢复](experiments/p2_planning_sft_upright_3b_gpu7_20261105/monitor/events.jsonl)。

**新版筛选结果接续 Qwen 规划 LoRA 已排队（2026-11-05）：** 用户授权筛选完成后开训；独立后台队列等待新版全量筛选 COMPLETE、来源SHA及媒体验收通过，再封存最终通过的全部 train，dev/test不进梯度。沿用原始Qwen2.5-Omni-3B新建语言层LoRA（r32/alpha64），node8（172.29.104.237）仅GPU7，micro8×累积32维持有效batch256，一轮、LR1e-4、每500步保存、关闭dev评估；复用严格匹配首帧PNG＋隔离AUTO4解码＋worker4/prefetch4、BF16/SDPA/fused Adam。18条新媒体/processor一致性、19项CPU回归及真实Trainer累积归一化对照通过；最终选集验收后会再做12条输入复核、GPU反传和保存后重载，才启动正式训练。[接续状态](experiments/p2_planning_sft_upright_3b_gpu7_20261105/QUEUE_STATUS.json) · [冻结配置](experiments/p2_planning_sft_upright_3b_gpu7_20261105/config.json) · [输入对照](experiments/p2_planning_sft_upright_3b_gpu7_20261105/PRELIMINARY_INPUT_EQUIVALENCE.json) · [累积归一化检查](experiments/p2_planning_sft_upright_3b_gpu7_20261105/GRADIENT_ACCUMULATION_VALIDATION.json) · [独立队列入口](scripts/queue_j_upright_planning_lora.py)。

**新版全量 Qwen 规划 SFT 筛选（2026-11-05）：** 按用户指定仅使用 node8（172.29.104.237）GPU7；输入转正后的全部 1,107,313 条新标签，保留原 train/dev/test，独立于 AV500h 选集。冻结历史严格判断及代码否决，自动接 train/dev 非零音频发声恢复、短引文／原文句子约束重试，test 仅严格判断；未决样本不进入训练，caption 正文保留。新旧样本索引逐条核对，提供新版媒体读取器；筛选完成后由独立队列接续新版LoRA，见上方记录。[阶段状态](experiments/listen_j_upright_response_sft_20261105/RUN_STATUS.json) · [严格筛查进度](experiments/listen_j_upright_response_sft_20261105/strict/status.json) · [启动实测与4,096条独立复验](experiments/listen_j_upright_response_sft_20261105/STARTUP_ACCEPTANCE.json) · [规则一致性核验](experiments/listen_j_upright_response_sft_20261105/FROZEN_POLICY_VALIDATION.json) · [首条真实输入读取](experiments/listen_j_upright_response_sft_20261105/STARTUP_MEDIA_READBACK.json) · [新版数据读取器](scripts/j_upright_response_sft_data.py)。

**AV＋Event＋Feature＋speaker Whisper已正式四机31卡开训（2026-11-05）：** node17／node12／node4各8卡，node8仅0–6卡；**node8（172.29.104.237）GPU7留给用户**。沿用旧500h／285,638条train、原始H和1／0.05／0.025／0.025损失、CPU预取2批＋512MiB通信与GPU Adam；有效batch31，duration496，完整一轮9215步、每500步保存。为31卡从相同预训练初始权重重新开始，旧18步仅作测试；2313个初始张量逐值一致，31rank真实更新／五组梯度已通过。全部真实样本与噪声ordinal一一保留，尾批27条；同样本量的32卡6000步约对应31卡6194步，不能把batch变化视为完全相同优化轨迹。[正式运行、状态、预留卡与恢复](experiments/p2_j_whisper_event_feature_train31_20261105/README.md) · [当前进度](experiments/p2_j_whisper_event_feature_train31_20261105/status.json) · [启动验收](experiments/p2_j_whisper_event_feature_train31_20261105/STARTUP_ACCEPTANCE.json)。

**当前train＋dev版500h五项训练缓存全部完成（2026-11-05）：** Qwen H、Gemma、AV VAE、speaker Whisper、listener Feature teacher均覆盖284,218段／500.00125913h，保留源split、排除test。AV/Whisper/teacher只补增量，旧文件直接引用；全量验收及五项联合读取通过。本缓存任务未启动训练。[统一缓存入口、增量及恢复记录](experiments/listen_j_upright_av500h_train_dev_v3_20261105/CACHE_README.md)。

**Seamless200评分无损GT解码缓存已启用（2026-11-05）：** 固定200条GT跨权重复用完整RGB，真实短/长样本像素与时间审计完全一致；实际GPU评分两条的decode总耗时14.90→0.45秒、3.17→0.12秒，全部指标及FID/FVD特征逐值一致。17项回归通过；后续worker接入，原评分/推理不停。[实现、实测、缓存及读取审计](experiments/p2_seamless200_tenway_24gpu_20261105/DECODE_CACHE.md)。

**本轮评测提速更新：24卡推理＋node8四卡提前评分（2026-11-05）：** node8 `172.29.104.237`仅GPU0–3加入，先计算基础四组指标，与三机24卡原生推理重叠；全部生成完成后24卡自动并入评分，共28卡。node8每卡8路、其余三机释放后每卡4路；共享GT特征及batch64等历史配置不变；推理完成后的控制器加入同一评分队列，不重复开评分。13项队列／评测回归通过。[本轮安排、状态和结果](experiments/p2_seamless200_tenway_24gpu_20261105/README.md)。

**原画转正与重复标签八卡重打标已完成（2026-11-05）：** 全量新标签保留1,107,313条；747个原画横置来源／1,699段中1,690段成功转正并重新检测打标，其余不可用段剔除；全部36条重复标记重打标，36条保留、0条剔除。正常宽屏与歪头、自动通过与待复核均保留，原音频／时间／split不变。全部1,107,323个目标归账、原字节保留与新素材SHA独立验收通过。[新全量统一入口](experiments/listen_j_labels_upright_20261105/README.md) · [执行及八卡实测](experiments/listen_j_upright_regen_20261105/README.md)。

**当前：停止组合损失训练，三机24卡重测10组（2026-11-05）：** 用户要求立即转评测；训练正常保存并停止于完整第7610步，node17／node12／node4各8卡用于新Seamless200。历史9组加原始H＋AV＋Event＋Feature第7000步，共2000输出；7000步累计190000条，对照原32卡6000步的192000条，差约1.04%。复用此前800条采样及656条成片，node8自动接续已取消，不占用node8。沿用原采样参数及GPU评分配置。[本轮配置、进度与结果](experiments/p2_seamless200_tenway_24gpu_20261105/README.md)。下方九组node8等待／三机继续训练条目为历史状态。

**Seamless200九组评测已暂停并等待空机（2026-11-05）：** 用户要在node8运行其他任务；本评测8卡worker及原重启守护已停止，完整结果保留。轻量等待进程确认其他GPU任务运行并结束、8卡连续空闲120秒后自动续跑；三机训练未动。[暂停、已保留结果与接续状态](experiments/p2_seamless200_nineway_20261105/README.md)。

**原画转正已验收，全部重复标签八卡重打标已排队（2026-11-05）：** 全量方向候选及额外宽屏原画逐来源核对；747个横置来源／1,699段均完成原生裁框转正、原PTS保留及重新测量，1,690段可用、9段按原清晰度规则剔除；正常宽屏及歪头保留。全部36条重复标记与转正素材合并为1,734条重打标任务，真实输入检查通过；新node8的8卡当前被九组Seamless200评测占用，已部署本机自动等待接续、打标、导出及独立验收。新全量入口尚未发布。[修复依据、执行与实时状态](experiments/listen_j_upright_regen_20261105/README.md)。

**新Seamless200快速验证集及历史9组重测（2026-11-05）：** 从已接入6624条test中读取11月5日最新caption，固定200条／19.99分钟及48条核心集；64条实际静音、116条短声学活动、20条较长活动，200个独立pair，180条人脸p10≥224。全部新旧媒体／最终裁框／净化音轨逐条兼容核对及4项CPU检查通过。全部历史9组、1800输出已在新node8 `172.29.104.237` 本机8卡队列启动，沿用历史采样和GPU评分提速配置。[选集、原文与推理入口](experiments/seamless_quick200_latest_20261105/README.md) · [9组配置、状态和结果](experiments/p2_seamless200_nineway_20261105/README.md)。

**原始H＋AV＋Event＋Feature已切换三机24卡继续（2026-11-05）：** 用户指定暂不等待node8；node17/node12/node4各8卡，从原5500步模型及Adam逐值迁移，global28→24，按384条时长分组，剩余131638条真实样本/噪声ordinal全部保留，末批仅补齐2条，最终10985步。原损失权重、数据、teacher及每500步保存不变。7项CPU检查及模型/Adam/24份所选RNG逐值校验通过；24rank真实恢复更新、组合损失公式及四组梯度已验收，见新目录STARTUP_ACCEPTANCE.json。[当前配置、迁移、状态和日志](experiments/p2_j_base_h_av_event_feature_continue24_20261105/README.md)。

**已排队：Whisper第8000步完整保存后恢复组合损失实验（2026-11-05）：** 当前训练先正常保存并退出，再严格从原始H＋AV＋Event＋Feature第5500步恢复；沿用原四机28rank/global28、duration448、原损失权重和每500步保存，目标10202步。真实检查点、104源码、数据／teacher身份与28rank RNG预检通过，4项接续测试通过；Whisper8000步已完整保存并独立SHA通过。node8约62.7万PID1僵尸导致进程配额耗尽，已停止本接续的首次启动；后台接续加入每机至少4096进程／线程空位检查，等待平台重启node8后恢复。[接续协议、实时状态与日志](experiments/p2_whisper8000_to_event_feature_20261105/README.md)。

**新标签分组人工抽检页已准备（2026-11-05，待用户检查）：** 从本轮新标签固定抽取自动通过24条（P01–P24）和带复核标记36条（R01–R36），覆盖非零音频／静音、不同人脸大小、5类主要提示及生成异常；60条来自不同pair。沿用最终裁框素材复核页及float32双音轨同步播放器，展示新caption、实际输入线索、原复核原因，支持分组／原因筛选和人工意见保存导出。60条视频加载、120路PCM哈希、音轨切换／拖动及意见导出通过，分层样本不能直接估计全量准确率。[检查通过组24条（8876）](http://localhost:8876/listen_benchmark_review/j_caption_quality_20261105/?group=pass) · [检查复核组36条（8876）](http://localhost:8876/listen_benchmark_review/j_caption_quality_20261105/?group=review) · [抽样规则、编号与验收](experiments/listen_j_caption_quality_review_20261105/README.md)。

**全量新标签统一入口已导出并完成覆盖核对（2026-11-05）：** 以最新 `listen_j_reaction_vocal_omni_20261103` 打标结果导出1,107,322条新caption；全部1,107,323个目标ID均已归账，另1条新版检测 `no_detected_face` 无新标签，单列例外且未回填旧标签。独立重读全量清单及哈希，重复／遗漏／来源元数据错配／group跨split均0；686条原始标签分层回读通过。原生成复核标记全部保留，尚未执行新的语义筛选或行为统计。[后续处理统一入口与读取说明](experiments/listen_j_labels_manifest_20261105/README.md) · [新标签清单](experiments/listen_j_labels_manifest_20261105/labels.jsonl) · [覆盖核对](experiments/listen_j_labels_manifest_20261105/COVERAGE_AUDIT.json) · [唯一缺项](experiments/listen_j_labels_manifest_20261105/exceptions.jsonl)。

**九小时四机巡检已完成（2026-11-05）：** 原定9次小时检查全部完成，四机32个训练PID／start_ticks保持一致，未发生训练重启、掉卡或异常停步；训练平均5.032秒／步，第6,500步完整检查点独立SHA通过。窗口末6,573步，终检时6,595步；巡检退出，四机训练及原守护继续。[逐项验收、全部小时报告与当前状态](experiments/p2_j_qwen_whisper_av_tuned_20261104/monitor_nine_hours/README.md#九小时结束与验收)。

**Whisper 32卡已恢复256MiB通信桶正式训练（2026-11-04）：** 从完整第209步继承模型、优化器及32卡RNG，全层重算、duration512和GPU驻留Adam保持不变；同512个样本实测5.4357→5.0634秒／步，步时减少6.85%，梯度／损失逐值一致。真实恢复的210–225步平均5.223秒／步，32卡最长样本及首步loss对照通过；粗估剩余13–15小时，按后续实测更新。四分之三重算虽为4.8221秒，但峰值133.998GiB，余量不足，未采用。[当前配置、验收、状态、快照与实测](experiments/p2_j_qwen_whisper_av_tuned_20261104/README.md)。

**前阶段：Whisper立即时长分组验证（2026-11-04）：** 保留完整第2步检查点，从第3步启用duration512，首个完整512样本窗口平均5.444秒／步；32卡带完整Adam状态的最长片段检查通过。原500h／285,638条、global32、8,927步、每500步保存、原Qwen H＋Gemma＋Whisper且仅AV目标均不变。复用全层重算、NCCL和32MiB FP64流式归约；改用GPU驻留Adam状态，避免每步CPU往返。按首窗口粗估剩余14–16小时，含存盘余量；以随后实测更新。[当前配置、迁移验收、状态、140文件快照与加速依据](experiments/p2_j_qwen_whisper_av_duration_now_20261104/README.md)。

**缓存验收后四机训练初始检查已完成（2026-11-04）：** speaker Whisper缓存285,638条已全部提取和验收；原始Qwen H＋Gemma＋Whisper双路条件，仅AV loss。原运行完成32卡最长样本前后向及正式2步检查，旧检查点完整保留；后续转到上方立即分组运行。[初始配置、状态与日志](experiments/p2_j_qwen_whisper_av_20261104/queue/README.md)。

**Whisper缓存已补齐历史提速策略并更新接续（2026-11-04）：** 四机32卡、每卡batch16／4读取线程、双缓冲音频预取＋单线程异步写盘、FP32关闭TF32，终检16线程。每卡启动先验证短音频与27秒跨窗合批／逐条一致性。80项CPU回归通过，真实样本对旧缓存最大误差0；GPU吞吐尚待运行实测。新缓存目录为`p2_j_speaker_whisper_cache_20261104_prefetch_v2`，接续守护已恢复，当前打标未停。[实现、配置和验证](related/P2_Whisper缓存流水线提速与接续_20261104.md)。

**speaker Whisper全量缓存已自动排队（2026-11-04）：** 等当前四机打标完成固定清单验收并释放32卡，自动提取原500h／285,638条speaker音频的冻结Whisper特征，随后验收缓存；不启动训练。13项接续检查通过，后台守护已运行。[接续说明与状态](experiments/p2_j_speaker_whisper_cache_20261104/queue/README.md)。

**Qwen H＋Whisper双路条件，仅AV实验代码已准备（2026-11-04）：** 保留原始未SFT Qwen H、Gemma及原500h目标缓存，新增冻结Whisper-large-v3 speaker特征和独立音频／视频cross-attention残差，采用共同秒数时间RoPE；只训练AV目标，Event／Feature不运行。59项CPU回归、两进程Gloo更新逐值一致、真实单条Whisper提取与原缓存联合读取通过；未启动全量缓存或训练，22B GPU显存／吞吐及32卡NCCL待验收。[结构、缓存、运行与验收交接](related/P2_QwenH加Whisper双路条件_AV训练交接_20261104.md) · [实验配置](configs/p2_j_qwen_whisper_av_20261104.json) · [统一入口](scripts/p2_j_whisper_av.py) · [speaker专用缓存入口](scripts/p2_j_speaker_whisper_cache.py) · [检查记录](experiments/p2_j_qwen_whisper_av_codecheck_20261104/README.md) · [133文件源码快照](releases/p2_qwen_whisper_av_v1_20261104/README.md)。

**2026-11-04 输入准备加速已推广32卡：** 每卡6个CPU准备进程、预取12条；365秒生产窗口约24.16段/秒，32卡PID稳定。检测已完成并停止空转监督。[实测与当前有效配置](related/P2_四机32卡检测打标重叠运行_20261103.md) · [整机测量](experiments/listen_j_reaction_vocal_omni_20261103/cluster_control/prep6_global_measurement_result.json) · [代码与配置快照](releases/listen_caption_prep6_20261104/effective_profile.json)。

**四机32卡检测／打标接续（2026-11-03）：** 第5500步检查点完成并校验后已停止训练，32个修复版worker均已产出；CPU检测与GPU打标重叠运行；最终J切分retain清单、修正版裁框、官方时间并集净化listener音轨保持不变。[分工、参数、检查点门控与90分钟巡检](related/P2_四机32卡检测打标重叠运行_20261103.md) · [实时控制状态](experiments/listen_j_reaction_vocal_omni_20261103/cluster_control/state.json)。

**原始H＋AV＋Event＋Feature已暂停于5500步完整检查点（2026-11-03）：** 后续按用户要求切换四机32卡检测／打标，未恢复训练。原运行配置： 用户指定改用node17/node8/node12各8卡及node4的0–3卡；global batch28、完整一轮10202步，AV＋0.05 Event＋0.025视频Feature＋0.025音频Feature，冻结原始Qwen H。复用最终裁框、净化音频和已验收teacher；duration448从第1步、每500步保存。45项CPU回归、真实3条读取、28rank16步完整反传和断点恢复均通过，已继续正式训练。[实际配置、状态与恢复](experiments/p2_j_base_h_av_event_feature_20261103/README.md)。

**ResponseNet200三组6000步模型Demo汇报精选展示页已上线（2026-11-03）：** 针对 `av_event_6000`、`av_feature_6000`、`sftqwen_av_event_6000` 各选4条代表样本（2条高指标标杆＋2条丰富Backchannel回应代表，共12条／108个媒体文件）。每条并排展示GT与三组模型的 `speaker audio + listener` 等增益对齐叠加视频（支持一键切换仅listener音视频）、Speaker音频与ASR转写、原生Qwen与SFT Qwen推理生成指导Prompt及完整Gemma正向Prompt、逐样本指标。[打开Demo汇报展示页（8876）](http://localhost:8876/listen_benchmark_review/p2_responsenet200_demo_20261103/) · [直接下载完整离线压缩包（74MB .zip）](http://localhost:8876/listen_benchmark_review/p2_responsenet200_demo_20261103/p2_responsenet200_demo_20261103.zip) · [汇总表格CSV](experiments/p2_responsenet200_demo_20261103/cases_summary.csv) · [数据清单](experiments/p2_responsenet200_demo_20261103/data.json) · [构建脚本](scripts/build_p2_responsenet200_demo.py)。

**ResponseNet200新增交叉对照（2026-11-03）：** 已加入SFT规划LoRA＋原始H Event6000，共5组／1000输出；新增组与SFT H Event组直接共享条件缓存。旧权重GPU重放的视频/音频latent逐值一致，13项CPU检查通过。SSH失败退避与评分查询容错已修复，原输出保留并恢复队列。[本轮协议与状态](experiments/p2_response200_feature_sft_event_20261103/README.md)。

**新两组四权重的ResponseNet200评测已接续（2026-11-03）：** 使用用户固定200条，原始H＋AV＋Feature、SFT新H＋AV＋0.05 Event各6000/8927步，共800输出；各组使用训练对应Qwen规划器，输入仅speaker音频＋listener首帧。等当前训练结束并释放32卡后自动冒烟与全量生成，沿用已优化的九项评测与共享GT缓存。[协议、入口和结果](experiments/p2_response200_feature_sft_event_20261103/README.md) · [队列状态](experiments/p2_response200_feature_sft_event_20261103/status.json)。

**点头＋官方门控内ASR发声线索融入Omni自然语言打标（v4统一版＋三级完备规则）已定版并归档（2026-11-03）：** 针对此前自然语言描述漏写发声的问题，在官方listener词／句／VAD并集门控保留的真实发声区间内，补充我们的ASR与官方转写／音频事件线索，交由Qwen3-Omni-30B在不输入旧caption的前提下一次性生成整段英文描述。采用三级完备ASR规范化规则（全量21,393种音频事件原文校验0非ASCII残留）：Tier 0纯英文原样保留；Tier 1有官方英文词时优先采用官方词消解谐音偏差；Tier 2封闭拟声白名单映射为精确英文拟声音节（`嗯/हम्म/うん→mhm`、`哈→ha`、`呵/嘿→heh`、`哼→hm`、`哇→wow`等）并要求显式带引号写入；Tier 3其余跨语种实义词幻觉（如`对/是/我/não`）置空文本仅保留发声时间窗供Omni直接听辨波形。v4 Prompt保持正文1–5段与`first_plus_evidence`逐字一致。在`172.29.189.119`完成24条有声样本（10条原50试点＋14条J大脸原漏发声样本）完整重跑，24/24正常结束、100%纯ASCII英文且全部找回具体词句与拟声音节指导。[三列有声对照HTML（8876）](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_nod_vocal24_20261103/) · [实验说明与v4规则](experiments/listen_native_caption_nod_vocal24_20261103/README.md) · [定版推理脚本](scripts/run_listen_native_caption_vocal_pilot.py) · [完整源码归档与校验（160文件）](releases/listen_j_nod_omni_caption_v1_20261103/README.md) · [外部Git备份库副本](../audio-driven-v5-data-processing/event_labeling/j_nod_omni_caption_20261103/README.md)。

**SpeakerVid 测试原片大脸 listener 池（2026-11-03）：** 口型能分开、清晰度保留、listener 脸短边 p10≥224 的片段 **1514 段／2.503 小时／37 条原片**。清单每行一条，含起止、脸短边、口型分数、listener 脸和切之前的整幅画面路径。正脸候选 7967 段／11.61 小时；其中能分开的 7368 段／11.23 小时／85 条原片已在总览页，本清单是其中再过脸尺寸的子集。已抽出测试子集 400 段／播放 0.974 小时／27 条原片：播放文件改用 Seamless 最终头肩裁剪框，listener 脸不少于 5 秒，同一原片起点至少隔开 25 秒，同一原片里优先留更锐的脸。[清单与媒体位置](experiments/speakervid_listener_face224_20261103/README.md) · [1514 条列表](experiments/speakervid_listener_face224_20261103/pool.jsonl) · [400 条测试子集](experiments/speakervid_listener_face224_20261103/selected_400.jsonl)。

**规划SFT新H接入AV＋Event（2026-11-02）：** 已补齐独立backend、来源校验和训练配置，冻结SFT新H＋AV＋0.05 Event；复用原500h文本、最终裁框和净化音频目标，Feature关闭。35项CPU回归、3条真实读取通过，285,638条全量preflight通过（约93秒）；已排队：现有AV＋Feature完成后自动使用四机32卡，先验收3步再续跑完整8927步，每500步保存。[接入与运行说明](related/P2_规划SFT新H接入AVEVENT训练交接_20261102.md) · [验收目录](experiments/p2_j_sft_h_av_event_ready_20261102/README.md)。

**四机32卡AV＋Feature已启动，原始Qwen H、Feature各0.025（2026-11-02）：** 先使用未经过规划SFT的原始Qwen2.5-Omni-3B guided H，对当前500h做AV＋Feature对照；视觉与音频Feature各0.025，等价于整体均值Feature权重0.05。已接入全量teacher缓存，保持原文、最终裁框和净化音频，关闭Event分支；新SFT H保留供后续比较。继承原AV时长分组及训练规格；前3步完整反传与检查点验收通过；按用户要求从第71步立即启用duration512、每500步保存，第162步出现停顿后从第70步恢复，改用three_quarters梯度重算，继续至8,927步。[当前恢复运行](experiments/p2_j_guided_av_feature_base025_20261102/duration_memorysafe_save500/README.md)。[本轮配置、读取、验证及命令](experiments/p2_j_guided_av_feature_base025_20261102/README.md)。

**规划 SFT 后的 500h hidden states 已全部完成（2026-11-02）：** 285,638／285,638条train，失败0，96个提取worker全部正常退出，32进程全量审计24.25秒通过；四机32卡已释放。最新2,697步Qwen2.5-Omni-3B规划LoRA的新guided H独立保存（约2.66TB），原标注、Gemma、AV latent及独立teacher继续复用。32条GPU验收与8条原生/禁用LoRA逐值对照通过；每卡2→3进程生产窗口约85→97条/秒。未启动训练。[读取与交接](related/P2_规划SFT后500h隐藏特征接续提取_20261102.md) · [结果入口](experiments/p2_j_train500h_sft_hidden_20261102/README.md) · [全量验收](experiments/p2_j_train500h_sft_hidden_20261102/COMPLETE.json)。

**最新规划LoRA同24条推理已完成（2026-11-02）：** 最新1,155h／2,697步adapter与未微调模型各24条全部正常EOS；同prompt基线重放24/24逐token一致。旧两轮4条虚构补充信息→新模型0条，发声未说明1→0；新模型5条安排发声，但18条仍含一次点头，未证明语义规划／AV提升。当前样本12条见于新train，原4条dev未参与梯度训练。按用户后续授权在本机GPU6/7叠加完成，缓存未停止。[可播放新旧对照](http://localhost:8876/listen_benchmark_review/planning_sft_filtered_compare24_20261102/) · [设置、逐条结论、权重与复现](experiments/p2_planning_sft_filtered_compare24_20261102/README.md)。

**当前500h的Feature teacher已扩为四机32卡提取（2026-11-02）：** 285,638条train，使用最终listener裁框及原官方并集净化音频，原Qwen2.5-Omni-3B视觉／音频塔冻结提取；原guided文本条件和AV缓存复用，不加载规划LoRA、不新增媒体文件。32条真实样本的输入与teacher逐值对照通过，采用双模型进程／卡、CPU并行准备、短seek和完成顺序预取；共享队列续跑及最终验收已接续。285,638条已全部提取并通过全量验收，未启动AV＋Feature训练。[数据、模型、配置、验证与读取](experiments/p2_j_teacher500h_node8_20261102/README.md) · [四机实时状态](experiments/p2_j_teacher500h_node8_20261102/production/cluster_status.json)。

**规划LoRA三机接续已完成（2026-11-02）：** 最终2,697步完成，完整结果见对应目录COMPLETE.json；三机GPU已转入上述teacher缓存提取。以下为接续时的配置与测量记录： node17／node12／node4，从完整checkpoint-500恢复，已核对推进到686步，node8释放。旧首帧PNG严格来源核验后复用＋独立CPU AUTO4解码＋prefetch4；解码超时只重启对应子进程并回退原读取器。首次尝试575步停滞的未落盘501–575已重算。global batch仍256，总2697步、每500步保存、不跑dev；50条输入完全一致、29项回归通过，24卡模型／优化器／RNG恢复核验通过。最近100步约1.45秒/步（原32卡窗口4.49秒，不同样本非严格AB）。[当前运行／续跑／代码快照](experiments/p2_planning_sft_filtered_3b_24gpu_inputfast_isolated_20261102/README.md) · [接续审计](experiments/p2_planning_sft_filtered_3b_24gpu_inputfast_isolated_20261102/switch_verified.json)。

**历史首次三机接续（已在575步因解码worker停滞停止）：** 按用户要求从完整checkpoint-500接续，使用node17／node12／node4，node8释放。复用核验PNG＋AUTO4解码＋预取4；全局batch仍256，每卡10/11条，全部全局样本分组与原32卡一致，总2,697步、每500步保存、不跑dev。adapter、504份optimizer状态、scheduler与24卡RNG恢复验证通过，28项回归通过；连续窗口约4.49→1.38秒/步（不同样本窗口）。[当前运行、配置与恢复](experiments/p2_planning_sft_filtered_3b_24gpu_inputfast_20261102/README.md) · [接续和吞吐证据](experiments/p2_planning_sft_filtered_3b_24gpu_inputfast_20261102/switch_verified.json)。

**新规划LoRA供数优化已通过对照，等待500步接续（2026-11-02）：** 复用可验证的旧首帧PNG，其余AUTO四线程等价解码，预取2→4；50条输入/PTS/音频/processor张量逐值一致，输入读取125.1→13.9秒。四机原训练继续，单次接续器在完整checkpoint-500后保留模型、优化器和32卡RNG切换；不改batch/LR/数据顺序，不从头重跑。[实现、验证、切换协议](related/P2_Qwen规划LoRA供数加速与无损接续_20261102.md) · [当前状态](experiments/p2_planning_sft_filtered_3b_32gpu_20261102/performance/migration_state.json)。

**新规划LoRA供数瓶颈已定位（2026-11-02）：** 旧方案使用已缓存首帧PNG；新清单虽沿用计算/组批加速，但按需解码原视频造成供数等待。8条profile中首帧占约96%耗时；缩短seek及AUTO多帧解码候选均通过8条像素/PTS对照，尚未部署或证明四机端到端收益，正式训练保持运行。[原策略核对、瓶颈与候选实测](experiments/p2_planning_sft_filtered_3b_32gpu_20261102/performance/README.md)。

**最新筛选数据Qwen规划LoRA四机训练已启动（2026-11-02）：** node17／node8／node12／node4各8卡，train 690,337条／1,155.44h，原始Qwen2.5-Omni-3B初始化新LoRA，global batch256、1轮2,697步；每500步保存，关闭全部dev评估。32卡首批真实反传通过，LoRA梯度有限且非零、冻结参数无梯度，沿用既有加速设置。[运行入口、日志及恢复](experiments/p2_planning_sft_filtered_3b_32gpu_20261102/README.md)。

**最新筛选数据已接入Qwen规划LoRA（2026-11-02）：** train 690,337条／1,155.44h，dev 5,238条／9.10h；SQLite按需读取speaker音频＋listener最终裁框首帧，目标描述仅作assistant监督。新3B LoRA默认从原始基座初始化；按最新要求关闭周期和结束时dev评估，每500步保存，结束另存最终权重。全清单／索引校验、12条真实媒体与标签检查、14项回归通过；GPU试训尚待空卡，未启动正式训练。[代码、配置、验证和运行交接](related/P2_Qwen规划LoRA新筛选数据接入_20261102.md)。

**ResponseNet四权重推理与评测全部完成（2026-11-02 14:25北京时间）：** 2,344条生成、128个评分分片和全量汇总完成，视频／音频执行错误均0；UTMOS因GT回应过短全部N/A。最终采用GPU检脸／关键点／情绪、batch64、像素线程4、四机每卡最多4 worker、评测机空卡最多8 worker及共享GT缓存。[结果](experiments/p2_testsets_4ckpt_24gpu_20261102/RESULTS.json) · [最终配置与验收](experiments/p2_metrics_acceleration_20261102/README.md#最终采用配置与完成状态)。

**ResponseNet评测加速已启用（2026-11-02）：** GPU检脸／关键点／表情、共享GT特征缓存、4线程像素指标及推理结束后释放GPU分片评分。两片段总耗时144.45→46.64秒，缓存复用42.82秒；图像指标／分布特征与情绪覆盖一致，ΔResp最大绝对差4.21e-5，额外128帧检脸对照和47项测试通过。已确认四机32卡接入评分；后续按用户要求把四机每卡worker上限从2增至4，全部排队分片已启动，batch128无稳定收益而保留64。旧CPU评分归档，四权重统一新后端重评；不代表完整评分已结束。[实测、代码与调度](experiments/p2_metrics_acceleration_20261102/README.md)。


**已切换为仅ResponseNet四权重推理（2026-11-02，最新范围）：** 按用户要求暂停seamless后续生成／评分，只跑ResponseNet完整586条×4权重＝2,344条。三机24卡已重新分片并启动，复用第一组已完成240条采样和24条导出；旧数据、结果和协议全部保留。ResponseNet九项评分链路检查通过，评测仍在172.29.189.119空卡独立接续。预计切换后约1.5–2小时完成全部生成与导出，评分汇总另计。[当前范围与队列](experiments/p2_testsets_4ckpt_24gpu_20261102/README.md) · [实时状态](experiments/p2_testsets_4ckpt_24gpu_20261102/status.json)。

**两项后续工作交接（2026-11-02）：** 新筛选train 690,337／dev 5,238的规划LoRA数据入口、旧SFT代码及Dataset接入背景（当前已完成，见上方新交接）；ResponseNet 586条音轨检查、ASR来源、128条候选筛选建议和四机队列状态。未启动新训练或切换测试子集。[交接文档](related/交接_Qwen规划LoRA数据代码与ResponseNet筛选_20261102.md)。


**ResponseNet-only已扩至四机32卡（2026-11-02）：** 按最新要求加入node8，node17／node12／node4／node8各8卡；四权重依次完成586条ResponseNet，seamless继续暂停。缓存和已完成结果保留，分片改为32份，指标仍在172.29.189.119空卡并行运行。当前目录名24gpu为历史名称。[当前配置及预计时长](experiments/p2_testsets_4ckpt_24gpu_20261102/README.md)。


**推理合批优化已实测，正式配置保持不变（2026-11-02）：** node17两卡对同4条长／短输入比较单样本、合并CFG、双样本及组合，固定权重／seed／30步。最佳组合短片段18.07→15.73秒（−13.0%）、长片段53.07→51.41秒（−3.1%），仅采样阶段；原生重复逐值一致，合批后最终latent有数值差异，尚未验证感知质量保持。候选未用于正式结果，借用的两张卡已恢复原队列。[完整实测、数值差异与原始结果](experiments/p2_inference_batch_benchmark_20261102_v3/README.md)。

**三机24卡逐权重推理已启动（2026-11-02，替代原32卡安排）：** 两组训练均完成8,927步。按用户指定只使用node17／node12／node4，按AV＋Event 6,000→8,927→仅AV 6,000→8,927顺序，每个权重先完成ResponseNet 586条和seamless 6,624条再换权重，全部512×512。24卡48条真实生成／导出验证通过，全部7,210条共享条件已完成，现已开始AV＋Event step6,000全量推理。九项指标含UTMOSv2移到172.29.189.119，等待空卡并与后续推理并行；当前该机8卡均忙，GPU评分验证尚待执行。旧队列PyAV环境混用已修复，旧记录保留。[新队列协议与输出](experiments/p2_testsets_4ckpt_24gpu_20261102/README.md) · [推理状态](experiments/p2_testsets_4ckpt_24gpu_20261102/status.json) · [评测状态](experiments/p2_testsets_4ckpt_24gpu_20261102/metrics_status.json)。

**四组权重的双测试集32卡推理已排队（2026-11-02）：** 等当前仅AV训练完成至8,927步后，使用AV＋Event／仅AV各6,000和8,927步，覆盖ResponseNet全部586条与seamless全部6,624条，共28,840个输出。规划器按用户选择使用未微调Qwen-Omni-3B，标准输入仅speaker音频＋listener首帧。两集均生成512×512；ResponseNet参考图和GT显式bicubic缩放，保留原帧率。先执行32卡完整链路验证，通过后自动全量推理并计算既有八项视频指标，FID/FVD合并全部特征重算。CPU检查通过，GPU验证及正式生成尚未开始。[协议、队列和输出说明](experiments/p2_testsets_4ckpt_32gpu_20261102/README.md) · [实时状态](experiments/p2_testsets_4ckpt_32gpu_20261102/status.json)。

**responsenet／seamless 完整评测接入（2026-11-02）：** [OmniResponse统一入口与运行说明](../OmniResponse/evaluation/listener_metrics/README.md)。复用已有RGB修正版GLARE四项；新增音频活动事件匹配与静音／短回应分流，接入UTMOSv2和SyncNet，分别报告检测失败、N/A及有效覆盖。已接测试清单不变，未启动正式模型生成或全量跑分。

**历史检查点清理已完成（2026-11-02）：** 按用户确认规则，仅清理本轮AV＋Event与仅AV的5个正式训练／迁移目录；删除91个`.pt`及91个完成标记，权重文件合计118.24GiB。保留每组0／2,000／4,000／6,000步、AV＋Event最终8,927步、4,201步迁移前后版本，以及仅AV当时最新7,600／7,800步；共13个文件。全部保留权重SHA256校验通过，训练在清理后继续推进至7,957步。历史记录中的其他检查点路径可能已删除，以此清单为准；后续仍按每200步及最终步保存。[保留清单与完成记录](experiments/p2_checkpoint_cleanup_20261102/complete.json) · [逐项删除记录](experiments/p2_checkpoint_cleanup_20261102/deletions.jsonl)。

**四机32卡双机守护已部署（2026-11-01）：** node8＋node17独立常驻，每分钟检测、每小时记录；训练进程缺失或步数停滞经确认后从最近完整权重恢复。北京时间11月2日02:38专项检查仅AV接续，至少守护至08:47，之后继续到两组完成。12项测试和真实双机补位验证通过；4小时及10小时事件仍待实际发生。[巡检与自动恢复说明](experiments/p2_j_training_watchdog_20261101/README.md) · [最新状态](experiments/p2_j_training_watchdog_20261101/state.json)。

**AV＋Event已恢复，后接仅AV自动队列（2026-11-01）：** 32卡验证通过后，按用户要求从4,201步恢复AV＋Event至原定8,927步，再自动启动从第0步初始化的仅AV对照、完整1轮8,927步。两组使用隔层重算、四机32卡、每200步保存与TensorBoard；检查点迁移的模型／优化器／32卡RNG逐位不变，4项队列测试通过。后组不会继承前组已训练权重。[自动队列说明](experiments/p2_j_av_event_then_av_20261101/README.md) · [实时状态](experiments/p2_j_av_event_then_av_20261101/state.json)。

**仅AV对照通过32卡验证，已选隔层重算（2026-11-01）：** AV＋Event在第4,200步保存后正常停止，退出检查点为4,201步。仅AV完成112次真实32卡测试更新，最长11.95秒样本通过；重算策略切换及拆段恢复的模型、优化器、32卡RNG均逐位一致。相同32批次步时4.03→3.64秒，降低9.6%，默认配置已更新。正式仅AV训练尚未启动，仍为1轮8,927步、每200步保存。[实测与验证](experiments/p2_j_av_only_benchmark_20261101/README.md) · [启动说明](experiments/p2_j_guided_av_only_s1_32gpu_20261101/README.md)。

**train/dev聊天前缀修复与捞回已完成（2026-11-02）：** 15,033条带前缀：此前已保留7,921条清理前缀；另7,112条复判后新增捞回4,091条（train 4,041／dev 50，约9.85h），拒绝3,021条、待复核0条。当前合并train **690,337条／1,155.44h**、dev **5,238条／9.10h**，此前通过及有声捞回数据全部保留。沿用原严格筛选及有声捞回规则，引用格式失败项追加原文句段选择复判；只删除前缀，不改写正文。原文、修改和历次判断均归档，test入口不变。[当前train](experiments/listen_j_train_dev_response_prefixfix_20261102/current/train.jsonl) · [当前dev](experiments/listen_j_train_dev_response_prefixfix_20261102/current/dev.jsonl) · [读取、拒绝原因及验收说明](experiments/listen_j_full_response_filter_20261101/README.md#traindev前缀修复与当前入口2026-11-02)。

**test聊天前缀修复与复判已完成（2026-11-02）：** 实查共144条带`Assistant:`前缀（原先143条此前因前缀拒绝，另1条此前因外观描述拒绝）。仅删除开头前缀，用原Qwen3-30B及相同严格规则复判，新增通过87条，另57条仍拒绝。当前通过**6,624条／11.04717h**，拒绝5,643条，待复核17条；共12,284条完整保留。读取器支持可追溯的前缀删除，原文、媒体／裁框引用和前后判断均保留；全量ID、哈希和索引核对、144条原始统计校验、3条真实媒体读取通过。此入口替代下方初筛通过清单。[当前通过清单](experiments/listen_j_test_response_prefixfix_20261102/current/test.jsonl) · [拒绝清单](experiments/listen_j_test_response_prefixfix_20261102/current/rejected/test.jsonl) · [逐条拒绝原因](experiments/listen_j_test_response_prefixfix_20261102/current/rejected/reasons.jsonl) · [完整读取与归档说明](experiments/listen_j_full_response_filter_20261101/README.md#test前缀修复与当前清单2026-11-02)。

**test原文直接可用筛查已完成（2026-11-02）：** 对原始test 12,284条／20.51h使用与train/dev初筛逐字相同的prompt、相同判断及后置规则、同一Qwen3-30B和推理配置。通过6,537条／10.85h（53.22%），剔除5,730条／9.64h，待复核17条；通过项中1,941条净化listener音轨非零。test独立导出，原文及引用行未改写。三路ID完整且无重复、全部清单哈希及6,537个索引位置核对通过，GPU7已释放。[通过清单](experiments/listen_j_test_response_filter_20261102/filtered/test.jsonl) · [8条通过样例](experiments/listen_j_test_response_filter_20261102/passed_examples.json) · [统计](experiments/listen_j_test_response_filter_20261102/filtered/summary.json) · [验收](experiments/listen_j_test_response_filter_20261102/FINAL_AUDIT.json)。

**有声回应捞回及合并已完成（2026-11-02 09:36北京时间）：** 已处理原先未保留且净化listener音轨非零的201,203条候选，新增捞回130,358条／230.77h片段（train 129,606／dev 752），另剔除63,747、待复核7,098。按用户要求放宽衣服、背景等细节，原文保留；Qwen先摘录发声证据再判断，并用独立规则拦截明确的外部证据说明。与旧保留集合并后train 686,296条／1,145.72h、dev 5,188条／8.97h。全量清单哈希、ID去重／划分、28条索引与原文回读检查通过；音轨非零＋文本判断不等于逐条媒体复核，test不参与。GPU7已释放，未启动训练。[读取及运行说明](experiments/listen_j_full_response_filter_20261101/README.md#有声回应捞回2026-11-02) · [合并数据入口](experiments/listen_j_voiced_recovery_20261102/combined/summary.json) · [验收记录](experiments/listen_j_voiced_recovery_20261102/FINAL_AUDIT.json) · [完整prompt](experiments/listen_j_voiced_recovery_20261102/system.txt)。

**全量train＋dev的Qwen-30B初筛已完成（2026-11-01 22:54北京时间）：** 全部1,095,039条已处理。train保留556,690／剔除527,523／待复核2,004；dev保留4,436／剔除4,363／待复核23，test不参与。保留合计561,126条、923.92h，原文未改写，ID／划分／原行哈希验收完成。后续有声捞回按上方新口径独立处理。[读取说明](experiments/listen_j_full_response_filter_20261101/README.md) · [完成标记](experiments/listen_j_full_response_filter_20261101/COMPLETE.json) · [完整统计](experiments/listen_j_full_response_filter_20261101/filtered/summary.json)。

**Qwen-Omni回应文本监督已扩展到全量（2026-11-01）：** 已包含当前500h试选，覆盖全部画质通过且有标注数据：train 1,086,217段／1,820.42h，dev 8,822段／15.32h，test 12,284段／20.51h。输入为speaker音频＋listener最终裁框首帧，目标为原回应caption；每条标记是否属于500h。全量ID／来源／划分及成员标记核验通过，16条媒体抽查通过；引用读取不复制全量媒体，未启动训练。[当前全量数据入口、读取代码与验收](experiments/listen_j_full_response_sft_20261101/README.md)。此前小脸清单保留为历史子集。

**问题标注改写Few-shot对照已完成（2026-11-01）：** 在上一轮剔除的同59条上，以相同规则对比无示例／8组完整示例，共118次生成。按严格保留原文细节的助手逐条文字检查，34条可改写样本通过数从9增至24；示例版仍有3条细节压缩待复核、7条未通过。23条证据不足样本均返回复核，另有1条原本可用样本误拒。这是针对已有难例的试验，不是独立泛化准确率；原标签与训练未变。[59条三列对照](http://localhost:8876/listen_benchmark_review/planning_rewrite_examples59_20261101/) · [prompt、检查口径及具体案例](related/P2_规划标签改写FewShot59条对照_20261101.md)。

**规划标签100条：改写转为判断筛选试验（2026-11-01）：** GPU7实测自由改写存在确定化和漏声音问题，原文删除也有误删姿态；按用户新方案改做原样保留／剔除。模型保留50条，助手按规则逐条检查发现9条漏拒、2条误拒；补充背景／头发外观规则后保留41条候选。此为分层难例且规则在本批调整，不能据此估算全量比例或泛化准确率；原caption、AV缓存及在训清单均未改动。建议仅用于规划LoRA的SFT筛选。[100条原文与判断依据](http://localhost:8876/listen_benchmark_review/planning_caption_filter100_20261101/) · [设置、失败案例、保留分布及限制](related/P2_规划标签清洗与判断筛选100条试验_20261101.md)。

**小脸数据已整理为Qwen-Omni回应文本监督清单（2026-11-01）：** 从最终画质通过、原始人脸p10<224px的候选中排除500h补充部分；train 800,579段／1,320.42h，dev 5,806段／9.77h，test 7,341段／12.09h，保持独立。输入为配对speaker音频＋最终修正版裁框listener首帧，目标为原始listener回应caption。提供引用清单、随机索引和按需读取器，不新增批量WAV/图片；16条真实媒体及5项错配检查通过，未启动训练。[数据位置、读取示例与验收](experiments/listen_j_smallface_response_sft_20261101/README.md)。

**回应规划LoRA两轮同24条对照已完成（2026-11-01）：** 172.29.189.119 GPU7上并行比较未微调、1轮和2轮，统一当前SFT prompt，72条全部正常结束。训练后描述更完整，但第二轮没有一致优于第一轮；两轮各4条虚构“补充信息”依据。20条train见过、4条dev未参与梯度训练；另24条静音对照中22条输出变化，表明音频影响生成，但不代表已证明语义规划能力或AV提升。结果逐条保留，GPU7已释放。[可播放对照页](http://localhost:8876/listen_benchmark_review/planning_sft_compare24_20261101/) · [实验设置、案例与结论](related/P2_Qwen规划LoRA两轮24条对照_20261101.md)。

**已排队：LoRA第2轮结束后恢复AV＋Event，每200步保存（2026-11-01）：** 后台队列等待LoRA累计2,232步完成及四机显存释放，再从S1第2,301步接续至原预算8,927步。已修正原先写死的100步保存逻辑；下一个周期检查点为2,400。迁移后的模型、优化器和32卡RNG逐位不变，46项测试通过，原时长分组与TensorBoard保留。当前队列状态以实时文件为准。[队列状态](experiments/p2_j_av_event_resume_after_lora2_20261101/state.json) · [AV接续目录](experiments/p2_j_guided_av_event_s1_32gpu_resume200_20261101/) · [接续说明](related/P2_Qwen回应规划LoRA_SFT代码准备_20261101.md#第2轮结束后自动恢复avevent改为200步保存)。

**回应规划LoRA第2轮已在四机32卡续训（2026-11-01）：** 第一轮1,116步已完成；按用户要求从`checkpoint-1116`保留权重、AdamW动量和32卡随机状态继续完整1轮，累计目标2,232步。已确认首个新增步为1,117，未重跑第一轮。延长cosine预算后续训起始LR约5.12e-5，其他数据、batch及提速设置不变；第一轮结果独立保留。[第2轮运行目录](experiments/p2_planning_sft_3b_32gpu_epoch2_20261101/) · [续训记录](related/P2_Qwen回应规划LoRA_SFT代码准备_20261101.md#用户追加第2轮2026-11-01)。

**S1已暂停，四机32卡回应规划LoRA已启动（2026-11-01）：** 按用户要求验证第2,300步S1权重后正常停止；退出时额外保存第2,301步，两份均保留。Omni-3B采用teacher forcing回答token交叉熵，全部285,638条train、global batch256、1轮约1,116步。已开启BF16／SDPA／fused AdamW／时长分组／并行预取／短音频padding，并根据实测关闭梯度重算。32卡梯度检查通过，第71—100步平均0.692秒／步、约370条／秒（不含保存评估）。[训练与切换记录](related/P2_Qwen回应规划LoRA_SFT代码准备_20261101.md#s1切换与32卡正式运行2026-11-01-1052-utc) · [正式运行目录](experiments/p2_planning_sft_3b_32gpu_20261101/)。

**回应规划LoRA SFT代码已准备（2026-11-01）：** 原始Qwen-Omni-3B，音频＋最终裁框首帧→现有listener描述；仅回答token交叉熵，语言层LoRA rank32，约5,987万可训练参数。默认四机32卡、global batch256、LR=1e-4、1轮，每100步保存及记录dev loss，支持TensorBoard与断点恢复。原24条输入／标签、真实反传、恢复续训及adapter重载生成通过；四机32进程CPU通信通过，正式GPU训练尚未启动。BF16恢复对照有数值差异，不宣称逐位一致。[代码、配置、启动与验证](related/P2_Qwen回应规划LoRA_SFT代码准备_20261101.md)。

**VITA-1.5方案与未微调3B自主规划试验已完成（2026-11-01）：** 已记录多阶段文本SFT与回应描述LoRA候选路线。原24条仅输入speaker音频＋最终裁框参考首帧＋system/user指令，24／24正常生成，单批生成7.91秒；未输入目标描述或listener音轨。初步检查发现全部安排轻微点头、声音状态经常不明确，#23复述speaker长句；也有语义适配的短回应。保留冻结3B基线，建议先做prompt对照再判断是否需要规划LoRA；尚未测AV效果。使用指定推理机GPU7，未改动当前S1。[方案与实测记录](related/P2_VITA15多阶段训练与自主回应规划试验_20261101.md) · [24条音频与原始规划](http://localhost:8876/listen_benchmark_review/omni3b_planning_20261101/)。

**首轮 S1 已采用时长对齐加速（2026-11-01）：** 双通路文本指导＋AV＋Event，四机32卡、完整数据1轮、global batch=32、8,927步。32卡ABBA实测随机组批6.698秒/步、512窗口时长对齐3.961秒/步，步时降低40.9%；中途恢复的模型、优化器、各卡随机状态逐位一致。保留原训练前100步，从该检查点接续；测试更新不计入正式预算。[训练与加速记录](related/P2_J整段backchannel辅助监督_20261101.md#32-卡时长对齐实测与接续) · [接续状态](experiments/p2_j_guided_av_event_s1_32gpu_20261101/performance/duration512/state.json) · [正式训练目录](experiments/p2_j_guided_av_event_s1_32gpu_duration512_20261101/)。

**J Connector 已改为 1024 维、4 层（2026-11-01）：** 直接／guided 共用多层 Qwen hidden＋外部时空查询，16 个注意力头，Cross-Attention→Self-Attention→FFN，接原音视频投影及整段多标签分支。八组 S1／S2 配置已更新，缓存无需重提；46 项测试及真实缓存双条件反传通过，正式训练未启动。[结构与验证](related/P2_J整段backchannel辅助监督_20261101.md#connector1024-维4-层)。

**整段监督已切换为19类（2026-11-01）：** 八组训练配置统一采用正例数≥5,000的19类（视觉16／音频3），覆盖83.63%的正类记录；完整2,726类标签数据库保留，全部285,638条数据继续训练，无19类正例的17,304条仍参与AV。55项测试、真实缓存双条件读取及1024维4层Connector反传通过；缓存无需重提，正式训练未启动。[实现与验证](related/P2_J整段backchannel辅助监督_20261101.md#19-类训练实现) · [类别与数据保留理由](related/P2_J整段backchannel辅助监督_20261101.md#类别收敛与数据保留讨论)。

**J 整段 backchannel 辅助监督已实现（2026-11-01）：** 对条件 U 做整段 mask 池化、多标签 BCE，使用现有开放行为名称，不要求事件时间。285,638 条 train 标签已派生（视觉 2,625／音频 101 类）；45 项测试、真实24条缓存双条件读取及生产规格 Connector 反传通过。缓存继续复用，正式训练未启动。[实现、标签策略、配置与验证](related/P2_J整段backchannel辅助监督_20261101.md)。

**J 新训练输入缓存已完成并验收（2026-11-01）：** 285,638 条／500.00028h train 清单已冻结。原 24 条复核样本的输入和原生训练接口通过，Qwen 两种条件及 AV latent 加速前后逐值一致；四机 32 卡大批量验证通过。生产使用 Gemma batch=32、Qwen＋VAE 每卡 2 个常驻进程、并行 ROI 解码与异步写盘。V5 已处理一条真实缺帧样本并完成其32条批次；四种缓存各285,638条全部验收通过，COMPLETE.json已生成。[配置、数值证据与读取入口](related/P2_J新数据缓存与四机32卡提取_20261101.md) · [运行目录](experiments/p2_j_train500h_cache_20261101/)。

**当前数据筛选与500h扩充方案已记录（2026-11-01）：** 保留train大脸主体264,054段／462.63h，补入21,584段／37.37h稍小脸有声候选，试选合计285,638段／500.00028h；补充实际最低人脸p10约186px，净化后非零音频片段占34.86%。dev/test独立，原文及音频不改；已认可方向并保存候选试选，正式缓存 manifest 已冻结，新缓存正在四机提取，状态见上方入口。[筛选口径、统计、试选规则和全部入口](related/P2_J数据筛选与500h有声补充方案记录_20261101.md)。

**train＋dev人脸p10≥224素材复核（2026-11-01）：** 从267,070段／468.18h固定随机抽20条train＋4条dev，统一最终修正版裁框，含listener音视频、speaker音频、参考首帧、原文与统计。11条历史标注框与最终框不同已标明；该复核页展示源素材；随后同一24条训练输入与缓存验证已通过，见上方新缓存入口。[复核说明及清单](experiments/listen_j_face224_training_review_20261101/README.md) · [打开HTML](http://localhost:8876/listen_benchmark_review/j_face224_training_20261101/)。

**情绪字段分析与倾听片段复核（2026-11-01）：** 已记录最终V00覆盖量、人脸尺寸限制、5段视频复核及类别映射未确认的问题。当前建议保留情绪字段作辅助，暂不用于硬筛选或生成prompt；尚未修改训练配置。[完整分析记录](related/P2_Seamless情绪字段覆盖与倾听片段复核记录_20261101.md)。

**最终 V00 情绪字段覆盖（2026-11-01）：** 494,277段／774.76小时，来自40,239条原始 listener 参与者片段（原42,948条的93.69%）；已按train/dev/test统计，情绪逐帧有效性未筛查。[统计、明细与NPZ抽查](experiments/listen_j_final_quality_selection_20261101/emotion_v00/README.md)。

**Test数据概览（2026-11-01）：** 最终保留12,284段／20.51小时；开放候选名称689个（视觉660、音频29）。人脸p10≥224的有4,943段／8.42小时，叠加至少3类为2,616段／4.79小时。[类别、人脸分布和完整表格](experiments/listen_j_final_quality_selection_20261101/overview_test/README.md)。

**train＋dev合并统计（2026-11-01）：** 最终画质通过数据共1,095,039段／1,835.74小时；开放候选名称4,751个（视觉4,560、音频191）。已重算类别时长、人脸分布和容量，test不计入，原split不变。[完整概览与表格](experiments/listen_j_final_quality_selection_20261101/overview_train_dev/README.md)。

**最终数据统计概览（2026-11-01）：** 1,107,323段／1,856.25小时；4,764个开放候选名称（视觉4,572、音频192），非互斥语义大类。已统计每类覆盖时长、原生人脸p10/p50时长分布、片长、丰富度、视听支持及训练池容量，无图表。[完整概览](experiments/listen_j_final_quality_selection_20261101/overview/README.md) · [全部类别](experiments/listen_j_final_quality_selection_20261101/overview/all_categories.csv)。

**最终画质复筛已完成（2026-11-01）：** 全量回查早期已打标数据的后补画质结果，最终保留1,107,323段／1,856.25小时（train 1,086,217段／1,820.42小时）。从已有caption中另排除6,237段、移入待核6,105段；原标签与原媒体保留，未因无回应或统计局部待核而删数据。[最终清单、分流、统计与读取](experiments/listen_j_final_quality_selection_20261101/README.md) · [独立验收](experiments/listen_j_final_quality_selection_20261101/audit.json)。

**用户授权的十小时接续监控已完成（北京时间2026-11-01 01:25）：** 全量打标、文本整理、CPU统计及独立验收完成；32卡模型进程正常结束，显存释放，监控按约定自动停止。全量覆盖1,151,805段，1,119,665段有可用统计，其余有明确分流记录。[监控、修复与验收记录](experiments/listen_richness_full_j_20261031/goal_monitor_20261031/README.md)。

**全量开放行为统计已完成（2026-10-31 UTC）：** Qwen3-30B-A3B＋已确认v3，512并发／65536调度token；全量1,119,665段、2,187个chunk，0任务失败、0截断、0无法解析，34,729段局部待核仍保留。原文、来源、裁框、官方音频与点头证据保持可追溯；4,877个开放候选名称尚含同义细分，不等于4,877种互斥行为。未执行训练子集筛选。[结果及读取入口](experiments/listen_richness_full_j_20261031/README.md) · [完整交接](related/P2_全量开放行为统计32卡接续_20261031.md) · [全量独立验收](experiments/listen_richness_full_j_20261031/goal_monitor_20261031/final_audit.json)。

**Omni尾部跨卡协作已启用（2026-10-31）：** 四机32卡全部恢复产出；队列充足时仍每卡最多预领4来源，待领取来源少于64时降至1，空闲卡每次最多分担8个未开始片段。片段独占锁保护在途推理与保存，原来源拥有者收齐结果后提交一次回执。63项测试、2项子测试、32路512条调度模拟、四机锁检查和32条真实输入来源核验通过；尚未进入尾部，不宣称已测得尾部提速幅度。[策略、参数、代码、验收与回退](related/P2_Omni尾部跨卡协作调度_20261031.md)。

**四机32卡整体提速已验收（2026-10-31 10:22 UTC）：** 32卡实际产出从前置窗口10.82段/秒提升到两轮17.99～20.38段/秒，平均19.19段/秒（观察增幅77.3%）；GPU均值26.8%→46.2%。新增保留帧属性的YUV裁切、单线程转色复用和10-bit原平面共享，模型与输入规格不变。6条真实输入重放、每卡一条共32条来源核验通过，两轮无重启和新增失败。不同片段窗口存在波动，未宣称满载或全局最优。[配置、代码、完整四机测速和备份](related/P2_四机32卡整体吞吐优化_20261031.md)。

**历史第一轮 Omni 输入优化（2026-10-31，后续整体提速见上方）：** 像素一致裁剪、修正点头／Omni 按条件复用解码、最多 4 来源连续预取，并修正预解码占用点头槽位的问题；32 卡均恢复正式产出。80 项测试、6＋2 条真实输入重放通过。初版固定 8 条 ABBA 对照约 2.07 倍；最终生产短窗 8.10 段/秒、GPU 20.7%，尚未证明整体提速，不能承诺全量翻倍或卡已满载。最终当次 16 条正式结果来源抽查通过，任务队列失败为 0。[实现、参数、生产测速与验收](related/P2_Omni输入裁剪解码复用与跨来源预取_20261031.md)。

**历史修正版裁框流水线性能复查（2026-10-31）：** 完成 16→24 并发试验并回退；像素一致裁剪组件小样本准备耗时降低约 19.6%。当时尚未接入生产，后续接入见上方。[测量、配置与优化顺序](related/P2_修正版裁框流水线性能复查_20261031.md)。

**J 打标裁框修正接续（2026-10-31）：** 后续输入采用确认的修正版；已完成旧框标签保留并注明实际来源，变框片段的点头与 Omni 使用一致坐标。[规则、代码、结果字段与部署验收](related/P2_J修正版裁框接续与旧标签保留_20261031.md)。

# audio-driven-v5 总索引

**当前类型与次数统计方案已整理记录（2026-10-31）：** 采用v3文本提取结果＋CPU逐段／分类统计，保留未知、待核和全部原文；不是LLM整体评分。已冻结代码、prompt与参数，并明确通用入口默认v4。全量流式分片与接续现已实现，见上方32卡条目；此处保留初始冻结方案。[当前方案、复现命令和全量前改造清单](related/P2_开放行为类型次数统计与筛选当前方案交接_20261031.md) · [本地配置与源码快照](releases/listen_richness_statistics_record_20261031/README.md)。此为历史冻结记录，全量状态见页首。

**50段开放行为类型与数量统计（2026-10-31）：** 复用v3提取，CPU整理出逐片段与按类别两张表，全部50段保留（含原3个校验失败案例）。区分明示次数、分次推得、未知和持续状态组；26种采用候选类型，局部待核项另存。未使用整体评分替代数量，未执行全量筛选或改写训练数据。[结果、规则与复现](experiments/listen_richness_statistics50_20261031/README.md) · [直接查看两张表](experiments/listen_richness_statistics50_20261031/tables_v2/tables.md) · [前10／后10排序效果与例外](experiments/listen_richness_statistics50_20261031/ranking_check/README.md)。

**50段回应丰富度提取prompt试验（2026-10-31）：** 172.29.189.119仅GPU7，用Qwen3-30B-A3B比较同一50段的4版prompt／协议。v4结构及引用校验50/50通过，仍有漏动作、明确次数丢失和同义分散，未通过语义验收，未推广全量。原句ID取证与取消关系图有效；保留全部原文和失败记录。[四版结果、资源与具体问题](experiments/listen_richness_gpu7_pilot_20261031/README.md) · [50段原文对照](experiments/listen_richness_gpu7_pilot_20261031/review50.md)。首批50段生成10.62秒，GPU均值97.8%，约129GiB显存；任务结束已释放GPU7。

**整段描述的回应丰富度统计与筛选代码已准备（2026-10-31）：** 原始caption保持不变，复用点头／官方音频时间与原图人脸测量，由文本模型提取开放行为及原文依据，再做主体集＋稀有行为补充。[规则、prompt、命令与使用边界](related/P2_自然语言打标丰富度统计与子集筛选交接_20261031.md) · [入口](scripts/listen_richness.py) · [初版配置](configs/listen_richness_selection_v1.json) · [29项CPU回归及12条真实输入检查](experiments/listen_richness_codecheck_20261031/README.md)。此为初版代码记录；后续已完成上方50段试验，尚未推广全量或替换训练数据。

**J 输入与历史分片接续已复核修复（2026-10-31）：** 四机8条真实媒体重放的帧、音频与prompt指纹一致；完整清单仍为60,869来源／1,151,805段。补接历史第三目录的1,295来源／62,488段，按真实所有者读取6,647个委托来源的点头结果，复用已完成结果，未重启32个GPU进程。旧两机范围与旧点头剩余量估算不能继续使用。[复核、修复、范围与证据](related/P2_J打标输入链路与历史分片接续复核_20261031.md)。

**点头检测跨机 CPU 分担（2026-10-31）：** 按用户授权，将其他机器可用 CPU 用于原 node17 点头队列；保留画质筛查与 Omni 输入准备资源。沿用原任务锁、结果目录、25 fps 与 recall 检测参数，仅按执行主机分配解码资源；已部署并验收：点头队列有效进程 168，专用画质进程 160；末两窗口点头整源提交约 6.6–7.4 段/秒，Omni 32卡持续处理。[实施、验收、配置与回退](related/P2_点头检测跨机CPU分担_20261031.md)。

**画质前置筛查＋四机32卡（2026-10-31）：** Omni 32卡和128个点头worker已接入画质关口。32卡采用已验证的自适应请求提交（同样本端到端约1.19～1.24倍，输出一致）；早期专用画质CPU为256进程；当前按上方跨机点头方案调整为160进程，保留后台生产与故障恢复。六小时巡检窗口已结束，用户已暂停 goal；后台生产与故障恢复继续。[流程、规则、资源、输出与巡检交接](related/P2_画质前置筛查与四机32卡运行_20261031.md)。

**历史三机24卡提速与10小时巡检（已被上述新授权替换）：** 保持新J输入和已确认prompt，先做CPU供数及同样本端到端对照；每半小时留存资源报告、恢复本流水线故障，授权窗口至2026-10-31 00:46:02 UTC。[执行规则、实测及巡检位置](related/P2_Omni24卡提速与10小时巡检_20261030.md)。

**J新切分已接续 Omni 自然语言打标（2026-10-30）：** 用户确认采用“第一版描述方式＋精简点头／静音说明”，并直接用官方词／句／VAD并集净化，不补ASR。12项CPU测试、16条真实J片段与跨机器锁检查通过后，node17＋node8由旧VLM转为Omni，按追加授权加入node4（172.29.225.51）全部8卡，共三机24卡；原160个CPU点头worker保留，边产点头边打标。新范围1,151,805段，旧结果保留，不新增WAV。[统一交接](related/P2_J切分_点头接续Omni自然语言打标交接_20261030.md) · [运行、输出与续跑](experiments/listen_j_omni_20261030/README.md) · [最新状态](experiments/listen_j_omni_20261030/status.json) · [完整源码归档与校验（160文件，含2026-11-03发声ASR补充v4定版）](releases/listen_j_nod_omni_caption_v1_20261103/README.md) · [外部备份库副本](../audio-driven-v5-data-processing/event_labeling/j_nod_omni_caption_20261103/README.md)。

**第一版描述方式＋精简证据说明已完成50条试验（2026-10-30）：** 保留第一版段落要求，只增加简短点头字段与真实静音说明，node4八卡50条首次正常完成。S0699恢复两次Okay、S0805写出前后两段点头、S1337不再虚构说话；仍有漏动作、推演和把空候选当不存在的问题。当前为对照试验，未替换v1或全量标签。[三版有声对照](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_first_evidence/) · [prompt、代码、结果与限制](experiments/listen_native_caption_first_evidence50_20261030/README.md)。

**Omni 整段自然语言最终 prompt v1 已固定（2026-10-30）：** 原生视频8 fps＋净化 listener 音轨＋四字段点头候选＋波形核验的全静音标记，不输入旧段落。node4 八卡同批50条全部正常完成，无重试；四条已知静音虚构说话案例未重现，但 S1337 仍有静音说话描述，且有漏点头、推演和时间数字残留。运行验证通过不等于语义验收。[最新有声复核](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_nod_final/) · [冻结 prompt、实际结果、限制及源码](experiments/listen_native_caption_nod_final50_20261030/README.md)。

**Omni＋点头线索独立重新打标（2026-10-30）：** 50条全部重新推理，实际输入无旧段落，点头条目仅`id/start/end/observation`，无候选的12条也重新生成。49条正常结束、1条重复截断，并发现全静音片段虚构说话等问题；完整原始输出保留，尚未通过质量验收。[新有声复核页](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_nod_direct/) · [prompt、代码、实际结果和验证](experiments/listen_native_caption_nod_direct50_20261030/README.md)。

**高召回点头融入 Omni 自然语言描述试验（2026-10-30）：** 同一批50条中38条检出50个点头候选区间，八卡修订后29条明确写入 nod、4条改写为头部运动、5条原文未变；其余12条无候选，原文直接保留。结果仍需复核动作、时序和推演残留。[补充前后有声复核](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption_nods/) · [方法、prompt、代码、结果与验收](experiments/listen_native_caption_nod50_20261030/README.md)。

**Omni 净化音轨＋原生视听自然语言描述试验已完成（2026-10-30）：** 沿用 `native_simple` 的50条，node4八卡生成整段英文描述，50条正常结束；不输出逐事件时间戳表，不做音轨净化消融对照。输入帧与历史结果逐条一致，播放音轨与实际净化波形50条哈希一致；仍存在漏词和推演措辞，待人工复核。[有声HTML](http://localhost:8876/listen_benchmark_review/native_simple/natural_caption/) · [范围、prompt、代码、结果与验收](experiments/listen_native_caption_clean50_20261030/README.md)。

**本轮30B整合结果已提供有声复核页：** [65段主试验＋8段历史回归，原文与整合结果并排查看](http://localhost:8876/listen_benchmark_review/semantic_qwen30b_20261030/)。包含自动通过结果、未通过提案及具体失败原因，视频置左；可下载带声视频和对照JSON。沿用Cursor转发端口8876。本页展示已完成的v4结果，尚未实施后续讨论的局部整合／失败回退方案。

**Qwen3-30B 原文整合四卡优化已完成试验（2026-10-30）：** 本机0–3卡比较原方案及原文编号／固定归属表，新增可选解码CUDA Graph；单次对照耗时约降低44%～67%。最终单归属表在65段中29段通过现有程序与模型自检，8个历史回归通过6段；59项CPU测试及已发布项重放通过。仍有跨独立区间合并、推演残留，尚未推广为全量最终标签，原始数据及训练缓存保留。[代码、规则、配置与复用交接](related/P2_Qwen30B_原文整合优化与四卡验证_20261030.md) · [完整试验结果、失败明细与本地源码快照](experiments/listen_qwen30b_assignments_20261030/README.md)。

**J全量打标已扩展至node17＋node8两机16卡（2026-10-30）：** node8实际地址172.29.104.233，已按用户授权停止旧GPU任务及自动恢复进程。两机沿用已验证加速方案，分别承担649,665段（含此前已完成）与502,140段；来源互斥、合计1,151,805段，node17原工作进程保持运行。[部署、分片、结果位置与恢复](experiments/listen_j_annotation_two_nodes_20261030/README.md) · [最近一次两机检查](experiments/listen_j_annotation_two_nodes_20261030/status.json)。两机各有独立队列与守护；未恢复goal自动续行或定时消息。

**当前打标加速配置已固定（2026-10-30）：** node17八卡采用batch512／调度token32768、CPU进程解码、连续补入与输入预处理重叠；停止探索新优化，后台全量打标与故障恢复继续。[加速策略、有效配置、实测及多机复用交接](related/P2_打标加速固定配置与多机复用交接_20261030.md) · [固定参数快照](experiments/listen_j_annotation_node17_20261029/frozen_config_20261030/recommended_reference.json)。多机复用需独立工作目录、来源分片及启动器适配，当前不是已验收的多机部署。下方早期batch256测速记录为历史阶段。

**J 数据 node17 八卡打标与八小时监督已启动（2026-10-29）：** 64,428 条成功来源中的 1,151,805 个合格终端片段进入队列，上游 194 条失败另存。采用原生视频 8 fps＋GLARE recall 点头补充；八卡重复测速后选择 batch512／32768 token，通过逐卡更新生效。观察式 C prompt、裁框、采样和生成长度不变。[范围、代码、测速、结果与恢复](experiments/listen_j_annotation_node17_20261029/README.md) · [实时监督状态](experiments/listen_j_annotation_node17_20261029/supervisor.json) · [守护状态](experiments/listen_j_annotation_node17_20261029/watchdog.json)。监督至少持续到 10 月 29 日 23:57:32 UTC／北京时间 10 月 30 日 07:57:32；全量打标尚未完成。

**Seamless 全量 J v1 四机 CPU 切分已启动（2026-10-29）：** 用户将任务扩大为八小时内完成原始全量，冻结 **64,622 条来源**，包括旧目录 64,087 条及保持身份隔离标记的 535 条。四机分批升至每机约 24 个 worker、上限 32 个，每 worker 主要计算 2 线程，音质 CPU、缓存优先、不生成审核视频；按实时吞吐调整，持续训练变慢则降载。目标截止 **10 月 29 日 22:19 UTC／北京时间 10 月 30 日 06:19**。[运行、结果与恢复](data_corpora/seamless_listen_j_full_20261029/README.md) · [实时完成数／并发／ETA](data_corpora/seamless_listen_j_full_20261029/status.json) · [持续集群入口](scripts/run_listen_pair_j_full_cluster.py) · [配置](configs/listen_pair_j_full_cpu_cluster.json)。原 J 规则未改；17 项测试、4 条原样本等价与真实断点恢复、2 条无旧缓存原始来源验证通过。全量尚在处理，旧结果与训练输入保持原样。

**当前 pair 切分代码入口：固定 J v1（2026-10-29，整理完成）。** [统一交接：代码分工、参数、运行与输出](releases/listen_pair_segmentation_j_v1_20261029/README.md) · [实际 J 规则模块](scripts/listen_local_recovery_semantic_trial.py) · [已验证的运行入口（参数 `j`）](experiments/pair_semantic_random5_20261029/run.py) · [源码指纹与归档校验](releases/listen_pair_segmentation_j_v1_20261029/verify.py) · [五条最新 HTML 与全部差异](experiments/pair_semantic_random5_20261029/README.md)。当前仍为 2～11.95 秒、超过 5 秒停顿分段，先切完整候选再筛音质／局部恢复；≥3 秒或 ≥4 秒尚未启用。旧全量脚本仍使用旧规则，不能直接作为 J 入口；本次仅整理交接，未重切或启动全量。

**原生 8 fps＋点头的 CPU 轻量整理已供查看（2026-10-29）：** 15 段 365 条有效观测→359 条，仅折叠 6 条完全重复，16 个点头候选保留；原文、窗口和时间来源完整保留，尚未做 LLM 语义去重。按用户要求停止本机 30B 叠加试跑，无已完成的 LLM 整理结果。[整理前后 HTML](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/semantic_light/) · [代码、结果与验证](experiments/listen_native8_nod_semantic15_20261029/README.md)。

**J 切分规则已固定为 v1（2026-10-29）：** 在用户认可此前七条后，再随机抽取五条未检查来源（seed=20261031）。H→J 初始候选 150→150，140 段起止不变；10 处均为保护完整句末、收回下一句残尾。旧全量已保留内容在 J 完整候选中只减少 I／I／He 三个孤立残尾词。最终自动通过仍须区分：旧全量 131→J 113，92 段不变；37 段未自动通过，其中 18 段保留整段供复核。未启动全量或自动放行音质疑点。[五条 HTML、全部起止差异与验证](experiments/pair_semantic_random5_20261029/README.md) · [固定版本与源码指纹](releases/listen_pair_segmentation_j_v1_20261029/README.md)。

**当前确认的视觉打标方案：原生视频 8 fps＋高召回点头补充（2026-10-29）。** Qwen3-VL-8B＋观察式 C prompt，listener 512×512 预裁框，整段及 3 秒／1 秒步长窗口；点头用同片段 25 fps 姿态的 GLARE recall。专用摇头停用，其他四类专用检测仅留历史实验，VLM 所有原文仍保留。[完整交接：模型、prompt、代码、参数、运行与读取](related/P2_原生视频8fps与点头补充打标交接.md) · [固定配置](configs/listen_visual_native8_nod_v1.json) · [统一入口](scripts/listen_visual_native8_nod.py) · [15 段已有结果样例及验收](experiments/listen_visual_native8_nod_20261029/README.md)。已加入多窗口批量提交、后台预取和跨卡工作量均衡（[加速参数与验收](related/P2_原生视频8fps与点头补充打标交接.md#8-2026-10-29-打标调度加速)）；已在 node17 完成 8 组各 50 条测速，当前最快为跨片段 batch256＋16384 token（[实测、推荐参数及边界](related/P2_原生视频8fps打标加速实测_20261029.md)）；未启动全量或训练。

**GLARE 摇头混淆修正试验（2026-10-29）：** 用户反馈点头被标成摇头后，对旧 20 个摇头候选追加原始角度主方向与真实反向运动检查，暂保留 2 个待复核候选；16 个点头及其余表情完全不变。旧候选与筛除原因均可查看，不以点头重叠直接删除。[更新后的原页面](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/) · [规则、结果与新版输出](experiments/listen_glare_reactions15_20261029/headshake_direction_v1/README.md)。这是保守试验，尚未证明两个保留项是真实摇头；未应用生产数据。

**GLARE 六类视觉补充已完成 15 段（2026-10-29）：** 点头沿用用户认可的高召回候选，新增摇头、微笑、视觉大笑、皱眉、惊讶样表情，共 63 个独立候选区间，保留多类同时发生与全部 VLM 原文。[原页面查看六类对照](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/) · [代码、输出、参数与验收](experiments/listen_glare_reactions15_20261029/README.md)。FER 在 S0694／S0343 人脸检出覆盖不足，页面已提示；新类别仍待人工复核，未写入生产标签或训练缓存。

**GLARE 点头小样本已运行（2026-10-29）：** 按要求先处理此前人工复核的 15 段，使用现有 512×512／25 fps 缓存、CPU 两路并发。默认参数 0 个候选；平衡参数 8 段／8 个候选区间；高召回参数 13 段／16 个候选区间。候选尚非确认点头，不覆盖原 VLM 或训练标签。[打开三组参数与原文对照](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/glare_nodding/) · [方法、结果与复现](experiments/listen_glare_nodding50_20261029/README.md)。

**J 规则原样扩展到原抽样剩余五条（2026-10-29），已供人工检查。** 未新增抽样、未改变用户看过的 J 代码；五条初始切分合计 177→175 段，157 段起止不变、17 组变化。原 HTML 保留旧全量／H／J 三列，逐段起止、原文及音质状态齐全，共 61 组对照；原先两条 J 页面保持可用。重点：398 秒来源的完整短句 I like that. 未再成段，442 秒来源 I mean...／Yeah, 也需试听，不能一概视作残尾。[五条入口、完整差异与统计](experiments/pair_semantic_remaining5_20261029/README.md) · [398 秒重点检查](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S2098_I00000195&fixed=1&compare=remaining_initial1#fixedPairReview) · [442 秒短回应检查](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=remaining_initial3#fixedPairReview)。5 条实际数据验收与 194 次对照播放通过；同规则、串行处理，未启动全量。

**两条句末完整性试修 J 已供人工检查（2026-10-29）。** 仅串行处理 170 秒／190 秒来源，各 20 个初始候选；前者 16 个初始切分起止不变，后者 20 个全不变。170 秒长句恢复为 25.160–35.498；He’s／He／What 残尾不再追加；190 秒 San Diego、During 和末尾求职整句保持修复且通过。恢复子段不能确认完整时保留原整段候选供人工复核，未放宽音质规则；两条最终均 16 段自动通过，另各 4 段整段候选未自动通过。Man／What 本次未成 pair，明确记录，不宣称无覆盖损失或全量可用。[170 秒三列对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=trial_long_sentence#fixedPairReview) · [190 秒对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&compare=trial_during#fixedPairReview) · [规则、逐段表与验收](experiments/pair_semantic_trial2_20261029/README.md)。5 项回归与 50 次对照播放检查通过；仅三段新音质／媒体计算，其余精确复用；其他来源和旧全量未重跑。

**已定位旧版切分更好的具体原因（2026-10-29）：保留“先切完整候选”的流程。** 串行对七条做 V1/F/G/L/H 同输入切分试算：短尾补回函数会把原来的完整句末补成下一句碎片，13 处与旧全量→最新版的实际最终通过差异精确对应（例如 He's mad.→He's mad. He's）；5 秒停顿与贪心策略耦合，第一条把 Man 与后续长句连在一起，导致 Mufasa / movie theater 在筛音质前已被拆开；恢复子段虽标记 recovered_partial_sentence，仍可能音质通过。问题不是“先切候选”方向错误，而是覆盖量优先、缺少完整句末保护及后续句子的整体切分评估。[逐项复现、13 处起止与代码依据](experiments/pair_segmentation_diagnosis_20261029/README.md)。原 HTML 已标明原因，未修改切分代码或启动全量。

**历史 H 对照：此前认可的旧全量版 → H 版，七条统一基线（2026-10-29；当前 J 见顶部入口）。** 旧结果直接取 `seamless_listen_local_full/results`，不是 L 版、小批审核或同版重跑。最终通过 **205→206 段，159 段起止完全一致，39 组变化**。原 HTML 左右只放最终通过片段，未通过提案单独折叠；逐段起止和原文含全部不变片段。第 1、2、6 条分别有 25／23／6 个原对齐词未再覆盖，尚不能判为只有明确优化，未启动全量。[全部七条统计与 39 组起止／内容](experiments/pair_head_seven_20261029/PRODUCTION_COMPARISON.md) · [第一条前后试听](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=production_diff1#fixedPairReview) · [第六条 maybe two or three days 减少](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&compare=production_diff2#fixedPairReview)。已确认累计切分改动之外，实验入口还从“既有音质阻断参与切分”变为“先切完整候选再筛音质”，不能把所有变化归为小边界修复。下方 L→H 和混合基线记录是历史排查，不是本次主结论。

**七来源新旧比较口径纠正：统一 L→H，逐段起止与内容已补齐。** 上一条“前两条历史 B→H、后五条 H 同版重跑”不能当作统一新旧方案比较。本轮补齐前两条 L 版：七条最终通过共 206 段，204 段起止完全相同；仅 442 秒来源 Yeah 的起点 55.070→54.378、Okay 的起点 213.983→213.143，终点均不变。Right 扩展未通过并完整回退；其余六条所有候选与终端片段起止、原文及音质结论一致。历史 B→H 的较大减少仍另列，不是最后 L→H 补接造成。[七条完整起止／原文表](experiments/pair_head_seven_20261029/ADJACENT_INTERVALS.md) · [原 HTML](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1#adjacentIntervals)。未启动全量。

**历史七来源 H 版核验（2026-10-29）：未满足全量启动条件，未启动全量。** 按指定七条串行重跑，之前验证过的五条与上轮 H 结果完全一致；最早两条此前未应用 H，与原连续审核版相比，最终通过覆盖分别 96.826→83.124 秒、48.428→41.593 秒，25＋23 个原通过对齐词未进入新版通过集。它们仍在初始候选内，但未通过整段音质筛选且局部恢复未全部保回，不能直接认定是误杀或改善。另两处结尾加入下一句开头 He’s／He，需要判断边界。[170 秒来源七组对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3610_I00000088&fixed=1&compare=seven_diff1#fixedPairReview) · [196 秒来源完整话语减少](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V01_S0122_I00000125&fixed=1&compare=seven_diff2#fixedPairReview) · [七条统计、基线口径与证据](experiments/pair_head_seven_20261029/README.md)。切分规则本轮未修改，生产与训练未替换。

**原生视频8／16 fps轻量复核与测速（2026-10-29）：** [打开复核页](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/fps_decision/) · [操作、抽样与测速说明](experiments/listen_visual_input50_20261029/native_fps_decision/README.md)。复用50条原文，9条关键分歧＋6条随机一致项，默认甲乙盲化，只需回答具体问题；全50条仍可查看。用户确认可在原训练上叠加单GPU低并发测速；三次试跑收到SIGKILL，暂未得到完整速度结果，已停止继续重启。复核页已通过浏览器检查，结果不作为人工真值。

**C prompt 原生视频输入对照已完成（2026-10-29）：** [带声音的三组对照](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/?comparison=inputs) · [结果与判断](experiments/listen_visual_input50_20261029/RESULT.md) · [原文案例](experiments/listen_visual_input50_20261029/REVIEW_NOTES.md) · [实验规格与续跑](experiments/listen_visual_input50_20261029/README.md)。同50条、879次请求：多图8fps／原生视频8fps／原生视频16fps，有效请求分别281／284／284（各293次）；整段三组均50/50通过结构检查。原生视频减少部分碎片，但存在宽区间概括与规则重复；16fps没有稳定优势。三机分散运行、每台最多1路源视频解码；已结束，未替换生产标签。原 B/C/D 对照及音轨切换均保留。

**历史五来源验证：句首短回应补接，五条来源验证完成（2026-10-29）。** [Okay 补回对照](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=okay_head#fixedPairReview) · [Yeah 补回对照](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=yeah_head#fixedPairReview) · [Right 扩展失败及完整回退](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=right_head#fixedPairReview)。H 版只补接未覆盖且在 0.3 秒内接着同一人正式发言的短反馈，扩展必须通过原音质规则，否则恢复原完整 pair。前四条切分不变；第五条最终通过仍 55 段，53 段不变，只提前 Okay／Yeah 两段起点，无旧覆盖词丢失。Right 扩展待复核，原片段保留。29 项测试、17 个对照视频串行浏览器验收通过；原生产与训练未修改，历史 L/G/F 对照保留。[规则、统计与恢复](experiments/pair_head_attach5_20261029/README.md)。

**新增三条原始长视频验证（2026-10-29）：发现一处新版退步。** [176 秒：19→19 段，全不变](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S5228_I00000302&fixed=1&compare=unchanged#fixedPairReview) · [398 秒：44→44 段，全不变](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S2098_I00000195&fixed=1&compare=unchanged#fixedPairReview) · [442 秒：54→55 段，重点看 Okay 遗漏](http://127.0.0.1:8875/listen_full_source/?pair=improvised_V00_S0862_I00000794&fixed=1&compare=change3#fixedPairReview)。seed=20261030，比较前固定三条新来源，未按改善效果挑选。第三条 48 段边界不变、5 组变化：句末更合理、补回 it’s、两处几十毫秒留白变化，以及遗漏 213.343–213.963 秒的 Okay。只读分项试算定位到“局部反馈分类后未接回随后的正式发言”；句末排序单独启用不丢该词。原 HTML 展示所有变化和完整候选；规则在本轮验证中未继续修改，不应据原两条成功例子直接推广全量。[方法、逐处判断与验收](experiments/pair_boundary_holdout3_20261029/README.md)。

**四版视觉 prompt 的 50 条 HTML 对照（2026-10-29）：** [打开复核页](http://localhost:8876/listen_benchmark_review/prompt_versions_20261029/) · [页面、代码与验收说明](data_previews/listen_benchmark_review/prompt_versions_20261029/README.md)。左侧固定视频、右侧原文对照；包含原始简单版、追加禁止推演版、观察式 v1、允许动作命名版，以及三轮同期对照。已标出动作遗漏、推演残留、重复短区间、过度合并和否定动作措辞等人工复核点；保留全部 50 条与失败响应。Cursor 转发远端 8876；未修改生产标签。

**允许动作命名的观察式prompt验证（2026-10-29）：** [50条配对结果](experiments/listen_visual_actions50_20261029/RESULT.md) · [原文对照](experiments/listen_visual_actions50_20261029/REVIEW_NOTES.md) · [完整prompt与复现](experiments/listen_visual_actions50_20261029/README.md)。新增动作命名许可、非穷尽动作清单及点头运动说明，与观察式v1同期重跑，共586次输出。新版38条nod提及均是否定／未见明显点头的措辞，不能当作检出增加；推演词线索请求比例16.1%→19.4%，仍有重复短区间和过度合并。暂不替换正式方案；原始结果、失败响应及文本核对均保留，未覆盖生产标签。

**历史 L 版：局部 backchannel 分类＋真正句末优先，仅两条长视频（2026-10-29）。** [San Diego／yeah 处理前后](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&compare=local_feedback#fixedPairReview) · [During…marry her 整句对照](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&compare=sentence_priority#fixedPairReview) · [第二条两组切点变化](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205&fixed=1&compare=sentence_priority_2a#fixedPairReview)。相较 G 版，第一条 21→20 段（18 段边界不变），第二条仍 42 段（38 段边界不变）；音质待复核仍 4＋6 段，语义提示为 10＋18 段。新增独立 L 版，原 HTML 默认展示，保留 G／F 历史版。两处目标均通过现有音质规则；During 整句为 11.95 秒、句尾额外留白为 0，仍保留边界提示，需试听结尾。5 秒停顿、150ms 实质发言安全余量和 QC 规则不变；旧版已覆盖 speaker 单词未丢失，前两处修复保留，未应用全量或生产数据。[实现、运行与验收](experiments/pair_boundary_real_audit_20261029/localbc_pairs/README.md)。

**新观察式prompt的50条验证（2026-10-29）：** [结果与限制](experiments/listen_visual_observation50_20261029/RESULT.md) · [三组原文对照例子](experiments/listen_visual_observation50_20261029/REVIEW_NOTES.md) · [完整prompt与运行方式](experiments/listen_visual_observation50_20261029/README.md)。复用同50条，Qwen3-VL-8B同期重跑旧／新prompt，共586次输出；含明确推演词线索的有效请求占比98.3%→14.9%，短于0.2秒候选67.0%→53.5%，但时长中位数仍0.133秒，仍有重复、推演残留及12个无效新组请求（含2个截断）。关键词统计不是准确率；未跑方案A、未覆盖生产标签。

**视觉prompt禁止推演的50条配对验证（2026-10-29）：** [完整结果与原文例子](experiments/listen_visual_observable50_20261029/RESULT.md) · [模型、完整prompt与复现命令](experiments/listen_visual_observable50_20261029/README.md)。从新50h随机抽50条，Qwen3-VL-8B新旧prompt同输入重跑，共586次输出；新增约束后，含明确推演词线索的有效请求占比从98.3%降至37.4%，仍未消除推演，且重复短区间和动作描述覆盖变化仍需检查。该比例是关键词诊断，不是准确率。原生产标注、训练标签与缓存未覆盖。

**总索引疑问细节补充（2026-10-29）：** [数据处理、模型对照、去重／静音、筛选、16组pipeline、提示词、Event/BCE及评测问答](related/P2_总索引问答与疑问细节补充.md)。汇总本轮问答与原始依据，集中保存HTML和下载入口；历史数字注明版本，运行状态以最新记录为准。

**历史 G 版语义边界待复核入口：** [第一条 11 段](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&version=local-listening-boundary-repair-gap5-3&filter=semantic#fixedPairReview) · [第二条 20 段](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205&fixed=1&version=local-listening-boundary-repair-gap5-3&filter=semantic#fixedPairReview)。每段列出边界原因、首尾转写，支持“听开头／听结尾”和跳回完整原视频的句首／句尾附近。提示不等于已确认截断：包括词边界片段、推测分句、长停顿及源转写首尾未确认，部分标签继承自覆盖语句，需要人工核实实际切口。

**历史 G 版：5 秒停顿阈值，仅两条长视频（2026-10-29）。** [190 秒来源：21 个初始候选，4 个音质疑点／11 个语义边界提示](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&version=local-listening-boundary-repair-gap5-3#fixedPairReview) · [354 秒来源：42 个初始候选，6 个音质疑点／20 个语义边界提示](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205&fixed=1&version=local-listening-boundary-repair-gap5-3#fixedPairReview) · [“It was crazy” 同段播放：G1-14](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&version=local-listening-boundary-repair-gap5-3&clip=G1-14#fixedPairReview)。相邻词间隔超过 5 秒才按长停顿拆分，2～11.95 秒及其余角色／音质约束不变。共 63 个初始候选和 7 个恢复子段；6 个新边界片段重算 QC3/QC4，64 个精确匹配片段复用证据并重新路由。该历史版本展示 10 个音质待复核候选，可另筛语义边界提示，两类可能重叠；“候选时长”不再暗示语义完整。下拉可回看 3 秒历史版，新编号 G 与旧编号 F 分开。未扩展全量或修改生产数据。[运行与验收](experiments/pair_boundary_real_audit_20261029/gap5_pairs/README.md)。

**历史 3 秒版：两条长视频的边界修复＋完整 pair 误杀复核（2026-10-29）：** [190 秒来源：21 个完整候选，3 个待复核](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&fixed=1&version=local-listening-boundary-repair-2#fixedPairReview) · [354 秒来源：43 个完整候选，5 个待复核](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205&fixed=1&version=local-listening-boundary-repair-2#fixedPairReview)。先修复短尾回退和角色区间回应尾部问题，再重新切分这两条来源、按未改动的 QC3/QC4 和音质路由检查。新边界的 11 段重新计算音质，其余 58 段仅在边界与归一化音轨完全一致时复用证据；共 64 个初始完整候选及 5 个局部恢复子段。该历史版本展示 8 个暂缓／排除的完整父 pair，播放器不加上下文，黄色标出规则疑点；可查看全部候选、局部恢复子段及记录人工结论。两处遗漏内容已完整保留并通过规则检查。此前 52 段未覆盖区间改为历史参考。仅这两条来源，未扩展全量、未修改旧生产结果或训练。[实现与验收](experiments/pair_boundary_real_audit_20261029/fixed_pairs/README.md) · [完整候选清单](experiments/pair_boundary_real_audit_20261029/review/fixed_pairs/manifest.json)。

**真实 pair 切分问题复现已加入连续审核（2026-10-29）：** [完整原视频审核入口（原两条＋新增两条）](http://127.0.0.1:8875/listen_full_source/) · [短尾整句遗漏：190 秒会话，定位 183 秒](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&t=183) · [句尾与回应遗漏：353.987 秒会话，定位 333 秒](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205&t=333)。新增样本展示完整长视频、生产 pair 及未覆盖区间、裁框参考、原始转写；红色标注遗漏，蓝色为诊断替代切法，并提供问题跳转、当前／替代视频对照和原始双路 WAV。第一例遗漏 184.921–186.642 秒整句，第二例遗漏 speaker 的 “up.”（335.787–335.928 秒）和 listener 的 “Oh, yeah.”（336.115–336.355 秒）。替代切法未重新做音质准入，未修改生产切分或训练。[复现与媒体证据](experiments/pair_boundary_real_audit_20261029/review/) · [发布脚本](experiments/pair_boundary_real_audit_20261029/publish_review.py)。

- **局部处理前／处理后对照（直接在原 HTML 内）：** [问题一](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175#case1) · [问题二](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205#case2)。左侧为处理前的连续原视频局部，右侧为当前生产规则切分后保留的同位置片段；两边源时间起点一致，右侧在实际 pair 终点停止。诊断替代切法另放在折叠区，不与“处理后”混淆。

**疑似误杀片段已单独提取供人工复核（2026-10-29）：** [190 秒来源：17 段](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V02_S3579_I00000175&review=1#manualReview) · [354 秒来源：35 段](http://127.0.0.1:8875/listen_full_source/?pair=naturalistic_V03_S1909_I00000205&review=1#manualReview)。原 HTML 内新增复核区，覆盖这两条来源的全部连续未纳入区间（含极短间隙，不预判均为误杀），优先展示音质／阻断疑点。每段保留前后最多 3 秒上下文，黄色标出待复核范围，可单听 A/B、查看具体排除证据、保存结论并导出 JSON；生产数据未修改。[提取清单与媒体](experiments/pair_boundary_real_audit_20261029/review/manual_review/manifest.json) · [提取脚本](experiments/pair_boundary_real_audit_20261029/export_manual_review.py)。

**Qwen自主规划已有3条视频预览：** [打开12个视频（AV/AV＋Event×双seed）](http://127.0.0.1:8876/p2_av_review/autonomous_preview_20261028/) · [范围及失败记录](experiments/p2_autonomous_guided2_test24_20261028/README.md)。仅3/24条规划通过；12个视频及这3条的同样本配对计分已完成；完整24条评测尚未完成，不能代表整体结果。

**Qwen自主规划推理代码已准备（未启动GPU）：** [链路、运行命令与验收边界](related/P2_Qwen自主规划双次forward推理评测.md) · [原test24任务清单](experiments/p2_autonomous_test24_20261028/README.md) · [脚本](scripts/p2_autonomous_eval.py)。speaker音频＋listener首帧→自主规划→同一规划进入Qwen第二次forward与Gemma；与真实标注指导结果隔离。当前冻结Qwen的旧4组/新版Stage1适用；S2 LoRA明确拒绝共用冻结hidden。

**本批评测已上线（6组288输出）：** [原HTML总入口](http://127.0.0.1:8876/p2_av_review/) · [重新整合标注后的test24结果](http://127.0.0.1:8876/p2_av_review/reintegrated_20261028/) · [发布验收](experiments/p2_reintegrated_final_eval_20261028/HTML_PUBLISHED.json)。原历史结果保留；新版AV＋Feature等待最终1249步权重后另补48输出。

**当前评估范围已纠正：所有pipeline统一原独立test24。** [准确范围、代码与结果入口](experiments/p2_reintegrated_final_eval_20261028/README.md) · [24卡实时进度](experiments/p2_reintegrated_final_eval_20261028/eval_status.json)。每组24条×2seed；旧四组＋新版AV/AV＋Event共288输出，AV＋Feature最终权重待补48。只更新整合指导文本，不加入train、不扩充test成员；此前576/672的宽范围队列已停止。训练已按新指令从第587步恢复，继续三机24卡原队列及后续LoRA加速接续。

**Omni LoRA后续接续已准备：** [运行与恢复说明](experiments/p2_full50h_omni_accel_20261028/README.md) · [等待/测试/训练状态](experiments/p2_full50h_omni_accel_20261028/status.json)。当前四组冻结Qwen完成后，自动比较“24卡串行”和“四组各6卡＋时长对齐”；正确性及吞吐通过才并行并动态扩卡，否则自动串行。当前S1不中断，新LoRA GPU对照尚未执行。

**统一加速策略与下次复用入口：** [跨卡时长对齐＋冻结文本编码缓存：配置、实测、适用条件与复用步骤](related/P2_训练加速策略总览与自动接续.md)。记录384条窗口/24卡共同采样、Gemma缓存与在线Qwen LoRA边界、ABBA耗时降低31%的证据、未采用的重计算策略，以及对照公平性和断点恢复要求。

**当前正式训练：完整50h＋跨卡时长对齐（2026-10-28）。** [新八组运行、结果与恢复入口](experiments/p2_full50h_duration384_20261028/README.md) · [24卡实时训练状态](experiments/p2_full50h_duration384_20261028/training_queue/state.json) · [加速实现说明](related/P2_训练加速策略总览与自动接续.md)。24卡ABBA短测从7.62降至5.26秒/步（耗时降低31%），保存恢复后模型/优化器逐位一致。已自动切换八组统一采样；完整29,975条、每阶段1,249步不变，当前S1单组独占24卡；后续LoRA卡分配见上方自动接续入口。原随机顺序第53步已归档，不混入新版本。

**当前正式目标：完整新50h、29,975条全部保留（2026-10-28）。** [恢复规则、缓存和自动训练入口](experiments/p2_full50h_hard_keep_20261028/README.md) · [标签恢复进度](experiments/p2_full50h_hard_keep_20261028/RECOVERY_PROGRESS.json) · [三机24卡缓存状态](experiments/p2_full50h_hard_keep_20261028/pool/state.json)。模型二次审核仅记录；硬处理失败事件保留原文，整段不排除。每阶段1,249步，先四组冻结Qwen，再四组Qwen LoRA，每组独占24卡。此前10,893条小子集实验已保存第144步并停止，下面旧入口仅作历史记录。

**当前新50h训练顺序：单组独占三机24卡，先四组冻结Qwen，再四组Qwen LoRA（2026-10-28）。** [调度顺序与batch配置](experiments/p2_new50h_guided_chain_20261027/training_schedule.json) · [运行状态](experiments/p2_new50h_guided_chain_20261027/README.md)。每卡1样本、梯度累积1次、global batch=24；覆盖此前新50h四组各6卡并行方案。

**2026-10-28加速更新：** [新50h运行说明](experiments/p2_new50h_guided_chain_20261027/README.md)：最终GPU S0改为四条配对链并行；该条原训练并行配置已被上方单组24卡顺序覆盖。当前阶段及是否真正开始训练，以实时状态为准。

**已启动：新50h方案A整合 → 新条件缓存 → 标注指导8组自动接续。** [运行、成员口径和恢复说明](experiments/p2_new50h_guided_chain_20261027/README.md) · [三机24卡任务状态](experiments/p2_new50h_guided_chain_20261027/pool/state.json) · [接续状态](experiments/p2_new50h_guided_chain_20261027/director/status.json)。严格使用交接代码最终成功发布的同一批样本，完整保留拒绝清单；每阶段global batch=24、seed=20261023、ceil(N/24)步。原四组固定文本LoRA训练继续运行；新正式训练须等缓存及真实GPU验证通过后自动启动。

**新50h方案A重新整合交接：** [29,975段输入、三机24卡命令、续跑验收与缓存边界](related/P2_新50h方案A打标信息重新整合交接.md)。原交接说明保留；整套新50h整合现已启动，运行状态见上方自动接续入口，原标签及缓存保留。

**最新30B方案A复核页：** [打开原文截取与去重结果](http://127.0.0.1:8876/listen_benchmark_review/semantic_prompt_pilot_20261027/)。100条、60条通过／40条拒绝，默认定位S1400；保留原文、来源、音视频及被拒提案。旧页已备份，Cursor转发远端8876查看。

**方案A原文截取去重修复（2026-10-27）：** [检查发现、严格原文输出、动作遗漏防护与脚本入口](related/P2_方案A原文截取去重校验与交接.md)。使用Qwen3-30B-A3B-Instruct-2507；模型选择原文片段和去重来源，代码拒绝改写／遗漏，不再静默修补。运行、缓存恢复、集群汇总、CPU验收和展示构建已同步修复。原始数据及训练缓存保留；验证结果以文档关联报告为准。

**历史加速配置：原50h四组Qwen LoRA并行训练（2026-10-27，未用于当前新50h队列）。** [完整调优、验证与恢复说明](experiments/p2_legacy50h_qwen_lora_20261027/performance/README.md)、[历史四组进度及输出路径](experiments/p2_legacy50h_qwen_lora_20261027/parallel_queue/state.json)、[历史三机配置](experiments/p2_legacy50h_qwen_lora_20261027/parallel_cluster.json)、[历史配置与GPU分配](experiments/p2_legacy50h_qwen_lora_20261027/parallel_queue/jobs.json)。

> **2026-10-28核查时状态：** `p2_full50h_duration384_20261028`原队列在冻结Qwen和Qwen LoRA两个阶段都执行“单组24卡串行”；八组均已应用384窗口时长对齐及Gemma缓存，但没有启用本段的“四组各6卡并行＋完成后动态扩到12/24卡”。下述并行成绩来自旧数据、固定prompt的实验。现已另挂接`p2_full50h_omni_accel_20261028`，等待四组S1结束后重新实测并选择LoRA执行方案；不能直接叠加历史提速比例。

- **并行方式：** AV／AV＋Event／AV＋Feature／AV＋Event＋Feature四组同时运行，初始每组6卡（三机各2卡），合计24卡。每卡microbatch=1、梯度累积4次，每组global batch=24；初始每机GPU分配为E0=0/1、E1=2/3、E0F=4/5、E1F=6/7。
- **保持的实验条件：** 原始30,415段50h、固定Gemma文本、seed=20261023、原损失系数；各组从对应原第1,268步权重接续Qwen LoRA，再训练1,268步。E0调优前已完整保存本阶段第105步，现从该步继续。原权重对应关系见[训练说明](experiments/p2_legacy50h_qwen_lora_20261027/README.md)；与新29,975段后处理数据实验分开记录。
- **实际采用的执行配置：** 保留LTX全部层及Qwen梯度重计算，训练allocator上限128GiB。增大显存占用、减少重计算的候选在24卡短测中未稳定提速，因此正式训练采用提高实验并行度的方案。
- **自动调度与恢复：** 某组完成后，未完成组保存完整checkpoint，再扩展到12／24卡，累积次数相应改为2／1，global batch始终为24。保留每100步保存、断点恢复和TensorBoard；[调度入口](scripts/p2_legacy_parallel.py)、[启动验收](experiments/p2_legacy50h_qwen_lora_20261027/parallel_queue/startup_verified.json)。旧串行`queue/STOP`保留，当前应查看`parallel_queue`。
- **效果与验证：** 四组6卡并行短测的每组中位步时约18.3秒；同批5步后，6卡／8卡方案与原24卡方案的模型及优化器状态逐位一致。并行启动时估计四组约7–9小时完成，该估计需随实际步速更新；实时进度以上方状态文件为准。

**生成指导语义去冗余试验（2026-10-27）：** [模型、代码约束与复现方式](related/P2_生成用语义去冗余实现.md)、[100条实测结果与问题样本](experiments/listen_semantic_prompt_pilot_20261027/README.md)。Qwen3-VL-8B仅处理现有视觉描述，代码检查来源和时间；音频原文与有效区间保持不变。66/100通过程序与模型自检，34条拒绝，11项回归通过；当前模型仍有语义残留问题，不具备全量自动发布条件。输出为独立派生prompt，未替换原标签、事件缓存或训练配置。

**16组训练代码与新50h缓存（2026-10-27）：** [训练实现、矩阵与运行说明](related/P2_十六组训练实现与运行.md)、[设计记录](related/P2_十六组文本条件与QwenLoRA训练设计.md)、[16份配置索引](configs/p2_matrix16/matrix.json)。固定／时段标注 × AV／AV＋Event／AV＋Feature／AV＋Event＋Feature × 冻结Qwen／二阶段Qwen LoRA；阶段二在线hidden接收生成与辅助损失，不做规划SFT。支持每100步保存、严格恢复、TensorBoard及新checkpoint条件推理接口。相关88项测试通过；GPU验证及限制见运行说明。尚未启动正式训练。

> **新50h新增缓存全部完成（2026-10-27 07:19 UTC）：** Gemma、输入＋计划Qwen hidden、静音后Feature teacher均为29,975/29,975。[完成记录与日志](experiments/p2_plan_cache_train50h_verbatim_20261027/README.md)、[条件缓存完成验收](experiments/p2_plan_conditions_train50h_verbatim_20261027/COMPLETE.json)、[teacher完成验收](experiments/p2_matrix16_teacher50h_20261027/COMPLETE.json)。条件缓存14/14分片校验成功，teacher48/48分片成功；覆盖无遗漏、无重复。空卡为任务正常完成后释放，未启动16组正式训练。

**最终数据处理代码备份（2026-10-27）：** [覆盖核查、GitHub固定版本与恢复说明](related/P2_最终后处理与筛选代码备份恢复.md)。补齐全量音频后处理、新50h／双300h、test48及原文读取代码；140项CPU回归通过，另保存163个生产运行时冻结文件。数据和缓存仍在原机器。

> **显式规划条件缓存读取接口：** [实现、命令与读取API](related/P2_显式规划条件缓存实现与运行.md)。逐样本Gemma及A2输入＋计划hidden、原指导模板、分片和断点复用；最新策略为保留原标注，不新增翻译。已与新16组训练入口衔接。

**最终数据与缓存位置总览（2026-10-27）：** [全量、新50h、两份300h、test48/test24及共享缓存读取索引](related/P2_最终数据集与缓存位置总览.md)。明确当前配置、manifest、原文／七类标签、静音配方、原媒体、视频／音频latent与事件缓存的实际位置，以及旧50h区别、test缓存范围和原文读取接口。

> **显式回应规划首版约束已确认（2026-10-27）**：[Qwen规划→计划回填hidden＋Gemma双通路实验方案](related/P2_显式回应规划双通路实验方案草案.md)。统一使用重新筛选的29,975段后处理50h及512×512几何；首版Qwen全冻结，双通路使用与此前英文指导评测相同格式的文本。沿用A0/A1/A2对照，验证数据待用户提供，具体预算未冻结；尚未实现新pipeline或启动训练。

> **四组正式训练代码备份已补齐（2026-10-27）**：[AV／AV＋Event／AV＋Feature／AV＋Event＋Feature版本与恢复索引](related/P2_四组训练代码备份与恢复索引.md)、[GitHub固定快照](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/p2-fourway-actual-runs-20261027/p2_training/fourway_actual_runs_20261027)。按实际run逐一匹配73/70/73/73个源码指纹，补齐旧备份遗漏的preflight与Feature版本；四组副本681次CPU测试通过。权重、媒体和缓存仍留在原机器。

> **全英文指导重推理与双口径评测完成（2026-10-27）**：[运行、翻译记录与验收](experiments/p2_guided_en_full_eval_20261027/README.md)。384个英文指导输出全部重新生成、评测并替换页面中的旧指导输出；加上固定版共768个。GLARE可切换严格版（train2/24、test7/24）与去额外门槛版（两套均24/24）。[打开HTML](http://127.0.0.1:8876/p2_av_review/)，浏览器数值、播放及媒体来源验证全部通过。

**双300h训练子集已完成并验收（2026-10-27）：** [七类版：176,147段](experiments/p2_train300h_seven_20261027/README.md)、[全部原文版：159,056段](experiments/p2_train300h_open_20261027/README.md)、[筛选和整合约定](related/P2_双300h训练子集筛选方案.md)。按用户确认独立筛选、人脸p10≥160、每pair最多24段，源时间不重叠且保持train/dev/test隔离。原文版沿用旧去重整合，保留七类之外原始视觉观察，不新增语义归类；音频沿用已验收后处理。两版逐条校验及34／39条实际CPU读取通过，复用媒体缓存，不新增WAV；原文版提供独立原文读取器，未改造固定七类监督头。各目录 `COMPLETE.json` 为最终验收依据，未启动训练。

> **GLARE检测修复完成（2026-10-27）**：[RGB人脸检测／BGR情绪分类修复与重测](experiments/p2_glare_detector_fix_20261027/README.md)，816个独立视频统一重测、28组评分及HTML发布完成。test八条件共同有效片段0→10/24；检测门槛与指标公式不变，原结果保留。浏览器逐数值核对通过。

**独立 test 48 条快速验证子集（2026-10-27，已验收）：** [数据入口、类别统计与读取说明](experiments/p2_test48_backchannel_face_20261027/README.md)。从已完成后处理的独立 test 候选中，沿用 50h 的大脸、回应丰富度、七类覆盖、源区间去重及身份隔离标准选出 48 条，共 285.1945 秒；全部原始人脸短边 p10≥304 像素，七类均覆盖。含原 test24 的全部 24 条，另增 24 条；配对 speaker 音频、listener 裁剪视频、原文标签和虚拟静音配方齐备，48 条媒体及波形读取验证通过。不新增处理后 WAV，未创建训练 latent 或启动训练。

> **后处理音频训练入口修正（2026-10-27）**：[新训练版本与命令](releases/p2_audio_postprocess_v1/README.md)，支持50h历史缓存与全量新缓存、更新事件标签及共享预检查；[64条CPU／8条CUDA读取验证](experiments/p2_audio_training_reader_20261027/README.md)。未启动新训练。

> **固定／原文标注指导完整配对评测（2026-10-27）**：训练集24条与独立test24条，各组各48个输出（双seed），指导版共384个。唯一干预是Gemma正向文本；完整指标自动接续并更新[有声可视化](http://127.0.0.1:8876/p2_av_review/)。[范围、任务清单与进度入口](experiments/p2_guided_full_eval_20261027/README.md)。

**音视频可视化页已更新（2026-10-27）：** [打开页面（8876）](http://127.0.0.1:8876/p2_av_review/)、[来源与验证说明](experiments/p2_av_review_20261027/README.md)。GT有声播放、实际speaker音频及48份ASR、listener单独/叠加speaker切换、全样本连续下滑、各pipeline统一指标均已接入。覆盖train24/test24、四组权重、固定／标注指导和双seed，共768个输出，每组每条件每数据集48个；完整评测与GLARE修正版发布状态见上方入口。

**原文标注→Gemma逐样本指导推理已实现（2026-10-27）：** [实现与命令](related/P2_原文标注Gemma指导推理实现.md)、[10条GPU验证与输出](experiments/p2_guided_prompt_train24_20261027/README.md)、[test24文本与缓存](experiments/p2_guided_prompt_test24_20261027/README.md)。保留原描述/原时间，合并音频描述与发声文本；48份文本缓存完整编码，支持1024/2048/4096分档，不截断。四组权重、长文本及固定prompt控制完成验证，35项新增测试与37项原评测测试通过。属于目标标注指导诊断，不替代source-only评测；未启动训练或修改已有HTML。

**全量开放描述与事件时间统计（2026-10-27，无图）：** [完整原文目录、时间口径与逐事件入口](experiments/p2_full_open_description_times_20261027/README.md)、[整合结果全部原文类型](experiments/p2_full_open_description_times_20261027/retained_全部原文类型.csv)、[未纳入七类的视觉原文类型](experiments/p2_full_open_description_times_20261027/unassigned_visual_全部原文类型.csv)、[七类时间对照](experiments/p2_full_open_description_times_20261027/七类时间对照.csv)。722,706 段全部保留；补充七类之外的原始观察描述、每种原文的区间并集时长和持续时间分布，以及片段内／源视频起止时间。原文类型不等于去重后的语义类别，额外视觉观察不自动视为已确认回应。[此前七类报告](experiments/p2_full_backchannel_report_20261027/README.md)仅供对照；“未检出七类事件”不代表没有其他行为。未修改标签、训练准入或缓存。后续未明确要求画图时，只出表格与文字。


**独立test24四组权重评测完成（2026-10-27）：** [对比结果](experiments/p2_test24_fourway_eval_20261027/report.md)、[音频监督与近静音核查](experiments/p2_test24_fourway_eval_20261027/audio_investigation.md)。192条生成和AV指标计算完成，与四组实际旧训练集无身份交集；GLARE共同成功仅4条/3组。已确认训练含listener音频，但生成大多近静音，不能将UTMOS/SyncNet分数当作有效回应质量结论。HTML暂未修改。

**四组最终权重：训练集随机24条快速验证完成（2026-10-27）：** [比较结论与指标表](experiments/p2_train24_fourway_eval_20261027/report.md)、[生成盲看页](experiments/p2_train24_fourway_eval_20261027/review/index.html)。AV、AV＋Event、AV＋Feature、AV＋Event＋Feature均用第1,268步、同24条、同两个seed，192条生成及AV指标计算全部成功。GLARE共同成功子集仅5条/2组，尚不能证明Event改善回应；Feature存在UTMOS数值下降信号，但后续发现输出近静音，不能直接解释为有效发声音质下降。原实验计划H1、独立留出和speaker替换等证据边界已逐项对照。

**独立 test 24 条快速验证子集已完成（2026-10-27）：** [数据、配方与筛选说明](experiments/p2_test24_backchannel_face_20261027/README.md)。24 条、154.594 秒，覆盖七类，全部原始人脸短边 p10≥310 像素；与新 50h train 的会话／参与者／连通组无交集。已补做同规格模型标注及音频后处理，保留原文与虚拟静音配方，媒体和实际读取已验证；标签仍是模型弱标注。

**静音后重新筛选 50h 训练集已完成（2026-10-27）：** [数据入口、筛选规则与读取说明](experiments/p2_train50h_postprocessed_reselected_20261027/README.md)。29,975 段、50 小时零 0.532 秒，全部 train；使用后处理后保留的事件重新评分，七类均覆盖，99.17% 原始人脸短边 p10≥224。全部缓存合同与标签配方校验通过，并完成 34 段真实 CPU 训练读取；复用新音频／事件和原视频缓存，不新增 WAV，未启动训练。

**E0＋Feature → E1＋Feature → 722,706 段全量音频处理已接续（2026-10-26）：** [全量运行与恢复说明](experiments/p2_full_audio_postprocess_crop384_20261026/README.md)、[全量实时状态](experiments/p2_full_audio_postprocess_crop384_20261026/status.json)、[范围与处理规则](related/P2_音频后处理增量缓存与训练读取交接.md)。全量索引已冻结，CPU 配方/标签准备已启动；三机 24 卡音频缓存等待 E1 完成后自动接手。30 项测试、100 条波形对照、真实 VAE CPU 编码和缺失缓存恢复验证通过；三机备用接管实测约 6.5 秒。全量 GPU 缓存及最终验收尚未完成，不启动全量训练。

**E0＋Feature → E1＋Feature 训练自动接续：** [运行与恢复说明](experiments/p2_e1_feature01_train50h_20261026/README.md)、[队列实时状态](experiments/p2_e1_feature01_train50h_20261026/queue/state.json)。三机冗余守护，E0 完成后自动以同数据/seed/初始化启动 E1；失败从完整 checkpoint 重试。22 项测试、共享锁互斥及备用接管演练通过；15:20 UTC 的 NCCL 超时后已自动从第 800 步恢复并重新出步。实时步数以状态文件为准。

**音频后处理代码备份（2026-10-26）：** [GitHub固定版本](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/listener-audio-postprocess-train50h-20261026/event_labeling/audio_postprocess_20261026)、[本地恢复说明](../audio-driven-v5-data-processing/event_labeling/audio_postprocess_20261026/README.md)。覆盖50h音频后处理、三机24卡调度、训练读取适配、依赖源码和验收记录；备份副本54项CPU测试通过。

**50h训练子集音频后处理（三机24卡，2026-10-26）：** [代码、规则与运行入口](related/P2_50h音频后处理三机运行说明.md)、[当前状态](experiments/p2_train50h_audio_postprocess_20261026/status.json)。固定30,415段已全部处理并验收通过，保留23,958个音频事件、12,300段有发声标签，七类仍全部覆盖。新WAV为0，新增音频latent约1.26 GB；训练读取已接入 [新配置](experiments/p2_train50h_audio_postprocess_20261026/view/config.json)。原媒体、视觉标签和原缓存保留，未启动训练。

**AV＋Feature 50h 三机 24 卡正式训练已启动（2026-10-26）：** [配置、验证与运行说明](experiments/p2_feature01_train50h_20261026/README.md)、[实时状态](experiments/p2_feature01_train50h_20261026/status.json)。AV＋0.1×视频 Feature＋0.1×音频 Feature，不加生成 Event 监督；沿用 E0 的 30,415 段、seed=20261023、batch=24、1,268 步，从相同共享初始化重新训练。158 项 CPU 测试、全量 teacher 审计、最长片段实测及 24 卡逐位一致恢复通过；正式启动验收已完成至少 14 步，52 项 TensorBoard 指标正常，每 100 步保存。

**虚拟静音音轨＋音频缓存增量更新代码就绪（2026-10-26）：** [实现与使用说明](related/P2_虚拟静音音轨与音频缓存增量更新.md)、[验证记录与100条配置](experiments/listen_audio_virtual_validation_20261026/README.md)。全量无需额外保存处理后WAV：保存配方，读取时静音，单独更新音频latent，训练时组合原视频缓存。固定100条已完整运行并逐条验收：98条更新音频缓存、2条原样复用，波形对照、训练读取与续跑全部通过；26项测试通过。未新增WAV或视频缓存，验证仅用CPU，全量转换和训练未启动。

**OmniResponse 音视频评测已接入（2026-10-26）：** [实现与运行](related/P2_OmniResponse评测实现与运行.md)、[输入接口](examples/p2_omni_eval/README.md)、[真实 CPU 验收](experiments/p2_omni_eval_validation_20261026/README.md)。默认 FVD-32、UTMOSv2、SyncNet LSE-D/LSE-C；可选提供特征后的 FD，以及有转写后的 METEOR、ROUGE-L、BERTScore、Distinct。71 项回归通过，真实 I3D/UTMOS/SyncNet/文本模型、缓存复用及共同支持集合比较已验收；修复 SyncNet 上游转码延迟引起的帧数偏移。支持原版 E1 与预检查修复版 E0 的生成入口。训练源码/环境未修改；正式评测数据仍待提供，工程自配对不作为效果结果。

**固定100条：多字中文ASR移除并静音（2026-10-26）：** [执行规则](related/P2_多字中文ASR移除与静音规则.md)、[本轮数据与复现](experiments/listen_official_chinese100_20261026/README.md)、[Cursor转发8876试听](http://127.0.0.1:8876/listen_benchmark_review/audio_official_chinese100/)。保留9条单字“嗯／哈／呵／哎”标注，移出“南京地铁／我唔记得咗／不能”3条并静音对应区间。最终38条有发声标注、62条无发声标注；支持筛选前后试听，原数据及训练缓存未替换。

**同100条：原文保留与官方时间对齐（2026-10-26）：** [对齐规则](related/P2_官方时间对齐与原文保留规则.md)、[结果与代码入口](experiments/listen_official_aligned100_20261026/README.md)、[Cursor转发8876左右布局复核页](http://127.0.0.1:8876/listen_benchmark_review/audio_official_aligned100/)。41条仍有发声标注、59条没有；保留66个音频事件，其中15个采用官方词时间、1个采用单次VAD，余下按交集保留。转写不一致仍保留我们的原文。视频左、标注与音轨右；原始标签、音频和训练缓存未替换。

**预检查修复验收完成，E0 三机 24 卡已启动（2026-10-26）：** [实现、验证及重启说明](experiments/p2_preflight_fix_20261026/README.md)、[实时状态](experiments/p2_preflight_fix_20261026/status.json)。旧 E0 已按用户要求在预检查阶段停止，无训练更新；E1 保留。167 项 CPU 回归＋14 项配对/资源测试通过，三机 6 卡 E0/E1 与原版及断点恢复逐位一致。30,415 段三机 24 rank 首次审计约 10 分 38 秒，再次复用约 1 分 57 秒；数据与类别统计精确匹配 E1。用户已授权停止第三台机器占卡的推理任务；已释放资源并于 11:07 UTC 启动新版 E0，已通过配对核验并进入实际训练更新（启动验收至少完成 15 步）。沿用 seed=20261023、batch=24、1,268 步，实际步数见实时状态。

**官方 listener 区间并集静音，固定100条试听（2026-10-26）：** [新规则](related/P2_官方音频时间并集静音规则.md)、[处理结果与复现](experiments/listen_official_gate100_20261026/README.md)、[Cursor 转发8876试听页](http://127.0.0.1:8876/listen_benchmark_review/audio_official_gate100/)。保留官方词／句／VAD并集，前后各加0.1秒，边界10毫秒淡变，其余静音。同一批100条中52条输出全静音、48条有保留区间；支持原音、处理后和移除部分对比。只生成媒体与候选标签副本，未替换训练数据或缓存。

**GLARE 四项反应评测已接通（2026-10-26）：** [实现、口径与运行命令](related/P2_GLARE四项反应评测实现与运行.md)、[数据接口](examples/p2_glare_eval/README.md)、[工程验收](experiments/p2_glare_eval_validation_20261026/README.md)。支持最终检查点分片生成、独立视频反应检测、R-F1 / R-tIoU / R-ATD / R-FID、多 seed、断点复用和失败审计。独立 CPU 环境，未改训练源码；真实视频与 FID 链路已通过工程验证。**正式评测数据待用户提供，尚未启动 E1 效果评测。**

**随机100条官方 VAD 复核（2026-10-26）：** [固定抽样、统计与复核说明](experiments/listen_official_random100_20261026/README.md)、[Cursor 转发 8876 复核页](http://127.0.0.1:8876/listen_benchmark_review/audio_official_random100/)。从50h训练子集均匀抽取100条新样本，展示双方转写/VAD及现有音频事件，分别反馈VAD内杂音、VAD外漏回应；支持导出。仅复核，未应用门控或淘汰。

**六条标注与官方时间戳对比（2026-10-26）：** [逐样本发现与对齐规则](experiments/listen_official_compare6_20261026/README.md)、[Cursor 转发 8876 对比页](http://127.0.0.1:8876/listen_benchmark_review/audio_official_compare_6/)。对齐我们的音频事件与官方双方句/词/VAD，支持事件试听；未匹配只作差异，不自动视为杂音。

**六条 listener 杂音处理试听（2026-10-26）：** [试验方法、结果与复现](experiments/listen_audio_cleanup_6_20261026/README.md)、[Cursor 转发服务端 8876 试听页](http://127.0.0.1:8876/listen_benchmark_review/audio_cleanup_6/)。4 条有杂音＋2 条干净对照，比较 DeepFilterNet3 两档与视听提取，支持原音／处理后／移除部分切换。仅试听副本，尚未替换训练数据。

**50 小时 E1 训练（2026-10-26，已完成 1,268 步）：** [本轮配置、数据准入、启动与恢复说明](experiments/p2_e1_train50h_lora_tb_20261026/README.md)。按用户确认使用三机 24×H200，global batch=24、1,268 步、seed=20261023，覆盖 30,415 段一轮；AV＋Event＋text-attention LoRA，无 feature decoder / teacher 读取，每 100 步保存、每步 TensorBoard。已实际通过 [第 100 步检查点与 43 项 TensorBoard 指标核验](experiments/p2_e1_train50h_lora_tb_20261026/checkpoint_000100_validation.json)，本轮现已完成，最终完成状态见运行目录 status.json。训练记录见 [全局指标](experiments/p2_e1_train50h_lora_tb_20261026/runs/20261023/E1/metrics.jsonl) 和 [节点状态](experiments/p2_e1_train50h_lora_tb_20261026/node0.json)。

**当前训练版固定备份（2026-10-26）：** [GitHub 固定版本：AV＋Event＋text-attention LoRA＋H200 优化＋TensorBoard](https://github.com/liqilin-eudaemonia/audio-driven-v5-data-processing/tree/p2-av-event-text-lora-h200-tensorboard-20261026/p2_training/av_event_lora_tensorboard_20261026)、[本地备份与恢复说明](../audio-driven-v5-data-processing/p2_training/av_event_lora_tensorboard_20261026/README.md)。包含 P2/v4 复用代码、实际 LTX 包源码、配置、依赖记录和验证证据；副本重跑 146 项 CPU 测试及两进程检查通过。固定标签 `p2-av-event-text-lora-h200-tensorboard-20261026`；旧无 LoRA 标签保留。权重、检查点、媒体和缓存仍在原机器。

**50 小时训练子集随机 HTML 复核（2026-10-26）：** [Cursor 转发服务端 8876 页面](http://127.0.0.1:8876/listen_benchmark_review/train50h_random/)、[抽样说明与检查记录](data_previews/listen_benchmark_review/train50h_random/README.md)。整体随机 50 段＋七类各 5 段，去重后 85 段；保留原文和时间，支持 listener 音视频播放、人脸尺寸查看、事件跳转及分类筛选。

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


**RealTalk 按说话人时段保留原声试验（2026-11-01）：** [6条三列对照与移除原声](http://127.0.0.1:8798/speaker_time_gate/)、[代码与来源](../../realtalk_processing/speaker_time_gate_20261101/README.md)。参考vidchat_sametime_audio.md的等长静音＋区间贴回，正确读取speaker_json而非顶层A；RealTalk时间日志由原混音PyAnnote新生成，不是SpeakerVid官方或3D-Speaker日志。固定旧pair和裁剪，保留区间PCM逐样本不变、保留＋移除精确重构。6条仅保留0.879秒，3条全静音；039已知yeah范围被漏掉，未手工补回。本次未通过效果验证，时间日志仍需改进，未准入训练。

**RealTalk 按指定顺序重新切分试验（2026-10-30）：** [完整镜头与10个新pair审核](http://127.0.0.1:8798/glare_ordered_trial/)、[代码及参数](../../realtalk_processing/glare_ordered_trial_20261030/README.md)。按用户指定原混音→Asteroid→各路降噪→PyAnnote活动→新pair执行，不以人工审核为前置，不复用旧pair边界或联合拼接聚类。3段连续镜头107.16秒，生成10个≥2秒候选共32.12秒，55个视频长度核验通过；全部源帧记账。3段人物通道映射均未达启发式支持条件，一段32秒因活动碎片化没有≥2秒输出；保留完整对照，不把这些未输出数据判为不可用，未准入训练。

**RealTalk GLARE流程纠正（2026-10-30）：** [分离/降噪人工审核](http://127.0.0.1:8798/glare_audio_trial/)。按论文A.3，必须先人工排除严重分离失败，再做PyAnnote活动边界及配对。此前“原混音＋分离音轨拼接聚类”是自定义诊断，已撤出主流程；论文没有公开PyAnnote输入组织。当前8条未完成前置人工验收，不视为严格复现。下方聚类记录仅为历史诊断。

**RealTalk 完整说话人聚类补充（2026-10-30）：** [8条可点击聚类时间线](http://127.0.0.1:8798/glare_audio_trial/)、[实现与限制](../../realtalk_processing/glare_audio_trial_20261030/README.md#full-speaker-diarization-added)。已运行官方speaker-diarization-3.1分段、声纹与聚类；以原对话上下文和两路降噪音轨拼接进行联合分析，时间映射明确保留。4条主speaker簇有交叉支持，listener簇均未独立确认；7处其他声音区间待审核，未自动改边界或静音。该试验适配不是作者GLARE原版，亦不证明音轨干净。

**RealTalk 借鉴GLARE音频步骤试验（2026-10-30）：** [8条三列对照](http://127.0.0.1:8798/glare_audio_trial/)、[方法与限制](../../realtalk_processing/glare_audio_trial_20261030/README.md)。固定v2边界及listener画面；Asteroid DPRNN分离→DeepFilterNet3降噪→PyAnnote segmentation-3.0逐通道活动标注，再以SyncNet交叉评分核对人物。使用本次自选公开模型，非GLARE原版复现，也未实现作者未公开的完整diarization流水线。40个视频长度核验通过，5条speaker通道有支持、0条双方可靠绑定；保留全部匿名通道，人工听辨待完成，未应用全量。

**RealTalk 新抽样50条：响度统一与SyncNet（2026-10-26）：** [两列试听审核](http://127.0.0.1:8798/aec_random50/)、[逐条分数CSV](http://127.0.0.1:8798/aec_random50/scores.csv)、[代码与规则](../../realtalk_processing/aec_random50_20261026/README.md)。新抽取50条≥2秒样本，共330.64秒；默认listener画面＋响度处理后AEC音轨，左列为响度处理前。19条达到−23 LUFS目标、4条受限、27条未检出语音而不提升音量；49条正常评分、1条2秒暂定评分，本批无原SyncNet阈值通过样本，全部保留人工审核。

**RealTalk 成对参考去串音试验（2026-10-26）：** [16组≥2秒样本三列对照](http://127.0.0.1:8798/seamless_aec_trial/)、[代码与规则](../../realtalk_processing/seamless_aec_trial_20261026/README.md)。采用公开DTLN-aec 512，比较原分离、旧门控、参考式AEC及回应保护版本；借鉴Seamless流程，不是Beryl复现。24组全部完成，96个对照视频长度校验与浏览器联动试听检查通过，音质待人工审核。

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

- **下一轮显式规划（首版约束已确认，尚未实施）：** [显式回应规划与双通路生成方案](related/P2_显式回应规划双通路实验方案草案.md)。使用新29,975段后处理50h、512×512、冻结Qwen和既有全英文指导格式；沿用A0/A1/A2对照。验证数据由用户后续提供，训练预算与正式评价协议待细化。
- **已完成四组训练的版本依据：** [四组代码、固定备份、checkpoint与恢复索引](related/P2_四组训练代码备份与恢复索引.md)。下方早期“待正式数据”等内容为历史计划，不替代四组最终run记录。

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

**当前论文写作方案（2026-11-05）：** 采用“倾听阶段视听反馈的任务定义 → 数据构建 → 专门评测 → 生成方法 → 实验分析”主线，将数据与 benchmark 列为核心贡献候选，记录方法职责、目标贡献、摘要草案及待验证的证据边界。见 [倾听阶段视听反馈生成论文写作方案](writing/倾听阶段视听反馈生成论文写作方案.md)；参考 [GLARE 与 OmniPersona 写法分析](writing/GLARE与OmniPersona论文写作分析.md) 和 [倾听与对话生成论文叙事分析](writing/倾听与对话生成论文写作叙事套路与技巧.md)。

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

**RealTalk 后续最短时长规则（用户确认）：** 不足2秒不再继续处理；正好2秒保留。已生成[后续处理清单与统计](../../realtalk_processing/full_v2_20261025/reports/downstream_min2s/summary.json)。
# Event＋Feature＋Whisper：8000步停训后自动评测（2026-11-06）

当前31卡训练按用户要求在第8000步完整保存后停训，随后自动评测本轮6000／8000两个权重：固定Seamless200＋ResponseNet分层随机200，共800条生成，历史采样与结果沿用。复用条件、speaker Whisper、GT解码及特征缓存；node8 GPU7继续预留，巡检goal保持暂停。[接续配置、验证、状态与结果入口](experiments/p2_combined_whisper_eval_20261106/README.md)。

