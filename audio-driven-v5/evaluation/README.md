**当前重要权重已统一归档说明（2026-11-13）：** 主候选Qwen7500＋LTX6800、四组256条双seed组合、Qwen保留adapter、完整LTX候选、六份消融晋级权重及GLARE两条训练线均列出实际路径与评测状态；注明step2000排队、完整test代码尚未启动及恢复入口。[权重位置与评测交接](../related/P2_当前重要权重与评测入口汇总_20261113.md)。

**GLARE早期step2000已接入后续256条队列（2026-11-13）：** 采用早期batch8训练的best.pt，内部step2000及SHA已固定；相同Seamless256／seed137／NFE10／CFG2，输入与官方韵律缓存复用，256条预测重新生成。等待当前6组消融256条推理评分完成并释放控制器后，自动使用node12＋node11共16卡，node8不参与；6项CPU检查通过，完成自动更新同256条结果页。[运行、结果与恢复](../experiments/glare2000_test256_after_ablation_20261113/README.md) · [接续状态](../experiments/glare2000_test256_after_ablation_20261113/QUEUE_STATE.json)。

**消融6候选已启动固定256条测评（2026-11-13）：** 用户明确先与完整模型比较，暂不要求高于GLARE。相同64条下，去Event6500／6573、去Feature5500／6643、去Whisper5500／6643均至少6/10项数值较差，视觉／反应至少5/8项较差；去Event6000不满足，已替换。仅node12＋node11共16卡；共1536条，复用384条、新增1152条，固定Qwen7500／seed137及原采样评分配置。全部12份结果保留；此为按test子集成绩筛选的探索性比较，不能称各消融最优，256条排序可能变化。[筛选依据、运行与恢复](../experiments/p2_ablation6_test256_qwen7500_16gpu_20261113/README.md) · [实时状态](../experiments/p2_ablation6_test256_qwen7500_16gpu_20261113/status.json)。

**Qwen7500＋LTX6800完整双seed三机代码已准备（未启动GPU，2026-11-13）：** 最新指令先完成全部推理与导出，顺序Seamless6624 seed137→238，再ResponseNet完整586 seed137→238；不自动评分、不做双seed汇总。三机24卡，24路缓存／22路生成＋2路常驻解码，复用512条同身份结果，新增13908条；7210条Whisper与输入缓存核验、7项CPU测试通过。启动前要求24卡释放，评分入口单独保留。[代码、输出位置、启动与恢复](../experiments/p2_qwen7500_ltx6800_full_dualseed_24gpu_20261113/README.md)。

**后续GLARE3935改用两机16卡并填充空卡（2026-11-13）：** 按用户最新要求暂不使用node8；node12/node11各0–7卡动态领取全部24个逻辑分片，不留分片等待node8。消融768条视频已全部生成，GLARE可在剩余评分之外的空卡接续，自动避开仍在评分的卡；该评分完成后两机16卡全部可用。样本／seed／采样／评分不变，完成后自动更新结果页。[运行与恢复](../experiments/glare3935_test256_after_ablation_20261113/README.md)。


**消融测评动态补卡已启用（2026-11-13）：** 修复每份权重固定最多2路、导致空卡不能接剩余慢组的问题；独立`accelerate.py`按剩余组数动态分配生成worker，优先node12/node11，node8现有额外占用按剩余显存准入。切换保留1315份完成收据且SHA不变，权重／输入／seed／采样／评分及冻结原worker代码不改，GLARE3935后续队列仍有效。[迁移、运行与恢复](../experiments/p2_ablation12_test64_qwen7500_24gpu_20261113/README.md#2026-11-13动态调度修复)。


**GLARE早期batch8 step3935已接入后续队列（2026-11-13）：** 等当前三组消融12权重×64条的推理和评分全部完成、控制器退出后，自动用node8/node12/node11三机24卡运行相同Seamless固定256条、seed137。原权重inode与SHA已固定，源媒体／官方韵律缓存复用；原生NFE10、CFG2、解码batch4及评分加速配置保持。5项CPU检查通过，尚未启动本轮GPU；结果完成后自动加入同seed结果表，明确标注早期batch8独立训练线。[队列、结果与恢复](../experiments/glare3935_test256_after_ablation_20261113/README.md) · [接续状态](../experiments/glare3935_test256_after_ablation_20261113/QUEUE_STATE.json)。


**固定256条联合模型／GLARE对比已加入结果页（2026-11-13）：** seed137包含四组Qwen/LTX及GLARE10618／19050；seed238和双seed均值单独排名，均值表不混入只有一个seed的GLARE。所有数值来自已完成原始JSON，FID/FVD按完整256条特征汇总，保留检测有效覆盖与音频N/A；第一名加粗、第二名下划线。[直接打开同seed对比](http://localhost:8876/listen_benchmark_review/vico_tables_20261112/#seamless256-seed137) · [双seed均值](http://localhost:8876/listen_benchmark_review/vico_tables_20261112/#seamless256-mean)。


**三组全LoRA消融12权重快速测评已挂载（2026-11-13）：** 去Event 5500／6000／6500／6573，去Feature与去Whisper各5500／6000／6500／6643，统一Qwen7500、原test64、seed137，共768条。授权node8／node12／node11 GPU0–7，按剩余显存交错生成／解码／评分，保留其他任务；实际进度以状态及加载收据为准。复用64条条件缓存，5项CPU检查通过，全部结果保留；GLARE对照及256条晋级待用户确认。[运行、结果与恢复](../experiments/p2_ablation12_test64_qwen7500_24gpu_20261113/README.md) · [实时状态](../experiments/p2_ablation12_test64_qwen7500_24gpu_20261113/status.json)。


**GLARE同256条三机测评已接续（2026-11-13）：** 按用户指定先step19050，再step10618，沿用相同固定256及当前GT；分别复用248／89条有效旧生成，补8／167条，全部重新评分、全256特征重算FID/FVD。node8、node12、node11参与，node11利用剩余显存，原不可见占用未清空。GLARE seed137，音频指标N/A；完成情况见运行状态。[运行、结果与恢复](../experiments/glare_test256_3nodes_20261113/README.md)。

**四组256条双seed接续已启动（2026-11-13）：** seed137于北京时间17:50:45完成；seed238于17:52:57自动启动，同四组、同256条与完全相同条件，仅改变LTX采样seed，1024条重新生成和评分。仅node8＋node12共16卡。第二轮完成自动汇总各指标双seed均值、样本标准差及原值，FID／FVD逐seed全局计算后再平均，N/A不补零。[运行与最终平均表入口](../experiments/p2_joint_qwen_ltx_test256_two_seeds_20261113/README.md) · [接续状态](../experiments/p2_joint_qwen_ltx_test256_two_seeds_20261113/status.json)。

**四组Qwen×LTX固定256条已启动（2026-11-13）：** 8256＋6800、7500＋6800、2752＋6000及原生Qwen＋6800，仅node8／node12共16卡。原64条全部保留，复用256条生成与16个评分分片，新增768条生成；仅补提576份Qwen条件。生成上限14并发，解码／评分动态接续，最后按全部256条特征重算FID／FVD。node11保持暂停。[运行、结果与恢复](../experiments/p2_joint_qwen_ltx_test256_16gpu_20261113/README.md) · [实时状态](../experiments/p2_joint_qwen_ltx_test256_16gpu_20261113/status.json)。

**Qwen×LTX两机16卡联合测评已完成（2026-11-13 北京时间16:30:56）：** 2752／7500／8256 × 五份LTX，加原生Qwen×6800控制组，共16组×固定test64；复用320条、新增704条，全部生成、评分与汇总完成，本轮工作进程退出。图像／分布和音频均覆盖64条；部分表情／ΔResp按实际有效覆盖，7500有1条规划截断按真实输出保留。[结果表](../experiments/p2_joint_qwen_ltx_test64_16gpu_20261113/RESULTS.md) · [运行与恢复](../experiments/p2_joint_qwen_ltx_test64_16gpu_20261113/README.md)。

**node11本项目测评已全部退出（北京时间2026-11-13 14:57:18）：** 原Qwen/LTX比较已结束；GPU7参考方法11个worker、Monitor/controller及ResponseNet接续守护均已停止，自动恢复关闭，产物保留。Qwen×LTX联合筛选现已在node8＋node12共16卡启动。[退出、断点与恢复记录](../experiments/p2_joint_qwen_ltx_test64_16gpu_20261113/NODE11_PAUSE_README.md)。

**Qwen追加两轮的规划文字对照已完成（2026-11-13）：** 原生＋旧6份＋7500／8000／8256，同固定200，共2000条（新增600），均正常EOS、零重试。8000/8256有92条逐字相同，平均逐词相似度83.4%；2752/8256仅5条相同，50.5%。26条文字复核发现改善和退步并存，建议暂不为找回中期权重重训，优先2752与8256接LTX比较；非全200语义准确率排名。node11任务已按要求停止，本机完成收尾；后续仅node8＋node12。[对照页](http://localhost:8876/listen_benchmark_review/planning_ckpt_extended_seamless200_20261113/) · [复核报告](../experiments/p2_planning_ckpt_extended_seamless200_20261113/REVIEW.md)。

**下一轮LTX候选已选5份（2026-11-13）：** 6000、6300、6500、6700、6800；6000／6500／6800为主要候选，6700补充ΔResp最优，6300补充FID第二。依据固定test64＋SFT2752的互补指标表现选择，非最终test严格前五；原权重全部保留，下一轮未启动。[候选清单与权重路径](../experiments/p2_ltx15_test64_sft2752_20261113/CANDIDATES_5.json) · [结果表](../experiments/p2_ltx15_test64_sft2752_20261113/RESULTS.md)。

**15权重test64初筛已完成（2026-11-13 北京时间14:11）：** 实际三机21卡完成960条生成与15份汇总，固定SFT2752；6800在PSNR／SSIM／LPIPS／表情准确率领先，6000在FID／FVD领先。像素／FID／FVD和音频均覆盖64条；同一条样本部分表情／ΔResp不可测，UTMOS因发声过短N/A。按预先冻结的保守规则无自动淘汰项。[全部结果](../experiments/p2_ltx15_test64_sft2752_20261113/RESULTS.md) · [最终验收](../experiments/p2_ltx15_test64_sft2752_20261113/cluster29/FINAL_AUDIT.json)。

**15份续训LTX已接GPU4初筛（2026-11-13）：** 用户要求暂停GPU4的GLARE10618（保留2223条生成及已有评分），先测同一次28卡续训5600～7000、每100步共15份。固定新test64＋SFT2752、seed137／Euler30／CFG3、7，共960条；64条规划／hidden／Gemma／Whisper缓存全复用，常驻LTX切权重、旁路解码评分。15份step／Adam／身份／源码核对通过；5600首条真实latent已产出并核验权重／SFT／seed，解码自动接续，当前GPU4利用率98%。其余权重完整SHA与生成流水进行。仅淘汰明显综合落后候选，接近或存在取舍的保留，不删除原权重。[运行、结果与恢复](../experiments/p2_ltx15_test64_sft2752_20261113/README.md) · [实时状态](../experiments/p2_ltx15_test64_sft2752_20261113/status.json)。

**Seamless test快速筛权重64条已固定（2026-11-13）：** 按用户要求直接从当前完整test6624抽取；naturalistic49／improvised15，listener处理后音轨非零19／全零45，<4秒22／4–8秒23／≥8秒19。覆盖64条原视频、56个来源组，平均6.039秒（全集6.004秒），不读取模型预测或评分。用于Qwen/LTX权重选择，明确为test调参子集，不能称独立dev；当前未启动推理。[清单、比例与接入](../experiments/seamless_test_tuning64_20261113/README.md) · [机器入口](SEAMLESS_TUNING64.json) · [逐条起止时间CSV](../experiments/seamless_test_tuning64_20261113/selection.csv)。

**七组Qwen固定Seamless诊断200已完成（2026-11-13）：** 仅node11 GPU7；原模型＋SFT500/1000/1500/2000/2500/2752，共1400条规划。使用当前转正200条、真实speaker音频和listener首帧，同原生推理参数；现有自动标注只用于并列对照。GPU4的GLARE继续。1400/1400均非空、正常EOS、零重试；对照页已生成。四条文字抽查发现漏笑声/短回应及额外点头，尚无全量语义准确率排名。[逐条对照页](http://localhost:8876/listen_benchmark_review/planning_ckpt_seamless200_20261113/)。[运行、结果与恢复](../experiments/p2_planning_ckpt_seamless200_20261113/README.md)。

**GLARE10618已接上当前Seamless完整6624测试（2026-11-13）：** 用户指定 `before_extension20epochs.pt`，内部实际10618步；仅node11 GPU4，4个原生常驻生成worker／decode4，生成期间2路评分、结束后8路，batch64／GPU检脸和情绪／像素4线程。6624条音轨与官方韵律缓存已核验复用；所有预测独立生成。使用当前转正来源，其中33条与历史GLARE全量来源不同，比较时保留版本说明。原LoRA6000 Seamless与LoRA6500 ResponseNet队列已暂停、产物保留，本任务不占GPU7。已核验实际10618加载、视频产出及评分接续，完整成绩尚未完成。[实时状态](../experiments/glare10618_seamless_test6624_20261113/status.json) · [运行配置、输出与恢复](../experiments/glare10618_seamless_test6624_20261113/RUN_INFO.json) · [现场验收](../experiments/glare10618_seamless_test6624_20261113/LIVE_ACCEPTANCE.json)。

**已加入ResponseNet完整test586后续评测（2026-11-12）：** 同五个参考方法与L2L四套权重，在Seamless完整6624结束后自动接续，仍仅GPU7（2026-11-13最新资源安排）空闲资源；复用历史200条，新增386条。[完整586进度/结果](../experiments/p2_reference_methods_responsenet586_20261112/RESULTS.md) · [后续队列与资源规则](../experiments/p2_reference_methods_responsenet586_20261112/README.md)。

**五个参考方法扩至Seamless完整test6624（2026-11-12）：** L2L四套权重、ListenFormer、DyStream、Avatar Forcing、DualTalk沿用双200已验收链路。双200已完成并独立保留；全量只用node11 GPU7，GPU4禁止。2026-11-13改为模型常驻，取消小批次退出和定时重载，评分模型跨分片复用；前台任务占卡时暂停计算，随后恢复同一进程。已完成缓存继续复用，完整ResponseNet586在本轮完成后接续。[完整6624结果与进度](../experiments/p2_reference_methods_seamless6624_20261112/RESULTS.md) · [范围、模型与Monitor说明](../experiments/p2_reference_methods_seamless6624_20261112/README.md)。

**GPU4／GPU7九小时巡检已启动（2026-11-12）：** 用户要求每小时复核并修复退出／明显降速，固定至11月12日23:52 UTC／北京时间11月13日07:52。node11两卡初查均100%；GPU4继续Seamless全LoRA6000完整6624，GPU7优先完成ResponseNet6000／6500完整586。后台每分钟核验精确进程、每小时保留吞吐证据并在退出后有界恢复；GPU7两权重均验收结束且整卡释放后，按共享样本锁分担Seamless生成，避免重复计算。[巡检、证据与恢复](../experiments/p2_gpu4_gpu7_watch9h_20261112/README.md)。

**GPU4 Seamless LoRA6000全量队列已断点恢复（2026-11-12）：** 原退出原因为Gemma batch16触发48GiB进程配额，整卡并未耗尽；现Gemma batch8／缓存64GiB，已完成Whisper、Qwen、196条旧生成和4条新增latent全部复用，模型／SFT／采样保持。[故障与恢复](../experiments/p2_lora6000_seamless_full_gpu4_20261112/RECOVERY_20261112.json) · [状态](../experiments/p2_lora6000_seamless_full_gpu4_20261112/status.json)。

**LoRA6000＋SFT2752已启动Seamless完整test6624（2026-11-12）：** 按用户要求仅node11 GPU4，采用总索引当前upright版本，33条来源相对历史全集更新。正确原权重SHA3c2c78e8…、SFT2752、seed137／Euler30／CFG3、7；复用196条来源不变的生成、200条已核验条件，补齐6428条。CPU输入准备完成，模型1632处LoRA严格加载通过，GPU4已实跑；缓存→生成→解码→视频／音频评分及全局特征合并自动接续，GPU7的ResponseNet队列不动。尚未完成，不填全量成绩。[运行、进度、结果与恢复](../experiments/p2_lora6000_seamless_full_gpu4_20261112/README.md)。

**GLARE已加入总结果页（2026-11-12）：** Seamless固定200新增GLARE19050／32940，现22组／9组对照／41项差值；新增全LoRA6000、6428及原完整6000→GLARE19050、GLARE19050→32940。逐ID核验200条GT RGB、帧数、裁框与倾听区间一致；声音指标N/A，不记0、不参与音频排名，CSV同步。总页新增ViCo43、ResponseNet完整586、Seamless固定200及完整test6624独立汇总，第一名加粗、第二名下划线；ResponseNet随机200保持18组，完整586不混入随机200。19050全集仍在全局汇总／验收，32940全集按用户要求暂停，未完成不填成绩；完成后渲染入口自动同步总页表格。994项数值／差值、浏览器、移动端、CSV及原视频音轨检查通过。[GLARE方法对照](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=glare#ablation-glare) · [各测试范围汇总](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless#checkpoint-results)。

**GLARE最新结果已同步到指定汇总页（2026-11-12）：** [ViCo／ResponseNet／Seamless结果表](http://localhost:8876/listen_benchmark_review/vico_tables_20261112/)。19050步Seamless完整6624条的生成、209个评分块、全局FID／FVD及最终指纹验收已完成；图像／分布有效6619条、表情6576条、ΔResp6569段，原素材缺帧及检测支持不足保留N/A，完整覆盖见表格与[原始结果](../experiments/glare19050_seamless_test6624_20261112/test6624_RESULTS.json)。每个测试范围单独一表，第一名加粗、第二名下划线；32940步全集仍暂停。

**旧联合8000 vs 全LoRA6500：固定Seamless200双seed已全部完成（2026-11-12）：** 同SFT2752、seed137／238，共800条与四轮验收通过，耗时约43分钟。全LoRA均值FID14.677403／FVD173.409345，旧联合14.965719／179.976908；全LoRA画质／分布更好，旧联合情绪准确率与ΔResp更好，UTMOS按原支持规则为N/A。仅使用四机28卡，node11为0–3。[每seed原值、双seed均值与完整结果](../experiments/p2_seamless200_pipeline_two_seeds_20261112/RESULTS.md) · [运行与验收](../experiments/p2_seamless200_pipeline_two_seeds_20261112/README.md)。

**检测流程已按数据标注对齐（2026-11-12）：** 复用生产动作／表情门控及ASR规范化，补齐抬眉辅助线索和眉部观察用语，移除评测额外的100%人脸覆盖硬改标签。仍输出七类整段状态；不另调阈值，不以补齐人评作为运行门槛。[实现与验证](../experiments/event_annotation_alignment_20261112/README.md)。


**人工核验结果已录入（2026-11-12）：** 核验者01提交56项；按媒体及分模态波形身份确认45项可沿用到转正版本，余11项留空。新版页面已预填，可直接补视觉8项、音频3项。[原始标签、迁移依据与检测分歧](../experiments/event_presence_human_review_20261112/README.md)。


**Seamless评测转正来源已迁移（2026-11-12）：** 固定200条中4条、核心48中1条、完整6624中33条接入已发布upright来源，样本与时间保持。旧清单和旧分数留作历史；后续评测使用[当前机器入口](SEAMLESS_CURRENT.json)。[修复、增量生成与核验页状态](../experiments/seamless_upright_eval_fix_20261112/README.md)。


**事件人工核验页面已支持点选与导出（2026-11-12）：** [视觉四类核验](http://localhost:8876/listen_benchmark_review/event_presence_pilot_20261108/review_visual.html)支持逐段点选、浏览器自动保存、旋转观看和 CSV 导出；声音三类页面同步更新。当前 8 段、视觉 32 项／音频 24 项，尚未完成真实人工核验。[操作与验证记录](../experiments/event_review_ui_20261112/README.md)。


**五个参考方法的历史双200评测（2026-11-12，执行中）：** 用户指定本轮只测固定ResponseNet200＋Seamless200。L2L四套人物权重、ListenFormer、DyStream、Avatar Forcing、DualTalk沿用已接通的推理链路；DyStream复核历史结果后复用，DualTalk原生动作与视频指标分开列。仅node11 GPU7，按样本就绪接续生成与评分。[数值对比表](../experiments/p2_reference_methods_dualtest_20261112/RESULTS.md) · [范围、模型与恢复说明](../experiments/p2_reference_methods_dualtest_20261112/README.md)。

**参考方法真实训练接入测试（2026-11-09）：** 七个方法已用最终500h选集中的真实片段执行完整原生网络更新与断点重放；新进程恢复、日志和权重见[训练验收](../experiments/baseline_training_realdata_20261109/README.md)。当前为链路短跑，尚未完成500h全量训练，也未据此新增论文指标或LM Listener评分。

**全LoRA6000／6428已加入总结果页（2026-11-09）：** Seamless固定200现20组／8组消融／37项对照，ResponseNet仍18组／7组／30项（全LoRA两组尚未评测）。新增原范围完整6000→全LoRA6000／6428、6000→6428和AV／Event／Feature／Whisper基线对照，B−A箭头、原值、排名前两名配色、音频覆盖及CSV同步；注明时长分组512→496、非严格单变量。旧Seamless视频页指标总览同步20组，原有14组视频选择与音轨保持。938项数值／差值及双页浏览器、移动端、CSV、播放检查通过。[打开全LoRA对照](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=all_lora#ablation-all_lora)。

**全LoRA完整组6000已四机32卡实际评测（2026-11-09）：** 原训练6428完整保存、用户释放资源后自动接续；四机32个原生模型严格加载与每卡1632处LoRA目标全部通过，已有生成保留。固定Seamless200、同SFT2752与seed137，代码按卡／就绪分片接解码及视频／音频评分，原vLLM保留。[状态、权重及实际启动验收](../experiments/p2_ltx_all_lora6000_seamless200_sft2752_eval32_20261109/README.md)。

**全attention／FFN LoRA完整组6000已准备Seamless200评测（2026-11-09）：** 正确原权重SHA3c2c78e8…，含Event；固定同200条、SFT2752、seed137、Euler30／CFG3、7。完整模型／Adam／31RNG及逐张量推理导出核对、4项回归通过；配置支持四机32卡共享生成和就绪分片流水评分。资源准备期间其余31卡被从6208恢复的完整组训练占用，目前仅node11 GPU0实际推理，已完成真实产出检查；32卡扩容等待训练暂停选择，不停止其他任务。[状态、正确权重与入口](../experiments/p2_ltx_all_lora6000_seamless200_sft2752_eval32_20261109/README.md)。

**Seamless两组6000新seed复测已启动（2026-11-08）：** node11仅GPU0，A＝当前500h AV＋Whisper6000＋SFT2752，B＝主干Event＋Feature＋Whisper6000＋SFT2752；固定同200条，改生成seed137→238，规划文本／H／Gemma／Whisper缓存及Euler30／CFG3、7保持。独立输出400条，旧结果保留；A真实GPU严格加载及首条seed238有限latent已通过，B随后自动生成、解码、评分。沿用4路解码与8路CPU预取评分／GT缓存；不占其他GPU、不改其他训练。[配置、状态及输出](../experiments/p2_seamless_whisper_joint_seed238_node11gpu0_20261108/README.md)。

**当前Feature／Whisper6000评测完成并更新总结果页（2026-11-08）：** 两组均使用最新SFT2752，双200共800条生成／解码／视频与音频评分和最终验收完成；四机接续耗时约23分钟。总页现为两测试集各18组、30项对照；新增当前AV→Feature、AV→Whisper、Feature→Whisper及两组分别→联合6000，原值、B−A箭头、第一／第二名颜色、音频覆盖与CSV同步。840项数值／差值及浏览器验收通过。[打开总结果页](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-joint_whisper)。

**四机评测改为按卡／分片流水推进（2026-11-08）：** 去除两模型全部生成后才解码评分的等待：每卡完成生成即可解码，分片媒体齐全即可在已释放GPU评分，节点资源／存活异步探测，node11慢任务不阻塞其他节点。各模型可独立汇总，最终仍要求固定双200完整并用原生全量bind逐行核对；指标公式及已完成输出保留。11项CPU回归通过，GPU阶段以实时状态为准。[配置、代码及状态](../experiments/p2_current_feature_whisper6000_eval32_20261108/README.md#2026-11-08去除整机全量阶段等待)。

**Feature6000＋Whisper6000四机32卡评测接续已设置（2026-11-08）：** Whisper6000完整保存后安全暂停该31卡训练，将释放卡转评测；保留Feature已有产出及GPU7当前样本，跨卡共享领取、每卡完成Feature后可直接转Whisper。相同SFT2752与双200条件，沿用4路解码、32评分分片／模型、CPU预取及GT缓存；Feature推理权重导出逐值一致，CPU与四机共享文件检查、4项回归通过。6000已完整保存且原训练退出，四机协调器已接续；GPU模型验收与实际进度以运行状态为准。[自动顺序、配置、代码与状态](../experiments/p2_current_feature_whisper6000_eval32_20261108/README.md)。

**当前500h AV＋Feature6000＋SFT2752已接续评测（2026-11-08）：** 固定Seamless200＋ResponseNet200，直接共享纯AV与联合组的400条SFT2752条件；无Event／Whisper，Feature权重双路各0.025。CPU完整权重／源码检查、同训练与推理条件核对及2项测试通过；仅node11 GPU7按显存余量运行，后接4路解码、8路CPU预取评分和全量汇总。[配置、代码和状态入口](../experiments/p2_current500h_feature6000_sft2752_dual200_20261108/README.md)。

**当前500h AV-only6000＋SFT2752双200评测已完成并加入总结果页（2026-11-08）：** 400条最终验收通过；ResponseNet／Seamless各16组、25项差值对照，新增“相同SFT2752条件：当前仅AV→联合6000”，历史AV基线独立保留。总体指标、音频覆盖、第一／第二名配色、B−A箭头及CSV同步更新；700项数值／差值和浏览器验收通过。[总结果页](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-joint_whisper)。

**当前AV-only6000正式评分已叠加启动（2026-11-08）：** node11仅GPU7，用户确认叠加后已替换闲置协调器，最多8个流水线评分worker；原28.3GB任务保持运行，400条生成／解码与预检保留，按显存余量准入并预留16GiB。[当前状态与执行说明](../experiments/p2_current500h_av6000_sft2752_dual200_20261107/overlay_scoring/README.md)。

**当前500h AV-only6000＋最新SFT2752已排队（2026-11-07）：** 固定Seamless200＋ResponseNet分层随机200，直接复用联合6000／6500的400条SFT规划／H／Gemma条件；LTX为纯AV，无Whisper分支。node11仅GPU7，连续两次确认无compute进程后开跑，不叠加或停止其他任务。CPU权重／源码／条件检查和3项测试通过，真实GPU验收待空闲；采用4路解码、8评分分片及CPU预取流水线。[排队、配置、代码与结果入口](../experiments/p2_current500h_av6000_sft2752_dual200_20261107/README.md)。

**评测CPU读取／像素计算流水化已验证（2026-11-07）：** 后台有界预取＋常驻4线程计算原生PSNR／SSIM，与当前样本GPU评分重叠；同4条Seamless总耗时59.20→35.65秒、ResponseNet78.03→55.45秒，逐条指标、FID/FVD特征与汇总完全一致，GT缓存保持；7项测试通过。已提供后续队列入口，未重算已完成6500正式结果。[代码、配置、实测与复用说明](../related/P2_评测CPU预取与像素计算流水化提速_20261107.md)。

**6500评分已增至GPU7同卡8路（2026-11-07）：** 解码400条完成，评分4→8路直接接入剩余分片，原有分片不重启；104秒新增64条视频评分，约0.616条/秒，预热后15次GPU采样平均52.2%。指标算法、batch64、GT缓存及全200特征汇总保持，仅node11 GPU7。[配置与实测](../experiments/p2_ltx6500_latest_sft_dual200_20261107/card7/metrics8/README.md)。

**6500评测单卡解码提速已实测（2026-11-07）：** 仅node11 GPU7，原1进程＋3辅助进程交错VAE解码与CPU编码；83秒完成59条，生产吞吐约为此前单进程的3.71倍，15次GPU采样平均80.4%。原生数值／codec／逐样本锁及已完成输出保持，评分仍由原队列接续；不同样本窗口，仅作生产提速参考。[实测与入口](../experiments/p2_ltx6500_latest_sft_dual200_20261107/card7/decode_assist/README.md)。

**最新step6000＋SFT2752已加入原测评页面（2026-11-07）：** ResponseNet／Seamless各15组，7组消融、24项对照；新增主干Event＋Feature＋Whisper6000＋SFT2752，与历史联合6000、仅AV6000及AV＋Event6000对比。原值、B−A、指标箭头、前两名配色和音频覆盖均保留；注明数据／Event读出／规划器多项变化，不能归因于单一组件。672项数值／差值及浏览器验收通过。[打开新增对照](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper#ablation-whisper)。

**step6500已按最新要求改为仅node11 GPU7（2026-11-07）：** 保留已完成163条latent，剩余生成、解码、评分只使用物理7号卡；0–6卡交还其他任务。相同双200清单、最新SFT2752条件及采样协议不变。step6000组400条已完成且最终验收通过；未恢复聊天监控或Goal。[GPU7接续说明](../experiments/p2_ltx6500_latest_sft_dual200_20261107/card7/README.md) · [当前状态](../experiments/p2_ltx6500_latest_sft_dual200_20261107/status.json)。

**历史排队配置：step6500已接在step6000后评测（2026-11-07）：** 使用三机24卡续训目录 `p2_ltx_clip_train24_reserve_node11_20261107/step_006500.pt`，相同LTX主干整段Event＋Feature＋Whisper、最新SFT2752及双200清单；直接共享6000组400条H／Gemma／speaker条件，seed和采样不变。权重导出逐值、原生来源与条件复用检查通过；node11八卡等待6000组评分与验收完成后自动接续。[配置、代码及结果入口](../experiments/p2_ltx6500_latest_sft_dual200_20261107/README.md) · [队列状态](../experiments/p2_ltx6500_latest_sft_dual200_20261107/status.json)。 当前资源安排以上方GPU7接续说明为准。

**LTX主干整段Event＋Feature＋Whisper6000＋最新SFT2752已完成（2026-11-07）：** 固定双200共400条生成与视频／音频评分完成，最终验收通过。[结果](../experiments/p2_ltx6000_latest_sft_dual200_20261107/RESULTS.md) · [验收](../experiments/p2_ltx6000_latest_sft_dual200_20261107/FINAL_AUDIT.json)。

**6309步双200评测已扩至node11八卡（2026-11-07）：** GPU0–6新增7个worker，GPU7原renderer持续运行；同一共享样本队列、400条固定输入与采样参数不变，已完成结果保留。新controller统一等待render完整后接8卡解码与原生评分，训练不恢复。[八卡接管、代码与状态](../experiments/p2_event_whisper6309_dual200_node11_20261107/expansion8/README.md) · [实时进度](../experiments/p2_event_whisper6309_dual200_node11_20261107/status.json) · [八卡真实推理与结果保留验收](../experiments/p2_event_whisper6309_dual200_node11_20261107/expansion8/LIVE_ACCEPTANCE.json)。

**L2L复现依据重新核对（2026-11-07）：** 当前全部L2L训练已停止。原论文补充与公开代码在warmup、VQ隐藏维度和优化器上存在差异；官方README明确VQ loss可能先升后降，不能据此断定训练故障。短预热和码本初始化均只保留为诊断，不计作复现。[逐项证据、纠正与执行顺序](../experiments/l2l_literature_protocol_audit_20261106/PAPER_GROUNDED_REPRODUCTION_AUDIT_20261107.md)。

**Event＋Whisper 6309步双200评测已启动（2026-11-07）：** 仅node11（172.29.189.119）GPU7，固定ResponseNet分层随机200＋Seamless固定200；复用原始Qwen/Gemma和speaker Whisper条件、GT缓存及历史seed137／Euler30／CFG3、7。推理后自动解码、视频／音频评分和全200特征汇总，训练不恢复。用户允许同卡叠加，不停止其他任务。[运行、输出与协议](../experiments/p2_event_whisper6309_dual200_node11_20261107/README.md) · [当前状态](../experiments/p2_event_whisper6309_dual200_node11_20261107/status.json) · [GPU7实际推理验收](../experiments/p2_event_whisper6309_dual200_node11_20261107/STARTUP_ACCEPTANCE.json)。

**L2L可靠重训已重新授权（2026-11-07）：** 用户允许临时使用node11 GPU7并与现有推理叠加，主种子137的VQ因量化项不稳定已在27749步保存并停止；3000步码本诊断已结束，另一个短预热predictor诊断已在569步停止；当前全部L2L训练停止，先核对论文协议。新轮已通过官方更新／断点恢复／小样本学习预检，恢复官方预热；通过验证学习检查后才进入test43；旧两轮第1步最优结果仍不算有效基线。[当前方案、进度和验收入口](../experiments/p2_l2l_reliable_retrain_20261107/README.md)。

**参考方法测评当前入口（2026-11-06）：** 统一实测与展示已完成；L2L本地重训按用户要求在第二轮VQ20000步后暂停，predictor/test43未启动，最优权重仍为第1步，不作为有效基线。[完整进度、已确认口径与后续方案](../related/P2_参考方法统一测评与L2L重训暂停交接_20261106.md) · [汇总结果及全部视频](../experiments/common_method_comparison_20261106/README.md) · [机器可读状态](../experiments/common_method_comparison_20261106/HANDOFF_STATE.json)。下方早期启动/完成信息为历史记录，本轮状态以交接为准。

**Seamless固定200条 DyStream执行完成（2026-11-06）：** 200/200视频、30,100评分帧及全局FID/FVD完成，独立验收通过；仅node11 GPU6，耗时28分01秒。画质五项覆盖200条，情绪／ΔResp覆盖199条；同一历史支持不足样本保留N/A，状态partial。PSNR15.4589、SSIM0.66241、FID23.2626、FVD243.2780。[结果与覆盖说明](../experiments/p2_compare_seamless200_20261106/README.md)。

**Seamless固定200条 DyStream已启动（2026-11-06）：** 固定19.99分钟清单，仅node11 GPU6，12路生成后接8路原Seamless视频评分；使用同一最终裁框及无损GT缓存，listener自身音频置零。[当前状态与协议](../experiments/p2_compare_seamless200_20261106/README.md)。Learning2Listen仍暂缓。

**ResponseNet DyStream固定200条已完成（2026-11-06）：** 200/200视频、44,046评分帧与全部视频指标完成，执行错误0；node11仅0–6卡补齐61条并评分，保留139条原输出，接续6分34秒，卡已释放；7卡未使用。PSNR16.4107、SSIM0.64069、FID17.5310、FVD184.4808。[结果与独立验收](../experiments/p2_compare_responsenet_random200_20261106/README.md)。Learning2Listen仍暂缓。

**ResponseNet DyStream续跑（2026-11-06）：** 已按用户要求转至node11的0–6卡（7卡保留），保留139条有效视频，28进程续跑剩余61条，自动接7卡ResponseNet视频评分。[当前状态与协议](../experiments/p2_compare_responsenet_random200_20261106/README.md)。

**ResponseNet随机200条 DyStream已挂接（2026-11-06）：** 按用户最新要求，仅node8物理GPU7跑200条视频及既有ResponseNet视频评分；Learning2Listen暂缓。生成已提至12路，实测约31.2原生帧/秒，显存113GB；生成后自动四路评分、全200条FID/FVD汇总。进度与协议见[本轮记录](../experiments/p2_compare_responsenet_random200_20261106/README.md)，完成以最终审计标记为准。

**ViCo三种方法统一展示（2026-11-06）：** [联合模型8000／DyStream／Learning2Listen评分与同片段对照](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/)。speaker、GT、联合模型、DyStream四路同步播放，两个生成方法一起切换512／256和音轨；L2L四套动作评分与曲线同页展示，缺写实视频明确标注。43组两种分辨率、152项汇总数值／CSV与手机布局检查通过。[统一结果说明](../experiments/p2_compare_vico43_20261106/three_method_review/RESULTS.md)。

**compare参考工作ViCo实测已完成可运行部分（2026-11-06）：** DyStream43条视频、Learning2Listen四人物各43条动作（共172条）及评分／独立验收通过；其余六项缺核心生成权重或代码，未计为完成。[全部视频与动作曲线](http://localhost:8876/listen_benchmark_review/p2_vico43_compare_20261106/) · [结果与缺项说明](../experiments/p2_compare_vico43_20261106/RESULTS.md)。

**ViCo联合8000步生成可视化（2026-11-06）：** [打开全部43条speaker／GT／生成同步对照](http://localhost:8876/listen_benchmark_review/p2_vico43_combined8000_20261106/)。复用现有ViCo审核页布局与同步播放方式；支持原生512×512／25fps及评分256×256／30fps切换，speaker／生成／混合声音、同步拖动与变速、逐条运动指标和规划文本。原[ViCo原画幅页](http://localhost:8876/vico_paper_test/)也增加了入口。

**ViCo运动指标补测已完成（2026-11-06）：** 联合8000步test43，129路3DMM／48,864帧完成，FD、FID_delta_fm、SND、V-D覆盖43条，RTLCC42条、RWTLCC41条，短片段N/A；7项数值回归及独立验收通过，node8 GPU7已释放。原始系数、±2秒lag、4秒窗口／1秒步长、逐视频等权；公开定义复现，不宣称严格复现CustomListener原表。[结果表与完整协议](../experiments/p2_vico_motion8000_20261106/RESULTS.md)。

**本轮联合Whisper已加入原可视化（2026-11-06）：** 两套测试集各14组，7组消融、21项对照。新增联合6000／8000，明确与仅AV、AV＋Event、AV＋Feature、原Event＋Feature和仅AV＋Whisper比较；含6000→8000及预算不等的终点参考。原值、B−A、箭头、第一／第二名颜色、音频及覆盖保留。 [ResponseNet联合消融](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=responsenet&ablation=joint_whisper) · [Seamless联合消融](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=joint_whisper) · [有声视频基线对照](http://localhost:8876/listen_benchmark_review/p2_seamless200_20261105/?preset=joint_av&metrics=1)。588项数值／差值检查、浏览器与32份新增媒体／PCM核验通过。

**已完成（2026-11-06 02:23 UTC，北京时间10:23）：** 6000／8000步×Seamless200／ResponseNet200，共800条生成、导出、62个评分分片及全局FID/FVD汇总完成，最终逐条验收通过。Seamless两组各1条情绪／运动支持不足（199/200），其余图像指标200/200；音频执行错误0，UTMOS因短回应为N/A。训练保持停在8000步，巡检goal保持暂停。 [结果与验收](../experiments/p2_combined_whisper_eval_20261106/RESULTS.md)。

**十二组数值与消融增减表已更新：** [ResponseNet随机200／Seamless固定200可切换页面](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/)；各套12组总体值，6组消融／11项对照，展开显示基线、对照、B−A，百分比用百分点。原Seamless视频页同步消融表。308项数值／差值及浏览器检查通过。[代码、导出与验收](../experiments/p2_responsenet_random200_twelveway_20261105/DASHBOARD_README.md)。

**ResponseNet分层随机200条／12组全部完成（2026-11-05 08:48 UTC）：** 2400条生成导出、384个评分分片及总体FID/FVD合并全部结束，视频／音频执行错误0，每组200条指标覆盖。UTMOS按短回应／静音规则为N/A。GPU已结束本轮工作并释放。[完整结果表](../experiments/p2_responsenet_random200_twelveway_20261105/RESULTS.md) · [CSV](../experiments/p2_responsenet_random200_twelveway_20261105/summary.csv) · [最终验收](../experiments/p2_responsenet_random200_twelveway_20261105/FINAL_AUDIT.json)。

**ResponseNet随机200条32卡运行已验收（2026-11-05）：** 修复旧分片尾部27卡空等，先用空卡解码SFT已生成结果，再切到四机32卡共享领取；组合组已产生首批32条，实测各机8/8卡计算、利用率98%–100%。[验收与资源快照](../experiments/p2_responsenet_random200_twelveway_20261105/expansion32/LIVE_ACCEPTANCE.json)。

**ResponseNet随机200条评测扩至32卡（2026-11-05）：** 用户新增node8的4–7卡，已实际接入Whisper并行推理；当前阶段正常收尾后自动统一32卡共享领取，减少固定分片尾部等待，固定200条／12组及采样不变。[扩容状态与验收](../experiments/p2_responsenet_random200_twelveway_20261105/expansion32/README.md)。

**Seamless200 Whisper直连cross-attention补测已完成（2026-11-05）：** 6000／8000各200条，共400条生成和全量指标完成，已加入原HTML成为12组，新增AV／Whisper6000和Whisper6000／8000快捷切换。6000相对AV6000，PSNR15.89→16.38、SSIM0.6662→0.6769、LPIPS0.3619→0.3356；发声漏检仍严重。每组1条人脸／关键点覆盖不足，UTMOS无有效窗口保留N/A；音频执行错误0。[打开Whisper对照](http://localhost:8876/listen_benchmark_review/p2_seamless200_20261105/?preset=whisper&metrics=1) · [结果与验收](../experiments/p2_seamless200_whisper_20261105/README.md)。

**2026-11-05 ResponseNet固定分层随机200条／12组评测：** 从完整586条按GT音频×时长×ASR来源等比例随机，seed20261105，不参考模型得分。200条／25.08分钟／91个来源，17条非零音轨；全部12组统一清单，复用1160条、新生成1240条，四机授权28卡接续。[清单、协议与结果](../experiments/p2_responsenet_random200_twelveway_20261105/README.md)。旧精选200条与旧586条结果保留。

# 评测索引

这里只登记评测侧的入口。Seamless 训练切分仍以项目根目录 [README.md](../README.md) 里的固定 J v1 为准，本目录不替换那套规则。

## ViCo / DIM 对齐：工程完成，原表等价性待确认（2026-11-08）

**2026-11-09 已核对用户提供的飞书本地副本：** DIM原表8行96个数值核查一致；Avatar Forcing表4转引的RLHG／L2L／DIM三行24个数值完全一致。其他截图的FD分别可能为动作L1或分布Fréchet，且有乘100／1000单位；不能合并排名。†的转引／预训练含义也已区分。[截图与原论文逐表结论](../experiments/baseline_training_readiness_20261109/FEISHU_VICO.md)。

**最新Var选择：** 用户指定采用逐系数总体方差的平均值，即`mean_j Var_t(x[t,j])`、`ddof=0`。GT及两版模型已统一重算，其他五项指标与有效帧保持，原公式成绩另存。[本轮结果／公式／完整JSON](../experiments/p2_vico_dim_var_per_coefficient_20261108_v2/README.md) · [配置](../configs/p2_vico_dim_variance_per_coefficient_v1.json)。十次择优队列已按其他任务中的用户要求暂停，本次只改评分定义，未恢复GPU任务。

指定EMOCA_v2_lr_mse_20，正式test43，speaker／GT／联合8000/base-Qwen／联合8000/SFT2752共172组动作特征已提取；两版FD、P-FD、MSE、SID、Var、rPCC评分完成。全43条548个数值复算最大误差0，原封存文件及源码／权重指纹核验通过。前期审计的“尚未接入EMOCA”属于历史状态，不再表示当前进度。

GT统计与论文尚不一致：exp Var约1.45611对0.93，pose Var约0.02327对0.01；实际测试ID／有效范围、EMOCA预处理、主表采样选择、speaker及指标汇总仍需原作者材料确认。因此工程完成不等同于已严格复现DIM原表。七类事件pilot是独立工作，不作为此项完成条件。

[结果与全部指标](../experiments/p2_vico_literature_protocol_20261108/RESULTS.md) · [本轮验收／复算](../experiments/p2_vico_dim_handoff_20261108/README.md) · [需要你double-check的清单及提问草稿](../experiments/p2_vico_dim_handoff_20261108/DOUBLE_CHECK.md)。

**后续方向已更新：** 用户无法联系作者，改由公开论文／代码证据推断配置，不以作者回复为前置条件。已完成GT侧Var／SID定义诊断并记录推荐设置；原double-check表作为疑点目录保留。[推断配置、依据与诊断](../experiments/p2_vico_dim_inferred_protocol_20261108/README.md)。Var差异可能包含统计定义因素，不能仅据GT不一致认定提取器错误，也不按接近论文数字的程度挑设置。

**配置敏感性已实测：** 两版模型×正式43条／同时间戳speaker／加载器范围投影三套设置，六项DIM指标及Var／SID替代定义全部计算；6项测试、48个历史汇总数一致性检查通过。[推荐设置与实际差异](../experiments/p2_vico_dim_inferred_protocol_20261108/sensitivity_v1/RESULTS.md) · [固定配置](../configs/p2_vico_dim_inferred_v1.json)。一帧偏移在本批影响较小，测试范围对FD的影响更明显。现有结果仍为单次采样，未执行10次GT择优。

**DIM采样预算补测已排队：** 此前ViCo最新SFT2752版固定seed137–146，复用已有43条、补387条；发布代码的完整56维GT择优独立成补充表。4项测试与原环境CPU预检通过，仅node11 GPU7、余量足够才自动运行；当前等待显存，尚无best10最终分数。[队列、实时状态及采样说明](../experiments/p2_vico_dim_best10_20261108/README.md)。

## 整段事件类别一致性（2026-11-08）

**线索传递遗漏已修复（2026-11-12）：** 检测提示升级为v2，接入窗口内微笑blendshape摘要，避免缺失测量被当成零；ASR提示仅使用规范化结果，长视频按窗口计算人脸覆盖率。22项CPU测试与8个真实媒体的14份提示回放通过，原计分结果复算一致；未重跑GPU检测，尚不声称检测准确率提升。[修复与验证](../experiments/event_presence_cue_fix_20261112/README.md)。后续评测需新目录绑定，不覆盖原pilot。

已按用户确认的[事件级框架](../writing/事件级评测框架_当前确认版.md)接入独立事件阶段：视觉“点头、摇头、微笑／笑态、皱眉”，音频“可听笑声、简短词语回应、非词汇回应”。重复同类只记存在，不要求与GT时间重合；每类跨测试集累计TP/FP/FN，视觉、音频分别报告Macro-P／Macro-R／Macro-F1和逐类结果。

主宏平均类别取冻结参考中有正例的类别，无参考正例类单列误报；双空不加分，未知／检测失败不作负例，多方法按逐类共同有效ID比较。它衡量参考类别一致性，语境适当性和生成内部视听协调性仍单列。GLARE R-F1及音频活动时序F1继续使用各自定义，不冒充本项。

检测复用现有MediaPipe／反应线索、ASR和Qwen3-Omni标注链路，分别重读GT与生成最终媒体；生成音频保留自身完整音轨，不套GT发声门控。新提示输出七类显式状态，未提及不自动判无。当前仅用于pilot，独立人工核验和阈值校准未完成，不预写正式benchmark分数。

[完整定义、运行命令、输出与人工核验接口](event_presence/README.md) · [默认配置](../configs/p2_event_presence_v1.json) · [独立入口](../scripts/p2_event_presence.py) · [小规模实际验证目录](../experiments/p2_event_presence_pilot_20261108/) · [已确认设计](../docs/superpowers/specs/2026-11-08-event-presence-evaluation-design.md)。实际执行状态与完成证据以模块说明及运行记录为准，不改写旧结果或在运行评分任务。

小规模真实验收已完成：四对生成＋四对GT自对照、八个媒体，技术失败0、一个音频类别unknown保留；15项CPU测试与28张逐类计数表独立重算通过。[实测结果和限制](../experiments/p2_event_presence_pilot_20261108/README.md) · [盲化核验页](http://localhost:8876/listen_benchmark_review/event_presence_pilot_20261108/)。人工可靠性与校准仍pending。

## 无损GT解码缓存提速（2026-11-05）

适用于同一Seamless测试集比较多个权重：缓存一次GT完整uint8 RGB及原FPS、PTS、最终裁框审计，跨权重共享只读mmap的无损`.npy`；生成视频仍逐系统解码。当前实现针对Seamless lazy GT，其他数据集尚未接入这套解码缓存。[完整实现、实测与审计说明](../experiments/p2_seamless200_tenway_24gpu_20261105/DECODE_CACHE.md)。

实测需区分三个口径：

| 范围 | 原耗时 | 缓存后耗时 | 证据 |
|---|---:|---:|---|
| GT单独解码，51帧短片段 | 3.82秒 | 0.036秒 | [像素／时间审计对照](../experiments/p2_seamless200_tenway_24gpu_20261105/DECODE_CACHE_PARITY.json) |
| GT单独解码，297帧长片段 | 16.00秒 | 0.190秒 | 同上 |
| 同两条样本实际评分中的GT＋生成视频合计解码 | 14.90／3.17秒 | 0.45／0.12秒 | [GPU指标与特征逐值对照](../experiments/p2_seamless200_tenway_24gpu_20261105/DECODE_CACHE_GPU_PARITY.json) |
| 首批正式生产34次缓存命中 | — | 平均0.84秒，中位数0.19秒 | [生产窗口记录](../experiments/p2_seamless200_tenway_24gpu_20261105/DECODE_CACHE_HIT_TIMINGS.json) |

原先约7.8秒/条的统计包含GT和生成视频两部分。上述耗时只反映解码环节；整轮收益还取决于像素指标、人脸／表情、分布特征提取与汇总。冷缓存首次构建仍需原解码成本，不能把命中样本的倍数当作全窗口或整轮提速倍数。

### 复用配置与命令

在评测工作目录的`protocol.json`显式设置绝对路径：

```json
{
  "metrics_decoded_gt_cache": "/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_seamless200_tenway_24gpu_20261105/decoded_gt_cache"
}
```

[评分调度](../src/p2_testsets_eval/metrics.py)为后续启动的Seamless评分worker调用[缓存包装入口](../scripts/p2_cached_video_metrics.py)。已完成评分和正在运行的进程保留，不为启用缓存重跑。新测试集建议使用自己的缓存目录；相同目录中的不同冻结输入也会按身份隔离。

准备当前固定测试池的缓存并进行短／长样本对照：

```bash
cd /private/liqi_workspace/project_code/audio-driven-v5
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /private/software/conda/envs/AnyTalker/bin/python \
  scripts/p2_decoded_gt_benchmark.py \
  --work experiments/p2_seamless200_tenway_24gpu_20261105 \
  --workers 4 --prewarm
```

这条命令只读取工作目录协议指定的测试池，不扫描全量数据、不占GPU。实际缓存条数和字节数见工作目录`DECODE_CACHE_PREWARMED.json`；当前200条估计约24GB，为可重建的评测缓存，不新增WAV，不作为训练输入。

### 一致性与失效规则

[缓存实现](../src/p2_testsets_eval/decoded_gt.py)绑定冻结行、最终裁框、源媒体位置／时间、reader／缓存代码、库版本及源档大小／mtime／ctime；每次读取仍验证冻结输入和完整RGB SHA256。不同身份产生不同缓存键，像素损坏会拒绝复用；互斥构建、临时目录写入及原子发布避免读取半成品。

原OmniResponse reader/scorer文件保持不变；缓存包装的代码身份、命中／未命中、耗时和RGB摘要记录在每份评分的`decoded_gt_cache.json`及`contract.json`的`execution.decoded_gt_cache`。两条真实GPU对照中，PSNR、SSIM、LPIPS、表情指标、可计算的ΔResp，以及GT／生成侧FID/FVD特征逐值一致；17项相关回归通过。原不可计算项、覆盖率和全量特征汇总口径保持原规则。

**GLARE最终检查点的ViCo／ResponseNet评测已完成（2026-11-12）：** 最后第32940步，正式ViCo43＋完整ResponseNet586，共629条、1258份媒体完整解码验收，执行错误0；ViCo五项、BFM运动和DIM12项、ResponseNet8项均完成。DIM公开代码16项数值误差0，采用既有同一评分环境；无listener生成音轨，不计算音频分数。node11 GPU4生成约98%–99%，最高16评分分片与4动作提取重叠。[完整结果](../experiments/glare_final_vico43_responsenet586_20261112/RESULTS.md) · [协议、视频与验收](../experiments/glare_final_vico43_responsenet586_20261112/README.md)。

**GLARE第19050步的ViCo／ResponseNet及诊断200已完成（2026-11-12）：** 用户指定`before_extension30epochs.pt`，实际step19050；ViCo43／ResponseNet586／Seamless固定200共829条生成和评分完成，已加入[对比表](VICO_ALL_LORA6000_6500_RESULTS_20261112.md)。Seamless最终范围已按用户更正为完整6624条，见下方独立全量队列，200条只作子集对比。[本阶段结果与验收](../experiments/glare19050_vico43_responsenet586_seamless200_20261112/RESULTS.json) · [协议与恢复](../experiments/glare19050_vico43_responsenet586_seamless200_20261112/README.md)。

**19050步范围已按用户更正为完整Seamless test6624（2026-11-12）：** ViCo43、ResponseNet586及固定200的829条测评已完成，已加入[同表对比](VICO_ALL_LORA6000_6500_RESULTS_20261112.md)。另已启动该权重完整6624条：复用全部输入／韵律和已完成200条视频／评分，补齐其余6424条。按最新指令，node11 GPU4只继续19050步，当前4路生成＋2路评分；32940步控制器及评分进程已暂停，已有视频、评分块和缓存保留。[19050完整6624：协议、进度与恢复](../experiments/glare19050_seamless_test6624_20261112/README.md) · [32940暂停记录](../experiments/glare_final_seamless_test6624_20261112/PAUSED.json)。

**GLARE第32940步的Seamless全量队列已暂停（2026-11-12）：** 按用户最新指令，node11 GPU4只继续19050步。32940步已生成6624/6624条，已完成79个评分块／2472条；已停止控制器及在途评分进程，全部结果与共享输入／韵律缓存保留，全量评分尚未完成。[暂停记录](../experiments/glare_final_seamless_test6624_20261112/PAUSED.json) · [运行与恢复](../experiments/glare_final_seamless_test6624_20261112/README.md)。

## ResponseNet

**全LoRA6000／6500的ResponseNet完整586条正在评测（2026-11-12 00:01 UTC快照）：** 6000已导出70/586条视频，6500排队；条件586条齐全。GPU7已自动交回原常驻模型，资源许可时并行解码，九小时巡检结束后后台队列继续。完整分数尚未生成；原fps／512×512、全量汇总FID／FVD，缓存批次浮点差异及72条本地词时间来源均保留披露。[配置、进度、视频与结果入口](../experiments/p2_responsenet_full586_all_lora6000_6500_20261111/README.md)。

官方 140 条。说话人固定为 Speaker / Human · 观察对象，Agent 自己说话的区间不反向切。音质模型不跑。

140 条里 135 条有观察对象 pair，共 586 条。测评读取方式：[responsenet_speaker_j/HANDOFF.md](responsenet_speaker_j/HANDOFF.md)。处理代码和文件位置：[responsenet_speaker_j/README.md](responsenet_speaker_j/README.md)，逐条清单是 [responsenet_speaker_j/CUTS.md](responsenet_speaker_j/CUTS.md)。

已放上原视频审核页的两条：

- http://localhost:8876/seamless_face_listening_v1/listen_full_source/?pair=6V1HXfo
- http://localhost:8876/seamless_face_listening_v1/listen_full_source/?pair=9lhidn7

## VidChatBench

本地内容验收过的 8 条分轨，不是官方 501 条。说话人固定为 human。官方 ASR 没有词时间，切分用本地已有的词级转写。音质模型不跑。

处理代码和页面：[vidchatbench_speaker_j/README.md](vidchatbench_speaker_j/README.md)。

## ResponseNet固定200条：新两组四权重

使用用户确认的[200条原始清单](../experiments/responsenet_initial200_20261102/references.jsonl)与[CSV时间范围](../experiments/responsenet_initial200_20261102/selection.csv)。原始H＋AV＋Feature、SFT新H＋AV＋0.05 Event各6000/8927步，合计800个输出。四机32卡自动接续当前训练，沿用GPU加速推理及九项评测。[本轮运行、模型对应关系与结果入口](../experiments/p2_response200_feature_sft_event_20261103/README.md)。

新增交叉对照：SFT LoRA Qwen＋原始H Event6000，同200条并与SFT H Event组共享条件。现为5组1000输出；旧权重兼容重放与SSH恢复详见[本轮记录](../experiments/p2_response200_feature_sft_event_20261103/README.md)。

## ViCo

**全LoRA step6000／step6500结果对比已整理（2026-11-12）：** ViCo正式test43两组均完成，汇总画质／身份五项、DIM表情／姿态十二项、运动补测与覆盖数，附权重来源、Var公式差异、视频及复核入口；ResponseNet全586条另标为运行中，尚无全量分数。[打开两权重结果汇总](VICO_ALL_LORA6000_6500_RESULTS_20261112.md) · [只看表格（HTML）](VICO_ALL_LORA6000_6500_RESULTS_20261112.html)。

**全attention／FFN LoRA6500已完成test43（2026-11-11）：** 官方五项、运动补测、DIM Exp／Pose十二项与独立数值及修复来源复核通过；43条、86个视频副本、DIM有效16,237帧，原公开代码对照误差0。保留6428后batch28来源、辅助关键点诊断修复及覆盖差异。6000结果保持。[结果](../experiments/p2_vico43_all_lora6500_sft2752_20261111/RESULTS.md) · [43条视频](../experiments/p2_vico43_all_lora6500_sft2752_20261111/VIDEOS.md)。

**全attention／FFN LoRA6000的test43评测已全部完成（2026-11-11）：** 用户指定step006000，仅node11物理GPU7，固定SFT2752／seed137／Euler30／CFG3、7，512／25→256／30。43条推理、86个分辨率副本完整解码、官方画质／身份5项、CustomListener风格运动补测、DIM十二项均完成。官方结果PSNR19.0860、SSIM0.64727、FID31.2077、CSIM0.59348、CPBD0.14220。DIM共同有效16,238帧，原代码16项数值误差0；主Var沿用用户指定逐系数方差平均，原公式另存。评测进程已退出，不影响其他任务；未接AF SPECTRE。[完整结果](../experiments/p2_vico43_all_lora6000_sft2752_20261111/RESULTS.md) · [43条视频与音轨](../experiments/p2_vico43_all_lora6000_sft2752_20261111/VIDEOS.md) · [复核记录](../experiments/p2_vico43_all_lora6000_sft2752_20261111/FINAL_USER_REVIEW.json)。

论文数据集，不是挑战赛那 30 条无听者真值的提交包。测试集 43 条，域外测试 42 条，训练集 398 条。听者真值是裁剪后的听者视频。

**AF Table4原始视野一轮已完整验收、数值未复现（2026-11-10）：** 43条／13575帧输出、129路官方RetinaFace／FAN2／JPEG前端提取、8项评分和独立重算完成。主重建与完整DIM代码候选均0/8；不能据此声称与论文对齐。[8项实测及差值](../experiments/p2_avatarforcing_table4_framegrid_20261110_v2/OFFICIAL_RESULTS.md) · [完整验收](../experiments/p2_avatarforcing_table4_framegrid_20261110_v2/FINAL_AUDIT.json)。

**GT来源受控核对已完整完成（2026-11-10）：** 同一批生成视频，改官方256 GT／speaker后主口径与完整DIM候选均0/8；Exp FD20.00→42.82、Pose FD0.0682→0.0800，Var按程序断言保持相同。该切换没有修复当前差距，不能反推作者私有GT已确认。[完整结果](../experiments/p2_avatarforcing_table4_canonical_gt_20261110_v4/RESULTS.md) · [独立验收](../experiments/p2_avatarforcing_table4_canonical_gt_20261110_v4/FINAL_AUDIT.json)。IIANet音源分离43对／86路已统一在CPU完成并通过独立验收，相关进程已退出。官方代码及权重来源、正式test43身份、采样数、嘴部帧数、波形完整性和FLOAT WAV误差均已核对。按用户指示到此停止，未启动后续AF生成或评分；仍没有分离后新的Table4成绩。[音轨清单](../experiments/p2_avatarforcing_table4_audio_preparation_cpu_20261110_v5e/separated_audio_manifest.jsonl) · [完成验收](../experiments/p2_avatarforcing_table4_audio_preparation_cpu_20261110_v5e/SEPARATION_COMPLETE.md)。[音频准备](../experiments/p2_avatarforcing_table4_audio_preparation_cpu_20261110_v5e/README.md)。

**参考图预处理对照已完整完成（2026-11-10）：** 43条／13575帧、129路及8项独立重算通过；主口径与完整DIM候选均1/8。Exp FD22.98→20.00、Pose FD0.0897→0.0682，论文16.64／0.05；仅Pose rPCC两位小数一致，仍未复现。当前接续两条公开演示读取及固定生成、改官方256 GT／speaker来源的对照；Var只依赖生成，不能靠更换GT改变。[本轮完整结果](../experiments/p2_avatarforcing_table4_reference_mean_20261110_v3/CONTROLLED_REFERENCE_RESULTS.md) · [官方权重来源核验](../experiments/p2_avatarforcing_table4_reference_mean_20261110_v3/checkpoint_source_audit/README.md)。

**Avatar Forcing Table4独立复现检查已完成（2026-11-10）：** 官方权重恢复真实speaker视觉，43条生成＋129路SPECTRE动作＋8项评分及独立最终核验完成。8项中0项按论文两位小数一致；当前公开配置重建未复现Table4，不能宣称评测已严格对齐。SID K15／9、逐系数Var；不等同DIM的EMOCA／K40／20、P-FD或MSE验证。[逐项实测与论文差值](../experiments/p2_avatarforcing_table4_recheck_20261110/RESULTS.md) · [冻结协议与来源](../experiments/p2_avatarforcing_table4_recheck_20261110/README.md)。

**Table4差异定位与工程修正（2026-11-10）：** ViCo官方BFM／逐帧L1、DIM的EMOCA／K40／20与AF的SPECTRE／K15／9必须分开；AF没有发布整表评分入口，FD／SID／Var跨片段汇总仍不能由引用旧基线反推。两条正式补充视频已用音轨匹配到test，证明其画面视野与本地256紧裁脸不同。另以同权重同21段随机噪声的单变量干预确认cuDNN状态和官方min/max导出会产生1.326倍亮度差，导致单片段FD漂移；完整原脚本可字节复现，独立稳定入口已实现并验证。原43条成绩未改，修正入口只验证1条，不能当作新43条结果。[定位报告、六条成对消融和原始证据](../experiments/p2_avatarforcing_table4_diagnosis_20261110/README.md) · [稳定入口](../scripts/p2_avatarforcing_table4_generate_stable.py) · [实际验证](../experiments/p2_avatarforcing_table4_stable_runtime_20261110_v2/FIX_VALIDATION.json)。

**Avatar Forcing已具备权重并完成43条实测（2026-11-10回查确认，原执行2026-11-07）：** 官方`flow_transformer.pth`与`motion_autoencoder.pth`均在本地，SHA与冻结生成协议一致；推理适配器实际加载327个核心参数，43条原生视频／43条评分副本及最终评分完成。共同输入为speaker音频＋listener首帧，自身音频及speaker视觉条件置零，单次seed137，原生512／25→评分256／30。已完成ViCo画质／身份五项和BFM／DECA动作评分；这份历史实测不等同于DIM的EMOCA十二项或十次择优复现。[完整记录与代码入口](../experiments/p2_vico43_unified_20261107/README.md) · [实际结果](../experiments/p2_vico43_unified_20261107/RESULTS.md) · [推理适配器](../compare/deployment/vico_avatarforcing.py) · [43条视频对照页](http://localhost:8876/listen_benchmark_review/vico43_unified_20261107/)。

读取方式和算分命令：[vico/README.md](vico/README.md)。机器可读清单是 [vico/index.json](vico/index.json)。画面和权重在 `/public/0_DATA/2_Avatar/ViCo`。本轮43条输出验收及画质／身份五项评分均已完成，运动／同步补测已独立完成，结果见上方。

**2026-11-06 画质／身份五项实测已完成：** 最新AV＋Event＋Feature＋Whisper 8000步，正式test 43/43条、16,288帧；512×512/25fps生成，256×256/30fps官方评分。PSNR **19.6683**、SSIM **0.66245**、FID **30.1163**、CSIM **0.59684**、CPBD **0.13913**。逐条独立验收通过，node8 GPU7已释放。这份五项结果不含运动指标，独立补测结果见上方。[结果与协议](vico/README.md) · [最终验收](../experiments/p2_vico256_combined8000_20261106/FINAL_AUDIT.json)。

测试集 43 条可以在审核页上看：[http://localhost:8876/vico_paper_test/](http://localhost:8876/vico_paper_test/)。页面播的是原画幅，算分仍用 `/public` 里的 256×256 裁剪。

## LM Listener

**2026-11-09 官方评分器CPU实测通过，并已开始恢复原媒体。** 发布包只有系数／转写和时间，共36个条目、原始媒体0个；已另外确认5个原始公开视频来源、覆盖118条记录，并取回1个源视频及音频，导出候选片段和listener首帧。与官方系数的时间、speaker音轨及裁框仍待验证，当前模型尚未生成或打分。[逐项阻塞、派生名单和评分验收](../experiments/lm_listener_readiness_20261109/README.md)。

压缩包是 `evaluation/lm-listener/data_release/lm_listener_data.zip`，解压后的名单在 `evaluation/lm-listener/dataset/`。官方说明是 [lm-listener/README.md](lm-listener/README.md)。以下为发布文件的原始行数。

| 目录 | 训练 | 验证 | 原测试 | 去重叠测试 |
|---|---:|---:|---:|---:|
| `trevor` | 5379 | 446 | 2885 | 2853 |
| `conanglobal` | 3234 | 181 | 574 | 446 |
| `stephenglobal` | 4803 | 793 | 808 | 没有这份文件 |
| `trevorconanstephen` | 13416 | 1420 | 4267 | 4107 |

作者2026年6月更新提供去重叠测试。逐字段／逐张量比对确认：Trevor2853＋Conan446＋Stephen原test808，含重复次数的多重集精确等于官方合并no-overlap4107行。Stephen缺单独文件不再阻塞名单确定；其808行含4行重复帧区间，合并共有4103个独立区间／63个来源视频。[4107行清单](../experiments/lm_listener_readiness_20261109/test_manifest.jsonl)保留源行号和重复，正式生成前须固定重复权重。

官方评分器用一条60帧GT自对照已实际完成：MSE与valence误差为0，FD约7.45e-9、P-FD约−2.28e-6（原实现数值舍入），不是模型成绩。官方README示例仍读val，接入测试时须显式选择测试名单；系数比半开评分区间多一帧，不能直接全算。

DIM论文的LM Listener表只用Trevor2366／222／543段，并非当前三人测试；其Exp／Pose六项指标也不同于原官方56维／valence评分。当前模型尚未生成或打分，不能把CPU自对照记作benchmark完成。

## VideoFDB

公开打分实现在 [OpenVideoFDB](OpenVideoFDB/)。数据目录固定为 `/public/0_DATA/2_Avatar/VideoFDB`。本地入口是 [videofdb_local.py](videofdb_local.py)：`snapshot` 下载，`manifest validation|test` 写清单，`bind` 把已经生成的 listener wav/mp4 放进打分目录，`score` 计算时机和评分细则。生成清单每行一个 `clip_id`，以及 `audio` 或 `video`。评分员是本地 Qwen3-Omni：把模型目录写进 `OpenVideoFDB/.env` 的 `QWEN3_OMNI_PATH`。`score` 默认使用 node8 的 7 号卡，可用环境变量 `VIDEOFDB_CUDA` 改卡。

## Seamless

完整接入为独立test 6624条／11.05小时，见[OmniResponse接入与读取](../../OmniResponse/evaluation/seamless/README.md)。

2026-11-05固定200条／19.99分钟快速诊断集，并含48条核心子集；选自该6624条，使用最新caption、最终裁框和净化listener音轨。[筛选、数据及输入入口](../experiments/seamless_quick200_latest_20261105/README.md) · [历史9组重测](../experiments/p2_seamless200_nineway_20261105/README.md)。


2026-11-05最新执行：用户停止三机训练后，改用node17／node12／node4共24卡重测历史9组＋组合损失7000步，共10组／2000输出。旧node8队列自动恢复已取消，完整产物复用。[协议、状态与结果](../experiments/p2_seamless200_tenway_24gpu_20261105/README.md)。

当前提速更新：node8仅0–3卡提前评分，与24卡生成重叠；自检通过，node8容量按CPU与显存余量提高至8路/卡，生成24卡释放后4路/卡，共28卡评分。控制器加入同一评分队列而非重复提交；采样及指标口径不变。见本轮`METRICS_OVERLAP_LIVE_VERIFIED.json`及`SCORING_CONCURRENCY_EXPANDED.json`。

Seamless200新增[无损GT解码缓存](../experiments/p2_seamless200_tenway_24gpu_20261105/DECODE_CACHE.md)：复用完整RGB而不改变原reader/scorer；两条真实GPU指标及分布特征逐值一致，后续worker自动采用，已有评分保留。

2026-11-05 05:53 UTC：本轮十组／2000输出、240评分分片及全局汇总均结束；[RESULTS.json](../experiments/p2_seamless200_tenway_24gpu_20261105/RESULTS.json)已生成。完成标记为partial：每组1–2条缺人脸/运动支持，UTMOS无有效窗口；覆盖和N/A保留，初始化与音频执行错误均0。


## Seamless200十组结果分析与视频复核

[打开可播放对照（8876）](http://localhost:8876/listen_benchmark_review/p2_seamless200_20261105/) · [完整分析](../experiments/p2_seamless200_analysis_20261105/ANALYSIS.md) · [总体CSV](../experiments/p2_seamless200_analysis_20261105/summary.csv) · [源group配对bootstrap区间](../experiments/p2_seamless200_analysis_20261105/paired_bootstrap.json)。

16条目的性/随机样本中每条都有全部10组媒体，同步对照GT与任选两个系统，支持精确listener PCM、speaker音轨及统一增益混合试听、原始规划prompt和下载。旧ResponseNet展示页有新Seamless入口。选例规则公开，不以典型案例估计总体表现；初始化、声音、覆盖和指标N/A保持正式口径。浏览器验证16条/10组、192路PCM SHA、播放/拖动/切音轨/模型切换/导出与手机布局均通过。[验收记录](../experiments/p2_seamless200_analysis_20261105/BROWSER_AUDIT.json)。


**2026-11-05 Seamless200消融补充：** 总体指标第一名绿色、第二名蓝色，按未舍入值／并列同色；Emo-Div按与GT差距排序。[六维消融分析](../experiments/p2_seamless200_analysis_20261105/ABLATIONS.md)。[Whisper speaker cross-attention 6000／8000补测队列](../experiments/p2_seamless200_whisper_20261105/README.md)：同200条／采样协议，等待已授权28卡释放后自动接续；原十组结果保持不变。
**Event＋Feature＋Whisper自动接续（2026-11-06）：** 当前31卡训练第8000步完整落盘并停训后，自动测本轮6000／8000步，固定Seamless200＋ResponseNet分层随机200，共800条。保持seed137、Euler30、CFG3/7与历史输入／评分协议；独立联合模型加载器、共享任务池、历史条件与GT缓存复用，node8 GPU7预留。CPU输入预演与5项回归通过，真实GPU验收待接续执行。[运行与结果](../experiments/p2_combined_whisper_eval_20261106/README.md)。

**五个参考方法三数据集结果汇总（2026-11-13）：** L2L四套、ListenFormer、DyStream、Avatar Forcing、DualTalk的ViCo43、Seamless固定200、ResponseNet固定200成绩已整理，包含图像／视频指标、原生及统一动作指标、有效覆盖和权重说明。完整Seamless6624／ResponseNet586尚未完成，暂停进度单列；附428项精确数值CSV及封存来源核验。[打开结果汇总](REFERENCE_METHODS_RESULTS_20261113.md) · [精确数值CSV](REFERENCE_METHODS_RESULTS_20261113.csv)。

