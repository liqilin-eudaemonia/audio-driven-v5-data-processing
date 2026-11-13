# J 全量标签：原画转正及重复标签重打标（2026-11-05）

用户最新要求：自动通过与待复核均保留；原画旋转90度的素材优先恢复正常方向，无法可靠使用的剔除；方向正常的宽屏保留。全部重复标记重打标，指定新node8（172.29.104.237，SSH22，port提示20491）的8卡，优化吞吐，重试后仍异常则剔除。

## 对象与方向核对

冻结父入口为 `../listen_j_labels_manifest_20261105/labels.jsonl`，1,107,322条标签；1,107,323个目标中另1条原检测无脸缺项保持单列，不回填旧标签。

全量1,107,323条原始质量测量已回读、逐文件核对SHA，执行错误0。人脸接近90度姿态只用于找候选；再逐一看未经裁框的原画，确认整幅场景和身体是否横置。候选764个来源中21个为正常歪头，保留；额外全查73个未在候选中的宽屏来源，其中4个实际横置，其余69个保留。最终747个来源／1,699段需要转正，90个已核对的正常来源保留。11个混合姿态来源另外核对非候选片段，场景仍横置。

逐来源原画判断和图片SHA：`../listen_j_clean_regen_20261105/orientation_source_decisions_final.json`。转正方向在 `rotation_sources_confirmed.json`：逐一检查全部747张转正原画，742个逆时针90度，S0233、S0242、S0249、S0251、S0563五个顺时针90度。`rotation_plus90_check_01.jpg` 至 `12.jpg` 留存方向核对依据。

## 素材修复与重新验收

`restoration_summary.json`：1,699段均完成物理转正及重新测量，1,690段可继续打标，9段未通过原有清晰度规则，后续剔除。每段使用已经验收的最终原生裁框，做精确90度像素转置，不做空间缩放；以libx264rgb CRF0无损编码，使用NUT容器保留原始PTS，再写入独立TAR。保持原sample ID、pair、participant、split、group、起止时间、净化音频配方和音频波形。每段验证完整解码帧首尾时间支持，并将代表帧与原画裁框直接转置逐像素核对。

修复后的原生素材在 `derived_media/<sample_id>/upright.tar`，`provenance.json` 含原始素材、裁框、方向、完整命令、归档SHA、帧数、首尾PTS及像素误差。`restored_rows/` 和 `quality_measurements/` 保存新的来源及5fps质量测量。来源和质量合同另立版本，见 `upright_quality_contract.json`。旧素材、测量、caption及缓存保持原样。

MP4旧版muxer的edit-list会使绝对PTS轻微偏移，因此正式修复素材使用NUT；`prototype.mp4` 为未采用的探索结果，不作为数据入口。前期冒烟错误保留在 `smoke_execution_errors_resolved/`，正式全量转正执行无该类错误。`RESTORATION_AUDIT.json`独立回读全部1,699份修复、归档及测量，核对时间、音频／参与者／split、完整SHA和像素误差。

## 八卡重打标

全部36条 `repeated_generation` 标签都纳入，包括最初34条长循环／截断，以及R34、R35两条短重复。修复素材1,699段与重复标签36条交集1条，统一任务1,734条，见 `regenerate_candidates.jsonl`。重复ID见 `repeated_sample_ids.json`。此前34条的第一轮修复结果在 `../listen_j_clean_regen_20261105/`，本目录按照最新要求重新处理全部36条，不将第一轮失败自动视为本轮失败。

每GPU一个独立模型副本，BF16、memory utilization 0.86、最多4条并发；每GPU8个CPU准备进程、预取16条、就绪优先。修复素材重新运行25fps点头及视觉反应检测，复用同次解码的8fps Omni帧；音频与父标签encoder波形SHA必须一致。重复而未转正的数据还必须与父标签视频帧SHA一致。所有新caption保存父文件及SHA、实际提示词／渲染文本、音视频输入SHA、新身份、重试记录和复核标记。

两轮尝试仍空、截断或病态长循环的剔除；对于原36条重复标签，新结果若仍触发repeated_generation也必须重试，仍未消除则剔除；其他普通复核标记保留。新配置缩短冗余描述并用重复惩罚抑制循环。8个worker相互独立，不使用跨rank通信端口；20491保留在启动记录，SSH使用22。

当前正式入口为 `../../scripts/regenerate_j_upright_captions_v4.py`。第一次启动发现输入准备进程的scripts目录遮蔽src中的p2_listen包；已修正为src优先，并在GPU启动前加入独立输入验收。失败尝试记录保留在 `first_launch_initialization_stall/`、`v2_before_import_order_fix/`；不能将这些基础设施错误当作数据质量问题剔除。

历史资源等待：此前发现另一个 `p2_seamless200_nineway_20261105/base_standard` 评测已接管node8全部8卡。自动接续已部署在node8本机，等待该评测控制器及worker全部退出、8卡显存释放后启动；不抢评测阶段之间的空隙。用户若选择打标优先，再协调暂停评测。目前未停止该任务。`input_preflight.json` 为本轮正常重复样本与转正R25的真实输入检查，已通过。`QUEUE_STATUS.json`、`QUEUE_LAUNCHED.json`和`queue_pid.json`保存实时阶段及进程身份；队列还会自动导出全量入口、独立验收并更新索引。正常情况下无需另行人工接续；基础设施失败会标明failed_requires_fix而不会当作坏数据跳过。

## 发布条件及下一步入口

待1,734条全部有明确结果后，运行 `../../scripts/export_j_upright_labels.py export` 和 `audit`，将全量统一入口发布至 `../listen_j_labels_upright_20261105/`。必须有 `EXPORT_COMPLETE.json` 及 `UPRIGHT_AUDIT.json` 才算发布完成。所有不受影响标签逐行原字节保留；横置原标签必须被转正新版本替换或明确剔除；36条旧重复标签必须被替换或明确剔除。全量ID覆盖、split／group／时间／音频、衍生素材SHA及新检测输入均独立核对。

这一步准备新的可用标签池。后续需要基于新caption重算行为统计，再按既有选择规则选取实际训练／测试集，并为转正素材重建受影响的首帧、视觉及条件缓存；历史500h清单及正在运行的实验不在本步骤中改写。

## 机器释放后的实际接续与读取修复

2026-11-05用户通知机器释放后，自动队列已在node8接续。V3完成134条有效标签后遇到部分原始音频事件记录缺失的样本：这些样本需要回查官方转写，而转正衍生TAR只包含视频，不能向该TAR索取原始JSON。此错误是输入读取错误，不作为数据质量剔除依据，失败日志及退出码保留在v3_missing_vocal_metadata_attempt/。

V4将转正视频与官方转写来源明确分开：视频读取转正素材；转写读取原始listener归档，保持原始参与者、时间轴和净化音频。8个原始来源共20段真实出错路径全部重放通过，声学SHA和vocal_supplement与父标签一致，见VOCAL_ORIGINAL_ARCHIVE_PREFLIGHT.json。已有134条成功结果和1条视觉排除保留，接续只处理未完成项。64个CPU准备进程、128条预取、64个反应检测槽及八卡每卡最多4条并发；实时状态以QUEUE_STATUS.json、regeneration_workers/和node8_gpu_telemetry.jsonl为准。

旧选集受本轮修复影响的ID已整理在affected_previous_selections.json：旧AV500h中1,039条，旧规划SFT中1,285条（train 1,276、dev 9），旧test选择33条。这些旧清单与缓存未被改写；后续重选和缓存更新需使用本轮新来源。

## 最终完成结果

已完成八卡重打标及全量独立验收。新统一入口为[../listen_j_labels_upright_20261105/README.md](../listen_j_labels_upright_20261105/README.md)，保留1,107,313条，剔除9条；原1条缺项单列。CAPTION_REGENERATION_COMPLETE.json保存逐新标签SHA与八卡实测峰值，QUEUE_STATUS.json为complete。前文等待／失败启动为历史记录。
