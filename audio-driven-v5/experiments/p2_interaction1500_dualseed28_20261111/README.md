# 候选1661 vs 历史基线1500：4机28卡、64条、双seed

用户选择复用旧batch31基线1500，按累计训练样本数将新batch28候选终点改为1661。
基线46,500条，候选46,508条；batch、分组与训练调度仍不同，结果属于实际模型效果比较，不是严格控制变量的loss系数消融。

- 基线：p2_ltx_all_attention_ffn_lora31_20261108/step_001500.pt，E=.05，双路F=.025。
- 候选：p2_event_feature_interaction28_20261111/runs/event020_feature010/step_001661.pt，E=.20，双路F=.10。
- 当前训练先安全保存并由resume1661.py完整恢复；1500不停止，1661完整保存并通过STAGE_1661校验后接续。
- driver.py 等待STAGE_1661以及原分配锁释放，随后导出候选、复用已验证基线导出，确认28卡释放再启动rolling_driver。
- 同Seamless64、SFT2752规划/H/Gemma/Whisper缓存、Euler30、CFG3/7、512、seed137/238。两组共256生成结果。
- node12/node8/node4 GPU0–7，node11 GPU0–3。跨组共享动态生成、流水解码/指标、GPU租约事件评分，复用GT缓存。
- 不再训练基线，不自动选择赢家，不自动恢复后续训练。

CPU校验：3项测试通过；候选路由/错误权重拒绝、原生调度导入、样本与条件不变。旧基线完整模型/Adam/31RNG、来源与导出逐值检查通过。尚未进行1661的GPU生成验收，须待候选权重保存。

入口：driver.py；等待状态HANDOFF_STATUS.json；错误HANDOFF_FAILURE.json；评测状态STATUS.json与ROLLING_VIDEO_STATUS.json；最终COMPLETE.json、COMPARISON.json。
目录名中的1500保留为历史名称，实际候选终点以BASELINE_SELECTION.json及protocol arms.step=1661为准。


2026-11-11事件评分并行修复：parallel_events.media_ready严格校验全64条generated complete、checkpoint/step/seed、规划身份、无未来输入、独立音频及28份配对清单。rolling_driver按媒体就绪提前启动事件CPU准备/候选/GPU检测，与视觉音频指标共享GPU租约并行；SCORING_COMPLETE仍必须等待FINAL_AUDIT，不能提前宣称完整完成。旧batch31基线缺loss_search字段采用明确arm名识别，避免事件末尾KeyError。旧控制器安全退出、256条生成和解码保留，PLAN/ROLLING_PLAN同步新调度源码hash；验收 early_events_patch/ACCEPTED.json。

2026-11-11旧基线评分封存兼容补齐：历史31卡配置没有loss_search字段，原reference_events.freeze和stage_driver.verify_scoring会KeyError。driver仅在本次进程适配原生函数：旧基线必须匹配BASELINE_SELECTION精确权重路径及baseline1500 arm；所有来源、媒体、GT与结果哈希检查保留，不修改历史协议或权重identity。实际baseline seed238已通过freeze/封存/verify全链路验收，复用检测结果，剩余组恢复。证据historical_seal_fix/ACCEPTED.json。


2026-11-11新增旧基线6000快速双seed：用户要求比较同一原全LoRA完整模型1500→6000，系数E=.05/F=.025。新增baseline6000_6000_seed137/238；6000权重SHA3c2c78e8…，严格核验原200条seed137中同64条的shared item、模型SHA、planning SHA、SFT身份、采样、媒体完成标记并直接链接复用全部生成/latent。seed238新增生成64条。两seed均在同64条重算原生指标和事件评分，不复用200条FID/FVD。最终BASELINE_1500_VS_6000.json只比较旧基线两个阶段；COMPARISON.json同时保留候选1661。仅两检查点可判断净改善，不能证明中途单调提升。所有旧协议备份baseline6000_extension，扩大检查点白名单未改变历史模型身份或生成数据。SSH握手被拒绝增加仅针对kex失败的有界重试；保留guard stdin取消，仅停止自有SSH子进程。测试已通过。

2026-11-11复用6000 seed137路径校验修复：64条逐步清单与原生完整绑定仅generated路径不同（符号链接路径vs原目录真实路径）。复用目录改为真实目录+逐文件同inode硬链接，媒体字节不变；旧binding缓存归档后重建，64条canonical_audit已实际通过。没有跳过校验、没有重新生成。证据baseline6000_extension/PATH_BINDING_FIXED.json。
