**结果分析与可播放对照：** [十组总体分析、配对区间及限制](../p2_seamless200_analysis_20261105/ANALYSIS.md) · [16条GT/A/B视频与声音对照（8876）](http://localhost:8876/listen_benchmark_review/p2_seamless200_20261105/)。每条可切全部10组，旧ResponseNet入口也已加入新页签。

**Seamless200十组评测计算已结束（2026-11-05 05:53 UTC）：** 2000条生成、240个评分分片及全量FID/FVD汇总均结束；GPU已无待处理任务。正式结果保留`partial`：每组1–2条缺有效人脸／相邻关键点支持，UTMOS无有效评分窗口，未补0或删除样本；后端初始化和音频执行错误均0。[正式结果](RESULTS.json) · [完成及覆盖核对](FINAL_RUN_AUDIT.json)。下方运行中记录为本轮历史阶段。

**node8评分空档已接续（2026-11-05）：** 前7组评分完成后，node8的0–3卡提前协助导出SFT Event6000的完整200条，已完成并接上该组评分；原三机采样没有重启。导出新增逐样本互斥，和后续24卡导出共用完整结果；该组评分使用不可变清单快照，避免整组后续绑定时改写文件。后续启动的评分调度已改为先均衡分配到空GPU，再追加并发，17项相关回归通过；本轮已处理过半的分片不搬迁重跑。见`NODE8_GAP_RECOVERY.json`及`node8_decode_assist/`。

**新增解码提速已验证并启用：** 固定200条GT完整RGB跨权重复用，真实两条的GT＋生成视频解码总耗时14.90→0.45秒、3.17→0.12秒；全部指标和FID/FVD特征逐值一致，17项回归通过。后续worker自动接入，正在运行和已完成评分不重算。[实现、数据身份、实测与缓存位置](DECODE_CACHE.md)。

**当前执行更新：** node8 `172.29.104.237`的GPU0–3已授权并加入指标计算，4–7卡排除。评分自检已通过，node8先以4卡×8进程处理已成片系统（192核实测CPU26%、4路/卡时显存23–24GB后提高并发），与三机24卡推理并行；全部生成完成后才开放这24张推理卡给评分，共28卡。此前node8八卡九组推理自动恢复仍取消，不会复活旧任务。推理结束后后续控制器等待已有评分队列完成，失败时可取得已释放队列锁，不重复提交。队列与既有评测回归13项通过。下文原“全部生成后再评分”表述由本段更新。

# Seamless200：三机24卡，历史九组＋组合损失7000步

用户要求停止当前训练并优先评测。训练已完整保存并停止于7610步，不再自动恢复。评测节点为node17 `172.29.15.91`、node12 `172.29.252.37`、node4 `172.29.225.51`，各GPU0–7；不占用node8，其旧自动接续已取消。

## 对比范围

| 系统 | 权重步数 | 规划器 |
|---|---:|---|
| av_event_6000 | 6000 | unmodified_Qwen2.5-Omni-3B |
| av_event_8927 | 8927 | unmodified_Qwen2.5-Omni-3B |
| av_only_6000 | 6000 | unmodified_Qwen2.5-Omni-3B |
| av_only_8927 | 8927 | unmodified_Qwen2.5-Omni-3B |
| base_feature_6000 | 6000 | unmodified_Qwen2.5-Omni-3B |
| base_feature_8927 | 8927 | unmodified_Qwen2.5-Omni-3B |
| base_event_feature_7000 | 7000 | unmodified_Qwen2.5-Omni-3B |
| sft_event_6000 | 6000 | planning_SFT_Qwen2.5-Omni-3B |
| sft_event_8927 | 8927 | planning_SFT_Qwen2.5-Omni-3B |
| sft_on_base_event_6000 | 6000 | planning_SFT_Qwen2.5-Omni-3B |

组合损失为AV＋0.05 Event＋0.025视频Feature＋0.025音频Feature，冻结原始Qwen H。它先在28卡训练5500步，再在24卡继续，因此7000步累计190000条，接近其他32卡组6000步的192000条（少1.04%）；不是把三机7000步直接等同四机6000步。7610步仅为停止时保存，不作为本轮比较组。

## 数据与复用

[固定200条／19.99分钟](../seamless_quick200_latest_20261105/README.md)，全部系统使用相同样本、seed137、30步Euler、video CFG3／audio CFG7、512×512、25fps及原pair完整时长。实际输入只有speaker音频＋最终裁框listener首帧；GT描述及listener未来音视频不输入模型。

从[此前九组运行](../p2_seamless200_nineway_20261105/README.md)复用800条完整latent、656条完整成片，原产物不覆盖。原始规划器的200条Qwen H／Gemma条件及已验证相同的负向条件复用；SFT规划器单独计算并校验对应LoRA身份。缺少的1200条采样及1344条视频继续补齐。

## 执行与加速

原生推理，不改变采样数学、步数、seed或CFG。24卡按原生帧数均衡分片，每卡8–9条；每卡常驻模型、CPU四线程准备。架构兼容权重共用LTX／VAE加载；完整产物按提交标记跳过。基础四组只补导出，其后Feature两组＋组合损失一组，再SFT三组。Qwen／Gemma沿用batch16，未采用历史会OOM的batch32。

全部生成后24卡切换GPU评分：每卡最多4个评分进程，动态领取240个分片任务，共享GT特征，LPIPS／FID batch64、GPU检脸／关键点和表情。沿用PSNR、SSIM、FID、FVD、LPIPS、Emo-Acc、Emo-Div、ΔResp、UTMOSv2及音频活动统计；FID／FVD用全部200条特征合并重算，UTMOS静音／过短为N/A并报告覆盖率。Seamless face-padding为0。

## 入口与状态

- [预检查](PRECHECK.json)、[协议及全部权重路径](protocol.json)、[资源配置](cluster.json)、[复用审计](REUSE_AUDIT.json)。
- [总状态](status.json)、`queue.log`、`watch.json`；子目录`base_standard`、`base_feature`、`sft_event`有各阶段进度及日志。
- `generated/<系统>/seamless/<id>/generated.mp4`及`listener.wav`：成片与独立listener音轨。
- 最终`RESULTS.json`、`UTMOS_COMMON_WINDOWS.json`、`COMPLETE.json`；仅有`GENERATION_COMPLETE.json`不代表指标已完成。失败和覆盖不足明确保留。

准备入口：`scripts/p2_seamless200_continue24.py`。运行与有限故障恢复入口：`scripts/p2_seamless200_nineway.py run|watch --work <本目录>`。沿用入口名称，运行参数已支持24卡及10组。控制器持有排他锁，不得重复启动。守护随本评测结束退出，不创建定时用户汇报。

组合7000权重SHA及模型结构已核对；其唯一新增源码是三机迁移专用训练调度入口，精确摘要在`compatibility/base_event_feature_7000.json`，其余推理源码一致；未关闭来源检查。历史权重继续使用此前GPU重放验证过的精确兼容记录。
