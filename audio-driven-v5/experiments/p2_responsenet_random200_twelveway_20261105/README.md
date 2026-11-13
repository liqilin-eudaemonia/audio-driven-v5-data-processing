**全LoRA6000／6428已加入总结果页（2026-11-09）：** Seamless固定200现20组／8组消融／37项对照，ResponseNet仍18组／7组／30项（全LoRA两组尚未评测）。新增原范围完整6000→全LoRA6000／6428、6000→6428和AV／Event／Feature／Whisper基线对照，B−A箭头、原值、排名前两名配色、音频覆盖及CSV同步；注明时长分组512→496、非严格单变量。旧Seamless视频页指标总览同步20组，原有14组视频选择与音轨保持。938项数值／差值及双页浏览器、移动端、CSV、播放检查通过。[打开全LoRA对照](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/?dataset=seamless&ablation=all_lora#ablation-all_lora)。

**可视化数值已更新：** [打开两测试集十二组指标与消融增减表](http://localhost:8876/listen_benchmark_review/p2_responsenet_random200_20261105/)。[使用、重建与数值验收](DASHBOARD_README.md)。

**当前状态：全部完成（2026-11-05 08:48 UTC）。** 12组×200条＝2400条生成导出，384评分分片与全量汇总结束，视频／音频执行错误0；UTMOS为N/A。GPU已释放。结果见[RESULTS.md](RESULTS.md)、[summary.csv](summary.csv)与[最终验收](FINAL_AUDIT.json)。下方为运行过程记录。

**四机32卡现已实际运行：** 共享队列已启用，组合监督组已提交首批32条；实测四机各8/8卡计算，利用率98%–100%。[当前运行验收](expansion32/LIVE_ACCEPTANCE.json)。

**2026-11-05已加入node8的4–7卡：** 额外4卡已实际运行Whisper，原28卡继续当前组；按安全阶段边界自动升级为四机32卡共享领取。测试集、权重和采样不变。[扩容与动态调度记录](expansion32/README.md)。下文28卡为初次启动配置。

# ResponseNet分层随机200条：十二组统一评测

用户确认改用分层随机200条，替代本轮原拟完整586条。旧精选200条及原完整586条结果保持原样。本轮不参考任何生成得分选样。

## 数据与抽样

- 母集：已接入的完整586条ResponseNet pair，135个源视频，冻结父references SHA。
- 随机种子20261105；按**GT音频状态×片段时长×官方／本地ASR来源**联合分层。时长分为<5、5–8、≥8秒；音频分为严格全零、非零但未达活动阈值、有声学活动。
- 按母集比例用最大余数法分配配额，同余数随机打破并在层内均匀随机抽样。没有按模型画面、行为丰富度、生成分数或描述挑样；不强制每个源视频等量。
- 固定200条，91个源视频，1504.605秒（25.08分钟）；183条严格静音、17条非零，其中13条达到声学活动阈值。母集48/586为非零；本轮17/200，比例接近。
- 每层配额与来源分布见`selection_summary.json`；清单`selection.csv`、`selection.jsonl`；实际输入参考为`references/references.jsonl`。200条全部保留原始媒体、起止时间与来源字段，不新切分。
- 原精选200条是全部48条非零＋按模型得分选152条静音；这轮与它只重合72条。两份清单用途不同，不混合报告。

## 十二组

| 条件／监督 | 步数 | 推理Qwen |
|---|---|---|
| 原始H＋仅AV | 6000、8927 | 原生 |
| 原始H＋AV＋Event | 6000、8927 | 原生 |
| 原始H＋AV＋Feature | 6000、8927 | 原生 |
| SFT H＋AV＋Event | 6000、8927 | 规划SFT LoRA |
| SFT规划器＋原始H训练的Event | 6000 | 规划SFT LoRA；文本和推理H一起变化 |
| 原始H＋AV＋Event＋Feature | 7000 | 原生；约190000训练展示，近似匹配32卡6000的192000 |
| 原始H＋仅AV＋speaker Whisper cross-attention | 6000、8000 | 原生；额外Whisper分支 |

合计2400条输出。原四组在586条上的输出均已存在；其余五组各有72条可复用。总计复用1160条，新生成1240条。每份复用记录核对权重、step、seed、采样参数及未输入GT未来信息，见`REUSED_GENERATIONS.json`。

## 输入和公平性

仅输入speaker audio＋listener首帧。listener未来视频、listener音轨和GT回应描述均不作为条件。原生组复用原586条Qwen H／Gemma／首帧；SFT组三个权重共享匹配的SFT条件，仅补128条缺少的规划缓存。

统一512×512、各pair原始帧率（23.98／24／25／29.97／30fps）、seed137、30步Euler、videoCFG3/audioCFG7、原片段时长；先按原生8n+1采样，再导出原帧数和音频时长。Whisper只编码当前pair的speaker WAV，重采样16k，时间从0开始、保留静音；复用与原十组完全相同的采样函数和噪声精度，在CFG两条路径保留Whisper条件。

Whisper8000仅是训练进展参考，与AV8927不等步数；各训练时长分组启动时刻不同。这些是既有pipeline对照，不宣称全部训练变量严格相同。

## 调度和速度

四机授权28卡：node17、node12、node4各0–7卡，node8（172.29.104.237）0–3卡。只使用已释放显存的授权卡，不终止其他任务。接在已完成的Seamless Whisper补测之后。

复用现有模型常驻、按视频token量均衡分片、CPU预处理、Qwen/Gemma batch16、每卡单生成进程、兼容权重连续切换、VAE常驻导出。GT特征缓存沿用已有目录。评分阶段GPU检测／关键点／表情、batch64、像素线程4；动态平衡多卡评分，每卡上限4个worker。

四个完整586条旧组的逐条视频／音频得分与FID/FVD特征行，在当前200条ID、GT、生成文件和时间完全一致后直接复用；其余八组计算当前200条。**FID/FVD从当前200条全部特征全局重算，不把586条分数拿来替代，不平均分片分数。** 分片复用收据见`REUSED_SCORES.json`，代码`src/p2_testsets_eval/reuse_scores.py`。旧评分核心源码逐文件核对与当前一致。

## 入口和结果

- 运行入口：`scripts/p2_responsenet_random200.py prepare|queue --work experiments/p2_responsenet_random200_twelveway_20261105`。
- `status.json`：总阶段；`CONTROLLER.json`：后台PID；`controller.log`及各group的`logs/`：诊断日志。
- `PRECHECK.json`：CPU准入／复用数量；`resources.json`：最新资源检查；`progress/metrics`与各group的`progress`：进度。
- 最终`RESULTS.json`、`summary.csv`、`RESULTS.md`、`COMPLETE.json`；逐条指标在`metrics/<arm>/responsenet/`。
- 测试集输入仍是`references/references.jsonl`；生成文件在`generated/<arm>/responsenet/<id>/`，包括`generated.mp4`和独立`listener.wav`。
- 原有九项指标、声学活动匹配、短回应／静音分流与有效覆盖率保持不变。UTMOS无有效窗口时N/A；不补零、不删除失败样本冒充完成。
- 本轮只有13条GT活动样本，音频比例指标波动较大；小幅模型差异不直接当作稳定优势。若最终前两名接近，再补完整586条确认。

CPU回归：随机稳定性与不使用模型分数、层配额、分布特征偏移、Whisper音轨和时间轴3项通过。真实GPU新生成以运行进度和最终验收为准。
