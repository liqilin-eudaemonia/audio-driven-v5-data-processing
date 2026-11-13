**2026-11-03新增第5组：SFT LoRA Qwen＋原始H训练的Event第6000步，共200条。总计5组／1000输出。** 新组 `sft_on_base_event_6000`，原权重来自 `p2_j_guided_av_event_s1_32gpu_alternating_20261101/step_006000.pt`；`sft_on_base_event/conditions` 和 `negative` 直接链接当前SFT组目录，确保两组规划文本、H、Gemma条件一致。seed和生成参数完全一致。旧原生Qwen＋原始H Event6000的历史结果位于上轮200条选择目录，可作为A组参考。

**恢复记录：** 原队列在SFT6000导出187／200条时因node8 SSH握手中断而停止；评分队列因评测机SSH查询255而退出。均保留原结果和失败日志。GPU启动增加单机2个并发握手、连接失败退避重试；评分资源查询失败只跳过本次host并下轮重试，不再结束整条队列。已恢复原队列并在末尾接上新增组。此前30–50分钟估计因连接故障失效。

**旧权重兼容验收：** 旧训练源码中有4个文件已做过Feature支持扩展；生成核心Connector、LoRA、LTX及采样文件身份不变。`sft_on_base_event/source_compatibility.json` 显式绑定旧源码指纹、当前指纹及checkpoint SHA，仅认可该已审核组合。用旧原生Qwen条件对同一权重单样本GPU重放，视频和音频latent均与历史结果逐值相同，见 `LEGACY_REPLAY_VERIFIED.json`；不是普遍关闭来源校验。13项CPU回归通过。

比较B（新增SFT规划器＋原始H Event）和C（SFT规划器＋SFT H Event6000）可更直接检查训练条件匹配作用。两次既有训练的时长分组启动点不同，因此不是严格只改变一个因素的重新训练实验。以下保留最初4组协议说明，最新组数及总量以 `protocol.json` 为准。

# ResponseNet固定200条：Feature／SFT Event各6000与8927步

本轮只使用用户确认的 `experiments/responsenet_initial200_20261102/selection.csv` 和 `references.jsonl`，不重新筛选。200条×4权重＝800个输出。原200条文件及其sealed SHA保持不变；本目录 `references/` 是兼容原bind接口的只读内容副本，保留原始选择seal与CSV哈希。

## 四组与输入

| 组 | LTX/Connector权重 | Qwen规划与H |
|---|---|---|
| base_feature_6000 | 原始H＋AV＋0.025视觉Feature＋0.025音频Feature，第6000步 | 原始Qwen2.5-Omni-3B |
| base_feature_8927 | 同组第8927步 | 同上，与6000共享条件 |
| sft_event_6000 | SFT新H＋AV＋0.05 Event，第6000步 | 对应2697步规划SFT LoRA |
| sft_event_8927 | 同组第8927步 | 同上，与6000共享条件 |

权重路径、规划LoRA文件SHA、参考清单SHA均记录在 `protocol.json`、各子目录protocol及 `checkpoint_hashes.json`。两种架构分别用独立GPU阶段加载，避免Feature decoder和Event模型结构混用；每个checkpoint都核对其原训练代码身份和资产。

只输入speaker音频＋listener首帧，不读GT回应描述、后续画面或listener音轨作生成输入。原始组复用旧586条推理中这200条的原始Qwen/Gemma条件（逐文件SHA和refs逐条对照通过），SFT组独立计算规划、H和Gemma缓存；两种规划器不会混用缓存。两组比较同时改变H/规划器及辅助损失，因此是两套pipeline的比较，不能单独归因于某一项。

每条seed137，30步Euler，video CFG3/audio CFG7，512×512，原FPS与GT帧数；首帧和GT统一bicubic缩放。沿用已有三阶段常驻GPU模型、32卡按帧数均衡分片、CPU四线程供数、Qwen/Gemma batch16、完成标记续跑。64条GPU冒烟（每卡2条）对每种pipeline分别执行，通过后进入全量；验证样本可复用，不重复计数。

## 自动接续

入口 `scripts/p2_response200_fourway.py queue --work <本目录>`，后台PID/命令在 `launch.json`，日志 `queue.log`。**控制器已运行时不要重复启动。**

等待当前SFT新H＋AV＋Event完整结束、训练controller complete、四个checkpoint校验成功、32卡无其他compute进程并释放足够显存，然后开始。训练不暂停、不叠加GPU推理。顺序为base Feature 6000→8927，再SFT Event 6000→8927。各pipeline内部复用对应规划条件。

`base_feature/`、`sft_event/` 保存各自条件、latent、生成MP4/独立listener WAV及元数据；父目录 `generated/<arm>/`、`maps/`、`bindings/`、`metric_manifests/` 链接到实际输出。父 `status.json` 为总队列，子status记录具体GPU阶段。开始前CPU检查通过不代表GPU生成已通过。

## 评测

沿用上轮九项：PSNR、SSIM、FID、FVD、LPIPS、Emo-Acc、Emo-Div、ΔResp、UTMOSv2。GPU检脸/关键点/表情，batch64，像素线程4，共享原GT缓存，每卡最多4个评分worker。先用172.29.189.119空卡；全部生成结束后加入四机已释放GPU。评分只申请空卡，不终止其他程序。

生成和评分独立推进：一组pipeline输出绑定后可评分，后组继续生成。FID/FVD由200条全量分片特征重算；不能平均分片指标。UTMOS对短回应/静音保留N/A及覆盖率，并在全部4组结束后按共同有效窗口重新汇总，不将静音当普通语音。

最终 `RESULTS.json`、`metrics/<arm>/responsenet/summary.json`、逐条分数与 `UTMOS_COMMON_WINDOWS.json`。`GENERATION_COMPLETE.json`只表示800条生成完成；`COMPLETE.json`还需查看complete/partial和指标覆盖率。

CPU接入：11项测试通过，2条真实source输入首帧逐像素复现旧输入，音频长度和时间语义通过；200条源文件存在；已存在的3个checkpoint SHA通过，第4个等待训练结束后校验。见 `CPU_CHECK.json`、`CPU_INPUT_CHECK.json`、`cpu_tests.xml`。

父目录 `STOP` 会阻止启动下一pipeline；正在执行的子pipeline如需停下，也应在对应子目录创建STOP。已完成文件保留，恢复时先确认旧控制器已退出。GPU或评分失败写failed并保留日志，不跳过样本冒充完成。
