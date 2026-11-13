**当前已按用户要求停止（2026-11-05 05:00 UTC）：** 完整第7610步检查点已落盘；不是完整一轮训练结束。三机24卡转入[新Seamless200十组测评](../p2_seamless200_tenway_24gpu_20261105/README.md)，其中采用第7000步组合损失权重以接近其他实验6000步的累计样本量。两个STOP文件保留，禁止自动恢复训练。下文为此前训练启动记录。

**真实启动已验收：** 三机24个rank均在运行，从5500步恢复后已持续更新，损失公式、四组非零梯度和有限性通过；见`STARTUP_ACCEPTANCE.json`、`GPU_RUNNING_VERIFIED.json`与`RESUME_VERIFIED.json`。

# 原始H＋AV＋Event＋Feature：三机24卡继续训练

2026-11-05用户指定不再等待node8，使用node17、node12、node4各8卡继续。当前训练控制器已启动，实际启动验收以`RESUME_VERIFIED.json`及`metrics.jsonl`为准。

- 输入仍为原500h、285638条train，原始未SFT Qwen H、Gemma完整打标文本、最终512×512修正版listener裁框、官方时间并集净化listener音轨，以及已验收Feature teacher；没有Whisper条件。
- 损失保持`AV + 0.05 Event + 0.025 video Feature + 0.025 audio Feature`。模型、Connector、Event head、Feature decoder、text-attention LoRA及梯度路径不变。
- 从原28卡实验完整`step_005500.pt`迁移。原模型及Adam状态逐值一致；24份已存在的CPU/CUDA随机状态按`MIGRATION.json`中的原rank映射继承。未将不匹配的28卡检查点冒充严格24卡检查点。原运行及权重保留不变。
- global batch由28改为24，每卡每步1条。原5500步已消耗154000条真实样本；剩余131638条真实样本/ordinal全部保留，按384条（16步）时长窗口分组，重新组织剩余批次，不更换逐样本噪声种子。
- 最后一批补齐2条确定性重复样本，去掉原28卡日程中额外的16条尾部padding；没有丢弃真实训练数据。后续5485步，最终计数10985；该步数包括原5500步，不能按10985×24反推总样本数。
- 每500步保存及最终保存，下一次正常保存6000步；保留BF16、three_quarters梯度重算、FP64梯度SUM/32MiB桶、原学习率/优化器及既有单HCA RoCE配置。

批大小和剩余批次组织改变，因此后续优化轨迹不同，不能声称与28卡原运行逐更新一致。比较其他实验时优先用完整一轮结果；同6000步的累计样本量也已不同。

## 入口和证据

- [固定配置](config.json)、[三机集群](cluster.json)、[迁移审计](MIGRATION.json)、[实时状态](status.json)、[控制器阶段](queue/state.json)。
- 统一脚本[`../../scripts/p2_j_event_feature_continue24.py`](../../scripts/p2_j_event_feature_continue24.py)，仅新增独立迁移/调度入口，未改动104个原训练源码。
- 原28卡完整全量preflight按相同缓存/teacher/Event身份复用，未重新扫描285638条媒体或提取缓存。
- CPU共7项检查通过，其中3项专测实际285638条预算与全部未消耗ordinal保留、24rank每步各一条、缺失样本拒绝；另外4项为检查点接续安全与PID配额门控。
- `MIGRATION.json`确认模型/Adam逐值一致、所选24份RNG一致、仅新增本入口源码指纹；原损失、模型、数据配置未改。
- `RESUME_VERIFIED.json`仅在真实GPU更新出现、损失公式和四组梯度验证后写出；准备完成不等于真实开训验收。

控制器持有`queue/controller.lock`，不能重复启动。队列会检查三机24卡空闲及各机PID/线程余量，完全排除node8。启动失败保留日志并最多从最新完整检查点重试5次；只管理本运行持有的进程。`queue/STOP`请求目标训练正常保存后退出；`output/STOP`为训练器自己的保存停止信号，不能随意删除。

原Whisper实验已保留完整8000步权重及独立SHA校验，位于`../p2_j_qwen_whisper_av_tuned_20261104/step_008000.pt`，没有继续到8927步。原28卡等待队列已关闭，防止node8恢复后另起冲突训练。
