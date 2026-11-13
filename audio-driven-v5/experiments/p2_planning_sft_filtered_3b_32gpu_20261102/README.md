# 最新筛选数据 Qwen-Omni 规划 LoRA 四机训练

2026-11-02 用户授权启动。四机真实训练已开始，32卡首批反传通过：LoRA梯度有限且非零，冻结参数无梯度。启动验收见 [launch_verified.json](launch_verified.json)（一次性快照，不是实时状态）。node17、node8、node12、node4 各8卡，32个DDP rank；后台启动记录见 [launcher.json](launcher.json)，主日志见 [launcher.log](launcher.log)，各机日志在 `launch_logs/`。

- 数据：train 690,337条／1,155.44h，最新 `listen_j_train_dev_response_prefixfix_20261102/current/train.jsonl`；dev 5,238条保留但不做训练期或结束评估，也不并入train。
- 输入：speaker音频＋listener最终裁框首帧；目标为原文回应描述，只监督assistant token及EOS。
- 基座：原始Qwen2.5-Omni-3B；新LoRA、新优化器，独立于历史500h训练。
- 每卡batch8，32卡global batch256，梯度累积1，1轮约2,697步；沿用既有加速参数。
- 每500步保存，`save_total_limit=3`保留最近3份周期检查点；完整结束另存`final_adapter/`。不执行dev评估。
- 配置：[训练](../../configs/p2_planning_sft_filtered_3b_32gpu_20261102.json)、[四机](../../configs/p2_planning_cluster_filtered_20261102.json)；[数据和代码交接](../../related/P2_Qwen规划LoRA新筛选数据接入_20261102.md)。

运行后的`run.json`记录数据、代码和world size；`gradient_audit.json`记录真实梯度；`step_metrics.jsonl`记录训练步时；TensorBoard在`tensorboard/`。只有`COMPLETE.json`才表示正式完成，不以进程启动代替完成。

恢复只能使用本目录的完整`checkpoint-*`，配置、代码、数据及world size须一致。主调度器为独立后台会话；没有新增定时通知任务。
