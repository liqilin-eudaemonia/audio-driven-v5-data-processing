**已全部完成（2026-11-09 11:36:58 UTC／北京时间19:36:58）：** 200条生成、200条解码、28份评分分片及全量合并验收通过；评测进程已释放GPU。完整指标`RESULTS.json`，通过证明`FINAL_AUDIT.json`、`HANDOFF_COMPLETE.json`。

**已四机28卡实际启动：** 28个原生模型严格加载通过，每卡1632处LoRA目标、四个实际主机核对通过；node11仅GPU0–3，本轮未向4–7派发任务。完整权重及290份来源、6项回归通过；生成及后处理结果以`status.json`和最终验收为准。

# 全attention／FFN LoRA完整组6428：Seamless200，四机28卡

最新完整权重：`../p2_ltx_all_lora_full31_resumed_20261109/step_006428.pt`。来源SHA256：`d26f867f6030d929358bddcb9b27d70edead2e087ede77d73e0951914dd61274`。保留AV＋整段Event＋双路Feature＋speaker Whisper，48层1632处rank8 LoRA；不是无Event组。

固定与6000完全相同的Seamless200、SFT2752规划/H/Gemma、speaker Whisper、listener首帧、seed137、Euler30、video CFG3／audio CFG7、512和原FPS／帧数。不重新规划，不输入listener未来媒体或GT caption。与6000独立输出。

资源：node12／node8／node4各GPU0–7，node11只GPU0–3，共28卡。禁止本轮使用node11 GPU4–7；不停止现存vLLM。按剩余显存接入，不能用显存占用代替实际算力利用率。

沿用共享样本锁、最长片段优先、模型只读mmap、异步四机资源探测、就绪分片流水解码及评分、CPU视频读取预取／像素并行、完整200特征汇总FID/FVD。生成和解码清空PYTHONPATH，只有指标进程使用vendor目录。每卡生成结束即可接续后处理，不等待最慢节点。

完整checkpoint先核对模型／Adam／31 RNG及290份源码（282份原生＋8份迁移适配）。只导出模型张量，逐值核对；GPU仍严格载入全部1632目标。迁移来源身份在进程内校验，不修改原训练源码或历史评测。

入口：`scripts/p2_all_lora6428_seamless_eval28_20261109.py prepare|queue|worker`；适配：`src/p2_all_lora6428_eval_20261109.py`。实际进度见`status.json`，GPU启动验收见`RUNNING28_ACCEPTED.json`（生成后），日志见`logs/`。

输出：`generated/ltx_all_lora_event_feature_whisper_6428_sft2752/seamless/<id>/generated.mp4`及`listener.wav`。最终指标`RESULTS.json`；与6000对比`COMPARISON.json`。完成必须通过`PROGRESSIVE_BIND_AUDIT.json`、`FINAL_AUDIT.json`和`HANDOFF_COMPLETE.json`。当前准备／运行状态以实际收据为准。

根目录创建STOP仅停止本轮；不恢复训练，不创建Goal或周期聊天汇报。
