# 全LoRA完整pipeline step7500 · Seamless200

**2026-11-10按新授权扩至node11 GPU4–7：** 5000／5500／7000均已完成；剩余7500从单卡切换四卡，保留已完成的83条latent，四卡均已严格加载1632处LoRA并实际产出，采样／输入／指标协议未变。复用原四卡逐卡释放、共享任务锁及并行解码评分；10项共享队列与调度测试通过。原目录名中的gpu7是历史命名，不代表当前卡数。[实际四卡验收](resume_gpu4_20261110/ACCEPTED.json) · [运行身份](RUNTIME_GPU4_20261110.json)。

当前入口：`scripts/p2_all_lora7500_node11_gpu4_20261110.py queue`；实际状态见`status.json`，CPU准备见`CPU_ACCEPTANCE.json`，最终完成见`FINAL_AUDIT.json`与`HANDOFF_COMPLETE.json`，指标`RESULTS.json`。原[接续总队列](../p2_ltx_all_lora_followups_gpu7_20261109/README.md)的前三轮完成记录保留，最后一轮完成后自动补齐总队列收据。创建本目录`STOP`停止当前四卡队列。
