# 2026-10-26：P2 LoRA 联合训练 GPU 验证与提速

已完成；24 张 GPU 已释放，没有启动正式效果实验。

- [主报告与启动说明](../../related/P2_24卡联合训练GPU验证与提速.md)
- [最终验收汇总](summary.json)、[性能数据](benchmark_summary.json)、[GPU 释放核对](final_gpu_inventory.json)
- [加速前后模型/优化器/RNG 逐位一致](optimization_equivalence.json)
- [加速版单/双卡](optimized_pair_comparison.json)、[E0 恢复](optimized_replay_E0.json)、[E1 恢复](optimized_replay_E1.json)
- [140 项 CPU 测试日志](cpu_tests_final.log)、[两进程 CPU DDP](cpu_ddp_optimized.json)
- [24 卡正式入口 S0](optimized_24gpu/runs/s0_20261026/E1/)、[单卡及实际续训](optimized_1gpu/runs/s0_20261026/)、[双卡](optimized_2gpu/runs/s0_20261026/)
- [独立正常生成](optimized_generate_normal/output/)、[移除 U 的生成](optimized_generate_remove_u/output/)
- [修改前源码](code_before/)、[修改后源码](code_after/)、[当前源码校验清单](final_code_manifest.json)、[修改边界](source_changes.json)

`bench_*` 为有限性能试验，不用于效果结论；每组 6 步，第一步预热。`probe_*` 为跨机网络校验。首次默认 RoCE/GPU Direct 试验失败的原因、停止的试验及通过的替代配置记录在 `summary.json`，没有未说明的失败或仍运行的任务。不要用 debug 检查点做正式效果比较。
